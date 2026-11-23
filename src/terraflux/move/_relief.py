# SPDX-License-Identifier: GPL-3.0-or-later
"""The length of a step on the relief: the 3D distance over a DEM."""

from __future__ import annotations

import numpy as np

from .. import _core
from .._threads import resolve
from ._along import _grid
from ._ssf import _layers
from ._tracks import info_of


def _geographic(crs) -> bool:
    if crs is None:
        return False
    from pyproj import CRS

    return bool(CRS.from_user_input(crs).is_geographic)


def surface_distance(steps, dem, *, spacing: float | None = None, x1: str = "x1_", y1: str = "y1_",
                     x2: str = "x2_", y2: str = "y2_", name: str = "sl_surface_", n_threads: int | None = None):
    """The length of each step measured on the relief, not on the plane.

    Not in amt (nor in R's move2): on a steep slope the planar length ``sl_`` of a
    step underestimates the distance the animal walks, and with it its speed
    (Signer et al. 2019). Here the step is followed over a DEM: points are taken
    along the segment every ``spacing`` (the last stretch, shorter, ends exactly
    at the end of the step), plus its start and end; the height of each point is
    the bilinear interpolation between the centres of the four nearest cells
    (``terra::extract(method = "bilinear")``); the length is the sum, in order, of
    ``sqrt(dxy^2 + dz^2)`` of the chords between consecutive points. It is never
    shorter than the planar length, and equals it on a flat DEM. On a plane (a
    constant slope) the bilinear heights are exact, so the length is exactly the
    planar one times ``sqrt(1 + slope^2)`` along the step.

    Edges: a point in the outer half cell (between the edge of the raster and the
    outermost centres) takes the height of the nearest edge cell (clamped); a point
    outside the raster's extent gives NaN. The height is NaN, and so the step,
    when any of the four cells it is interpolated from has no data. A step of
    length 0 inside the DEM is 0 (outside, NaN). A step with a NaN coordinate
    (as ``random_steps`` draws from a zero-length step) is NaN.

    Parameters
    ----------
    steps
        The :func:`steps` (or :func:`random_steps`) of tracks, in a projected CRS
        in metres.
    dem
        A ``(y, x)`` DataArray with the heights **in metres**, north up, in the
        same CRS as the steps. Geographic coordinates (steps or DEM in degrees) are
        an error: heights in metres and distances in degrees do not add.
    spacing
        Distance between points along the step, in the units of the coordinates
        (default: a quarter of the smaller cell side).
    x1, y1, x2, y2
        The columns of the start and the end of the steps.
    name
        The new column.
    n_threads
        Threads (default: :func:`terraflux.default_threads`); the result does not depend on them.

    Returns
    -------
    pandas.DataFrame
        A copy of ``steps`` with the column ``name`` (metres) added.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> import terraflux as tf
    >>> dem = xr.DataArray(np.tile(np.arange(5.0) * 3, (5, 1)), dims=("y", "x"),
    ...                    coords={"y": [4.5, 3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5, 4.5]})
    >>> steps = pd.DataFrame({"x1_": [1.0], "y1_": [2.5], "x2_": [4.0], "y2_": [2.5]})
    >>> tf.move.surface_distance(steps, dem)["sl_surface_"].round(6).tolist()   # 3 m on a 3:1 slope
    [9.486833]
    """
    cols = (x1, y1, x2, y2)
    missing = [c for c in cols if c not in getattr(steps, "columns", ())]
    if missing:
        raise TypeError(f"steps: the steps of tf.move.steps or tf.move.random_steps (missing {', '.join(missing)})")
    layers = _layers(dem)
    if len(layers) != 1:
        raise ValueError("dem: one layer (a (y, x) DataArray)")
    _, r, z = layers[0]
    info = info_of(steps)
    scrs, dcrs = info.get("crs"), r.rio.crs
    if info.get("lonlat") or _geographic(scrs) or _geographic(dcrs):
        raise ValueError("surface_distance needs steps and a DEM in a projected CRS in metres: heights in metres and "
                         "distances in degrees do not add (project them with tf.reproject first)")
    if scrs is not None and dcrs is not None:
        from pyproj import CRS

        if CRS.from_user_input(scrs) != CRS.from_user_input(dcrs):
            raise ValueError("steps and dem must be in the same CRS (reproject the DEM or the tracks)")
    tr = _grid(r)
    if tr.b != 0 or tr.d != 0:
        raise ValueError("rotated rasters are not supported")
    if not (tr.a > 0 and tr.e < 0):
        raise ValueError("dem: rasters with north up (cells growing eastwards and southwards) are expected")
    dx, dy = float(tr.a), float(-tr.e)
    if spacing is None:
        spacing = 0.25 * min(dx, dy)
    elif not (np.isfinite(spacing) and spacing > 0):
        raise ValueError("spacing: a positive distance")
    a = [np.ascontiguousarray(steps[c].to_numpy(np.float64)) for c in cols]
    got = _core.move_surface_distance(*a, np.ascontiguousarray(z), float(tr.c), float(tr.f), dx, dy,
                                      float(spacing), resolve(n_threads))
    out = steps.copy()
    out.attrs = dict(steps.attrs)
    out[name] = got
    return out
