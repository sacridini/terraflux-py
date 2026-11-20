# SPDX-License-Identifier: GPL-3.0-or-later
"""Values along steps: the cells a step crosses (amt's extract_covariates_along) and their summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from ._ssf import _layers
from ._tracks import info_of

STATS = ("mean", "min", "max", "ascent", "descent", "delta")


def _grid(r):
    """The affine transform of a raster as its file gave it (the ``GeoTransform`` text rioxarray keeps, exact), else
    the one :func:`terraflux.io._transform` derives from the coordinates, which can differ in the last bits."""
    from affine import Affine

    from ..io import _transform

    tr = _transform(r)
    spatial = r.coords.get("spatial_ref")
    text = None if spatial is None else spatial.attrs.get("GeoTransform")
    try:
        c, a, b, f, d, e = (float(v) for v in text.split())
    except (AttributeError, ValueError):
        return tr
    tol = 1e-6 * max(abs(tr.a), abs(tr.e))
    if max(abs(a - tr.a), abs(e - tr.e), abs(c - tr.c), abs(f - tr.f)) > tol or b != 0 or d != 0:
        return tr   # the text is stale (a slice, a resampling): the coordinates are right
    return Affine(a, 0.0, c, 0.0, e, f)


def _walk(steps, covariates, n_threads):
    """(layers, values per layer (flat), offsets, cell, rank) of the steps on the raster's grid."""
    from ..sdm._points import to_crs

    if not all(c in getattr(steps, "columns", ()) for c in ("x1_", "y1_", "x2_", "y2_")):
        raise TypeError("steps: the steps of tf.move.steps or tf.move.random_steps (columns x1_, y1_, x2_, y2_)")
    layers = _layers(covariates)
    if not layers:
        raise ValueError("covariates: no layers")
    first = layers[0][1]
    tr = _grid(first)
    for _, r, v in layers:
        t2 = _grid(r)
        if tuple(t2)[:6] != tuple(tr)[:6] or v.shape != layers[0][2].shape:
            raise ValueError("covariates: the layers must share one grid (the same extent and cell size)")
    if tr.b != 0 or tr.d != 0:
        raise ValueError("rotated rasters are not supported")
    if not (tr.a > 0 and tr.e < 0):
        raise ValueError("covariates: rasters with north up (cells growing eastwards and southwards) are expected")
    ny, nx = layers[0][2].shape
    xmin, ymax = tr.c, tr.f
    xmax, ymin = xmin + nx * tr.a, ymax + ny * tr.e
    x1, y1, x2, y2 = (np.ascontiguousarray(steps[c].to_numpy(np.float64)) for c in ("x1_", "y1_", "x2_", "y2_"))
    crs = info_of(steps).get("crs")
    x1, y1 = to_crs(x1, y1, crs, first.rio.crs)
    x2, y2 = to_crs(x2, y2, crs, first.rio.crs)
    if np.isinf(np.r_[x1, y1, x2, y2]).any():
        raise ValueError("steps: infinite coordinates (terra's extract fails on them)")
    # amt pastes the coordinates into a WKT text: 15 significant digits (round15)
    offsets, cell, rank = _core.move_along_cells(x1, y1, x2, y2, float(xmin), float(xmax), float(ymin), float(ymax),
                                                 int(nx), int(ny), True, resolve(n_threads))
    return layers, offsets, cell, rank


def extract_covariates_along(steps, covariates, *, n_threads: int | None = None) -> list:
    """The values of rasters in the cells each step crosses (amt's ``extract_covariates_along``).

    As amt: ``terra::extract`` of the line from the start to the end of each step,
    with terra's defaults for lines (every cell the line touches, GDAL's
    ``ALL_TOUCHED`` line walk on the grid terra crops to the line; a step that
    touches no cell takes the cells of its two ends). The coordinates are
    rounded to 15 significant digits first, as amt's ``paste0`` does, and taken
    to the raster's CRS. A NaN coordinate (the steps ``random_steps`` draws from a
    zero-length step) gives, as in R, the cell of the other end alone (NaN when that
    one is outside too); infinite ones are an error. The cells come in terra's order, the cell number
    (row by row from the top left), not along the step; :func:`along_summary`
    puts them along it. A step outside the raster gives one NaN.

    Parameters
    ----------
    steps
        The :func:`steps` (or :func:`random_steps`) of tracks: columns ``x1_``,
        ``y1_``, ``x2_``, ``y2_``.
    covariates
        A ``(y, x)`` or ``(band, y, x)`` DataArray, or a dict (name: raster) or
        list of them, all on one grid.
    n_threads
        Threads (default: :func:`terraflux.default_threads`); the result does not depend on them.

    Returns
    -------
    list
        One entry per step, in the order of ``steps``: a 1-D array of the cell
        values for one covariate, a DataFrame (a column per covariate, named
        as in :func:`extract_covariates`) for several. NaN on no data.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> import terraflux as tf
    >>> dem = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("y", "x"),
    ...                    coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]}, name="dem")
    >>> steps = pd.DataFrame({"x1_": [0.5], "y1_": [3.5], "x2_": [3.5], "y2_": [3.5]})
    >>> tf.move.extract_covariates_along(steps, dem)[0].tolist()
    [0.0, 1.0, 2.0, 3.0]
    """
    layers, offsets, cell, _ = _walk(steps, covariates, n_threads)
    flat = {}
    for name, _, v in layers:
        w = v.ravel()
        flat[name] = np.where(cell >= 0, w[np.where(cell >= 0, cell, 0)], np.nan)
    cuts = offsets[1:-1]
    if len(layers) == 1:
        return list(np.split(flat[layers[0][0]], cuts))
    table = np.column_stack([flat[name] for name, _, _ in layers])
    names = [name for name, _, _ in layers]
    return [pd.DataFrame(part, columns=names) for part in np.split(table, cuts)]


def along_summary(steps, covariates, stats=STATS, *, n_threads: int | None = None):
    """Summaries of the raster values along each step: whether it climbs a hill or follows a contour line.

    Not in amt: a table made for step selection functions, from the cells of
    :func:`extract_covariates_along`. For each covariate and each of ``stats``
    a column ``<covariate>_along_<stat>``:

    * ``mean``, ``min``, ``max``: of the cells the step crosses (cells with no
      data left out; the mean as R's ``mean(x, na.rm = TRUE)``);
    * ``ascent``, ``descent``: with the cells in the order along the step, from
      its start to its end, the sum of the positive differences between
      successive cells and the sum of the negative ones, as a positive number
      (cells with no data are skipped, the difference is between the cells
      around them);
    * ``delta``: ``|last - first|`` of the values in that order.

    A step that follows a contour line has a small ascent and descent, one that
    goes straight up a slope has ``ascent == delta``. NaN where no cell has a value.
    The order along the step is the order GDAL's line walk burns the cells in
    (a step touching no cell: its start cell, then its end cell).

    Parameters
    ----------
    steps, covariates, n_threads
        As :func:`extract_covariates_along`.
    stats
        A subset of ``"mean"``, ``"min"``, ``"max"``, ``"ascent"``, ``"descent"``, ``"delta"``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``steps`` with the new columns.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> import terraflux as tf
    >>> dem = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("y", "x"),
    ...                    coords={"y": [3.5, 2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]}, name="dem")
    >>> steps = pd.DataFrame({"x1_": [0.5], "y1_": [3.5], "x2_": [3.5], "y2_": [3.5]})
    >>> tf.move.along_summary(steps, dem, ("mean", "ascent")).iloc[0].tolist()[-2:]
    [1.5, 3.0]
    """
    stats = (stats,) if isinstance(stats, str) else tuple(stats)
    bad = [s for s in stats if s not in STATS]
    if bad:
        raise ValueError(f"stats: {', '.join(STATS)}; not {', '.join(map(repr, bad))}")
    layers, offsets, cell, rank = _walk(steps, covariates, n_threads)
    out = steps.copy()
    out.attrs = dict(steps.attrs)
    threads = resolve(n_threads)
    for name, _, v in layers:
        table = _core.move_along_stats(offsets, cell, rank, np.ascontiguousarray(v.ravel()), threads)
        for s in stats:
            out[f"{name}_along_{s}"] = table[:, STATS.index(s)]
    return out
