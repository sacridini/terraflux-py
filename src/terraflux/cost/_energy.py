# SPDX-License-Identifier: GPL-3.0-or-later
"""Energy landscapes: the cost of walking over a DEM (R enerscape 1.2.0)."""

from __future__ import annotations

import numpy as np

from .. import _core
from .._threads import resolve
from ._transition import _crs_lonlat, read_grid_values

_UNITS = ("joule", "kcal")
_MODELS = ("animal", "human")
_OUTPUTS = ("energy", "conductance", "both")


def _wrap(values: np.ndarray, template, name: str):
    if template is None:
        return values
    import xarray as xr

    out = xr.DataArray(values, coords={k: template.coords[k] for k in template.dims}, dims=template.dims, name=name,
                       attrs={"long_name": name})
    crs = template.rio.crs if hasattr(template, "rio") else None
    if crs is not None:
        from ..io import _with_georef

        out = _with_georef(out, crs=crs, transform=template.rio.transform())
    return out


def energy_landscape(dem, mass: float, *, unit: str = "joule", neigh: int = 8, model: str = "animal",
                     v: float | None = None, output: str = "energy", n_threads: int | None = None):
    """The energy cost of walking over a terrain (enerscape's ``enerscape`` and ``humanscape``).

    The cost of getting onto each cell, averaged over its ``neigh`` neighbours: for each one, the energy of walking
    from it onto the cell, over the distance in three dimensions and the slope between their heights. It is a
    resistance surface (log it to plot it; it is skewed to the right) for least-cost paths and circuit theory.
    Bit for bit with enerscape 1.2.0 (Berti et al. 2022) on R, the cells' means in 80-bit arithmetic as Rcpp's.

    ``model="animal"`` is Pontzer's (2016) allometric cost of locomotion, as corrected by enerscape 1.1.0: the cost per
    metre is ``(8 m^-0.34 + 50 (1 + sin((2 s - 74) / 180 pi)) m^-0.12) m`` joules for a body mass ``m`` (kg) and a slope
    ``s`` in degrees, lowest on a gentle descent (-8 degrees). ``model="human"`` is Looney et al.'s (2019) equation of
    the energy of walking at the speed ``v`` (m/s), level, uphill and downhill (enerscape's ``humanscape``).

    The border cells (one cell wide), the cells with a missing height in their neighbourhood (the 3 x 3 cells for
    ``neigh=8``, the cross for 4) and costs of exactly zero are NaN, as in R. Where R is silent, this function raises:
    ``neigh`` other than 4 or 8 (R returns an all-NA raster), a mass or a speed that is not positive, a CRS in degrees
    (R only says it assumes metres), cells that are not square.

    Parameters
    ----------
    dem
        Elevation in metres: a path, a ``DataArray`` or an array. The CRS must be projected in metres (the cell size
        is taken as metres when there is none); the cells must be square (the sizes in x and y differ by 0.01 or less).
    mass
        Body mass (kg), above zero.
    unit
        ``"joule"`` or ``"kcal"`` (1 kcal = 4184 J).
    neigh
        Neighbours of a cell, 4 (the cross) or 8.
    model
        ``"animal"`` (Pontzer 2016) or ``"human"`` (Looney et al. 2019).
    v
        Walking speed (m/s) of the human model (enerscape's default, 1.39); not for the animal one.
    output
        ``"energy"``; ``"conductance"`` (1 / energy, NaN where the energy is NaN or 0: an extension to enerscape);
        ``"both"``: a ``Dataset`` of both for a ``DataArray``, else a tuple (energy, conductance).
    n_threads
        Threads (default: :func:`terraflux.default_threads`); the result does not depend on them.

    Returns
    -------
    DataArray or ndarray
        The grid and CRS of ``dem``, in float64, named ``"energy"`` (or ``"conductance"``).

    References
    ----------
    Berti, E., Davoli, M., Buitenwerf, R., Dyer, A., Hansen, O. L. P., Hirt, M., Svenning, J.-C., Terlau, J. F.,
    Brose, U. & Vollrath, F. (2022). The r package enerscape: a general energy landscape framework for terrestrial
    movement ecology. Methods in Ecology and Evolution 13, 60-67.

    Pontzer, H. (2016). A unified theory for the energy cost of legged locomotion. Biology Letters 12, 20150935.

    Looney, D. P., Santee, W. R., Hansen, E. O., Bonventre, P. J., Chalmers, C. R. & Potter, A. W. (2019). Estimating
    energy expenditure during level, uphill, and downhill walking. Medicine & Science in Sports & Exercise 51,
    1954-1960.

    Examples
    --------
    >>> import terraflux as tf
    >>> en = tf.cost.energy_landscape(dem, 50, unit="kcal")        # a 50 kg animal, in kcal
    >>> cost = tf.cost.energy_landscape(dem, 70, model="human", v=1.2)
    """
    if unit not in _UNITS:
        raise ValueError(f"unit must be one of {_UNITS}, got {unit!r}")
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    if output not in _OUTPUTS:
        raise ValueError(f"output must be one of {_OUTPUTS}, got {output!r}")
    if isinstance(neigh, bool) or neigh not in (4, 8):
        raise ValueError(f"neigh must be 4 or 8, got {neigh!r}")
    mass = float(mass)
    if not (mass > 0 and np.isfinite(mass)):
        raise ValueError(f"mass must be a positive number (kg), got {mass!r}")
    if model == "animal" and v is not None:
        raise ValueError("v (the walking speed) is for model='human'")
    speed = 1.39 if v is None else float(v)
    if not (speed > 0 and np.isfinite(speed)):
        raise ValueError(f"v must be a positive speed (m/s), got {v!r}")

    values, grid, template = read_grid_values(dem)
    if grid.lonlat or _crs_lonlat(grid.crs):
        raise ValueError("dem: the CRS is in degrees; project it to metres first (the energies need metres)")
    dx, dy = grid.res
    if not abs(dx - dy) <= 1e-2:
        raise ValueError(f"dem: the cells must be square (sizes {dx} x {dy}); resample it to square cells")
    en = _core.cost_energy_landscape(values, mass, float(dx), int(neigh), model == "human", speed, unit == "kcal",
                                     resolve(n_threads))
    if output == "energy":
        return _wrap(en, template, "energy")
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(en != 0, 1.0 / en, np.nan)
    if output == "conductance":
        return _wrap(cond, template, "conductance")
    e, c = _wrap(en, template, "energy"), _wrap(cond, template, "conductance")
    if template is None:
        return e, c
    import xarray as xr

    return xr.Dataset({"energy": e, "conductance": c})
