# SPDX-License-Identifier: GPL-3.0-or-later
"""Home ranges of tracks: minimum convex polygons, kernel density estimates and local convex hulls.

R's amt 0.3.1 (Signer, Fieberg & Avgar 2019), the same numbers: ``hr_mcp``, ``hr_kde`` with the bandwidths
``hr_kde_ref``, ``hr_kde_ref_scaled``, ``hr_kde_pi`` and ``hr_kde_lscv``, ``hr_locoh``, ``hr_isopleths``,
``hr_area``, ``hr_ud``, ``hr_cud``, ``hr_overlap``, ``intensity_use`` and ``make_trast``. The estimators are
computed per track (amt works on one track at a time): the result of :func:`hr_mcp`, :func:`hr_kde` and
:func:`hr_locoh` is a :class:`HomeRange` with one polygon per track and level, and for the kernel density
estimate a ``(track, y, x)`` DataArray with one band per track.

The numbers are R's: the kernel estimate is KernSmooth's ``bkde2D`` (linear binning and R's own fft, the
bandwidth ``dpik`` and the cross-validation of ``lscv`` included), the isopleth lines are R's ``contourLines``
(what ``terra::as.contour`` uses) and the polygons are GEOS's, as in ``sf``. Tracks in longitude / latitude must be
projected first (amt's areas of them come from ``s2`` on a plane computed in degrees; see :func:`aeqd_crs`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from ._clean import _points
from ._tracks import _TRACK_NAMES, _find, track_codes

_REFERENCE = ("Signer, J., Fieberg, J. & Avgar, T. (2019). Animal movement tools (amt): R package for managing "
              "tracking data and conducting habitat selection analyses. Ecology and Evolution 9, 880-890. "
              "doi:10.1002/ece3.4823")
_RESCALE = ("unitvar", "xvar", "none")
_OVERLAP = ("hr", "phr", "vi", "ba", "udoi", "hd")
_EXTENT = "terraflux_extent"   # the exact (xmin, xmax, ymin, ymax) of a template, in the attrs of its DataArray


# ---------------------------------------------------------------------------------------------- the tracks

@dataclass
class _Track:
    label: object
    x: np.ndarray
    y: np.ndarray


def _tracks_of(tracks, track: str | None, what: str) -> tuple[list[_Track], object, str]:
    """The locations of each track (in the order of their first appearance), the CRS and the track column."""
    x, y, crs, lonlat = _points(tracks, what)
    if lonlat:
        raise ValueError(f"{what}: the tracks are in longitude / latitude, where amt's areas come from s2 on a plane "
                         "computed in degrees: project them first (tracks.to_crs(tf.move.aeqd_crs(tracks)))")
    try:
        col = _find(tracks, track, _TRACK_NAMES, "track")
    except ValueError:
        col = None   # no track column: one track
    if col is None or len(x) == 0:
        return [_Track(1, x, y)], crs, "track"
    values = tracks[col].to_numpy()
    codes = track_codes(values)
    out = []
    for k in range(int(codes.max()) + 1 if len(codes) else 0):
        rows = np.flatnonzero(codes == k)
        out.append(_Track(values[rows[0]], np.ascontiguousarray(x[rows]), np.ascontiguousarray(y[rows])))
    return out, crs, col


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: a number")
    return float(value)


def _levels(levels, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    v = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if v.ndim != 1 or v.size < 1 or not np.all(np.isfinite(v)) or np.any(v < lower) or np.any(v > upper):
        raise ValueError(f"levels: numbers between {lower:g} and {upper:g}")
    return np.sort(v)


# ------------------------------------------------------------------------------------------- the template

def _grid_of(trast):
    """(xmin, xmax, ymin, ymax, ncol, nrow) of a template DataArray (north up)."""
    nrow, ncol = int(trast.sizes["y"]), int(trast.sizes["x"])
    ext = trast.attrs.get(_EXTENT)
    if ext is not None and len(ext) == 4:
        return float(ext[0]), float(ext[1]), float(ext[2]), float(ext[3]), ncol, nrow
    xs = np.asarray(trast["x"].values, np.float64)
    ys = np.asarray(trast["y"].values, np.float64)
    if ncol > 1:
        dx = (xs[-1] - xs[0]) / (ncol - 1)
    else:
        dx = abs(float(trast.rio.resolution()[0]))
    if nrow > 1:
        dy = (ys[0] - ys[-1]) / (nrow - 1)
    else:
        dy = abs(float(trast.rio.resolution()[1]))
    if dy <= 0 or dx <= 0:
        raise ValueError("the template raster must be north up: x increasing and y decreasing")
    return xs[0] - dx / 2, xs[-1] + dx / 2, ys[-1] - dy / 2, ys[0] + dy / 2, ncol, nrow


def _template(xmin: float, xmax: float, ymin: float, ymax: float, ncol: int, nrow: int, crs):
    import xarray as xr

    dx = (xmax - xmin) / ncol     # terra's xres, yres
    dy = (ymax - ymin) / nrow
    xs = xmin + (np.arange(ncol, dtype=np.float64) + 0.5) * dx    # xFromCol, yFromRow
    ys = ymax - (np.arange(nrow, dtype=np.float64) + 0.5) * dy
    da = xr.DataArray(np.zeros((nrow, ncol)), dims=("y", "x"), coords={"y": ys, "x": xs}, name="trast")
    if crs is not None:
        da = da.rio.write_crs(crs)
    da.attrs[_EXTENT] = [float(xmin), float(xmax), float(ymin), float(ymax)]
    return da


def _round_half_up(v: float) -> int:
    f = math.floor(v)
    return int(f) + (1 if v - f >= 0.5 else 0)


def _terra_rast(xmin, xmax, ymin, ymax, res: float):
    """terra::rast(extent, res = res): the columns and rows are the rounded ``(xmax - xmin) / res`` and
    ``(ymax - ymin) / res`` (at least 1), and the extent is moved to hold them from its lower left corner."""
    ncol = max(1, _round_half_up((xmax - xmin) / res))
    nrow = max(1, _round_half_up((ymax - ymin) / res))
    return xmin + ncol * res, ymin + nrow * res, ncol, nrow


def _extent_max(x: np.ndarray, y: np.ndarray) -> float:
    return float(max(np.max(x) - np.min(x), np.max(y) - np.min(y)))


def make_trast(tracks, factor: float = 1.5, res: float | None = None, *, track: str | None = None):
    """A template raster for the estimates of a track's utilisation distribution (amt's ``make_trast``).

    The bounding box of the locations grown by ``factor`` (1.5: half of the largest extent on every side),
    cut in square cells of ``res``: by default 1/100 of that largest extent (at least 1e-9).

    Parameters
    ----------
    tracks
        A GeoDataFrame of points: one track, or several (the box then holds all of them).
    factor
        At least 1.
    res
        Cell size in the CRS's units (at least 1e-10).

    Returns
    -------
    xarray.DataArray
        ``(y, x)``, zeros, north up, with the CRS of the tracks; the exact extent is kept in its attrs.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(30), tracks=1, seed=3)
    >>> r = tf.move.make_trast(tr)
    >>> max(r.shape)          # the larger side holds 200 cells: the box is twice the largest extent
    200
    """
    xs, ys = _all_xy(tracks)
    return _make_trast(xs, ys, tracks.crs, factor, res)


def _all_xy(tracks):
    x, y, _, _ = _points(tracks, "make_trast")
    if len(x) == 0:
        raise ValueError("tracks: no locations")
    return x, y


def _make_trast(x, y, crs, factor, res):
    factor = _number(factor, "factor")
    if factor < 1:
        raise ValueError("factor: a number of at least 1")
    me = _extent_max(x, y)
    if res is None:
        res = max(me / 100, 1e-09)
    res = _number(res, "res")
    if res < 1e-10:
        raise ValueError("res: a number of at least 1e-10")
    bu = me * factor - me
    xmin, xmax = float(np.min(x)) + bu * -1, float(np.max(x)) + bu * 1
    ymin, ymax = float(np.min(y)) + bu * -1, float(np.max(y)) + bu * 1
    xmax, ymax, ncol, nrow = _terra_rast(xmin, xmax, ymin, ymax, res)
    return _template(xmin, xmax, ymin, ymax, ncol, nrow, crs)


def _ring_area(ring) -> float:
    return _core.move_hr_ring_area(np.ascontiguousarray(np.asarray(ring.coords, dtype=np.float64)[:, :2]))


def _area(g) -> float:
    """``sf::st_area`` of a geometry in a projected CRS: GDAL's OGR sums (not GEOS's area)."""
    t = g.geom_type
    if t == "Polygon":
        if g.is_empty:
            return 0.0
        a = _ring_area(g.exterior)
        for r in g.interiors:
            a -= _ring_area(r)
        return a
    if t in ("MultiPolygon", "GeometryCollection"):
        s = 0.0
        for p in g.geoms:
            s += _area(p)
        return s
    return 0.0


# ------------------------------------------------------------------------------------------ the container

@dataclass
class HomeRange:
    """Home-range estimates of one or several tracks (amt's ``hr`` objects, per track).

    Made by :func:`hr_mcp`, :func:`hr_kde` and :func:`hr_locoh`; read with :func:`hr_isopleths`,
    :func:`hr_area`, :func:`hr_ud` and :func:`hr_cud`, compared with :func:`hr_overlap`.
    """
    estimator: str                       #: ``"mcp"``, ``"kde"`` or ``"locoh"``
    levels: np.ndarray                   #: the levels asked for, ascending
    crs: object                          #: the CRS of the tracks (None for none)
    tracks: list                         #: the track labels, in order
    polygons: object = None              #: GeoDataFrame (track, level, what, area, geometry): mcp and locoh
    ud: object = None                    #: DataArray (track, y, x): the kernel density estimate (kde), not normalised
    trast: object = None                 #: the template raster of the estimate (kde)
    h: pd.DataFrame | None = None        #: the bandwidth of each track (kde): columns hx, hy
    params: dict = field(default_factory=dict)
    data: object = None                  #: the tracks, when kept
    track_name: str = "track"
    _iso: object = field(default=None, repr=False)

    @property
    def probabilistic(self) -> bool:
        """True for the estimators with a utilisation distribution (``"kde"``)."""
        return self.ud is not None

    def __repr__(self) -> str:
        extra = f", h={self.h.iloc[0].tolist()}" if self.h is not None and len(self.h) == 1 else ""
        return (f"HomeRange({self.estimator!r}, tracks={self.tracks}, levels={self.levels.tolist()}{extra})")

    def sel(self, label) -> HomeRange:
        """The estimate of one track."""
        if label not in self.tracks:
            raise KeyError(f"track {label!r} not found")
        out = HomeRange(self.estimator, self.levels, self.crs, [label], params=dict(self.params),
                        track_name=self.track_name)
        if self.polygons is not None:
            out.polygons = self.polygons[self.polygons[self.track_name] == label].reset_index(drop=True)
        if self.ud is not None:
            out.ud = self.ud.sel(track=[label])
            out.trast = self.trast
            out.h = self.h.loc[[label]] if self.h is not None else None
        if self.data is not None:
            col = self.params.get("track_column")
            out.data = self.data[self.data[col] == label] if col in getattr(self.data, "columns", ()) else self.data
        return out


def _single_ud(ud, label=None) -> np.ndarray:
    """The 2-D values of a (track, y, x) or (y, x) DataArray."""
    if "track" in ud.dims:
        ud = ud.sel(track=label) if label is not None else ud.isel(track=0)
    return np.ascontiguousarray(ud.values, dtype=np.float64)


def _ud_stack(values: list[np.ndarray], labels, trast, crs, name: str):
    import xarray as xr

    da = xr.DataArray(np.stack(values), dims=("track", "y", "x"),
                      coords={"track": list(labels), "y": trast["y"].values, "x": trast["x"].values}, name=name)
    if crs is not None:
        da = da.rio.write_crs(crs)
    da.attrs[_EXTENT] = list(trast.attrs.get(_EXTENT, _grid_of(trast)[:4]))
    return da


# ---------------------------------------------------------------------------------------------------- MCP

def hr_mcp(tracks, levels=0.95, keep_data: bool = True, *, track: str | None = None) -> HomeRange:
    """Minimum convex polygons of each track (amt's ``hr_mcp``).

    The polygon is the convex hull (R's ``chull``) of the locations whose squared distance to the mean location
    is at most the quantile (type 7) of that level; ``levels=1`` is the hull of all of them.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points in a projected CRS (or without one), several tracks allowed.
    levels
        Numbers in [0, 1]: the share of locations in each polygon (default 0.95).
    keep_data
        Keep ``tracks`` in the result.
    track
        The track column, if not recorded (:func:`set_columns`).

    Returns
    -------
    HomeRange
        ``polygons`` has a row per track and level: ``track``, ``level``, ``what`` ("estimate"), ``area``,
        ``geometry`` (a Polygon).

    References
    ----------
    Worton, B. J. (1989). Kernel methods for estimating the utilization distribution in home-range studies.
    Ecology 70, 164-168. doi:10.2307/1938423

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(40), tracks=2, seed=1)
    >>> hr = tf.move.hr_mcp(tr, levels=[0.5, 0.95])
    >>> tf.move.hr_area(hr).shape
    (4, 4)
    """
    levels = _levels(levels)
    items, crs, col = _tracks_of(tracks, track, "hr_mcp")
    rows = []
    for t in items:
        for lv, geom in zip(levels, _mcp_polygons(t.x, t.y, levels)):
            rows.append((t.label, lv, "estimate", _area(geom), geom))
    out = HomeRange("mcp", levels, crs, [t.label for t in items], _polygon_frame(rows, crs, col), params={},
                    data=tracks if keep_data else None, track_name=col)
    out.params["track_column"] = col
    return out


def _polygon_frame(rows, crs, track_name):
    import geopandas as gpd

    df = pd.DataFrame(rows, columns=[track_name, "level", "what", "area", "geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)


def _mcp_polygons(x: np.ndarray, y: np.ndarray, levels: np.ndarray):
    from shapely import Polygon

    mx, my = _core.move_hr_colmean(x), _core.move_hr_colmean(y)
    dx, dy = x - mx, y - my
    sqd = dx * dx + dy * dy
    qts = _core.move_hr_quantile7(sqd, levels)
    out = []
    for q in qts:
        sel = np.flatnonzero(sqd <= q)
        hull = sel[_core.move_hr_chull(np.ascontiguousarray(x[sel]), np.ascontiguousarray(y[sel]))]
        ring = np.column_stack([x[hull], y[hull]])
        ring = np.vstack([ring, ring[:1]])
        while len(ring) < 4:      # one or two vertices (too few locations, or all on a line): R's polygon has no area
            ring = np.vstack([ring, ring[-1:]])
        out.append(Polygon(ring))
    return out


# ---------------------------------------------------------------------------------------------- bandwidths

def _prep_xy(x, y, rescale: str, what: str):
    if rescale not in _RESCALE:
        raise ValueError(f"{what}: rescale: not one of unitvar, xvar or none")
    xs, ys = x, y
    if rescale == "unitvar":
        xs = xs / _sd(xs)
        ys = ys / _sd(ys)
    elif rescale == "xvar":
        ys = (ys / _sd(ys)) * _sd(xs)
    return xs, ys


def _sd(v) -> float:
    return math.sqrt(_core.move_hr_var(np.ascontiguousarray(v, np.float64)))


def _ref_one(t: _Track, rescale: str) -> tuple[float, float]:
    xs, ys = _prep_xy(t.x, t.y, rescale, "hr_kde_ref")
    n = len(t.x)
    vx = _core.move_hr_var(np.ascontiguousarray(xs, np.float64))
    vy = _core.move_hr_var(np.ascontiguousarray(ys, np.float64))
    h = math.sqrt(0.5 * (vx + vy)) * _core.move_hr_pow(float(n), -1 / 6)
    if rescale == "unitvar":
        return h * _sd(t.x), h * _sd(t.y)
    return h, h


def _bandwidths(rows: dict, tracks_col: str) -> pd.DataFrame:
    return pd.DataFrame(rows, index=["hx", "hy"]).T.rename_axis(tracks_col)


def hr_kde_ref(tracks, rescale: str = "none", *, track: str | None = None) -> pd.DataFrame:
    """The reference bandwidth of each track (amt's ``hr_kde_ref``): ``sqrt((var(x) + var(y)) / 2) * n^(-1/6)``.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, several tracks allowed.
    rescale
        ``"none"`` (default); ``"unitvar"`` rescales each coordinate to unit variance first (the bandwidth is
        returned in the original units); ``"xvar"`` rescales y to the variance of x.

    Returns
    -------
    pandas.DataFrame
        Indexed by track: ``hx`` and ``hy`` (equal here).

    References
    ----------
    Worton, B. J. (1989). Ecology 70, 164-168. doi:10.2307/1938423

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(40), tracks=2, seed=1)
    >>> tf.move.hr_kde_ref(tr).shape
    (2, 2)
    """
    items, _, col = _tracks_of(tracks, track, "hr_kde_ref")
    return _bandwidths({t.label: _ref_one(t, rescale) for t in items}, col)


def hr_kde_pi(tracks, rescale: str = "none", correct: bool = True, *, scalest: str = "minim", level: int = 2,
              kernel: str = "normal", canonical: bool = False, gridsize: int = 401, range_x=None,
              truncate: bool = True, track: str | None = None) -> pd.DataFrame:
    """The plug-in bandwidth of each track (amt's ``hr_kde_pi``): KernSmooth's ``dpik`` for x and y.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, several tracks allowed.
    rescale
        As in :func:`hr_kde_ref`.
    correct
        Raise the bandwidths to the power 5/6, as amt does by default (the plug-in is for a univariate density).
    scalest, level, kernel, canonical, gridsize, range_x, truncate
        The arguments of ``KernSmooth::dpik`` (``scalest`` ``"minim"``, ``"stdev"`` or ``"iqr"``; ``level`` 0
        to 5; ``range_x`` is its ``range.x``, by default the range of each coordinate).

    Returns
    -------
    pandas.DataFrame
        Indexed by track: ``hx`` and ``hy``.

    References
    ----------
    Wand, M. P. & Jones, M. C. (1995). Kernel Smoothing. Chapman & Hall; Sheather, S. J. & Jones, M. C. (1991).
    J. R. Stat. Soc. B 53, 683-690.

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> tf.move.hr_kde_pi(tr).shape
    (1, 2)
    """
    items, _, col = _tracks_of(tracks, track, "hr_kde_pi")
    out = {}
    for t in items:
        xs, ys = _prep_xy(t.x, t.y, rescale, "hr_kde_pi")
        hx = _dpik(xs, scalest, level, kernel, canonical, gridsize, range_x, truncate)
        hy = _dpik(ys, scalest, level, kernel, canonical, gridsize, range_x, truncate)
        h = np.array([hx, hy])
        if rescale == "unitvar":
            h = np.array([hx, hx]) * np.array([_sd(t.x), _sd(t.y)])
        if correct:
            h = np.array([_core.move_hr_pow(float(v), 5 / 6) for v in h])
        out[t.label] = (float(h[0]), float(h[1]))
    return _bandwidths(out, col)


_KERNELS = ("normal", "box", "epanech", "biweight", "triweight")
_SCALEST = ("minim", "stdev", "iqr")


def _dpik(x, scalest, level, kernel, canonical, gridsize, range_x, truncate) -> float:
    if scalest not in _SCALEST:
        raise ValueError("scalest: one of minim, stdev, iqr")
    if kernel not in _KERNELS:
        raise ValueError("kernel: one of " + ", ".join(_KERNELS))
    x = np.ascontiguousarray(x, np.float64)
    a, b = (float(np.min(x)), float(np.max(x))) if range_x is None else (float(range_x[0]), float(range_x[1]))
    return _core.move_hr_dpik(x, a, b, _SCALEST.index(scalest), int(level), _KERNELS.index(kernel), bool(canonical),
                              int(gridsize), bool(truncate))


@dataclass
class LscvFit:
    """The cross-validation of one track (:func:`hr_kde_lscv`)."""
    h: tuple                    #: the chosen bandwidth (hx, hy)
    converged: bool             #: False when the choice is at the end of the range
    res: np.ndarray             #: the criterion at each bandwidth of ``range``
    which_min: str              #: ``"global"`` or ``"local"``
    rescale: str
    range: np.ndarray           #: the bandwidths tried


@dataclass
class Lscv:
    """The result of :func:`hr_kde_lscv` for each track."""
    h: pd.DataFrame             #: indexed by track: ``hx``, ``hy``
    converged: pd.Series        #: by track
    fits: dict                  #: track -> :class:`LscvFit`


def _local_minima(x: np.ndarray) -> np.ndarray:
    """amt's ``local_minima`` (1-based positions)."""
    y = np.diff(np.r_[2147483647.0, x]) < 0
    change = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
    ends = np.r_[change[1:], len(y)]                 # the end of each run, 1-based
    runs = ends[::2]
    if x[0] == x[1]:
        runs = runs[1:]
    return runs


def hr_kde_lscv(tracks, range=None, which_min: str = "global", rescale: str = "none", *,
                track: str | None = None, n_threads: int | None = None) -> Lscv:
    """Least-squares cross-validation of the bandwidth of each track (amt's ``hr_kde_lscv``).

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, several tracks allowed.
    range
        The bandwidths to try (default: 100 values from 0.1 to 2 times :func:`hr_kde_ref` of each track).
    which_min
        ``"global"`` (the bandwidth with the lowest criterion) or ``"local"`` (the largest local minimum).
    rescale
        As in :func:`hr_kde_ref`.
    n_threads
        Threads (default: most of the cores); the result does not depend on it.

    Returns
    -------
    Lscv
        ``h`` (a DataFrame by track), ``converged`` and ``fits`` (the criterion and range of each track). A
        warning when the choice is at an end of the range.

    References
    ----------
    Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis. Chapman & Hall.

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(50), tracks=1, seed=2)
    >>> fit = tf.move.hr_kde_lscv(tr)
    >>> fit.h.shape
    (1, 2)
    """
    if rescale not in _RESCALE:
        raise ValueError("rescale: not one of unitvar, xvar or none")
    if which_min not in ("global", "local"):
        raise ValueError('which_min: "global" or "local"')
    threads = resolve(n_threads)
    items, _, col = _tracks_of(tracks, track, "hr_kde_lscv")
    hs, conv, fits = {}, {}, {}
    for t in items:
        xs, ys = _prep_xy(t.x, t.y, rescale, "hr_kde_lscv")
        if range is None:
            href = _ref_one(t, "none")[0]
            rng = _core.move_hr_seq(href * 0.1, href * 2, 100)
        else:
            rng = np.ascontiguousarray(np.asarray(range, dtype=np.float64))
        res = _core.move_hr_lscv(np.ascontiguousarray(xs, np.float64), np.ascontiguousarray(ys, np.float64), rng,
                                 threads)
        if which_min == "global":
            h = rng[int(np.nanargmin(res))]
        else:
            minima = _local_minima(res)
            if len(minima) == 0:
                raise ValueError("hr_kde_lscv: the criterion has no local minimum")
            h = rng[int(minima.max()) - 1]
        converged = True
        if rng[0] == h or rng[-1] == h:
            converged = False
            warnings.warn("hr_kde_lscv: lscv did not converge.", UserWarning, stacklevel=2)
        hh = (float(h * _sd(t.x)), float(h * _sd(t.y))) if rescale == "unitvar" else (float(h), float(h))
        hs[t.label], conv[t.label] = hh, converged
        fits[t.label] = LscvFit(hh, converged, res, which_min, rescale, rng)
    return Lscv(_bandwidths(hs, col), pd.Series(conv, name="converged").rename_axis(col), fits)


# ---------------------------------------------------------------------------------------------------- KDE

def _bandwidth_of(h, items, col) -> dict:
    """The (hx, hy) of each track from ``h`` (None: the reference bandwidth)."""
    if h is None:
        return {t.label: _ref_one(t, "none") for t in items}
    if isinstance(h, pd.DataFrame):
        cols = ["hx", "hy"] if {"hx", "hy"} <= set(h.columns) else list(h.columns[:2])
        return {t.label: (float(h.loc[t.label, cols[0]]), float(h.loc[t.label, cols[1]])) for t in items}
    if isinstance(h, dict):
        return {t.label: _one_h(h[t.label]) for t in items}
    one = _one_h(h)
    return {t.label: one for t in items}


def _one_h(h) -> tuple:
    if isinstance(h, str) or not np.all([isinstance(v, (int, float, np.integer, np.floating)) for v in np.atleast_1d(h)]):
        raise TypeError("hr_kde: bandwidth should be numeric")
    v = np.atleast_1d(np.asarray(h, dtype=np.float64))
    if v.size > 2:
        warnings.warn("hr_kde: h only first 2 elements used", UserWarning, stacklevel=4)
        v = v[:2]
    elif v.size < 2:
        warnings.warn("hr_kde: same bandwidth is used in x and y direction", UserWarning, stacklevel=4)
        v = np.repeat(v, 2)
    return float(v[0]), float(v[1])


def _kde_values(x, y, h, grid) -> np.ndarray:
    xmin, xmax, ymin, ymax, ncol, nrow = grid
    if not (h[0] > 0 and h[1] > 0):
        raise ValueError("'bandwidth' must be strictly positive")
    fhat, warn = _core.move_hr_bkde2d(x, y, h[0], h[1], xmin, xmax, ymin, ymax, ncol, nrow)
    if warn:
        warnings.warn("Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'",
                      UserWarning, stacklevel=3)
    return np.ascontiguousarray(fhat[:, ::-1].T)    # kde$fhat[, ncol(kde$fhat):1], the rows from the top


def hr_kde(tracks, h=None, trast=None, levels=0.95, keep_data: bool = True, *, track: str | None = None) -> HomeRange:
    """Kernel density estimates of each track (amt's ``hr_kde``: KernSmooth's ``bkde2D``, a bivariate normal kernel).

    Parameters
    ----------
    tracks
        A GeoDataFrame of points in a projected CRS (or without one), several tracks allowed.
    h
        The bandwidth: a number, a pair (x, y), a dict or DataFrame by track (:func:`hr_kde_pi`,
        :func:`hr_kde_lscv`...); by default :func:`hr_kde_ref` of each track.
    trast
        The template raster (:func:`make_trast`): its cells are the estimate's cells. By default one made from
        the locations of all the tracks (for a single track, amt's default).
    levels
        Numbers in [0, 1] for the isopleths (:func:`hr_isopleths`, :func:`hr_area`).
    keep_data
        Keep ``tracks`` in the result.
    track
        The track column, if not recorded (:func:`set_columns`).

    Returns
    -------
    HomeRange
        ``ud`` is a ``(track, y, x)`` DataArray with the estimate as KernSmooth gives it (amt's ``x$ud``;
        :func:`hr_ud` normalises it to sum to 1), ``h`` the bandwidths, ``trast`` the template. The estimate is
        evaluated at ``xmin + i * (xmax - xmin) / (ncol - 1)``, the grid of ``bkde2D`` laid over the template's
        edges, as amt does, not at the cell centres.

    References
    ----------
    Worton, B. J. (1989). Ecology 70, 164-168. doi:10.2307/1938423

    Wand, M. P. & Jones, M. C. (1995). Kernel Smoothing. Chapman & Hall.

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> hr = tf.move.hr_kde(tr, levels=[0.5, 0.95])
    >>> ud = tf.move.hr_ud(hr)
    >>> bool(abs(float(ud.sum()) - 1) < 1e-12)
    True
    """
    levels = _levels(levels)
    items, crs, col = _tracks_of(tracks, track, "hr_kde")
    bw = _bandwidth_of(h, items, col)
    if trast is None:
        allx = np.concatenate([t.x for t in items])
        ally = np.concatenate([t.y for t in items])
        trast = _make_trast(allx, ally, crs, 1.5, None)
    grid = _grid_of(trast)
    values = [_kde_values(t.x, t.y, bw[t.label], grid) for t in items]
    labels = [t.label for t in items]
    ud = _ud_stack(values, labels, trast, crs, "ud")
    hdf = _bandwidths(bw, col)
    out = HomeRange("kde", levels, crs, labels, ud=ud, trast=trast, h=hdf, data=tracks if keep_data else None,
                    track_name=col)
    out.params["track_column"] = col
    return out


# ---------------------------------------------------------------------------------------------- isopleths

def _grid_cell(grid):
    xmin, xmax, ymin, ymax, ncol, nrow = grid
    return xmin, ymax, (xmax - xmin) / ncol, (ymax - ymin) / nrow


def _isopleth_geometries(cud2d: np.ndarray, grid, levels: np.ndarray):
    """The union of the contour lines of each level as a MultiPolygon (amt's ``hr_isopleths`` of a raster)."""
    import shapely
    from shapely import MultiPolygon, Polygon

    xmin, ymax, dx, dy = _grid_cell(grid)
    if cud2d.size > 100000:
        warnings.warn("the template has more than 1e5 cells: amt's isopleths (terra::as.contour, maxcells = 1e5) are "
                      "those of a regular sample of 1e5 cells; here the contours are those of the whole grid",
                      UserWarning, stacklevel=3)
    out = []
    for lv in levels:
        lines = _core.move_hr_contour_lines(cud2d, xmin, ymax, dx, dy, float(lv))
        if lines is None:
            raise ValueError("hr_isopleths: all the values of the raster are equal: no isopleths")
        if len(lines) == 0:
            raise ValueError(f"hr_isopleths: no contour line at level {lv:g}")
        polys = []
        for ln in lines:
            if len(ln) < 4 or not np.array_equal(ln[0], ln[-1]):
                raise ValueError(f"hr_isopleths: the contour at level {lv:g} reaches the edge of the raster (an open "
                                 "line is not a polygon): use a larger template raster (make_trast(factor=...))")
            polys.append(Polygon(ln))
        u = shapely.union_all(polys)
        out.append(u if isinstance(u, MultiPolygon) else MultiPolygon([u]) if u.geom_type == "Polygon" else u)
    return out


def _cud2d(values2d: np.ndarray) -> np.ndarray:
    flat = np.ascontiguousarray(values2d.reshape(-1))
    return _core.move_hr_cud(flat).reshape(values2d.shape)


def _kde_isopleths(hr: HomeRange) -> object:
    if hr._iso is not None:
        return hr._iso
    grid = _grid_of(hr.trast)
    rows = []
    for k, label in enumerate(hr.tracks):
        geoms = _isopleth_geometries(_cud2d(hr.ud.values[k]), grid, hr.levels)
        for lv, g in zip(hr.levels, geoms):
            rows.append((label, lv, "estimate", _area(g), g))
    hr._iso = _polygon_frame(rows, hr.crs, hr.track_name)
    return hr._iso


def hr_isopleths(x, levels=None, descending: bool = True):
    """The isopleths of home-range estimates as polygons (amt's ``hr_isopleths``).

    For the minimum convex polygons and LoCoH hulls these are the polygons made; for a kernel density estimate
    they are the contour lines (R's ``contourLines``, as ``terra::as.contour``) of its cumulative distribution
    (:func:`hr_cud`) at each level, each line closed into a polygon and all united (so a hole of one level is
    filled, as in amt).

    Parameters
    ----------
    x
        A :class:`HomeRange`, or a raster DataArray ``(y, x)`` / ``(track, y, x)`` with a distribution (it is
        normalised to sum to 1).
    levels
        For a raster: the levels (numbers in [0, 1]). Ignored for a :class:`HomeRange`, which has its own.
    descending
        Order the rows by level, the highest first (amt's default).

    Returns
    -------
    geopandas.GeoDataFrame
        ``track``, ``level``, ``what`` ("estimate"), ``area`` (CRS units squared), ``geometry``. For a raster
        without tracks, no ``track`` column.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> iso = tf.move.hr_isopleths(tf.move.hr_kde(tr, levels=[0.5, 0.95]))
    >>> iso["level"].tolist()
    [0.95, 0.5]
    """
    if isinstance(x, HomeRange):
        df = _kde_isopleths(x) if x.estimator == "kde" else x.polygons
        track_name = x.track_name
        df = df.copy()
    else:
        if levels is None:
            raise ValueError("levels: the levels of the isopleths")
        lv = _levels(levels)
        grid = _grid_of(x)
        crs = x.rio.crs
        has_track = "track" in x.dims
        labels = list(x["track"].values) if has_track else [None]
        rows = []
        for k, label in enumerate(labels):
            geoms = _isopleth_geometries(_cud2d(_single_ud(x, label if has_track else None)), grid, lv)
            for v, g in zip(lv, geoms):
                rows.append((label, v, "estimate", _area(g), g))
        df = _polygon_frame(rows, crs, "track")
        if not has_track:
            df = df.drop(columns="track")
        track_name = "track"
    keys = [c for c in (track_name,) if c in df.columns]
    if keys:
        order = np.lexsort((-df["level"].to_numpy() if descending else df["level"].to_numpy(),
                            pd.factorize(df[track_name])[0]))
    else:
        order = np.argsort(-df["level"].to_numpy() if descending else df["level"].to_numpy(), kind="stable")
    return df.iloc[order].reset_index(drop=True)


def hr_area(x, units: bool = False, level: float = 0.95):
    """The areas of the isopleths (amt's ``hr_area``).

    Parameters
    ----------
    x
        A :class:`HomeRange` (a table by track and level) or a raster DataArray (the area at ``level``, a number
        or a Series by track).
    units
        Add the column ``unit``: the CRS's area unit (``"m^2"``...).
    level
        For a raster, the level.

    Returns
    -------
    pandas.DataFrame or float
        ``track``, ``level``, ``what``, ``area`` for a :class:`HomeRange`.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(40), tracks=1, seed=1)
    >>> list(tf.move.hr_area(tf.move.hr_mcp(tr)).columns)
    ['track', 'level', 'what', 'area']
    """
    if isinstance(x, HomeRange):
        iso = hr_isopleths(x)
        out = pd.DataFrame(iso.drop(columns="geometry"))
        if units:
            out["unit"] = _area_unit(x.crs)
        return out
    iso = hr_isopleths(x, level)
    if "track" in iso.columns:
        return iso.set_index("track")["area"]
    return float(iso["area"].iloc[0])


def _area_unit(crs) -> str:
    if crs is None:
        return ""
    try:
        name = crs.axis_info[0].unit_name
    except (IndexError, AttributeError):
        return ""
    return {"metre": "m^2", "meter": "m^2", "kilometre": "km^2"}.get(name, f"{name}^2")


def hr_ud(x):
    """The utilisation distribution of a kernel density estimate, summing to 1 (amt's ``hr_ud``).

    Parameters
    ----------
    x
        A :class:`HomeRange` with a distribution (:func:`hr_kde`), or a raster DataArray.

    Returns
    -------
    xarray.DataArray
        ``(track, y, x)`` (or as the input), each band the cells divided by their sum.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> ud = tf.move.hr_ud(tf.move.hr_kde(tr))
    >>> bool(abs(float(ud.sum()) - 1) < 1e-12)
    True
    """
    da = _ud_of(x)
    out = da.copy(deep=True)
    flat = out.values.reshape(-1, out.sizes["y"] * out.sizes["x"]) if "track" in out.dims else \
        out.values.reshape(1, -1)
    for k in range(flat.shape[0]):
        total = _core.move_r_sum(np.ascontiguousarray(flat[k]))
        flat[k] = flat[k] / total
    out.values = flat.reshape(out.shape)
    return out


def hr_cud(x):
    """The cumulative distribution of a utilisation distribution (amt's ``hr_cud``).

    Each cell holds the share of the distribution in the cells at least as dense as it is: 0.95 bounds the
    smallest area holding 95 % of the use.

    Parameters
    ----------
    x
        A :class:`HomeRange` with a distribution, or a raster DataArray (normalised to sum to 1 first).

    Returns
    -------
    xarray.DataArray
        As :func:`hr_ud`.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> cud = tf.move.hr_cud(tf.move.hr_kde(tr))
    >>> bool(cud.max() <= 1 + 1e-12)
    True
    """
    da = _ud_of(x)
    out = da.copy(deep=True)
    vals = out.values.reshape(-1, out.sizes["y"] * out.sizes["x"]) if "track" in out.dims else out.values.reshape(1, -1)
    for k in range(vals.shape[0]):
        vals[k] = _core.move_hr_cud(np.ascontiguousarray(vals[k]))
    out.values = vals.reshape(out.shape)
    return out


def _ud_of(x):
    if isinstance(x, HomeRange):
        if x.ud is None:
            raise ValueError(f"the {x.estimator} estimate has no utilisation distribution")
        return x.ud
    if not hasattr(x, "dims") or "y" not in x.dims or "x" not in x.dims:
        raise TypeError("x: a HomeRange or a raster DataArray with y and x")
    return x


# ----------------------------------------------------------------------------------------------- LoCoH

_LOCOH = ("k", "a", "r")


def hr_locoh(tracks, n=10, type: str = "k", levels=0.95, keep_data: bool = True, rand_buffer: float = 1e-5, *,
             track: str | None = None, n_threads: int | None = None) -> HomeRange:
    """Local convex hulls of each track (amt's ``hr_locoh``; Getz et al. 2007).

    A hull is made around each location and its neighbours, buffered by ``rand_buffer``; the hulls are taken
    from the smallest and united until they hold the share of the locations asked for.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points in a projected CRS (or without one), several tracks allowed.
    n
        ``type="k"``: the number of nearest neighbours (default 10); ``"r"``: the radius; ``"a"``: the sum of
        distances to the neighbours, nearest first.
    type
        ``"k"``, ``"a"`` or ``"r"``.
    levels
        Numbers in [0, 1]: the share of locations covered. As in amt, the level of each polygon is the share
        actually reached, rounded to 2 digits (the nearest to the one asked for).
    keep_data
        Keep ``tracks`` in the result.
    rand_buffer
        Buffer of the hulls (the CRS's units), so that a hull of collinear points has an area.
    track
        The track column, if not recorded (:func:`set_columns`).
    n_threads
        Threads (default: most of the cores); the result does not depend on it.

    Returns
    -------
    HomeRange
        ``polygons`` with ``track``, ``level``, ``what``, ``area``, ``geometry``.

    References
    ----------
    Getz, W. M., Fortmann-Roe, S., Cross, P. C., Lyons, A. J., Ryan, S. J., Wilmers, C. C. (2007). LoCoH:
    nonparameteric kernel methods for constructing home ranges and utilization distributions. PLoS ONE 2, e207.
    doi:10.1371/journal.pone.0000207

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> hr = tf.move.hr_locoh(tr, n=8, levels=[0.5, 0.9])
    >>> len(tf.move.hr_isopleths(hr))
    2
    """
    if type not in _LOCOH:
        raise ValueError("hr_loch: incorrect type")
    levels = _levels(levels)
    n = _number(n, "n")
    threads = resolve(n_threads)
    items, crs, col = _tracks_of(tracks, track, "hr_locoh")
    rows = []
    for t in items:
        if type == "k" and n > len(t.x):
            raise ValueError("hr_locoh, type k, n > number of points")
        if type == "k" and n >= len(t.x):
            raise ValueError("hr_locoh, type k: n must be less than the number of points (FNN's k nearest "
                             "neighbours of a point exclude itself)")
        for lv, g, area in _locoh_polygons(t.x, t.y, n, type, levels, rand_buffer, threads):
            rows.append((t.label, lv, "estimate", area, g))
    out = HomeRange("locoh", levels, crs, [t.label for t in items], _polygon_frame(rows, crs, col),
                    params={"n": n, "type": type, "rand_buffer": rand_buffer, "track_column": col},
                    data=tracks if keep_data else None, track_name=col)
    return out


def _r_round(x: float, dig: int = 2) -> float:
    """R's ``round(x, dig)`` (4.0.0 and later: the nearest of the two candidates, a tie to the even one)."""
    if not math.isfinite(x) or x == 0:
        return x
    sgn = -1.0 if x < 0 else 1.0
    x = abs(x)
    l10x = 0.301029995663981195213738894724 * (0.5 + math.floor(math.log2(x)))
    if l10x + dig > 15:
        return sgn * x
    p10 = 10.0 ** dig
    x10 = p10 * x
    i10 = math.floor(x10)
    xd = i10 / p10
    xu = math.ceil(x10) / p10
    du, dd = xu - x, x - xd
    return sgn * (xd if (dd < du or (dd == du and i10 % 2 == 0)) else xu)


def _locoh_polygons(x, y, n, type, levels, rand_buffer, threads):
    import shapely
    from shapely import MultiPoint

    m = len(x)
    off, ids = _core.move_hr_locoh_neighbours(x, y, _LOCOH_CODE[type], float(n), threads)
    multi = []
    members = []
    for i in range(m):
        idx = np.concatenate([[i], ids[off[i]:off[i + 1]]]).astype(np.int64)
        members.append(idx)
        multi.append(MultiPoint(np.column_stack([x[idx], y[idx]])))
    hulls = shapely.convex_hull(np.array(multi, dtype=object))
    buffered = shapely.buffer(hulls, rand_buffer, quad_segs=30)
    areas = np.array([_area(g) for g in buffered])
    order = np.argsort(areas, kind="stable")
    # the first polygon (in the order of the areas) that holds each location
    seen = np.zeros(m, dtype=bool)
    first = []
    for rank, i in enumerate(order, start=1):
        for p in members[i]:
            if not seen[p]:
                seen[p] = True
                first.append(rank)
    id_poly = np.array(first, dtype=np.int64)                 # non-decreasing
    all_polys = np.arange(1, m + 1)
    pp = np.searchsorted(id_poly, all_polys, side="right") / m
    out = []
    for lv in levels:
        w = int(np.argmin(np.abs(pp - lv))) + 1
        g = shapely.union_all(buffered[order[:w]])
        out.append((_r_round(float(pp[w - 1]), 2), g, _area(g)))
    return out


_LOCOH_CODE = {"k": 0, "r": 1, "a": 2}


# ------------------------------------------------------------------------------------------------ overlap

def _units(x) -> list[tuple[object, HomeRange]]:
    """(label, single-track estimate) of what ``hr_overlap`` is given: a HomeRange, a list or a dict of them."""
    if isinstance(x, HomeRange):
        return [(t, x.sel(t)) for t in x.tracks]
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            for t in v.tracks:
                out.append((k if len(v.tracks) == 1 else (k, t), v.sel(t)))
        return out
    out = []
    for k, v in enumerate(x, start=1):
        if not isinstance(v, HomeRange):
            raise TypeError("not all x are home-range estimates")
        for t in v.tracks:
            out.append((k if len(v.tracks) == 1 else (k, t), v.sel(t)))
    return out


def _base_ov(gx, gy) -> float:
    if gx.intersects(gy):
        return float(_area(gx.intersection(gy)) / _area(gx))
    return 0.0


def _hr_base(x: HomeRange, y: HomeRange) -> pd.DataFrame:
    ix = hr_isopleths(x)
    iy = hr_isopleths(y)
    if len(ix) != len(iy):
        raise ValueError("Not all levels of x in y.")
    vals = [_base_ov(ix.geometry.iloc[i], iy.geometry.iloc[i]) for i in range(len(ix))]
    return pd.DataFrame({"level": ix["level"].to_numpy(), "overlap": vals})


def _cond_ud(x: HomeRange, level: float) -> np.ndarray:
    cud = hr_cud(x).values[0]
    ud = hr_ud(x).values[0].copy()
    ud[cud > level] = 0
    return (ud / level).reshape(-1)


def _pair(a: HomeRange, b: HomeRange, type: str, conditional: bool) -> pd.DataFrame:
    if type == "hr":
        return _hr_base(a, b)
    if not (a.probabilistic and b.probabilistic):
        raise ValueError("Both x and y should be probabilistic estimators.")
    ga, gb = _grid_of(a.trast), _grid_of(b.trast)
    same = (ga[4:] == gb[4:] and np.allclose(ga[:4], gb[:4], rtol=1.5e-8, atol=0))
    if not same:
        raise ValueError("Extend and resolution differ, make sure to use a template raster.")
    code = _OVERLAP.index(type) - 1
    if conditional:
        if not np.array_equal(a.levels, b.levels):
            raise ValueError("x and y should have the same level for conditional hr_overlap")
        vals = [_core.move_hr_vol_base(np.ascontiguousarray(_cond_ud(a, lv)), np.ascontiguousarray(_cond_ud(b, lv)),
                                       code) for lv in a.levels]
        return pd.DataFrame({"level": a.levels, "overlap": vals})
    ua, ub = hr_ud(a).values[0].reshape(-1), hr_ud(b).values[0].reshape(-1)
    return pd.DataFrame({"level": [1.0], "overlap": [_core.move_hr_vol_base(np.ascontiguousarray(ua),
                                                                             np.ascontiguousarray(ub), code)]})


def hr_overlap(x, y=None, type: str = "hr", conditional: bool = False, which: str = "consecutive", labels=None):
    """Overlap between home-range estimates (amt's ``hr_overlap``).

    ``x`` and ``y`` are estimates; the overlap of ``x`` with ``y`` is computed track by track (tracks of the same
    label, or the only track of each). Without ``y``, the tracks of ``x`` (or the elements of a list or dict of
    estimates) are compared: consecutive ones, all pairs, or the first with the others (``which``).

    Parameters
    ----------
    x, y
        :class:`HomeRange` objects; ``x`` alone may be a list or a dict of them.
    type
        ``"hr"``: the share of the polygon of ``x`` covered by the polygon of ``y`` at each level; the rest
        need two kernel estimates on the same template (:func:`make_trast`): ``"phr"`` (the probability of the
        use of ``y`` in the home range of ``x``), ``"vi"`` (volume of intersection), ``"ba"``
        (Bhattacharyya's affinity), ``"udoi"`` (utilisation distribution overlap index) and ``"hd"``
        (Hellinger's distance); Fieberg & Kochanny (2005).
    conditional
        For the distribution indices: use the distribution conditional on the level (the cells inside the
        isopleth, rescaled), for each level of the estimates, instead of the whole distribution.
    which
        Without ``y``: ``"consecutive"``, ``"all"`` or ``"one_to_all"``.
    labels
        Names for the estimates without ``y`` (default: the track labels, or 1, 2... for a list).

    Returns
    -------
    pandas.DataFrame
        ``from``, ``to``, ``level`` and ``overlap``. With ``y``: ``track`` instead of ``from`` and ``to``. The
        level is 1 for the indices of the whole distribution.

    References
    ----------
    Fieberg, J. & Kochanny, C. O. (2005). Quantifying home-range overlap: the importance of the utilization
    distribution. Journal of Wildlife Management 69, 1346-1359. doi:10.2193/0022-541X(2005)69[1346:QHOTIO]2.0.CO;2

    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=2, seed=4)
    >>> hr = tf.move.hr_mcp(tr, levels=[0.5, 0.95])
    >>> tf.move.hr_overlap(hr)["overlap"].shape
    (2,)
    """
    if type not in _OVERLAP:
        raise ValueError("type should be one of: hr, phr, vi, ba, udoi, or hd")
    if not isinstance(conditional, (bool, np.bool_)):
        raise TypeError("conditional: True or False")
    if y is not None:
        ux, uy = _units(x), _units(y)
        rows = []
        if len(ux) == 1 and len(uy) == 1:
            pairs = [(ux[0][0], ux[0][1], uy[0][1])]
        else:
            ydict = dict(uy)
            pairs = [(k, a, ydict[k]) for k, a in ux if k in ydict]
            if not pairs:
                raise ValueError("x and y have no track in common")
        for k, a, b in pairs:
            t = _pair(a, b, type, conditional)
            t.insert(0, "track", k)
            rows.append(t)
        return pd.concat(rows, ignore_index=True)
    if which not in ("all", "consecutive", "one_to_all"):
        raise ValueError("Which should be one of 'all', 'consecutive' or 'one_to_all'")
    units = _units(x)
    if len(units) < 2:
        raise ValueError("At least two home range estimates are needed")
    names = [k for k, _ in units]
    if labels is not None:
        labels = [str(v) for v in labels]
        if len(labels) == len(names):
            names = labels
        else:
            warnings.warn("Length of `labels` does not match the number of cases. Using default instead.",
                          UserWarning, stacklevel=2)
    if any(isinstance(n, str) and len(n) == 0 for n in names):
        raise ValueError("Names must be different from ''")
    if len(set(names)) != len(names):
        raise ValueError("Duplicated names are not permitted")
    lv0 = units[0][1].levels
    if not all(np.array_equal(lv0, u.levels) for _, u in units[1:]):
        raise ValueError("Not all levels of first home range est are in the others as well.")
    if which == "consecutive":
        grid = [(i, i + 1) for i in range(len(units) - 1)]
    elif which == "all":
        grid = [(i, j) for i in range(len(units)) for j in range(len(units)) if i != j]
    else:
        grid = [(0, j) for j in range(1, len(units))]
    rows = []
    for i, j in grid:
        t = _pair(units[i][1], units[j][1], type, conditional)
        t.insert(0, "to", names[j])
        t.insert(0, "from", names[i])
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# -------------------------------------------------------------------------------------- the rest of amt

def hr_kde_ref_scaled(tracks, range=None, trast=None, num_parts: int = 1, levels=0.95, tol: float = 0.1,
                      max_it: int = 500, *, track: str | None = None) -> pd.DataFrame:
    """The reference bandwidth scaled to give no more than ``num_parts`` polygons (amt's ``hr_kde_ref_scaled``).

    A bisection between the two ends of ``range``: the bandwidth is lowered while the isopleth at ``levels``
    has more polygons than asked and raised while it has no more.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, several tracks allowed.
    range
        The two ends of the search (default 0.01 to 1 times the reference bandwidth of each track).
    trast
        The template raster (default :func:`make_trast`, as :func:`hr_kde`).
    num_parts
        The most polygons wanted.
    levels
        One level in [1e-4, 0.9999].
    tol
        Stop when the bandwidth changes by no more than this and the polygons are few enough.
    max_it
        The most iterations.

    Returns
    -------
    pandas.DataFrame
        Indexed by track: ``h`` (the bandwidth for x and y), ``n_iter`` and ``success``. A warning when it did not
        converge.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), tracks=1, seed=2)
    >>> tf.move.hr_kde_ref_scaled(tr).shape
    (1, 3)
    """
    lv = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if lv.size != 1 or not (1e-4 <= lv[0] <= 0.9999):
        raise ValueError("levels: one number between 1e-4 and 0.9999")
    if not isinstance(max_it, (int, np.integer)):
        raise TypeError("max_it: an integer")
    items, crs, col = _tracks_of(tracks, track, "hr_kde_ref_scaled")
    if trast is None:
        trast = _make_trast(np.concatenate([t.x for t in items]), np.concatenate([t.y for t in items]), crs, 1.5, None)
    grid = _grid_of(trast)
    rows = {}
    for t in items:
        if range is None:
            rng = _ref_one(t, "none")[0] * np.array([0.01, 1.0])
        else:
            rng = np.asarray(range, dtype=np.float64)
            if rng.size != 2:
                raise ValueError("range: two numbers")
        hmin, hmax = float(rng.min()), float(rng.max())
        hcur = _mean2(hmin, hmax)
        success = False
        hnew = hcur
        i = 0
        for i in builtins_range(1, int(max_it) + 1):
            if i > 1:
                hcur = hnew
            values = _kde_values(t.x, t.y, (hcur, hcur), grid)
            geom = _isopleth_geometries(_cud2d(values), grid, lv)[0]
            n_holes = len(geom.geoms)
            if n_holes <= num_parts:
                hmax = hcur
            else:
                hmin = hcur
            hnew = _mean2(hmax, hmin)
            if abs(hcur - hnew) <= tol and n_holes <= num_parts:
                success = True
                break
        if not success:
            warnings.warn("`hr_kde_ref_scaled` did not converge.", UserWarning, stacklevel=2)
        rows[t.label] = (hcur, i, success)
    df = pd.DataFrame(rows, index=["h", "n_iter", "success"]).T.rename_axis(col)
    df["n_iter"] = df["n_iter"].astype(int)
    df["success"] = df["success"].astype(bool)
    df["h"] = df["h"].astype(float)
    return df


builtins_range = range   # `range` is a parameter of some functions here


def _mean2(a: float, b: float) -> float:
    return _core.move_hr_mean(np.array([a, b]))


def intensity_use(tracks, *, track: str | None = None):
    """The intensity of use of each track (amt's ``intensity_use``): its length over the square root of the area
    of its 100 % minimum convex polygon.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points in a projected CRS (or without one), several tracks allowed.
    track
        The track column, if not recorded (:func:`set_columns`).

    Returns
    -------
    pandas.Series
        Indexed by track.

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(40), tracks=2, seed=1)
    >>> tf.move.intensity_use(tr).shape
    (2,)
    """
    items, _, col = _tracks_of(tracks, track, "intensity_use")
    out = {}
    for t in items:
        d = _core.move_distances(t.x, t.y, np.zeros(len(t.x), np.int64), False, 1)[:-1]
        length = _core.move_r_sum(np.ascontiguousarray(d)) if len(d) else 0.0
        area = _area(_mcp_polygons(t.x, t.y, np.array([1.0]))[0])
        out[t.label] = length / _core.move_hr_pow(float(area), 0.5)
    return pd.Series(out, name="intensity_use", dtype=np.float64).rename_axis(col)


def home_range(tracks, method: str = "mcp", levels=0.95, **kwargs) -> HomeRange:
    """A home range by one of the methods: ``"mcp"`` (:func:`hr_mcp`), ``"kde"`` (:func:`hr_kde`) or ``"locoh"``
    (:func:`hr_locoh`).

    Parameters
    ----------
    tracks
        A GeoDataFrame of points in a projected CRS, several tracks allowed.
    method
        ``"mcp"``, ``"kde"`` or ``"locoh"``.
    levels
        The levels (numbers in [0, 1]).
    **kwargs
        The other arguments of the method's function.

    Returns
    -------
    HomeRange

    References
    ----------
    Signer, J., Fieberg, J. & Avgar, T. (2019). Ecology and Evolution 9, 880-890. doi:10.1002/ece3.4823

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(40), tracks=1, seed=1)
    >>> tf.move.home_range(tr, "mcp", levels=0.9).estimator
    'mcp'
    """
    fn = {"mcp": hr_mcp, "kde": hr_kde, "locoh": hr_locoh}.get(method)
    if fn is None:
        raise ValueError('method: "mcp", "kde" or "locoh"')
    return fn(tracks, levels=levels, **kwargs)
