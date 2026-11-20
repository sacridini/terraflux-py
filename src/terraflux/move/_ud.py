# SPDX-License-Identifier: GPL-3.0-or-later
"""The dynamic Brownian bridge movement model: the motion variance of a track, its utilization distribution and what
goes with them.

From R's move2utils (Kranstauber, Safi & Scharf 2026, bioRxiv doi:10.64898/2026.07.07.736908), the same numbers:
``mt_dbbmm_variance``, ``mt_dbbmm_ud``, ``mt_motion_variance``, ``mt_suggest_dbbmm_window``, ``mt_mask_segments``,
``ud_volume`` and ``ud_outer_probability``, on the C kernels of the ``move`` package (Kranstauber et al. 2012)
that move2utils keeps. The tracks are GeoDataFrames of points (:func:`terraflux.load_tracks`); the variance goes
into their columns (``motion_variance``, ``in_windows``, ``interest``) and the distribution is a raster with one
band per track (or per burst, the bursts :func:`resample` gives).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from .. import _core
from .._threads import limit, resolve
from ._clean import _find_column, _points, _seconds
from ._geometry import aeqd_crs
from ._steps import BURST
from ._tracks import check_order, columns, track_codes

DBBMM = "terraflux.dbbmm"          # the key of the variance's settings in ``attrs``
_VARIANCE_COLUMNS = ("motion_variance", "in_windows", "interest")
_LOCATION_ERROR = "location_error"
_EXT = 4.0                          # the window of a bridge step, in standard deviations
_NA_HOW = {"median": 0, "mean": 1, "zero": 2, "approx": 3}
_ARGOS = {"3": 250.0, "2": 500.0, "1": 1500.0, "0": 5000.0, "A": 5000.0, "B": 10000.0, "Z": np.nan}
_INF = math.inf


# ------------------------------------------------------------------------------------------------ units

@dataclass
class _Unit:
    """A track (or a burst of one) with what the bridge needs, in the computing CRS."""
    label: str
    track: object
    burst: object
    start: int
    stop: int
    x: np.ndarray
    y: np.ndarray
    time_mins: np.ndarray
    loc_err: np.ndarray | None = None
    variance: np.ndarray | None = None
    in_windows: np.ndarray | None = None
    interest: np.ndarray | None = None
    breaks: list = field(default_factory=list)


def _burst_column(tracks, burst) -> str | None:
    if burst is None or burst is False:
        return None
    if burst == "auto" or burst is True:
        return BURST if BURST in tracks.columns else None
    if burst not in tracks.columns:
        raise KeyError(f"burst column {burst!r} not found")
    return burst


def _unit_codes(tracks, time, track, burst):
    """(time column, unit code per row, labels, track values, burst values): a unit is a track, or a burst of a track."""
    t, k = columns(tracks, time, track)
    tcode = track_codes(tracks[k])
    b = _burst_column(tracks, burst)
    if b is None:
        _, first = np.unique(tcode, return_index=True)       # the codes are in order of appearance
        vals = tracks[k].to_numpy()[first]
        return t, tcode, [str(v) for v in vals], list(vals), [None] * len(vals)
    bv = tracks[b]
    if bv.isna().any():
        raise ValueError(f"the burst column {b!r} has missing values: drop those locations first")
    codes = pd.MultiIndex.from_arrays([tcode, bv.to_numpy()]).factorize()[0].astype(np.int64)
    first = np.zeros(codes.max() + 1 if len(codes) else 0, dtype=np.int64)
    first[codes[::-1]] = np.arange(len(codes))[::-1]
    tv = tracks[k].to_numpy()[first]
    bvv = bv.to_numpy()[first]
    return t, codes, [f"{a}__{c}" for a, c in zip(tv, bvv)], list(tv), list(bvv)


def _bounds(code: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of each unit (their rows consecutive)."""
    if not len(code):
        return []
    cut = np.flatnonzero(np.diff(code) != 0) + 1
    starts = np.r_[0, cut]
    stops = np.r_[cut, len(code)]
    return list(zip(starts.tolist(), stops.tolist()))


def _suitable(crs) -> bool:
    """A CRS the bridge can run in: none (plain coordinates), or projected with the metre as unit (R's
    ``.crs_is_suitable``; longitude / latitude and feet are not)."""
    if crs is None:
        return True
    if crs.is_geographic:
        return False
    try:
        unit = crs.axis_info[0].unit_name
    except IndexError:
        return False
    return "met" in unit.lower()


def _transform_xy(x, y, src, dst):
    if src is None or dst is None or src == dst:
        return x, y
    from pyproj import Transformer

    ox, oy = Transformer.from_crs(src, dst, always_xy=True).transform(x, y)
    return np.ascontiguousarray(ox, dtype=np.float64), np.ascontiguousarray(oy, dtype=np.float64)


# ----------------------------------------------------------------------------------------- location error

def _argos_sigma(values) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            continue
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            v = int(v)
        out[i] = _ARGOS.get(str(v).upper(), np.nan)
    return out


def _clean_sigma(v) -> np.ndarray:
    sig = np.asarray(pd.to_numeric(pd.Series(v), errors="coerce"), dtype=np.float64).copy()
    sig[~np.isfinite(sig) | (sig < 0)] = np.nan
    return sig


def _resolve_error(location_error, tracks, n: int):
    """move2utils' ``.resolve_location_error`` on a whole table: ``None``, a float, or a length-n array with NaN
    where the error is unknown."""
    if location_error is None:
        return None
    if isinstance(location_error, (bool, np.bool_)):
        raise TypeError("location_error: None, a number, an array, a column name or 'auto'")
    if isinstance(location_error, str):
        if location_error == "auto":
            col = _find_column(tracks, "eobs_horizontal_accuracy_estimate")
            if col is not None:
                return _clean_sigma(tracks[col])
            col = _find_column(tracks, "argos_lc")
            if col is not None:
                return _argos_sigma(tracks[col].to_numpy())
            warnings.warn("location_error='auto': no recognised quality column (eobs_horizontal_accuracy_estimate, "
                          "argos_lc) found, no location error is used", UserWarning, stacklevel=3)
            return None
        col = location_error if location_error in tracks.columns else _find_column(tracks, location_error)
        if col is None:
            raise KeyError(f"location_error names a column ({location_error!r}) that is not in the tracks")
        return _clean_sigma(tracks[col])
    arr = np.asarray(location_error, dtype=np.float64)
    if arr.ndim == 0 or arr.size == 1:
        v = float(arr.reshape(-1)[0])
        if not np.isfinite(v) or v < 0:
            raise ValueError("location_error: a scalar must be a non-negative finite number")
        return v
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(f"location_error: a number, or one value per location ({n}), got {arr.shape}")
    return _clean_sigma(arr)


def _error_of(resolved, start: int, stop: int, how: str) -> np.ndarray:
    """The error of one unit: zeros for none, the number repeated, or its slice with the missing values filled."""
    n = stop - start
    if resolved is None:
        return np.zeros(n)
    if isinstance(resolved, float):
        return np.full(n, resolved)
    e = np.ascontiguousarray(resolved[start:stop])
    if np.isnan(e).any():
        e, status = _core.move_impute_location_error(e, _NA_HOW[how])
        if status == 1:
            raise ValueError(f"location_error is entirely missing (n = {n}): it cannot be filled with "
                             f"location_error_na={how!r}; 'zero' only if no error at all is intended")
        if status == 2:
            raise ValueError("location_error_na='approx' needs at least two known errors in the track")
    return e


# --------------------------------------------------------------------------------------------- variance

def _check_window(window_size, margin) -> tuple[int, int]:
    for name, v in (("window_size", window_size), ("margin", margin)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"{name}: a whole number")
    window_size, margin = int(window_size), int(margin)
    if window_size % 2 != 1 or margin % 2 != 1 or margin < 1:
        raise ValueError("margin and window_size must both be odd (and positive)")
    if window_size < 2 * margin + 1:
        raise ValueError("window_size must be at least 2 * margin + 1")
    return window_size, margin


def _check_how(how) -> str:
    if how not in _NA_HOW:
        raise ValueError(f"location_error_na: one of {', '.join(_NA_HOW)}")
    return how


def _units_from(tracks, x, y, secs, code, labels, tvals, bvals) -> list[_Unit]:
    check_order(code, secs, strictly=True)
    units = []
    for u, (s, e) in enumerate(_bounds(code)):
        tm = secs[s:e] / 60.0                      # R: seconds since 1970 over 60, minutes
        units.append(_Unit(labels[u], tvals[u], bvals[u], s, e, x[s:e], y[s:e], tm))
    return units


def _fit_variance(units: list[_Unit], resolved, na_how: str, window_size: int, margin: int, threads: int) -> list[_Unit]:
    """The variance of each unit with enough locations (the others are returned with no variance, as skipped)."""
    skipped = [u.label for u in units if len(u.x) < window_size]
    if skipped and len(units) == 1:
        raise ValueError(f"window_size ({window_size}) cannot be larger than the number of locations ({len(units[0].x)})")
    if skipped:
        if len(skipped) == len(units):
            raise ValueError("No track had enough locations for the given window_size")
        warnings.warn(f"Tracks skipped (fewer locations than window_size): {', '.join(skipped)}", UserWarning,
                      stacklevel=3)
    for u in units:
        n = len(u.x)
        if n < window_size:
            continue
        u.loc_err = _error_of(resolved, u.start, u.stop, na_how)
        lag = np.append(np.diff(u.time_mins), 0.0)
        r = _core.move_dbbmm_variance(u.x, u.y, lag, u.loc_err, window_size, margin, threads)
        u.variance = r["variance"]
        ne = r["n_estim"]
        u.in_windows = np.where(ne > 0, ne.astype(np.float64), np.nan)
        u.interest = np.zeros(n, dtype=bool)
        if (ne > 0).any():
            u.interest[ne == ne.max()] = True
        u.breaks = (r["break_pos"].astype(np.int64) - 1).tolist()      # 0-based positions in the track
    return units


def _prepare(tracks, time, track, burst, what: str):
    x, y, crs, lonlat = _points(tracks, what)
    t, code, labels, tvals, bvals = _unit_codes(tracks, time, track, burst)
    secs = _seconds(tracks[t])
    if np.isnan(secs).any():
        raise ValueError(f"some locations have no time: remove them before {what}")
    return x, y, crs, lonlat, code, labels, tvals, bvals, secs


def dbbmm_variance(tracks, location_error, window_size: int = 31, margin: int = 11, location_error_na: str = "median",
                   *, time: str | None = None, track: str | None = None, burst="auto", n_threads: int | None = None):
    """The motion variance of the dynamic Brownian bridge along each track (move2utils' ``mt_dbbmm_variance``).

    A window of ``window_size`` locations slides along the track; in each, the Brownian motion variance that best
    explains the locations between their neighbours (every second one, left out in turn, as a bridge between its
    neighbours) is found by maximum likelihood, and so is one with a break point in it; the break is kept when
    it lowers the BIC. The variance of a location is the mean over the windows that cover it between the margins,
    ``margin - 1`` locations at the start and ``margin`` at the end have none. Locations covered by every window
    are the ``interest`` of the later distribution. A track with fewer locations than ``window_size`` is skipped
    with a warning (an error when it is the only one).

    Coordinates in longitude / latitude are projected to the azimuthal equidistant CRS of :func:`aeqd_crs` (the
    centre of the locations, the corrected one: move2's own is a kilometre off), the variance is in m^2 per minute.
    Other CRS are used as they are (the units of the CRS squared per minute).

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, one row per location, each track's rows consecutive and ordered by time, with
        no empty location and no repeated time (:func:`terraflux.load_tracks`, :func:`filter_unique`).
    location_error
        The standard deviation of the location error (metres): ``None`` (no error), a number, an array with one
        value per location, the name of a column, or ``"auto"`` (the column ``eobs_horizontal_accuracy_estimate``,
        else the classes of ``argos_lc``: 3 = 250 m, 2 = 500, 1 = 1500, 0 = 5000, A = 5000, B = 10000, Z =
        unknown). Missing values (and the negative or infinite ones) are filled in each track as
        ``location_error_na`` says. Column names are found with underscores, dashes or Movebank's colon
        (``eobs:horizontal-accuracy-estimate``, which move2utils itself does not find).
    window_size, margin
        Odd numbers, ``window_size >= 2 * margin + 1``; see :func:`suggest_dbbmm_window`.
    location_error_na
        ``"median"`` (default), ``"mean"``, ``"zero"`` or ``"approx"`` (linear along the track, the nearest known
        value beyond the ends): what a missing error becomes.
    time, track
        The columns of the time and of the track, if not recorded (:func:`set_columns`).
    burst
        ``"auto"`` (default): when the tracks have the column ``burst_`` (:func:`resample`), each burst is a track
        of its own, as the vignette "UD_bursted_uds" of move2utils does with a composite track id; ``None``: not;
        or the name of a column.
    n_threads
        Threads (default: most of the cores); the windows run in parallel and are added in their order, the same
        result with any number of threads.

    Returns
    -------
    GeoDataFrame
        A copy of ``tracks`` with the columns ``motion_variance`` (NaN where no window covers), ``in_windows`` (the
        number of windows, NaN where none), ``interest`` (True where every window covers) and ``location_error``
        (the errors used, the missing ones filled). ``attrs["terraflux.dbbmm"]`` holds the window, the margin, the
        break points of each track (``breaks``: the 0-based position of the break of every window that has one)
        and the CRS of the computation, which :func:`dbbmm_ud` uses.

    References
    ----------
    Kranstauber, B., Kays, R., LaPoint, S. D., Wikelski, M., Safi, K. (2012). A dynamic Brownian bridge movement
    model to estimate utilization distributions for heterogeneous animal movement. Journal of Animal Ecology 81,
    738-746. doi:10.1111/j.1365-2656.2012.01955.x

    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(120), sigma=2.0, tracks=2, seed=3)
    >>> v = tf.move.dbbmm_variance(tr, 5.0, window_size=31, margin=11)
    >>> int(v["motion_variance"].notna().sum()), int(v["interest"].sum())
    (198, 162)
    """
    window_size, margin = _check_window(window_size, margin)
    na_how = _check_how(location_error_na)
    threads = resolve(n_threads)
    x, y, crs, lonlat, code, labels, tvals, bvals, secs = _prepare(tracks, time, track, burst, "estimating the variance")
    work = aeqd_crs(tracks, "centroid") if lonlat else crs
    px, py = _transform_xy(x, y, crs, work)
    return _variance_table(tracks, px, py, code, labels, tvals, bvals, secs, location_error, na_how, window_size,
                           margin, threads, work)


def _variance_table(tracks, px, py, code, labels, tvals, bvals, secs, location_error, na_how, window_size, margin,
                    threads, work):
    units = _units_from(tracks, px, py, secs, code, labels, tvals, bvals)
    resolved = _resolve_error(location_error, tracks, len(tracks))
    with limit(threads):
        units = _fit_variance(units, resolved, na_how, window_size, margin, threads)
    n = len(tracks)
    cols = {"motion_variance": np.full(n, np.nan), "in_windows": np.full(n, np.nan), "interest": np.zeros(n, bool),
            _LOCATION_ERROR: np.full(n, np.nan)}
    for u in units:
        if u.variance is None:
            continue
        cols["motion_variance"][u.start:u.stop] = u.variance
        cols["in_windows"][u.start:u.stop] = u.in_windows
        cols["interest"][u.start:u.stop] = u.interest
        cols[_LOCATION_ERROR][u.start:u.stop] = u.loc_err
    out = tracks.copy()
    out.attrs = dict(tracks.attrs)
    for k, v in cols.items():
        out[k] = v
    out.attrs[DBBMM] = {"window_size": window_size, "margin": margin, "location_error_na": na_how,
                        "breaks": {u.label: u.breaks for u in units if u.variance is not None},
                        "crs": None if work is None else work.to_wkt(),
                        "orig_crs": None if tracks.crs is None else tracks.crs.to_wkt()}
    return out


def motion_variance(tracks) -> np.ndarray:
    """The motion variance of each location (move2utils' ``mt_motion_variance``): the column ``motion_variance`` of
    :func:`dbbmm_variance`, NaN at the margins of each track.

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(60), sigma=2.0, tracks=1, seed=3)
    >>> tf.move.motion_variance(tf.move.dbbmm_variance(tr, 5.0, 31, 11)).shape
    (60,)
    """
    if "motion_variance" not in tracks.columns:
        raise ValueError("no motion_variance column: run dbbmm_variance first")
    return tracks["motion_variance"].to_numpy(np.float64)


# ------------------------------------------------------------------------------------------ the window

class WindowSuggestion(NamedTuple):
    """A window for :func:`dbbmm_variance`."""
    window_size: int
    margin: int
    median_dt_secs: float
    target_hours: float
    window_hours: float
    n: int
    n_windows: int


def _nearest_odd(z: float) -> int:
    k = int(np.rint(min(max(z, -2e9), 2e9)))   # R's round: halves to even
    if k % 2 == 0:
        k += 1
    return max(k, 1)


def _floor_odd(z: float) -> int:
    k = math.floor(min(max(z, -2e9), 2e9))
    if k % 2 == 0:
        k -= 1
    return max(k, 1)


def suggest_dbbmm_window(tracks, target_hours: float = 4.0, *, time: str | None = None, track: str | None = None):
    """A window size and margin for :func:`dbbmm_variance` from the sampling of each track (move2utils'
    ``mt_suggest_dbbmm_window``, without its plot).

    The window covers about ``target_hours`` at the median time lag (an odd number of locations, at least 11 and
    at most a quarter of the track), the margin is a quarter of it (odd, at least 3, at most ``(window_size - 1) /
    2``). A track with fewer than 11 locations or fewer than two time lags above zero gets ``None`` (with a
    warning).

    Returns
    -------
    WindowSuggestion or dict or None
        For one track the :class:`WindowSuggestion` (``window_size``, ``margin``, ``median_dt_secs``,
        ``target_hours``, ``window_hours`` and ``n_windows``, the whole windows that fit); for several, a dict
        of them by track.

    Examples
    --------
    >>> import numpy as np
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(np.arange(0, 3600 * 60, 600.0), sigma=2.0, tracks=1, seed=3)
    >>> tf.move.suggest_dbbmm_window(tr, 4).window_size
    25
    """
    if not (isinstance(target_hours, (int, float, np.number)) and np.isfinite(target_hours) and target_hours > 0):
        raise ValueError("target_hours: a positive number")
    t, k = columns(tracks, time, track)
    code = track_codes(tracks[k])
    secs = _seconds(tracks[t])
    out = {}
    _, first = np.unique(code, return_index=True)
    ids = tracks[k].to_numpy()[np.sort(first)]
    for (s, e), tid in zip(_bounds(code), ids):
        n = e - s
        if n < 11:
            warnings.warn(f"Too few locations for dBBMM (< 11) in {tid}: no suggestion", UserWarning, stacklevel=2)
            out[tid] = None
            continue
        dt = np.diff(secs[s:e])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) < 2:
            warnings.warn(f"Too few finite time lags in {tid}: no suggestion", UserWarning, stacklevel=2)
            out[tid] = None
            continue
        med = float(np.median(dt))
        raw_w = target_hours * 3600 / med
        cap_w = max(11, _floor_odd(n / 4))
        w = max(11, min(_nearest_odd(raw_w), cap_w))
        m_int = max(3, _nearest_odd((w - 1) / 4))
        m = min(m_int, _floor_odd((w - 1) / 2))
        out[tid] = WindowSuggestion(int(w), int(m), med, float(target_hours), w * med / 3600, int(n), int(n // w))
    return next(iter(out.values())) if len(out) == 1 else out


# --------------------------------------------------------------------------------------------- masking

def mask_segments(tracks, segments, *, time: str | None = None, track: str | None = None, burst="auto"):
    """Leave segments out of the distribution (move2utils' ``mt_mask_segments``): ``interest`` is False for them.

    The segment ``i`` of a track is the bridge from its location ``i`` to ``i + 1`` (0-based; R counts from 1),
    the one that the time between two fixes spans. Leave out the bridges across long gaps, for example, as the
    vignette "UD_gap_aware_ud" does.

    Parameters
    ----------
    tracks
        Tracks with the columns of :func:`dbbmm_variance`.
    segments
        Whole numbers in ``0 .. n - 2`` of each track, applied to every track (a mistake for a track with fewer
        locations); a boolean array with one value per row of ``tracks`` (True: leave out the segment that starts
        there); or a dict by track (label of :func:`dbbmm_ud`'s bands) of numbers or of booleans of the length
        ``n - 1`` or ``n`` of that track. Missing values are ignored.

    Returns
    -------
    GeoDataFrame
        A copy of ``tracks`` with ``interest`` updated.

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(120), sigma=2.0, tracks=1, seed=3)
    >>> v = tf.move.dbbmm_variance(tr, 5.0, 31, 11)
    >>> int(tf.move.mask_segments(v, [40, 41])["interest"].sum()) - int(v["interest"].sum())
    -2
    """
    if "interest" not in tracks.columns:
        raise ValueError("no interest column: run dbbmm_variance first")
    _, code, labels, _, _ = _unit_codes(tracks, time, track, burst)
    bounds = _bounds(code)
    interest = tracks["interest"].to_numpy(bool).copy()

    def cut(label, s, e, seg):
        n = e - s
        seg = np.asarray(seg)
        if seg.dtype == bool:
            if len(seg) not in (n - 1, n):
                raise ValueError(f"logical segments must have length n-1 ({n - 1}) or n ({n}); got {len(seg)}")
            idx = np.flatnonzero(seg)
        else:
            f = seg.astype(np.float64) if seg.dtype.kind in "iuf" else None
            if f is None:
                raise TypeError("segments: an integer or logical array")
            f = f[~np.isnan(f)]
            if len(f) and (f.min() < 0 or f.max() > n - 2):
                raise ValueError(f"segments must be integers in 0..{n - 2} (track {label})")
            idx = f.astype(np.int64)
        interest[s + idx] = False

    if isinstance(segments, dict):
        unknown = [k for k in segments if k not in labels]
        if unknown:
            raise KeyError(f"segments names no track: {unknown}")
        for u, (s, e) in enumerate(bounds):
            if labels[u] in segments:
                cut(labels[u], s, e, segments[labels[u]])
    else:
        seg = np.asarray(segments)
        if seg.dtype == bool and len(seg) == len(tracks):
            interest[seg] = False
        else:
            for u, (s, e) in enumerate(bounds):
                cut(labels[u], s, e, seg)
    out = tracks.copy()
    out.attrs = dict(tracks.attrs)
    out["interest"] = interest
    return out


# --------------------------------------------------------------------------------------------- the grid

class Grid(NamedTuple):
    """The grid of a distribution: terra's extent and counts (square cells that need not be exactly square)."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    nrow: int
    ncol: int

    @property
    def xres(self) -> float:
        return (self.xmax - self.xmin) / self.ncol

    @property
    def yres(self) -> float:
        return (self.ymax - self.ymin) / self.nrow

    def vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """The centres of the columns and of the rows, both ascending (terra's xFromCol and yFromRow, the rows from
        the bottom up as the kernel takes them)."""
        xs = self.xmin + (np.arange(self.ncol, dtype=np.float64) + 0.5) * self.xres
        rows = np.arange(self.nrow - 1, -1, -1, dtype=np.float64)
        ys = self.ymax - ((rows + 0.5) * self.yres)
        return xs, ys


def _extcalc(px, py, ext):
    """The bounding box of the points widened by ``ext`` times its width and height on each side (``.extcalc``)."""
    xmin, xmax, ymin, ymax = float(px.min()), float(px.max()), float(py.min()), float(py.max())
    xr, yr = xmax - xmin, ymax - ymin
    return xmin - ext * xr, xmax + ext * xr, ymin - ext * yr, ymax + ext * yr


def _make_grid(px, py, cell_size, dim_size, ext) -> Grid:
    """``.make_raster``: the widened box with its sides rounded up to whole cells, the excess split evenly."""
    xmin, xmax, ymin, ymax = _extcalc(px, py, ext)
    xr, yr = xmax - xmin, ymax - ymin
    cell = cell_size if cell_size is not None else max(xr, yr) / dim_size
    if not (cell > 0 and np.isfinite(cell)):
        raise ValueError("the locations have no extent to build a grid on: give like=")
    ymin2 = ymin - (float(np.ceil(yr / cell)) * cell - yr) / 2
    ymax2 = ymax + (float(np.ceil(yr / cell)) * cell - yr) / 2
    xmin2 = xmin - (float(np.ceil(xr / cell)) * cell - xr) / 2
    xmax2 = xmax + (float(np.ceil(xr / cell)) * cell - xr) / 2
    nr = int(np.rint((ymax2 - ymin2) / cell))
    nc = int(np.rint((xmax2 - xmin2) / cell))
    if nr < 1 or nc < 1:
        raise ValueError("the grid would have no cells: the locations are on a line, give like= or a larger ext_factor")
    return Grid(xmin2, xmax2, ymin2, ymax2, nr, nc)


def _needed_range(box, env, cell, extra=0.0):
    """The extent a grid of cells of size ``cell`` needs so that every window stays inside it (``.ud_needed_range``)."""
    return (min(box[0], env[0] - (0.5 + extra) * cell), max(box[1], env[1] + (0.5 + extra) * cell),
            min(box[2], env[2] - (1.5 + extra) * cell), max(box[3], env[3] + (0.5 + extra) * cell))


def _grid_from_range(rng, cell) -> Grid:
    xr, yr = rng[1] - rng[0], rng[3] - rng[2]
    nc = max(1, int(np.ceil(xr / cell - 1e-9)))
    nr = max(1, int(np.ceil(yr / cell - 1e-9)))
    xpad = (nc * cell - xr) / 2
    ypad = (nr * cell - yr) / 2
    return Grid(rng[0] - xpad, rng[1] + xpad, rng[2] - ypad, rng[3] + ypad, nr, nc)


@dataclass
class _Kernel:
    """The inputs of the kernel for one unit (the same for the probe and the distribution)."""
    unit: _Unit
    x: np.ndarray
    y: np.ndarray
    s1: np.ndarray
    t: np.ndarray
    le: np.ndarray
    interest: np.ndarray
    time_step: float


def _kernel_inputs(u: _Unit, le: np.ndarray, time_step) -> _Kernel:
    n = len(u.x)
    if n < 2:
        raise ValueError(f"A UD needs at least two fixes; got {n} in {u.label}")
    lag = np.append(np.diff(u.time_mins), 0.0)
    derived = time_step is None
    ts = float(lag[:-1].min()) / 15 if derived else float(time_step)
    if not (np.isfinite(ts) and ts > 0):
        raise ValueError("The default time_step is a fraction of the shortest time lag, which is not positive here "
                         "(duplicate or unsorted times?): give time_step in minutes" if derived else
                         "time_step must be a single positive number (minutes)")
    var = np.nan_to_num(u.variance, nan=0.0)
    return _Kernel(u, u.x, u.y, np.ascontiguousarray(var), np.ascontiguousarray(u.time_mins - u.time_mins.min()),
                   np.ascontiguousarray(le), np.ascontiguousarray(u.interest.view(np.uint8)), ts)


class _Probe(NamedTuple):
    env: tuple
    sides: tuple      # left, right, bottom, top
    safe: bool


def _probe(k: _Kernel, g: Grid) -> _Probe:
    xs, ys = g.vectors()
    if len(xs) < 2 or len(ys) < 2:
        return _Probe((_INF, -_INF, _INF, -_INF), (len(xs) < 2, len(xs) < 2, len(ys) < 2, len(ys) < 2), False)
    r = _core.move_ud_probe(k.x, k.y, k.s1, k.t, k.le, k.interest, xs, ys, k.time_step, _EXT)
    idx = r[4:]
    sides = (bool(idx[0] < 0), bool(idx[1] > len(xs) - 1), bool(idx[2] > len(ys) - 1), bool(idx[3] < 0))
    return _Probe(tuple(r[:4]), sides, not any(sides))


def _union(probes) -> tuple:
    e = np.array([p.env for p in probes])
    return float(e[:, 0].min()), float(e[:, 1].max()), float(e[:, 2].min()), float(e[:, 3].max())


def _check_cells(g: Grid, res, dim_size, max_cells):
    n = g.nrow * g.ncol
    if n > max_cells:
        what = f"res={res:g}" if res is not None else f"dim_size={dim_size}"
        raise ValueError(f"The grid from {what} would have {g.ncol} x {g.nrow} = {n:.3g} cells, above the limit of "
                         f"{max_cells:.3g} cells (max_cells): use a {'larger res' if res is not None else 'smaller dim_size'}"
                         ", or raise max_cells")


def _auto_grid(px, py, res, dim_size, ext_factor, kernels, max_cells) -> tuple[Grid, bool]:
    """``.ud_auto_grid``: the grid of ``res`` or ``dim_size`` on the widened box, kept when every bridge window
    fits, else enlarged to the windows (the cell size grows with ``dim_size``, the cells are added with ``res``)."""
    g = _make_grid(px, py, res, None if res is not None else dim_size, ext_factor)
    _check_cells(g, res, dim_size, max_cells)
    probes = [_probe(k, g) for k in kernels]
    if all(p.safe for p in probes):
        return g, False
    env = _union(probes)
    box = _extcalc(px, py, ext_factor)
    cell = res if res is not None else max(box[1] - box[0], box[3] - box[2]) / dim_size
    if res is None:
        # the fixed point of cell = the longest side of the needed range / dim_size (converges for dim_size >= 3)
        for _ in range(500):
            need = _needed_range(box, env, cell)
            new = max(need[1] - need[0], need[3] - need[2]) / dim_size
            if new <= cell * (1 + 1e-12):
                break
            cell = new
        else:
            raise ValueError(f"dim_size={dim_size} is too small to hold the bridges' kernel windows with the margin "
                             "the kernel needs (at least 3): use a larger dim_size, or give res")
    for attempt in range(4):
        if res is None and attempt > 0:
            cell = cell * (1 + 1e-9)
        need = _needed_range(box, env, cell, extra=float(attempt) if res is not None else 0.0)
        if res is not None:
            nc = max(1, math.ceil((need[1] - need[0]) / cell - 1e-9))
            nr = max(1, math.ceil((need[3] - need[2]) / cell - 1e-9))
            if nc * nr > max_cells:
                raise ValueError(f"Enlarging the grid so that every bridge's window fits would need {nc} x {nr} = "
                                 f"{nc * nr:.3g} cells of {cell:g} map units, above the limit of {max_cells:.3g} cells "
                                 "(max_cells): use a larger res, or give dim_size, which fixes the number of cells")
        g = _grid_from_range(need, cell)
        if all(_probe(k, g).safe for k in kernels):
            warnings.warn(f"The grid was enlarged so that every bridge's window fits: {g.ncol} x {g.nrow} cells of "
                          f"{g.xres:.4g} map units (the box of ext_factor={ext_factor:g} was too small)",
                          UserWarning, stacklevel=4)
            return g, True
    raise RuntimeError("could not build a UD grid large enough for the kernel windows (please report this)")


def _template(like):
    """(Grid, pyproj CRS or None) of a raster to compute on."""
    from pyproj import CRS

    from .._layers import _one
    from ..io import _transform

    da = _one(like)
    if da.ndim not in (2, 3):
        raise ValueError("like: a raster (y, x) or (band, y, x)")
    tr = _transform(da)
    if tr.b != 0 or tr.d != 0 or tr.e >= 0 or tr.a <= 0:
        raise ValueError("like: a north-up raster without rotation")
    ny, nx = da.shape[-2:]
    xmin, ymax = tr.c, tr.f
    g = Grid(xmin, xmin + nx * tr.a, ymax + ny * tr.e, ymax, int(ny), int(nx))
    rc = da.rio.crs
    return g, (None if rc is None else CRS.from_wkt(rc.to_wkt()))


def _check_template(g: Grid, kernels):
    probes = [_probe(k, g) for k in kernels]
    if all(p.safe for p in probes):
        return
    if g.nrow < 2 or g.ncol < 2:
        raise ValueError(f"like has {g.nrow} row(s) and {g.ncol} column(s); the kernel needs at least two of each")
    sides = [n for n, f in zip(("left", "right", "bottom", "top"), np.any([p.sides for p in probes], axis=0)) if f]
    need = _needed_range((g.xmin, g.xmax, g.ymin, g.ymax), _union(probes), max(g.xres, g.yres))
    raise ValueError("like is too small: the bridges' kernel windows reach beyond its " + ", ".join(sides) +
                     " edge(s). A template is never enlarged. A raster with the extent "
                     f"x {math.floor(need[0])} .. {math.ceil(need[1])}, y {math.floor(need[2])} .. {math.ceil(need[3])} "
                     "keeps its resolution and is large enough. Or leave like out and give res or dim_size to build "
                     "a grid large enough")


def _grid_args(like, res, dim_size, ext_factor, ext):
    if ext is not None:
        warnings.warn("ext is deprecated: use ext_factor (the fraction by which the bounding box of the track is "
                      "widened)", DeprecationWarning, stacklevel=3)
        ext_factor = ext
    if isinstance(ext_factor, (bool, np.bool_)) or not isinstance(ext_factor, (int, float, np.number)) \
            or not np.isfinite(ext_factor) or ext_factor < 0:
        raise ValueError("ext_factor: a non-negative number")
    given = [n for n, v in (("like", like), ("res", res), ("dim_size", dim_size)) if v is not None]
    if len(given) > 1:
        raise ValueError(f"give only one of like (a template), res (a cell size) or dim_size (cells along the longest "
                         f"side); got {' and '.join(given)}")
    if res is not None and not (isinstance(res, (int, float, np.number)) and np.isfinite(res) and res > 0):
        raise ValueError("res: a positive number, the cell size in the units of the computing CRS")
    if dim_size is not None and not (isinstance(dim_size, (int, np.integer)) and not isinstance(dim_size, bool)
                                     and dim_size >= 2):
        raise ValueError("dim_size: a whole number of at least 2 (cells along the longest side)")
    if not given:
        dim_size = 100
    return (None if res is None else float(res)), dim_size, float(ext_factor)


# ------------------------------------------------------------------------------------------ the output

def _raster(stack: np.ndarray, g: Grid, labels, tvals, bvals, crs, name: str, attrs: dict) -> xr.DataArray:
    from affine import Affine

    xs = g.xmin + (np.arange(g.ncol, dtype=np.float64) + 0.5) * g.xres
    ys = g.ymax - ((np.arange(g.nrow, dtype=np.float64) + 0.5) * g.yres)
    coords = {"band": list(labels), "track": ("band", [str(v) for v in tvals]), "y": ys, "x": xs}
    if any(b is not None for b in bvals):
        coords["burst"] = ("band", list(bvals))
    attrs = {**attrs, "extent": (g.xmin, g.xmax, g.ymin, g.ymax)}      # exact, as the grid was made
    da = xr.DataArray(stack, dims=("band", "y", "x"), coords=coords, name=name, attrs=attrs)
    if crs is not None:
        da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(Affine(g.xres, 0.0, g.xmin, 0.0, -g.yres, g.ymax), inplace=True)
    return da


def _to_input_crs(stack: np.ndarray, g: Grid, src, dst):
    """The bands warped to another CRS, bilinear, the negative and missing cells made 0 and every band summed to
    1 again (move2utils' ``.warp_ud_to_target``; GDAL's warp chooses the grid, as terra's ``project`` does)."""
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    src_tr = Affine(g.xres, 0.0, g.xmin, 0.0, -g.yres, g.ymax)
    tr, w, h = calculate_default_transform(src, dst, g.ncol, g.nrow, left=g.xmin, bottom=g.ymin, right=g.xmax,
                                           top=g.ymax)
    out = np.full((len(stack), h, w), np.nan)
    for i, band in enumerate(stack):
        reproject(np.ascontiguousarray(band), out[i], src_transform=src_tr, src_crs=src, dst_transform=tr,
                  dst_crs=dst, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    out[~np.isfinite(out) | (out < 0)] = 0.0
    for band in out:
        s = math.fsum(band.ravel())
        if np.isfinite(s) and s > 0:
            band /= s
    return out, Grid(tr.c, tr.c + w * tr.a, tr.f + h * tr.e, tr.f, int(h), int(w))


# --------------------------------------------------------------------------------------------- the UD

_NOTE = ("computed and returned in the azimuthal equidistant CRS centred on the tracks (metres); crs='input' warps "
         "it back to the CRS of the tracks (bilinear, cells below 0 made 0, every band summed to 1 again)")


def dbbmm_ud(tracks, location_error=None, *, like=None, res: float | None = None, dim_size: int | None = None,
             ext_factor: float = 0.5, ext: float | None = None, time_step: float | None = None, window_size: int = 31,
             margin: int = 11, location_error_na: str = "median", crs: str = "projected", max_cells: float = 1e8,
             time: str | None = None, track: str | None = None, burst="auto", n_threads: int | None = None):
    """The utilization distribution of the dynamic Brownian bridge, one band per track (move2utils' ``mt_dbbmm_ud``).

    Each pair of consecutive locations is a Brownian bridge whose variance is the motion variance of the
    locations (:func:`dbbmm_variance`) plus the location error at the ends; it is walked in steps of
    ``time_step`` and each step spreads a bivariate normal over the cells within four standard deviations of its
    position. The cell values add the steps' densities times the cell area and each band is divided by its sum:
    the probability of the animal being in the cell, summing to 1. Only the segments of ``interest`` count
    (:func:`mask_segments`). The tracks share one grid, so the bands compare directly.

    Parameters
    ----------
    tracks
        Tracks as for :func:`dbbmm_variance`, with or without its columns: without them the variance is computed
        here with ``window_size``, ``margin``, ``location_error`` and ``location_error_na``; with them (and the
        settings kept in ``attrs``) it is used as it is, and ``location_error``, if given (a number, an array
        with one value per location, no missing values, or a dict by track), replaces the errors they hold.
    like
        A raster whose grid to compute on (its CRS must be projected in metres, or the tracks' own coordinates are
        taken into it). It is never enlarged: an error says how large it has to be. Or give one of ``res``,
        ``dim_size``:
    res
        The cell size in the units of the computing CRS. With the grid of the box of the locations widened by
        ``ext_factor`` times its sides, enlarged by cells when the bridge windows do not fit.
    dim_size
        The cells along the longest side (the cell size grows when the windows do not fit); 100 when none of
        ``like``, ``res`` and ``dim_size`` is given.
    ext_factor
        The fraction of the box's sides added to each side of it (0.5).
    ext
        Deprecated name of ``ext_factor``.
    time_step
        The step of the integration over time, in minutes (default: a fifteenth of the shortest time lag of the
        track). The result depends on it only through the discretisation of the bridges.
    crs
        ``"projected"`` (default): the distribution in the CRS it was computed in, a metric one (the tracks' own
        if projected in metres, else the azimuthal equidistant of :func:`aeqd_crs`, centred on the middle of the
        ranges of longitude and latitude); ``"input"``: warped back to the CRS of the tracks when that is not it
        (bilinear, then cells below zero made zero and each band summed to 1 again: not the same bits on every
        machine, the sums agree to ~1e-3 with the original). With ``like`` the distribution is on its grid.
    max_cells
        The most cells a grid built from ``res`` or ``dim_size`` may have (move2utils' ``ud_max_cells``).
    time, track, burst, n_threads
        As in :func:`dbbmm_variance`. The cells are computed in parallel strips; every cell adds its steps in their
        order, so the result is the same with any number of threads.

    Returns
    -------
    xarray.DataArray
        ``(band, y, x)``, one band per track (``band`` is the track, ``track`` the track's coordinate;
        with bursts the band is ``"<track>__<burst>"`` and ``burst`` is also a coordinate), each summing to 1, with
        the CRS written. ``attrs["note"]`` says where it was computed.

    Notes
    -----
    A track with fewer than ``window_size`` locations gets no band (a warning). move2utils' ``mt_dbbmm_ud``
    returns the distribution in the CRS of the input (longitude / latitude included) through terra and GDAL; here
    that is ``crs="input"``, not bit for bit.

    References
    ----------
    Kranstauber, B., Kays, R., LaPoint, S. D., Wikelski, M., Safi, K. (2012). A dynamic Brownian bridge movement
    model to estimate utilization distributions for heterogeneous animal movement. Journal of Animal Ecology 81,
    738-746. doi:10.1111/j.1365-2656.2012.01955.x

    Horne, J. S., Garton, E. O., Krone, S. M., Lewis, J. S. (2007). Analyzing animal movements using Brownian
    bridges. Ecology 88, 2354-2363. doi:10.1890/06-0957.1

    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(120), sigma=2.0, tracks=2, seed=3)
    >>> ud = tf.move.dbbmm_ud(tr, 5.0, dim_size=60)
    >>> ud.dims, ud.shape[0], round(float(ud.isel(band=0).sum()), 12)
    (('band', 'y', 'x'), 2, 1.0)
    """
    if crs not in ("projected", "input"):
        raise ValueError('crs: "projected" or "input"')
    res, dim_size, ext_factor = _grid_args(like, res, dim_size, ext_factor, ext)
    threads = resolve(n_threads)
    x, y, tcrs, _, code, labels, tvals, bvals, secs = _prepare(tracks, time, track, burst, "the distribution")
    info = tracks.attrs.get(DBBMM)
    has_cols = all(c in tracks.columns for c in _VARIANCE_COLUMNS)
    tmpl = _template(like) if like is not None else None
    if has_cols and info is not None and info.get("crs") is not None:
        from pyproj import CRS

        work = CRS.from_wkt(info["crs"])
    elif tmpl is not None:
        work = tmpl[1]
        if work is not None and not _suitable(work):
            raise ValueError("like: its CRS must be projected in metres (the distribution is computed on its grid)")
    else:
        work = tcrs if _suitable(tcrs) else aeqd_crs(tracks, "center")
    if tmpl is not None and tmpl[1] is not None and work is not None and work != tmpl[1]:
        raise ValueError("like: its CRS must be the CRS the variance was computed in: re-project it, or give the "
                         "tracks without the variance columns")
    px, py = _transform_xy(x, y, tcrs, work)
    check_order(code, secs, strictly=True)
    if has_cols:
        units = _units_from(tracks, px, py, secs, code, labels, tvals, bvals)
        if _LOCATION_ERROR not in tracks.columns:
            raise ValueError("the tracks have the variance columns but no location_error column: run "
                             "dbbmm_variance again")
        mv = tracks["motion_variance"].to_numpy(np.float64)
        le_all = tracks[_LOCATION_ERROR].to_numpy(np.float64)
        interest = tracks["interest"].to_numpy(bool)
        inw = tracks["in_windows"].to_numpy(np.float64)
        keep = []
        for u in units:
            u.variance, u.interest, u.in_windows = mv[u.start:u.stop], interest[u.start:u.stop], inw[u.start:u.stop]
            u.loc_err = le_all[u.start:u.stop]
            if np.isnan(u.loc_err).all() and np.isnan(u.variance).all():
                continue                                    # a track the variance skipped
            keep.append(u)
        override = _expand_error(location_error, tracks, keep)
    else:
        window_size, margin = _check_window(window_size, margin)
        na_how = _check_how(location_error_na)
        units = _units_from(tracks, px, py, secs, code, labels, tvals, bvals)
        resolved = _resolve_error(location_error, tracks, len(tracks))
        with limit(threads):
            units = _fit_variance(units, resolved, na_how, window_size, margin, threads)
        keep = [u for u in units if u.variance is not None]
        override = {}
    kernels = [_kernel_inputs(u, override.get(u.label, u.loc_err), time_step) for u in keep]
    if tmpl is not None:
        grid = tmpl[0]
        _check_template(grid, kernels)
    else:
        px_all = np.concatenate([u.x for u in keep])
        py_all = np.concatenate([u.y for u in keep])
        grid, _ = _auto_grid(px_all, py_all, res, dim_size, ext_factor, kernels, float(max_cells))
    xs, ys = grid.vectors()
    stack = np.empty((len(kernels), grid.nrow, grid.ncol))
    with limit(threads):
        for i, k in enumerate(kernels):
            stack[i] = _core.move_dbbmm_ud(k.x, k.y, k.s1, k.t, k.le, k.interest, xs, ys, k.time_step, _EXT, threads)
    names = [u.label for u in keep]
    attrs = {"note": _NOTE, "ext": _EXT, "time_step": [k.time_step for k in kernels]}
    if crs == "input" and tmpl is None and work is not None and tcrs is not None and work != tcrs:
        stack, grid = _to_input_crs(stack, grid, work, tcrs)
        attrs["note"] = "warped to the CRS of the tracks (bilinear, negative cells made 0, bands summed to 1 again)"
        work = tcrs
    return _raster(stack, grid, names, [u.track for u in keep], [u.burst for u in keep], work, "dbbmm_ud", attrs)


def _expand_error(location_error, tracks, units) -> dict:
    """The errors given to ``dbbmm_ud`` for tracks that already have a variance: by unit label."""
    if location_error is None:
        return {}
    n = len(tracks)

    def one(v, size):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0 or a.size == 1:
            a = np.full(size, float(a.reshape(-1)[0]))
        if len(a) != size:
            raise ValueError("location_error must be one value or one value per location")
        if np.isnan(a).any():
            raise ValueError("location_error must not contain missing values")
        if (a < 0).any():
            raise ValueError("location_error must be non-negative")
        return a

    if isinstance(location_error, dict):
        return {u.label: one(location_error[u.label], u.stop - u.start) for u in units if u.label in location_error}
    a = np.asarray(location_error, dtype=np.float64)
    if a.ndim == 0 or a.size == 1:
        return {u.label: one(a, u.stop - u.start) for u in units}
    a = one(a, n)
    return {u.label: np.ascontiguousarray(a[u.start:u.stop]) for u in units}


# ------------------------------------------------------------------------------------ volume and probability

def _bands(ud) -> np.ndarray:
    a = np.asarray(ud.values, dtype=np.float64)
    if a.ndim == 2:
        return a[None]
    if a.ndim != 3:
        raise ValueError("ud: a raster (y, x) or (band, y, x)")
    return a


def ud_volume(ud):
    """The volume of a distribution: each cell's cumulative probability when the cells are added from the largest
    value down (move2utils' ``ud_volume``). The 0.95 contour of it bounds the smallest area with 95 % of the
    probability. Per band; a band whose total is not positive is left as it is; missing cells stay missing; cells
    of equal value are taken in the order of the cells.

    Examples
    --------
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(120), sigma=2.0, tracks=1, seed=3)
    >>> v = tf.move.ud_volume(tf.move.dbbmm_ud(tr, 5.0, dim_size=40))
    >>> float(v.max()) <= 1.0 + 1e-12
    True
    """
    a = _bands(ud)
    out = np.stack([_core.move_ud_volume(np.ascontiguousarray(b.reshape(-1))).reshape(b.shape) for b in a])
    return ud.copy(data=out.reshape(np.shape(ud.values)))


def ud_outer_probability(points, ud, *, band=None) -> np.ndarray:
    """The outer probability of locations: the volume of the distribution at each one (move2utils'
    ``ud_outer_probability``), the share of the probability in the cells at least as likely as the cell that
    holds the location (small: inside the core, near 1: far out). NaN for a location outside the raster.

    Parameters
    ----------
    points
        A GeoDataFrame or GeoSeries of points (taken to the raster's CRS), or an array with the x and y of the
        locations in its first two columns, in the raster's CRS.
    ud
        A distribution (:func:`dbbmm_ud`) of one band, or of several with ``band`` naming one.
    band
        The band (label or position) of a stack.

    Examples
    --------
    >>> import numpy as np
    >>> import terraflux as tf
    >>> tr = tf.move.simulate_brownian(range(120), sigma=2.0, tracks=1, seed=3)
    >>> ud = tf.move.dbbmm_ud(tr, 5.0, dim_size=40)
    >>> xy = np.c_[tr.geometry.x, tr.geometry.y]
    >>> p = tf.move.ud_outer_probability(xy, ud)
    >>> bool(np.nanmax(p) <= 1.0 + 1e-12)
    True
    """
    from ..io import _transform  # noqa: F401  (the rio transform of the ud)
    from ..sdm._points import grid_cells, to_crs

    if "band" in ud.dims and ud.sizes["band"] > 1:
        if band is None:
            raise ValueError("ud has several bands: name one with band=")
        ud = ud.sel(band=band) if band in list(ud["band"].values) else ud.isel(band=int(band))
    elif "band" in ud.dims:
        ud = ud.isel(band=0)
    v = ud_volume(ud)
    geom = getattr(points, "geometry", points)
    if hasattr(geom, "x") and hasattr(geom, "geom_type"):
        xs, ys = geom.x.to_numpy(np.float64), geom.y.to_numpy(np.float64)
        xs, ys = to_crs(xs, ys, getattr(points, "crs", None), ud.rio.crs)
    else:
        a = np.asarray(points, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError("points: a GeoDataFrame of points or an array with at least two columns (x, y)")
        xs, ys = np.ascontiguousarray(a[:, 0]), np.ascontiguousarray(a[:, 1])
    cells = grid_cells(xs, ys, v)
    vals = np.full(len(cells), np.nan)
    ok = cells >= 0
    vals[ok] = np.asarray(v.values, dtype=np.float64).reshape(-1)[cells[ok]]
    return vals


__all__ = ["Grid", "WindowSuggestion", "dbbmm_ud", "dbbmm_variance", "mask_segments", "motion_variance",
           "suggest_dbbmm_window", "ud_outer_probability", "ud_volume"]
