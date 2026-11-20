# SPDX-License-Identifier: GPL-3.0-or-later
"""Cleaning and thinning of tracks: the physiological speed limit, GPS quality, thinning in time and in distance.

The simple part of R's move2utils (Kranstauber, Safi & Scharf 2026, bioRxiv doi:10.64898/2026.07.07.736908), the
same numbers: ``v_phys_estimate``, ``mt_flag_speed_cap`` with a hard cap, ``mt_filter_gps_quality``,
``mt_thin_time`` and ``mt_thin_distance``.
"""

from __future__ import annotations

import datetime
import re
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from ._tracks import _TRACK_NAMES, _find, _nanoseconds, check_order, columns, length_factor, track_codes

_MODES = ("flying", "running", "swimming")
_HIRT_RANGE = (3e-8, 1.084e5)  # kg: the masses of the data Hirt et al. fitted
_REFERENCE = "Hirt et al. (2017) Nat. Ecol. Evol., doi:10.1038/s41559-017-0241-4"


class VPhys(NamedTuple):
    """The allometric speed limit of :func:`v_phys_estimate` (floats for one mass, arrays for several)."""
    v_max: float | np.ndarray      #: the prediction, m/s
    lower: float | np.ndarray      #: lower bound of the parameter-uncertainty interval, m/s
    upper: float | np.ndarray      #: upper bound, m/s
    kmh: float | np.ndarray        #: the prediction in km/h (Hirt et al.'s unit)
    mass: float | np.ndarray       #: the body mass, kg
    mode: str                      #: ``"flying"``, ``"running"`` or ``"swimming"``
    ci_level: float                #: the level of the interval
    reference: str = _REFERENCE


def _mode_of(mode) -> str:
    """R's ``match.arg``: the full name or a unique beginning of it."""
    if not isinstance(mode, str):
        raise ValueError(f"mode: one of {', '.join(_MODES)}")
    hits = [m for m in _MODES if m == mode] or [m for m in _MODES if mode and m.startswith(mode)]
    if len(hits) != 1:
        raise ValueError(f"mode: one of {', '.join(_MODES)}")
    return hits[0]


def v_phys_estimate(mass, mode: str = "flying", ci_level: float = 0.95) -> VPhys:
    """Allometric estimate of the physiological maximum speed of an animal (move2utils' ``v_phys_estimate``).

    The scaling law of Hirt et al. (2017), a power law of the body mass saturated by a time-dependent term,
    ``v = a M^b (1 - exp(-h M^i))``, with the parameters of its Supplementary Table 4 for each locomotion
    ``mode`` (``M`` in kg, ``v`` in km/h, returned in m/s). The interval is the parameter uncertainty of the
    fit by the delta method (the standard errors of the table), not the scatter between species (a further
    factor of about 2, R^2 = 0.893). The model is for horizontal speeds and bursts: a ceiling for speeds
    that cannot be real (see :func:`flag_speed`), not a cruising speed; specialists (cheetah, pronghorn) can
    exceed it, and a published value for the species is better when there is one.

    Parameters
    ----------
    mass
        Body mass in kg: a positive number or an array of them. A warning when outside the data of Hirt et
        al. (3e-8 to 1.084e5 kg).
    mode
        ``"flying"`` (default, as in R), ``"running"`` or ``"swimming"``.
    ci_level
        Level of the interval, in (0, 1).

    Returns
    -------
    VPhys
        ``v_max`` (m/s), ``lower`` and ``upper`` (m/s), ``kmh``, ``mass``, ``mode``, ``ci_level``, ``reference``.

    References
    ----------
    Hirt, M. R., Jetz, W., Rall, B. C., Brose, U. (2017). A general scaling law reveals why the largest animals
    are not the fastest. Nature Ecology & Evolution 1, 1116-1122. doi:10.1038/s41559-017-0241-4

    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> v = tf.move.v_phys_estimate(5, "flying")         # a golden eagle
    >>> round(v.v_max, 2), round(v.lower, 2), round(v.upper, 2)
    (30.89, 0.9, 60.88)
    """
    mode = _mode_of(mode)
    scalar = np.ndim(mass) == 0
    m = np.atleast_1d(np.asarray(mass, dtype=np.float64))
    if m.size == 0 or not np.all(m > 0):  # NaN is not > 0
        raise ValueError("mass: positive numbers (kg)")
    if not isinstance(ci_level, (int, float, np.number)) or not 0 < ci_level < 1:
        raise ValueError("ci_level: a number in (0, 1)")
    if np.any((m < _HIRT_RANGE[0]) | (m > _HIRT_RANGE[1])):
        warnings.warn(f"mass outside the range of the Hirt et al. (2017) data ({_HIRT_RANGE[0]:g}, "
                      f"{_HIRT_RANGE[1]:g}) kg: the prediction is an extrapolation", UserWarning, stacklevel=2)
    v, lo, hi, kmh = _core.move_v_phys(np.ascontiguousarray(m), _MODES.index(mode), float(ci_level))
    if scalar:
        return VPhys(float(v[0]), float(lo[0]), float(hi[0]), float(kmh[0]), float(m[0]), mode, float(ci_level))
    return VPhys(v, lo, hi, kmh, m, mode, float(ci_level))


# ---------------------------------------------------------------------------------------------- inputs

def _seconds(values) -> np.ndarray:
    """Times as float seconds (as R's ``as.numeric`` of a POSIXct), NaN where missing."""
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        # the whole seconds and the fraction apart: ns / 1e9 would round twice near 1e9 s (POSIXct's doubles hold
        # about 2.4e-7 s there)
        ns = _nanoseconds(s)
        whole = np.floor_divide(ns, 1_000_000_000)
        out = whole.astype(np.float64) + (ns - whole * 1_000_000_000).astype(np.float64) / 1e9
        out[s.isna().to_numpy()] = np.nan
        return out
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        out = s.to_numpy("timedelta64[ns]").view(np.int64).astype(np.float64) / 1e9
        out[s.isna().to_numpy()] = np.nan
        return out
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.to_numpy(np.float64, na_value=np.nan)
    raise TypeError("the time column: datetimes or numbers")


def _points(tracks, what: str):
    """x, y of tracks that are points with a location (the error of R's ``.reject_empty_geometry``)."""
    geom = getattr(tracks, "geometry", None)
    if geom is None:
        raise TypeError("tracks: a GeoDataFrame of points (tf.load_tracks, tf.move.as_tracks)")
    if len(geom) and (geom.is_empty.any() or geom.isna().any()):
        raise ValueError(f"some locations are empty: drop them before {what} "
                         "(tracks[~tracks.geometry.is_empty], or tf.move.filter_gps_quality)")
    if len(geom) and not (geom.geom_type == "Point").all():
        raise ValueError("the geometry must be points")
    x = np.ascontiguousarray(geom.x.to_numpy(np.float64))
    y = np.ascontiguousarray(geom.y.to_numpy(np.float64))
    if len(x) and not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"some locations have non-finite coordinates: remove them before {what}")
    crs = tracks.crs
    return x, y, crs, bool(crs is not None and crs.is_geographic)


_DURATION_UNITS = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
                   "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
                   "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
                   "d": 86400.0, "day": 86400.0, "days": 86400.0,
                   "w": 604800.0, "week": 604800.0, "weeks": 604800.0,
                   "month": 2629800.0, "months": 2629800.0, "year": 31557600.0, "years": 31557600.0}


def _duration(value, name: str) -> float:
    """Seconds of a number (seconds), a timedelta or a text (``"45 mins"``, lubridate's; ``"45min"``)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}: a number of seconds, a time difference or a text such as '45 mins'")
    if isinstance(value, (pd.Timedelta, datetime.timedelta, np.timedelta64)):   # np.timedelta64 is an np.integer
        out = pd.Timedelta(value).total_seconds()
    elif isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
    elif isinstance(value, str):
        hit = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)\s*", value)
        if hit and hit.group(2).lower() in _DURATION_UNITS:
            out = float(hit.group(1)) * _DURATION_UNITS[hit.group(2).lower()]
        else:
            try:
                out = pd.Timedelta(value).total_seconds()
            except ValueError as e:
                raise ValueError(f"{name}: cannot read {value!r} as a duration (e.g. '45 mins')") from e
    else:
        raise TypeError(f"{name}: a number of seconds, a time difference or a text such as '45 mins'")
    if not np.isfinite(out):
        raise ValueError(f"{name}: a finite duration")
    return out


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: a number")
    return float(value)


# ------------------------------------------------------------------------------------------- speed cap

def flag_speed(tracks, v_max: float | None = None, *, mass: float | None = None, mode: str | None = None,
               jitter: float | None = None, remove: bool = False, time: str | None = None,
               track: str | None = None, n_threads: int | None = None):
    """Flag the locations next to a step faster than a speed limit (move2utils' ``mt_flag_speed_cap``, hard cap).

    The speed of a step is its length over its time lag (metres per second): on a sphere of 6 371 008.8 m
    (haversine) for longitude / latitude, the straight line in the CRS's units otherwise. A location is flagged
    when the step leaving it or the step arriving at it is faster than ``v_max``: from the data alone the
    detector cannot tell which end of the step is wrong, so it flags both. Locations without a time are left
    out of the track for this (their neighbours are joined), steps with a zero or negative lag have no speed.
    Only the hard cap of ``mt_flag_speed_cap`` is here; its automatic thresholds are not.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, one row per location, each track's rows consecutive and ordered by time
        (:func:`terraflux.load_tracks`, :func:`as_tracks`), with no empty location (drop them first, e.g. with
        :func:`filter_gps_quality`).
    v_max
        The limit in m/s (in the CRS's units per second for a projected CRS). Or give ``mass`` and ``mode``,
        then it is :func:`v_phys_estimate`'s prediction, as move2utils' ``mt_clean_track`` takes it, with no
        margin added (the README of move2utils suggests about 25 % above for sprints: pass ``v_max`` for that).
    mass, mode
        Body mass (kg) and locomotion mode (``"flying"``, ``"running"``, ``"swimming"``) for the limit.
    jitter
        Also flag the locations next to a step shorter than this (metres or the CRS's units), the noise floor
        of the device: column ``is_step_below_jitter``.
    remove
        Return only the locations that are not flagged.
    time, track
        The columns of the time and of the track, if not recorded (:func:`set_columns`).
    n_threads
        Threads (default: most of the cores); the tracks are scored in parallel, the result does not depend on it.

    Returns
    -------
    GeoDataFrame
        A copy of ``tracks`` with the columns of move2utils: ``is_outlier`` (any flag), ``step_speed`` (the speed
        of the step leaving the location, NaN where there is none), ``is_speed_above_cap``, ``is_step_below_jitter``
        (with ``jitter``), ``flagged_by_speed`` (the same as ``is_outlier``) and ``loglr_speed`` (log(speed / the
        least flagged speed) of the track, one-sided: not below 0). ``is_speed_above_cap`` and
        ``is_step_below_jitter`` are pandas ``boolean`` (missing for the locations without a time). The limit
        used is in ``attrs["v_max_used"]`` (and ``attrs["threshold_type"] == "hard"``).

    References
    ----------
    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Hirt, M. R., Jetz, W., Rall, B. C., Brose, U. (2017). A general scaling law reveals why the largest animals
    are not the fastest. Nature Ecology & Evolution 1, 1116-1122. doi:10.1038/s41559-017-0241-4

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02", "2024-01-01 00:03"], utc=True)
    >>> pts = [Point(0, 0), Point(30, 0), Point(5000, 0), Point(5060, 0)]       # a jump of 5 km in a minute
    >>> tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=pts, crs=32723), "time", "id")
    >>> tf.move.flag_speed(tr, v_max=20)["is_outlier"].tolist()
    [False, True, True, False]
    >>> tf.move.flag_speed(tr, mass=5, mode="flying").attrs["v_max_used"] > 20
    True
    """
    if v_max is None:
        if mass is None or mode is None:
            raise ValueError("give v_max (m/s), or mass (kg) and mode")
        v_max = v_phys_estimate(mass, mode).v_max
    elif mass is not None or mode is not None:
        raise ValueError("give v_max, or mass and mode, not both")
    v_max = _number(v_max, "v_max")
    if not (np.isfinite(v_max) and v_max > 0):
        raise ValueError("v_max: a positive number (m/s)")
    if jitter is not None:
        jitter = _number(jitter, "jitter")
        if not (np.isfinite(jitter) and jitter >= 0):
            raise ValueError("jitter: a non-negative number (metres) or None")
    t, k = columns(tracks, time, track)
    x, y, _, lonlat = _points(tracks, "flagging speeds")
    code = track_codes(tracks[k])
    check_order(code, None)
    speed, above, below, loglr = _core.move_speed_cap(x, y, np.ascontiguousarray(_seconds(tracks[t])), code, lonlat,
                                                      v_max, np.nan if jitter is None else jitter,
                                                      resolve(n_threads))
    out = tracks.copy()
    out["is_outlier"] = (above == 1) | (below == 1)
    out["step_speed"] = speed
    out["is_speed_above_cap"] = _nullable(above)
    if jitter is not None:
        out["is_step_below_jitter"] = _nullable(below)
    out["flagged_by_speed"] = out["is_outlier"]
    out["loglr_speed"] = loglr
    out.attrs["v_max_used"] = v_max
    out.attrs["threshold_type"] = "hard"
    return out[~out["is_outlier"]] if remove else out


def _nullable(flags: np.ndarray) -> pd.Series:
    """0 / 1 flags as a boolean column, missing where -1."""
    return pd.array(np.where(flags < 0, pd.NA, flags == 1), dtype="boolean")


# ----------------------------------------------------------------------------------------- GPS quality

def _find_column(tracks, underscore_name: str) -> str | None:
    """Movebank's column ``gps_satellite_count`` by its names: ``gps_satellite_count``, ``gps-satellite-count``
    and ``gps:satellite-count`` (the one Movebank exports)."""
    dashed = underscore_name.replace("_", "-")
    for name in (underscore_name, dashed, re.sub(r"^([a-z]+)-", r"\1:", dashed)):
        if name in tracks.columns:
            return name
    return None


def gps_quality(tracks, sat_min: float | None = 5, dop_max: float | None = 10, hacc_max: float | None = 100, *,
                drop_empty: bool = True) -> np.ndarray:
    """Which locations have a reliable GPS geometry (move2utils' ``mt_filter_gps_quality``): True = kept.

    Too few satellites (a fix needs four, five is the usual floor), a high dilution of precision or a large
    horizontal-accuracy estimate put a fix far from the true position. The columns are Movebank's, found
    as ``gps:satellite-count``, ``gps:dop`` and ``eobs:horizontal-accuracy-estimate`` (also with underscores,
    ``gps_dop``, or dashes, ``gps-dop``: move2utils itself finds only these two forms, not the names
    Movebank exports, and so filters nothing from a Movebank file); a missing column skips its
    criterion, and so does ``None``. Values that are missing or not numbers do not drop a location.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points (:func:`terraflux.load_tracks`, :func:`as_tracks`).
    sat_min
        Keep locations with at least this many satellites (default 5).
    dop_max
        Keep locations with a dilution of precision of at most this (default 10).
    hacc_max
        Keep locations with a horizontal-accuracy estimate (eobs, metres) of at most this (default 100).
    drop_empty
        Drop the locations without a position (default), which no detector can score.

    Returns
    -------
    numpy.ndarray
        One boolean per location, True where kept. :func:`filter_gps_quality` applies it.

    References
    ----------
    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"], utc=True)
    >>> df = gpd.GeoDataFrame({"time": t, "id": "a", "gps:satellite-count": [9, 4, 7], "gps:dop": [1.2, 1.0, 14.0]},
    ...                       geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs=32723)
    >>> tf.move.gps_quality(tf.move.as_tracks(df, "time", "id"))
    array([ True, False, False])
    """
    keep = np.ones(len(tracks), dtype=bool)
    if drop_empty:
        geom = getattr(tracks, "geometry", None)
        if geom is None:
            raise TypeError("tracks: a GeoDataFrame of points (tf.load_tracks, tf.move.as_tracks)")
        keep &= ~(geom.is_empty.to_numpy(bool) | geom.isna().to_numpy(bool))
    for column, limit, low in (("gps_satellite_count", sat_min, True), ("gps_dop", dop_max, False),
                               ("eobs_horizontal_accuracy_estimate", hacc_max, False)):
        if limit is None:
            continue
        col = _find_column(tracks, column)
        if col is None:
            continue
        v = pd.to_numeric(tracks[col], errors="coerce").astype("float64").to_numpy()
        with np.errstate(invalid="ignore"):  # a missing value compares False: it does not drop the location
            keep &= ~((v < limit) if low else (v > limit))
    return keep


def filter_gps_quality(tracks, sat_min: float | None = 5, dop_max: float | None = 10, hacc_max: float | None = 100,
                       *, drop_empty: bool = True):
    """``tracks`` without the locations of an unreliable GPS geometry (see :func:`gps_quality`).

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"], utc=True)
    >>> df = gpd.GeoDataFrame({"time": t, "id": "a", "gps:satellite-count": [9, 4, 7], "gps:dop": [1.2, 1.0, 14.0]},
    ...                       geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs=32723)
    >>> len(tf.move.filter_gps_quality(tf.move.as_tracks(df, "time", "id")))
    1
    """
    return tracks[gps_quality(tracks, sat_min, dop_max, hacc_max, drop_empty=drop_empty)]


# ----------------------------------------------------------------------------------------- thinning

def _criterion(criterion: str) -> bool:
    if criterion not in ("closest", "first"):
        raise ValueError('criterion: "closest" or "first"')
    return criterion == "closest"


def thin_time(tracks, interval, tolerance=None, criterion: str = "closest", *, time: str | None = None,
              track: str | None = None, n_threads: int | None = None) -> np.ndarray:
    """Which locations to keep so that the kept ones of a track are about ``interval`` apart (move2utils'
    ``mt_thin_time``, the move2 version of ``move::thinTrackTime``).

    The longest chain of locations whose successive time lags are within ``interval`` +/- ``tolerance``: a
    dynamic programme over the locations sorted by time, f(i) = 1 + max f(j) over the earlier j with
    t_i - t_j in [interval - tolerance, interval + tolerance]. A gap longer than ``interval + tolerance``
    splits the track there and thinning never bridges it (the lag across it stays what it was). It is not
    :func:`per_interval`: the interval is a lag between kept locations, not a clock bucket. The cost is that
    of the locations inside the tolerance window of each location, so it is linear in the number of
    locations; tracks run in parallel.

    Parameters
    ----------
    tracks
        A GeoDataFrame with a time and a track column (:func:`terraflux.load_tracks`, :func:`as_tracks`); the
        geometry is not read. A track need not be sorted: its locations are taken in order of time (ties in
        their order), those without a time are never kept (but the lone location of a one-location track).
    interval
        The target lag: seconds, a ``pandas.Timedelta`` / ``datetime.timedelta``, or text (``"45 mins"``,
        ``"45min"``, ``"2 hours"``).
    tolerance
        Half-width of the accepted lags, in the same form (default ``interval / 10``; at most ``interval``).
    criterion
        Among chains of the same length: ``"closest"`` (default) the least total deviation from ``interval``,
        or ``"first"`` the one ending first. In both the last candidate in time wins the ties of each step.
    time, track
        The columns of the time and of the track, if not recorded (:func:`set_columns`).
    n_threads
        Threads (default: most of the cores); the result does not depend on it.

    Returns
    -------
    numpy.ndarray
        One boolean per location, True where kept. :func:`filter_thin_time` applies it.

    References
    ----------
    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime("2024-01-01") + pd.to_timedelta([0, 10, 20, 30, 40, 50, 60, 70, 80], unit="min")
    >>> tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=[Point(i, 0) for i in range(9)],
    ...                                         crs=32723), "time", "id")
    >>> tf.move.thin_time(tr, "30 min", "1 min").astype(int)      # every third location, 0, 30, 60
    array([1, 0, 0, 1, 0, 0, 1, 0, 0])
    """
    closest = _criterion(criterion)
    interval_s = _duration(interval, "interval")
    tolerance_s = interval_s / 10 if tolerance is None else _duration(tolerance, "tolerance")
    if interval_s <= 0:
        raise ValueError("interval: positive")
    if tolerance_s < 0:
        raise ValueError("tolerance: not negative")
    if tolerance_s > interval_s:
        raise ValueError("tolerance: not above the interval")
    t, k = columns(tracks, time, track)
    n = len(tracks)
    if n == 0:
        return np.zeros(0, dtype=bool)
    code = track_codes(tracks[k])
    secs = _seconds(tracks[t])
    order = np.lexsort((np.where(np.isnan(secs), np.inf, secs), code))   # by track, then time, NaN last (stable)
    sel = _core.move_thin_time(np.ascontiguousarray(secs[order]), np.ascontiguousarray(code[order]), interval_s,
                               tolerance_s, closest, resolve(n_threads))
    out = np.zeros(n, dtype=bool)
    out[order] = sel.astype(bool)
    return out


def filter_thin_time(tracks, interval, tolerance=None, criterion: str = "closest", **kwargs):
    """``tracks`` thinned to about ``interval`` between locations (see :func:`thin_time`).

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime("2024-01-01") + pd.to_timedelta([0, 10, 20, 30, 40, 50, 60, 70, 80], unit="min")
    >>> tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=[Point(i, 0) for i in range(9)],
    ...                                         crs=32723), "time", "id")
    >>> tf.move.filter_thin_time(tr, "30 min", "1 min")["time"].dt.strftime("%H:%M").tolist()
    ['00:00', '00:30', '01:00']
    """
    return tracks[thin_time(tracks, interval, tolerance, criterion, **kwargs)]


def thin_distance(tracks, distance: float, method: str = "step", tolerance: float | None = None,
                  criterion: str = "closest", *, track: str | None = None, n_threads: int | None = None) -> np.ndarray:
    """Which locations to keep so that the kept ones of a track are about ``distance`` metres of travel apart
    (move2utils' ``mt_thin_distance``).

    The distance of each step is :func:`distance`'s, in metres whatever the CRS's unit (on the sphere of S2
    for longitude / latitude, planar otherwise), added along the track in the order of the rows
    (R's ``cumsum``, in extended precision). ``method``:

    * ``"step"`` (default): the first location, then each one at which the travel since the last kept one
      reaches ``distance``; kept ones are at least ``distance`` apart, more where the step crossed it;
    * ``"interval"``: the longest chain whose successive travel is within ``distance`` +/- ``tolerance``,
      by the dynamic programme of :func:`thin_time` over the cumulative distance (a single step longer
      than ``distance + tolerance`` splits the track).

    Parameters
    ----------
    tracks
        A GeoDataFrame of points with a track column, with no empty location (drop them first, e.g. with
        :func:`filter_gps_quality`).
    distance
        The target travel between kept locations, metres (> 0).
    method
        ``"step"`` or ``"interval"``.
    tolerance
        ``"interval"`` only: half-width of the accepted travel, metres (default ``distance / 10``; at most
        ``distance``). A warning if given with ``"step"``, which ignores it.
    criterion
        ``"interval"`` only, as in :func:`thin_time`.
    track
        The column of the track, if not recorded (:func:`set_columns`).
    n_threads
        Threads (default: most of the cores); the result does not depend on it.

    Returns
    -------
    numpy.ndarray
        One boolean per location, True where kept. :func:`filter_thin_distance` applies it.

    References
    ----------
    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime("2024-01-01") + pd.to_timedelta(range(6), unit="min")
    >>> tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=[Point(60 * i, 0) for i in range(6)],
    ...                                         crs=32723), "time", "id")
    >>> tf.move.thin_distance(tr, 100).astype(int)       # 60 m a step: one in two
    array([1, 0, 1, 0, 1, 0])
    """
    if method not in ("step", "interval"):
        raise ValueError('method: "step" or "interval"')
    closest = _criterion(criterion)
    distance = _number(distance, "distance")
    if not (np.isfinite(distance) and distance > 0):
        raise ValueError("distance: a positive number (metres)")
    tol = 0.0
    if method == "interval":
        tol = distance / 10 if tolerance is None else _number(tolerance, "tolerance")
        if tol < 0:
            raise ValueError("tolerance: not negative")
        if tol > distance:
            raise ValueError("tolerance: not above the distance")
    elif tolerance is not None:
        warnings.warn('tolerance is ignored when method = "step"; it applies only to method = "interval"',
                      UserWarning, stacklevel=2)
    k = _find(tracks, track, _TRACK_NAMES, "track")
    n = len(tracks)
    if n == 0:
        return np.zeros(0, dtype=bool)
    x, y, crs, lonlat = _points(tracks, "thinning by distance")
    code = track_codes(tracks[k])
    order = np.argsort(code, kind="stable")        # the rows of each track, in their order
    threads = resolve(n_threads)
    step = _core.move_distances(np.ascontiguousarray(x[order]), np.ascontiguousarray(y[order]),
                                np.ascontiguousarray(code[order]), lonlat, threads)
    f = length_factor("m", crs, lonlat)
    if f != 1.0:
        step = step / f
    sel = _core.move_thin_distance(step, np.ascontiguousarray(code[order]), 0 if method == "step" else 1, distance,
                                   tol, closest, threads)
    out = np.zeros(n, dtype=bool)
    out[order] = sel.astype(bool)
    return out


def filter_thin_distance(tracks, distance: float, method: str = "step", tolerance: float | None = None,
                         criterion: str = "closest", **kwargs):
    """``tracks`` thinned by distance travelled (see :func:`thin_distance`).

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> t = pd.to_datetime("2024-01-01") + pd.to_timedelta(range(6), unit="min")
    >>> tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=[Point(60 * i, 0) for i in range(6)],
    ...                                         crs=32723), "time", "id")
    >>> len(tf.move.filter_thin_distance(tr, 100))
    3
    """
    return tracks[thin_distance(tracks, distance, method, tolerance, criterion, **kwargs)]


__all__ = ["VPhys", "filter_gps_quality", "filter_thin_distance", "filter_thin_time", "flag_speed", "gps_quality",
           "thin_distance", "thin_time", "v_phys_estimate"]
