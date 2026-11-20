# SPDX-License-Identifier: GPL-3.0-or-later
"""Corridors seen in a track: fast, straight segments among others going the same way (move2utils' ``mt_corridor``).

LaPoint et al.'s (2013) idea as R's move2utils (Kranstauber, Safi & Scharf 2026) computes it, the same numbers: a segment
is part of a corridor where the animal moves fast and, among the segments whose midpoints lie close to its own, the
directions agree.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve
from ._tracks import check_order, columns, lags, speed_factor, time_seconds, track_codes

__all__ = ["corridor", "corridor_lines"]

_LABELS = ["corridor", "not corridor"]


def quantile7(x, p: float) -> float:
    """R's ``quantile(x, p, type = 7)`` of values without missing ones (NaN for none)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0:
        return float("nan")
    index = 1 + max(n - 1, 0) * p
    lo = int(np.floor(index))
    hi = int(np.ceil(index))
    qs = float(x[lo - 1])
    if index > lo and x[hi - 1] != qs:
        h = index - lo
        qs = (1 - h) * qs + h * float(x[hi - 1])
    return qs


def _threshold(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name}: a number, or None for the quantile of the tracks")
    if not np.isfinite(value):
        raise ValueError(f"{name}: a finite number")
    return float(value)


def corridor(tracks, speed_threshold: float | None = None, circvar_threshold: float | None = None,
             min_segments: float = 2, *, time: str | None = None, track: str | None = None,
             n_threads: int | None = None):
    """Mark the segments of the tracks that run along a corridor (move2utils' ``mt_corridor``, LaPoint et al. 2013).

    A segment is the step from a location to the next of its track. It is a corridor segment when

    1. it is fast (its speed is at least ``speed_threshold``),
    2. the directions of the segments around it agree (the circular variance of their *pseudo-azimuths*,
       ``(2 * azimuth) mod 2 pi``, which makes two animals walking the same path in opposite directions agree, is at
       most ``circvar_threshold``): the neighbours are the segments, of all tracks together, whose midpoints lie
       within half its own length from its midpoint (the circle of that radius as GEOS draws it with ``sf::st_buffer``,
       a polygon of 120 vertices, the boundary included), itself included, and
    3. it has at least ``min_segments`` neighbours and strictly more neighbours that satisfy 1 and 2 than not.

    Where a threshold is left out, it is a quantile of the tracks: the 0.75 quantile of the speeds, the 0.25 quantile
    of the circular variances (R's ``quantile`` type 7), with a warning naming it. For corridors that can be compared
    between animals, give the thresholds, computed once on all of them. The search radius assumes a regular sampling
    interval (a warning where the interval is very irregular). Longitude / latitude is projected to a local azimuthal
    equidistant projection (:func:`aeqd_crs`, centred on the locations) for the neighbourhoods and the directions.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points, one row per location, each track's rows consecutive and ordered by time
        (:func:`terraflux.load_tracks`, :func:`as_tracks`). Empty locations are left out of the computation and
        come back with missing values.
    speed_threshold
        The speed (m/s) from which a segment is fast.
    circvar_threshold
        The circular variance, in [0, 1], up to which the directions around a segment agree.
    min_segments
        The fewest neighbours (counting all of them, the segment itself included) a corridor segment has.
    time, track
        The columns of the time and of the track, if not recorded (:func:`set_columns`).
    n_threads
        Threads (default: most of the cores); the same result with any number.

    Returns
    -------
    GeoDataFrame
        A copy of ``tracks`` with ``corridor`` (a categorical of ``"corridor"`` and ``"not corridor"``),
        ``corridor_speed`` (m/s), ``corridor_azimuth`` (degrees, on the working projection), ``corridor_circvar``
        (NaN with fewer than two neighbours) and ``corridor_n_neighbours`` (nullable integer): one value for the
        segment leaving each location, missing for the last location of a track and for the empty locations (their
        ``corridor`` is ``"not corridor"``). The thresholds used are in ``attrs["corridor"]``.

    References
    ----------
    LaPoint, S., Gallery, P., Wikelski, M., Kays, R. (2013). Animal behavior, cost-based corridor models, and real
    corridors. Landscape Ecology 28, 1615-1630. doi:10.1007/s10980-013-9910-0

    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> n = 20
    >>> t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    >>> a = [Point(10.0 * i, 0.3 * (i % 3)) for i in range(n)]          # a straight, fast walk
    >>> b = [Point(10.0 * i + 2.5, 0.5) for i in range(n)]              # another animal beside it
    >>> df = gpd.GeoDataFrame({"time": list(t) * 2, "id": ["a"] * n + ["b"] * n}, geometry=a + b, crs=32723)
    >>> out = tf.move.corridor(tf.move.as_tracks(df, "time", "id"), speed_threshold=0.1, circvar_threshold=0.1)
    >>> out["corridor"].value_counts().to_dict()
    {'corridor': 38, 'not corridor': 2}
    """
    speed_threshold = _threshold(speed_threshold, "speed_threshold")
    circvar_threshold = _threshold(circvar_threshold, "circvar_threshold")
    if isinstance(min_segments, (bool, np.bool_)) or not isinstance(min_segments, (int, float, np.integer, np.floating)):
        raise TypeError("min_segments: a number")
    t, k = columns(tracks, time, track)
    geom = getattr(tracks, "geometry", None)
    if geom is None:
        raise TypeError("tracks: a GeoDataFrame of points (tf.load_tracks, tf.move.as_tracks)")
    empty = (geom.is_empty | geom.isna()).to_numpy()
    keep = np.flatnonzero(~empty)
    if len(keep) < 2:
        raise ValueError("need at least 2 non-empty locations for corridor detection")
    if not (geom.iloc[keep].geom_type == "Point").all():
        raise ValueError("the geometry must be points")
    sub = tracks.iloc[keep]
    threads = resolve(n_threads)
    x = np.ascontiguousarray(sub.geometry.x.to_numpy(np.float64))
    y = np.ascontiguousarray(sub.geometry.y.to_numpy(np.float64))
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("some locations have non-finite coordinates: remove them first")
    code = track_codes(sub[k])
    check_order(code, time_seconds(sub[t]))
    crs = tracks.crs
    if crs is None:
        raise ValueError("the tracks have no CRS: the speeds (m/s) and the search radius need one (set_crs)")
    lonlat = bool(crs.is_geographic)

    # the segments: from each location to the next of its track
    n_kept = len(keep)
    has_next = np.zeros(n_kept, bool)
    has_next[:-1] = code[1:] == code[:-1]
    seg = np.flatnonzero(has_next)
    if len(seg) < 2:
        raise ValueError("need at least 2 segments across all tracks for corridor detection")
    dist = _core.move_distances(x, y, code, lonlat, threads)
    lag = lags(sub[t], code)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = dist / lag
    length, duration = speed_factor("m/s", crs, lonlat)
    if (length, duration) != (1.0, 1.0):
        v = v / length * duration
    speed = v[seg]
    # the buffer's radius is half the segment's length in the units of the working coordinates (metres for
    # longitude / latitude, the CRS's units otherwise)
    radius = dist[seg] / 2

    # irregular sampling: the search radius assumes roughly uniform sampling
    tl = lag[np.isfinite(lag) & (lag > 0)]
    if len(tl) >= 2:
        med = float(np.median(tl))
        iqr = quantile7(tl, 0.75) - quantile7(tl, 0.25)
        if med > 0 and iqr / med > 1:
            warnings.warn(f"Sampling interval is highly irregular (IQR / median = {iqr / med:.2g}). The half-segment-length "
                          "search radius assumes roughly uniform sampling; corridor flags on very short or very long "
                          "segments may be misleading.", stacklevel=2)

    # the working coordinates: a local projection for longitude / latitude
    if lonlat:
        from pyproj import Transformer

        from ._geometry import aeqd_crs

        work = aeqd_crs(sub)
        wx, wy = Transformer.from_crs(crs, work, always_xy=True).transform(x, y)
        wx, wy = np.ascontiguousarray(wx, dtype=np.float64), np.ascontiguousarray(wy, dtype=np.float64)
    else:
        wx, wy = x, y
    nb = _core.move_corridor_neighbourhoods(np.ascontiguousarray(wx[seg]), np.ascontiguousarray(wy[seg]),
                                            np.ascontiguousarray(wx[seg + 1]), np.ascontiguousarray(wy[seg + 1]),
                                            np.ascontiguousarray(radius), threads)
    circvar = nb["circvar"]
    n_nb = np.diff(nb["offset"])

    if speed_threshold is None:
        speed_threshold = quantile7(speed[~np.isnan(speed)], 0.75)
        warnings.warn(f"`speed_threshold` not supplied; using within-object 0.75 quantile of segment speed = "
                      f"{speed_threshold:.4g} m/s. For cross-individual or cross-population comparability, supply an "
                      "explicit value.", stacklevel=2)
    if circvar_threshold is None:
        valid = circvar[~np.isnan(circvar)]
        circvar_threshold = quantile7(valid, 0.25) if len(valid) >= 2 else float("nan")
        warnings.warn(f"`circvar_threshold` not supplied; using within-object 0.25 quantile of valid circular variance = "
                      f"{circvar_threshold:.4g}. For cross-individual or cross-population comparability, supply an "
                      "explicit value.", stacklevel=2)

    fast = ~np.isnan(speed) & (speed >= speed_threshold)
    if np.isnan(circvar_threshold):
        # R: cv <= NA is NA, so a fast segment with a variance makes `if (any(is_candidate))` fail
        if np.any(fast & ~np.isnan(circvar)):
            raise ValueError("the circular variance threshold is missing (fewer than two valid variances): "
                             "pass circvar_threshold")
        is_corridor = np.zeros(len(seg), bool)
    else:
        consistent = ~np.isnan(circvar) & (circvar <= circvar_threshold)
        candidate = fast & consistent
        if candidate.any():
            is_corridor = _core.move_corridor_flags(nb["offset"], nb["index"], candidate.astype(np.uint8),
                                                    float(min_segments), threads).astype(bool)
        else:
            is_corridor = np.zeros(len(seg), bool)

    # one value per input row
    n = len(tracks)
    label = np.zeros(n, bool)
    c_speed = np.full(n, np.nan)
    c_azimuth = np.full(n, np.nan)
    c_circvar = np.full(n, np.nan)
    rows = keep[seg]
    label[rows] = is_corridor
    c_speed[rows] = speed
    c_azimuth[rows] = nb["azimuth"] * 180 / np.pi
    c_circvar[rows] = circvar
    nb_full = np.zeros(n, dtype=np.int64)
    nb_full[rows] = n_nb
    mask = np.zeros(n, bool)
    mask[rows] = True
    c_nb = pd.array(np.where(mask, nb_full, 0), dtype="Int64")
    c_nb[~mask] = pd.NA
    out = tracks.copy()
    out["corridor"] = pd.Categorical(np.where(label, _LABELS[0], _LABELS[1]), categories=_LABELS)
    out["corridor_speed"] = c_speed
    out["corridor_azimuth"] = c_azimuth
    out["corridor_circvar"] = c_circvar
    out["corridor_n_neighbours"] = c_nb
    out.attrs["corridor"] = {"speed_threshold": float(speed_threshold), "circvar_threshold": float(circvar_threshold),
                             "min_segments": min_segments}
    return out


def corridor_lines(tracks, *, time: str | None = None, track: str | None = None):
    """The corridor segments of :func:`corridor` as lines.

    Each segment flagged ``"corridor"`` as a two-point line from its location to the next of its track (in the CRS
    of ``tracks``), with the track, the time of its start and the columns of :func:`corridor`
    (``speed``, ``azimuth``, ``circvar``, ``n_neighbours``).

    Examples
    --------
    >>> import terraflux as tf
    >>> import geopandas as gpd, pandas as pd
    >>> from shapely import Point
    >>> n = 20
    >>> t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    >>> a = [Point(10.0 * i, 0.3 * (i % 3)) for i in range(n)]
    >>> b = [Point(10.0 * i + 2.5, 0.5) for i in range(n)]
    >>> df = gpd.GeoDataFrame({"time": list(t) * 2, "id": ["a"] * n + ["b"] * n}, geometry=a + b, crs=32723)
    >>> out = tf.move.corridor(tf.move.as_tracks(df, "time", "id"), speed_threshold=0.1, circvar_threshold=0.1)
    >>> tf.move.corridor_lines(out).shape[0]
    38
    """
    import geopandas as gpd
    from shapely import linestrings

    if "corridor" not in tracks.columns:
        raise ValueError("tracks: the result of tf.move.corridor")
    t, k = columns(tracks, time, track)
    geom = tracks.geometry
    pos = np.flatnonzero(~(geom.is_empty | geom.isna()).to_numpy())
    flagged = np.flatnonzero((tracks["corridor"].astype(str) == _LABELS[0]).to_numpy())
    where = np.searchsorted(pos, flagged)
    ok = (where + 1 < len(pos))
    start, end = pos[where[ok]], pos[where[ok] + 1]
    gx, gy = geom.x.to_numpy(), geom.y.to_numpy()
    coords = np.stack([np.c_[gx[start], gy[start]], np.c_[gx[end], gy[end]]], axis=1)
    sel = tracks.iloc[start]
    data = {k: sel[k].to_numpy(), t: sel[t].to_numpy()}
    for name in ("speed", "azimuth", "circvar", "n_neighbours"):
        data[name] = sel[f"corridor_{name}"].to_numpy()
    return gpd.GeoDataFrame(data, geometry=linestrings(coords) if len(start) else [], crs=tracks.crs,
                            index=sel.index)
