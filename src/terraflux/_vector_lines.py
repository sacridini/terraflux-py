# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear referencing, shortest lines and closest points of :mod:`terraflux.vector`.

``a`` is a path, a GeoDataFrame or a GeoSeries; ``b`` / ``lines`` the same, a shapely geometry or a list, reprojected
to the CRS of ``a`` when both are known and differ. The result is a GeoDataFrame in the CRS of ``a``, the index reset.
``out`` (a path) also writes it with :func:`terraflux.write_vector`.

The arithmetic of linear referencing is fixed here (numpy, not shapely), so that other implementations can match it
bit for bit. A line has vertices p_0..p_n; the segment lengths are ``L_k = hypot(x_{k+1}-x_k, y_{k+1}-y_k)``; the
cumulative lengths are ``c_0 = 0``, ``c_{k+1} = c_k + L_k`` summed sequentially in vertex order. The point at measure
``m`` (``0 <= m <= c_n``) lies on the first segment ``k`` with ``c_k <= m <= c_{k+1}`` and ``L_k > 0`` (zero-length
segments are skipped): ``f = (m - c_k) / L_k``, ``x = x_k + f * (x_{k+1} - x_k)``, ``y`` likewise. ``m = 0`` gives p_0
and ``m = c_n`` gives p_n exactly. A MultiLineString is handled part by part, in order; measures restart at 0 on each.
"""

from __future__ import annotations

import numpy as np

from ._vector import _as_geodataframe, _gpd, to_crs, write_vector

__all__ = [
    "closest_point",
    "locate",
    "points_along",
    "shortest_line",
    "split_lines",
    "substring",
]

_LINE_TYPES = (1, 2, 5)  # LineString, LinearRing, MultiLineString


# ------------------------------------------------------------------------------------------------------ helpers


def _one(a):
    return _as_geodataframe(a).reset_index(drop=True)


def _load(a, b):
    """a and b as GeoDataFrames, b in the CRS of a (index of both reset)."""
    ga = _as_geodataframe(a)
    gb = _as_geodataframe(b)
    if ga.crs is not None and gb.crs is not None and ga.crs != gb.crs:
        gb = to_crs(gb, ga.crs)
    return ga.reset_index(drop=True), gb.reset_index(drop=True)


def _geoms(gdf) -> np.ndarray:
    return np.asarray(gdf.geometry.values, dtype=object)


def _objects(items) -> np.ndarray:
    arr = np.empty(len(items), dtype=object)
    for i, g in enumerate(items):
        arr[i] = g
    return arr


def _write(result, out):
    if out is not None:
        write_vector(result, out)
    return result


def _present(geoms) -> np.ndarray:
    import shapely

    return ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))


def _rename(names, taken, suffix):
    out, seen = {}, set(taken)
    for c in names:
        new = c
        while new in seen:
            new = f"{new}{suffix}"
        seen.add(new)
        out[c] = new
    return out


def _take(bdf, ib):
    """The rows ib of bdf (a plain DataFrame); -1 gives missing values: NaN for numbers, None for texts."""
    ib = np.asarray(ib, dtype=np.int64)
    if len(bdf.columns) == 0:
        return bdf.iloc[:0].reindex(range(len(ib)))
    miss = ib < 0
    taken = bdf.reindex(np.where(miss, -1, ib))
    taken = taken.reset_index(drop=True)
    if miss.any():
        for c in taken.columns:
            if taken[c].dtype == object:
                taken[c] = taken[c].astype(object).where(~miss, None)
    return taken


def _nearest(ga, gb):
    """(index of the nearest feature of b (lowest on ties, -1 for none), its distance (NaN for none)), per a."""
    import shapely

    n = len(ga)
    idx = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.nan)
    if n and _present(gb).any():
        tree = shapely.STRtree(gb)
        pairs, d = tree.query_nearest(ga, return_distance=True, all_matches=True)
        if pairs.shape[1]:
            ia, ib = pairs[0].astype(np.int64), pairs[1].astype(np.int64)
            order = np.lexsort((ib, ia))
            ia, ib, d = ia[order], ib[order], np.asarray(d)[order]
            first = np.r_[True, ia[1:] != ia[:-1]]
            idx[ia[first]] = ib[first]
            dist[ia[first]] = d[first]
    return idx, dist


def _check_lines(geoms, what):
    import shapely

    ok = _present(geoms)
    types = shapely.get_type_id(geoms)
    if np.any(ok & ~np.isin(types, _LINE_TYPES)):
        raise ValueError(f"{what} must be lines (LineString or MultiLineString)")


def _line_parts(geom):
    """The parts of a line geometry as (part index, coordinates (n, 2)); empty or missing parts are skipped."""
    import shapely

    if geom is None or shapely.is_empty(geom):
        return []
    out = []
    for i, p in enumerate(shapely.get_parts(geom)):
        if not shapely.is_empty(p):
            out.append((i, shapely.get_coordinates(p)))
    return out


class _Line:
    """One line part: vertices, segment lengths and cumulative lengths, and the point at a measure."""

    def __init__(self, xy):
        self.x = np.ascontiguousarray(xy[:, 0], dtype=np.float64)
        self.y = np.ascontiguousarray(xy[:, 1], dtype=np.float64)
        self.seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.c = np.concatenate([[0.0], np.cumsum(self.seg)])
        self.length = float(self.c[-1])

    def at(self, m):
        """The points (x, y arrays) at the measures m (clamped to [0, length])."""
        m = np.atleast_1d(np.asarray(m, dtype=np.float64))
        j = np.searchsorted(self.c[1:], m, side="left")
        j = np.minimum(j, len(self.seg) - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = (m - self.c[j]) / self.seg[j]
            xs = self.x[j] + f * (self.x[j + 1] - self.x[j])
            ys = self.y[j] + f * (self.y[j + 1] - self.y[j])
        xs = np.where(m >= self.length, self.x[-1], xs)
        ys = np.where(m >= self.length, self.y[-1], ys)
        xs = np.where(m <= 0, self.x[0], xs)
        ys = np.where(m <= 0, self.y[0], ys)
        return xs, ys

    def piece(self, ms, me):
        """The coordinates (k, 2) of the line between the measures ms <= me: the point at ms, the vertices strictly
        between, the point at me."""
        x0, y0 = self.at(ms)
        x1, y1 = self.at(me)
        inner = np.flatnonzero((self.c > ms) & (self.c < me))
        return np.vstack([[x0[0], y0[0]], np.column_stack([self.x[inner], self.y[inner]]), [x1[0], y1[0]]])


def _expand(gdf, rows, cols, geoms):
    """The features ``rows`` of gdf (repeated as needed) with the columns ``cols`` appended and the geometries."""
    gpd = _gpd()
    out = gdf.iloc[np.asarray(rows, dtype=np.int64)].reset_index(drop=True).copy()
    ren = _rename(list(cols), gdf.columns, "_2")
    for k, v in cols.items():
        out[ren[k]] = v
    name = gdf.geometry.name
    out[name] = gpd.GeoSeries(_objects(geoms), index=out.index, crs=gdf.crs)
    return gpd.GeoDataFrame(out, geometry=name, crs=gdf.crs)


def _lines_in(a):
    gdf = _one(a)
    g = _geoms(gdf)
    _check_lines(g, "a")
    return gdf, g


# -------------------------------------------------------------------------------------------- points and cuts


def points_along(a, spacing=None, n=None, start=0.0, out=None):
    """Points along each line, at a fixed spacing or in a fixed number.

    Exactly one of ``spacing`` and ``n``. With ``spacing`` the points lie at the measures ``start + k * spacing``
    (k = 0, 1, ... computed as that product) while they are <= the length of the part. With ``n`` (>= 1) at
    ``length * k / (n - 1)`` for k = 0..n-1 (``n = 1``: the midpoint). A MultiLineString is taken part by part and its
    measures restart at 0 on each. One feature per point, the attributes of ``a`` repeated, the columns ``part`` (the
    index of the part, int) and ``measure`` (float).

    PostGIS ``ST_LineInterpolatePoints``; QGIS *Points along geometry*.

    >>> tf.vector.points_along("roads.gpkg", spacing=100, out="stations.gpkg")  # doctest: +SKIP
    """
    import shapely

    if (spacing is None) == (n is None):
        raise ValueError("give exactly one of spacing and n")
    if spacing is not None and not (np.isfinite(spacing) and spacing > 0):
        raise ValueError(f"spacing must be a finite number > 0, not {spacing!r}")
    if n is not None and (int(n) != n or n < 1):
        raise ValueError(f"n must be an integer >= 1, not {n!r}")
    if not np.isfinite(start) or start < 0:
        raise ValueError(f"start must be a finite number >= 0, not {start!r}")
    gdf, g = _lines_in(a)
    rows, parts, meas, pts = [], [], [], []
    for i, geom in enumerate(g):
        for pi, xy in _line_parts(geom):
            ln = _Line(xy)
            if spacing is not None:
                kmax = int(np.floor((ln.length - start) / spacing)) + 2 if ln.length >= start else 0
                m = start + np.arange(max(kmax, 0) + 1, dtype=np.float64) * spacing
                m = m[m <= ln.length]
            elif n == 1:
                m = np.array([ln.length / 2])
            else:
                m = ln.length * np.arange(int(n), dtype=np.float64) / (int(n) - 1)
                m = np.minimum(m, ln.length)
            if not len(m):
                continue
            xs, ys = ln.at(m)
            pts.extend(shapely.points(xs, ys))
            rows.extend([i] * len(m))
            parts.extend([pi] * len(m))
            meas.extend(m.tolist())
    cols = {"part": np.asarray(parts, dtype=np.int64), "measure": np.asarray(meas, dtype=np.float64)}
    return _write(_expand(gdf, rows, cols, pts), out)


def _cut(geom, ms_of, me_of):
    """The geometry of a line cut part by part between ms_of(length) and me_of(length)."""
    import shapely

    pieces = []
    for _, xy in _line_parts(geom):
        ln = _Line(xy)
        ms, me = ms_of(ln.length), me_of(ln.length)
        pieces.append(shapely.linestrings(ln.piece(ms, me)))
    if not pieces:
        return geom
    return pieces[0] if len(pieces) == 1 else shapely.multilinestrings(pieces)


def substring(a, start, end, normalized=False, out=None):
    """Each line (each part of a MultiLineString) cut between the measures ``start`` and ``end``.

    With ``normalized=True`` they are fractions of the length of the part (``m = fraction * length``). Both are
    clamped to [0, length] part by part; ``start > end`` is a ValueError. The piece is the point at ``start``, every
    vertex strictly between, and the point at ``end``; a piece of zero length stays a 2-point LineString of equal
    points. A MultiLineString gives a MultiLineString when it has several parts. Empty or missing geometries stay as
    they are.

    PostGIS ``ST_LineSubstring``; QGIS *Line substring*.

    >>> tf.vector.substring("roads.gpkg", 0.25, 0.75, normalized=True)  # doctest: +SKIP
    """
    if not (np.isfinite(start) and np.isfinite(end)):
        raise ValueError("start and end must be finite numbers")
    if start > end:
        raise ValueError(f"start ({start!r}) must not be greater than end ({end!r})")
    gdf, g = _lines_in(a)

    def bound(v):
        if normalized:
            return lambda length: min(max(v * length, 0.0), length)
        return lambda length: min(max(float(v), 0.0), length)

    geoms = [_cut(geom, bound(start), bound(end)) for geom in g]
    return _write(_expand(gdf, np.arange(len(gdf)), {}, geoms), out)


def split_lines(a, length, out=None):
    """Each line cut into consecutive pieces of at most ``length``.

    Each part is cut at the measures ``k * length`` (the last piece is the shorter one; no piece has zero length), one
    feature per piece, the attributes of ``a`` repeated and the column ``part`` (int, the sequence number of the piece
    within its feature, counting over all its parts). A feature without length gives no piece.

    QGIS *Split lines by maximum length*; PostGIS: ``ST_LineSubstring`` over a ``generate_series``.

    >>> tf.vector.split_lines("roads.gpkg", 500, out="segments.gpkg")  # doctest: +SKIP
    """
    import shapely

    if not (np.isfinite(length) and length > 0):
        raise ValueError(f"length must be a finite number > 0, not {length!r}")
    gdf, g = _lines_in(a)
    rows, seq, geoms = [], [], []
    for i, geom in enumerate(g):
        k = 0
        for _, xy in _line_parts(geom):
            ln = _Line(xy)
            if ln.length <= 0:
                continue
            ks = np.arange(int(np.ceil(ln.length / length)) + 2, dtype=np.float64)
            ks = ks[ks * length < ln.length]
            for kk in ks:
                s, e = kk * length, min((kk + 1) * length, ln.length)
                geoms.append(shapely.linestrings(ln.piece(float(s), float(e))))
                rows.append(i)
                seq.append(k)
                k += 1
    cols = {"part": np.asarray(seq, dtype=np.int64)}
    return _write(_expand(gdf, rows, cols, geoms), out)


# ------------------------------------------------------------------------------------- nearest and closest


def locate(a, lines, out=None):
    """For each feature of ``a``, the nearest line of ``lines`` and the measure along it of the closest point.

    One row per feature of ``a`` with the columns ``line`` (the index into ``lines``, -1 for none; the lowest index
    on ties), ``measure`` (GEOS's ``line_locate_point`` of the point of the line closest to the feature: the second
    point of ``shortest_line``; for a MultiLineString GEOS measures along its parts in sequence) and ``distance``
    (planar, CRS units). An empty or missing geometry of ``a``, or no lines, gives -1 and NaN. A column of ``a`` with
    the same name gets the suffix ``_2``.

    PostGIS ``ST_LineLocatePoint``; QGIS *Locate points along lines* (plugin) / *Join by nearest*.

    >>> tf.vector.locate("stops.gpkg", "routes.gpkg")  # doctest: +SKIP
    """
    import shapely

    ga, gl = _load(a, lines)
    la = _geoms(gl)
    _check_lines(la, "lines")
    g = _geoms(ga)
    idx, dist = _nearest(g, la)
    n = len(ga)
    measure = np.full(n, np.nan)
    ok = idx >= 0
    if ok.any():
        sl = shapely.shortest_line(g[ok], la[idx[ok]])
        q = shapely.get_point(sl, 1)
        measure[ok] = shapely.line_locate_point(la[idx[ok]], q)
    cols = {"line": idx, "measure": measure, "distance": dist}
    return _write(_expand(ga, np.arange(n), cols, g), out)


def shortest_line(a, b, suffix="_2", out=None):
    """The line from each feature of ``a`` to its nearest feature of ``b``.

    One row per feature of ``a``: the geometry is ``shapely.shortest_line`` (GEOS nearest points) between the feature
    and the nearest feature of ``b`` (the lowest index on ties), with the attributes of ``a``, those of ``b`` (a
    column that ``a`` already has gets ``suffix``) and ``distance``. An empty or missing geometry of ``a``, or an
    empty ``b``, gives an empty line and missing values (NaN numbers, None texts).

    PostGIS ``ST_ShortestLine``; QGIS *Shortest line between features*.

    >>> tf.vector.shortest_line("houses.gpkg", "rivers.gpkg", out="links.gpkg")  # doctest: +SKIP
    """
    import pandas as pd
    import shapely

    gpd = _gpd()
    ga, gb = _load(a, b)
    g, h = _geoms(ga), _geoms(gb)
    idx, dist = _nearest(g, h)
    ok = idx >= 0
    geoms = np.full(len(ga), shapely.from_wkt("LINESTRING EMPTY"), dtype=object)
    if ok.any():
        geoms[ok] = shapely.shortest_line(g[ok], h[idx[ok]])
    bdf = pd.DataFrame(gb.drop(columns=gb.geometry.name))
    bdf = bdf.rename(columns=_rename(list(bdf.columns), ga.columns, suffix))
    ren = _rename(["distance"], list(ga.columns) + list(bdf.columns), suffix)
    left = pd.DataFrame(ga.drop(columns=ga.geometry.name)).reset_index(drop=True)
    full = pd.concat([left, _take(bdf, idx), pd.DataFrame({ren["distance"]: dist})], axis=1)
    name = ga.geometry.name
    full[name] = gpd.GeoSeries(geoms, index=full.index, crs=ga.crs)
    return _write(gpd.GeoDataFrame(full, geometry=name, crs=ga.crs), out)


def closest_point(a, b, max_distance=None, out=None):
    """Each point of ``a`` moved to the nearest point of the nearest feature of ``b``.

    ``a`` must be points. The target is the second point of ``shortest_line`` (a point on a segment, not only a
    vertex). With ``max_distance`` a point whose distance to the target (``hypot`` of the coordinate differences) is
    greater stays where it is. Columns ``moved`` (float: the distance moved, 0 for a point left as it is, with nothing
    to snap to, or empty) and ``snapped`` (bool: the point was set on a target, even when it already lay on it).

    PostGIS ``ST_ClosestPoint``; QGIS *Snap geometries to layer* (point to segment).

    >>> tf.vector.closest_point("gps.gpkg", "roads.gpkg", max_distance=30)  # doctest: +SKIP
    """
    import shapely

    ga, gb = _load(a, b)
    g, h = _geoms(ga), _geoms(gb)
    present = _present(g)
    if np.any(present & (shapely.get_type_id(g) != 0)):
        raise ValueError("a must be points")
    if max_distance is not None and not (max_distance >= 0):
        raise ValueError(f"max_distance must be a number >= 0, not {max_distance!r}")
    idx, _ = _nearest(g, h)
    idx = np.where(present, idx, -1)
    n = len(ga)
    moved = np.zeros(n)
    snapped = np.zeros(n, dtype=bool)
    geoms = g.copy()
    ok = idx >= 0
    if ok.any():
        sl = shapely.shortest_line(g[ok], h[idx[ok]])
        src = shapely.get_coordinates(shapely.get_point(sl, 0))
        dst = shapely.get_coordinates(shapely.get_point(sl, 1))
        d = np.hypot(dst[:, 0] - src[:, 0], dst[:, 1] - src[:, 1])
        take = np.ones(len(d), dtype=bool) if max_distance is None else d <= max_distance
        rows = np.flatnonzero(ok)[take]
        geoms[rows] = shapely.points(dst[take, 0], dst[take, 1])
        moved[rows] = d[take]
        snapped[rows] = True
    cols = {"moved": moved, "snapped": snapped}
    return _write(_expand(ga, np.arange(n), cols, geoms), out)
