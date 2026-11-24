# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoothing and polygon coverages of :mod:`terraflux.vector`: Chaikin corner cutting, the shared vertices of a
coverage, coverage simplification and coverage validation.

``a`` is a path, a GeoDataFrame or a GeoSeries; the result is a GeoDataFrame in the CRS of ``a`` with its attributes,
the index reset. ``out`` (a path) also writes it with :func:`terraflux.write_vector`.
"""

from __future__ import annotations

import numpy as np

from ._vector import _as_geodataframe, _gpd, write_vector

__all__ = ["coverage_node", "coverage_simplify", "coverage_validate", "smooth"]


def _one(a):
    """a as a GeoDataFrame with the index reset (a copy: the input is never changed)."""
    return _as_geodataframe(a).reset_index(drop=True).copy()


def _write(result, out):
    if out is not None:
        write_vector(result, out)
    return result


def _replace(gdf, geoms, out):
    name = gdf.geometry.name
    gdf[name] = _gpd().GeoSeries(np.asarray(geoms, dtype=object), index=gdf.index, crs=gdf.crs)
    return _write(gdf, out)


def _array(gdf) -> np.ndarray:
    return np.asarray(gdf.geometry.values, dtype=object)


def _cut(p: np.ndarray, q: np.ndarray):
    """The two points of Chaikin's cut of every segment p_i -> q_i, each coordinate as ``0.75 * a + 0.25 * b``."""
    return 0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q


def _chaikin_line(xy: np.ndarray, preserve_ends: bool) -> np.ndarray:
    n = len(xy)
    if n < 3:
        return xy
    q, r = _cut(xy[:-1], xy[1:])
    mid = np.empty((2 * (n - 1), 2))
    mid[0::2], mid[1::2] = q, r
    if preserve_ends:
        return np.vstack([xy[:1], mid, xy[-1:]])
    return mid


def _chaikin_ring(xy: np.ndarray) -> np.ndarray:
    if len(xy) < 4:
        return xy
    q, r = _cut(xy[:-1], xy[1:])  # every segment, the closing one included (the ring repeats its first vertex)
    m = len(q)
    out = np.empty((2 * m + 1, 2))
    out[0:2 * m:2], out[1:2 * m:2] = q, r
    out[-1] = out[0]
    return out


def _smooth_geom(g, iterations: int, preserve_ends: bool):
    import shapely

    if g is None or g.is_empty:
        return g
    kind = g.geom_type
    if kind in ("LineString", "LinearRing"):
        xy = shapely.get_coordinates(g)
        for _ in range(iterations):
            xy = _chaikin_line(xy, preserve_ends)
        return shapely.LineString(xy)
    if kind == "Polygon":
        def ring(r):
            xy = shapely.get_coordinates(r)
            for _ in range(iterations):
                xy = _chaikin_ring(xy)
            return xy

        return shapely.Polygon(ring(g.exterior), [ring(h) for h in g.interiors])
    if kind in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        parts = [_smooth_geom(p, iterations, preserve_ends) for p in g.geoms]
        return type(g)(parts)
    return g  # points and multipoints


def smooth(a, iterations=1, preserve_ends=True, out=None):
    """Lines and polygons smoothed by Chaikin's corner cutting, ``iterations`` times (1 to 10).

    Each iteration replaces every segment (p, q) by the two points 0.75 p + 0.25 q and 0.25 p + 0.75 q. An open
    line keeps its first and last vertex with ``preserve_ends`` (the default), else they are cut too; lines of fewer
    than 3 vertices stay as they are. Polygon rings (exteriors and holes) are smoothed all around, closing segment
    included, and ``preserve_ends`` does not apply to them: the ring starts at the first cut of its first segment.
    Points, multipoints and empty geometries are unchanged, multi-geometries are smoothed part by part. Z and M
    values are dropped. The vertex count doubles at each iteration and the area of a polygon shrinks, so the
    result may stop being valid for very narrow shapes.

    PostGIS ``ST_ChaikinSmoothing`` (which also keeps the line ends; its handling of the start point of a ring and
    of Z/M may differ from ours, which is fixed to this arithmetic so that the Terraflux app reproduces it bit for
    bit); QGIS *Smooth*.

    >>> tf.vector.smooth("coast.gpkg", iterations=2, out="coast_smooth.gpkg")  # doctest: +SKIP
    """
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or not 1 <= iterations <= 10:
        raise ValueError(f"iterations must be an integer from 1 to 10, got {iterations!r}")
    gdf = _one(a)
    geoms = [_smooth_geom(g, int(iterations), bool(preserve_ends)) for g in _array(gdf)]
    return _replace(gdf, geoms, out)


def _need_coverage(name):
    import shapely

    if not hasattr(shapely, name):
        raise RuntimeError(f"{name} needs shapely 2.1 (GEOS 3.12 or later)")
    return getattr(shapely, name)


def _polygons(gdf, what):
    g = _array(gdf)
    for x in g:
        if x is None or x.is_empty:
            continue
        if x.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"{what} needs polygons, got a {x.geom_type}")
    return g


def coverage_simplify(a, tolerance, preserve_boundary=False, out=None):
    """The polygons of a coverage simplified together (Visvalingam-Whyatt, GEOS), in the units of the CRS.

    Neighbours keep sharing their edges, so no gap or overlap opens, which separate simplification of each polygon
    (:func:`simplify`) cannot promise. ``a`` must be polygons forming a valid coverage (see
    :func:`coverage_validate`). With ``preserve_boundary`` the outer boundary of the coverage is left as it is and
    only the edges between polygons are simplified. Needs shapely 2.1 (GEOS 3.12).

    PostGIS ``ST_CoverageSimplify``; QGIS *Simplify coverage* (3.36+).

    >>> tf.vector.coverage_simplify("parcels.gpkg", 5, preserve_boundary=True)  # doctest: +SKIP
    """
    func = _need_coverage("coverage_simplify")
    gdf = _one(a)
    g = _polygons(gdf, "coverage_simplify")
    return _replace(gdf, func(g, tolerance, simplify_boundary=not preserve_boundary), out)


def coverage_validate(a, gap_width=0.0, out=None):
    """The edges of a polygon coverage that overlap or leave a gap with their neighbours, per feature.

    The result has the attributes of ``a``, a bool column ``valid`` (the feature has no invalid edge) and as
    geometry the invalid edges of the feature (lines; empty when valid). ``gap_width`` also flags edges closer than
    that to a neighbour (narrow gaps, in the units of the CRS). Needs shapely 2.1 (GEOS 3.12).

    PostGIS ``ST_CoverageInvalidEdges``; QGIS *Validate coverage*.

    >>> tf.vector.coverage_validate("parcels.gpkg", gap_width=0.5)  # doctest: +SKIP
    """
    import shapely

    func = _need_coverage("coverage_invalid_edges")
    gdf = _one(a)
    g = _polygons(gdf, "coverage_validate")
    edges = np.asarray(func(g, gap_width=gap_width), dtype=object)
    edges = np.array([shapely.LineString() if e is None else e for e in edges], dtype=object)
    gdf["valid"] = shapely.is_empty(edges)
    return _replace(gdf, edges, out)


def coverage_node(a, tolerance=0.0, out=None):
    """The polygons with the vertices they share with their neighbours: every vertex of another feature that lies
    on an edge of a polygon is added to that edge.

    The polygons of :func:`terraflux.vector.polygonize`-like tools (GDAL's polygonize, ``rasterio.features.shapes``)
    have only their own corners: where the junction of two patches meets the straight edge of a third, that edge has
    no vertex, and GEOS does not take the layer for a coverage (:func:`coverage_validate` flags those edges, and
    :func:`coverage_simplify` cannot keep the neighbours together). This adds the missing vertices, so that shared
    edges match vertex for vertex. Nothing else changes: the features, their order, their attributes, the start and
    orientation of every ring and every vertex already there; a column ``added`` tells how many vertices each feature
    gained. It does not remove overlaps or close gaps.

    A vertex ``v`` of another feature is added to the segment ``p -> q`` when it lies on it, strictly between its
    ends: the distance from ``v`` to the line through ``p`` and ``q`` is at most ``tolerance`` (``|cross| <=
    tolerance * |q - p|``, exactly 0 by default, which suits coordinates on a grid), its projection falls strictly
    inside the segment, and it is farther than ``tolerance`` from both ends. The added vertices are the neighbour's
    coordinates, in their order along the segment (repeated ones once). A small ``tolerance`` (in the units of the
    CRS) helps with coordinates that went through a reprojection; it moves an edge by at most that much.

    ``a`` must be polygons or multipolygons; empty and missing geometries are kept. The vertices of a feature are not
    added to its own edges (parts of one multipolygon that share edges stay as they are).

    Like PostGIS's ``ST_Snap(geom, neighbour, tolerance)`` restricted to adding vertices (no vertex moves); QGIS
    *Snap geometries to layer* with *Prefer closest point, insert extra vertices where required*.

    >>> cover = tf.vector.coverage_node("patches.gpkg")             # doctest: +SKIP
    >>> tf.vector.coverage_simplify(cover, 60)                      # doctest: +SKIP
    """
    import shapely

    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be a finite number >= 0, not {tolerance!r}")
    tol = float(tolerance)
    gdf = _one(a)
    g = _polygons(gdf, "coverage_node")
    n = len(g)
    added = np.zeros(n, dtype=np.int64)
    parts, part_feature = shapely.get_parts(g, return_index=True)
    rings, ring_part = shapely.get_rings(parts, return_index=True)
    xy, coord_ring = shapely.get_coordinates(rings, return_index=True)
    if len(xy) == 0:
        gdf["added"] = added
        return _write(gdf, out)
    coord_feature = part_feature[ring_part[coord_ring]]
    # the segments: a coordinate followed by one of the same ring; the vertices: every coordinate but a ring's last
    last = np.r_[coord_ring[1:] != coord_ring[:-1], True]
    seg = np.flatnonzero(~last)
    vert = seg
    p, q = xy[seg], xy[seg + 1]
    lo = np.minimum(p, q) - tol
    hi = np.maximum(p, q) + tol
    tree = shapely.STRtree(shapely.points(xy[vert]))
    iseg, ivert = tree.query(shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]))
    iv = vert[ivert]
    other = coord_feature[seg[iseg]] != coord_feature[iv]
    iseg, iv = iseg[other], iv[other]
    px, py = p[iseg, 0], p[iseg, 1]
    qx, qy = q[iseg, 0], q[iseg, 1]
    vx, vy = xy[iv, 0], xy[iv, 1]
    dx, dy = qx - px, qy - py
    wx, wy = vx - px, vy - py
    ux, uy = vx - qx, vy - qy
    cross = dx * wy - dy * wx
    dot = dx * wx + dy * wy
    len2 = dx * dx + dy * dy
    on = (
        (np.abs(cross) <= tol * np.sqrt(len2))
        & (dot > 0)
        & (dot < len2)
        & (wx * wx + wy * wy > tol * tol)
        & (ux * ux + uy * uy > tol * tol)
    )
    iseg, iv, dot = iseg[on], iv[on], dot[on]
    if len(iseg) == 0:
        gdf["added"] = added
        return _write(gdf, out)
    # along each segment by the projection, then x, then y; a coordinate repeated (two neighbours) once
    order = np.lexsort((xy[iv, 1], xy[iv, 0], dot, iseg))
    iseg, iv = iseg[order], iv[order]
    keep = np.r_[True, (iseg[1:] != iseg[:-1]) | np.any(xy[iv[1:]] != xy[iv[:-1]], axis=1)]
    iseg, iv = iseg[keep], iv[keep]
    # the new coordinates: each coordinate, then what is added after it (on the segment it starts), in that order
    at = seg[iseg]
    pos = np.r_[np.arange(len(xy)), at]
    sub = np.r_[np.zeros(len(xy), dtype=np.int64), np.arange(1, len(at) + 1)]
    new = np.lexsort((sub, pos))
    new_xy = np.r_[xy, xy[iv]][new]
    new_ring = np.r_[coord_ring, coord_ring[at]][new]
    np.add.at(added, coord_feature[at], 1)
    changed = np.zeros(len(rings), dtype=bool)
    changed[coord_ring[at]] = True
    rebuilt = shapely.linearrings(new_xy, indices=new_ring)
    new_rings = np.where(changed, rebuilt, rings)
    polys = np.array(parts, dtype=object)  # a part without a ring (an empty polygon) stays as it is
    made = shapely.polygons(new_rings, indices=ring_part)
    polys[: len(made)] = made
    geoms = np.array(g, dtype=object)
    for i in np.flatnonzero(added):
        mine = polys[part_feature == i]
        geoms[i] = mine[0] if g[i].geom_type == "Polygon" else shapely.multipolygons(mine)
    gdf["added"] = added
    return _replace(gdf, geoms, out)
