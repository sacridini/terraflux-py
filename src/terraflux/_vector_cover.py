# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoothing and polygon coverages of :mod:`terraflux.vector`: Chaikin corner cutting, coverage simplification and
coverage validation.

``a`` is a path, a GeoDataFrame or a GeoSeries; the result is a GeoDataFrame in the CRS of ``a`` with its attributes,
the index reset. ``out`` (a path) also writes it with :func:`terraflux.write_vector`.
"""

from __future__ import annotations

import numpy as np

from ._vector import _as_geodataframe, _gpd, write_vector

__all__ = ["coverage_simplify", "coverage_validate", "smooth"]


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
