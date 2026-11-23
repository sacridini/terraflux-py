# SPDX-License-Identifier: GPL-3.0-or-later
"""Voronoi polygons, Delaunay triangles and distance clustering (DBSCAN) of :mod:`terraflux.vector`.

``a`` is a path, a GeoDataFrame or a GeoSeries; the result is a GeoDataFrame in the CRS of ``a``, the index reset.
``out`` (a path) also writes it with :func:`terraflux.write_vector`.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ._vector import _as_geodataframe, _gpd, to_crs, write_vector

__all__ = ["cluster", "delaunay", "voronoi"]


def _write(result, out):
    if out is not None:
        write_vector(result, out)
    return result


def _one(a):
    return _as_geodataframe(a).reset_index(drop=True).copy()


def _geoms(gdf) -> np.ndarray:
    return np.asarray(gdf.geometry.values, dtype=object)


def _present(geoms) -> np.ndarray:
    import shapely

    return ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))


def _shapely_at_least(major: int, minor: int) -> bool:
    import shapely

    return tuple(int(v) for v in shapely.__version__.split(".")[:2]) >= (major, minor)


def _set_geometry(gdf, geoms):
    name = gdf.geometry.name
    gdf[name] = _gpd().GeoSeries(np.asarray(geoms, dtype=object), index=gdf.index, crs=gdf.crs)
    return gdf


def _clip_geometry(clip, crs):
    """The union of the polygons of ``clip`` (path, frame, series, geometry or list), in ``crs``."""
    import shapely

    if isinstance(clip, shapely.geometry.base.BaseGeometry):
        geoms = np.array([clip], dtype=object)
    else:
        gdf = _as_geodataframe(clip).reset_index(drop=True)
        if crs is not None and gdf.crs is not None and gdf.crs != crs:
            gdf = to_crs(gdf, crs)
        geoms = _geoms(gdf)
    geoms = geoms[_present(geoms)]
    if len(geoms) == 0:
        return None
    types = set(shapely.get_type_id(geoms))
    if not types <= {3, 6, 7}:
        raise ValueError("clip must hold polygons")
    return shapely.union_all(geoms)


def voronoi(a, clip=None, tolerance=0.0, out=None):
    """The Voronoi polygon of each point: one polygon per point, with the attributes of ``a``, in its order.

    ``a`` must hold points (anything else is a ValueError). The extent is GEOS's default: the envelope of the points
    enlarged by GEOS's own margin, so the outer cells are finite; with ``clip`` (polygons) each cell is intersected
    with the union of the polygons of ``clip`` (reprojected to the CRS of ``a``) and a row whose cell becomes empty is
    kept with an empty geometry. Duplicate points share one GEOS cell: every duplicate row gets that same cell.
    Empty or missing points get an empty geometry, and so do the points of a set with a single distinct site (GEOS
    needs two).

    ``tolerance`` is GEOS's snapping tolerance. With shapely >= 2.1 on GEOS >= 3.12 the cells come in input order
    (``ordered=True``); otherwise each cell is matched to the point it contains (the same result).

    PostGIS ``ST_VoronoiPolygons``; QGIS *Voronoi polygons*.

    >>> tf.vector.voronoi("wells.gpkg", clip="area.gpkg", out="cells.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    g = _geoms(gdf)
    ok = _present(g)
    if ok.any() and not set(shapely.get_type_id(g[ok])) <= {0}:
        raise ValueError("voronoi needs points")
    cells = np.full(len(g), None, dtype=object)
    cells[:] = shapely.Polygon()
    idx = np.flatnonzero(ok)
    if len(idx):
        xy = shapely.get_coordinates(g[idx])
        uniq, first, inverse = np.unique(xy, axis=0, return_index=True, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        # distinct sites in the order of their first appearance
        order = np.argsort(first, kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        sites = uniq[order]
        if len(sites) >= 2:
            multi = shapely.multipoints(sites)
            site_cells = None
            if _shapely_at_least(2, 1):
                try:
                    res = shapely.voronoi_polygons(multi, tolerance=float(tolerance), ordered=True)
                    parts = shapely.get_parts(res)
                    if len(parts) == len(sites):
                        site_cells = parts
                except (TypeError, shapely.errors.GEOSException):
                    site_cells = None
            if site_cells is None:
                res = shapely.voronoi_polygons(multi, tolerance=float(tolerance))
                parts = shapely.get_parts(res)
                tree = shapely.STRtree(parts)
                pts = shapely.points(sites)
                ip, ic = tree.query(pts, predicate="intersects")
                site_cells = np.full(len(sites), shapely.Polygon(), dtype=object)
                seen = np.zeros(len(sites), dtype=bool)
                for p, c in sorted(zip(ip.tolist(), ic.tolist())):
                    if not seen[p]:
                        seen[p] = True
                        site_cells[p] = parts[c]
            cells[idx] = np.asarray(site_cells, dtype=object)[rank[inverse]]
    if clip is not None:
        region = _clip_geometry(clip, gdf.crs)
        if region is None:
            cells[:] = shapely.Polygon()
        else:
            present = _present(cells)
            cells[present] = shapely.intersection(cells[present], region)
    return _write(_set_geometry(gdf, cells), out)


def delaunay(a, edges=False, tolerance=0.0, out=None):
    """The Delaunay triangulation of all the vertices of ``a``: triangles, or the edges as lines with ``edges=True``.

    One feature per triangle (or edge), in the order GEOS returns the parts of its result, with no attributes
    besides the geometry; the CRS is that of ``a``. Fewer than three non-collinear vertices give no features.
    ``tolerance`` is GEOS's snapping tolerance. For ``n`` vertices with ``h`` on the convex hull there are
    ``2n - 2 - h`` triangles.

    PostGIS ``ST_DelaunayTriangles``; QGIS *Delaunay triangulation*.

    >>> tf.vector.delaunay("wells.gpkg", out="tin.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _as_geodataframe(a)
    g = _geoms(gdf)
    g = g[~shapely.is_missing(g)]
    xy = shapely.get_coordinates(g)
    parts = np.empty(0, dtype=object)
    if len(xy):
        res = shapely.delaunay_triangles(shapely.multipoints(xy), tolerance=float(tolerance), only_edges=bool(edges))
        parts = shapely.get_parts(res)
        parts = parts[~shapely.is_empty(parts)]
    result = _gpd().GeoDataFrame(geometry=_gpd().GeoSeries(parts, crs=gdf.crs))
    return _write(result, out)


def _neighbours(g, distance):
    """The sorted neighbour lists (itself included) of each feature, by the distance between geometries."""
    import shapely

    n = len(g)
    idx = np.flatnonzero(_present(g))
    sub = g[idx]
    tree = shapely.STRtree(sub)
    if _shapely_at_least(2, 1):
        pairs = tree.query(sub, predicate="dwithin", distance=distance)
    else:
        bounds = shapely.bounds(sub)
        boxes = shapely.box(bounds[:, 0] - distance, bounds[:, 1] - distance, bounds[:, 2] + distance,
                            bounds[:, 3] + distance)
        pairs = tree.query(boxes)
        keep = shapely.distance(sub[pairs[0]], sub[pairs[1]]) <= distance
        pairs = pairs[:, keep]
    ia, ib = idx[pairs[0]], idx[pairs[1]]
    order = np.lexsort((ib, ia))
    ia, ib = ia[order], ib[order]
    starts = np.searchsorted(ia, np.arange(n + 1))
    return [ib[starts[i]:starts[i + 1]] for i in range(n)]


def _dbscan(neigh, min_points):
    n = len(neigh)
    core = np.array([len(x) >= min_points for x in neigh], dtype=bool)
    label = np.full(n, -1, dtype=np.int64)
    next_id = 0
    for i in range(n):
        if label[i] != -1 or not core[i]:
            continue
        label[i] = next_id
        queue = deque([i])
        while queue:
            k = queue.popleft()
            for j in neigh[k]:
                if label[j] == -1:
                    label[j] = next_id
                    if core[j]:
                        queue.append(j)
        next_id += 1
    # ids already follow the lowest core member; renumber by the lowest member of any kind to honour the rule
    out = np.full(n, -1, dtype=np.int64)
    new = {}
    for i in range(n):
        c = label[i]
        if c >= 0:
            if c not in new:
                new[c] = len(new)
            out[i] = new[c]
    return out


def cluster(a, distance, min_points=1, column="cluster", out=None):
    """The features of ``a`` with an int64 column of cluster ids: DBSCAN on the distance between geometries.

    The neighbours of feature ``i`` are the features ``j`` (``i`` included) with ``distance(i, j) <= distance``
    (inclusive, as ``ST_DWithin``), for geometries of any type. A feature is a core when it has at least
    ``min_points`` neighbours (itself included). The algorithm is deterministic: features are visited in index order;
    at an unvisited core feature a new cluster starts and grows breadth first, the queue in FIFO order and the
    neighbours of each feature added in ascending index; a non-core feature reached by the growth and not yet in a
    cluster joins it (a border feature joins the first cluster that reaches it) but is not expanded; a feature in no
    cluster at the end is noise, ``-1``. The clusters are then numbered 0, 1, 2... in the order of their lowest member
    index. ``min_points=1`` is ``ST_ClusterWithin``: the connected components of the "within distance" graph, with no
    noise. Empty or missing geometries are noise. An existing ``column`` is overwritten.

    PostGIS ``ST_ClusterDBSCAN(geom, eps, minpoints)`` / ``ST_ClusterWithin``; QGIS *DBSCAN clustering*.

    >>> tf.vector.cluster("trees.gpkg", 50.0, min_points=4, out="clustered.gpkg")  # doctest: +SKIP
    """
    distance = float(distance)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError(f"distance must be a finite number >= 0, not {distance!r}")
    if int(min_points) != min_points or min_points < 1:
        raise ValueError(f"min_points must be an integer >= 1, not {min_points!r}")
    gdf = _one(a)
    g = _geoms(gdf)
    ids = _dbscan(_neighbours(g, distance), int(min_points))
    gdf[column] = ids
    return _write(gdf, out)
