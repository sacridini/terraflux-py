# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometry operations of :mod:`terraflux.vector` (centroids, hulls, simplify, envelopes, validity, parts, boundaries
and measures), one result feature per input feature unless said otherwise.

``a`` is a path, a GeoDataFrame or a GeoSeries; the result is a GeoDataFrame in the CRS of ``a`` with its attributes,
the index reset. ``out`` (a path) also writes it with :func:`terraflux.write_vector`. Empty or missing geometries stay
empty.
"""

from __future__ import annotations

import numpy as np

from ._vector import _as_geodataframe, _gpd, to_crs, write_vector

__all__ = [
    "boundary",
    "centroid",
    "envelope",
    "explode",
    "hull",
    "make_valid",
    "measures",
    "simplify",
]


def _one(a):
    """a as a GeoDataFrame with the index reset (a copy: the input is never changed)."""
    return _as_geodataframe(a).reset_index(drop=True).copy()


def _write(result, out):
    if out is not None:
        write_vector(result, out)
    return result


def _replace(gdf, geoms, out):
    """The features of ``gdf`` (already a copy) with their geometry replaced by ``geoms`` (same order)."""
    name = gdf.geometry.name
    gdf[name] = _gpd().GeoSeries(np.asarray(geoms, dtype=object), index=gdf.index, crs=gdf.crs)
    return _write(gdf, out)


def _array(gdf) -> np.ndarray:
    return np.asarray(gdf.geometry.values, dtype=object)


def centroid(a, inside=False, out=None):
    """The centroid of each feature, as a point.

    With ``inside=True`` a point guaranteed to lie on the feature (GEOS's point on surface) instead: the centroid of
    a concave polygon (a "C", an "L") can fall outside it.

    PostGIS ``ST_Centroid`` / ``ST_PointOnSurface``; QGIS *Centroids* / *Point on surface*.

    >>> tf.vector.centroid("parks.gpkg", inside=True, out="labels.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    g = _array(gdf)
    return _replace(gdf, shapely.point_on_surface(g) if inside else shapely.centroid(g), out)


def _group_codes(values):
    """(codes, first) of the groups of ``values`` in the order of their first appearance (missing values together)."""
    import pandas as pd

    codes, _ = pd.factorize(values, use_na_sentinel=False)
    _, first = np.unique(codes, return_index=True)
    return codes, first


def hull(a, kind="convex", ratio=0.3, allow_holes=False, by=None, dissolve=False, out=None):
    """The convex or concave hull of each feature.

    Parameters
    ----------
    kind
        ``"convex"`` or ``"concave"``. The concave hull needs GEOS 3.11 (shapely 2.0 with it or later).
    ratio, allow_holes
        The concave hull's (``ratio`` 1 is the convex hull, 0 the tightest; ``allow_holes`` lets it have holes).
    by
        A column: one hull per value of it, over the union of the group's geometries, in the order of the first
        feature of each group. The result has the columns ``by`` and ``count`` (the features of the group), as
        ``vector.dissolve`` of the app.
    dissolve
        One hull of every feature (the only attribute is ``count``). Not together with ``by``.

    PostGIS ``ST_ConvexHull`` / ``ST_ConcaveHull``; QGIS *Convex hull* / *Concave hull* / *Minimum bounding geometry*.

    >>> tf.vector.hull("trees.gpkg", by="species", kind="concave", ratio=0.5)  # doctest: +SKIP
    """
    import shapely

    if kind not in ("convex", "concave"):
        raise ValueError(f"unknown hull kind {kind!r} (convex or concave)")
    if by is not None and dissolve:
        raise ValueError("by and dissolve cannot be used together")
    if kind == "concave" and not hasattr(shapely, "concave_hull"):
        raise RuntimeError("the concave hull needs shapely 2.1 or GEOS 3.11 or later")

    def one(g):
        if kind == "convex":
            return shapely.convex_hull(g)
        try:
            return shapely.concave_hull(g, ratio=ratio, allow_holes=allow_holes)
        except shapely.errors.UnsupportedGEOSVersionError as err:
            raise RuntimeError("the concave hull needs GEOS 3.11 or later") from err

    gdf = _one(a)
    g = _array(gdf)
    if by is None and not dissolve:
        return _replace(gdf, one(g), out)
    gpd = _gpd()
    name = gdf.geometry.name
    if dissolve:
        codes, first = np.zeros(len(gdf), dtype=np.int64), np.array([0] if len(gdf) else [], dtype=np.int64)
    else:
        if by not in gdf.columns:
            raise KeyError(f"the features have no column {by!r}")
        codes, first = _group_codes(gdf[by].to_numpy())
    ngroups = len(first)
    geoms = [shapely.union_all(g[codes == k]) for k in range(ngroups)]
    data = {} if dissolve else {by: gdf[by].to_numpy()[first]}
    data["count"] = np.bincount(codes, minlength=ngroups).astype(np.int64)
    data[name] = np.asarray(one(np.asarray(geoms, dtype=object)), dtype=object) if ngroups else []
    result = gpd.GeoDataFrame(data, geometry=name, crs=gdf.crs)
    return _write(result, out)


def simplify(a, tolerance, preserve_topology=True, out=None):
    """Each geometry simplified (Douglas-Peucker) with ``tolerance``, in the units of the CRS.

    With ``preserve_topology`` (the default) the result stays valid and no ring collapses or crosses another.

    PostGIS ``ST_SimplifyPreserveTopology`` / ``ST_Simplify``; QGIS *Simplify*.

    >>> tf.vector.simplify("coast.gpkg", 50, out="coast_50m.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    g = _array(gdf)
    return _replace(gdf, shapely.simplify(g, tolerance, preserve_topology=preserve_topology), out)


def envelope(a, out=None):
    """The bounding box of each feature, as a polygon.

    As GEOS does, the envelope of a point is the point itself, and of a horizontal or vertical line a line.

    PostGIS ``ST_Envelope``; QGIS *Bounding boxes*.

    >>> tf.vector.envelope("parcels.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    return _replace(gdf, shapely.envelope(_array(gdf)), out)


def make_valid(a, out=None):
    """Each invalid geometry repaired (GEOS's linework method: a bow-tie polygon becomes a multipolygon); valid ones
    are kept as they are.

    PostGIS ``ST_MakeValid``; QGIS *Fix geometries*.

    >>> tf.vector.make_valid("parcels.gpkg", out="parcels_ok.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    return _replace(gdf, shapely.make_valid(_array(gdf)), out)


def explode(a, out=None):
    """Multi-part features split into single parts, one feature per part with the attributes repeated, in the order
    of the features and, in each, of the parts. The index is reset; no column is added. Empty geometries are kept
    (one empty feature).

    PostGIS ``ST_Dump``; QGIS *Multipart to singleparts*.

    >>> tf.vector.explode("islands.gpkg", out="islands_single.gpkg")  # doctest: +SKIP
    """
    gdf = _one(a)
    result = gdf.explode(index_parts=False).reset_index(drop=True)
    return _write(result, out)


def boundary(a, out=None):
    """The boundary of each feature: the rings of a polygon as lines, the end points of a line as points (a closed
    line has none: an empty geometry); the boundary of a point is empty.

    PostGIS ``ST_Boundary``; QGIS *Boundary*.

    >>> tf.vector.boundary("parks.gpkg", out="park_edges.gpkg")  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    return _replace(gdf, shapely.boundary(_array(gdf)), out)


def measures(a, geodesic=False, out=None):
    """The features with the columns ``area``, ``perimeter``, ``length``, ``x`` and ``y``, as the app's
    ``vector.geometry`` (QGIS's *Add geometry attributes*), in the units of the CRS.

    ``area`` and ``perimeter`` are those of polygons (holes out of the area, their rings in the perimeter; 0 for
    other kinds), ``length`` that of lines (0 for other kinds), ``x`` and ``y`` the centroid (of the area, the length
    or the points; NaN when empty). With ``geodesic=True`` the area is in square metres on the WGS84 ellipsoid
    (the features are reprojected to EPSG:4326 when in another CRS), as GDAL's ``OGR_G_GeodesicArea``; the other
    columns stay in the units of the CRS.

    PostGIS ``ST_Area`` / ``ST_Perimeter`` / ``ST_Length`` / ``ST_X(ST_Centroid())``; QGIS *Add geometry attributes*.

    >>> tf.vector.measures("parks.gpkg", geodesic=True)  # doctest: +SKIP
    """
    import shapely

    gdf = _one(a)
    g = _array(gdf)
    kind = shapely.get_type_id(g)
    polygons, lines = np.isin(kind, (3, 6)), np.isin(kind, (1, 5))
    area = np.where(polygons, shapely.area(g), 0.0)
    c = shapely.centroid(g)
    if geodesic:
        if gdf.crs is None:
            raise ValueError("the geodesic area needs the CRS of the features")
        from pyproj import Geod

        geo = _array(to_crs(gdf, 4326)) if not gdf.crs.equals("EPSG:4326") else g
        geod = Geod(ellps="WGS84")
        area = np.array(
            [abs(geod.geometry_area_perimeter(x)[0]) if p and not shapely.is_empty(x) else 0.0
             for x, p in zip(geo, polygons)],
            dtype="float64",
        )
    gdf["area"] = area
    gdf["perimeter"] = np.where(polygons, shapely.length(g), 0.0)
    gdf["length"] = np.where(lines, shapely.length(g), 0.0)
    full = ~shapely.is_empty(c) & ~shapely.is_missing(c)
    x, y = np.full(len(g), np.nan), np.full(len(g), np.nan)
    x[full], y[full] = shapely.get_x(c[full]), shapely.get_y(c[full])
    gdf["x"], gdf["y"] = x, y
    return _write(gdf, out)
