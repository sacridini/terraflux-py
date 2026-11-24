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
from ._vector_lines import _expand, _load, _objects, _rename

__all__ = [
    "boundary",
    "centroid",
    "envelope",
    "explode",
    "hull",
    "make_valid",
    "measures",
    "shape",
    "simplify",
    "split",
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


# ------------------------------------------------------------------------------------------------------- split

_CUTTABLE = (1, 2, 3, 5, 6)  # LineString, LinearRing, Polygon, MultiLineString, MultiPolygon


def _blade_of(g):
    """The blade a geometry gives: a line as it is, a polygon its boundary (None for none)."""
    import shapely

    if g is None or shapely.is_empty(g):
        return None
    return shapely.boundary(g) if shapely.get_type_id(g) in (3, 6) else g


def _rank_order(pieces, size):
    """The order of pieces: the larger ``size`` first; on a tie, the larger y of the point on surface first, then the
    smaller x; then the position (stable)."""
    import shapely

    pos = shapely.point_on_surface(pieces)
    return np.lexsort((np.arange(len(pieces)), shapely.get_x(pos), -shapely.get_y(pos), -size))


def _cut_polygon(poly, blade):
    """The pieces of a Polygon cut by ``blade`` (as shapely.ops.split): the faces polygonized from the union of its
    boundary and the blade whose point on surface lies within the polygon."""
    import shapely

    faces = shapely.get_parts(shapely.polygonize([shapely.union(shapely.boundary(poly), blade)]))
    if len(faces) == 0:
        return [poly]
    pts = shapely.point_on_surface(faces)
    return [f for f, ok in zip(faces, shapely.contains(poly, pts)) if ok]


def split(a, lines, out=None):
    """Each polygon or line of ``a`` cut by the lines of ``lines``, one row per piece.

    ``lines`` holds lines or polygons (a polygon gives its boundary as blades); points in ``lines`` or in ``a`` raise
    ``ValueError``, as does a collection. For each feature of ``a`` the blade is the union of the features of
    ``lines`` that intersect it (an STRtree finds them; empty or missing ones are ignored).

    - A feature with no blade, or one that gives no more pieces than it has parts (a blade that only dangles inside a
      polygon, touches it from outside, or covers a line entirely; a multi-part feature none of whose parts is cut),
      is kept unchanged: one piece, its geometry as it was. When one part of a multi-part feature is cut, every part
      and piece is a row of its own.
    - A polygon is cut as ``shapely.ops.split`` does: the faces polygonized from the union of its boundary and the
      blade, keeping those whose point on surface lies within the polygon. A MultiPolygon is cut part by part (the
      same blade for every part). Holes are kept (a blade across a hole cuts through it).
    - A line is cut at its crossings with the blade: ``difference(line, blade)``, whose parts are the pieces (a
      stretch lying along a blade is removed, not cut). A MultiLineString is cut as a whole.
    - An empty or missing geometry of ``a`` is kept as one row: ``part`` 1, ``parts`` 1.

    Rows follow the order of ``a`` and, within a feature, ``part``. Each row has the attributes of the feature of
    ``a`` and the columns ``part`` (int, 1..n: 1 the largest piece by area for polygons, by length for lines; on a
    tie the one whose point on surface has the larger y first, then the smaller x, then the order GEOS gave them) and
    ``parts`` (int, n of that feature). A column ``a`` has with those names gets the suffix ``_2``.

    PostGIS ``ST_Split``; QGIS *Split with lines*; shapely ``shapely.ops.split``.

    >>> tf.vector.split("parcels.gpkg", "roads.gpkg", out="lots.gpkg")  # doctest: +SKIP
    """
    import shapely

    ga, gl = _load(a, lines)
    g = np.asarray(ga.geometry.values, dtype=object)
    bl = np.asarray(gl.geometry.values, dtype=object)
    for arr, what in ((g, "a"), (bl, "lines")):
        live = arr[~(shapely.is_missing(arr) | shapely.is_empty(arr))]
        kinds = shapely.get_type_id(live)
        if len(kinds) and not np.isin(kinds, _CUTTABLE).all():
            raise ValueError(f"{what} must hold lines or polygons (no points or collections)")
    blades = _objects([_blade_of(x) for x in bl])
    keep = np.flatnonzero([x is not None for x in blades])
    tree = shapely.STRtree(blades[keep]) if len(keep) else None
    rows, part, parts, geoms = [], [], [], []
    for i, geom in enumerate(g):
        pieces = None
        if geom is not None and not shapely.is_empty(geom) and tree is not None:
            hit = np.sort(keep[tree.query(geom, predicate="intersects")])
            if len(hit):
                blade = shapely.union_all(blades[hit])
                if shapely.get_type_id(geom) in (3, 6):
                    pieces = [p for part_ in shapely.get_parts(geom) for p in _cut_polygon(part_, blade)]
                    size = shapely.area(pieces)
                else:
                    pieces = list(shapely.get_parts(shapely.difference(geom, blade)))
                    size = shapely.length(pieces)
                if len(pieces) <= shapely.get_num_geometries(geom):  # no part was cut: the feature as it was
                    pieces = None
                else:
                    order = _rank_order(np.asarray(pieces, dtype=object), np.asarray(size))
                    pieces = [pieces[k] for k in order]
        if pieces is None:
            pieces = [geom]
        for k, p in enumerate(pieces):
            rows.append(i)
            part.append(k + 1)
            parts.append(len(pieces))
            geoms.append(p)
    cols = {"part": np.asarray(part, dtype=np.int64), "parts": np.asarray(parts, dtype=np.int64)}
    return _write(_expand(ga, rows, cols, geoms), out)


# ------------------------------------------------------------------------------------------------------- shape

_SHAPE_GEOMETRY = ("feature", "inscribed", "bounding", "rectangle")


def shape(a, tolerance=None, geometry="feature", out=None):
    """Shape descriptors of polygons, one row per feature of ``a`` (polygons and multipolygons only: other kinds
    raise ``ValueError``; empty or missing geometries give NaN in every column).

    New columns (float, in the units of the CRS; a name ``a`` already has gets the suffix ``_2``):

    - ``inscribed_radius``: the radius of the maximum inscribed circle (the length of the line
      ``shapely.maximum_inscribed_circle`` returns), the "depth" of the patch: the farthest any point is from its
      edge. ``tolerance`` stops the search when the area searched is smaller than it; ``None`` is shapely's default,
      ``max(width, height) / 1000`` of each feature's envelope. PostGIS ``ST_MaximumInscribedCircle``.
    - ``bounding_radius``: the radius of the minimum bounding circle (``shapely.minimum_bounding_radius``). PostGIS
      ``ST_MinimumBoundingCircle``.
    - ``circle``: FRAGSTATS's related circumscribing circle (McGarigal), ``1 - area / (pi * bounding_radius ** 2)``:
      0 for a circle, near 1 for a long thin patch (NaN when the radius is 0).
    - ``rect_length`` and ``rect_width``: the long and short sides of the minimum rotated rectangle
      (``shapely.oriented_envelope``; PostGIS ``ST_OrientedEnvelope``).
    - ``elongation``: ``1 - rect_width / rect_length``, 0 for a square, near 1 for an elongated patch (NaN when the
      length is 0).
    - ``orientation``: the direction of the long side in degrees clockwise from north (the +y axis), in [0, 180).
      With c0, c1, c2 the first coordinates of the rectangle's exterior as GEOS returns them, side A = c1 - c0 and
      side B = c2 - c1, the long side is A if |A| >= |B| else B, and ``orientation = degrees(atan2(dx, dy)) mod
      180`` of it (180 becomes 0). A degenerate rectangle GEOS gives as a line has A = the line and B = 0; as a point,
      length and width 0 and orientation NaN.

    ``geometry`` is what the result's geometry is: ``"feature"`` (default) the feature's own, ``"inscribed"`` the
    maximum inscribed circle as a polygon (the buffer of its centre by the radius, shapely's default ``quad_segs=8``),
    ``"bounding"`` the minimum bounding circle (``shapely.minimum_bounding_circle``), ``"rectangle"`` the oriented
    envelope. Anything else raises ``ValueError``.

    FRAGSTATS: the CIRCLE metric (McGarigal & Marks 1995); PostGIS: ``ST_MaximumInscribedCircle``,
    ``ST_MinimumBoundingCircle``, ``ST_OrientedEnvelope``.

    >>> tf.vector.shape("patches.gpkg", geometry="inscribed", out="cores.gpkg")  # doctest: +SKIP
    """
    import shapely

    if geometry not in _SHAPE_GEOMETRY:
        raise ValueError(f"unknown geometry {geometry!r} (one of {', '.join(_SHAPE_GEOMETRY)})")
    gdf = _one(a)
    g = _array(gdf)
    live = ~(shapely.is_missing(g) | shapely.is_empty(g))
    if live.any() and not np.isin(shapely.get_type_id(g[live]), (3, 6)).all():
        raise ValueError("shape needs polygons or multipolygons")
    n = len(g)
    names = ("inscribed_radius", "bounding_radius", "circle", "rect_length", "rect_width", "elongation", "orientation")
    cols = {k: np.full(n, np.nan) for k in names}
    geoms = g.copy()
    idx = np.flatnonzero(live)
    if len(idx):
        p = g[idx]
        mic = shapely.maximum_inscribed_circle(p, tolerance)
        radius = shapely.length(mic)
        brad = shapely.minimum_bounding_radius(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["circle"][idx] = np.where(brad > 0, 1.0 - shapely.area(p) / (np.pi * brad ** 2), np.nan)
        cols["inscribed_radius"][idx], cols["bounding_radius"][idx] = radius, brad
        rect = shapely.oriented_envelope(p)
        length, width, orient = np.full(len(p), np.nan), np.full(len(p), np.nan), np.full(len(p), np.nan)
        for k, r in enumerate(rect):
            xy = shapely.get_coordinates(r)
            if len(xy) < 2:
                length[k] = width[k] = 0.0
                continue
            ab = xy[1] - xy[0]
            bb = xy[2] - xy[1] if len(xy) > 2 else np.zeros(2)
            la, lb = np.hypot(*ab), np.hypot(*bb)
            long_, (dx, dy) = (la, ab) if la >= lb else (lb, bb)
            length[k], width[k] = long_, min(la, lb)
            o = np.degrees(np.arctan2(dx, dy)) % 180.0
            orient[k] = 0.0 if o >= 180.0 else o
        cols["rect_length"][idx], cols["rect_width"][idx], cols["orientation"][idx] = length, width, orient
        with np.errstate(divide="ignore", invalid="ignore"):
            cols["elongation"][idx] = np.where(length > 0, 1.0 - width / length, np.nan)
        if geometry == "inscribed":
            geoms[idx] = shapely.buffer(shapely.get_point(mic, 0), radius)
        elif geometry == "bounding":
            geoms[idx] = shapely.minimum_bounding_circle(p)
        elif geometry == "rectangle":
            geoms[idx] = rect
    ren = _rename(list(cols), gdf.columns, "_2")
    for k, v in cols.items():
        gdf[ren[k]] = v
    return _replace(gdf, geoms, out)
