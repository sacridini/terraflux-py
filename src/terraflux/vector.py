# SPDX-License-Identifier: GPL-3.0-or-later
"""Vector analysis as in PostGIS: predicates, selection by location, joins, nearest feature, overlay and buffers.

One function per operation, named as in PostGIS / GDAL::

    import terraflux as tf

    inside = tf.vector.within("points.gpkg", "parks.gpkg")        # points in a park
    near = tf.vector.dwithin(roads, schools, 500)                 # roads within 500 m of a school
    joined = tf.vector.join(points, parks, "within", how="first")  # the park's attributes on each point
    closest = tf.vector.nearest(houses, hospitals)                # the nearest hospital and its distance
    pieces = tf.vector.overlay(soils, parcels, "intersection")    # a polygon where a soil and a parcel meet
    owners = tf.vector.dissolve(parcels, by="owner")              # one feature per owner
    ranges = tf.vector.hull(sightings, by="animal")               # a convex hull per animal

Conventions, for every function:

- ``a`` is a path (read with :func:`terraflux.load_vector`), a GeoDataFrame or a GeoSeries. ``b`` is the same, or a
  shapely geometry, or a list of them (taken to be in the CRS of ``a``).
- When both CRSs are known and differ, ``b`` is reprojected to the CRS of ``a``. The result is in the CRS of ``a``.
- The result is a GeoDataFrame with the attributes and the geometry of ``a``, the index reset (0..n-1), in the order of
  ``a``. ``out`` (a path) also writes it with :func:`terraflux.write_vector`.
- An empty or missing geometry never satisfies a predicate (``disjoint`` included: an empty geometry is not kept).
- A predicate ``p`` is evaluated as ``p(a_i, b_j)``: the feature of ``a`` first. A feature of ``a`` is kept when it
  holds with **at least one** feature of ``b`` (``WHERE EXISTS (SELECT 1 FROM b WHERE p(a.geom, b.geom))``).
"""

from __future__ import annotations

import numpy as np

from ._vector import _as_geodataframe, _gpd, to_crs, write_vector
from ._vector_cluster import cluster, delaunay, voronoi
from ._vector_cover import coverage_node, coverage_simplify, coverage_validate, smooth
from ._vector_lines import (
    closest_point,
    locate,
    points_along,
    shortest_line,
    split_lines,
    substring,
)
from ._vector_shapes import (
    boundary,
    centroid,
    envelope,
    explode,
    hull,
    make_valid,
    measures,
    shape,
    simplify,
    split,
)

__all__ = [
    "PREDICATES",
    "boundary",
    "buffer",
    "centroid",
    "clip",
    "closest_point",
    "cluster",
    "contains",
    "contains_properly",
    "count",
    "coverage_node",
    "coverage_simplify",
    "coverage_validate",
    "covered_by",
    "covers",
    "crosses",
    "delaunay",
    "difference",
    "disjoint",
    "dissolve",
    "distance",
    "dwithin",
    "envelope",
    "equals",
    "explode",
    "hull",
    "identity",
    "intersection",
    "intersects",
    "join",
    "locate",
    "make_valid",
    "measures",
    "nearest",
    "overlaps",
    "overlay",
    "points_along",
    "relate",
    "select",
    "shape",
    "shortest_line",
    "simplify",
    "smooth",
    "split",
    "split_lines",
    "substring",
    "symmetric_difference",
    "touches",
    "union",
    "voronoi",
    "within",
]

PREDICATES = (
    "intersects",
    "within",
    "contains",
    "touches",
    "crosses",
    "overlaps",
    "covers",
    "covered_by",
    "contains_properly",
    "equals",
    "disjoint",
    "dwithin",
)
_TREE_PREDICATES = {"intersects", "within", "contains", "touches", "crosses", "overlaps", "covers", "covered_by",
                    "contains_properly"}


# ---------------------------------------------------------------------------------------------------- machinery


def _load(a, b):
    """a and b as GeoDataFrames, b in the CRS of a (index of both reset)."""
    ga = _as_geodataframe(a)
    gb = _as_geodataframe(b)
    if ga.crs is not None and gb.crs is not None and ga.crs != gb.crs:
        gb = to_crs(gb, ga.crs)
    return ga.reset_index(drop=True), gb.reset_index(drop=True)


def _geoms(gdf) -> np.ndarray:
    return np.asarray(gdf.geometry.values, dtype=object)


def _check(predicate, distance):
    if predicate not in PREDICATES:
        raise ValueError(f"unknown predicate {predicate!r}; use one of {', '.join(PREDICATES)}")
    if predicate == "dwithin":
        if distance is None:
            raise ValueError("predicate 'dwithin' needs a distance")
        if not np.isfinite(distance) or distance < 0:
            raise ValueError(f"distance must be a finite number >= 0, not {distance!r}")


def _present(geoms) -> np.ndarray:
    import shapely

    return ~(shapely.is_missing(geoms) | shapely.is_empty(geoms))


def _dwithin_pairs(tree, ga, gb, distance):
    import shapely

    if tuple(int(v) for v in shapely.__version__.split(".")[:2]) >= (2, 1):
        return tree.query(ga, predicate="dwithin", distance=distance)
    keep = np.flatnonzero(_present(ga))
    bounds = shapely.bounds(ga[keep])
    boxes = shapely.box(bounds[:, 0] - distance, bounds[:, 1] - distance, bounds[:, 2] + distance,
                        bounds[:, 3] + distance)
    ia, ib = tree.query(boxes)
    ia = keep[ia]
    ok = shapely.distance(ga[ia], gb[ib]) <= distance
    return np.vstack([ia[ok], ib[ok]])


def _pairs(ga, gb, predicate, distance=None):
    """The pairs (i, j), sorted by i then j, with predicate(a_i, b_j) (disjoint: the pairs that intersect)."""
    import shapely

    tree = shapely.STRtree(gb)
    if predicate == "dwithin":
        pairs = _dwithin_pairs(tree, ga, gb, float(distance))
    elif predicate in ("equals", "disjoint"):
        pairs = tree.query(ga, predicate="intersects")
        if predicate == "equals" and pairs.shape[1]:
            pairs = pairs[:, shapely.equals(ga[pairs[0]], gb[pairs[1]])]
    else:
        pairs = tree.query(ga, predicate=predicate)
    ia, ib = pairs[0].astype(np.int64), pairs[1].astype(np.int64)
    order = np.lexsort((ib, ia))
    return ia[order], ib[order]


def _write(result, out):
    if out is not None:
        write_vector(result, out)
    return result


# ------------------------------------------------------------------------------------------ select and friends


def relate(a, b, predicate="intersects", distance=None) -> np.ndarray:
    """Which features of ``a`` satisfy ``predicate`` with at least one feature of ``b``: a bool array, one per feature.

    This is what :func:`select` keeps. ``predicate`` is one of :data:`PREDICATES`; ``distance`` is needed by
    ``"dwithin"``. ``"disjoint"`` is True for the features of ``a`` that meet no feature of ``b`` (an empty or missing
    geometry is False). With an empty ``b``, everything is False (``disjoint``: True for the non-empty ones).

    PostGIS: ``EXISTS (SELECT 1 FROM b WHERE ST_<Predicate>(a.geom, b.geom))``.

    >>> tf.vector.relate(points, parks, "within")          # doctest: +SKIP
    array([ True, False,  True])
    """
    _check(predicate, distance)
    ga, gb = _load(a, b)
    return _relate(_geoms(ga), _geoms(gb), predicate, distance)


def _relate(ga, gb, predicate, distance):
    ia, _ = _pairs(ga, gb, predicate, distance)
    hit = np.zeros(len(ga), dtype=bool)
    hit[ia] = True
    if predicate == "disjoint":
        return _present(ga) & ~hit
    return hit


def count(a, b, predicate="intersects", distance=None) -> np.ndarray:
    """How many features of ``b`` satisfy ``predicate`` with each feature of ``a``: an int64 array, one per feature.

    ``"disjoint"`` counts the features of ``b`` that the feature does **not** meet (``len(b non-empty)`` minus the
    ones it intersects). An empty or missing geometry of ``a`` counts 0.

    PostGIS: ``SELECT count(b.*) FROM b WHERE ST_<Predicate>(a.geom, b.geom)`` per feature of ``a``.

    >>> tf.vector.count(parks, trees, "contains")          # doctest: +SKIP
    array([12,  0,  7])
    """
    _check(predicate, distance)
    ga, gb = _load(a, b)
    a_geoms, b_geoms = _geoms(ga), _geoms(gb)
    ia, _ = _pairs(a_geoms, b_geoms, predicate, distance)
    n = np.bincount(ia, minlength=len(a_geoms)).astype(np.int64)
    if predicate == "disjoint":
        n = np.where(_present(a_geoms), int(_present(b_geoms).sum()) - n, 0).astype(np.int64)
    return n


def distance(a, b) -> np.ndarray:
    """The distance from each feature of ``a`` to the nearest feature of ``b``: a float64 array, one per feature.

    In the units of the CRS of ``a`` (planar). 0 when they touch or overlap. NaN for an empty or missing geometry of
    ``a`` and when ``b`` has no geometry.

    PostGIS: ``min(ST_Distance(a.geom, b.geom))``; QGIS: "Distance to nearest hub" / "Join by nearest".

    >>> tf.vector.distance(houses, hospitals)              # doctest: +SKIP
    array([ 120.5, 3041.2,    0. ])
    """
    ga, gb = _load(a, b)
    return _nearest(_geoms(ga), _geoms(gb))[1]


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


def select(a, b, predicate="intersects", distance=None, invert=False, out=None):
    """The features of ``a`` for which ``predicate`` holds with at least one feature of ``b`` (selection by location).

    ``predicate`` is one of :data:`PREDICATES` (``intersects``, ``within``, ``contains``, ``touches``, ``crosses``,
    ``overlaps``, ``covers``, ``covered_by``, ``contains_properly``, ``equals``, ``disjoint``, ``dwithin``);
    ``distance`` is required by ``"dwithin"``. ``invert=True`` keeps the other features instead (including the empty
    and missing geometries, which never satisfy a predicate). ``"disjoint"`` keeps the features that meet no feature
    of ``b``. Returns a GeoDataFrame with the attributes of ``a``, in its order, the index reset.

    PostGIS: ``SELECT a.* FROM a WHERE [NOT] EXISTS (SELECT 1 FROM b WHERE ST_<Predicate>(a.geom, b.geom))``;
    QGIS: "Select by location" / "Extract by location".

    >>> tf.vector.select(points, parks, "within", invert=True)   # points outside every park   # doctest: +SKIP
    """
    _check(predicate, distance)
    ga, gb = _load(a, b)
    keep = _relate(_geoms(ga), _geoms(gb), predicate, distance)
    if invert:
        keep = ~keep
    result = ga[keep].reset_index(drop=True)
    return _write(result, out)


def within(a, b, *, out=None):
    """The features of ``a`` that lie inside at least one feature of ``b``.

    No point of the feature is outside the feature of ``b`` and their interiors meet, so a point on the edge of a
    polygon is **not** within it (use :func:`covered_by` to count the edge). A polygon within itself is.

    Short for ``select(a, b, "within")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Within(a.geom, b.geom)``; QGIS: *Select by location*, 'are within'.

    >>> inside = tf.vector.within(trees, parks)          # the trees in a park  # doctest: +SKIP
    """
    return select(a, b, "within", out=out)


def contains(a, b, *, out=None):
    """The features of ``a`` that contain at least one feature of ``b``.

    The converse of :func:`within`: no point of the feature of ``b`` is outside the feature of ``a`` and their
    interiors meet, so a polygon does **not** contain a point on its edge (use :func:`covers` for that).

    Short for ``select(a, b, "contains")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Contains(a.geom, b.geom)``; QGIS: *Select by location*, 'contain'.

    >>> with_trees = tf.vector.contains(parks, trees)    # the parks that hold a tree  # doctest: +SKIP
    """
    return select(a, b, "contains", out=out)


def intersects(a, b, *, out=None):
    """The features of ``a`` that share at least one point with a feature of ``b``.

    The broadest predicate: touching counts, as does lying inside. Its negation is :func:`disjoint`.

    Short for ``select(a, b, "intersects")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Intersects(a.geom, b.geom)``; QGIS: *Select by location*, 'intersect'.

    >>> crossed = tf.vector.intersects(parcels, roads)   # the parcels a road reaches  # doctest: +SKIP
    """
    return select(a, b, "intersects", out=out)


def touches(a, b, *, out=None):
    """The features of ``a`` that touch at least one feature of ``b``.

    They share boundary points but no interior point: two parcels with a common edge or corner, a line ending on
    a polygon's edge. Two points never touch (a point has no boundary).

    Short for ``select(a, b, "touches")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Touches(a.geom, b.geom)``; QGIS: *Select by location*, 'touch'.

    >>> neighbours = tf.vector.touches(parcels, parcel)  # the parcels next to one  # doctest: +SKIP
    """
    return select(a, b, "touches", out=out)


def crosses(a, b, *, out=None):
    """The features of ``a`` that cross at least one feature of ``b``.

    They share interior points, and what they share has fewer dimensions than the larger of the two: a road
    crossing a river (a point), a line passing through a polygon (part of the line, part of it outside).
    Two polygons never cross (see :func:`overlaps`).

    Short for ``select(a, b, "crosses")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Crosses(a.geom, b.geom)``; QGIS: *Select by location*, 'cross'.

    >>> bridges = tf.vector.crosses(roads, rivers)       # the roads over a river  # doctest: +SKIP
    """
    return select(a, b, "crosses", out=out)


def overlaps(a, b, *, out=None):
    """The features of ``a`` that overlap at least one feature of ``b``.

    Both have the same dimension, they share part of their interiors, and neither is inside the other: two
    polygons partly on top of each other, two lines along a common stretch. A polygon inside another does not
    overlap it (that is :func:`within`).

    Short for ``select(a, b, "overlaps")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Overlaps(a.geom, b.geom)``; QGIS: *Select by location*, 'overlap'.

    >>> clashes = tf.vector.overlaps(claims, reserves)   # claims partly in a reserve  # doctest: +SKIP
    """
    return select(a, b, "overlaps", out=out)


def covers(a, b, *, out=None):
    """The features of ``a`` that cover at least one feature of ``b``.

    No point of the feature of ``b`` is outside the feature of ``a``; unlike :func:`contains`, a feature lying on
    the edge counts (a polygon covers a point on its edge, or its own boundary).

    Short for ``select(a, b, "covers")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Covers(a.geom, b.geom)``; QGIS: *Select by location*, 'cover'.

    >>> with_trees = tf.vector.covers(parks, trees)      # the edge counts  # doctest: +SKIP
    """
    return select(a, b, "covers", out=out)


def covered_by(a, b, *, out=None):
    """The features of ``a`` covered by at least one feature of ``b``.

    No point of the feature of ``a`` is outside the feature of ``b``; unlike :func:`within`, the edge counts (a
    point on the edge of a polygon is covered by it).

    Short for ``select(a, b, "covered_by")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_CoveredBy(a.geom, b.geom)``; QGIS: *Select by location*, 'are covered by'.

    >>> in_or_on = tf.vector.covered_by(trees, parks)    # the trees in a park or on its edge  # doctest: +SKIP
    """
    return select(a, b, "covered_by", out=out)


def contains_properly(a, b, *, out=None):
    """The features of ``a`` that contain at least one feature of ``b`` properly.

    The feature of ``b`` lies in the interior of the feature of ``a`` and does not touch its boundary: a
    polygon does not contain itself properly, nor a line that runs to its edge. Faster than :func:`contains`
    with many points against few polygons.

    Short for ``select(a, b, "contains_properly")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_ContainsProperly(a.geom, b.geom)``; QGIS: *Select by location*, 'contain properly'.

    >>> core = tf.vector.contains_properly(reserves, nests)  # off the edges  # doctest: +SKIP
    """
    return select(a, b, "contains_properly", out=out)


def equals(a, b, *, out=None):
    """The features of ``a`` that are topologically equal to at least one feature of ``b``.

    The same set of points, whatever the vertices: a ring may start elsewhere, run the other way or have an extra
    vertex along an edge. Exact coordinates count: a vertex moved by a rounding is another geometry.

    Short for ``select(a, b, "equals")``. :func:`select` also takes ``invert``, :func:`relate` gives
    one bool per feature and :func:`count` how many features of ``b``. An empty or missing geometry never
    qualifies. The result is a GeoDataFrame with the attributes of ``a`` in its order, the index reset.

    PostGIS: ``ST_Equals(a.geom, b.geom)``; QGIS: *Select by location*, 'are equal'.

    >>> unchanged = tf.vector.equals(parcels_2024, parcels_2020)  # shapes kept  # doctest: +SKIP
    """
    return select(a, b, "equals", out=out)


def disjoint(a, b, *, out=None):
    """The features of ``a`` that meet **no** feature of ``b`` (not ``intersects`` with any).

    Unlike the other predicates this is not "with at least one": it is the useful meaning, the features of ``a`` far
    from all of ``b`` (``NOT EXISTS ... ST_Intersects``). An empty or missing geometry of ``a`` is not kept; with an
    empty ``b`` every non-empty feature is.

    PostGIS: ``ST_Disjoint(a.geom, b.geom)`` in a ``NOT EXISTS``; QGIS: Select by location, 'are disjoint'.

    >>> tf.vector.disjoint(points, parks)          # points in no park   # doctest: +SKIP
    """
    return select(a, b, "disjoint", out=out)


def dwithin(a, b, distance, *, out=None):
    """The features of ``a`` within ``distance`` (in the CRS units, planar, inclusive) of at least one feature of ``b``.

    A feature at exactly ``distance`` qualifies. PostGIS: ``ST_DWithin(a.geom, b.geom, distance)``;
    QGIS: Select by location, 'are within a distance' (via a buffer). An empty or missing geometry never qualifies.

    >>> tf.vector.dwithin(roads, schools, 500)          # doctest: +SKIP
    """
    return select(a, b, "dwithin", distance=distance, out=out)


# ------------------------------------------------------------------------------------------------------- join


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


def join(a, b, predicate="intersects", how="first", fields=None, suffix="_2", distance=None, out=None):
    """Spatial join: attributes of ``b`` on the features of ``a`` that satisfy ``predicate`` with them.

    - ``how="first"``: one row per feature of ``a``, with the attributes of the matching feature of ``b`` of the
      **lowest index**; no match gives NaN (numbers) or None (texts).
    - ``how="all"``: one row per (a, b) pair, ordered by the index of ``a`` then of ``b``; a feature of ``a`` with no
      match appears once with missing values (a left join, as ``geopandas.sjoin(..., how="left")``).
    - ``how="summary"``: one row per feature of ``a`` with ``count`` (the matches) and, for each name in ``fields`` (a
      list or a comma-separated string), ``{f}_sum`` and ``{f}_mean`` over the matches (NaN values skipped; the sum is
      0 and the mean NaN with no valid value).

    The geometry column of ``b`` is dropped; a column of ``b`` that ``a`` already has gets ``suffix``. The geometry is
    that of ``a``. ``disjoint`` is not allowed (it has no matching feature to take attributes from). ``distance`` is
    needed by ``"dwithin"``.

    PostGIS: ``a LEFT JOIN b ON ST_<Predicate>(a.geom, b.geom)``; QGIS: "Join attributes by location" (one-to-one
    uses the first match, one-to-many all; summary is "Join attributes by location (summary)").

    >>> tf.vector.join(points, parks, "within", how="first")      # doctest: +SKIP
    >>> tf.vector.join(parks, trees, "contains", how="summary", fields="height")   # doctest: +SKIP
    """
    _check(predicate, distance)
    if predicate == "disjoint":
        raise ValueError("a join needs matching features: 'disjoint' is not allowed (use select or relate)")
    if how not in ("first", "all", "summary"):
        raise ValueError(f"unknown how {how!r}; use 'first', 'all' or 'summary'")
    import pandas as pd

    gpd = _gpd()
    ga, gb = _load(a, b)
    ia, ib = _pairs(_geoms(ga), _geoms(gb), predicate, distance)
    n = len(ga)
    bdf = pd.DataFrame(gb.drop(columns=gb.geometry.name))
    ren = _rename(list(bdf.columns), ga.columns, suffix)
    bdf = bdf.rename(columns=ren)

    if how == "summary":
        cols = {"count": np.bincount(ia, minlength=n).astype(np.int64)}
        if fields is None:
            fields = []
        elif isinstance(fields, str):
            fields = [f.strip() for f in fields.split(",") if f.strip()]
        for f in fields:
            if f not in ren:
                raise ValueError(f"field {f!r} is not a column of b")
            v = pd.to_numeric(bdf[ren[f]], errors="coerce").to_numpy(dtype=float)[ib]
            ok = ~np.isnan(v)
            s = np.bincount(ia[ok], weights=v[ok], minlength=n)
            k = np.bincount(ia[ok], minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"{f}_sum"] = s
                cols[f"{f}_mean"] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
        part = pd.DataFrame(_rename_dict(cols, ga.columns, suffix))
        left = ga
    else:
        if how == "first":
            first = np.r_[True, ia[1:] != ia[:-1]] if len(ia) else np.zeros(0, dtype=bool)
            sel_a, sel_b = ia[first], ib[first]
            pick = np.full(n, -1, dtype=np.int64)
            pick[sel_a] = sel_b
            rows_a, rows_b = np.arange(n), pick
        else:
            matched = np.zeros(n, dtype=bool)
            matched[ia] = True
            rest = np.flatnonzero(~matched)
            rows_a = np.r_[ia, rest]
            rows_b = np.r_[ib, np.full(len(rest), -1, dtype=np.int64)]
            order = np.lexsort((rows_b, rows_a))
            rows_a, rows_b = rows_a[order], rows_b[order]
        left = ga.iloc[rows_a].reset_index(drop=True)
        part = _take(bdf, rows_b)

    full = pd.concat([pd.DataFrame(left), part.reset_index(drop=True)], axis=1)
    result = gpd.GeoDataFrame(full, geometry=ga.geometry.name, crs=ga.crs)
    return _write(result, out)


def _rename_dict(cols, taken, suffix):
    ren = _rename(list(cols), taken, suffix)
    return {ren[k]: v for k, v in cols.items()}


def _knearest(ga, gb, k, max_distance, own):
    """The k nearest features of gb for each of ga: (i, j, distance, rank), one row per pair, sorted by i then rank.

    Exact: the k smallest by (``shapely.distance``, index of gb), found with an STRtree by querying ``dwithin`` with a
    distance that doubles until the k-th distance lies within it (or it reaches ``max_distance`` or covers all of gb).
    ``own`` leaves a feature out of its own neighbours (the same layer, by index). A feature of ga without neighbour
    is absent from the result.
    """
    import shapely

    live_a, live_b = np.flatnonzero(_present(ga)), np.flatnonzero(_present(gb))
    if not len(live_a) or not len(live_b):
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0), np.zeros(0, np.int64))
    tree = shapely.STRtree(gb)
    ba, bb = shapely.bounds(ga[live_a]), shapely.bounds(gb[live_b])
    lo = np.minimum(ba[:, :2].min(axis=0), bb[:, :2].min(axis=0))
    hi = np.maximum(ba[:, 2:].max(axis=0), bb[:, 2:].max(axis=0))
    span = float(np.hypot(*(hi - lo)))
    top = span + 1.0  # every feature of gb is within this of any of ga
    if max_distance is not None and max_distance < top:
        top = float(max_distance)
    d = np.full(len(live_a), min(top, max(span / np.sqrt(len(live_b)), 1e-9) if span > 0 else 1.0))
    ii, jj, dd, rr = [], [], [], []
    active = np.arange(len(live_a))
    while len(active):
        pairs = tree.query(ga[live_a[active]], predicate="dwithin", distance=d[active])
        pi, pj = active[pairs[0]], pairs[1].astype(np.int64)
        if own:
            ok = live_a[pi] != pj
            pi, pj = pi[ok], pj[ok]
        pd_ = shapely.distance(ga[live_a[pi]], gb[pj])
        order = np.lexsort((pj, pd_, pi))
        pi, pj, pd_ = pi[order], pj[order], pd_[order]
        start = np.searchsorted(pi, active)
        count = np.searchsorted(pi, active, side="right") - start
        kth = np.full(len(active), np.inf)
        full = count >= k
        kth[full] = pd_[start[full] + k - 1]
        done = (full & (kth <= d[active])) | (d[active] >= top)
        for n, a_ in enumerate(active):
            if done[n]:
                sl = slice(start[n], start[n] + min(count[n], k))
                m = len(pj[sl])
                ii.append(np.full(m, live_a[a_], dtype=np.int64))
                jj.append(pj[sl])
                dd.append(pd_[sl])
                rr.append(np.arange(1, m + 1, dtype=np.int64))
        active = active[~done]
        d[active] = np.minimum(d[active] * 2.0, top)
    i, j = np.concatenate(ii), np.concatenate(jj)
    dist, rank = np.concatenate(dd), np.concatenate(rr)
    if max_distance is not None:
        ok = dist <= max_distance
        i, j, dist, rank = i[ok], j[ok], dist[ok], rank[ok]
    order = np.lexsort((rank, i))
    return i[order], j[order], dist[order], rank[order]


def nearest(a, b=None, max_distance=None, k=1, distance_col="distance", suffix="_2", out=None):
    """Each feature of ``a`` with the attributes of the nearest feature of ``b`` and the distance to it.

    One row per feature of ``a``, in its order. The distance is planar, in the CRS units of ``a``, written in the
    column ``distance_col`` (0 when they touch). On a tie, the feature of ``b`` with the lowest index. With
    ``max_distance``, a feature with nothing within it (``distance <= max_distance`` is kept) gets missing values (NaN
    numbers, None texts) and a NaN distance; so does an empty or missing geometry of ``a`` or an empty ``b``. The
    geometry column of ``b`` is dropped; a column of ``b`` that ``a`` already has gets ``suffix``.

    ``k`` (an int >= 1, else ``ValueError``) asks for the ``k`` nearest features of ``b``: one row per pair, in the
    order of ``a`` and then by ``rank``, a new column (nullable ``Int64``, 1..k, ``<NA>`` for a feature without
    neighbour) that only exists when ``k > 1``. The neighbours of a feature are the ``k`` smallest by (``distance``,
    index of ``b``): at equal distance the lower index of ``b`` comes first. Fewer than ``k`` rows when ``b`` has
    fewer features or ``max_distance`` cuts them; a feature of ``a`` with no neighbour at all (empty ``b``, empty or
    missing geometry, nothing within ``max_distance``) keeps one row with missing values, a NaN distance and a
    missing rank. The distances are exactly ``shapely.distance``'s.

    ``b=None`` takes the layer itself: a feature is never its own neighbour (excluded by its index, not by geometry:
    a duplicate at distance 0 counts). The columns of the neighbour get ``suffix`` as all of them are in ``a``. With
    ``k=1`` this is FRAGSTATS's ENN, the Euclidean nearest-neighbour distance between patches (column ``distance``).

    PostGIS: ``ORDER BY a.geom <-> b.geom LIMIT k`` (a lateral join); GeoPandas: ``sjoin_nearest``;
    QGIS: "Join attributes by nearest".

    >>> tf.vector.nearest(houses, hospitals, max_distance=5000)      # doctest: +SKIP
    >>> tf.vector.nearest(patches, k=3)                              # doctest: +SKIP
    """
    import pandas as pd

    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be an integer >= 1, not {k!r}")
    gpd = _gpd()
    own = b is None
    if own:
        ga = _as_geodataframe(a).reset_index(drop=True)
        gb = ga
    else:
        ga, gb = _load(a, b)
    rank = None
    if k == 1 and not own:
        idx, dist = _nearest(_geoms(ga), _geoms(gb))
        if max_distance is not None:
            far = ~(dist <= max_distance)
            idx = np.where(far, -1, idx)
            dist = np.where(far, np.nan, dist)
        left = ga
    else:
        i, j, d, r = _knearest(_geoms(ga), _geoms(gb), int(k), max_distance, own)
        n = len(ga)
        has = np.zeros(n, dtype=bool)
        has[i] = True
        miss = np.flatnonzero(~has)  # a feature without neighbour: one row, missing values
        rows = np.concatenate([i, miss])
        order = np.lexsort((np.concatenate([r, np.zeros(len(miss), np.int64)]), rows))
        rows = rows[order]
        idx = np.concatenate([j, np.full(len(miss), -1, np.int64)])[order]
        dist = np.concatenate([d, np.full(len(miss), np.nan)])[order]
        rank = pd.array(np.concatenate([r, np.zeros(len(miss), np.int64)])[order], dtype="Int64")
        rank[idx < 0] = pd.NA
        left = ga.iloc[rows].reset_index(drop=True)
    bdf = pd.DataFrame(gb.drop(columns=gb.geometry.name))
    taken = set(ga.columns)
    bdf = bdf.rename(columns=_rename(list(bdf.columns), taken, suffix))
    part = _take(bdf, idx)
    dcol = distance_col
    while dcol in taken or dcol in part.columns:
        dcol = f"{dcol}{suffix}"
    part[dcol] = dist
    if k > 1:
        rcol = "rank"
        while rcol in taken or rcol in part.columns:
            rcol = f"{rcol}{suffix}"
        part[rcol] = rank
    full = pd.concat([pd.DataFrame(left), part], axis=1)
    result = gpd.GeoDataFrame(full, geometry=ga.geometry.name, crs=ga.crs)
    return _write(result, out)


# ---------------------------------------------------------------------------------------------------- overlay

_HOWS = ("intersection", "union", "difference", "symmetric_difference", "identity")
_POLYGONAL = ("Polygon", "MultiPolygon")
_DIM_NAMES = {0: "points", 1: "lines", 2: "polygons"}


def _dimension(geoms, what, allowed) -> int:
    """The one dimension (0 points, 1 lines, 2 polygons) of the non-empty geometries (2 when there are none)."""
    import shapely

    live = geoms[_present(geoms)]
    dims = {int(d) for d in np.unique(shapely.get_dimensions(live))} if len(live) else {2}
    if len(dims) != 1:
        raise ValueError(f"{what} mixes points, lines and polygons; it needs one kind")
    dim = dims.pop()
    if dim not in allowed:
        raise ValueError(f"{what} is {_DIM_NAMES[dim]}, it needs {' or '.join(_DIM_NAMES[d] for d in allowed)}")
    return dim


def _of_kind(g, dim):
    """The part of a geometry of a dimension (None when it has none): of a collection, its parts of that dimension
    (two polygons that overlap and also touch along an edge give a polygon and a line: the polygon is kept)."""
    import shapely

    if g is None or g.is_empty:
        return None
    simple, multi = (("Point", "MultiPoint"), ("LineString", "MultiLineString"), ("Polygon", "MultiPolygon"))[dim]
    if g.geom_type in (simple, multi):
        return g
    parts = [p for p in shapely.get_parts(g) if not p.is_empty and shapely.get_dimensions(p) == dim]
    parts = [q for p in parts for q in shapely.get_parts(p)]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return (shapely.multipoints, shapely.multilinestrings, shapely.multipolygons)[dim](parts)


def _less(g, others, dim):
    """g less the union of others (g itself when there are none; None when nothing of its kind is left)."""
    import shapely

    if len(others) == 0:
        return g
    return _of_kind(shapely.difference(g, shapely.union_all(others)), dim)


def _groups(key, n):
    """For each of 0..n-1, the positions where key (sorted ascending) equals it."""
    bounds = np.searchsorted(key, np.arange(n + 1))
    return [np.arange(bounds[k], bounds[k + 1]) for k in range(n)]


def _pieces(ga, gb, how):
    """The geometries of the overlay and, for each, the feature of a and of b it came from (-1: none)."""
    import shapely

    a_geoms, b_geoms = _geoms(ga), _geoms(gb)
    first = (0, 1, 2) if how in ("intersection", "difference", "identity") else (2,)
    dim = _dimension(a_geoms, "the first layer", first)
    _dimension(b_geoms, "the second layer", (2,))
    ia, ib = _pairs(a_geoms, b_geoms, "intersects")
    geoms, from_a, from_b = [], [], []

    def add(g, i, j):
        geoms.append(g)
        from_a.append(i)
        from_b.append(j)

    if how in ("intersection", "union", "identity"):
        cut = shapely.intersection(a_geoms[ia], b_geoms[ib]) if len(ia) else np.zeros(0, dtype=object)
        for c, i, j in zip(cut, ia, ib):
            g = _of_kind(c, dim)
            if g is not None:
                add(g, int(i), int(j))
    if how != "intersection":
        for i, m in enumerate(_groups(ia, len(a_geoms))):
            g = _less(a_geoms[i], b_geoms[ib[m]], dim)
            if g is not None:
                add(g, i, -1)
    if how in ("union", "symmetric_difference"):
        order = np.argsort(ib, kind="stable")
        for j, m in enumerate(_groups(ib[order], len(b_geoms))):
            g = _less(b_geoms[j], a_geoms[ia[order][m]], 2)
            if g is not None:
                add(g, -1, j)
    return geoms, np.asarray(from_a, dtype=np.int64), np.asarray(from_b, dtype=np.int64)


def overlay(a, b, how="intersection", suffix="_2", out=None):
    """Two layers overlaid: the pieces where their features meet, and the parts of one that the other does not cover.

    ``how``:

    - ``"intersection"``: a piece where a feature of ``a`` and one of ``b`` meet, with the attributes of both, in the
      order of ``a`` then ``b``. A column of ``b`` that ``a`` already has gets ``suffix``.
    - ``"union"``: those pieces, then each feature of ``a`` less the ``b`` it meets, then each feature of ``b`` less
      the ``a`` it meets. A remainder has only its own attributes (the other layer's are NaN / None).
    - ``"difference"``: each feature of ``a`` less the union of the ``b`` it meets, with the attributes of ``a`` only
      (a feature with nothing left is dropped).
    - ``"symmetric_difference"``: the remainders of ``a`` then those of ``b`` (the union without the pieces).
    - ``"identity"``: the intersection pieces, then the remainders of ``a`` (``b``'s attributes missing on them).

    ``b`` must be polygons. ``a`` may also be lines or points for ``intersection``, ``difference`` and ``identity``:
    the pieces keep its kind (the parts of a line inside a polygon, the points inside), as ``geopandas.overlay`` with
    ``keep_geom_type=True``; ``union`` and ``symmetric_difference`` need polygons in both. A piece that is not of the
    kind of ``a`` (two polygons that only share an edge) is dropped.

    The order is that of the app (the pieces by pair, then the remainders), not geopandas'. PostGIS:
    ``ST_Intersection``, ``ST_Difference``, ``ST_SymDifference`` and ``ST_Union`` over the pairs that ``ST_Intersects``;
    QGIS: "Intersection", "Difference", "Symmetrical difference", "Union", "Identity".

    >>> tf.vector.overlay(soils, parcels, "union")          # doctest: +SKIP
    """
    if how not in _HOWS:
        raise ValueError(f"unknown how {how!r}; use one of {', '.join(_HOWS)}")
    import pandas as pd

    gpd = _gpd()
    ga, gb = _load(a, b)
    geoms, from_a, from_b = _pieces(ga, gb, how)
    adf = pd.DataFrame(ga.drop(columns=ga.geometry.name))
    if how == "difference":
        part = _take(adf, from_a)
    else:
        bdf = pd.DataFrame(gb.drop(columns=gb.geometry.name))
        bdf = bdf.rename(columns=_rename(list(bdf.columns), adf.columns, suffix))
        part = pd.concat([_take(adf, from_a), _take(bdf, from_b)], axis=1)
    name = ga.geometry.name
    part[name] = gpd.GeoSeries(geoms, crs=ga.crs, index=part.index)
    result = gpd.GeoDataFrame(part, geometry=name, crs=ga.crs)
    return _write(result, out)


def intersection(a, b, suffix="_2", out=None):
    """The pieces where the features of ``a`` and ``b`` meet, with both attributes: ``overlay(a, b, "intersection")``.

    PostGIS: ``ST_Intersection(a.geom, b.geom)`` over the pairs that ``ST_Intersects``; QGIS: "Intersection".

    >>> tf.vector.intersection(soils, parcels)          # doctest: +SKIP
    """
    return overlay(a, b, "intersection", suffix, out)


def difference(a, b, suffix="_2", out=None):
    """Each feature of ``a`` less the ``b`` it meets, attributes of ``a``: ``overlay(a, b, "difference")``.

    PostGIS: ``ST_Difference(a.geom, ST_Union(b.geom))``; QGIS: "Difference".

    >>> tf.vector.difference(parcels, water)          # doctest: +SKIP
    """
    return overlay(a, b, "difference", suffix, out)


def symmetric_difference(a, b, suffix="_2", out=None):
    """What is in ``a`` or in ``b`` but not both: ``overlay(a, b, "symmetric_difference")``.

    PostGIS: ``ST_SymDifference``; QGIS: "Symmetrical difference".

    >>> tf.vector.symmetric_difference(before, after)          # doctest: +SKIP
    """
    return overlay(a, b, "symmetric_difference", suffix, out)


def identity(a, b, suffix="_2", out=None):
    """All of ``a``, cut where ``b`` meets it, with ``b``'s attributes there: ``overlay(a, b, "identity")``.

    QGIS: "Identity"; GeoPandas: ``overlay(how="identity")``.

    >>> tf.vector.identity(parcels, flood_zones)          # doctest: +SKIP
    """
    return overlay(a, b, "identity", suffix, out)


def union(a, b=None, suffix="_2", out=None):
    """With two layers, their overlay union; with one, all its features merged into one (``dissolve(a)``).

    PostGIS: ``ST_Union(a.geom, b.geom)`` per pair, and the aggregate ``ST_Union(geom)`` for one layer;
    QGIS: "Union" and "Dissolve".

    >>> tf.vector.union(parcels)                    # doctest: +SKIP
    >>> tf.vector.union(soils, parcels)             # doctest: +SKIP
    """
    if b is None:
        return dissolve(a, out=out)
    return overlay(a, b, "union", suffix, out)


# ------------------------------------------------------------------------------- clip, dissolve, buffer


def clip(a, mask, out=None):
    """Each feature of ``a`` cut to the area of ``mask`` (the union of its polygons); those outside it are left out.

    The attributes are kept, the order is that of ``a``. ``mask`` is polygons (any form ``b`` takes), reprojected to
    the CRS of ``a``.

    PostGIS: ``ST_Intersection(a.geom, ST_Union(mask.geom))`` where not empty; QGIS: "Clip"; GeoPandas: ``clip``.

    >>> tf.vector.clip(roads, county)          # doctest: +SKIP
    """
    import shapely

    ga, gm = _load(a, mask)
    m_geoms = _geoms(gm)
    _dimension(m_geoms, "the mask", (2,))
    a_geoms = _geoms(ga)
    raw = shapely.intersection(a_geoms, shapely.union_all(m_geoms))
    # each piece of the kind of its feature (a collection's parts of that kind, as the app)
    dims = shapely.get_dimensions(a_geoms)
    cut = np.array([None if g is None else _of_kind(g, int(d)) for g, d in zip(raw, dims)], dtype=object)
    keep = _present(cut)
    result = ga[keep].copy()
    result[ga.geometry.name] = cut[keep]
    return _write(result.reset_index(drop=True), out)


def dissolve(a, by=None, out=None):
    """The features merged: all into one, or one per value of the column ``by``, with its value and ``count``.

    The groups come in the order of their first feature; a missing value is a group of its own. The result has the
    column ``by`` (if given) and ``count`` (how many features were merged), and the geometry of the union.

    PostGIS: ``SELECT by, count(*), ST_Union(geom) FROM a GROUP BY by``; QGIS: "Dissolve"; GeoPandas: ``dissolve``.

    >>> tf.vector.dissolve(parcels, by="owner")          # doctest: +SKIP
    """
    import pandas as pd
    import shapely

    gpd = _gpd()
    ga = _as_geodataframe(a).reset_index(drop=True)
    if by:
        if by not in ga.columns:
            raise KeyError(f"the features have no attribute {by!r}")
        codes = pd.factorize(ga[by], use_na_sentinel=False)[0]
    else:
        codes = np.zeros(len(ga), dtype=int)
    members = [np.flatnonzero(codes == k) for k in range(codes.max() + 1 if len(codes) else 0)]
    data = {}
    if by:
        data[by] = ga[by].to_numpy()[np.array([m[0] for m in members], dtype=int)]
    data["count"] = np.array([len(m) for m in members], dtype=np.int64)
    geoms = _geoms(ga)
    merged = [shapely.union_all(geoms[m]) for m in members]
    result = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries(merged, crs=ga.crs), crs=ga.crs)
    return _write(result, out)


def buffer(a, distance, segments=16, dissolve=False, out=None):
    """The area within ``distance`` of every feature (``segments`` per quarter circle), its attributes kept.

    A negative distance shrinks polygons (and may empty them; the empty buffer is kept). With ``dissolve=True`` all
    the buffers are merged into one feature without attributes.

    PostGIS: ``ST_Buffer(geom, distance, 'quad_segs=16')`` (and ``ST_Union`` for ``dissolve``); QGIS: "Buffer".

    >>> tf.vector.buffer(wells, 250, dissolve=True)          # doctest: +SKIP
    """
    import shapely

    gpd = _gpd()
    segments = int(segments)
    if segments < 1 or not np.isfinite(distance):
        raise ValueError("a finite distance and segments >= 1")
    ga = _as_geodataframe(a).reset_index(drop=True)
    geoms = shapely.buffer(_geoms(ga), float(distance), quad_segs=segments)
    if dissolve:
        result = gpd.GeoDataFrame(geometry=gpd.GeoSeries([shapely.union_all(geoms)], crs=ga.crs), crs=ga.crs)
    else:
        result = ga.copy()
        result[ga.geometry.name] = geoms
    return _write(result, out)
