# SPDX-License-Identifier: GPL-3.0-or-later
"""Vector analysis as in PostGIS: spatial predicates, selection by location, spatial joins and nearest feature.

One function per operation, named as in PostGIS / GDAL::

    import terraflux as tf

    inside = tf.vector.within("points.gpkg", "parks.gpkg")        # points in a park
    near = tf.vector.dwithin(roads, schools, 500)                 # roads within 500 m of a school
    joined = tf.vector.join(points, parks, "within", how="first")  # the park's attributes on each point
    closest = tf.vector.nearest(houses, hospitals)                # the nearest hospital and its distance

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

__all__ = [
    "PREDICATES",
    "contains",
    "contains_properly",
    "count",
    "covered_by",
    "covers",
    "crosses",
    "disjoint",
    "distance",
    "dwithin",
    "equals",
    "intersects",
    "join",
    "nearest",
    "overlaps",
    "relate",
    "select",
    "touches",
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


def _predicate_function(name, summary, postgis, qgis):
    def fn(a, b, *, out=None):
        return select(a, b, name, out=out)

    fn.__name__ = fn.__qualname__ = name
    fn.__doc__ = (
        f"The features of ``a`` {summary} at least one feature of ``b``.\n\n"
        f"PostGIS: ``{postgis}``; QGIS: Select by location, '{qgis}'.\n\n"
        "An empty or missing geometry never qualifies. The result is a GeoDataFrame with the attributes of ``a`` in\n"
        "its order, the index reset (see :func:`select` for ``invert`` and :func:`relate` for a bool array).\n\n"
        f">>> tf.vector.{name}(points, polygons)          # doctest: +SKIP\n"
    )
    return fn


within = _predicate_function("within", "lying completely inside (interiors meet, no point outside)",
                             "ST_Within(a.geom, b.geom)", "are within")
contains = _predicate_function("contains", "that completely contain", "ST_Contains(a.geom, b.geom)", "contain")
intersects = _predicate_function("intersects", "that share at least one point with", "ST_Intersects(a.geom, b.geom)",
                                 "intersect")
touches = _predicate_function("touches", "that touch (share boundary points, no interior points) with",
                              "ST_Touches(a.geom, b.geom)", "touch")
crosses = _predicate_function("crosses", "that cross (share interior points, in a lower dimension) with",
                              "ST_Crosses(a.geom, b.geom)", "cross")
overlaps = _predicate_function("overlaps",
                               "that overlap (same dimension, partly shared, neither inside the other) with",
                               "ST_Overlaps(a.geom, b.geom)", "overlap")
covers = _predicate_function("covers", "that cover (no point of b outside a; a point on a's edge counts) with",
                             "ST_Covers(a.geom, b.geom)", "cover")
covered_by = _predicate_function("covered_by", "covered by (no point outside; the edge counts) with",
                                 "ST_CoveredBy(a.geom, b.geom)", "are covered by")
contains_properly = _predicate_function("contains_properly",
                                        "that contain properly (b touches only the interior of a) with",
                                        "ST_ContainsProperly(a.geom, b.geom)", "contain properly")
equals = _predicate_function("equals", "topologically equal to (same point set, any vertex order or start) with",
                             "ST_Equals(a.geom, b.geom)", "are equal")


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


def nearest(a, b, max_distance=None, distance_col="distance", suffix="_2", out=None):
    """Each feature of ``a`` with the attributes of the nearest feature of ``b`` and the distance to it.

    One row per feature of ``a``, in its order. The distance is planar, in the CRS units of ``a``, written in the
    column ``distance_col`` (0 when they touch). On a tie, the feature of ``b`` with the lowest index. With
    ``max_distance``, a feature with nothing within it (``distance <= max_distance`` is kept) gets missing values (NaN
    numbers, None texts) and a NaN distance; so does an empty or missing geometry of ``a`` or an empty ``b``. The
    geometry column of ``b`` is dropped; a column of ``b`` that ``a`` already has gets ``suffix``.

    PostGIS: ``ORDER BY a.geom <-> b.geom LIMIT 1`` (a lateral join); GeoPandas: ``sjoin_nearest``;
    QGIS: "Join attributes by nearest".

    >>> tf.vector.nearest(houses, hospitals, max_distance=5000)      # doctest: +SKIP
    """
    import pandas as pd

    gpd = _gpd()
    ga, gb = _load(a, b)
    idx, dist = _nearest(_geoms(ga), _geoms(gb))
    if max_distance is not None:
        far = ~(dist <= max_distance)
        idx = np.where(far, -1, idx)
        dist = np.where(far, np.nan, dist)
    bdf = pd.DataFrame(gb.drop(columns=gb.geometry.name))
    taken = set(ga.columns)
    bdf = bdf.rename(columns=_rename(list(bdf.columns), taken, suffix))
    part = _take(bdf, idx)
    dcol = distance_col
    while dcol in taken or dcol in part.columns:
        dcol = f"{dcol}{suffix}"
    part[dcol] = dist
    full = pd.concat([pd.DataFrame(ga), part], axis=1)
    result = gpd.GeoDataFrame(full, geometry=ga.geometry.name, crs=ga.crs)
    return _write(result, out)
