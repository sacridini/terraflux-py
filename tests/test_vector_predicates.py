# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.vector: predicates, selection by location, joins and nearest, against geopandas and by hand."""

import numpy as np
import pytest

import terraflux as tf

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, Point, Polygon, box

CRS = "EPSG:32633"
SJOIN = ["intersects", "within", "contains", "touches", "crosses", "overlaps", "covers", "covered_by",
         "contains_properly"]


def _layers():
    rng = np.random.default_rng(7)
    pts = gpd.GeoDataFrame({"pid": np.arange(200), "val": rng.random(200)},
                           geometry=[Point(x, y) for x, y in rng.random((200, 2)) * 100], crs=CRS)
    polys = []
    for i in range(30):
        x, y = rng.random(2) * 80
        w, h = 5 + rng.random(2) * 25
        p = box(x, y, x + w, y + h)
        if i % 4 == 0:
            p = Polygon(p.exterior.coords, [box(x + w * .3, y + h * .3, x + w * .6, y + h * .6).exterior.coords])
        polys.append(p)
    # a polygon equal to another with other vertices, and one that shares an edge with another
    polys.append(box(*polys[1].bounds))
    polys.append(box(polys[2].bounds[2], polys[2].bounds[1], polys[2].bounds[2] + 4, polys[2].bounds[3]))
    pg = gpd.GeoDataFrame({"gid": np.arange(len(polys)), "w": rng.random(len(polys)) * 10, "name": [f"p{i}" for i in range(len(polys))]},
                          geometry=polys, crs=CRS)
    lines = gpd.GeoDataFrame({"lid": np.arange(20)}, geometry=[
        LineString(rng.random((3, 2)) * 100) for _ in range(20)], crs=CRS)
    return pts, pg, lines


PTS, POLY, LINES = _layers()
PAIRS = [("pts", PTS, "poly", POLY), ("poly", POLY, "pts", PTS), ("lines", LINES, "poly", POLY),
         ("poly", POLY, "lines", LINES), ("lines", LINES, "lines", LINES), ("poly", POLY, "poly", POLY),
         ("pts", PTS, "lines", LINES)]


def _sj(a, b, pred, **kw):
    return gpd.sjoin(a[["geometry"]], b[["geometry"]], predicate=pred, how="left", **kw)


@pytest.mark.parametrize("pred", SJOIN)
@pytest.mark.parametrize("na,a,nb,b", PAIRS, ids=[f"{p[0]}-{p[2]}" for p in PAIRS])
def test_against_sjoin(pred, na, a, nb, b):
    sj = _sj(a, b, pred)
    idx = sj.index.to_numpy()
    j = sj["index_right"].to_numpy()
    expect = np.zeros(len(a), dtype=bool)
    expect[idx[~np.isnan(j)]] = True
    cnt = np.zeros(len(a), dtype=np.int64)
    np.add.at(cnt, idx[~np.isnan(j)], 1)
    assert (tf.vector.relate(a, b, pred) == expect).all()
    assert (tf.vector.count(a, b, pred) == cnt).all()
    sel = tf.vector.select(a, b, pred)
    assert sel["geometry"].geom_equals_exact(a[expect].geometry.reset_index(drop=True), 0).all()
    assert list(sel.index) == list(range(len(sel)))
    assert len(tf.vector.select(a, b, pred, invert=True)) == len(a) - expect.sum()
    # join all against sjoin left, with b's attributes
    ref = gpd.sjoin(a, b, predicate=pred, how="left").reset_index()
    ref = ref.sort_values(["index", "index_right"], kind="stable").reset_index(drop=True)
    got = tf.vector.join(a, b, pred, how="all")
    assert len(got) == len(ref)
    first_col = next(c for c in b.columns if c != "geometry")
    suff = first_col + "_2" if first_col in a.columns else first_col
    assert np.allclose(got[suff].to_numpy(dtype=float), ref[first_col if first_col not in a.columns else first_col + "_right"].to_numpy(dtype=float), equal_nan=True)


def test_first_summary_and_dwithin_against_sjoin():
    got = tf.vector.join(PTS, POLY, "within", how="first")
    ref = gpd.sjoin(PTS, POLY, predicate="within", how="left")
    ref = ref.reset_index().sort_values(["index", "index_right"]).drop_duplicates("index")
    assert len(got) == len(PTS)
    assert np.allclose(got["gid"].to_numpy(dtype=float), ref["gid"].to_numpy(dtype=float), equal_nan=True)
    assert any(v is None for v in got["name"]) == bool(np.isnan(got["gid"].to_numpy(dtype=float)).any())
    summ = tf.vector.join(POLY, PTS, "contains", how="summary", fields="val, pid")
    assert (summ["count"].to_numpy() == tf.vector.count(POLY, PTS, "contains")).all()
    for i in (0, 5, 11):
        m = PTS[POLY.geometry.iloc[i].contains(PTS.geometry)]
        assert summ["val_sum"].iloc[i] == pytest.approx(m["val"].sum())
        if len(m):
            assert summ["val_mean"].iloc[i] == pytest.approx(m["val"].mean())
        else:
            assert np.isnan(summ["val_mean"].iloc[i])
    if hasattr(gpd.GeoDataFrame, "sjoin"):
        ref = gpd.sjoin(PTS[["geometry"]], POLY[["geometry"]], predicate="dwithin", distance=7.5, how="left")
        ok = np.zeros(len(PTS), dtype=bool)
        ok[ref.index[~ref["index_right"].isna()]] = True
        assert (tf.vector.relate(PTS, POLY, "dwithin", distance=7.5) == ok).all()


def test_equals_disjoint_against_shapely():
    a = POLY
    got = tf.vector.relate(a, a, "equals")
    assert got.all()
    eq = tf.vector.count(a, a, "equals")
    assert eq[1] == 2 and eq[a.shape[0] - 2] == 2
    dj = tf.vector.relate(PTS, POLY, "disjoint")
    assert (dj == ~tf.vector.relate(PTS, POLY, "intersects")).all()
    nd = tf.vector.count(PTS, POLY, "disjoint")
    assert (nd == len(POLY) - tf.vector.count(PTS, POLY, "intersects")).all()


@pytest.mark.parametrize("na,a,nb,b", PAIRS[:4], ids=[f"{p[0]}-{p[2]}" for p in PAIRS[:4]])
def test_nearest_against_sjoin_nearest(na, a, nb, b):
    got = tf.vector.nearest(a, b)
    ref = gpd.sjoin_nearest(a, b, distance_col="d", how="left").reset_index()
    ref = ref.sort_values(["index", "index_right"], kind="stable").drop_duplicates("index")
    assert len(got) == len(a)
    assert np.allclose(got["distance"].to_numpy(), ref["d"].to_numpy(), atol=1e-12, rtol=0)
    col = next(c for c in b.columns if c != "geometry")
    rc = col if col not in a.columns else col + "_right"
    assert np.allclose(got[col if col not in a.columns else col + "_2"].to_numpy(dtype=float), ref[rc].to_numpy(dtype=float))
    assert np.allclose(tf.vector.distance(a, b), got["distance"].to_numpy(), atol=1e-12, rtol=0)


def test_nearest_tie_lowest_index_and_max_distance():
    a = gpd.GeoDataFrame({"k": [1]}, geometry=[Point(0, 0)], crs=CRS)
    b = gpd.GeoDataFrame({"v": [10, 20, 30], "t": ["x", "y", "z"]},
                         geometry=[Point(5, 0), Point(-1, 0), Point(1, 0)], crs=CRS)
    r = tf.vector.nearest(a, b)
    assert r["v"].iloc[0] == 20 and r["distance"].iloc[0] == 1.0
    b2 = b.iloc[[0, 2, 1]].reset_index(drop=True)
    b2.loc[1, "v"] = 99
    assert tf.vector.nearest(a, b2)["v"].iloc[0] == 99  # the lowest index among the tied
    far = tf.vector.nearest(a, b, max_distance=0.5)
    assert np.isnan(far["v"].iloc[0]) and far["t"].iloc[0] is None and np.isnan(far["distance"].iloc[0])
    assert tf.vector.nearest(a, b, max_distance=1.0)["v"].iloc[0] == 20
    e = tf.vector.nearest(a, b.iloc[:0])
    assert np.isnan(e["distance"].iloc[0]) and np.isnan(tf.vector.distance(a, b.iloc[:0])[0])


def test_column_suffix_and_missing_values():
    a = gpd.GeoDataFrame({"id": [1, 2], "name": ["a", "b"]}, geometry=[Point(1, 1), Point(50, 50)], crs=CRS)
    b = gpd.GeoDataFrame({"id": [7], "name": ["zone"], "n": [3]}, geometry=[box(0, 0, 2, 2)], crs=CRS)
    r = tf.vector.join(a, b, "within")
    assert list(r.columns) == ["id", "name", "geometry", "id_2", "name_2", "n"]
    assert r["id_2"].iloc[0] == 7 and np.isnan(r["id_2"].iloc[1])
    assert r["name_2"].iloc[1] is None and r["n"].iloc[0] == 3
    assert r.crs == a.crs and isinstance(r, gpd.GeoDataFrame)
    # how="all": a left join, a without match appears once
    p = gpd.GeoDataFrame({"q": [1, 2]}, geometry=[box(0, 0, 3, 3), box(0, 0, 4, 4)], crs=CRS)
    allr = tf.vector.join(a, p, "within", how="all")
    assert list(allr["q"].fillna(-1)) == [1, 2, -1]
    assert list(allr["id"]) == [1, 1, 2]


def test_hand_cases():
    sq = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=CRS)
    on_edge = [Point(10, 5)]
    pe = gpd.GeoDataFrame(geometry=on_edge, crs=CRS)
    assert len(tf.vector.within(pe, sq)) == 0
    assert len(tf.vector.covered_by(pe, sq)) == 1
    assert len(tf.vector.intersects(pe, sq)) == 1
    assert len(tf.vector.touches(pe, sq)) == 1
    assert len(tf.vector.contains(sq, pe)) == 0 and len(tf.vector.covers(sq, pe)) == 1
    inner = gpd.GeoDataFrame(geometry=[Point(5, 5)], crs=CRS)
    assert len(tf.vector.contains_properly(sq, inner)) == 1 and len(tf.vector.contains_properly(sq, pe)) == 0
    nb = gpd.GeoDataFrame(geometry=[box(10, 0, 20, 10)], crs=CRS)
    assert len(tf.vector.touches(sq, nb)) == 1 and len(tf.vector.overlaps(sq, nb)) == 0
    ov = gpd.GeoDataFrame(geometry=[box(5, 0, 20, 10)], crs=CRS)
    assert len(tf.vector.overlaps(sq, ov)) == 1
    line = gpd.GeoDataFrame(geometry=[LineString([(-5, 5), (15, 5)])], crs=CRS)
    assert len(tf.vector.crosses(line, sq)) == 1
    assert len(tf.vector.intersects(line, sq)) == 1
    same = gpd.GeoDataFrame(geometry=[Polygon([(10, 10), (10, 0), (0, 0), (0, 10)])], crs=CRS)
    assert len(tf.vector.equals(sq, same)) == 1
    # dwithin at exactly the distance
    p = gpd.GeoDataFrame(geometry=[Point(13, 5), Point(14, 5)], crs=CRS)
    assert len(tf.vector.dwithin(p, sq, 3)) == 1
    assert len(tf.vector.dwithin(p, sq, 2.999)) == 0
    assert len(tf.vector.dwithin(p, sq, 4)) == 2
    assert len(tf.vector.disjoint(p, sq)) == 2
    assert len(tf.vector.disjoint(pe, sq)) == 0
    # empty and missing geometries never satisfy, disjoint included
    weird = gpd.GeoDataFrame(geometry=[Point(), None, Point(1, 1)], crs=CRS)
    assert list(tf.vector.relate(weird, sq, "intersects")) == [False, False, True]
    assert list(tf.vector.relate(weird, sq, "disjoint")) == [False, False, False]
    assert list(tf.vector.relate(weird, sq, "dwithin", distance=1)) == [False, False, True]
    assert len(tf.vector.select(weird, sq, "intersects", invert=True)) == 2
    assert list(tf.vector.count(weird, sq, "disjoint")) == [0, 0, 0]
    assert list(tf.vector.relate(p, sq.iloc[:0], "disjoint")) == [True, True]


def test_b_forms_and_crs(tmp_path):
    a = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(500000, 4649776), Point(0, 0)], crs=CRS)
    # EPSG:4326 polygon around the UTM zone 33 central meridian, 42 N (500000 E, ~4649776 N)
    b = gpd.GeoDataFrame(geometry=[box(14.9, 41.9, 15.1, 42.1)], crs="EPSG:4326")
    r = tf.vector.within(a, b)
    assert list(r["id"]) == [1] and r.crs == a.crs
    # a shapely geometry and a list, taken in a's CRS
    assert len(tf.vector.within(a, box(499000, 4649000, 501000, 4651000))) == 1
    assert len(tf.vector.within(a, [box(499000, 4649000, 501000, 4651000), box(-1, -1, 1, 1)])) == 2
    assert len(tf.vector.within(a.geometry, box(-1, -1, 1, 1))) == 1
    # paths and out=
    pa, pb, po = tmp_path / "a.gpkg", tmp_path / "b.gpkg", tmp_path / "o.gpkg"
    tf.write_vector(a, pa)
    tf.write_vector(b.to_crs(CRS), pb)
    r = tf.vector.select(str(pa), pb, "within", out=po)
    assert po.exists() and list(r["id"]) == [1]
    back = tf.load_vector(po)
    assert list(back["id"]) == [1] and back.crs == a.crs
    assert len(tf.vector.join(pa, pb, "within", out=tmp_path / "j.gpkg")) == 2


def test_errors():
    with pytest.raises(ValueError, match="unknown predicate"):
        tf.vector.select(PTS, POLY, "nearby")
    with pytest.raises(ValueError, match="distance"):
        tf.vector.select(PTS, POLY, "dwithin")
    with pytest.raises(ValueError, match="distance"):
        tf.vector.relate(PTS, POLY, "dwithin")
    with pytest.raises(ValueError, match="disjoint"):
        tf.vector.join(PTS, POLY, "disjoint")
    with pytest.raises(ValueError, match="how"):
        tf.vector.join(PTS, POLY, how="many")
    with pytest.raises(ValueError, match="not a column"):
        tf.vector.join(POLY, PTS, "contains", how="summary", fields="nope")
