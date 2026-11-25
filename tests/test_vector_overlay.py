# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.vector: overlay (intersection, union, difference, symmetric_difference, identity), clip, dissolve and buffer."""

import numpy as np
import pytest

import terraflux as tf

gpd = pytest.importorskip("geopandas")
pd = pytest.importorskip("pandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

CRS = "EPSG:32633"
HOWS = ["intersection", "union", "difference", "symmetric_difference", "identity"]


def _polys(rng, n, tag):
    out = []
    for i in range(n):
        x, y = rng.random(2) * 70
        w, h = 8 + rng.random(2) * 30
        p = box(x, y, x + w, y + h)
        if i % 3 == 0:  # a hole
            p = Polygon(p.exterior.coords, [box(x + w * .3, y + h * .3, x + w * .6, y + h * .6).exterior.coords])
        out.append(p)
    return out


def _layers():
    rng = np.random.default_rng(11)
    pa = _polys(rng, 14, "a")
    pb = _polys(rng, 12, "b")
    pb.append(box(500, 500, 510, 510))  # meets nothing
    pa.append(box(-300, -300, -290, -290))  # meets nothing
    a = gpd.GeoDataFrame({"aid": np.arange(len(pa)), "name": [f"a{i}" for i in range(len(pa))]}, geometry=pa, crs=CRS)
    b = gpd.GeoDataFrame({"bid": np.arange(len(pb)), "w": rng.random(len(pb))}, geometry=pb, crs=CRS)
    return a, b


A, B = _layers()


def _key(df, cols):
    """Sorted (ids..., area) of the pieces; a missing id is -1."""
    rows = []
    for _, r in df.iterrows():
        ids = tuple(-1 if r[c] != r[c] else int(r[c]) for c in cols)
        rows.append((ids, float(r.geometry.area)))
    return sorted(rows)


def _same(ours, theirs, cols):
    ko, kt = _key(ours, cols), _key(theirs, cols)
    assert [k[0] for k in ko] == [k[0] for k in kt]
    np.testing.assert_allclose([k[1] for k in ko], [k[1] for k in kt], rtol=1e-9)


@pytest.mark.parametrize("how", HOWS)
def test_overlay_matches_geopandas(how):
    ours = tf.vector.overlay(A, B, how)
    theirs = gpd.overlay(A, B, how=how, keep_geom_type=True)
    cols = ["aid"] if how == "difference" else ["aid", "bid"]
    if how == "symmetric_difference":
        cols = ["aid", "bid"]
    _same(ours, theirs, cols)
    assert ours.crs == A.crs
    assert list(ours.index) == list(range(len(ours)))
    assert len(ours) > 0


def test_overlay_order_and_attributes():
    ours = tf.vector.overlay(A, B, "union")
    n_pairs = int((ours.aid.notna() & ours.bid.notna()).sum())
    both = ours.iloc[:n_pairs]
    # the pieces by (a, b), then a's remainders, then b's
    pairs = list(zip(both.aid, both.bid))
    assert pairs == sorted(pairs)
    rest = ours.iloc[n_pairs:]
    a_rest = rest[rest.aid.notna()]
    b_rest = rest[rest.aid.isna()]
    assert len(a_rest) + len(b_rest) == len(rest)
    assert list(rest.index[: len(a_rest)]) == [n_pairs + k for k in range(len(a_rest))]
    assert a_rest.bid.isna().all() and a_rest.w.isna().all()
    assert b_rest.name.isna().all() or all(v is None for v in b_rest.name)
    assert list(b_rest.bid) == sorted(b_rest.bid)
    assert list(ours.columns) == ["aid", "name", "bid", "w", "geometry"]


def test_difference_keeps_only_a():
    d = tf.vector.difference(A, B)
    assert list(d.columns) == ["aid", "name", "geometry"]
    far = d[d.aid == len(A) - 1]
    assert len(far) == 1 and far.geometry.iloc[0].equals(A.geometry.iloc[-1])
    assert d.aid.is_monotonic_increasing
    # a feature wholly inside b disappears
    inner = gpd.GeoDataFrame({"k": [1]}, geometry=[box(1, 1, 2, 2)], crs=CRS)
    outer = gpd.GeoDataFrame({"z": [9]}, geometry=[box(0, 0, 5, 5)], crs=CRS)
    assert len(tf.vector.difference(inner, outer)) == 0


def test_shortcuts_equal_overlay():
    for name in ("intersection", "difference", "symmetric_difference", "identity"):
        got = getattr(tf.vector, name)(A, B)
        want = tf.vector.overlay(A, B, name)
        assert got.equals(want)
    assert tf.vector.union(A, B).equals(tf.vector.overlay(A, B, "union"))


def test_suffix_for_clashing_names():
    b = B.rename(columns={"bid": "aid"})
    got = tf.vector.intersection(A, b)
    assert list(got.columns) == ["aid", "name", "aid_2", "w", "geometry"]
    got = tf.vector.intersection(A, b, suffix="_b")
    assert "aid_b" in got.columns


def test_hand_case_union():
    a = gpd.GeoDataFrame({"x": [1.0]}, geometry=[box(0, 0, 2, 2)], crs=CRS)
    b = gpd.GeoDataFrame({"y": ["t"]}, geometry=[box(1, 1, 3, 3)], crs=CRS)
    u = tf.vector.overlay(a, b, "union")
    assert [round(g.area, 9) for g in u.geometry] == [1.0, 3.0, 3.0]
    assert u.x.tolist()[:2] == [1.0, 1.0] and np.isnan(u.x.iloc[2])
    assert u.y.tolist()[0] == "t" and pd.isna(u.y.iloc[1]) and u.y.iloc[2] == "t"
    s = tf.vector.symmetric_difference(a, b)
    assert [round(g.area, 9) for g in s.geometry] == [3.0, 3.0]
    i = tf.vector.identity(a, b)
    assert [round(g.area, 9) for g in i.geometry] == [1.0, 3.0]
    assert i.y.iloc[0] == "t" and pd.isna(i.y.iloc[1])


def test_edge_sharing_pieces_are_dropped():
    a = gpd.GeoDataFrame({"x": [1]}, geometry=[box(0, 0, 1, 1)], crs=CRS)
    b = gpd.GeoDataFrame({"y": [2]}, geometry=[box(1, 0, 2, 1)], crs=CRS)
    assert len(tf.vector.intersection(a, b)) == 0
    assert len(tf.vector.difference(a, b)) == 1


def test_reprojects_b_to_a():
    b = B.to_crs("EPSG:4326")
    ours = tf.vector.intersection(A, b)
    base = tf.vector.intersection(A, B)
    assert ours.crs == A.crs
    assert len(ours) == len(base)
    assert ours.area.sum() == pytest.approx(base.area.sum(), rel=1e-6)


def test_shapely_geometry_as_b_and_out(tmp_path):
    got = tf.vector.intersection(A, box(0, 0, 40, 40), out=tmp_path / "o.gpkg")
    assert list(got.columns) == ["aid", "name", "geometry"]
    back = gpd.read_file(tmp_path / "o.gpkg")
    assert len(back) == len(got)


# ------------------------------------------------------------------------------------ lines and points as a


def _wide():
    return gpd.GeoDataFrame({"z": [1, 2]}, geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=CRS)


def test_lines_intersection_difference_identity():
    lines = gpd.GeoDataFrame({"l": [0, 1, 2]}, geometry=[
        LineString([(-5, 5), (15, 5)]),  # 10 inside the first
        LineString([(5, 5), (25, 5)]),  # 5 in the first, 5 in the second
        LineString([(40, 0), (50, 0)]),  # outside
    ], crs=CRS)
    inter = tf.vector.intersection(lines, _wide())
    assert inter.l.tolist() == [0, 1, 1]
    assert inter.z.tolist() == [1, 1, 2]
    assert inter.length.tolist() == pytest.approx([10, 5, 5])
    diff = tf.vector.difference(lines, _wide())
    assert diff.l.tolist() == [0, 1, 2]
    assert diff.length.tolist() == pytest.approx([10, 10, 10])
    ident = tf.vector.identity(lines, _wide())
    assert ident.length.sum() == pytest.approx(lines.length.sum())
    assert ident.l.tolist() == [0, 1, 1, 0, 1, 2]
    theirs = gpd.overlay(lines, _wide(), how="intersection", keep_geom_type=True)
    assert sorted(theirs.length) == pytest.approx(sorted(inter.length))


def test_points():
    pts = gpd.GeoDataFrame({"p": [0, 1, 2, 3]}, geometry=[Point(5, 5), Point(25, 5), Point(15, 5), Point(10, 5)],
                           crs=CRS)
    inter = tf.vector.intersection(pts, _wide())
    assert inter.p.tolist() == [0, 1, 3]  # the one on the edge meets (as GEOS intersects)
    assert inter.z.tolist() == [1, 2, 1]
    diff = tf.vector.difference(pts, _wide())
    assert diff.p.tolist() == [2]
    ident = tf.vector.identity(pts, _wide())
    assert len(ident) == 4 and ident.p.tolist() == [0, 1, 3, 2]
    assert all(g.geom_type == "Point" for g in ident.geometry)


def test_line_clipped_to_corner_keeps_only_lines():
    # a line along the edge shared with a polygon: the piece is a line; a line touching at one point gives nothing
    lines = gpd.GeoDataFrame({"l": [0, 1]}, geometry=[LineString([(0, 0), (0, 10)]), LineString([(10, 10), (12, 12)])],
                             crs=CRS)
    b = gpd.GeoDataFrame({"z": [1]}, geometry=[box(0, 0, 10, 10)], crs=CRS)
    got = tf.vector.intersection(lines, b)
    assert got.l.tolist() == [0]
    assert got.length.iloc[0] == pytest.approx(10)


# --------------------------------------------------------------------------------------------- union of one


def test_union_one_layer_is_dissolve():
    one = tf.vector.union(A)
    assert one.equals(tf.vector.dissolve(A))
    assert len(one) == 1 and one["count"].iloc[0] == len(A)
    assert one.geometry.iloc[0].area == pytest.approx(shapely.union_all(list(A.geometry)).area, rel=1e-12)


# ------------------------------------------------------------------------------------------------ errors


def test_errors():
    with pytest.raises(ValueError, match="unknown how"):
        tf.vector.overlay(A, B, "xor")
    lines = gpd.GeoDataFrame({"l": [0]}, geometry=[LineString([(0, 0), (1, 1)])], crs=CRS)
    with pytest.raises(ValueError, match="second layer is lines"):
        tf.vector.overlay(A, lines)
    for how in ("union", "symmetric_difference"):
        with pytest.raises(ValueError, match="first layer is lines"):
            tf.vector.overlay(lines, B, how)
    mixed = gpd.GeoDataFrame({"m": [0, 1]}, geometry=[Point(0, 0), LineString([(0, 0), (1, 1)])], crs=CRS)
    with pytest.raises(ValueError, match="mixes"):
        tf.vector.overlay(mixed, B)
    with pytest.raises(ValueError, match="mask is lines"):
        tf.vector.clip(A, lines)
    with pytest.raises(KeyError):
        tf.vector.dissolve(A, by="nope")
    with pytest.raises(ValueError):
        tf.vector.buffer(A, float("nan"))
    with pytest.raises(ValueError):
        tf.vector.buffer(A, 1, segments=0)


# -------------------------------------------------------------------------- clip, dissolve and buffer


def test_clip_matches_geopandas():
    mask = gpd.GeoDataFrame(geometry=[box(10, 10, 50, 50), box(40, 40, 90, 90)], crs=CRS)
    ours = tf.vector.clip(A, mask)
    theirs = gpd.clip(A, mask)
    _same(ours, theirs, ["aid"])
    assert list(ours.index) == list(range(len(ours)))
    assert list(ours.columns) == list(A.columns)
    # features outside the mask are left out; the order is that of a
    assert len(ours) < len(A) and ours.aid.is_monotonic_increasing
    assert (ours.aid != len(A) - 1).all()


def test_clip_lines_and_reprojected_mask():
    lines = gpd.GeoDataFrame({"l": [0, 1]}, geometry=[LineString([(0, 5), (20, 5)]), LineString([(0, 50), (5, 50)])],
                             crs=CRS)
    mask = gpd.GeoDataFrame(geometry=[box(5, 0, 10, 10)], crs=CRS).to_crs("EPSG:4326")
    got = tf.vector.clip(lines, mask)
    assert got.l.tolist() == [0] and got.length.iloc[0] == pytest.approx(5, rel=1e-6)
    assert got.crs == lines.crs


def test_dissolve_matches_geopandas():
    grp = A.assign(g=["x", "y", "x", "z", "y", "x", "x", "z", "y", "x", "y", "z", "x", "y", "z"][: len(A)])
    ours = tf.vector.dissolve(grp, by="g")
    assert ours.g.tolist() == ["x", "y", "z"]  # the order of the first feature
    assert ours["count"].tolist() == [int((grp.g == k).sum()) for k in "xyz"]
    theirs = grp[["g", "geometry"]].dissolve(by="g")
    for k, geom in zip(ours.g, ours.geometry):
        assert geom.area == pytest.approx(theirs.geometry.loc[k].area, rel=1e-12)
    assert list(ours.columns) == ["g", "count", "geometry"]
    allone = tf.vector.dissolve(grp)
    assert list(allone.columns) == ["count", "geometry"]
    assert allone["count"].tolist() == [len(grp)]
    assert allone.geometry.iloc[0].area == pytest.approx(theirs.union_all().area, rel=1e-12)
    assert ours.crs == A.crs


def test_dissolve_missing_values_form_a_group():
    gdf = gpd.GeoDataFrame({"g": ["a", None, "a", None]}, geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(1, 0, 2, 1),
                                                                      box(5, 5, 6, 6)], crs=CRS)
    got = tf.vector.dissolve(gdf, by="g")
    assert got["count"].tolist() == [2, 2]
    assert got.geometry.iloc[0].area == pytest.approx(2)


def test_buffer_matches_geopandas():
    ours = tf.vector.buffer(A, 3.5, segments=8)
    theirs = A.geometry.buffer(3.5, resolution=8)
    assert [g.equals(h) for g, h in zip(ours.geometry, theirs)] == [True] * len(A)
    assert list(ours.columns) == list(A.columns) and ours.crs == A.crs
    assert tf.vector.buffer(A, 1).geometry.iloc[0].equals(A.geometry.iloc[0].buffer(1, quad_segs=16))


def test_buffer_dissolve_and_points():
    pts = gpd.GeoDataFrame({"p": [0, 1]}, geometry=[Point(0, 0), Point(1, 0)], crs=CRS)
    one = tf.vector.buffer(pts, 1, dissolve=True)
    assert len(one) == 1 and list(one.columns) == ["geometry"] and one.crs == CRS
    assert one.geometry.iloc[0].equals(shapely.union_all([Point(0, 0).buffer(1), Point(1, 0).buffer(1)]))
    # a circle of 16 segments per quarter is a 64-gon, area a little under pi
    c = tf.vector.buffer(gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=CRS), 1)
    assert 3.13 < c.area.iloc[0] < np.pi
    # a negative distance can empty a polygon
    assert tf.vector.buffer(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=CRS), -2).geometry.iloc[0].is_empty


def test_multipolygon_inputs():
    mp = MultiPolygon([box(0, 0, 2, 2), box(5, 0, 7, 2)])
    a = gpd.GeoDataFrame({"x": [1]}, geometry=[mp], crs=CRS)
    b = gpd.GeoDataFrame({"y": [1]}, geometry=[box(1, 0, 6, 2)], crs=CRS)
    got = tf.vector.intersection(a, b)
    assert len(got) == 1 and got.area.iloc[0] == pytest.approx(4)
    assert got.geometry.iloc[0].geom_type == "MultiPolygon"


def test_a_piece_that_also_touches_along_an_edge_keeps_its_area():
    """Two polygons overlapping and also touching along an edge: GEOS gives two polygons and a line; both stay."""
    a = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 4, 4)], crs="EPSG:32723")
    # a U whose arms overlap a's left and right sides and whose base touches a's bottom edge between them
    u = Polygon([(-1, -1), (5, -1), (5, 4), (3, 4), (3, 0), (1, 0), (1, 4), (-1, 4)])
    b = gpd.GeoDataFrame({"k": [7]}, geometry=[u], crs="EPSG:32723")
    for how in ("intersection", "identity", "union"):
        got = tf.vector.overlay(a, b, how)
        piece = got[got["k"].notna() & got["id"].notna()]
        assert len(piece) == 1, how
        assert piece.geometry.iloc[0].geom_type == "MultiPolygon"
        assert piece.geometry.iloc[0].area == pytest.approx(8.0, abs=1e-12)
