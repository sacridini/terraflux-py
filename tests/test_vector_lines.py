# SPDX-License-Identifier: GPL-3.0-or-later
"""Linear referencing, shortest lines and closest points of tf.vector (src/terraflux/_vector_lines.py)."""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, MultiLineString, Point

from terraflux import _vector_lines as vl

L_LINE = LineString([(0, 0), (10, 0), (10, 5)])


def _gdf(geoms, **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs="EPSG:32633")


def _xy(gdf):
    return [(g.x, g.y) for g in gdf.geometry if not g.is_empty]


def test_points_along_spacing_exact():
    r = vl.points_along(_gdf([L_LINE], id=[7]), spacing=4)
    assert list(r["measure"]) == [0, 4, 8, 12]
    assert _xy(r) == [(0, 0), (4, 0), (8, 0), (10, 2)]
    assert list(r["part"]) == [0, 0, 0, 0] and list(r["id"]) == [7] * 4
    assert r.crs == "EPSG:32633" and list(r.index) == [0, 1, 2, 3]


def test_points_along_includes_end_when_exact_and_start_offset():
    r = vl.points_along(_gdf([L_LINE]), spacing=5)
    assert list(r["measure"]) == [0, 5, 10, 15]
    assert _xy(r)[-1] == (10, 5)
    r = vl.points_along(_gdf([L_LINE]), spacing=4, start=3)
    assert list(r["measure"]) == [3, 7, 11, 15]
    assert _xy(r) == [(3, 0), (7, 0), (10, 1), (10, 5)]
    assert len(vl.points_along(_gdf([L_LINE]), spacing=4, start=20)) == 0


def test_points_along_n():
    r = vl.points_along(_gdf([L_LINE]), n=4)
    assert list(r["measure"]) == [0, 5, 10, 15]
    assert _xy(r) == [(0, 0), (5, 0), (10, 0), (10, 5)]
    r = vl.points_along(_gdf([L_LINE]), n=1)
    assert list(r["measure"]) == [7.5] and _xy(r) == [(7.5, 0)]


def test_points_along_multi_parts_restart_measures():
    m = MultiLineString([[(0, 0), (10, 0)], [(0, 5), (0, 8)]])
    r = vl.points_along(_gdf([m]), spacing=4)
    assert list(r["part"]) == [0, 0, 0, 1]
    assert list(r["measure"]) == [0, 4, 8, 0]
    assert _xy(r) == [(0, 0), (4, 0), (8, 0), (0, 5)]


def test_points_along_zero_length_segment():
    ln = LineString([(0, 0), (4, 0), (4, 0), (4, 6)])
    r = vl.points_along(_gdf([ln]), spacing=2)
    assert list(r["measure"]) == [0, 2, 4, 6, 8, 10]
    assert _xy(r) == [(0, 0), (2, 0), (4, 0), (4, 2), (4, 4), (4, 6)]


def test_points_along_errors_and_missing():
    with pytest.raises(ValueError):
        vl.points_along(_gdf([L_LINE]))
    with pytest.raises(ValueError):
        vl.points_along(_gdf([L_LINE]), spacing=1, n=3)
    with pytest.raises(ValueError):
        vl.points_along(_gdf([L_LINE]), spacing=0)
    with pytest.raises(ValueError):
        vl.points_along(_gdf([L_LINE]), n=0)
    with pytest.raises(ValueError):
        vl.points_along(_gdf([Point(0, 0)]), spacing=1)
    r = vl.points_along(_gdf([LineString(), L_LINE]), spacing=15)
    assert list(r["measure"]) == [0, 15]


def test_substring_keeps_vertices():
    r = vl.substring(_gdf([L_LINE]), 3, 12)
    assert list(r.geometry[0].coords) == [(3, 0), (10, 0), (10, 2)]
    r = vl.substring(_gdf([L_LINE]), 0.2, 0.8, normalized=True)
    assert list(r.geometry[0].coords) == [(3, 0), (10, 0), (10, 2)]


def test_substring_clamp_zero_length_multi_and_errors():
    r = vl.substring(_gdf([L_LINE]), -5, 100)
    assert list(r.geometry[0].coords) == list(L_LINE.coords)
    r = vl.substring(_gdf([L_LINE]), 4, 4)
    assert list(r.geometry[0].coords) == [(4, 0), (4, 0)]
    m = MultiLineString([[(0, 0), (10, 0)], [(0, 5), (0, 8)]])
    r = vl.substring(_gdf([m]), 2, 5)
    g = r.geometry[0]
    assert g.geom_type == "MultiLineString"
    assert [list(p.coords) for p in g.geoms] == [[(2, 0), (5, 0)], [(0, 7), (0, 8)]]
    with pytest.raises(ValueError):
        vl.substring(_gdf([L_LINE]), 5, 4)
    with pytest.raises(ValueError):
        vl.substring(_gdf([Point(0, 0)]), 0, 1)


def test_split_lines():
    r = vl.split_lines(_gdf([L_LINE], id=[1]), 6)
    assert [list(g.coords) for g in r.geometry] == [
        [(0, 0), (6, 0)],
        [(6, 0), (10, 0), (10, 2)],
        [(10, 2), (10, 5)],
    ]
    assert [round(g.length, 9) for g in r.geometry] == [6, 6, 3]
    assert list(r["part"]) == [0, 1, 2] and list(r["id"]) == [1, 1, 1]
    # exact multiple: no piece of zero length
    assert len(vl.split_lines(_gdf([L_LINE]), 5)) == 3
    # sequence counts over parts
    m = MultiLineString([[(0, 0), (10, 0)], [(0, 5), (0, 8)]])
    assert list(vl.split_lines(_gdf([m]), 6)["part"]) == [0, 1, 2]
    with pytest.raises(ValueError):
        vl.split_lines(_gdf([L_LINE]), 0)


def test_locate_name_clash():
    lines = [LineString([(0, 0), (10, 0)])]
    r = vl.locate(_gdf([Point(4, 1)], line=["x"], distance=[9]), lines)
    assert r["line"].tolist() == ["x"] and r["line_2"].tolist() == [0]
    assert r["distance_2"].tolist() == [1] and r["measure"].tolist() == [4]


def test_locate_columns():
    lines = [LineString([(0, 0), (10, 0)]), LineString([(0, 2), (10, 2)]), L_LINE]
    r = vl.locate(_gdf([Point(4, 1), Point(10, 3), Point(2, 1)]), lines)
    assert r["line"].tolist() == [0, 2, 0]  # the first is a tie of lines 0 and 1: lowest index
    assert r["measure"].tolist() == [4, 13, 2]
    assert r["distance"].tolist() == [1, 0, 1]
    r = vl.locate(_gdf([Point(1, 1), Point()]), [])
    assert r["line"].tolist() == [-1, -1] and r["measure"].isna().all() and r["distance"].isna().all()
    with pytest.raises(ValueError):
        vl.locate(_gdf([Point(1, 1)]), [Point(0, 0)])


def test_shortest_line():
    houses = _gdf([Point(5, 3), Point(20, 0)], id=[1, 2], name=["a", "b"])
    roads = _gdf([LineString([(0, 0), (10, 0)]), LineString([(0, 6), (10, 6)])], id=[10, 20], kind=["p", "q"])
    r = vl.shortest_line(houses, roads)
    assert [list(g.coords) for g in r.geometry] == [[(5, 3), (5, 0)], [(20, 0), (10, 0)]]
    assert r["distance"].tolist() == [3, 10]
    assert r["id"].tolist() == [1, 2] and r["id_2"].tolist() == [10, 10] and r["kind"].tolist() == ["p", "p"]
    r = vl.shortest_line(houses, _gdf([]))
    assert all(g.is_empty for g in r.geometry) and r["distance"].isna().all()
    r = vl.shortest_line(_gdf([Point(1, 1), None]), roads)
    assert r.geometry[1].is_empty and np.isnan(r["distance"][1]) and r["kind"][1] is None


def test_closest_point():
    pts = _gdf([Point(5, 3), Point(5, 40), Point(3, 0), Point()], id=[1, 2, 3, 4])
    roads = [LineString([(0, 0), (10, 0)])]
    r = vl.closest_point(pts, roads)
    assert _xy(r) == [(5, 0), (5, 0), (3, 0)]
    assert r["moved"].tolist() == [3, 40, 0, 0]
    assert r["snapped"].tolist() == [True, True, True, False]
    r = vl.closest_point(pts, roads, max_distance=10)
    assert _xy(r) == [(5, 0), (5, 40), (3, 0)]
    assert r["moved"].tolist() == [3, 0, 0, 0]
    assert r["snapped"].tolist() == [True, False, True, False]
    assert r.geometry[3].is_empty and r["id"].tolist() == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        vl.closest_point(_gdf([L_LINE]), roads)


def test_reprojects_b_to_a():
    a = _gdf([Point(500000, 10)])
    b = gpd.GeoDataFrame(geometry=[LineString([(15, 0), (15, 1)])], crs="EPSG:4326").to_crs("EPSG:32633")
    r = vl.shortest_line(a, b)
    assert r.crs == "EPSG:32633" and r["distance"][0] < 1e3


def _random_line(rng, n):
    return LineString(np.cumsum(rng.uniform(-5, 5, size=(n, 2)), axis=0))


def test_against_shapely_random():
    from shapely.ops import substring as sh_sub

    rng = np.random.default_rng(7)
    for _ in range(30):
        ln = _random_line(rng, int(rng.integers(2, 12)))
        if ln.length == 0:
            continue
        r = vl.points_along(_gdf([ln]), spacing=ln.length / 7.3)
        for g, m in zip(r.geometry, r["measure"], strict=True):
            e = ln.interpolate(m)
            assert abs(g.x - e.x) < 1e-12 and abs(g.y - e.y) < 1e-12
        a, b = sorted(rng.uniform(0, ln.length, 2))
        got = vl.substring(_gdf([ln]), a, b).geometry[0]
        exp = sh_sub(ln, a, b)
        assert got.geom_type == "LineString"
        ca, ce = np.asarray(got.coords), np.asarray(exp.coords)
        assert ca.shape == ce.shape and np.abs(ca - ce).max() < 1e-12
        # the pieces of split_lines add up to the line
        s = vl.split_lines(_gdf([ln]), ln.length / 4.1)
        assert abs(sum(g.length for g in s.geometry) - ln.length) < 1e-9


def test_path_input_and_out(tmp_path):
    p = tmp_path / "in.gpkg"
    _gdf([L_LINE], id=[1]).to_file(p, driver="GPKG")
    o = tmp_path / "out.gpkg"
    r = vl.points_along(p, spacing=5, out=o)
    assert len(r) == 4
    back = gpd.read_file(o)
    assert len(back) == 4 and back["id"].tolist() == [1] * 4
    r = vl.split_lines(str(p), 6, out=tmp_path / "s.gpkg")
    assert len(gpd.read_file(tmp_path / "s.gpkg")) == len(r) == 3
