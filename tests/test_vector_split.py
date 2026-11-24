# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.vector.split: polygons and lines cut by lines (PostGIS ST_Split, QGIS Split with lines)."""

import numpy as np
import pytest

import terraflux as tf

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

CRS = "EPSG:32633"


def _gdf(geoms, **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs=CRS)


def test_square_cut_by_one_line_into_two():
    a = _gdf([box(0, 0, 10, 10)], name=["sq"])
    r = tf.vector.split(a, _gdf([LineString([(3, -1), (3, 11)])]))
    assert r.crs == a.crs
    assert list(r["name"]) == ["sq", "sq"]
    assert list(r["part"]) == [1, 2] and list(r["parts"]) == [2, 2]
    assert r.geometry.area.tolist() == [70.0, 30.0]  # the larger first
    assert shapely.union_all(r.geometry.values).equals(box(0, 0, 10, 10))


def test_equal_pieces_tie_larger_y_then_smaller_x():
    r = tf.vector.split(_gdf([box(0, 0, 10, 10)]), _gdf([LineString([(5, -1), (5, 11)]), LineString([(-1, 5), (11, 5)])]))
    assert len(r) == 4 and set(r["parts"]) == {4}
    c = [(g.centroid.x, g.centroid.y) for g in r.geometry]
    assert c == [(2.5, 7.5), (7.5, 7.5), (2.5, 2.5), (7.5, 2.5)]


def test_cross_cuts_into_four_unequal():
    cross = _gdf([LineString([(3, -1), (3, 11)]), LineString([(-1, 4), (11, 4)])])
    r = tf.vector.split(_gdf([box(0, 0, 10, 10)]), cross)
    assert r.geometry.area.tolist() == [42.0, 28.0, 18.0, 12.0]
    assert list(r["part"]) == [1, 2, 3, 4]


def test_line_that_does_not_reach_across_does_not_cut():
    p = box(0, 0, 10, 10)
    a = _gdf([p], k=[7])
    r = tf.vector.split(a, _gdf([LineString([(5, 5), (5, 20)])]))  # enters, ends inside
    assert len(r) == 1 and r["part"][0] == 1 and r["parts"][0] == 1
    assert r.geometry[0].equals_exact(p, 0)  # the geometry as it was
    r = tf.vector.split(a, _gdf([LineString([(5, 20), (5, 30)])]))  # nowhere near
    assert len(r) == 1 and r.geometry[0].equals_exact(p, 0)


def test_multipolygon_cut_part_by_part_and_order_of_a():
    mp = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    a = _gdf([box(100, 100, 101, 101), mp], id=[1, 2])
    r = tf.vector.split(a, _gdf([LineString([(2, -1), (2, 11)]), LineString([(25, -1), (25, 11)])]))
    assert list(r["id"]) == [1, 2, 2, 2, 2]
    assert list(r["part"]) == [1, 1, 2, 3, 4] and list(r["parts"]) == [1, 4, 4, 4, 4]
    assert r.geometry.area.tolist()[1:] == [80.0, 50.0, 50.0, 20.0]  # largest first; the 50s: smaller x first
    assert r.geometry[2].centroid.x < r.geometry[3].centroid.x


def test_polygon_with_hole_cut_through_hole():
    p = Polygon(box(0, 0, 10, 10).exterior.coords, [box(4, 4, 6, 6).exterior.coords])
    r = tf.vector.split(_gdf([p]), _gdf([LineString([(5, -1), (5, 11)])]))
    assert len(r) == 2 and np.allclose(r.geometry.area, [48.0, 48.0])
    assert sum(r.geometry.area) == p.area


def test_lines_cut_by_lines():
    a = _gdf([LineString([(0, 0), (10, 0)]), LineString([(0, 5), (10, 5)])], id=[1, 2])
    blades = _gdf([LineString([(2, -1), (2, 1)]), LineString([(6, -1), (6, 1)])])
    r = tf.vector.split(a, blades)
    assert list(r["id"]) == [1, 1, 1, 2]
    assert list(r["part"]) == [1, 2, 3, 1] and list(r["parts"]) == [3, 3, 3, 1]
    assert r.geometry.length.tolist()[:3] == [4.0, 4.0, 2.0]  # tie 4 and 4: smaller x first
    assert r.geometry[0].coords[0] == (2.0, 0.0) and r.geometry[1].coords[0] == (6.0, 0.0)
    assert r.geometry[3].equals_exact(a.geometry[1], 0)


def test_line_touching_blade_at_end_is_not_cut():
    r = tf.vector.split(_gdf([LineString([(0, 0), (10, 0)])]), _gdf([LineString([(10, -1), (10, 1)])]))
    assert len(r) == 1 and r["parts"][0] == 1


def test_polygon_blades_use_their_boundaries():
    a = _gdf([LineString([(-5, 5), (15, 5)]), box(0, 0, 10, 10)])
    r = tf.vector.split(a, _gdf([box(3, -2, 20, 20)]))
    assert list(r["parts"]) == [2, 2, 2, 2]  # the line at x=3, the square along x=3 and y 0..10 of the boundary
    assert r.geometry.geom_type.tolist() == ["LineString", "LineString", "Polygon", "Polygon"]
    assert r.geometry.area.tolist()[2:] == [70.0, 30.0]


def test_empty_and_missing_kept_with_part_one():
    a = _gdf([Polygon(), None, box(0, 0, 2, 2)])
    r = tf.vector.split(a, _gdf([LineString([(1, -1), (1, 3)])]))
    assert list(r["part"]) == [1, 1, 1, 2] and list(r["parts"]) == [1, 1, 2, 2]
    assert r.geometry[0].is_empty and r.geometry[1] is None


def test_errors_and_no_blades():
    sq = _gdf([box(0, 0, 1, 1)])
    with pytest.raises(ValueError):
        tf.vector.split(sq, _gdf([Point(0.5, 0.5)]))
    with pytest.raises(ValueError):
        tf.vector.split(_gdf([Point(0, 0)]), _gdf([LineString([(0, 0), (1, 1)])]))
    r = tf.vector.split(sq, _gdf([LineString([(0, 0), (1, 1)])]).iloc[:0])
    assert len(r) == 1 and r["parts"][0] == 1


def test_column_names_taken_get_suffix_and_out(tmp_path):
    a = _gdf([box(0, 0, 4, 4)], part=["x"])
    out = tmp_path / "pieces.gpkg"
    r = tf.vector.split(a, _gdf([LineString([(2, -1), (2, 5)])]), out=out)
    assert "part_2" in r.columns and "parts" in r.columns and list(r["part"]) == ["x", "x"]
    assert len(gpd.read_file(out)) == 2


def test_reprojects_lines_to_crs_of_a():
    a = _gdf([box(0, 0, 10, 10)])
    lines = gpd.GeoDataFrame(geometry=[LineString([(3, -1), (3, 11)])], crs=CRS).to_crs(4326)
    r = tf.vector.split(a, lines)
    assert len(r) == 2 and np.allclose(sorted(r.geometry.area), [30, 70], atol=1e-3)


def test_multipart_feature_with_no_part_cut_is_kept_whole():
    mp = MultiPolygon([box(0, 0, 10, 10), box(20, 0, 30, 10)])
    touch = _gdf([LineString([(10, 5), (15, 5)])])  # meets the first part's edge, cuts nothing
    r = tf.vector.split(_gdf([mp], id=[1]), touch)
    assert len(r) == 1 and list(r["parts"]) == [1] and r.geometry[0].equals(mp)
    one = tf.vector.split(_gdf([mp], id=[1]), _gdf([LineString([(5, -1), (5, 11)])]))
    assert list(one["parts"]) == [3, 3, 3]  # one part cut: its two pieces and the other part
