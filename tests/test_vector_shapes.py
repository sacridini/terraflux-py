"""Geometry operations of tf.vector (src/terraflux/_vector_shapes.py)."""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from terraflux import _vector_shapes as vs


def _gdf(geoms, crs="EPSG:32633", **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs=crs)


L_SHAPE = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10)])


def test_centroid_square_and_attributes():
    g = _gdf([box(0, 0, 2, 2), box(10, 10, 14, 12)], name=["a", "b"])
    r = vs.centroid(g)
    assert r.crs == g.crs
    assert list(r["name"]) == ["a", "b"]
    assert (r.geometry.x.tolist(), r.geometry.y.tolist()) == ([1.0, 12.0], [1.0, 11.0])
    assert r.geometry.geom_equals(g.geometry.centroid).all()


def test_centroid_concave_outside_vs_point_on_surface():
    c_shape = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 9), (10, 9), (10, 10), (0, 10)])
    g = _gdf([c_shape])
    assert not c_shape.covers(vs.centroid(g).geometry[0])
    assert c_shape.covers(vs.centroid(g, inside=True).geometry[0])


def test_hull_convex_each_and_geopandas():
    g = _gdf([MultiPolygon([box(0, 0, 1, 1), box(3, 3, 4, 4)]), LineString([(0, 0), (1, 1), (2, 0)])])
    r = vs.hull(g)
    assert r.geometry.geom_equals(g.geometry.convex_hull).all()
    assert r.geometry[1].area == 1.0


def test_hull_by_group_with_count():
    pts = [Point(0, 0), Point(4, 0), Point(0, 3), Point(10, 10), Point(12, 10), Point(10, 13), Point(1, 1)]
    g = _gdf(pts, kind=["b", "b", "b", "a", "a", "a", "b"])
    r = vs.hull(g, by="kind")
    assert list(r["kind"]) == ["b", "a"]  # order of first appearance
    assert list(r["count"]) == [4, 3]
    assert r.crs == g.crs
    assert r.geometry[0].area == pytest.approx(6.0)
    assert r.geometry[1].area == pytest.approx(3.0)


def test_hull_dissolve_all():
    g = _gdf([Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3), Point(2, 1)], k=[1, 2, 3, 4, 5])
    r = vs.hull(g, dissolve=True)
    assert list(r.columns) == ["count", "geometry"]
    assert list(r["count"]) == [5]
    assert r.geometry[0].area == pytest.approx(12.0)


def test_hull_concave_not_larger_than_convex():
    g = _gdf([L_SHAPE.centroid, *[Point(x, 0) for x in range(11)], *[Point(0, y) for y in range(11)],
              *[Point(x, 1) for x in range(1, 11)], *[Point(1, y) for y in range(1, 11)]])
    cv = vs.hull(g, dissolve=True)
    cc = vs.hull(g, kind="concave", ratio=0.2, dissolve=True)
    assert cc.geometry[0].area <= cv.geometry[0].area
    assert cc.geometry[0].area < cv.geometry[0].area  # the L's notch is cut out


def test_hull_errors():
    g = _gdf([Point(0, 0), Point(1, 1)], k=[1, 1])
    with pytest.raises(ValueError, match="kind"):
        vs.hull(g, kind="round")
    with pytest.raises(ValueError, match="together"):
        vs.hull(g, by="k", dissolve=True)
    with pytest.raises(KeyError):
        vs.hull(g, by="zzz")


def test_simplify_zigzag():
    line = LineString([(0, 0), (1, 0.1), (2, -0.1), (3, 0.1), (4, 0)])
    g = _gdf([line])
    r = vs.simplify(g, 0.5)
    assert list(r.geometry[0].coords) == [(0.0, 0.0), (4.0, 0.0)]
    assert list(vs.simplify(g, 0.0).geometry[0].coords) == list(line.coords)


def test_envelope_rotated_square_and_point():
    sq = Polygon([(2, 0), (4, 2), (2, 4), (0, 2)])
    g = _gdf([sq, Point(3, 3)])
    r = vs.envelope(g)
    assert r.geometry[0].equals(box(0, 0, 4, 4))
    assert r.geometry[1].equals(Point(3, 3))
    assert r.geometry.geom_equals(g.geometry.envelope).all()


def test_make_valid_bowtie():
    bow = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    g = _gdf([bow, box(0, 0, 1, 1)])
    r = vs.make_valid(g)
    assert r.geometry.is_valid.all()
    assert r.geometry[0].geom_type == "MultiPolygon"
    assert r.geometry[0].area == pytest.approx(2.0)
    assert r.geometry[1].equals(box(0, 0, 1, 1))


def test_explode_multipolygon_repeats_attributes():
    mp = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3), box(5, 5, 6, 6)])
    g = _gdf([box(9, 9, 10, 10), mp], name=["one", "multi"], n=[1, 2])
    r = vs.explode(g)
    assert len(r) == 4
    assert list(r.index) == [0, 1, 2, 3]
    assert list(r["name"]) == ["one", "multi", "multi", "multi"]
    assert list(r["n"]) == [1, 2, 2, 2]
    assert list(r.columns) == ["name", "n", "geometry"]
    assert r.geometry[2].equals(box(2, 2, 3, 3))
    expected = g.explode(index_parts=False).reset_index(drop=True)
    assert r.geometry.geom_equals(expected.geometry).all()


def test_boundary_polygon_with_hole_and_line():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (4, 2), (4, 4), (2, 4)]])
    g = _gdf([poly, LineString([(0, 0), (3, 4)])])
    r = vs.boundary(g)
    assert r.geometry[0].geom_type == "MultiLineString"
    assert len(r.geometry[0].geoms) == 2
    assert r.geometry[1].geom_type == "MultiPoint"
    assert r.geometry.geom_equals(g.geometry.boundary).all()


def test_measures_triangle_line_points():
    tri = Polygon([(0, 0), (3, 0), (0, 4)])
    donut = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (4, 2), (4, 4), (2, 4)]])
    g = _gdf([tri, donut, LineString([(0, 0), (3, 4)]), Point(5, 7), shapely.from_wkt("POINT EMPTY")])
    r = vs.measures(g)
    assert r["area"].tolist() == [6.0, 96.0, 0.0, 0.0, 0.0]
    assert r["perimeter"].tolist() == [12.0, 48.0, 0.0, 0.0, 0.0]
    assert r["length"].tolist() == [0.0, 0.0, 5.0, 0.0, 0.0]
    assert r["x"][0] == pytest.approx(1.0) and r["y"][0] == pytest.approx(4 / 3)
    assert (r["x"][2], r["y"][2]) == (1.5, 2.0)
    assert (r["x"][3], r["y"][3]) == (5.0, 7.0)
    assert np.isnan(r["x"][4]) and np.isnan(r["y"][4])


def test_measures_geodesic_cell_near_equator():
    g = _gdf([box(10, 0, 11, 1)], crs="EPSG:4326")
    r = vs.measures(g, geodesic=True)
    assert r["area"][0] == pytest.approx(12308e6, rel=0.005)
    assert vs.measures(g)["area"][0] == 1.0
    # the same cell in UTM is reprojected back
    utm = g.to_crs(32632)
    assert vs.measures(utm, geodesic=True)["area"][0] == pytest.approx(r["area"][0], rel=1e-3)
    with pytest.raises(ValueError, match="CRS"):
        vs.measures(_gdf([box(0, 0, 1, 1)], crs=None), geodesic=True)


def test_path_input_out_and_geoseries(tmp_path):
    g = _gdf([box(0, 0, 2, 2), box(5, 5, 6, 6)], name=["a", "b"])
    src = tmp_path / "in.gpkg"
    g.to_file(src, driver="GPKG")
    out = tmp_path / "c.gpkg"
    r = vs.centroid(src, out=out)
    assert out.exists()
    back = gpd.read_file(out)
    assert len(back) == 2 and list(back["name"]) == ["a", "b"]
    assert r.geometry.geom_equals(back.geometry).all()
    s = vs.envelope(g.geometry)
    assert isinstance(s, gpd.GeoDataFrame) and len(s) == 2
    out2 = tmp_path / "h.gpkg"
    vs.hull(src, by="name", out=out2)
    assert list(gpd.read_file(out2)["count"]) == [1, 1]


def test_input_not_changed_and_index_reset():
    g = _gdf([box(0, 0, 1, 1), box(2, 2, 3, 3)], v=[1, 2]).iloc[[1, 0]]
    before = g.geometry.copy()
    r = vs.simplify(g, 1)
    assert list(r.index) == [0, 1]
    assert list(r["v"]) == [2, 1]
    assert g.geometry.geom_equals(before).all() and list(g.index) == [1, 0]
