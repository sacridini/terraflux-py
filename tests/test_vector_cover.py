# SPDX-License-Identifier: GPL-3.0-or-later
import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

from terraflux import _vector_cover as vc

needs_cov = pytest.mark.skipif(not hasattr(shapely, "coverage_simplify"), reason="needs shapely 2.1")


def gdf(geoms, **cols):
    return gpd.GeoDataFrame(cols, geometry=list(geoms), crs="EPSG:32633")


def coords(g):
    return [tuple(c) for c in shapely.get_coordinates(g)]


def test_smooth_line_exact():
    g = gdf([LineString([(0, 0), (4, 0), (4, 4)])])
    got = coords(vc.smooth(g).geometry[0])
    assert got == [(0, 0), (1, 0), (3, 0), (4, 1), (4, 3), (4, 4)]
    got = coords(vc.smooth(g, preserve_ends=False).geometry[0])
    assert got == [(1, 0), (3, 0), (4, 1), (4, 3)]


def test_smooth_ring_exact_and_closed():
    sq = Polygon([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)])
    r = vc.smooth(gdf([sq]), preserve_ends=False).geometry[0]
    assert coords(r.exterior) == [(1, 0), (3, 0), (4, 1), (4, 3), (3, 4), (1, 4), (0, 3), (0, 1), (1, 0)]
    assert r.exterior.is_closed and len(r.exterior.coords) == 9
    assert r.area < sq.area and r.is_valid


def test_smooth_hole_and_iterations():
    p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (7, 3), (7, 7), (3, 7)]])
    one = vc.smooth(gdf([p])).geometry[0]
    assert len(one.exterior.coords) == 9 and len(one.interiors[0].coords) == 9
    assert one.is_valid and one.area < p.area
    two = vc.smooth(gdf([p]), iterations=2).geometry[0]
    twice = vc.smooth(vc.smooth(gdf([p]))).geometry[0]
    assert coords(two) == coords(twice)
    assert len(two.exterior.coords) == 17


def test_smooth_staircase_gets_smoother():
    st = Polygon([(0, 0), (4, 0), (4, 1), (3, 1), (3, 2), (2, 2), (2, 3), (1, 3), (1, 4), (0, 4)])
    s = vc.smooth(gdf([st]), iterations=2).geometry[0]
    assert len(s.exterior.coords) > len(st.exterior.coords)
    assert s.length < st.length and s.is_valid


def test_smooth_unchanged_and_multi():
    pts = [Point(1, 2), MultiPoint([(0, 0), (1, 1)]), LineString([(0, 0), (1, 1)]), LineString()]
    res = vc.smooth(gdf(pts)).geometry
    assert all(r.equals_exact(p, 0) for r, p in zip(res, pts))
    ml = shapely.MultiLineString([[(0, 0), (4, 0), (4, 4)], [(0, 0), (1, 1)]])
    r = vc.smooth(gdf([ml])).geometry[0]
    assert r.geom_type == "MultiLineString" and len(r.geoms[0].coords) == 6 and len(r.geoms[1].coords) == 2
    mp = shapely.MultiPolygon([box(0, 0, 4, 4), box(10, 0, 14, 4)])
    assert len(vc.smooth(gdf([mp])).geometry[0].geoms) == 2


def test_smooth_keeps_attributes_crs_index():
    g = gdf([LineString([(0, 0), (4, 0), (4, 4)])] * 2, k=[1, 2]).iloc[[1, 0]]
    r = vc.smooth(g)
    assert list(r.k) == [2, 1] and list(r.index) == [0, 1] and r.crs == g.crs


def test_smooth_errors():
    g = gdf([LineString([(0, 0), (4, 0), (4, 4)])])
    for bad in (0, 11, 1.5, True):
        with pytest.raises(ValueError):
            vc.smooth(g, iterations=bad)


def test_smooth_path_and_out(tmp_path):
    src = tmp_path / "in.gpkg"
    gdf([LineString([(0, 0), (4, 0), (4, 4)])]).to_file(src)
    dst = tmp_path / "out.gpkg"
    r = vc.smooth(str(src), out=str(dst))
    assert len(r.geometry[0].coords) == 6
    assert len(gpd.read_file(dst).geometry[0].coords) == 6


def grid(jagged=True):
    cells = [box(i, j, i + 1, j + 1) for j in range(3) for i in range(3)]
    if jagged:
        # the shared edge x=1 between cells (0,1) and (1,1) zig-zags, on both sides
        zig = [(1, 1), (1.05, 1.25), (0.95, 1.5), (1.05, 1.75), (1, 2)]
        left = Polygon([(0, 1), *zig, (0, 2)])
        mid = Polygon([(1, 1), (2, 1), (2, 2), (1, 2), *reversed(zig[1:-1])])
        cells[3], cells[4] = left, mid
    return gdf(cells, n=range(9))


@needs_cov
def test_coverage_simplify():
    g = grid()
    assert shapely.coverage_is_valid(_arr(g))
    r = vc.coverage_simplify(g, 0.5, preserve_boundary=True)
    assert len(r) == 9 and list(r.n) == list(range(9))
    assert shapely.coverage_is_valid(_arr(r))
    assert abs(shapely.union_all(_arr(r)).area - shapely.union_all(_arr(g)).area) < 1e-9
    assert sum(shapely.get_num_coordinates(_arr(r))) < sum(shapely.get_num_coordinates(_arr(g)))
    free = vc.coverage_simplify(g, 0.5)
    assert shapely.coverage_is_valid(_arr(free))


@needs_cov
def test_coverage_simplify_errors():
    with pytest.raises(ValueError):
        vc.coverage_simplify(gdf([Point(0, 0)]), 1)


def _arr(g):
    return np.asarray(g.geometry.values, dtype=object)


@needs_cov
def test_coverage_validate_valid():
    r = vc.coverage_validate(grid(False), out=None)
    assert r.valid.all() and r.geometry.is_empty.all() and list(r.n) == list(range(9))
    assert r.crs == grid().crs


@needs_cov
def test_coverage_validate_overlap_and_gap():
    ov = gdf([box(0, 0, 2, 2), box(1, 0, 3, 2), box(10, 0, 11, 1)])
    r = vc.coverage_validate(ov)
    assert list(r.valid) == [False, False, True]
    assert not r.geometry[0].is_empty and r.geometry[2].is_empty
    gap = gdf([box(0, 0, 1, 1), box(1.1, 0, 2, 1), box(10, 0, 11, 1)])
    assert vc.coverage_validate(gap).valid.all()
    r = vc.coverage_validate(gap, gap_width=0.5)
    assert list(r.valid) == [False, False, True]
    assert r.geometry[0].geom_type in ("LineString", "MultiLineString")
    assert not r.geometry[0].is_empty


@needs_cov
def test_coverage_path_and_out(tmp_path):
    src = tmp_path / "c.gpkg"
    grid().to_file(src)
    dst = tmp_path / "s.gpkg"
    r = vc.coverage_simplify(str(src), 0.5, out=str(dst))
    assert len(gpd.read_file(dst)) == len(r) == 9
    dst2 = tmp_path / "v.gpkg"
    vc.coverage_validate(str(src), out=str(dst2))
    assert dst2.exists()


def test_coverage_needs_shapely_21(monkeypatch):
    monkeypatch.delattr(shapely, "coverage_simplify", raising=False)
    with pytest.raises(RuntimeError, match="shapely 2.1"):
        vc.coverage_simplify(grid(False), 1)
