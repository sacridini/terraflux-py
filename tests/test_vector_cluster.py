# SPDX-License-Identifier: GPL-3.0-or-later
"""Voronoi, Delaunay and DBSCAN clustering of terraflux.vector (the _vector_cluster module)."""

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

from shapely.geometry import Point, box

from terraflux import _vector_cluster as vc


def _pts(xy, crs="EPSG:32633", **cols):
    return gpd.GeoDataFrame(cols, geometry=[Point(x, y) for x, y in xy], crs=crs)


def _random(n, seed=3):
    return np.random.default_rng(seed).uniform(0, 100, size=(n, 2))


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_voronoi_square_tiles_extent():
    gdf = _pts(SQUARE, name=list("abcd"))
    r = vc.voronoi(gdf)
    assert list(r["name"]) == list("abcd")
    assert r.crs == gdf.crs
    for cell, p in zip(r.geometry, gdf.geometry):
        assert cell.contains(p)
    env = shapely.envelope(shapely.union_all(r.geometry.values))
    assert r.geometry.area.sum() == pytest.approx(env.area)
    assert env.area > 100  # GEOS enlarges the envelope of the points


def test_voronoi_random_each_cell_contains_its_point():
    gdf = _pts(_random(50))
    r = vc.voronoi(gdf)
    assert len(r) == 50
    assert all(c.contains(p) for c, p in zip(r.geometry, gdf.geometry))


def test_voronoi_duplicates_and_empties():
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(10, 0), Point(0, 0), Point(5, 9), Point()], crs="EPSG:32633")
    r = vc.voronoi(gdf)
    assert r.geometry.iloc[0].equals(r.geometry.iloc[2])
    assert not r.geometry.iloc[0].is_empty
    assert r.geometry.iloc[4].is_empty
    assert r.geometry.iloc[1].contains(gdf.geometry.iloc[1])


def test_voronoi_clip():
    gdf = _pts(SQUARE)
    clip = box(-1, -1, 4, 11)
    r = vc.voronoi(gdf, clip=clip)
    assert len(r) == 4
    assert r.geometry.area.sum() == pytest.approx(clip.area)
    r = vc.voronoi(gdf, clip=gpd.GeoDataFrame(geometry=[box(-1, -1, 4, 4)], crs=gdf.crs))
    assert r.geometry.iloc[0].area == pytest.approx(25)
    assert r.geometry.iloc[2].is_empty
    assert len(r) == 4


def test_voronoi_clip_reprojected():
    gdf = _pts([(x + 500_000, y + 4_000_000) for x, y in SQUARE])
    clip = gpd.GeoDataFrame(geometry=[box(499_999, 3_999_999, 500_011, 4_000_011)], crs="EPSG:32633")
    clip = clip.to_crs("EPSG:4326")
    r = vc.voronoi(gdf, clip=clip)
    assert r.geometry.area.sum() == pytest.approx(144, rel=1e-6)


def test_voronoi_non_points():
    with pytest.raises(ValueError):
        vc.voronoi(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]))


def test_delaunay_square():
    gdf = _pts(SQUARE, name=list("abcd"))
    t = vc.delaunay(gdf)
    assert len(t) == 2 and list(t.columns) == ["geometry"]
    assert t.crs == gdf.crs
    assert t.geometry.area.sum() == pytest.approx(100)
    e = vc.delaunay(gdf, edges=True)
    assert len(e) == 5
    assert all(g.geom_type == "LineString" for g in e.geometry)


def test_delaunay_random_count():
    xy = _random(40, seed=8)
    t = vc.delaunay(_pts(xy))
    h = len(shapely.get_coordinates(shapely.MultiPoint(xy).convex_hull)) - 1
    assert len(t) == 2 * 40 - 2 - h


def test_delaunay_polygon_vertices():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    assert len(vc.delaunay(g)) == 2


def test_cluster_two_groups_and_noise():
    gdf = _pts([(0, 0), (50, 50), (1, 0), (51, 50), (500, 500), (0, 1), (50, 51)])
    r = vc.cluster(gdf, 2.0, min_points=2)
    assert list(r["cluster"]) == [0, 1, 0, 1, -1, 0, 1]
    assert r["cluster"].dtype == np.int64


def test_cluster_numbering_by_lowest_member():
    gdf = _pts([(100, 0), (0, 0), (100, 1), (1, 0)])
    assert list(vc.cluster(gdf, 2.0)["cluster"]) == [0, 1, 0, 1]


def test_cluster_chain_inclusive():
    gdf = _pts([(i * 5.0, 0) for i in range(6)])
    assert set(vc.cluster(gdf, 5.0)["cluster"]) == {0}
    assert set(vc.cluster(gdf, 4.99)["cluster"]) == set(range(6))


def test_cluster_border_goes_to_first_reaching_cluster():
    # two squares of 4 cores (eps 1.5) and a border point B (3 neighbours < 4) within eps of one corner of each
    left = [(0, 0), (1, 0), (0, 1), (1, 1)]
    right = [(4, 0), (5, 0), (4, 1), (5, 1)]
    border = (2.5, 0)
    r = vc.cluster(_pts([border] + right + left), 1.5, min_points=4)
    assert list(r["cluster"]) == [0] * 5 + [1] * 4  # the right square starts first and reaches B first
    r = vc.cluster(_pts([border] + left + right), 1.5, min_points=4)
    assert list(r["cluster"]) == [0] * 5 + [1] * 4
    r = vc.cluster(_pts(left + right + [border]), 1.5, min_points=4)
    assert list(r["cluster"]) == [0] * 4 + [1] * 4 + [0]


def test_cluster_polygons_edge_distance():
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(10, 0, 11, 1)])
    assert list(vc.cluster(gdf, 1.0)["cluster"]) == [0, 0, 1]
    assert list(vc.cluster(gdf, 0.5)["cluster"]) == [0, 1, 2]


def test_cluster_empty_and_overwrite():
    gdf = gpd.GeoDataFrame({"cluster": [9, 9, 9]}, geometry=[Point(0, 0), Point(), Point(1, 0)])
    r = vc.cluster(gdf, 2.0)
    assert list(r["cluster"]) == [0, -1, 0]
    assert list(gdf["cluster"]) == [9, 9, 9]
    assert "grp" in vc.cluster(gdf, 2.0, column="grp")


def test_cluster_validation():
    gdf = _pts(SQUARE)
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            vc.cluster(gdf, bad)
    with pytest.raises(ValueError):
        vc.cluster(gdf, 1.0, min_points=0)


def _same_partition(x, y):
    pairs = set(zip(x.tolist(), y.tolist()))
    return len(pairs) == len(set(x.tolist())) == len(set(y.tolist()))


@pytest.mark.parametrize("min_points", [2, 4])
def test_cluster_matches_sklearn(min_points):
    cl = pytest.importorskip("sklearn.cluster")
    xy = _random(300, seed=11)
    mine = vc.cluster(_pts(xy), 6.0, min_points=min_points)["cluster"].to_numpy()
    ref = cl.DBSCAN(eps=6.0, min_samples=min_points).fit(xy)
    assert set(np.flatnonzero(mine == -1)) == set(np.flatnonzero(ref.labels_ == -1))
    core = np.zeros(len(xy), dtype=bool)
    core[ref.core_sample_indices_] = True
    assert _same_partition(mine[core], ref.labels_[core])


def test_cluster_min_points_one_is_components():
    sp = pytest.importorskip("scipy.sparse")
    from scipy.sparse.csgraph import connected_components

    xy = _random(200, seed=5)
    mine = vc.cluster(_pts(xy), 7.0)["cluster"].to_numpy()
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    _, lab = connected_components(sp.csr_matrix(d <= 7.0), directed=False)
    assert (mine >= 0).all()
    assert _same_partition(mine, lab)
    assert mine[0] == 0 and mine.max() == len(set(lab)) - 1


def test_paths_and_out(tmp_path):
    src = tmp_path / "p.gpkg"
    _pts(SQUARE, name=list("abcd")).to_file(src, driver="GPKG")
    for name, call in (
        ("v", lambda o: vc.voronoi(str(src), out=o)),
        ("d", lambda o: vc.delaunay(str(src), out=o)),
        ("c", lambda o: vc.cluster(str(src), 20.0, out=o)),
    ):
        out = tmp_path / f"{name}.gpkg"
        r = call(str(out))
        assert out.exists()
        assert len(gpd.read_file(out)) == len(r)
    assert list(gpd.read_file(tmp_path / "c.gpkg")["cluster"]) == [0, 0, 0, 0]


def test_old_shapely_fallback(monkeypatch):
    monkeypatch.setattr(vc, "_shapely_at_least", lambda major, minor: False)
    gdf = _pts(_random(30, seed=2))
    r = vc.voronoi(gdf)
    assert all(c.contains(p) for c, p in zip(r.geometry, gdf.geometry))
    xy = _random(80, seed=4)
    a = vc.cluster(_pts(xy), 8.0, min_points=3)["cluster"]
    monkeypatch.undo()
    assert list(a) == list(vc.cluster(_pts(xy), 8.0, min_points=3)["cluster"])
