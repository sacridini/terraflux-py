# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move.validate: the use of the animals against a connectivity surface (Zeller et al. 2018; not in R).

Spearman's coefficient is compared with scipy's, the rest with numbers worked out by hand.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import spearmanr
from shapely import LineString, Point

import terraflux as tf
from terraflux.move import _validate as V

N = 12
X = np.arange(N) * 10.0 + 5
Y = (np.arange(N)[::-1]) * 10.0 + 5


def raster(a, x=X, y=Y, crs="EPSG:32723"):
    da = xr.DataArray(np.asarray(a, dtype=np.float64), coords={"y": y, "x": x}, dims=("y", "x"))
    return da.rio.write_crs(crs) if crs else da


def ramp():
    """A surface that grows to the east: the column index, ties along each column."""
    return raster(np.tile(np.arange(N, dtype=float), (N, 1)))


def stack(bands, names):
    da = xr.DataArray(np.stack(bands), coords={"band": names, "y": Y, "x": X}, dims=("band", "y", "x"))
    return da.rio.write_crs("EPSG:32723")


def test_spearman_is_scipys():
    rng = np.random.default_rng(3)
    for n, ties in [(50, False), (200, True), (7, True), (1000, True)]:
        x = rng.normal(size=n)
        y = x + rng.normal(size=n)
        if ties:
            x, y = np.round(x, 1), np.round(y, 0)
        assert V.spearman(x, y) == pytest.approx(spearmanr(x, y)[0], rel=1e-12, abs=1e-14)
    assert V.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isnan(V.spearman([1, 2], [1, 2]))
    assert np.isnan(V.spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_average_ranks():
    assert V._ranks(np.array([10.0, 20.0, 20.0, 5.0, 20.0])).tolist() == [2.0, 4.0, 4.0, 1.0, 4.0]


def test_ud_against_a_ramp_by_hand():
    # a UD on the 4 columns 8..11 (north to south uniform): all its mass is east; the ramp's best third are corridors
    u = np.zeros((N, N))
    u[:, 8:] = 1.0
    ud = raster(u / u.sum())
    v = tf.move.validate(ud, ramp(), threshold=8)
    r = v.table.iloc[0]
    assert r["track"] == "ud_1"
    assert r["use_fraction"] == pytest.approx(1.0)
    assert r["area_fraction"] == pytest.approx(4 / 12)
    assert r["use_ratio"] == pytest.approx(3.0)
    assert r["surface_used"] == pytest.approx(9.5)
    assert r["surface_available"] == pytest.approx(5.5)
    assert r["coverage"] == 1.0
    assert v.threshold == 8
    # the corridor is the surface >= 8, as a raster
    assert np.array_equal(v.corridor.values[:, :8], np.zeros((N, 8))) and v.corridor.values[:, 8:].min() == 1
    # the correlation over the cells with use (48 cells, all with surface 8..11 and equal use): use is constant
    assert np.isnan(r["rho"])


def test_rho_over_the_cells_is_scipys():
    rng = np.random.default_rng(5)
    s = rng.normal(size=(N, N))
    u = np.exp(s * 0.5 + rng.normal(size=(N, N)))
    u = u / u.sum()
    v = tf.move.validate(raster(u), raster(s), support=1, quantile=0.75)
    assert v.table["rho"][0] == pytest.approx(spearmanr(u.ravel(), s.ravel())[0], rel=1e-12)
    # support < 1: only the cells inside the volume contour
    order = np.argsort(-u.ravel(), kind="stable")
    cum = np.cumsum(u.ravel()[order])
    k = int(np.searchsorted(cum, 0.5 * cum[-1], side="left")) + 1
    keep = order[:k]
    v = tf.move.validate(raster(u), raster(s), support=0.5, quantile=0.75)
    assert v.table["rho"][0] == pytest.approx(spearmanr(u.ravel()[keep], s.ravel()[keep])[0], rel=1e-12)
    assert v.table["n_cells"][0] == k
    # the use fraction: the UD's mass over the top quarter of the surface
    top = s >= np.quantile(s, 0.75)
    assert v.table["use_fraction"][0] == pytest.approx(u[top].sum())
    assert v.table["area_fraction"][0] == pytest.approx(top.mean())


def test_resistance_is_the_surface_turned_over():
    rng = np.random.default_rng(8)
    s = rng.uniform(1, 10, size=(N, N))
    u = rng.uniform(size=(N, N))
    a = tf.move.validate(raster(u), raster(s), kind="current", quantile=0.7, support=1)
    b = tf.move.validate(raster(u), raster(-s), kind="resistance", quantile=0.7, support=1)
    # the best 30 % of the cells: the highest currents, the lowest resistances of the negated map
    c = tf.move.validate(raster(u), raster(1 / s), kind="resistance", quantile=0.7, support=1)
    assert b.table["rho"][0] == pytest.approx(spearmanr(u.ravel(), s.ravel())[0], rel=1e-12)
    assert c.table["rho"][0] == pytest.approx(spearmanr(u.ravel(), s.ravel())[0], rel=1e-12)
    assert a.table["rho"][0] == b.table["rho"][0]
    assert a.table["use_fraction"][0] == b.table["use_fraction"][0]
    assert a.threshold == -b.threshold
    # with a threshold: a resistance up to it
    t = tf.move.validate(raster(u), raster(s), kind="resistance", threshold=4.0)
    assert t.threshold == 4.0
    assert t.corridor.values[s <= 4.0].min() == 1 and t.corridor.values[s > 4.0].max() == 0


def test_binary_surface_is_its_own_corridor():
    s = np.zeros((N, N))
    s[:, 3:5] = 1
    u = np.zeros((N, N))
    u[:, 3] = 1
    u[:, 6] = 1
    v = tf.move.validate(raster(u / u.sum()), raster(s), kind="binary")
    r = v.table.iloc[0]
    assert r["use_fraction"] == pytest.approx(0.5)
    assert r["area_fraction"] == pytest.approx(2 / 12)
    assert r["use_ratio"] == pytest.approx(3.0)
    assert np.isnan(v.threshold)
    with pytest.raises(ValueError, match="binary"):
        tf.move.validate(raster(u), raster(s), kind="binary", threshold=0.5)
    with pytest.raises(ValueError, match="no corridor"):
        tf.move.validate(raster(u), raster(np.zeros((N, N))), kind="binary")


def test_stack_of_uds_and_the_pooled_row():
    a = np.zeros((N, N))
    a[:, 10] = 1
    b = np.zeros((N, N))
    b[:, 2] = 3
    v = tf.move.validate(stack([a, b], ["east", "west"]), ramp(), threshold=8)
    assert v.table["track"].tolist() == ["east", "west", "pooled"]
    assert v.table["use_fraction"].tolist() == pytest.approx([1.0, 0.0, 0.5])
    # the pooled UD is the mean of the normalised bands
    assert v.table["surface_used"].tolist() == pytest.approx([10.0, 2.0, 6.0])
    assert v.note.startswith("pooled: 50% of the use")


def test_ud_on_another_grid_is_read_at_the_surface_cells():
    # the UD on a grid shifted by half a cell, 10 columns instead of 12: its cell centres fall on the surface's cells
    x2 = X[:10] + 2.0
    y2 = Y[:10] - 3.0
    u = np.zeros((10, 10))
    u[:, 6:] = 1.0
    v = tf.move.validate(raster(u / u.sum(), x2, y2), ramp(), threshold=8)
    r = v.table.iloc[0]
    # columns 6..9 of the UD sit in the surface's columns 6..9 (x 62..92 lies in cells 6..9): ramp values 6..9
    assert r["surface_used"] == pytest.approx(7.5)
    assert r["coverage"] == 1.0
    # a UD that sticks out of the surface loses its mass there
    xo = X + 60.0
    u = np.ones((N, N)) / N**2
    v = tf.move.validate(raster(u, xo, Y), ramp(), quantile=0.5)
    assert v.table["coverage"][0] == pytest.approx(0.5)
    assert "Only 50% of the use" in v.note
    with pytest.raises(ValueError, match="no mass"):
        tf.move.validate(raster(u, X + 500.0, Y), ramp())


def test_ud_with_another_crs_is_reprojected():
    # a UD in longitude / latitude over one spot of a UTM surface: the surface is read at its centres in its own CRS
    from pyproj import Transformer

    surf = raster(np.tile(np.arange(N, dtype=float), (N, 1)), X + 500000.0, Y + 7_000_000.0)
    lon, lat = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True).transform(500045.0, 7000045.0)
    d = 1e-7
    u_ll = raster(np.ones((3, 3)) / 9, np.array([lon - d, lon, lon + d]), np.array([lat + d, lat, lat - d]),
                  crs="EPSG:4326")
    v = tf.move.validate(u_ll, surf, quantile=0.5)
    assert v.table["coverage"][0] == 1.0
    assert v.table["surface_used"][0] == pytest.approx(4.0)


def test_points_count_in_their_cells():
    # 6 points: 4 in the east third of the surface, 2 in the west
    pts = [Point(95, 55), Point(97, 56), Point(105, 5), Point(115, 15), Point(15, 15), Point(25, 105)]
    df = gpd.GeoDataFrame({"track": ["a"] * 6}, geometry=pts, crs="EPSG:32723")
    v = tf.move.validate(df, ramp(), threshold=8)
    r = v.table.iloc[0]
    assert r["use_fraction"] == pytest.approx(4 / 6)
    # the bounding box of the points' cells: columns 1..11, rows 1..11 -> 11 columns, of which 4 are corridors
    assert r["area_fraction"] == pytest.approx(4 / 11)
    assert r["use_ratio"] == pytest.approx((4 / 6) / (4 / 11))
    assert r["n_cells"] == 11 * 11
    assert r["surface_used"] == pytest.approx((9 + 9 + 10 + 11 + 1 + 2) / 6)
    # Spearman over the cells of the box: the counts against the column index
    box = ramp().values[1:12, 1:12]
    use = np.zeros((11, 11))
    for p in pts:
        use[int((N * 10 - p.y) // 10) - 1, int(p.x // 10) - 1] += 1
    assert r["rho"] == pytest.approx(spearmanr(use.ravel(), box.ravel())[0], rel=1e-12)
    # two tracks and the pooled row
    df["track"] = ["a", "a", "b", "b", "a", "b"]
    v = tf.move.validate(df, ramp(), threshold=8)
    assert v.table["track"].tolist() == ["a", "b", "pooled"]
    assert v.table["use_fraction"].tolist() == pytest.approx([2 / 3, 2 / 3, 4 / 6])
    # no track column: one track
    v = tf.move.validate(df[["geometry"]], ramp(), threshold=8)
    assert v.table["track"].tolist() == ["use"]
    # a point off the surface or on a cell without a value is left out
    s = ramp()
    s.values[11, 9] = np.nan
    s.values[0, 0] = np.nan
    far = gpd.GeoDataFrame(geometry=pts + [Point(1000, 1000), Point(95, 5)], crs="EPSG:32723")
    v = tf.move.validate(far, s, threshold=8)
    assert v.table["coverage"][0] == pytest.approx(6 / 8)


def test_points_in_another_crs_are_reprojected():
    from pyproj import Transformer

    pts = gpd.GeoDataFrame({"track": ["a"] * 3}, geometry=[Point(95, 55), Point(105, 5), Point(15, 15)], crs="EPSG:32723")
    surf = raster(np.tile(np.arange(N, dtype=float), (N, 1)), X + 300000.0, Y + 7_000_000.0)
    pts.geometry = [Point(p.x + 300000.0, p.y + 7_000_000.0) for p in pts.geometry]
    a = tf.move.validate(pts, surf, threshold=8)
    ll = pts.to_crs(4326)
    b = tf.move.validate(ll, surf, threshold=8)
    for c in ("use_fraction", "area_fraction", "surface_used"):
        assert a.table[c][0] == pytest.approx(b.table[c][0])
    assert Transformer is not None


def test_lines_count_their_length():
    # a line along row 5 over the columns 2..10 (a length of 80 m) and one inside the corridor: columns 8..11
    lines = [LineString([(25, 65), (105, 65)]), LineString([(85, 25), (115, 25)])]
    df = gpd.GeoDataFrame({"track": ["a", "a"]}, geometry=lines, crs="EPSG:32723")
    v = tf.move.validate(df, ramp(), threshold=8, spacing=1.0)
    r = v.table.iloc[0]
    # the first line is in columns 2..10; 8 of its 80 m (x from 80) are on columns >= 8: 25..105 -> 80..105 is 25 m
    assert r["use_fraction"] == pytest.approx((25 + 30) / 110)
    # the lines' boxes: columns 2..11 -> 10 columns, 4 of them corridors
    assert r["area_fraction"] == pytest.approx(4 / 10)
    # the default spacing (half a cell) gives the same here: the pieces are midpoints of 5 m
    w = tf.move.validate(df, ramp(), threshold=8)
    assert w.table["use_fraction"][0] == pytest.approx(r["use_fraction"])
    assert w.table["surface_used"][0] == pytest.approx(r["surface_used"], rel=0.05)
    # the length is what counts, not the number of vertices: a polyline cut in two gives the same
    df2 = gpd.GeoDataFrame({"track": ["a", "a"]}, geometry=[LineString([(25, 65), (65, 65), (105, 65)]), lines[1]],
                           crs="EPSG:32723")
    assert tf.move.validate(df2, ramp(), threshold=8).table["use_fraction"][0] == pytest.approx(w.table["use_fraction"][0])
    with pytest.raises(ValueError, match="not both"):
        tf.move.validate(gpd.GeoDataFrame(geometry=[lines[0], Point(5, 5)], crs="EPSG:32723"), ramp())


def test_corridor_segments_of_move_corridor():
    n = 20
    t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    a = [Point(10.0 * i, 0.3 * (i % 3)) for i in range(n)]
    b = [Point(10.0 * i + 2.5, 0.5) for i in range(n)]
    df = gpd.GeoDataFrame({"time": list(t) * 2, "id": ["a"] * n + ["b"] * n}, geometry=a + b, crs=32723)
    tr = tf.move.corridor(tf.move.as_tracks(df, "time", "id"), speed_threshold=0.1, circvar_threshold=0.1)
    lines = tf.move.corridor_lines(tr)
    surf = raster(np.ones((6, 24)) * np.linspace(0, 1, 24), np.arange(24) * 10.0 + 5, np.arange(6)[::-1] * 10.0 + 5)
    v = tf.move.validate(lines, surf, quantile=0.5)
    assert v.table["track"].tolist() == ["a", "b", "pooled"]
    assert (v.table["coverage"] > 0.9).all()
    # the points of the same tracks are in the same cells
    p = tf.move.validate(tr, surf, quantile=0.5)
    assert p.table["track"].tolist() == ["a", "b", "pooled"]


def test_note_reads_the_numbers():
    s = ramp()
    east = np.zeros((N, N))
    east[:, 9:] = 1
    west = np.zeros((N, N))
    west[:, :3] = 1
    flat = np.ones((N, N))
    up = tf.move.validate(raster(east / east.sum()), s, threshold=8)
    assert "favour the corridors" in up.note
    down = tf.move.validate(raster(west / west.sum()), s, threshold=8)
    assert "avoid the corridors" in down.note
    even = tf.move.validate(raster(flat / flat.sum()), s, threshold=8)
    assert "about what chance gives" in even.note and "cannot be computed" in even.note
    assert "the use is higher where the surface is worse" in tf.move.validate(
        raster(west / west.sum() + 1e-3), s, threshold=8, support=1).note


def test_errors():
    u = raster(np.ones((N, N)))
    with pytest.raises(ValueError, match="kind"):
        tf.move.validate(u, ramp(), kind="x")
    with pytest.raises(ValueError, match="quantile"):
        tf.move.validate(u, ramp(), quantile=1.0)
    with pytest.raises(ValueError, match="support"):
        tf.move.validate(u, ramp(), support=0)
    with pytest.raises(ValueError, match="threshold"):
        tf.move.validate(u, ramp(), threshold=float("nan"))
    with pytest.raises(ValueError, match="spacing"):
        tf.move.validate(u, ramp(), spacing=-1)
    with pytest.raises(TypeError, match="use"):
        tf.move.validate(3, ramp())
    with pytest.raises(ValueError, match="no negative"):
        tf.move.validate(raster(-np.ones((N, N))), ramp())
    with pytest.raises(ValueError, match="no cell has a value"):
        tf.move.validate(u, raster(np.full((N, N), np.nan)))
    with pytest.raises(ValueError, match="no mass"):
        tf.move.validate(raster(np.zeros((N, N))), ramp())
    with pytest.raises(TypeError, match="surface"):
        tf.move.validate(u, np.ones((N, N)))
    polys = gpd.GeoDataFrame(geometry=[Point(5, 5).buffer(3)], crs="EPSG:32723")
    with pytest.raises(ValueError, match="points"):
        tf.move.validate(polys, ramp())
    with pytest.raises(KeyError, match="track column"):
        tf.move.validate(gpd.GeoDataFrame(geometry=[Point(5, 5)], crs="EPSG:32723"), ramp(), track="nope")


def test_the_same_with_any_surface_orientation():
    # a surface with y increasing gives the same numbers
    rng = np.random.default_rng(2)
    s = rng.normal(size=(N, N))
    u = rng.uniform(size=(N, N))
    a = tf.move.validate(raster(u), raster(s), quantile=0.6)
    sf = raster(s[::-1], X, Y[::-1])
    uf = raster(u[::-1], X, Y[::-1])
    b = tf.move.validate(uf, sf, quantile=0.6)
    pd.testing.assert_frame_equal(a.table, b.table)
    assert np.array_equal(b.corridor.values, a.corridor.values[::-1])
