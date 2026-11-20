# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move.extract_covariates_along against R's amt 0.3.1 and terra 1.9-50 (GDAL 3.9.3), and along_summary.

The references (tests/move_cases.py, group "amt_along"; tools/amt_fixed.R's amt_along) are the values amt's
extract_covariates_along gives, bit for bit, cell by cell and in terra's order. along_summary is not in amt: it is
tested against a plain computation from the values and the order along the step, and on a few steps of a known DEM.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

import terraflux as tf
from terraflux import _core
from move_cases import ALONG_CASES, ALONG_LAYERS, ALONG_RASTERS, along_raster
from test_move import FISHERS, same

REF = Path(__file__).parent / "data" / "reference" / "move"


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "amt_along.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs_along.npz"))


def raster(name, layers=3):
    """The DataArrays (a dict, one per layer) of a raster of ALONG_RASTERS."""
    r = along_raster(name)
    out = {}
    for k in ALONG_LAYERS[:layers]:
        da = xr.DataArray(r["values"][k], dims=("y", "x"), name=k,
                          coords={"y": r["ymax"] - r["yres"] * (np.arange(r["nrows"]) + 0.5),
                                  "x": r["xmin"] + r["xres"] * (np.arange(r["ncols"]) + 0.5)})
        # rioxarray's transform() recomputes it from the coordinates (1.3 becomes 1.2999999999999998): the grid is
        # the one of the GeoTransform it writes, as R has the extent
        if r["crs"]:
            da = da.rio.write_crs(r["crs"])
        out[k] = da.rio.write_transform(Affine(r["xres"], 0, r["xmin"], 0, -r["yres"], r["ymax"]))
    return out


def stored_steps(inputs, name):
    return pd.DataFrame({"x1_": inputs[f"{name}/x"], "y1_": inputs[f"{name}/y"], "x2_": inputs[f"{name}/x2"],
                         "y2_": inputs[f"{name}/y2"]})


def fishers_steps():
    f = tf.load_tracks(FISHERS, sort=False)
    return tf.move.steps(f[~f.geometry.is_empty])


def flat(got, layers):
    """amt_along's flattened form of a list of the values of the steps: (offsets, values of each layer)."""
    parts = [g.to_numpy() if isinstance(g, pd.DataFrame) else g.reshape(-1, 1) for g in got]
    off = np.r_[0, np.cumsum([len(p) for p in parts])]
    table = np.concatenate(parts)
    return off, [table[:, k] for k in range(layers)]


def check(ref, case, got, layers):
    off, vals = flat(got, layers)
    same(off, ref[f"{case}/off"], f"{case}: cells per step")
    for k, v in enumerate(vals):
        want = ref[f"{case}/v{k + 1}"]
        bad = ~((v == want) | (np.isnan(v) & np.isnan(want)))
        assert not bad.any(), f"{case}: layer {k + 1}: {bad.sum()} of {len(v)} values differ, first at {np.flatnonzero(bad)[:5]}"
    return off


@pytest.mark.parametrize("case", list(ALONG_CASES))
def test_values_are_amts(ref, inputs, case):
    steps = stored_steps(inputs, case)
    got = tf.move.extract_covariates_along(steps, raster(ALONG_CASES[case]))
    assert len(got) == len(steps) and all(isinstance(g, pd.DataFrame) for g in got)
    assert list(got[0].columns) == list(ALONG_LAYERS)
    off = check(ref, case, got, 3)
    assert off[-1] > len(steps) or case in ("along_one", "along_one_out", "along_zero")


@pytest.mark.parametrize("case, source", [("along_edges_one", "along_edges"), ("along_origin_one", "along_origin")])
def test_one_layer_gives_vectors(ref, inputs, case, source):
    got = tf.move.extract_covariates_along(stored_steps(inputs, source), raster(ALONG_CASES[source], 1)["cover"])
    assert all(isinstance(g, np.ndarray) and g.ndim == 1 for g in got)
    check(ref, case, got, 1)


def test_fishers_are_amts(ref):
    steps = fishers_steps()
    got = tf.move.extract_covariates_along(steps, raster("fishers"))
    off = check(ref, "along_fishers", got, 3)
    assert len(steps) == 32896 and off[-1] > 2 * len(steps)
    first = tf.move.extract_covariates_along(steps.iloc[:5000], raster("fishers", 1)["cover"])
    check(ref, "along_fishers_one", first, 1)


def test_cases_reach_the_edge_cases(ref, inputs):
    """The references hold what the cases are for: NaN cells, rows with an NA cell, one cell steps, many cells."""
    e = ref["along_outside/off"]
    n = np.diff(e)
    assert np.isnan(ref["along_edges/v1"]).any() and np.isnan(ref["along_edges/v2"]).any()
    wholly_out = ref["along_outside/v3"][e[:-1][n == 1].astype(int)]
    assert np.isnan(wholly_out).sum() > 300
    assert (np.diff(ref["along_zero/off"]) == 1).all()
    assert np.diff(ref["along_edges/off"]).max() > 12
    assert not np.isnan(ref["along_axis/v3"]).all()


def test_threads_do_not_matter(inputs):
    steps = stored_steps(inputs, "along_edges")
    r = raster("edges")
    a = tf.move.extract_covariates_along(steps, r, n_threads=1)
    b = tf.move.extract_covariates_along(steps, r, n_threads=4)
    for p, q in zip(a, b):
        same(p.to_numpy(), q.to_numpy())
    one = tf.move.along_summary(steps, r, n_threads=1)
    four = tf.move.along_summary(steps, r, n_threads=4)
    pd.testing.assert_frame_equal(one, four)


def test_crs_of_the_steps():
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    r = raster("utm")
    steps = pd.DataFrame({"x1_": [187500.5, 188000.0], "y1_": [8251000.5, 8252000.0], "x2_": [189000.25, 188500.0],
                          "y2_": [8251400.5, 8250500.0]})
    want = tf.move.extract_covariates_along(steps, r)
    t = Transformer.from_crs("EPSG:32723", "EPSG:4326", always_xy=True)
    ll = steps.copy()
    ll["x1_"], ll["y1_"] = t.transform(steps["x1_"], steps["y1_"])
    ll["x2_"], ll["y2_"] = t.transform(steps["x2_"], steps["y2_"])
    ll.attrs["terraflux.move"] = {"crs": "EPSG:4326"}
    got = tf.move.extract_covariates_along(ll, r)
    for p, q in zip(got, want):
        same(p.to_numpy(), q.to_numpy())


def test_errors(inputs):
    steps = stored_steps(inputs, "along_one")
    r = raster("edges")
    with pytest.raises(TypeError, match="steps"):
        tf.move.extract_covariates_along(steps[["x1_", "y1_"]], r)
    bad = steps.copy()
    bad.loc[0, "x2_"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        tf.move.extract_covariates_along(bad, r)
    with pytest.raises(ValueError, match="stats"):
        tf.move.along_summary(steps, r, ("median",))
    other = dict(raster("edges"), hill=raster("tenth")["hill"])
    with pytest.raises(ValueError, match="one grid"):
        tf.move.extract_covariates_along(steps, other)


# ---------------------------------------------------------------------------------------------- along_summary

def entry_order(x1, y1, x2, y2, ids, r):
    """The cells of a step ordered along it by the parameter where the segment enters each (Liang-Barsky)."""
    nc = r["ncols"]
    t_in = []
    for c in ids:
        row, col = divmod(int(c), nc)
        x0, x1c = r["xmin"] + col * r["xres"], r["xmin"] + (col + 1) * r["xres"]
        y1c, y0 = r["ymax"] - row * r["yres"], r["ymax"] - (row + 1) * r["yres"]
        lo, hi = 0.0, 1.0
        for p, q in ((-(x2 - x1), x1 - x0), (x2 - x1, x1c - x1), (-(y2 - y1), y1 - y0), (y2 - y1, y1c - y1)):
            if p == 0:
                continue
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
        assert lo <= hi + 1e-9, "the cell is not on the segment"
        t_in.append(lo)
    return np.argsort(t_in, kind="stable")


def by_hand(values, order):
    """The six summaries from the values (terra's order) and the order along the step (R's mean: move_r_mean,
    which test_move_amt checks against R)."""
    ok = values[~np.isnan(values)]
    if not len(ok):
        return [np.nan] * 6
    along = values[order]
    along = along[~np.isnan(along)]
    d = np.diff(along)
    up = down = 0.0
    for v in d:   # one after the other, as the sums are made (numpy's sum adds in pairs)
        if v > 0:
            up += v
        elif v < 0:
            down -= v
    return [_core.move_r_mean(np.ascontiguousarray(ok)), ok.min(), ok.max(), up, down,
            abs(along[-1] - along[0])]


def test_summary_is_computed_from_values_and_order():
    r = along_raster("utm")
    rng = np.random.default_rng(5)
    n = 600
    x1 = rng.uniform(r["xmin"] + 100, r["xmax"] - 100, n)
    y1 = rng.uniform(r["ymin"] + 100, r["ymax"] - 100, n)
    ang = rng.uniform(0, 2 * np.pi, n)
    length = rng.uniform(40, 900, n)
    steps = pd.DataFrame({"x1_": x1, "y1_": y1, "x2_": x1 + length * np.cos(ang), "y2_": y1 + length * np.sin(ang)})
    ids = raster("utm")["cover"].copy(data=np.arange(r["nrows"] * r["ncols"], dtype=float).reshape(r["nrows"], -1))
    dem = raster("utm")["hill"].copy(data=np.sin(np.arange(r["nrows"] * r["ncols"]).reshape(r["nrows"], -1) / 37.0) * 20)
    dem.values[5::17, ::3] = np.nan
    both = {"id": ids, "dem": dem}
    cells = tf.move.extract_covariates_along(steps, both)
    got = tf.move.along_summary(steps, both)
    assert list(got.columns) == list(steps.columns) + [f"{k}_along_{s}" for k in ("id", "dem") for s in tf.move._along.STATS]
    flat_dem = dem.values.ravel()
    for i, c in enumerate(cells):
        cid = c["id"].to_numpy().astype(int)
        order = entry_order(*steps.iloc[i], cid, r)
        want = by_hand(flat_dem[cid], order)
        have = got.iloc[i][[f"dem_along_{s}" for s in tf.move._along.STATS]].to_numpy(float)
        np.testing.assert_array_equal(have, np.array(want), err_msg=f"step {i}")
        want_id = by_hand(cid.astype(float), order)
        np.testing.assert_array_equal(got.iloc[i][[f"id_along_{s}" for s in tf.move._along.STATS]].to_numpy(float),
                                      np.array(want_id), err_msg=f"step {i} ids")
    assert got["dem_along_ascent"].gt(0).any() and got["dem_along_descent"].gt(0).any()
    assert got["dem_along_mean"].isna().sum() == 0 and not got["dem_along_delta"].isna().all()


def known_dem(z):
    """A raster of 10 x 10 unit cells whose values are z(row, col)."""
    row, col = np.meshgrid(np.arange(10.0), np.arange(10.0), indexing="ij")
    da = xr.DataArray(z(row, col), dims=("y", "x"), name="dem", coords={"y": 9.5 - np.arange(10.0), "x": np.arange(10.0) + 0.5})
    return da.rio.write_transform(Affine(1, 0, 0, 0, -1, 10))


def summary(dem, *steps):
    df = pd.DataFrame(steps, columns=["x1_", "y1_", "x2_", "y2_"])
    return tf.move.along_summary(df, dem).iloc[:, 4:].rename(columns=lambda c: c.split("_along_")[1])


def test_summary_on_a_known_slope():
    east = known_dem(lambda row, col: 3.0 * col)       # rises to the east: contour lines run north-south
    s = summary(east, (0.5, 5.5, 9.5, 5.5), (9.5, 5.5, 0.5, 5.5), (4.5, 0.5, 4.5, 9.5), (4.5, 9.5, 4.5, 0.5),
                (0.5, 5.1, 9.5, 5.9))
    # up the slope, down it, along a contour line in both directions, up the slope at a slant
    assert s["ascent"].tolist() == [27.0, 0.0, 0.0, 0.0, 27.0]
    assert s["descent"].tolist() == [0.0, 27.0, 0.0, 0.0, 0.0]
    assert s["delta"].tolist() == [27.0, 27.0, 0.0, 0.0, 27.0]
    assert s["mean"].tolist() == [13.5, 13.5, 12.0, 12.0, 13.5]
    assert s["min"].tolist() == [0.0, 0.0, 12.0, 12.0, 0.0] and s["max"].tolist() == [27.0, 27.0, 12.0, 12.0, 27.0]
    south = known_dem(lambda row, col: 2.0 * row)      # row 0 is the northern one: it rises to the south
    t = summary(south, (5.5, 9.5, 5.5, 0.5), (5.5, 0.5, 5.5, 9.5), (0.5, 5.5, 9.5, 5.5))
    assert t["ascent"].tolist() == [18.0, 0.0, 0.0] and t["descent"].tolist() == [0.0, 18.0, 0.0]
    assert t["delta"].tolist() == [18.0, 18.0, 0.0]


def test_summary_of_a_hill():
    hill = known_dem(lambda row, col: 100.0 - (col - 4.5) ** 2 * 2)    # a ridge from north to south: 59.5 ... 99.5, 99.5 ...
    s = summary(hill, (0.5, 5.5, 9.5, 5.5), (9.5, 5.5, 0.5, 5.5), (0.5, 5.5, 4.5, 5.5), (4.5, 0.5, 4.5, 9.5))
    # over the ridge ascent = descent and the ends are as high; to the top ascent = delta; along the crest nothing
    assert s["ascent"].tolist() == [40.0, 40.0, 40.0, 0.0]
    assert s["descent"].tolist() == [40.0, 40.0, 0.0, 0.0]
    assert s["delta"].tolist() == [0.0, 0.0, 40.0, 0.0]
    assert s["max"].tolist() == [99.5, 99.5, 99.5, 99.5] and s["min"].tolist() == [59.5, 59.5, 59.5, 99.5]


def test_summary_skips_cells_without_data():
    z = np.array([10.0, np.nan, 14.0, 13.0, np.nan, np.nan, 20.0, 5.0, np.nan, 7.0])
    dem = known_dem(lambda row, col: z[col.astype(int)])
    s = summary(dem, (0.5, 5.5, 9.5, 5.5), (9.5, 5.5, 0.5, 5.5), (1.5, 5.5, 1.5, 5.5), (4.5, 5.5, 5.5, 5.5))
    # 10, 14, 13, 20, 5, 7: up 4, down 1, up 7, down 15, up 2
    assert s["ascent"].tolist()[0] == 13.0 and s["descent"].tolist()[0] == 16.0 and s["delta"].tolist()[0] == 3.0
    assert s["ascent"].tolist()[1] == 16.0 and s["descent"].tolist()[1] == 13.0 and s["delta"].tolist()[1] == 3.0
    assert s["mean"].tolist()[0] == 69 / 6 and s["min"].tolist()[0] == 5.0 and s["max"].tolist()[0] == 20.0
    # a step on a cell without data, and on two of them: all NaN
    assert s.iloc[2].isna().all() and s.iloc[3].isna().all()


def test_summary_of_one_cell():
    dem = known_dem(lambda row, col: row * 10 + col)
    s = summary(dem, (2.5, 2.5, 2.7, 2.6))
    assert s.iloc[0].tolist() == [72.0, 72.0, 72.0, 0.0, 0.0, 0.0]


def test_summary_columns_and_stats():
    dem = known_dem(lambda row, col: col)
    df = pd.DataFrame({"x1_": [0.5], "y1_": [5.5], "x2_": [9.5], "y2_": [5.5], "id": [7]})
    out = tf.move.along_summary(df, dem, stats="ascent")
    assert list(out.columns) == ["x1_", "y1_", "x2_", "y2_", "id", "dem_along_ascent"] and out["id"].tolist() == [7]
    assert list(df.columns) == ["x1_", "y1_", "x2_", "y2_", "id"]   # the steps are left as they were
    assert tf.move.along_summary(df, dem, ())["id"].tolist() == [7]


def test_summary_of_the_fishers_runs_on_the_data():
    steps = fishers_steps().iloc[:3000]
    out = tf.move.along_summary(steps, raster("fishers"), ("mean", "ascent", "descent", "delta"))
    assert (out["hill_along_ascent"] >= 0).all() and (out["hill_along_delta"] <= out["hill_along_ascent"] + out["hill_along_descent"]).all()
    assert np.isfinite(out["hill_along_mean"]).all()
