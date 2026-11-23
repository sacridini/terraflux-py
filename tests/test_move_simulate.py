# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's simulation from a step selection model (redistribution_kernel, simulate_path, resistance, path_density,
issf) against R's amt 0.3.1 (tests/move_sim_cases.py, group "amt_sim"; tools/amt_fixed.R) and by properties."""

import dataclasses
import warnings
from pathlib import Path

import move_sim_cases as cases
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import BITWISE
from test_move import FISHERS

import terraflux as tf

REF = Path(__file__).parent / "data" / "reference" / "move"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "amt_sim.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs_sim.npz"))


def same(got, want, err=""):
    """R's bits (the x87 functions of R's Windows build); within 1e-9 where the machine has none."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    if BITWISE:
        np.testing.assert_array_equal(got, want, err_msg=err)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=err)


def map_a(values):
    x = np.arange(5.0, 1000.0, 10.0)
    return xr.DataArray(np.asarray(values, float).reshape(100, 100), dims=("y", "x"), coords={"y": x[::-1], "x": x},
                        name="cov").rio.write_crs("EPSG:32633")


def distr(spec):
    return tf.move.make_distribution(spec[0], **spec[1])


def model_of(c):
    return tf.move.make_issf_model(c["coefs"], distr(c["sl"]), distr(c["ta"]))


def start_of(c):
    return tf.move.make_start(c["start"], c["ta0"], time=c["time"], dt=c["dt"])


def seconds(t):
    return np.asarray((t - EPOCH).dt.total_seconds())


def run(c, cov, k_kwargs=None, n_steps=None):
    k = tf.move.redistribution_kernel(model_of(c), start_of(c), cov, n_control=c["n_control"],
                                      tolerance_outside=c["tol"], seed=c["seed"], **(k_kwargs or {}))
    return k, tf.move.simulate_path(k, c["n_steps"] if n_steps is None else n_steps)


def check_path(p, ref, case, which):
    same(p["x_"], ref[f"{case}/{which}_x"], f"{which} x")
    same(p["y_"], ref[f"{case}/{which}_y"], f"{which} y")
    same(seconds(p["t_"]), ref[f"{case}/{which}_t"], f"{which} t")


@pytest.mark.external_reference
def test_example_a_is_amts_and_the_corrected_amts(ref, inputs):
    c = cases.A
    cov = map_a(inputs["sim_a/in_map"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        k, p = run(c, cov)
    # the first cycle is the kernel; the path starts from the second
    same([k.sample.x_[0], k.sample.y_[0]], ref["sim_a/orig_first"])
    same([k.sample.x_[0], k.sample.y_[0]], ref["sim_a/fixed_first"])
    assert (p["x_"][1], p["y_"][1]) != (k.sample.x_[0], k.sample.y_[0])
    # amt's own functions (with a `fun` that makes log_sl_ and cos_ta_) and the corrected ones: the same path
    check_path(p, ref, "sim_a", "orig")
    check_path(p, ref, "sim_a", "fixed")
    assert p["x_"].iloc[0] == 500.0 and len(p) == 21 and p["step_"].tolist() == list(range(21))
    same(p["dt"].dt.total_seconds(), ref["sim_a/orig_dt"])
    assert k.stopped is None and k.sample.columns.tolist() == ["x_", "y_", "t2_"]
    assert k.sample.t2_[0] == pd.Timestamp("2020-01-01 01:00", tz="UTC")


def pd_raster(inputs):
    ext = inputs["sim_b/in_pd_ext"]
    nr, nc = (int(v) for v in inputs["sim_b/in_pd_dim"])
    xs = ext[0] + (ext[1] - ext[0]) / nc * (np.arange(nc) + 0.5)
    ys = ext[3] - (ext[3] - ext[2]) / nr * (np.arange(nr) + 0.5)
    return xr.DataArray(inputs["sim_b/in_pd_vals"].reshape(nr, nc), dims=("y", "x"), coords={"y": ys, "x": xs},
                        name="popden").rio.write_crs("EPSG:5070")


def b_tracks(inputs):
    import geopandas as gpd

    return gpd.GeoDataFrame({"time": pd.to_datetime(inputs["sim_b/in_leroy_t"], unit="s", utc=True), "track": "Leroy"},
                            geometry=gpd.points_from_xy(inputs["sim_b/in_leroy_x"], inputs["sim_b/in_leroy_y"]))


@pytest.mark.external_reference
def test_example_b_a_fitted_model_of_the_fishers(ref, inputs):
    """amt's tutorial on the fisher Leroy: the data, the fit and the simulation from it are R's bits."""
    b = cases.B
    popden = pd_raster(inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = tf.move.issf(b_tracks(inputs), popden, interval="30min", tolerance="5min", n_control=10, where="both",
                         formula="case_ ~ popden_end + sl_ + log(sl_) + cos(ta_)", seed=b["seed_fit"])
    same(m.coef.to_numpy(), ref["sim_b/coef"], "the coefficients")
    assert m.nevent == ref["sim_b/n_steps_fit"][0] - 57   # the steps with a turn angle
    assert m.sl_distr.params["shape"] == ref["sim_b/sl_shape"][0] and m.sl_distr.params["scale"] == ref["sim_b/sl_scale"][0]
    assert m.ta_distr.params["kappa"] == ref["sim_b/ta_kappa"][0]
    x0, y0 = inputs["sim_b/in_start"]
    st = tf.move.make_start((x0, y0), b["ta0"], time=b["time"], dt=b["dt"])
    kw = {"n_control": b["n_control"], "tolerance_outside": b["tol"], "seed": b["seed"]}
    # the fitted model (log(sl_), cos(ta_) as terms) and the same numbers as a manual model (log_sl_, cos_ta_)
    manual = tf.move.make_issf_model(dict(zip(["popden_end", "sl_", "log_sl_", "cos_ta_"], ref["sim_b/coef"])),
                                     m.sl_distr, m.ta_distr)
    for model in (m, manual):
        k = tf.move.redistribution_kernel(model, st, popden, **kw)
        p = tf.move.simulate_path(k, b["n_steps"])
        same([k.sample.x_[0], k.sample.y_[0]], ref["sim_b/orig_first"])
        check_path(p, ref, "sim_b", "orig")
        check_path(p, ref, "sim_b", "fixed")
        same(p["dt"].dt.total_seconds(), np.full(21, 1800.0))


@pytest.mark.external_reference
@pytest.mark.parametrize("name, c", [("sim_c", cases.C), ("sim_d", cases.D)])
def test_interactions_functions_and_other_distributions(ref, inputs, name, c):
    """A half-normal step length (amt fails on its default max.dist: qhnorm), a log-normal one, interactions, I(), exp()
    and a covariate at the start of the step."""
    cov = map_a(inputs["sim_a/in_map"])
    _, p = run(c, cov)
    check_path(p, ref, name, "fixed")
    check_path(p, ref, name, "orig")   # with the max.dist given where amt needs it
    same(seconds(p["t_"]), ref[f"{name}/orig_t"])


@pytest.mark.external_reference
def test_amt_ends_at_the_first_candidate_outside(ref, inputs):
    """tolerance.outside = 0 (the default): amt gives NULL when one candidate in a thousand ends outside the map, though it
    has no covariate and no weight and is never drawn; here it is not counted."""
    c = cases.E
    assert ref["sim_e/orig_null"][0] == 1 and ref["sim_e/fixed_null"][0] == 0
    cov = map_a(inputs["sim_a/in_map"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        k, p = run(c, cov)
    same([k.sample.x_[0], k.sample.y_[0]], ref["sim_e/fixed_first"])
    check_path(p, ref, "sim_e", "fixed")


@pytest.mark.external_reference
def test_the_path_stops_where_candidates_that_could_be_drawn_are_outside(ref, inputs):
    """No covariate at the end of the step: a candidate outside has a weight, and above the tolerance the path ends with a
    warning, the rest NaN, as amt (whose rule is the same here)."""
    c = cases.F
    cov = map_a(inputs["sim_a/in_map"])
    k = tf.move.redistribution_kernel(model_of(c), start_of(c), cov, n_control=c["n_control"], tolerance_outside=c["tol"],
                                      seed=c["seed"])
    with pytest.warns(RuntimeWarning, match=r"stopped after 10 time steps"):
        p = tf.move.simulate_path(k, c["n_steps"])
    assert p["x_"].isna().sum() == 30 and p["y_"].notna().sum() == 11
    for which in ("orig", "fixed"):
        same(p["x_"], ref[f"sim_f/{which}_x"])
        same(p["y_"], ref[f"sim_f/{which}_y"])
    assert (p["t_"].diff().dropna() == pd.Timedelta("1h")).all()
    # the kernel itself stops with the same rule
    start = tf.move.make_start((20.0, 500.0), 3.0, time=0, dt=1)
    with pytest.warns(RuntimeWarning, match="Terminating simulations here"):
        k2 = tf.move.redistribution_kernel(model_of(c), start, cov, n_control=300, tolerance_outside=0.01, seed=1)
    assert k2.sample is None and k2.stopped


# -------------------------------------------------------------------------------------------- properties


@pytest.fixture(scope="module")
def land():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(60, 60))
    x = np.arange(0.5, 60.5)
    return xr.DataArray(z, dims=("y", "x"), coords={"y": x[::-1], "x": x}, name="forest").rio.write_crs("EPSG:32633")


def toy_model(**coefs):
    return tf.move.make_issf_model(coefs or {"forest_end": 1.0, "log_sl_": 0.2},
                                   tf.move.make_distribution("gamma", shape=2, scale=2),
                                   tf.move.make_distribution("vonmises", kappa=1))


def test_the_same_seed_is_the_same_path_and_the_kernel_keeps_the_generator(land):
    kw = {"n_control": 400, "tolerance_outside": 1.0}
    a = tf.move.redistribution_kernel(toy_model(), (30, 30), land, seed=5, **kw)
    b = tf.move.redistribution_kernel(toy_model(), (30, 30), land, seed=5, **kw)
    pa, pb = tf.move.simulate_path(a, 12), tf.move.simulate_path(b, 12)
    pd.testing.assert_frame_equal(pa, pb)
    c = tf.move.redistribution_kernel(toy_model(), (30, 30), land, seed=6, **kw)
    assert not np.array_equal(tf.move.simulate_path(c, 12)["x_"], pa["x_"])
    # one path of 12 steps is a path of 5 and the next 7 from where it ended (the generator goes on)
    d = tf.move.redistribution_kernel(toy_model(), (30, 30), land, seed=5, **kw)
    p1 = tf.move.simulate_path(d, 5)
    ta = np.arctan2(p1.y_[5] - p1.y_[4], p1.x_[5] - p1.x_[4])
    p2 = tf.move.simulate_path(d, 7, start=tf.move.make_start((p1.x_[5], p1.y_[5]), ta, time=a.start.t, dt=a.start.dt))
    np.testing.assert_array_equal(np.r_[p1["x_"], p2["x_"][1:]], pa["x_"])
    np.testing.assert_array_equal(np.r_[p1["y_"], p2["y_"][1:]], pa["y_"])
    # the kernel takes a cycle: the path is not the generator's first one
    assert (pa["x_"][1], pa["y_"][1]) != (a.sample.x_[0], a.sample.y_[0])


def test_selection_pulls_the_path_to_the_good_cells(land):
    from terraflux.sdm._points import grid_cells

    k = tf.move.redistribution_kernel(toy_model(forest_end=3.0), (30, 30), land, n_control=500, tolerance_outside=1.0, seed=1)
    p = tf.move.simulate_path(k, 150)
    cell = grid_cells(p.x_.to_numpy(), p.y_.to_numpy(), land)
    used = land.values.ravel()[cell[cell >= 0]]
    assert used.mean() > land.values.mean() + 0.5


def test_a_model_without_the_distributions_or_a_discrete_landscape(land):
    m = tf.move.make_issf_model({"forest_end": 1.0})
    assert m.sl_distr.name == "exp" and m.ta_distr.name == "unif"   # amt's defaults
    m.sl_distr = None
    with pytest.raises(ValueError, match="sl_distr and ta_distr"):
        tf.move.redistribution_kernel(m, (30, 30), land, n_control=10)
    with pytest.raises(NotImplementedError, match="discrete"):
        tf.move.redistribution_kernel(toy_model(), (30, 30), land, landscape="discrete")
    with pytest.raises(ValueError, match="not a covariate"):
        tf.move.redistribution_kernel(tf.move.make_issf_model({"rocks_end": 1.0}), (30, 30), land, n_control=10)
    with pytest.raises(ValueError, match="too few positive"):
        tf.move.redistribution_kernel(toy_model(), (1000.0, 1000.0), land, n_control=50, tolerance_outside=1.0)
    with pytest.raises(ValueError, match="distribution of turn angles"):
        tf.move.redistribution_kernel(toy_model(), (30, 30), land, ta_distr=tf.move.make_distribution("exp"))


def test_a_factor_of_a_covariate_selects_its_levels():
    rng = np.random.default_rng(2)
    n = 600
    d = pd.DataFrame({"s": np.repeat(np.arange(n // 6), 6), "lu_end": rng.choice([1.0, 2.0, 3.0], n)})
    d["case_"] = np.tile([True] + [False] * 5, n // 6)
    d.loc[d.case_, "lu_end"] = rng.choice([1.0, 2.0, 3.0], n // 6, p=[0.1, 0.8, 0.1])
    m = tf.move.fit_issf(d, "case_ ~ factor(lu_end) + strata(s)")
    assert m.coef.index.tolist() == ["factor(lu_end)2", "factor(lu_end)3"] and m.coef.iloc[0] > 1.0
    m.sl_distr = tf.move.make_distribution("gamma", shape=2, scale=8)
    m.ta_distr = tf.move.make_distribution("unif")
    x = np.arange(0.5, 60.5)
    lu = xr.DataArray(np.tile(np.repeat([1.0, 2.0, 3.0], 20), (60, 1)), dims=("y", "x"), coords={"y": x[::-1], "x": x},
                      name="lu").rio.write_crs("EPSG:32633")

    def share(model):
        k = tf.move.redistribution_kernel(model, (20.0, 30.0), lu, n_control=4000, n_sample=500, tolerance_outside=1.0, seed=1)
        return (k.sample.x_.to_numpy() >= 20).mean()   # the share that goes to level 2 (the other half is level 1)

    up = share(m)
    flipped = dataclasses.replace(m, coef=-m.coef)   # the same factor and levels, the opposite selection
    assert up > 0.7 and share(flipped) < 0.3
    bad = lu.where(lu != 3.0, 7.0)
    with pytest.raises(ValueError, match="none of its levels"):
        tf.move.redistribution_kernel(m, (30.0, 30.0), bad, n_control=2000, tolerance_outside=1.0, seed=1)


def test_make_issf_model():
    # the interaction in either order (amt reads a:b as b:a), the columns as R names them: by the first appearance
    m = tf.move.make_issf_model({"b": 2.0, "a": 1.0, "a:b": 0.5})
    assert m.coef.index.tolist() == ["b", "a", "b:a"] and m.coef.tolist() == [2.0, 1.0, 0.5]
    assert m.formula == "case_ ~ b + a + a:b + strata(step_id_)"
    with pytest.raises(ValueError, match="factor"):
        tf.move.make_issf_model({"factor(a)": 1.0})
    with pytest.raises(ValueError, match="finite"):
        tf.move.make_issf_model({"a": float("nan")})
    # the model matrix of the formula, as amt's
    mm = m.model_matrix(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    assert mm["b:a"].tolist() == [3.0, 8.0]


def test_the_distributions_of_random_steps_go_to_the_fit():
    tr = tf.move.simulate_brownian(range(60), tracks=2, seed=1)
    rs = tf.move.random_steps(tf.move.steps(tr), n_control=5, seed=2)
    rs["x"] = np.random.default_rng(1).normal(size=len(rs)) + 0.5 * rs["case_"]
    rs["stratum_"] = rs.groupby(["track", "step_id_"], sort=False).ngroup()
    m = tf.move.fit_issf(rs, "case_ ~ x + strata(stratum_)")
    assert m.sl_distr is None   # one gamma for each track: not one for the model
    one = tf.move.random_steps(tf.move.steps(tr[tr.track == tr.track.iloc[0]]), n_control=5, seed=2)
    one["x"] = np.random.default_rng(1).normal(size=len(one)) + 0.5 * one["case_"]
    m = tf.move.fit_issf(one, "case_ ~ x + strata(step_id_)")
    assert m.sl_distr.name == "gamma" and m.ta_distr.name == "vonmises"


# --------------------------------------------------------------------------------------------- resistance


def small_landscape():
    f = np.linspace(0.0, 1.0, 24).reshape(4, 6)
    g = np.arange(24.0).reshape(4, 6) % 5
    x, y = np.arange(0.5, 6.5), np.arange(3.5, -0.5, -1.0)
    return {"forest": xr.DataArray(f, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:32633"),
            "slope": xr.DataArray(g, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:32633")}


@pytest.mark.parametrize("transform", ["negexp", "linear", "inverse", "log"])
def test_resistance_follows_the_selection_down(transform):
    cov = small_landscape()
    m = tf.move.make_issf_model({"forest_end": 2.0, "slope_end": -0.3})
    r = tf.move.resistance(m, cov, transform=transform)
    eta = 2.0 * cov["forest"].values - 0.3 * cov["slope"].values
    order = np.argsort(eta.ravel())
    res = r.values.ravel()[order]
    assert (np.diff(res) <= 1e-12).all()   # the more selected, the lower the resistance
    assert r.dims == ("y", "x") and r.name == "resistance" and r.rio.crs == cov["forest"].rio.crs
    np.testing.assert_array_equal(r["x"], cov["forest"]["x"])
    if transform in ("negexp", "linear"):
        assert res.max() == pytest.approx(100.0) and res.min() == pytest.approx(1.0)


def test_resistance_formulas_and_the_limits():
    cov = small_landscape()
    m = tf.move.make_issf_model({"forest_end": 2.0})
    f = cov["forest"].values
    w = np.exp(2.0 * f)
    h = (w - w.min()) / (w.max() - w.min())
    lin = tf.move.resistance(m, cov, transform="linear", r_max=50.0).values
    np.testing.assert_allclose(lin, 50.0 - 49.0 * h, rtol=1e-12)
    ne = tf.move.resistance(m, cov, c=4.0, r_max=80.0).values
    np.testing.assert_allclose(ne, 80.0 - 79.0 * (1 - np.exp(-4.0 * h)) / (1 - np.exp(-4.0)), rtol=1e-12)
    mx = tf.move.resistance(m, cov, transform="linear", scale="max", r_max=50.0).values
    np.testing.assert_allclose(mx, 50.0 - 49.0 * (w / w.max()), rtol=1e-12)
    np.testing.assert_allclose(tf.move.resistance(m, cov, transform="inverse").values, 1 / w, rtol=1e-12)
    np.testing.assert_allclose(tf.move.resistance(m, cov, transform="log", c=3.0).values,
                               1 + 3.0 * (np.log(w.max()) - np.log(w)), rtol=1e-12)
    # the negative exponential falls faster than the line at the start
    assert (ne.ravel()[np.argsort(f.ravel())][1] < lin.ravel()[np.argsort(f.ravel())][1] + 30)
    with pytest.raises(ValueError, match="transform"):
        tf.move.resistance(m, cov, transform="sigmoid")
    with pytest.raises(ValueError, match="r_max"):
        tf.move.resistance(m, cov, r_max=0.5)


def test_resistance_leaves_out_what_a_cell_does_not_have():
    cov = small_landscape()
    m = tf.move.make_issf_model({"forest_end": 2.0, "log_sl_": 0.3, "forest_end:cos_ta_": 0.2, "slope_start": 0.5})
    with pytest.warns(UserWarning) as w:
        r = tf.move.resistance(m, cov)
    text = str(w[0].message)
    assert "log_sl_" in text and "forest_end:cos_ta_" in text and "slope_start" in text
    only = tf.move.resistance(tf.move.make_issf_model({"forest_end": 2.0}), cov)
    np.testing.assert_array_equal(r.values, only.values)
    with pytest.raises(ValueError, match="no term of habitat"), pytest.warns(UserWarning):
        tf.move.resistance(tf.move.make_issf_model({"sl_": 1.0}), cov)


def test_resistance_is_nan_where_a_covariate_is():
    cov = small_landscape()
    cov["forest"] = cov["forest"].where(cov["forest"] > 0.1)
    r = tf.move.resistance(tf.move.make_issf_model({"forest_end": 2.0, "slope_end": 0.1}), cov)
    assert np.isnan(r.values[0, 0]) and np.isnan(r.values).sum() == int(np.isnan(cov["forest"].values).sum())
    assert np.nanmin(r.values) == pytest.approx(1.0) and np.nanmax(r.values) == pytest.approx(100.0)


# ----------------------------------------------------------------------------------------- path density


def like_raster(nan=False):
    v = np.zeros((4, 5))
    if nan:
        v[0, 0] = np.nan
    x, y = np.arange(0.5, 5.5), np.arange(3.5, -0.5, -1.0)
    return xr.DataArray(v, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:32633")


def test_path_density_sums_to_one():
    like = like_raster()
    p = pd.DataFrame({"x_": [0.1, 0.2, 2.5, 4.9, np.nan, 9.0], "y_": [0.1, 0.3, 1.5, 3.9, 1.0, 1.0],
                      "step_": [0, 1, 2, 3, 4, 5]})
    d = tf.move.path_density(p, like)
    assert d.dims == ("y", "x") and d.name == "density"
    assert float(d.sum()) == pytest.approx(1.0)
    # the start is not a step end; the NaN and the point outside the grid are not counted
    assert d.values[3, 0] == pytest.approx(1 / 3) and d.values[2, 2] == pytest.approx(1 / 3) and d.values[0, 4] == pytest.approx(1 / 3)
    all_ = tf.move.path_density(p, like, include_start=True)
    assert all_.values[3, 0] == pytest.approx(0.5) and float(all_.sum()) == pytest.approx(1.0)
    # a list of paths is pooled; by gives one band each
    q = p.assign(track=["a", "a", "a", "b", "b", "b"])
    both = tf.move.path_density([p, p], like)
    np.testing.assert_allclose(both.values, d.values)
    bands = tf.move.path_density(q, like, by="track")
    assert bands.dims == ("track", "y", "x") and bands["track"].values.tolist() == ["a", "b"]
    np.testing.assert_allclose(bands.sum(("y", "x")).values, [1.0, 1.0])
    with pytest.raises(ValueError, match="x_ and y_"):
        tf.move.path_density(pd.DataFrame({"x": [1.0]}), like)
    with pytest.raises(ValueError, match="no position"):
        tf.move.path_density(pd.DataFrame({"x_": [50.0], "y_": [50.0]}), like)


def test_path_density_is_nan_where_the_landscape_is():
    like = like_raster(nan=True)
    p = pd.DataFrame({"x_": [0.5, 1.5, 1.5], "y_": [3.5, 3.5, 3.5]})   # one on the NaN cell
    d = tf.move.path_density(p, like)
    assert np.isnan(d.values[0, 0]) and d.values[0, 1] == pytest.approx(1.0) and np.nansum(d.values) == pytest.approx(1.0)


def test_the_density_of_simulated_paths_is_where_the_selection_is(land):
    k = tf.move.redistribution_kernel(toy_model(forest_end=2.0), (30, 30), land, n_control=300, tolerance_outside=1.0, seed=4)
    p = tf.move.simulate_path(k, 400)
    d = tf.move.path_density(p, land)
    assert float(d.sum()) == pytest.approx(1.0)
    used = d.values > 0
    assert land.values[used].mean() > land.values.mean()


# ---------------------------------------------------------------------------------------------- issf


def fishers_landscape():
    import rioxarray as rx

    elev = rx.open_rasterio(Path(tf.move.__file__).parent / "data" / "elevation.tif").isel(band=0, drop=True)
    elev.name = "elevation"
    f = tf.load_tracks(FISHERS, sort=False)
    f = f[~f.geometry.is_empty].to_crs(elev.rio.crs)
    return f, elev


def test_issf_is_the_sequence_by_hand():
    f, elev = fishers_landscape()
    first = f["individual-local-identifier"].unique()[:2] if "individual-local-identifier" in f.columns else None
    tr = f if first is None else f[f["individual-local-identifier"].isin(first)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = tf.move.issf(tr, elev, interval="2h", tolerance="30min", n_control=8, seed=3)
        # by hand: the same calls, the stratum of each track and step
        t = tf.move.filter_min_n_burst(tf.move.resample(tr, "2h", 1800.0), 3)
        rs = tf.move.random_steps(tf.move.steps(t), 8, seed=3)
        ex = tf.move.extract_covariates(rs, elev, where="end")
        track = tf.move._tracks.info_of(rs)["track"]
        ex["stratum_"] = ex.groupby([track, "step_id_"], sort=False).ngroup() + 1
        hand = tf.move.fit_issf(ex, "case_ ~ elevation + log(sl_) + cos(ta_) + strata(stratum_)")
    assert m.coef.index.tolist() == ["elevation", "log(sl_)", "cos(ta_)"]
    np.testing.assert_array_equal(m.coef.to_numpy(), hand.coef.to_numpy())
    np.testing.assert_array_equal(m.vcov.to_numpy(), hand.vcov.to_numpy())
    assert m.steps is not None and len(m.steps) == len(ex) and "stratum_" in m.steps.columns
    assert m.formula.endswith("strata(stratum_)")
    assert "iSSF of 2 track(s)" in m.report and "elevation" in m.report and m.n == hand.n
    assert m.sl_distr is None or m.sl_distr.name == "gamma"
    # the strata of different tracks are not merged: as many as the steps
    assert ex["stratum_"].nunique() == int(ex["case_"].sum())
    # a model of it simulates
    m.sl_distr = tf.move.make_distribution("gamma", shape=1.0, scale=300.0)
    m.ta_distr = tf.move.make_distribution("vonmises", kappa=0.5)
    cx = float(elev.x.mean())
    cy = float(elev.y.mean())
    k = tf.move.redistribution_kernel(m, (cx, cy), elev, n_control=200, tolerance_outside=1.0, seed=2)
    assert len(tf.move.simulate_path(k, 5)) == 6
