# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's step selection (fit_issf, log_rss) against R's survival 3.7-0 and amt 0.3.1
(tests/move_issf_cases.py, group "amt_issf"; tools/amt_fixed.R)."""

import math
import warnings
from pathlib import Path

import move_issf_cases as cases
import numpy as np
import pandas as pd
import pytest
from conftest import BITWISE

import terraflux as tf

REF = Path(__file__).parent / "data" / "reference" / "move"


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "amt_issf.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs_issf.npz"))


def table(inputs, name, **extra):
    d = pd.DataFrame({"step_id_": inputs[f"{name}/x"], "step_txt": inputs[f"{name}/track"].astype(str),
                      **{k: inputs[f"{name}/{k}"] for k in ("case_", "elevation", "sl_", "ta_", "landuse")}, **extra})
    return d


def same(got, want, err=""):
    """The bits of R's (the x87 functions of R's Windows build); within 1e-9 where the machine has none."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    if BITWISE:
        np.testing.assert_array_equal(got, want, err_msg=err)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-14, equal_nan=True, err_msg=err)


def same_fit(m, ref, case, prefix=""):
    r = lambda k: ref[f"{case}/{prefix}{k}"]
    same(m.coef.to_numpy(), r("coef"), "coef")
    same(m.vcov.to_numpy().ravel(order="F"), r("var"), "var")
    same(m.loglik, r("loglik"), "loglik")
    same(m.score, r("score"), "score")
    same(m.wald, r("wald"), "wald")
    assert m.iter == r("iter")[0] and m.n == r("n")[0] and m.nevent == r("nevent")[0]
    same(m.means.to_numpy(), r("means"), "means")
    s = m.summary()
    same(s["exp_coef"], r("exp"), "exp(coef)")
    same(s["se"], r("se"), "se")
    same(s["z"], r("z"), "z")
    same(s["p"], r("p"), "p")
    same(m.confint().to_numpy().ravel(order="F"), r("ci95"), "confint")
    same(m.confint(0.9).to_numpy().ravel(order="F"), r("ci90"), "confint 0.9")
    same(m.aic, r("aic"), "aic")
    assert m.edf == r("edf")[0]


def points(d):
    return pd.DataFrame(d["x1"]), pd.DataFrame(d["x2"])


def same_lr(res, ref, case, prefix=""):
    same(res["log_rss"], ref[f"{case}/{prefix}log_rss"], f"{prefix}log_rss")
    if "lwr" in res:
        same(res["lwr"], ref[f"{case}/{prefix}lwr"], f"{prefix}lwr")
        same(res["upr"], ref[f"{case}/{prefix}upr"], f"{prefix}upr")


@pytest.mark.external_reference
def test_the_fit_is_survivals(ref, inputs):
    d = table(inputs, "issf_a")
    m = tf.move.fit_issf(d, cases.F_A)
    same_fit(m, ref, "issf_a")
    assert m.n == len(d) - 2 and m.n_omitted == 2 and m.nevent == int(d.case_.sum()) - 1
    assert m.coef.index.tolist() == ["elevation", "log(sl_)", "cos(ta_)"]
    assert m.converged
    # a fit of True / False cases is the fit of 0 / 1
    d["case_"] = d.case_ == 1
    same(tf.move.fit_issf(d, cases.F_A).coef, m.coef)


@pytest.mark.external_reference
def test_the_log_rss_is_amts(ref, inputs):
    m = tf.move.fit_issf(table(inputs, "issf_a"), cases.F_A)
    x1, x2 = points(cases.LR_A)
    r = tf.move.log_rss(m, x1, x2)
    assert r.columns.tolist() == ["elevation_x1", "sl_x1", "ta_x1", "log_rss"]
    same_lr(r, ref, "issf_a")
    same_lr(tf.move.log_rss(m, x1, x2, ci="se"), ref, "issf_a", "se_")
    same_lr(tf.move.log_rss(m, x1, x2, ci="se", ci_level=0.9), ref, "issf_a", "se90_")


@pytest.mark.external_reference
def test_interactions_factors_and_transformations(ref, inputs):
    d = table(inputs, "issf_a")
    m = tf.move.fit_issf(d, cases.F_B)
    assert m.coef.index.tolist() == ["sl_", "log(sl_)", "factor(landuse)30", "factor(landuse)50", "factor(landuse)70",
                                     "factor(landuse)210", "I(elevation^2)", "log(sl_):elevation"]
    same_fit(m, ref, "issf_b")
    assert m.iter == 4
    assert (m.means.to_numpy()[2:6] == 0).all()   # the dummies are not centred


@pytest.mark.external_reference
def test_a_factor_column_and_its_log_rss(ref, inputs):
    lu = pd.Categorical(d_lu(inputs), categories=cases.LEVELS)
    d = table(inputs, "issf_a", lu=lu)
    m = tf.move.fit_issf(d, cases.F_FAC)
    assert m.coef.index.tolist()[:4] == ["lu0", "lu30", "lu50", "lu210"]
    assert "lu0:elevation" in m.coef.index
    same_fit(m, ref, "issf_fac")
    x1, x2 = points(cases.LR_FAC)
    x1["lu"] = pd.Categorical(x1["lu"], categories=cases.LEVELS)
    x2["lu"] = pd.Categorical(x2["lu"], categories=cases.LEVELS)
    same_lr(tf.move.log_rss(m, x1, x2), ref, "issf_fac")
    same_lr(tf.move.log_rss(m, x1, x2, ci="se"), ref, "issf_fac", "se_")
    # levels that are not the fit's
    bad = x1.assign(lu=pd.Categorical(x1["lu"].astype(str), categories=["0", "30", "210", "70", "50"]))
    with pytest.raises(ValueError, match="levels"):
        tf.move.log_rss(m, bad, x2)
    # strings of the levels work too; a value that is no level does not
    same_lr(tf.move.log_rss(m, x1.assign(lu=x1["lu"].astype(str)), x2.assign(lu=x2["lu"].astype(str))), ref, "issf_fac")
    with pytest.raises(ValueError, match="new levels"):
        tf.move.log_rss(m, x1.assign(lu=["30", "0", "999"]), x2.assign(lu="70"))


def d_lu(inputs):
    return np.asarray(inputs["issf_a/landuse"]).astype(int).astype(str)


@pytest.mark.external_reference
def test_tied_cases_in_a_stratum(ref, inputs):
    d = table(inputs, "issf_ties")
    k = d.groupby("step_txt").case_.sum()
    assert k.min() == 1 and k.max() == 4
    # the strata as text: "10" before "2" (the sums go in that order); as numbers in the numeric order
    same_fit(tf.move.fit_issf(d, cases.F_TIES_TXT), ref, "issf_ties", "t_")
    same_fit(tf.move.fit_issf(d, cases.F_TIES_NUM), ref, "issf_ties", "n_")
    # the strata kept apart: the argument instead of the term
    same_fit(tf.move.fit_issf(d, "case_ ~ elevation + sl_", strata="step_txt"), ref, "issf_ties", "t_")


@pytest.mark.external_reference
def test_collinear_columns_have_no_coefficient(ref, inputs):
    d = table(inputs, "issf_a").assign(k1=1.0)
    d["elev2"] = 2 * d["elevation"]
    m = tf.move.fit_issf(d, cases.F_COL)
    assert np.isnan(m.coef["elev2"]) and not np.isnan(m.coef["elevation"])
    assert (m.vcov.loc["elev2"] == 0).all() and (m.vcov["elev2"] == 0).all()
    same_fit(m, ref, "issf_col")
    s = m.summary()
    assert s.loc["elev2", "se"] == 0 and np.isnan(s.loc["elev2", ["z", "p"]]).all()
    x1, x2 = points(cases.LR_COL)
    same_lr(tf.move.log_rss(m, x1, x2, ci="se"), ref, "issf_col", "se_")
    # a column of ones: no coefficient, and the others as if it were not there
    m1 = tf.move.fit_issf(d, cases.F_ONE)
    assert np.isnan(m1.coef["k1"])
    same_fit(m1, ref, "issf_col", "one_")


@pytest.mark.external_reference
def test_the_bootstrap_is_amts_with_its_errors_corrected(ref, inputs):
    d = table(inputs, "issf_a")
    x1, x2 = points(cases.LR_BOOT)
    m = tf.move.fit_issf(d, cases.F_BOOT)
    r = tf.move.log_rss(m, x1, x2, ci="boot", ci_level=cases.BOOT_A["ci_level"], n_boot=cases.BOOT_A["n_boot"],
                        seed=cases.BOOT_A["seed"])
    assert r.columns.tolist() == ["elevation_x1", "sl_x1", "log_rss", "lwr", "upr"]
    same_lr(r, ref, "issf_boot", "a_")
    # log(sl_) in the formula and one row in x1: amt fails with both
    x1, x2 = points(cases.LR_BOOT1)
    m = tf.move.fit_issf(d, cases.F_BOOT_LOG)
    r = tf.move.log_rss(m, x1, x2, ci="boot", ci_level=cases.BOOT_B["ci_level"], n_boot=cases.BOOT_B["n_boot"],
                        seed=cases.BOOT_B["seed"])
    same_lr(r, ref, "issf_boot", "b_")


def small(seed=1, n=60, per=5):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"step_id_": np.repeat(np.arange(n), per), "x": rng.normal(size=n * per),
                      "z": rng.normal(size=n * per) * 3 + 1})
    d["case_"] = np.tile([True] + [False] * (per - 1), n)
    d.loc[d.case_, "x"] += 0.7
    return d


def test_the_aliases_and_the_defaults():
    assert tf.move.fit_ssf is tf.move.fit_issf and tf.move.fit_clogit is tf.move.fit_issf
    m = tf.move.fit_issf(small(), "case_ ~ x + z + strata(step_id_)")
    assert isinstance(m, tf.move.ISSF)
    assert m.nevent == 60 and m.n == 300 and m.n_omitted == 0 and m.converged
    assert 0 < m.coef["x"] < 2 and abs(m.coef["z"]) < 0.5
    assert m.aic == -2 * m.loglik[1] + 4 and m.loglik[0] < m.loglik[1]
    s = m.summary()
    assert s.columns.tolist() == ["coef", "exp_coef", "se", "z", "p"] and s.index.tolist() == ["x", "z"]
    assert (s["p"] < 1).all() and (s["p"] >= 0).all() and s["p"]["x"] < 0.05
    ci = m.confint()
    assert ci.columns.tolist() == ["2.5 %", "97.5 %"]
    np.testing.assert_allclose((ci["2.5 %"] + ci["97.5 %"]) / 2, m.coef, rtol=1e-12)


def test_a_fit_does_not_depend_on_the_order_of_the_rows_but_in_the_last_bits():
    d = small(3)
    f = "case_ ~ x + z + strata(step_id_)"
    a = tf.move.fit_issf(d, f)
    b = tf.move.fit_issf(d.sample(frac=1, random_state=4), f)   # rows shuffled, within the strata too
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=1e-9)
    assert a.iter == b.iter
    np.testing.assert_allclose(a.loglik, b.loglik, rtol=1e-12)


def test_the_intercept_terms_of_the_formula_are_ignored():
    d = small()
    f = tf.move.fit_issf(d, "case_ ~ x + z + strata(step_id_)")
    for g in ("case_ ~ x + z - 1 + strata(step_id_)", "case_ ~ 0 + x + z + strata(step_id_)",
              "case_ ~ 1 + x + z + strata(step_id_)", "case_~x+z+strata( step_id_ )"):
        pd.testing.assert_series_equal(tf.move.fit_issf(d, g).coef, f.coef)


def test_the_names_and_the_order_of_the_columns():
    d = small().assign(g=np.random.default_rng(9).choice(list("abcde"), 300), b=lambda t: t.z > 1)
    m = tf.move.fit_issf(d, "case_ ~ x*z + g + I(x^2) + log(abs(z) + 1) + b + strata(step_id_)")
    assert m.coef.index.tolist() == ["x", "z", "gb", "gc", "gd", "ge", "I(x^2)", "log(abs(z) + 1)", "bTRUE", "x:z"]
    m = tf.move.fit_issf(d, "case_ ~ x:g + x + strata(step_id_)")
    assert m.coef.index.tolist() == ["x", "x:gb", "x:gc", "x:gd", "x:ge"]
    # without the main effect of x, the factor has a column for each of its levels
    m = tf.move.fit_issf(d, "case_ ~ g:x + z + strata(step_id_)")
    assert m.coef.index.tolist() == ["z", "ga:x", "gb:x", "gc:x", "gd:x", "ge:x"]
    # a model matrix of other data: the columns of the fit, a missing value is a NaN
    new = pd.DataFrame({"x": [0.5, np.nan], "z": [1.0, 2.0], "g": ["a", "e"]})
    mm = m.model_matrix(new)
    assert mm.columns.tolist() == m.coef.index.tolist() and mm.shape == (2, len(m.coef))
    assert np.isnan(mm.iloc[1, 1:]).all() and mm.loc[0, "ga:x"] == 0.5 and mm.loc[0, "gb:x"] == 0.0 and mm.loc[1, "z"] == 2.0


def test_missing_values_leave_the_row_out():
    d = small()
    d.loc[3, "x"] = np.nan
    d.loc[9, "z"] = np.inf * 0
    d.loc[10, "step_id_"] = np.nan
    m = tf.move.fit_issf(d, "case_ ~ x + z + strata(step_id_)")
    assert m.n == 297 and m.n_omitted == 3 and m.nevent == 59
    d2 = d.drop([3, 9, 10]).reset_index(drop=True)
    pd.testing.assert_series_equal(m.coef, tf.move.fit_issf(d2, "case_ ~ x + z + strata(step_id_)").coef)
    # a transformation that is missing: log of a negative number
    m = tf.move.fit_issf(small(), "case_ ~ log(x) + strata(step_id_)")
    assert m.n < 300


def test_errors_of_the_formula_and_the_data():
    d = small()
    err = pytest.raises
    with err(ValueError, match="No strata"):
        tf.move.fit_issf(d, "case_ ~ x")
    with err(ValueError, match="both|too"):
        tf.move.fit_issf(d, "case_ ~ x + strata(step_id_)", strata="step_id_")
    with err(ValueError, match="not supported"):
        tf.move.fit_issf(d, "case_ ~ x + tan(z) + strata(step_id_)")
    with err(ValueError, match="not supported"):
        tf.move.fit_issf(d, "case_ ~ poly(x, 2) + strata(step_id_)")
    with err(ValueError, match="one strata"):
        tf.move.fit_issf(d, "case_ ~ x + strata(step_id_) + strata(z)")
    with err(ValueError, match="one argument"):
        tf.move.fit_issf(d, "case_ ~ x + strata(step_id_, z)")
    with err(ValueError, match="~"):
        tf.move.fit_issf(d, "x + strata(step_id_)")
    with err(ValueError, match="not in the data"):
        tf.move.fit_issf(d, "case_ ~ w + strata(step_id_)")
    with err(ValueError, match="not in the data"):
        tf.move.fit_issf(d, "case_ ~ x + strata(nope)")
    with err(ValueError, match="cannot read"):
        tf.move.fit_issf(d, "case_ ~ x + $z + strata(step_id_)")
    with err(ValueError, match="unexpected"):
        tf.move.fit_issf(d, "case_ ~ x + + * strata(step_id_)")
    with err(ValueError, match="not supported"):
        tf.move.fit_issf(d, "case_ ~ x %in% z + strata(step_id_)")
    with err(ValueError, match="no covariates"):
        tf.move.fit_issf(d, "case_ ~ strata(step_id_)")
    with err(ValueError, match="cases"):
        tf.move.fit_issf(d.assign(case_=d.case_.astype(int) + 1), "case_ ~ x + strata(step_id_)")
    with err(ValueError, match="no case"):
        tf.move.fit_issf(d.assign(case_=False), "case_ ~ x + strata(step_id_)")
    with err(ValueError, match="infinite"):
        tf.move.fit_issf(d.assign(z=0.0), "case_ ~ x + log(z) + strata(step_id_)")
    with err(ValueError, match="constant"):
        tf.move.fit_issf(d.assign(k=5.0), "case_ ~ x + k + strata(step_id_)")
    with err(ValueError, match="fewer than 2 levels"):
        tf.move.fit_issf(d.assign(f="a"), "case_ ~ x + f + strata(step_id_)")
    with err(TypeError):
        tf.move.log_rss("fit", pd.DataFrame(), pd.DataFrame())


def test_a_constant_of_ones_is_a_column_without_coefficient_and_zero_is_too():
    d = small().assign(one=1.0, zero=0.0)
    m = tf.move.fit_issf(d, "case_ ~ x + one + zero + strata(step_id_)")
    assert m.coef.isna().tolist() == [False, True, True]
    assert m.edf == 1 and m.aic == -2 * m.loglik[1] + 2


def test_log_rss_arguments_and_values():
    d = small()
    m = tf.move.fit_issf(d, "case_ ~ x + z + strata(step_id_)")
    x1, x2 = pd.DataFrame({"x": [0.0, 1.0, 2.0], "z": [0.0, 0.0, 1.0]}), pd.DataFrame({"x": [0.0], "z": [0.0]})
    r = tf.move.log_rss(m, x1, x2)
    # the same as (x1 - x2) b, up to the last digits
    np.testing.assert_allclose(r["log_rss"], x1["x"] * m.coef["x"] + x1["z"] * m.coef["z"], rtol=1e-12, atol=1e-15)
    assert r.columns.tolist() == ["x_x1", "z_x1", "log_rss"]
    se = tf.move.log_rss(m, x1, x2, ci="se")
    assert se["lwr"].iloc[0] == se["upr"].iloc[0] == 0.0   # x1 = x2 there
    v = m.vcov.to_numpy()
    for i in (1, 2):
        a = (x1.iloc[i] - x2.iloc[0]).to_numpy()
        half = 1.959963984540054 * np.sqrt(a @ v @ a)
        np.testing.assert_allclose([se["lwr"].iloc[i], se["upr"].iloc[i]], r["log_rss"].iloc[i] + np.array([-half, half]),
                                   rtol=1e-12)
    wide = tf.move.log_rss(m, x1, x2, ci="se", ci_level=0.99)
    assert (wide["upr"] - wide["lwr"]).iloc[1] > (se["upr"] - se["lwr"]).iloc[1]
    # the names amt gives the columns: _x1 is added unless it is there, x1 after a trailing _
    from terraflux.move._issf import _append_x1
    assert [_append_x1(s) for s in ("elevation", "sl_", "z_x1", "x")] == ["elevation_x1", "sl_x1", "z_x1", "x_x1"]
    # columns the formula does not use are fine
    assert len(tf.move.log_rss(m, x1.assign(other="a"), x2.assign(other="a"))) == 3
    with pytest.raises(ValueError, match="exactly 1 row"):
        tf.move.log_rss(m, x1, x1)
    with pytest.raises(ValueError, match="ci"):
        tf.move.log_rss(m, x1, x2, ci="percentile")
    with pytest.raises(ValueError, match="ci_level"):
        tf.move.log_rss(m, x1, x2, ci="se", ci_level=1.5)
    with pytest.raises(ValueError, match="n_boot"):
        tf.move.log_rss(m, x1, x2, ci="boot", n_boot=0)
    with pytest.raises(TypeError, match="DataFrame"):
        tf.move.log_rss(m, {"x": [0.0]}, x2)
    with pytest.raises(ValueError, match="not in the data"):
        tf.move.log_rss(m, x1[["x"]], x2)
    nomodel = tf.move.fit_issf(d, "case_ ~ x + z + strata(step_id_)", model=False)
    assert nomodel._data is None
    same(tf.move.log_rss(nomodel, x1, x2)["log_rss"], r["log_rss"])
    with pytest.raises(ValueError, match="model=True"):
        tf.move.log_rss(nomodel, x1, x2, ci="boot")


def test_the_bootstrap_draws_new_strata_and_repeats_with_the_seed():
    d = small(5, n=40)
    m = tf.move.fit_issf(d, "case_ ~ x + log(abs(z) + 1) + strata(step_id_)")
    x1, x2 = pd.DataFrame({"x": [1.0], "z": [0.5]}), pd.DataFrame({"x": [0.0], "z": [0.5]})
    a = tf.move.log_rss(m, x1, x2, ci="boot", n_boot=12, seed=3)
    b = tf.move.log_rss(m, x1, x2, ci="boot", n_boot=12, seed=3)
    c = tf.move.log_rss(m, x1, x2, ci="boot", n_boot=12, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert a["lwr"].iloc[0] != c["lwr"].iloc[0]
    assert a["lwr"].iloc[0] < a["log_rss"].iloc[0] < a["upr"].iloc[0]


def test_strata_given_as_values_and_the_order_of_their_levels():
    d = small()
    a = tf.move.fit_issf(d, "case_ ~ x + z + strata(step_id_)")
    b = tf.move.fit_issf(d, "case_ ~ x + z", strata=d["step_id_"].to_numpy())
    pd.testing.assert_series_equal(a.coef, b.coef)
    # strata of a track and a step: made into one column, the steps of two tracks do not share a stratum
    two = pd.concat([d, d.assign(step_id_=d.step_id_)], ignore_index=True)
    merged = tf.move.fit_issf(two, "case_ ~ x + z + strata(step_id_)")
    assert merged.nevent == 120
    apart = tf.move.fit_issf(two.assign(sid=two.step_id_.astype(str) + np.repeat(["a", "b"], 300)),
                             "case_ ~ x + z + strata(sid)")
    assert apart.nevent == 120 and not np.allclose(merged.coef, apart.coef, rtol=1e-6)


def test_no_convergence_is_a_warning():
    d = small()
    d["perfect"] = d.case_.astype(float)   # separates the cases from the other steps
    with pytest.warns(RuntimeWarning, match="beta may be infinite|did not converge"):
        tf.move.fit_issf(d, "case_ ~ perfect + x + strata(step_id_)")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tf.move.fit_issf(d, "case_ ~ x + z + strata(step_id_)")


def test_the_pieces_of_r_arithmetic():
    from terraflux import _core
    # pchisq(1, 1, lower.tail = FALSE), 2 * pnorm(-1)
    p = _core.move_issf_pchisq(np.array([1.0, 0.0, 3.841458820694124, 400.0]), 1.0, False)
    np.testing.assert_allclose(p, [math.erfc(1 / math.sqrt(2)), 1.0, 0.05, math.erfc(20 / math.sqrt(2))], rtol=1e-12)
    assert _core.move_issf_pchisq(np.array([np.nan]), 1.0, False)[0] != _core.move_issf_pchisq(np.array([np.nan]), 1.0, False)[0]
    np.testing.assert_allclose(_core.move_issf_qnorm(0.975), 1.959963984540054, rtol=1e-15)
    assert _core.move_issf_qnorm(0.0) == -np.inf and np.isnan(_core.move_issf_qnorm(1.5))
    draws = _core.move_issf_sample(1, 10, 3, 2)
    assert draws.shape == (2, 3) and draws.min() >= 1 and draws.max() <= 10
    np.testing.assert_array_equal(draws, _core.move_issf_sample(1, 10, 3, 4)[:2])
