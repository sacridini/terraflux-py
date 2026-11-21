# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's directional Brownian bridge (dBGB) against R's move2utils 0.4.6 (tests/move_cases.py, group
"move2utils_ud"; tools/move2utils_fixed.R) and, by tolerance, against the legacy ``move`` package's dynBGBvariance.

As for the dBBMM (tests/test_move_ud.py): the variance and the distribution are compared bit for bit with move2utils in
R, run with one thread, from the projected coordinates R's PROJ gave where the tracks are in longitude / latitude;
from longitude / latitude itself, by tolerance.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import terraflux as tf
from conftest import BITWISE, UCRT_LIBM
from move_cases import UD_DBGB, UD_DBGB_VARIANCE
from test_move_ud import libm, make_like, make_tracks, projected_tracks, same, with_projected

REF = Path(__file__).parent / "data" / "reference" / "move"
DATA = Path(__file__).parent / "data" / "move"


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "move2utils_ud.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs_move2utils_ud.npz"))


@pytest.fixture(scope="module")
def fishers():
    tr = tf.load_tracks(DATA / "fishers.csv.gz", sort=False)
    return tr[~tr.geometry.is_empty]


# ------------------------------------------------------------------------------------------------ acos

def test_acos_is_the_one_of_r(ref):
    from terraflux import _core

    a = ref["acos/x"]
    got = _core.move_r_acos(a)
    if UCRT_LIBM:          # R on Windows takes acos from the UCRT; other libraries differ in the last bit
        same(got, ref["acos/y"], "acos")
    else:
        np.testing.assert_allclose(got, ref["acos/y"], rtol=1e-15, atol=0)
    assert np.isnan(_core.move_r_acos(np.array([1.5, np.nan, -2.0]))).all()


# ------------------------------------------------------------------------------------- the variance

VARIANCE_CALLS = {
    "gvar_fishers": ("fishers", None, dict(location_error=25)),
    "gvar_m1": ("fishers", "M1", dict(location_error=25)),
    "gvar_m1_w21": ("fishers", "M1", dict(location_error=15, window_size=21, margin=5)),
    "gvar_utm": ("ud_utm", None, dict(location_error=10, window_size=21, margin=7)),
    "gvar_noerr": ("ud_utm", None, dict(location_error=None, window_size=21, margin=7)),
    "gvar_ll": ("ud_ll", None, dict(location_error=5)),
    "gvar_burst": ("ud_burst", "burst", dict(location_error="le", window_size=21, margin=7, location_error_na="approx")),
    "gvar_err_median": ("ud_err", None, dict(location_error="le", location_error_na="median")),
    "gvar_err_approx": ("ud_err", None, dict(location_error="le", location_error_na="approx")),
    "gvar_argos": ("ud_err", "argos", dict(location_error="auto")),
}
LONLAT = ("gvar_fishers", "gvar_m1", "gvar_m1_w21", "gvar_ll")


def test_every_variance_case_is_tested():
    assert set(VARIANCE_CALLS) == set(UD_DBGB_VARIANCE)


def check_variance(out, ref, case):
    kept = out["dbgb_location_error"].notna().to_numpy()
    assert kept.sum() == ref[f"{case}/n"].sum()
    libm(out["para_sd"][kept], ref[f"{case}/para_sd"], "para_sd")
    libm(out["orth_sd"][kept], ref[f"{case}/orth_sd"], "orth_sd")
    same(out["n_estim"][kept], ref[f"{case}/n_estim"], "n_estim")
    same(out["seg_interest"][kept], ref[f"{case}/seg_interest"], "seg_interest")
    libm(out["dbgb_location_error"][kept], ref[f"{case}/le"], "location error")


@pytest.mark.parametrize("case", sorted(VARIANCE_CALLS))
def test_dbgb_variance_is_move2utils(ref, inputs, fishers, case):
    data, prep, kw = VARIANCE_CALLS[case]
    tr = make_tracks(inputs, fishers, data, prep)
    if case in LONLAT:
        kept = tf.move.dbgb_variance(tr, **kw)["dbgb_location_error"].notna().to_numpy()
        crs = tf.move.aeqd_crs(tr, "centroid")
        tr = with_projected(tr[kept], [ref[f"{case}/x"], ref[f"{case}/y"]], crs)
    check_variance(tf.move.dbgb_variance(tr, **kw), ref, case)


@pytest.mark.parametrize("case", LONLAT)
def test_the_variance_of_lonlat_tracks_agrees_with_move2utils(ref, inputs, fishers, case):
    """From longitude / latitude: pyproj's PROJ and sf's differ by units in the last place of the coordinates."""
    data, prep, kw = VARIANCE_CALLS[case]
    tr = make_tracks(inputs, fishers, data, prep)
    out = tf.move.dbgb_variance(tr, **kw)
    kept = out["dbgb_location_error"].notna().to_numpy()
    for key in ("para_sd", "orth_sd"):
        got, want = out[key][kept].to_numpy(), ref[f"{case}/{key}"]
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4, equal_nan=True)
        ok = ~np.isnan(got)
        assert np.median(np.abs(got[ok] - want[ok]) / np.maximum(np.abs(want[ok]), 1e-9)) < 1e-6
    same(out["n_estim"][kept], ref[f"{case}/n_estim"], "n_estim")


def test_the_variance_agrees_with_the_legacy_move_package(ref, inputs, fishers):
    """move2utils is not bit for bit with ``move``'s dynBGBvariance: the tolerances are those of its own test."""
    tr = make_tracks(inputs, fishers, "fishers", "M1")
    xy = ref["move_leroy_dbgb/xy"].reshape(2, -1)
    import geopandas as gpd

    plain = gpd.GeoDataFrame(pd.DataFrame(tr.drop(columns=tr.geometry.name)),
                             geometry=gpd.points_from_xy(xy[0], xy[1]))
    v = tf.move.dbgb_variance(plain, 25, 31, 11)
    for key, want in (("para_sd", ref["move_leroy_dbgb/para"]), ("orth_sd", ref["move_leroy_dbgb/orth"])):
        got = v[key].to_numpy()
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        d = np.abs(got[~np.isnan(got)] - want[~np.isnan(want)])
        assert np.quantile(d, 0.9) < 1e-2 and np.median(d) < 1e-3
    # and what R itself gave for this: move2utils' own object
    same(v["para_sd"], ref["move_leroy_dbgb/my_para"], key)
    same(v["orth_sd"], ref["move_leroy_dbgb/my_orth"], key)


# ------------------------------------------------------------------------------------ the distribution

UD_CALLS = {
    "gud_m1_res": ("fishers", "M1", True, dict(location_error=25, res=100)),
    "gud_m1_dim": ("fishers", "M1", True, dict(location_error=25, dim_size=120, ext_factor=0.3, window_size=21, margin=5)),
    "gud_m1_enlarge_dim": ("fishers", "M1", True, dict(location_error=25, dim_size=60, ext_factor=0)),
    "gud_m1_enlarge_res": ("fishers", "M1", True, dict(location_error=25, res=200, ext_factor=0)),
    "gud_m1_step": ("fishers", "M1", True, dict(location_error=25, res=150, time_step=3)),
    "gud_m1_like": ("fishers", "M1", True, dict(location_error=25, like=True)),
    "gud_fishers500": ("fishers", "first500", True, dict(location_error=25, dim_size=150)),
    "gud_utm": ("ud_utm", None, False, dict(location_error=10, dim_size=90, window_size=21, margin=7)),
    "gud_ll": ("ud_ll", None, True, dict(location_error=5, dim_size=80)),
    "gud_argos": ("ud_err", "argos", False, dict(location_error="auto", dim_size=100)),
    "gud_eobs": ("ud_err", None, False, dict(location_error="auto", res=40)),
    "gud_burst": ("ud_burst", "burst", False, dict(location_error=5, window_size=21, margin=7, res=40)),
    "gud_masked": ("ud_utm", "u1", False, dict(res=20)),
}


def test_every_distribution_case_is_tested():
    assert set(UD_CALLS) == set(UD_DBGB)


@pytest.mark.parametrize("case", sorted(UD_CALLS))
def test_dbgb_ud_is_move2utils(ref, inputs, fishers, case):
    data, prep, projected, kw = UD_CALLS[case]
    kw = dict(kw)
    if prep == "first500":
        tr = fishers[fishers.groupby("individual-local-identifier", sort=False).cumcount() < 500]
    elif prep == "u1":
        tr = make_tracks(inputs, fishers, data, None)
        tr = tr[tr["track"] == "u1"]
    else:
        tr = make_tracks(inputs, fishers, data, prep)
    crs = None
    if projected:
        tr, crs = projected_tracks(ref, case, tr)
    nr, nc, nl = ref[f"{case}/dim"].astype(int)
    ext = ref[f"{case}/ext"]
    if kw.pop("like", False):
        kw["like"] = make_like(ext, nr, nc, crs)
    if case == "gud_masked":
        v = tf.move.dbgb_variance(tr, 10, 21, 7)
        v = tf.move.mask_segments(v, np.r_[np.arange(30, 41), 90, 150] - 1)       # R counts from 1
        same(v["seg_interest"], ref[f"{case}/seg_interest"], "seg_interest")
        ud = tf.move.dbgb_ud(v, **kw)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)     # the track of ud_utm shorter than the window
            ud = tf.move.dbgb_ud(tr, **kw)
    assert ud.shape == (nl, nr, nc)
    assert ud.attrs["extent"] == tuple(ext)
    got, want = ud.values.reshape(nl, -1), ref[f"{case}/ud"].reshape(nl, -1)
    if case == "gud_m1_like" and BITWISE:
        # this grid reaches cells 8 km from the bridges, where a sum of 256 terms of 1e-22 rests on the last bit of
        # each: one cell of 24 024 comes out one unit in the last place from R's (its sin(), not exp or acos, which
        # are R's bit for bit; see the report), every other one is R's bit for bit
        assert np.count_nonzero(got != want) <= 1
        np.testing.assert_allclose(got, want, rtol=3e-16, atol=0)
    else:
        libm(got, want, "the distribution")
    np.testing.assert_allclose(ud.values.reshape(nl, -1).sum(axis=1), 1.0, rtol=1e-12)
    assert ud.name == "dbgb_ud" and ud.attrs["ext"] == 5.0
    if case == "gud_burst":
        assert list(ud["band"].values) == ["b1__1", "b1__2", "b2__1", "b2__2"]
    if case == "gud_utm":
        assert list(ud["band"].values) == ["u1", "u2"]


def test_the_distribution_of_longitude_latitude_agrees_with_move2utils(ref, inputs, fishers):
    tr = make_tracks(inputs, fishers, "ud_ll", None)
    ud = tf.move.dbgb_ud(tr, 5, dim_size=80)
    nr, nc, nl = ref["gud_ll/dim"].astype(int)
    assert ud.shape == (nl, nr, nc)
    np.testing.assert_allclose(ud.values.reshape(nl, -1), ref["gud_ll/ud"].reshape(nl, -1), rtol=1e-4, atol=1e-10)


# ------------------------------------------------------------------------------------------- by hand

def sim(n=160, tracks=2, sigma=0.3, seed=5, crs=None):
    tr = tf.move.simulate_brownian(np.arange(n) * 600.0, sigma=sigma, tracks=tracks, seed=seed)
    return tr.set_crs(crs) if crs else tr


def test_the_columns_and_their_shape():
    tr = sim(120, 2)
    v = tf.move.dbgb_variance(tr, 5.0, 31, 15)
    assert len(v) == len(tr) and v.crs == tr.crs
    assert {"para_sd", "orth_sd", "n_estim", "seg_interest", "dbgb_location_error"} <= set(v.columns)
    assert not {"motion_variance", "interest"} & set(v.columns)
    for _, g in v.groupby("track"):
        covered = g["para_sd"].notna().to_numpy()
        assert covered.sum() == 120 - 31 + 2 and (g["para_sd"][covered] >= 0).all()           # 2 positions per window
        np.testing.assert_array_equal(covered, g["orth_sd"].notna().to_numpy())
        assert g["n_estim"].max() == 2 and (g["seg_interest"] == (g["n_estim"] == 2)).all()
    info = v.attrs["terraflux.dbgb"]
    assert (info["window_size"], info["margin"]) == (31, 15) and info["crs"] is None
    mv = tf.move.motion_variance(v)
    np.testing.assert_array_equal(mv["para"], v["para_sd"] ** 2)
    np.testing.assert_array_equal(mv["orth"], v["orth_sd"] ** 2)
    assert "para_sd" not in tr.columns
    with pytest.raises(ValueError, match="run dbgb_variance"):
        tf.move.motion_variance(tr, model="dbgb")


def test_a_track_across_its_direction_has_a_larger_orthogonal_variance():
    n = 220
    rng = np.random.default_rng(2)
    x = np.arange(n) * 20.0 + rng.normal(0, 0.5, n)                  # along x at a steady pace
    y = rng.normal(0, 12.0, n)                                        # but wandering across it
    import geopandas as gpd

    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": pd.to_datetime(np.arange(n) * 600.0, unit="s", utc=True),
                                             "track": "a"}, geometry=gpd.points_from_xy(x, y)), "time", "track")
    v = tf.move.dbgb_variance(tr, 0.0, 31, 15)
    assert np.nanmedian(v["orth_sd"]) > 2 * np.nanmedian(v["para_sd"])


def test_short_tracks_the_window_and_the_masks():
    long, short = sim(80, 1), sim(20, 1)
    both = tf.move.as_tracks(pd.concat([long, short.assign(track=2)], ignore_index=True), "time", "track", sort=False)
    with pytest.warns(UserWarning, match="skipped.*2"):
        v = tf.move.dbgb_variance(both, 5.0)
    assert v["para_sd"].iloc[80:].isna().all() and v["dbgb_location_error"].iloc[80:].isna().all()
    with pytest.warns(UserWarning, match="skipped"):
        ud = tf.move.dbgb_ud(both, 5.0, dim_size=30)
    assert list(ud["band"].values) == ["1"]
    with pytest.raises(ValueError, match="cannot be larger"):
        tf.move.dbgb_variance(short, 5.0)
    with pytest.raises(ValueError, match="odd|at least"):
        tf.move.dbgb_variance(long, 5.0, 31, 14)
    with pytest.raises(ValueError, match="at least"):
        tf.move.dbgb_variance(long, 5.0, 21, 15)
    # the masks: by model when the tracks hold both
    v = tf.move.dbgb_variance(long, 5.0)
    n0 = int(v["seg_interest"].sum())
    m = tf.move.mask_segments(v, [40, 41])
    assert int(m["seg_interest"].sum()) == n0 - 2 and "interest" not in m.columns
    w = tf.move.dbbmm_variance(v, 5.0)
    with pytest.raises(ValueError, match="both"):
        tf.move.mask_segments(w, [40])
    assert int(tf.move.mask_segments(w, [40], model="dbgb")["seg_interest"].sum()) == n0 - 1
    assert int(tf.move.mask_segments(w, [40], model="dbbmm")["interest"].sum()) == int(w["interest"].sum()) - 1
    with pytest.raises(ValueError, match="no seg_interest"):
        tf.move.mask_segments(tf.move.dbbmm_variance(long, 5.0), [1], model="dbgb")


def test_ud_with_the_columns_and_without_are_the_same_and_the_grid_is_quiet():
    tr = sim(120, 2)
    one = tf.move.dbgb_ud(tr, 5.0, res=15)
    two = tf.move.dbgb_ud(tf.move.dbgb_variance(tr, 5.0), res=15)
    np.testing.assert_array_equal(one.values, two.values)
    both = tf.move.dbgb_ud(tf.move.dbbmm_variance(tf.move.dbgb_variance(tr, 5.0), 5.0), res=15)   # the two models together
    np.testing.assert_array_equal(both.values, one.values)
    assert one.name == "dbgb_ud" and float(one.isel(band=1).sum()) == pytest.approx(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # the grid was enlarged (a box of 0.5 is too small here): nothing said
        tf.move.dbgb_ud(sim(100, 1, sigma=0.6), 5.0, res=50, ext_factor=0)
    with pytest.warns(UserWarning, match="enlarged"):
        tf.move.dbgb_ud(sim(100, 1, sigma=0.6), 5.0, res=50, ext_factor=0, verbose=True)


def test_the_distribution_needs_a_bridge_of_interest():
    tr = sim(100, 1)
    v = tf.move.dbgb_variance(tr, 5.0)
    full = tf.move.dbgb_ud(v, res=20)
    with pytest.raises(ValueError, match="at least one bridge"):
        tf.move.dbgb_ud(tf.move.mask_segments(v, np.arange(0, 99)), like=full)
    part = tf.move.dbgb_ud(tf.move.mask_segments(v, np.arange(0, 50)), like=full)
    assert float(part.sum()) == pytest.approx(1.0) and not np.array_equal(part.values, full.values)
    with pytest.raises(ValueError, match="time_step"):
        tf.move.dbgb_ud(v, time_step=0)


def test_one_and_four_threads_give_the_same():
    tr = sim(1500, 2, sigma=0.5)
    a = tf.move.dbgb_variance(tr, 5.0, n_threads=1)
    b = tf.move.dbgb_variance(tr, 5.0, n_threads=4)
    pd.testing.assert_frame_equal(pd.DataFrame(a.drop(columns="geometry")), pd.DataFrame(b.drop(columns="geometry")))
    assert a["seg_interest"].sum() > 2000
    one = tf.move.dbgb_ud(a, dim_size=300, n_threads=1)
    four = tf.move.dbgb_ud(a, dim_size=300, n_threads=4)
    np.testing.assert_array_equal(one.values, four.values)
