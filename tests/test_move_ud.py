# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's dynamic Brownian bridge against R's move2utils 0.4.6 (tests/move_cases.py, group "move2utils_ud";
tools/move2utils_fixed.R) and, by tolerance, against the legacy ``move`` package.

The motion variance and the distribution are compared bit for bit with move2utils in R, run with one thread (its
sums of the variance are added in no fixed order with more). Where it starts from longitude / latitude the references
also hold the projected coordinates its PROJ gave, and the distribution is computed from those (move2utils warps it
back to longitude / latitude through terra and GDAL, which is not portable: tf.move returns it in the projected CRS,
``crs="input"`` warps it, within a tolerance). Where move2utils' ``mt_aeqd_crs`` is wrong (it centres the projection
on a point moved to EPSG:4236), the references use the corrected one, as ``tf.move.aeqd_crs`` does.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely import Point

import terraflux as tf
from conftest import BITWISE, X87_EXACT
from move_cases import UD_DIST, UD_VARIANCE

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


def frame(inputs, name):
    """The tracks of a set stored with the references."""
    x, y, t = inputs[f"{name}/x"], inputs[f"{name}/y"], inputs[f"{name}/t"]
    crs = str(inputs[f"{name}/crs"]) or None
    time = pd.to_datetime((t * 1e9).astype(np.int64), utc=True)
    df = {"time": time, "track": inputs[f"{name}/track"]}
    for key in ("le", "lc", "eobs_horizontal_accuracy_estimate", "burst"):
        if f"{name}/{key}" in inputs:
            df[key] = inputs[f"{name}/{key}"]
    return tf.move.as_tracks(gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x, y), crs=crs), "time", "track",
                             sort=False)


def make_tracks(inputs, fishers, data, prep):
    """The tracks a case runs on: its data set with the same steps as the R code (`prep`)."""
    tr = fishers if data == "fishers" else frame(inputs, data)
    if prep == "M1":
        tr = tr[tr["individual-local-identifier"].astype(str) == "M1"]
    elif prep == "burst":
        tr = tr.copy()
        tr["burst_"] = tr.pop("burst").astype(int)
    elif prep == "argos":
        tr = tr.drop(columns="eobs_horizontal_accuracy_estimate").copy()
        tr["argos_lc"] = np.array(["3", "2", "1", "0", "A", "B", "Z"])[tr["lc"].to_numpy().astype(int) - 1]
    return tr


def same(got, want, err=""):
    np.testing.assert_array_equal(np.asarray(got, float), np.asarray(want, float), err_msg=err)


def libm(got, want, err="", rtol=1e-9):
    """Bit for bit where the references were made (an Intel CPU's x87 functions, see ``conftest.X87_EXACT``); elsewhere
    within ``rtol`` (1e-9 where only the last bits of a sum move)."""
    if X87_EXACT:
        same(got, want, err)
    else:
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), rtol=rtol, atol=1e-12,
                                   equal_nan=True, err_msg=err)


# Measured off the references' CPU: the variance of the likelihood's optimum moves by up to 3.7e-8 of itself on AMD
# x86 (Windows and Linux CI) and 2.7e-7 on ARM64 (a standard deviation of the dBGB by 1.1e-6 there, on Linux and
# macOS alike); the distributions by 2.7e-10 and 3.7e-9.
VARIANCE_RTOL = 1e-6 if BITWISE else 1e-5
UD_RTOL = 1e-7


def with_projected(tr, xy, crs):
    """The tracks with the projected coordinates R's PROJ gave."""
    xy = np.asarray(xy, float).reshape(2, -1)
    out = gpd.GeoDataFrame(pd.DataFrame(tr.drop(columns=tr.geometry.name)),
                           geometry=gpd.points_from_xy(xy[0], xy[1]), crs=crs)
    out.attrs = dict(tr.attrs)
    return out


# ------------------------------------------------------------------------------------- the variance

VARIANCE_CALLS = {
    "var_fishers": ("fishers", None, dict(location_error=25)),
    "var_m1": ("fishers", "M1", dict(location_error=25)),
    "var_m1_w21": ("fishers", "M1", dict(location_error=15, window_size=21, margin=5)),
    "var_utm": ("ud_utm", None, dict(location_error=10, window_size=21, margin=7)),
    "var_noerr": ("ud_utm", None, dict(location_error=None, window_size=21, margin=7)),
    "var_ll": ("ud_ll", None, dict(location_error=5)),
    "var_burst": ("ud_burst", "burst", dict(location_error=5, window_size=21, margin=7)),
    "var_burst_err": ("ud_burst", "burst", dict(location_error="le", window_size=21, margin=7,
                                                 location_error_na="approx")),
    "var_err_median": ("ud_err", None, dict(location_error="le", location_error_na="median")),
    "var_err_mean": ("ud_err", None, dict(location_error="le", location_error_na="mean")),
    "var_err_zero": ("ud_err", None, dict(location_error="le", location_error_na="zero")),
    "var_err_approx": ("ud_err", None, dict(location_error="le", location_error_na="approx")),
    "var_err_vec": ("ud_err", None, dict(location_error="le")),
    "var_argos": ("ud_err", "argos", dict(location_error="auto")),
    "var_eobs": ("ud_err", None, dict(location_error="auto")),
}


def test_every_case_is_tested():
    assert set(VARIANCE_CALLS) == set(UD_VARIANCE)


LONLAT = ("var_fishers", "var_m1", "var_m1_w21", "var_ll")


def check_variance(out, ref, case):
    """`out` (the tracks with their variance) is what move2utils gave, bit for bit, for the tracks it did not skip."""
    kept = out["location_error"].notna().to_numpy()                   # the tracks R did not skip are the ones with errors
    assert kept.sum() == ref[f"{case}/n"].sum()
    libm(out["motion_variance"][kept], ref[f"{case}/variance"], "variance", VARIANCE_RTOL)
    same(out["in_windows"][kept], ref[f"{case}/in_windows"], "in_windows")
    same(out["interest"][kept], ref[f"{case}/interest"], "interest")
    libm(out["location_error"][kept], ref[f"{case}/le"], "location error")
    info = out.attrs["terraflux.dbbmm"]
    breaks = [np.asarray(b) + 1 for b in info["breaks"].values()]       # R counts from 1
    same([len(b) for b in breaks], ref[f"{case}/nbreaks"], "number of breaks")
    same(np.concatenate(breaks) if breaks else [], ref[f"{case}/breaks"], "breaks")


@pytest.mark.parametrize("case", sorted(VARIANCE_CALLS))
def test_dbbmm_variance_is_move2utils(ref, inputs, fishers, case):
    data, prep, kw = VARIANCE_CALLS[case]
    tr = make_tracks(inputs, fishers, data, prep)
    if case == "var_err_vec":
        kw = dict(location_error=tr["le"].to_numpy())
    if case in LONLAT:
        # from the projected coordinates R's PROJ gave: the variance is R's bit for bit
        kept = tf.move.dbbmm_variance(tr, **kw)["location_error"].notna().to_numpy()
        crs = tf.move.aeqd_crs(tr, "centroid")
        tr = with_projected(tr[kept], [ref[f"{case}/x"], ref[f"{case}/y"]], crs)
    check_variance(tf.move.dbbmm_variance(tr, **kw), ref, case)


@pytest.mark.parametrize("case", LONLAT)
def test_the_variance_of_lonlat_tracks_agrees_with_move2utils(ref, inputs, fishers, case):
    """From longitude / latitude the projection is PROJ's in pyproj (9.5) and in sf (9.4): coordinates that differ by
    a few units in the last place (1e-9 m), which the likelihoods of nearly still stretches carry to ~1e-7."""
    from pyproj import Transformer

    data, prep, kw = VARIANCE_CALLS[case]
    tr = make_tracks(inputs, fishers, data, prep)
    out = tf.move.dbbmm_variance(tr, **kw)
    kept = out["location_error"].notna().to_numpy()
    crs = tf.move.aeqd_crs(tr, "centroid")
    t = tr[kept]
    x, y = Transformer.from_crs(tr.crs, crs, always_xy=True).transform(t.geometry.x.to_numpy(), t.geometry.y.to_numpy())
    np.testing.assert_allclose(x, ref[f"{case}/x"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(y, ref[f"{case}/y"], rtol=0, atol=1e-7)
    got, want = out["motion_variance"][kept].to_numpy(), ref[f"{case}/variance"]
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-6, equal_nan=True)
    ok = ~np.isnan(got)
    assert np.median(np.abs(got[ok] - want[ok]) / np.maximum(np.abs(want[ok]), 1e-12)) < 1e-6
    same(out["in_windows"][kept], ref[f"{case}/in_windows"], "in_windows")


# ------------------------------------------------------------------------------------ the distribution

UD_CALLS = {
    "ud_m1_res": ("fishers", "M1", True, dict(location_error=25, res=100)),
    "ud_m1_dim": ("fishers", "M1", True, dict(location_error=25, dim_size=120, ext_factor=0.3, window_size=21, margin=5)),
    "ud_m1_enlarge_dim": ("fishers", "M1", True, dict(location_error=25, dim_size=60, ext_factor=0)),
    "ud_m1_enlarge_res": ("fishers", "M1", True, dict(location_error=25, res=200, ext_factor=0)),
    "ud_m1_step": ("fishers", "M1", True, dict(location_error=25, res=150, time_step=3)),
    "ud_m1_like": ("fishers", "M1", True, dict(location_error=25, like=True)),
    "ud_fishers500": ("fishers", "first500", True, dict(location_error=25, dim_size=150)),
    "ud_utm": ("ud_utm", None, False, dict(location_error=10, dim_size=90, window_size=21, margin=7)),
    "ud_ll": ("ud_ll", None, True, dict(location_error=5, dim_size=80)),
    "ud_argos": ("ud_err", "argos", False, dict(location_error="auto", dim_size=100)),
    "ud_eobs": ("ud_err", None, False, dict(location_error="auto", res=40)),
    "ud_err_approx": ("ud_err", None, False, dict(location_error="le", location_error_na="approx", res=40)),
    "ud_burst": ("ud_burst", "burst", False, dict(location_error=5, window_size=21, margin=7, res=40)),
    "ud_masked": ("ud_utm", "u1", False, dict(res=20)),
}


def test_every_distribution_case_is_tested():
    assert set(UD_CALLS) == set(UD_DIST)


def make_like(ext, nrow, ncol, crs):
    from affine import Affine

    xmin, xmax, ymin, ymax = ext
    xres, yres = (xmax - xmin) / ncol, (ymax - ymin) / nrow
    da = xr.DataArray(np.zeros((nrow, ncol)), dims=("y", "x"),
                      coords={"y": ymax - (np.arange(nrow) + 0.5) * yres, "x": xmin + (np.arange(ncol) + 0.5) * xres})
    da.rio.write_crs(crs, inplace=True)
    da.rio.write_transform(Affine(xres, 0.0, xmin, 0.0, -yres, ymax), inplace=True)
    return da


def projected_tracks(ref, case, tr):
    crs = tf.move.aeqd_crs(tr, "center")
    return with_projected(tr, ref[f"{case}/xy"], crs), crs


@pytest.mark.parametrize("case", sorted(UD_CALLS))
def test_dbbmm_ud_is_move2utils(ref, inputs, fishers, case):
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
    if case == "ud_masked":
        v = tf.move.dbbmm_variance(tr, 10, 21, 7)
        v = tf.move.mask_segments(v, np.r_[np.arange(30, 41), 90, 150] - 1)       # R counts from 1
        same(v["interest"], ref[f"{case}/interest"], "interest")
        ud = tf.move.dbbmm_ud(v, **kw)
    elif "enlarge" in case:
        with pytest.warns(UserWarning, match="enlarged"):
            ud = tf.move.dbbmm_ud(tr, verbose=True, **kw)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)     # the track of ud_utm shorter than the window
            ud = tf.move.dbbmm_ud(tr, **kw)
    assert ud.shape == (nl, nr, nc)
    assert ud.attrs["extent"] == tuple(ext)
    libm(ud.values.reshape(nl, -1), ref[f"{case}/ud"].reshape(nl, -1), "the distribution", UD_RTOL)
    np.testing.assert_allclose(ud.values.reshape(nl, -1).sum(axis=1), 1.0, rtol=1e-12)
    assert ud.dims == ("band", "y", "x")
    if case == "ud_burst":
        assert list(ud["band"].values) == ["b1__1", "b1__2", "b2__1", "b2__2"]
        assert list(ud["track"].values) == ["b1", "b1", "b2", "b2"]
    if case == "ud_utm":
        assert list(ud["band"].values) == ["u1", "u2"]                  # u3 has fewer locations than the window


def test_the_distribution_of_longitude_latitude_is_computed_in_the_projection_of_sf(ref, inputs, fishers):
    """Without R's projected coordinates: the same distribution to the precision of PROJ's and pyproj's coordinates."""
    tr = make_tracks(inputs, fishers, "ud_ll", None)
    ud = tf.move.dbbmm_ud(tr, 5, dim_size=80)
    nr, nc, nl = ref["ud_ll/dim"].astype(int)
    assert ud.shape == (nl, nr, nc)
    got, want = ud.values.reshape(nl, -1), ref["ud_ll/ud"].reshape(nl, -1)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(ud.attrs["extent"], ref["ud_ll/ext"], rtol=1e-12)
    from pyproj import CRS

    got_crs, want_crs = CRS.from_user_input(ud.rio.crs).to_dict(), tf.move.aeqd_crs(tr, "center").to_dict()
    assert got_crs["proj"] == "aeqd" and all(abs(got_crs[k] - want_crs[k]) < 1e-9 for k in ("lat_0", "lon_0"))


def test_volume_and_outer_probability_are_move2utils(ref, inputs, fishers):
    data, prep, _, kw = UD_CALLS["ud_m1_res"]
    tr, _ = projected_tracks(ref, "ud_m1_res", make_tracks(inputs, fishers, data, prep))
    ud = tf.move.dbbmm_ud(tr, **kw)
    libm(tf.move.ud_volume(ud).values.reshape(-1), ref["ud_volume/vol"], "volume")
    xy = np.c_[tr.geometry.x.to_numpy(), tr.geometry.y.to_numpy()]
    pts = np.vstack([xy[::7], [0.0, 0.0]])
    out = tf.move.ud_outer_probability(pts, ud)
    libm(out, ref["ud_volume/outer"], "outer probability")
    assert not np.isnan(out).any()                                     # the centre of the projection is in the raster
    libm(tf.move.ud_outer_probability(tr.iloc[:40], ud), ref["ud_volume/outer_sf"], "outer probability of points")


def test_volume_of_cells_that_tie_is_move2utils(ref):
    v1 = np.tile([0.1, 0.05, 0.05, 0.2, 0.3, 0, 0.05], 6)
    v1[[3, 19, 32]] = np.nan
    v2 = np.r_[np.tile([1, 3, 3, 3, 2, 2, 3], 5), 7, 7, 1, 0, 0, 0, 0]
    stack = np.stack([v1, v2, np.zeros(42)]).reshape(3, 6, 7)
    da = xr.DataArray(stack, dims=("band", "y", "x"))
    same(tf.move.ud_volume(da).values.reshape(-1), ref["ud_volume_ties/vol"])
    same(tf.move.ud_volume(da.isel(band=0)).values.reshape(-1), ref["ud_volume_ties/vol"][:42])      # one layer alone


def flat(s):
    keys = ("window_size", "margin", "median_dt_secs", "window_hours", "n_windows")
    items = [s] if isinstance(s, tf.move.WindowSuggestion) or s is None else list(s.values())
    return np.concatenate([[np.nan] * 5 if x is None else [getattr(x, k) for k in keys] for x in items])


def test_suggest_dbbmm_window_is_move2utils(ref, inputs, fishers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for h, key in ((1, "h1"), (4, "h4"), (12, "h12")):
            same(flat(tf.move.suggest_dbbmm_window(fishers, h)), ref[f"suggest_window/{key}"], key)
        m1 = fishers[fishers["individual-local-identifier"].astype(str) == "M1"]
        same(flat(tf.move.suggest_dbbmm_window(m1, 4)), ref["suggest_window/m1_h4"])
        same(flat(tf.move.suggest_dbbmm_window(m1, 0.3)), ref["suggest_window/m1_h0_3"])
        utm = frame(inputs, "ud_utm")
        for h, key in ((1, "h1"), (5, "h5")):
            same(flat(tf.move.suggest_dbbmm_window(utm, h)), ref[f"suggest_window_utm/{key}"], key)


# --------------------------------------------------------------------------- the legacy move package

def test_the_variance_and_the_distribution_agree_with_the_legacy_move_package(ref, inputs, fishers):
    """move2utils is not bit for bit with ``move`` (the optimiser's tolerance, the constant of the kernel): the
    tolerances are the ones of move2utils' own parity tests."""
    tr = make_tracks(inputs, fishers, "fishers", "M1")
    xy = ref["move_leroy/xy"].reshape(2, -1)
    plain = gpd.GeoDataFrame(pd.DataFrame(tr.drop(columns=tr.geometry.name)),
                             geometry=gpd.points_from_xy(xy[0], xy[1]))
    v = tf.move.dbbmm_variance(plain, 25, 31, 11)
    want = ref["move_leroy/var"]
    got = v["motion_variance"].to_numpy()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(got)
    diff = np.abs(got[ok] - want[ok])
    assert np.median(diff / np.maximum(np.abs(want[ok]), 1e-12)) < 1e-6
    assert diff.max() < 1e-3
    ud = tf.move.dbbmm_ud(v, res=100, ext_factor=2)
    nr, nc = ref["move_leroy/dim"].astype(int)
    legacy = ref["move_leroy/ud"]
    mine = ud.values.reshape(-1)
    assert (nr, nc) == ud.shape[1:]
    legacy = legacy / np.nansum(legacy)
    np.testing.assert_allclose(ud.attrs["extent"], ref["move_leroy/ext"], rtol=1e-12)       # the same grid
    assert np.abs(np.sort(mine) - np.sort(legacy)).max() < 1e-6                              # move2utils' own test
    assert np.abs(mine - legacy).max() < 1e-6                                                # cell by cell


# ------------------------------------------------------------------------------------------- by hand

def sim(n=160, tracks=2, sigma=0.3, seed=5, crs=None):
    """Brownian tracks 10 minutes apart (steps of a few metres), in plain coordinates or a CRS."""
    tr = tf.move.simulate_brownian(np.arange(n) * 600.0, sigma=sigma, tracks=tracks, seed=seed)
    return tr.set_crs(crs) if crs else tr


def test_variance_columns_and_their_shape():
    tr = sim(120, 2)
    v = tf.move.dbbmm_variance(tr, 5.0, 31, 11)
    assert len(v) == len(tr) and v.crs == tr.crs
    assert {"motion_variance", "in_windows", "interest", "location_error"} <= set(v.columns)
    for _, g in v.groupby("track"):
        mv = g["motion_variance"].to_numpy()
        assert np.isnan(mv[:10]).all() and np.isnan(mv[-11:]).all() and not np.isnan(mv[10:-11]).any()   # margin - 1, margin
        inw = g["in_windows"].to_numpy()
        assert np.nanmax(inw) == 10 == 31 - 2 * 11 + 1
        np.testing.assert_array_equal(g["interest"].to_numpy(), inw == 10)
    info = v.attrs["terraflux.dbbmm"]
    assert (info["window_size"], info["margin"]) == (31, 11) and set(info["breaks"]) == {"1", "2"}
    assert (v["location_error"] == 5.0).all()
    np.testing.assert_array_equal(tf.move.motion_variance(v), v["motion_variance"])
    assert "motion_variance" not in tr.columns                          # the tracks given are not changed
    with pytest.raises(ValueError, match="run dbbmm_variance"):
        tf.move.motion_variance(tr)


def test_the_variance_of_a_faster_stretch_is_larger():
    n = 200
    t = np.arange(n) * 600.0
    rng = np.random.default_rng(3)
    steps = rng.normal(0, 1, (n - 1, 2)) * np.r_[np.full(100, 2.0), np.full(99, 40.0)][:, None]
    xy = np.vstack([[0, 0], np.cumsum(steps, axis=0)])
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": pd.to_datetime(t, unit="s", utc=True), "track": "a"},
                                            geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1])), "time", "track")
    v = tf.move.dbbmm_variance(tr, 0.0, 31, 11)["motion_variance"].to_numpy()
    assert np.nanmean(v[10:80]) * 20 < np.nanmean(v[120:188])
    assert len(tf.move.dbbmm_variance(tr, 0.0, 31, 11).attrs["terraflux.dbbmm"]["breaks"]["a"]) > 0


def test_short_tracks_are_skipped_or_refused():
    long, short = sim(80, 1), sim(20, 1)
    both = pd.concat([long, short.assign(track=2)], ignore_index=True)
    both = tf.move.as_tracks(both, "time", "track", sort=False)
    with pytest.warns(UserWarning, match="skipped.*2"):
        v = tf.move.dbbmm_variance(both, 5.0, 31, 11)
    assert v["motion_variance"].iloc[80:].isna().all() and v["location_error"].iloc[80:].isna().all()
    assert not v["interest"].iloc[80:].any() and v["motion_variance"].iloc[:80].notna().sum() == 80 - 21
    with pytest.warns(UserWarning, match="skipped"):
        ud = tf.move.dbbmm_ud(both, 5.0, dim_size=30)
    assert list(ud["band"].values) == ["1"]
    with pytest.raises(ValueError, match="cannot be larger"):
        tf.move.dbbmm_variance(short, 5.0, 31, 11)
    with pytest.raises(ValueError, match="No track"):
        tf.move.dbbmm_variance(pd.concat([short, short.assign(track=2)]), 5.0, 31, 11)


@pytest.mark.parametrize("ws, margin", [(30, 11), (31, 10), (31, 0), (21, 11), (31, -1)])
def test_the_window_must_be_odd_and_hold_the_margins(ws, margin):
    with pytest.raises(ValueError, match="odd|at least"):
        tf.move.dbbmm_variance(sim(60, 1), 5.0, ws, margin)
    with pytest.raises(TypeError):
        tf.move.dbbmm_variance(sim(60, 1), 5.0, 31.0, 11)


def test_the_tracks_must_be_in_order_and_whole():
    tr = sim(60, 1)
    with pytest.raises(ValueError, match="times|consecutive"):
        tf.move.dbbmm_variance(tr.iloc[::-1], 5.0, 31, 11)
    with pytest.raises(ValueError, match="share their time"):
        tf.move.dbbmm_variance(pd.concat([tr, tr.iloc[[10]]]).sort_values("time"), 5.0, 31, 11)
    with pytest.raises(ValueError, match="empty"):
        tf.move.dbbmm_variance(tr.assign(geometry=[Point()] + list(tr.geometry[1:])), 5.0, 31, 11)


def test_location_error_forms():
    tr = sim(100, 1)
    z = tf.move.dbbmm_variance(tr, None)
    assert (z["location_error"] == 0).all()
    assert (tf.move.dbbmm_variance(tr, 7)["location_error"] == 7).all()
    col = np.linspace(2, 9, 100)
    for name in ("le_col", "le-col"):                       # a column by its name, with underscores or dashes
        out = tf.move.dbbmm_variance(tr.assign(**{name: col}), "le_col")
        np.testing.assert_array_equal(out["location_error"], col)
    out = tf.move.dbbmm_variance(tr.assign(**{"eobs:horizontal-accuracy-estimate": col}), "auto")   # Movebank's own
    np.testing.assert_array_equal(out["location_error"], col)
    np.testing.assert_array_equal(tf.move.dbbmm_variance(tr, col)["location_error"], col)
    for bad in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="scalar"):
            tf.move.dbbmm_variance(tr, bad)
    with pytest.raises(ValueError, match="one value per location"):
        tf.move.dbbmm_variance(tr, col[:50])
    with pytest.raises(KeyError, match="names a column"):
        tf.move.dbbmm_variance(tr, "nothing")
    with pytest.warns(UserWarning, match="no recognised"):
        assert (tf.move.dbbmm_variance(tr, "auto")["location_error"] == 0).all()
    with pytest.raises(TypeError):
        tf.move.dbbmm_variance(tr, True)


def test_argos_classes_and_the_missing_errors():
    tr = sim(40, 1, seed=2)
    lc = np.array(["3", "2", "1", "0", "A", "B", "Z", "z", None, "9"] * 4, dtype=object)
    out = tf.move.dbbmm_variance(tr.assign(argos_lc=lc), "auto", 15, 5)["location_error"].to_numpy()
    known = np.array([250, 500, 1500, 5000, 5000, 10000, np.nan, np.nan, np.nan, np.nan])
    valid = np.tile(known, 4)
    fill = np.median(valid[~np.isnan(valid)])
    np.testing.assert_array_equal(out, np.where(np.isnan(valid), fill, valid))
    # the numbers of a numeric class column are classes too
    num = tr.assign(argos_lc=np.tile([3.0, 2.0, 1.0, 0.0], 10))
    np.testing.assert_array_equal(tf.move.dbbmm_variance(num, "auto", 15, 5)["location_error"],
                                  np.tile([250.0, 500.0, 1500.0, 5000.0], 10))


@pytest.mark.parametrize("how", ["median", "mean", "zero", "approx"])
def test_the_missing_errors_are_filled_as_r_does(how):
    e = np.array([np.nan, 4.0, np.nan, np.nan, 7.0, 1.0, np.nan, 9.5, 2.0, np.nan])
    from terraflux import _core

    out, status = _core.move_impute_location_error(e, ["median", "mean", "zero", "approx"].index(how))
    ok = ~np.isnan(e)
    want = {"median": np.where(ok, e, np.median(e[ok])), "mean": np.where(ok, e, e[ok].mean()),
            "zero": np.where(ok, e, 0.0), "approx": np.interp(np.arange(10), np.flatnonzero(ok), e[ok])}[how]
    assert status == 0
    np.testing.assert_allclose(out, want, rtol=1e-15)
    if how == "median":                                       # an even count: the mean of the two in the middle
        e4 = np.array([np.nan, 4.0, 1.0, np.nan, 7.0, 2.0])
        assert _core.move_impute_location_error(e4, 0)[0][0] == (2.0 + 4.0) / 2
    assert np.isnan(e[0])                                     # the argument is not changed


def test_the_missing_errors_that_cannot_be_filled():
    from terraflux import _core

    assert _core.move_impute_location_error(np.full(5, np.nan), 0)[1] == 1
    assert _core.move_impute_location_error(np.full(5, np.nan), 2)[0].tolist() == [0.0] * 5
    assert _core.move_impute_location_error(np.array([np.nan, 3.0, np.nan]), 3)[1] == 2
    tr = sim(60, 1)
    e = np.full(60, np.nan)
    with pytest.raises(ValueError, match="entirely missing"):
        tf.move.dbbmm_variance(tr.assign(le=e), "le")
    assert (tf.move.dbbmm_variance(tr.assign(le=e), "le", location_error_na="zero")["location_error"] == 0).all()
    e[5] = 4.0
    with pytest.raises(ValueError, match="at least two"):
        tf.move.dbbmm_variance(tr.assign(le=e), "le", location_error_na="approx")
    with pytest.raises(ValueError, match="location_error_na"):
        tf.move.dbbmm_variance(tr, 1.0, location_error_na="max")


def test_negative_and_infinite_errors_are_missing_ones():
    tr = sim(60, 1)
    e = np.full(60, 6.0)
    e[[3, 30]] = [-2.0, np.inf]
    out = tf.move.dbbmm_variance(tr.assign(le=e), "le")["location_error"].to_numpy()
    assert (out == 6.0).all()


# ----------------------------------------------------------------------------------------- the window

def test_suggest_dbbmm_window_by_hand():
    tr = sim(400, 1)                                           # 10 minutes apart
    s = tf.move.suggest_dbbmm_window(tr, 4)
    assert s == tf.move.WindowSuggestion(25, 7, 600.0, 4.0, 25 * 600.0 / 3600, 400, 16)
    assert tf.move.suggest_dbbmm_window(tr, 0.5).window_size == 11              # the floor
    assert tf.move.suggest_dbbmm_window(tr, 1000).window_size == 99             # a quarter of the track, odd
    big = tf.move.suggest_dbbmm_window(tr, 1000)
    assert big.margin == 25 and big.window_size >= 2 * big.margin + 1
    many = tf.move.suggest_dbbmm_window(sim(400, 3), 4)
    assert list(many) == ["1", "2", "3"] or list(many) == [1, 2, 3]
    with pytest.warns(UserWarning, match="Too few locations"):
        assert tf.move.suggest_dbbmm_window(sim(10, 1)) is None
    with pytest.raises(ValueError, match="target_hours"):
        tf.move.suggest_dbbmm_window(tr, 0)


# ----------------------------------------------------------------------------------------- masking

def test_mask_segments_by_hand():
    tr = sim(120, 2)
    v = tf.move.dbbmm_variance(tr, 5.0, 31, 11)
    n0 = int(v["interest"].sum())
    a = tf.move.mask_segments(v, [30, 31, 40])                                  # every track, from 0
    assert int(a["interest"].sum()) == n0 - 6 and int(v["interest"].sum()) == n0
    for _, g in a.groupby("track"):
        assert not g["interest"].to_numpy()[[30, 31, 40]].any()
    b = tf.move.mask_segments(v, {"2": [50]})
    assert int(b["interest"].sum()) == n0 - 1 and not b["interest"].to_numpy()[120 + 50]
    flags = np.zeros(len(v), bool)
    flags[[35, 36]] = True
    c = tf.move.mask_segments(v, flags)
    assert int(c["interest"].sum()) == n0 - 2
    d = tf.move.mask_segments(v, {"1": np.arange(119) == 33})                   # one flag per segment
    assert not d["interest"].to_numpy()[33] and int(d["interest"].sum()) == n0 - 1
    e = tf.move.mask_segments(v, [np.nan, 40.0])
    assert int(e["interest"].sum()) == n0 - 2
    with pytest.raises(ValueError, match="0..118"):
        tf.move.mask_segments(v, [119])
    with pytest.raises(ValueError, match="0..118"):
        tf.move.mask_segments(v, [-1])
    with pytest.raises(KeyError, match="no track"):
        tf.move.mask_segments(v, {"9": [1]})
    with pytest.raises(ValueError, match="length"):
        tf.move.mask_segments(v, {"1": np.zeros(50, bool)})
    with pytest.raises(ValueError, match="run dbbmm_variance"):
        tf.move.mask_segments(tr, [1])


def test_masked_segments_leave_the_distribution_out():
    tr = sim(100, 1)
    v = tf.move.dbbmm_variance(tr, 5.0, 31, 11)
    full = tf.move.dbbmm_ud(v, dim_size=60)
    part = tf.move.dbbmm_ud(tf.move.mask_segments(v, np.arange(0, 60)), like=full)
    assert float(part.sum()) == pytest.approx(1.0) and not np.array_equal(full.values, part.values)
    with pytest.raises(RuntimeError, match="zero or non-finite"):
        tf.move.dbbmm_ud(tf.move.mask_segments(v, np.arange(0, 99)), like=full)


# ----------------------------------------------------------------------------------------- the grid

def test_grid_arguments():
    tr = sim(100, 1)
    v = tf.move.dbbmm_variance(tr, 5.0, 31, 11)
    like = tf.move.dbbmm_ud(v, dim_size=40)
    for kw, msg in ((dict(res=5, dim_size=40), "only one"), (dict(res=0), "res"), (dict(res=-3), "res"),
                    (dict(dim_size=1), "dim_size"), (dict(dim_size=10.5), "dim_size"),
                    (dict(ext_factor=-1), "ext_factor"), (dict(ext_factor=np.nan), "ext_factor"),
                    (dict(like=like, dim_size=40), "only one"), (dict(time_step=0), "time_step"),
                    (dict(time_step=-1), "time_step"), (dict(crs="geographic"), "crs")):
        with pytest.raises(ValueError, match=msg):
            tf.move.dbbmm_ud(v, **kw)
    with pytest.raises(ValueError, match="max_cells"):
        tf.move.dbbmm_ud(v, dim_size=200, max_cells=1000)
    with pytest.raises(ValueError, match="max_cells"):
        tf.move.dbbmm_ud(v, res=0.01, max_cells=1000)
    with pytest.warns(DeprecationWarning, match="ext_factor"):
        old = tf.move.dbbmm_ud(v, dim_size=30, ext=0.7)
    assert old.attrs["extent"] == tf.move.dbbmm_ud(v, dim_size=30, ext_factor=0.7).attrs["extent"]


def test_the_default_grid_has_100_cells_along_the_longest_side():
    ud = tf.move.dbbmm_ud(sim(100, 1), 5.0)
    assert max(ud.shape[1:]) in (100, 101)
    assert float(ud.sum()) == pytest.approx(1.0)


def test_a_template_is_used_as_it_is_and_never_enlarged():
    tr = sim(100, 1, crs="EPSG:32723")
    v = tf.move.dbbmm_variance(tr, 5.0, 31, 11)
    ud = tf.move.dbbmm_ud(v, res=2)
    again = tf.move.dbbmm_ud(v, like=ud)
    np.testing.assert_array_equal(again.values, ud.values)
    np.testing.assert_allclose(again.attrs["extent"], ud.attrs["extent"], rtol=1e-14)     # rebuilt from the transform
    np.testing.assert_array_equal(again["x"], ud["x"])
    assert again.rio.crs == ud.rio.crs and again.rio.crs.to_epsg() == 32723
    small = ud.isel(x=slice(10, -10), y=slice(10, -10))
    with pytest.raises(ValueError, match="too small.*(left|right|top|bottom)"):
        tf.move.dbbmm_ud(v, like=small)
    with pytest.raises(ValueError, match="at least two"):
        tf.move.dbbmm_ud(v, like=ud.isel(x=slice(0, 1)))
    other = ud.copy()
    other.rio.write_crs("EPSG:32722", inplace=True)
    with pytest.raises(ValueError, match="CRS the variance was computed in"):
        tf.move.dbbmm_ud(v, like=other)
    geographic = ud.copy()
    geographic.rio.write_crs("EPSG:4326", inplace=True)
    with pytest.raises(ValueError, match="projected in metres"):
        tf.move.dbbmm_ud(tr, 5.0, like=geographic)
    # tracks without the variance are taken into the CRS of the template
    elsewhere = tr.to_crs("EPSG:32722")
    moved = tf.move.dbbmm_ud(elsewhere, 5.0, like=ud)
    assert moved.rio.crs.to_epsg() == 32723 and float(moved.sum()) == pytest.approx(1.0)


def test_the_grid_grows_when_the_bridges_do_not_fit():
    tr = sim(100, 1, sigma=0.6)
    box = (tr.geometry.x.min(), tr.geometry.x.max(), tr.geometry.y.min(), tr.geometry.y.max())
    for kw in (dict(res=50), dict(dim_size=50)):
        with pytest.warns(UserWarning, match="enlarged"):
            ud = tf.move.dbbmm_ud(tr, 5.0, ext_factor=0, verbose=True, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                                  # as in R, nothing is said by default
            again = tf.move.dbbmm_ud(tr, 5.0, ext_factor=0, **kw)
        np.testing.assert_array_equal(again.values, ud.values)
        xmin, xmax, ymin, ymax = ud.attrs["extent"]
        assert xmin < box[0] and xmax > box[1] and ymin < box[2] and ymax > box[3]          # beyond the track's box
        assert float(ud.sum()) == pytest.approx(1.0)
        if "res" in kw:
            assert (xmax - xmin) / ud.shape[2] == pytest.approx(50.0, rel=1e-12)             # the cells keep their size
        else:
            assert max(ud.shape[1:]) in (50, 51)                                            # and with dim_size their number


# ------------------------------------------------------------------------------------ the distribution

def test_ud_with_variance_columns_and_without_are_the_same():
    tr = sim(120, 2)
    one = tf.move.dbbmm_ud(tr, 5.0, res=15)
    two = tf.move.dbbmm_ud(tf.move.dbbmm_variance(tr, 5.0), res=15)
    np.testing.assert_array_equal(one.values, two.values)
    assert one.attrs["extent"] == two.attrs["extent"] and list(one["band"]) == ["1", "2"]


def test_location_error_given_to_a_distribution_with_a_variance():
    tr = sim(100, 1)
    v = tf.move.dbbmm_variance(tr, 5.0)
    base = tf.move.dbbmm_ud(v, res=15)
    np.testing.assert_array_equal(tf.move.dbbmm_ud(v, 5.0, res=15).values, base.values)
    np.testing.assert_array_equal(tf.move.dbbmm_ud(v, np.full(100, 5.0), res=15).values, base.values)
    np.testing.assert_array_equal(tf.move.dbbmm_ud(v, {"1": 5.0}, res=15).values, base.values)
    other = tf.move.dbbmm_ud(v, 25.0, res=15)
    assert not np.array_equal(other.values, base.values)
    with pytest.raises(ValueError, match="missing"):
        tf.move.dbbmm_ud(v, np.r_[np.nan, np.full(99, 5.0)], res=15)
    with pytest.raises(ValueError, match="non-negative"):
        tf.move.dbbmm_ud(v, -1.0, res=15)
    with pytest.raises(ValueError, match="one value"):
        tf.move.dbbmm_ud(v, np.ones(7), res=15)
    with pytest.raises(ValueError, match="no location_error column"):
        tf.move.dbbmm_ud(v.drop(columns="location_error"), res=15)


def test_bursts_make_a_band_each():
    n = 90
    t = np.r_[np.arange(40), 60 + np.arange(50)] * 600.0                   # a gap of 20 steps
    tr = tf.move.simulate_brownian(t, sigma=0.3, tracks=1, seed=4)
    b = tf.move.resample(tr, rate="10 min", tolerance="1 min")
    assert set(b["burst_"]) == {1, 2} and len(b) == n
    ud = tf.move.dbbmm_ud(b, 3.0, window_size=15, margin=5, res=20)
    assert list(ud["band"].values) == ["1__1", "1__2"] and list(ud["burst"].values) == [1, 2]
    one = tf.move.dbbmm_ud(b, 3.0, window_size=15, margin=5, res=20, burst=None)
    assert list(one["band"].values) == ["1"] and one.shape[0] == 1
    v = tf.move.dbbmm_variance(b, 3.0, 15, 5)
    assert v["motion_variance"].notna().sum() == (40 - 9) + (50 - 9)       # n - 2 * margin + 1 of each burst
    with pytest.raises(KeyError, match="burst column"):
        tf.move.dbbmm_ud(b, 3.0, burst="nothing")


def test_one_and_four_threads_give_the_same():
    tr = sim(1500, 2, sigma=0.5)
    a = tf.move.dbbmm_variance(tr, 5.0, n_threads=1)
    b = tf.move.dbbmm_variance(tr, 5.0, n_threads=4)
    pd.testing.assert_frame_equal(pd.DataFrame(a.drop(columns="geometry")), pd.DataFrame(b.drop(columns="geometry")))
    assert a.attrs["terraflux.dbbmm"]["breaks"] == b.attrs["terraflux.dbbmm"]["breaks"]
    assert a["interest"].sum() > 2000
    one = tf.move.dbbmm_ud(a, dim_size=300, n_threads=1)
    four = tf.move.dbbmm_ud(a, dim_size=300, n_threads=4)
    np.testing.assert_array_equal(one.values, four.values)
    np.testing.assert_array_equal(tf.move.ud_volume(one).values, tf.move.ud_volume(four).values)


def test_the_distribution_in_the_crs_of_the_tracks():
    lon = tf.move.simulate_brownian(np.arange(100) * 600.0, sigma=0.01, tracks=1, start=(-47.9, -15.8), seed=8)
    lon = lon.set_crs("EPSG:4326")
    proj = tf.move.dbbmm_ud(lon, 30.0, dim_size=80)
    assert proj.rio.crs.is_projected and "azimuthal equidistant" in proj.attrs["note"]
    back = tf.move.dbbmm_ud(lon, 30.0, dim_size=80, crs="input")
    assert back.rio.crs.is_geographic and float(back.sum()) == pytest.approx(1.0)
    assert (back.values >= 0).all() and np.isfinite(back.values).all()
    # the centre of mass is the same place
    from pyproj import Transformer

    def centre(da):
        w = da.values[0]
        xx, yy = np.meshgrid(da["x"].values, da["y"].values)
        return (w * xx).sum(), (w * yy).sum()

    cx, cy = Transformer.from_crs(proj.rio.crs, "EPSG:4326", always_xy=True).transform(*centre(proj))
    bx, by = centre(back)
    assert abs(cx - bx) < 3 * abs(back["x"].values[1] - back["x"].values[0])
    assert abs(cy - by) < 3 * abs(back["y"].values[1] - back["y"].values[0])
    plain = sim(100, 1)
    same_crs = tf.move.dbbmm_ud(plain, 5.0, dim_size=30, crs="input")                 # nothing to warp
    assert "azimuthal" in same_crs.attrs["note"]


def test_volume_and_outer_probability_by_hand():
    tr = sim(100, 1)
    ud = tf.move.dbbmm_ud(tr, 5.0, dim_size=40)
    vol = tf.move.ud_volume(ud)
    assert vol.shape == ud.shape and float(vol.max()) == pytest.approx(1.0, abs=1e-12)
    v, u = vol.values[0].ravel(), ud.values[0].ravel()
    order = np.argsort(-u, kind="stable")
    np.testing.assert_allclose(v[order], np.cumsum(u[order]) / u.sum(), rtol=1e-12)
    two = tf.move.dbbmm_ud(sim(100, 2), 5.0, dim_size=40)
    assert tf.move.ud_volume(two).shape == two.shape
    with pytest.raises(ValueError, match="several bands"):
        tf.move.ud_outer_probability(np.zeros((1, 2)), two)
    xy = np.c_[tr.geometry.x, tr.geometry.y]
    p1 = tf.move.ud_outer_probability(xy, two, band="1")
    p2 = tf.move.ud_outer_probability(xy, two, band=0)
    np.testing.assert_array_equal(p1, p2)
    assert (p1 >= 0).all() and (p1 <= 1 + 1e-12).all()
    far = tf.move.ud_outer_probability(np.array([[1e9, 1e9], [np.nan, 0.0]]), ud)
    assert np.isnan(far).all()
    with pytest.raises(ValueError, match="two columns"):
        tf.move.ud_outer_probability(np.zeros((3, 1)), ud)
    # points in another CRS are taken to the raster's
    geo = tf.move.simulate_brownian(np.arange(100) * 600.0, sigma=0.01, tracks=1, start=(-47.9, -15.8), seed=8)
    geo = geo.set_crs("EPSG:4326")
    d = tf.move.dbbmm_ud(geo, 30.0, dim_size=40)
    q = tf.move.ud_outer_probability(geo, d)
    assert np.isfinite(q).all() and (q <= 1 + 1e-12).all()
    np.testing.assert_allclose(q, tf.move.ud_outer_probability(geo.to_crs(d.rio.crs), d), rtol=0, atol=0)
