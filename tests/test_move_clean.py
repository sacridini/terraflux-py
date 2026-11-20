# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's cleaning and thinning against R's move2utils 0.4.6 (tests/move_cases.py, group "move2utils";
tools/move2utils_fixed.R; tests/data/move/m2u_synthetic.csv.gz is move2utils' synthetic_tracks.csv.gz, GPL-3.0-or-later).

Everything is compared bit for bit with the references. The steps of the speed cap on a sphere (haversine, R's
sin, cos and asin) are R's bits where the references were made, on an Intel CPU (the x87 unit, as in test_move.py:
EXACT); elsewhere they agree to a few units in the last place.
"""

import datetime
import platform
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Point

import terraflux as tf
from conftest import BITWISE, UCRT_LIBM
from move_cases import (SPEED, SPEED_MASS, SPEED_MODE, THIN_DISTANCE, THIN_TIME)

REF = Path(__file__).parent / "data" / "reference" / "move"
DATA = Path(__file__).parent / "data" / "move"
EXACT = UCRT_LIBM and BITWISE and "GenuineIntel" in platform.processor()


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "move2utils.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs_move2utils.npz"))


@pytest.fixture(scope="module")
def fishers():
    return tf.load_tracks(DATA / "fishers.csv.gz", sort=False)


@pytest.fixture(scope="module")
def synth():
    return tf.load_tracks(DATA / "m2u_synthetic.csv.gz", sort=False)


def frame(inputs, name):
    """The tracks of a set stored with the references (a missing time is NaT, a missing coordinate an empty point)."""
    x, y, t = inputs[f"{name}/x"], inputs[f"{name}/y"], inputs[f"{name}/t"]
    crs = str(inputs[f"{name}/crs"]) or None
    geom = [Point() if (np.isnan(a) or np.isnan(b)) else Point(a, b) for a, b in zip(x, y)]
    time = pd.Series(pd.to_datetime((np.nan_to_num(t) * 1e9).astype(np.int64), utc=True)).where(~np.isnan(t))
    df = {"time": time, "track": inputs[f"{name}/track"]}
    for key in ("gps_satellite_count", "gps_dop", "eobs_horizontal_accuracy_estimate", "gps-satellite-count",
                "gps-dop", "eobs-horizontal-accuracy-estimate"):
        if f"{name}/{key}" in inputs:
            df[key] = inputs[f"{name}/{key}"]
    return tf.move.as_tracks(gpd.GeoDataFrame(df, geometry=geom, crs=crs), "time", "track", sort=False)


@pytest.fixture(scope="module")
def tracks(inputs, fishers, synth):
    cache = {"fishers": fishers, "m2u_synthetic": synth}

    def get(name):
        if name not in cache:
            cache[name] = frame(inputs, name)
        return cache[name]

    return get


def same(got, want, err=""):
    np.testing.assert_array_equal(np.asarray(got, float), np.asarray(want, float), err_msg=err)


def same_libm(got, want, lonlat, err=""):
    """Bit for bit, but on a sphere (R's sin, cos, asin) only where the references were made (EXACT)."""
    if EXACT or not lonlat:
        same(got, want, err)
    else:
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), rtol=1e-12, atol=0,
                                   equal_nan=True, err_msg=err)


def as_float(flags):
    return pd.Series(flags).astype("Float64").to_numpy(np.float64, na_value=np.nan)


# ------------------------------------------------------------------------------------------ v_phys_estimate

def test_v_phys_estimate_is_move2utils(ref):
    masses = [1e-9, 3e-8, 0.001, 0.02, 0.5, 1, 2, 5, 6, 70, 250, 1000, 25000, 1.084e5, 2e5]
    got = {"v": [], "lo": [], "hi": [], "kmh": []}
    for lv in (0.95, 0.9, 0.99):
        for mode in ("flying", "running", "swimming"):
            with pytest.warns(UserWarning, match="extrapolation"):
                r = tf.move.v_phys_estimate(masses, mode, ci_level=lv)
            one = tf.move.v_phys_estimate(5, mode, ci_level=lv)    # a number gives numbers, the array the same
            assert isinstance(one.v_max, float) and r.mode == mode and r.ci_level == lv
            assert (one.v_max, one.lower, one.upper, one.kmh) == (r.v_max[7], r.lower[7], r.upper[7], r.kmh[7])
            got["v"] += list(r.v_max)
            got["lo"] += list(r.lower)
            got["hi"] += list(r.upper)
            got["kmh"] += list(r.kmh)
    for key, name in (("v", "v"), ("lo", "lo"), ("hi", "hi"), ("kmh", "kmh")):
        if BITWISE:
            same(got[key], ref[f"v_phys/{name}"], key)
        else:  # the x87 powers, exp and log of R: the system's differ in the last bits
            np.testing.assert_allclose(got[key], ref[f"v_phys/{name}"], rtol=1e-13, atol=0, err_msg=key)


def test_v_phys_estimate_values_and_checks():
    v = tf.move.v_phys_estimate(5, "flying")
    assert (round(v.v_max, 2), round(v.kmh, 2), round(v.lower, 2), round(v.upper, 2)) == (30.89, 111.19, 0.9, 60.88)
    assert v.reference.startswith("Hirt et al. (2017)") and v.mass == 5.0
    assert tf.move.v_phys_estimate(5).mode == "flying"            # R's default
    assert tf.move.v_phys_estimate(5, "run").mode == "running"    # match.arg
    assert tf.move.v_phys_estimate(5, "running").v_max < tf.move.v_phys_estimate(5, "flying").v_max
    for bad in (0, -1, np.nan, [1.0, -2.0]):
        with pytest.raises(ValueError, match="mass"):
            tf.move.v_phys_estimate(bad)
    for bad in ("walking", "x", "", None):
        with pytest.raises(ValueError, match="mode"):
            tf.move.v_phys_estimate(5, bad)
    for bad in (0, 1, -0.5, 1.5):
        with pytest.raises(ValueError, match="ci_level"):
            tf.move.v_phys_estimate(5, ci_level=bad)
    with pytest.warns(UserWarning, match="extrapolation"):
        tf.move.v_phys_estimate(2e5)
    with pytest.warns(UserWarning, match="extrapolation"):
        tf.move.v_phys_estimate(1e-9)


# ------------------------------------------------------------------------------------------ flag_speed

def speed_args(v_max, jitter):
    kw = {"jitter": jitter}
    if v_max is None:
        kw.update(mass=SPEED_MASS, mode=SPEED_MODE)
    else:
        kw["v_max"] = v_max
    return kw


@pytest.mark.parametrize("case", sorted(SPEED))
def test_flag_speed_is_move2utils(ref, tracks, case):
    data, v_max, jitter, _ = SPEED[case]
    tr = tracks(data)
    if data == "fishers":
        tr = tr[~tr.geometry.is_empty]
    out = tf.move.flag_speed(tr, **speed_args(v_max, jitter))
    lonlat = data in ("cl_ll", "fishers", "m2u_synthetic")
    same_libm(out["step_speed"], ref[f"{case}/speed"], lonlat, "step_speed")
    same(as_float(out["is_speed_above_cap"]), ref[f"{case}/above"], "is_speed_above_cap")
    if jitter is not None:
        same(as_float(out["is_step_below_jitter"]), ref[f"{case}/below"], "is_step_below_jitter")
    else:
        assert "is_step_below_jitter" not in out.columns and f"{case}/below" not in ref
    same(out["is_outlier"], ref[f"{case}/outlier"], "is_outlier")
    same(out["flagged_by_speed"], ref[f"{case}/flagged"], "flagged_by_speed")
    same_libm(out["loglr_speed"], ref[f"{case}/loglr"], lonlat, "loglr_speed")
    assert out.attrs["v_max_used"] == ref[f"{case}/vmax"][0] and out.attrs["threshold_type"] == "hard"
    assert len(out) == len(tr) and out.index.equals(tr.index)
    cols = ["is_outlier", "step_speed", "is_speed_above_cap"] + (["is_step_below_jitter"] if jitter is not None else []) \
        + ["flagged_by_speed", "loglr_speed"]
    assert list(out.columns[len(tr.columns):]) == cols             # the columns of move2utils, in its order
    for c in tr.columns:                                           # the input is untouched
        assert c in out.columns
    assert "is_outlier" not in tr.columns
    gone = tf.move.flag_speed(tr, remove=True, **speed_args(v_max, jitter))
    assert len(gone) == int((~ref[f"{case}/outlier"].astype(bool)).sum())
    assert not gone["is_outlier"].any()


def test_flag_speed_the_limit_of_the_mass(tracks):
    tr = tracks("cl_utm")
    v = tf.move.v_phys_estimate(SPEED_MASS, SPEED_MODE).v_max
    by_mass = tf.move.flag_speed(tr, mass=SPEED_MASS, mode=SPEED_MODE)
    by_limit = tf.move.flag_speed(tr, v_max=v)
    assert by_mass.attrs["v_max_used"] == v                         # no margin added
    pd.testing.assert_frame_equal(pd.DataFrame(by_mass.drop(columns="geometry")),
                                  pd.DataFrame(by_limit.drop(columns="geometry")))
    with pytest.raises(ValueError, match="not both"):
        tf.move.flag_speed(tr, v_max=3, mass=2, mode="running")
    with pytest.raises(ValueError, match="mass"):
        tf.move.flag_speed(tr, mass=2)
    with pytest.raises(ValueError, match="v_max"):
        tf.move.flag_speed(tr)
    for bad in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="v_max"):
            tf.move.flag_speed(tr, v_max=bad)
    with pytest.raises(ValueError, match="jitter"):
        tf.move.flag_speed(tr, v_max=1, jitter=-1)


def test_flag_speed_refuses_empty_locations(fishers):
    with pytest.raises(ValueError, match="empty"):
        tf.move.flag_speed(fishers, v_max=1)


def test_flag_speed_by_hand():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02", "2024-01-01 00:03",
                        "2024-01-01 00:03", "2024-01-01 00:04"], utc=True)
    pts = [Point(0, 0), Point(30, 0), Point(5000, 0), Point(5060, 0), Point(5061, 0), Point(5062, 0)]
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=pts, crs=32723), "time", "id")
    out = tf.move.flag_speed(tr, v_max=20, jitter=2)
    # speeds: 0.5, 82.8, 1, (zero lag: none), 0.0167 m/s; flagged: the ends of the step of 82.8 m/s, and the jitter
    np.testing.assert_array_equal(out["is_speed_above_cap"].to_numpy(bool), [False, True, True, False, False, False])
    np.testing.assert_array_equal(out["is_step_below_jitter"].to_numpy(bool), [False, False, False, True, True, True])
    assert np.isnan(out["step_speed"].iloc[[3, 5]]).all()          # a zero lag and the last location
    assert out["is_outlier"].tolist() == [False, True, True, True, True, True]
    # loglr: the least flagged speed is the boundary (the one-sided hard cap clamps at 0)
    s = out["step_speed"].to_numpy()
    lr = out["loglr_speed"].to_numpy()
    assert lr[1] == np.log(s[1])                                    # above the least flagged speed (1 m/s)
    assert lr[0] == 0.0 and lr[2] == 0.0 and lr[4] == 0.0 and np.isnan(lr[3]) and np.isnan(lr[5])


def test_flag_speed_locations_without_time_are_left_out():
    t = pd.to_datetime(["2024-01-01 00:00", None, "2024-01-01 00:02", "2024-01-01 00:03"], utc=True)
    pts = [Point(0, 0), Point(10000, 0), Point(60, 0), Point(120, 0)]
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=pts, crs=32723), "time", "id",
                           sort=False)
    out = tf.move.flag_speed(tr, v_max=20)
    # the location without time is skipped: the step is from the first to the third (0.5 m/s)
    assert out["step_speed"].iloc[0] == 0.5 and np.isnan(out["step_speed"].iloc[1])
    assert out["is_speed_above_cap"].isna().tolist() == [False, True, False, False]
    assert out["is_outlier"].tolist() == [False] * 4


def test_flag_speed_the_tracks_one_by_one(tracks):
    tr = tracks("cl_utm")
    both = tf.move.flag_speed(tr, v_max=0.1, jitter=0.5)
    for name in ("p", "q"):
        part = tf.move.flag_speed(tr[tr["track"] == name], v_max=0.1, jitter=0.5)
        a = part.drop(columns="geometry").reset_index(drop=True)
        b = both[both["track"] == name].drop(columns="geometry").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------------------------------ gps_quality

QUALITY = {"default": {}, "nosat": {"sat_min": None}, "strict": {"sat_min": 8, "dop_max": 3, "hacc_max": 40},
           "keepempty": {"drop_empty": False}, "nodop": {"dop_max": None, "hacc_max": 60},
           "none": {"sat_min": None, "dop_max": None, "hacc_max": None, "drop_empty": False}}


@pytest.mark.parametrize("case,data", [("gps_cl_q", "cl_q"), ("gps_cl_qd", "cl_qd"), ("gps_cl_q_fixed", "cl_q"),
                                       ("gps_synth", "m2u_synthetic"), ("gps_fishers", "fishers")])
def test_gps_quality_is_move2utils(ref, tracks, case, data):
    tr = tracks(data)
    for key, kw in QUALITY.items():
        if data == "m2u_synthetic" and key == "strict":
            kw = {"sat_min": 8, "dop_max": 3}
        mask = tf.move.gps_quality(tr, **kw)
        assert mask.dtype == bool and len(mask) == len(tr)
        same(mask, ref[f"{case}/{key}"], f"{case} {key}")
        assert len(tf.move.filter_gps_quality(tr, **kw)) == int(mask.sum())
    assert not ref[f"{case}/default"].all()                         # it filters something


def test_gps_quality_finds_what_move2utils_does_not(ref, tracks):
    """mt_filter_gps_quality does not know Movebank's own names (gps:satellite-count, gps:dop, ...): on the
    synthetic tracks of its own example it keeps every location; tf.move filters what it was meant to."""
    synth = tracks("m2u_synthetic")
    assert ref["gps_synth_orig/default"].all() and ref["gps_synth_orig/strict"].all()
    assert not tf.move.gps_quality(synth).all()
    assert int(tf.move.gps_quality(synth).sum()) == 11487 == int(ref["gps_synth/default"].sum())
    # the tracks of move2utils' fixed function: fishers' own accuracy column is empty, only the empty locations go
    fishers = tracks("fishers")
    assert not fishers["eobs:horizontal-accuracy-estimate"].notna().any()
    assert int(tf.move.gps_quality(fishers).sum()) == int((~fishers.geometry.is_empty).sum())


def test_gps_quality_by_hand():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"], utc=True)
    df = gpd.GeoDataFrame({"time": t, "id": "a", "gps:satellite-count": ["9", "4", "x", "7"],
                           "gps:dop": [1.2, 1.0, 14.0, np.nan]},
                          geometry=[Point(0, 0), Point(1, 1), Point(2, 2), Point()], crs=32723)
    tr = tf.move.as_tracks(df, "time", "id")
    # a number as text counts, text that is not a number and a missing value do not drop a location
    np.testing.assert_array_equal(tf.move.gps_quality(tr), [True, False, False, False])
    np.testing.assert_array_equal(tf.move.gps_quality(tr, drop_empty=False), [True, False, False, True])
    np.testing.assert_array_equal(tf.move.gps_quality(tr, sat_min=None, dop_max=None, drop_empty=False), [True] * 4)
    np.testing.assert_array_equal(tf.move.gps_quality(tr, sat_min=4, dop_max=14, drop_empty=False), [True] * 4)
    np.testing.assert_array_equal(tf.move.gps_quality(tr[["time", "id", "geometry"]]), [True, True, True, False])
    assert len(tf.move.filter_gps_quality(tr)) == 1


# ------------------------------------------------------------------------------------------ thinning

@pytest.mark.parametrize("case", sorted(THIN_TIME))
def test_thin_time_is_move2utils(ref, tracks, case):
    data, interval, tolerance, criterion = THIN_TIME[case]
    tr = tracks(data)
    mask = tf.move.thin_time(tr, interval, tolerance, criterion)
    assert mask.dtype == bool and len(mask) == len(tr)
    same(mask, ref[f"{case}/selected"], case)
    assert len(tf.move.filter_thin_time(tr, interval, tolerance, criterion)) == int(mask.sum())


@pytest.mark.parametrize("case", sorted(THIN_DISTANCE))
def test_thin_distance_is_move2utils(ref, tracks, case):
    data, distance, tolerance, criterion, method = THIN_DISTANCE[case]
    tr = tracks(data)
    if data == "fishers":
        tr = tr[~tr.geometry.is_empty]
    mask = tf.move.thin_distance(tr, distance, method, tolerance if method == "interval" else None, criterion)
    assert mask.dtype == bool and len(mask) == len(tr)
    same(mask, ref[f"{case}/selected"], case)
    assert len(tf.move.filter_thin_distance(tr, distance, method, tolerance if method == "interval" else None,
                                            criterion)) == int(mask.sum())


def test_the_thinned_tracks_keep_their_lags(tracks):
    tr = tracks("m2u_synthetic")
    out = tf.move.filter_thin_time(tr, "45 mins", "5 mins")
    lags = tf.move.time_lags(out, "min")
    lags = lags[~np.isnan(lags)]
    assert (lags >= 40 - 1e-9).all()      # no kept pair is closer than interval - tolerance; the longer ones cross gaps
    assert 0.5 < ((lags <= 50 + 1e-9).mean()) and len(out) < len(tr) / 2


def r_dp_run(ts, d, tol, closest):
    """R's .thin_dp_run, line by line."""
    m = len(ts)
    if m == 1:
        return [True]
    lo, hi = d - tol, d + tol
    f, dev, pred = [1] * m, [0.0] * m, [-1] * m
    for i in range(1, m):
        best_f, best_j, best_dv = 1, -1, np.inf
        j = i - 1
        while j >= 0:
            gap = ts[i] - ts[j]
            if gap > hi:
                break
            if gap >= lo:
                fj, dvj = f[j] + 1, dev[j] + abs(gap - d)
                if fj > best_f or (fj == best_f and closest and dvj < best_dv):
                    best_f, best_j, best_dv = fj, j, dvj
            j -= 1
        f[i], dev[i], pred[i] = best_f, (0.0 if best_j < 0 else best_dv), best_j
    cand = [i for i in range(m) if f[i] == max(f)]
    end = cand[int(np.argmin([dev[i] for i in cand]))] if closest else cand[0]
    sel = [False] * m
    while end >= 0:
        sel[end] = True
        end = pred[end]
    return sel


def r_thin(ts, d, tol, closest):
    """R's .thin_dp_track."""
    n = len(ts)
    if n == 1:
        return [True]
    run = np.r_[0, np.cumsum(np.diff(ts) > d + tol)]
    out = [False] * n
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        for i, s in zip(idx, r_dp_run([ts[i] for i in idx], d, tol, closest)):
            out[i] = s
    return out


@pytest.mark.parametrize("seed", range(6))
def test_thin_time_is_the_literal_dp(seed):
    rng = np.random.default_rng(seed)
    step = rng.choice([1.0, 2.0, 3.0, 5.0, 10.0, 0.5])    # multiples of half a second: exact as nanoseconds
    ts = np.cumsum(rng.choice([0.0, step, 2 * step, 4.0, 9.5, 31.0], 700))
    t0 = pd.Timestamp("2024-03-01", tz="UTC")
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t0 + pd.to_timedelta(ts, unit="s"), "id": "a"},
                                            geometry=[Point(float(i), 0) for i in range(len(ts))], crs=32723),
                           "time", "id", sort=False)
    for d, tol in ((10.0, 2.0), (6.0, 6.0), (20.0, 0.0), (4.0, 1.5), (30.0, 5.0)):
        for criterion in ("closest", "first"):
            want = r_thin(list(ts), d, tol, criterion == "closest")
            np.testing.assert_array_equal(tf.move.thin_time(tr, d, tol, criterion), want, err_msg=f"{d} {tol}")


def test_thin_time_sorts_each_track_by_time():
    rng = np.random.default_rng(3)
    parts = []
    for k in "abcd":
        t = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(np.cumsum(rng.integers(1, 20, 100)) * 60.0, unit="s")
        parts.append(gpd.GeoDataFrame({"time": t, "id": k}, geometry=[Point(float(i), 0) for i in range(100)],
                                      crs=32723))
    ordered = pd.concat(parts, ignore_index=True)
    n = len(ordered)
    sorted_ = tf.move.as_tracks(ordered, "time", "id", sort=False)
    a = tf.move.thin_time(sorted_, "30 min", "4 min")
    assert 0 < a.sum() < n
    # the rows of each track shuffled
    perm = np.concatenate([rng.permutation(np.arange(100)) + 100 * i for i in range(4)])
    shuffled = tf.move.as_tracks(ordered.iloc[perm], "time", "id", sort=False)
    np.testing.assert_array_equal(tf.move.thin_time(shuffled, "30 min", "4 min"), a[perm])
    # the tracks need not be consecutive: their rows interleaved, each track in its order
    inter = np.argsort(np.tile(np.arange(100), 4) + rng.random(n) * 0.5, kind="stable")
    mixed = tf.move.as_tracks(ordered.iloc[inter], "time", "id", sort=False)
    np.testing.assert_array_equal(tf.move.thin_time(mixed, "30 min", "4 min"), a[inter])
    np.testing.assert_array_equal(tf.move.thin_distance(mixed, 5.5), tf.move.thin_distance(sorted_, 5.5)[inter])
    np.testing.assert_array_equal(tf.move.thin_distance(mixed, 5.5, "interval"),
                                  tf.move.thin_distance(sorted_, 5.5, "interval")[inter])


def test_thin_time_by_hand():
    def make(minutes, ids=None):
        t = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(minutes, unit="min")
        return tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": ids if ids is not None else "a"},
                                                  geometry=[Point(float(i), 0) for i in range(len(minutes))],
                                                  crs=32723), "time", "id", sort=False)
    # the lone location and the first of a track with no lag in the window: kept; a gap longer than the window
    # splits the track and is crossed as it was
    got = tf.move.thin_time(make([0, 10, 20, 30, 40, 50, 60, 70, 80]), "30 min", "1 min")
    np.testing.assert_array_equal(got, [1, 0, 0, 1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(tf.move.thin_time(make([0]), "30 min"), [True])
    np.testing.assert_array_equal(tf.move.thin_time(make([0, 5]), "30 min", "1 min"), [True, False])
    np.testing.assert_array_equal(tf.move.thin_time(make([0, 30]), "30 min", "1 min"), [True, True])
    np.testing.assert_array_equal(tf.move.thin_time(make([0, 30, 200, 230]), "30 min", "1 min"), [1, 1, 1, 1])
    # of equal chains the latest earlier location is taken: 60 follows the second 30
    np.testing.assert_array_equal(tf.move.thin_time(make([0, 30, 30, 60]), "30 min", "1 min"), [1, 0, 1, 1])
    assert len(tf.move.thin_time(make([0, 1])[:0], "30 min")) == 0
    # the same text, seconds and time differences
    ref_ = tf.move.thin_time(make(np.arange(0, 600, 7.0)), "45 min", "5 min")
    for interval, tol in ((2700, 300), (pd.Timedelta("45min"), pd.Timedelta(minutes=5)), ("45 mins", "5 min"),
                          (datetime.timedelta(minutes=45), np.timedelta64(5, "m")), ("0.75 hours", "300 secs"),
                          ("45min", "5 minutes")):
        np.testing.assert_array_equal(tf.move.thin_time(make(np.arange(0, 600, 7.0)), interval, tol), ref_)
    # the default tolerance is a tenth of the interval
    np.testing.assert_array_equal(tf.move.thin_time(make(np.arange(0, 600, 7.0)), "45 min"),
                                  tf.move.thin_time(make(np.arange(0, 600, 7.0)), "45 min", "4.5 min"))


def test_thin_time_checks():
    t = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:30"], utc=True)
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=[Point(0, 0), Point(1, 0)], crs=32723),
                           "time", "id")
    for kw in ({"interval": 0}, {"interval": -5}):
        with pytest.raises(ValueError, match="interval"):
            tf.move.thin_time(tr, **kw)
    with pytest.raises(ValueError, match="tolerance"):
        tf.move.thin_time(tr, "10 min", "11 min")
    with pytest.raises(ValueError, match="tolerance"):
        tf.move.thin_time(tr, "10 min", -1)
    with pytest.raises(ValueError, match="criterion"):
        tf.move.thin_time(tr, "10 min", criterion="all")
    with pytest.raises(ValueError, match="duration"):
        tf.move.thin_time(tr, "soon")
    with pytest.raises(TypeError, match="interval"):
        tf.move.thin_time(tr, [1, 2])
    with pytest.raises(ValueError, match="positive"):
        tf.move.thin_distance(tr, 0)
    with pytest.raises(ValueError, match="method"):
        tf.move.thin_distance(tr, 10, "greedy")
    with pytest.raises(ValueError, match="tolerance"):
        tf.move.thin_distance(tr, 10, "interval", 11)
    with pytest.warns(UserWarning, match="ignored"):
        tf.move.thin_distance(tr, 10, "step", 2)
    with pytest.raises(ValueError, match="empty"):
        tf.move.thin_distance(tr.assign(geometry=[Point(0, 0), Point()]), 10)


def test_thin_distance_by_hand():
    t = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(range(6), unit="min")
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=[Point(60.0 * i, 0) for i in range(6)],
                                            crs=32723), "time", "id")
    np.testing.assert_array_equal(tf.move.thin_distance(tr, 100), [1, 0, 1, 0, 1, 0])      # 0, 120, 240
    np.testing.assert_array_equal(tf.move.thin_distance(tr, 60), [1] * 6)
    np.testing.assert_array_equal(tf.move.thin_distance(tr, 1000), [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(tf.move.thin_distance(tr, 120, "interval", 1), [1, 0, 1, 0, 1, 0])
    # in feet the same metres: 60 m = 196.85 ft a step
    ft = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"},
                                            geometry=[Point(60.0 * i / 0.3048006096012192, 0) for i in range(6)],
                                            crs=2263), "time", "id")
    np.testing.assert_array_equal(tf.move.thin_distance(ft, 100), [1, 0, 1, 0, 1, 0])


# ------------------------------------------------------------------------------------------ threads

def big_tracks(n_tracks=8, per=7000, seed=11, lonlat=False):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_tracks):
        step = rng.choice([30.0, 45.0, 60.0, 90.0, 600.0], per)
        t = pd.Timestamp("2024-05-01", tz="UTC") + pd.to_timedelta(np.cumsum(step), unit="s")
        scale = 1e-4 if lonlat else 8.0
        xy = np.cumsum(rng.normal(0, scale, (per, 2)), axis=0) + ([10.0, 50.0] if lonlat else [5e5, 5e6])
        xy[rng.integers(0, per, 20)] += rng.normal(0, scale * 500, (20, 2))
        rows.append(gpd.GeoDataFrame({"time": t, "id": f"t{k}"}, geometry=[Point(a, b) for a, b in xy],
                                     crs=4326 if lonlat else 32632))
    return tf.move.as_tracks(pd.concat(rows, ignore_index=True), "time", "id", sort=False)


@pytest.mark.parametrize("lonlat", [False, True])
def test_one_and_four_threads_give_the_same(lonlat):
    tr = big_tracks(lonlat=lonlat)
    assert len(tr) > 20000                                          # above the size from which the loops run in parallel
    a = tf.move.flag_speed(tr, v_max=1.5, jitter=2.0, n_threads=1)
    b = tf.move.flag_speed(tr, v_max=1.5, jitter=2.0, n_threads=4)
    pd.testing.assert_frame_equal(pd.DataFrame(a.drop(columns="geometry")), pd.DataFrame(b.drop(columns="geometry")))
    assert 0 < a["is_outlier"].sum() < len(tr)
    for kw in ({"interval": "20 min", "tolerance": "3 min"}, {"interval": "5 min", "criterion": "first"}):
        x = tf.move.thin_time(tr, n_threads=1, **kw)
        np.testing.assert_array_equal(x, tf.move.thin_time(tr, n_threads=4, **kw))
        assert 0 < x.sum() < len(tr)
    for kw in ({"distance": 500.0}, {"distance": 500.0, "method": "interval", "tolerance": 100.0}):
        x = tf.move.thin_distance(tr, n_threads=1, **kw)
        np.testing.assert_array_equal(x, tf.move.thin_distance(tr, n_threads=4, **kw))
        assert 0 < x.sum() < len(tr)
