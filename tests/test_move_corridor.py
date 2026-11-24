# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's corridors and distance between UDs against R's move2utils 0.4.6 (tests/move_cases.py, group
"move2utils_corridor"; tools/move2utils_fixed.R).

The corridors are compared bit for bit with the references: the segments, speeds, azimuths, circular variances, neighbours
and the flags, where R's mt_corridor is right (projected in metres) or corrected (tools/move2utils_fixed.R: the projection of
longitude / latitude, the radius in a CRS not in metres). On a sphere (s2's asin, the UCRT's) the bits are R's where the
references were made, on an Intel CPU (EXACT, as in test_move_clean.py); elsewhere they agree to rounding.
The Sinkhorn distance is compared to a relative 1e-12 (R's reference BLAS sums in the order reproduced here: the results
are, in fact, the same bits where the references were made), the exact one bit for bit.
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
from terraflux import _core
from conftest import BITWISE, X87_EXACT
from move_cases import CORRIDOR

REF = Path(__file__).parent / "data" / "reference" / "move"
DATA = Path(__file__).parent / "data" / "move"
EXACT = X87_EXACT


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "move2utils_corridor.npz"))


@pytest.fixture(scope="module")
def inputs():
    out = dict(np.load(REF / "inputs.npz"))
    out.update(np.load(REF / "inputs_move2utils.npz"))
    out.update(np.load(REF / "inputs_corridor.npz"))
    return out


@pytest.fixture(scope="module")
def fishers():
    return tf.load_tracks(DATA / "fishers.csv.gz", sort=False)


@pytest.fixture(scope="module")
def tracks(inputs, fishers):
    cache = {"fishers": fishers}

    def get(name, one=None):
        if name not in cache:
            x, y, t = inputs[f"{name}/x"], inputs[f"{name}/y"], inputs[f"{name}/t"]
            time = pd.to_datetime((t * 1e9).astype(np.int64), utc=True)
            df = gpd.GeoDataFrame({"time": time, "track": inputs[f"{name}/track"]},
                                  geometry=[Point(a, b) for a, b in zip(x, y)], crs=str(inputs[f"{name}/crs"]) or None)
            cache[name] = tf.move.as_tracks(df, "time", "track", sort=False)
        out = cache[name]
        if one is not None:
            out = out[out[tf.move._tracks.columns(out)[1]] == one]
        return out

    return get


def same(got, want, err=""):
    np.testing.assert_array_equal(np.asarray(got, float), np.asarray(want, float), err_msg=err)


def run(tr, args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tf.move.corridor(tr, **args, **kwargs)


def columns_of(out):
    return {"corridor": (out["corridor"] == "corridor").to_numpy().astype(float),
            "speed": out["corridor_speed"].to_numpy(), "azimuth": out["corridor_azimuth"].to_numpy(),
            "circvar": out["corridor_circvar"].to_numpy(),
            "nb": out["corridor_n_neighbours"].astype("float64").to_numpy()}


# ------------------------------------------------------------------------------------------ GEOS' buffer

def test_buffer_vertices_and_membership_are_geos(ref):
    """The 120 vertices sf's st_buffer gives a point and the answer of st_intersects for points on, around and off the
    polygon (its vertices, edges, one ulp off, the lattice, between the polygon and the circle)."""
    cx, cy, r = ref["cor_buffer/cx"], ref["cor_buffer/cy"], ref["cor_buffer/r"]
    vx, vy = ref["cor_buffer/vx"].reshape(-1, 120), ref["cor_buffer/vy"].reshape(-1, 120)
    px, py = ref["cor_buffer/px"].reshape(len(cx), -1), ref["cor_buffer/py"].reshape(len(cx), -1)
    hit = ref["cor_buffer/hit"].reshape(len(cx), -1).astype(bool)
    n_in = 0
    for b in range(len(cx)):
        mx, my = _core.move_buffer_vertices(cx[b], cy[b], r[b])
        if BITWISE:
            same(mx, vx[b], f"buffer {b} x")
            same(my, vy[b], f"buffer {b} y")
        else:
            np.testing.assert_allclose(mx, vx[b], rtol=1e-14, atol=0)
        for k in range(px.shape[1]):
            got = _core.move_ring_intersects(vx[b], vy[b], px[b, k], py[b, k])
            assert got == hit[b, k], (b, k)
        n_in += int(hit[b].sum())
    assert 0.3 < n_in / hit.size < 0.7   # points on both sides


def test_ring_intersects_boundary_counts():
    vx, vy = _core.move_buffer_vertices(10.0, 20.0, 5.0)
    assert _core.move_ring_intersects(vx, vy, 10.0, 20.0)          # the centre
    assert _core.move_ring_intersects(vx, vy, 15.0, 20.0)          # a vertex
    assert not _core.move_ring_intersects(vx, vy, 15.001, 20.0)
    assert not _core.move_ring_intersects(vx, vy, 10.0 + 5.0 * 0.7072, 20.0 + 5.0 * 0.7072)   # outside the circle
    # between the polygon and the circle: the midpoint of an edge is inside the circle only by the sagitta
    mx, my = (vx[0] + vx[1]) / 2, (vy[0] + vy[1]) / 2
    ex, ey = 10.0 + (mx - 10.0) * 1.0002, 20.0 + (my - 20.0) * 1.0002
    assert not _core.move_ring_intersects(vx, vy, ex, ey) and np.hypot(ex - 10.0, ey - 20.0) < 5.0


# ------------------------------------------------------------------------------------------ mt_corridor

@pytest.mark.parametrize("case", list(CORRIDOR))
def test_corridor_is_move2utils(case, ref, tracks):
    data, one, args, _ = CORRIDOR[case]
    lonlat = data in ("fishers", "ll")
    out = run(tracks(data, one), args)
    got = columns_of(out)
    strict = BITWISE and (EXACT or not lonlat)
    # R divides by the lag in the unit its difftime picks (minutes...) and converts the speed to m/s: a few ulps off
    # d / lag in seconds; its POSIXct seconds in a double (the fishers have fractions of a second) move a lag by 1e-7
    # (longitude / latitude without the x87 unit: the geodesics' sines and cosines come from the system library, 1.8e-8
    # of the speed at most on ARM64)
    speed_rtol = 1e-7 if data == "fishers" or (lonlat and not BITWISE) else 1e-14
    for key, value in got.items():
        want = ref[f"{case}/{key}"]
        assert len(value) == len(want)
        if key == "speed":
            np.testing.assert_allclose(value, want, rtol=speed_rtol, atol=0, equal_nan=True, err_msg=f"{case} {key}")
        elif key in ("azimuth", "circvar") and lonlat:
            # the working projection of longitude / latitude is PROJ's aeqd: R's has PROJ 9.4.1's geodesics, pyproj's
            # PROJ 9.5.1's, which put the coordinates about 1e-8 m apart: the directions agree to 1e-7 of themselves
            # (to 1.02e-6 on ARM64, whose sines and cosines are the system's libm's)
            np.testing.assert_allclose(value, want, rtol=1e-6 if BITWISE else 1e-5, atol=1e-8, equal_nan=True,
                                       err_msg=f"{case} {key}")
        elif strict:
            same(value, want, f"{case} {key}")
        elif key in ("azimuth", "circvar"):
            np.testing.assert_allclose(value, want, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=f"{case} {key}")
        else:   # a midpoint at the border of a buffer can go either way in the last bits (NaN is NaN)
            assert np.mean((value == want) | (np.isnan(value) & np.isnan(want))) > 0.999, f"{case} {key}"
    if not args:     # the quantile thresholds R resolved
        used = out.attrs["corridor"]
        want = ref[f"{case}/thr"]
        assert used["speed_threshold"] == pytest.approx(want[0], rel=speed_rtol, abs=0)
        if lonlat:
            assert used["circvar_threshold"] == pytest.approx(want[1], rel=1e-6, abs=1e-9)
        elif strict:
            assert (np.isnan(want[1]) and np.isnan(used["circvar_threshold"])) or used["circvar_threshold"] == want[1]
    assert out.attrs["corridor"]["min_segments"] == args.get("min_segments", 2)


def test_corridor_has_something_of_everything(ref):
    """The cases would pass trivially on all-false flags: the references have corridor segments, non-corridor segments,
    zero-length steps (no neighbours, no variance) and neighbours at the border."""
    for case in ("cor_fishers_default", "cor_m1_min2", "cor_utm_loose", "cor_grid_given", "cor_lattice_given", "cor_zero_given",
                 "cor_ft_loose", "cor_ll_default"):
        flag = ref[f"{case}/corridor"]
        assert 0 < flag.sum() < len(flag), case
    nb = ref["cor_zero_given/nb"]
    assert (nb == 0).any() and (nb > 5).any()                  # zero-length segments have no neighbour at all
    assert np.isnan(ref["cor_zero_given/circvar"][nb == 0]).all()
    assert np.isnan(ref["cor_fishers_default/speed"]).sum() > 14000   # the empty locations and the ends of the tracks
    # more segments are corridors with fewer neighbours required
    assert ref["cor_m1_min1/corridor"].sum() >= ref["cor_m1_min2/corridor"].sum() >= ref["cor_m1_min3/corridor"].sum() \
        >= ref["cor_m1_min4/corridor"].sum()


@pytest.mark.parametrize("case", ["cor_fishers_given", "cor_grid_default", "cor_zero_default", "cor_ll_default"])
def test_corridor_threads_agree(case, tracks):
    data, one, args, _ = CORRIDOR[case]
    tr = tracks(data, one)
    a = columns_of(run(tr, args, n_threads=1))
    b = columns_of(run(tr, args, n_threads=4))
    for key in a:
        same(a[key], b[key], key)


def test_corridor_in_feet_uses_the_radius_in_feet(tracks):
    """move2utils gives st_buffer the radius in metres where the CRS is in feet; here it is in the CRS's units: the
    same segments in feet and in metres have the same neighbours."""
    tr = tracks("cl_ft")
    metres = tr.copy()
    f = 1200 / 3937
    metres["geometry"] = gpd.GeoSeries([Point(g.x * f, g.y * f) for g in tr.geometry], index=tr.index)
    metres = metres.set_crs(None, allow_override=True).set_crs(32723)
    a = run(tr, {"speed_threshold": 0.01, "circvar_threshold": 0.5})
    b = run(metres, {"speed_threshold": 0.01, "circvar_threshold": 0.5})
    np.testing.assert_array_equal(a["corridor_n_neighbours"].astype("float64"), b["corridor_n_neighbours"].astype("float64"))
    np.testing.assert_allclose(a["corridor_speed"], b["corridor_speed"], rtol=1e-9)
    assert a["corridor_n_neighbours"].max() > 1


def test_corridor_on_a_straight_walk_and_its_checks():
    n = 20
    t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    pts = [Point(10.0 * i, 0.3 * (i % 3)) for i in range(n)] + [Point(10.0 * i + 2.5, 0.5) for i in range(n)]
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": list(t) * 2, "id": ["a"] * n + ["b"] * n}, geometry=pts, crs=32723),
                           "time", "id")
    out = tf.move.corridor(tr, speed_threshold=0.1, circvar_threshold=0.1)
    assert out["corridor"].dtype.name == "category" and list(out["corridor"].cat.categories) == ["corridor", "not corridor"]
    assert (out["corridor"] == "corridor").sum() == 38 and out["corridor"].iloc[n - 1] == "not corridor"
    assert pd.isna(out["corridor_speed"].iloc[n - 1]) and out["corridor_n_neighbours"].isna().iloc[-1]
    assert out["corridor_n_neighbours"].dtype.name == "Int64"
    assert "corridor" not in tr.columns                          # a copy
    assert out.attrs["corridor"] == {"speed_threshold": 0.1, "circvar_threshold": 0.1, "min_segments": 2}
    # nothing is fast enough
    assert (tf.move.corridor(tr, speed_threshold=5, circvar_threshold=0.1)["corridor"] == "corridor").sum() == 0
    # the lines of the corridor segments
    lines = tf.move.corridor_lines(out)
    assert len(lines) == 38 and (lines.geom_type == "LineString").all() and lines.crs == tr.crs
    assert lines.geometry.iloc[3].coords[0] == (30.0, 0.0) and lines.geometry.iloc[3].coords[1] == (40.0, 0.3)
    assert {"id", "time", "speed", "azimuth", "circvar", "n_neighbours"} <= set(lines.columns)
    with pytest.raises(ValueError, match="no CRS"):
        tf.move.corridor(tr.set_crs(None, allow_override=True), speed_threshold=0.1, circvar_threshold=0.1)
    with pytest.raises(ValueError, match="at least 2"):
        tf.move.corridor(tr.iloc[:1], speed_threshold=0.1, circvar_threshold=0.1)
    with pytest.raises(ValueError, match="finite"):
        tf.move.corridor(tr, speed_threshold=float("nan"))
    with pytest.raises(TypeError):
        tf.move.corridor(tr, speed_threshold="fast")
    with pytest.raises(ValueError, match="corridor"):
        tf.move.corridor_lines(tr)


def test_corridor_warns_where_r_does_and_leaves_empty_rows_alone():
    n = 30
    t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    pts = [Point(10.0 * i, 0.5 * (i % 2)) if i != 5 else Point() for i in range(n)]
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t, "id": "a"}, geometry=pts, crs=32723), "time", "id")
    with pytest.warns(UserWarning) as w:
        out = tf.move.corridor(tr)
    texts = " ".join(str(x.message) for x in w)
    assert "`speed_threshold` not supplied" in texts and "`circvar_threshold` not supplied" in texts
    assert out["corridor"].iloc[5] == "not corridor" and np.isnan(out["corridor_speed"].iloc[5])
    # the segment before the empty location runs to the one after it
    assert out["corridor_speed"].iloc[4] == pytest.approx(20.0 / 120.0, rel=1e-12)
    # an irregular sampling interval
    t2 = pd.to_datetime(np.cumsum(np.r_[0, np.random.default_rng(1).choice([30, 60, 3600, 20000], n - 1)]) + 1.7e9, unit="s",
                        utc=True)
    irr = tf.move.as_tracks(gpd.GeoDataFrame({"time": t2, "id": "a"}, geometry=[Point(10.0 * i, 0) for i in range(n)],
                                            crs=32723), "time", "id")
    with pytest.warns(UserWarning, match="highly irregular"):
        tf.move.corridor(irr, speed_threshold=0.0, circvar_threshold=1.0)


def test_corridor_with_a_missing_variance_threshold():
    """Fewer than two neighbourhoods with a variance: R's quantile gives NA, and `if (any(NA))` fails where a fast segment
    has a variance; here the error says so."""
    t = pd.date_range("2024-01-01", periods=2, freq="min", tz="UTC")
    # a segment of 20 m (radius 10) whose neighbourhood holds the midpoint of a segment of 4 m (radius 2) 5 m off: only
    # the first has two neighbours
    pts = [Point(0, 0), Point(20, 0), Point(13, 0), Point(17, 0)]
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": list(t) * 2, "id": ["a", "a", "b", "b"]}, geometry=pts, crs=32723),
                           "time", "id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = tf.move.corridor(tr, speed_threshold=100.0)       # no segment is fast: no corridor, no error
        assert (out["corridor"] == "corridor").sum() == 0
        assert list(out["corridor_n_neighbours"]) == [2, pd.NA, 1, pd.NA]
        assert out["corridor_circvar"].notna().sum() == 1
        assert np.isnan(out.attrs["corridor"]["circvar_threshold"])
        with pytest.raises(ValueError, match="circular variance threshold is missing"):
            tf.move.corridor(tr, speed_threshold=0.0)
        # with both given there is a candidate, but one of two neighbours is not a strict majority
        both = tf.move.corridor(tr, speed_threshold=0.0, circvar_threshold=1.0)
        assert (both["corridor"] == "corridor").sum() == 0 and both["corridor_circvar"].notna().sum() == 1


def test_corridor_min_segments_counts_all_neighbours():
    """move2utils compares min_segments with the number of neighbours (its manual says qualifying ones): kept."""
    n = 12
    t = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    a = [Point(10.0 * i, 0) for i in range(n)]
    b = [Point(10.0 * i + 2.5, 0.5) for i in range(n)]           # the midpoints of two tracks 2.55 m apart, 10 m along
    tr = tf.move.as_tracks(gpd.GeoDataFrame({"time": list(t) * 2, "id": ["a"] * n + ["b"] * n}, geometry=a + b, crs=32723),
                           "time", "id")
    got = {m: int((tf.move.corridor(tr, speed_threshold=0.0, circvar_threshold=1.0, min_segments=m)["corridor"]
                   == "corridor").sum()) for m in (0, 1, 2, 3, 4)}
    nb = tf.move.corridor(tr, speed_threshold=0.0, circvar_threshold=1.0)["corridor_n_neighbours"].dropna()
    assert (nb == 2).all()                       # itself and the other track's, whatever min_segments
    assert got[0] == got[1] == got[2] == 2 * (n - 1) and got[3] == got[4] == 0


# ------------------------------------------------------------------------------------------ emd

def uds(ref, case, suffix="", names=None):
    nrow, ncol, nl = (int(v) for v in ref[f"{case}/dim{suffix}"])
    ud = ref[f"{case}/ud{suffix}"].reshape(nl, nrow, ncol)
    return xr.DataArray(ud, coords={"band": list(names or [f"ud{i}" for i in range(nl)]), "y": ref[f"{case}/ys{suffix}"],
                                    "x": ref[f"{case}/xs{suffix}"]}, dims=("band", "y", "x"))


FISHERS = ("F1", "F2", "M1")
# the R calls of the references: key -> arguments of emd(); the distances are of (1, 2), (1, 3), (2, 3)
SINKHORN = {"sink": {}, "mask09": {"mask_quantile": 0.9}, "mask05": {"mask_quantile": 0.5}, "mask1": {"mask_quantile": 1},
            "reg01": {"reg": 0.1}, "reg002": {"reg": 0.02, "max_iter": 2000}, "tol13": {"tol": 1e-13},
            "thr": {"threshold": 1500}, "iter3": {"max_iter": 3}}


@pytest.mark.parametrize("key", list(SINKHORN))
def test_sinkhorn_distance_is_move2utils(key, ref):
    # mask_quantile = 1 keeps every cell: on the coarse grid (the other UDs are at 100 m)
    stack = uds(ref, "emd_fishers", "_c" if key == "mask1" else "", names=FISHERS)
    table = tf.move.ud_distance(stack, **SINKHORN[key])
    assert list(table.columns) == ["a", "b", "distance"]
    assert [(a, b) for a, b in zip(table["a"], table["b"])] == [("F1", "F2"), ("F1", "M1"), ("F2", "M1")]
    np.testing.assert_allclose(table["distance"], ref[f"emd_fishers/{key}"], rtol=1e-12, atol=0, err_msg=key)
    assert (table["distance"] > 0).all()


def test_sinkhorn_distance_on_the_coarse_grid_and_between_two(ref):
    stack = uds(ref, "emd_fishers", "_c", names=FISHERS)
    table = tf.move.ud_distance(stack)
    np.testing.assert_allclose(table["distance"], ref["emd_fishers/sink_c"], rtol=1e-12, atol=0)
    one = tf.move.ud_distance(stack.sel(band="F1"), stack.sel(band="M1"))      # a number for two UDs
    assert isinstance(one, float) and one == table["distance"].iloc[1]
    two = tf.move.ud_distance(stack.sel(band=["F2", "M1"]))
    assert isinstance(two, float) and two == table["distance"].iloc[2]
    assert tf.move.ud_distance({"p": stack.sel(band="F1"), "q": stack.sel(band="M1")}) == one
    assert tf.move.ud_distance([stack.sel(band="F1"), stack.sel(band="M1")]) == one


def test_sinkhorn_identical_uds_and_the_regularisation(ref):
    stack = uds(ref, "emd_fishers", names=FISHERS)
    f1 = stack.sel(band="F1")
    same_ud = tf.move.ud_distance(f1, f1)
    np.testing.assert_allclose(same_ud, ref["emd_fishers/same"][0], rtol=1e-12, atol=0)
    far = tf.move.ud_distance(f1, stack.sel(band="M1"))
    assert 0 <= same_ud < 0.2 * far              # the entropic distance of a UD to itself is small, not 0
    # a smaller regularisation is closer to the exact distance
    coarse = uds(ref, "emd_fishers", "_c", names=FISHERS)
    exact = tf.move.ud_distance(coarse.sel(band="F1"), coarse.sel(band="M1"), method="exact")
    d05 = tf.move.ud_distance(coarse.sel(band="F1"), coarse.sel(band="M1"), reg=0.05)
    d01 = tf.move.ud_distance(coarse.sel(band="F1"), coarse.sel(band="M1"), reg=0.01, max_iter=3000)
    assert abs(d01 - exact) < abs(d05 - exact)


def test_exact_distance_is_emdist_bit_for_bit(ref):
    stack = uds(ref, "emd_fishers", "_c", names=FISHERS)
    table = tf.move.ud_distance(stack, method="exact")
    same(table["distance"], ref["emd_fishers/exact"], "exact")
    same(tf.move.ud_distance(stack, method="exact", mask_quantile=0.9)["distance"], ref["emd_fishers/exact_q"], "exact 0.9")
    f1 = stack.sel(band="F1")
    assert tf.move.ud_distance(f1, f1, method="exact") == ref["emd_fishers/same_exact"][0] == 0.0
    # Sinkhorn is within a few percent of it on these UDs
    np.testing.assert_allclose(ref["emd_fishers/sink_c"], ref["emd_fishers/exact"], rtol=0.05)
    with pytest.raises(ValueError, match="gc"):
        tf.move.ud_distance(stack, method="exact", gc=True)
    with pytest.raises(ValueError, match="threshold"):
        tf.move.ud_distance(stack, method="exact", threshold=100)


@pytest.mark.parametrize("key,args", [("gc", {"gc": True}), ("gc_q", {"gc": True, "mask_quantile": 0.9}),
                                       ("gc_thr", {"gc": True, "threshold": 20000}),
                                       ("gc_reg", {"gc": True, "reg": 0.02, "max_iter": 1500}),
                                       ("deg", {}), ("deg_thr", {"threshold": 0.1})])
def test_sinkhorn_on_a_longitude_latitude_grid(key, args, ref):
    stack = uds(ref, "emd_lonlat", names="abc").rio.write_crs(4326)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = tf.move.ud_distance(stack, **args)
    assert list(table["a"]) == ["a", "a", "b"] and list(table["b"]) == ["b", "c", "c"]
    # the haversine goes through R's x87 sin, cos and atan2: R's bits where they exist
    rtol = 1e-12 if (BITWISE or not args.get("gc")) else 1e-9
    np.testing.assert_allclose(table["distance"], ref[f"emd_lonlat/{key}"], rtol=rtol, atol=0, err_msg=key)


def test_ud_distance_warns_on_degrees_and_checks(ref):
    stack = uds(ref, "emd_lonlat", names="abc").rio.write_crs(4326)
    with pytest.warns(UserWarning, match="gc=True"):
        tf.move.ud_distance(stack)
    plain = uds(ref, "emd_fishers", names=FISHERS)
    with pytest.raises(ValueError, match="at least two"):
        tf.move.ud_distance(plain.sel(band="F1"), ud_b=None)
    with pytest.raises(ValueError, match="same raster geometry"):
        tf.move.ud_distance(plain.sel(band="F1"), plain.sel(band="F2").assign_coords(x=plain.x + 1))
    with pytest.raises(ValueError, match="method"):
        tf.move.ud_distance(plain, method="lp")
    with pytest.raises(ValueError, match="mask_quantile"):
        tf.move.ud_distance(plain, mask_quantile=0)
    with pytest.raises(ValueError, match="reg"):
        tf.move.ud_distance(plain, reg=-1)
    with pytest.raises(ValueError, match="mass"):
        tf.move.ud_distance(plain.sel(band="F1"), plain.sel(band="M1") * 0)
    with pytest.raises(TypeError):
        tf.move.ud_distance(plain.values)


def test_sinkhorn_threads_agree(ref):
    stack = uds(ref, "emd_fishers", names=FISHERS)
    a = tf.move.ud_distance(stack, n_threads=1)["distance"].to_numpy()
    b = tf.move.ud_distance(stack, n_threads=4)["distance"].to_numpy()
    same(a, b, "threads")


def test_ud_distance_of_shifted_blobs_is_the_shift():
    x, y = np.arange(20) * 10.0 + 5, np.arange(20)[::-1] * 10.0 + 5
    gx, gy = np.meshgrid(x, y)

    def blob(cx, cy):
        d = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 20.0 ** 2))
        return xr.DataArray(d / d.sum(), coords={"y": y, "x": x}, dims=("y", "x"))

    exact = tf.move.ud_distance(blob(60, 100), blob(140, 100), method="exact", mask_quantile=1)
    assert exact == pytest.approx(80.0, rel=0.01)
    sink = tf.move.ud_distance(blob(60, 100), blob(140, 100), mask_quantile=1, reg=0.01, max_iter=2000)
    assert sink == pytest.approx(80.0, rel=0.05)
    assert tf.move.ud_distance(blob(60, 100), blob(60, 100), method="exact") == 0.0
