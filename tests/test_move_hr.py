# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move's home ranges against R's amt 0.3.1 (tests/move_cases.py, group "amt_hr"; tools/amt_hr.R).

The minimum convex polygons, the bandwidths (``hr_kde_ref``, ``hr_kde_pi``: KernSmooth's ``dpik``, ``hr_kde_lscv``),
the kernel density estimate (KernSmooth's ``bkde2D`` with R's fft), the isopleths (R's ``contourLines`` and GEOS), the
local convex hulls and the overlap indices are compared with the references bit for bit where the references were made
(an Intel CPU on Windows: the x87 functions of R's build, see ``conftest.BITWISE``); elsewhere the numbers agree to a
few units in the last place. The polygons of GEOS (the union of the contour lines, LoCoH's buffers) are compared by
vertices and areas: the same bits where the GEOS is the one of the references (3.12), otherwise the same polygons.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import Point

import terraflux as tf
from conftest import BITWISE
from terraflux import _core

REF = Path(__file__).parent / "data" / "reference" / "move"
LEVELS = [0.1, 0.5, 0.75, 0.95, 1.0]


@pytest.fixture(scope="module")
def ref():
    return dict(np.load(REF / "amt_hr.npz"))


@pytest.fixture(scope="module")
def inputs():
    return dict(np.load(REF / "inputs_hr.npz"))


@pytest.fixture(scope="module")
def sets(inputs):
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = frame(inputs, name)
        return cache[name]

    return get


def frame(inputs, name):
    x, y, t = inputs[f"{name}/x"], inputs[f"{name}/y"], inputs[f"{name}/t"]
    crs = str(inputs[f"{name}/crs"]) or None
    geom = [Point(a, b) for a, b in zip(x, y)]
    time = pd.to_datetime((t * 1e9).astype(np.int64), utc=True)
    df = gpd.GeoDataFrame({"time": time, "track": inputs[f"{name}/track"]}, geometry=geom, crs=crs)
    return tf.move.as_tracks(df, "time", "track", sort=False)


def labels(tr):
    return list(pd.unique(tr["track"]))


def one(tr, label):
    return tr[tr["track"] == label]


def same(got, want, err="", rtol=1e-9):
    """Bit for bit with the references (the same up to rounding where the x87 functions are not available)."""
    got, want = np.asarray(got, float), np.asarray(want, float)
    if BITWISE:
        np.testing.assert_array_equal(got, want, err_msg=err)
    else:
        np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-12 * max(1.0, np.nanmax(np.abs(want), initial=0)),
                                   equal_nan=True, err_msg=err)


def enc(g):
    """A Polygon or MultiPolygon as tools/amt_hr.R's hr_enc: the polygons, their rings, the vertices (x then y)."""
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
    out = [len(polys)]
    for p in polys:
        rings = [p.exterior, *p.interiors]
        out.append(len(rings))
        for r in rings:
            c = np.asarray(r.coords)
            out += [len(c), *c[:, 0], *c[:, 1]]
    return out


def decode(v):
    """The rings of an encoding, as arrays (n, 2), grouped by polygon."""
    v = list(map(float, v))
    i = 1
    polys = []
    for _ in range(int(v[0])):
        nr = int(v[i]); i += 1
        rings = []
        for _ in range(nr):
            n = int(v[i]); i += 1
            rings.append(np.column_stack([v[i:i + n], v[i + n:i + 2 * n]]))
            i += 2 * n
        polys.append(rings)
    return polys, i


def split_geoms(stream, count):
    """The encodings of ``count`` geometries one after the other in a vector."""
    out, i = [], 0
    for _ in range(count):
        _, used = decode(stream[i:])
        out.append(np.asarray(stream[i:i + used]))
        i += used
    assert i == len(stream)
    return out


def same_geom(got, want, err=""):
    """The polygon's rings are the same; as written by this GEOS (its vertices and their order are R's where it
    is the same) or, if not, the same set of vertices and the same area."""
    if len(got) == len(want) and np.array_equal(got, want):
        return
    pg, _ = decode(got)
    pw, _ = decode(want)
    assert len(pg) == len(pw), err
    key = lambda p: (len(p[0]), round(float(p[0][:, 0].sum()), 6))
    for a, b in zip(sorted(pg, key=key), sorted(pw, key=key)):
        assert len(a) == len(b), err
        for ra, rb in zip(a, b):
            assert len(ra) == len(rb), err
            sa = np.unique(np.round(ra, 8), axis=0)
            sb = np.unique(np.round(rb, 8), axis=0)
            np.testing.assert_allclose(sa, sb, rtol=0, atol=1e-6, err_msg=err)


def check_geoms(gdf, stream, err=""):
    wants = split_geoms(np.asarray(stream), len(gdf))
    for k, (g, w) in enumerate(zip(gdf.geometry, wants)):
        same_geom(np.asarray(enc(g)), w, f"{err} geometry {k}")


# --------------------------------------------------------------------------------------------- make_trast

def grid8(da):
    xmin, xmax, ymin, ymax = da.attrs["terraflux_extent"]
    ncol, nrow = da.sizes["x"], da.sizes["y"]
    return [ncol, nrow, xmin, xmax, ymin, ymax, (xmax - xmin) / ncol, (ymax - ymin) / nrow]


@pytest.mark.parametrize("case,name", [("hr_trast_utm", "hr_utm"), ("hr_trast_plain", "hr_plain"),
                                       ("hr_trast_fishers", "hr_fishers")])
def test_make_trast_is_amt(ref, sets, case, name):
    tr = sets(name)
    want = {k.split("/")[1]: v for k, v in ref.items() if k.startswith(case + "/")}
    kinds = {"def": {}, "res": {"res": 13.3}, "fac": {"factor": 2, "res": 7}, "big": {"res": 1000}}
    for key, kw in kinds.items():
        if key in want:
            got = np.concatenate([grid8(tf.move.make_trast(one(tr, t), **kw)) for t in labels(tr)])
            np.testing.assert_array_equal(got, want[key], err_msg=f"{case} {key}")
    pooled = (("pooled", {"factor": 1.2}),) if case == "hr_trast_plain" else (("pooled", {}), ("pooled_res", {"res": 5}))
    for key, kw in pooled:
        if key in want:
            np.testing.assert_array_equal(grid8(tf.move.make_trast(tr, **kw)), want[key], err_msg=f"{case} {key}")
    r = tf.move.make_trast(one(tr, labels(tr)[0]))
    assert r.dims == ("y", "x") and bool((r.values == 0).all())
    assert (r.rio.crs is None) == (tr.crs is None)


def test_make_trast_checks(sets):
    tr = sets("hr_utm")
    with pytest.raises(ValueError, match="factor"):
        tf.move.make_trast(tr, factor=0.5)
    with pytest.raises(ValueError, match="res"):
        tf.move.make_trast(tr, res=1e-12)
    with pytest.raises(TypeError, match="number"):
        tf.move.make_trast(tr, res="x")
    # a template grows with the factor and the coordinates of its cells are those of terra's xFromCol / yFromRow
    r = tf.move.make_trast(tr, res=50)
    xmin, xmax, ymin, ymax = r.attrs["terraflux_extent"]
    dx = (xmax - xmin) / r.sizes["x"]
    assert r["x"].values[0] == xmin + 0.5 * dx and r["y"].values[0] == ymax - 0.5 * ((ymax - ymin) / r.sizes["y"])


# ------------------------------------------------------------------------------------------------- MCP

@pytest.mark.parametrize("name", ["utm", "plain", "fishers", "small"])
def test_hr_mcp_is_amt(ref, sets, name):
    tr = sets(f"hr_{name}")
    case = f"hr_mcp_{name}"
    hr = tf.move.hr_mcp(tr, levels=LEVELS)
    iso = hr.polygons
    assert iso.shape == (len(labels(tr)) * len(LEVELS), 5)
    assert list(iso.columns) == ["track", "level", "what", "area", "geometry"]
    assert hr.levels.tolist() == LEVELS and (iso["what"] == "estimate").all()
    same(iso["area"], ref[f"{case}/area"], "areas")
    np.testing.assert_array_equal(iso["level"], ref[f"{case}/level"])
    check_geoms(iso, ref[f"{case}/geom"], case)
    same(tf.move.intensity_use(tr), ref[f"{case}/iu"], "intensity_use")
    assert list(tf.move.intensity_use(tr).index) == labels(tr)
    assert iso.crs == tr.crs


def test_hr_mcp_options(sets):
    tr = sets("hr_utm")
    hr = tf.move.hr_mcp(tr, levels=[0.95, 0.5])      # sorted, as amt does
    assert hr.levels.tolist() == [0.5, 0.95]
    assert hr.polygons["level"].tolist() == [0.5, 0.95] * 3
    iso = tf.move.hr_isopleths(hr)                   # the highest level first
    assert iso["level"].tolist() == [0.95, 0.5] * 3 and iso["track"].tolist() == ["a", "a", "b", "b", "c", "c"]
    asc = tf.move.hr_isopleths(hr, descending=False)
    assert asc["level"].tolist() == [0.5, 0.95] * 3
    area = tf.move.hr_area(hr)
    assert list(area.columns) == ["track", "level", "what", "area"] and (area["area"] == iso["area"]).all()
    assert tf.move.hr_area(hr, units=True)["unit"].iloc[0] == "m^2"
    assert hr.data is tr and tf.move.hr_mcp(tr, keep_data=False).data is None
    assert tf.move.home_range(tr, "mcp", levels=0.9).polygons.equals(tf.move.hr_mcp(tr, levels=0.9).polygons)
    # the polygon of each level holds that share of the locations (the squared distance to the mean location)
    a = one(tr, "a")
    p = tf.move.hr_mcp(a, levels=0.9).polygons.geometry.iloc[0]
    inside = sum(p.buffer(1e-6).contains(g) for g in a.geometry)
    assert inside >= int(0.9 * len(a)) - 1
    for bad in (1.5, -0.1, [], np.nan):
        with pytest.raises(ValueError, match="levels"):
            tf.move.hr_mcp(tr, levels=bad)
    one_track = tf.move.hr_mcp(tr.drop(columns="track"), levels=0.5)           # no track column: one track
    assert one_track.tracks == [1]
    with pytest.raises(ValueError, match="longitude"):
        tf.move.hr_mcp(tr.set_crs(4326, allow_override=True))


def test_degenerate_tracks_have_no_area(ref, sets):
    """One, two or three locations, or all on a line: R's polygon has no area (it is not an error)."""
    tr = sets("hr_edge")
    mcp = ref["hr_edge/mcp_area"].reshape(-1, 2)
    ok = ref["hr_edge/mcp_ok"]
    hr = tf.move.hr_mcp(tr, levels=[0.5, 1])
    got = hr.polygons["area"].to_numpy().reshape(-1, 2)
    assert ok.all()
    same(got, mcp, "areas of degenerate polygons")
    assert got[labels(tr).index("e1")].tolist() == [0.0, 0.0] and got[labels(tr).index("col")].tolist() == [0.0, 0.0]


# -------------------------------------------------------------------------------------------- bandwidths

def concat(df):
    return df.to_numpy().reshape(-1)


def test_bandwidths_are_amt(ref, sets):
    tr = sets("hr_utm")
    w = lambda k: ref[f"hr_bw_utm/{k}"]
    same(concat(tf.move.hr_kde_ref(tr)), w("ref"), "ref")
    same(concat(tf.move.hr_kde_ref(tr, "unitvar")), w("refu"), "ref unitvar")
    same(concat(tf.move.hr_kde_ref(tr, "xvar")), w("refx"), "ref xvar")
    same(concat(tf.move.hr_kde_pi(tr)), w("pi"), "pi")
    same(concat(tf.move.hr_kde_pi(tr, correct=False)), w("pi_raw"), "pi raw")
    same(concat(tf.move.hr_kde_pi(tr, rescale="unitvar")), w("piu"), "pi unitvar")
    same(concat(tf.move.hr_kde_pi(tr, rescale="xvar")), w("pix"), "pi xvar")
    for level in (0, 1, 3, 4, 5):
        same(concat(tf.move.hr_kde_pi(tr, level=level)), w(f"pi_l{level}"), f"pi level {level}")
    same(concat(tf.move.hr_kde_pi(tr, scalest="stdev")), w("pi_sd"), "pi stdev")
    same(concat(tf.move.hr_kde_pi(tr, scalest="iqr")), w("pi_iqr"), "pi iqr")
    same(concat(tf.move.hr_kde_pi(tr, kernel="epanech", gridsize=201, truncate=False)), w("pi_k"), "pi epanech")
    h = tf.move.hr_kde_ref(tr)
    assert list(h.columns) == ["hx", "hy"] and list(h.index) == labels(tr) and (h["hx"] == h["hy"]).all()


def test_bandwidths_on_the_fishers(ref, sets):
    tr = sets("hr_fishers")
    same(concat(tf.move.hr_kde_ref(tr)), ref["hr_bw_fishers/ref"], "ref")
    same(concat(tf.move.hr_kde_ref(tr, "unitvar")), ref["hr_bw_fishers/refu"], "ref unitvar")
    same(concat(tf.move.hr_kde_pi(tr)), ref["hr_bw_fishers/pi"], "pi")
    same(concat(tf.move.hr_kde_pi(tr, rescale="unitvar")), ref["hr_bw_fishers/piu"], "pi unitvar")


def test_bandwidth_checks(sets):
    tr = sets("hr_utm")
    with pytest.raises(ValueError, match="rescale"):
        tf.move.hr_kde_ref(tr, "nope")
    with pytest.raises(ValueError, match="scalest"):
        tf.move.hr_kde_pi(tr, scalest="nope")
    with pytest.raises(ValueError, match="Level"):
        tf.move.hr_kde_pi(tr, level=7)
    with pytest.raises(ValueError, match="kernel"):
        tf.move.hr_kde_pi(tr, kernel="nope")


def test_lscv_is_amt(ref, sets):
    tr = sets("hr_small")
    w = lambda k: ref[f"hr_lscv/{k}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = tf.move.hr_kde_lscv(tr)
        b = tf.move.hr_kde_lscv(tr, which_min="local")
        u = tf.move.hr_kde_lscv(tr, rescale="unitvar")
        r = tf.move.hr_kde_lscv(tr, range=np.linspace(2, 60, 30))
    cat = lambda res, key: np.concatenate([getattr(res.fits[t], key) for t in labels(tr)])
    same(concat(a.h), w("h"), "h")
    same(cat(a, "res"), w("res"), "criterion")
    same(cat(a, "range"), w("range"), "range")
    np.testing.assert_array_equal(a.converged.to_numpy().astype(float), w("conv"))
    same(concat(b.h), w("hl"), "local h")
    same(cat(b, "res"), w("resl"), "local criterion")
    same(concat(u.h), w("hu"), "unitvar h")
    same(cat(u, "res"), w("resu"), "unitvar criterion")
    same(concat(r.h), w("hr"), "range h")
    same(cat(r, "res"), w("resr"), "range criterion")
    assert list(a.h.index) == labels(tr) and list(a.converged.index) == labels(tr)


def test_lscv_threads_do_not_matter(sets):
    tr = sets("hr_small")
    a = tf.move.hr_kde_lscv(one(tr, "t"), n_threads=1)
    b = tf.move.hr_kde_lscv(one(tr, "t"), n_threads=4)
    np.testing.assert_array_equal(a.fits["t"].res, b.fits["t"].res)
    assert a.h.equals(b.h)


def test_lscv_warns_at_the_end_of_the_range(sets):
    tr = one(sets("hr_small"), "s")
    with pytest.warns(UserWarning, match="did not converge"):
        fit = tf.move.hr_kde_lscv(tr, range=np.linspace(100, 200, 10))
    assert not fit.converged.iloc[0]
    with pytest.raises(ValueError, match="which_min"):
        tf.move.hr_kde_lscv(tr, which_min="x")


def test_local_minima_is_amt():
    from terraflux.move._homerange import _local_minima

    assert _local_minima(np.array([5.0, 3, 4, 1, 2, 0, 7])).tolist() == [2, 4, 6]
    assert _local_minima(np.array([3.0, 3, 4, 1, 2])).tolist() == [4]    # amt drops the first run of a flat start
    assert _local_minima(np.array([1.0, 2, 3])).tolist() == [1]


# --------------------------------------------------------------------------------------------------- KDE

def check_kde(ref, case, tr, **kw):
    w = lambda k: ref[f"{case}/{k}"]
    hr = tf.move.hr_kde(tr, levels=[0.5, 0.9, 0.95], **kw)
    ud = hr.ud
    assert ud.dims == ("track", "y", "x") and list(ud["track"].values) == labels(tr)
    same(hr.h.to_numpy().reshape(-1), w("h"), "bandwidths")
    np.testing.assert_array_equal(grid8(hr.trast), w("grid") if w("grid").size == 8 else w("grid")[:8])
    same(np.concatenate([ud.values[k].reshape(-1) for k in range(ud.sizes["track"])]), w("ud"), "ud")
    same(tf.move.hr_ud(hr).values[0].reshape(-1), w("udn"), "hr_ud")
    same(tf.move.hr_cud(hr).values[0].reshape(-1), w("cud"), "hr_cud")
    iso = tf.move.hr_isopleths(hr)
    np.testing.assert_array_equal(iso["level"], w("level"))
    same(iso["area"], w("area"), "areas")
    check_geoms(iso, w("geom"), case)
    return hr


def test_kde_is_amt(ref, sets):
    tr = sets("hr_utm")
    hr = check_kde(ref, "hr_kde_utm", tr, trast=tf.move.make_trast(tr))
    assert hr.estimator == "kde" and hr.probabilistic and hr.crs == tr.crs
    assert hr.ud.rio.crs == tr.crs and float(abs(hr.ud.sum(("y", "x")) - hr.ud.sum(("y", "x"))).sum()) == 0.0
    bands = tf.move.hr_ud(hr).sum(("y", "x")).values
    np.testing.assert_allclose(bands, 1.0, rtol=0, atol=1e-12)
    # no template: one for all the tracks (amt's, for one track)
    pooled = tf.move.hr_kde(tr, levels=0.9)
    assert grid8(pooled.trast) == grid8(tf.move.make_trast(tr))


def test_kde_with_h_is_amt(ref, sets):
    tr = sets("hr_utm")
    check_kde(ref, "hr_kde_utm_h", tr, trast=tf.move.make_trast(tr), h=(25, 40))


def test_kde_without_crs_is_amt(ref, sets):
    tr = sets("hr_plain")
    hr = check_kde(ref, "hr_kde_plain", tr, trast=tf.move.make_trast(tr))
    assert hr.ud.rio.crs is None and hr.polygons is None


def test_kde_on_the_fishers_is_amt(ref, sets):
    tr = sets("hr_fishers")
    check_kde(ref, "hr_kde_fishers", tr, trast=tf.move.make_trast(tr))


def test_kde_of_one_track_has_amts_defaults(ref, sets):
    tr = sets("hr_utm")
    w = lambda k: ref[f"hr_kde_single/{k}"]
    uds, hs, isos = [], [], []
    for t in labels(tr):
        hr = tf.move.hr_kde(one(tr, t), levels=[0.5, 0.95])
        uds.append(hr.ud.values[0].reshape(-1))
        hs.append(hr.h.to_numpy().reshape(-1))
        isos.append(tf.move.hr_isopleths(hr))
    same(np.concatenate(hs), w("h"), "h")
    same(np.concatenate(uds), w("ud"), "ud")
    iso = pd.concat(isos)
    np.testing.assert_array_equal(iso["level"], w("level"))
    same(iso["area"], w("area"), "areas")
    check_geoms(iso, w("geom"), "single")


def test_plug_in_and_cross_validated_bandwidths_in_the_kde(ref, sets):
    tr = sets("hr_utm")
    tp = tf.move.make_trast(tr)
    w = lambda k: ref[f"hr_kde_pi_ud/{k}"]
    hr = tf.move.hr_kde(tr, h=tf.move.hr_kde_pi(tr), trast=tp, levels=0.95)
    same(hr.h.to_numpy().reshape(-1), w("h"), "h")
    same(tf.move.hr_area(hr)["area"], w("area"), "area")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lscv = tf.move.hr_kde_lscv(tr)
    same(concat(lscv.h), w("lscv_h"), "lscv h")


def test_kde_ref_scaled_is_amt(ref, sets):
    tr = sets("hr_utm")
    tp = tf.move.make_trast(tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = tf.move.hr_kde_ref_scaled(tr, trast=tp, levels=0.95)
        b = tf.move.hr_kde_ref_scaled(tr, trast=tp, levels=0.9, num_parts=2, tol=0.5)
    for res, key in ((a, "one"), (b, "two")):
        want = ref[f"hr_kde_scaled/{key}"].reshape(-1, 3)
        same(res["h"], want[:, 0], key)
        np.testing.assert_array_equal(res["n_iter"], want[:, 1])
        np.testing.assert_array_equal(res["success"].astype(float), want[:, 2])


def test_kde_checks(sets):
    tr = sets("hr_utm")
    one_t = one(tr, "a")
    with pytest.warns(UserWarning, match="same bandwidth"):
        a = tf.move.hr_kde(one_t, h=30)
    b = tf.move.hr_kde(one_t, h=(30, 30))
    np.testing.assert_array_equal(a.ud.values, b.ud.values)
    with pytest.warns(UserWarning, match="first 2"):
        tf.move.hr_kde(one_t, h=(30, 30, 30))
    with pytest.raises(TypeError, match="numeric"):
        tf.move.hr_kde(one_t, h="a")
    with pytest.raises(ValueError, match="strictly positive"):
        tf.move.hr_kde(one_t, h=(0, 3))
    with pytest.raises(ValueError, match="levels"):
        tf.move.hr_kde(one_t, levels=2)
    # a bandwidth by track (the table of hr_kde_ref), and a template without the extent attribute
    h = tf.move.hr_kde_ref(tr)
    tp = tf.move.make_trast(tr)
    plain = tp.copy()
    plain.attrs = {}
    x = tf.move.hr_kde(tr, h=h, trast=tp)
    y = tf.move.hr_kde(tr, h=h, trast=plain)
    np.testing.assert_allclose(x.ud.values, y.ud.values, rtol=1e-6, atol=1e-12)
    # a bandwidth so small that the grid is too coarse for it
    with pytest.warns(UserWarning, match="too coarse"):
        tf.move.hr_kde(one_t, h=(0.01, 0.01), trast=tp)


def test_isopleths_of_a_raster(sets):
    tr = sets("hr_utm")
    hr = tf.move.hr_kde(tr, levels=[0.5, 0.95])
    iso = tf.move.hr_isopleths(hr)
    ud = tf.move.hr_ud(hr)
    again = tf.move.hr_isopleths(ud, [0.5, 0.95])
    assert again["track"].tolist() == iso["track"].tolist() and np.array_equal(again["area"], iso["area"])
    one_band = tf.move.hr_isopleths(ud.sel(track="a"), 0.95)
    assert "track" not in one_band.columns and one_band["area"].iloc[0] == iso[(iso["track"] == "a") & (iso["level"] == 0.95)]["area"].iloc[0]
    assert tf.move.hr_area(ud.sel(track="a"), level=0.95) == one_band["area"].iloc[0]
    with pytest.raises(ValueError, match="levels"):
        tf.move.hr_isopleths(ud)
    # an isopleth that reaches the edge of the template is not a polygon
    small = tf.move.make_trast(tr, factor=1.0)
    with pytest.raises(ValueError, match="edge"):
        tf.move.hr_isopleths(tf.move.hr_kde(tr, trast=small, levels=0.99))


# -------------------------------------------------------------------------------------------------- LoCoH

# The hulls are buffered by 1e-5 and united: the buffer of GEOS 3.12 (the references') has 134 vertices where the one of
# 3.13 has 128 (the arcs of the corners are cut in another number of segments), so the areas agree to the last digits
# of the buffer's area (1e-8 of the area), not to the bit; the hulls, their order, the levels and the rings of the
# united polygons (same corners) are the references'.
LOCOH_RTOL = 1e-6


def check_locoh(ref, case, tr, **kw):
    w = lambda k: ref[f"{case}/{k}"]
    hr = tf.move.hr_locoh(tr, **kw)
    poly = hr.polygons
    np.testing.assert_array_equal(poly["level"], w("level"))
    np.testing.assert_allclose(poly["area"], w("area"), rtol=LOCOH_RTOL, atol=0, err_msg="areas")
    wants = split_geoms(np.asarray(w("geom")), len(poly))
    for k, (g, want) in enumerate(zip(poly.geometry, wants)):
        mine, _ = decode(np.asarray(enc(g)))
        theirs, _ = decode(want)
        assert [len(p) for p in mine] == [len(p) for p in theirs], f"{case}: rings of geometry {k}"
    return hr


def test_locoh_k_is_amt(ref, sets):
    tr = sets("hr_utm")
    hr = check_locoh(ref, "hr_locoh_k_utm", tr, n=8, type="k", levels=[0.5, 0.9, 0.95])
    assert hr.estimator == "locoh" and hr.params["type"] == "k"
    check_locoh(ref, "hr_locoh_k_small", sets("hr_small"), n=4, type="k", levels=[0.3, 0.8, 1])
    check_locoh(ref, "hr_locoh_plain", sets("hr_plain"), n=6, levels=0.95)
    check_locoh(ref, "hr_locoh_k_fishers", sets("hr_fishers_s"), n=10, levels=[0.5, 0.95])


def test_locoh_r_and_a_are_amt(ref, sets):
    tr = sets("hr_utm")
    check_locoh(ref, "hr_locoh_r_utm", tr, n=60, type="r", levels=[0.5, 0.9, 0.95])
    check_locoh(ref, "hr_locoh_a_utm", tr, n=400, type="a", levels=[0.5, 0.9, 0.95])


def test_locoh_checks_and_threads(sets):
    tr = sets("hr_utm")
    one_t = one(tr, "c")
    with pytest.raises(ValueError, match="incorrect type"):
        tf.move.hr_locoh(one_t, type="z")
    with pytest.raises(ValueError, match="n > number"):
        tf.move.hr_locoh(one_t, n=1000)
    with pytest.raises(ValueError, match="less than"):
        tf.move.hr_locoh(one_t, n=len(one_t))
    with pytest.raises(TypeError, match="number"):
        tf.move.hr_locoh(one_t, n="x")
    a = tf.move.hr_locoh(tr, n=6, levels=[0.5, 0.9], n_threads=1).polygons
    b = tf.move.hr_locoh(tr, n=6, levels=[0.5, 0.9], n_threads=4).polygons
    assert a["geometry"].geom_equals_exact(b["geometry"], 0).all() and a["area"].equals(b["area"])
    x, y = np.asarray(one_t.geometry.x), np.asarray(one_t.geometry.y)
    for type_, d in ((0, 5.0), (1, 40.0), (2, 300.0)):
        o1, i1 = _core.move_hr_locoh_neighbours(x, y, type_, d, 1)
        o4, i4 = _core.move_hr_locoh_neighbours(x, y, type_, d, 4)
        assert np.array_equal(o1, o4) and np.array_equal(i1, i4)


# ------------------------------------------------------------------------------------------------ overlap

def test_overlap_is_amt(ref, sets):
    for case, name in (("hr_overlap_utm", "hr_utm"), ("hr_overlap_fishers", "hr_fishers_s")):
        tr = sets(name)
        w = lambda k: ref[f"{case}/{k}"]
        tp = tf.move.make_trast(tr)
        tracks = labels(tr)
        kk = {t: tf.move.hr_kde(one(tr, t), trast=tp, levels=[0.5, 0.9]) for t in tracks}
        mm = {t: tf.move.hr_mcp(one(tr, t), levels=[0.5, 0.9]) for t in tracks}
        ll = {t: tf.move.hr_locoh(one(tr, t), n=6, levels=[0.5, 0.9]) for t in tracks}
        ov = lambda e, ty, cond=False, which="all": tf.move.hr_overlap(e, type=ty, conditional=cond, which=which)
        same(ov(kk, "hr")["overlap"], w("hr_kde"), "hr kde")
        same(ov(mm, "hr")["overlap"], w("hr_mcp"), "hr mcp")
        np.testing.assert_allclose(ov(ll, "hr")["overlap"], w("hr_locoh"), rtol=LOCOH_RTOL, atol=1e-12, err_msg="hr locoh")
        same(ov(mm, "hr", which="consecutive")["overlap"], w("hr_cons"), "consecutive")
        same(ov(mm, "hr", which="one_to_all")["overlap"], w("hr_one"), "one to all")
        for ty in ("phr", "vi", "ba", "udoi", "hd"):
            # phr is the share of one polygon in another: a cell of the isopleth that the last bits of the density move
            # across its threshold (ARM64, no x87: 0.4299 for 0.4146 in one pair, the rest within 5e-8) moves it; the
            # other indices are sums over the cells, 1.5e-9 off there (ba)
            rtol = 0.05 if ty == "phr" else 1e-7
            same(ov(kk, ty)["overlap"], w(ty), ty, rtol)
            same(ov(kk, ty, True)["overlap"], w("c" + ty), "conditional " + ty, rtol)
        same(ov(kk, "vi", which="consecutive")["overlap"], w("cons"), "vi consecutive")
        pair = tf.move.hr_overlap(kk[tracks[0]], kk[tracks[1]], type="hr")
        same(pair["overlap"], w("pair"), "pair")
        pairu = tf.move.hr_overlap(kk[tracks[1]], kk[tracks[0]], type="udoi")
        same(pairu["overlap"], w("pairu"), "pair udoi")


def test_overlap_tables_and_checks(sets):
    tr = sets("hr_utm")
    tp = tf.move.make_trast(tr)
    hr = tf.move.hr_kde(tr, trast=tp, levels=[0.5, 0.9])
    t = tf.move.hr_overlap(hr, type="vi", which="consecutive")
    assert list(t.columns) == ["from", "to", "level", "overlap"] and t[["from", "to"]].drop_duplicates().shape[0] == 2
    assert t["level"].tolist() == [1.0, 1.0]
    c = tf.move.hr_overlap(hr, type="ba", conditional=True, which="one_to_all")
    assert c["level"].tolist() == [0.5, 0.9, 0.5, 0.9] and c["from"].tolist() == ["a"] * 4
    lab = tf.move.hr_overlap(hr, type="hd", labels=["u", "v", "w"])
    assert set(lab["from"]) | set(lab["to"]) == {"u", "v", "w"}
    with pytest.warns(UserWarning, match="labels"):
        tf.move.hr_overlap(hr, type="hd", labels=["u"])
    # with y: the tracks of both, matched by their label
    y = tf.move.hr_kde(tr[tr["track"].isin(["b", "c"])], trast=tp, levels=[0.5, 0.9])
    m = tf.move.hr_overlap(hr, y, type="phr")
    assert m["track"].tolist() == ["b", "c"]
    single = tf.move.hr_overlap(hr.sel("a"), y.sel("b"), type="phr")
    assert single["track"].tolist() == ["a"]
    with pytest.raises(ValueError, match="should be one of"):
        tf.move.hr_overlap(hr, type="zz")
    with pytest.raises(ValueError, match="Which"):
        tf.move.hr_overlap(hr, which="zz")
    with pytest.raises(ValueError, match="At least two"):
        tf.move.hr_overlap(hr.sel("a"))
    with pytest.raises(ValueError, match="in common"):
        tf.move.hr_overlap(hr.sel("a").sel("a") if False else tf.move.hr_kde(tr[tr["track"].isin(["a", "b"])], trast=tp),
                           tf.move.hr_kde(tr[tr["track"].isin(["c"])], trast=tp).sel("c") if False else
                           tf.move.hr_kde(tr[tr["track"].isin(["c", "b"])].assign(track=lambda d: d["track"].replace({"b": "q", "c": "r"})),
                                          trast=tp))
    mc = tf.move.hr_mcp(tr, levels=[0.5, 0.9])
    with pytest.raises(ValueError, match="probabilistic"):
        tf.move.hr_overlap(mc, type="vi")
    other = tf.move.hr_kde(tr, trast=tf.move.make_trast(tr, res=40), levels=[0.5, 0.9])
    with pytest.raises(ValueError, match="Extend and resolution"):
        tf.move.hr_overlap(hr.sel("a"), other.sel("b"), type="vi")
    lv = tf.move.hr_kde(tr, trast=tp, levels=[0.5])
    with pytest.raises(ValueError, match="same level"):
        tf.move.hr_overlap(hr.sel("a"), lv.sel("b"), type="vi", conditional=True)


# ---------------------------------------------------------------------------------------- degenerate sets

def test_kde_and_locoh_of_degenerate_tracks(ref, sets):
    tr = sets("hr_edge")
    tp = tf.move.make_trast(tr)
    names = labels(tr)
    kde_ok = ref["hr_edge/kde_ok"].astype(bool)
    kde_area = ref["hr_edge/kde_area"]
    got_ok, got_area = [], []
    for t in names:
        try:
            hr = tf.move.hr_kde(one(tr, t), trast=tp, levels=0.9)
            got_area.append(tf.move.hr_isopleths(hr)["area"].iloc[0])
            got_ok.append(True)
        except ValueError:
            got_ok.append(False)
            got_area.append(np.nan)
    assert got_ok == kde_ok.tolist()
    same(got_area, kde_area, "kde areas")
    # locoh: R's k nearest neighbours of a point with fewer than k + 1 others are NA (FNN prints an error): here an error
    loc_ok = ref["hr_edge/locoh_ok"].astype(bool)
    area = ref["hr_edge/locoh_area"].reshape(-1, 2)
    for k, t in enumerate(names):
        if len(one(tr, t)) <= 2:
            with pytest.raises(ValueError):
                tf.move.hr_locoh(one(tr, t), n=2, levels=[0.5, 0.9])
            continue
        assert loc_ok[k]
        if t == "dup":   # FNN drops the first of the k + 1 nearest as if it were the point: not with twin locations
            assert (tf.move.hr_locoh(one(tr, t), n=2, levels=[0.5, 0.9]).polygons["area"] > 0).all()
            continue
        np.testing.assert_allclose(tf.move.hr_locoh(one(tr, t), n=2, levels=[0.5, 0.9]).polygons["area"], area[k],
                                   rtol=LOCOH_RTOL, atol=1e-12, err_msg=f"locoh {t}")


# ------------------------------------------------------------------------------------------ R's own pieces

def test_r_round_is_rs():
    from terraflux.move._homerange import _r_round

    # R 4.4.2's round(x, 2)
    for v, want in ((0.945, 0.94), (0.123456, 0.12), (0.5, 0.5), (0.125, 0.12), (0.135, 0.14), (1 / 3, 0.33),
                    (2 / 3, 0.67), (0.9999, 1.0), (0.005, 0.0), (0.015, 0.01), (0.025, 0.03), (0.955, 0.96),
                    (0.8449999, 0.84), (0.65, 0.65), (0.285, 0.28)):
        assert _r_round(v, 2) == want, v


def test_fft_is_numpys_for_powers_of_two():
    rng = np.random.default_rng(3)
    for n in (2, 4, 8, 16, 32, 64, 128, 512):
        re, im = rng.normal(size=n), rng.normal(size=n)
        a, b = _core.move_hr_fft(re, im, n, 1, False)
        np.testing.assert_allclose(a + 1j * b, np.fft.fft(re + 1j * im), rtol=0, atol=1e-12 * n)
        c, d = _core.move_hr_fft(a, b, n, 1, True)
        np.testing.assert_allclose((c + 1j * d) / n, re + 1j * im, rtol=0, atol=1e-13)
    re, im = rng.normal(size=16 * 32), rng.normal(size=16 * 32)
    a, b = _core.move_hr_fft(re, im, 16, 32, False)
    np.testing.assert_allclose((a + 1j * b).reshape(32, 16), np.fft.fft2((re + 1j * im).reshape(32, 16)), atol=1e-11)
    with pytest.raises(ValueError, match="powers of two"):
        _core.move_hr_fft(np.zeros(6), np.zeros(6), 6, 1, False)


def test_chull_has_rs_vertices_and_order():
    # R 4.4.2's chull(): the C code's hull, then sorted by the angle around the mean of its vertices
    x = np.array([0.0, 1, 1, 0, 0.5, 0.5, 0.0])      # a square with a point inside, one on an edge, a repeated vertex
    y = np.array([0.0, 0, 1, 1, 0.5, 0.0, 0.0])
    assert _core.move_hr_chull(x, y).tolist() == [1, 0, 3, 2]                                       # R: 2 1 4 3
    assert _core.move_hr_chull(np.array([2.0]), np.array([3.0])).tolist() == [0]                    # R: 1
    assert _core.move_hr_chull(np.array([2.0, 5]), np.array([3.0, 4])).tolist() == [0, 1]           # R: 1 2
    assert _core.move_hr_chull(np.arange(5.0), np.arange(5.0)).tolist() == [0, 4]                   # R: 1 5
    assert _core.move_hr_chull(np.array([1.0, 1, 1]), np.array([1.0, 2, 3])).tolist() == [0, 2]    # R: 1 3
    assert _core.move_hr_chull(np.array([0, 0, 1, 1, 0.5]), np.array([0, 1, 1, 0, 0.5])).tolist() == [3, 0, 1, 2]  # R: 4 1 2 3


def test_quantile_and_seq_are_rs():
    x = np.array([5.0, 1, 4, 2, 3])
    assert _core.move_hr_quantile7(x, np.array([0, 0.25, 0.5, 0.9, 1])).tolist() == [1.0, 2.0, 3.0, 4.6, 5.0]
    assert _core.move_hr_seq(0.0, 1.0, 5).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    s = _core.move_hr_seq(1.1, 2.3, 7)
    assert s[0] == 1.1 and s[-1] == 2.3
    assert _core.move_hr_seq(3.0, 3.0, 3).tolist() == [3.0, 3.0, 3.0]


def test_areas_grow_with_the_level_for_every_method(sets):
    tr = one(sets("hr_utm"), "a")
    levels = [0.5, 0.6, 0.75, 0.9, 0.95, 0.99]
    for method, kw in (("mcp", {}), ("kde", {}), ("locoh", {"n": 8})):
        hr = tf.move.home_range(tr, method, levels=levels, **kw)
        area = tf.move.hr_area(hr).sort_values("level")["area"].to_numpy()
        assert np.all(np.diff(area) > 0) or method == "locoh", method        # LoCoH's levels can fall on the same hull
        assert np.all(np.diff(area) >= 0), method
        assert repr(hr).startswith(f"HomeRange('{method}'")
    sub = tf.move.hr_kde(sets("hr_utm"), levels=0.9).sel("b")
    assert sub.tracks == ["b"] and sub.ud.sizes["track"] == 1 and sub.h.index.tolist() == ["b"]
    with pytest.raises(KeyError):
        sub.sel("zzz")
