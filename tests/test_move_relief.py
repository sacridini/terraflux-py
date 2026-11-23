# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.move.surface_distance: the 3D length of a step over a DEM (not in amt: tested by properties)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

import terraflux as tf
from test_move import FISHERS

ELEVATION = Path(tf.move.__file__).parent / "data" / "elevation.tif"


def make_dem(z, res=10.0, x0=0.0, ymax=None, crs="EPSG:32633"):
    """A (y, x) DEM of cells of ``res`` metres with the top left corner at (x0, ymax); z(x, y) of the centres."""
    z = np.asarray(z, dtype=np.float64)
    ny, nx = z.shape
    ymax = ny * res if ymax is None else ymax
    da = xr.DataArray(z, dims=("y", "x"), coords={"y": ymax - res * (np.arange(ny) + 0.5),
                                                   "x": x0 + res * (np.arange(nx) + 0.5)}, name="dem")
    return da.rio.write_crs(crs).rio.write_transform(Affine(res, 0, x0, 0, -res, ymax))


def plane_dem(a, b, n=60, res=10.0):
    x = res * (np.arange(n) + 0.5)
    y = n * res - res * (np.arange(n) + 0.5)
    return make_dem(a * x[None, :] + b * y[:, None], res)


def frame(x1, y1, x2, y2, crs="EPSG:32633"):
    out = pd.DataFrame({"x1_": x1, "y1_": y1, "x2_": x2, "y2_": y2})
    out.attrs["terraflux.move"] = {"crs": crs, "lonlat": False}
    return out


def random_inside(rng, n, lo=40.0, hi=560.0):
    p = rng.uniform(lo, hi, (4, n))
    return frame(*p)


def test_plane_is_analytic():
    a, b = 0.37, -0.52
    rng = np.random.default_rng(1)
    steps = random_inside(rng, 200, 10.0, 590.0)   # inside the outermost centres: the bilinear is exact
    got = tf.move.surface_distance(steps, plane_dem(a, b))["sl_surface_"].to_numpy()
    dx, dy = steps.x2_ - steps.x1_, steps.y2_ - steps.y1_
    want = np.hypot(dx, dy) * np.sqrt(1 + ((a * dx + b * dy) / np.hypot(dx, dy)) ** 2)
    assert np.allclose(got, want, rtol=1e-12, atol=0)


def test_flat_dem_is_the_planar_length():
    flat = make_dem(np.full((60, 60), 123.0))
    steps = random_inside(np.random.default_rng(2), 100)
    got = tf.move.surface_distance(steps, flat)["sl_surface_"].to_numpy()
    want = np.hypot(steps.x2_ - steps.x1_, steps.y2_ - steps.y1_).to_numpy()
    assert np.allclose(got, want, rtol=4e-16 * 8, atol=0)


def test_flat_dem_equals_steps_sl():
    import rioxarray as rx

    elev = rx.open_rasterio(ELEVATION).isel(band=0, drop=True)
    dem = elev * 0.0 + 100.0
    f = tf.load_tracks(FISHERS, sort=False)
    st = tf.move.steps(f[~f.geometry.is_empty].to_crs(elev.rio.crs))
    out = tf.move.surface_distance(st, dem)
    ok = out["sl_surface_"].notna()
    assert ok.sum() > 100
    assert np.allclose(out.loc[ok, "sl_surface_"], st.loc[ok, "sl_"], rtol=1e-13, atol=0)
    assert list(out.columns) == list(st.columns) + ["sl_surface_"]
    assert "sl_surface_" not in st.columns
    assert out.attrs == st.attrs


def test_never_shorter_than_planar():
    rng = np.random.default_rng(5)
    dem = make_dem(rng.normal(0, 30, (60, 60)))
    steps = random_inside(rng, 300, 5.0, 595.0)
    got = tf.move.surface_distance(steps, dem)["sl_surface_"].to_numpy()
    planar = np.hypot(steps.x2_ - steps.x1_, steps.y2_ - steps.y1_).to_numpy()
    assert np.isfinite(got).all()
    assert (got >= planar * (1 - 1e-15)).all()
    assert (got > planar).mean() > 0.99


def test_denser_points_change_little_on_a_smooth_dem():
    n = 80
    x = 10.0 * (np.arange(n) + 0.5)
    y = n * 10.0 - 10.0 * (np.arange(n) + 0.5)
    dem = make_dem(40 * np.sin(x[None, :] / 90) * np.cos(y[:, None] / 70))
    steps = random_inside(np.random.default_rng(6), 100, 20.0, 780.0)
    a = tf.move.surface_distance(steps, dem, spacing=2.5)["sl_surface_"].to_numpy()
    b = tf.move.surface_distance(steps, dem, spacing=1.25)["sl_surface_"].to_numpy()
    assert np.allclose(a, b, rtol=1e-3)
    assert np.abs(a - b).max() < 0.05


def test_outside_nan_cell_and_null_step():
    z = np.zeros((10, 10))
    z[5, 5] = np.nan   # the cell at x 50-60, y 40-50 (rows from the top)
    dem = make_dem(z)
    steps = frame([15.0, 15.0, 15.0, 15.0, 55.0, 15.0, np.nan],
                  [15.0, 15.0, 15.0, 15.0, 45.0, 15.0, 15.0],
                  [85.0, 120.0, -5.0, 15.0, 55.0, 15.0, 25.0],
                  [85.0, 20.0, 20.0, 15.0, 45.0, 85.0, 25.0])
    got = tf.move.surface_distance(steps, dem)["sl_surface_"].to_numpy()
    assert np.isnan(got[0])            # crosses the cell with no data
    assert np.isnan(got[1]) and np.isnan(got[2])   # ends outside the extent
    assert got[3] == 0.0               # a null step inside
    assert np.isnan(got[4])            # a null step on a NaN cell
    assert np.isfinite(got[5]) and got[5] == pytest.approx(70.0)   # beside the NaN cell, flat
    assert np.isnan(got[6])            # a NaN coordinate
    # a step that stays clear of the NaN cell is fine
    assert np.isfinite(tf.move.surface_distance(frame([15.0], [15.0], [15.0], [85.0]), dem)["sl_surface_"]).all()


def test_edge_half_cell_takes_the_edge_cell():
    dem = make_dem(np.tile(np.arange(10.0), (10, 1)) * 10)   # z = x - 5 at the centres
    out = tf.move.surface_distance(frame([0.0], [50.0], [5.0], [50.0]), dem)["sl_surface_"].iloc[0]
    assert out == pytest.approx(5.0)   # flat in the outer half cell (clamped)


def test_threads_give_the_same_bits():
    rng = np.random.default_rng(7)
    dem = make_dem(rng.normal(0, 20, (60, 60)))
    steps = random_inside(rng, 5000, 5.0, 595.0)
    steps.loc[::97, "x2_"] = 700.0
    one = tf.move.surface_distance(steps, dem, n_threads=1)["sl_surface_"].to_numpy()
    four = tf.move.surface_distance(steps, dem, n_threads=4)["sl_surface_"].to_numpy()
    assert np.array_equal(one.view(np.int64), four.view(np.int64))


def test_degrees_are_an_error():
    dem = make_dem(np.zeros((10, 10)))
    with pytest.raises(ValueError, match="projected"):
        tf.move.surface_distance(frame([1.0], [1.0], [2.0], [2.0], "EPSG:4326"), dem)
    deg = make_dem(np.zeros((10, 10)), res=0.001, crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected"):
        tf.move.surface_distance(frame([0.001], [0.001], [0.002], [0.002]), deg)
    with pytest.raises(ValueError, match="same CRS"):
        tf.move.surface_distance(frame([1.0], [1.0], [2.0], [2.0], "EPSG:32632"), dem)
    with pytest.raises(ValueError, match="spacing"):
        tf.move.surface_distance(frame([1.0], [1.0], [2.0], [2.0]), dem, spacing=0)
    with pytest.raises(TypeError):
        tf.move.surface_distance(pd.DataFrame({"a": [1]}), dem)


def test_custom_columns_and_name():
    dem = plane_dem(0.5, 0.0)
    st = pd.DataFrame({"a": [50.0], "b": [50.0], "c": [150.0], "d": [50.0]})
    out = tf.move.surface_distance(st, dem, x1="a", y1="b", x2="c", y2="d", name="len3d")
    assert out["len3d"].iloc[0] == pytest.approx(100 * np.sqrt(1.25), rel=1e-12)


def test_runs_on_the_fishers_steps():
    import rioxarray as rx

    dem = rx.open_rasterio(ELEVATION).isel(band=0, drop=True)
    f = tf.load_tracks(FISHERS, sort=False)
    f = f[~f.geometry.is_empty].to_crs(dem.rio.crs)
    st = tf.move.steps(f)
    rs = tf.move.random_steps(st.iloc[:300].dropna(subset=["ta_"]), n_control=5, seed=1)
    out = tf.move.surface_distance(rs, dem)
    assert len(out) == len(rs) and out["sl_surface_"].notna().any()
    ok = out["sl_surface_"].notna()
    assert (out.loc[ok, "sl_surface_"] >= out.loc[ok, "sl_"] * (1 - 1e-12)).all()
