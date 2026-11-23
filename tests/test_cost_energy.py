# SPDX-License-Identifier: GPL-3.0-or-later
"""tf.cost.energy_landscape against R's enerscape 1.2.0 (bit for bit) and unit checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import terraflux as tf

REF = Path(__file__).parent / "data" / "reference" / "cost" / "energy"


def _csv(name):
    return np.genfromtxt(REF / f"{name}.csv", delimiter=",")  # "NA" reads as NaN


def _dem(z, x0, y1, res, crs="EPSG:32632"):
    ny, nx = z.shape
    da = xr.DataArray(z, dims=("y", "x"),
                      coords={"y": y1 - res * (np.arange(ny) + 0.5), "x": x0 + res * (np.arange(nx) + 0.5)})
    return da.rio.write_crs(crs) if crs else da


def _small():
    return _dem(_csv("small_dem"), 500000.0, 5000450.0, 30.0)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.int64)[~np.isnan(a)], b.view(np.int64)[~np.isnan(b)]) and \
        np.array_equal(np.isnan(a), np.isnan(b))


@pytest.mark.external_reference
@pytest.mark.parametrize("name, kw", [
    ("small_m50_j_n8_out", dict(mass=50, neigh=8)),
    ("small_m50_j_n4_out", dict(mass=50, neigh=4)),
    ("small_m50_kcal_n8_out", dict(mass=50, neigh=8, unit="kcal")),
    ("small_human_m70_v139_j_n8_out", dict(mass=70, model="human", v=1.39)),
])
def test_small_dem_matches_enerscape(name, kw):
    out = tf.cost.energy_landscape(_small(), **kw)
    assert _same(out.values, _csv(name))


@pytest.mark.external_reference
@pytest.mark.parametrize("key, kw", [
    ("j_n8", dict(mass=10, neigh=8, unit="joule")),
    ("kcal_n4", dict(mass=10, neigh=4, unit="kcal")),
])
def test_sirente_matches_enerscape(key, kw):
    z = np.load(REF / "sirente.npz")
    dem = _dem(z["dem"].astype(np.float64), 879340.0, 4676810.0, 10.0)
    out = tf.cost.energy_landscape(dem, **kw)
    assert _same(out.values, z[key])
    assert int(np.isnan(out.values).sum()) == 1970


def test_human_default_speed_is_enerscapes():
    a = tf.cost.energy_landscape(_small(), 70, model="human")
    b = tf.cost.energy_landscape(_small(), 70, model="human", v=1.39)
    assert _same(a.values, b.values)


def test_keeps_the_grid_and_crs():
    dem = _small()
    out = tf.cost.energy_landscape(dem, 50)
    assert out.name == "energy" and out.dtype == np.float64
    np.testing.assert_array_equal(out.x, dem.x)
    np.testing.assert_array_equal(out.y, dem.y)
    assert out.rio.crs == dem.rio.crs
    assert out.rio.transform() == dem.rio.transform()


def test_border_and_neighbourhood_of_a_missing_height_are_nan():
    z = _csv("small_dem")
    z[5, 5] = np.nan
    for neigh, expect in ((8, 66 + 9), (4, 66 + 5)):
        out = tf.cost.energy_landscape(_dem(z, 0.0, 450.0, 30.0), 50, neigh=neigh).values
        assert int(np.isnan(out).sum()) == expect
        assert np.isnan(out[[0, -1]]).all() and np.isnan(out[:, [0, -1]]).all()
        assert np.isnan(out[4:7, 5]).all() and np.isnan(out[5, 4:7]).all()
    assert not np.isnan(tf.cost.energy_landscape(_dem(z, 0.0, 450.0, 30.0), 50, neigh=4).values[4, 4])


def test_flat_ground_is_pontzers_cost_per_metre():
    z = np.full((5, 6), 100.0)
    out = tf.cost.energy_landscape(_dem(z, 0.0, 50.0, 10.0), 20.0).values
    ar = 8.0 * 20.0 ** -0.34
    mec = 50.0 * (1.0 + np.sin(-74.0 / 180.0 * 3.1415926535)) * 20.0 ** -0.12
    inner = out[1:-1, 1:-1]
    # four straight steps of 10 m, four diagonal ones of 10 sqrt(2) m
    expect = (4 * 10.0 + 4 * 10.0 * np.sqrt(2)) / 8 * (ar + mec) * 20.0
    np.testing.assert_allclose(inner, expect, rtol=1e-12)


def test_kcal_is_joule_over_4184():
    j = tf.cost.energy_landscape(_small(), 50).values
    k = tf.cost.energy_landscape(_small(), 50, unit="kcal").values
    np.testing.assert_allclose(k, j / 4184.0, rtol=1e-14, equal_nan=True)


def test_conductance_is_the_reciprocal():
    dem = _small()
    en = tf.cost.energy_landscape(dem, 50).values
    cond = tf.cost.energy_landscape(dem, 50, output="conductance")
    assert cond.name == "conductance"
    assert np.array_equal(cond.values, 1.0 / en, equal_nan=True)
    both = tf.cost.energy_landscape(dem, 50, output="both")
    assert isinstance(both, xr.Dataset) and set(both.data_vars) == {"energy", "conductance"}
    assert np.array_equal(both["energy"].values, en, equal_nan=True)
    assert np.array_equal(both["conductance"].values, cond.values, equal_nan=True)
    assert both["energy"].rio.crs == dem.rio.crs


def test_arrays_in_arrays_out_and_no_crs_is_metres():
    z = _csv("small_dem")
    e, c = tf.cost.energy_landscape(z, 50, output="both")
    assert isinstance(e, np.ndarray) and isinstance(c, np.ndarray)
    # unit cells (1 m), as terra::rast(matrix): the same as a DataArray of 1 m cells with no CRS
    ref = tf.cost.energy_landscape(_dem(z, 0.0, 15.0, 1.0, crs=None), 50)
    assert _same(e, ref.values)


def test_threads_do_not_matter():
    z = np.load(REF / "sirente.npz")["dem"].astype(np.float64)
    dem = _dem(z, 879340.0, 4676810.0, 10.0)
    for kw in (dict(neigh=8), dict(neigh=4, unit="kcal"), dict(model="human", v=1.2)):
        a = tf.cost.energy_landscape(dem, 10, n_threads=1, **kw).values
        b = tf.cost.energy_landscape(dem, 10, n_threads=4, **kw).values
        assert _same(a, b)


def test_tiny_grids_are_all_nan():
    out = tf.cost.energy_landscape(_dem(np.ones((2, 7)), 0.0, 2.0, 1.0), 10)
    assert np.isnan(out.values).all()


@pytest.mark.parametrize("neigh", [0, 3, 6, 16, True])
def test_neigh_other_than_4_or_8_is_an_error(neigh):
    with pytest.raises(ValueError, match="neigh"):
        tf.cost.energy_landscape(_small(), 50, neigh=neigh)


@pytest.mark.parametrize("mass", [0, -1, float("nan"), float("inf")])
def test_mass_must_be_positive(mass):
    with pytest.raises(ValueError, match="mass"):
        tf.cost.energy_landscape(_small(), mass)


def test_other_argument_errors():
    dem = _small()
    with pytest.raises(ValueError, match="unit"):
        tf.cost.energy_landscape(dem, 50, unit="watt")
    with pytest.raises(ValueError, match="model"):
        tf.cost.energy_landscape(dem, 50, model="cyclist")
    with pytest.raises(ValueError, match="output"):
        tf.cost.energy_landscape(dem, 50, output="path")
    with pytest.raises(ValueError, match="speed"):
        tf.cost.energy_landscape(dem, 50, v=1.2)  # v is for the human model
    with pytest.raises(ValueError, match="positive"):
        tf.cost.energy_landscape(dem, 50, model="human", v=0)


def test_geographic_crs_is_an_error():
    dem = _dem(_csv("small_dem"), 10.0, 46.0, 0.001, crs="EPSG:4326")
    with pytest.raises(ValueError, match="degrees"):
        tf.cost.energy_landscape(dem, 50)


def test_cells_must_be_square():
    z = _csv("small_dem")
    ny, nx = z.shape
    da = xr.DataArray(z, dims=("y", "x"), coords={"y": 450.0 - 20.0 * (np.arange(ny) + 0.5),
                                                  "x": 30.0 * (np.arange(nx) + 0.5)}).rio.write_crs("EPSG:32632")
    with pytest.raises(ValueError, match="square"):
        tf.cost.energy_landscape(da, 50)
    # within 1 cm: accepted, with the size in x
    z2 = _dem(z, 0.0, 15 * 30.005, 30.0)
    assert np.isfinite(tf.cost.energy_landscape(z2, 50).values[1:-1, 1:-1]).all()
