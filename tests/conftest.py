# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from gtb_reference import find_executable
from rasterio.errors import NotGeoreferencedWarning

from terraflux import _core

DATA = Path(__file__).parent / "data"

# tf.plot must not open windows while the tests run (nor matplotlib's: static figures)
os.environ.setdefault("TERRAFLUX_PLOT_SHOW", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

# The references of R (landscapemetrics, NLMR) and Perl (Biodiverse) come from x86-64,
# where log, sqrt, exp, x^y, sin, cos and atan2 run on the x87 unit; x86 builds do the
# same (lsm/rmath.hpp) and must give their bits. Other builds (ARM64) take these
# functions from the system library, which can differ in the last bits: there the
# comparisons allow a few units in the last place.
BITWISE = bool(_core.x87_math)

# The functions of the C library that the references take from the UCRT of Windows
# (atan, atanh...) and that terraflux calls as they are: other C libraries (glibc)
# give some of them other last bits, and the UCRT's code is not public, so off
# Windows the comparisons that go through them allow one unit in the last place.
UCRT_LIBM = sys.platform == "win32"

# References made by Python packages that call numpy's matrix products (MOLUSCE's
# correlation: np.corrcoef, a dgemm) carry the summation order of the BLAS kernel
# numpy used on Windows, where they were made; the same inputs, means and dot
# products give other last bits through another build's dgemm, so off Windows those
# values allow a few units in the last place.
WINDOWS_BLAS = sys.platform == "win32"

# The x87 functions themselves (fsin, fcos, fpatan, fyl2x, f2xm1) are microcoded, and the references of move2utils / amt
# were made on an Intel CPU on Windows: the likelihoods and the distributions of tf.move carry the last bits of those
# functions through an optimiser and long sums, so they are R's bits only there. On the CI runners (AMD x86 on
# Windows and Linux: the same numbers on both, not the references') they differ by up to 3.7e-8 of a variance
# and 2.7e-10 of a distribution, and without the x87 unit (ARM64) by up to 2.7e-7 and 3.7e-9: those comparisons
# allow the tolerances measured there (tests/test_move_ud.py, test_move_dbgb.py, test_move_corridor.py).
X87_EXACT = BITWISE and UCRT_LIBM and "GenuineIntel" in platform.processor()

# Off Windows, results that go through the C library's exp, pow, log10 or atan in a
# chain of float32 rasters (the InVEST models, D-infinity) can differ from references
# made on Windows by a unit in the last place of one term, which the chain carries on.
UCRT_RTOL = 0.0 if UCRT_LIBM else 1e-5


def assert_ucrt_close(a, b, err_msg: str = "") -> None:
    """``a`` equals ``b`` (NaN matching NaN) on Windows, within ``UCRT_RTOL`` elsewhere."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind == "f":
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=err_msg)
    if UCRT_LIBM:
        np.testing.assert_array_equal(a, b, err_msg=err_msg)
    else:
        np.testing.assert_allclose(a, b, rtol=UCRT_RTOL, atol=0, equal_nan=True, err_msg=err_msg)


# SWAT+ simulations carry the last bits of exp, log, pow... through years of daily
# state (soil water, plants, storages): off Windows the libm's other last bits move a
# value by up to ~1e-5 of its column's scale (a daily series of one quantity).
SWAT_TOL = 1e-4


def assert_swat_close(a, b, err_msg: str = "") -> None:
    """SWAT+ dump records ``a`` (rows = days and objects, columns = fields) equal ``b``
    on Windows; elsewhere ``|a - b| <= SWAT_TOL * (|b| + max |b| of the column)``."""
    a = np.asarray(a)
    b = np.asarray(b)
    if UCRT_LIBM:
        assert_ucrt_close(a, b, err_msg=err_msg)
        return
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=err_msg)
    a64 = np.nan_to_num(a.astype(np.float64))
    b64 = np.nan_to_num(b.astype(np.float64))
    scale = np.abs(b64).max(axis=0) if b64.ndim == 2 and len(b64) else np.abs(b64).max(initial=0.0)
    bad = np.abs(a64 - b64) > SWAT_TOL * (np.abs(b64) + scale)
    assert not bad.any(), f"{err_msg}: {int(bad.sum())} values beyond SWAT_TOL, first at {np.argwhere(bad)[0]}"


def same_bits(a, b, *, scale=None, ulps: float = 4) -> np.ndarray:
    """Element by element: ``a`` has the bits of ``b`` (NaN matching NaN).

    Without the x87 functions (see ``BITWISE``): within ``ulps`` units in the last
    place of ``b``, or of ``scale`` for results that add up many terms (the
    magnitude of the result, e.g. the largest value of a transform).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    if BITWISE:
        return same
    size = np.abs(b) if scale is None else np.maximum(np.abs(b), scale)
    with np.errstate(invalid="ignore"):
        return same | (np.abs(a - b) <= ulps * np.spacing(size))


def assert_same_bits(a, b, label: str = "", *, scale=None, ulps: float = 4) -> None:
    """Assert :func:`same_bits` for every element, naming the first differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{label}: shapes {a.shape} and {b.shape}"
    ok = same_bits(a, b, scale=scale, ulps=ulps)
    if not ok.all():
        bad = np.argwhere(~ok)[:5].tolist()
        raise AssertionError(f"{label}: {int((~ok).sum())} of {ok.size} values differ "
                             f"({'bit for bit' if BITWISE else f'beyond {ulps} ulp'}), first at {bad}: "
                             f"{a[~ok][:3].tolist()} != {b[~ok][:3].tolist()}")


@pytest.fixture(autouse=True)
def _quiet_georef_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        yield


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    return DATA / "reference"


@pytest.fixture(scope="session")
def mspa_exe() -> Path:
    exe = find_executable()
    if exe is None:
        pytest.skip("set TERRAFLUX_MSPA_EXE to the original GTB mspa executable to run live comparisons")
    return exe


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def mspa_input(shape, foreground=(), missing=()) -> np.ndarray:
    """Background image with the given foreground/missing slices set."""
    a = np.ones(shape, dtype=np.uint8)
    for sl in foreground:
        a[sl] = 2
    for sl in missing:
        a[sl] = 0
    return a
