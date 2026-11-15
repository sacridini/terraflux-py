# SPDX-License-Identifier: GPL-3.0-or-later
"""The compatibility package compat/landschaft (pip install landschaft): `landschaft` is terraflux."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMPAT = ROOT / "compat" / "landschaft"
SRC = COMPAT / "src"

# Each check runs in a fresh interpreter. The base environment may hold an older editable install of
# the real landschaft (a .pth line or a finder that can come before PYTHONPATH), so the code puts the
# compat source first in sys.path itself and checks, before importing, that "landschaft" is found there.
_PRELUDE = f"""
import sys, importlib.util, warnings
sys.path.insert(0, {str(SRC)!r})
sys.meta_path[:] = [f for f in sys.meta_path if "landschaft" not in type(f).__module__.lower()
                    and "landschaft" not in type(f).__name__.lower()]
_spec = importlib.util.find_spec("landschaft")
assert _spec.origin and _spec.origin.startswith({str(SRC)!r}), _spec.origin
"""


def run(code: str, *flags: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONWARNINGS")}
    env.update(OMP_NUM_THREADS="4", TERRAFLUX_THREADS="4")
    r = subprocess.run([sys.executable, *flags, "-c", _PRELUDE + code], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stderr


def test_same_modules():
    run("""
warnings.simplefilter("ignore")
import landschaft as ls, terraflux, terraflux.mspa, terraflux._core
assert ls is terraflux
assert ls.mspa is terraflux.mspa
import landschaft.circuit, landschaft.sdm
assert landschaft.circuit is terraflux.circuit and sys.modules["landschaft.sdm"] is terraflux.sdm
import landschaft._core
assert landschaft._core is terraflux._core
assert not [m for m in sys.modules if m.startswith("landschaft.") and sys.modules[m].__name__.startswith("landschaft")]
""")


def test_from_import():
    run("""
warnings.simplefilter("ignore")
import terraflux.sdm as t
name = next(n for n in dir(t) if not n.startswith("_") and callable(getattr(t, n)))
ns = {}
exec(f"from landschaft.sdm import {name} as obj", ns)
assert ns["obj"] is getattr(t, name)
from landschaft import mspa
assert mspa is terraflux.mspa
""".replace("import terraflux.sdm as t", "import terraflux, terraflux.sdm as t"))


def test_version():
    run("""
warnings.simplefilter("ignore")
import landschaft, terraflux
assert landschaft.__version__ == terraflux.__version__ != "0.0.0+unknown"
""")


def test_missing_submodule():
    run("""
warnings.simplefilter("ignore")
import landschaft
try:
    import landschaft.nonexistent_xyz
except ModuleNotFoundError:
    pass
else:
    raise AssertionError
""")


def test_warns_once():
    err = run("""
import landschaft
import landschaft.circuit
import landschaft as again
""", "-W", "always::DeprecationWarning")
    assert err.count("landschaft is now terraflux") == 1, err
    assert "<string>:" in err, err  # it points at the importing line, not at importlib or the package


def test_pickle_by_old_name():
    run("""
warnings.simplefilter("ignore")
import pickle, terraflux._jobs as j
# protocol 0 is text: the longer name does not break a length prefix
data = pickle.dumps(j.Shard, protocol=0).replace(b"terraflux._jobs", b"landschaft._jobs")
assert b"landschaft._jobs" in data
assert pickle.loads(data) is j.Shard
""")


def test_versions_agree():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+; the wheels test 3.10 too
    main = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    compat = tomllib.loads((COMPAT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert compat["name"] == "landschaft"
    assert compat["version"] == main["version"]
    assert compat["dependencies"] == [f"terraflux-py=={main['version']}"]
    assert compat["requires-python"] == main["requires-python"]
    assert compat["scripts"] == {"landschaft": "terraflux._cli:main"}
