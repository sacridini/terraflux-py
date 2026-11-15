# SPDX-License-Identifier: GPL-3.0-or-later
"""What landschaft (the name until 1.58) left behind keeps working: its environment variables, tags,
sessions, pickles, folders and batch records; and the thread rule (never all the cores)."""

from __future__ import annotations

import json
import pickle
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import terraflux as tf
from terraflux import _compat, _env, _jobs, _threads

_THREAD_VARIABLES = ["TERRAFLUX_THREADS", "LANDSCHAFT_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_JOB_ID",
                     "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_JOBID", "NCPUS", "LSB_JOBID",
                     "LSB_DJOB_NUMPROC"]


@pytest.fixture
def threads(monkeypatch):
    """A clean thread environment on a 20-processor machine, whatever this one is."""
    for v in _THREAD_VARIABLES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(_threads, "processors", lambda: (20, 20))
    monkeypatch.setattr(_threads, "_warned_cap", False)
    yield
    tf.set_default_threads(None)


# -- the thread rule ------------------------------------------------------------------------

@pytest.mark.parametrize("cores, most", [(1, 1), (2, 1), (4, 3), (8, 6), (20, 15)])
def test_most(cores, most, monkeypatch):
    assert _threads.most(cores) == most
    for v in _THREAD_VARIABLES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(_threads, "processors", lambda: (cores, cores))
    monkeypatch.setattr(_threads, "_warned_cap", True)           # (the warning has its own test)
    try:
        assert tf.default_threads() == most                       # the default
        assert tf.set_default_threads(cores) == most             # a count given for the process
        assert tf.default_threads() == most
        tf.set_default_threads(None)
        assert _threads.resolve(cores) == most                    # an explicit n_threads
        assert _threads.resolve(0) == _threads.resolve(None) == most
        assert _threads.resolve(1) == 1
    finally:
        tf.set_default_threads(None)


def test_requests_are_held_and_warn_once(threads):
    with pytest.warns(RuntimeWarning, match="never uses all the cores"):
        assert _threads.resolve(64) == 15
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _threads.resolve(64) == 15
        assert tf.set_default_threads(20) == 15
        assert tf.default_threads() == 15
        assert _threads.resolve(15) == 15
        with _threads.limit(100) as n:
            assert n == 15


def test_variables_are_held_too(threads, monkeypatch):
    monkeypatch.setenv("TERRAFLUX_THREADS", "20")
    assert tf.default_threads() == 15
    monkeypatch.setenv("TERRAFLUX_THREADS", "4")
    assert tf.default_threads() == 4
    # a scheduler's count: a quarter of the job's processors kept free
    monkeypatch.delenv("TERRAFLUX_THREADS")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert tf.default_threads() == 6


def test_threads_variable_and_the_old_name(threads, monkeypatch):
    monkeypatch.setenv("LANDSCHAFT_NUM_THREADS", "3")
    assert tf.default_threads() == 3
    assert _threads.detect().source == "LANDSCHAFT_NUM_THREADS"
    monkeypatch.setenv("TERRAFLUX_THREADS", "5")             # the new name wins
    assert tf.default_threads() == 5
    assert _threads.detect().source == "TERRAFLUX_THREADS"
    monkeypatch.delenv("TERRAFLUX_THREADS")
    assert tf.default_threads() == 3


def test_helper_children_get_terraflux_threads():
    from terraflux._dist._slots import child_env

    assert child_env(3)["TERRAFLUX_THREADS"] == "3"


# -- environment variables ------------------------------------------------------------------

def test_env_precedence(monkeypatch):
    monkeypatch.delenv("TERRAFLUX_RENAME_X", raising=False)
    monkeypatch.delenv("LANDSCHAFT_RENAME_X", raising=False)
    assert _env.get("RENAME_X") is None
    assert _env.get("RENAME_X", "dflt") == "dflt"
    monkeypatch.setenv("LANDSCHAFT_RENAME_X", "old")
    assert _env.get("RENAME_X", "dflt") == "old"              # the old name still read
    monkeypatch.setenv("TERRAFLUX_RENAME_X", "new")
    assert _env.get("RENAME_X", "dflt") == "new"              # the new one wins
    monkeypatch.setenv("TERRAFLUX_RENAME_X", "")
    assert _env.get("RENAME_X", "dflt") == ""                 # set (even empty) is set


def test_plot_switches_read_both_names(monkeypatch):
    from terraflux._plotting import _live

    monkeypatch.delenv("TERRAFLUX_PLOT_LIVE", raising=False)
    monkeypatch.setenv("LANDSCHAFT_PLOT_LIVE", "0")
    assert _live.enabled() is False
    monkeypatch.setenv("TERRAFLUX_PLOT_LIVE", "1")
    assert _live.enabled() is True


# -- files of landschaft --------------------------------------------------------------------

def _tif(path: Path, tags: dict) -> Path:
    data = np.array([[1, 2, 2], [1, 3, 3], [1, 1, 2]], dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=3, width=3, count=1, dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(500000, 4000000, 30, 30), nodata=0) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)
    return path


def test_geotiff_with_a_landschaft_tag(tmp_path):
    path = _tif(tmp_path / "old.tif", {"landschaft_product": "MSPA", "mspa_transition": "1", "mspa_connectivity": "8",
                                      "mspa_edge_width": "1", "mspa_internal": "1",
                                      "landschaft_other": "x"})
    da = tf.load_raster(path)
    assert da.attrs["terraflux_product"] == "MSPA"
    assert da.attrs["terraflux_other"] == "x"
    assert not [k for k in da.attrs if k.startswith("landschaft_")]
    # what consumes the product recognises it: the plot's colours and the colour table written back
    from terraflux import io
    from terraflux._plotting._palettes import product_style

    table, _ = product_style(da.attrs)
    assert table
    assert io._resolve_colormap("auto", da)
    # a file of 1.59 and later carries the new tag, which wins over an old one
    new = _tif(tmp_path / "new.tif", {"terraflux_product": "MSPA", "landschaft_product": "FAD_CLASSES"})
    assert tf.load_raster(new).attrs["terraflux_product"] == "MSPA"
    # and what is written now carries terraflux_*
    out = tmp_path / "out.tif"
    tf.write_raster(da, out)
    with rasterio.open(out) as src:
        tags = src.tags()
    assert tags["terraflux_product"] == "MSPA" and "landschaft_product" not in tags


def _site_bd():
    from bio_cases import BASEDATA_CASES, DATA

    case = BASEDATA_CASES["site_counts"]
    kw = {k: v for k, v in case["terraflux"].items() if v is not None}
    return tf.bio.BaseData.from_table(DATA / case["file"], **kw)


def _rewrite_format(src: Path, dst: Path, old_to_new: tuple[str, str]) -> None:
    """Copy a zip, naming the format in its JSON headers as landschaft did."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".json"):
                header = json.loads(data)
                assert header["format"] == old_to_new[0]
                header["format"] = old_to_new[1]
                data = json.dumps(header).encode()
            zout.writestr(item, data)


def test_bio_files_of_landschaft_load(tmp_path):
    pytest.importorskip("pyarrow")
    from terraflux.bio import _session

    assert _session.FORMAT == "terraflux-bio"
    bd = _site_bd()
    new = tmp_path / "new.lsbio"
    bd.save(new)
    old = tmp_path / "old.lsbio"
    _rewrite_format(new, old, ("terraflux-bio", "landschaft-bio"))
    back = tf.bio.BaseData.load(old)
    assert back.groups == bd.groups and back.labels == bd.labels
    assert (back.counts != bd.counts).nnz == 0
    # a session too
    session = tmp_path / "s.lsbio"
    tf.bio.save_session(session, sites=bd)
    old_session = tmp_path / "s_old.lsbio"
    _rewrite_format(session, old_session, ("terraflux-bio", "landschaft-bio"))
    assert tf.bio.load_session(old_session)["sites"].groups == bd.groups
    # something else is refused
    bad = tmp_path / "bad.lsbio"
    _rewrite_format(new, bad, ("terraflux-bio", "something-else"))
    with pytest.raises(ValueError, match="terraflux basedata"):
        tf.bio.BaseData.load(bad)


# -- pickles ---------------------------------------------------------------------------------

def test_pickle_of_a_landschaft_class_loads():
    value = _threads.Threads(3, 4, "TERRAFLUX_THREADS", 8, 8, None, None, 8, False)
    for protocol in (0, 2):
        data = pickle.dumps(value, protocol=protocol)
        assert b"terraflux._threads" in data
        old = data.replace(b"terraflux._threads", b"landschaft._threads")
        assert b"landschaft._threads" in old
        with pytest.raises(ImportError):
            pickle.loads(old)                         # the plain loader does not know it
        assert _compat.loads(old) == value
        import io

        assert _compat.load(io.BytesIO(old)) == value
    assert _compat.module_name("landschaft") == "terraflux"
    assert _compat.module_name("landschaft.sdm._models") == "terraflux.sdm._models"
    assert _compat.module_name("landschaftx.y") == "landschaftx.y"
    assert _compat.module_name("numpy.core") == "numpy.core"


def test_old_module_names_travel(tmp_path):
    from terraflux._dist import _pack
    from terraflux._dist._ops import build_estimator
    from terraflux.conn.nodes import Resistance

    # a task written by a helper of landschaft names its classes landschaft.*
    fields = {"values": 1.5, "extent": {"$tuple": [0.0, 1.0, 2.0, 3.0]}}
    spec = {"$dataclass": "landschaft.conn.nodes:Resistance", "fields": fields}
    value = _pack.decode(spec, tmp_path)
    assert isinstance(value, Resistance) and value.extent == (0.0, 1.0, 2.0, 3.0)
    est = build_estimator({"$estimator": "landschaft.sdm._maxnet:MaxNet", "params": {"regmult": 2.0}})
    assert type(est).__module__ == "terraflux.sdm._maxnet" and est.get_params()["regmult"] == 2.0
    # only terraflux's own classes, by either name
    with pytest.raises(ValueError, match="refusing"):
        _pack.decode({"$dataclass": "landschaftx.conn.nodes:Resistance", "fields": {}}, tmp_path)


# -- cache and configuration folders ----------------------------------------------------------

def test_migrate(tmp_path):
    old, new = tmp_path / "landschaft", tmp_path / "terraflux"
    # nothing at all: the new one
    assert _env.migrate(old, new) == new and not new.exists()
    # only the old one: moved, data kept
    (old / "sub").mkdir(parents=True)
    (old / "sub" / "data.bin").write_bytes(b"data")
    assert _env.migrate(old, new) == new
    assert not old.exists() and (new / "sub" / "data.bin").read_bytes() == b"data"
    # both: the new one, the old one untouched
    (old / "sub").mkdir(parents=True)
    (old / "other.txt").write_text("old")
    (new / "mine.txt").write_text("new")
    assert _env.migrate(old, new) == new
    assert (old / "other.txt").read_text() == "old" and not (new / "other.txt").exists()


def test_migrate_failure_uses_the_old_folder(tmp_path, monkeypatch):
    old, new = tmp_path / "landschaft", tmp_path / "terraflux"
    old.mkdir()
    (old / "keep.txt").write_text("kept")

    def refuse(self, target):
        raise PermissionError("no")

    monkeypatch.setattr(Path, "rename", refuse)
    assert _env.migrate(old, new) == old
    assert (old / "keep.txt").read_text() == "kept" and not new.exists()


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Cache and configuration roots of this platform inside tmp_path."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    for v in ("TERRAFLUX_CACHE", "LANDSCHAFT_CACHE", "TERRAFLUX_CONFIG_DIR", "LANDSCHAFT_CONFIG_DIR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg_cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    if sys.platform == "win32":
        return {"cache_old": tmp_path / "local" / "landschaft" / "cache",
                "cache_new": tmp_path / "local" / "terraflux" / "python" / "cache",
                "config_old": tmp_path / "roaming" / "landschaft", "config_new": tmp_path / "roaming" / "terraflux" / "python"}
    if sys.platform == "darwin":
        return {"cache_old": home / "Library" / "Caches" / "landschaft",
                "cache_new": home / "Library" / "Caches" / "terraflux" / "python",
                "config_old": tmp_path / "xdg_config" / "landschaft",
                "config_new": tmp_path / "xdg_config" / "terraflux" / "python"}
    return {"cache_old": tmp_path / "xdg_cache" / "landschaft", "cache_new": tmp_path / "xdg_cache" / "terraflux" / "python",
            "config_old": tmp_path / "xdg_config" / "landschaft", "config_new": tmp_path / "xdg_config" / "terraflux" / "python"}


def test_cache_of_landschaft_moves(homes):
    from terraflux import _download

    old, new = homes["cache_old"], homes["cache_new"]
    (old / "sdm").mkdir(parents=True)
    (old / "sdm" / "thing.zip").write_bytes(b"zip")
    assert _download.cache_dir() == new
    assert (new / "sdm" / "thing.zip").read_bytes() == b"zip" and not old.exists()


def test_cache_both_exist_uses_the_new_one(homes):
    from terraflux import _download

    old, new = homes["cache_old"], homes["cache_new"]
    old.mkdir(parents=True)
    (old / "old.bin").write_bytes(b"old")
    new.mkdir(parents=True)
    (new / "new.bin").write_bytes(b"new")
    assert _download.cache_dir() == new
    assert (new / "new.bin").read_bytes() == b"new"
    assert (old / "old.bin").exists() or (new / "old.bin").exists()      # never deleted


def test_cache_variable_names(homes, monkeypatch, tmp_path):
    from terraflux import _download

    monkeypatch.setenv("LANDSCHAFT_CACHE", str(tmp_path / "from_old_name"))
    assert _download.cache_dir() == tmp_path / "from_old_name"
    monkeypatch.setenv("TERRAFLUX_CACHE", str(tmp_path / "from_new_name"))
    assert _download.cache_dir() == tmp_path / "from_new_name"


def test_config_of_landschaft_moves(homes, monkeypatch, tmp_path):
    from terraflux._dist import _group

    old, new = homes["config_old"], homes["config_new"]
    old.mkdir(parents=True)
    (old / "group.json").write_text(json.dumps({"secret": "ab" * 16}))
    assert _group.config_dir() == new
    assert _group.load_secret() == bytes.fromhex("ab" * 16)             # the pairing goes on working
    assert not (old / "group.json").exists()
    monkeypatch.setenv("LANDSCHAFT_CONFIG_DIR", str(tmp_path / "elsewhere"))
    assert _group.config_dir() == tmp_path / "elsewhere"
    monkeypatch.setenv("TERRAFLUX_CONFIG_DIR", str(tmp_path / "elsewhere2"))
    assert _group.config_dir() == tmp_path / "elsewhere2"


def test_user_agent():
    from terraflux import _download

    agent = _download.user_agent()
    assert agent.startswith(f"terraflux/{tf.__version__} ") and "github.com/sacridini/terraflux-py" in agent


# -- batch records ---------------------------------------------------------------------------

def test_old_batch_record_is_picked_up(tmp_path):
    out = tmp_path / "out"
    old = out / ".landschaft-batch"
    old.mkdir(parents=True)
    (old / _jobs.SPLIT).write_text(json.dumps({"count": 4, "first": 1, "step": 2}))
    (old / _jobs.MANIFEST).write_text(json.dumps({"tool": "fad", "inputs": ["a.tif"], "markers": ["a.json"]}))
    assert _jobs.STATE == ".terraflux-batch"
    split = _jobs.read_split(out)
    assert split == {"count": 4, "first": 1, "step": 2}
    # renamed to the new name, the record intact
    assert not old.exists() and (out / ".terraflux-batch" / _jobs.MANIFEST).exists()
    assert _jobs.batch_status(out).tool == "fad"


def test_old_batch_record_read_in_place_when_it_cannot_move(tmp_path, monkeypatch):
    out = tmp_path / "out"
    old = out / ".landschaft-batch"
    old.mkdir(parents=True)
    (old / _jobs.SPLIT).write_text(json.dumps({"count": 2, "first": 0, "step": 1}))

    def refuse(self, target):
        raise PermissionError("no")

    monkeypatch.setattr(Path, "rename", refuse)
    assert _jobs.state_dir(out) == old
    assert _jobs.read_split(out) == {"count": 2, "first": 0, "step": 1}


def test_new_batch_record_wins(tmp_path):
    out = tmp_path / "out"
    (out / ".landschaft-batch").mkdir(parents=True)
    (out / ".terraflux-batch").mkdir()
    assert _jobs.state_dir(out) == out / ".terraflux-batch"
    assert (out / ".landschaft-batch").exists()
