# SPDX-License-Identifier: GPL-3.0-or-later
"""Threads of the parallel code: never all the cores (a quarter kept free),
configurable per call, for the process and by variables."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest

import terraflux as tf
from terraflux import _core, _threads


_VARIABLES = ["TERRAFLUX_THREADS", "LANDSCHAFT_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK",
              "SLURM_CPUS_ON_NODE", "PBS_JOBID", "NCPUS", "LSB_JOBID", "LSB_DJOB_NUMPROC"]


@pytest.fixture(autouse=True)
def _reset_default(monkeypatch):
    for v in _VARIABLES:
        monkeypatch.delenv(v, raising=False)
    yield
    tf.set_default_threads(None)


@pytest.fixture
def cores20(monkeypatch):
    """A machine of 20 processors, whatever this one is."""
    monkeypatch.setattr(_threads, "processors", lambda: (20, 20))


def test_default_leaves_a_quarter_free():
    assert tf.default_threads() == _threads.most(_threads.available_cores())
    if _core.openmp:
        assert _core.team_size() == tf.default_threads()


def test_limit_and_process_default(cores20):
    with _threads.limit(2):
        assert _core.team_size() == (2 if _core.openmp else 1)
        with _threads.limit(1):
            assert _core.team_size() == 1
        assert _core.team_size() == (2 if _core.openmp else 1)
    assert tf.set_default_threads(3) == 3
    assert tf.default_threads() == 3
    assert _threads.resolve(None) == _threads.resolve(0) == 3
    assert _threads.resolve(5) == 5
    if _core.openmp:
        assert _core.team_size() == 3
        # other Python threads follow the same rule
        seen = []
        t = threading.Thread(target=lambda: seen.append(_core.team_size()))
        t.start()
        t.join()
        assert seen == [3]
    tf.set_default_threads(None)
    assert tf.default_threads() == _threads.detect().threads
    with pytest.raises(ValueError):
        tf.set_default_threads(0)
    with pytest.raises(ValueError):
        _threads.resolve(-1)


def test_parallel_functions_take_n_threads():
    for fn in (tf.fad, tf.fad_multiscale, tf.landscape_mosaic, tf.spatial_entropy, tf.euclidean_distance,
               tf.parcellation, tf.morphological_change, tf.fos_change, tf.distance_to_resistance,
               tf.isochrone_map_ab, tf.raster_boundaries, tf.mspa, tf.accounting, tf.spa, tf.fos,
               tf.landscape_metrics, tf.sample_metrics, tf.window_metrics, tf.omniscape,
               tf.conn.protconn_raster, tf.conn.settlement_probabilities, tf.circuit.circuitscape_pairwise,
               tf.circuit.circuitscape_network_pairwise):
        assert "n_threads" in inspect.signature(fn).parameters, fn.__name__


def test_results_do_not_depend_on_the_threads():
    rng = np.random.default_rng(4)
    fg = rng.choice(np.array([1, 2], dtype=np.uint8), size=(120, 150), p=[0.4, 0.6])
    a = tf.fad(fg, n_threads=1)
    b = tf.fad(fg)
    c = tf.fad(fg, threads=3)  # the older name
    np.testing.assert_array_equal(np.asarray(a.density), np.asarray(b.density))
    np.testing.assert_array_equal(np.asarray(a.classes), np.asarray(c.classes))
    classes = rng.integers(1, 4, size=(80, 90))
    np.testing.assert_array_equal(tf.raster_boundaries(classes, n_threads=1), tf.raster_boundaries(classes))


def test_landscape_stores_a_count():
    from terraflux.lsm.landscape import core_options

    assert core_options({}, None)["n_threads"] == tf.default_threads()
    assert core_options({}, 2)["n_threads"] == 2


def test_batch_workers_default():
    from terraflux.batch import run_parallel

    src = inspect.getsource(run_parallel)
    assert "default_threads()" in src


def _decide(env: dict, available: int = 128, machine: int = 128, **kw):
    return _threads.decide(env.get, available, machine, **kw)


def test_rule_desktop_and_jobs():
    # a desktop: a quarter of the processors kept free
    d = _decide({}, 20, 20)
    assert (d.threads, d.source, d.reserved, d.job) == (15, None, True, None)
    assert d.describe() == "15 threads (20 processors; a quarter kept free)"
    # confined to fewer processors than the machine has (cpuset, taskset): a quarter of those
    assert _decide({}, 8, 20).threads == 6
    # a Slurm job confined by its cgroup: a quarter of its processors kept free
    d = _decide({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "8"}, 8, 128)
    assert (d.threads, d.source, d.reserved, d.scheduler) == (6, None, True, "Slurm")
    assert d.describe() == "6 threads (8 of 128 processors available to this process; Slurm job 123; a quarter kept free)"
    # a job that sees the whole node (no cgroup): a quarter less than the job's processors
    d = _decide({"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "8"})
    assert (d.threads, d.source) == (6, "SLURM_CPUS_PER_TASK")
    # --ntasks without --cpus-per-task: the job's processors on this node
    d = _decide({"SLURM_JOB_ID": "1", "SLURM_CPUS_ON_NODE": "16"})
    assert (d.threads, d.source) == (12, "SLURM_CPUS_ON_NODE")
    # the whole node allocated: still not every processor
    assert _decide({"SLURM_JOB_ID": "1", "SLURM_CPUS_ON_NODE": "128"}).threads == 96
    # PBS and LSF
    assert _decide({"PBS_JOBID": "9.pbs", "NCPUS": "12"}).threads == 9
    assert _decide({"LSB_JOBID": "77", "LSB_DJOB_NUMPROC": "6"}).threads == 5
    # the variables of a scheduler count only inside its job
    assert _decide({"NCPUS": "12"}).threads == 96
    assert _decide({"SLURM_CPUS_PER_TASK": "8"}).threads == 96


@pytest.mark.parametrize("cores, most", [(1, 1), (2, 1), (3, 2), (4, 3), (8, 6), (16, 12), (20, 15), (128, 96)])
def test_most(cores, most):
    assert _threads.most(cores) == most


def test_rule_order():
    job = {"SLURM_JOB_ID": "5", "SLURM_CPUS_PER_TASK": "16"}
    assert _decide({**job, "OMP_NUM_THREADS": "4"}).threads == 4
    assert _decide({**job, "OMP_NUM_THREADS": "4", "TERRAFLUX_THREADS": "10"}).threads == 10
    assert _decide({**job, "OMP_NUM_THREADS": "4", "LANDSCHAFT_NUM_THREADS": "10"}).threads == 10   # the old name
    d = _decide({"TERRAFLUX_THREADS": "7", "LANDSCHAFT_NUM_THREADS": "9"})
    assert (d.threads, d.source) == (7, "TERRAFLUX_THREADS")
    # a dask worker's threads: a quarter of them kept free too
    assert _decide({**job, "OMP_NUM_THREADS": "1"}, worker=6).threads == 5
    assert _decide({**job, "TERRAFLUX_THREADS": "3"}, worker=6).threads == 3
    assert _decide({**job, "TERRAFLUX_THREADS": "3"}, fixed=2).threads == 2
    # every count is held to the most of the processors
    assert _decide({"OMP_NUM_THREADS": "20"}, 20, 20).threads == 15
    assert _decide({"TERRAFLUX_THREADS": "99"}, 8, 8).threads == 6
    assert _decide({}, 8, 8, fixed=8).threads == 6
    # OpenMP's nested levels, Slurm's repeat counts, junk
    assert _decide({"OMP_NUM_THREADS": "8,2"}).threads == 8
    assert _decide({"SLURM_JOB_ID": "1", "SLURM_CPUS_ON_NODE": "4(x2)"}).threads == 3
    assert _decide({"OMP_NUM_THREADS": "many", "TERRAFLUX_THREADS": "0"}, 20, 20).threads == 15


def test_variables_reach_default_threads(monkeypatch, cores20):
    monkeypatch.setenv("TERRAFLUX_THREADS", "3")
    assert tf.default_threads() == 3
    if _core.openmp:
        assert _core.team_size() == 3
    assert tf.set_default_threads(2) == 2
    tf.set_default_threads(None)
    monkeypatch.delenv("TERRAFLUX_THREADS")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert tf.default_threads() == 1
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert tf.default_threads() == _threads.most(_threads.available_cores())   # in a job, a quarter kept free


def test_warns_once_for_a_low_omp_num_threads_in_a_job(monkeypatch):
    available = _threads.available_cores()
    if available < 2:
        pytest.skip("one processor")
    monkeypatch.setattr(_threads, "_warned", False)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.warns(RuntimeWarning, match=f"TERRAFLUX_THREADS={available}"):
        assert tf.default_threads() == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tf.default_threads() == 1
    # outside a job, no warning: OMP_NUM_THREADS is what the user asked for
    monkeypatch.setattr(_threads, "_warned", False)
    monkeypatch.delenv("SLURM_JOB_ID")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tf.default_threads() == 1


def test_dask_worker_threads(monkeypatch, cores20):
    class State:
        nthreads = 5

    class Worker:
        state = State()

    class Distributed:
        @staticmethod
        def get_worker():
            return Worker()

    monkeypatch.setitem(sys.modules, "distributed", Distributed)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # what the nanny sets
    assert tf.default_threads() == 4   # 5 threads, a quarter (at least one) kept free

    class Outside:
        @staticmethod
        def get_worker():
            raise ValueError("no worker found")

    monkeypatch.setitem(sys.modules, "distributed", Outside)
    assert tf.default_threads() == 1


def test_cgroup_quota(tmp_path):
    # cgroup v2: the smallest quota of the group and its parents
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("0::/system.slice/job_7/step_0\n")
    base = tmp_path / "sys/fs/cgroup"
    (base / "system.slice/job_7/step_0").mkdir(parents=True)
    (base / "system.slice/job_7/step_0/cpu.max").write_text("max 100000\n")
    (base / "system.slice/job_7/cpu.max").write_text("250000 100000\n")
    (base / "system.slice/cpu.max").write_text("800000 100000\n")
    assert _threads._cgroup_quota(tmp_path) == 3
    (base / "system.slice/job_7/cpu.max").write_text("max 100000\n")
    assert _threads._cgroup_quota(tmp_path) == 8
    (base / "system.slice/cpu.max").unlink()
    assert _threads._cgroup_quota(tmp_path) is None
    # cgroup v1
    v1 = tmp_path / "v1"
    (v1 / "proc/self").mkdir(parents=True)
    (v1 / "proc/self/cgroup").write_text("4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n")
    cpu = v1 / "sys/fs/cgroup/cpu,cpuacct"
    cpu.mkdir(parents=True)  # inside the container, the group is the mount's root
    (cpu / "cpu.cfs_quota_us").write_text("150000\n")
    (cpu / "cpu.cfs_period_us").write_text("100000\n")
    assert _threads._cgroup_quota(v1) == 2
    (cpu / "cpu.cfs_quota_us").write_text("-1\n")
    assert _threads._cgroup_quota(v1) is None
    assert _threads._cgroup_quota(tmp_path / "nothing") is None


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="Linux affinity")
def test_affinity_in_a_process():
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 3 or len(cpus) < (os.cpu_count() or 1):
        pytest.skip("fewer than 3 processors, or already confined")
    code = ("import os; os.sched_setaffinity(0, {%d, %d}); import terraflux as tf, terraflux._core as c; "
            "print(tf.default_threads(), c.team_size() if c.openmp else 2)" % (cpus[0], cpus[1]))
    env = {k: v for k, v in os.environ.items() if k not in _VARIABLES}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    # confined to 2 of the machine's processors: one of them (a quarter, at least one, kept free)
    assert out.stdout.split() == ["1", "1"]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**env, "SLURM_JOB_ID": "9", "SLURM_CPUS_PER_TASK": "1"})
    assert out.stdout.split()[0] == "1"


def test_helper_children_get_the_threads():
    from terraflux._dist._slots import child_env

    env = child_env(3)
    assert env["TERRAFLUX_THREADS"] == env["OMP_NUM_THREADS"] == "3"
