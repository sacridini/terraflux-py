# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate golden outputs of R's move2, amt and move2utils for tf.move.

Usage::

    python tools/make_reference_move.py [--case NAME ...]

Draws the track sets of ``tests/move_cases.py``, runs the cases through
``tools/move_reference.R`` (with ``tools/move2_fixed.R``,
``tools/amt_fixed.R`` and ``tools/move2utils_fixed.R``, the packages' corrected
functions) and stores the inputs in ``tests/data/reference/move/inputs.npz``
(those of move2 and amt; the sets of move2utils, in
``inputs_move2utils.npz``; the steps of extract_covariates_along, in ``inputs_along.npz``; those of amt's
home ranges in ``inputs_hr.npz``; those of the step selection cases, in ``inputs_issf.npz``) and the results in
``tests/data/reference/move/<group>.npz`` (``move2.npz``, ``amt.npz``,
``move2utils.npz``, ``amt_along.npz``, ``amt_hr.npz``, ``amt_issf.npz``; keys ``<case>/<key>``).
``--case`` regenerates only the named cases (the others are kept).
The along sets are drawn with tf.move.random_steps, so terraflux must be importable.

Needs Rscript (``TERRAFLUX_RSCRIPT`` as in ``tests/lsm_reference.py``) and the
library ``%LOCALAPPDATA%/R/landschaft-move-lib`` (``TERRAFLUX_MOVE_R_LIBS``) with
survival 3.7-0, move2 0.5.0, sf, s2, lwgeom, units, amt 0.3.1, fitdistrplus 1.2-6, circular
0.5-2, geosphere 1.6-8, suncalc 0.5.1, terra and move2utils 0.4.6 (commit b3304d6). Runs R with ``TZ=UTC`` (suncalc
takes the noon of the machine's time zone). The test-suite reads the stored files,
so it runs anywhere without R.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from terraflux import _env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import lsm_reference
import move_cases
import move_issf_cases

OUT = ROOT / "tests" / "data" / "reference" / "move"
R_SCRIPT = ROOT / "tools" / "move_reference.R"
DEFAULT_LIBS = Path(os.environ.get("LOCALAPPDATA", "")) / "R" / "landschaft-move-lib"


def r_env() -> dict[str, str]:
    env = lsm_reference.r_environment()
    libs = [_env.get("MOVE_R_LIBS", str(DEFAULT_LIBS))]
    user = sorted(DEFAULT_LIBS.parent.glob("win-library/*"))
    libs += [str(p) for p in user[-1:]]
    env["R_LIBS"] = os.pathsep.join([*libs, *filter(None, [env.get("R_LIBS")])])
    env["TOOLS"] = str(ROOT / "tools")
    env["OMP_NUM_THREADS"] = "1"
    env["TZ"] = "UTC"
    return env


def save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """np.savez_compressed, unless the file already holds these arrays (zip stamps the time: it would change the bytes)."""
    if path.exists():
        with np.load(path) as old:
            if set(old.files) == set(arrays) and all(
                    old[k].dtype == np.asarray(v).dtype and np.array_equal(old[k], v, equal_nan=old[k].dtype.kind == "f")
                    for k, v in arrays.items()):
                return
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()
    cases = {k: c for k, c in move_cases.CASES.items() if not args.case or k in args.case}
    rscript = lsm_reference.find_rscript()
    if rscript is None:
        sys.exit("Rscript not found: set TERRAFLUX_RSCRIPT")
    along = move_cases.along_track_sets()
    hr_sets = move_cases.hr_track_sets()
    issf_sets = move_issf_cases.issf_track_sets()
    sets = {**move_cases.track_sets(), **move_cases.m2u_track_sets(), **along,
            **move_cases.corridor_track_sets(), **move_cases.ud_track_sets(), **hr_sets, **issf_sets}
    inputs: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="move_ref_") as tmp:
        tmp = Path(tmp)
        job_sets = {}
        for name, s in sets.items():
            d = tmp / name
            d.mkdir()
            extra = [k for k in s if k not in ("x", "y", "t", "track", "crs")]
            for key in ("x", "y", "t", *extra):
                np.asarray(s[key], "<f8").tofile(d / f"{key}.bin")
                inputs[f"{name}/{key}"] = np.asarray(s[key], np.float64)
            inputs[f"{name}/track"] = np.asarray(s["track"]).astype("U")
            inputs[f"{name}/crs"] = np.array(s["crs"] or "")
            job_sets[name] = {"dir": d.as_posix(), "n": len(s["x"]), "crs": s["crs"],
                              "track": [str(v) for v in s["track"]], "extra": extra}
        job = {"sets": job_sets, "cases": [{"name": k, "data": c.data, "code": c.r} for k, c in cases.items()]}
        (tmp / "job.json").write_text(json.dumps(job), encoding="utf-8")
        out = tmp / "out"
        out.mkdir()
        subprocess.run([str(rscript), str(R_SCRIPT), str(tmp / "job.json"), str(out)], check=True, env=r_env())
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arrays = {f"{m['case']}/{m['key']}": np.fromfile(out / m["file"], "<f8") for m in manifest}
    OUT.mkdir(parents=True, exist_ok=True)
    for group in sorted({c.group for c in cases.values()}):
        mine = {k: v for k, v in arrays.items() if cases[k.split("/", 1)[0]].group == group}
        npz = OUT / f"{group}.npz"
        if args.case and npz.exists():
            with np.load(npz) as old:
                mine = {**{k: old[k] for k in old.files if k.split("/", 1)[0] not in cases}, **mine}
        save(npz, mine)
        print(f"{group}: {len(mine)} arrays -> {npz.relative_to(ROOT)}")
    m2u = {k: v for k, v in inputs.items() if k.split("/", 1)[0] in move_cases.m2u_track_sets()}
    alg = {k: v for k, v in inputs.items() if k.split("/", 1)[0] in along}
    cor = {k: v for k, v in inputs.items() if k.split("/", 1)[0] in move_cases.corridor_track_sets()}
    ud = {k: v for k, v in inputs.items() if k.split("/", 1)[0] in move_cases.ud_track_sets()}
    hr = {k: v for k, v in inputs.items() if k.split("/", 1)[0] in hr_sets}
    issf = {k: v for k, v in inputs.items() if k.split("/", 1)[0] in issf_sets}
    save(OUT / "inputs.npz", {k: v for k, v in inputs.items()
                              if k not in m2u and k not in alg and k not in cor and k not in ud
                              and k not in hr and k not in issf})
    save(OUT / "inputs_move2utils.npz", m2u)
    save(OUT / "inputs_along.npz", alg)
    save(OUT / "inputs_corridor.npz", cor)
    save(OUT / "inputs_move2utils_ud.npz", ud)
    save(OUT / "inputs_hr.npz", hr)
    save(OUT / "inputs_issf.npz", issf)


if __name__ == "__main__":
    main()
