# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases of tf.move's step selection (fit_issf, log_rss) against R's amt 0.3.1 and survival 3.7-0 (group "amt_issf").

The tables are drawn here once and stored with the references (tests/data/reference/move/inputs_issf.npz), as the
track sets of move_cases.py; ``tools/move_reference.R`` rebuilds them as data frames (``step_id_`` from ``x``, the
text of the strata ``step_txt`` from ``track``, the other columns as they are). ``make_cases(Case)`` gives the cases
for ``move_cases.CASES`` (the hook at the end of that file).
"""

from __future__ import annotations

import numpy as np

LEVELS = ["70", "0", "30", "50", "210"]   # the levels of the factor column ``lu``, the reference first
LANDUSE = [0, 30, 50, 70, 210]

# the points log_rss compares: x1 (several rows) against x2 (one)
LR_A = {"x1": {"elevation": [50.0, 100.0, 150.0, 200.5], "sl_": [100.0, 20.0, 300.0, 55.5],
               "ta_": [0.0, 1.5, -2.0, 0.3]},
        "x2": {"elevation": [100.0], "sl_": [100.0], "ta_": [0.0]}}
LR_FAC = {"x1": {"lu": ["30", "0", "210"], "elevation": [90.0, 100.0, 120.0], "sl_": [50.0, 60.0, 70.0]},
          "x2": {"lu": ["70"], "elevation": [100.0], "sl_": [60.0]}}
LR_COL = {"x1": {"elevation": [60.0, 120.0], "elev2": [120.0, 240.0], "sl_": [10.0, 20.0]},
          "x2": {"elevation": [100.0], "elev2": [200.0], "sl_": [15.0]}}
LR_BOOT = {"x1": {"elevation": [50.0, 150.0], "sl_": [100.0, 30.0]}, "x2": {"elevation": [100.0], "sl_": [100.0]}}
LR_BOOT1 = {"x1": {"elevation": [140.0], "sl_": [25.0]}, "x2": {"elevation": [100.0], "sl_": [100.0]}}
BOOT_A = {"seed": 123, "n_boot": 40, "ci_level": 0.9}
BOOT_B = {"seed": 7, "n_boot": 25, "ci_level": 0.95}

F_A = "case_ ~ elevation + log(sl_) + cos(ta_) + strata(step_id_)"
F_B = "case_ ~ sl_ + log(sl_) + factor(landuse) + elevation:log(sl_) + I(elevation^2) + strata(step_id_)"
F_FAC = "case_ ~ lu + elevation + lu:elevation + sl_ + strata(step_id_)"
F_TIES_TXT = "case_ ~ elevation + sl_ + strata(step_txt)"
F_TIES_NUM = "case_ ~ elevation + sl_ + strata(step_id_)"
F_COL = "case_ ~ elevation + elev2 + log(sl_) + strata(step_id_)"
F_ONE = "case_ ~ elevation + k1 + strata(step_id_)"
F_BOOT = "case_ ~ elevation + sl_ + strata(step_id_)"
F_BOOT_LOG = "case_ ~ elevation + log(sl_) + strata(step_id_)"


def _f32(v):
    return np.asarray(v, np.float32).astype(np.float64)   # a raster's values, as amt extracts them


def issf_track_sets() -> dict[str, dict]:
    """name -> {x (step_id_), y, t, track (the text of the strata), crs, case_, elevation, sl_, ta_, landuse}."""
    rng = np.random.default_rng(20261123)
    sets = {}
    # issf_a: 140 steps with 10 random steps each, the ids not in order, two rows with no elevation (one a case)
    ns, nc = 140, 10
    n = ns * (nc + 1)
    case = np.zeros(n)
    case[::nc + 1] = 1
    ids = rng.permutation(np.arange(3, 3 + 7 * ns, 7)).astype(np.float64)
    step = np.repeat(ids, nc + 1)
    elev = _f32(rng.normal(100, 25, n) + 6 * case)
    sl = rng.gamma(1.5, 40, n) * (1 - 0.15 * case) + 0.5
    ta = rng.uniform(-np.pi, np.pi, n)
    ta = np.where(case == 1, np.clip(rng.normal(0, 1.3, n), -np.pi, np.pi), ta)
    lu = np.where(case == 1, rng.choice(LANDUSE, n, p=[0.1, 0.4, 0.2, 0.2, 0.1]),
                  rng.choice(LANDUSE, n, p=[0.3, 0.2, 0.2, 0.15, 0.15])).astype(np.float64)
    elev[17] = np.nan
    elev[nc + 1 + 0] = np.nan   # the case of the second step
    sets["issf_a"] = {"x": step, "y": np.zeros(n), "t": np.zeros(n), "track": np.array([str(int(v)) for v in step]),
                      "crs": None, "case_": case, "elevation": elev, "sl_": sl, "ta_": ta, "landuse": lu}
    # issf_ties: strata of 1 to 4 cases among 7 rows (tied deaths: the exact recursions), the ids as text ("10" < "2")
    ns, per = 100, 7
    ids = rng.choice(np.arange(1, 500), ns, replace=False).astype(np.float64)
    k = rng.choice([1, 2, 3, 4], ns, p=[0.45, 0.3, 0.17, 0.08])
    case = np.zeros(ns * per)
    for s in range(ns):
        case[s * per + rng.permutation(per)[:k[s]]] = 1
    step = np.repeat(ids, per)
    elev = _f32(rng.normal(100, 25, ns * per) + 5 * case)
    sl = rng.gamma(1.5, 40, ns * per) + 0.5
    sets["issf_ties"] = {"x": step, "y": np.zeros(ns * per), "t": np.zeros(ns * per),
                         "track": np.array([str(int(v)) for v in step]), "crs": None, "case_": case,
                         "elevation": elev, "sl_": sl, "ta_": rng.uniform(-np.pi, np.pi, ns * per),
                         "landuse": np.zeros(ns * per)}
    return sets


def _frame(d: dict, factor: str | None = None) -> str:
    """R code of a data frame of numbers (and the factor column ``factor`` with the levels LEVELS)."""
    cols = []
    for k, v in d.items():
        if k == factor:
            cols.append(f"{k} = factor(c({', '.join(repr(s) for s in v).replace(chr(39), chr(34))}), levels = c("
                        f"{', '.join(repr(s) for s in LEVELS).replace(chr(39), chr(34))}))")
        else:
            cols.append(f"{k} = c({', '.join(repr(float(s)) for s in v)})")
    return f"data.frame({', '.join(cols)})"


def _fit(formula: str, extra: str = "") -> str:
    return f'{extra}m <- amt::fit_issf(x, as.formula("{formula}"), model = TRUE)\n'


def make_cases(Case) -> dict:
    g = "amt_issf"

    def lr(d, fac=None):
        return f"x1 <- {_frame(d['x1'], fac)}\nx2 <- {_frame(d['x2'], fac)}\n"

    cases = {
        "issf_a": Case("issf_a", _fit(F_A) + lr(LR_A) + """c(issf_out(m$model), lr_out(amt::log_rss(m, x1, x2)),
  lr_out(amt::log_rss(m, x1, x2, ci = "se"), "se_"), lr_out(amt::log_rss(m, x1, x2, ci = "se", ci_level = 0.9), "se90_"))""", g),
        "issf_b": Case("issf_a", _fit(F_B) + "issf_out(m$model)", g),
        "issf_fac": Case("issf_a", _fit(F_FAC, 'x$lu <- factor(as.character(as.integer(x$landuse)), levels = c('
                                    + ", ".join(f'"{s}"' for s in LEVELS) + "))\n")
                         + lr(LR_FAC, "lu") + """c(issf_out(m$model), lr_out(amt::log_rss(m, x1, x2)),
  lr_out(amt::log_rss(m, x1, x2, ci = "se"), "se_"))""", g),
        "issf_ties": Case("issf_ties", _fit(F_TIES_TXT) + f"""a <- issf_out(m$model, "t_")
m <- amt::fit_issf(x, as.formula("{F_TIES_NUM}"), model = TRUE)
c(a, issf_out(m$model, "n_"))""", g),
        "issf_col": Case("issf_a", _fit(F_COL, "x$elev2 <- 2 * x$elevation\nx$k1 <- 1\n") + lr(LR_COL)
                         + f"""a <- c(issf_out(m$model), lr_out(amt::log_rss(m, x1, x2, ci = "se"), "se_"))
m <- amt::fit_issf(x, as.formula("{F_ONE}"), model = TRUE)
c(a, issf_out(m$model, "one_"))""", g),
        "issf_boot": Case("issf_a", _fit(F_BOOT) + lr(LR_BOOT) + f"""mle <- amt::log_rss(m, x1, x2)$df$log_rss
set.seed({BOOT_A['seed']})
a <- log_rss_boot_fixed(x, as.formula("{F_BOOT}"), x1, x2, {BOOT_A['ci_level']}, {BOOT_A['n_boot']}, mle)
x1 <- {_frame(LR_BOOT1['x1'])}
m <- amt::fit_issf(x, as.formula("{F_BOOT_LOG}"), model = TRUE)
mle <- amt::log_rss(m, x1, x2)$df$log_rss
set.seed({BOOT_B['seed']})
b <- log_rss_boot_fixed(x, as.formula("{F_BOOT_LOG}"), x1, x2, {BOOT_B['ci_level']}, {BOOT_B['n_boot']}, mle)
c(lr_out(list(df = a), "a_"), lr_out(list(df = b), "b_"))""", g),
    }
    return cases
