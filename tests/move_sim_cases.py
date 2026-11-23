# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases of tf.move's simulation from a step selection model (redistribution_kernel, simulate_path) against R's amt 0.3.1
(group "amt_sim"; tools/amt_fixed.R holds ``sim_out``, ``redistribution_kernel_fixed`` and ``simulate_path_fixed``).

Each case runs amt's own functions (names ``orig_*``: where its errors let it) and the corrected ones (``fixed_*``) from
the same seed; the maps, the model's numbers and the tracks R used are in the output too, as ``in_*`` keys, which
``tools/make_reference_move.py`` stores in ``inputs_sim.npz``. ``make_cases(Case)`` gives the cases for
``move_cases.CASES`` (the hook at the end of that file).
"""

from __future__ import annotations

# the models: R's and Python's, the same numbers
A = {"coefs": {"cov_end": 1.5, "sl_": -0.002, "cos_ta_": 0.4}, "sl": ("gamma", {"shape": 2, "scale": 50}),
     "ta": ("vonmises", {"kappa": 2}), "start": (500.0, 500.0), "ta0": 0.0, "time": "2020-01-01 00:00:00", "dt": "1h",
     "n_control": 1000, "tol": 1.0, "n_steps": 20, "seed": 42}
C = {"coefs": {"cov_end": 1.2, "I(cov_end^2)": -0.3, "log_sl_": 0.15, "cov_end:sl_": -0.004},
     "sl": ("hnorm", {"sd": 60}), "ta": ("unif", {"min": -3.141592653589793, "max": 3.141592653589793}),
     "start": (480.0, 520.0), "ta0": 1.0, "time": "2021-06-01 12:00:00", "dt": "15min", "n_control": 500, "tol": 1.0,
     "n_steps": 15, "seed": 7, "max_dist": 500}
D = {"coefs": {"cov_end": 1.0, "cov_start": 0.3, "exp(cov_end)": -0.2, "cov_end:cos_ta_": 0.5, "log_sl_": -0.1},
     "sl": ("lnorm", {"meanlog": 3.5, "sdlog": 0.8}), "ta": ("vonmises", {"kappa": 1.5}), "start": (300.0, 700.0),
     "ta0": -2.0, "time": "2022-03-05 06:30:00", "dt": "2h", "n_control": 700, "tol": 1.0, "n_steps": 12, "seed": 99}
# a start near the edge with the default tolerance: amt stops before the first step; with no covariate at the end of
# the step a candidate outside has a weight, and the corrected version stops too (or not) as amt's rule
E = {**A, "start": (40.0, 500.0), "tol": 0.0, "n_steps": 10, "seed": 5, "n_control": 1000}
F = {"coefs": {"sl_": -0.01, "cos_ta_": 0.3}, "sl": ("gamma", {"shape": 2, "scale": 50}), "ta": ("vonmises", {"kappa": 1}),
     "start": (500.0, 300.0), "ta0": -1.5, "time": "2020-01-01 00:00:00", "dt": "1h", "n_control": 1000, "tol": 0.05,
     "n_steps": 40, "seed": 11}

B = {"seed_fit": 1, "seed": 2024, "n_control": 1000, "tol": 1.0, "n_steps": 20, "ta0": 0.5, "time": "2009-02-11 12:00:00",
     "dt": "30min", "formula": "case_ ~ popden_end + sl_ + log_sl_ + cos_ta_ + strata(step_id_)"}


def _distr(kind, name, params):
    f = {"gamma": "make_gamma_distr", "vonmises": "make_vonmises_distr", "hnorm": "make_hnorm_distr",
         "lnorm": "make_lnorm_distr", "unif": "make_unif_distr", "exp": "make_exp_distr"}[name]
    return f"amt::{f}({', '.join(f'{k} = {v!r}' for k, v in params.items())})"


def _r_model(c) -> str:
    coefs = ", ".join(f'"{k}" = {v!r}' for k, v in c["coefs"].items())
    return (f"m <- amt::make_issf_model(coefs = c({coefs}), sl = {_distr('sl', *c['sl'])}, ta = {_distr('ta', *c['ta'])})\n"
            f"st <- amt::make_start(c({c['start'][0]!r}, {c['start'][1]!r}), ta_ = {c['ta0']!r}, time = as.POSIXct(\"{c['time']}\", tz = \"UTC\"), "
            f"dt = lubridate::{_r_dt(c['dt'])})\n")


def _r_dt(dt: str) -> str:
    return {"1h": "hours(1)", "15min": "minutes(15)", "2h": "hours(2)", "30min": "minutes(30)"}[dt]


def _run(c, original=True) -> str:
    md = f"max.dist = {c['max_dist']!r}" if "max_dist" in c else "max.dist = NULL"
    return (f"sim_out(m, st, r, {float(c['n_control'])!r}, {float(c['tol'])!r}, {c['n_steps']}, {c['seed']}, \"\", {md}, "
            f"original = {'TRUE' if original else 'FALSE'})")


def make_cases(Case) -> dict:
    g = "amt_sim"

    def case(c, extra=""):
        return Case("sim", "r <- sim_map_a()\n" + _r_model(c) + extra + _run(c), g)

    return {
        "sim_a": Case("sim", "r <- sim_map_a()\n" + _r_model(A) + "c(list(in_map = as.vector(terra::values(r))), "
                      + _run(A) + ")", g),
        "sim_c": case(C),
        "sim_d": case(D),
        "sim_e": case(E),
        "sim_f": case(F),
        "sim_b": Case("sim", f"""cv <- amt::get_amt_fisher_covars()
pd <- cv$popden
leroy <- amt::amt_fisher |> dplyr::filter(name == "Leroy")
dat <- leroy |> amt::track_resample(rate = lubridate::minutes(30), tolerance = lubridate::minutes(5)) |>
  amt::filter_min_n_burst(3) |> amt::steps_by_burst()
set.seed({B['seed_fit']})
ssf <- dat |> amt::random_steps(n_control = 10) |> amt::extract_covariates(pd, where = "both") |>
  dplyr::mutate(log_sl_ = log(sl_), cos_ta_ = cos(ta_))
fit <- amt::fit_issf(ssf, {B['formula']}, model = TRUE)
r <- pd
m <- fit
st <- amt::make_start(c(dat$x1_[1], dat$y1_[1]), ta_ = {B['ta0']!r}, time = as.POSIXct("{B['time']}", tz = "UTC"),
                      dt = lubridate::minutes(30), crs = NA)
c(list(in_leroy_x = leroy$x_, in_leroy_y = leroy$y_, in_leroy_t = as.numeric(leroy$t_),
       in_pd_vals = as.vector(terra::values(pd)), in_pd_ext = as.vector(terra::ext(pd)),
       in_pd_dim = c(terra::nrow(pd), terra::ncol(pd)), in_start = c(dat$x1_[1], dat$y1_[1]), coef = unname(coef(fit)),
       sl_shape = fit$sl_$params$shape, sl_scale = fit$sl_$params$scale, ta_kappa = fit$ta_$params$kappa,
       n_steps_fit = nrow(dat)),
  sim_out(m, st, r, {float(B['n_control'])!r}, {float(B['tol'])!r}, {B['n_steps']}, {B['seed']}, "", max.dist = NULL))
""", g),
    }
