# SPDX-License-Identifier: GPL-3.0-or-later
"""Cases of tf.move against R's move2 (tools/make_reference_move.py, tools/move_reference.R).

Each track set is drawn here once and stored with the references
(tests/data/reference/move/inputs.npz), so the tests read the same inputs R
did on every platform. A case is R code run with ``x`` (the move2 object of
its track set) returning a named list of numbers; ``fishers`` is move2's
example (mt_read(mt_example()), also in tests/data/move/fishers.csv.gz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T0 = 1_600_000_000  # 2020-09-13 12:26:40 UTC


def _walk(rng, n, start, scales, lonlat):
    x, y = [start[0]], [start[1]]
    for i in range(n - 1):
        s = scales[i % len(scales)]
        dx, dy = rng.normal(0, s, 2)
        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
    x, y = np.array(x), np.array(y)
    if lonlat:
        x = (x + 180) % 360 - 180
        y = np.clip(y, -89.9, 89.9)
    return x, y


def track_sets() -> dict[str, dict]:
    """name -> {x, y, t (seconds), track (ids), crs}."""
    rng = np.random.default_rng(20261029)
    sets = {}
    # lon / lat: steps from millimetres to hundreds of km, a repeated location, the antimeridian, near a pole
    xs, ys, ts, ks = [], [], [], []
    specs = [("a", 120, (-47.9, -15.8), [1e-6, 1e-4, 1e-3, 0.01, 0.1]),
             ("b", 90, (179.7, 10.0), [0.05, 0.2, 1.0]),
             ("c", 60, (20.0, 88.5), [0.01, 0.3, 2.0])]
    for name, n, start, scales in specs:
        x, y = _walk(rng, n, start, scales, True)
        x[10], y[10] = x[9], y[9]  # same location twice
        t = T0 + np.cumsum(rng.choice([30, 45, 600, 3600, 7200], n)).astype(np.float64)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ks += [name] * n
    sets["ll"] = {"x": np.concatenate(xs), "y": np.concatenate(ys), "t": np.concatenate(ts),
                  "track": np.array(ks), "crs": "EPSG:4326"}
    # projected (UTM 23S, metres) and without a CRS
    xs, ys, ts, ks = [], [], [], []
    for name, n in [("p1", 80), ("p2", 70)]:
        x, y = _walk(rng, n, (190000.0, 8250000.0), [0.5, 25.0, 400.0], False)
        x[5], y[5] = x[4], y[4]
        t = T0 + np.cumsum(rng.choice([20, 60, 900], n)).astype(np.float64)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        ks += [name] * n
    base = {"x": np.concatenate(xs), "y": np.concatenate(ys), "t": np.concatenate(ts), "track": np.array(ks)}
    sets["utm"] = {**base, "crs": "EPSG:32723"}
    sets["plain"] = {**base, "x": base["x"] - 190000.0, "y": base["y"] - 8250000.0, "crs": None}
    # duplicated times: repeated records, records with less, records that differ
    t = T0 + np.array([0, 60, 60, 60, 120, 180, 180, 240, 300, 300, 300, 360], np.float64)
    sets["dup"] = {"x": np.array([0, 1, 1, 1, 2, 3, 3.5, 4, 5, 5, 5, 6], np.float64),
                   "y": np.zeros(12), "t": np.r_[t, t], "track": np.array(["u"] * 12 + ["v"] * 12),
                   "crs": None,
                   "v1": np.array([1, 2, 2, np.nan, 3, 4, 4, 5, 6, 6, np.nan, 7] * 2, np.float64),
                   "v2": np.array([1, 2, np.nan, 2, 3, 4, 5, 5, 6, 7, 7, 8] * 2, np.float64)}
    sets["dup"]["x"] = np.r_[sets["dup"]["x"], sets["dup"]["x"] + 10]
    sets["dup"]["y"] = np.r_[sets["dup"]["y"], sets["dup"]["y"]]
    # gaps to interpolate: lon / lat and plain, with records without a location (NaN)
    x, y = _walk(rng, 40, (-60.0, -3.0), [0.01, 0.5], True)
    t = T0 + np.cumsum(rng.choice([600, 1800, 5400], 40)).astype(np.float64)
    empty = np.zeros(40, bool)
    empty[[3, 4, 11, 20, 21, 22, 39]] = True
    x[empty], y[empty] = np.nan, np.nan
    sets["gap"] = {"x": x, "y": y, "t": t, "track": np.array(["g"] * 20 + ["h"] * 20), "crs": "EPSG:4326"}
    sets["gap_plain"] = {**sets["gap"], "x": x * 1000.0, "y": y * 1000.0, "crs": None}
    return sets


@dataclass(frozen=True)
class Case:
    data: str     # a track set or "fishers"
    r: str        # R code with x -> named list of numbers
    group: str = "move2"


_MEASURES = """list(
  distance = as.numeric(mt_distance(x)),
  azimuth = if (is.na(st_crs(x)) || st_is_longlat(x)) as.numeric(mt_azimuth(x)) else NULL,
  turnangle = if (is.na(st_crs(x)) || st_is_longlat(x)) as.numeric(mt_turnangle(x)) else NULL,
  time_lags = as.numeric(units::set_units(mt_time_lags(x), "s")),
  speed = if (all(as.numeric(mt_time_lags(x)) > 0, na.rm = TRUE)) as.numeric(mt_speed(x)) else NULL
)"""

CASES: dict[str, Case] = {
    "measures_ll": Case("ll", _MEASURES),
    "measures_utm": Case("utm", _MEASURES),
    "measures_plain": Case("plain", _MEASURES),
    "measures_fishers": Case("fishers", "x <- x[!st_is_empty(x), ]\n" + _MEASURES),
    "km_deg": Case("ll", """list(km = as.numeric(mt_distance(x, units = "km")),
                             deg = as.numeric(mt_azimuth(x, units = "degrees")))"""),
    "unique": Case("dup", """list(
  first = as.numeric(mt_unique(x, "first")), last = as.numeric(mt_unique(x, "last")),
  subsets = as.numeric(suppressWarnings(mt_unique(x, "subsets"))),
  subsets_equal = as.numeric(suppressWarnings(mt_unique(x, "subsets_equal"))),
  sample = { set.seed(7); as.numeric(mt_unique(x, "sample")) })"""),
    "per_interval": Case("fishers", """list(
  hour = as.numeric(mt_per_interval(x)), last = as.numeric(mt_per_interval(x, "last")),
  m15 = as.numeric(mt_per_interval(x, unit = "15 mins")), h2 = as.numeric(mt_per_interval(x, unit = "2 hours")),
  h5 = as.numeric(mt_per_interval(x, unit = "5 hours")), day = as.numeric(mt_per_interval(x, unit = "day")),
  d3 = as.numeric(mt_per_interval(x, unit = "3 days")), week = as.numeric(mt_per_interval(x, unit = "week")),
  month = as.numeric(mt_per_interval(x, unit = "month")),
  random = { set.seed(3); as.numeric(mt_per_interval(x, "random", unit = "day")) })"""),
    "visible": Case("fishers", "list(visible = as.numeric(mt_movebank_visible(x)))"),
    "interpolate_empty": Case("gap", """y <- mt_interpolate(x); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2])"""),
    "interpolate_times": Case("gap", """tt <- seq(min(mt_time(x)), max(mt_time(x)), by = "45 mins")
y <- mt_interpolate(x, tt, max_time_lag = units::as_units(2, "h")); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2], t = as.numeric(mt_time(y)), tr = as.numeric(factor(mt_track_id(y))))"""),
    "interpolate_hour_omit": Case("gap", """y <- mt_interpolate(x, "1 hour", omit = TRUE); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2], t = as.numeric(mt_time(y)))"""),
    "interpolate_plain": Case("gap_plain", """tt <- seq(min(mt_time(x)), max(mt_time(x)), by = "20 mins")
y <- mt_interpolate(x, tt); c <- st_coordinates(y)
list(x = c[, 1], y = c[, 2], t = as.numeric(mt_time(y)))"""),
    "aeqd": Case("fishers", """f <- function(p) as.numeric(regmatches(p, regexpr("(?<=lat_0=)[-0-9.]+", p, perl = TRUE)))
g <- function(p) as.numeric(regmatches(p, regexpr("(?<=lon_0=)[-0-9.]+", p, perl = TRUE)))
a <- mt_aeqd_crs_fixed(x)$proj4string; b <- mt_aeqd_crs_fixed(x, "center")$proj4string
list(centroid = c(g(a), f(a)), center = c(g(b), f(b)))"""),
    "brownian": Case("ll", """set.seed(11); b <- mt_sim_brownian_motion(t = c(0, 1, 2.5, 4, 8), sigma = c(1, 2, 0, 0.5), tracks = 3)
c1 <- st_coordinates(b)
set.seed(5); d <- mt_sim_brownian_motion()
c2 <- st_coordinates(d)
list(x1 = c1[, 1], y1 = c1[, 2], x2 = c2[, 1], y2 = c2[, 2])"""),
}

# amt (group "amt"): x's tracks as amt tracks (amt_tracks, tools/amt_fixed.R), results of the tracks in turn

_AMT_STEPS = """s <- lapply(amt_tracks(x), function(tr) amt::steps(tr{args}))
list(sl = amt_cat(s, "sl_"), dir = amt_cat(s, "direction_p"), ta = amt_cat(s, "ta_"),
     x2 = amt_cat(s, "x2_"), y2 = amt_cat(s, "y2_"), dt = amt_cat(s, "dt_"))"""

# the raster of the extract cases: 40 x 50 cells of 37 m, its left edge on the 7th location, its top on the 12th
EXTRACT_RASTER = {"nrows": 40, "ncols": 50, "res": 37.0, "x_from": 6, "y_from": 11, "na_every": 13}
_RASTER = """xy <- sf::st_coordinates(x)
X0 <- xy[7, 1]; Y1 <- xy[12, 2]
r <- terra::rast(nrows = 40, ncols = 50, xmin = X0, xmax = X0 + 50 * 37, ymin = Y1 - 40 * 37, ymax = Y1,
                 crs = "EPSG:32723")
v <- as.numeric(1:2000); v[v %% 13 == 0] <- NA; terra::values(r) <- v; names(r) <- "cover"
"""

CASES.update({
    "amt_steps_utm": Case("utm", _AMT_STEPS.format(args=""), "amt"),
    "amt_steps_plain": Case("plain", _AMT_STEPS.format(args=""), "amt"),
    "amt_steps_ll": Case("ll", _AMT_STEPS.format(args=", lonlat = TRUE"), "amt"),
    "amt_resample": Case("utm", """tr <- amt_tracks(x)
a <- lapply(tr, function(t) amt::track_resample(t, rate = lubridate::seconds(60), tolerance = lubridate::seconds(15)))
b <- lapply(tr, function(t) amt::track_resample(t, rate = lubridate::minutes(15), tolerance = lubridate::minutes(2),
                                                start = 3))
f <- lapply(a, amt::filter_min_n_burst, min_n = 4)
s <- lapply(a, function(t) suppressWarnings(amt::steps_by_burst(t)))
list(t_a = amt_cat(a, "t_"), burst_a = amt_cat(a, "burst_"), t_b = amt_cat(b, "t_"), burst_b = amt_cat(b, "burst_"),
     t_f = amt_cat(f, "t_"), burst_f = amt_cat(f, "burst_"),
     sl = amt_cat(s, "sl_"), dir = amt_cat(s, "direction_p"), ta = amt_cat(s, "ta_"), sburst = amt_cat(s, "burst_"))""",
                         "amt"),
    "amt_summaries_utm": Case("utm", """tr <- amt_tracks(x)
list(nsd = unlist(lapply(tr, amt::nsd)), msd = sapply(tr, amt::msd),
     cum = sapply(tr, function(t) as.numeric(amt::cum_dist(t))), tot = sapply(tr, function(t) as.numeric(amt::tot_dist(t))),
     straight = sapply(tr, function(t) as.numeric(amt::straightness(t))),
     sinuosity = sapply(tr, amt::sinuosity), tac = sapply(tr, amt::tac))""", "amt"),
    "amt_summaries_ll": Case("ll", """tr <- amt_tracks(x)
list(nsd = unlist(lapply(tr, amt::nsd)), msd = sapply(tr, amt::msd),
     cum = sapply(tr, function(t) as.numeric(amt::cum_dist(t))), tot = sapply(tr, function(t) as.numeric(amt::tot_dist(t))))""",
                             "amt"),
    "amt_fit": Case("utm", """out <- list()
for (tr in amt_tracks(x)) {
  s <- amt::steps(tr)
  g <- suppressMessages(amt::fit_distr(s$sl_, "gamma")); v <- amt::fit_distr(s$ta_, "vonmises")
  l <- amt::fit_distr(s$sl_[s$sl_ > 0], "lnorm")
  out <- c(out, list(c(g$params$shape, g$params$scale, as.vector(g$vcov), v$params$kappa, as.numeric(v$vcov)^2,
                       amt_fit_distr_fixed(s$sl_[s$sl_ > 0], "exp")$params$rate, l$params$meanlog, l$params$sdlog,
                       amt::fit_distr(s$sl_, "hnorm")$params$sd, unlist(amt::fit_distr(s$ta_, "unif")$params))))
}
list(fits = unlist(out))""", "amt"),
    "amt_random_numbers": Case("plain", """d <- list(amt::make_gamma_distr(0.4, 120), amt::make_gamma_distr(3.5, 20), amt::make_exp_distr(0.02),
          amt::make_lnorm_distr(3, 0.7), amt::make_hnorm_distr(50), amt::make_unif_distr(),
          amt::make_vonmises_distr(0), amt::make_vonmises_distr(2.5))
list(draws = unlist(lapply(d, function(z) { set.seed(17); amt::random_numbers(z, n = 300) })))""", "amt"),
    "amt_random_steps": Case("utm", """set.seed(42)
r <- lapply(amt_tracks(x), function(tr) amt::random_steps(amt::steps(tr), n_control = 5))
set.seed(3)
u <- lapply(amt_tracks(x), function(tr) amt::random_steps(amt::steps(tr), n_control = 3,
                                                          sl_distr = amt::make_exp_distr(0.01),
                                                          ta_distr = amt::make_unif_distr()))
list(x2 = amt_cat(r, "x2_"), y2 = amt_cat(r, "y2_"), sl = amt_cat(r, "sl_"), ta = amt_cat(r, "ta_"),
     case = amt_cat(r, "case_"), id = amt_cat(r, "step_id_"),
     ux2 = amt_cat(u, "x2_"), uy2 = amt_cat(u, "y2_"), usl = amt_cat(u, "sl_"), uta = amt_cat(u, "ta_"))""", "amt"),
    "amt_random_steps_bursts": Case("utm", """set.seed(7)
r <- lapply(amt_tracks(x), function(tr) {
  b <- amt::track_resample(tr, rate = lubridate::seconds(60), tolerance = lubridate::seconds(15))
  suppressWarnings(amt::random_steps(amt::steps_by_burst(b)))
})
list(x2 = amt_cat(r, "x2_"), y2 = amt_cat(r, "y2_"), sl = amt_cat(r, "sl_"), ta = amt_cat(r, "ta_"),
     case = amt_cat(r, "case_"), id = amt_cat(r, "step_id_"), burst = amt_cat(r, "burst_"))""", "amt"),
    "amt_time_of_day": Case("ll", """tr <- amt_tracks(x)
s <- lapply(tr, amt::steps)
list(tod = unlist(lapply(tr, function(t) as.numeric(amt::time_of_day(t)$tod_))),
     crep = unlist(lapply(tr, function(t) as.numeric(amt::time_of_day(t, include.crepuscule = TRUE)$tod_))),
     end = unlist(lapply(s, function(t) as.numeric(amt::time_of_day(t, include.crepuscule = TRUE)$tod_end_))),
     start = unlist(lapply(s, function(t) as.numeric(amt::time_of_day(t, where = "start")$tod_start_))))""", "amt"),
    "amt_time_of_day_fishers": Case("fishers", """x <- x[!sf::st_is_empty(x), ]
list(crep = unlist(lapply(amt_tracks(x), function(t) as.numeric(amt::time_of_day(t, include.crepuscule = TRUE)$tod_))))""",
                                    "amt"),
    "amt_extract": Case("utm", _RASTER + """tr <- amt_tracks(x)
s <- lapply(tr, amt::steps)
set.seed(42)
rs <- lapply(tr, function(t) amt::random_steps(amt::steps(t), n_control = 5))
list(track = unlist(lapply(tr, function(t) amt::extract_covariates(t, r)$cover)),
     both_start = unlist(lapply(s, function(t) amt::extract_covariates(t, r, where = "both")$cover_start)),
     both_end = unlist(lapply(s, function(t) amt::extract_covariates(t, r, where = "both")$cover_end)),
     start = unlist(lapply(s, function(t) amt::extract_covariates(t, r, where = "start")$cover)),
     random = unlist(lapply(rs, function(t) amt::extract_covariates(t, r)$cover)))""", "amt"),
})


# move2utils (group "move2utils"): the cleaning and thinning functions. Its sets are drawn apart from the others so
# that tests/data/reference/move/inputs.npz keeps its bytes: they are stored in inputs_move2utils.npz.

M2U_SYNTHETIC = "tests/data/move/m2u_synthetic.csv.gz"   # move2utils' synthetic_tracks.csv.gz (GPL-3.0-or-later)


def m2u_track_sets() -> dict[str, dict]:
    """name -> {x, y, t (seconds), track (ids), crs, extra columns}, for the cleaning and thinning cases."""
    rng = np.random.default_rng(20261120)
    sets = {}

    def walk_set(specs, crs, lonlat, steps=(600, 900, 900, 900, 1200, 3600)):
        xs, ys, ts, ks = [], [], [], []
        for name, n, start, scales in specs:
            x, y = _walk(rng, n, start, scales, lonlat)
            t = T0 + np.cumsum(rng.choice(steps, n)).astype(np.float64)
            xs.append(x), ys.append(y), ts.append(t), ks.extend([name] * n)
        return {"x": np.concatenate(xs), "y": np.concatenate(ys), "t": np.concatenate(ts), "track": np.array(ks),
                "crs": crs}

    # lon / lat: tracks of 70, 2 and 45 fixes; the last with a zero time lag, a repeated location and a 4 hour gap
    ll = walk_set([("a", 70, (-47.9, -15.8), [0.0005, 0.002, 0.004, 0.02, 0.0005, 0.0005]),
                   ("b", 2, (10.0, 50.0), [0.003]),
                   ("c", 45, (179.9, 10.0), [0.001, 0.003, 0.03, 0.001])], "EPSG:4326", True)
    c0 = 72  # the first fix of track c
    ll["t"][c0 + 7] = ll["t"][c0 + 6]
    ll["x"][c0 + 10], ll["y"][c0 + 10] = ll["x"][c0 + 9], ll["y"][c0 + 9]
    ll["t"][c0 + 20:] += 14400.0
    sets["cl_ll"] = ll
    # projected (UTM 23S, metres), without a CRS, and in US survey feet (EPSG:2263)
    utm = walk_set([("p", 60, (190000.0, 8250000.0), [0.5, 5.0, 20.0, 2.0, 0.5]),
                    ("q", 40, (191000.0, 8251000.0), [1.0, 8.0, 300.0, 3.0])], "EPSG:32723", False)
    utm["t"][5] = utm["t"][4]
    sets["cl_utm"] = utm
    sets["cl_plain"] = {**utm, "x": utm["x"] - 190000.0, "y": utm["y"] - 8250000.0, "crs": None}
    sets["cl_ft"] = walk_set([("f", 50, (1000000.0, 200000.0), [2.0, 15.0, 60.0, 3.0]),
                              ("g", 30, (1001000.0, 201000.0), [4.0, 30.0, 500.0])], "EPSG:2263", False)
    # tracks of one fix, of three and four fixes (a zero lag in the last), and with locations without a time (UTM)
    t = T0 + np.array([0.0, 0.0, 60.0, 120.0, 0.0, 0.0, 60.0, 120.0, 180.0, 0.0, 60.0, 60.0])
    sets["cl_edge"] = {"x": np.array([0.0, 0.0, 10.0, 5000.0, 7.0, 0.0, 10.0, 20.0, 5030.0, 1.0, 2.0, 3.0]),
                       "y": np.zeros(12), "t": t, "crs": "EPSG:32723",
                       "track": np.array(["s1", "s2", "s2", "s2", "s3", "s4", "s4", "s4", "s4", "s5", "s5", "s5"])}
    t = T0 + np.array([0.0, np.nan, 120.0, 180.0, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 60.0, 90.0, np.nan, 60.0,
                       120.0, np.nan, 0.0, 70.0])
    sets["cl_nat"] = {"x": np.array([0.0, 5.0, 10.0, 5000.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 10.0, 20.0, 100.0, 5.0, 6000.0,
                                     20.0, 0.0, 4000.0]),
                      "y": np.zeros(18), "t": t, "crs": "EPSG:32723",
                      "track": np.array(["n1"] * 4 + ["n2"] * 3 + ["n3"] + ["n4"] * 6 + ["n5"] * 2 + ["n6"] * 2)}
    # the dynamic programme's ties and limits: a regular grid, irregular steps, zero lags, gaps
    g1_t = np.arange(21) * 10.0
    g2_t = np.cumsum(rng.choice([5.0, 10.0, 15.0], 80))
    g3_t = np.repeat(np.arange(8) * 10.0, 2)
    g4_t = np.concatenate([np.arange(4) * 10.0, 500 + np.arange(4) * 10.0, 1000 + np.arange(4) * 10.0])
    ts = [g1_t, g2_t, g3_t, g4_t]
    xs = [g1_t, g2_t + np.cumsum(rng.choice([0.0, 2.0], 80)), g3_t, g4_t * 0.5]
    sets["cl_grid"] = {"x": np.concatenate(xs), "y": np.zeros(sum(map(len, ts))), "t": T0 + np.concatenate(ts),
                       "track": np.array(sum(([n] * len(a) for n, a in zip("wxyz", ts)), [])), "crs": "EPSG:32723"}
    # GPS quality: the columns by their names with underscores (and with dashes), missing values, two empty locations
    n = 40
    sat = rng.integers(3, 13, n).astype(np.float64)
    dop = np.round(rng.uniform(0.8, 14.0, n), 1)
    hacc = np.round(rng.uniform(5.0, 150.0, n), 1)
    sat[[4, 20]] = np.nan
    dop[[9, 30]] = np.nan
    hacc[[2, 12, 33]] = np.nan
    x = np.arange(n, dtype=np.float64) * 10.0
    x[[3, 17]] = np.nan
    q = {"x": x, "y": np.zeros(n), "t": T0 + np.arange(n) * 60.0, "track": np.array(["q"] * n), "crs": "EPSG:32723"}
    sets["cl_q"] = {**q, "gps_satellite_count": sat, "gps_dop": dop, "eobs_horizontal_accuracy_estimate": hacc}
    sets["cl_qd"] = {**q, "gps-satellite-count": sat, "gps-dop": dop, "eobs-horizontal-accuracy-estimate": hacc}
    return sets


# thinning cases: name -> (data, interval or distance, tolerance, criterion[, method]); the R call is made from these
THIN_TIME = {
    "tt_fishers": ("fishers", "1 hour", "10 mins", "closest"),
    "tt_fishers_first": ("fishers", 3600, 600, "first"),
    "tt_synth": ("m2u_synthetic", "45 mins", "5 mins", "closest"),
    "tt_synth_first": ("m2u_synthetic", "20 mins", None, "first"),
    "tt_ll": ("cl_ll", "30 mins", "5 mins", "closest"),
    "tt_ll_first": ("cl_ll", "30 mins", "5 mins", "first"),
    "tt_utm": ("cl_utm", "10 mins", None, "closest"),
    "tt_grid": ("cl_grid", 30, 10, "closest"),
    "tt_grid_first": ("cl_grid", 30, 10, "first"),
    "tt_grid_wide": ("cl_grid", 20, 20, "closest"),
    "tt_grid_tight": ("cl_grid", 20, 0, "closest"),
    "tt_edge": ("cl_edge", 60, 10, "closest"),
    "tt_edge_wide": ("cl_edge", 60, 60, "closest"),
    "tt_nat": ("cl_nat", 60, 10, "closest"),
    "tt_nat_first": ("cl_nat", 60, 10, "first"),
}
THIN_DISTANCE = {
    "td_utm_step": ("cl_utm", 200, None, "closest", "step"),
    "td_utm_int": ("cl_utm", 200, 40, "closest", "interval"),
    "td_utm_int_first": ("cl_utm", 200, 40, "first", "interval"),
    "td_ll_step": ("cl_ll", 5000, None, "closest", "step"),
    "td_ll_int": ("cl_ll", 5000, 1000, "closest", "interval"),
    "td_ft_step": ("cl_ft", 150, None, "closest", "step"),
    "td_ft_int": ("cl_ft", 150, 30, "closest", "interval"),
    "td_fishers_step": ("fishers", 300, None, "closest", "step"),
    "td_fishers_int": ("fishers", 300, 60, "closest", "interval"),
    "td_synth_step": ("m2u_synthetic", 100, None, "closest", "step"),
    "td_synth_int": ("m2u_synthetic", 100, 20, "closest", "interval"),
    "td_grid_step": ("cl_grid", 25, None, "closest", "step"),
    "td_grid_int": ("cl_grid", 30, 10, "closest", "interval"),
    "td_grid_int_first": ("cl_grid", 30, 10, "first", "interval"),
    "td_edge_step": ("cl_edge", 100, None, "closest", "step"),
    "td_edge_int": ("cl_edge", 100, 20, "closest", "interval"),
}
# hard speed caps: name -> (data, v_max, jitter, R function); no v_max: the limit of v_phys_estimate(MASS, MODE)
SPEED_MASS, SPEED_MODE = 2.0, "running"
SPEED = {
    "speed_cl_ll": ("cl_ll", 1.5, None, "mt_flag_speed_cap"),
    "speed_cl_ll_jitter": ("cl_ll", 1.5, 30.0, "mt_flag_speed_cap"),
    "speed_cl_utm": ("cl_utm", 0.1, None, "mt_flag_speed_cap"),
    "speed_cl_utm_jitter": ("cl_utm", 0.1, 0.5, "mt_flag_speed_cap"),
    "speed_cl_plain": ("cl_plain", 0.1, 0.5, "mt_flag_speed_cap"),
    "speed_cl_ft": ("cl_ft", 0.3, 1.0, "mt_flag_speed_cap"),
    "speed_fishers": ("fishers", 1.0, 5.0, "mt_flag_speed_cap"),
    "speed_synth": ("m2u_synthetic", 5.0, None, "mt_flag_speed_cap"),
    "speed_synth_phys": ("m2u_synthetic", None, 3.0, "mt_flag_speed_cap"),
    "speed_edge": ("cl_edge", 3.0, 5.0, "mt_flag_speed_cap_fixed"),
    "speed_nat": ("cl_nat", 3.0, 5.0, "mt_flag_speed_cap_fixed"),
}


def _r_arg(v):
    return "NULL" if v is None else (f'"{v}"' if isinstance(v, str) else repr(v))


_NUM = "num <- function(v) if (is.null(v)) NULL else as.numeric(v)\n"
_NONEMPTY = "x <- x[!st_is_empty(x), ]\n"


def _speed_case(data, v_max, jitter, fn):
    pre = _NONEMPTY if data == "fishers" else ""
    v = f'v <- as.numeric(v_phys_estimate({SPEED_MASS}, "{SPEED_MODE}"))\n' if v_max is None else ""
    return Case(data, pre + _NUM + v + f"""o <- {fn}(x, v_max = {'v' if v_max is None else v_max}, threshold_type = "hard", jitter = {_r_arg(jitter)}, plot = FALSE, silent = TRUE)
list(outlier = num(o[["is_outlier"]]), speed = num(o[["step_speed"]]), above = num(o[["is_speed_above_cap"]]),
     below = num(o[["is_step_below_jitter"]]), flagged = num(o[["flagged_by_speed"]]), loglr = num(o[["loglr_speed"]]),
     vmax = num(attr(o, "v_max_used")))""", "move2utils")


def _thin_case(kind, data, size, tolerance, criterion, method=None):
    pre = _NONEMPTY if (kind == "distance" and data == "fishers") else ""
    if kind == "time":
        call = (f"mt_thin_time(x, interval = {_r_arg(size)}, tolerance = {_r_arg(tolerance)}, "
                f'criterion = "{criterion}")')
    else:
        call = (f"mt_thin_distance(x, distance = {size}, tolerance = {_r_arg(tolerance)}, "
                f'criterion = "{criterion}", method = "{method}")')
    return Case(data, pre + f"o <- {call}\nlist(selected = as.numeric(o$thin_selected))", "move2utils")


_QUALITY = """x$rowid <- seq_len(nrow(x))
mask <- function(...) as.numeric(x$rowid %in% FN(x, ...)$rowid)
list(default = mask(), nosat = mask(sat_min = NULL), strict = mask(sat_min = 8, dop_max = 3, hacc_max = 40),
     keepempty = mask(drop_empty = FALSE), nodop = mask(dop_max = NULL, hacc_max = 60),
     none = mask(sat_min = NULL, dop_max = NULL, hacc_max = NULL, drop_empty = FALSE))"""
# the points of NaN coordinates the sets hold are empty locations for the filter
_EMPTY = "x$geometry[is.na(sf::st_coordinates(x)[, 1])] <- sf::st_point()\n"
_QUALITY_FIXED = _QUALITY.replace("FN", "mt_filter_gps_quality_fixed")
_QUALITY = "mt_filter_gps_quality_quiet <- function(x, ...) mt_filter_gps_quality(x, ..., verbose = FALSE)\n" \
    + _QUALITY.replace("FN", "mt_filter_gps_quality_quiet")

CASES.update({
    "v_phys": Case("cl_edge", """masses <- c(1e-9, 3e-8, 0.001, 0.02, 0.5, 1, 2, 5, 6, 70, 250, 1000, 25000, 1.084e5, 2e5)
res <- list()
for (lv in c(0.95, 0.9, 0.99)) for (md in c("flying", "running", "swimming")) for (m in masses) {
  r <- suppressWarnings(v_phys_estimate(m, md, ci_level = lv))
  res[[length(res) + 1]] <- c(as.numeric(r), attr(r, "ci"), attr(r, "kmh"))
}
m <- do.call(rbind, res)
list(v = m[, 1], lo = m[, 2], hi = m[, 3], kmh = m[, 4])""", "move2utils"),
    **{k: _speed_case(*v) for k, v in SPEED.items()},
    **{k: _thin_case("time", *v) for k, v in THIN_TIME.items()},
    **{k: _thin_case("distance", *v) for k, v in THIN_DISTANCE.items()},
    "gps_cl_q": Case("cl_q", _EMPTY + _QUALITY, "move2utils"),
    "gps_cl_qd": Case("cl_qd", _EMPTY + _QUALITY, "move2utils"),
    "gps_cl_q_fixed": Case("cl_q", _EMPTY + _QUALITY_FIXED, "move2utils"),
    "gps_synth": Case("m2u_synthetic", _QUALITY_FIXED.replace("sat_min = 8, dop_max = 3, hacc_max = 40",
                                                              "sat_min = 8, dop_max = 3"), "move2utils"),
    "gps_synth_orig": Case("m2u_synthetic", _QUALITY.replace("sat_min = 8, dop_max = 3, hacc_max = 40",
                                                             "sat_min = 8, dop_max = 3"), "move2utils"),
    "gps_fishers": Case("fishers", _QUALITY_FIXED, "move2utils"),
})


# amt's extract_covariates_along (group "amt_along", in amt_along.npz; the steps of its sets are stored in
# inputs_along.npz): the steps start at the points x, y of a set and end at its x2, y2; the rasters of ALONG_RASTERS
# have three layers made of integers, so that R and numpy build the same values.

ALONG_LAYERS = ("cover", "lvl", "hill")
ALONG_RASTERS = {
    "fishers": dict(nrows=355, ncols=1100, xmin=-73.94041, xres=0.0005, ymax=42.85101, yres=0.0004, crs="EPSG:4326"),
    "utm": dict(nrows=200, ncols=200, xmin=186999.63, xres=37.3, ymax=8253000.91, yres=37.3, crs="EPSG:32723"),
    "edges": dict(nrows=10, ncols=12, xmin=0.0, xres=1.0, ymax=10.0, yres=1.0, crs="EPSG:32723"),
    "tenth": dict(nrows=30, ncols=40, xmin=0.0, xres=0.1, ymax=3.0, yres=0.1, crs=None),
    "origin": dict(nrows=15, ncols=20, xmin=1.3, xres=2.5, ymax=17.2, yres=1.7, crs=None),
    "big": dict(nrows=40, ncols=60, xmin=500000.123, xres=30.0, ymax=7500000.456, yres=30.0, crs="EPSG:32723"),
}


def along_raster(name: str) -> dict:
    """The extent and the values (layer: (nrows, ncols) array, NaN for no data) of a raster of ALONG_RASTERS."""
    p = ALONG_RASTERS[name]
    nr, nc = p["nrows"], p["ncols"]
    cell = np.arange(nr * nc, dtype=np.float64)
    row, col = cell // nc, cell % nc
    cover = cell + 1
    cover[cover % 13 == 0] = np.nan
    lvl = (row * 31 + col * 17) % 101
    lvl[(row + col) % 11 == 0] = np.nan
    hill = (row - nr // 2) ** 2 + (col - nc // 2) ** 2
    values = {k: v.reshape(nr, nc) for k, v in zip(ALONG_LAYERS, (cover, lvl, hill))}
    return {**p, "xmax": p["xmin"] + nc * p["xres"], "ymin": p["ymax"] - nr * p["yres"], "values": values}


def _r_along_raster(name: str, layers: int = 3) -> str:
    p = ALONG_RASTERS[name]
    crs = "NA_character_" if p["crs"] is None else '"' + p["crs"] + '"'
    sel = "" if layers == 3 else "r <- r[[1]]\n"
    return f"""nr <- {p['nrows']}; nc <- {p['ncols']}
r <- terra::rast(nrows = nr, ncols = nc, xmin = {p['xmin']!r}, xmax = {p['xmin']!r} + nc * {p['xres']!r},
                 ymin = {p['ymax']!r} - nr * {p['yres']!r}, ymax = {p['ymax']!r}, nlyrs = 3, crs = {crs})
cell <- 0:(nr * nc - 1); row <- cell %/% nc; col <- cell %% nc
cover <- cell + 1; cover[cover %% 13 == 0] <- NA
lvl <- (row * 31 + col * 17) %% 101; lvl[(row + col) %% 11 == 0] <- NA
hill <- (row - nr %/% 2)^2 + (col - nc %/% 2)^2
terra::values(r) <- cbind(cover, lvl, hill); names(r) <- c("cover", "lvl", "hill")
{sel}"""


def _along_set(x1, y1, x2, y2, crs=None) -> dict:
    n = len(x1)
    return {"x": np.asarray(x1, np.float64), "y": np.asarray(y1, np.float64), "t": T0 + np.arange(n, dtype=np.float64),
            "track": np.array(["s"] * n), "crs": crs, "x2": np.asarray(x2, np.float64), "y2": np.asarray(y2, np.float64)}


def _ulps(v, k):
    """v moved by k units in the last place."""
    for _ in range(abs(k)):
        v = np.nextafter(v, np.inf if k > 0 else -np.inf)
    return v


def _random_steps_set() -> dict:
    """The steps of tf.move.random_steps on the utm set (this needs terraflux)."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    import terraflux as tf

    s = track_sets()["utm"]
    geom = [Point(a, b) for a, b in zip(s["x"], s["y"])]
    df = {"time": pd.to_datetime((s["t"] * 1e9).astype(np.int64), utc=True), "track": s["track"]}
    tr = tf.move.as_tracks(gpd.GeoDataFrame(df, geometry=geom, crs=s["crs"]), "time", "track", sort=False)
    rs = tf.move.random_steps(tf.move.steps(tr), n_control=5, seed=42)
    return _along_set(rs["x1_"], rs["y1_"], rs["x2_"], rs["y2_"], crs=s["crs"])


def along_track_sets() -> dict[str, dict]:
    """name -> the steps of the along cases, as the sets of track_sets (x, y, t, track, crs, x2, y2)."""
    rng = np.random.default_rng(20261121)
    sets = {}
    pick = lambda a, m: a[rng.integers(0, len(a), m)]  # noqa: E731

    def lattice(k, lo, hi, m=3000):
        return rng.integers(round(lo / k), round(hi / k) + 1, m) * k

    # on the edges and corners of a 12 x 10 raster of unit cells: lines along each edge and each diagonal of the
    # cells, and steps between points of lattices of 0.5 and 0.25, some of them outside
    sx, sy, ex, ey = [], [], [], []
    for i in range(0, 13):
        sx += [i, i]
        sy += [0, 10]
        ex += [i, i]
        ey += [10, 0]
    for j in range(0, 11):
        sx += [0, 12]
        sy += [j, j]
        ex += [12, 0]
        ey += [j, j]
    for i in range(0, 12):
        for j in range(0, 10):
            sx += [i, i + 1, i, i + 1]
            sy += [j, j, j + 1, j + 1]
            ex += [i + 1, i, i + 1, i]
            ey += [j + 1, j + 1, j, j]
    parts = [[np.array(a, np.float64) for a in (sx, sy, ex, ey)]]
    for k in (0.5, 0.25):
        parts.append([lattice(k, -1, 13), lattice(k, -1, 11), lattice(k, -1, 13), lattice(k, -1, 11)])
    sets["along_edges"] = _along_set(*(np.concatenate(c) for c in zip(*parts)), crs="EPSG:32723")
    # vertical and horizontal lines, and lines within 0.01 pixel of them (and of 1e-4 of the end of a cell)
    d_pos = np.array([0, 1e-5, 0.001, 0.005, 0.0099, 0.00999999, 0.01, 0.0100001, 0.02, 0.5, 0.999, 1 - 1e-5])
    d_pos = np.r_[d_pos, -d_pos]
    d_len = np.array([0, 1e-5, 1e-4, 0.00009999, 0.0001, 0.004, 0.0099, 0.01, 0.0100001, 0.02])
    d_len = np.r_[d_len, -d_len]
    m = 2000
    ends = np.r_[np.arange(-1, 13, 0.5), rng.uniform(-1, 13, 20)]
    vx = rng.integers(-1, 14, m) + rng.choice(d_pos, m)
    horiz_y = rng.integers(-1, 12, m) + rng.choice(d_pos, m)
    sets["along_axis"] = _along_set(np.r_[vx, pick(ends, m)], np.r_[pick(ends, m), horiz_y],
                                    np.r_[vx + rng.choice(d_len, m), pick(ends, m)],
                                    np.r_[pick(ends, m), horiz_y + rng.choice(d_len, m)], crs="EPSG:32723")
    # zero-length steps: on the edges and corners, inside, outside, on the border of the raster
    zx = np.r_[lattice(0.5, -1, 13), rng.uniform(-1, 13, 1000), [0, 12, 0, 12, 6, 6, 0, 12, 13, -1]]
    zy = np.r_[lattice(0.5, -1, 11), rng.uniform(-1, 11, 1000), [0, 0, 10, 10, 0, 10, 5, 5, 5, 5]]
    sets["along_zero"] = _along_set(zx, zy, zx, zy, crs="EPSG:32723")
    # partly and wholly outside, and touching the border from outside
    ox = rng.uniform(-6, 18, (3000, 2))
    oy = rng.uniform(-6, 16, (3000, 2))
    tx = np.array([[12, 14], [14, 12], [12, 14], [-2, 0], [0, -2], [5, 5], [5, 5], [5, 7], [7, 5], [-3, -1], [13, 15],
                   [12, 12], [0, 0], [12, 13], [0, -1], [12, 12], [0, 0], [6, 8], [6, 4], [-1, -1]], np.float64)
    ty = np.array([[5, 7], [5, 7], [3, 3], [5, 5], [5, 5], [10, 12], [0, -2], [10, 12], [-2, 0], [-1, 1], [5, 5],
                   [2, 5], [2, 5], [5, 5], [5, 5], [11, 13], [-1, -3], [10, 12], [0, -2], [3, 6]], np.float64)
    sets["along_outside"] = _along_set(np.r_[ox[:, 0], tx[:, 0]], np.r_[oy[:, 0], ty[:, 0]], np.r_[ox[:, 1], tx[:, 1]],
                                       np.r_[oy[:, 1], ty[:, 1]], crs="EPSG:32723")
    # coordinates that paste0's 15 significant digits change: multiples of 0.1 made in several ways, neighbours in ulps
    k = np.arange(-3, 44)
    base = np.r_[k * 0.1, k / 10, 0.1 + 0.2 + (k - 3) * 0.1, (k * 0.3) / 3, k * (1 / 10)]
    base = np.r_[base, [_ulps(b, u) for b in base[::3] for u in (-2, -1, 1, 2)]]
    k = np.arange(-3, 34)
    by = np.r_[k * 0.1, k / 10, [_ulps(b, u) for b in k[3:] * 0.1 for u in (-1, 1)]]
    m = 3000
    sets["along_digits"] = _along_set(pick(base, m), pick(by, m), pick(base, m), pick(by, m))
    # coordinates of the size of UTM ones near the edges, with digits beyond the 15th
    big = ALONG_RASTERS["big"]
    gx = big["xmin"] + np.arange(0, 61) * big["xres"]
    gy = big["ymax"] - np.arange(0, 41) * big["yres"]
    noise = np.array([0.0, 1e-10, -1e-10, 3e-10, -4.4e-10, 6e-10, -1e-9, 2e-9])

    def jitter(g, m):
        return pick(g, m) + rng.choice(noise, m) + rng.integers(0, 2, m) * rng.uniform(0, 1e-3, m)

    m = 3000
    sets["along_big"] = _along_set(jitter(gx, m), jitter(gy, m), jitter(gx, m), jitter(gy, m), crs="EPSG:32723")
    # a raster whose origin is not a multiple of its cell size: steps on and near the edges, and random ones
    o = ALONG_RASTERS["origin"]
    ex = o["xmin"] + np.arange(-1, 22) * o["xres"]
    ey = o["ymax"] - np.arange(-1, 17) * o["yres"]
    xmax, ymin = o["xmin"] + o["ncols"] * o["xres"], o["ymax"] - o["nrows"] * o["yres"]
    sets["along_origin"] = _along_set(
        np.r_[pick(ex, 1500), rng.uniform(o["xmin"] - 3, xmax + 3, 1500)], np.r_[pick(ey, 1500), rng.uniform(ymin - 3, o["ymax"] + 3, 1500)],
        np.r_[pick(ex, 1500), rng.uniform(o["xmin"] - 3, xmax + 3, 1500)], np.r_[pick(ey, 1500), rng.uniform(ymin - 3, o["ymax"] + 3, 1500)])
    # a single step, and a single one outside the raster
    sets["along_one"] = _along_set([0.5], [0.5], [9.5], [3.2], crs="EPSG:32723")
    sets["along_one_out"] = _along_set([-5.0], [-5.0], [-4.0], [-3.0], crs="EPSG:32723")
    sets["along_random"] = _random_steps_set()
    return sets


# case -> raster (the cases of the steps of a set; the fishers' steps are the data's own)
ALONG_CASES = {"along_edges": "edges", "along_axis": "edges", "along_zero": "edges", "along_outside": "edges",
               "along_digits": "tenth", "along_big": "big", "along_origin": "origin", "along_one": "edges",
               "along_one_out": "edges", "along_random": "utm"}
_ALONG_STEPS = "xy <- sf::st_coordinates(x); s <- data.frame(x1_ = xy[, 1], y1_ = xy[, 2], x2_ = x$x2, y2_ = x$y2)\n"
_FISHERS_STEPS = """x <- x[!st_is_empty(x), ]
s <- do.call(rbind, lapply(amt_tracks(x), function(t) {
  as.data.frame(suppressWarnings(amt::steps(t, lonlat = TRUE)))[, c("x1_", "y1_", "x2_", "y2_")] }))
"""

CASES.update({
    **{name: Case(name, _r_along_raster(raster) + _ALONG_STEPS + "amt_along(s, r)", "amt_along")
       for name, raster in ALONG_CASES.items()},
    "along_fishers": Case("fishers", _r_along_raster("fishers") + _FISHERS_STEPS + "amt_along(s, r)", "amt_along"),
    "along_fishers_one": Case("fishers", _r_along_raster("fishers", 1) + _FISHERS_STEPS
                              + "s <- s[1:5000, ]\namt_along(s, r)", "amt_along"),
    # one layer: a vector per step instead of a data frame
    "along_edges_one": Case("along_edges", _r_along_raster("edges", 1) + _ALONG_STEPS + "amt_along(s, r)", "amt_along"),
    "along_origin_one": Case("along_origin", _r_along_raster("origin", 1) + _ALONG_STEPS + "amt_along(s, r)",
                             "amt_along"),
})
