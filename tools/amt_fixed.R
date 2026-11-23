# SPDX-License-Identifier: GPL-3.0-or-later
# amt 0.3.1 with its errors corrected, the references tf.move is tested against,
# and the helpers of the amt cases (tests/move_cases.py).
#
# direction_abs(lonlat = TRUE): geosphere::bearing, clockwise from north, is
# taken as the anticlockwise angle from east; here it is converted, 90 - bearing.
# sinuosity: the turn angles, already in radians, are converted from degrees
# (cos(ta * pi / 180)); here they are taken as radians.
# hr_kde_pi, hr_kde_lscv (rescale = "unitvar"), hr_mcp (a track without a CRS): see hr_kde_pi_fixed,
# hr_kde_lscv_fixed and hr_mcp_fixed below.
# fit_distr(x, "exp"): fitdistrplus' BFGS fails for rates below 1e-3 (its
# numerical gradient steps the rate below 0); here the maximum likelihood rate,
# 1 / mean(x), from which amt's BFGS starts.
# redistribution_kernel() / simulate_path() (continuous space; the cases of tests/move_sim_cases.py): see
# redistribution_kernel_fixed() and simulate_path_fixed() at the end of this file.
# The rest is amt's code.

local({
  ns <- asNamespace("amt")

  direction_abs_fixed <- function(x, full_circle = FALSE, zero_dir = "E", clockwise = FALSE,
                                  append_last = TRUE, lonlat = FALSE, ...) {
    zero_dir <- toupper(zero_dir)
    if (!zero_dir %in% c("E", "N", "W", "S")) {
      stop("zero_dir should be one of 'E', 'N', 'W', or 'S'.")
    }
    if (zero_dir == "E") {
      zero_dir <- "East"
    }
    x$dx <- diff_x(x)
    x$dy <- diff_y(x)
    x$zero_step <- x$dx == 0 & x$dy == 0
    a <- if (!lonlat) {
      atan2(x$dy, x$dx)
    } else {
      xx <- sf::st_coordinates(as_sf(x))
      c(90 - geosphere::bearing(xx[-nrow(xx), ], xx[-1, ]), NA) * pi / 180
    }
    a[x$zero_step] <- NA
    a <- if (append_last) a else a[-length(a)]
    a <- ifelse(a < 0, 2 * pi + a, a)
    a <- switch(zero_dir,
      "East" = a,
      "S" = (2.5 * pi + a) %% (2 * pi),
      "W" = (3 * pi + a) %% (2 * pi),
      "N" = (3.5 * pi + a) %% (2 * pi)
    )
    a <- if (clockwise) (2 * pi - a) %% (2 * pi) else a
    a <- if (full_circle) a else ifelse(a > pi, (2 * pi - a) * -1, a)
    a
  }

  sinuosity_fixed <- function(x, ...) {
    x <- steps(x)
    p <- mean(x$sl_)
    cv <- sd(x$sl_) / p
    mc <- mean(cos(x$ta_), na.rm = TRUE)
    ms <- mean(sin(x$ta_), na.rm = TRUE)
    2 * (p * ((1 - mc^2 - ms^2) / ((1 - mc)^2 + ms^2) + cv^2))^(-0.5)
  }

  environment(direction_abs_fixed) <- ns
  environment(sinuosity_fixed) <- ns
  # UseMethod finds a method in the generic's namespace before the registered ones
  for (f in list(list("direction_abs.track_xy", direction_abs_fixed), list("sinuosity.track_xy", sinuosity_fixed))) {
    unlockBinding(f[[1]], ns)
    assign(f[[1]], f[[2]], envir = ns)
    lockBinding(f[[1]], ns)
  }
  registerS3method("direction_abs", "track_xy", direction_abs_fixed, envir = ns)
  registerS3method("sinuosity", "track_xy", sinuosity_fixed, envir = ns)
})

amt_fit_distr_fixed <- function(x, dist_name, na.rm = TRUE) {
  if (dist_name == "exp") {
    if (na.rm) x <- x[!is.na(x)]
    return(amt::make_exp_distr(rate = 1 / mean(x)))
  }
  amt::fit_distr(x, dist_name, na.rm = na.rm)
}

# One amt track_xyt per track of the move2 object x, in the order of x.
amt_tracks <- function(x) {
  xy <- sf::st_coordinates(x)
  id <- as.character(move2::mt_track_id(x))
  t <- move2::mt_time(x)
  crs <- sf::st_crs(x)
  lapply(split(seq_len(nrow(xy)), factor(id, levels = unique(id))), function(i) {
    amt::make_track(data.frame(x = xy[i, 1], y = xy[i, 2], t = t[i]), x, y, t,
                    crs = if (is.na(crs)) NA else crs)
  })
}

# The column `col` of each table of a list, one after the other.
amt_cat <- function(tables, col) unlist(lapply(tables, function(d) as.numeric(d[[col]])), use.names = FALSE)

# amt's extract_covariates_along on a table of steps (columns x1_, y1_, x2_, y2_), flattened for the references:
# off (the cells of each step: its values are [off[i], off[i + 1])) and v1, v2, ... (the values of each layer, the
# steps one after the other, in the order amt's list has them). A raster of one layer gives vectors, of several
# data frames; both are read here.
amt_along <- function(steps, r) {
  l <- amt:::extract_covariates_along.steps_xy(steps, r)
  m <- lapply(l, function(v) if (is.data.frame(v)) as.matrix(v) else matrix(as.numeric(v), ncol = 1))
  all <- do.call(rbind, m)
  out <- list(off = c(0, cumsum(vapply(m, nrow, 1L))))
  for (k in seq_len(ncol(all))) out[[paste0("v", k)]] <- all[, k]
  out
}

# hr_kde_pi(rescale = "unitvar") and hr_kde_lscv(rescale = "unitvar") multiply the bandwidth by sd(x[, 1]) and
# sd(x[, 2]): x[, 1] of a track (a tibble) is a one-column table, which sd() refuses ("'list' object cannot be
# coerced to type 'double'"); here the columns x_ and y_. The rest is amt's code.
hr_kde_pi_fixed <- function(x, rescale = "none", correct = TRUE, ...) {
  if (!rescale %in% c("unitvar", "xvar", "none")) {
    stop("rhrHpi: scale: not one of unitvar, xvar or none")
  }
  xs <- dplyr::pull(x, "x_")
  ys <- dplyr::pull(x, "y_")
  if (rescale == "unitvar") {
    xs <- xs / sd(xs)
    ys <- ys / sd(ys)
  } else if (rescale == "xvar") {
    ys <- (ys / sd(ys)) * sd(xs)
  }
  hx <- KernSmooth::dpik(xs, ...)
  hy <- KernSmooth::dpik(ys, ...)
  h <- c(hx, hy)
  if (rescale == "unitvar") {
    h <- KernSmooth::dpik(xs, ...)
    h <- h * c(sd(x$x_), sd(x$y_))
  }
  if (correct) {
    h <- h^(5 / 6)
  }
  h
}

hr_kde_lscv_fixed <- function(x, range = do.call(seq, as.list(c(amt::hr_kde_ref(x) * c(0.1, 2), length.out = 100))),
                              which_min = "global", rescale = "none", trast = amt::make_trast(x)) {
  if (!rescale %in% c("unitvar", "xvar", "none")) {
    stop("scale: not one of unit, sd or none")
  }
  xs <- dplyr::pull(x, "x_")
  ys <- dplyr::pull(x, "y_")
  if (rescale == "unitvar") {
    xs <- xs / sd(xs)
    ys <- ys / sd(ys)
  } else if (rescale == "xvar") {
    ys <- (ys / sd(ys)) * sd(xs)
  }
  converged <- TRUE
  res <- amt:::lscv(cbind(xs, ys), range)
  if (which_min == "global") {
    h <- range[which.min(res)]
  } else {
    h <- range[max(amt:::local_minima(res))]
  }
  if (range[1] == h | range[length(range)] == h) {
    converged <- FALSE
    warning("hr_kde_lscv: lscv did not converge.")
  }
  if (rescale == "unitvar") {
    h <- h * c(sd(x$x_), sd(x$y_))
  } else {
    h <- c(h, h)
  }
  list(h = h, converged = converged, res = res, which_min = which_min, rescale = rescale, range = range)
}

# hr_mcp(): st_as_sfc(geometry, crs = get_crs(x)) refuses the NA of a track without a CRS ("is.numeric(crs) ||
# is.character(crs) || inherits(crs, "crs") is not TRUE"), so no minimum convex polygon (nor intensity_use) of such a
# track; here the NA CRS of sf. The rest is amt's code.
hr_mcp_fixed <- function(x, levels = 0.95, keep.data = TRUE, ...) {
  checkmate::assert_numeric(levels, lower = 0, upper = 1, min.len = 1)
  levels <- sort(levels)
  xy <- x[, c("x_", "y_")]
  mxy <- colMeans(xy)
  sqd <- (xy$x_ - mxy[1])^2 + (xy$y_ - mxy[2])^2
  qts <- stats::quantile(sqd, levels)
  geometry <- lapply(qts, function(i) amt:::chull_mcp(xy[sqd <= i, ]))
  crs <- amt:::get_crs(x)
  geometry <- sf::st_as_sfc(geometry, crs = if (is.na(crs)) sf::NA_crs_ else crs)
  mcps <- sf::st_sf(level = levels, what = "estimate", area = sf::st_area(geometry), geometry)
  mcp <- list(mcp = mcps, levels = levels, estimator = "mcp", crs = crs, data = if (keep.data) x else NULL)
  class(mcp) <- c("mcp", "hr_geom", "hr")
  mcp
}

intensity_use_fixed <- function(x, ...) {
  amt::cum_dist(x) / amt::hr_area(hr_mcp_fixed(x, levels = 1))$area^0.5
}


# Step selection (the cases of tests/move_issf_cases.py): the numbers of a fit_issf() model and of log_rss().
# log_rss(ci = "boot") of amt 0.3.1 has three errors, corrected in log_rss_boot_fixed(): (1) resample_coxph() draws strata
# with replacement, but a stratum drawn twice keeps its step_id_, so the two merge into one stratum with two cases (the
# exact partial likelihood of tied deaths, not a bootstrap of the strata); here each draw is a stratum of its own
# (the ids 1, 2, ...); (2) the model frame it resamples holds log(sl_) as a column, and refitting the formula on it fails
# with "object 'sl_' not found" for any transformation; here the rows of the data are resampled; (3) with one row
# in x1, replicate(simplify = "array") gives a vector and apply() fails. The rest is amt's: the strata are the
# unique ones of the data with no missing value, sample() draws as many as there are cases, a replicate whose refit
# fails is drawn again (49 tries), the interval is mle + (quantile - mean) of the replicates.
issf_out <- function(m, prefix = "") {
  s <- summary(m)$coefficients
  ae <- extractAIC(m)
  o <- list(coef = unname(m$coefficients), var = as.vector(m$var), loglik = m$loglik, score = m$score,
            wald = m$wald.test, iter = m$iter, n = m$n, nevent = m$nevent, means = unname(m$means),
            exp = unname(s[, 2]), se = unname(s[, 3]), z = unname(s[, 4]), p = unname(s[, 5]),
            ci95 = as.vector(confint(m)), ci90 = as.vector(confint(m, level = 0.9)), aic = ae[2], edf = ae[1])
  names(o) <- paste0(prefix, names(o))
  o
}

lr_out <- function(r, prefix = "") {
  o <- list(log_rss = r$df$log_rss, lwr = r$df$lwr, upr = r$df$upr)
  names(o) <- paste0(prefix, names(o))
  o
}

log_rss_boot_fixed <- function(data, formula, x1, x2, ci_level, n_boot, mle) {
  mf <- model.frame(formula, data)
  dat <- data[as.integer(rownames(mf)), , drop = FALSE]
  tl <- attr(terms(formula, specials = "strata"), "term.labels")
  sname <- sub("^strata\\((.+)\\)$", "\\1", grep("^strata\\(.+\\)$", tl, value = TRUE))
  response <- as.character(formula[[2]])
  sid <- dat[[sname]]
  ids <- unique(sid)
  n_obs <- sum(dat[[response]])
  rows <- lapply(ids, function(i) which(sid == i))
  one <- function() {
    i <- 1
    res <- NULL
    while (is.null(res)) {
      draw <- sample.int(length(ids), n_obs, replace = TRUE)
      newdat <- dat[unlist(rows[draw]), , drop = FALSE]
      newdat[[sname]] <- rep(seq_along(draw), lengths(rows[draw]))
      res <- tryCatch({
        m <- suppressWarnings(amt::fit_issf(newdat, formula, model = TRUE))
        amt::log_rss(m, x1, x2, ci = NA)$df$log_rss
      }, error = function(e) NULL)
      i <- i + 1
      if (i == 50) stop("Bootstrapping failed.")
    }
    res
  }
  arr <- matrix(vapply(seq_len(n_boot), function(b) one(), numeric(nrow(x1))), nrow = nrow(x1))
  p_lwr <- (1 - ci_level) / 2
  p_upr <- 1 - p_lwr
  q_lwr <- apply(arr, 1, quantile, p_lwr)
  q_upr <- apply(arr, 1, quantile, p_upr)
  boot_mean <- apply(arr, 1, mean)
  list(log_rss = mle, lwr = mle + (q_lwr - boot_mean), upr = mle + (q_upr - boot_mean))
}


# Simulation from a fitted iSSF (the cases of tests/move_sim_cases.py, group "amt_sim"). amt 0.3.1's
# redistribution_kernel() and simulate_path() in continuous space have these errors, corrected in
# redistribution_kernel_fixed() and simulate_path_fixed(), which are amt's code otherwise (random_steps_simple(),
# extract_covariates(), ssf_weights(), sample.int()):
# (1) the default `fun` only extracts the covariates: log_sl_ and cos_ta_ are missing and model.matrix fails ("object
# 'cos_ta_' not found"); here they are made (sim_fun); (2) the test of the study area counts every candidate, so
# `tolerance.outside = 0` (the default) ends the simulation as soon as one in n.control candidates is outside, though a
# candidate with a missing covariate has no weight and is never drawn; here only the candidates with a weight above 0
# count; (3) simulate_path() makes the start of the next step with the time of the step before (xy$t_[i], not
# xy$t_[i + 1]) and without dt (back to the default of an hour); here the time and dt of the start; (4) the arguments
# are collected with as.list(environment()), which evaluates max.dist = get_max_dist(x): qhnorm, which does not exist,
# makes any model with a half-normal step length fail; here max.dist is not evaluated (continuous space does not use
# it).
sim_fun <- function(xy, map) {
  xy <- amt::extract_covariates(xy, map, where = "both")
  xy$cos_ta_ <- cos(xy$ta_)
  xy$log_sl_ <- log(xy$sl_)
  xy
}

sim_map_a <- function() {
  r <- terra::rast(nrows = 100, ncols = 100, xmin = 0, xmax = 1000, ymin = 0, ymax = 1000)
  rr <- rep(1:100, each = 100)
  cc <- rep(1:100, times = 100)
  terra::values(r) <- sin(cc / 5) + cos(rr / 7)
  names(r) <- "cov"
  r
}

redistribution_kernel_fixed <- function(x, start, map, n.control = 1e+06, n.sample = 1, tolerance.outside = 0) {
  arguments <- as.list(environment())
  xy <- amt:::random_steps_simple(start, sl_model = x$sl_, ta_model = x$ta_, n.control = n.control)
  xy$t1_ <- start$t_
  xy$t2_ <- start$t_ + start$dt
  xy <- sim_fun(xy, map)
  w <- amt:::ssf_weights(xy, x, compensate.movement = FALSE)
  bb.map <- as.vector(terra::ext(map))
  outside <- xy$x2_ < bb.map["xmin"] | xy$x2_ > bb.map["xmax"] | xy$y2_ < bb.map["ymin"] | xy$y2_ > bb.map["ymax"]
  fraction.outside <- mean(outside & w > 0)
  if (fraction.outside > tolerance.outside) {
    warning("candidates outside the study area. Terminating simulations here.")
    return(NULL)
  }
  r <- dplyr::select(xy[sample.int(nrow(xy), size = n.sample, prob = w), ], x_ = x2_, y_ = y2_, t2_)
  res <- list(args = arguments, redistribution.kernel = r)
  class(res) <- c("redistribution_kernel", "list")
  res
}

simulate_path_fixed <- function(x, n.steps = 100, start = x$args$start, ...) {
  mod <- x$args
  dt <- start$dt
  xy <- tibble::tibble(x_ = rep(NA_real_, n.steps + 1), y_ = NA_real_, t_ = start$t_ + dt * (0:n.steps), dt = dt)
  xy$x_[1] <- start$x_
  xy$y_[1] <- start$y_
  for (i in 1:n.steps) {
    rk <- redistribution_kernel_fixed(x = mod$x, start = start, map = mod$map, n.control = mod$n.control,
                                      n.sample = 1, tolerance.outside = mod$tolerance.outside)
    if (is.null(rk)) {
      warning(paste0("Simulation stopped after ", i - 1, " time steps, because the animal stepped out of the landscape."))
      return(xy)
    }
    rk <- rk$redistribution.kernel
    new.ta <- atan2(rk$y_[1] - start$y_[1], rk$x_[1] - start$x_[1])
    xy$x_[i + 1] <- rk$x_[1]
    xy$y_[i + 1] <- rk$y_[1]
    start <- amt::make_start(as.numeric(xy[i + 1, c("x_", "y_")]), new.ta, time = xy$t_[i + 1], dt = dt,
                             crs = attr(x$args$start, "crs"))
  }
  xy
}

# one path of both versions as numbers: the original amt (with its `fun` made here, and a max.dist when given) and
# the corrected one, from the same seed (the kernel first, as R's generator goes on from it); `tag` prefixes the names
sim_out <- function(m, st, map, n.control, tol, n.steps, seed, tag, max.dist = NULL, original = TRUE) {
  op <- options(warn = -1)
  on.exit(options(op))
  o <- list()
  if (original) {
    set.seed(seed)
    k <- if (is.null(max.dist)) {
      amt::redistribution_kernel(m, start = st, map = map, fun = sim_fun, n.control = n.control, tolerance.outside = tol)
    } else {
      amt::redistribution_kernel(m, start = st, map = map, fun = sim_fun, max.dist = max.dist, n.control = n.control,
                                 tolerance.outside = tol)
    }
    o$orig_null <- as.numeric(is.null(k))
    if (!is.null(k)) {
      o$orig_first <- c(k$redistribution.kernel$x_, k$redistribution.kernel$y_)
      o$orig_unif <- runif(1)
      set.seed(seed)
      k <- if (is.null(max.dist)) {
        amt::redistribution_kernel(m, start = st, map = map, fun = sim_fun, n.control = n.control, tolerance.outside = tol)
      } else {
        amt::redistribution_kernel(m, start = st, map = map, fun = sim_fun, max.dist = max.dist, n.control = n.control,
                                   tolerance.outside = tol)
      }
      s <- suppressWarnings(amt::simulate_path(k, n.steps = n.steps))
      o$orig_x <- s$x_
      o$orig_y <- s$y_
      o$orig_t <- as.numeric(s$t_)
      o$orig_dt <- as.numeric(s$dt, "seconds")
      o$orig_unif2 <- runif(1)
    }
  }
  set.seed(seed)
  k <- redistribution_kernel_fixed(m, st, map, n.control = n.control, tolerance.outside = tol)
  o$fixed_null <- as.numeric(is.null(k))
  if (!is.null(k)) {
    o$fixed_first <- c(k$redistribution.kernel$x_, k$redistribution.kernel$y_)
    o$fixed_unif <- runif(1)
    set.seed(seed)
    k <- redistribution_kernel_fixed(m, st, map, n.control = n.control, tolerance.outside = tol)
    s <- suppressWarnings(simulate_path_fixed(k, n.steps = n.steps))
    o$fixed_x <- s$x_
    o$fixed_y <- s$y_
    o$fixed_t <- as.numeric(s$t_)
    o$fixed_unif2 <- runif(1)
  }
  names(o) <- paste0(tag, names(o))
  o
}
