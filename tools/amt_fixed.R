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
