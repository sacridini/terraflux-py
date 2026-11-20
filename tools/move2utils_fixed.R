# SPDX-License-Identifier: GPL-3.0-or-later
# move2utils 0.4.6 with its errors corrected, the references tf.move is tested against.
#
# mt_filter_gps_quality: its find_col() tries gps_satellite_count, gps-satellite-count (and the other two
# columns likewise), but Movebank's own names, which move2::mt_read keeps, are gps:satellite-count, gps:dop
# and eobs:horizontal-accuracy-estimate: on them it finds no column and returns the tracks unchanged (the
# example of its manual does nothing). mt_filter_gps_quality_fixed also tries those. The rest is move2utils'
# code.
#
# mt_flag_speed_cap: .clean_track_dispatch lifts every result of length 1 to an attribute, meant for the
# scalars (v_max_used, threshold_type), but a track with one fix to score gives vectors of length 1 as well:
# is_outlier, step_speed and is_speed_above_cap are lost as columns, the result of that track has fewer
# columns than the others and the rbind of a set of tracks fails ("numbers of columns of the arguments do
# not match"). mt_flag_speed_cap_fixed lifts by name: the scalars to attributes, the rest to columns. The
# rest is move2utils' code (the functions are move2utils' own, patched, run in an environment where the
# patched copies are found first).

mt_filter_gps_quality_fixed <- function(x, sat_min = 5, dop_max = 10, hacc_max = 100, drop_empty = TRUE,
                                        verbose = FALSE) {
  src <- paste(deparse(move2utils::mt_filter_gps_quality, width.cutoff = 500L), collapse = "\n")
  from <- "        NA_character_\n    }"
  to <- paste0("        colon <- sub(\"^([a-z]+)-\", \"\\\\1:\", dashed)\n",
               "        if (colon %in% names(x)) return(colon)\n", from)
  stopifnot(grepl(from, src, fixed = TRUE))
  f <- eval(parse(text = sub(from, to, src, fixed = TRUE)))
  environment(f) <- asNamespace("move2utils")
  f(x, sat_min = sat_min, dop_max = dop_max, hacc_max = hacc_max, drop_empty = drop_empty, verbose = verbose)
}

.speedcap_fixed_env <- local({
  ns <- asNamespace("move2utils")
  env <- new.env(parent = ns)
  patch <- function(fn, from, to) {
    src <- paste(deparse(fn, width.cutoff = 500L), collapse = "\n")
    stopifnot(grepl(from, src, fixed = TRUE))
    f <- eval(parse(text = sub(from, to, src, fixed = TRUE)))
    environment(f) <- env
    f
  }
  env$.clean_track_dispatch <- patch(
    ns$.clean_track_dispatch,
    "if (length(val) == 1L) {",
    "if (length(val) == 1L && nm %in% c(\"v_max_used\", \"threshold_type\")) {")
  env$mt_flag_speed_cap <- {
    f <- ns$mt_flag_speed_cap
    environment(f) <- env
    f
  }
  env
})

mt_flag_speed_cap_fixed <- function(x, ...) .speedcap_fixed_env$mt_flag_speed_cap(x, ...)

# mt_corridor: (1) in longitude / latitude it projects to move2::mt_aeqd_crs(x_kept), which centres the projection on the
# centroid moved to EPSG:4236 (a typo for 4326) and so about a kilometre off (mt_aeqd_crs_fixed, tools/move2_fixed.R,
# centres it where meant); (2) it takes the search radius, half the segment's length, from mt_distance(x, units = "m")
# and gives the number to st_buffer() as if it were in the CRS's units: in a CRS in feet the neighbourhoods are 3.28
# times too small (and in kilometres 1000 times too large); mt_corridor_fixed takes the radius in the CRS's own
# units. The rest is move2utils' code.
mt_corridor_fixed <- function(x, ...) {
  src <- paste(deparse(move2utils::mt_corridor, width.cutoff = 500L), collapse = "\n")
  from1 <- "move2::mt_aeqd_crs(x_kept)"
  from2 <- "seg_radius_m <- seg_length_m/2"
  to2 <- paste0("seg_radius_m <- if (isTRUE(sf::st_is_longlat(x_kept))) seg_length_m/2 else ",
                "as.numeric(move2::mt_distance(x_kept))[is_seg]/2")
  stopifnot(grepl(from1, src, fixed = TRUE), grepl(from2, src, fixed = TRUE))
  src <- sub(from1, "mt_aeqd_crs_fixed(x_kept)", src, fixed = TRUE)
  src <- sub(from2, to2, src, fixed = TRUE)
  f <- eval(parse(text = src))
  environment(f) <- asNamespace("move2utils")
  f(x, ...)
}

# The dynamic Brownian bridge (mt_dbbmm_variance, mt_dbbmm_ud, ...).
#
# mt_aeqd_crs: move2utils calls move2::mt_aeqd_crs (.ensure_projected, mt_dbbmm_ud.move2), which centres the
# projection on a point transformed to EPSG:4236 where 4326 is meant (see move2_fixed.R). with_fixed_aeqd(expr)
# runs expr with move2's own function replaced by the corrected one and puts it back after. The rest is move2utils'
# code.
#
# dbbmm_ud_out(u) / dbbmm_var_out(o): the numbers of a UD raster and of the variance objects, in the order the
# tests read them.

with_fixed_aeqd <- function(expr) {
  ns <- asNamespace("move2")
  orig <- get("mt_aeqd_crs", envir = ns)
  unlockBinding("mt_aeqd_crs", ns)
  assign("mt_aeqd_crs", mt_aeqd_crs_fixed, envir = ns)
  on.exit({
    assign("mt_aeqd_crs", orig, envir = ns)
    lockBinding("mt_aeqd_crs", ns)
  })
  expr
}

project_center <- function(x, center = "center") sf::st_transform(x, mt_aeqd_crs_fixed(x, center))

dbbmm_var_out <- function(o) {
  vl <- if (inherits(o, "mt_dbbmm_variance")) list(o) else o
  td <- function(f) unlist(lapply(vl, function(v) f(v$track_data)))
  list(variance = unlist(lapply(vl, `[[`, "variance")), in_windows = unlist(lapply(vl, `[[`, "in_windows")),
       interest = as.numeric(unlist(lapply(vl, `[[`, "interest"))), le = td(function(d) d$location_error),
       n = vapply(vl, function(v) as.numeric(v$track_data$n_locs), numeric(1)),
       nbreaks = vapply(vl, function(v) as.numeric(length(v$break_list)), numeric(1)),
       breaks = as.numeric(unlist(lapply(vl, `[[`, "break_list"))),
       x = td(function(d) d$x), y = td(function(d) d$y))
}

dbbmm_ud_out <- function(u) {
  list(ud = as.numeric(terra::values(u)), dim = c(terra::nrow(u), terra::ncol(u), terra::nlyr(u)),
       ext = as.vector(terra::ext(u)))
}
