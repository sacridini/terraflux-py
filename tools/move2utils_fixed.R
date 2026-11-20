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
