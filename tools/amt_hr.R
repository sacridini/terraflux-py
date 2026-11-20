# SPDX-License-Identifier: GPL-3.0-or-later
# Helpers of the amt home-range cases (tests/move_cases.py, group "amt_hr"; run by tools/move_reference.R).
#
# amt 0.3.1's home-range functions are used as they are (their errors are described in tests/test_move_hr.py:
# hr_locoh's crs branch, hr_isopleths' ignored `descending`, hr_overlap's `x$levels`).

# A POLYGON or MULTIPOLYGON as numbers: the number of polygons, then for each its number of rings, and for each
# ring its number of vertices followed by the x and the y of the vertices.
hr_enc <- function(g) {
  g <- sf::st_geometry(g)[[1]]
  polys <- if (inherits(g, "MULTIPOLYGON")) unclass(g) else list(unclass(g))
  out <- length(polys)
  for (p in polys) {
    out <- c(out, length(p))
    for (r in p) out <- c(out, nrow(r), r[, 1], r[, 2])
  }
  out
}

# The geometries of the rows of an sf table, one after the other.
hr_enc_all <- function(d) unlist(lapply(seq_len(nrow(d)), function(i) hr_enc(d[i, ])))

# The template of all the locations of x together (amt's default is one per track).
hr_pooled <- function(x, ...) {
  xy <- sf::st_coordinates(x)
  crs <- sf::st_crs(x)
  tr <- amt::make_track(data.frame(x = xy[, 1], y = xy[, 2], t = move2::mt_time(x)), x, y, t,
                        crs = if (is.na(crs)) NA else crs)
  amt::make_trast(tr, ...)
}

# Columns, rows, extent and resolution of a raster.
hr_grid <- function(r) c(dim(r)[2], dim(r)[1], as.vector(terra::ext(r)), terra::res(r))

hr_values <- function(r) as.numeric(terra::values(r)[, 1])

# A result or NULL when the call fails.
hr_safe <- function(expr) tryCatch(suppressWarnings(expr), error = function(e) NULL)

# amt's functions by their names (amt is not attached: the other cases write amt:: and its functions mask stats::filter)
for (.f in c("make_trast", "hr_mcp", "hr_kde", "hr_kde_ref", "hr_kde_ref_scaled", "hr_kde_pi", "hr_kde_lscv", "hr_locoh",
             "hr_isopleths", "hr_area", "hr_ud", "hr_cud", "hr_overlap", "intensity_use")) {
  assign(.f, getExportedValue("amt", .f))
}
rm(.f)
