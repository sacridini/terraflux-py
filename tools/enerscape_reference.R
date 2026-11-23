# SPDX-License-Identifier: GPL-3.0-or-later
#
# The reference values of tests/test_cost_energy.py: R's enerscape 1.2.0 (R 4.4.2, terra, Rcpp 1.1.1, OMP_NUM_THREADS=1)
# on a small synthetic DEM and on the package's own DEM (Monte Sirente). Each matrix is written with 17 significant
# digits (exact doubles), row 0 = north, NA as "NA".
#
#   Rscript tools/enerscape_reference.R out_dir

suppressMessages({
  library(terra)
  library(enerscape)
})

out <- commandArgs(TRUE)[1]
dir.create(out, showWarnings = FALSE, recursive = TRUE)

write_matrix <- function(m, name) {
  s <- matrix(sprintf("%.17g", m), nrow(m), ncol(m))
  s[is.na(m)] <- "NA"
  write.table(s, file.path(out, paste0(name, ".csv")), sep = ",", quote = FALSE, row.names = FALSE, col.names = FALSE)
}
as_matrix <- function(r) matrix(values(r), nrow = nrow(r), ncol = ncol(r), byrow = TRUE)

# A) 15 x 20 cells of 30 m, UTM 32N. The heights are written once and read back (sin and cos of the generator come
# from the platform's libm: the CSV is the input of the tests).
nr <- 15; nc <- 20
i <- matrix(rep(1:nr, each = nc), nr, nc, byrow = TRUE)
j <- matrix(rep(1:nc, nr), nr, nc, byrow = TRUE)
z <- 100 + 40 * sin(i / 3) + 25 * cos(j / 4) + 0.5 * i * j / 10
write_matrix(z, "small_dem")
dem <- rast(z, crs = "EPSG:32632", extent = ext(500000, 500600, 5000000, 5000450))
write_matrix(as_matrix(enerscape(dem, 50, unit = "joule", neigh = 8)), "small_m50_j_n8_out")
write_matrix(as_matrix(enerscape(dem, 50, unit = "joule", neigh = 4)), "small_m50_j_n4_out")
write_matrix(as_matrix(enerscape(dem, 50, unit = "kcal", neigh = 8)), "small_m50_kcal_n8_out")
write_matrix(as_matrix(humanscape(dem, 70, v = 1.39, unit = "joule", neigh = 8)), "small_human_m70_v139_j_n8_out")

# B) Monte Sirente, 393 x 594 cells of 10 m, UTM 32N, no NA.
data("sirente")
dem <- rast(sirente, crs = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs",
            extent = ext(879340, 885280, 4672880, 4676810))
write_matrix(as_matrix(dem), "sirente_dem")
write_matrix(as_matrix(enerscape(dem, 10, neigh = 4, unit = "kcal")), "sirente_m10_kcal_n4_out")
write_matrix(as_matrix(enerscape(dem, 10, neigh = 8, unit = "joule")), "sirente_m10_j_n8_out")
