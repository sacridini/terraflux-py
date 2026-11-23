# Reference values of `tf.cost.energy_landscape`

Made by `tools/enerscape_reference.R` with R 4.4.2, enerscape 1.2.0 (GPL-3, Berti et al. 2022), terra and Rcpp 1.1.1 on
Windows, `OMP_NUM_THREADS=1`. The matrices have row 0 at the north, 17 significant digits, `NA` for missing.

- Small DEM (15 x 20 cells of 30 m, EPSG:32632, origin 500000 / 5000450): `small_dem.csv` and the results of
  `enerscape(dem, 50, "joule", 8)`, `enerscape(dem, 50, "joule", 4)`, `enerscape(dem, 50, "kcal", 8)` and
  `humanscape(dem, 70, v = 1.39, "joule", 8)` (`small_*_out.csv`).
- `sirente.npz`: enerscape's own DEM (Monte Sirente, 393 x 594 cells of 10 m, EPSG:32632, origin 879340 / 4676810;
  `dem` as float32, exact) and the results of `enerscape(dem, 10, neigh = 8, unit = "joule")` (`j_n8`) and
  `enerscape(dem, 10, neigh = 4, unit = "kcal")` (`kcal_n4`), float64.

The ground truth is the output of the package's compiled code, not an R re-implementation: R's vectorised arithmetic
differs from it in the last bit in rare cases.
