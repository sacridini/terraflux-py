// SPDX-License-Identifier: GPL-3.0-or-later
//
// Earth mover's distance between utilisation distributions (ls.move), as move2utils' emd() computes it: the cells
// of each UD inside the `mask_quantile` volume contour (.emd_prep_ud), the cost between the cells' centres
// (Euclidean, or haversine on 6378137 m with geosphere::distHaversine's arithmetic), and either Sinkhorn's
// iterations (.emd_sinkhorn, the arithmetic of R's reference BLAS dgemv: the same result up to the order of the
// sums that other BLAS libraries change; here it is R's own order) or Rubner's exact solver (emdist::emd, vendor/emdist).
#pragma once

#include <cstdint>
#include <vector>

namespace move {

using i64 = std::int64_t;

// .emd_prep_ud on the values of a UD (NaN: no data): the cells kept (ascending) and their weights, renormalised
// to sum 1. Throws std::invalid_argument when the total mass is zero or not finite.
void emd_prepare(const double* v, i64 n, double mask_quantile, std::vector<i64>& cells, std::vector<double>& weights);

// .emd_cost_matrix cell (i, j): the distance between (ax, ay) and (bx, by), clipped at `threshold` (NaN: none).
// gc: haversine in metres on the coordinates in degrees, else Euclidean.
double emd_cost(double ax, double ay, double bx, double by, bool gc, double threshold);

// .emd_sinkhorn for the cells (ax, ay) with weights p (na) and (bx, by) with weights q (nb). `iterations`
// receives the number of iterations made.
double emd_sinkhorn(const double* ax, const double* ay, const double* p, i64 na, const double* bx, const double* by,
                    const double* q, i64 nb, bool gc, double threshold, double reg, int max_iter, double tol,
                    int threads, int* iterations);

// .emd_exact: emdist::emd(cbind(weights, coords), ...) with its Euclidean cost, the weights and coordinates in
// float. `hit_max_iter` is set when the solver stopped at `max_iter` (R warns).
double emd_exact(const double* ax, const double* ay, const double* p, i64 na, const double* bx, const double* by,
                 const double* q, i64 nb, int max_iter, bool* hit_max_iter);

}  // namespace move
