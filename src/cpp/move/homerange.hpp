// SPDX-License-Identifier: GPL-3.0-or-later
//
// amt 0.3.1's home ranges (ls.move): the numbers of R's grDevices::chull, KernSmooth's bkde2D, dpik and linbin
// (with R's own fft: Singleton's mixed-radix code, here for the power-of-two lengths they use), the least-squares
// cross-validation of hr_kde_lscv, the neighbourhoods of hr_locoh and the contour lines of terra::as.contour (R's contourLines),
// all in R's arithmetic (long double sums, the x87 exp, log and pow of R's Windows build), so the same bits.
// Nothing here depends on the number of threads: each result is summed in a fixed order.
#pragma once

#include <cstdint>
#include <vector>

namespace move {

using i64 = std::int64_t;

// R's fft() of an n1 x n2 complex array z (re, im interleaved, column-major; n2 = 1 for a vector), unnormalised:
// the lengths must be powers of two (or 1).
void r_fft(double* z, i64 n1, i64 n2, bool inverse);

// R's seq(a, b, length.out = m): a + i * by, the ends exact.
void r_seq_len(double a, double b, i64 m, double* out);

// R's var() and sd() of a vector (the mean refined in long double, the squares summed in long double).
double r_var_of(const double* x, i64 n);

// colMeans of one column: the long double sum divided in long double.
double r_colmean_of(const double* x, i64 n);

// quantile(x, probs) type 7 (x unsorted, copied).
void r_quantile7(const double* x, i64 n, const double* probs, i64 np, double* out);

// R's dnorm(x) of the standard normal.
double r_dnorm(double x);

// grDevices::chull of the n points (x, y): the 0-based indices of the hull's vertices in R's order (the C code's
// clockwise, the first of duplicate points, then R's sort by the angle around the vertices' mean: atan2(y, -x));
// returns their number.
i64 r_chull(const double* x, const double* y, i64 n, i64* out);

// KernSmooth::linbin (a univariate linear binning on m points from a to b; `truncate`: observations outside the
// grid are left out) and linbin2D (observations outside the mesh are ignored): the bin counts.
void kern_linbin(const double* x, i64 n, double a, double b, i64 m, bool truncate, double* gcnts);
void kern_linbin2d(const double* x, const double* y, i64 n, double a1, double a2, double b1, double b2, i64 m1, i64 m2,
                   double* gcnts);

// KernSmooth::bkde2D(cbind(x, y), bandwidth = c(h1, h2), gridsize = c(m1, m2), range.x = list(c(a1, b1), c(a2, b2)))$fhat:
// m1 x m2, the first index fastest. Returns 0, or 1 when the binning grid is too coarse for the bandwidth (R warns).
int kern_bkde2d(const double* x, const double* y, i64 n, double h1, double h2, double a1, double b1, double a2,
                double b2, i64 m1, i64 m2, double* fhat);

// KernSmooth::dpik. scalest: 0 "minim", 1 "stdev", 2 "iqr"; kernel: 0 normal, 1 box, 2 epanech, 3 biweight,
// 4 triweight. Returns NaN with `error` set (1: level, 2: the scale estimate is zero) when R stops.
double kern_dpik(const double* x, i64 n, int scalest, int level, int kernel, bool canonical, i64 gridsize, double a,
                 double b, bool truncate, int* error);

// amt's lscv(cbind(x, y), hs): the criterion of each bandwidth.
void amt_lscv(const double* x, const double* y, i64 n, const double* hs, i64 nh, int threads, double* out);

// The area of a ring of n vertices (the last equal to the first) as GDAL's OGRSimpleCurve::get_LinearArea gives it,
// which is what sf::st_area returns: 0.5 * |sum of x[i] * (y[i + 1] - y[i - 1])|, the vertices cyclic. xy: x, y pairs.
double ogr_ring_area(const double* xy, i64 n);

// hr_cud: the cumulative distribution of the cells of v (each divided by the sum first), cells in their order.
void amt_cud(const double* v, i64 n, double* out);

// amt's vol_base for two UDs on the same grid (R's long double sums): type 0 phr, 1 vi, 2 ba, 3 udoi, 4 hd.
double amt_vol_base(const double* ui, const double* uj, i64 n, int type);

// hr_locoh's neighbourhoods: the 0-based neighbours of each point (not itself, except for types r and a where the
// point is among them), flattened with the offsets `off` (n + 1 values). type: 0 k (the k nearest by FNN), 1 r
// (distance <= d), 2 a (the points whose cumulative distance, nearest first, is <= d).
void amt_locoh_neighbours(const double* x, const double* y, i64 n, int type, double d, int threads,
                          std::vector<i64>& off, std::vector<i64>& ids);

// terra::as.contour at one level: R's grDevices::contourLines (src/main/contour-common.h) of the raster v (nr rows
// from the top, nc columns; cells of width dx and height dy, the top left corner at (xmin, ymax)). The lines in
// R's order, each as x0, y0, x1, y1... Returns false when all the values are equal (R warns and returns nothing).
bool r_contour_lines(const double* v, i64 nr, i64 nc, double xmin, double ymax, double dx, double dy, double level,
                     std::vector<std::vector<double>>& lines);

}  // namespace move
