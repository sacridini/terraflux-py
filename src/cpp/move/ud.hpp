// SPDX-License-Identifier: GPL-3.0-or-later
//
// The utilization distribution of the dynamic Brownian bridge (ls.move), as move2utils' dbbmm2_omp and
// ud_grid_probe compute it: the bridge between two locations is walked in time steps, each spreading a bivariate
// normal over the cells of a window of `ext` standard deviations around the bridge mean, the densities of the
// steps added cell by cell.
//
// The grid vectors are the centres of the columns (`xgrid`, ascending) and of the rows (`ygrid`, ascending: from
// the bottom row up); the cell size is read from their first two values, as the kernels do. The density sums
// are in R's layout, `ud[i * ny + j]` with i the column and j the row counted from the top, so that R's sum() over
// the matrix can be repeated in its order. The same walk of the time steps serves the probe, which says how far
// the windows reach, and the kernel; a second kernel with other windows (the directional bridge) can take its
// place in Walker::next without touching them.
#pragma once

#include <cstdint>
#include <vector>

namespace move {

using i64 = std::int64_t;

// The inputs of a kernel: n locations (x, y), the bridge variance after each (s1; a second, s2, for a kernel with
// two), their times t (minutes, from the first), the location errors le (standard deviations), the segments
// `interest`ing (1 byte per location: the bridge from i to i + 1), the time step and the window half-width in
// standard deviations.
enum class UdModel { Dbbmm = 0, Dbgb = 1 };

struct UdInput {
    const double* x;
    const double* y;
    const double* s1;
    const double* s2;
    const double* t;
    const double* le;
    const std::uint8_t* interest;
    i64 n;
    double time_step;
    double ext;
    UdModel model = UdModel::Dbbmm;   // Dbbmm: s1 is the variance, one circular bridge; Dbgb: s1, s2 the standard
                                      // deviations along and across the direction to the next location
};

// ud_grid_probe for the dBBMM: the envelope of the windows (min x - w, max x + w, min y - w, max y + w, +/-Inf
// when no step is walked) and the index extremes on the grid (min xStart, max xEnd, max yEnd, min yStart; as
// doubles, INT_MAX / INT_MIN when no step is walked). All 4 + 4 values are the probe's.
struct UdProbe {
    double env[4];
    double idx[4];
};
UdProbe ud_probe(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny);

// bgb_omp: the same for the directional bridge (in.model = Dbgb; `x`, `y`, `t` of the core locations, `t` as given).
void dbgb_ud(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny, int threads, double* ud);

// dbbmm2_omp: the sum of the densities of the steps (times the cell area) in `ud` (nx * ny, R's layout), on
// `threads` threads in strips of columns, every cell adding its steps in their order. Throws std::runtime_error
// when a window leaves the grid (the equality at the far edges, which R's kernel lets through to read outside the
// grid, counts as leaving it) and std::invalid_argument for inputs the kernel cannot walk.
void dbbmm_ud(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny, int threads, double* ud);

// R's sum() of the values in their order (extended precision), the error of the zero or non-finite total
// (std::runtime_error), then v / total. Returns the total.
double ud_normalise(double* v, i64 n);

// R's layout (nx * ny, i * ny + j) to the raster's rows (row j from the top, then column i).
void ud_to_rows(const double* v, i64 nx, i64 ny, double* rows);

// ud_volume of one layer in raster order: each cell's cumulative probability when the cells are taken from the
// largest value down (ties in the order of the cells; R's order() and cumsum()). NaN cells stay NaN; a layer
// whose total is not positive and finite is copied as it is.
void ud_volume(const double* v, i64 n, double* out);

}  // namespace move
