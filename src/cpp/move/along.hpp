// SPDX-License-Identifier: GPL-3.0-or-later
//
// The cells a step crosses, as terra 1.9-50 extracts them for a line (amt's extract_covariates_along:
// terra::extract(r, vect(LINESTRING (x1 y1, x2 y2))) with its defaults, touches = TRUE, small = TRUE), whose line
// walk is GDAL 3.9.3's ALL_TOUCHED GDALdllImageLineAllTouched on the grid terra crops to the line's extent. The
// steps run in parallel, each on its own, so the result does not depend on the number of threads.
#pragma once

#include <cstdint>
#include <vector>

namespace move {

using i64 = std::int64_t;

struct AlongCells {
    std::vector<i64> offsets;  // n + 1: step i has the cells [offsets[i], offsets[i + 1])
    // terra's cell numbers (0-based, row-major of the whole raster) in terra's order (the cropped grid's row-major
    // order, which is ascending cell number); -1: terra's one NA row (a step outside the raster)
    std::vector<i64> cell;
    // rank[k]: position (0 = first) of cell k of the step along the step, from (x1, y1) to (x2, y2)
    std::vector<i64> rank;
};

// strtod of "%.15g": the value that R's paste0 (as.character, 15 significant digits) gives to a double.
double r_fmt15(double x);

// Steps (x1, y1) -> (x2, y2), i = 0..n-1, on the raster with terra's extent [xmin, xmax] x [ymin, ymax] and nx x ny
// cells. round15: the coordinates go through r_fmt15 first (amt's paste0). A step with a non-finite coordinate
// gives the NA row.
AlongCells along_cells(const double* x1, const double* y1, const double* x2, const double* y2, i64 n, double xmin,
                       double xmax, double ymin, double ymax, i64 nx, i64 ny, bool round15, int threads);

// Summaries of the values `v` (one raster, row-major, ny * nx; -1 cells give NaN) along each step, six numbers a
// step in `out` (n * 6): the mean (R's, in terra's order of the cells), minimum and maximum of the cells that are
// not NaN, then, with those cells in the order along the step: the ascent (the sum of the positive differences
// between successive values), the descent (the sum of the negative ones, as a positive number) and the delta
// (|last - first|). All NaN when no cell has a value.
void along_stats(const AlongCells& cells, const double* v, i64 nv, int threads, double* out);

}  // namespace move
