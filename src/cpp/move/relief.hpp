// SPDX-License-Identifier: GPL-3.0-or-later
//
// The length of a step measured on the relief (tf.move.surface_distance): the 3D length of the segment over a DEM.
#pragma once

#include <cstdint>

namespace move {

// For each of the n steps (x1, y1) -> (x2, y2), the sum of the 3D chords of the points along the segment every
// `spacing` (the last, shorter, ends exactly at the end), the height of each point by bilinear interpolation between
// the centres of the cells of the grid z (ny rows from the top, nx columns, row-major; NaN is no data). The grid's
// top left corner is (xmin, ymax), the cells dx wide and dy high. A point in the half cell outside the outermost
// centres takes the nearest edge cell (the value is clamped); outside the extent, or with any of the 4 cells NaN, the
// height is NaN and so is the step. spacing <= 0: a quarter of the smaller cell side. Steps are independent: the
// result does not depend on threads. Reentrant (no global state).
void surface_distance(const double* x1, const double* y1, const double* x2, const double* y2, std::int64_t n,
                      const double* z, std::int64_t nx, std::int64_t ny, double xmin, double ymax, double dx, double dy,
                      double spacing, int threads, double* out);

}  // namespace move
