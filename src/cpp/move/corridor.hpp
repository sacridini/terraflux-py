// SPDX-License-Identifier: GPL-3.0-or-later
//
// Corridor detection (ls.move), as move2utils' mt_corridor computes it (LaPoint et al. 2013): the neighbourhood of
// a segment is the set of segment midpoints, of all tracks together, that sf's st_intersects finds in the buffer
// of its own midpoint of half the segment's length. sf's buffer of a point is GEOS' (BufferOp, CAP_ROUND, 30
// segments to the quadrant): a convex polygon of 120 vertices, not a circle; its vertices and the test of a
// point against it (GEOS' RayCrossingCounter with the robust orientation of CGAlgorithmsDD: the boundary
// counts) are reproduced here, so a midpoint between the polygon and the circle falls on the same side as in R.
// The segments run in parallel, each on its own: the result does not depend on the number of threads.
#pragma once

#include <cstdint>
#include <vector>

namespace move {

using i64 = std::int64_t;

// The 120 vertices GEOS' buffer of the point (cx, cy) at distance r > 0 has (without the closing one), in its
// order (clockwise, from (cx + r, cy)): x = cx + r * cos(-k * a), y = cy + r * sin(-k * a), a = 2 pi / 120.
constexpr int kBufferVertices = 120;
void buffer_vertices(double cx, double cy, double r, double* vx, double* vy);

// GEOS' Location of the point (px, py) in the ring of n vertices (closed implicitly): the point is in the polygon
// or on its boundary.
bool ring_intersects(const double* vx, const double* vy, int n, double px, double py);

// The neighbourhoods of n segments from (x0, y0) to (x1, y1) in the working CRS, with the radius of each buffer
// (half the segment's length; a radius <= 0 or NaN has an empty buffer, so no neighbour). CSR lists: the
// neighbours of segment i (indices of segments, ascending as sf returns them, the segment itself included when
// its radius is positive) are index[offset[i] .. offset[i + 1]).
//
// azimuth: atan2(x1 - x0, y1 - y0) (radians); circvar: 1 - |R| of the pseudo-azimuths ((2 * azimuth) %% (2 pi)) of
// the neighbours, NaN for fewer than 2 neighbours (.circ_var_per_neighbourhood).
struct Neighbourhoods {
    std::vector<double> azimuth;
    std::vector<double> circvar;
    std::vector<i64> offset;
    std::vector<std::int32_t> index;
};
Neighbourhoods corridor_neighbourhoods(const double* x0, const double* y0, const double* x1, const double* y1,
                                       const double* radius, i64 n, int threads);

// is_corridor: n_neighbours >= min_segments and strictly more candidates than non-candidates among the neighbours
// (a segment with no neighbour is not a corridor). candidate: one 0/1 per segment.
void corridor_flags(const i64* offset, const std::int32_t* index, const unsigned char* candidate, i64 n,
                    double min_segments, int threads, unsigned char* corridor);

}  // namespace move
