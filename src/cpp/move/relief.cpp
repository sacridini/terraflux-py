// SPDX-License-Identifier: GPL-3.0-or-later
//
// The length of a step measured on the relief (tf.move.surface_distance).

#include "move/relief.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace move {

namespace {

using i64 = std::int64_t;

struct Grid {
    const double* z;
    i64 nx, ny;
    double xmin, ymax, dx, dy;
};

// Bilinear height at (x, y): NaN outside the extent or with a NaN among the 4 cells.
double height(const Grid& g, double x, double y) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double xmax = g.xmin + static_cast<double>(g.nx) * g.dx;
    const double ymin = g.ymax - static_cast<double>(g.ny) * g.dy;
    if (!(x >= g.xmin && x <= xmax && y >= ymin && y <= g.ymax)) return nan;
    // position in cell units from the first centre; clamped to the outermost centres (nearest edge cell)
    double fx = (x - g.xmin) / g.dx - 0.5;
    double fy = (g.ymax - y) / g.dy - 0.5;
    fx = std::min(std::max(fx, 0.0), static_cast<double>(g.nx - 1));
    fy = std::min(std::max(fy, 0.0), static_cast<double>(g.ny - 1));
    i64 c0 = static_cast<i64>(std::floor(fx));
    i64 r0 = static_cast<i64>(std::floor(fy));
    c0 = std::min(c0, g.nx - 1);
    r0 = std::min(r0, g.ny - 1);
    const double tx = fx - static_cast<double>(c0);
    const double ty = fy - static_cast<double>(r0);
    const i64 c1 = std::min(c0 + 1, g.nx - 1);
    const i64 r1 = std::min(r0 + 1, g.ny - 1);
    const double a = g.z[r0 * g.nx + c0], b = g.z[r0 * g.nx + c1];
    const double c = g.z[r1 * g.nx + c0], d = g.z[r1 * g.nx + c1];
    if (std::isnan(a) || std::isnan(b) || std::isnan(c) || std::isnan(d)) return nan;
    const double top = a + (b - a) * tx;
    const double bot = c + (d - c) * tx;
    return top + (bot - top) * ty;
}

double one_step(const Grid& g, double x1, double y1, double x2, double y2, double spacing) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!(std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2))) return nan;
    const double len = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    double zp = height(g, x1, y1);
    if (std::isnan(zp)) return nan;
    if (len == 0.0) return 0.0;
    const double nseg_d = std::ceil(len / spacing);
    const i64 nseg = nseg_d < 1.0 ? 1 : static_cast<i64>(nseg_d);
    double xp = x1, yp = y1, sum = 0.0;
    for (i64 k = 1; k <= nseg; ++k) {
        double xq = x2, yq = y2;
        if (k < nseg) {
            const double t = static_cast<double>(k) * spacing / len;
            xq = x1 + (x2 - x1) * t;
            yq = y1 + (y2 - y1) * t;
        }
        const double zq = height(g, xq, yq);
        if (std::isnan(zq)) return nan;
        const double ex = xq - xp, ey = yq - yp, ez = zq - zp;
        sum += std::sqrt(ex * ex + ey * ey + ez * ez);
        xp = xq;
        yp = yq;
        zp = zq;
    }
    return sum;
}

}  // namespace

void surface_distance(const double* x1, const double* y1, const double* x2, const double* y2, i64 n, const double* z,
                      i64 nx, i64 ny, double xmin, double ymax, double dx, double dy, double spacing, int threads,
                      double* out) {
    const Grid g{z, nx, ny, xmin, ymax, dx, dy};
    if (!(spacing > 0.0)) spacing = 0.25 * std::min(dx, dy);
    if (threads < 1) threads = 1;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 i = 0; i < n; ++i) out[i] = one_step(g, x1[i], y1[i], x2[i], y2[i], spacing);
}

}  // namespace move
