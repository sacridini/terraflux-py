// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/corridor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "move/geo.hpp"
#include "move/track.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793;  // R's pi

// ------------------------------------------------------------------------------------------------ GEOS' DD
// The double-double of GEOS' DD.cpp (JTS' DD.java), as far as CGAlgorithmsDD::orientationIndex uses it.
struct DD {
    double hi = 0, lo = 0;
    DD() = default;
    explicit DD(double x) : hi(x) {}
    DD(double h, double l) : hi(h), lo(l) {}
    DD operator-() const { return {-hi, -lo}; }
    DD operator+(const DD& y) const {
        double S = hi + y.hi;
        double T = lo + y.lo;
        double e = S - hi;
        double f = T - lo;
        double s = S - e;
        double t = T - f;
        s = (y.hi - e) + (hi - s);
        t = (y.lo - f) + (lo - t);
        e = s + T;
        const double H = S + e;
        const double h = e + (S - H);
        e = t + h;
        const double zhi = H + e;
        const double zlo = e + (H - zhi);
        return {zhi, zlo};
    }
    DD operator-(const DD& y) const { return *this + (-y); }
    DD operator*(const DD& y) const {
        constexpr double kSplit = 134217729.0;  // 2^27 + 1
        double C = kSplit * hi;
        double hx = C - hi;
        double c = kSplit * y.hi;
        hx = C - hx;
        const double tx = hi - hx;
        double hy = c - y.hi;
        C = hi * y.hi;
        hy = c - hy;
        const double ty = y.hi - hy;
        c = ((((hx * hy - C) + hx * ty) + tx * hy) + tx * ty) + (hi * y.lo + lo * y.hi);
        const double zhi = C + c;
        hx = C - zhi;
        const double zlo = c + hx;
        return {zhi, zlo};
    }
    int signum() const {
        if (hi > 0) return 1;
        if (hi < 0) return -1;
        if (lo > 0) return 1;
        if (lo < 0) return -1;
        return 0;
    }
};

int sign_of(double d) { return d < 0 ? -1 : (d > 0 ? 1 : 0); }

// CGAlgorithmsDD::orientationIndex(p1, p2, q): 1 if q is left of p1 -> p2, -1 right, 0 collinear.
int orientation_index(double p1x, double p1y, double p2x, double p2y, double qx, double qy) {
    // orientationIndexFilter(pa = p1, pb = p2, pc = q)
    {
        double detsum;
        const double detleft = (p1x - qx) * (p2y - qy);
        const double detright = (p1y - qy) * (p2x - qx);
        const double det = detleft - detright;
        if (detleft > 0.0) {
            if (detright <= 0.0) return sign_of(det);
            detsum = detleft + detright;
        } else if (detleft < 0.0) {
            if (detright >= 0.0) return sign_of(det);
            detsum = -detleft - detright;
        } else {
            return sign_of(det);
        }
        const double errbound = 1e-15 * detsum;  // DP_SAFE_EPSILON
        if (det >= errbound || -det >= errbound) return sign_of(det);
    }
    const DD dx1 = DD(p2x) + DD(-p1x);
    const DD dy1 = DD(p2y) + DD(-p1y);
    const DD dx2 = DD(qx) + DD(-p2x);
    const DD dy2 = DD(qy) + DD(-p2y);
    return (dx1 * dy2 - dy1 * dx2).signum();
}

// ------------------------------------------------------------------------------------------------ the index

// A uniform grid over the midpoints.
struct Grid {
    double x0 = 0, y0 = 0, cell = 1;
    i64 nx = 1, ny = 1;
    std::vector<i64> start;          // nx * ny + 1
    std::vector<std::int32_t> item;  // the points of each cell, ascending

    i64 col(double x) const {
        const double c = std::floor((x - x0) / cell);
        return c < 0 ? 0 : (c >= static_cast<double>(nx) ? nx - 1 : static_cast<i64>(c));
    }
    i64 row(double y) const {
        const double c = std::floor((y - y0) / cell);
        return c < 0 ? 0 : (c >= static_cast<double>(ny) ? ny - 1 : static_cast<i64>(c));
    }
};

Grid build_grid(const std::vector<double>& mx, const std::vector<double>& my) {
    Grid g;
    const i64 n = static_cast<i64>(mx.size());
    double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin, ymin = xmin, ymax = -xmin;
    i64 m = 0;
    for (i64 i = 0; i < n; ++i) {
        if (!(std::isfinite(mx[i]) && std::isfinite(my[i]))) continue;
        xmin = std::min(xmin, mx[i]);
        xmax = std::max(xmax, mx[i]);
        ymin = std::min(ymin, my[i]);
        ymax = std::max(ymax, my[i]);
        ++m;
    }
    if (m == 0) {
        g.start.assign(2, 0);
        return g;
    }
    const double w = xmax - xmin, h = ymax - ymin;
    // about two points to a cell where the points spread over an area, along the longer side where they do not
    const double target = std::max<double>(1.0, static_cast<double>(m) / 2.0);
    double cell = (w > 0 && h > 0) ? std::sqrt(w * h / target) : std::max(w, h) / target;
    if (!(cell > 0) || !std::isfinite(cell)) cell = 1.0;
    // at most 4 million cells
    while ((std::floor(w / cell) + 1) * (std::floor(h / cell) + 1) > 4e6) cell *= 2;
    g.x0 = xmin;
    g.y0 = ymin;
    g.cell = cell;
    g.nx = static_cast<i64>(std::floor(w / cell)) + 1;
    g.ny = static_cast<i64>(std::floor(h / cell)) + 1;
    g.start.assign(static_cast<std::size_t>(g.nx * g.ny + 1), 0);
    for (i64 i = 0; i < n; ++i)
        if (std::isfinite(mx[i]) && std::isfinite(my[i])) ++g.start[static_cast<std::size_t>(g.row(my[i]) * g.nx + g.col(mx[i]) + 1)];
    for (std::size_t k = 1; k < g.start.size(); ++k) g.start[k] += g.start[k - 1];
    g.item.resize(static_cast<std::size_t>(m));
    std::vector<i64> fill(g.start.begin(), g.start.end() - 1);
    for (i64 i = 0; i < n; ++i)
        if (std::isfinite(mx[i]) && std::isfinite(my[i]))
            g.item[static_cast<std::size_t>(fill[static_cast<std::size_t>(g.row(my[i]) * g.nx + g.col(mx[i]))]++)] =
                static_cast<std::int32_t>(i);
    return g;
}

struct Scratch {
    std::vector<double> vx, vy, cs, sn;
    std::vector<std::int32_t> hit;
};

}  // namespace

void buffer_vertices(double cx, double cy, double r, double* vx, double* vy) {
    // OffsetSegmentGenerator::addDirectedFillet(p, 0, 2 pi, CLOCKWISE, r): nSegs = (int)(2 pi / (pi / 2 / 30) + 0.5)
    // = 120 vertices at the angles -k * (2 pi / 120), computed with R's (mingw-w64, x87) cos and sin as GEOS in
    // the R build does
    const double quantum = kPi / 2.0 / 30;
    const int n_segs = static_cast<int>(2.0 * kPi / quantum + 0.5);
    const double inc = (2.0 * kPi) / n_segs;
    for (int k = 0; k < n_segs; ++k) {
        const double angle = static_cast<double>(-k) * inc;
        vx[k] = cx + r * lsm::r_cos(angle);
        vy[k] = cy + r * lsm::r_sin(angle);
    }
}

bool ring_intersects(const double* vx, const double* vy, int n, double px, double py) {
    // RayCrossingCounter over the segments of the closed ring; the boundary counts
    i64 crossings = 0;
    for (int k = 0; k < n; ++k) {
        const int k2 = (k + 1 == n) ? 0 : k + 1;
        const double p1x = vx[k], p1y = vy[k], p2x = vx[k2], p2y = vy[k2];
        if (px == p2x && py == p2y) return true;
        if (p1y == py && p2y == py) {
            double minx = p1x, maxx = p2x;
            if (minx > maxx) std::swap(minx, maxx);
            if (px >= minx && px <= maxx) return true;
            continue;
        }
        if ((p1y > py && p2y <= py) || (p2y > py && p1y <= py)) {
            int sign = orientation_index(p1x, p1y, p2x, p2y, px, py);
            if (sign == 0) return true;
            if (p2y < p1y) sign = -sign;
            if (sign > 0) ++crossings;
        }
    }
    return (crossings % 2) == 1;
}

Neighbourhoods corridor_neighbourhoods(const double* x0, const double* y0, const double* x1, const double* y1,
                                       const double* radius, i64 n, int threads) {
    Neighbourhoods out;
    out.azimuth.resize(static_cast<std::size_t>(n));
    out.circvar.assign(static_cast<std::size_t>(n), kNaN);
    out.offset.assign(static_cast<std::size_t>(n) + 1, 0);
    std::vector<double> mx(static_cast<std::size_t>(n)), my(static_cast<std::size_t>(n));
    std::vector<double> cs(static_cast<std::size_t>(n)), sn(static_cast<std::size_t>(n));
    const bool par = threads > 1 && n > 2000;
    lsm::parallel_for(par, n, [&](i64 i) {
        // st_line_interpolate(dist = 0.5, normalized = TRUE) of a two-vertex line (GDAL's Value)
        planar_sample(x0[i], y0[i], x1[i], y1[i], 0.5, mx[static_cast<std::size_t>(i)], my[static_cast<std::size_t>(i)]);
        const double az = lsm::r_atan2(x1[i] - x0[i], y1[i] - y0[i]);
        out.azimuth[static_cast<std::size_t>(i)] = az;
        const double pseudo = r_fmod(2 * az, 2 * kPi);
        cs[static_cast<std::size_t>(i)] = lsm::r_cos(pseudo);
        sn[static_cast<std::size_t>(i)] = lsm::r_sin(pseudo);
    });
    const Grid grid = build_grid(mx, my);
    std::vector<std::vector<std::int32_t>> lists(static_cast<std::size_t>(n));
    lsm::parallel_for_scratch<Scratch>(par, n, 64, [&](i64 i, Scratch& s) {
        const double r = radius[i];
        const double cx = mx[static_cast<std::size_t>(i)], cy = my[static_cast<std::size_t>(i)];
        if (!(r > 0) || !std::isfinite(r) || !std::isfinite(cx) || !std::isfinite(cy) || grid.item.empty()) return;
        s.vx.resize(kBufferVertices);
        s.vy.resize(kBufferVertices);
        buffer_vertices(cx, cy, r, s.vx.data(), s.vy.data());
        // the polygon lies between its inscribed circle and the circle of radius r; the slack covers the
        // rounding of the vertices and of the distances
        const double slack = 64 * 2.220446049250313e-16 * (std::fabs(cx) + std::fabs(cy) + r);
        const double inner = r * std::cos(kPi / 120.0) - slack;
        const double outer = r + slack;
        const double inner2 = inner > 0 ? inner * inner : -1.0, outer2 = outer * outer;
        const i64 c0 = grid.col(cx - outer), c1 = grid.col(cx + outer);
        const i64 r0 = grid.row(cy - outer), r1 = grid.row(cy + outer);
        s.hit.clear();
        for (i64 gr = r0; gr <= r1; ++gr) {
            for (i64 gc = c0; gc <= c1; ++gc) {
                const std::size_t cell = static_cast<std::size_t>(gr * grid.nx + gc);
                for (i64 k = grid.start[cell]; k < grid.start[cell + 1]; ++k) {
                    const std::int32_t j = grid.item[static_cast<std::size_t>(k)];
                    const double px = mx[static_cast<std::size_t>(j)], py = my[static_cast<std::size_t>(j)];
                    const double dx = px - cx, dy = py - cy;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 > outer2) continue;
                    if (d2 <= inner2 || ring_intersects(s.vx.data(), s.vy.data(), kBufferVertices, px, py)) s.hit.push_back(j);
                }
            }
        }
        std::sort(s.hit.begin(), s.hit.end());
        lists[static_cast<std::size_t>(i)] = s.hit;
        // .circ_var_per_neighbourhood: sums of cos and sin (R's long double sum, in the neighbours' order)
        const i64 nl = static_cast<i64>(s.hit.size());
        if (nl >= 2) {
            s.cs.resize(static_cast<std::size_t>(nl));
            s.sn.resize(static_cast<std::size_t>(nl));
            for (i64 k = 0; k < nl; ++k) {
                s.cs[static_cast<std::size_t>(k)] = cs[static_cast<std::size_t>(s.hit[static_cast<std::size_t>(k)])];
                s.sn[static_cast<std::size_t>(k)] = sn[static_cast<std::size_t>(s.hit[static_cast<std::size_t>(k)])];
            }
            const double sum_cos = lsm::r_sum(s.cs.data(), nl);
            const double sum_sin = lsm::r_sum(s.sn.data(), nl);
            const double a = sum_cos / static_cast<double>(nl), b = sum_sin / static_cast<double>(nl);
            out.circvar[static_cast<std::size_t>(i)] = 1 - lsm::r_sqrt(a * a + b * b);
        }
    });
    i64 total = 0;
    for (i64 i = 0; i < n; ++i) {
        out.offset[static_cast<std::size_t>(i)] = total;
        total += static_cast<i64>(lists[static_cast<std::size_t>(i)].size());
    }
    out.offset[static_cast<std::size_t>(n)] = total;
    out.index.resize(static_cast<std::size_t>(total));
    for (i64 i = 0; i < n; ++i) {
        const auto& l = lists[static_cast<std::size_t>(i)];
        std::copy(l.begin(), l.end(), out.index.begin() + out.offset[static_cast<std::size_t>(i)]);
    }
    return out;
}

void corridor_flags(const i64* offset, const std::int32_t* index, const unsigned char* candidate, i64 n,
                    double min_segments, int threads, unsigned char* corridor) {
    lsm::parallel_for(threads > 1 && n > 20000, n, [&](i64 i) {
        const i64 nb = offset[i + 1] - offset[i];
        i64 cand = 0;
        for (i64 k = offset[i]; k < offset[i + 1]; ++k) cand += candidate[index[k]] ? 1 : 0;
        corridor[i] = (nb > 0 && static_cast<double>(nb) >= min_segments && cand > nb - cand) ? 1 : 0;
    });
}

}  // namespace move
