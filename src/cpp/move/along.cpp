// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/along.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

#include "lsm/parallel.hpp"
#include "move/amt.hpp"

namespace move {

namespace {

// static_cast<int>(floor(x)) as x86-64 gives it: INT_MIN when out of range.
int to_int(double d) {
    if (!(d >= -2147483648.0 && d < 2147483648.0)) return INT_MIN;
    return static_cast<int>(d);
}

// terra's math_utils is_equal (tolerance 10)
bool is_equal(double a, double b) {
    const double tol = std::max(10.0, std::abs(std::min(a, b))) * std::numeric_limits<double>::epsilon();
    return a == b || std::abs(a - b) < tol;
}

// The raster as terra holds it: the extent, the cells and what SpatRaster::resolution() and origin() give.
struct Grid {
    double xmin, xmax, ymin, ymax;
    i64 nx, ny;
    double xres, yres, ox, oy;

    Grid(double xmin_, double xmax_, double ymin_, double ymax_, i64 nx_, i64 ny_)
        : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), nx(nx_), ny(ny_) {
        xres = (xmax - xmin) / static_cast<double>(nx);
        yres = (ymax - ymin) / static_cast<double>(ny);
        // SpatRaster::origin(): the y origin comes from ymax
        ox = xmin - xres * std::round(xmin / xres);
        oy = ymax - yres * std::round(ymax / yres);
        if (is_equal(xres + ox, std::abs(ox))) ox = std::fabs(ox);
        if (is_equal(yres + oy, std::abs(oy))) oy = std::fabs(oy);
    }

    // SpatRaster::cellFromXY (the rule of points on edges: right / below, except on the last edge); -1 for NA
    i64 cell_of(double x, double y) const {
        if (std::isnan(x) || std::isnan(y)) return -1;
        const double yr_inv = static_cast<double>(ny) / (ymax - ymin);
        const double xr_inv = static_cast<double>(nx) / (xmax - xmin);
        double row = std::floor((ymax - y) * yr_inv);
        if (y == ymin) row = static_cast<double>(ny - 1);
        double col = std::floor((x - xmin) * xr_inv);
        if (x == xmax) col = static_cast<double>(nx - 1);
        if (!(row >= 0 && row < static_cast<double>(ny) && col >= 0 && col < static_cast<double>(nx))) return -1;
        return static_cast<i64>(row) * nx + static_cast<i64>(col);
    }
};

// GDAL 3.9.3's GDALdllImageLineAllTouched for one segment (bIntersectOnly = false, bAvoidBurningSamePoints =
// false, no variant): burn(iY, iX) for each cell, in the order GDAL burns them. Returns whether that order runs
// from the segment's end to its start: the walk goes by increasing x, and by increasing y in the vertical case.
template <class F>
bool line_all_touched(int nx, int ny, double dfX, double dfY, double dfXEnd, double dfYEnd, F&& burn) {
    constexpr double kEps = 1e-4;
    const double y0 = dfY;
    const double y1 = dfYEnd;
    const bool x_reversed = dfX > dfXEnd;
    // Skip segments that are off the target region.
    if ((dfY < 0.0 && dfYEnd < 0.0) || (dfY > ny && dfYEnd > ny) || (dfX < 0.0 && dfXEnd < 0.0) ||
        (dfX > nx && dfXEnd > nx))
        return x_reversed;
    if (dfX > dfXEnd) {
        std::swap(dfX, dfXEnd);
        std::swap(dfY, dfYEnd);
    }
    // Special case for vertical lines.
    if (std::floor(dfX) == std::floor(dfXEnd) || std::fabs(dfX - dfXEnd) < .01) {
        if (dfYEnd < dfY) std::swap(dfY, dfYEnd);
        const int iX = to_int(std::floor(dfXEnd));
        int iY = to_int(std::floor(dfY));
        int iYEnd = to_int(std::floor(dfYEnd - kEps));
        if (iX < 0 || iX >= nx) return y0 > y1;
        if (iY < 0) iY = 0;
        if (iYEnd >= ny) iYEnd = ny - 1;
        for (; iY <= iYEnd; iY++) burn(iY, iX);
        return y0 > y1;
    }
    // Special case for horizontal lines.
    if (std::floor(dfY) == std::floor(dfYEnd) || std::fabs(dfY - dfYEnd) < .01) {
        int iX = to_int(std::floor(dfX));
        const int iY = to_int(std::floor(dfY));
        int iXEnd = to_int(std::floor(dfXEnd - kEps));
        if (iY < 0 || iY >= ny) return x_reversed;
        if (iX < 0) iX = 0;
        if (iXEnd >= nx) iXEnd = nx - 1;
        for (; iX <= iXEnd; iX++) burn(iY, iX);
        return x_reversed;
    }
    // General case: left to right, sloped.
    const double dfSlope = (dfYEnd - dfY) / (dfXEnd - dfX);
    // Clip segment in X.
    if (dfXEnd > nx) {
        dfYEnd -= (dfXEnd - nx) * dfSlope;
        dfXEnd = nx;
    }
    if (dfX < 0.0) {
        dfY += (0.0 - dfX) * dfSlope;
        dfX = 0.0;
    }
    // Clip segment in Y.
    if (dfYEnd > dfY) {
        if (dfY < 0.0) {
            const double dfDiffX = (0.0 - dfY) / dfSlope;
            dfX += dfDiffX;
            dfY = 0.0;
        }
        if (dfYEnd >= ny) {
            dfXEnd += (dfYEnd - ny) / dfSlope;
            if (dfXEnd > nx) dfXEnd = nx;
        }
    } else {
        if (dfY >= ny) {
            const double dfDiffX = (ny - dfY) / dfSlope;
            dfX += dfDiffX;
            dfY = ny;
        }
        if (dfYEnd < 0.0) dfXEnd -= (dfYEnd - 0) / dfSlope;
    }
    // Step from pixel to pixel.
    while (dfX >= 0.0 && dfX < dfXEnd) {
        const int iX = to_int(std::floor(dfX));
        const int iY = to_int(std::floor(dfY));
        if (iY >= 0 && iY < ny) burn(iY, iX);
        double dfStepX = std::floor(dfX + 1.0) - dfX;
        double dfStepY = dfStepX * dfSlope;
        if (to_int(std::floor(dfY + dfStepY)) == iY) {
            dfX += dfStepX;
            dfY += dfStepY;
        } else if (dfSlope < 0) {
            dfStepY = iY - dfY;
            if (dfStepY > -0.000000001) dfStepY = -0.000000001;
            dfStepX = dfStepY / dfSlope;
            dfX += dfStepX;
            dfY += dfStepY;
        } else {
            dfStepY = (iY + 1) - dfY;
            if (dfStepY < 0.000000001) dfStepY = 0.000000001;
            dfStepX = dfStepY / dfSlope;
            dfX += dfStepX;
            dfY += dfStepY;
        }
    }
    return x_reversed;
}

// extract(r, LINESTRING) for one step: the cells in terra's order and their ranks along the step.
void one_step(const Grid& g, double x1, double y1, double x2, double y2, std::vector<i64>& cells,
              std::vector<i64>& rank) {
    cells.clear();
    rank.clear();
    const auto na = [&] {
        cells.assign(1, -1);
        rank.assign(1, 0);
    };
    // a coordinate that is NaN (the WKT says "NaN") burns nothing in GDAL: the cells of the other vertex follow
    const bool finite = std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2);
    std::vector<std::pair<int, int>> visits;  // (row, column) of the grid burnt on, in the order burnt
    bool reversed = false;
    i64 nc = g.nx, nr = g.ny;  // the grid burnt on, with its corner and cell size
    double cxmin = g.xmin, cymax = g.ymax, cxres = g.xres, cyres = g.yres;
    if (finite) {
        // SpatRaster::rasterizeCells: the line's extent, widened when it has no width or height
        double exmin = std::min(x1, x2), exmax = std::max(x1, x2);
        double eymin = std::min(y1, y2), eymax = std::max(y1, y2);
        if (exmin >= exmax) {
            const double xr = 0.1 * g.xres;
            exmin -= xr;
            exmax += xr;
        }
        if (eymin >= eymax) {
            const double yr = 0.1 * g.yres;
            eymin -= yr;
            eymax += yr;
        }
        exmin = std::max(g.xmin, exmin);
        exmax = std::min(g.xmax, exmax);
        eymin = std::max(g.ymin, eymin);
        eymax = std::min(g.ymax, eymax);
        if (!(exmax >= exmin && eymax >= eymin)) return na();

        // SpatRaster::crop(e, "out") gives the grid the line is burnt on. When the extent only touches the raster
        // (no width or height left) the crop fails, and terra burns on the error object, which has the geometry of
        // the whole raster.
        if (exmax > exmin && eymax > eymin) {
            // setExtent(e, keepRes = true, expand = false, "out"): align to the raster's grid, clip, count the cells
            double xmn = std::floor((exmin - g.ox) / g.xres) * g.xres + g.ox;
            double xmx = std::ceil((exmax - g.ox) / g.xres) * g.xres + g.ox;
            double ymn = std::floor((eymin - g.oy) / g.yres) * g.yres + g.oy;
            double ymx = std::ceil((eymax - g.oy) / g.yres) * g.yres + g.oy;
            if (xmn == xmx) {
                if (xmn < exmin) xmx = xmx + g.xres;
                else xmn = xmn - g.xres;
            }
            if (ymn == ymx) {
                if (ymn < eymin) ymx = ymx + g.yres;
                else ymn = ymn - g.yres;
            }
            xmn = std::max(xmn, g.xmin);
            xmx = std::min(xmx, g.xmax);
            ymn = std::max(ymn, g.ymin);
            ymx = std::min(ymx, g.ymax);
            nc = static_cast<i64>(std::max(1.0, std::round((xmx - xmn) / g.xres)));
            nr = static_cast<i64>(std::max(1.0, std::round((ymx - ymn) / g.yres)));
            xmx = xmn + static_cast<double>(nc) * g.xres;
            ymx = ymn + static_cast<double>(nr) * g.yres;
            cxmin = xmn;
            cymax = ymx;
            cxres = (xmx - xmn) / static_cast<double>(nc);
            cyres = (ymx - ymn) / static_cast<double>(nr);
        }
        // GDAL's transformer: the inverse of {xmin, xres, 0, ymax, 0, -yres}, applied as inv0 + X * inv1 + Y * inv2
        const double gt0 = cxmin, gt1 = cxres, gt3 = cymax, gt5 = -1 * cyres;
        const double i0 = -gt0 / gt1, i1 = 1.0 / gt1, i2 = 0.0, i3 = -gt3 / gt5, i4 = 0.0, i5 = 1.0 / gt5;
        const double px1 = i0 + x1 * i1 + y1 * i2, py1 = i3 + x1 * i4 + y1 * i5;
        const double px2 = i0 + x2 * i1 + y2 * i2, py2 = i3 + x2 * i4 + y2 * i5;
        reversed = line_all_touched(static_cast<int>(nc), static_cast<int>(nr), px1, py1, px2, py2,
                                               [&](int iy, int ix) { visits.emplace_back(iy, ix); });
    }

    if (visits.empty()) {
        // small = TRUE: the cells of the vertices, sorted and unique
        i64 a = g.cell_of(x1, y1), b = g.cell_of(x2, y2);
        if (a < 0 && b < 0) return na();
        if (a < 0) a = b;
        if (b < 0) b = a;
        if (a == b) {
            cells.assign(1, a);
            rank.assign(1, 0);
        } else {
            cells = {std::min(a, b), std::max(a, b)};
            rank = a < b ? std::vector<i64>{0, 1} : std::vector<i64>{1, 0};
        }
        return;
    }

    // the burnt cells once each, in the cropped grid's row-major order; the first visit gives the position
    const i64 m = static_cast<i64>(visits.size());
    std::vector<std::pair<i64, i64>> keyed(static_cast<std::size_t>(m));  // (key, visit)
    for (i64 k = 0; k < m; ++k)
        keyed[static_cast<std::size_t>(k)] = {static_cast<i64>(visits[static_cast<std::size_t>(k)].first) * nc +
                                                  visits[static_cast<std::size_t>(k)].second,
                                              k};
    std::sort(keyed.begin(), keyed.end());
    keyed.erase(std::unique(keyed.begin(), keyed.end(),
                            [](const std::pair<i64, i64>& a, const std::pair<i64, i64>& b) { return a.first == b.first; }),
                keyed.end());
    // along the step: the visits of the cells that stay, in order (the first visit of each)
    std::vector<i64> by_visit(keyed.size());
    for (std::size_t k = 0; k < keyed.size(); ++k) by_visit[k] = static_cast<i64>(k);
    std::sort(by_visit.begin(), by_visit.end(),
              [&](i64 a, i64 b) { return keyed[static_cast<std::size_t>(a)].second < keyed[static_cast<std::size_t>(b)].second; });
    const i64 q = static_cast<i64>(keyed.size());
    cells.resize(static_cast<std::size_t>(q));
    rank.resize(static_cast<std::size_t>(q));
    for (i64 p = 0; p < q; ++p) rank[static_cast<std::size_t>(by_visit[static_cast<std::size_t>(p)])] = reversed ? q - 1 - p : p;
    for (i64 k = 0; k < q; ++k) {
        const i64 key = keyed[static_cast<std::size_t>(k)].first;
        const i64 row = key / nc, col = key - row * nc;
        // as_points: the centre of the cell of the cropped grid; cellFromXY on the whole raster
        const double cx = cxmin + (static_cast<double>(col) + 0.5) * cxres;
        const double cy = cymax - (static_cast<double>(row) + 0.5) * cyres;
        cells[static_cast<std::size_t>(k)] = g.cell_of(cx, cy);
    }
}

}  // namespace

double r_fmt15(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", x);
    return std::strtod(buf, nullptr);
}

AlongCells along_cells(const double* x1, const double* y1, const double* x2, const double* y2, i64 n, double xmin,
                       double xmax, double ymin, double ymax, i64 nx, i64 ny, bool round15, int threads) {
    const Grid g(xmin, xmax, ymin, ymax, nx, ny);
    std::vector<std::vector<i64>> cells(static_cast<std::size_t>(n)), ranks(static_cast<std::size_t>(n));
    lsm::parallel_for_dynamic(threads > 1 && n > 256, n, [&](i64 i) {
        const auto s = static_cast<std::size_t>(i);
        double a = x1[i], b = y1[i], c = x2[i], d = y2[i];
        if (round15) {
            a = r_fmt15(a);
            b = r_fmt15(b);
            c = r_fmt15(c);
            d = r_fmt15(d);
        }
        one_step(g, a, b, c, d, cells[s], ranks[s]);
    });
    AlongCells out;
    out.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    for (i64 i = 0; i < n; ++i)
        out.offsets[static_cast<std::size_t>(i) + 1] =
            out.offsets[static_cast<std::size_t>(i)] + static_cast<i64>(cells[static_cast<std::size_t>(i)].size());
    out.cell.reserve(static_cast<std::size_t>(out.offsets.back()));
    out.rank.reserve(static_cast<std::size_t>(out.offsets.back()));
    for (i64 i = 0; i < n; ++i) {
        out.cell.insert(out.cell.end(), cells[static_cast<std::size_t>(i)].begin(), cells[static_cast<std::size_t>(i)].end());
        out.rank.insert(out.rank.end(), ranks[static_cast<std::size_t>(i)].begin(), ranks[static_cast<std::size_t>(i)].end());
    }
    return out;
}

void along_stats(const AlongCells& c, const double* v, i64 nv, int threads, double* out) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    const i64 n = static_cast<i64>(c.offsets.size()) - 1;
    lsm::parallel_for_dynamic(threads > 1 && n > 256, n, [&](i64 i) {
        const i64 a = c.offsets[static_cast<std::size_t>(i)];
        const i64 b = c.offsets[static_cast<std::size_t>(i) + 1];
        const i64 m = b - a;
        double* o = out + i * 6;
        std::vector<double> vals(static_cast<std::size_t>(m)), along(static_cast<std::size_t>(m));
        for (i64 k = 0; k < m; ++k) {
            const i64 cell = c.cell[static_cast<std::size_t>(a + k)];
            const double x = (cell >= 0 && cell < nv) ? v[cell] : kNaN;
            vals[static_cast<std::size_t>(k)] = x;
            along[static_cast<std::size_t>(c.rank[static_cast<std::size_t>(a + k)])] = x;
        }
        std::vector<double> ok;
        ok.reserve(static_cast<std::size_t>(m));
        for (const double x : vals)
            if (!std::isnan(x)) ok.push_back(x);
        if (ok.empty()) {
            for (int j = 0; j < 6; ++j) o[j] = kNaN;
            return;
        }
        o[0] = r_mean_of(ok.data(), static_cast<i64>(ok.size()));
        o[1] = *std::min_element(ok.begin(), ok.end());
        o[2] = *std::max_element(ok.begin(), ok.end());
        double up = 0, down = 0, first = kNaN, prev = kNaN;
        for (const double x : along) {
            if (std::isnan(x)) continue;
            if (std::isnan(first)) first = x;
            else if (x > prev) up += x - prev;
            else if (x < prev) down += prev - x;
            prev = x;
        }
        o[3] = up;
        o[4] = down;
        o[5] = std::fabs(prev - first);
    });
}

}  // namespace move
