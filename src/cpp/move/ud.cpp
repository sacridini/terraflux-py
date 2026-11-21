// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/ud.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"
#include "move/acos.hpp"

namespace move {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;  // M_PI

// One time step of the bridge: its mean, its variance and the cells of its window (the indices of the kernels).
struct Step {
    double mux, muy, sigma, w;
    double sp, so, bx, by;   // the directional bridge: the two standard deviations, the next location
    int xs, xe, ys, ye;
};

int to_index(double v, const char* what) {
    if (!(std::fabs(v) < 2.0e9))
        throw std::runtime_error(std::string("A bridge window is not finite (") + what +
                                 "): check the variance, the location error and the time step");
    return static_cast<int>(v);  // (int) of a whole number
}

// The loop `for (ti = xt[0] + fmod(xt[n-1] - xt[0], dT) / 2; ti <= xt[n-1]; ti += dT)` of the kernels, with their
// segment walk, their arithmetic and their `continue` on the segments out of `interest`. ti is added to, never
// computed as t0 + m * dT, as in the kernels.
class Walker {
public:
    Walker(const UdInput& in, double x0, double y0, double xres, double yres, i64 ny)
        : in_(in), x0_(x0), y0_(y0), xres_(xres), yres_(yres), ny_(ny) {
        ti_ = in.t[0] + (std::fmod((in.t[in.n - 1] - in.t[0]), in.time_step) / 2.0);
        end_ = in.t[in.n - 1];
    }

    bool next(Step& s) {
        const double* xt = in_.t;
        while (ti_ <= end_) {
            const double ti = ti_;
            ti_ += in_.time_step;
            if (!(ti_ > ti)) throw std::invalid_argument("time_step is too small for the times: it does not advance them");
            while (k_ + 1 < in_.n - 1 && xt[k_ + 1] < ti) ++k_;
            if (!in_.interest[k_]) continue;
            const double alpha = (ti - xt[k_]) / (xt[k_ + 1] - xt[k_]);
            const double mux = in_.x[k_] + (in_.x[k_ + 1] - in_.x[k_]) * alpha;
            const double muy = in_.y[k_] + (in_.y[k_ + 1] - in_.y[k_]) * alpha;
            // pow(v, 2) is v * v
            const double a1 = 1 - alpha;
            const double e0 = in_.le[k_], e1 = in_.le[k_ + 1];
            const double dt = xt[k_ + 1] - xt[k_];
            double sigma = 0, sigma_ext;
            if (in_.model == UdModel::Dbbmm) {
                sigma = dt * alpha * (1 - alpha) * in_.s1[k_] + (a1 * a1) * (e0 * e0) + (alpha * alpha) * (e1 * e1);
                sigma_ext = std::sqrt(sigma) * in_.ext;
            } else {
                const double sp = std::sqrt(dt * alpha * (1 - alpha) * (in_.s1[k_] * in_.s1[k_]) + (a1 * a1) * (e0 * e0) +
                                            (alpha * alpha) * (e1 * e1));
                const double so = std::sqrt(dt * alpha * (1 - alpha) * (in_.s2[k_] * in_.s2[k_]) + (a1 * a1) * (e0 * e0) +
                                            (alpha * alpha) * (e1 * e1));
                s.sp = sp;
                s.so = so;
                s.bx = in_.x[k_ + 1];
                s.by = in_.y[k_ + 1];
                const double max_sigma = (std::isnan(sp) || std::isnan(so)) ? sp + so : ((sp < so) ? so : sp);   // fmax2
                sigma_ext = max_sigma * in_.ext;
            }
            s.mux = mux;
            s.muy = muy;
            s.sigma = sigma;
            s.w = sigma_ext;
            s.xs = to_index(std::floor((mux - x0_) / xres_ - sigma_ext / xres_), "x");
            s.xe = to_index(std::ceil((mux - x0_) / xres_ + sigma_ext / xres_), "x");
            s.ye = to_index(static_cast<double>(static_cast<int>(ny_)) - std::floor((muy - y0_) / yres_ - sigma_ext / yres_), "y");
            s.ys = to_index(static_cast<double>(static_cast<int>(ny_)) - std::ceil((muy - y0_) / yres_ + sigma_ext / yres_), "y");
            return true;
        }
        return false;
    }

private:
    const UdInput& in_;
    double x0_, y0_, xres_, yres_;
    i64 ny_;
    double ti_, end_;
    i64 k_ = 0;
};

void check_inputs(const UdInput& in, i64 nx, i64 ny) {
    if (in.n < 2) throw std::invalid_argument("A UD needs at least two fixes");
    if (nx < 2 || ny < 2) throw std::invalid_argument("the grid needs at least two rows and two columns");
    if (!(in.time_step > 0) || !std::isfinite(in.time_step))
        throw std::invalid_argument("time_step must be a single positive number");
}

}  // namespace

UdProbe ud_probe(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny) {
    UdProbe p{{kInf, -kInf, kInf, -kInf}, {static_cast<double>(INT_MAX), static_cast<double>(INT_MIN),
                                           static_cast<double>(INT_MIN), static_cast<double>(INT_MAX)}};
    if (in.n < 2 || nx < 2 || ny < 2 || !(in.time_step > 0)) return p;
    Walker walk(in, xgrid[0], ygrid[0], xgrid[1] - xgrid[0], ygrid[1] - ygrid[0], ny);
    Step s;
    int min_xs = INT_MAX, max_xe = INT_MIN, max_ye = INT_MIN, min_ys = INT_MAX;
    while (walk.next(s)) {
        if (s.mux - s.w < p.env[0]) p.env[0] = s.mux - s.w;
        if (s.mux + s.w > p.env[1]) p.env[1] = s.mux + s.w;
        if (s.muy - s.w < p.env[2]) p.env[2] = s.muy - s.w;
        if (s.muy + s.w > p.env[3]) p.env[3] = s.muy + s.w;
        if (s.xs < min_xs) min_xs = s.xs;
        if (s.xe > max_xe) max_xe = s.xe;
        if (s.ye > max_ye) max_ye = s.ye;
        if (s.ys < min_ys) min_ys = s.ys;
    }
    p.idx[0] = min_xs;
    p.idx[1] = max_xe;
    p.idx[2] = max_ye;
    p.idx[3] = min_ys;
    return p;
}

namespace {

void run_ud(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny, int threads, double* ud) {
    check_inputs(in, nx, ny);
    const double x0 = xgrid[0];
    const double y0 = ygrid[0];
    const double xres = xgrid[1] - xgrid[0];
    const double yres = ygrid[1] - ygrid[0];
    const double cell_area = xres * yres;
    std::fill(ud, ud + nx * ny, 0.0);

    struct Dens {
        double mux, muy, inv_2pi_sigma, inv_2sigma;     // the directional bridge: 1 / (2 pi sp so), -0.5 / sp^2
        double inv_so2, c, bx, by;                      // -0.5 / so^2, the distance to the next location, and it
        int xs, xe, ys, ye;
    };
    const bool dbgb = in.model == UdModel::Dbgb;
    constexpr std::size_t kBlock = 1 << 17;
    std::vector<Dens> blk;
    blk.reserve(kBlock);
    // the strips of columns the steps are shared out by: narrow, since the steps of a block lie along one stretch of
    // the track and cover a part of the columns
    const i64 per = threads > 1 ? std::max<i64>(4, std::min<i64>(64, (nx + threads * 16 - 1) / (threads * 16))) : nx;
    const i64 n_strips = (nx + per - 1) / per;
    std::vector<i64> first(static_cast<std::size_t>(n_strips) + 1), next(static_cast<std::size_t>(n_strips));
    std::vector<std::int32_t> list;

    // the steps of d on the columns lo..hi
    auto add = [&](const Dens& d, i64 lo, i64 hi) {
        if (dbgb) {
            for (i64 ii = lo; ii <= hi; ++ii) {
                const double gx = xgrid[ii];
                const double bxx = gx - d.bx, axx = d.mux - gx;
                const double bx2 = bxx * bxx, ax2 = axx * axx;
                double* col = ud + ii * ny;
                for (i64 jj = d.ys; jj <= d.ye; ++jj) {
                    const double gy = ygrid[ny - jj - 1];
                    const double byy = gy - d.by, ayy = d.muy - gy;
                    const double B = std::sqrt(bx2 + byy * byy);
                    const double A = std::sqrt(ax2 + ayy * ayy);
                    double delta_para, delta_orth;
                    if (A == 0.0) {
                        delta_para = 0.0;
                        delta_orth = 0.0;
                    } else if (d.c == 0.0) {
                        const double sq = std::sqrt(0.5);
                        delta_orth = A * lsm::r_sin(r_acos(sq));
                        delta_para = A * lsm::r_cos(r_acos(sq));
                    } else {
                        double costheta = (B * B - A * A - d.c * d.c) / (2.0 * A * d.c);
                        if (costheta > 1.0) costheta = 1.0;
                        if (costheta < -1.0) costheta = -1.0;
                        const double theta = r_acos(costheta);
                        delta_orth = A * lsm::r_sin(theta);
                        delta_para = A * lsm::r_cos(theta);
                    }
                    col[jj] += d.inv_2pi_sigma * lsm::r_exp(d.inv_2sigma * delta_para * delta_para +
                                                            d.inv_so2 * delta_orth * delta_orth) * cell_area;
                }
            }
            return;
        }
        for (i64 ii = lo; ii <= hi; ++ii) {
            const double dx = xgrid[ii] - d.mux;
            const double dx2 = dx * dx;
            double* col = ud + ii * ny;
            for (i64 jj = d.ys; jj <= d.ye; ++jj) {
                const double dy = ygrid[ny - jj - 1] - d.muy;
                const double ztz = dx2 + dy * dy;
                col[jj] += d.inv_2pi_sigma * lsm::r_exp(d.inv_2sigma * ztz) * cell_area;
            }
        }
    };

    Walker walk(in, x0, y0, xres, yres, ny);
    Step s;
    for (;;) {
        blk.clear();
        while (blk.size() < kBlock) {
            if (!walk.next(s)) break;
            if (s.xs < 0)
                throw std::runtime_error("Lower x grid not large enough, consider extending the raster in that direction or enlarging the ext argument");
            if (s.xe > nx - 1)
                throw std::runtime_error("Higher x grid not large enough, consider extending the raster in that direction or enlarging the ext argument");
            if (s.ye > ny - 1)
                throw std::runtime_error("Lower y grid not large enough, consider extending the raster in that direction or enlarging the ext argument");
            if (s.ys < 0)
                throw std::runtime_error("Higher y grid not large enough, consider extending the raster in that direction or enlarging the ext argument");
            if (dbgb) {
                const double cx = s.mux - s.bx, cy = s.muy - s.by;
                blk.push_back({s.mux, s.muy, 1.0 / (2.0 * kPi * s.sp * s.so), -0.5 / (s.sp * s.sp), -0.5 / (s.so * s.so),
                               std::sqrt(cx * cx + cy * cy), s.bx, s.by, s.xs, s.xe, s.ys, s.ye});
            } else {
                blk.push_back({s.mux, s.muy, 1.0 / (2.0 * kPi * s.sigma), -1.0 / (2.0 * s.sigma), 0.0, 0.0, 0.0, 0.0,
                               s.xs, s.xe, s.ys, s.ye});
            }
        }
        if (blk.empty()) break;
        if (threads > 1 && n_strips > 1 && blk.size() >= 8) {
            // each strip gets the list of the block's steps that reach it, in their order: a cell adds its steps in
            // their order whatever the threads
            std::fill(first.begin(), first.end(), 0);
            for (const Dens& d : blk)
                for (i64 st = d.xs / per; st <= d.xe / per; ++st) ++first[static_cast<std::size_t>(st) + 1];
            for (i64 st = 0; st < n_strips; ++st)
                first[static_cast<std::size_t>(st) + 1] += first[static_cast<std::size_t>(st)];
            list.resize(static_cast<std::size_t>(first[static_cast<std::size_t>(n_strips)]));
            std::copy(first.begin(), first.end() - 1, next.begin());
            for (std::size_t k = 0; k < blk.size(); ++k)
                for (i64 st = blk[k].xs / per; st <= blk[k].xe / per; ++st)
                    list[static_cast<std::size_t>(next[static_cast<std::size_t>(st)]++)] = static_cast<std::int32_t>(k);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
            for (i64 st = 0; st < n_strips; ++st) {
                const i64 c0 = st * per;
                const i64 c1 = std::min(nx, c0 + per);
                for (i64 q = first[static_cast<std::size_t>(st)]; q < first[static_cast<std::size_t>(st) + 1]; ++q) {
                    const Dens& d = blk[static_cast<std::size_t>(list[static_cast<std::size_t>(q)])];
                    add(d, std::max<i64>(d.xs, c0), std::min<i64>(d.xe, c1 - 1));
                }
            }
        } else {
            for (const Dens& d : blk) add(d, d.xs, d.xe);
        }
    }
}

}  // namespace

void dbbmm_ud(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny, int threads, double* ud) {
    UdInput q = in;
    q.model = UdModel::Dbbmm;
    run_ud(q, xgrid, nx, ygrid, ny, threads, ud);
}

void dbgb_ud(const UdInput& in, const double* xgrid, i64 nx, const double* ygrid, i64 ny, int threads, double* ud) {
    UdInput q = in;
    q.model = UdModel::Dbgb;
    run_ud(q, xgrid, nx, ygrid, ny, threads, ud);
}

double ud_normalise(double* v, i64 n) {
    const double total = lsm::r_sum(v, n);
    if (total == 0 || !std::isfinite(total))
        throw std::runtime_error(
            "UD computation produced zero or non-finite values. The raster extent may not overlap the track.");
    for (i64 i = 0; i < n; ++i) v[i] = v[i] / total;
    return total;
}

void ud_to_rows(const double* v, i64 nx, i64 ny, double* rows) {
    constexpr i64 kTile = 64;
    for (i64 i0 = 0; i0 < nx; i0 += kTile)
        for (i64 j0 = 0; j0 < ny; j0 += kTile) {
            const i64 i1 = std::min(nx, i0 + kTile), j1 = std::min(ny, j0 + kTile);
            for (i64 i = i0; i < i1; ++i)
                for (i64 j = j0; j < j1; ++j) rows[j * nx + i] = v[i * ny + j];
        }
}

void ud_volume(const double* v, i64 n, double* out) {
    const double vtot = lsm::sum_of(0, n, [&](i64 i) { return v[i]; }, true);
    if (!std::isfinite(vtot) || vtot <= 0) {
        std::copy(v, v + n, out);
        return;
    }
    std::vector<double> vn(static_cast<std::size_t>(n));
    std::vector<i64> ord;
    ord.reserve(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) {
        vn[static_cast<std::size_t>(i)] = v[i] / vtot;
        if (!std::isnan(vn[static_cast<std::size_t>(i)])) ord.push_back(i);
    }
    // order(decreasing = TRUE), the radix method: ties stay in the order of the cells
    std::sort(ord.begin(), ord.end(), [&](i64 a, i64 b) {
        const double va = vn[static_cast<std::size_t>(a)], vb = vn[static_cast<std::size_t>(b)];
        return va > vb || (va == vb && a < b);
    });
    std::fill(out, out + n, std::numeric_limits<double>::quiet_NaN());
    lsm::Extended sum = 0.0;  // cumsum() adds in long double and rounds each partial sum
    for (const i64 i : ord) {
        sum += vn[static_cast<std::size_t>(i)];
        out[i] = static_cast<double>(sum);
    }
}

}  // namespace move
