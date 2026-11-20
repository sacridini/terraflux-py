// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/bridge.hpp"

#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;  // M_PI

// The leave-one-out pairs of a stretch of locations (precompute_loo): the time between the neighbours, the
// weight of the first, the squared distance of the middle one to the straight line between them, and their errors.
struct Pairs {
    std::vector<double> T, alpha, ztz, le1, le2;
    int n = 0;
    void reserve(int m) {
        const auto s = static_cast<std::size_t>(m);
        T.resize(s), alpha.resize(s), ztz.resize(s), le1.resize(s), le2.resize(s);
    }
};

void precompute_loo(const double* x, const double* y, const double* lag, const double* err, int n, Pairs& p) {
    int np = 0;
    int i = 1;
    while (i < n - 1) {
        const double t = lag[i - 1] + lag[i];
        p.T[static_cast<std::size_t>(np)] = t;
        const double a = lag[i - 1] / t;
        p.alpha[static_cast<std::size_t>(np)] = a;
        const double ux = x[i - 1] + a * (x[i + 1] - x[i - 1]);
        const double uy = y[i - 1] + a * (y[i + 1] - y[i - 1]);
        const double dx = x[i] - ux;
        const double dy = y[i] - uy;
        p.ztz[static_cast<std::size_t>(np)] = dx * dx + dy * dy;
        p.le1[static_cast<std::size_t>(np)] = err[i - 1];
        p.le2[static_cast<std::size_t>(np)] = err[i + 1];
        ++np;
        i += 2;
    }
    p.n = np;
}

// The negative log-likelihood of the variance `var` (neg_log_lik): the log of a product, as the C does it.
double neg_log_lik(double var, const Pairs& p) {
    double nll = 0.0;
    for (int i = 0; i < p.n; ++i) {
        const auto s = static_cast<std::size_t>(i);
        const double a = p.alpha[s];
        const double e1 = p.le1[s];
        const double e2 = p.le2[s];
        double v = p.T[s] * a * (1.0 - a) * var + (1.0 - a) * (1.0 - a) * e1 * e1 + a * a * e2 * e2;
        if (v <= 0.0) v = DBL_EPSILON;
        nll -= lsm::r_log(1.0 / (2.0 * kPi * v) * lsm::r_exp(-p.ztz[s] / (2.0 * v)));
    }
    return nll;
}

// Brent's minimisation of neg_log_lik on [a, b], line by line as move2utils' brent_min (its golden section
// constant is truncated, 0.3819660; the tolerance is relative to |x|).
double brent_min(double a, double b, const Pairs& p, double tol, double* f_min) {
    double x, w, v, fx, fw, fv, e, d, u, fu;
    double midpoint, tol1, tol2, pp, q, r;
    const double golden = 0.3819660;
    const int max_iter = 500;

    x = w = v = a + golden * (b - a);
    fx = fw = fv = neg_log_lik(x, p);
    e = 0.0;
    d = 0.0;

    for (int iter = 0; iter < max_iter; iter++) {
        midpoint = 0.5 * (a + b);
        tol1 = tol * std::fabs(x) + 1e-10;
        tol2 = 2.0 * tol1;

        if (std::fabs(x - midpoint) <= (tol2 - 0.5 * (b - a))) {
            *f_min = fx;
            return x;
        }

        if (std::fabs(e) > tol1) {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            pp = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) pp = -pp; else q = -q;
            r = e;
            e = d;

            if (std::fabs(pp) < std::fabs(0.5 * q * r) && pp > q * (a - x) && pp < q * (b - x)) {
                d = pp / q;
                u = x + d;
                if ((u - a) < tol2 || (b - u) < tol2) d = (x < midpoint) ? tol1 : -tol1;
            } else {
                e = (x < midpoint) ? b - x : a - x;
                d = golden * e;
            }
        } else {
            e = (x < midpoint) ? b - x : a - x;
            d = golden * e;
        }

        u = (std::fabs(d) >= tol1) ? x + d : x + ((d > 0) ? tol1 : -tol1);
        fu = neg_log_lik(u, p);

        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    *f_min = fx;
    return x;
}

// What one window found.
struct Window {
    double whole = kNaN, before = kNaN, after = kNaN;
    int best_b = 0;       // 1-based position of the break in the window, 0 for none
    bool done = false;    // false: no pairs, the window adds nothing
};

}  // namespace

DbbmmVariance dbbmm_variance(const double* x, const double* y, const double* time_lag, const double* loc_err, i64 n,
                             int window_size, int margin, int threads) {
    if (window_size > n)
        throw std::invalid_argument("window_size (" + std::to_string(window_size) + ") > number of locations (" +
                                    std::to_string(n) + ")");
    if (margin < 1 || window_size < 2 * margin + 1)
        throw std::invalid_argument("window_size (" + std::to_string(window_size) + ") < 2*margin+1 (" +
                                    std::to_string(2 * margin + 1) + ")");
    if ((window_size % 2) == 0 || (margin % 2) == 0)
        throw std::invalid_argument("window_size and margin must both be odd");

    const i64 n_windows = n - window_size + 1;
    const int n_range = window_size - 2 * margin + 2;
    const int n_output = n_range - 1;
    // candidate breaks: the odd positions margin .. window_size - margin + 1 inside a window (1-based)
    std::vector<int> uneven;
    for (int i = 0; i < n_range; ++i) {
        const int v = margin + i;
        if ((v % 2) == 1) uneven.push_back(v);
    }
    const int max_pairs = window_size / 2 + 1;
    std::vector<Window> win(static_cast<std::size_t>(n_windows));

    struct Scratch {
        Pairs p;
    };
    lsm::parallel_for_scratch<Scratch>(threads > 1 && n_windows > 1, n_windows, 16, [&](i64 w, Scratch& sc) {
        Pairs& p = sc.p;
        p.reserve(max_pairs);
        const double* wx = x + w;
        const double* wy = y + w;
        const double* wl = time_lag + w;
        const double* we = loc_err + w;
        const int wn = window_size;
        Window& out = win[static_cast<std::size_t>(w)];

        precompute_loo(wx, wy, wl, we, wn, p);
        if (p.n == 0) return;
        double f_whole;
        const double var_whole = brent_min(0.0, 1e15, p, 1e-8, &f_whole);
        const double bic_whole = 2.0 * f_whole + lsm::r_log(static_cast<double>(wn));

        double best_bic = kInf;
        int best_b = 0;
        double var_before = var_whole, var_after = var_whole;
        for (const int ub : uneven) {
            const int b = ub - 1;
            precompute_loo(wx, wy, wl, we, b + 1, p);
            if (p.n == 0) continue;
            double f_b;
            const double v_b = brent_min(0.0, 1e15, p, 1e-8, &f_b);
            precompute_loo(wx + b, wy + b, wl + b, we + b, wn - b, p);
            if (p.n == 0) continue;
            double f_a;
            const double v_a = brent_min(0.0, 1e15, p, 1e-8, &f_a);
            const double bic = 2.0 * (f_b + f_a) + 2.0 * lsm::r_log(static_cast<double>(wn));
            if (bic < best_bic) {
                best_bic = bic;
                best_b = ub;
                var_before = v_b;
                var_after = v_a;
            }
        }
        out.done = true;
        out.whole = var_whole;
        if (best_bic < bic_whole) {
            out.best_b = best_b;
            out.before = var_before;
            out.after = var_after;
        }
    });

    DbbmmVariance res;
    res.variance.assign(static_cast<std::size_t>(n), 0.0);
    res.n_estim.assign(static_cast<std::size_t>(n), 0);
    // the sums of the locations in the order of the windows (what R's single thread does)
    for (i64 w = 0; w < n_windows; ++w) {
        const Window& o = win[static_cast<std::size_t>(w)];
        if (!o.done) continue;
        if (o.best_b > 0) res.break_pos.push_back(static_cast<std::int32_t>(w + o.best_b));
        for (int i = 0; i < n_output; ++i) {
            const auto loc = static_cast<std::size_t>(w + margin + i - 1);
            const double v = o.best_b > 0 ? ((margin + i) < o.best_b ? o.before : o.after) : o.whole;
            res.variance[loc] += v;
            res.n_estim[loc] += 1;
        }
    }
    for (i64 i = 0; i < n; ++i) {
        const auto s = static_cast<std::size_t>(i);
        res.variance[s] = res.n_estim[s] > 0 ? res.variance[s] / res.n_estim[s] : kNaN;
    }
    return res;
}

}  // namespace move
