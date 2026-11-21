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
template <class F>
double brent_run(double a, double b, F&& nll, double tol, double* f_min) {
    double x, w, v, fx, fw, fv, e, d, u, fu;
    double midpoint, tol1, tol2, pp, q, r;
    const double golden = 0.3819660;
    const int max_iter = 500;

    x = w = v = a + golden * (b - a);
    fx = fw = fv = nll(x);
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
        fu = nll(u);

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

double brent_min(double a, double b, const Pairs& p, double tol, double* f_min) {
    return brent_run(a, b, [&](double v) { return neg_log_lik(v, p); }, tol, f_min);
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

// ------------------------------------------------------------------------------------------------- dBGB

namespace {

// One axis of the leave-one-out pairs of a window: a view of `n` pairs starting where the pointers do.
struct Axis {
    const double *T, *a, *e1, *e2, *z;
    int n;
};

double neg_log_lik_axis(double var, const Axis& q) {
    double nll = 0.0;
    for (int i = 0; i < q.n; ++i) {
        const double a = q.a[i];
        const double e1 = q.e1[i];
        const double e2 = q.e2[i];
        double v = q.T[i] * a * (1.0 - a) * var + (1.0 - a) * (1.0 - a) * e1 * e1 + a * a * e2 * e2;
        if (v <= 0.0) v = DBL_EPSILON;
        nll += 0.5 * lsm::r_log(v) + q.z[i] / (2.0 * v);
    }
    return nll;
}

double brent_axis(const Axis& q, double* f_min) {
    return brent_run(0.0, 1e15, [&](double v) { return neg_log_lik_axis(v, q); }, 1e-8, f_min);
}

struct PairsDbgb {
    std::vector<double> T, alpha, zp, zo, le1, le2;
    int n = 0;
    void reserve(int m) {
        const auto s = static_cast<std::size_t>(m);
        T.resize(s), alpha.resize(s), zp.resize(s), zo.resize(s), le1.resize(s), le2.resize(s);
    }
};

// precompute_loo_dbgb: the squared residual of each pair along the direction expected position -> next
// location, and across it (the whole squared residual shared when the two coincide).
void precompute_dbgb(const double* x, const double* y, const double* lag, const double* err, int n, PairsDbgb& p) {
    int np = 0;
    int i = 1;
    while (i < n - 1) {
        const auto s = static_cast<std::size_t>(np);
        const double t = lag[i - 1] + lag[i];
        p.T[s] = t;
        const double a = lag[i - 1] / t;
        p.alpha[s] = a;
        const double ux = x[i - 1] + a * (x[i + 1] - x[i - 1]);
        const double uy = y[i - 1] + a * (y[i + 1] - y[i - 1]);
        const double rx = x[i] - ux;
        const double ry = y[i] - uy;
        const double rn2 = rx * rx + ry * ry;
        const double axx = x[i + 1] - ux;
        const double axy = y[i + 1] - uy;
        const double axn2 = axx * axx + axy * axy;
        double rp2, ro2;
        if (axn2 > 0.0) {
            const double dot = rx * axx + ry * axy;
            rp2 = (dot * dot) / axn2;
            const double diff = rn2 - rp2;
            ro2 = (diff > 0.0) ? diff : 0.0;
        } else {
            rp2 = rn2 * 0.5;
            ro2 = rn2 * 0.5;
        }
        p.zp[s] = rp2;
        p.zo[s] = ro2;
        p.le1[s] = err[i - 1];
        p.le2[s] = err[i + 1];
        ++np;
        i += 2;
    }
    p.n = np;
}

struct WindowDbgb {
    double s2pb = kNaN, s2pa = kNaN, s2ob = kNaN, s2oa = kNaN;  // variances before / after the break of each axis
    int pb = 0, ob = 0;                                          // break positions in the window (0: none)
    bool done = false;
};

}  // namespace

DbgbVariance dbgb_variance(const double* x, const double* y, const double* time_mins, const double* loc_err, i64 n,
                           int window_size, int margin, int threads) {
    if (window_size > n)
        throw std::invalid_argument("window_size (" + std::to_string(window_size) + ") > number of locations (" +
                                    std::to_string(n) + ")");
    if (margin < 1 || window_size < 2 * margin + 1)
        throw std::invalid_argument("window_size (" + std::to_string(window_size) + ") < 2*margin+1 (" +
                                    std::to_string(2 * margin + 1) + ")");
    if ((window_size % 2) == 0 || (margin % 2) == 0)
        throw std::invalid_argument("window_size and margin must both be odd");

    // the candidate breaks: odd positions (1-based, in the window) from margin to 1 + window_size - margin
    std::vector<int> breaks;
    for (int p = 2; p < window_size; ++p)
        if (p >= margin && p <= (1 + window_size - margin) && (p % 2) == 1) breaks.push_back(p);
    if (breaks.empty()) throw std::invalid_argument("margin too large for window_size (no valid candidate breaks)");
    const int n_breaks = static_cast<int>(breaks.size());
    const int min_brk = breaks.front();
    const int max_brk = breaks.back();
    const i64 n_windows = n - window_size + 1;
    const int max_pairs = window_size / 2 + 1;
    std::vector<WindowDbgb> win(static_cast<std::size_t>(n_windows));

    struct Scratch {
        PairsDbgb p;
        std::vector<double> lag, fp, s2pb, s2pa, fo, s2ob, s2oa;
    };
    lsm::parallel_for_scratch<Scratch>(threads > 1 && n_windows > 1, n_windows, 16, [&](i64 w, Scratch& sc) {
        PairsDbgb& p = sc.p;
        p.reserve(max_pairs);
        const auto nb = static_cast<std::size_t>(n_breaks);
        sc.lag.resize(static_cast<std::size_t>(window_size));
        sc.fp.resize(nb), sc.s2pb.resize(nb), sc.s2pa.resize(nb), sc.fo.resize(nb), sc.s2ob.resize(nb),
            sc.s2oa.resize(nb);
        const double* wx = x + w;
        const double* wy = y + w;
        const double* wt = time_mins + w;
        const double* we = loc_err + w;
        for (int i = 0; i < window_size - 1; ++i) sc.lag[static_cast<std::size_t>(i)] = wt[i + 1] - wt[i];
        sc.lag[static_cast<std::size_t>(window_size - 1)] = 0.0;
        WindowDbgb& out = win[static_cast<std::size_t>(w)];

        precompute_dbgb(wx, wy, sc.lag.data(), we, window_size, p);
        const int np = p.n;
        if (np < 1) return;
        const double logn = lsm::r_log(static_cast<double>(window_size));
        auto axis = [&](const std::vector<double>& z, int from, int count) {
            const auto o = static_cast<std::size_t>(from);
            return Axis{p.T.data() + o, p.alpha.data() + o, p.le1.data() + o, p.le2.data() + o, z.data() + o, count};
        };

        // 1: no break
        double f_p_full, f_o_full;
        const double s2_p_full = brent_axis(axis(p.zp, 0, np), &f_p_full);
        const double s2_o_full = brent_axis(axis(p.zo, 0, np), &f_o_full);
        const double bic_none = 2.0 * (f_p_full + f_o_full) + 2.0 * logn;

        double best_bic = bic_none;
        int best_pb = 0, best_ob = 0;
        double best_s2_pb = s2_p_full, best_s2_pa = s2_p_full;
        double best_s2_ob = s2_o_full, best_s2_oa = s2_o_full;

        // 2 and 3: a break on the parallel axis only, on the orthogonal axis only
        for (int bi = 0; bi < n_breaks; ++bi) {
            const auto b = static_cast<std::size_t>(bi);
            const int b_fix = breaks[b];
            const int k = b_fix / 2;
            if (k <= 0 || k >= np) {
                sc.fp[b] = kInf;
                sc.fo[b] = kInf;
                sc.s2pb[b] = sc.s2pa[b] = sc.s2ob[b] = sc.s2oa[b] = kNaN;
                continue;
            }
            double f_pb, f_pa;
            const double s2_pb = brent_axis(axis(p.zp, 0, k), &f_pb);
            const double s2_pa = brent_axis(axis(p.zp, k, np - k), &f_pa);
            sc.fp[b] = f_pb + f_pa;
            sc.s2pb[b] = s2_pb;
            sc.s2pa[b] = s2_pa;
            const double bic_para_only = 2.0 * (sc.fp[b] + f_o_full) + 3.0 * logn;
            if (bic_para_only < best_bic) {
                best_bic = bic_para_only;
                best_pb = b_fix;
                best_ob = 0;
                best_s2_pb = s2_pb;
                best_s2_pa = s2_pa;
                best_s2_ob = s2_o_full;
                best_s2_oa = s2_o_full;
            }
            double f_ob, f_oa;
            const double s2_ob = brent_axis(axis(p.zo, 0, k), &f_ob);
            const double s2_oa = brent_axis(axis(p.zo, k, np - k), &f_oa);
            sc.fo[b] = f_ob + f_oa;
            sc.s2ob[b] = s2_ob;
            sc.s2oa[b] = s2_oa;
            const double bic_orth_only = 2.0 * (f_p_full + sc.fo[b]) + 3.0 * logn;
            if (bic_orth_only < best_bic) {
                best_bic = bic_orth_only;
                best_pb = 0;
                best_ob = b_fix;
                best_s2_pb = s2_p_full;
                best_s2_pa = s2_p_full;
                best_s2_ob = s2_ob;
                best_s2_oa = s2_oa;
            }
        }

        // 4: given the break of one axis, the break of the other (the orthogonal one first)
        if (best_pb != 0) {
            int idx = -1;
            for (int bi = 0; bi < n_breaks; ++bi)
                if (breaks[static_cast<std::size_t>(bi)] == best_pb) {
                    idx = bi;
                    break;
                }
            if (idx >= 0) {
                const double f_para_break = sc.fp[static_cast<std::size_t>(idx)];
                for (int bi = 0; bi < n_breaks; ++bi) {
                    const auto b = static_cast<std::size_t>(bi);
                    if (!std::isfinite(sc.fo[b])) continue;
                    const double bic_both = 2.0 * (f_para_break + sc.fo[b]) + 4.0 * logn;
                    if (bic_both < best_bic) {
                        best_bic = bic_both;
                        best_ob = breaks[b];
                        best_s2_ob = sc.s2ob[b];
                        best_s2_oa = sc.s2oa[b];
                    }
                }
            }
        }
        if (best_ob != 0) {
            int idx = -1;
            for (int bi = 0; bi < n_breaks; ++bi)
                if (breaks[static_cast<std::size_t>(bi)] == best_ob) {
                    idx = bi;
                    break;
                }
            if (idx >= 0) {
                const double f_orth_break = sc.fo[static_cast<std::size_t>(idx)];
                for (int bi = 0; bi < n_breaks; ++bi) {
                    const auto b = static_cast<std::size_t>(bi);
                    if (!std::isfinite(sc.fp[b])) continue;
                    const double bic_both = 2.0 * (sc.fp[b] + f_orth_break) + 4.0 * logn;
                    if (bic_both < best_bic) {
                        best_bic = bic_both;
                        best_pb = breaks[b];
                        best_s2_pb = sc.s2pb[b];
                        best_s2_pa = sc.s2pa[b];
                    }
                }
            }
        }
        out.done = true;
        out.pb = best_pb;
        out.ob = best_ob;
        out.s2pb = best_s2_pb;
        out.s2pa = best_s2_pa;
        out.s2ob = best_s2_ob;
        out.s2oa = best_s2_oa;
    });

    DbgbVariance res;
    std::vector<double> sum_p(static_cast<std::size_t>(n), 0.0), sum_o(static_cast<std::size_t>(n), 0.0);
    res.n_estim.assign(static_cast<std::size_t>(n), 0);
    // the sums of the locations in the order of the windows
    for (i64 w = 0; w < n_windows; ++w) {
        const WindowDbgb& o = win[static_cast<std::size_t>(w)];
        if (!o.done) continue;
        for (int fix_pos = min_brk; fix_pos < max_brk; ++fix_pos) {
            const auto loc = static_cast<std::size_t>(w + fix_pos - 1);
            sum_p[loc] += (o.pb == 0 || fix_pos < o.pb) ? o.s2pb : o.s2pa;
            sum_o[loc] += (o.ob == 0 || fix_pos < o.ob) ? o.s2ob : o.s2oa;
            res.n_estim[loc] += 1;
        }
    }
    res.para_sd.assign(static_cast<std::size_t>(n), kNaN);
    res.orth_sd.assign(static_cast<std::size_t>(n), kNaN);
    for (i64 i = 0; i < n; ++i) {
        const auto s = static_cast<std::size_t>(i);
        if (res.n_estim[s] > 0) {
            const double mp = sum_p[s] / res.n_estim[s];
            const double mo = sum_o[s] / res.n_estim[s];
            res.para_sd[s] = (mp > 0.0) ? std::sqrt(mp) : 0.0;
            res.orth_sd[s] = (mo > 0.0) ? std::sqrt(mo) : 0.0;
        }
    }
    return res;
}

}  // namespace move
