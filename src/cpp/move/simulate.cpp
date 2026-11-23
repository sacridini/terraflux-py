// SPDX-License-Identifier: GPL-3.0-or-later
//
// amt's redistribution_kernel and simulate_path in continuous space (see simulate.hpp).

#include "move/simulate.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "move/distr.hpp"
#include "nlm/rng.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr i64 kChunk = 4096;

// the largest stack of a program, and its check (the programs come from Python: a malformed one is an error here, not
// a crash in the parallel loop)
int program_depth(const SimFormula& f, int v) {
    int d = 0, top = 0;
    for (int k = f.var_ptr[static_cast<std::size_t>(v)]; k < f.var_ptr[static_cast<std::size_t>(v) + 1]; ++k) {
        const int op = f.op[static_cast<std::size_t>(k)];
        int pop = 0;
        if (op == kOpConst || op == kOpEnd || op == kOpStart || op == kOpSl || op == kOpTa || op == kOpLogSl ||
            op == kOpCosTa)
            pop = 0;
        else if (op >= kOpNeg && op <= kOpLog1p)
            pop = 1;
        else if (op >= kOpAdd && op <= kOpPow)
            pop = 2;
        else
            throw std::invalid_argument("simulate: unknown operation in a formula");
        if (top < pop) throw std::invalid_argument("simulate: malformed formula program");
        top = top - pop + 1;
        if (top > d) d = top;
    }
    if (top != 1) throw std::invalid_argument("simulate: a formula program must leave one value");
    return d;
}

struct Cycle {
    const SimSpec& s;
    i64 n;
    double x0, y0;
    const double* x2;
    const double* y2;
    const double* sl;
    const double* ta;
    std::vector<double> start_val;  // each layer at the start of the step
    int nvar;
    std::vector<int> depth;
    int max_depth = 1;
    std::atomic<int> bad_level{0};
};

// the values of variable v for the candidates [b, e): stack machine over chunks
void eval_var(const Cycle& c, int v, i64 b, i64 e, double* out, double* stack) {
    const SimFormula& f = c.s.f;
    const i64 m = e - b;
    int top = 0;   // number of arrays on the stack
    auto at = [&](int k) { return stack + static_cast<std::size_t>(k) * kChunk; };
    for (int k = f.var_ptr[static_cast<std::size_t>(v)]; k < f.var_ptr[static_cast<std::size_t>(v) + 1]; ++k) {
        const int op = f.op[static_cast<std::size_t>(k)];
        const double arg = f.arg[static_cast<std::size_t>(k)];
        if (op < kOpNeg) {
            double* d = at(top++);
            switch (op) {
                case kOpConst:
                    for (i64 i = 0; i < m; ++i) d[i] = arg;
                    break;
                case kOpEnd: {
                    const SimLayer& g = c.s.layers[static_cast<std::size_t>(arg)];
                    for (i64 i = 0; i < m; ++i) {
                        const i64 cell = sim_grid_cell(g, c.x2[b + i], c.y2[b + i]);
                        d[i] = cell < 0 ? kNaN : g.v[cell];
                    }
                    break;
                }
                case kOpStart: {
                    const double sv = c.start_val[static_cast<std::size_t>(arg)];
                    for (i64 i = 0; i < m; ++i) d[i] = sv;
                    break;
                }
                case kOpSl:
                    for (i64 i = 0; i < m; ++i) d[i] = c.sl[b + i];
                    break;
                case kOpTa:
                    for (i64 i = 0; i < m; ++i) d[i] = c.ta[b + i];
                    break;
                case kOpLogSl:
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_log(c.sl[b + i]);
                    break;
                default:  // kOpCosTa
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_cos(c.ta[b + i]);
                    break;
            }
        } else if (op < kOpAdd) {
            double* d = at(top - 1);
            switch (op) {
                case kOpNeg:
                    for (i64 i = 0; i < m; ++i) d[i] = -d[i];
                    break;
                case kOpAbs:
                    for (i64 i = 0; i < m; ++i) d[i] = std::fabs(d[i]);
                    break;
                case kOpLog:
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_log(d[i]);
                    break;
                case kOpExp:
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_exp(d[i]);
                    break;
                case kOpSqrt:
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_sqrt(d[i], false);
                    break;
                case kOpSin:
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_sin(d[i]);
                    break;
                case kOpCos:
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_cos(d[i]);
                    break;
                default:  // kOpLog1p
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_log1p(d[i]);
                    break;
            }
        } else {
            double* r = at(top - 1);
            const double* l = at(top - 2);
            double* d = at(top - 2);
            switch (op) {
                case kOpAdd:
                    for (i64 i = 0; i < m; ++i) d[i] = l[i] + r[i];
                    break;
                case kOpSub:
                    for (i64 i = 0; i < m; ++i) d[i] = l[i] - r[i];
                    break;
                case kOpMul:
                    for (i64 i = 0; i < m; ++i) d[i] = l[i] * r[i];
                    break;
                case kOpDiv:
                    for (i64 i = 0; i < m; ++i) d[i] = l[i] / r[i];
                    break;
                default:  // kOpPow
                    for (i64 i = 0; i < m; ++i) d[i] = lsm::r_pow(l[i], r[i]);
                    break;
            }
            --top;
        }
    }
    const double* res = at(0);
    for (i64 i = 0; i < m; ++i) out[i] = res[i];
}

// the model matrix (column-major, n x p) of the candidates
void model_matrix(Cycle& c, std::vector<double>& X) {
    const SimFormula& f = c.s.f;
    const i64 n = c.n;
    const int p = static_cast<int>(f.coef.size());
    const i64 nchunk = (n + kChunk - 1) / kChunk;
    X.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(p), 0.0);
    // the variables the columns use
    std::vector<char> used(static_cast<std::size_t>(c.nvar), 0);
    for (int v : f.col_var) used[static_cast<std::size_t>(v)] = 1;
    lsm::parallel_for(nchunk > 1, nchunk, [&](i64 ch) {
        const i64 b = ch * kChunk, e = std::min<i64>(n, b + kChunk), m = e - b;
        std::vector<double> stack(static_cast<std::size_t>(c.max_depth) * kChunk);
        std::vector<double> vals(static_cast<std::size_t>(c.nvar) * kChunk);
        for (int v = 0; v < c.nvar; ++v)
            if (used[static_cast<std::size_t>(v)])
                eval_var(c, v, b, e, vals.data() + static_cast<std::size_t>(v) * kChunk, stack.data());
        for (int col = 0; col < p; ++col) {
            double* out = X.data() + static_cast<std::size_t>(col) * static_cast<std::size_t>(n) + b;
            bool first = true;
            for (int k = f.col_ptr[static_cast<std::size_t>(col)]; k < f.col_ptr[static_cast<std::size_t>(col) + 1]; ++k) {
                const int v = f.col_var[static_cast<std::size_t>(k)];
                const int lv = f.col_level[static_cast<std::size_t>(k)];
                const double* x = vals.data() + static_cast<std::size_t>(v) * kChunk;
                for (i64 i = 0; i < m; ++i) {
                    double val;
                    if (lv < 0) {
                        val = x[i];
                    } else if (std::isnan(x[i])) {
                        val = kNaN;
                    } else {
                        // the indicator of the level: the value must be one of the factor's levels
                        const int l0 = f.level_ptr[static_cast<std::size_t>(v)], l1 = f.level_ptr[static_cast<std::size_t>(v) + 1];
                        int code = -1;
                        for (int q = l0; q < l1; ++q)
                            if (f.level[static_cast<std::size_t>(q)] == x[i]) {
                                code = q - l0;
                                break;
                            }
                        if (code < 0) {
                            c.bad_level.store(1);
                            val = kNaN;
                        } else {
                            val = code == lv ? 1.0 : 0.0;
                        }
                    }
                    out[i] = first ? val : out[i] * val;
                }
                first = false;
            }
        }
    });
}

}  // namespace

i64 sim_grid_cell(const SimLayer& g, double x, double y) {
    i64 col = -1, row = -1;
    if (x >= g.xmin && x < g.xmax)
        col = static_cast<i64>(std::trunc((x - g.xmin) / g.xres));
    else if (x == g.xmax)
        col = g.nx - 1;
    if (y > g.ymin && y <= g.ymax)
        row = static_cast<i64>(std::trunc((g.ymax - y) / g.yres));
    else if (y == g.ymin)
        row = g.ny - 1;
    if (col < 0 || row < 0 || col >= g.nx || row >= g.ny) return -1;
    return row * g.nx + col;
}

int sim_cycle(nlm::RRng& rng, const SimSpec& s, double x0, double y0, double ta0, i64 n_sample, double* ox,
              double* oy) {
    const SimFormula& f = s.f;
    const i64 n = s.n_control;
    if (n < 1) throw std::invalid_argument("simulate: n_control must be 1 or more");
    const int p = static_cast<int>(f.coef.size());
    if (p < 1) throw std::invalid_argument("simulate: the model has no coefficients");
    const int nvar = static_cast<int>(f.var_ptr.size()) - 1;
    Cycle c{s, n, x0, y0, nullptr, nullptr, nullptr, nullptr, {}, nvar, {}, 1, {}};
    c.depth.resize(static_cast<std::size_t>(nvar));
    for (int v = 0; v < nvar; ++v) {
        c.depth[static_cast<std::size_t>(v)] = program_depth(f, v);
        c.max_depth = std::max(c.max_depth, c.depth[static_cast<std::size_t>(v)]);
    }
    // random_steps_simple: all the lengths, then all the angles; x1 + sl * cos(ta0 + ta)
    std::vector<double> sl(static_cast<std::size_t>(n)), ta(static_cast<std::size_t>(n)), x2(static_cast<std::size_t>(n)),
        y2(static_cast<std::size_t>(n));
    random_numbers(rng, s.sl_name.c_str(), s.sl_par, n, sl.data());
    random_numbers(rng, s.ta_name.c_str(), s.ta_par, n, ta.data());
    lsm::parallel_for(n > 20000, n, [&](i64 i) {
        const double a = ta0 + ta[static_cast<std::size_t>(i)];
        x2[static_cast<std::size_t>(i)] = x0 + sl[static_cast<std::size_t>(i)] * lsm::r_cos(a);
        y2[static_cast<std::size_t>(i)] = y0 + sl[static_cast<std::size_t>(i)] * lsm::r_sin(a);
    });
    c.x2 = x2.data();
    c.y2 = y2.data();
    c.sl = sl.data();
    c.ta = ta.data();
    c.start_val.resize(s.layers.size());
    for (std::size_t l = 0; l < s.layers.size(); ++l) {
        const i64 cell = sim_grid_cell(s.layers[l], x0, y0);
        c.start_val[l] = cell < 0 ? kNaN : s.layers[l].v[cell];
    }
    // ssf_weights: model.matrix(na.pass) %*% coef, exp(w - mean(finite w)), the rest 0
    std::vector<double> X;
    model_matrix(c, X);
    if (c.bad_level.load()) throw std::invalid_argument("a factor in the formula has a value that is none of its levels");
    // R's %*%: the BLAS (dgemv: column by column, in double) unless a value is NaN or infinite, then the long double
    // sums of simple_matprod
    bool plain = true;
    for (std::size_t k = 0; k < X.size() && plain; ++k) plain = std::isfinite(X[k]);
    for (std::size_t j = 0; j < f.coef.size() && plain; ++j) plain = std::isfinite(f.coef[j]);
    std::vector<double> w(static_cast<std::size_t>(n));
    const std::size_t un = static_cast<std::size_t>(n);
    if (plain) {
        lsm::parallel_for(n > 20000, n, [&](i64 i) {
            double acc = 0.0;
            for (int j = 0; j < p; ++j) acc = acc + f.coef[static_cast<std::size_t>(j)] * X[static_cast<std::size_t>(j) * un + static_cast<std::size_t>(i)];
            w[static_cast<std::size_t>(i)] = acc;
        });
    } else {
        lsm::parallel_for(n > 20000, n, [&](i64 i) {
            std::vector<double> prod(static_cast<std::size_t>(p));
            for (int j = 0; j < p; ++j) prod[static_cast<std::size_t>(j)] = X[static_cast<std::size_t>(j) * un + static_cast<std::size_t>(i)] * f.coef[static_cast<std::size_t>(j)];
            w[static_cast<std::size_t>(i)] = lsm::r_sum(prod.data(), p);
        });
    }
    X.clear();
    X.shrink_to_fit();
    std::vector<double> fin;
    fin.reserve(un);
    for (double v : w)
        if (std::isfinite(v)) fin.push_back(v);
    const double mean = lsm::r_mean(fin.data(), static_cast<i64>(fin.size()));   // NaN when none is finite
    fin.clear();
    fin.shrink_to_fit();
    lsm::parallel_for(n > 20000, n, [&](i64 i) {
        const double v = lsm::r_exp(w[static_cast<std::size_t>(i)] - mean);
        w[static_cast<std::size_t>(i)] = std::isfinite(v) ? v : 0.0;
    });
    // the candidates that could be drawn and end outside the study area (amt counts them all, and the candidates
    // outside have no covariates: no weight)
    i64 outside = 0;
    for (std::size_t i = 0; i < un; ++i)
        if (w[i] > 0 && (x2[i] < s.xmin || x2[i] > s.xmax || y2[i] < s.ymin || y2[i] > s.ymax)) ++outside;
    const double fraction = static_cast<double>(lsm::Extended(outside) / lsm::Extended(n));   // R's mean() of logicals
    if (fraction > s.tolerance_outside) return kSimOutside;
    bool any = false;
    for (std::size_t i = 0; i < un && !any; ++i) any = w[i] > 0;
    if (!any) return kSimNoWeight;
    const auto pick = rng.sample_int(n, n_sample, false, &w);
    for (std::size_t k = 0; k < pick.size(); ++k) {
        ox[k] = x2[static_cast<std::size_t>(pick[k] - 1)];
        oy[k] = y2[static_cast<std::size_t>(pick[k] - 1)];
    }
    return kSimOk;
}

int sim_path(nlm::RRng& rng, const SimSpec& s, double x0, double y0, double ta0, i64 n_steps, double* xs, double* ys,
             i64* done) {
    for (i64 i = 0; i <= n_steps; ++i) xs[i] = ys[i] = kNaN;
    xs[0] = x0;
    ys[0] = y0;
    *done = 0;
    for (i64 i = 0; i < n_steps; ++i) {
        double nx, ny;
        const int st = sim_cycle(rng, s, x0, y0, ta0, 1, &nx, &ny);
        if (st != kSimOk) return st;
        ta0 = lsm::r_atan2(ny - y0, nx - x0);
        x0 = nx;
        y0 = ny;
        xs[i + 1] = x0;
        ys[i + 1] = y0;
        *done = i + 1;
    }
    return kSimOk;
}

}  // namespace move
