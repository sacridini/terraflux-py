// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/emd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "move/vendor/emdist/emd_rubner.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793;  // R's pi

// R's long double sum() fed in blocks (the running sum stays extended between them).
struct LongSum {
#if defined(LSM_X87_MASM)
    alignas(16) unsigned char acc[16] = {};  // an 80-bit +0
    void add(const double* x, i64 n) { lsm_x87_accumulate(x, n, acc); }
    double get() const { return lsm_x87_to_double(acc); }
#else
    lsm::Extended s = 0.0;
    void add(const double* x, i64 n) {
        for (i64 i = 0; i < n; ++i) s += x[i];
    }
    double get() const { return static_cast<double>(s); }
#endif
};

}  // namespace

void emd_prepare(const double* v, i64 n, double mask_quantile, std::vector<i64>& cells, std::vector<double>& weights) {
    cells.clear();
    weights.clear();
    // vtot = sum(v[ok]); w = v / vtot
    std::vector<double> ok_values;
    ok_values.reserve(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i)
        if (!std::isnan(v[i])) ok_values.push_back(v[i]);
    const double vtot = lsm::r_sum(ok_values.data(), static_cast<i64>(ok_values.size()));
    if (!std::isfinite(vtot) || vtot <= 0) throw std::invalid_argument("UD has zero or non-finite total mass");
    std::vector<double> w(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = v[i] / vtot;
    // ord <- order(w, decreasing = TRUE): radix, stable (ties in the cells' order), NaN last
    std::vector<i64> ord(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) ord[static_cast<std::size_t>(i)] = i;
    std::stable_sort(ord.begin(), ord.end(), [&](i64 a, i64 b) {
        const double wa = w[static_cast<std::size_t>(a)], wb = w[static_cast<std::size_t>(b)];
        if (std::isnan(wb)) return !std::isnan(wa);
        if (std::isnan(wa)) return false;
        return wa > wb;
    });
    // cum[ord] <- cumsum(w[ord]): R's cumsum adds in long double and rounds each element
    std::vector<char> keep(static_cast<std::size_t>(n), 0);
    lsm::Extended run = 0.0;
    bool any = false;
    for (i64 k = 0; k < n; ++k) {
        const i64 cell = ord[static_cast<std::size_t>(k)];
        const double wk = w[static_cast<std::size_t>(cell)];
        if (std::isnan(wk)) break;  // only the cells without data are left
        run += wk;
        const double cum = static_cast<double>(run);
        if (cum <= mask_quantile) {
            keep[static_cast<std::size_t>(cell)] = 1;
            any = true;
        }
    }
    if (!any) {
        // guarantee at least one cell survives: keep[ord[1]]
        keep[static_cast<std::size_t>(ord[0])] = 1;
    }
    std::vector<double> kept;
    for (i64 i = 0; i < n; ++i) {
        if (keep[static_cast<std::size_t>(i)]) {
            cells.push_back(i);
            kept.push_back(w[static_cast<std::size_t>(i)]);
        }
    }
    const double s = lsm::r_sum(kept.data(), static_cast<i64>(kept.size()));
    weights.resize(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) weights[i] = kept[i] / s;
}

double emd_cost(double ax, double ay, double bx, double by, bool gc, double threshold) {
    double m;
    if (gc) {
        // geosphere::distHaversine(a, b, r = 6378137), as distm calls it
        const double to_rad = kPi / 180;
        const double lon1 = ax * to_rad, lat1 = ay * to_rad, lon2 = bx * to_rad, lat2 = by * to_rad;
        const double dlat = lat2 - lat1;
        const double dlon = lon2 - lon1;
        const double s1 = lsm::r_sin(dlat / 2);
        const double s2 = lsm::r_sin(dlon / 2);
        double a = s1 * s1 + lsm::r_cos(lat1) * lsm::r_cos(lat2) * (s2 * s2);
        if (a > 1) a = 1;  // pmin(a, 1)
        m = 2 * lsm::r_atan2(lsm::r_sqrt(a), lsm::r_sqrt(1 - a)) * 6378137.0;
    } else {
        const double dx = ax - bx, dy = ay - by;
        m = lsm::r_sqrt(dx * dx + dy * dy);
    }
    if (!std::isnan(threshold) && m > threshold) m = threshold;
    return m;
}

double emd_sinkhorn(const double* ax, const double* ay, const double* p, i64 na, const double* bx, const double* by,
                    const double* q, i64 nb, bool gc, double threshold, double reg, int max_iter, double tol,
                    int threads, int* iterations) {
    if (iterations) *iterations = 0;
    if (na < 1 || nb < 1) throw std::invalid_argument("no cells");
    if (max_iter < 1) throw std::invalid_argument("max_iter: at least 1");
    const bool par = threads > 1 && na * nb > 100000;
    const auto cost = [&](i64 i, i64 j) { return emd_cost(ax[i], ay[i], bx[j], by[j], gc, threshold); };
    // M_scale <- max(M)
    std::vector<double> col_max(static_cast<std::size_t>(nb));
    lsm::parallel_for(par, nb, [&](i64 j) {
        double m = -std::numeric_limits<double>::infinity();
        for (i64 i = 0; i < na; ++i) {
            const double c = cost(i, j);
            if (c > m || std::isnan(c)) m = c;
        }
        col_max[static_cast<std::size_t>(j)] = m;
    });
    double scale = -std::numeric_limits<double>::infinity();
    for (double m : col_max) {
        if (std::isnan(m)) throw std::invalid_argument("the cost matrix has missing values");
        scale = std::max(scale, m);
    }
    if (scale == 0) return 0.0;
    // K <- exp(-M / M_scale / reg), column-major na x nb
    std::vector<double> K(static_cast<std::size_t>(na) * static_cast<std::size_t>(nb));
    lsm::parallel_for(par, nb, [&](i64 j) {
        double* col = K.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(na);
        for (i64 i = 0; i < na; ++i) col[i] = lsm::r_exp(-(cost(i, j) / scale) / reg);
    });
    std::vector<double> u(static_cast<std::size_t>(na), 1.0), unew(static_cast<std::size_t>(na)),
        v(static_cast<std::size_t>(nb), 0.0);
    constexpr i64 kRows = 2048;
    const i64 nblocks = (na + kRows - 1) / kRows;
    int it = 0;
    for (it = 1; it <= max_iter; ++it) {
        // v <- q / crossprod(K, u): dgemv('T'), the dot product of each column with u in order
        lsm::parallel_for(par, nb, [&](i64 j) {
            const double* col = K.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(na);
            double temp = 0.0;
            for (i64 i = 0; i < na; ++i) temp += col[i] * u[static_cast<std::size_t>(i)];
            double x = q[j] / temp;
            if (!std::isfinite(x)) x = 0.0;
            v[static_cast<std::size_t>(j)] = x;
        });
        // u_new <- p / (K %*% v): dgemv('N'), y[i] += v[j] * K[i, j] for j in turn, from 0
        lsm::parallel_for(par, nblocks, [&](i64 b) {
            const i64 i0 = b * kRows, i1 = std::min(na, i0 + kRows);
            for (i64 i = i0; i < i1; ++i) unew[static_cast<std::size_t>(i)] = 0.0;
            for (i64 j = 0; j < nb; ++j) {
                const double temp = v[static_cast<std::size_t>(j)];
                const double* col = K.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(na);
                for (i64 i = i0; i < i1; ++i) unew[static_cast<std::size_t>(i)] += temp * col[i];
            }
            for (i64 i = i0; i < i1; ++i) {
                double x = p[i] / unew[static_cast<std::size_t>(i)];
                if (!std::isfinite(x)) x = 0.0;
                unew[static_cast<std::size_t>(i)] = x;
            }
        });
        double change = 0.0;
        for (i64 i = 0; i < na; ++i) change = std::max(change, std::fabs(u[static_cast<std::size_t>(i)] - unew[static_cast<std::size_t>(i)]));
        if (change < tol) {
            u.swap(unew);
            break;
        }
        u.swap(unew);
    }
    if (iterations) *iterations = std::min(it, max_iter);
    // sum(((u * K) * v) * M), column by column, in R's long double
    LongSum total;
    const i64 cols_per_block = std::max<i64>(1, (1 << 22) / std::max<i64>(na, 1));
    std::vector<double> terms(static_cast<std::size_t>(std::min(nb, cols_per_block) * na));
    for (i64 j0 = 0; j0 < nb; j0 += cols_per_block) {
        const i64 j1 = std::min(nb, j0 + cols_per_block);
        lsm::parallel_for(par, j1 - j0, [&](i64 jj) {
            const i64 j = j0 + jj;
            const double* col = K.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(na);
            double* out = terms.data() + static_cast<std::size_t>(jj) * static_cast<std::size_t>(na);
            const double vj = v[static_cast<std::size_t>(j)];
            for (i64 i = 0; i < na; ++i) out[i] = ((u[static_cast<std::size_t>(i)] * col[i]) * vj) * cost(i, j);
        });
        total.add(terms.data(), (j1 - j0) * na);
    }
    return total.get();
}

double emd_exact(const double* ax, const double* ay, const double* p, i64 na, const double* bx, const double* by,
                 const double* q, i64 nb, int max_iter, bool* hit_max_iter) {
    if (na < 1 || nb < 1) throw std::invalid_argument("no cells");
    if (na >= 2000000000 || nb >= 2000000000) throw std::invalid_argument("too many cells");
    // emdist keeps its working memory in global variables
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock);
    std::vector<feature_t> fa(static_cast<std::size_t>(na)), fb(static_cast<std::size_t>(nb));
    std::vector<float> wa(static_cast<std::size_t>(na)), wb(static_cast<std::size_t>(nb));
    for (i64 i = 0; i < na; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        for (int d = 0; d < FDIM; ++d) fa[k].loc[d] = 0.0f;
        fa[k].loc[0] = static_cast<float>(ax[i]);
        fa[k].loc[1] = static_cast<float>(ay[i]);
        wa[k] = static_cast<float>(p[i]);
    }
    for (i64 j = 0; j < nb; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);
        for (int d = 0; d < FDIM; ++d) fb[k].loc[d] = 0.0f;
        fb[k].loc[0] = static_cast<float>(bx[j]);
        fb[k].loc[1] = static_cast<float>(by[j]);
        wb[k] = static_cast<float>(q[j]);
    }
    signature_t sa, sb;
    sa.n = static_cast<int>(na);
    sa.Features = fa.data();
    sa.Weights = wa.data();
    sb.n = static_cast<int>(nb);
    sb.Features = fb.data();
    sb.Weights = wb.data();
    const float d = emd_rubner(&sa, &sb, nullptr, nullptr, 0, calc_dist_L2, max_iter);
    if (hit_max_iter) *hit_max_iter = emd_rubner_hit_max_iter();
    return static_cast<double>(d);
}

}  // namespace move
