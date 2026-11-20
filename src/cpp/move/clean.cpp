// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/clean.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"
#include "nlm/rng.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.141592653589793;  // R's pi

// The consecutive locations of each track: [begin, end) pairs.
std::vector<std::pair<i64, i64>> segments(const i64* track, i64 n) {
    std::vector<std::pair<i64, i64>> out;
    i64 s = 0;
    for (i64 i = 1; i <= n; ++i) {
        if (i == n || track[i] != track[s]) {
            out.emplace_back(s, i);
            s = i;
        }
    }
    return out;
}

// .step_lengths_from_cc for one pair: R's vector arithmetic, in its order (sin, cos on the x87 as mingw-w64
// runs them, a vector sqrt, the UCRT's asin).
double step_length(double x1, double y1, double x2, double y2, bool lonlat) {
    if (!(std::isfinite(x1) && std::isfinite(y1) && std::isfinite(x2) && std::isfinite(y2))) return kNaN;
    if (lonlat) {
        const double R = 6371008.8;
        const double rad = kPi / 180;
        const double lon1 = x1 * rad, lat1 = y1 * rad, lon2 = x2 * rad, lat2 = y2 * rad;
        const double dlat = lat2 - lat1;
        const double dlon = lon2 - lon1;
        const double s1 = lsm::r_sin(dlat / 2);
        const double s2 = lsm::r_sin(dlon / 2);
        const double a = s1 * s1 + lsm::r_cos(lat1) * lsm::r_cos(lat2) * (s2 * s2);
        double r = lsm::r_sqrt(a);
        if (r > 1) r = 1;  // pmin(sqrt(a), 1)
        return 2 * R * std::asin(r);
    }
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return lsm::r_sqrt(dx * dx + dy * dy);
}

struct Hirt {
    double a, a_se, b, b_se, h, h_se, i, i_se;
};

const Hirt kHirt[3] = {{142.8, 16.7, 0.24, 0.01, 2.4, 1.4, -0.72, 0.26},
                       {25.5, 0.84, 0.26, 0.006, 22, 7.6, -0.6, 0.05},
                       {11.2, 0.91, 0.36, 0.02, 19.5, 13.6, -0.56, 0.07}};

// .thin_dp_run: the longest chain of `m` sorted times whose steps are within [d - tol, d + tol].
// The R code scans every earlier fix of each fix; only those within the window can be taken, and they are
// scanned in the same order (latest first), so the choices (and the floating-point sums) are the same.
void thin_run(const double* ts, i64 m, double d, double tol, bool closest, unsigned char* sel) {
    if (m <= 0) return;
    if (m == 1) {
        sel[0] = 1;
        return;
    }
    const double lo = d - tol;
    const double hi = d + tol;
    std::vector<int> f(static_cast<std::size_t>(m), 1);
    std::vector<double> dev(static_cast<std::size_t>(m), 0.0);
    std::vector<i64> pred(static_cast<std::size_t>(m), -1);
    i64 jp = -1;  // the latest j < i with ts[i] - ts[j] >= lo (it only moves forward)
    for (i64 i = 1; i < m; ++i) {
        const double ti = ts[i];
        while (jp + 1 < i && ti - ts[jp + 1] >= lo) ++jp;
        int best_f = 1;
        i64 best_j = -1;
        double best_dv = kInf;
        for (i64 j = jp; j >= 0; --j) {
            const double gap = ti - ts[j];
            if (gap > hi) break;
            const int fj = f[static_cast<std::size_t>(j)] + 1;
            const double dvj = dev[static_cast<std::size_t>(j)] + std::fabs(gap - d);
            if (fj > best_f || (fj == best_f && closest && dvj < best_dv)) {
                best_f = fj;
                best_j = j;
                best_dv = dvj;
            }
        }
        f[static_cast<std::size_t>(i)] = best_f;
        dev[static_cast<std::size_t>(i)] = best_j < 0 ? 0.0 : best_dv;
        pred[static_cast<std::size_t>(i)] = best_j;
    }
    int max_f = 0;
    for (i64 i = 0; i < m; ++i) max_f = std::max(max_f, f[static_cast<std::size_t>(i)]);
    i64 end = -1;
    for (i64 i = 0; i < m; ++i) {
        if (f[static_cast<std::size_t>(i)] != max_f) continue;
        if (end < 0 || (closest && dev[static_cast<std::size_t>(i)] < dev[static_cast<std::size_t>(end)])) end = i;
        if (!closest) break;
    }
    for (i64 k = end; k >= 0; k = pred[static_cast<std::size_t>(k)]) sel[k] = 1;
}

// .thin_dp_track of `m` sorted finite times: split at the gaps longer than d + tol, a DP in each part.
void thin_track(const double* ts, i64 m, double d, double tol, bool closest, unsigned char* sel) {
    if (m <= 0) return;
    if (m == 1) {
        sel[0] = 1;
        return;
    }
    const double cut = d + tol;
    i64 s = 0;
    for (i64 i = 1; i <= m; ++i) {
        if (i == m || ts[i] - ts[i - 1] > cut) {
            thin_run(ts + s, i - s, d, tol, closest, sel + s);
            s = i;
        }
    }
}

}  // namespace

void v_phys(const double* mass, i64 n, int mode, double ci_level, double* v_ms, double* lo_ms, double* hi_ms,
            double* kmh) {
    const Hirt& p = kHirt[mode];
    const double z = nlm::r_qnorm(0.5 + ci_level / 2);
    for (i64 k = 0; k < n; ++k) {
        const double M = mass[k];
        const double v_kmh = p.a * lsm::r_pow(M, p.b) * (1 - lsm::r_exp(-p.h * lsm::r_pow(M, p.i)));
        const double q = lsm::r_pow(M, p.b);
        const double r = lsm::r_pow(M, p.i);
        const double s = lsm::r_exp(-p.h * r);
        const double lm = lsm::r_log(M);
        const double dv_da = q * (1 - s);
        const double dv_db = p.a * q * lm * (1 - s);
        const double dv_dh = p.a * q * s * r;
        const double dv_di = p.a * q * s * p.h * r * lm;
        const double t1 = dv_da * p.a_se, t2 = dv_db * p.b_se, t3 = dv_dh * p.h_se, t4 = dv_di * p.i_se;
        const double var = t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4;
        const double se = std::sqrt(var);
        const double lo = v_kmh + (-1 * z) * se;
        const double hi = v_kmh + (1 * z) * se;
        kmh[k] = v_kmh;
        v_ms[k] = v_kmh / 3.6;
        lo_ms[k] = (lo < 0 ? 0 : lo) / 3.6;  // pmax(ci, 0)
        hi_ms[k] = (hi < 0 ? 0 : hi) / 3.6;
    }
}

void speed_cap_hard(const double* x, const double* y, const double* t, const i64* track, i64 n, bool lonlat,
                    double v_max, double jitter, int threads, double* speed, signed char* above, signed char* below,
                    double* loglr) {
    const bool has_jitter = !std::isnan(jitter);
    const auto segs = segments(track, n);
    lsm::parallel_for_dynamic(threads > 1 && n > 20000, static_cast<i64>(segs.size()), [&](i64 si) {
        const i64 s = segs[static_cast<std::size_t>(si)].first;
        const i64 e = segs[static_cast<std::size_t>(si)].second;
        std::vector<i64> act;
        act.reserve(static_cast<std::size_t>(e - s));
        for (i64 i = s; i < e; ++i) {
            speed[i] = kNaN;
            above[i] = -1;
            below[i] = -1;
            loglr[i] = kNaN;
            if (std::isfinite(t[i])) act.push_back(i);
        }
        const auto m = static_cast<i64>(act.size());
        std::vector<double> step(static_cast<std::size_t>(m), kNaN);
        for (i64 k = 0; k + 1 < m; ++k) {
            const i64 a = act[static_cast<std::size_t>(k)];
            const i64 b = act[static_cast<std::size_t>(k + 1)];
            const double st = step_length(x[a], y[a], x[b], y[b], lonlat);
            const double dt = t[b] - t[a];
            step[static_cast<std::size_t>(k)] = st;
            speed[a] = (std::isfinite(st) && std::isfinite(dt) && dt > 0) ? st / dt : kNaN;
        }
        for (i64 k = 0; k < m; ++k) {
            const i64 a = act[static_cast<std::size_t>(k)];
            const double sp_in = k > 0 ? speed[act[static_cast<std::size_t>(k - 1)]] : kNaN;
            const double st_in = k > 0 ? step[static_cast<std::size_t>(k - 1)] : kNaN;
            above[a] = (std::isfinite(speed[a]) && speed[a] > v_max) || (std::isfinite(sp_in) && sp_in > v_max);
            below[a] = has_jitter && ((std::isfinite(step[static_cast<std::size_t>(k)]) &&
                                       step[static_cast<std::size_t>(k)] < jitter) ||
                                      (std::isfinite(st_in) && st_in < jitter));
        }
        // .attach_speedcap_loglr: surprisal log(step_speed); the boundary is the least surprising flagged step
        std::vector<double> neg(static_cast<std::size_t>(e - s), kNaN);
        double thr_flagged = kInf, mx = -kInf;
        bool any_flagged = false, any_finite = false;
        for (i64 i = s; i < e; ++i) {
            const double v = speed[i];
            if (std::isfinite(v) && v > 0) {
                const double l = lsm::r_log(v);
                if (std::isfinite(l)) {
                    neg[static_cast<std::size_t>(i - s)] = l;
                    any_finite = true;
                    mx = std::max(mx, l);
                    if (above[i] == 1) {
                        any_flagged = true;
                        thr_flagged = std::min(thr_flagged, l);
                    }
                }
            }
        }
        const double thr = any_flagged ? thr_flagged : (any_finite ? mx + 1e-9 : 0.0);
        for (i64 i = s; i < e; ++i) {
            const double l = neg[static_cast<std::size_t>(i - s)] - thr;
            loglr[i] = std::isnan(l) ? kNaN : (l < 0 ? 0.0 : l);  // pmax(lr, 0): the hard cap's one-sided evidence
        }
    });
}

void thin_time(const double* t, const i64* track, i64 n, double interval, double tolerance, bool closest, int threads,
               unsigned char* selected) {
    const auto segs = segments(track, n);
    lsm::parallel_for_dynamic(threads > 1 && n > 20000, static_cast<i64>(segs.size()), [&](i64 si) {
        const i64 s = segs[static_cast<std::size_t>(si)].first;
        const i64 e = segs[static_cast<std::size_t>(si)].second;
        for (i64 i = s; i < e; ++i) selected[i] = 0;
        if (e - s == 1) {
            selected[s] = 1;
            return;
        }
        i64 m = 0;  // the finite times come first
        while (s + m < e && std::isfinite(t[s + m])) ++m;
        if (m == 0) {
            selected[s] = 1;  // R's first run holds the first fix whatever its time
            return;
        }
        thin_track(t + s, m, interval, tolerance, closest, selected + s);
    });
}

void thin_distance(const double* step, const i64* track, i64 n, int method, double distance, double tolerance,
                   bool closest, int threads, unsigned char* selected) {
    const auto segs = segments(track, n);
    lsm::parallel_for_dynamic(threads > 1 && n > 20000, static_cast<i64>(segs.size()), [&](i64 si) {
        const i64 s = segs[static_cast<std::size_t>(si)].first;
        const i64 e = segs[static_cast<std::size_t>(si)].second;
        const i64 m = e - s;
        // cumsum(c(0, head(d, -1))) with d's NAs as 0: R's cumsum adds in long double
        std::vector<double> cum(static_cast<std::size_t>(m));
        lsm::Extended sum = 0.0;
        cum[0] = 0.0;
        for (i64 k = 1; k < m; ++k) {
            const double d = step[s + k - 1];
            sum += std::isnan(d) ? 0.0 : d;
            cum[static_cast<std::size_t>(k)] = static_cast<double>(sum);
        }
        for (i64 i = s; i < e; ++i) selected[i] = 0;
        if (method == 0) {  // .thin_dist_step
            selected[s] = 1;
            double last = cum[0];
            for (i64 k = 1; k < m; ++k) {
                if (cum[static_cast<std::size_t>(k)] - last >= distance) {
                    selected[s + k] = 1;
                    last = cum[static_cast<std::size_t>(k)];
                }
            }
        } else {
            thin_track(cum.data(), m, distance, tolerance, closest, selected + s);
        }
    });
}

}  // namespace move
