// SPDX-License-Identifier: GPL-3.0-or-later
//
// Energy landscapes of a DEM (R enerscape 1.2.0): see energy.hpp. The operations are R's, in R's order and with its
// constants (PI is the truncated literal of the package; no fused multiply-add), sin() is mingw-w64's (the x87
// fsin, lsm::r_sin: the UCRT's differs in the last bit on 8% of the cells), and each cell's mean is Rcpp sugar's:
// an 80-bit sum, divided, refined once with the residuals. atan, tan, pow and sqrt of the C library agree with it.

#include "cost/energy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "lsm/extended.hpp"
#include "lsm/rmath.hpp"

namespace cost {

namespace {

using i64 = std::int64_t;
using lsm::Extended;

const double kPi = 3.1415926535;  // enerscape's own literal, not M_PI
const double kKcal = 4184.0;

// slope.cpp, in degrees (or radians for the walking model): the rise over the horizontal distance.
inline double slope_deg(double h, double res, bool diag) {
    const double run = diag ? std::sqrt(2.0) * res : res;
    return std::atan(h / run) * 180 / kPi;
}
inline double slope_rad(double h, double res, bool diag) {
    const double run = diag ? std::sqrt(2.0) * res : res;
    return std::atan(h / run);
}

// distance.cpp: the 3-D distance.
inline double dist3d(double h, double res, bool diag) {
    const double r2 = res * res;
    const double h2 = h * h;
    return diag ? std::sqrt(2 * r2 + h2) : std::sqrt(r2 + h2);
}

// Rcpp::mean (sugar): long double sum in order, divided by n, then one pass over the residuals.
double rcpp_mean(const double* v, int n) {
    Extended s = Extended(0.0);
    for (int k = 0; k < n; ++k) s = s + Extended(v[k]);
    const Extended cnt = Extended(n);
    s = s / cnt;
    if (std::isfinite(static_cast<double>(s))) {
        Extended t = Extended(0.0);
        for (int k = 0; k < n; ++k) t = t + (Extended(v[k]) - s);
        s = s + t / cnt;
    }
    return static_cast<double>(s);
}

struct Model {
    bool human;
    bool kcal;
    double mass;
    double speed;
    double ar = 0, pm = 0, flat = 0;  // the terms that do not depend on the edge

    Model(bool human_, bool kcal_, double mass_, double speed_) : human(human_), kcal(kcal_), mass(mass_), speed(speed_) {
        if (human) {
            flat = 1.44 + 1.94 * std::pow(speed, 0.34) + 0.24 * std::pow(speed, 4);
        } else {
            ar = 8.0 * std::pow(mass, -0.34);
            pm = std::pow(mass, -0.12);
        }
    }

    // The cost of the edge from a neighbour h below the cell (h = cell - neighbour) at the given distances.
    double edge(double h, double res, bool diag) const {
        const double d = dist3d(h, res, diag);
        double work;
        if (human) {
            const double sl = slope_rad(h, res, diag);
            const double hill = 34 * speed * std::tan(sl) * (1 - std::pow(1.05, 1 - std::pow(1.1, 100 * std::tan(sl) + 32)));
            work = (flat + hill) / speed * mass * d;
        } else {
            const double sl = slope_deg(h, res, diag);
            const double mec = 50.0 * (1.0 + lsm::r_sin((2.0 * sl - 74.0) / 180.0 * kPi)) * pm;
            work = (ar + mec) * mass * d;
        }
        return kcal ? work / kKcal : work;
    }
};

}  // namespace

void energy_landscape(const double* x, i64 ny, i64 nx, double mass, double res, int neigh, bool human, double speed,
                      bool kcal, int threads, double* out) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::fill(out, out + ny * nx, nan);
    if (ny < 3 || nx < 3) return;
    const Model model(human, kcal, mass, speed);
    // neighbours in R's order: (row offset, column offset, diagonal)
    static const int k4[4][3] = {{-1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {1, 0, 0}};
    static const int k8[8][3] = {{-1, -1, 1}, {0, -1, 0}, {1, -1, 1}, {-1, 0, 0}, {1, 0, 0}, {-1, 1, 1}, {0, 1, 0}, {1, 1, 1}};
    const int (*nb)[3] = neigh == 4 ? k4 : k8;
    const int nn = neigh;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (i64 i = 1; i < ny - 1; ++i) {
        double en[8];
        for (i64 j = 1; j < nx - 1; ++j) {
            const double c = x[i * nx + j];
            for (int k = 0; k < nn; ++k) {
                const double z = x[(i + nb[k][0]) * nx + (j + nb[k][1])];
                en[k] = model.edge(c - z, res, nb[k][2] != 0);
            }
            const double m = rcpp_mean(en, nn);
            out[i * nx + j] = m == 0 ? nan : m;
        }
    }
}

}  // namespace cost
