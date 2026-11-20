// SPDX-License-Identifier: GPL-3.0-or-later
#include "move/locerr.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "lsm/rmath.hpp"

namespace move {

namespace {

using i64 = std::int64_t;

// stats::median of the values (sorted here): the middle one, or R's mean of the two middle ones
double r_median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    const auto n = static_cast<i64>(v.size());
    const i64 half = (n + 1) / 2;
    if (n % 2 == 1) return v[static_cast<std::size_t>(half - 1)];
    return lsm::r_mean(v.data() + (half - 1), 2);
}

// R's approx1(): the value at v of the line through the valid points (x, y), n >= 2, rule = 2.
double approx1(double v, const double* x, const double* y, i64 n) {
    i64 i = 0, j = n - 1;
    if (v < x[i]) return y[0];
    if (v > x[j]) return y[n - 1];
    while (i < j - 1) {
        const i64 ij = (i + j) / 2;
        if (v < x[ij]) j = ij; else i = ij;
    }
    if (v == x[j]) return y[j];
    if (v == x[i]) return y[i];
    return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]));
}

}  // namespace

int impute_loc_err(double* e, i64 n, ImputeHow how) {
    std::vector<double> ok;
    std::vector<double> idx;
    for (i64 i = 0; i < n; ++i) {
        if (!std::isnan(e[i])) {
            ok.push_back(e[i]);
            idx.push_back(static_cast<double>(i + 1));
        }
    }
    if (static_cast<i64>(ok.size()) == n) return 0;
    if (ok.empty() && how != ImputeHow::Zero) return 1;
    if (how == ImputeHow::Approx) {
        if (ok.size() < 2) return 2;
        for (i64 i = 0; i < n; ++i)
            if (std::isnan(e[i]))
                e[i] = approx1(static_cast<double>(i + 1), idx.data(), ok.data(), static_cast<i64>(ok.size()));
        return 0;
    }
    double fill = 0.0;
    if (how == ImputeHow::Median) {
        std::vector<double> tmp = ok;
        fill = r_median(tmp);
    } else if (how == ImputeHow::Mean) {
        fill = lsm::r_mean(ok.data(), static_cast<i64>(ok.size()));
    }
    for (i64 i = 0; i < n; ++i)
        if (std::isnan(e[i])) e[i] = fill;
    return 0;
}

}  // namespace move
