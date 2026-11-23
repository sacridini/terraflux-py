// SPDX-License-Identifier: GPL-3.0-or-later
//
// Conditional logistic regression as R's survival::clogit fits it (amt's fit_issf, fit_ssf and fit_clogit):
// coxph(Surv(rep(1, n), case) ~ x + strata(stratum), method = "exact"), that is coxexact.fit and the C code of
// coxexact.c, cholesky2.c, chsolve2.c and chinv2.c of survival 3.7-0 (Therneau; LGPL >= 2), in R's arithmetic (the x87
// exp and log of R's Windows build, long double sums in scale()), so the same bits.
// Reentrant and without threads: the fit of a few thousand rows is small.
#pragma once

#include <cstdint>
#include <vector>

namespace move {

struct ClogitFit {
    std::vector<double> coef;         // p, NaN for the columns found collinear
    std::vector<double> var;          // p x p, column-major (symmetric), the zero rows of the collinear columns
    std::vector<double> coef_scaled;  // the coefficients of the scaled columns, before the NaN of the collinear ones
    std::vector<double> var_scaled;   // p x p of the scaled columns
    std::vector<double> u;            // the score at the end (scaled)
    std::vector<double> means;        // the centres of scale() (0 for the columns of -1, 0 and 1)
    std::vector<double> rescale;      // the scales of scale() (1 for the columns of -1, 0 and 1)
    double loglik0 = 0, loglik1 = 0;  // at zero and at the end
    double score = 0;                 // the score test
    int flag = 0;                     // rank (negative if not non-negative definite), 1000: no convergence
    int iter = 0;
};

// x: p columns of n values (column-major), stratum: n codes (strata in the order of their codes; the rows of
// a stratum in their order), status: n values of 0 or 1. Throws std::invalid_argument when a column of x is
// constant and not made of -1, 0 and 1 (R: NA/NaN/Inf in foreign function call).
ClogitFit clogit_fit(const double* x, std::int64_t n, std::int64_t p, const std::int32_t* stratum,
                     const std::uint8_t* status, int iter_max, double eps, double toler_chol, bool nocenter);

// coxph.wtest: b' var^-1 b through the Cholesky of var (k x k, column-major); the n = 1 case b * b / var.
double clogit_wald(const double* var, const double* b, int k, double toler_chol);

}  // namespace move
