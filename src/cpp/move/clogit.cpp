// SPDX-License-Identifier: GPL-3.0-or-later
//
// survival 3.7-0's coxexact.fit (R/coxexact.fit.R) and its C code (src/coxexact.c, cholesky2.c, chsolve2.c, chinv2.c,
// coxph_wtest.c), for the data of a conditional logistic regression: time 1 for every row, so each stratum is one
// death time with as many deaths as it has cases. The elementary symmetric polynomials of the exact partial
// likelihood (Gail et al. 1981) are the memoised recursions coxd0, coxd1 and coxd2 of the C code; here they are
// the same sums in the same order, computed forwards over the rows of the stratum (a cell depends only on cells
// of the column before), so the bits are the C code's.
#include "move/clogit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "lsm/rmath.hpp"
#include "move/homerange.hpp"

namespace move {

namespace {

using i64 = std::int64_t;

// cholesky2.c: C = F D F', the factors in the lower triangle of the ragged array (matrix[col][row]); a column found
// redundant (a pivot below toler * the largest diagonal) gets a zero diagonal
int cholesky2(double** matrix, int n, double toler) {
    int nonneg = 1;
    double eps = 0;
    for (int i = 0; i < n; i++) {
        if (matrix[i][i] > eps) eps = matrix[i][i];
        for (int j = i + 1; j < n; j++) matrix[j][i] = matrix[i][j];
    }
    if (eps == 0)
        eps = toler;  // no positive diagonals
    else
        eps *= toler;
    int rank = 0;
    for (int i = 0; i < n; i++) {
        const double pivot = matrix[i][i];
        if (!std::isfinite(pivot) || pivot < eps) {
            matrix[i][i] = 0;
            if (pivot < -8 * eps) nonneg = -1;
        } else {
            rank++;
            for (int j = i + 1; j < n; j++) {
                const double temp = matrix[j][i] / pivot;
                matrix[j][i] = temp;
                matrix[j][j] -= temp * temp * pivot;
                for (int k = j + 1; k < n; k++) matrix[k][j] -= temp * matrix[k][i];
            }
        }
    }
    return rank * nonneg;
}

// chsolve2.c: y <- the solution b of A b = y, from cholesky2's factors
void chsolve2(double** matrix, int n, double* y) {
    for (int i = 0; i < n; i++) {  // F b = y
        double temp = y[i];
        for (int j = 0; j < i; j++) temp -= y[j] * matrix[i][j];
        y[i] = temp;
    }
    for (int i = n - 1; i >= 0; i--) {  // D F' z = b
        if (matrix[i][i] == 0) {
            y[i] = 0;
        } else {
            double temp = y[i] / matrix[i][i];
            for (int j = i + 1; j < n; j++) temp -= y[j] * matrix[j][i];
            y[i] = temp;
        }
    }
}

// chinv2.c: the inverse of A in the upper triangle and the diagonal, from cholesky2's factors
void chinv2(double** matrix, int n) {
    for (int i = 0; i < n; i++) {  // invert the Cholesky factor in the lower triangle
        if (matrix[i][i] > 0) {
            matrix[i][i] = 1 / matrix[i][i];
            for (int j = i + 1; j < n; j++) {
                matrix[j][i] = -matrix[j][i];
                for (int k = 0; k < i; k++) matrix[j][k] += matrix[j][i] * matrix[i][k];
            }
        }
    }
    for (int i = 0; i < n; i++) {  // F' D F
        if (matrix[i][i] == 0) {   // a singular row
            for (int j = 0; j < i; j++) matrix[j][i] = 0;
            for (int j = i; j < n; j++) matrix[i][j] = 0;
        } else {
            for (int j = i + 1; j < n; j++) {
                const double temp = matrix[j][i] * matrix[j][j];
                if (j != i) matrix[i][j] = temp;
                for (int k = i; k < j; k++) matrix[i][k] += temp * matrix[j][k];
            }
        }
    }
}

// a p x p ragged array over flat storage: m[col][row]
struct Ragged {
    std::vector<double> flat;
    std::vector<double*> col;
    int p = 0;
    explicit Ragged(int p_) : flat(static_cast<std::size_t>(p_) * static_cast<std::size_t>(p_), 0.0), col(static_cast<std::size_t>(p_)), p(p_) {
        for (int i = 0; i < p_; i++) col[static_cast<std::size_t>(i)] = flat.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(p_);
    }
    void zero() { std::fill(flat.begin(), flat.end(), 0.0); }
    double** data() { return col.data(); }
};

// the sorted, scaled problem
struct Problem {
    int p = 0;
    i64 n = 0;
    std::vector<const double*> x;     // the p columns
    const std::uint8_t* status = nullptr;
    std::vector<i64> start;           // the first row of each stratum, then n
    int max_deaths = 0;
};

struct Scratch {
    std::vector<double> score, d0, d1, d2, d1j;
    explicit Scratch(const Problem& pb)
        : score(static_cast<std::size_t>(pb.n)),
          d0(static_cast<std::size_t>(pb.max_deaths) + 1),
          d1(static_cast<std::size_t>(pb.p) * (static_cast<std::size_t>(pb.max_deaths) + 1)),
          d2(static_cast<std::size_t>(pb.p) * static_cast<std::size_t>(pb.p + 1) / 2 *
             (static_cast<std::size_t>(pb.max_deaths) + 1)),
          d1j(static_cast<std::size_t>(pb.p)) {}
};

// The log-likelihood, the score and the information at beta: the three loops of coxexact.c (the first iteration, each
// iteration, the last recomputation), the same in all.
void evaluate(const Problem& pb, const double* beta, Scratch& sc, double& newlk, double* u, Ragged& imat) {
    const int p = pb.p;
    newlk = 0;
    for (int i = 0; i < p; i++) u[i] = 0;
    imat.zero();
    double** im = imat.data();
    for (std::size_t s = 0; s + 1 < pb.start.size(); ++s) {
        const i64 a = pb.start[s], b = pb.start[s + 1];
        int ndeath = 0;
        for (i64 i = a; i < b; ++i) {
            double zbeta = 0.0;  // the offset
            for (int j = 0; j < p; j++) zbeta += pb.x[static_cast<std::size_t>(j)][i] * beta[j];
            sc.score[static_cast<std::size_t>(i)] = lsm::r_exp(zbeta);
            if (pb.status[i] == 1) {
                newlk += zbeta;
                for (int j = 0; j < p; j++) u[j] += pb.x[static_cast<std::size_t>(j)][i];
                ndeath++;
            }
        }
        if (ndeath == 0) continue;
        const i64 nrisk = b - a;
        const std::size_t w = static_cast<std::size_t>(ndeath) + 1;
        double* D0 = sc.d0.data();
        D0[0] = 1.0;
        for (i64 n = 1; n <= nrisk; ++n) {
            const double sn = sc.score[static_cast<std::size_t>(a + n - 1)];
            const int dm = static_cast<int>(std::min<i64>(n, ndeath));
            for (int d = dm; d >= 1; --d) {
                const bool lt = d < n;
                for (int j = 0; j < p; j++) {
                    const double cj = pb.x[static_cast<std::size_t>(j)][a + n - 1];
                    double* D1j = sc.d1.data() + static_cast<std::size_t>(j) * w;
                    for (int k = 0; k <= j; k++) {
                        const double ck = pb.x[static_cast<std::size_t>(k)][a + n - 1];
                        const double* D1k = sc.d1.data() + static_cast<std::size_t>(k) * w;
                        double* c2 = sc.d2.data() + static_cast<std::size_t>(j * (j + 1) / 2 + k) * w;
                        double v = ((D0[d - 1] * sn) * cj) * ck;
                        if (lt) v += c2[d];
                        if (d > 1) v += sn * ((c2[d - 1] + cj * D1k[d - 1]) + ck * D1j[d - 1]);
                        c2[d] = v;
                    }
                }
                for (int j = 0; j < p; j++) {
                    const double cj = pb.x[static_cast<std::size_t>(j)][a + n - 1];
                    double* D1j = sc.d1.data() + static_cast<std::size_t>(j) * w;
                    double v = (sn * cj) * D0[d - 1];
                    if (lt) v += D1j[d];
                    if (d > 1) v += sn * D1j[d - 1];
                    D1j[d] = v;
                }
                double v0 = sn * D0[d - 1];
                if (lt) v0 += D0[d];
                D0[d] = v0;
            }
        }
        const double d0 = D0[ndeath];
        newlk -= lsm::r_log(d0);
        double* d1 = sc.d1j.data();
        for (int j = 0; j < p; j++) {
            d1[j] = sc.d1[static_cast<std::size_t>(j) * w + static_cast<std::size_t>(ndeath)] / d0;
            u[j] -= d1[j];
            for (int k = 0; k <= j; k++) {
                const double temp = sc.d2[static_cast<std::size_t>(j * (j + 1) / 2 + k) * w + static_cast<std::size_t>(ndeath)];
                im[k][j] += temp / d0 - d1[j] * d1[k];
            }
        }
    }
}

void symmetrise(Ragged& m) {
    for (int i = 1; i < m.p; i++)
        for (int j = 0; j < i; j++) m.col[static_cast<std::size_t>(i)][j] = m.col[static_cast<std::size_t>(j)][i];
}

}  // namespace

ClogitFit clogit_fit(const double* x, i64 n, i64 p64, const std::int32_t* stratum, const std::uint8_t* status,
                     int maxiter, double eps, double toler, bool nocenter) {
    if (n < 1 || p64 < 1) throw std::invalid_argument("clogit: at least one row and one column");
    const int p = static_cast<int>(p64);
    // order(strata, -time): the strata in the order of their codes, the rows as they come (a stable sort)
    std::vector<i64> ord(static_cast<std::size_t>(n));
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](i64 a, i64 b) { return stratum[a] < stratum[b]; });

    Problem pb;
    pb.p = p;
    pb.n = n;
    std::vector<std::uint8_t> st(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) st[static_cast<std::size_t>(i)] = status[ord[static_cast<std::size_t>(i)]];
    pb.status = st.data();
    for (i64 i = 0; i < n; ++i)
        if (i == 0 || stratum[ord[static_cast<std::size_t>(i)]] != stratum[ord[static_cast<std::size_t>(i - 1)]]) pb.start.push_back(i);
    pb.start.push_back(n);
    for (std::size_t s = 0; s + 1 < pb.start.size(); ++s) {
        int d = 0;
        for (i64 i = pb.start[s]; i < pb.start[s + 1]; ++i) d += st[static_cast<std::size_t>(i)];
        pb.max_deaths = std::max(pb.max_deaths, d);
    }

    // scale(x[sorted, ]), then the columns of -1, 0 and 1 as they are
    std::vector<double> xs(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
    ClogitFit fit;
    fit.means.assign(static_cast<std::size_t>(p), 0.0);
    fit.rescale.assign(static_cast<std::size_t>(p), 1.0);
    std::vector<double> v(static_cast<std::size_t>(n));
    for (int j = 0; j < p; j++) {
        double* c = xs.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
        const double* src = x + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
        for (i64 i = 0; i < n; ++i) c[i] = src[ord[static_cast<std::size_t>(i)]];
        bool zero_one = true;
        for (i64 i = 0; i < n && zero_one; ++i) zero_one = (c[i] == -1.0 || c[i] == 0.0 || c[i] == 1.0);
        if (nocenter && zero_one) {
            pb.x.push_back(c);
            continue;
        }
        const double center = r_colmean_of(c, n);
        for (i64 i = 0; i < n; ++i) c[i] -= center;
        for (i64 i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = c[i] * c[i];
        const double sd = std::sqrt(lsm::r_sum(v.data(), n) / std::max(1.0, static_cast<double>(n - 1)));
        for (i64 i = 0; i < n; ++i) {
            c[i] /= sd;
            if (!std::isfinite(c[i]))
                throw std::invalid_argument("NA/NaN/Inf in foreign function call (arg 5): a column has no variance");
        }
        fit.means[static_cast<std::size_t>(j)] = center;
        fit.rescale[static_cast<std::size_t>(j)] = sd;
        pb.x.push_back(c);
    }

    Scratch sc(pb);
    Ragged imat(p);
    std::vector<double> beta(static_cast<std::size_t>(p), 0.0), oldbeta(static_cast<std::size_t>(p)),
        u(static_cast<std::size_t>(p)), d1(static_cast<std::size_t>(p));
    // init * rescale: zeros
    double newlk = 0, loglik1 = 0;
    int flag = 0, iter = 0;
    double score = 0, loglik0 = 0;
    bool done = false;

    evaluate(pb, beta.data(), sc, newlk, u.data(), imat);
    loglik0 = newlk;
    loglik1 = newlk;
    for (int i = 0; i < p; i++) d1[static_cast<std::size_t>(i)] = u[static_cast<std::size_t>(i)];
    flag = cholesky2(imat.data(), p, toler);
    chsolve2(imat.data(), p, u.data());
    score = 0;
    for (int i = 0; i < p; i++) score += u[static_cast<std::size_t>(i)] * d1[static_cast<std::size_t>(i)];

    if (maxiter == 0 || !std::isfinite(loglik0)) {
        iter = 0;
        chinv2(imat.data(), p);
        symmetrise(imat);
        done = true;
    }
    if (!done) {
        // never complain about convergence on the first step
        for (int i = 0; i < p; i++) {
            oldbeta[static_cast<std::size_t>(i)] = beta[static_cast<std::size_t>(i)];
            beta[static_cast<std::size_t>(i)] = beta[static_cast<std::size_t>(i)] + u[static_cast<std::size_t>(i)];
        }
        bool halving = false;
        for (iter = 1; iter <= maxiter; iter++) {
            evaluate(pb, beta.data(), sc, newlk, u.data(), imat);
            flag = cholesky2(imat.data(), p, toler);
            int notfinite = 0;
            for (int i = 0; i < p; i++) {
                if (!std::isfinite(u[static_cast<std::size_t>(i)])) notfinite = 2;
                for (int j = 0; j < p; j++)
                    if (!std::isfinite(imat.col[static_cast<std::size_t>(i)][j])) notfinite = 3;
            }
            if (!std::isfinite(newlk)) notfinite = 4;
            if (notfinite == 0 && std::fabs(1 - (loglik1 / newlk)) <= eps && !halving) {
                loglik1 = newlk;
                chinv2(imat.data(), p);
                symmetrise(imat);
                done = true;
                break;
            }
            if (iter == maxiter) break;  // skip the step halving
            if (notfinite > 0 || newlk < loglik1) {
                halving = true;
                for (int i = 0; i < p; i++)
                    beta[static_cast<std::size_t>(i)] = (oldbeta[static_cast<std::size_t>(i)] + beta[static_cast<std::size_t>(i)]) / 2;
            } else {
                halving = false;
                loglik1 = newlk;
                chsolve2(imat.data(), p, u.data());
                for (int i = 0; i < p; i++) {
                    oldbeta[static_cast<std::size_t>(i)] = beta[static_cast<std::size_t>(i)];
                    beta[static_cast<std::size_t>(i)] = beta[static_cast<std::size_t>(i)] + u[static_cast<std::size_t>(i)];
                }
            }
        }
        if (!done) {
            // out of iterations: the last good beta (as the C code, whose chinv2 then takes the information
            // matrix as it is, not factored)
            if (maxiter > 1) {
                for (int i = 0; i < p; i++) beta[static_cast<std::size_t>(i)] = oldbeta[static_cast<std::size_t>(i)];
                evaluate(pb, beta.data(), sc, newlk, u.data(), imat);
            }
            loglik1 = newlk;
            flag = 1000;
            chinv2(imat.data(), p);
            symmetrise(imat);
        }
    }

    fit.loglik0 = loglik0;
    fit.loglik1 = loglik1;
    fit.score = score;
    fit.flag = flag;
    fit.iter = iter;
    fit.u = u;
    fit.coef_scaled = beta;
    fit.var_scaled = imat.flat;
    // coxexact.fit.R: the singular columns have a zero on the diagonal of var
    fit.coef = beta;
    if (flag < p)
        for (int i = 0; i < p; i++)
            if (imat.col[static_cast<std::size_t>(i)][i] == 0) fit.coef[static_cast<std::size_t>(i)] = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < p; i++) fit.coef[static_cast<std::size_t>(i)] /= fit.rescale[static_cast<std::size_t>(i)];
    // scmat %*% var %*% scmat with scmat = diag(1 / rescale)
    std::vector<double> inv(static_cast<std::size_t>(p));
    for (int i = 0; i < p; i++) inv[static_cast<std::size_t>(i)] = 1.0 / fit.rescale[static_cast<std::size_t>(i)];
    fit.var.resize(imat.flat.size());
    for (int c = 0; c < p; c++)
        for (int r = 0; r < p; r++)
            fit.var[static_cast<std::size_t>(c) * static_cast<std::size_t>(p) + static_cast<std::size_t>(r)] =
                (inv[static_cast<std::size_t>(r)] * imat.col[static_cast<std::size_t>(c)][r]) * inv[static_cast<std::size_t>(c)];
    return fit;
}

double clogit_wald(const double* var, const double* b, int k, double toler) {
    if (k == 1) return b[0] * b[0] / var[0];
    Ragged m(k);
    std::copy(var, var + static_cast<std::size_t>(k) * static_cast<std::size_t>(k), m.flat.begin());
    cholesky2(m.data(), k, toler);
    std::vector<double> solve(b, b + k);
    chsolve2(m.data(), k, solve.data());
    double sum = 0;
    for (int j = 0; j < k; j++) sum += b[j] * solve[static_cast<std::size_t>(j)];
    return sum;
}

}  // namespace move
