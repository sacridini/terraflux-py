// SPDX-License-Identifier: GPL-3.0-or-later
//
// amt's home ranges with the arithmetic of R (see homerange.hpp). Ported from R 4.4's grDevices (chull.c: Eddy's
// ACM TOMS 523 as translated by R), stats (fft.c, after Singleton's "Programs for Digital Signal Processing") and
// main (contour-common.h, R's contourLines), and KernSmooth 2.23 (linbin.f, linbin2D.f, bkde2D, dpik, bkfe),
// all GPL-2-or-later as is this file's license (GPL-3.0-or-later).
#include "move/homerange.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "lsm/extended.hpp"
#include "lsm/parallel.hpp"
#include "lsm/rmath.hpp"

namespace move {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.141592653589793238462643383280;
constexpr double kPi4 = 0.785398163397448309615660845820;  // M_PI_4
constexpr double kInvSqrt2Pi = 0.398942280401432677939946059934;  // M_1_SQRT_2PI

inline int imin2(int a, int b) { return a < b ? a : b; }

// ---------------------------------------------------------------------------------------------- fft

// Singleton's fftmx of R's fft.c for the factors 2 and 4 (the lengths are powers of two): the data a (real parts)
// and b (imaginary parts) are interleaved with the increment |isn| = 2. The odd factors, and the permutation for
// square-free factors, are not reached by a power of two and are left out.
void fftmx(double* a, double* b, int ntot, int n, int nspan, int isn, int m, int kt, double* at, double* bt, int* np,
           int* nfac) {
    double aj, ajm, ajp, ak, akm, akp;
    double bj, bjm, bjp, bk, bkm, bkp;
    double c1, c2 = 0, c3 = 0, cd;
    double dr, rad;
    double s1, s2 = 0, s3 = 0, sd;
    int i, inc, jc, jj, k, k1, k2, k3 = 0, kk, klim, ks, kspan, kspnn, lim, mm, nn, nt, j;

    a--;
    b--;
    at--;
    bt--;
    np--;

    inc = std::abs(isn);
    nt = inc * ntot;
    ks = inc * nspan;
    rad = kPi4;
    if (isn <= 0) rad = -rad;

    kspan = ks;
    nn = nt - inc;
    jc = ks / n;
    lim = 32;
    klim = lim * jc;
    i = 0;

L_start:
    dr = (8.0 * jc) / kspan;
    cd = lsm::r_sin(0.5 * dr * rad);
    cd = 2.0 * cd * cd;
    sd = lsm::r_sin(dr * rad);
    kk = 1;
    i++;
    if (nfac[i - 1] != 2) goto L110;

    // factor 2 (with the rotation)
    kspan /= 2;
    k1 = kspan + 2;
    do {
        do {
            k2 = kk + kspan;
            ak = a[k2];
            bk = b[k2];
            a[k2] = a[kk] - ak;
            b[k2] = b[kk] - bk;
            a[kk] += ak;
            b[kk] += bk;
            kk = k2 + kspan;
        } while (kk <= nn);
        kk -= nn;
    } while (kk <= jc);

    if (kk > kspan) goto L_fin;
L60:
    c1 = 1.0 - cd;
    s1 = sd;
    mm = imin2(k1 / 2, klim);
    goto L80;

L70:
    ak = c1 - (cd * c1 + sd * s1);
    s1 = (sd * c1 - cd * s1) + s1;
    c1 = ak;

L80:
    do {
        k2 = kk + kspan;
        ak = a[kk] - a[k2];
        bk = b[kk] - b[k2];
        a[kk] += a[k2];
        b[kk] += b[k2];
        a[k2] = c1 * ak - s1 * bk;
        b[k2] = s1 * ak + c1 * bk;
        kk = k2 + kspan;
    } while (kk < nt);
    k2 = kk - nt;
    c1 = -c1;
    kk = k1 - k2;
    if (kk > k2) goto L80;
    kk += jc;
    if (kk <= mm) goto L70;
    if (kk >= k2) {
        k1 = k1 + inc + inc;
        kk = (k1 - kspan) / 2 + jc;
        if (kk <= jc + jc) goto L60;
        goto L_start;
    }

    s1 = ((kk - 1) / jc) * dr * rad;
    c1 = lsm::r_cos(s1);
    s1 = lsm::r_sin(s1);
    mm = imin2(k1 / 2, mm + klim);
    goto L80;

    // factor 4
L110:
    if (nfac[i - 1] != 4) throw std::invalid_argument("fft: the length must be a power of two");
    kspnn = kspan;
    kspan /= 4;
L120:
    c1 = 1.0;
    s1 = 0;
    mm = imin2(kspan, klim);
    goto L150;
L130:
    c2 = c1 - (cd * c1 + sd * s1);
    s1 = (sd * c1 - cd * s1) + s1;
    c1 = c2;

L140:
    c2 = c1 * c1 - s1 * s1;
    s2 = c1 * s1 * 2.0;
    c3 = c2 * c1 - s2 * s1;
    s3 = c2 * s1 + s2 * c1;

L150:
    k1 = kk + kspan;
    k2 = k1 + kspan;
    k3 = k2 + kspan;
    akp = a[kk] + a[k2];
    akm = a[kk] - a[k2];
    ajp = a[k1] + a[k3];
    ajm = a[k1] - a[k3];
    a[kk] = akp + ajp;
    ajp = akp - ajp;
    bkp = b[kk] + b[k2];
    bkm = b[kk] - b[k2];
    bjp = b[k1] + b[k3];
    bjm = b[k1] - b[k3];
    b[kk] = bkp + bjp;
    bjp = bkp - bjp;
    if (isn < 0) goto L180;
    akp = akm - bjm;
    akm = akm + bjm;
    bkp = bkm + ajm;
    bkm = bkm - ajm;
    if (s1 == 0.0) goto L190;
L160:
    a[k1] = akp * c1 - bkp * s1;
    b[k1] = akp * s1 + bkp * c1;
    a[k2] = ajp * c2 - bjp * s2;
    b[k2] = ajp * s2 + bjp * c2;
    a[k3] = akm * c3 - bkm * s3;
    b[k3] = akm * s3 + bkm * c3;
    kk = k3 + kspan;
    if (kk <= nt) goto L150;
L170:
    kk = kk - nt + jc;
    if (kk <= mm) goto L130;
    if (kk < kspan) goto L200;
    kk = kk - kspan + inc;
    if (kk <= jc) goto L120;
    if (kspan == jc) goto L_fin;
    goto L_start;
L180:
    akp = akm + bjm;
    akm = akm - bjm;
    bkp = bkm - ajm;
    bkm = bkm + ajm;
    if (s1 != 0.0) goto L160;
L190:
    a[k1] = akp;
    b[k1] = bkp;
    a[k2] = ajp;
    b[k2] = bjp;
    a[k3] = akm;
    b[k3] = bkm;
    kk = k3 + kspan;
    if (kk <= nt) goto L150;
    goto L170;
L200:
    s1 = ((kk - 1) / jc) * dr * rad;
    c1 = lsm::r_cos(s1);
    s1 = lsm::r_sin(s1);
    mm = imin2(kspan, mm + klim);
    goto L140;

    // permute the results to normal order, in two stages: the permutation for the square factors of n
L_fin:
    np[1] = ks;
    if (kt == 0) return;
    k = kt + kt + 1;
    if (m < k) k--;
    np[k + 1] = jc;
    for (j = 1; j < k; j++, k--) {
        np[j + 1] = np[j] / nfac[j - 1];
        np[k] = np[k + 1] * nfac[j - 1];
    }
    k3 = np[k + 1];
    kspan = np[2];
    kk = jc + 1;
    k2 = kspan + 1;
    j = 1;

    if (n == ntot) {
        // permutation for a single-variate transform
    L370:
        do {
            ak = a[kk];
            a[kk] = a[k2];
            a[k2] = ak;
            bk = b[kk];
            b[kk] = b[k2];
            b[k2] = bk;
            kk += inc;
            k2 += kspan;
        } while (k2 < ks);
    L380:
        do {
            k2 -= np[j];
            j++;
            k2 += np[j + 1];
        } while (k2 > np[j]);
        j = 1;
        do {
            if (kk < k2) goto L370;
            kk += inc;
            k2 += kspan;
        } while (k2 < ks);
        if (kk < ks) goto L380;
        jc = k3;
    } else {
        // permutation for a multivariate transform
    L400:
        k = kk + jc;
        do {
            ak = a[kk];
            a[kk] = a[k2];
            a[k2] = ak;
            bk = b[kk];
            b[kk] = b[k2];
            b[k2] = bk;
            kk += inc;
            k2 += inc;
        } while (kk < k);
        kk += ks - jc;
        k2 += ks - jc;
        if (kk < nt) goto L400;
        k2 += -nt + kspan;
        kk += -nt + jc;
        if (k2 < ks) goto L400;

        do {
            do {
                k2 -= np[j];
                j++;
                k2 += np[j + 1];
            } while (k2 > np[j]);
            j = 1;
            do {
                if (kk < k2) goto L400;
                kk += jc;
                k2 += kspan;
            } while (k2 < ks);
        } while (kk < ks);
        jc = k3;
    }
    // a power of two has no square-free factor (2 * kt + 1 >= m): nothing more to do
    (void)jj;
    (void)at;
    (void)bt;
    (void)c3;
    (void)s3;
    (void)aj;
    (void)bj;
}

// fft_factor of R's fft.c: the factors of n (the squares first); false for a length that is not a power of two.
bool fft_factor(int n, int* nfac, int& m_fac, int& kt, int& maxf, int& maxp) {
    int j, jj, k, sqrtk, kchanged;
    m_fac = 0;
    k = n;
    if (k == 1) return false;
    while (k % 16 == 0) {
        nfac[m_fac++] = 4;
        k /= 16;
    }
    kchanged = 0;
    sqrtk = static_cast<int>(std::sqrt(static_cast<double>(k)));
    for (j = 3; j <= sqrtk; j += 2) {
        jj = j * j;
        while (k % jj == 0) {
            nfac[m_fac++] = j;
            k /= jj;
            kchanged = 1;
        }
        if (kchanged) {
            kchanged = 0;
            sqrtk = static_cast<int>(std::sqrt(static_cast<double>(k)));
        }
    }
    maxp = 0;
    if (k <= 4) {
        kt = m_fac;
        nfac[m_fac] = k;
        if (k != 1) m_fac++;
    } else {
        if (k % 4 == 0) {
            nfac[m_fac++] = 2;
            k /= 4;
        }
        kt = m_fac;
        maxp = std::max(kt + kt + 2, k - 1);
        j = 2;
        do {
            if (k % j == 0) {
                nfac[m_fac++] = j;
                k /= j;
            }
            j = ((j + 1) / 2) * 2 + 1;
        } while (j <= k);
    }
    if (m_fac <= kt + 1) maxp = m_fac + kt + 1;
    if (m_fac + kt > 20) return false;
    if (kt != 0) {
        j = kt;
        while (j != 0) nfac[m_fac++] = nfac[--j];
    }
    maxf = nfac[m_fac - kt - 1];
    if (kt > 0) maxf = std::max(nfac[kt - 1], maxf);
    if (kt > 1) maxf = std::max(nfac[kt - 2], maxf);
    if (kt > 2) maxf = std::max(nfac[kt - 3], maxf);
    return true;
}

bool is_pow2(i64 n) { return n >= 1 && (n & (n - 1)) == 0; }

}  // namespace

void r_fft(double* z, i64 n1, i64 n2, bool inverse) {
    if (!is_pow2(n1) || !is_pow2(n2)) throw std::invalid_argument("fft: the lengths must be powers of two");
    const int inv = inverse ? 2 : -2;
    const i64 dims[2] = {n1, n2};
    i64 nseg = n1 * n2, n = 1, nspn = 1;
    for (const i64 d : dims) {
        if (d <= 1) continue;
        nspn *= n;
        n = d;
        nseg /= n;
        int nfac[20] = {};
        int m_fac = 0, kt = 0, maxf = 0, maxp = 0;
        if (!fft_factor(static_cast<int>(n), nfac, m_fac, kt, maxf, maxp))
            throw std::invalid_argument("fft factorization error");
        std::vector<double> work(static_cast<std::size_t>(4 * maxf) + 8);
        std::vector<int> iwork(static_cast<std::size_t>(maxp) + 8);
        const int nspan = static_cast<int>(n * nspn);
        const int ntot = static_cast<int>(nspan * nseg);
        fftmx(z, z + 1, ntot, static_cast<int>(n), nspan, inv, m_fac, kt, work.data(), work.data() + 2 * maxf,
              iwork.data(), nfac);
    }
}

// ---------------------------------------------------------------------------------------- R's small pieces

void r_seq_len(double a, double b, i64 m, double* out) {
    if (m <= 0) return;
    if (m == 1) {
        out[0] = a;
        return;
    }
    if (m == 2) {
        out[0] = a;
        out[1] = b;
        return;
    }
    if (a == b) {
        for (i64 i = 0; i < m; ++i) out[i] = a;
        return;
    }
    const i64 n1 = m - 1;
    const double by = (b - a) / static_cast<double>(n1);
    out[0] = a;
    for (i64 i = 1; i < n1; ++i) out[i] = a + static_cast<double>(i) * by;
    out[n1] = b;
}

double r_var_of(const double* x, i64 n) {
    if (n < 2) return kNaN;
    const double mean = lsm::r_mean(x, n);
#if defined(LSM_X87_MASM)
    return lsm_x87_ssd(x, n, mean, n - 1);
#else
    const lsm::Extended m = mean;
    lsm::Extended ss = 0.0;
    for (i64 i = 0; i < n; ++i) {
        const lsm::Extended d = lsm::Extended(x[i]) - m;
        ss += d * d;
    }
    return static_cast<double>(ss / lsm::Extended(n - 1));
#endif
}

double r_colmean_of(const double* x, i64 n) {
    lsm::Extended s = 0.0;
    for (i64 i = 0; i < n; ++i) s += x[i];
    s /= lsm::Extended(n);
    return static_cast<double>(s);
}

void r_quantile7(const double* x, i64 n, const double* probs, i64 np, double* out) {
    std::vector<double> s(x, x + n);
    std::sort(s.begin(), s.end());
    for (i64 k = 0; k < np; ++k) {
        const double p = std::max(0.0, std::min(1.0, probs[k]));
        const double index = 1 + std::max(static_cast<double>(n) - 1, 0.0) * p;
        const double lo = std::floor(index), hi = std::ceil(index);
        double qs = s[static_cast<std::size_t>(lo) - 1];
        const double xh = s[static_cast<std::size_t>(hi) - 1];
        if (index > lo && xh != qs) {
            const double h = index - lo;
            qs = (1 - h) * qs + h * xh;
        }
        out[k] = qs;
    }
}

double r_dnorm(double x) {
    if (std::isnan(x)) return x;
    if (!std::isfinite(x)) return 0.0;
    x = std::fabs(x);
    if (x >= 2 * std::sqrt(std::numeric_limits<double>::max())) return 0.0;
    if (x < 5) return kInvSqrt2Pi * lsm::r_exp(-0.5 * x * x) / 1.0;
    if (x > std::sqrt(-2 * 0.693147180559945309417232121458 * (-1021 + 1 - 53))) return 0.0;
    const double x1 = std::ldexp(std::nearbyint(std::ldexp(x, 16)), -16);
    const double x2 = x - x1;
    return kInvSqrt2Pi / 1.0 * (lsm::r_exp(-0.5 * x1 * x1) * lsm::r_exp((-0.5 * x2 - x1) * x2));
}

// ------------------------------------------------------------------------------------------------ chull

namespace {

// R's chull.c (Eddy's algorithm 523): the arrays are 1-based as in the original.
void chull_split(int n, double* x, int m, int* in, int ii, int jj, int s, int* iabv, int* na, int* maxa, int* ibel,
                 int* nb, int* maxb) {
    double a = 0, b = 0, down, d1, up, xt, z;
    int i, is;
    bool vert, neg_dir = false;

    --x;
    xt = x[ii];
    vert = (x[jj] == xt);
    d1 = x[jj + n] - x[ii + n];
    if (vert) {
        neg_dir = ((s > 0 && d1 < 0.) || (s < 0 && d1 > 0.));
    } else {
        a = d1 / (x[jj] - xt);
        b = x[ii + n] - a * xt;
    }
    up = 0.;
    *na = 0;
    *maxa = 0;
    down = 0.;
    *nb = 0;
    *maxb = 0;
    for (i = 0; i < m; ++i) {
        is = in[i];
        if (vert) {
            if (neg_dir)
                z = xt - x[is];
            else
                z = x[is] - xt;
        } else {
            z = x[is + n] - a * x[is] - b;
        }
        if (z > 0.) {
            if (s == -2) continue;
            iabv[*na] = is;
            ++(*na);
            if (z >= up) {
                up = z;
                *maxa = *na;
            }
        } else if (s != 2 && z < 0.) {
            ibel[*nb] = is;
            ++(*nb);
            if (z <= down) {
                down = z;
                *maxb = *nb;
            }
        }
    }
}

void in_chull(int* n, double* x, int* m, int* in, int* ia, int* ib, int* ih, int* nh, int* il) {
#define Y(k) x[k + x_dim1]
    bool mine, maxe;
    int i, j, ilinh, ma, mb, kn, mm, kx, mx, mp1, mbb, nia, nib, inh, min, mxa, mxb, mxbb;
    int x_dim1, x_offset;
    double d1;

    x_dim1 = *n;
    x_offset = 1;
    x -= x_offset;
    --il;
    --ih;
    --ib;
    --ia;
    --in;

    if (*m == 1) goto L_1pt;
    il[1] = 2;
    il[2] = 1;
    kn = in[1];
    kx = in[2];
    if (*m == 2) goto L_2pts;
    mp1 = *m + 1;
    min = 1;
    mx = 1;
    kx = in[1];
    maxe = false;
    mine = false;
    for (i = 2; i <= *m; ++i) {
        j = in[i];
        if ((d1 = x[j] - x[kx]) < 0.) {
        } else if (d1 == 0) {
            maxe = true;
        } else {
            maxe = false;
            mx = i;
            kx = j;
        }
        if ((d1 = x[j] - x[kn]) < 0.) {
            mine = false;
            min = i;
            kn = j;
        } else if (d1 == 0) {
            mine = true;
        }
    }

    if (kx == kn) goto L_vertical;

    if (maxe || mine) {
        if (maxe) {
            for (i = 1; i <= *m; ++i) {
                j = in[i];
                if (x[j] != x[kx]) continue;
                if (Y(j) <= Y(kx)) continue;
                mx = i;
                kx = j;
            }
        }
        if (mine) {
            for (i = 1; i <= *m; ++i) {
                j = in[i];
                if (x[j] != x[kn]) continue;
                if (Y(j) >= Y(kn)) continue;
                min = i;
                kn = j;
            }
        }
    }

    ih[1] = kx;
    ih[2] = kn;
    *nh = 3;
    inh = 1;
    nib = 1;
    ma = *m;
    in[mx] = in[*m];
    in[*m] = kx;
    mm = *m - 2;
    if (min == *m) min = mx;
    in[min] = in[*m - 1];
    in[*m - 1] = kn;
    chull_split(*n, &x[x_offset], mm, &in[1], ih[1], ih[2], 0, &ia[1], &mb, &mxa, &ib[1], &ia[ma], &mxbb);

    // the left half of the tree, starting with the left son
L8:
    nib += ia[ma];
    --ma;
    do {
        if (mxa != 0) {
            il[*nh] = il[inh];
            il[inh] = *nh;
            ih[*nh] = ia[mxa];
            ia[mxa] = ia[mb];
            --mb;
            ++(*nh);
            if (mb != 0) {
                ilinh = il[inh];
                chull_split(*n, &x[x_offset], mb, &ia[1], ih[inh], ih[ilinh], 1, &ia[1], &mbb, &mxa, &ib[nib],
                            &ia[ma], &mxb);
                mb = mbb;
                goto L8;
            }
            inh = il[inh];
        }
        do {
            inh = il[inh];
            ++ma;
            nib -= ia[ma];
            if (ma >= *m) goto L12;
        } while (ia[ma] == 0);
        ilinh = il[inh];
        chull_split(*n, &x[x_offset], ia[ma], &ib[nib], ih[inh], ih[ilinh], 2, &ia[1], &mb, &mxa, &ib[nib], &mbb,
                    &mxb);
        ia[ma] = mbb;
    } while (true);

    // the right half of the tree
L12:
    mxb = mxbb;
    ma = *m;
    mb = ia[ma];
    nia = 1;
    ia[ma] = 0;
L13:
    nia += ia[ma];
    --ma;
    do {
        if (mxb != 0) {
            il[*nh] = il[inh];
            il[inh] = *nh;
            ih[*nh] = ib[mxb];
            ib[mxb] = ib[mb];
            --mb;
            ++(*nh);
            if (mb != 0) {
                ilinh = il[inh];
                chull_split(*n, &x[x_offset], mb, &ib[nib], ih[inh], ih[ilinh], -1, &ia[nia], &ia[ma], &mxa,
                            &ib[nib], &mbb, &mxb);
                mb = mbb;
                goto L13;
            }
            inh = il[inh];
        }
        do {
            inh = il[inh];
            ++ma;
            if (ma == mp1) goto Finis;
            nia -= ia[ma];
        } while (ia[ma] == 0);
        ilinh = il[inh];
        chull_split(*n, &x[x_offset], ia[ma], &ia[nia], ih[inh], ih[ilinh], -2, &ia[nia], &mbb, &mxa, &ib[nib], &mb,
                    &mxb);
    } while (true);

L_vertical:
    kx = in[1];
    kn = in[1];
    for (i = 1; i <= *m; ++i) {
        j = in[i];
        if (Y(j) > Y(kx)) {
            mx = i;
            kx = j;
        }
        if (Y(j) < Y(kn)) {
            min = i;
            kn = j;
        }
    }
    if (kx == kn) goto L_1pt;

L_2pts:
    ih[1] = kx;
    ih[2] = kn;
    if (x[kn] == x[kx] && Y(kn) == Y(kx))
        *nh = 2;
    else
        *nh = 3;
    goto Finis;

L_1pt:
    *nh = 2;
    ih[1] = in[1];
    il[1] = 1;

Finis:
    --(*nh);
    for (i = 1; i <= *nh; ++i) ia[i] = ih[i];
    j = il[1];
    for (i = 2; i <= *nh; ++i) {
        ih[i] = ia[j];
        j = il[j];
    }
    (void)mbb;
    (void)mxb;
#undef Y
}

}  // namespace

i64 r_chull(const double* x, const double* y, i64 n, i64* out) {
    if (n <= 0) return 0;
    const int nn = static_cast<int>(n);
    std::vector<double> xy(static_cast<std::size_t>(2 * n));
    for (i64 i = 0; i < n; ++i) {
        xy[static_cast<std::size_t>(i)] = x[i];
        xy[static_cast<std::size_t>(n + i)] = y[i];
    }
    std::vector<int> in(static_cast<std::size_t>(n)), ih(static_cast<std::size_t>(4 * n) + 8);
    for (int i = 0; i < nn; ++i) in[static_cast<std::size_t>(i)] = i + 1;
    int nh = 0, m = nn, nvar = nn;
    in_chull(&nvar, xy.data(), &m, in.data(), ih.data() + n, ih.data() + 2 * n, ih.data(), &nh, ih.data() + 3 * n);
    for (int i = 0; i < nh; ++i) out[i] = ih[static_cast<std::size_t>(nh - 1 - i)] - 1;
    if (nh < 2) return nh;
    // grDevices::chull's R code fixes the order: xx <- sweep(x[res, ], 2, colMeans(x[res, ])); angs <- atan2(xx[, 2],
    // -xx[, 1]); res[order(angs)]
    std::vector<double> hx(static_cast<std::size_t>(nh)), hy(static_cast<std::size_t>(nh)), ang(static_cast<std::size_t>(nh));
    for (int i = 0; i < nh; ++i) {
        hx[static_cast<std::size_t>(i)] = x[out[i]];
        hy[static_cast<std::size_t>(i)] = y[out[i]];
    }
    const double cx = r_colmean_of(hx.data(), nh), cy = r_colmean_of(hy.data(), nh);
    for (int i = 0; i < nh; ++i)
        ang[static_cast<std::size_t>(i)] = lsm::r_atan2(hy[static_cast<std::size_t>(i)] - cy, -(hx[static_cast<std::size_t>(i)] - cx));
    std::vector<int> ord(static_cast<std::size_t>(nh));
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return ang[static_cast<std::size_t>(a)] < ang[static_cast<std::size_t>(b)]; });
    std::vector<i64> sorted(static_cast<std::size_t>(nh));
    for (int i = 0; i < nh; ++i) sorted[static_cast<std::size_t>(i)] = out[ord[static_cast<std::size_t>(i)]];
    for (int i = 0; i < nh; ++i) out[i] = sorted[static_cast<std::size_t>(i)];
    return nh;
}

// ------------------------------------------------------------------------------------------- KernSmooth

void kern_linbin(const double* x, i64 n, double a, double b, i64 m, bool truncate, double* gcnts) {
    for (i64 i = 0; i < m; ++i) gcnts[i] = 0.0;
    const double delta = (b - a) / static_cast<double>(m - 1);
    for (i64 i = 0; i < n; ++i) {
        const double lxi = ((x[i] - a) / delta) + 1;
        // Fortran's int(): the truncation toward zero (a value out of range of an integer is left out)
        if (!(std::fabs(lxi) < 2147483647.0)) {
            if (!truncate && lxi >= static_cast<double>(m)) gcnts[m - 1] += 1;
            if (!truncate && lxi <= -2147483647.0) gcnts[0] += 1;
            continue;
        }
        const i64 li = static_cast<i64>(lxi);
        const double rem = lxi - static_cast<double>(li);
        if (li >= 1 && li < m) {
            gcnts[li - 1] = gcnts[li - 1] + (1 - rem);
            gcnts[li] = gcnts[li] + rem;
        }
        if (li < 1 && !truncate) gcnts[0] = gcnts[0] + 1;
        if (li >= m && !truncate) gcnts[m - 1] = gcnts[m - 1] + 1;
    }
}

void kern_linbin2d(const double* x, const double* y, i64 n, double a1, double a2, double b1, double b2, i64 m1, i64 m2,
                   double* gcnts) {
    for (i64 i = 0; i < m1 * m2; ++i) gcnts[i] = 0.0;
    const double delta1 = (b1 - a1) / static_cast<double>(m1 - 1);
    const double delta2 = (b2 - a2) / static_cast<double>(m2 - 1);
    for (i64 i = 0; i < n; ++i) {
        const double lxi1 = ((x[i] - a1) / delta1) + 1;
        const double lxi2 = ((y[i] - a2) / delta2) + 1;
        if (!(std::fabs(lxi1) < 2147483647.0) || !(std::fabs(lxi2) < 2147483647.0)) continue;
        const i64 li1 = static_cast<i64>(lxi1);
        const i64 li2 = static_cast<i64>(lxi2);
        const double rem1 = lxi1 - static_cast<double>(li1);
        const double rem2 = lxi2 - static_cast<double>(li2);
        if (li1 >= 1 && li2 >= 1 && li1 < m1 && li2 < m2) {
            const i64 ind1 = m1 * (li2 - 1) + li1;  // 1-based as in the Fortran
            const i64 ind2 = ind1 + 1;
            const i64 ind3 = m1 * li2 + li1;
            const i64 ind4 = ind3 + 1;
            gcnts[ind1 - 1] = gcnts[ind1 - 1] + (1 - rem1) * (1 - rem2);
            gcnts[ind2 - 1] = gcnts[ind2 - 1] + rem1 * (1 - rem2);
            gcnts[ind3 - 1] = gcnts[ind3 - 1] + (1 - rem1) * rem2;
            gcnts[ind4 - 1] = gcnts[ind4 - 1] + rem1 * rem2;
        }
    }
}

namespace {

// 2^(ceiling(log(v) / log(2))) as R computes it (the x87 log, then 2^k exactly)
double pow2_ceiling_log2(double v) {
    const double k = std::ceil(lsm::r_log(v) / lsm::r_log(2.0));
    return lsm::r_pow(2.0, k);
}

}  // namespace

int kern_bkde2d(const double* x, const double* y, i64 n, double h1, double h2, double a1, double b1, double a2,
                double b2, i64 M1, i64 M2, double* fhat) {
    const double tau = 3.4;
    const double h[2] = {h1, h2};
    const double a[2] = {a1, a2}, b[2] = {b1, b2};
    const i64 M[2] = {M1, M2};
    std::vector<double> gcounts(static_cast<std::size_t>(M1 * M2));
    kern_linbin2d(x, y, n, a1, a2, b1, b2, M1, M2, gcounts.data());

    double L[2];
    std::vector<double> kapid[2];
    for (int id = 0; id < 2; ++id) {
        L[id] = std::min(std::floor(tau * h[id] * static_cast<double>(M[id] - 1) / (b[id] - a[id])),
                         static_cast<double>(M[id] - 1));
        const i64 Lid = static_cast<i64>(L[id]);
        const double facid = (b[id] - a[id]) / (h[id] * static_cast<double>(M[id] - 1));
        std::vector<double> z(static_cast<std::size_t>(Lid + 1));
        for (i64 l = 0; l <= Lid; ++l) z[static_cast<std::size_t>(l)] = r_dnorm(static_cast<double>(l) * facid) / h[id];
        // tot <- sum(c(z, rev(z[-1]))) * facid * h
        std::vector<double> both;
        both.reserve(static_cast<std::size_t>(2 * Lid + 1));
        for (const double v : z) both.push_back(v);
        for (i64 l = Lid; l >= 1; --l) both.push_back(z[static_cast<std::size_t>(l)]);
        const double tot = lsm::r_sum(both.data(), static_cast<i64>(both.size())) * facid * h[id];
        kapid[id].resize(z.size());
        for (std::size_t l = 0; l < z.size(); ++l) kapid[id][l] = z[l] / tot;
    }
    const i64 L1 = static_cast<i64>(L[0]), L2 = static_cast<i64>(L[1]);
    const int warn = (std::min(L[0], L[1]) == 0) ? 1 : 0;
    // kapp <- kapid[[1]] %*% t(kapid[[2]]) / n
    const double dn = static_cast<double>(n);
    std::vector<double> kapp(static_cast<std::size_t>((L1 + 1) * (L2 + 1)));
    for (i64 j = 0; j <= L2; ++j)
        for (i64 i = 0; i <= L1; ++i)
            kapp[static_cast<std::size_t>(i + j * (L1 + 1))] =
                (kapid[0][static_cast<std::size_t>(i)] * kapid[1][static_cast<std::size_t>(j)]) / dn;

    const i64 P1 = static_cast<i64>(pow2_ceiling_log2(static_cast<double>(M1) + L[0]));
    const i64 P2 = static_cast<i64>(pow2_ceiling_log2(static_cast<double>(M2) + L[1]));
    // rp: P1 x P2 complex; sp likewise
    std::vector<double> rp(static_cast<std::size_t>(2 * P1 * P2), 0.0), sp(static_cast<std::size_t>(2 * P1 * P2), 0.0);
    auto RP = [&](i64 i, i64 j) -> double& { return rp[static_cast<std::size_t>(2 * (i + j * P1))]; };
    for (i64 j = 0; j <= L2; ++j)
        for (i64 i = 0; i <= L1; ++i) RP(i, j) = kapp[static_cast<std::size_t>(i + j * (L1 + 1))];
    if (L1)
        for (i64 j = 0; j <= L2; ++j)
            for (i64 q = 0; q < L1; ++q)  // rp[(P1 - L1 + 1):P1, j] <- kapp[(L1 + 1):2, j]
                RP(P1 - L1 + q, j) = kapp[static_cast<std::size_t>((L1 - q) + j * (L1 + 1))];
    if (L2)
        for (i64 q = 0; q < L2; ++q)  // rp[, (P2 - L2 + 1):P2] <- rp[, (L2 + 1):2]
            for (i64 i = 0; i < P1; ++i) RP(i, P2 - L2 + q) = RP(i, L2 - q);
    for (i64 j = 0; j < M2; ++j)
        for (i64 i = 0; i < M1; ++i) sp[static_cast<std::size_t>(2 * (i + j * P1))] = gcounts[static_cast<std::size_t>(i + j * M1)];
    r_fft(rp.data(), P1, P2, false);
    r_fft(sp.data(), P1, P2, false);
    // rp * sp: R's complex product
    for (i64 k = 0; k < P1 * P2; ++k) {
        const double ar = rp[static_cast<std::size_t>(2 * k)], ai = rp[static_cast<std::size_t>(2 * k + 1)];
        const double br = sp[static_cast<std::size_t>(2 * k)], bi = sp[static_cast<std::size_t>(2 * k + 1)];
        rp[static_cast<std::size_t>(2 * k)] = ar * br - ai * bi;
        rp[static_cast<std::size_t>(2 * k + 1)] = ar * bi + ai * br;
    }
    r_fft(rp.data(), P1, P2, true);
    const double total = static_cast<double>(P1 * P2);
    for (i64 j = 0; j < M2; ++j)
        for (i64 i = 0; i < M1; ++i) {
            const double v = rp[static_cast<std::size_t>(2 * (i + j * P1))] / total;
            fhat[i + j * M1] = v * (v > 0 ? 1.0 : 0.0);
        }
    return warn;
}

namespace {

// KernSmooth::bkfe(gcounts, drv, bandwidth, range.x = c(a, b), binned = TRUE)
double bkfe_binned(const double* gcounts, i64 M, int drv, double h, double a, double b) {
    const double n = lsm::r_sum(gcounts, M);
    const double delta = (b - a) / static_cast<double>(M - 1);
    const double tau = 4 + drv;
    const i64 L = static_cast<i64>(std::min(std::floor(tau * h / delta), static_cast<double>(M)));
    std::vector<double> arg(static_cast<std::size_t>(L + 1)), kappam(static_cast<std::size_t>(L + 1));
    const double hp = lsm::r_pow(h, static_cast<double>(drv + 1));
    for (i64 l = 0; l <= L; ++l) {
        arg[static_cast<std::size_t>(l)] = static_cast<double>(l) * delta / h;
        kappam[static_cast<std::size_t>(l)] = r_dnorm(arg[static_cast<std::size_t>(l)]) / hp;
    }
    std::vector<double> hmold0(arg.size(), 1.0), hmold1 = arg, hmnew(arg.size(), 1.0);
    if (drv >= 2) {
        for (int i = 2; i <= drv; ++i) {
            for (std::size_t l = 0; l < arg.size(); ++l)
                hmnew[l] = arg[l] * hmold1[l] - static_cast<double>(i - 1) * hmold0[l];
            hmold0 = hmold1;
            hmold1 = hmnew;
        }
    }
    for (std::size_t l = 0; l < kappam.size(); ++l) kappam[l] = hmnew[l] * kappam[l];
    const i64 P = static_cast<i64>(pow2_ceiling_log2(static_cast<double>(M + L + 1)));
    // kappam <- c(kappam, rep(0, P - 2L - 1), rev(kappam[-1]))
    std::vector<double> kz(static_cast<std::size_t>(2 * P), 0.0), gz(static_cast<std::size_t>(2 * P), 0.0);
    for (i64 l = 0; l <= L; ++l) kz[static_cast<std::size_t>(2 * l)] = kappam[static_cast<std::size_t>(l)];
    for (i64 q = 0; q < L; ++q) kz[static_cast<std::size_t>(2 * (P - L + q))] = kappam[static_cast<std::size_t>(L - q)];
    for (i64 i = 0; i < M; ++i) gz[static_cast<std::size_t>(2 * i)] = gcounts[i];
    r_fft(kz.data(), P, 1, false);
    r_fft(gz.data(), P, 1, false);
    for (i64 k = 0; k < P; ++k) {
        const double ar = kz[static_cast<std::size_t>(2 * k)], ai = kz[static_cast<std::size_t>(2 * k + 1)];
        const double br = gz[static_cast<std::size_t>(2 * k)], bi = gz[static_cast<std::size_t>(2 * k + 1)];
        kz[static_cast<std::size_t>(2 * k)] = ar * br - ai * bi;
        kz[static_cast<std::size_t>(2 * k + 1)] = ar * bi + ai * br;
    }
    r_fft(kz.data(), P, 1, true);
    const double dP = static_cast<double>(P);
    const double s = lsm::sum_of(0, M, [&](i64 i) { return gcounts[i] * (kz[static_cast<std::size_t>(2 * i)] / dP); });
    return s / (n * n);
}

}  // namespace

double kern_dpik(const double* x, i64 n, int scalest, int level, int kernel, bool canonical, i64 gridsize, double a,
                 double b, bool truncate, int* error) {
    *error = 0;
    if (level > 5 || level < 0) {
        *error = 1;
        return kNaN;
    }
    double del0 = 1;
    if (!canonical) {
        switch (kernel) {
            case 0: del0 = 1 / lsm::r_pow(4 * kPi, 1.0 / 10); break;
            case 1: del0 = lsm::r_pow(9.0 / 2, 1.0 / 5); break;
            case 2: del0 = lsm::r_pow(15.0, 1.0 / 5); break;
            case 3: del0 = lsm::r_pow(35.0, 1.0 / 5); break;
            default: del0 = lsm::r_pow(9450.0 / 143, 1.0 / 5); break;
        }
    }
    const i64 M = gridsize;
    const double dn = static_cast<double>(n);
    // scalest
    double q[2];
    const double probs[2] = {3.0 / 4, 1.0 / 4};
    r_quantile7(x, n, probs, 2, q);
    const double iqr = (q[0] - q[1]) / 1.349;
    const double sdv = std::sqrt(r_var_of(x, n));
    double sc;
    if (scalest == 1) sc = sdv;
    else if (scalest == 2) sc = iqr;
    else sc = std::min(iqr, sdv);
    if (sc == 0) {
        *error = 2;
        return kNaN;
    }
    const double mx = lsm::r_mean(x, n);
    std::vector<double> sx(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) sx[static_cast<std::size_t>(i)] = (x[i] - mx) / sc;
    const double sa = (a - mx) / sc, sb = (b - mx) / sc;
    std::vector<double> gp(static_cast<std::size_t>(M));
    r_seq_len(sa, sb, M, gp.data());
    std::vector<double> gcounts(static_cast<std::size_t>(M));
    kern_linbin(sx.data(), n, gp[0], gp[static_cast<std::size_t>(M - 1)], M, truncate, gcounts.data());
    const double sqrt2 = std::sqrt(2.0);
    const double sp = std::sqrt(2 / kPi);
    auto bk = [&](int drv, double alpha) { return bkfe_binned(gcounts.data(), M, drv, alpha, sa, sb); };
    double psi4hat;
    if (level == 0) {
        psi4hat = 3 / (8 * std::sqrt(kPi));
    } else if (level == 1) {
        const double alpha = lsm::r_pow(2 * lsm::r_pow(sqrt2, 7) / (5 * dn), 1.0 / 7);
        psi4hat = bk(4, alpha);
    } else if (level == 2) {
        double alpha = lsm::r_pow(2 * lsm::r_pow(sqrt2, 9) / (7 * dn), 1.0 / 9);
        const double psi6hat = bk(6, alpha);
        alpha = lsm::r_pow(-3 * sp / (psi6hat * dn), 1.0 / 7);
        psi4hat = bk(4, alpha);
    } else if (level == 3) {
        double alpha = lsm::r_pow(2 * lsm::r_pow(sqrt2, 11) / (9 * dn), 1.0 / 11);
        const double psi8hat = bk(8, alpha);
        alpha = lsm::r_pow(15 * sp / (psi8hat * dn), 1.0 / 9);
        const double psi6hat = bk(6, alpha);
        alpha = lsm::r_pow(-3 * sp / (psi6hat * dn), 1.0 / 7);
        psi4hat = bk(4, alpha);
    } else if (level == 4) {
        double alpha = lsm::r_pow(2 * lsm::r_pow(sqrt2, 13) / (11 * dn), 1.0 / 13);
        const double psi10hat = bk(10, alpha);
        alpha = lsm::r_pow(-105 * sp / (psi10hat * dn), 1.0 / 11);
        const double psi8hat = bk(8, alpha);
        alpha = lsm::r_pow(15 * sp / (psi8hat * dn), 1.0 / 9);
        const double psi6hat = bk(6, alpha);
        alpha = lsm::r_pow(-3 * sp / (psi6hat * dn), 1.0 / 7);
        psi4hat = bk(4, alpha);
    } else {
        double alpha = lsm::r_pow(2 * lsm::r_pow(sqrt2, 15) / (13 * dn), 1.0 / 15);
        const double psi12hat = bk(12, alpha);
        alpha = lsm::r_pow(945 * sp / (psi12hat * dn), 1.0 / 13);
        const double psi10hat = bk(10, alpha);
        alpha = lsm::r_pow(-105 * sp / (psi10hat * dn), 1.0 / 11);
        const double psi8hat = bk(8, alpha);
        alpha = lsm::r_pow(15 * sp / (psi8hat * dn), 1.0 / 9);
        const double psi6hat = bk(6, alpha);
        alpha = lsm::r_pow(-3 * sp / (psi6hat * dn), 1.0 / 7);
        psi4hat = bk(4, alpha);
    }
    return sc * del0 * lsm::r_pow(1 / (psi4hat * dn), 1.0 / 5);
}

// ------------------------------------------------------------------------------------------------ lscv

void amt_lscv(const double* x, const double* y, i64 n, const double* hs, i64 nh, int threads, double* out) {
    // f <- as.matrix(dist(x)); f <- f[lower.tri(f)]; then f^2: the squared distances in R's order
    const i64 m = n * (n - 1) / 2;
    std::vector<double> f2(static_cast<std::size_t>(m));
    {
        i64 k = 0;
        for (i64 j = 0; j < n; ++j)
            for (i64 i = j + 1; i < n; ++i) {
                double dist = 0;
                double dev = x[i] - x[j];
                dist += dev * dev;
                dev = y[i] - y[j];
                dist += dev * dev;
                const double f = std::sqrt(dist);
                f2[static_cast<std::size_t>(k++)] = f * f;
            }
    }
    const double dn = static_cast<double>(n);
    lsm::parallel_for_dynamic(threads > 1 && nh > 1, nh, [&](i64 k) {
        const double h = hs[k];
        const double hh = h * h;
        const double d1 = 4 * hh, d2 = 2 * hh;
        const double sum = lsm::sum_of(0, m, [&](i64 i) {
            const double v = f2[static_cast<std::size_t>(i)];
            return lsm::r_exp(-v / d1) - 4 * lsm::r_exp(-v / d2);
        });
        out[k] = 1 / (kPi * hh * dn) + (2 * sum - 3 * dn) / (kPi * 4 * hh * (dn * dn));
    });
}

double ogr_ring_area(const double* xy, i64 n) {
    if (n < 2) return 0.0;
    if (xy[0] != xy[2 * (n - 1)] || xy[1] != xy[2 * (n - 1) + 1]) return 0.0;   // not closed
    auto X = [&](i64 i) { return xy[2 * i]; };
    auto Y = [&](i64 i) { return xy[2 * i + 1]; };
    double sum = X(0) * (Y(1) - Y(n - 1));
    for (i64 i = 1; i < n - 1; ++i) sum += X(i) * (Y(i + 1) - Y(i - 1));
    sum += X(n - 1) * (Y(0) - Y(n - 2));
    return 0.5 * std::fabs(sum);
}

// ------------------------------------------------------------------------------------------------- cud

void amt_cud(const double* v, i64 n, double* out) {
    const double total = lsm::r_sum(v, n);
    std::vector<double> w(static_cast<std::size_t>(n));
    for (i64 i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = v[i] / total;
    std::vector<i64> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    // order(-v): ascending in -v, ties in their order
    std::stable_sort(idx.begin(), idx.end(),
                     [&](i64 p, i64 q) { return -w[static_cast<std::size_t>(p)] < -w[static_cast<std::size_t>(q)]; });
    lsm::Extended s = 0.0;
    for (i64 k = 0; k < n; ++k) {
        s += w[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
        out[idx[static_cast<std::size_t>(k)]] = static_cast<double>(s);
    }
}

double amt_vol_base(const double* ui, const double* uj, i64 n, int type) {
    switch (type) {
        case 0: {  // sum(ud_j[ud_i > 0])
            std::vector<double> sel;
            for (i64 k = 0; k < n; ++k)
                if (ui[k] > 0) sel.push_back(uj[k]);
            return lsm::r_sum(sel.data(), static_cast<i64>(sel.size()));
        }
        case 1:  // sum(pmin(ud_i, ud_j))
            return lsm::sum_of(0, n, [&](i64 k) { return (std::isnan(ui[k]) || std::isnan(uj[k])) ? kNaN : std::min(ui[k], uj[k]); });
        case 2:
        case 4: {  // sum(sqrt(ud_i) * sqrt(ud_j)) (R's vector sqrt), and hd: 2 * (1 - that)
            const double s = lsm::sum_of(0, n, [&](i64 k) { return lsm::r_sqrt(ui[k]) * lsm::r_sqrt(uj[k]); });
            return type == 2 ? s : 2 * (1 - s);
        }
        default: {  // udoi: A_ij * sum(ud_i * ud_j)
            i64 a = 0;
            for (i64 k = 0; k < n; ++k)
                if (ui[k] > 0 && uj[k] > 0) ++a;
            return static_cast<double>(a) * lsm::sum_of(0, n, [&](i64 k) { return ui[k] * uj[k]; });
        }
    }
}

// ----------------------------------------------------------------------------------------------- LoCoH

void amt_locoh_neighbours(const double* x, const double* y, i64 n, int type, double d, int threads,
                          std::vector<i64>& off, std::vector<i64>& ids) {
    std::vector<std::vector<i64>> per(static_cast<std::size_t>(n));
    lsm::parallel_for_dynamic(threads > 1 && n > 200, n, [&](i64 i) {
        std::vector<i64>& out = per[static_cast<std::size_t>(i)];
        if (type == 0) {
            const i64 k = static_cast<i64>(d);
            std::vector<std::pair<double, i64>> dd;
            dd.reserve(static_cast<std::size_t>(n - 1));
            for (i64 j = 0; j < n; ++j) {
                if (j == i) continue;
                const double dx = x[i] - x[j], dy = y[i] - y[j];
                dd.emplace_back(dx * dx + dy * dy, j);
            }
            const std::size_t kk = std::min<std::size_t>(static_cast<std::size_t>(k), dd.size());
            std::partial_sort(dd.begin(), dd.begin() + static_cast<std::ptrdiff_t>(kk), dd.end());
            for (std::size_t q = 0; q < kk; ++q) out.push_back(dd[q].second);
        } else if (type == 1) {
            for (i64 j = 0; j < n; ++j) {
                const double dx = x[j] - x[i], dy = y[j] - y[i];
                if (lsm::r_sqrt(dx * dx + dy * dy) <= d) out.push_back(j);
            }
        } else {
            std::vector<std::pair<double, i64>> di(static_cast<std::size_t>(n));
            for (i64 j = 0; j < n; ++j) {
                const double dx = x[j] - x[i], dy = y[j] - y[i];
                di[static_cast<std::size_t>(j)] = {lsm::r_sqrt(dx * dx + dy * dy), j};
            }
            std::stable_sort(di.begin(), di.end(),
                             [](const std::pair<double, i64>& p, const std::pair<double, i64>& q) {
                                 return p.first < q.first;
                             });
            lsm::Extended s = 0.0;
            for (const auto& p : di) {
                s += p.first;
                if (static_cast<double>(s) <= d) out.push_back(p.second);
            }
        }
    });
    off.assign(static_cast<std::size_t>(n) + 1, 0);
    for (i64 i = 0; i < n; ++i) off[static_cast<std::size_t>(i) + 1] = off[static_cast<std::size_t>(i)] + static_cast<i64>(per[static_cast<std::size_t>(i)].size());
    ids.clear();
    ids.reserve(static_cast<std::size_t>(off[static_cast<std::size_t>(n)]));
    for (i64 i = 0; i < n; ++i) ids.insert(ids.end(), per[static_cast<std::size_t>(i)].begin(), per[static_cast<std::size_t>(i)].end());
}

// ------------------------------------------------------------------------------------------- contour

namespace {

// R's contour-common.h (the segment database of a level, R_alloc replaced by a deque that keeps the pointers).
struct Seg {
    Seg* next;
    double x0, y0, x1, y1;
};

using SegDb = std::vector<Seg*>;

bool ctr_intersect(double z0, double z1, double zc, double* f) {
    if ((z0 < zc) != (z1 < zc) && z0 != zc && z1 != zc) {
        *f = (zc - z0) / (z1 - z0);
        return true;
    }
    return false;
}

Seg* ctr_newseg(std::deque<Seg>& pool, double x0, double y0, double x1, double y1, Seg* prev) {
    pool.push_back(Seg{prev, x0, y0, x1, y1});
    return &pool.back();
}

void ctr_swapseg(Seg* seg) {
    const double x = seg->x0, y = seg->y0;
    seg->x0 = seg->x1;
    seg->y0 = seg->y1;
    seg->x1 = x;
    seg->y1 = y;
}

inline bool match(double a, double b) { return std::fabs(a - b) == 0; }

int ctr_segdir(double xend, double yend, const double* x, const double* y, int* i, int* j, int nx, int ny) {
    if (match(yend, y[*j])) {
        if (*j == 0) return 0;
        *j = *j - 1;
        return 3;
    }
    if (match(xend, x[*i])) {
        if (*i == 0) return 0;
        *i = *i - 1;
        return 4;
    }
    if (match(yend, y[*j + 1])) {
        if (*j >= ny - 1) return 0;
        *j = *j + 1;
        return 1;
    }
    if (match(xend, x[*i + 1])) {
        if (*i >= nx - 1) return 0;
        *i = *i + 1;
        return 2;
    }
    return 0;
}

Seg* ctr_segupdate(double xend, double yend, int dir, bool tail, Seg* seglist, Seg** seg) {
    if (seglist == nullptr) {
        *seg = nullptr;
        return nullptr;
    }
    switch (dir) {
        case 1:
        case 3:
            if (match(yend, seglist->y0)) {
                if (!tail) ctr_swapseg(seglist);
                *seg = seglist;
                return seglist->next;
            }
            if (match(yend, seglist->y1)) {
                if (tail) ctr_swapseg(seglist);
                *seg = seglist;
                return seglist->next;
            }
            break;
        case 2:
        case 4:
            if (match(xend, seglist->x0)) {
                if (!tail) ctr_swapseg(seglist);
                *seg = seglist;
                return seglist->next;
            }
            if (match(xend, seglist->x1)) {
                if (tail) ctr_swapseg(seglist);
                *seg = seglist;
                return seglist->next;
            }
            break;
    }
    seglist->next = ctr_segupdate(xend, yend, dir, tail, seglist->next, seg);
    return seglist;
}

// contourLines(): the segments of each cell for the level zc. z[i + j * nx]: column i, row j from the bottom.
SegDb contour_segments(std::deque<Seg>& pool, const double* x, int nx, const double* y, int ny, const double* z,
                       double zc, double atom) {
    double f, xl, xh, yl, yh, zll, zhl, zlh, zhh, xx[4], yy[4];
    int i, j, k, l, m, nacode;
    Seg* seglist;
    SegDb db(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), nullptr);
    for (i = 0; i < nx - 1; i++) {
        xl = x[i];
        xh = x[i + 1];
        for (j = 0; j < ny - 1; j++) {
            yl = y[j];
            yh = y[j + 1];
            k = i + j * nx;
            zll = z[k];
            zhl = z[k + 1];
            zlh = z[k + nx];
            zhh = z[k + nx + 1];
            // a corner exactly at the level is moved by a tiny amount
            if (zll == zc) zll += atom;
            if (zhl == zc) zhl += atom;
            if (zlh == zc) zlh += atom;
            if (zhh == zc) zhh += atom;
            nacode = 0;
            if (std::isfinite(zll)) nacode += 1;
            if (std::isfinite(zhl)) nacode += 2;
            if (std::isfinite(zlh)) nacode += 4;
            if (std::isfinite(zhh)) nacode += 8;
            k = 0;
            switch (nacode) {
                case 15:
                    if (ctr_intersect(zll, zhl, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yl;
                        k++;
                    }
                    if (ctr_intersect(zll, zlh, zc, &f)) {
                        yy[k] = yl + f * (yh - yl);
                        xx[k] = xl;
                        k++;
                    }
                    if (ctr_intersect(zhl, zhh, zc, &f)) {
                        yy[k] = yl + f * (yh - yl);
                        xx[k] = xh;
                        k++;
                    }
                    if (ctr_intersect(zlh, zhh, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yh;
                        k++;
                    }
                    break;
                case 14:
                    if (ctr_intersect(zhl, zhh, zc, &f)) {
                        yy[k] = yl + f * (yh - yl);
                        xx[k] = xh;
                        k++;
                    }
                    if (ctr_intersect(zlh, zhh, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yh;
                        k++;
                    }
                    if (ctr_intersect(zlh, zhl, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yh + f * (yl - yh);
                        k++;
                    }
                    break;
                case 13:
                    if (ctr_intersect(zll, zlh, zc, &f)) {
                        yy[k] = yl + f * (yh - yl);
                        xx[k] = xl;
                        k++;
                    }
                    if (ctr_intersect(zlh, zhh, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yh;
                        k++;
                    }
                    if (ctr_intersect(zll, zhh, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yl + f * (yh - yl);
                        k++;
                    }
                    break;
                case 11:
                    if (ctr_intersect(zhl, zhh, zc, &f)) {
                        yy[k] = yl + f * (yh - yl);
                        xx[k] = xh;
                        k++;
                    }
                    if (ctr_intersect(zll, zhl, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yl;
                        k++;
                    }
                    if (ctr_intersect(zll, zhh, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yl + f * (yh - yl);
                        k++;
                    }
                    break;
                case 7:
                    if (ctr_intersect(zll, zlh, zc, &f)) {
                        yy[k] = yl + f * (yh - yl);
                        xx[k] = xl;
                        k++;
                    }
                    if (ctr_intersect(zll, zhl, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yl;
                        k++;
                    }
                    if (ctr_intersect(zlh, zhl, zc, &f)) {
                        xx[k] = xl + f * (xh - xl);
                        yy[k] = yh + f * (yl - yh);
                        k++;
                    }
                    break;
            }
            seglist = nullptr;
            if (k > 0) {
                if (k == 2) {
                    seglist = ctr_newseg(pool, xx[0], yy[0], xx[1], yy[1], seglist);
                } else if (k == 4) {
                    for (k = 3; k >= 1; k--) {
                        m = k;
                        xl = xx[k];
                        for (l = 0; l < k; l++) {
                            if (xx[l] > xl) {
                                xl = xx[l];
                                m = l;
                            }
                        }
                        if (m != k) {
                            xl = xx[k];
                            yl = yy[k];
                            xx[k] = xx[m];
                            yy[k] = yy[m];
                            xx[m] = xl;
                            yy[m] = yl;
                        }
                    }
                    seglist = ctr_newseg(pool, xx[0], yy[0], xx[1], yy[1], seglist);
                    seglist = ctr_newseg(pool, xx[2], yy[2], xx[3], yy[3], seglist);
                } else {
                    throw std::runtime_error("contour: k should be 2 or 4");
                }
            }
            db[static_cast<std::size_t>(i + j * nx)] = seglist;
        }
    }
    return db;
}

}  // namespace

bool r_contour_lines(const double* v, i64 nr, i64 nc, double xmin, double ymax, double dx, double dy, double level,
                     std::vector<std::vector<double>>& lines) {
    lines.clear();
    const int nx = static_cast<int>(nc), ny = static_cast<int>(nr);
    std::vector<double> x(static_cast<std::size_t>(nx)), y(static_cast<std::size_t>(ny)),
        z(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    // xFromCol(1:ncol), yFromRow(nrow:1), z = t(matrix[nrow:1, ])
    for (int i = 0; i < nx; ++i) x[static_cast<std::size_t>(i)] = xmin + (static_cast<double>(i) + 0.5) * dx;
    for (int j = 0; j < ny; ++j) y[static_cast<std::size_t>(j)] = ymax - (static_cast<double>(ny - 1 - j) + 0.5) * dy;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            z[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)] =
                v[static_cast<i64>(ny - 1 - j) * nc + i];
    double zmin = std::numeric_limits<double>::max(), zmax = -std::numeric_limits<double>::max();
    for (const double a : z)
        if (std::isfinite(a)) {
            if (zmax < a) zmax = a;
            if (zmin > a) zmin = a;
        }
    if (zmin >= zmax) return false;
    const double atom = 1e-3 * (zmax - zmin);
    std::deque<Seg> pool;
    SegDb db = contour_segments(pool, x.data(), nx, y.data(), ny, z.data(), level, atom);
    constexpr int kMaxSegments = 25000;
    // addContourLines: follow the contours: grab a segment, follow its tail, follow its head, save the line
    for (int i = 0; i < nx - 1; i++)
        for (int j = 0; j < ny - 1; j++) {
            Seg* seglist;
            while ((seglist = db[static_cast<std::size_t>(i + j * nx)]) != nullptr) {
                int ii = i, jj = j, dir, ns;
                Seg *start, *end, *seg = nullptr, *s;
                start = end = seglist;
                db[static_cast<std::size_t>(i + j * nx)] = seglist->next;
                double xend = seglist->x1, yend = seglist->y1;
                while ((dir = ctr_segdir(xend, yend, x.data(), y.data(), &ii, &jj, nx, ny)) != 0) {
                    db[static_cast<std::size_t>(ii + jj * nx)] =
                        ctr_segupdate(xend, yend, dir, true, db[static_cast<std::size_t>(ii + jj * nx)], &seg);
                    if (!seg) break;
                    end->next = seg;
                    end = seg;
                    xend = end->x1;
                    yend = end->y1;
                }
                end->next = nullptr;
                ii = i;
                jj = j;
                xend = seglist->x0;
                yend = seglist->y0;
                while ((dir = ctr_segdir(xend, yend, x.data(), y.data(), &ii, &jj, nx, ny)) != 0) {
                    db[static_cast<std::size_t>(ii + jj * nx)] =
                        ctr_segupdate(xend, yend, dir, false, db[static_cast<std::size_t>(ii + jj * nx)], &seg);
                    if (!seg) break;
                    seg->next = start;
                    start = seg;
                    xend = start->x0;
                    yend = start->y0;
                }
                s = start;
                ns = 0;
                while (s && ns < kMaxSegments) {
                    ns++;
                    s = s->next;
                }
                if (ns == kMaxSegments) throw std::runtime_error("contour(): circular or too long a line of segments");
                std::vector<double> line;
                line.reserve(static_cast<std::size_t>(2 * (ns + 1)));
                s = start;
                line.push_back(s->x0);
                line.push_back(s->y0);
                while (s->next) {
                    s = s->next;
                    line.push_back(s->x0);
                    line.push_back(s->y0);
                }
                line.push_back(s->x1);
                line.push_back(s->y1);
                lines.push_back(std::move(line));
            }
        }
    return true;
}

}  // namespace move
