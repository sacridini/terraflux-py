// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of amt's step selection models (terraflux._core.move_issf_*): survival's exact conditional logistic
// regression and the pieces of R's arithmetic the formula and the summary need.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lsm/rmath.hpp"
#include "move/clogit.hpp"
#include "move/rnmath.hpp"
#include "nlm/rng.hpp"

namespace py = pybind11;

namespace move {

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using F64F = py::array_t<double, py::array::f_style | py::array::forcecast>;
using I32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

}  // namespace

void register_python_issf(py::module_& m) {
    m.def(
        "move_issf_fit",
        [](const F64F& x, const U8& status, const I32& stratum, int iter_max, double eps, double toler_chol,
           bool nocenter) {
            if (x.ndim() != 2 || status.size() != x.shape(0) || stratum.size() != x.shape(0))
                throw std::invalid_argument("x: an (n, p) array; status and stratum: n values");
            const auto n = static_cast<std::int64_t>(x.shape(0)), p = static_cast<std::int64_t>(x.shape(1));
            ClogitFit f;
            {
                py::gil_scoped_release release;
                f = clogit_fit(x.data(), n, p, stratum.data(), status.data(), iter_max, eps, toler_chol, nocenter);
            }
            auto vec = [](const std::vector<double>& v) {
                py::array_t<double> a(static_cast<py::ssize_t>(v.size()));
                std::copy(v.begin(), v.end(), a.mutable_data());
                return a;
            };
            auto mat = [&](const std::vector<double>& v) {
                // column-major, symmetric
                py::array_t<double> a({static_cast<py::ssize_t>(p), static_cast<py::ssize_t>(p)});
                for (std::int64_t r = 0; r < p; ++r)
                    for (std::int64_t c = 0; c < p; ++c) a.mutable_data()[r * p + c] = v[static_cast<std::size_t>(c * p + r)];
                return a;
            };
            py::dict out;
            out["coef"] = vec(f.coef);
            out["var"] = mat(f.var);
            out["coef_scaled"] = vec(f.coef_scaled);
            out["var_scaled"] = mat(f.var_scaled);
            out["u"] = vec(f.u);
            out["means"] = vec(f.means);
            out["rescale"] = vec(f.rescale);
            out["loglik"] = py::make_tuple(f.loglik0, f.loglik1);
            out["score"] = f.score;
            out["flag"] = f.flag;
            out["iter"] = f.iter;
            return out;
        },
        py::arg("x"), py::arg("status"), py::arg("stratum"), py::arg("iter_max") = 20, py::arg("eps") = 1e-9,
        py::arg("toler_chol") = 1.8189894035458565e-12, py::arg("nocenter") = true,
        "survival's coxexact.fit on the rows of x (n, p), their cases (0 / 1) and strata (integer codes).");

    m.def(
        "move_issf_wald",
        [](const F64& var, const F64& b) {
            const auto k = static_cast<int>(b.size());
            if (k < 1 || var.size() != static_cast<py::ssize_t>(k) * k) throw std::invalid_argument("var: (k, k), b: k");
            // var as R passes it: column-major (the caller gives the transpose of a row-major array)
            return clogit_wald(var.data(), b.data(), k, 1.8189894035458565e-12);
        },
        py::arg("var"), py::arg("b"), "survival's coxph.wtest: b' var^-1 b through the Cholesky factors.");

    m.def(
        "move_issf_fn",
        [](const std::string& name, const F64& x) {
            double (*f)(double) = nullptr;
            if (name == "log") f = [](double v) { return lsm::r_log(v); };
            else if (name == "exp") f = [](double v) { return lsm::r_exp(v); };
            else if (name == "sin") f = [](double v) { return lsm::r_sin(v); };
            else if (name == "cos") f = [](double v) { return lsm::r_cos(v); };
            else if (name == "log1p") f = [](double v) { return lsm::r_log1p(v); };
            else if (name == "sqrt") f = [](double v) { return lsm::r_sqrt(v, false); };
            else if (name == "sqrt1") f = [](double v) { return lsm::r_sqrt(v, true); };
            else throw std::invalid_argument("unknown function " + name);
            F64 out(x.size());
            for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = f(x.data()[i]);
            return out;
        },
        py::arg("name"), py::arg("x"), "R's log, exp, sin, cos, log1p and sqrt (sqrt1: of one number) of each value.");

    m.def(
        "move_issf_pow",
        [](const F64& x, const F64& y) {
            if (y.size() != x.size()) throw std::invalid_argument("x and y: the same length");
            F64 out(x.size());
            for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = lsm::r_pow(x.data()[i], y.data()[i]);
            return out;
        },
        py::arg("x"), py::arg("y"), "R's x^y.");

    m.def(
        "move_issf_sum", [](const F64& x) { return lsm::r_sum(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "R's sum() of doubles (long double).");

    m.def(
        "move_issf_pchisq",
        [](const F64& x, double df, bool lower_tail) {
            F64 out(x.size());
            for (py::ssize_t i = 0; i < x.size(); ++i) out.mutable_data()[i] = r_pchisq(x.data()[i], df, lower_tail);
            return out;
        },
        py::arg("x"), py::arg("df"), py::arg("lower_tail") = true, "R's pchisq(x, df, lower.tail).");

    m.def(
        "move_issf_qnorm",
        [](double p) {
            if (std::isnan(p)) return p;
            if (p < 0 || p > 1) return std::numeric_limits<double>::quiet_NaN();
            if (p == 0) return -std::numeric_limits<double>::infinity();
            if (p == 1) return std::numeric_limits<double>::infinity();
            return nlm::r_qnorm(p);
        },
        py::arg("p"), "R's qnorm(p).");

    m.def(
        "move_issf_sample",
        [](std::int32_t seed, std::int64_t n, std::int64_t size, std::int64_t draws) {
            // set.seed(seed), then `draws` times sample.int(n, size, replace = TRUE): the 1-based indices
            if (n < 1 || size < 0 || draws < 0) throw std::invalid_argument("n >= 1, size >= 0, draws >= 0");
            nlm::RRng rng(seed);
            py::array_t<std::int64_t> out({static_cast<py::ssize_t>(draws), static_cast<py::ssize_t>(size)});
            for (std::int64_t k = 0; k < draws; ++k) {
                const auto s = rng.sample_int(n, size, true);
                for (std::int64_t j = 0; j < size; ++j) out.mutable_data()[k * size + j] = s[static_cast<std::size_t>(j)];
            }
            return out;
        },
        py::arg("seed"), py::arg("n"), py::arg("size"), py::arg("draws"), "R's sample.int(n, size, TRUE) repeated.");
}

}  // namespace move
