// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of amt's home ranges (terraflux._core.move_hr_*).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "lsm/rmath.hpp"
#include "move/homerange.hpp"

namespace py = pybind11;

namespace move {

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

}  // namespace

void register_python_hr(py::module_& m) {
    m.def(
        "move_hr_chull",
        [](const F64& x, const F64& y) {
            if (x.ndim() != 1 || y.size() != x.size()) throw std::invalid_argument("x and y: 1-D arrays of the same length");
            const auto n = static_cast<std::int64_t>(x.size());
            std::vector<std::int64_t> out(static_cast<std::size_t>(n) + 1);
            const std::int64_t nh = r_chull(x.data(), y.data(), n, out.data());
            py::array_t<std::int64_t> res(nh);
            for (std::int64_t i = 0; i < nh; ++i) res.mutable_data()[i] = out[static_cast<std::size_t>(i)];
            return res;
        },
        py::arg("x"), py::arg("y"), "grDevices::chull: the 0-based indices of the hull's vertices in R's order.");

    m.def(
        "move_hr_quantile7",
        [](const F64& x, const F64& probs) {
            if (x.size() < 1) throw std::invalid_argument("x: at least one value");
            F64 out(probs.size());
            r_quantile7(x.data(), static_cast<std::int64_t>(x.size()), probs.data(),
                        static_cast<std::int64_t>(probs.size()), out.mutable_data());
            return out;
        },
        py::arg("x"), py::arg("probs"), "R's quantile(x, probs), type 7.");
    m.def(
        "move_hr_var", [](const F64& x) { return r_var_of(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "R's var() of a vector.");
    m.def(
        "move_hr_mean", [](const F64& x) { return lsm::r_mean(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "R's mean() of a vector.");
    m.def(
        "move_hr_colmean", [](const F64& x) { return r_colmean_of(x.data(), static_cast<std::int64_t>(x.size())); },
        py::arg("x"), "R's colMeans() of one column.");
    m.def(
        "move_hr_pow", [](double x, double y) { return lsm::r_pow(x, y); }, py::arg("x"), py::arg("y"), "R's x^y.");
    m.def(
        "move_hr_seq",
        [](double a, double b, std::int64_t m) {
            if (m < 0) throw std::invalid_argument("m: not negative");
            F64 out(m);
            r_seq_len(a, b, m, out.mutable_data());
            return out;
        },
        py::arg("a"), py::arg("b"), py::arg("m"), "R's seq(a, b, length.out = m).");

    m.def(
        "move_hr_fft",
        [](const F64& re, const F64& im, std::int64_t n1, std::int64_t n2, bool inverse) {
            if (re.size() != n1 * n2 || im.size() != n1 * n2) throw std::invalid_argument("re and im: n1 * n2 values");
            std::vector<double> z(static_cast<std::size_t>(2 * n1 * n2));
            for (std::int64_t i = 0; i < n1 * n2; ++i) {
                z[static_cast<std::size_t>(2 * i)] = re.data()[i];
                z[static_cast<std::size_t>(2 * i + 1)] = im.data()[i];
            }
            r_fft(z.data(), n1, n2, inverse);
            F64 ore(n1 * n2), oim(n1 * n2);
            for (std::int64_t i = 0; i < n1 * n2; ++i) {
                ore.mutable_data()[i] = z[static_cast<std::size_t>(2 * i)];
                oim.mutable_data()[i] = z[static_cast<std::size_t>(2 * i + 1)];
            }
            return py::make_tuple(ore, oim);
        },
        py::arg("re"), py::arg("im"), py::arg("n1"), py::arg("n2") = 1, py::arg("inverse") = false,
        "R's fft() of an n1 x n2 complex array (column-major; powers of two).");

    m.def(
        "move_hr_bkde2d",
        [](const F64& x, const F64& y, double h1, double h2, double a1, double b1, double a2, double b2,
           std::int64_t m1, std::int64_t m2) {
            if (x.size() != y.size() || m1 < 2 || m2 < 2) throw std::invalid_argument("x, y: the same length; gridsize >= 2");
            F64 fhat(std::vector<py::ssize_t>{static_cast<py::ssize_t>(m1), static_cast<py::ssize_t>(m2)});
            int warn;
            {
                py::gil_scoped_release release;
                // the result is m1 x m2 with the first index fastest: written into a Fortran-order copy
                std::vector<double> buf(static_cast<std::size_t>(m1 * m2));
                warn = kern_bkde2d(x.data(), y.data(), static_cast<std::int64_t>(x.size()), h1, h2, a1, b1, a2, b2,
                                   m1, m2, buf.data());
                double* o = fhat.mutable_data();
                for (std::int64_t j = 0; j < m2; ++j)
                    for (std::int64_t i = 0; i < m1; ++i) o[i * m2 + j] = buf[static_cast<std::size_t>(i + j * m1)];
            }
            return py::make_tuple(fhat, warn);
        },
        py::arg("x"), py::arg("y"), py::arg("h1"), py::arg("h2"), py::arg("a1"), py::arg("b1"), py::arg("a2"),
        py::arg("b2"), py::arg("m1"), py::arg("m2"),
        "KernSmooth::bkde2D's fhat (m1 x m2) and whether R warns that the binning grid is too coarse.");

    m.def(
        "move_hr_dpik",
        [](const F64& x, double a, double b, int scalest, int level, int kernel, bool canonical,
           std::int64_t gridsize, bool truncate) {
            int err = 0;
            const double h = kern_dpik(x.data(), static_cast<std::int64_t>(x.size()), scalest, level, kernel, canonical,
                                       gridsize, a, b, truncate, &err);
            if (err == 1) throw std::invalid_argument("Level should be between 0 and 5");
            if (err == 2) throw std::invalid_argument("scale estimate is zero for input data");
            return h;
        },
        py::arg("x"), py::arg("a"), py::arg("b"), py::arg("scalest") = 0, py::arg("level") = 2, py::arg("kernel") = 0,
        py::arg("canonical") = false, py::arg("gridsize") = 401, py::arg("truncate") = true,
        "KernSmooth::dpik (direct plug-in bandwidth).");

    m.def(
        "move_hr_lscv",
        [](const F64& x, const F64& y, const F64& hs, int threads) {
            if (x.size() != y.size()) throw std::invalid_argument("x and y: the same length");
            F64 out(hs.size());
            {
                py::gil_scoped_release release;
                amt_lscv(x.data(), y.data(), static_cast<std::int64_t>(x.size()), hs.data(),
                         static_cast<std::int64_t>(hs.size()), threads, out.mutable_data());
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("hs"), py::arg("threads") = 1, "amt's lscv: the criterion of each h.");

    m.def(
        "move_hr_ring_area",
        [](const F64& xy) {
            if (xy.ndim() != 2 || xy.shape(1) != 2) throw std::invalid_argument("xy: an (n, 2) array");
            return ogr_ring_area(xy.data(), static_cast<std::int64_t>(xy.shape(0)));
        },
        py::arg("xy"), "The area of a closed ring as GDAL's OGRSimpleCurve::get_LinearArea (sf::st_area).");

    m.def(
        "move_hr_cud",
        [](const F64& v) {
            F64 out(v.size());
            amt_cud(v.data(), static_cast<std::int64_t>(v.size()), out.mutable_data());
            return out;
        },
        py::arg("v"), "amt's hr_cud of the cells v.");

    m.def(
        "move_hr_vol_base",
        [](const F64& ui, const F64& uj, int type) {
            if (ui.size() != uj.size()) throw std::invalid_argument("ud_i and ud_j: the same length");
            return amt_vol_base(ui.data(), uj.data(), static_cast<std::int64_t>(ui.size()), type);
        },
        py::arg("ud_i"), py::arg("ud_j"), py::arg("type"), "amt's vol_base: 0 phr, 1 vi, 2 ba, 3 udoi, 4 hd.");

    m.def(
        "move_hr_locoh_neighbours",
        [](const F64& x, const F64& y, int type, double d, int threads) {
            if (x.size() != y.size()) throw std::invalid_argument("x and y: the same length");
            std::vector<std::int64_t> off, ids;
            {
                py::gil_scoped_release release;
                amt_locoh_neighbours(x.data(), y.data(), static_cast<std::int64_t>(x.size()), type, d, threads, off,
                                     ids);
            }
            py::array_t<std::int64_t> o(static_cast<py::ssize_t>(off.size())), s(static_cast<py::ssize_t>(ids.size()));
            for (std::size_t i = 0; i < off.size(); ++i) o.mutable_data()[i] = off[i];
            for (std::size_t i = 0; i < ids.size(); ++i) s.mutable_data()[i] = ids[i];
            return py::make_tuple(o, s);
        },
        py::arg("x"), py::arg("y"), py::arg("type"), py::arg("d"), py::arg("threads") = 1,
        "hr_locoh's neighbours (offsets, 0-based ids). type 0 k, 1 r, 2 a.");

    m.def(
        "move_hr_contour_lines",
        [](const F64& v, double xmin, double ymax, double dx, double dy, double level) -> py::object {
            if (v.ndim() != 2) throw std::invalid_argument("v: a 2-D array (rows from the top)");
            std::vector<std::vector<double>> lines;
            const bool ok = r_contour_lines(v.data(), v.shape(0), v.shape(1), xmin, ymax, dx, dy, level, lines);
            if (!ok) return py::none();
            py::list out;
            for (const auto& l : lines) {
                py::array_t<double> a({static_cast<py::ssize_t>(l.size() / 2), static_cast<py::ssize_t>(2)});
                std::copy(l.begin(), l.end(), a.mutable_data());
                out.append(a);
            }
            return out;
        },
        py::arg("v"), py::arg("xmin"), py::arg("ymax"), py::arg("dx"), py::arg("dy"), py::arg("level"),
        "terra::as.contour at one level (R's contourLines): the lines as (n, 2) arrays; None when all values are equal.");
}

}  // namespace move
