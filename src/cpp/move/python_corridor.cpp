// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the corridors and the distance between utilisation distributions
// (terraflux._core.move_corridor_*, move_emd_*, move_ud_prepare).

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

#include "move/corridor.hpp"
#include "move/emd.hpp"
#include "move/python.hpp"

namespace py = pybind11;

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I64 = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using I32 = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

template <class A>
py::array_t<typename A::value_type> copy_out(const std::vector<typename A::value_type>& v) {
    py::array_t<typename A::value_type> out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

void same_size(py::ssize_t n, std::initializer_list<py::ssize_t> sizes, const char* what) {
    for (auto s : sizes)
        if (s != n) throw std::invalid_argument(what);
}

}  // namespace

namespace move {

void register_python_corridor(py::module_& m) {
    m.def(
        "move_corridor_neighbourhoods",
        [](const F64& x0, const F64& y0, const F64& x1, const F64& y1, const F64& radius, int threads) {
            const py::ssize_t n = x0.size();
            same_size(n, {y0.size(), x1.size(), y1.size(), radius.size()}, "x0, y0, x1, y1, radius: 1-D arrays of the same length");
            Neighbourhoods r;
            {
                py::gil_scoped_release release;
                r = corridor_neighbourhoods(x0.data(), y0.data(), x1.data(), y1.data(), radius.data(), n, threads);
            }
            py::dict out;
            out["azimuth"] = copy_out<F64>(r.azimuth);
            out["circvar"] = copy_out<F64>(r.circvar);
            out["offset"] = copy_out<I64>(r.offset);
            out["index"] = copy_out<I32>(r.index);
            return out;
        },
        py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"), py::arg("radius"), py::arg("threads") = 1,
        "The neighbours of segments as move2utils' mt_corridor finds them (GEOS' 120-gon buffer of each midpoint, "
        "st_intersects), their azimuths and circular variances.");

    m.def(
        "move_corridor_flags",
        [](const I64& offset, const I32& index, const U8& candidate, double min_segments, int threads) {
            const py::ssize_t n = candidate.size();
            if (offset.size() != n + 1) throw std::invalid_argument("offset: one more than candidate");
            if (offset.size() < 1 || offset.data()[n] != index.size()) throw std::invalid_argument("offset and index do not fit");
            U8 out(n);
            {
                py::gil_scoped_release release;
                corridor_flags(offset.data(), index.data(), candidate.data(), n, min_segments, threads, out.mutable_data());
            }
            return out;
        },
        py::arg("offset"), py::arg("index"), py::arg("candidate"), py::arg("min_segments"), py::arg("threads") = 1,
        "is_corridor: enough neighbours and a strict majority of candidates among them.");

    m.def(
        "move_buffer_vertices",
        [](double cx, double cy, double r) {
            F64 vx(kBufferVertices), vy(kBufferVertices);
            buffer_vertices(cx, cy, r, vx.mutable_data(), vy.mutable_data());
            return py::make_tuple(vx, vy);
        },
        py::arg("cx"), py::arg("cy"), py::arg("r"), "The 120 vertices of GEOS' buffer of a point (sf's st_buffer).");

    m.def(
        "move_ring_intersects",
        [](const F64& vx, const F64& vy, double px, double py) {
            if (vx.size() != vy.size() || vx.size() < 3) throw std::invalid_argument("vx, vy: the vertices of a ring");
            return ring_intersects(vx.data(), vy.data(), static_cast<int>(vx.size()), px, py);
        },
        py::arg("vx"), py::arg("vy"), py::arg("px"), py::arg("py"),
        "GEOS' point-in-ring test (boundary included) with its robust orientation.");

    m.def(
        "move_ud_prepare",
        [](const F64& values, double mask_quantile) {
            std::vector<std::int64_t> cells;
            std::vector<double> w;
            {
                py::gil_scoped_release release;
                emd_prepare(values.data(), static_cast<std::int64_t>(values.size()), mask_quantile, cells, w);
            }
            return py::make_tuple(copy_out<I64>(cells), copy_out<F64>(w));
        },
        py::arg("values"), py::arg("mask_quantile"),
        "move2utils' .emd_prep_ud: the cells inside the volume contour and their weights.");

    m.def(
        "move_emd_sinkhorn",
        [](const F64& ax, const F64& ay, const F64& p, const F64& bx, const F64& by, const F64& q, bool gc,
           double threshold, double reg, int max_iter, double tol, int threads) {
            same_size(ax.size(), {ay.size(), p.size()}, "ax, ay, p: the same length");
            same_size(bx.size(), {by.size(), q.size()}, "bx, by, q: the same length");
            double d = 0;
            int iterations = 0;
            {
                py::gil_scoped_release release;
                d = emd_sinkhorn(ax.data(), ay.data(), p.data(), ax.size(), bx.data(), by.data(), q.data(), bx.size(),
                                 gc, threshold, reg, max_iter, tol, threads, &iterations);
            }
            return py::make_tuple(d, iterations);
        },
        py::arg("ax"), py::arg("ay"), py::arg("p"), py::arg("bx"), py::arg("by"), py::arg("q"), py::arg("gc") = false,
        py::arg("threshold") = std::numeric_limits<double>::quiet_NaN(), py::arg("reg") = 0.05,
        py::arg("max_iter") = 500, py::arg("tol") = 1e-9, py::arg("threads") = 1,
        "move2utils' .emd_sinkhorn between two sets of weighted cells: (distance, iterations).");

    m.def(
        "move_emd_exact",
        [](const F64& ax, const F64& ay, const F64& p, const F64& bx, const F64& by, const F64& q, int max_iter) {
            same_size(ax.size(), {ay.size(), p.size()}, "ax, ay, p: the same length");
            same_size(bx.size(), {by.size(), q.size()}, "bx, by, q: the same length");
            double d = 0;
            bool hit = false;
            {
                py::gil_scoped_release release;
                d = emd_exact(ax.data(), ay.data(), p.data(), ax.size(), bx.data(), by.data(), q.data(), bx.size(),
                              max_iter, &hit);
            }
            return py::make_tuple(d, hit);
        },
        py::arg("ax"), py::arg("ay"), py::arg("p"), py::arg("bx"), py::arg("by"), py::arg("q"),
        py::arg("max_iter") = 500, "emdist::emd (Rubner's exact solver, in float): (distance, stopped at max_iter).");
}

}  // namespace move
