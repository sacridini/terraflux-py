// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the length of a step on the relief (terraflux._core.move_surface_distance).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

#include "move/relief.hpp"

namespace py = pybind11;

namespace move {

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;

}  // namespace

void register_python_relief(py::module_& m) {
    m.def(
        "move_surface_distance",
        [](const F64& x1, const F64& y1, const F64& x2, const F64& y2, const F64& z, double xmin, double ymax,
           double dx, double dy, double spacing, int threads) {
            const auto n = static_cast<std::int64_t>(x1.size());
            if (x1.ndim() != 1 || y1.size() != x1.size() || x2.size() != x1.size() || y2.size() != x1.size())
                throw std::invalid_argument("x1, y1, x2, y2: 1-D arrays of the same length");
            if (z.ndim() != 2) throw std::invalid_argument("z: a 2-D grid (rows from the top)");
            if (!(dx > 0.0 && dy > 0.0)) throw std::invalid_argument("dx, dy: positive cell sizes");
            F64 out(n);
            {
                py::gil_scoped_release release;
                surface_distance(x1.data(), y1.data(), x2.data(), y2.data(), n, z.data(), z.shape(1), z.shape(0),
                                 xmin, ymax, dx, dy, spacing, threads, out.mutable_data());
            }
            return out;
        },
        py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("z"), py::arg("xmin"), py::arg("ymax"),
        py::arg("dx"), py::arg("dy"), py::arg("spacing") = 0.0, py::arg("threads") = 1,
        "The 3D length of each step over a DEM (bilinear heights along the segment every spacing).");
}

}  // namespace move
