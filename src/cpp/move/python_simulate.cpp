// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the simulation from a step selection model (terraflux._core.move_sim_*): amt's
// redistribution_kernel and simulate_path in continuous space. The generator is R's, carried between calls as 625
// words (its Mersenne-Twister state and mti).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "move/simulate.hpp"
#include "nlm/rng.hpp"

namespace py = pybind11;

namespace move {

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using U32 = py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>;

template <class T>
std::vector<T> vec_of(const py::dict& d, const char* key) {
    return py::cast<std::vector<T>>(d[key]);
}

// the spec of a simulation; `keep` holds the arrays the layers point into
SimSpec read_spec(const py::dict& d, std::vector<F64>& keep) {
    SimSpec s;
    s.sl_name = py::cast<std::string>(d["sl_name"]);
    s.ta_name = py::cast<std::string>(d["ta_name"]);
    s.sl_par = vec_of<double>(d, "sl_par");
    s.ta_par = vec_of<double>(d, "ta_par");
    const auto bbox = vec_of<double>(d, "bbox");
    if (bbox.size() != 4) throw std::invalid_argument("bbox: xmin, xmax, ymin, ymax");
    s.xmin = bbox[0];
    s.xmax = bbox[1];
    s.ymin = bbox[2];
    s.ymax = bbox[3];
    s.n_control = py::cast<std::int64_t>(d["n_control"]);
    s.tolerance_outside = py::cast<double>(d["tolerance_outside"]);
    const py::list layers = d["layers"];     // 2-D arrays (y, x), NaN for no data
    const F64 grids = py::cast<F64>(d["grids"]);   // one row per layer: xmin, xmax, ymin, ymax, xres, yres
    if (grids.ndim() != 2 || grids.shape(1) != 6 || static_cast<std::size_t>(grids.shape(0)) != layers.size())
        throw std::invalid_argument("grids: (layers, 6)");
    keep.reserve(layers.size());
    for (std::size_t l = 0; l < layers.size(); ++l) {
        keep.push_back(py::cast<F64>(layers[l]));
        const F64& a = keep.back();
        if (a.ndim() != 2) throw std::invalid_argument("layers: 2-D arrays");
        const double* g = grids.data() + 6 * l;
        SimLayer lay;
        lay.v = a.data();
        lay.ny = a.shape(0);
        lay.nx = a.shape(1);
        lay.xmin = g[0];
        lay.xmax = g[1];
        lay.ymin = g[2];
        lay.ymax = g[3];
        lay.xres = g[4];
        lay.yres = g[5];
        s.layers.push_back(lay);
    }
    const py::dict f = d["formula"];
    s.f.var_ptr = vec_of<int>(f, "var_ptr");
    s.f.op = vec_of<int>(f, "op");
    s.f.arg = vec_of<double>(f, "arg");
    s.f.var_factor = vec_of<int>(f, "var_factor");
    s.f.level_ptr = vec_of<int>(f, "level_ptr");
    s.f.level = vec_of<double>(f, "level");
    s.f.col_ptr = vec_of<int>(f, "col_ptr");
    s.f.col_var = vec_of<int>(f, "col_var");
    s.f.col_level = vec_of<int>(f, "col_level");
    s.f.coef = vec_of<double>(f, "coef");
    const std::size_t nv = s.f.var_ptr.size() ? s.f.var_ptr.size() - 1 : 0;
    if (s.f.op.size() != s.f.arg.size() || s.f.level_ptr.size() != nv + 1 || s.f.var_factor.size() != nv ||
        s.f.col_ptr.size() != s.f.coef.size() + 1 || s.f.col_var.size() != s.f.col_level.size())
        throw std::invalid_argument("formula: arrays of inconsistent sizes");
    for (int v : s.f.col_var)
        if (v < 0 || static_cast<std::size_t>(v) >= nv) throw std::invalid_argument("formula: a column uses an unknown variable");
    for (std::size_t k = 0; k < s.f.op.size(); ++k) {
        if ((s.f.op[k] == kOpEnd || s.f.op[k] == kOpStart) &&
            (!(s.f.arg[k] >= 0) || static_cast<std::size_t>(s.f.arg[k]) >= s.layers.size()))
            throw std::invalid_argument("formula: a covariate that is not among the layers");
    }
    return s;
}

nlm::RRng rng_from(const U32& state) {
    if (state.ndim() != 1 || state.size() != nlm::RRng::kStateWords)
        throw std::invalid_argument("state: 625 words");
    nlm::RRng rng;
    rng.set_state(state.data());
    return rng;
}

U32 state_of(const nlm::RRng& rng) {
    U32 out(nlm::RRng::kStateWords);
    rng.get_state(out.mutable_data());
    return out;
}

}  // namespace

void register_python_simulate(py::module_& m) {
    m.def(
        "move_sim_seed",
        [](std::int32_t seed) {
            nlm::RRng rng(seed);
            return state_of(rng);
        },
        py::arg("seed"), "R's generator after set.seed(seed): its state (625 words).");

    m.def(
        "move_sim_kernel",
        [](const U32& state, double x0, double y0, double ta0, std::int64_t n_sample, const py::dict& spec) {
            std::vector<F64> keep;
            const SimSpec s = read_spec(spec, keep);
            if (n_sample < 1) throw std::invalid_argument("n_sample: 1 or more");
            nlm::RRng rng = rng_from(state);
            std::vector<double> ox(static_cast<std::size_t>(n_sample)), oy(static_cast<std::size_t>(n_sample));
            int status;
            {
                py::gil_scoped_release release;
                status = sim_cycle(rng, s, x0, y0, ta0, n_sample, ox.data(), oy.data());
            }
            py::dict out;
            out["status"] = status;
            out["x"] = py::array_t<double>(static_cast<py::ssize_t>(status == kSimOk ? ox.size() : 0), ox.data());
            out["y"] = py::array_t<double>(static_cast<py::ssize_t>(status == kSimOk ? oy.size() : 0), oy.data());
            out["state"] = state_of(rng);
            return out;
        },
        py::arg("state"), py::arg("x0"), py::arg("y0"), py::arg("ta0"), py::arg("n_sample"), py::arg("spec"),
        "One cycle of amt's redistribution_kernel (continuous space): the sampled end points of the candidate steps.");

    m.def(
        "move_sim_path",
        [](const U32& state, double x0, double y0, double ta0, std::int64_t n_steps, const py::dict& spec) {
            std::vector<F64> keep;
            const SimSpec s = read_spec(spec, keep);
            if (n_steps < 0) throw std::invalid_argument("n_steps: 0 or more");
            nlm::RRng rng = rng_from(state);
            py::array_t<double> xs(static_cast<py::ssize_t>(n_steps + 1)), ys(static_cast<py::ssize_t>(n_steps + 1));
            std::int64_t done = 0;
            int status;
            {
                py::gil_scoped_release release;
                status = sim_path(rng, s, x0, y0, ta0, n_steps, xs.mutable_data(), ys.mutable_data(), &done);
            }
            py::dict out;
            out["status"] = status;
            out["x"] = xs;
            out["y"] = ys;
            out["done"] = done;
            out["state"] = state_of(rng);
            return out;
        },
        py::arg("state"), py::arg("x0"), py::arg("y0"), py::arg("ta0"), py::arg("n_steps"), py::arg("spec"),
        "amt's simulate_path (continuous space): n_steps cycles of one draw, each from the step before.");
}

}  // namespace move
