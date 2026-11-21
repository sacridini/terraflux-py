// SPDX-License-Identifier: GPL-3.0-or-later
//
// Python bindings of the dynamic Brownian bridge (terraflux._core.move_dbbmm_*, move_ud_*, move_impute_*).

#include "move/python_ud.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "lsm/rmath.hpp"
#include "move/acos.hpp"
#include "move/bridge.hpp"
#include "move/locerr.hpp"
#include "move/ud.hpp"

namespace py = pybind11;

namespace {

using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using U8 = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

void same_length(std::int64_t n, std::initializer_list<std::int64_t> sizes, const char* what) {
    for (const auto s : sizes)
        if (s != n) throw std::invalid_argument(what);
}

// The inputs of a kernel, checked to have the same length.
move::UdInput ud_input(const F64& x, const F64& y, const F64& s1, const F64& s2, const F64& t, const F64& le,
                       const U8& interest, double time_step, double ext) {
    const auto n = static_cast<std::int64_t>(x.size());
    same_length(n, {static_cast<std::int64_t>(y.size()), static_cast<std::int64_t>(s1.size()),
                    static_cast<std::int64_t>(s2.size()), static_cast<std::int64_t>(t.size()),
                    static_cast<std::int64_t>(le.size()), static_cast<std::int64_t>(interest.size())},
                "x, y, the variances, t, the location errors and interest: 1-D arrays of the same length");
    return {x.data(), y.data(), s1.data(), s2.data(), t.data(), le.data(), interest.data(), n, time_step, ext};
}

}  // namespace

namespace move {

void register_python_ud(py::module_& m) {
    m.def(
        "move_dbbmm_variance",
        [](const F64& x, const F64& y, const F64& time_lag, const F64& loc_err, int window_size, int margin,
           int threads) {
            const auto n = static_cast<std::int64_t>(x.size());
            same_length(n, {static_cast<std::int64_t>(y.size()), static_cast<std::int64_t>(time_lag.size()),
                            static_cast<std::int64_t>(loc_err.size())},
                        "x, y, time_lag and loc_err: 1-D arrays of the same length");
            DbbmmVariance r;
            {
                py::gil_scoped_release release;
                r = dbbmm_variance(x.data(), y.data(), time_lag.data(), loc_err.data(), n, window_size, margin, threads);
            }
            py::dict out;
            py::array_t<double> var(static_cast<py::ssize_t>(r.variance.size()), r.variance.data());
            out["variance"] = var;
            py::array_t<std::int32_t> ne(static_cast<py::ssize_t>(r.n_estim.size()), r.n_estim.data());
            py::array_t<std::int32_t> bp(static_cast<py::ssize_t>(r.break_pos.size()), r.break_pos.data());
            out["n_estim"] = ne;
            out["break_pos"] = bp;
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("time_lag"), py::arg("loc_err"), py::arg("window_size"), py::arg("margin"),
        py::arg("threads") = 1,
        "move2utils' bm_variance_track_c: the motion variance of the dynamic Brownian bridge, per location "
        "(NaN where no window covers it), the windows that cover each, and the break of each window with one.");

    m.def(
        "move_impute_location_error",
        [](F64 err, int how) {
            if (err.ndim() != 1) throw std::invalid_argument("err: a 1-D array");
            F64 out(err.size());
            std::copy(err.data(), err.data() + err.size(), out.mutable_data());
            const int status = impute_loc_err(out.mutable_data(), static_cast<std::int64_t>(out.size()),
                                              static_cast<ImputeHow>(how));
            return py::make_tuple(out, status);
        },
        py::arg("err"), py::arg("how"),
        "The NaN of a location error filled (how: 0 median, 1 mean, 2 zero, 3 approx) and a status: 0 filled or "
        "nothing missing, 1 every value missing, 2 approx with fewer than two valid values.");

    m.def(
        "move_ud_probe",
        [](const F64& x, const F64& y, const F64& s1, const F64& t, const F64& le, const U8& interest, const F64& xgrid,
           const F64& ygrid, double time_step, double ext) {
            const UdInput in = ud_input(x, y, s1, s1, t, le, interest, time_step, ext);
            UdProbe p;
            {
                py::gil_scoped_release release;
                p = ud_probe(in, xgrid.data(), static_cast<std::int64_t>(xgrid.size()), ygrid.data(),
                             static_cast<std::int64_t>(ygrid.size()));
            }
            F64 out(8);
            double* o = out.mutable_data();
            for (int i = 0; i < 4; ++i) {
                o[i] = p.env[i];
                o[4 + i] = p.idx[i];
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("s1"), py::arg("t"), py::arg("le"), py::arg("interest"), py::arg("xgrid"),
        py::arg("ygrid"), py::arg("time_step"), py::arg("ext"),
        "move2utils' ud_grid_probe for the dBBMM: the envelope of the kernel windows (4) and the extreme cell "
        "indices they reach on the grid (4: min xStart, max xEnd, max yEnd, min yStart).");

    m.def(
        "move_dbbmm_ud",
        [](const F64& x, const F64& y, const F64& s1, const F64& t, const F64& le, const U8& interest, const F64& xgrid,
           const F64& ygrid, double time_step, double ext, int threads) {
            const UdInput in = ud_input(x, y, s1, s1, t, le, interest, time_step, ext);
            const auto nx = static_cast<std::int64_t>(xgrid.size());
            const auto ny = static_cast<std::int64_t>(ygrid.size());
            F64 rows({static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)});
            {
                py::gil_scoped_release release;
                std::vector<double> ud(static_cast<std::size_t>(nx * ny));
                dbbmm_ud(in, xgrid.data(), nx, ygrid.data(), ny, threads, ud.data());
                ud_normalise(ud.data(), nx * ny);
                ud_to_rows(ud.data(), nx, ny, rows.mutable_data());
            }
            return rows;
        },
        py::arg("x"), py::arg("y"), py::arg("s1"), py::arg("t"), py::arg("le"), py::arg("interest"), py::arg("xgrid"),
        py::arg("ygrid"), py::arg("time_step"), py::arg("ext"), py::arg("threads") = 1,
        "move2utils' dbbmm2_omp and the normalisation of mt_dbbmm_ud: the utilization distribution on a grid of "
        "cell centres (xgrid ascending, ygrid ascending), as (ny, nx) rows from the top, summing to 1.");

    m.def(
        "move_dbgb_variance",
        [](const F64& x, const F64& y, const F64& time_mins, const F64& loc_err, int window_size, int margin,
           int threads) {
            const auto n = static_cast<std::int64_t>(x.size());
            same_length(n, {static_cast<std::int64_t>(y.size()), static_cast<std::int64_t>(time_mins.size()),
                            static_cast<std::int64_t>(loc_err.size())},
                        "x, y, time_mins and loc_err: 1-D arrays of the same length");
            DbgbVariance r;
            {
                py::gil_scoped_release release;
                r = dbgb_variance(x.data(), y.data(), time_mins.data(), loc_err.data(), n, window_size, margin, threads);
            }
            py::dict out;
            out["para_sd"] = py::array_t<double>(static_cast<py::ssize_t>(r.para_sd.size()), r.para_sd.data());
            out["orth_sd"] = py::array_t<double>(static_cast<py::ssize_t>(r.orth_sd.size()), r.orth_sd.data());
            out["n_estim"] = py::array_t<std::int32_t>(static_cast<py::ssize_t>(r.n_estim.size()), r.n_estim.data());
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("time_mins"), py::arg("loc_err"), py::arg("window_size"),
        py::arg("margin"), py::arg("threads") = 1,
        "move2utils' bgb_var_break_track_c: the standard deviations of the directional bridge along and across the "
        "direction to the next location, per location (NaN where no window covers it), and the windows that cover each.");

    m.def(
        "move_dbgb_probe",
        [](const F64& x, const F64& y, const F64& s1, const F64& s2, const F64& t, const F64& le, const U8& interest,
           const F64& xgrid, const F64& ygrid, double time_step, double ext) {
            UdInput in = ud_input(x, y, s1, s2, t, le, interest, time_step, ext);
            in.model = UdModel::Dbgb;
            UdProbe p;
            {
                py::gil_scoped_release release;
                p = ud_probe(in, xgrid.data(), static_cast<std::int64_t>(xgrid.size()), ygrid.data(),
                             static_cast<std::int64_t>(ygrid.size()));
            }
            F64 out(8);
            double* o = out.mutable_data();
            for (int i = 0; i < 4; ++i) {
                o[i] = p.env[i];
                o[4 + i] = p.idx[i];
            }
            return out;
        },
        py::arg("x"), py::arg("y"), py::arg("s1"), py::arg("s2"), py::arg("t"), py::arg("le"), py::arg("interest"),
        py::arg("xgrid"), py::arg("ygrid"), py::arg("time_step"), py::arg("ext"),
        "move2utils' ud_grid_probe for the dBGB (see move_ud_probe).");

    m.def(
        "move_dbgb_ud",
        [](const F64& x, const F64& y, const F64& s1, const F64& s2, const F64& t, const F64& le, const U8& interest,
           const F64& xgrid, const F64& ygrid, double time_step, double ext, int threads) {
            const UdInput in = ud_input(x, y, s1, s2, t, le, interest, time_step, ext);
            const auto nx = static_cast<std::int64_t>(xgrid.size());
            const auto ny = static_cast<std::int64_t>(ygrid.size());
            F64 rows({static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)});
            {
                py::gil_scoped_release release;
                std::vector<double> ud(static_cast<std::size_t>(nx * ny));
                dbgb_ud(in, xgrid.data(), nx, ygrid.data(), ny, threads, ud.data());
                ud_normalise(ud.data(), nx * ny);
                ud_to_rows(ud.data(), nx, ny, rows.mutable_data());
            }
            return rows;
        },
        py::arg("x"), py::arg("y"), py::arg("s1"), py::arg("s2"), py::arg("t"), py::arg("le"), py::arg("interest"),
        py::arg("xgrid"), py::arg("ygrid"), py::arg("time_step"), py::arg("ext"), py::arg("threads") = 1,
        "move2utils' bgb_omp and the normalisation of mt_dbgb_ud: the utilization distribution of the directional "
        "bridge, as (ny, nx) rows from the top, summing to 1.");

    m.def(
        "move_r_acos",
        [](const F64& v) {
            const auto n = static_cast<std::int64_t>(v.size());
            F64 out(n);
            for (std::int64_t i = 0; i < n; ++i) out.mutable_data()[i] = r_acos(v.data()[i]);
            return out;
        },
        py::arg("v"), "acos as mingw-w64 computes it for R (x87), for each value.");

    m.def(
        "move_r_exp",
        [](const F64& v) {
            const auto n = static_cast<std::int64_t>(v.size());
            F64 out(n);
            for (std::int64_t i = 0; i < n; ++i) out.mutable_data()[i] = lsm::r_exp(v.data()[i]);
            return out;
        },
        py::arg("v"), "exp as R computes it (the x87 of mingw-w64), for each value.");

    m.def(
        "move_ud_volume",
        [](const F64& v) {
            const auto n = static_cast<std::int64_t>(v.size());
            F64 out(n);
            {
                py::gil_scoped_release release;
                ud_volume(v.data(), n, out.mutable_data());
            }
            return out;
        },
        py::arg("v"), "move2utils' ud_volume of one layer (values in raster order): the cumulative probability.");
}

}  // namespace move
