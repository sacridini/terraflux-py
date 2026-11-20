// SPDX-License-Identifier: GPL-3.0-or-later
//
// The Python functions of the dynamic Brownian bridge (move_dbbmm_*, move_ud_*), registered into
// terraflux._core.
#pragma once

#include <pybind11/pybind11.h>

namespace move {

void register_python_ud(pybind11::module_& m);

}  // namespace move
