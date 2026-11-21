// SPDX-License-Identifier: GPL-3.0-or-later
//
// acos() as R's C code gets it on Windows. Unlike sin, cos, exp and log (the x87 functions of mingw-w64, lsm/rmath.hpp)
// it is the C library's: on the 10108 arguments of the references, R's acos equals the UCRT's bit for bit, and the
// x87 formulas of mingw-w64's math library (atan2 of sqrt((1 - x)(1 + x)) and x, in extended precision) differ in the
// last bit in one argument of twenty. The same function as std::acos, then, named for what the kernels mean.
#pragma once

#include <cmath>

namespace move {

inline double r_acos(double x) { return std::acos(x); }

}  // namespace move
