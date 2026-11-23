// SPDX-License-Identifier: GPL-3.0-or-later
//
// Simulation from a fitted integrated step selection function (amt 0.3.1's redistribution_kernel and simulate_path
// in continuous space, Signer et al. 2024), with R's numbers: the candidate steps drawn from the tentative step
// length and turn angle distributions (all the lengths, then all the angles), the covariates at their ends and
// starts (terra's cellFromXY), the model matrix of the formula, the weights exp(X b - mean) as ssf_weights makes
// them (R's %*%: dgemv, or long double sums when a value is NaN or infinite) and sample.int(n, size, prob = w).
//
// The formula comes as a compact description (no Python): programs of reverse Polish operations for each variable
// (covariates at the end or start of the step, sl_, ta_, log_sl_, cos_ta_, constants, + - * / ^, log, exp, sqrt, sin,
// cos, log1p, abs, unary minus), the factor levels, and the model matrix's columns as products of variables (or of a
// factor's level indicators), with the coefficients.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nlm {
class RRng;
}

namespace move {

using i64 = std::int64_t;

// One band of covariates: values (row-major from the top row, NaN for no data) on a grid.
struct SimLayer {
    const double* v = nullptr;
    i64 nx = 0, ny = 0;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0, xres = 1, yres = 1;
};

// terra::cellFromXY: the cell (row-major, 0-based) of a point, -1 outside; a point on the edge between two cells is in
// the cell to its right / below, on the right and bottom edges of the raster in the last column / row.
i64 sim_grid_cell(const SimLayer& g, double x, double y);

enum SimOp : int {
    kOpConst = 0,   // arg: the value
    kOpEnd = 1,     // arg: the layer; its value at the end of the candidate step
    kOpStart = 2,   // arg: the layer; its value at the start of the step
    kOpSl = 3,      // the length of the candidate step
    kOpTa = 4,      // its turn angle
    kOpLogSl = 5,   // log(sl_)
    kOpCosTa = 6,   // cos(ta_)
    kOpNeg = 10, kOpAbs = 11, kOpLog = 12, kOpExp = 13, kOpSqrt = 14, kOpSin = 15, kOpCos = 16, kOpLog1p = 17,
    kOpAdd = 20, kOpSub = 21, kOpMul = 22, kOpDiv = 23, kOpPow = 24,
};

struct SimFormula {
    std::vector<int> var_ptr;     // the program of variable v: ops var_ptr[v] .. var_ptr[v + 1]
    std::vector<int> op;
    std::vector<double> arg;
    std::vector<int> var_factor;  // 1 for a factor (its value is compared with the levels)
    std::vector<int> level_ptr;   // the levels of variable v: level[level_ptr[v] .. level_ptr[v + 1]]
    std::vector<double> level;
    std::vector<int> col_ptr;     // the factors of column c: col_var / col_level [col_ptr[c] .. col_ptr[c + 1]]
    std::vector<int> col_var;     // (multiplied in this order)
    std::vector<int> col_level;   // the level index of a factor, -1 for a number
    std::vector<double> coef;     // one per column
};

struct SimSpec {
    std::string sl_name, ta_name;
    std::vector<double> sl_par, ta_par;
    std::vector<SimLayer> layers;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;  // the study area
    SimFormula f;
    i64 n_control = 1000000;
    double tolerance_outside = 0.0;
};

enum SimStatus : int { kSimOk = 0, kSimOutside = 1, kSimNoWeight = 2 };

// One cycle of redistribution_kernel: n_control candidates from (x0, y0) with the previous direction ta0, their weights
// and n_sample of them drawn without replacement (written to ox / oy). kSimOutside: more than tolerance_outside of the
// candidates that could be drawn end outside the study area (the cycle stops before drawing); kSimNoWeight: no
// candidate has a positive weight (R's "too few positive probabilities").
int sim_cycle(nlm::RRng& rng, const SimSpec& s, double x0, double y0, double ta0, i64 n_sample, double* ox,
              double* oy);

// simulate_path: n_steps cycles of one draw each, each from the step before with its direction (atan2). xs and ys
// have n_steps + 1 values (the first is the start; NaN after a stop). `done`: the steps made.
int sim_path(nlm::RRng& rng, const SimSpec& s, double x0, double y0, double ta0, i64 n_steps, double* xs, double* ys,
             i64* done);

}  // namespace move
