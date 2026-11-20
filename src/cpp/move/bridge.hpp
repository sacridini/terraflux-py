// SPDX-License-Identifier: GPL-3.0-or-later
//
// The dynamic Brownian bridge movement model's motion variance (ls.move), as move2utils' bm_variance_track_c
// computes it: a moving window over the locations, a Brent search of the variance that maximises the likelihood
// of the leave-one-out pairs, and a BIC test of one break point per window. The windows run in parallel, each on
// its own; their results are added to the locations in the order of the windows, so the result does not depend
// on the number of threads (R's OpenMP loop adds with `atomic` in no fixed order).
#pragma once

#include <cstdint>
#include <vector>

namespace move {

using i64 = std::int64_t;

struct DbbmmVariance {
    std::vector<double> variance;          // per location, NaN where no window covers it
    std::vector<std::int32_t> n_estim;     // windows that cover each location
    std::vector<std::int32_t> break_pos;   // per window with a break: w + best_b (1-based position in the track)
};

// time_lag = c(diff(time_mins), 0): minutes between a location and the next (the last 0); loc_err in the units of
// x and y (a standard deviation); the variance comes out in units^2 per minute. window_size and margin must be
// odd, margin >= 1 and window_size >= 2 * margin + 1, window_size <= n (std::invalid_argument otherwise).
DbbmmVariance dbbmm_variance(const double* x, const double* y, const double* time_lag, const double* loc_err, i64 n,
                             int window_size, int margin, int threads);

}  // namespace move
