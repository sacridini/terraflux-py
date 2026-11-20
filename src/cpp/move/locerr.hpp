// SPDX-License-Identifier: GPL-3.0-or-later
//
// The missing values of a track's location error, filled as move2utils' .impute_loc_err_na does it.
#pragma once

#include <cstdint>

namespace move {

enum class ImputeHow { Median = 0, Mean = 1, Zero = 2, Approx = 3 };

// Fills the NaN values of e[0, n) in place. Median and mean are R's (a long double mean of the valid values; the
// median of an even count is the mean of the two middle ones); Approx is approx(rule = 2) along the position:
// linear between the valid neighbours, the nearest valid value outside them.
// Returns 0, 1 when every value is missing (and `how` is not Zero) and 2 when Approx has fewer than two valid
// values (R's approx() stops there); e is left as it was in those two cases.
int impute_loc_err(double* e, std::int64_t n, ImputeHow how);

}  // namespace move
