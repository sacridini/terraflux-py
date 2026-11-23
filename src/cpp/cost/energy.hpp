// SPDX-License-Identifier: GPL-3.0-or-later
//
// Energy landscapes of a DEM: the R package enerscape 1.2.0 (Berti et al. 2022), the cost of walking onto each cell
// from its neighbours (animals: Pontzer 2016; people: Looney et al. 2019), bit for bit with it.
#pragma once

#include <cstdint>

namespace cost {

// The mean, over the 4 or 8 neighbours, of the cost of walking from each neighbour onto the cell, for a DEM `x`
// (ny rows by nx columns, row 0 = north, NaN = NA). `res` is the (square) cell size in metres. The cells of the
// border, and those with a NaN in their neighbourhood, are NaN, as are costs of exactly 0 (R's `ans[ans == 0] <- NA`).
// `human` takes Looney et al.'s walking model at speed `speed` (m/s) in place of Pontzer's. `kcal` gives kcal, else J.
// The rows go to `threads` threads; the result does not depend on them. `neigh` must be 4 or 8.
void energy_landscape(const double* x, std::int64_t ny, std::int64_t nx, double mass, double res, int neigh, bool human,
                      double speed, bool kcal, int threads, double* out);

}  // namespace cost
