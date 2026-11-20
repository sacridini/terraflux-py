// SPDX-License-Identifier: GPL-3.0-or-later
//
// Track cleaning and thinning (ls.move), as move2utils' v_phys_estimate, mt_flag_speed_cap (threshold_type =
// "hard") and mt_thin_time / mt_thin_distance compute them. The locations of a track are consecutive (`track`
// holds a code per location); the tracks run in parallel, each on its own, so the result does not depend on
// the number of threads.
#pragma once

#include <cstdint>

namespace move {

using i64 = std::int64_t;

// v_phys_estimate (Hirt et al. 2017, Supplementary Table 4), for each mass in kg: the prediction in m/s, the
// delta-method interval at `ci_level` in m/s and the prediction in km/h. mode: 0 flying, 1 running, 2 swimming.
void v_phys(const double* mass, i64 n, int mode, double ci_level, double* v_ms, double* lo_ms, double* hi_ms,
            double* kmh);

// .speed_cap_fn_core with a hard cap, for each track: the fixes with a finite time (`t`, seconds; the rest are
// left out of the track) in their order give the steps (haversine on a sphere of 6371008.8 m in lon / lat,
// Euclidean otherwise), the speeds (NaN where the step or the time lag is not finite or the lag is not
// positive), the fixes next to a step above `v_max` (above: 1, 0; -1 for a fix left out) and, when `jitter`
// is not NaN, those next to a step shorter than it (below, as above). `loglr` is .attach_speedcap_loglr's
// (one-sided in the hard mode). The arrays have one value per location.
void speed_cap_hard(const double* x, const double* y, const double* t, const i64* track, i64 n, bool lonlat,
                    double v_max, double jitter, int threads, double* speed, signed char* above, signed char* below,
                    double* loglr);

// .thin_dp_track on each track of locations sorted by time (NaN times last): `selected` marks the fixes kept.
// closest: criterion "closest" (else "first").
void thin_time(const double* t, const i64* track, i64 n, double interval, double tolerance, bool closest, int threads,
               unsigned char* selected);

// mt_thin_distance: `step` is the distance from each location to the next of its track (NaN where unknown,
// counted as 0), in the rows' order. method 0: the greedy walk "step"; 1: "interval" (the DP over the
// cumulative distance).
void thin_distance(const double* step, const i64* track, i64 n, int method, double distance, double tolerance,
                   bool closest, int threads, unsigned char* selected);

}  // namespace move
