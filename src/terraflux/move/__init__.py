# SPDX-License-Identifier: GPL-3.0-or-later
"""Movement ecology: animal tracks, their steps and their relation to the landscape.

Tracks are GeoDataFrames of points, one row per location, ordered by track and
time (:func:`terraflux.load_tracks` reads Movebank's files, :func:`as_tracks`
any table of locations). The first part is R's move2 (Kranstauber et al.
2024), the same numbers: the distance, speed, direction and turn of each step
(``sf`` / ``s2`` and ``lwgeom`` on the sphere and the ellipsoid), duplicates and
one location per interval, interpolation in time, tracks as lines, a local
projection, Brownian tracks. Computed in C++ with OpenMP. From R's move2utils
(Kranstauber, Safi & Scharf 2026) comes the simple part of cleaning a track: the
allometric speed limit (:func:`v_phys_estimate`, Hirt et al. 2017), the locations
beside steps above it (:func:`flag_speed`), GPS quality (:func:`filter_gps_quality`)
and thinning in time and in distance (:func:`filter_thin_time`, :func:`filter_thin_distance`).
From amt, the values of a raster along each step (:func:`extract_covariates_along`, the cells terra's
``extract`` takes for the line) and, added here, their summaries per step (:func:`along_summary`: mean, range,
ascent, descent and delta, which tell a step up a hill from one along a contour line).
From move2utils again, the corridors the animals use (:func:`corridor`, LaPoint et al. 2013: fast, straight
passages among the neighbouring segments; :func:`corridor_lines`) and the earth mover's distance between
utilisation distributions (:func:`ud_distance`, Sinkhorn or exact). The dynamic Brownian bridge of move2utils
(:func:`dbbmm_variance`, :func:`dbbmm_ud`: the motion variance by moving windows and the utilisation distribution, with
:func:`ud_volume` for its contours), and amt's home ranges (:func:`hr_mcp`, :func:`hr_kde` with its bandwidths,
:func:`hr_locoh`, :func:`hr_isopleths`, :func:`hr_overlap`; :func:`home_range` for short). Last, :func:`validate`
(not in R): the use of the animals, as a UD, points or corridor segments, against a connectivity surface, after
Zeller et al. (2018).

Typical use::

    import terraflux as tf

    tracks = tf.load_tracks("study.csv")                    # Movebank's export
    tracks = tf.move.filter_unique(tracks)                  # one record per time and track
    tracks["speed"] = tf.move.speed(tracks)                 # m/s, NaN at the last location
    tracks["turn"] = tf.move.turn_angle(tracks, "deg")
    hourly = tf.move.filter_per_interval(tracks, "hour")
    lines = tf.move.track_lines(tracks)                     # one line per animal
    tracks = tf.move.filter_gps_quality(tracks)             # satellites, DOP, accuracy
    flagged = tf.move.flag_speed(tracks, mass=5, mode="flying")   # beside a step above Hirt's limit
    thin = tf.move.filter_thin_time(tracks, "45 min")       # about 45 minutes between locations
    tf.write_vector(lines, "tracks.gpkg")
"""

from ._along import along_summary, extract_covariates_along
from ._clean import (VPhys, filter_gps_quality, filter_thin_distance, filter_thin_time, flag_speed, gps_quality,
                     thin_distance, thin_time, v_phys_estimate)
from ._corridor import corridor, corridor_lines
from ._emd import ud_distance
from ._examples import example_data
from ._filters import (filter_movebank_visible, filter_per_interval, filter_unique, movebank_visible, per_interval,
                       unique)
from ._geometry import aeqd_crs, interpolate, segments, track_data, track_lines
from ._issf import ISSF, fit_clogit, fit_issf, fit_ssf, log_rss
from ._homerange import (HomeRange, Lscv, LscvFit, home_range, hr_area, hr_cud, hr_isopleths, hr_kde, hr_kde_lscv,
                         hr_kde_pi, hr_kde_ref, hr_kde_ref_scaled, hr_locoh, hr_mcp, hr_overlap, hr_ud, intensity_use,
                         make_trast)
from ._measures import azimuth, distance, speed, time_lags, turn_angle
from ._read import as_tracks, load_tracks
from ._relief import surface_distance
from ._simulate import simulate_brownian
from ._ssf import Distribution, extract_covariates, fit_distr, make_distribution, random_numbers, random_steps
from ._steps import (cum_dist, filter_min_n_burst, msd, nsd, resample, sinuosity, steps, straightness, tac,
                     time_of_day, tot_dist)
from ._tracks import set_columns
from ._validate import Validation, validate
from ._ud import (WindowSuggestion, dbbmm_ud, dbbmm_variance, dbgb_ud, dbgb_variance, mask_segments, motion_variance, suggest_dbbmm_window,
                  ud_outer_probability, ud_volume)

__all__ = ["aeqd_crs", "along_summary", "as_tracks", "azimuth", "cum_dist", "distance", "Distribution", "example_data",
           "extract_covariates", "extract_covariates_along",
           "filter_gps_quality", "filter_min_n_burst", "filter_movebank_visible", "filter_per_interval",
           "filter_thin_distance", "filter_thin_time", "filter_unique", "fit_distr", "flag_speed", "gps_quality",
           "interpolate", "load_tracks", "make_distribution", "movebank_visible", "msd", "nsd", "per_interval",
           "random_numbers", "random_steps", "resample", "segments", "set_columns", "simulate_brownian", "sinuosity",
           "speed", "steps", "straightness", "tac", "thin_distance", "thin_time", "time_lags", "time_of_day",
           "tot_dist", "track_data", "track_lines", "turn_angle", "unique", "v_phys_estimate", "VPhys",
           "corridor", "corridor_lines", "ud_distance",
           "WindowSuggestion", "dbbmm_ud", "dbbmm_variance", "dbgb_ud", "dbgb_variance",
           "mask_segments", "motion_variance",
           "suggest_dbbmm_window", "ud_outer_probability", "ud_volume",
           "ISSF", "fit_clogit", "fit_issf", "fit_ssf", "log_rss",
           "HomeRange", "Lscv", "LscvFit", "home_range", "hr_area", "hr_cud", "hr_isopleths", "hr_kde", "hr_kde_lscv",
           "hr_kde_pi", "hr_kde_ref", "hr_kde_ref_scaled", "hr_locoh", "hr_mcp", "hr_overlap", "hr_ud",
           "intensity_use", "make_trast", "Validation", "validate", "surface_distance"]

