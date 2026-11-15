# SPDX-License-Identifier: GPL-3.0-or-later
"""Statistics of maps: trends along series of maps, spatial autocorrelation and interpolation of points.

Computed in C++ with OpenMP, cell by cell, on maps larger than PySAL's and
pymannkendall's loops handle comfortably, with their numbers: the Mann-Kendall
test and Sen's slope of every cell (pymannkendall's ``original_test``, the same
bits), the global and local Moran's I and Getis and Ord's Gi* over the
contiguity of the cells (PySAL's esda, to rounding), and the interpolation of
points onto a map as R's gstat: sample variograms and their fit, inverse
distance weighting, ordinary kriging. The same as Terraflux's ``stats.*``
operators.

Typical use::

    import terraflux as tf

    ndvi = tf.load_raster("ndvi_2000_2024.tif")       # a band per year
    t = tf.stats.trend(ndvi)
    t.slope, t.p, t.trend                              # Sen's slope, the test's p, up / down / none

    tf.stats.moran(t.slope)                            # global Moran's I of the slopes
    hot = tf.stats.getis_ord(tf.load_raster("density.tif")).spots

    pts = tf.load_vector("soil_samples.gpkg")          # points with a value
    v = tf.stats.variogram(pts, "carbon", model="exponential")
    k = tf.stats.kriging(pts, dem, v, "carbon")       # onto the cells of a map
    k.prediction, k.variance
"""

from ._autocorr import LISA, GetisOrd, Moran, getis_ord, lisa, moran
from ._interp import Kriging, Variogram, VariogramModel, idw, kriging, variogram
from ._raster import LocalCorrelation, cell_stats, local_correlation
from ._trend import Trend, mann_kendall, trend

__all__ = ["LISA", "GetisOrd", "Kriging", "LocalCorrelation", "Moran", "Trend", "Variogram", "VariogramModel",
           "cell_stats", "getis_ord", "idw", "kriging", "lisa", "local_correlation", "mann_kendall", "moran", "trend",
           "variogram"]
