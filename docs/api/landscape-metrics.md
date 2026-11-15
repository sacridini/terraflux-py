# Landscape metrics

Patch, class and landscape metrics of categorical maps, their sampling in
plots, windows and zones, time series, spatial signatures, plots and
utilities. Tutorial: [landscape metrics](../tutorials/landscape-metrics.md).

## Metrics of a map

::: terraflux.Landscape
    options:
      members:
        - patch_metrics
        - class_metrics
        - landscape_metrics
        - metrics
        - write_metrics
        - write_patch_metrics
        - to_raster
        - labels
        - patches
        - classes
        - n_patches
        - patch_classes
        - extent
        - out_of_core
        - centroids
        - nearest_neighbours
        - circumscribing_circles
        - adjacencies
        - boundaries
        - extract_metrics
        - show_patches
        - show_cores
        - show_lsm

::: terraflux.landscape_metrics

::: terraflux.list_metrics

## Plots, windows and zones

::: terraflux.sample_metrics

::: terraflux.scale_sample

::: terraflux.extract_metrics

::: terraflux.window_metrics

::: terraflux.zonal_metrics

::: terraflux.grid_zones

Metric tables (and zonal metrics as GeoDataFrames) are written with
[`tf.write_table`](data.md#terraflux.write_table) or [`tf.save`](data.md#terraflux.save).

## Time series and spatial signatures

::: terraflux.SpatioTemporalAnalysis

::: terraflux.SpatialSignatureAnalysis

::: terraflux.lsm.PCAResult
    options:
      show_root_full_path: false

::: terraflux.lsm.NMFResult

::: terraflux.lsm.ClustergramResult

## Plots

::: terraflux.show_patches

::: terraflux.show_cores

::: terraflux.show_lsm

::: terraflux.show_correlation

## Utilities

::: terraflux.check_landscape

::: terraflux.calculate_correlation

::: terraflux.raster_boundaries

::: terraflux.construct_buffer

::: terraflux.pad_raster

::: terraflux.unpad_raster

::: terraflux.raster_to_points

::: terraflux.get_unique_values
