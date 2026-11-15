# API reference

Every public function and class, with its parameters, what it returns and an
example you can run. Each page covers one topic; the
[tutorials](../tutorials/index.md) show the same tools at work on real maps.

<!-- no-run -->
```python
import terraflux as tf             # most tools: tf.mspa, tf.fad, tf.Landscape...
tf.conn.protconn(...)               # habitat networks and protected areas
from terraflux import circuit      # Circuitscape
tf.cost.transition(r, "mean", 8)    # distances and routes on grids
tf.surface.metrics(z)               # surface texture of continuous rasters
tf.nlm.random(100, 100, seed=1)     # neutral landscape models
tf.bio.BaseData.from_table(...)     # biodiversity analysis
tf.prioritize.zonation(features)    # spatial conservation prioritization
tf.sdm.prepare(occ, env)            # species distribution models
```

!!! example "Every example runs"
    The examples use the maps that ship with terraflux (`tf.example_data`) or
    small arrays built in place, and they are run as tests, outputs included.
    Copy one into a Python session to try a function.

<div class="grid cards" markdown>

-   :material-database-outline:{ .lg .middle } **[Data & configuration](data.md)**

    ---

    Read, write and reclassify rasters, collections, colour tables, threads.

-   :material-cloud-download-outline:{ .lg .middle } **[Data on demand](download.md)**

    ---

    WorldClim, CHELSA, Copernicus DEM, SoilGrids, MapBiomas, Hydrography90m, STAC, onto a map's grid.

-   :material-shape-outline:{ .lg .middle } **[Habitat patterns](patterns.md)**

    ---

    MSPA, SPA, objects by size, parcels and contortion.

-   :material-view-grid-outline:{ .lg .middle } **[Fragmentation](fragmentation.md)**

    ---

    Density, fragmentation classes, landscape mosaics, contagion, entropy.

-   :material-chart-box-outline:{ .lg .middle } **[Landscape metrics](landscape-metrics.md)**

    ---

    Patch, class and landscape metrics; windows, plots, zones, time series.

-   :material-graph-outline:{ .lg .middle } **[Habitat networks](networks.md)**

    ---

    Habitat graphs, IIC/PC/ECA, importance, ProtConn, grids, metapopulations, MSPA networks.

-   :material-lightning-bolt-outline:{ .lg .middle } **[Circuit connectivity](circuits.md)**

    ---

    Omniscape and Circuitscape: current, resistance distance, networks.

-   :material-terrain:{ .lg .middle } **[Surface metrics](surface.md)**

    ---

    Texture of continuous rasters (ISO 25178): roughness, shape, trend surfaces.

-   :material-waves:{ .lg .middle } **[Terrain hydrology](hydro.md)**

    ---

    Filling, flow directions and accumulation, streams, watersheds, HAND, wetness indices, river networks.

-   :material-water-outline:{ .lg .middle } **[Watershed model (SWAT+)](swat.md)**

    ---

    SWAT+ projects from TxtInOut folders or rasters, daily water, sediment and nutrients, results on maps.

-   :material-routes:{ .lg .middle } **[Grid distances](cost.md)**

    ---

    Transitions, cost surfaces, least-cost and random-walk distances and paths.

-   :material-map-marker-distance:{ .lg .middle } **[Distance & restoration](distance.md)**

    ---

    Distances, resistance, least-cost paths, corridors, zones of influence, original vegetation, intactness.

-   :material-compare-horizontal:{ .lg .middle } **[Change detection](change.md)**

    ---

    Habitat, fragmentation and landscape-mosaic change between dates.

-   :material-layers-triple-outline:{ .lg .middle } **[Large data & batch](batch.md)**

    ---

    Folders of rasters, tiled MSPA, maps larger than memory.

-   :material-dice-multiple-outline:{ .lg .middle } **[Neutral landscape models](nlm.md)**

    ---

    Simulated landscapes as null models: random, gradients, clusters, fractals.

-   :material-paw:{ .lg .middle } **[Biodiversity analysis](biodiversity.md)**

    ---

    Species counts per cell, phylogenies and dissimilarity matrices (Biodiverse).

-   :material-star-circle-outline:{ .lg .middle } **[Spatial prioritization](prioritization.md)**

    ---

    Zonation's ranking of the cells for conservation: rules, weights, costs, curves.

-   :material-map-search-outline:{ .lg .middle } **[Species distributions](sdm.md)**

    ---

    Clean records, choose predictors and background, spatial folds, fit, evaluate, project, disperse.

-   :material-map-marker-path:{ .lg .middle } **[Animal movement](move.md)**

    ---

    Movebank tracks, steps, speeds and turns, filters, track lines (move2); random steps for step selection (amt).

-   :material-timeline-clock-outline:{ .lg .middle } **[Land change simulation](simulate.md)**

    ---

    Land use change: transitions, potentials, CLUE / Dinamica / PLUS allocation, validation.

-   :material-target:{ .lg .middle } **[Map accuracy & area](accuracy.md)**

    ---

    Stratified samples of a map, its accuracy and the area of its classes (Olofsson et al. 2014).

-   :material-pine-tree:{ .lg .middle } **[Deforestation allocation (VT0007)](redd.md)**

    ---

    Verra's VT0007 as UDef-ARP: risk threshold, vulnerability, fit, allocation, assessment.

-   :material-gavel:{ .lg .middle } **[Brazil's Forest Code](forest_code.md)**

    ---

    APPs of rivers, lakes, springs, slopes, hilltops; the Reserva Legal's balance per property.

-   :material-leaf-circle-outline:{ .lg .middle } **[Ecosystem services](services.md)**

    ---

    InVEST's models: carbon, habitat quality, water yield and hydropower, sediment delivery.

-   :material-plus-minus-variant:{ .lg .middle } **[Uncertainty](uncertainty.md)**

    ---

    Totals, products, the delta method, Monte Carlo and confidence intervals.

-   :material-tune-variant:{ .lg .middle } **[Sensitivity analysis](sensitivity.md)**

    ---

    Morris's elementary effects and Sobol's indices of a model's parameters (SALib's).

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[Statistics of maps](stats.md)**

    ---

    Trends of every cell, Moran's I, LISA, Gi*; kriging, IDW; cell statistics, local correlation.

</div>

## All functions and classes

### Data & configuration

| Name | Summary |
|---|---|
| [`tf.load_raster`](data.md#terraflux.load_raster) | Read one raster, or collect many |
| [`tf.write_raster`](data.md#terraflux.write_raster) | Write a raster: one band or several, any format GDAL writes |
| [`tf.match_grid`](data.md#terraflux.match_grid) | Put a raster on the grid of another: same CRS, cells and coordinates |
| [`tf.reproject`](data.md#terraflux.reproject) | Reproject a raster or vector to another CRS, or onto the grid of another raster |
| [`tf.clip`](data.md#terraflux.clip) | Cut a raster to a region: the study area, a biome, a municipality |
| [`tf.reclassify`](data.md#terraflux.reclassify) | Give the values of a raster new values: from a table, or into a habitat map |
| [`tf.load_vector`](data.md#terraflux.load_vector) | Read a vector layer (polygons, lines, points) as a GeoDataFrame |
| [`tf.write_vector`](data.md#terraflux.write_vector) | Write a vector layer, the format given by the file extension |
| [`tf.rasterize`](data.md#terraflux.rasterize) | Burn vector features into a raster on the grid of another raster |
| [`tf.zonal_stats`](data.md#terraflux.zonal_stats) | Statistics of a raster in every zone: polygons, or the labels of a raster |
| [`tf.zonal_map`](data.md#terraflux.zonal_map) | Every cell given a statistic of its zone (zonal statistics as a raster) |
| [`tf.tabulate_area`](data.md#terraflux.tabulate_area) | The area of every class of a map in every zone of another (ArcGIS's Tabulate Area) |
| [`tf.extract`](data.md#terraflux.extract) | The values of a raster at points |
| [`tf.load_table`](data.md#terraflux.load_table) | Read a table, the format given by the file extension |
| [`tf.write_table`](data.md#terraflux.write_table) | Write a table, the format given by the file extension |
| [`tf.save`](data.md#terraflux.save) | Save data or a result, each part in the format that suits it |
| [`tf.load_result`](data.md#terraflux.load_result) | Read back a result folder written by `save` (or ``result.save``) |
| [`tf.plot`](data.md#terraflux.plot) | Plot anything: a raster, a vector layer, a table or the result of a tool, as it calls for |
| [`tf.Plot`](data.md#terraflux.Plot) | A plot made by `plot`: interactive maps and charts (or a static figure) |
| [`tf.example_data`](data.md#terraflux.example_data) | Path of an example data set shipped with the package |
| [`tf.RasterCollection`](data.md#terraflux.RasterCollection) | Many rasters handled as one object; pixels are read only when processed |
| [`tf.RasterInfo`](data.md#terraflux.RasterInfo) | Header of one raster of a collection (the pixels are not read) |
| [`tf.validate_mspa_input`](data.md#terraflux.validate_mspa_input) | Check that ``data`` is a habitat map MSPA can process |
| [`tf.validate_fad_input`](data.md#terraflux.validate_fad_input) | Check that ``data`` is a habitat map the density and object tools accept |
| [`tf.InvalidInputError`](data.md#terraflux.InvalidInputError) | The raster cannot be processed by the requested tool |
| [`tf.mspa_colormap`](data.md#terraflux.mspa_colormap) | Colour table of MSPA class maps (`mspa`) |
| [`tf.fad_colormap`](data.md#terraflux.fad_colormap) | Colour table of density maps of `fad`, `fad_multiscale` and `fos` (6 classes) |
| [`tf.fad5_colormap`](data.md#terraflux.fad5_colormap) | Colour table of 5-class fragmentation maps (`fos` with ``classes=5``) |
| [`tf.lm_colormap`](data.md#terraflux.lm_colormap) | Colour table of landscape mosaic maps (`landscape_mosaic`) |
| [`tf.accounting_colormap`](data.md#terraflux.accounting_colormap) | Colour table of object size-class maps (`accounting`) |
| [`tf.change_colormap`](data.md#terraflux.change_colormap) | Colour table of change maps (`simple_change`, `morphological_change`) |
| [`tf.foschange_colormap`](data.md#terraflux.foschange_colormap) | Colour table of fragmentation change maps (`fos_change`) |
| [`tf.dist_colormap`](data.md#terraflux.dist_colormap) | Colour table of distance maps (`euclidean_distance`) |
| [`tf.entropy_colormap`](data.md#terraflux.entropy_colormap) | Colour table of spatial entropy maps (`spatial_entropy_map`) |
| [`tf.contortion_colormap`](data.md#terraflux.contortion_colormap) | Colour table of contortion maps (`contortion`) |
| [`tf.resistance_colormap`](data.md#terraflux.resistance_colormap) | Colour table of resistance and isochrone maps (`distance_to_resistance`, `isochrone_map`) |
| [`tf.gtn_components_colormap`](data.md#terraflux.gtn_components_colormap) | Colour table of network component maps (`gtn_components`) |
| [`tf.gtn_importance_colormap`](data.md#terraflux.gtn_importance_colormap) | Colour table of network importance maps (`gtn_importance`) |
| [`tf.default_threads`](data.md#terraflux.default_threads) | The number of threads a parallel function uses when not given one |
| [`tf.set_default_threads`](data.md#terraflux.set_default_threads) | Set the default thread count of every parallel function for this process |
| [`tf.max_threads`](data.md#terraflux.max_threads) | Number of processors the compiled core can use |
| [`tf.doctor`](data.md#terraflux.doctor) | Print what terraflux finds on this computer |

### Data on demand

| Name | Summary |
|---|---|
| [`tf.download.worldclim`](download.md#terraflux.download.worldclim) | WorldClim 2.1's climate on a grid: the historical climate (1970-2000) or a CMIP6 future |
| [`tf.download.chelsa`](download.md#terraflux.download.chelsa) | CHELSA 2.1's bioclimatic variables on a grid: 1981-2010, or a CMIP6 future |
| [`tf.download.copernicus_dem`](download.md#terraflux.download.copernicus_dem) | The Copernicus DEM (GLO-30 or GLO-90) on a grid, from its tiles of a degree over the grid |
| [`tf.download.soilgrids`](download.md#terraflux.download.soilgrids) | SoilGrids 2.0's properties on a grid, at depths |
| [`tf.download.mapbiomas`](download.md#terraflux.download.mapbiomas) | MapBiomas Brazil's land use and cover of years on a grid (collections 9, 10 and 11) |
| [`tf.download.hydrography90m`](download.md#terraflux.download.hydrography90m) | Hydrography90m (Amatulli et al. 2022) on a grid: the river network of the world at 90 m |
| [`tf.download.environment90m`](download.md#terraflux.download.environment90m) | The environment of every sub-catchment of a map of Hydrography90m's sub-catchments, from Environment90m |
| [`tf.download.Environment90m`](download.md#terraflux.download.Environment90m) | The environment of Hydrography90m's sub-catchments (`environment90m`) |
| [`tf.download.raster`](download.md#terraflux.download.raster) | Any raster GDAL reads (a URL, a file), or several, warped onto a grid; only the part the grid covers is read |
| [`tf.download.stac`](download.md#terraflux.download.stac) | The scenes of a STAC catalogue over a grid, each asset warped onto it and the scenes combined |

### Habitat patterns

| Name | Summary |
|---|---|
| [`terraflux.mspa`](patterns.md#terraflux.mspa) | Morphological Spatial Pattern Analysis (MSPA) |
| [`tf.mspa_stats`](patterns.md#terraflux.mspa_stats) | Area of every MSPA class |
| [`terraflux.spa`](patterns.md#terraflux.spa) | Simplified Pattern Analysis (GuidosToolbox *Pattern -> Morphological -> SPA2/3/5/6*) |
| [`tf.SPAResult`](patterns.md#terraflux.SPAResult) | Result of `spa` |
| [`terraflux.accounting`](patterns.md#terraflux.accounting) | Accounting: foreground objects grouped into size classes |
| [`tf.AccountingResult`](patterns.md#terraflux.AccountingResult) | Result of `accounting` |
| [`tf.parcellation`](patterns.md#terraflux.parcellation) | How finely the landscape is divided into parcels |
| [`tf.ParcellationResult`](patterns.md#terraflux.ParcellationResult) | Results from landscape parcellation analysis |
| [`tf.contortion`](patterns.md#terraflux.contortion) | How convoluted the boundary of every habitat object is |

### Fragmentation

| Name | Summary |
|---|---|
| [`terraflux.fad`](fragmentation.md#terraflux.fad) | Forest Area Density (FAD) & Multiscale Fragmentation Analysis |
| [`tf.FADResult`](fragmentation.md#terraflux.FADResult) | Result of single-scale FAD analysis (`fad`) |
| [`tf.fad_multiscale`](fragmentation.md#terraflux.fad_multiscale) | Compute Multiscale Forest Area Density (FAD-MS) matching GuidosToolbox |
| [`tf.FADMultiscaleResult`](fragmentation.md#terraflux.FADMultiscaleResult) | Result of multiscale FAD analysis (`fad_multiscale`) |
| [`terraflux.fos`](fragmentation.md#terraflux.fos) | Fragmentation at a Fixed Observation Scale (GuidosToolbox *Fragmentation -> FOS*) |
| [`tf.FOSResult`](fragmentation.md#terraflux.FOSResult) | Result of `fos` |
| [`tf.landscape_mosaic`](fragmentation.md#terraflux.landscape_mosaic) | Compute Landscape Mosaic (LM) moving-window metric matching GuidosToolbox batch_lm |
| [`tf.landscape_mosaic_stats`](fragmentation.md#terraflux.landscape_mosaic_stats) | Compute Landscape Mosaic 103-class frequency distribution matching GTB heatmap.csv |
| [`tf.landscape_mosaic_heatmap`](fragmentation.md#terraflux.landscape_mosaic_heatmap) | Heatmap of a Landscape Mosaic, as GTB computes it |
| [`tf.LMHeatmap`](fragmentation.md#terraflux.LMHeatmap) | Landscape Mosaic heatmap of one map (GTB ``*_heatmap.sav`` / ``.csv``) |
| [`tf.contagion_pff`](fragmentation.md#terraflux.contagion_pff) | Compute moving-window Contagion (Pff) matching GuidosToolbox batch_pff |
| [`tf.clustering_fac`](fragmentation.md#terraflux.clustering_fac) | Compute moving-window Foreground Area Clustering (FAC) matching GuidosToolbox batch_fac |
| [`tf.spatial_entropy`](fragmentation.md#terraflux.spatial_entropy) | Compute global normalized spatial joint directional entropy index |
| [`tf.SpatialEntropyResult`](fragmentation.md#terraflux.SpatialEntropyResult) | Result of global spatial entropy analysis (`spatial_entropy`) |
| [`tf.spatial_entropy_map`](fragmentation.md#terraflux.spatial_entropy_map) | Compute moving-window spatial entropy map matching GuidosToolbox batch_ent |
| [`tf.gray_spatial_convolution`](fragmentation.md#terraflux.gray_spatial_convolution) | Compute moving-window spatial convolution metric over a grayscale raster |

### Landscape metrics

| Name | Summary |
|---|---|
| [`tf.Landscape`](landscape-metrics.md#terraflux.Landscape) | Categorical raster prepared for landscape metrics |
| [`tf.landscape_metrics`](landscape-metrics.md#terraflux.landscape_metrics) | Landscape metrics of one or several rasters, in long format |
| [`tf.list_metrics`](landscape-metrics.md#terraflux.list_metrics) | Available landscape metrics |
| [`tf.sample_metrics`](landscape-metrics.md#terraflux.sample_metrics) | Landscape metrics of sample plots, like landscapemetrics' ``sample_lsm`` |
| [`tf.scale_sample`](landscape-metrics.md#terraflux.scale_sample) | Landscape metrics of buffers of several sizes around points, like landscapemetrics' ``scale_sample`` |
| [`tf.extract_metrics`](landscape-metrics.md#terraflux.extract_metrics) | Patch metrics of the patches under points, like landscapemetrics' ``extract_lsm`` |
| [`tf.window_metrics`](landscape-metrics.md#terraflux.window_metrics) | Landscape-level metrics in a moving window, like landscapemetrics' ``window_lsm`` |
| [`tf.zonal_metrics`](landscape-metrics.md#terraflux.zonal_metrics) | Landscape and class metrics of every zone, as a GeoDataFrame |
| [`tf.grid_zones`](landscape-metrics.md#terraflux.grid_zones) | Square zones tiling a raster, for zonal analyses |
| [`tf.SpatioTemporalAnalysis`](landscape-metrics.md#terraflux.SpatioTemporalAnalysis) | Landscape metrics along a time series of rasters (pylandstats' ``SpatioTemporalAnalysis`` and its zonal, buffer and grid variants) |
| [`tf.SpatialSignatureAnalysis`](landscape-metrics.md#terraflux.SpatialSignatureAnalysis) | Landscapes (or zones) described by their metrics: the "spatial signature" matrix, its principal components and clusters (pylandstats' ``SpatialSignatureAnalysis``) |
| [`tf.lsm.PCAResult`](landscape-metrics.md#terraflux.lsm.PCAResult) | Principal components of the spatial signatures |
| [`tf.lsm.NMFResult`](landscape-metrics.md#terraflux.lsm.NMFResult) | Non-negative factors of the spatial signatures (min-max scaled metrics ~ scores @ components.T) |
| [`tf.lsm.ClustergramResult`](landscape-metrics.md#terraflux.lsm.ClustergramResult) | Clusters for a range of k (Schonlau's clustergram, as the clustergram package that pylandstats' ``get_cgram`` uses) |
| [`tf.show_patches`](landscape-metrics.md#terraflux.show_patches) | Map of the patches, each in its own colour (landscapemetrics' ``show_patches``) |
| [`tf.show_cores`](landscape-metrics.md#terraflux.show_cores) | Core and edge cells of the patches (landscapemetrics' ``show_cores``): core cells dark, edge cells light, other classes blank |
| [`tf.show_lsm`](landscape-metrics.md#terraflux.show_lsm) | A patch-level metric painted on the patches (landscapemetrics' ``show_lsm``), with a colour bar; ``labels`` writes the values |
| [`tf.show_correlation`](landscape-metrics.md#terraflux.show_correlation) | Heatmap of the correlations between metrics (landscapemetrics' ``show_correlation``), one panel per level |
| [`tf.check_landscape`](landscape-metrics.md#terraflux.check_landscape) | Is a raster suitable for landscape metrics? Like landscapemetrics' ``check_landscape`` |
| [`tf.calculate_correlation`](landscape-metrics.md#terraflux.calculate_correlation) | Correlation between metrics, like landscapemetrics' ``calculate_correlation`` |
| [`tf.raster_boundaries`](landscape-metrics.md#terraflux.raster_boundaries) | landscapemetrics' ``get_boundaries`` on the values of a raster |
| [`tf.construct_buffer`](landscape-metrics.md#terraflux.construct_buffer) | Square or circular sample plots around points, like landscapemetrics' ``construct_buffer`` |
| [`tf.pad_raster`](landscape-metrics.md#terraflux.pad_raster) | Add rows and columns of ``pad_raster_value`` around a raster, like landscapemetrics' ``pad_raster`` |
| [`tf.unpad_raster`](landscape-metrics.md#terraflux.unpad_raster) | Remove ``unpad_raster_cells`` rows and columns on each side of a raster, like landscapemetrics' ``unpad_raster`` (the inverse of `pad_raster`) |
| [`tf.raster_to_points`](landscape-metrics.md#terraflux.raster_to_points) | Cell centres and values, like landscapemetrics' ``raster_to_points`` |
| [`tf.get_unique_values`](landscape-metrics.md#terraflux.get_unique_values) | The distinct values of rasters (NA left out, doubles truncated to integers), sorted, like landscapemetrics' ``get_unique_values`` |

### Habitat networks

| Name | Summary |
|---|---|
| [`tf.HabitatGraph`](networks.md#terraflux.HabitatGraph) | Habitat patches (nodes with an attribute) and the connections between them |
| [`tf.conn.node_table`](networks.md#terraflux.conn.node_table) | Node ids and attributes, as Makurhini's ``nodesfile`` |
| [`tf.conn.distance_table`](networks.md#terraflux.conn.distance_table) | Euclidean distances between nodes, as Makurhini's ``distancefile`` |
| [`tf.conn.settlement_probabilities`](networks.md#terraflux.conn.settlement_probabilities) | Probabilities of settling in one patch from another under a spatial absorbing Markov chain |
| [`tf.conn.boundary_vertices`](networks.md#terraflux.conn.boundary_vertices) | Vertices of a polygon picked as Makurhini's ``SR_vertices`` does (systematic) |
| [`tf.conn.protconn`](networks.md#terraflux.conn.protconn) | Protected Connected land (ProtConn) and its fractions, as Makurhini's MK_ProtConn (Saura et al. 2017, 2018) |
| [`tf.conn.protconn_regions`](networks.md#terraflux.conn.protconn_regions) | ProtConn of several regions, one row per region and distance (Makurhini's ``MK_ProtConnMult``) |
| [`tf.conn.protconn_raster`](networks.md#terraflux.conn.protconn_raster) | ProtConn of a region from a raster of protected areas, as Makurhini's ``MK_ProtConn_raster`` |
| [`tf.conn.protconn_nodes`](networks.md#terraflux.conn.protconn_nodes) | Nodes of ProtConn in a region, as Makurhini's ``Protconn_nodes`` builds them |
| [`tf.conn.ProtConnNodes`](networks.md#terraflux.conn.ProtConnNodes) | The nodes of a ProtConn analysis of one region (`protconn_nodes`) |
| [`tf.conn.protconn_indicators`](networks.md#terraflux.conn.protconn_indicators) | ProtConn and its fractions from prepared nodes and distances (Makurhini's ``get_protconn_grid``) |
| [`tf.conn.make_grid`](networks.md#terraflux.conn.make_grid) | Grid of cells of `cell_area` over a region, as Makurhini's ``make_grid`` |
| [`tf.conn.connectivity_grid`](networks.md#terraflux.conn.connectivity_grid) | ProtConn (or PC and ECA) in every cell of a grid, as Makurhini's ``MK_Connect_grid`` |
| [`tf.conn.eca_change_grid`](networks.md#terraflux.conn.eca_change_grid) | ECA change between times in every cell of a grid, as Makurhini's ``MK_dECA_grid`` |
| [`tf.conn.eca_change`](networks.md#terraflux.conn.eca_change) | ECA of habitat graphs over time and its change, as Makurhini's ``MK_dECA`` |
| [`tf.conn.focal_nodes`](networks.md#terraflux.conn.focal_nodes) | IIC or PC of every node's neighbourhood and the node's importance in it (``MK_Focal_nodes``) |
| [`tf.conn.metapopulation`](networks.md#terraflux.conn.metapopulation) | A metapopulation in a network of habitat patches: Hanski's incidence function model as MetaLandSim simulates it |
| [`tf.conn.Metapopulation`](networks.md#terraflux.conn.Metapopulation) | A metapopulation's simulation (`metapopulation`) |
| [`tf.gtn_components`](networks.md#terraflux.gtn_components) | Components of the habitat network formed by MSPA Cores and Bridges |
| [`tf.GTNComponentsResult`](networks.md#terraflux.GTNComponentsResult) | Result of `gtn_components` |
| [`tf.gtn_importance`](networks.md#terraflux.gtn_importance) | Importance of every link (Bridge) and stepping-stone node (Core) of the MSPA network |
| [`tf.GTNImportanceResult`](networks.md#terraflux.GTNImportanceResult) | Result of `gtn_importance` |
| [`tf.conefor_inputs`](networks.md#terraflux.conefor_inputs) | Nodes and links of the MSPA network as Conefor Sensinode input files |

### Circuit connectivity

| Name | Summary |
|---|---|
| [`tf.omniscape`](circuits.md#terraflux.circuit.omniscape.omniscape) | Omnidirectional connectivity with Omniscape's moving-window circuits |
| [`circuit.omniscape_ini`](circuits.md#terraflux.circuit.omniscape_ini) | Run an Omniscape INI file, as ``run_omniscape(path)`` does |
| [`circuit.source_from_resistance`](circuits.md#terraflux.circuit.source_from_resistance) | Source strengths from the resistance, as Omniscape's ``source_from_resistance`` |
| [`circuit.read_reclass_table`](circuits.md#terraflux.circuit.read_reclass_table) | Read an Omniscape reclass table |
| [`circuit.reclassify`](circuits.md#terraflux.circuit.reclassify) | Reclassify a resistance array, as Omniscape's ``reclassify_resistance!`` |
| [`circuit.parse_ini`](circuits.md#terraflux.circuit.parse_ini) | Read an INI file as Omniscape's ``parse_cfg`` does |
| [`circuit.circuitscape_pairwise`](circuits.md#terraflux.circuit.circuitscape_pairwise) | Circuitscape's pairwise mode on rasters (``scenario = pairwise``) |
| [`circuit.circuitscape_one_to_all`](circuits.md#terraflux.circuit.circuitscape_one_to_all) | Circuitscape's one-to-all (or, with ``all_to_one``, all-to-one) mode on rasters |
| [`circuit.circuitscape_ini`](circuits.md#terraflux.circuit.circuitscape_ini) | Run a Circuitscape INI file, as ``Circuitscape.compute(path)`` does |
| [`circuit.CircuitscapeResult`](circuits.md#terraflux.circuit.CircuitscapeResult) | Circuitscape's outputs of a raster run |
| [`circuit.FocalPoints`](circuits.md#terraflux.circuit.FocalPoints) | Focal points as Circuitscape reads them: 1-based rows and columns and ids, sorted by id |
| [`circuit.IncludedPairs`](circuits.md#terraflux.circuit.IncludedPairs) | Circuitscape's include / exclude pairs |
| [`circuit.circuitscape_advanced`](circuits.md#terraflux.circuit.circuitscape_advanced) | Circuitscape's advanced mode on rasters (``scenario = advanced``) |
| [`circuit.advanced`](circuits.md#terraflux.circuit.advanced.advanced) | Currents and voltages of a raster circuit with sources and grounds |
| [`circuit.AdvancedResult`](circuits.md#terraflux.circuit.AdvancedResult) | Node currents and voltages of `advanced` |
| [`circuit.circuitscape_network_pairwise`](circuits.md#terraflux.circuit.circuitscape_network_pairwise) | Circuitscape's pairwise mode on a network (``data_type = network``, ``scenario = pairwise``) |
| [`circuit.circuitscape_network_advanced`](circuits.md#terraflux.circuit.circuitscape_network_advanced) | Circuitscape's advanced mode on a network (``data_type = network``, ``scenario = advanced``) |
| [`circuit.NetworkResult`](circuits.md#terraflux.circuit.NetworkResult) | Circuitscape's network outputs, with the node numbers it writes (from 1) |
| [`circuit.read_graph`](circuits.md#terraflux.circuit.read_graph) | Read a Circuitscape edge list, as its ``load_graph`` |

### Surface metrics

| Name | Summary |
|---|---|
| [`tf.surface.metrics`](surface.md#terraflux.surface.metrics) | Surface metrics of a raster, as a Series indexed by metric |
| [`tf.surface.list_metrics`](surface.md#terraflux.surface.list_metrics) | The surface metrics `metrics` computes |
| [`tf.surface.sa`](surface.md#terraflux.surface.sa) | Average roughness (Sa): the mean absolute deviation from the mean height |
| [`tf.surface.sq`](surface.md#terraflux.surface.sq) | Root mean square roughness (Sq): the root mean square of the heights about their mean, sqrt(sum((z - mean(z))^2) / N) (ISO 25178; geodiv uses the sample standard deviation, over N - 1) |
| [`tf.surface.ssk`](surface.md#terraflux.surface.ssk) | Skewness of the height distribution (Ssk) |
| [`tf.surface.sku`](surface.md#terraflux.surface.sku) | Kurtosis of the height distribution (Sku) |
| [`tf.surface.sv`](surface.md#terraflux.surface.sv) | Maximum valley depth (Sv): the absolute value of the lowest height |
| [`tf.surface.sph`](surface.md#terraflux.surface.sph) | Maximum peak height (Sp): the absolute value of the highest height |
| [`tf.surface.smean`](surface.md#terraflux.surface.smean) | Mean peak height: the mean of the positive heights |
| [`tf.surface.sdq`](surface.md#terraflux.surface.sdq) | Root mean square slope (Sdq) |
| [`tf.surface.sdq6`](surface.md#terraflux.surface.sdq6) | Area root mean square slope (Sdq6), from seven-point derivatives |
| [`tf.surface.sdr`](surface.md#terraflux.surface.sdr) | Developed interfacial area ratio (Sdr), in percent |
| [`tf.surface.surface_area`](surface.md#terraflux.surface.surface_area) | Area of the surface, triangulated, in geodiv's units |
| [`tf.surface.flat_area`](surface.md#terraflux.surface.flat_area) | Area of the surface's projection on a plane (geodiv's ``flatsa``) |
| [`tf.surface.find_peaks`](surface.md#terraflux.surface.find_peaks) | The peaks of a surface: cells higher than their 8 neighbours and than 0 |
| [`tf.surface.find_valleys`](surface.md#terraflux.surface.find_valleys) | The valleys of a surface: cells lower than their 8 neighbours and than 0 |
| [`tf.surface.sds`](surface.md#terraflux.surface.sds) | Density of summits (Sds): peaks per cell |
| [`tf.surface.ssc`](surface.md#terraflux.surface.ssc) | Mean summit curvature (Ssc) |
| [`tf.surface.s10z`](surface.md#terraflux.surface.s10z) | Ten-point height (S10z) |
| [`tf.surface.bearing_area`](surface.md#terraflux.surface.bearing_area) | The bearing area (Abbott-Firestone) curve: the share of the surface at or above each height |
| [`tf.surface.height_ba`](surface.md#terraflux.surface.height_ba) | Height of the bearing area curve at a bearing fraction |
| [`tf.surface.sdc`](surface.md#terraflux.surface.sdc) | Areal material ratio height difference (Sdc) |
| [`tf.surface.sbi`](surface.md#terraflux.surface.sbi) | Surface bearing index (Sbi): Sq divided by the height at 5 % bearing |
| [`tf.surface.svi`](surface.md#terraflux.surface.svi) | Valley fluid retention index (Svi): the void volume of the valley zone |
| [`tf.surface.sci`](surface.md#terraflux.surface.sci) | Core fluid retention index (Sci): the void volume of the core zone |
| [`tf.surface.sk`](surface.md#terraflux.surface.sk) | Core roughness depth (Sk): the height range of the core line |
| [`tf.surface.spk`](surface.md#terraflux.surface.spk) | Reduced peak height (Spk): the peaks above the core |
| [`tf.surface.svk`](surface.md#terraflux.surface.svk) | Reduced valley depth (Svk): the valleys below the core |
| [`tf.surface.aacf`](surface.md#terraflux.surface.aacf) | Areal autocorrelation function of a surface |
| [`tf.surface.scl`](surface.md#terraflux.surface.scl) | Correlation lengths: how far the autocorrelation takes to decay |
| [`tf.surface.stxr`](surface.md#terraflux.surface.stxr) | Texture aspect ratio (Str): fastest over slowest decay length |
| [`tf.surface.std`](surface.md#terraflux.surface.std) | Texture direction (Std) and texture direction index (Stdi) |
| [`tf.surface.srw`](surface.md#terraflux.surface.srw) | Dominant radial wavelength (Srw), its index (Srwi) and the half-power wavelength (Shw) |
| [`tf.surface.sfd`](surface.md#terraflux.surface.sfd) | Fractal dimension of the surface (Sfd), by triangular prisms |
| [`tf.surface.window_metrics`](surface.md#terraflux.surface.window_metrics) | Surface metrics in a moving window: one texture image per metric |
| [`tf.surface.terrain`](surface.md#terraflux.surface.terrain) | Slope or aspect of a DEM, as terra's ``terrain`` |
| [`tf.surface.tpi`](surface.md#terraflux.surface.tpi) | Topographic position index: the elevation minus the mean of the window |
| [`tf.surface.tri`](surface.md#terraflux.surface.tri) | Terrain ruggedness index (Riley et al. 1999) |
| [`tf.surface.vrm`](surface.md#terraflux.surface.vrm) | Vector ruggedness measure (Sappington et al. 2007) |
| [`tf.surface.srr`](surface.md#terraflux.surface.srr) | Surface relief ratio (Pike & Wilson 1971): (mean - min) / (max - min) of the window |
| [`tf.surface.dissection`](surface.md#terraflux.surface.dissection) | Martonne's modified dissection: (z - min) / (max - min) of the window |
| [`tf.surface.curvature`](surface.md#terraflux.surface.curvature) | Surface curvature of a DEM |
| [`tf.surface.spherical_sd`](surface.md#terraflux.surface.spherical_sd) | Spherical standard deviation of the slope and aspect (terrain heterogeneity) |
| [`tf.surface.hsp`](surface.md#terraflux.surface.hsp) | Hierarchical slope position (Murphy et al. 2010) |
| [`tf.surface.trasp`](surface.md#terraflux.surface.trasp) | Transformed aspect (Roberts & Cooper 1989): (1 - cos(aspect - 30 deg)) / 2 |
| [`tf.surface.hli`](surface.md#terraflux.surface.hli) | Heat load index (McCune & Keon 2002) |
| [`tf.surface.geomorphons`](surface.md#terraflux.surface.geomorphons) | Landforms of a DEM by geomorphons (Jasiewicz & Stepinski 2013) |
| [`tf.surface.Geomorphons`](surface.md#terraflux.surface.Geomorphons) | The geomorphons of a DEM (`geomorphons`) |
| [`tf.surface.hilltop`](surface.md#terraflux.surface.hilltop) | Hilltops as permanent protection areas (APP de topo de morro) |
| [`tf.surface.Hilltop`](surface.md#terraflux.surface.Hilltop) | The hilltops of a DEM (`hilltop`) |
| [`tf.surface.viewshed`](surface.md#terraflux.surface.viewshed) | How many observers see every cell of an elevation map (gdal_viewshed, GRASS's r.viewshed, for many observers) |
| [`tf.surface.gravity`](surface.md#terraflux.surface.gravity) | Gravitational interaction potential of source cells |
| [`tf.surface.fit_plane`](surface.md#terraflux.surface.fit_plane) | The least-squares polynomial surface of a given order |
| [`tf.surface.best_fit_plane`](surface.md#terraflux.surface.best_fit_plane) | The polynomial surface of order 0 to 3 that fits a surface best |
| [`tf.surface.remove_plane`](surface.md#terraflux.surface.remove_plane) | A surface minus its best-fitting polynomial surface (geodiv's ``remove_plane``) |
| [`tf.surface.example_data`](surface.md#terraflux.surface.example_data) | Path of an example surface |

### Terrain hydrology

| Name | Summary |
|---|---|
| [`tf.hydro.fill`](hydro.md#terraflux.hydro.fill) | Fill the depressions of a DEM, so that water can flow from every cell to the edge |
| [`tf.hydro.breach`](hydro.md#terraflux.hydro.breach) | Remove the depressions of a DEM by cutting channels through their rims |
| [`tf.hydro.flow_direction`](hydro.md#terraflux.hydro.flow_direction) | The direction each cell drains to |
| [`tf.hydro.flow_accumulation`](hydro.md#terraflux.hydro.flow_accumulation) | The area that drains through each cell (the cell itself included) |
| [`tf.hydro.streams`](hydro.md#terraflux.hydro.streams) | The stream cells: those whose flow accumulation is at least ``threshold`` |
| [`tf.hydro.distance_to_channel`](hydro.md#terraflux.hydro.distance_to_channel) | The mean distance water travels from each cell to a stream, along pygeoprocessing's multiple flow directions (as InVEST's sediment delivery model measures its downslope component) |
| [`tf.hydro.stream_links`](hydro.md#terraflux.hydro.stream_links) | The links of a stream network: the stretches between sources, junctions and outlets |
| [`tf.hydro.watersheds`](hydro.md#terraflux.hydro.watersheds) | The area draining to each seed: every cell labelled with the first seed below it |
| [`tf.hydro.snap_outlets`](hydro.md#terraflux.hydro.snap_outlets) | The stream cell each outlet point stands for |
| [`tf.hydro.snap_points`](hydro.md#terraflux.hydro.snap_points) | Points moved onto the stream network, as hydrographr's ``snap_to_network`` |
| [`tf.hydro.upstream`](hydro.md#terraflux.hydro.upstream) | The land upstream of every link of a stream network, or of every point on it, summarised |
| [`tf.hydro.Upstream`](hydro.md#terraflux.hydro.Upstream) | The land upstream of links or points (`upstream`) |
| [`tf.hydro.upstream_catchment`](hydro.md#terraflux.hydro.upstream_catchment) | The catchment of every point (a gauge, a dam, a site), as hydrographr's ``get_upstream_catchment`` |
| [`tf.hydro.Catchments`](hydro.md#terraflux.hydro.Catchments) | The catchments of points (`upstream_catchment`) |
| [`tf.hydro.network_distance`](hydro.md#terraflux.hydro.network_distance) | The distance along the river network between every two points, as hydrographr's ``get_distance`` |
| [`tf.hydro.along_network`](hydro.md#terraflux.hydro.along_network) | The stretches of river up and down from every point within a distance along the network |
| [`tf.hydro.NetworkReach`](hydro.md#terraflux.hydro.NetworkReach) | The stretches of river up and down from points (`along_network`) |
| [`tf.hydro.centrality`](hydro.md#terraflux.hydro.centrality) | The place of every link in its river network, as hydrographr's ``get_centrality`` on the graph of the links |
| [`tf.hydro.dci`](hydro.md#terraflux.hydro.dci) | The dendritic connectivity index of a river network cut by barriers (Cote et al. 2009), as R's riverconn |
| [`tf.hydro.DCI`](hydro.md#terraflux.hydro.DCI) | The dendritic connectivity index of a river network cut by barriers (`dci`) |
| [`tf.hydro.hand`](hydro.md#terraflux.hydro.hand) | Height Above the Nearest Drainage (HAND; Rennó et al. 2008, Nobre et al. 2011) |
| [`tf.hydro.twi`](hydro.md#terraflux.hydro.twi) | Topographic wetness index: ln(a / tan(slope)) (Beven & Kirkby 1979) |
| [`tf.hydro.spi`](hydro.md#terraflux.hydro.spi) | Stream power index: a · tan(slope) (Moore, Grayson & Ladson 1991), the erosive power of overland flow |
| [`tf.hydro.sti`](hydro.md#terraflux.hydro.sti) | Sediment transport index, the LS factor of the USLE from the contributing area (Moore & Burch 1986): (m + 1) (a / 22.13)^m (sin(slope) / 0.0896)^n |
| [`tf.hydro.connectivity_index`](hydro.md#terraflux.hydro.connectivity_index) | The index of connectivity of Borselli et al. (2008), as InVEST's SDR model computes it |

### Watershed model (SWAT+)

| Name | Summary |
|---|---|
| [`tf.hydro.swat.load_txtinout`](swat.md#terraflux.hydro.swat.load_txtinout) | Read a SWAT+ TxtInOut folder, or a zip file of one (the outputs of earlier runs in it are left out) |
| [`tf.hydro.swat.SwatProject`](swat.md#terraflux.hydro.swat.SwatProject) | The input files of a SWAT+ model (the TxtInOut folder), by name |
| [`tf.hydro.swat.run`](swat.md#terraflux.hydro.swat.run) | Simulate a SWAT+ project, as SWAT+ 62.0.1 does |
| [`tf.hydro.swat.SwatResult`](swat.md#terraflux.hydro.swat.SwatResult) | What `run` simulated |
| [`tf.hydro.swat.watershed`](swat.md#terraflux.hydro.swat.watershed) | Divide a watershed into subbasins, channels and HRUs, as QSWAT+ does |
| [`tf.hydro.swat.SwatWatershed`](swat.md#terraflux.hydro.swat.SwatWatershed) | A watershed divided as QSWAT+ divides it, ready for the SWAT+ Editor's import |
| [`tf.hydro.swat.from_tables`](swat.md#terraflux.hydro.swat.from_tables) | A SWAT+ project from QSWAT+'s tables and the weather, as the SWAT+ Editor makes it |
| [`tf.hydro.swat.from_rasters`](swat.md#terraflux.hydro.swat.from_rasters) | A SWAT+ project from a DEM, land use and soil maps, tables and the weather |
| [`tf.hydro.swat.load_qswat_project`](swat.md#terraflux.hydro.swat.load_qswat_project) | The tables of a QSWAT+ project database, for `from_tables` |
| [`tf.hydro.swat.example_landscape`](swat.md#terraflux.hydro.swat.example_landscape) | The inputs of a small watershed model, from the package's example rasters |
| [`tf.hydro.swat.climate`](swat.md#terraflux.hydro.swat.climate) | The weather of a SWAT+ project from tables |
| [`tf.hydro.swat.SwatClimate`](swat.md#terraflux.hydro.swat.SwatClimate) | The weather of a SWAT+ project (made by `climate`) |
| [`tf.hydro.swat.wgn_from_series`](swat.md#terraflux.hydro.swat.wgn_from_series) | SWAT's weather generator statistics (WXGEN) of each station, from daily series |
| [`tf.hydro.swat.hru_map`](swat.md#terraflux.hydro.swat.hru_map) | An HRU result on the map: each cell takes the value of its HRU |
| [`tf.hydro.swat.channel_map`](swat.md#terraflux.hydro.swat.channel_map) | A channel result on the stream cells (by default the mean discharge, m3/s) |
| [`tf.hydro.swat.reach_discharge`](swat.md#terraflux.hydro.swat.reach_discharge) | The daily discharge leaving each channel (m3/s, ``channel_sd``'s ``flo_out``): dates x channel number. ``result``: a run that printed ``channel_sd`` by day |

### Grid distances

| Name | Summary |
|---|---|
| [`tf.cost.transition`](cost.md#terraflux.cost.transition) | Transition matrix from a raster: gdistance's ``transition()`` |
| [`tf.cost.Transition`](cost.md#terraflux.cost.Transition) | A transition matrix on a raster grid: gdistance's ``TransitionLayer`` |
| [`tf.cost.geo_correction`](cost.md#terraflux.cost.geo_correction) | Correct a transition for the distances between cell centres: gdistance's ``geoCorrection()`` |
| [`tf.cost.normalize`](cost.md#terraflux.cost.normalize) | Normalise a transition matrix: gdistance's ``normalize()`` |
| [`tf.cost.sum_reciprocal`](cost.md#terraflux.cost.sum_reciprocal) | Combine two transitions as resistances in series: gdistance's ``sumReciprocal()`` |
| [`tf.cost.acc_cost`](cost.md#terraflux.cost.acc_cost) | Accumulated cost surface from the nearest origin: gdistance's ``accCost()`` |
| [`tf.cost.cost_distance`](cost.md#terraflux.cost.cost_distance) | Least-cost distances between points: gdistance's ``costDistance()`` |
| [`tf.cost.shortest_path`](cost.md#terraflux.cost.shortest_path) | Least-cost path from an origin to one or more goals: gdistance's ``shortestPath()`` |
| [`tf.cost.commute_distance`](cost.md#terraflux.cost.commute_distance) | Expected commute distances of a random walk between points: gdistance's ``commuteDistance()`` |
| [`tf.cost.passage`](cost.md#terraflux.cost.passage) | Passages of a random walk from origins to goals: gdistance's ``passage()`` |
| [`tf.cost.rsp_distance`](cost.md#terraflux.cost.rsp_distance) | Randomised-shortest-path distances between points: gdistance's ``rSPDistance()`` |
| [`tf.cost.path_inc`](cost.md#terraflux.cost.path_inc) | Overlap and non-overlap of the routes from a common origin to each location: gdistance's ``pathInc()`` |
| [`tf.cost.overlap`](cost.md#terraflux.cost.overlap) | gdistance's ``overlap``: the flow two routes share on each connection, ``min(a, b)`` |
| [`tf.cost.nonoverlap`](cost.md#terraflux.cost.nonoverlap) | gdistance's ``nonoverlap``: ``max(max(a, b) * (1 - min(a, b)) - min(a, b), 0)`` |
| [`tf.cost.barriers`](cost.md#terraflux.cost.barriers) | Barrier transitions: gdistance's ``transition(x, "barriers", directions)`` |
| [`tf.cost.areas`](cost.md#terraflux.cost.areas) | Area transitions: gdistance's ``transition(x, "areas", directions)`` |
| [`tf.cost.example_data`](cost.md#terraflux.cost.example_data) | Path of an example data set for the grid distances |

### Distance & restoration

| Name | Summary |
|---|---|
| [`tf.euclidean_distance`](distance.md#terraflux.euclidean_distance) | Euclidean distance to the habitat edge, inside and outside the habitat (GTB's ``batch_eucldist``) |
| [`tf.EuclideanDistanceResult`](distance.md#terraflux.EuclideanDistanceResult) | Result of `euclidean_distance` |
| [`tf.distance_to_resistance`](distance.md#terraflux.distance_to_resistance) | Resistance growing with the distance to the foreground (GTB Setup Tools -> Distance -> Resistance) |
| [`tf.restoration_status`](distance.md#terraflux.restoration_status) | How connected the habitat is and how much reconnecting it could add (GTB's Restoration Status, ``batch_rss``) |
| [`tf.RestorationStatusResult`](distance.md#terraflux.RestorationStatusResult) | Result of `restoration_status` |
| [`tf.cost_distance`](distance.md#terraflux.cost_distance) | Accumulated cost of travelling from the source cells across a resistance surface (8 neighbours) |
| [`tf.least_cost_path`](distance.md#terraflux.least_cost_path) | The least-cost path between two areas A and B |
| [`tf.LeastCostPathResult`](distance.md#terraflux.LeastCostPathResult) | Result of `least_cost_path` |
| [`tf.restoration_corridor`](distance.md#terraflux.restoration_corridor) | Connectivity gain of restoring the least-cost corridor between two areas (GTB's Restoration Planner) |
| [`tf.RestorationCorridorResult`](distance.md#terraflux.RestorationCorridorResult) | Result of `restoration_corridor` |
| [`tf.isochrone_map`](distance.md#terraflux.isochrone_map) | Accumulated cost from object A (GTB Restoration Planner -> Isochrone Map A) |
| [`tf.isochrone_map_ab`](distance.md#terraflux.isochrone_map_ab) | Cheapest route between objects A and B (GTB Restoration Planner -> Isochrone Map AB) |
| [`tf.IsochroneResult`](distance.md#terraflux.IsochroneResult) | Result of `isochrone_map` / `isochrone_map_ab` |
| [`tf.influence_zones`](distance.md#terraflux.influence_zones) | Zones of influence of the foreground objects (GTB Distance -> Influence Zones) |
| [`tf.InfluenceZonesResult`](distance.md#terraflux.InfluenceZonesResult) | Result of `influence_zones` |
| [`tf.proximity`](distance.md#terraflux.proximity) | Proximity between neighbouring objects (GTB Distance -> Proximity) |
| [`tf.ProximityResult`](distance.md#terraflux.ProximityResult) | Result of `proximity` |
| [`tf.original_vegetation`](distance.md#terraflux.original_vegetation) | The original vegetation of converted land, and what restoration brings back |
| [`tf.OriginalVegetation`](distance.md#terraflux.OriginalVegetation) | Result of `original_vegetation` |
| [`terraflux.intactness`](distance.md#terraflux.intactness) | Ecological intactness from the human footprint (Beyer et al. 2020) |
| [`tf.intactness_gradient`](distance.md#terraflux.intactness_gradient) | What restoring each cell adds to the total ecological intactness |

### Change detection

| Name | Summary |
|---|---|
| [`tf.simple_change`](change.md#terraflux.simple_change) | Pixel-by-pixel change of a habitat map between two dates |
| [`tf.SimpleChangeResult`](change.md#terraflux.SimpleChangeResult) | Result of `simple_change` |
| [`tf.morphological_change`](change.md#terraflux.morphological_change) | Change of a habitat map that separates real structural change from edge noise (GTB's Morphological Change Detection) |
| [`tf.MorphologicalChangeResult`](change.md#terraflux.MorphologicalChangeResult) | Result of `morphological_change` |
| [`tf.fos_change`](change.md#terraflux.fos_change) | Change of fragmentation between two dates (GTB's FOS change) |
| [`tf.FOSChangeResult`](change.md#terraflux.FOSChangeResult) | Result of `fos_change` |
| [`tf.fad_multiscale_change`](change.md#terraflux.fad_multiscale_change) | Fragmentation class change between two FAD-MS analyses (GTB Change -> FAD Multiscale) |
| [`tf.FADMultiscaleChangeResult`](change.md#terraflux.FADMultiscaleChangeResult) | Result of `fad_multiscale_change` |
| [`tf.FADChangeScale`](change.md#terraflux.FADChangeScale) | FAD change A -> B at one observation scale (one GTB ``FAD_change_<scale>`` table) |
| [`tf.landscape_mosaic_change`](change.md#terraflux.landscape_mosaic_change) | Change of the Landscape Mosaic heatmap from A to B (GTB Change -> Landscape Mosaic Heatmap) |
| [`tf.LMHeatmapChange`](change.md#terraflux.LMHeatmapChange) | Result of `landscape_mosaic_change`: the heatmap difference B - A (GTB's ``delta_heatmap.csv``) |

### Large data & batch

| Name | Summary |
|---|---|
| [`terraflux.batch`](batch.md#terraflux.batch) | Parallel processing of many rasters, one thread per raster |
| [`tf.mspa_batch`](batch.md#terraflux.mspa_batch) | Run MSPA on many rasters in parallel and write the results |
| [`tf.BatchResult`](batch.md#terraflux.BatchResult) | Outcome of one input of `batch` / `mspa_batch` |
| [`tf.batch_status`](batch.md#terraflux.batch_status) | What the runs of `batch` / `mspa_batch` did in an output folder: the inputs done, failed (with their error) and not done, and the job array tasks to submit again |
| [`terraflux.batch.Stopped`](batch.md#terraflux.batch.Stopped) | A batch stopped by SIGTERM (a scheduler's warning before the job's time limit, or its cancellation): no input started after it; ``results`` holds those that ended, in input order (``None`` for the inputs not processed) |
| [`tf.helpers`](batch.md#terraflux.parallel.helpers) | This computer and the helpers of its group, for ``on=`` (see `Helpers`) |
| [`tf.Helpers`](batch.md#terraflux.parallel.Helpers) | This computer and the helpers of its group, for the ``on=`` argument of terraflux's functions that run many independent tasks |
| [`tf.parallel.map`](batch.md#terraflux.parallel.map) | Run a tool on many inputs, on this computer or on helpers, in parallel |
| [`tf.parallel.tiled`](batch.md#terraflux.parallel.tiled) | Run a neighbourhood tool on a large raster tile by tile, on this computer or on helpers |
| [`tf.parallel.call`](batch.md#terraflux.parallel.call) | A registered tool with its arguments, the drawn ones marked by `param` |
| [`tf.parallel.param`](batch.md#terraflux.parallel.param) | The ``index``-th drawn parameter, in the arguments of a `call` |
| [`tf.parallel.run_worker`](batch.md#terraflux.parallel.run_worker) | Lend this computer to the group: wait for tasks and run them, until Ctrl+C |
| [`tf.parallel.invite_helpers`](batch.md#terraflux.parallel.invite_helpers) | Show an invitation code and let other computers join this one's group with ``terraflux-py worker --join <code>``, for ``minutes`` |
| [`tf.parallel.list_helpers`](batch.md#terraflux.parallel.list_helpers) | The helpers of this computer's group waiting for work now (busy ones are not looking): ``[{"name", "address", "cores", "version"}]`` |
| [`tf.parallel.leave_group`](batch.md#terraflux.parallel.leave_group) | Forget this computer's group (it no longer helps nor uses helpers) |
| [`tf.parallel.open_firewall`](batch.md#terraflux.parallel.open_firewall) | Let helpers reach this computer: the Windows firewall rules for the local network |
| [`tf.mspa_tiled`](batch.md#terraflux.mspa_tiled) | MSPA of a large map by overlapping tiles, identical to GTB's MSPA Tiling |
| [`tf.tile_layout`](batch.md#terraflux.tile_layout) | GTB's tiles along one axis of ``size`` pixels |

### Neutral landscape models

| Name | Summary |
|---|---|
| [`tf.nlm.random`](nlm.md#terraflux.nlm.random) | Spatially random neutral landscape model (NLMR's ``nlm_random``) |
| [`tf.nlm.percolation`](nlm.md#terraflux.nlm.percolation) | Percolation neutral landscape model (NLMR's ``nlm_percolation``) |
| [`tf.nlm.planar_gradient`](nlm.md#terraflux.nlm.planar_gradient) | Planar gradient neutral landscape model (NLMR's ``nlm_planargradient``) |
| [`tf.nlm.edge_gradient`](nlm.md#terraflux.nlm.edge_gradient) | Edge gradient neutral landscape model (NLMR's ``nlm_edgegradient``) |
| [`tf.nlm.distance_gradient`](nlm.md#terraflux.nlm.distance_gradient) | Distance gradient neutral landscape model (NLMR's ``nlm_distancegradient``) |
| [`tf.nlm.mpd`](nlm.md#terraflux.nlm.mpd) | Midpoint displacement neutral landscape model (NLMR's ``nlm_mpd``) |
| [`tf.nlm.fbm`](nlm.md#terraflux.nlm.fbm) | Fractional Brownian motion neutral landscape model (NLMR's ``nlm_fbm``) |
| [`tf.nlm.gaussian_field`](nlm.md#terraflux.nlm.gaussian_field) | Gaussian random field neutral landscape model (NLMR's ``nlm_gaussianfield``) |
| [`tf.nlm.mosaic_field`](nlm.md#terraflux.nlm.mosaic_field) | Mosaic random field neutral landscape model (NLMR's ``nlm_mosaicfield``, Schlather 1999) |
| [`tf.nlm.mosaic_tess`](nlm.md#terraflux.nlm.mosaic_tess) | Polygonal (Voronoi) mosaic neutral landscape model (NLMR's ``nlm_mosaictess``, Gaucherel 2008) |
| [`tf.nlm.mosaic_gibbs`](nlm.md#terraflux.nlm.mosaic_gibbs) | Gibbs-process Voronoi mosaic (NLMR's ``nlm_mosaicgibbs``, Gauchel et al. 2008) |
| [`tf.nlm.random_cluster`](nlm.md#terraflux.nlm.random_cluster) | Random cluster neutral landscape model (NLMR's ``nlm_randomcluster``, Saura & Martínez-Millán 2000) |
| [`tf.nlm.neigh`](nlm.md#terraflux.nlm.neigh) | Neighbourhood-based clusters (NLMR's ``nlm_neigh``, Scherer et al. 2016) |
| [`tf.nlm.random_rectangular_cluster`](nlm.md#terraflux.nlm.random_rectangular_cluster) | Random rectangular clusters (NLMR's ``nlm_randomrectangularcluster``, Gaucherel 2008) |
| [`tf.nlm.curds`](nlm.md#terraflux.nlm.curds) | Hierarchical curdling (NLMR's ``nlm_curds``, Keitt 2000) |
| [`tf.nlm.classify`](nlm.md#terraflux.nlm.classify) | Classify a continuous landscape into categories (NLMR's ``landscape_classify``) |
| [`tf.nlm.binarize`](nlm.md#terraflux.nlm.binarize) | Habitat / matrix maps at given habitat shares (landscapetools' ``util_binarize``) |
| [`tf.nlm.rescale`](nlm.md#terraflux.nlm.rescale) | Linearly rescale values to 0-1 (NLMR's ``util_rescale``) |
| [`tf.nlm.merge`](nlm.md#terraflux.nlm.merge) | Merge neutral landscapes (landscapetools' ``util_merge``) |

### Biodiversity analysis

| Name | Summary |
|---|---|
| [`tf.bio.BaseData`](biodiversity.md#terraflux.bio.BaseData) | Groups × labels sample counts with the group coordinates |
| [`tf.bio.snap_coordinates`](biodiversity.md#terraflux.bio.snap_coordinates) | Cell centres as Biodiverse's text import computes them (before naming) |
| [`tf.bio.example_data`](biodiversity.md#terraflux.bio.example_data) | Path of Biodiverse's example data (GPL-3, Laffan et al.) |
| [`tf.bio.read_occurrences`](biodiversity.md#terraflux.bio.read_occurrences) | Occurrence records of a GBIF download or a Darwin Core Archive, cleaned |
| [`tf.bio.neighbours`](biodiversity.md#terraflux.bio.neighbours) | Neighbour sets of every group, as Biodiverse's spatial analysis finds them |
| [`tf.bio.definition_query`](biodiversity.md#terraflux.bio.definition_query) | Boolean mask of the groups passing a definition query |
| [`tf.bio.SpatialCondition`](biodiversity.md#terraflux.bio.SpatialCondition) | A Biodiverse spatial condition (Perl syntax) compiled for a BaseData |
| [`tf.bio.Neighbourhoods`](biodiversity.md#terraflux.bio.Neighbourhoods) | Neighbour sets of the processing groups |
| [`tf.bio.spatial_analysis`](biodiversity.md#terraflux.bio.spatial_analysis) | Biodiverse's spatial analysis: indices over the neighbourhood of every group |
| [`tf.bio.list_calculations`](biodiversity.md#terraflux.bio.list_calculations) | The calculations available to `spatial_analysis` |
| [`tf.bio.SpatialResult`](biodiversity.md#terraflux.bio.SpatialResult) | Results of `spatial_analysis` |
| [`tf.bio.randomise`](biodiversity.md#terraflux.bio.randomise) | Randomise the basedata of a spatial analysis and compare the results |
| [`tf.bio.Randomisation`](biodiversity.md#terraflux.bio.Randomisation) | Randomisations of a `SpatialResult` (see `randomise`) |
| [`tf.bio.cluster`](biodiversity.md#terraflux.bio.cluster) | Cluster the groups by the dissimilarity of their labels (Biodiverse's cluster analysis) |
| [`tf.bio.region_grower`](biodiversity.md#terraflux.bio.region_grower) | Grow regions by joining the pairs with the highest value of an index (Biodiverse's RegionGrower) |
| [`tf.bio.ClusterResult`](biodiversity.md#terraflux.bio.ClusterResult) | Results of `cluster` |
| [`tf.bio.Tree`](biodiversity.md#terraflux.bio.Tree) | A rooted phylogenetic tree |
| [`tf.bio.read_tree`](biodiversity.md#terraflux.bio.read_tree) | Read one tree (the first, or the one with index/name ``index``) |
| [`tf.bio.read_trees`](biodiversity.md#terraflux.bio.read_trees) | Read every tree in a Newick, Phylip (several Newicks) or Nexus file |
| [`tf.bio.Matrix`](biodiversity.md#terraflux.bio.Matrix) | A symmetric matrix of values between elements (usually labels) |
| [`tf.bio.read_matrix`](biodiversity.md#terraflux.bio.read_matrix) | Read a Biodiverse matrix file (full or lower/upper triangle, or sparse pairs) |
| [`tf.bio.hex_centres`](biodiversity.md#terraflux.bio.hex_centres) | Centres of the hexagons (circumradius ``size``) containing each point |
| [`tf.bio.h3_cells`](biodiversity.md#terraflux.bio.h3_cells) | H3 cell of each point (``h3`` package) |
| [`tf.bio.save_session`](biodiversity.md#terraflux.bio.save_session) | Save basedatas, trees, matrices and analysis results in one file |
| [`tf.bio.load_session`](biodiversity.md#terraflux.bio.load_session) | Load the objects saved with `save_session` (a dict by name) |
| [`tf.bio.read_properties`](biodiversity.md#terraflux.bio.read_properties) | Read a Biodiverse element properties file (label or group properties) |

### Spatial prioritization

| Name | Summary |
|---|---|
| [`tf.prioritize.zonation`](prioritization.md#terraflux.prioritize.zonation) | Rank the cells of a landscape for conservation, as Zonation 5 does |
| [`tf.prioritize.ZonationResult`](prioritization.md#terraflux.prioritize.ZonationResult) | The result of `zonation` |
| [`tf.prioritize.run_project`](prioritization.md#terraflux.prioritize.run_project) | Run a Zonation 5 project (see `read_project`) |
| [`tf.prioritize.read_project`](prioritization.md#terraflux.prioritize.read_project) | The `zonation` arguments of a Zonation 5 project |
| [`tf.prioritize.Kernel`](prioritization.md#terraflux.prioritize.Kernel) | A dispersal kernel of Zonation 5's connectivity transforms |
| [`tf.prioritize.Interaction`](prioritization.md#terraflux.prioritize.Interaction) | An interaction connectivity transform (Zonation's ``-i``/``-I``): a feature weighted by the connectivity of a resource (``positive``) or reduced by it |
| [`tf.prioritize.convolve`](prioritization.md#terraflux.prioritize.convolve) | Convolve a raster with a kernel normalized to sum 1, as Zonation 5 does |
| [`tf.prioritize.AreaStatistics`](prioritization.md#terraflux.prioritize.AreaStatistics) | Statistics of areas of a ranking (`area_statistics`) |
| [`tf.prioritize.ImpactScenario`](prioritization.md#terraflux.prioritize.ImpactScenario) | An impact assessment scenario (Zonation's ``-s``): land-use change at sources |
| [`tf.prioritize.ImpactSource`](prioritization.md#terraflux.prioritize.ImpactSource) | An impact source of an impact group (a line of Zonation's impact buffers file) |
| [`tf.prioritize.CumulativeScenario`](prioritization.md#terraflux.prioritize.CumulativeScenario) | A cumulative impact assessment scenario (Zonation's ``-e``) |
| [`tf.prioritize.Pressure`](prioritization.md#terraflux.prioritize.Pressure) | A pressure raster of a cumulative impact scenario |
| [`tf.prioritize.problem`](prioritization.md#terraflux.prioritize.problem) | A conservation planning problem (prioritizr's ``problem()``) |
| [`tf.prioritize.Problem`](prioritization.md#terraflux.prioritize.Problem) | A conservation planning problem (prioritizr's ``problem()``): see `problem` |
| [`tf.prioritize.Solution`](prioritization.md#terraflux.prioritize.Solution) | A solution of a `Problem` |
| [`tf.prioritize.Portfolio`](prioritization.md#terraflux.prioritize.Portfolio) | Several solutions of a problem (`add_cuts_portfolio` and the others) |
| [`tf.prioritize.OptimizationProblem`](prioritization.md#terraflux.prioritize.OptimizationProblem) | A compiled model: ``modelsense`` of ``obj @ x`` subject to ``A x (sense) rhs``, ``lb <= x <= ub`` and ``x`` of type ``vtype`` (``"B"`` binary, ``"C"`` continuous, ``"I"`` integer); the layout of prioritizr's ``compile()`` |
| [`tf.prioritize.h3_units`](prioritization.md#terraflux.prioritize.h3_units) | H3 hexagons as planning units: the hexagons of an H3 ``resolution`` that cover a cost raster, with their cost |
| [`tf.prioritize.example_data`](prioritization.md#terraflux.prioritize.example_data) | Path of an example data set for prioritization: **Serra Verde**, an imaginary coastal region of 60 x 90 cells of 1 km (SIRGAS 2000 / UTM 23S) |
| [`tf.prioritize.category_layer`](prioritization.md#terraflux.prioritize.category_layer) | The zone of every cell of a stack of zone layers (prioritizr's ``category_layer()``): the number (1, 2, ...) of the layer with the largest value above 0, 0 where none is, NaN where every layer is NaN. ``stack``: a raster with a layer per zone (a solution's ``selection``) or a mapping zone -> raster |
| [`tf.prioritize.binary_stack`](prioritization.md#terraflux.prioritize.binary_stack) | One 0 / 1 raster per zone from a raster of zone numbers (prioritizr's ``binary_stack()``): a mapping zone -> raster, ready for `add_locked_in_constraints` (for example an existing zoning plan). ``zones``: the zones' names in the order of their numbers (default: those of a solution's ``zones`` map, else ``"1"``, ``"2"``, ...); cells with 0 or NaN are in no zone |
| [`tf.prioritize.boundary_matrix`](prioritization.md#terraflux.prioritize.boundary_matrix) | The length of the boundary every two units share, and every unit's perimeter on the diagonal (prioritizr's ``boundary_matrix()``): raster cells with data share ``xres`` with the cells above and below and ``yres`` with those left and right, perimeter ``2 (xres + yres)``; polygons share the length of their common paths, perimeters their own, rounded to 8 decimals |
| [`tf.prioritize.adjacency_matrix`](prioritization.md#terraflux.prioritize.adjacency_matrix) | 1 for every two adjacent units (prioritizr's ``adjacency_matrix()``): raster cells with data that share an edge (``directions`` 4) or also a corner (8); polygons that intersect, touching at a corner included |
| [`tf.prioritize.connectivity_matrix`](prioritization.md#terraflux.prioritize.connectivity_matrix) | Connectivity between adjacent units (prioritizr's ``connectivity_matrix()``): the length of their shared boundary times the mean of ``y`` in the two (``y``: a raster on the units' grid; with polygons a raster summed over each, or a column name) |
| [`tf.prioritize.proximity_matrix`](prioritization.md#terraflux.prioritize.proximity_matrix) | 1 for every two units within ``distance`` of each other (prioritizr's ``proximity_matrix()``): raster cell centres, or polygons by their nearest points |
| [`tf.prioritize.decay_connectivity`](prioritization.md#terraflux.prioritize.decay_connectivity) | Connectivity from distances between units as a dispersal probability (Conefor's negative exponential): ``exp(-ln(2) d / median)``, 0.5 at the ``median`` dispersal distance, dropped below ``min_probability`` (sparse), 0 on the diagonal and for unreachable pairs (``inf``, NaN) |
| [`tf.prioritize.unit_points`](prioritization.md#terraflux.prioritize.unit_points) | A point per unit, (n, 2) coordinates: raster cell centres (every cell), or a point inside every polygon (shapely's point on surface), to measure distances between units with `cost_distance` or `commute_distance` |
| [`tf.prioritize.eval_n_summary`](prioritization.md#terraflux.prioritize.eval_n_summary) | The number of selected planning units (the sum of the solution's values) |
| [`tf.prioritize.eval_cost_summary`](prioritization.md#terraflux.prioritize.eval_cost_summary) | The solution's total cost |
| [`tf.prioritize.eval_boundary_summary`](prioritization.md#terraflux.prioritize.eval_boundary_summary) | The solution's boundary length: the edges between selected and other units, plus ``edge_factor`` x the selected units' edges on the outside of the planning units |
| [`tf.prioritize.eval_connectivity_summary`](prioritization.md#terraflux.prioritize.eval_connectivity_summary) | The connectivity among the selected units: the sum of ``data[i, j]`` x the solution's values of units i and j (a symmetric matrix over the units) |
| [`tf.prioritize.eval_asym_connectivity_summary`](prioritization.md#terraflux.prioritize.eval_asym_connectivity_summary) | As `eval_connectivity_summary` for an asymmetric matrix (its diagonal counted once) |
| [`tf.prioritize.eval_feature_representation_summary`](prioritization.md#terraflux.prioritize.eval_feature_representation_summary) | Every feature's total amount and the amount the solution holds, absolute and relative |
| [`tf.prioritize.eval_target_coverage_summary`](prioritization.md#terraflux.prioritize.eval_target_coverage_summary) | Every target's feature, whether it is met, the feature's total, and the target, the amount held and the shortfall, absolute and relative (shares of the total; 0 when the total is 0) |
| [`tf.prioritize.eval_replacement_importance`](prioritization.md#terraflux.prioritize.eval_replacement_importance) | The replacement cost of every selected unit (Cabeza & Moilanen 2006): how much the objective worsens when the unit is left out and the problem solved again with the other selected units kept (infinite when nothing can replace it), rescaled to 0.01-1 |
| [`tf.prioritize.eval_ferrier_importance`](prioritization.md#terraflux.prioritize.eval_ferrier_importance) | Ferrier et al. (2000)'s irreplaceability of the selected units for every feature and in total (prioritizr's version: the problem needs one ">=" target per feature, amounts >= 0) |
| [`tf.prioritize.eval_rare_richness_importance`](prioritization.md#terraflux.prioritize.eval_rare_richness_importance) | The rarity-weighted richness of the selected units (Albuquerque & Beier 2015): the sum over the features of the unit's amount / the feature's largest amount / its total, rescaled to 0.01-1 |
| [`tf.prioritize.eval_rank_importance`](prioritization.md#terraflux.prioritize.eval_rank_importance) | Rank importance (Jung et al. 2021, prioritizr's eval_rank_importance): the problem is solved with increasing budgets (the min set objective becomes min shortfall), only the solution's units allowed and each step keeping the units of the step before; a unit's rank is how early it enters (n = the first budget, ..., 1 = the last), rescaled to 0.01-1. ``n`` budgets evenly spaced between the locked-in units' cost and the solution's, or the ``budgets`` themselves |
| [`tf.prioritize.eval_extinction_risk`](prioritization.md#terraflux.prioritize.eval_extinction_risk) | Every feature's extinction risk by the species-area relationship, 1 - (held / total)^z (z = 0.25 by default, one value or one per feature), and their sum (the expected number of features lost) |
| [`tf.prioritize.eval_landscape_metrics`](prioritization.md#terraflux.prioritize.eval_landscape_metrics) | Landscape metrics of a raster solution (`landscape_metrics` of the map of the selected cells: class 1 selected, 0 not, outside the planning units no data): e.g. the number, sizes and shapes of the selected patches |
| [`tf.prioritize.eval_mspa`](prioritization.md#terraflux.prioritize.eval_mspa) | MSPA (`mspa`) of a raster solution: the selected cells as foreground, the other planning units as background, the cells outside them missing |
| [`tf.prioritize.eval_landscape_identification`](prioritization.md#terraflux.prioritize.eval_landscape_identification) | Landscape identification: the separate areas of the top fraction of a priority surface, and what each holds (Zonation's area statistics of separate areas, over any surface) |
| [`tf.prioritize.eval_landscape_comparison`](prioritization.md#terraflux.prioritize.eval_landscape_comparison) | Landscape comparison: how much the top fractions of two priority surfaces overlap (Zonation's solution comparison at several fractions, prioripy's ``eval_landscape_comparison``) |
| [`tf.prioritize.position_rank`](prioritization.md#terraflux.prioritize.position_rank) | Zonation's ranks of a surface by position: the i-th lowest of the N finite cells gets float32(i * (1 / N)); ties go to the earlier cell in row order first; NaN stays NaN |
| [`tf.prioritize.LandscapeAreas`](prioritization.md#terraflux.prioritize.LandscapeAreas) | The areas of the top fraction of a priority surface (`eval_landscape_identification`) |
| [`tf.prioritize.LandscapeComparison`](prioritization.md#terraflux.prioritize.LandscapeComparison) | Two priority surfaces compared at several top fractions (`eval_landscape_comparison`) |
| [`tf.prioritize.eval_proxirr`](prioritization.md#terraflux.prioritize.eval_proxirr) | ProxIRR (Baisero et al. 2022) of every planning unit for the problem's targets, as the CRAN package proxirr computes it |
| [`tf.prioritize.eval_proxirr_marxan`](prioritization.md#terraflux.prioritize.eval_proxirr_marxan) | ProxIRR of a Marxan problem's planning units (proxirr's ``marxan_alphas``, ``marxan_betas`` and ``marxan_gammas``), from the tables `marxan` takes |
| [`tf.prioritize.ProxIRR`](prioritization.md#terraflux.prioritize.ProxIRR) | ProxIRR of every planning unit (`eval_proxirr`, `eval_proxirr_marxan`) |
| [`tf.prioritize.calibrate_penalty`](prioritization.md#terraflux.prioritize.calibrate_penalty) | The blended approach: the problem solved with a range of penalty weights (``values``; default ``n`` values evenly spaced in log scale over 1/100 to 100 times the Cohon et al. penalty), every candidate's cost and criterion (boundary length, connectivity, ...), and the candidates TOPSIS and the knee of the curve choose. ``p``: the problem without the penalty |
| [`tf.prioritize.calibrate_budget`](prioritization.md#terraflux.prioritize.calibrate_budget) | The hierarchical approach: the least cost solution first (``p``, a min set problem without the penalty, solved to optimality), then the least penalty within cost thresholds above its cost (``thresholds``, or the cost x (1 + ``inflation``); default inflation 1e-5 to 4 in 9 steps, the thresholds rounded up as in prioritizr's tutorial) |
| [`tf.prioritize.cohon_penalty`](prioritization.md#terraflux.prioritize.cohon_penalty) | The penalty of Cohon et al. (1979): the trade-off rate between the solution with the least cost (``p`` as given, a min set problem without the penalty) and the one with the least penalty (min penalties within a budget above every unit's cost): \|cost difference / criterion difference\|, rounded to 5 decimals |
| [`tf.prioritize.pareto_front`](prioritization.md#terraflux.prioritize.pareto_front) | The trade-off between cost and representation: the least total shortfall of the targets (min shortfall objective: the sum of the shortfalls as shares of their targets) within ``n`` budgets evenly spaced up to the least-cost solution's cost (above the cost of the locked-in units, which every budget includes), or the ``budgets`` given. ``p``: a problem with targets (its objective is replaced) |
| [`tf.prioritize.calibrate`](prioritization.md#terraflux.prioritize.calibrate) | Any trade-off parameter: ``make(value)`` gives the problem for every value (a feature's weight, a target, a penalty of any kind...), ``criterion(problem, solution)`` the quantity traded off against the cost (smaller is better) |
| [`tf.prioritize.TradeOffs`](prioritization.md#terraflux.prioritize.TradeOffs) | Candidate solutions over a trade-off parameter |
| [`tf.prioritize.topsis`](prioritization.md#terraflux.prioritize.topsis) | TOPSIS (as R's ``topsis`` package): the alternatives (rows of ``decision``) scored by their closeness to the ideal (``impacts``: "+" larger is better, "-" smaller is better, per column; equal ``weights`` by default) |
| [`tf.prioritize.knee_point`](prioritization.md#terraflux.prioritize.knee_point) | The knee (elbow) of a trade-off curve: the point farthest from the straight line between its ends, both axes scaled to 0-1 (the points in the order of x) |
| [`tf.prioritize.mean_sd`](prioritization.md#terraflux.prioritize.mean_sd) | plangea's choice of weights (``post_choose_weights``): every criterion (a column of ``decision``) rescaled to 0-1 over the alternatives (a constant one to 1), the "-" ones (smaller is better) inverted, and every alternative scored by the mean of its criteria divided by their standard deviation (good on all of them at once) |
| [`tf.prioritize.iterative_solve`](prioritization.md#terraflux.prioritize.iterative_solve) | Solve, change the problem, solve again: ``update(step, solution, problem)`` gives the next problem (for instance from the landscape a ``tf.simulate`` step leaves, with the solution's cells protected by `protected_cells` in the simulation's rules), or ``None`` to stop |
| [`tf.prioritize.protected_cells`](prioritization.md#terraflux.prioritize.protected_cells) | The selected cells of a raster solution as a boolean raster (for ``tf.simulate``'s restrictions) |
| [`tf.prioritize.zonation_features`](prioritization.md#terraflux.prioritize.zonation_features) | The features as Zonation 5 transforms them before ranking (condition, connectivity kernels, interactions, matrix connectivity, retention, impacts: see `zonation` for each), back in the features' own units, for an optimization problem (``problem(cost, zonation_features(...))``) |
| [`tf.prioritize.compare_zonation`](prioritization.md#terraflux.prioritize.compare_zonation) | Compare an optimization solution of ``p`` (solved if not given) with a Zonation ranking of the same features and costs (run with ``rule`` and ``zonation_kw`` if not given: a `ZonationResult` or a rank raster) cut at the solution's cost |
| [`tf.prioritize.ZonationComparison`](prioritization.md#terraflux.prioritize.ZonationComparison) | A Zonation ranking and an optimization solution with the same budget |
| [`tf.prioritize.sdm_features`](prioritization.md#terraflux.prioritize.sdm_features) | Features from species distribution models (``tf.sdm``): every result's prediction on ``env`` (its ensemble when it has several models, else its model), as a stack for `problem` |
| [`tf.prioritize.report`](prioritization.md#terraflux.prioritize.report) | Solve ``p`` (unless a solution is given) and evaluate the solution: the number of units, cost, boundary (raster and polygon units), the features' representation and the targets' coverage; with ``importance`` the Ferrier irreplaceability of the selected units (problems with one target per feature) or else their rarity-weighted richness; with ``extinction_z`` the species-area extinction risk; with ``zonation`` the comparison with a Zonation ranking at the same cost (raster units) |
| [`tf.prioritize.PrioritizationReport`](prioritization.md#terraflux.prioritize.PrioritizationReport) | A solution and its evaluation in one object: ``tf.plot(report)`` draws it, ``tf.plot(report).save( "report.html")`` writes it as a page that works offline, ``tf.save(report, folder)`` its tables and maps |
| [`tf.prioritize.coarsen`](prioritization.md#terraflux.prioritize.coarsen) | A problem's rasters on blocks of ``factor`` x ``factor`` cells (for continental problems: solve the coarse problem, then `refine` the selection): every block's cost and amounts are the sums of its cells' with a cost (the fine planning units; a block without any has none), so the refined selection holds what the coarse one does |
| [`tf.prioritize.refine`](prioritization.md#terraflux.prioritize.refine) | A selection on coarse blocks (a `Solution` of a `coarsen`-ed problem, or its selection raster) back on the fine grid of ``like`` (the fine cost raster): every cell takes its block's value, NaN where ``like`` has none |
| [`tf.prioritize.actions_data`](prioritization.md#terraflux.prioritize.actions_data) | The inputs of a multi-action problem, checked and completed as prioriactions's ``inputData()`` |
| [`tf.prioritize.actions_raster_data`](prioritization.md#terraflux.prioritize.actions_raster_data) | The inputs of a multi-action problem from rasters (see `actions_data` for the tables) |
| [`tf.prioritize.ActionsData`](prioritization.md#terraflux.prioritize.ActionsData) | Checked inputs of a multi-action problem (prioriactions's ``inputData()``): see `actions_data` |
| [`tf.prioritize.ActionsProblem`](prioritization.md#terraflux.prioritize.ActionsProblem) | A multi-action problem (prioriactions's ``problem()``): see `problem` |
| [`tf.prioritize.ActionsSolution`](prioritization.md#terraflux.prioritize.ActionsSolution) | A solution of an `ActionsProblem` |
| [`tf.prioritize.ActionsPortfolio`](prioritization.md#terraflux.prioritize.ActionsPortfolio) | Several solutions of an `ActionsProblem` (`add_cuts_portfolio` and the others) |
| [`tf.prioritize.restoration_problem`](prioritization.md#terraflux.prioritize.restoration_problem) | A restoration problem: how much of every converted cell to restore (and of natural land to conserve or convert) |
| [`tf.prioritize.RestorationProblem`](prioritization.md#terraflux.prioritize.RestorationProblem) | A restoration problem (see `restoration_problem`), built up by its ``add_*`` methods; every method returns a new problem |
| [`tf.prioritize.RestorationSolution`](prioritization.md#terraflux.prioritize.RestorationSolution) | A restoration plan |
| [`tf.prioritize.RestorationTradeOff`](prioritization.md#terraflux.prioritize.RestorationTradeOff) | Plans over a trade-off (`tradeoff`) or over sets of weights (`compare_weights`) |
| [`tf.prioritize.ecosystem_areas`](prioritization.md#terraflux.prioritize.ecosystem_areas) | The natural area of every ecosystem (ecoregion) now and before conversion, and what restoring a cell gives it: the input of `add_extinction_risk_benefit` for the risk of ecosystem collapse (plangea's ``ec``) |
| [`tf.prioritize.ecosystem_targets`](prioritization.md#terraflux.prioritize.ecosystem_targets) | Share a restoration target among ecosystems for the least risk of collapse (plangea's ecosystemic sub-region targets, ``harmonize_sr_ecosys``) |
| [`tf.prioritize.eval_risk_reduction`](prioritization.md#terraflux.prioritize.eval_risk_reduction) | The reduction of the extinction risk a plan brings (plangea's ``multi_exrisk_aggregate``), with the uncertainty of the species-area exponent z |
| [`tf.prioritize.RiskReduction`](prioritization.md#terraflux.prioritize.RiskReduction) | Result of `eval_risk_reduction`: per species, its habitat now (``current``), without the plan (``baseline``: less than now when conservation's baseline loses land) and gained by it, its risk before and after; the total reduction (negative when conversion raises the risk) and its standard deviation from the uncertainty of z |
| [`tf.prioritize.marxan`](prioritization.md#terraflux.prioritize.marxan) | Run Marxan: ``num_reps`` independent runs of annealing and iterative improvement (or the greedy heuristics) minimising cost + ``blm`` x boundary + penalties for the features' shortfalls |
| [`tf.prioritize.MarxanResult`](prioritization.md#terraflux.prioritize.MarxanResult) | The runs of `marxan` |

### Species distributions

| Name | Summary |
|---|---|
| [`tf.sdm.prepare`](sdm.md#terraflux.sdm.prepare) | Prepare occurrence records and predictors for a species distribution model, in one call |
| [`tf.sdm.download_gbif`](sdm.md#terraflux.sdm.download_gbif) | Occurrence records of a species from GBIF, with coordinates |
| [`tf.sdm.clean_occurrences`](sdm.md#terraflux.sdm.clean_occurrences) | Flag or drop occurrence records with suspicious coordinates |
| [`tf.sdm.reference_data`](sdm.md#terraflux.sdm.reference_data) | A geographic reference data set for cleaning occurrence records |
| [`tf.sdm.one_per_cell`](sdm.md#terraflux.sdm.one_per_cell) | Keep one occurrence record per cell of the predictor grid |
| [`tf.sdm.thin`](sdm.md#terraflux.sdm.thin) | Thin occurrence records so that no two kept records are closer than ``distance`` |
| [`tf.sdm.accessible_area`](sdm.md#terraflux.sdm.accessible_area) | The area the species could have reached: where its background is drawn |
| [`tf.sdm.select_predictors`](sdm.md#terraflux.sdm.select_predictors) | Drop predictors that the others already explain (collinearity) |
| [`tf.sdm.vif`](sdm.md#terraflux.sdm.vif) | Variance inflation factor of each predictor: how well the others predict it |
| [`tf.sdm.background`](sdm.md#terraflux.sdm.background) | Background points: a sample of the conditions available to the species |
| [`tf.sdm.bias_surface`](sdm.md#terraflux.sdm.bias_surface) | Sampling effort as a surface: a Gaussian kernel density of the records of a group |
| [`tf.sdm.dataset`](sdm.md#terraflux.sdm.dataset) | Put presences, background points and predictor values together |
| [`tf.sdm.SDMData`](sdm.md#terraflux.sdm.SDMData) | Presences and background (or absences) with their predictor values, ready for a model |
| [`tf.sdm.spatial_blocks`](sdm.md#terraflux.sdm.spatial_blocks) | Folds made of spatial blocks: no fold is a neighbour of the records it tests |
| [`tf.sdm.autocorrelation_range`](sdm.md#terraflux.sdm.autocorrelation_range) | The range of spatial autocorrelation of each predictor (and their median), in metres |
| [`tf.sdm.spatial_clusters`](sdm.md#terraflux.sdm.spatial_clusters) | Folds from k-means clusters of the records' coordinates (or of their environment) |
| [`tf.sdm.buffer_folds`](sdm.md#terraflux.sdm.buffer_folds) | Leave-one-out folds with a buffer: each presence is a test, nothing within ``size`` trains |
| [`tf.sdm.quadrant_folds`](sdm.md#terraflux.sdm.quadrant_folds) | Four folds from the occurrences split at their median latitude and longitude (ENMeval "block") |
| [`tf.sdm.checkerboard_folds`](sdm.md#terraflux.sdm.checkerboard_folds) | Folds from a checkerboard of aggregated cells (ENMeval "checkerboard") |
| [`tf.sdm.random_folds`](sdm.md#terraflux.sdm.random_folds) | Random k-fold of the presences, background always training (ENMeval "randomkfold") |
| [`tf.sdm.jackknife_folds`](sdm.md#terraflux.sdm.jackknife_folds) | Leave-one-out: each presence its own fold, background always training (ENMeval "jackknife") |
| [`tf.sdm.MaxNet`](sdm.md#terraflux.sdm.MaxNet) | Maxent fitted as R's maxnet does it, natively: the same model, faster |
| [`tf.sdm.evaluate`](sdm.md#terraflux.sdm.evaluate) | Evaluate predictions at presences against background points (or absences) |
| [`tf.sdm.Evaluation`](sdm.md#terraflux.sdm.Evaluation) | How well predictions separate presences from background (or absences) |
| [`tf.sdm.boyce`](sdm.md#terraflux.sdm.boyce) | Continuous Boyce index: does predicted suitability rise with the frequency of presences? |
| [`tf.sdm.threshold`](sdm.md#terraflux.sdm.threshold) | A threshold that turns continuous predictions into presence / absence |
| [`tf.sdm.omission_rate`](sdm.md#terraflux.sdm.omission_rate) | Share of test presences predicted below a threshold set on the training presences |
| [`tf.sdm.mess`](sdm.md#terraflux.sdm.mess) | Multivariate environmental similarity surface: how far conditions lie outside the reference |
| [`tf.sdm.mop`](sdm.md#terraflux.sdm.mop) | Mobility-oriented parity: where new conditions are outside or far from the reference ones |
| [`tf.sdm.exdet`](sdm.md#terraflux.sdm.exdet) | Extrapolation detection: univariate (NT1) and combinatorial (NT2) novelty |
| [`tf.sdm.fit`](sdm.md#terraflux.sdm.fit) | Fit models to SDMData, cross-validate them on its folds, and fit them to all the data |
| [`tf.sdm.SDMResult`](sdm.md#terraflux.sdm.SDMResult) | Fitted models, their cross-validated scores and what is needed to project them |
| [`tf.sdm.predict`](sdm.md#terraflux.sdm.predict) | Project fitted models onto rasters of predictors (present or future conditions) |
| [`tf.sdm.response_curves`](sdm.md#terraflux.sdm.response_curves) | How each model's suitability changes along one predictor, the others held at their mean (or median) |
| [`tf.sdm.importance`](sdm.md#terraflux.sdm.importance) | Permutation importance: how much AUC drops when a predictor's values are shuffled |
| [`tf.sdm.migrate`](sdm.md#terraflux.sdm.migrate) | A species' dispersal through the habitat suitability of successive periods, as MigClim simulates it |
| [`tf.sdm.Migration`](sdm.md#terraflux.sdm.Migration) | A dispersal simulation (`migrate`) |
| [`tf.sdm.area_of_habitat`](sdm.md#terraflux.sdm.area_of_habitat) | Area of habitat of species: their ranges refined by habitats and elevations |
| [`tf.sdm.HabitatAreas`](sdm.md#terraflux.sdm.HabitatAreas) | Result of `area_of_habitat` |
| [`tf.sdm.cell_area_km2`](sdm.md#terraflux.sdm.cell_area_km2) | The area of every cell of a raster's grid in km²: the product of the resolutions in a projected coordinate system (in its linear unit), the cell on the ellipsoid in a geographic one |
| [`tf.sdm.example_data`](sdm.md#terraflux.sdm.example_data) | Path of an example data set for species distribution modelling |

### Animal movement

| Name | Summary |
|---|---|
| [`tf.load_tracks`](move.md#terraflux.move.load_tracks) | Read a Movebank file into tracks (move2's ``mt_read``): a GeoDataFrame of points in EPSG:4326 |
| [`tf.move.example_data`](move.md#terraflux.move.example_data) | Path of an example data set for movement ecology |
| [`tf.move.as_tracks`](move.md#terraflux.move.as_tracks) | Tracks from a table of locations (move2's ``mt_as_move2``): a GeoDataFrame ``tf.move`` understands |
| [`tf.move.set_columns`](move.md#terraflux.move.set_columns) | Record which columns hold the time and the track (``tracks.attrs``), so later calls need not name them |
| [`tf.move.simulate_brownian`](move.md#terraflux.move.simulate_brownian) | Tracks of Brownian motion (move2's ``mt_sim_brownian_motion``) |
| [`tf.move.distance`](move.md#terraflux.move.distance) | Distance from each location to the next location of its track |
| [`tf.move.time_lags`](move.md#terraflux.move.time_lags) | Time from each location to the next of its track (move2's ``mt_time_lags``), in seconds |
| [`tf.move.speed`](move.md#terraflux.move.speed) | Speed of each step: `distance` over `time_lags` (move2's ``mt_speed``) |
| [`tf.move.azimuth`](move.md#terraflux.move.azimuth) | Direction of each step, clockwise from north (move2's ``mt_azimuth``) |
| [`tf.move.turn_angle`](move.md#terraflux.move.turn_angle) | Change of direction at each location, in [-pi, pi) (move2's ``mt_turnangle``) |
| [`tf.move.unique`](move.md#terraflux.move.unique) | Which locations to keep so that no two of a track share their time (move2's ``mt_unique``) |
| [`tf.move.filter_unique`](move.md#terraflux.move.filter_unique) | ``tracks`` without duplicated times within a track (move2's ``mt_filter_unique``; see `unique`) |
| [`tf.move.per_interval`](move.md#terraflux.move.per_interval) | Which locations to keep for one per time interval of each track (move2's ``mt_per_interval``) |
| [`tf.move.filter_per_interval`](move.md#terraflux.move.filter_per_interval) | ``tracks`` with one location per time interval of each track (see `per_interval`) |
| [`tf.move.movebank_visible`](move.md#terraflux.move.movebank_visible) | Which locations Movebank shows (move2's ``mt_movebank_visible``): a boolean mask |
| [`tf.move.filter_movebank_visible`](move.md#terraflux.move.filter_movebank_visible) | ``tracks`` without the locations Movebank hides (see `movebank_visible`) |
| [`tf.move.interpolate`](move.md#terraflux.move.interpolate) | Locations at given times between the recorded ones (move2's ``mt_interpolate``) |
| [`tf.move.track_lines`](move.md#terraflux.move.track_lines) | Each track as a line through its locations in time order (move2's ``mt_track_lines``) |
| [`tf.move.segments`](move.md#terraflux.move.segments) | The step from each location to the next of its track as a line (move2's ``mt_segments``) |
| [`tf.move.track_data`](move.md#terraflux.move.track_data) | One row per track with the columns whose value does not change within a track (move2's ``mt_track_data``) |
| [`tf.move.aeqd_crs`](move.md#terraflux.move.aeqd_crs) | An azimuthal equidistant projection centred on the tracks (move2's ``mt_aeqd_crs``) |
| [`tf.move.resample`](move.md#terraflux.move.resample) | Locations at a regular rate, in bursts (amt's ``track_resample``, the same locations and bursts) |
| [`tf.move.filter_min_n_burst`](move.md#terraflux.move.filter_min_n_burst) | The bursts (`resample`) of at least ``min_n`` locations (amt's ``filter_min_n_burst``) |
| [`tf.move.steps`](move.md#terraflux.move.steps) | The steps between consecutive locations of each track (amt's ``steps``, ``steps_by_burst``) |
| [`tf.move.time_of_day`](move.md#terraflux.move.time_of_day) | Day or night at each location, or at the end or start of each step (amt's ``time_of_day``) |
| [`tf.move.nsd`](move.md#terraflux.move.nsd) | Net squared displacement: the squared distance of each location from the first of its track (amt's ``nsd``) |
| [`tf.move.cum_dist`](move.md#terraflux.move.cum_dist) | The length of each track: the sum of its steps (amt's ``cum_dist``), metres in longitude / latitude |
| [`tf.move.tot_dist`](move.md#terraflux.move.tot_dist) | The distance from the first to the last location of each track (amt's ``tot_dist``) |
| [`tf.move.straightness`](move.md#terraflux.move.straightness) | `tot_dist` over `cum_dist`: 1 for a straight track, near 0 for a tortuous one (amt's ``straightness``) |
| [`tf.move.msd`](move.md#terraflux.move.msd) | Mean squared distance of the locations of each track from their centre (amt's ``msd``) |
| [`tf.move.sinuosity`](move.md#terraflux.move.sinuosity) | Sinuosity of each track (Benhamou 2004; amt's ``sinuosity``) |
| [`tf.move.tac`](move.md#terraflux.move.tac) | Turn angle correlation of each track (Dray et al. 2010; amt's ``tac``) |
| [`tf.move.make_distribution`](move.md#terraflux.move.make_distribution) | A distribution of step lengths or turn angles (amt's ``make_distribution``, ``make_*_distr``) |
| [`tf.move.Distribution`](move.md#terraflux.move.Distribution) | A distribution of step lengths or turn angles (amt's ``amt_distr``) |
| [`tf.move.fit_distr`](move.md#terraflux.move.fit_distr) | A distribution fitted to step lengths or turn angles (amt's ``fit_distr``, the same numbers) |
| [`tf.move.random_numbers`](move.md#terraflux.move.random_numbers) | Draws of a distribution of steps (amt's ``random_numbers``) |
| [`tf.move.random_steps`](move.md#terraflux.move.random_steps) | Each observed step with ``n_control`` random steps from its start (amt's ``random_steps``) |
| [`tf.move.extract_covariates`](move.md#terraflux.move.extract_covariates) | The values of rasters at locations, or at the ends or starts of steps (amt's ``extract_covariates``) |

### Land change simulation

| Name | Summary |
|---|---|
| [`tf.simulate.changes`](simulate.md#terraflux.simulate.changes) | Land change along a series of categorical maps |
| [`tf.simulate.Changes`](simulate.md#terraflux.simulate.Changes) | Land change along a series of categorical maps (see `changes`) |
| [`tf.simulate.IntensityAnalysis`](simulate.md#terraflux.simulate.IntensityAnalysis) | Intensity analysis of land change (Aldwaik & Pontius 2012), as OpenLand computes it |
| [`tf.simulate.crosstab`](simulate.md#terraflux.simulate.crosstab) | Cells of each class of one map that hold each class of another (a transition matrix of counts) |
| [`tf.simulate.difference_components`](simulate.md#terraflux.simulate.difference_components) | Agreement and the components of difference of each category and overall (diffeR's ``diffTablej``) |
| [`tf.simulate.overall_difference`](simulate.md#terraflux.simulate.overall_difference) | The overall difference and its components (diffeR's ``overallDiff``, ``overallQtyD``...) |
| [`tf.simulate.difference_resolutions`](simulate.md#terraflux.simulate.difference_resolutions) | Quantity, exchange and shift differences at increasingly coarse resolutions (diffeR's ``differenceMR``) |
| [`tf.simulate.composite_matrix`](simulate.md#terraflux.simulate.composite_matrix) | Cross-tabulation at a coarser resolution by the composite method (diffeR's ``composite``) |
| [`tf.simulate.memberships`](simulate.md#terraflux.simulate.memberships) | The share of each category in each block of factor x factor cells (diffeR's ``memberships``) |
| [`tf.simulate.mad`](simulate.md#terraflux.simulate.mad) | Mean absolute deviation between two continuous maps and its components (diffeR's ``MAD``) |
| [`tf.simulate.sample_to_population`](simulate.md#terraflux.simulate.sample_to_population) | A sample's cross-tabulation scaled to the population of each row category (diffeR's ``sample2pop``) |
| [`tf.simulate.Markov`](simulate.md#terraflux.simulate.Markov) | A Markov chain of land change: the transition probabilities of a period and of each time step |
| [`tf.simulate.WeightsOfEvidence`](simulate.md#terraflux.simulate.WeightsOfEvidence) | Weights of evidence of land use transitions, calibrated on two maps (Dinamica EGO's method) |
| [`tf.simulate.weights_of_evidence`](simulate.md#terraflux.simulate.weights_of_evidence) | Calibrate weights of evidence in one call: ``WeightsOfEvidence(ranges, **kwargs).fit(t0, t1, drivers)`` |
| [`tf.simulate.LogisticRegression`](simulate.md#terraflux.simulate.LogisticRegression) | Logistic regression of the presence of each land use class on the drivers (lulcc's ``glmModels``) |
| [`tf.simulate.Estimator`](simulate.md#terraflux.simulate.Estimator) | Potential of each class from a scikit-learn classifier, one model per class |
| [`tf.simulate.MultiCriteria`](simulate.md#terraflux.simulate.MultiCriteria) | Potential from weighted factors, the weights from pairwise comparisons (AHP; MOLUSCE's MCE) |
| [`tf.simulate.RasterPotential`](simulate.md#terraflux.simulate.RasterPotential) | A potential given as maps: one layer per class (or transition), as computed elsewhere |
| [`tf.simulate.Model`](simulate.md#terraflux.simulate.Model) | A land change model: potential, demand, allocation and rules |
| [`tf.simulate.Simulation`](simulate.md#terraflux.simulate.Simulation) | The maps of a simulation and what happened at each step |
| [`tf.simulate.AreaDemand`](simulate.md#terraflux.simulate.AreaDemand) | Cells each class must cover at each time step (CLUE-S's demand) |
| [`tf.simulate.Rules`](simulate.md#terraflux.simulate.Rules) | Decision rules of the allocation (lulcc's ``rules``, ``elas``, ``mask``, ``hist``, ``neighb``) |
| [`tf.simulate.Neighbourhood`](simulate.md#terraflux.simulate.Neighbourhood) | Neighbourhood thresholds (lulcc's ``NeighbRasterStack`` and ``nb.rules``) |
| [`tf.simulate.CLUE`](simulate.md#terraflux.simulate.CLUE) | CLUE-S allocation (Verburg et al. 2002), as lulcc's ``CluesModel`` |
| [`tf.simulate.Ordered`](simulate.md#terraflux.simulate.Ordered) | The ordered allocation of Fuchs et al. (2013) with stochastic choice, as lulcc's ``OrderedModel`` |
| [`tf.simulate.distance_to`](simulate.md#terraflux.simulate.distance_to) | Distance from each cell to the nearest cell of some classes |
| [`tf.simulate.share_of`](simulate.md#terraflux.simulate.share_of) | Share of the cells around each cell that hold some classes |
| [`tf.simulate.gravity_of`](simulate.md#terraflux.simulate.gravity_of) | Gravitational interaction potential of the cells of some classes |
| [`tf.simulate.Regional`](simulate.md#terraflux.simulate.Regional) | A demand per region (municipalities, biomes, basins...) |
| [`tf.simulate.CLUMondo`](simulate.md#terraflux.simulate.CLUMondo) | CLUMondo's demand-driven allocation of land systems |
| [`tf.simulate.ServiceDemand`](simulate.md#terraflux.simulate.ServiceDemand) | The demand for goods and services at each time step (CLUMondo's ``demand.in``) |
| [`tf.simulate.Supply`](simulate.md#terraflux.simulate.Supply) | What one cell of each land system supplies of each good or service (CLUMondo's ``lusmatrix``) |
| [`tf.simulate.NeighbourhoodEffect`](simulate.md#terraflux.simulate.NeighbourhoodEffect) | CLUMondo's neighbourhood function (``neighmat.txt`` and ``alloc2.reg``) |
| [`tf.simulate.LandSystemStack`](simulate.md#terraflux.simulate.LandSystemStack) | Binary layers, possibly overlapping, as land systems (bit masks) |
| [`tf.simulate.Roles`](simulate.md#terraflux.simulate.Roles) | What each transition between land systems does to their layers |
| [`tf.simulate.Cap`](simulate.md#terraflux.simulate.Cap) | At most so many cells (or so much area) a year may make a role's transitions (or given pairs) |
| [`tf.simulate.Succession`](simulate.md#terraflux.simulate.Succession) | Autonomous change after some years, whatever the demand (CLUMondo's code 1000 + years) |
| [`tf.simulate.ExpanderPatcher`](simulate.md#terraflux.simulate.ExpanderPatcher) | Dinamica's Expander and Patcher allocation of transitions (``AllocateTransitions``) |
| [`tf.simulate.ContinuousPatcher`](simulate.md#terraflux.simulate.ContinuousPatcher) | New patches that take a stock (biomass, carbon, timber) up to a quota |
| [`tf.simulate.StockAllocation`](simulate.md#terraflux.simulate.StockAllocation) | What `allocate` took |
| [`tf.simulate.StockSimulation`](simulate.md#terraflux.simulate.StockSimulation) | What `run` did, step by step |
| [`tf.simulate.RoadBuilder`](simulate.md#terraflux.simulate.RoadBuilder) | New roads each step, from the network towards attractive places |
| [`tf.simulate.RoadDistance`](simulate.md#terraflux.simulate.RoadDistance) | Distance to a growing road network: a dynamic variable (see `dynamic`) |
| [`tf.simulate.friction_cost`](simulate.md#terraflux.simulate.friction_cost) | Cost of reaching every cell from the nearest source over a friction surface |
| [`tf.simulate.destinations`](simulate.md#terraflux.simulate.destinations) | Cells drawn by attraction in quadrants of a map |
| [`tf.simulate.pathways`](simulate.md#terraflux.simulate.pathways) | Paths from source cells to a network down a cost surface |
| [`tf.simulate.CARS`](simulate.md#terraflux.simulate.CARS) | PLUS's CA based on multi-type random patch seeds (Liang et al. 2021) |
| [`tf.simulate.MolusceWoE`](simulate.md#terraflux.simulate.MolusceWoE) | MOLUSCE's weights of evidence (``WoeManager``): one model per transition, persistence included |
| [`tf.simulate.MolusceCA`](simulate.md#terraflux.simulate.MolusceCA) | MOLUSCE's simulator: each iteration makes the observed transitions where the model is most confident |
| [`tf.simulate.PatchGrowing`](simulate.md#terraflux.simulate.PatchGrowing) | FUTURES' patch growing allocation (r.futures.simulation) |
| [`tf.simulate.DevelopmentDemand`](simulate.md#terraflux.simulate.DevelopmentDemand) | Cells each subregion develops at each step (r.futures.simulation's ``demand`` file) |
| [`tf.simulate.DevelopmentPotential`](simulate.md#terraflux.simulate.DevelopmentPotential) | The development potential's coefficients per subregion (r.futures.potential's table) |
| [`tf.simulate.development_pressure`](simulate.md#terraflux.simulate.development_pressure) | The development pressure of each cell (r.futures.devpressure) |
| [`tf.simulate.Flooding`](simulate.md#terraflux.simulate.Flooding) | FUTURES' flood submodel (Sanchez et al. 2023): floods, damage, response, abandonment, migration |
| [`tf.simulate.patch_library`](simulate.md#terraflux.simulate.patch_library) | The sizes of the patches developed between two maps (r.futures.calib's patch library) |
| [`tf.simulate.calibrate_patches`](simulate.md#terraflux.simulate.calibrate_patches) | Calibrate FUTURES' patch compactness and discount factor (r.futures.calib) |
| [`tf.simulate.three_map_comparison`](simulate.md#terraflux.simulate.three_map_comparison) | Compare the reference maps at the start and the end with the simulated end at several resolutions |
| [`tf.simulate.ThreeMapComparison`](simulate.md#terraflux.simulate.ThreeMapComparison) | Three maps compared at several resolutions (lulcc's ``ThreeMapComparison``) |
| [`tf.simulate.compare_maps`](simulate.md#terraflux.simulate.compare_maps) | MOLUSCE's comparison of two maps (``DependenceCoef``): kappa, agreement and dependence |
| [`tf.simulate.error_budget`](simulate.md#terraflux.simulate.error_budget) | MOLUSCE's error budget (Pontius 2002): agreement for levels of information, at coarser resolutions |
| [`tf.simulate.futures_validation`](simulate.md#terraflux.simulate.futures_validation) | FUTURES' validation (``r.futures.validation``): disagreement, kappa, kappa simulation, change metrics |
| [`tf.simulate.futures_validation_grid`](simulate.md#terraflux.simulate.futures_validation_grid) | FUTURES' validation per tile or per region (``r.futures.gridvalidation``) |
| [`tf.simulate.fuzzy_similarity`](simulate.md#terraflux.simulate.fuzzy_similarity) | Fuzzy similarity of two categorical maps (Dinamica EGO's ``CalcReciprocalSimilarityMap``) |
| [`tf.simulate.FuzzySimilarity`](simulate.md#terraflux.simulate.FuzzySimilarity) | The fuzzy similarity of two maps (`fuzzy_similarity`) |
| [`tf.simulate.differences`](simulate.md#terraflux.simulate.differences) | The map of differences: each cell that changed between the maps, in its new class; NaN elsewhere |
| [`tf.simulate.similarity_of_differences`](simulate.md#terraflux.simulate.similarity_of_differences) | Dinamica EGO's validation: fuzzy similarity of the observed and the simulated change, per window |
| [`tf.simulate.validate`](simulate.md#terraflux.simulate.validate) | Compare a simulated map with the observed one in every way the land change tools do |
| [`tf.simulate.Validation`](simulate.md#terraflux.simulate.Validation) | Every comparison of a simulated map with the observed one (`validate`) |
| [`tf.simulate.calibrate`](simulate.md#terraflux.simulate.calibrate) | Search the settings of a model that best reproduce an observed map |
| [`tf.simulate.Calibration`](simulate.md#terraflux.simulate.Calibration) | The result of `calibrate` |
| [`tf.simulate.run`](simulate.md#terraflux.simulate.run) | Fit a land change model on two maps, check it on the observed period and project it |
| [`tf.simulate.Projection`](simulate.md#terraflux.simulate.Projection) | The result of `run` |
| [`tf.simulate.agreement`](simulate.md#terraflux.simulate.agreement) | How a simulated map matches the observed one, cell by cell and in quantity |
| [`tf.simulate.presets`](simulate.md#terraflux.simulate.presets) | The classic land change models, assembled from tf.simulate's pieces |
| [`tf.simulate.terraflux`](simulate.md#terraflux.simulate.terraflux) | Run a Terraflux model with the ``terraflux`` command; the simulated maps ``(step, y, x)`` |
| [`tf.simulate.scenarios`](simulate.md#terraflux.simulate.scenarios) | Run a model under several scenarios, several times each, in parallel |
| [`tf.simulate.Ensemble`](simulate.md#terraflux.simulate.Ensemble) | The simulations of `scenarios`: every scenario, replicate and year |
| [`tf.simulate.example_data`](simulate.md#terraflux.simulate.example_data) | Path of an example data set for land change modelling |

### Map accuracy & area

| Name | Summary |
|---|---|
| [`tf.accuracy.sample_size`](accuracy.md#terraflux.accuracy.sample_size) | The size of a stratified random sample for the accuracy and the area of a map, and its share of each stratum |
| [`tf.accuracy.SampleSize`](accuracy.md#terraflux.accuracy.SampleSize) | The size of a stratified sample and its share of each stratum (`sample_size`) |
| [`tf.accuracy.allocate`](accuracy.md#terraflux.accuracy.allocate) | The sample of each stratum, and whether the minima asked for more than ``n`` |
| [`tf.accuracy.sample`](accuracy.md#terraflux.accuracy.sample) | A stratified random sample of a map's cells for its accuracy and area (Olofsson et al. 2014) |
| [`tf.accuracy.assess`](accuracy.md#terraflux.accuracy.assess) | The accuracy and the area of a map's classes from a stratified random sample labelled with the reference |
| [`tf.accuracy.Accuracy`](accuracy.md#terraflux.accuracy.Accuracy) | The accuracy and the area of a map's classes from a stratified sample (`assess`) |
| [`tf.accuracy.qnorm`](accuracy.md#terraflux.accuracy.qnorm) | The standard normal's quantile (Wichura's AS 241, as R's ``qnorm``) |

### Deforestation allocation (VT0007)

| Name | Summary |
|---|---|
| [`tf.redd.edge_distance`](redd.md#terraflux.redd.edge_distance) | The distance of every forest cell to the nearest cell without forest, in map units (VT0007's input) |
| [`tf.redd.nrt`](redd.md#terraflux.redd.nrt) | VT0007's negligible risk threshold (UDef-ARP's ``nrt_calculation``) |
| [`tf.redd.benchmark`](redd.md#terraflux.redd.benchmark) | VT0007's benchmark vulnerability map (UDef-ARP's ``geometric_classification``) |
| [`tf.redd.alternative`](redd.md#terraflux.redd.alternative) | VT0007's alternative vulnerability map (UDef-ARP's ``geometric_classification_alternative``) |
| [`tf.redd.fit`](redd.md#terraflux.redd.fit) | VT0007's fit of a vulnerability map to a period's deforestation (UDef-ARP's fitting) |
| [`tf.redd.Fit`](redd.md#terraflux.redd.Fit) | The fit of a vulnerability map to a period's deforestation (`fit`) |
| [`tf.redd.allocate`](redd.md#terraflux.redd.allocate) | VT0007's allocation of the expected deforestation (UDef-ARP's prediction) |
| [`tf.redd.Allocation`](redd.md#terraflux.redd.Allocation) | VT0007's allocation of the expected deforestation (`allocate`) |
| [`tf.redd.assess`](redd.md#terraflux.redd.assess) | VT0007's assessment of an allocation (UDef-ARP's model evaluation) |
| [`tf.redd.Assessment`](redd.md#terraflux.redd.Assessment) | VT0007's assessment of an allocation (`assess`) |

### Brazil's Forest Code

| Name | Summary |
|---|---|
| [`tf.forest_code.app`](forest_code.md#terraflux.forest_code.app) | The Áreas de Preservação Permanente of Brazil's Forest Code (Lei 12.651/2012, art. 4) |
| [`tf.forest_code.APP`](forest_code.md#terraflux.forest_code.APP) | The Áreas de Preservação Permanente of a map (`app`) |
| [`tf.forest_code.river_margin`](forest_code.md#terraflux.forest_code.river_margin) | Art. 4, I: the margin of a river of a width (metres): 30 m below 10 m wide, 50 to 50, 100 to 200, 200 to 600, 500 |
| [`tf.forest_code.consolidated_river`](forest_code.md#terraflux.forest_code.consolidated_river) | Art. 61-A, §§ 1-4: a river's margin in a consolidated area by the property's fiscal modules |
| [`tf.forest_code.consolidated_lake`](forest_code.md#terraflux.forest_code.consolidated_lake) | Art. 61-A, § 6: a lake's margin in a consolidated area: 5, 8, 15 m up to 1, 2, 4 modules; 30 m beyond |
| [`tf.forest_code.reserve`](forest_code.md#terraflux.forest_code.reserve) | The balance of every rural property's Reserva Legal (Lei 12.651/2012) |
| [`tf.forest_code.ReservaLegal`](forest_code.md#terraflux.forest_code.ReservaLegal) | The Reserva Legal's balance of every rural property (`reserve`) |

### Ecosystem services

| Name | Summary |
|---|---|
| [`tf.services.carbon`](services.md#terraflux.services.carbon) | Carbon storage of a landscape and its change between two maps (InVEST's carbon model) |
| [`tf.services.Carbon`](services.md#terraflux.services.Carbon) | Carbon stored in a landscape, and its change (`carbon`) |
| [`tf.services.valuation_constant`](services.md#terraflux.services.valuation_constant) | The factor turning a change in carbon into its net present value (InVEST's) |
| [`tf.services.emissions`](services.md#terraflux.services.emissions) | The carbon emitted and removed by land change over a series of maps (the IPCC's stock-difference method) |
| [`tf.services.Emissions`](services.md#terraflux.services.Emissions) | The carbon emitted and removed along a series of land use maps (`emissions`) |
| [`tf.services.water_yield`](services.md#terraflux.services.water_yield) | Annual water yield, consumption and hydropower value (InVEST's annual water yield model) |
| [`tf.services.WaterYield`](services.md#terraflux.services.WaterYield) | Annual water yield of a landscape (`water_yield`) |
| [`tf.services.seasonal_water_yield`](services.md#terraflux.services.seasonal_water_yield) | Quickflow, local recharge and baseflow by month and year (InVEST's seasonal water yield model) |
| [`tf.services.SeasonalWaterYield`](services.md#terraflux.services.SeasonalWaterYield) | Quickflow, recharge and baseflow of a landscape (`seasonal_water_yield`) |
| [`tf.services.sdr`](services.md#terraflux.services.sdr) | Sediment export, deposition and retention (InVEST's sediment delivery ratio model) |
| [`tf.services.SDR`](services.md#terraflux.services.SDR) | Sediment export and retention of a landscape (`sdr`) |
| [`tf.services.ndr`](services.md#terraflux.services.ndr) | Nutrient export and retention (InVEST's nutrient delivery ratio model) |
| [`tf.services.NDR`](services.md#terraflux.services.NDR) | Nutrient export and retention of a landscape (`ndr`) |
| [`tf.services.habitat_quality`](services.md#terraflux.services.habitat_quality) | Habitat degradation, quality and rarity (InVEST's habitat quality model) |
| [`tf.services.HabitatQuality`](services.md#terraflux.services.HabitatQuality) | Habitat degradation, quality and rarity (`habitat_quality`) |
| [`tf.services.pollination`](services.md#terraflux.services.pollination) | Pollinator abundance and crop pollination (InVEST's crop pollination model) |
| [`tf.services.Pollination`](services.md#terraflux.services.Pollination) | Pollinator supply and abundance, and the yield of farms (`pollination`) |
| [`tf.services.coastal_vulnerability`](services.md#terraflux.services.coastal_vulnerability) | Exposure of the coast to erosion and flooding (InVEST's coastal vulnerability model) |
| [`tf.services.CoastalVulnerability`](services.md#terraflux.services.CoastalVulnerability) | Exposure of the coast to erosion and flooding (`coastal_vulnerability`) |

### Uncertainty

| Name | Summary |
|---|---|
| [`tf.uncertainty.sum_sd`](uncertainty.md#terraflux.uncertainty.sum_sd) | Standard deviation of a total of uncertain values, sum of w_i v_i |
| [`tf.uncertainty.product_sd`](uncertainty.md#terraflux.uncertainty.product_sd) | Standard deviation of the product of two uncertain values, x v |
| [`tf.uncertainty.delta_method`](uncertainty.md#terraflux.uncertainty.delta_method) | Uncertainty of a result from the uncertainty of its parameters (delta method) |
| [`tf.uncertainty.DeltaResult`](uncertainty.md#terraflux.uncertainty.DeltaResult) | Result of `delta_method` |
| [`tf.uncertainty.monte_carlo`](uncertainty.md#terraflux.uncertainty.monte_carlo) | Uncertainty of a result by drawing its parameters (Monte Carlo) |
| [`tf.uncertainty.MonteCarloResult`](uncertainty.md#terraflux.uncertainty.MonteCarloResult) | Result of `monte_carlo` |
| [`tf.uncertainty.interval`](uncertainty.md#terraflux.uncertainty.interval) | The normal confidence interval of a value: value -/+ z sd, z the normal quantile of the level |

### Sensitivity analysis

| Name | Summary |
|---|---|
| [`tf.sensitivity.design`](sensitivity.md#terraflux.sensitivity.design) | The runs of a global sensitivity analysis (SALib's samplers, the same values) |
| [`tf.sensitivity.Design`](sensitivity.md#terraflux.sensitivity.Design) | The runs of a sensitivity analysis (`design`) |
| [`tf.sensitivity.evaluate`](sensitivity.md#terraflux.sensitivity.evaluate) | The outputs of a model at every run of a design |
| [`tf.sensitivity.analyze`](sensitivity.md#terraflux.sensitivity.analyze) | The indices of a global sensitivity analysis from its runs and their outputs (SALib's, the same numbers) |
| [`tf.sensitivity.Sensitivity`](sensitivity.md#terraflux.sensitivity.Sensitivity) | The indices of a sensitivity analysis (`analyze`) |

### Statistics of maps

| Name | Summary |
|---|---|
| [`tf.stats.trend`](stats.md#terraflux.stats.trend) | The monotonic trend of every cell along a series of maps (pymannkendall's ``original_test``, the same numbers) |
| [`tf.stats.Trend`](stats.md#terraflux.stats.Trend) | Every cell's trend along a series of maps (`trend`) |
| [`tf.stats.mann_kendall`](stats.md#terraflux.stats.mann_kendall) | The Mann-Kendall test and Sen's slope of one series (pymannkendall's ``original_test``, the same bits) |
| [`tf.stats.cell_stats`](stats.md#terraflux.stats.cell_stats) | A statistic of a stack's layers at every cell (ArcGIS's Cell Statistics, terra's ``app``) |
| [`tf.stats.local_correlation`](stats.md#terraflux.stats.local_correlation) | The correlation and the least squares line between two maps in a moving window (terra's ``focalPairs``, ``focalReg``) |
| [`tf.stats.LocalCorrelation`](stats.md#terraflux.stats.LocalCorrelation) | The correlation and least squares line of two maps in a moving window (`local_correlation`) |
| [`tf.stats.moran`](stats.md#terraflux.stats.moran) | The global Moran's I of a map (PySAL's ``esda.Moran``, to rounding) |
| [`tf.stats.Moran`](stats.md#terraflux.stats.Moran) | The global Moran's I of a map (`moran`) |
| [`tf.stats.lisa`](stats.md#terraflux.stats.lisa) | The local Moran's I of every cell (Anselin's LISA; PySAL's ``esda.Moran_Local``, to rounding) |
| [`tf.stats.LISA`](stats.md#terraflux.stats.LISA) | The local Moran's I of every cell (`lisa`) |
| [`tf.stats.getis_ord`](stats.md#terraflux.stats.getis_ord) | Getis and Ord's Gi* of every cell (PySAL's ``esda.G_Local`` with ``star`` and row-standardised weights) |
| [`tf.stats.GetisOrd`](stats.md#terraflux.stats.GetisOrd) | Getis and Ord's Gi* of every cell (`getis_ord`) |
| [`tf.stats.variogram`](stats.md#terraflux.stats.variogram) | The sample variogram of points and a model fitted to it (R's gstat ``variogram`` and ``fit.variogram``) |
| [`tf.stats.Variogram`](stats.md#terraflux.stats.Variogram) | The sample variogram of points and the model fitted to it (`variogram`) |
| [`tf.stats.VariogramModel`](stats.md#terraflux.stats.VariogramModel) | A variogram model: a nugget and a structure of a shape, its partial sill and range (gstat's ``vgm``) |
| [`tf.stats.idw`](stats.md#terraflux.stats.idw) | Inverse distance weighting of points onto a map (R's gstat ``idw``, the same numbers) |
| [`tf.stats.kriging`](stats.md#terraflux.stats.kriging) | Ordinary kriging of points onto a map (R's gstat ``krige``, to rounding: 1e-10) |
| [`tf.stats.Kriging`](stats.md#terraflux.stats.Kriging) | Ordinary kriging of points onto a map (`kriging`) |
