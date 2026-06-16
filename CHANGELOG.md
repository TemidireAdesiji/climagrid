# Changelog

All notable changes to climagrid will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `climagrid.forecast()` and the `climagrid forecast` CLI command: medium-range probabilistic forecasting of daily asset stress features with prediction intervals, benchmarked against persistence and climatology baselines in a rolling-origin backtest (`climagrid.forecasting`, new `[ml]` optional extra). Forecasts environmental stress, not equipment failure.
- Conformal calibration of the prediction intervals (`ForecastConfig(calibrate_intervals=True)`), with season-conditional `"mondrian"` (default), `"normalized"`, and `"constant"` methods
- `LightGBMForecaster.save()` / `.load()` for train-once, reuse-anywhere inference; `evaluate()` and `history_ablation()` for honest skill scoring
- `examples/forecasting.ipynb` (showcase), `examples/kaggle_training.ipynb` (train and compare history windows, save models), and `examples/analyze_kaggle_run.ipynb`

## [0.2.1] - 2026-06-08

### Fixed
- README links to the quickstart notebook and the docs pages now use absolute GitHub URLs, so they resolve on the PyPI project page (relative links returned 404 there)

### Changed
- README DOI badge and `CITATION.cff` now use the Zenodo concept DOI (`10.5281/zenodo.20256535`), which always resolves to the latest version
- The README-media regeneration workflow opens pull requests (respecting branch protection) using a token, instead of pushing to the default branch; refreshed the demo gif and quickstart map

[0.2.1]: https://github.com/TemidireAdesiji/climagrid/releases/tag/v0.2.1

## [0.2.0] - 2026-05-19

### Added
- `climagrid report` CLI command and `[report]` optional extra: turns an asset CSV into a ranked inspection-priority PDF and an inspection-list CSV, no programming required (`climagrid.outputs.rank_assets`, `climagrid.outputs.generate_report`)
- Per-asset fetching for point-based sources: NASA POWER, NCEI, and NRCS now fetch one weather location per asset (via a `point_based` flag and `fetch_points`), so geographically spread asset registries each get local weather
- `examples/asset_stress_prioritization.ipynb`: end-to-end demo on real OpenStreetMap substations, including an exact verification of the thermal-aging feature against the IEEE C57.91 formula
- `docs/validation_notes.md`: a clear statement of what climagrid does and does not establish

### Changed
- The orchestrator now builds a bounding box covering the full asset extent for grid and station sources, not a single centroid
- Bundled sample registries (`sample_assets.csv`, `tx_assets.csv`) are now real electric substations from OpenStreetMap
- Documentation corrected to remove unproven failure-prediction and outcome claims; citation reworded from "for" to "to support" predictive maintenance; `feat_conductor_sag_index` labeled "IEEE 738 (simplified)" to reflect its simplified heat balance

### Fixed
- NCEI schema columns now match the adapter output (`ncei_temperature_max`/`min`, `ncei_precipitation_daily`)
- `BoundingBox.from_center` clamps to the valid WGS-84 range (no crash near the poles or antimeridian)
- Thermal-aging reference temperature corrected to 383.15 K so the IEEE C57.91 reference hotspot (110 C) yields exactly 1.0
- Documentation accuracy fixes (correct adapter class names, USDA vs NOAA for NRCS)

[0.2.0]: https://github.com/TemidireAdesiji/climagrid/releases/tag/v0.2.0

## [0.1.0] - 2026-05-17

### Added
- `NasaPowerAdapter`: hourly surface meteorology and solar irradiance from NASA POWER (MERRA-2)
- `NceiAdapter`: temperature, wind, and precipitation from NOAA NCEI CDO station network
- `HrrrAdapter`: 3 km atmospheric fields from NOAA HRRR (requires optional `herbie-data` dependency)
- `NrcsAdapter`: soil moisture and temperature from USDA NRCS SCAN/SNOTEL network
- `WfigsAdapter`: active wildfire perimeters from USFS NIFC WFIGS
- `feat_thermal_aging_factor`: transformer insulation aging per IEEE C57.91
- `feat_conductor_sag_index`: overhead conductor sag risk per IEEE 738-2012
- `feat_ice_loading_risk`: ice accretion risk per ASCE 7-22
- `feat_freeze_thaw_cycles`: daily freeze-thaw cycle count from temperature time series
- `feat_soil_saturation_index`: soil saturation risk from precipitation and soil moisture
- `feat_wildfire_proximity`: normalized proximity score to active fire perimeters
- `climagrid.run()`: single-call pipeline API (fetch, join, compute features, return DataFrame)
- CLI: `climagrid fetch` with CSV, Parquet, and long-form Parquet output
- `AssetRegistry`: loads utility asset records from CSV or GeoJSON
- `AssetEnvironmentJoiner`: spatial nearest-neighbour join of assets to environmental data
- Schema validation for all output columns
- Dockerfile for containerised deployment
- Apache 2.0 license

[0.1.0]: https://github.com/TemidireAdesiji/climagrid/releases/tag/v0.1.0
