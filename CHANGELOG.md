# Changelog

All notable changes to climagrid will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

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
