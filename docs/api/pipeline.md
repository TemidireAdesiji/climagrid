# Pipeline

The high-level `run()` function is the primary entry point for most users.

## climagrid.run

```{eval-rst}
.. autofunction:: climagrid.pipeline.orchestrator.run
```

## Source and feature names

Valid `sources` values:

- `"nasa_power"`: {doc}`NASA POWER adapter </data_sources>`
- `"noaa_hrrr"`: NOAA HRRR adapter (requires `pip install "climagrid[noaa-nwp]"`)
- `"noaa_ncei"`: NOAA NCEI CDO adapter (requires free API token)
- `"usda_nrcs"`: USDA NRCS AWDB adapter
- `"usfs_wfigs"`: USFS NIFC WFIGS wildfire adapter

Valid `features` values:

- `"thermal"`: `feat_thermal_aging_factor`, `feat_heat_hours_above_35c`
- `"conductor_sag"`: `feat_conductor_sag_index`
- `"freeze_thaw"`: `feat_freeze_thaw_cycles`
- `"ice_loading"`: `feat_ice_loading_risk`
- `"soil"`: `feat_soil_saturation_index`
- `"wildfire"`: `feat_wildfire_proximity`
