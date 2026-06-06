# Related Work

This page documents the existing ecosystem of open-source climate and weather data tools, and explains precisely where climagrid fits relative to each one.

---

## Comparison table

| Project | Data sources | Output | Grid engineering features | Asset-level join | License |
|---------|-------------|--------|--------------------------|-----------------|---------|
| **climagrid** | NASA POWER, NOAA HRRR, NCEI, USDA NRCS, USFS WFIGS | Pandas DataFrame, Parquet | Yes (IEEE C57.91, 738) | Yes (cKDTree) | Apache 2.0 |
| atlite | ERA5, SARAH, MERRA-2, CMSAF | xarray Dataset | Solar/wind capacity factors only | No | MIT |
| herbie | HRRR, GFS, NAM, ECMWF | xarray/Pandas | None | No | MIT |
| wetterdienst | DWD, NOAA, ECCC, EA, NWS | Pandas | None | No | MIT |
| metpy | Any (post-processing) | Pandas/xarray | Thermodynamic calculations | No | BSD-3 |
| pvlib | NASA POWER, TMY | Pandas | Solar PV only | No | BSD-3 |
| xarray-spatial | Raster inputs | xarray | Spatial analysis | No | BSD-3 |
| Herbie + xarray | HRRR | xarray | None | No | MIT |
| NREL WindToolkit | Wind resource | HDF5 | Wind power density | No | BSD-3 |
| climata | NOAA, USGS | Pandas | None | No | MIT |

---

## Detailed comparisons

### atlite

atlite converts ERA5 and other reanalysis data into renewable energy capacity factor time series (wind, solar, hydro). It is excellent for energy system planning and is widely used in European grid studies.

**Gap:** atlite targets capacity planning at the regional scale, not asset-level predictive maintenance. It produces capacity factors (how much electricity a turbine would generate), not equipment stress indices (how fast a transformer is aging). It has no concept of individual utility assets, no IEEE standard implementations, and no USDA soil or USFS wildfire data.

A planned `from_atlite_cutout()` adapter (not yet implemented) would let users who already have atlite cutout files reuse their atmospheric data as climagrid inputs without re-downloading.

### herbie

herbie is the most capable NOAA HRRR/GFS/NOMADS access library in Python. climagrid uses herbie internally for its `HrrrAdapter`.

**Gap:** herbie fetches and decodes GRIB2 files. It does not join data to point assets, compute engineering stress indices, or integrate multiple government data sources into a single standardized schema. It is a transport layer, not an end-to-end feature pipeline.

### wetterdienst

wetterdienst implements the adapter/provider pattern that climagrid's source architecture is modeled after. It provides excellent historical station observation access for European and North American weather services.

**Gap:** wetterdienst's adapters return raw meteorological observations: no IEEE-standard engineering features, no spatial join to utility assets, no wildfire or soil data, no designed integration with predictive maintenance pipelines.

### metpy

metpy provides professional meteorological calculations: thermodynamic parameters, hodographs, sounding analysis. It is the gold standard for operational meteorology in Python.

**Gap:** metpy is a calculation toolkit, not a data pipeline. It requires the user to supply the input data from whatever source they prefer, and it does not include power systems engineering models (transformer aging, conductor sag, ice loading).

### pvlib

pvlib is specialized for photovoltaic system performance modeling. It uses NASA POWER as one data source (climagrid's default source uses the same NASA POWER API).

**Gap:** pvlib's scope is solar PV yield estimation. It has no models for transmission/distribution equipment stress, no wildfire or soil data integration, and no asset registry concept.

---

## What climagrid adds

The gap that none of the above tools fill is the combination of:

1. **Multiple government sources in one normalized schema:** NOAA + NASA + USDA + USFS
2. **Asset-level spatial join:** matches environmental data to specific utility assets
3. **IEEE/ASCE engineering stress models:** C57.91 transformer aging, 738-2012 conductor sag, ASCE 7-22 ice loading
4. **Output is a feature matrix, not raw weather data:** ready for predictive maintenance ML pipelines
5. **Open license:** Apache 2.0, free for commercial utility use including patent grants

---

## Upstream dependencies

climagrid depends on and is grateful to:

- **herbie:** HRRR/NOMADS access
- **wetterdienst:** adapter pattern inspiration
- **pvlib:** NASA POWER API documentation
- **geopandas + shapely:** spatial operations
- **scipy:** cKDTree nearest-neighbor join
- **pydantic:** schema validation
