# Column Schema

Every adapter and feature function in climagrid produces DataFrames that conform to a shared column contract. Adding a new column to any output requires defining it in `climagrid.schema` first.

The schema is machine-readable via `climagrid.schema_summary()` and exportable as JSON via `climagrid.outputs.to_json_schema()`.

---

## Index columns

Always present, never null. Uniquely identify each observation.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `asset_id` | str | n/a | Utility asset identifier from AssetRegistry |
| `timestamp` | datetime64[ns, UTC] | n/a | UTC timestamp of the observation hour |
| `lat` | float64 | degrees | Asset latitude (WGS-84) |
| `lon` | float64 | degrees | Asset longitude (WGS-84) |

---

## NOAA HRRR columns

3 km CONUS hourly NWP analysis fields. Require `pip install "climagrid[noaa-nwp]"`.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `hrrr_temperature_2m` | float64 | °C | 2-metre air temperature |
| `hrrr_wind_speed_10m` | float64 | m/s | 10-metre wind speed (magnitude) |
| `hrrr_wind_direction_10m` | float64 | degrees | 10-metre wind direction (meteorological convention) |
| `hrrr_relative_humidity_2m` | float64 | % | 2-metre relative humidity |
| `hrrr_precipitation_rate` | float64 | mm/hr | Hourly accumulated precipitation |
| `hrrr_solar_irradiance_ghi` | float64 | W/m² | Downward short-wave radiation at surface |
| `hrrr_snow_depth` | float64 | m | Snow depth at surface |

---

## NASA POWER columns

MERRA-2-based hourly surface meteorology. Global coverage, no API key.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `nasa_temperature_2m` | float64 | °C | 2-metre air temperature (MERRA-2 based) |
| `nasa_wind_speed_10m` | float64 | m/s | 10-metre wind speed |
| `nasa_solar_irradiance_ghi` | float64 | W/m² | Global horizontal irradiance |
| `nasa_relative_humidity_2m` | float64 | % | Relative humidity at 2 m |
| `nasa_precipitation` | float64 | mm | Hourly precipitation |

---

## NOAA NCEI columns

Surface station observations from nearest GHCN station.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `ncei_temperature_max` | float64 | °C | Daily maximum temperature |
| `ncei_temperature_min` | float64 | °C | Daily minimum temperature |
| `ncei_wind_speed` | float64 | m/s | Average daily wind speed |
| `ncei_precipitation_daily` | float64 | mm | Daily precipitation |
| `ncei_relative_humidity` | float64 | % | Average daily relative humidity |

---

## USDA NRCS columns

SCAN / SNOTEL soil and snowpack sensors.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `nrcs_soil_moisture_pct` | float64 | % | Volumetric soil moisture (nearest SCAN station) |
| `nrcs_soil_temperature` | float64 | °C | Soil temperature at 2-inch depth |
| `nrcs_snow_water_equivalent` | float64 | mm | Snow water equivalent (nearest SNOTEL) |
| `nrcs_station_distance_km` | float64 | km | Distance from asset to nearest NRCS station |

---

## USFS WFIGS columns

Active wildfire perimeter proximity.

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `wfigs_nearest_fire_km` | float64 | km | Distance to nearest active fire perimeter edge |
| `wfigs_fire_active` | bool | n/a | True if any active fire within 50 km |
| `wfigs_fire_area_ha` | float64 | ha | Area of nearest active fire in hectares |

---

## Feature (stress index) columns

Computed by `climagrid.features.*`. Values are dimensionless indices or physical counts suitable for direct use as ML model inputs.

| Column | Type | Units | Standard | Description |
|--------|------|-------|----------|-------------|
| `feat_thermal_aging_factor` | float64 | per-unit | IEEE C57.91 | Arrhenius FAA relative to 110°C reference. Values >1 indicate accelerated aging. |
| `feat_heat_hours_above_35c` | float64 | hours | n/a | Cumulative hours with temperature >35°C in rolling 168-hour window |
| `feat_freeze_thaw_cycles` | float64 | count | n/a | Freeze-thaw transition count over the rolling window |
| `feat_ice_loading_risk` | float64 | 0-1 | ASCE 7-22 (simplified) | Normalized ice accretion risk (temperature × precipitation × wind composite) |
| `feat_soil_saturation_index` | float64 | 0-1 | n/a | Normalized soil saturation proxy (0=dry, 1=saturated) |
| `feat_wildfire_proximity` | float64 | 0-1 | n/a | Normalized fire proximity score (0=no risk, 1=adjacent to active fire) |
| `feat_conductor_sag_index` | float64 | 0-1 | IEEE 738 (simplified) | Normalized thermal sag index (0=no sag risk, 1=at design maximum) |

---

## Programmatic access

```python
import climagrid

# Human-readable summary
print(climagrid.schema_summary())

# Machine-readable JSON (useful for SCADA integration)
from climagrid.outputs import to_json_schema
schema = to_json_schema("schema.json")
```
