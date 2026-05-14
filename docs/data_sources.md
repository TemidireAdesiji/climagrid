# Data Sources

climagrid fetches from five U.S. government open-data APIs. All are free and require no registration except NOAA NCEI CDO (free token, issued in seconds).

---

## NASA POWER

**Module:** `climagrid.sources.nasa_power.NasaPowerAdapter`

NASA's Prediction of Worldwide Energy Resources (POWER) project provides hourly surface meteorology derived from the MERRA-2 atmospheric reanalysis at any lat/lon point globally.

| Property | Value |
|----------|-------|
| Temporal resolution | Hourly |
| Spatial resolution | ~50 km (MERRA-2 native) |
| Latency | ~3 days |
| Coverage | Global |
| API key | Not required |
| Endpoint | `https://power.larc.nasa.gov/api/temporal/hourly/point` |

**Columns produced:**

| Column | Units | Description |
|--------|-------|-------------|
| `nasa_temperature_2m` | °C | 2-metre air temperature |
| `nasa_wind_speed_10m` | m/s | 10-metre wind speed |
| `nasa_solar_irradiance_ghi` | W/m² | Global horizontal irradiance |
| `nasa_relative_humidity_2m` | % | Relative humidity at 2 m |
| `nasa_precipitation` | mm | Hourly precipitation |

**Best for:** Global coverage, no API key, consistent long historical records back to 1981. Preferred for rural assets outside HRRR coverage or for long-term trend analysis.

---

## NOAA HRRR

**Module:** `climagrid.sources.noaa_hrrr.NoaaHrrrAdapter`

NOAA's High-Resolution Rapid Refresh (HRRR) is a 3-km CONUS numerical weather prediction model updated hourly. It is the highest-resolution publicly available NWP product over the continental United States.

| Property | Value |
|----------|-------|
| Temporal resolution | Hourly |
| Spatial resolution | 3 km |
| Latency | ~1 hour |
| Coverage | CONUS only |
| API key | Not required |
| Library | `herbie-data` (optional dep `[noaa-nwp]`) |

**Columns produced:**

| Column | Units | Description |
|--------|-------|-------------|
| `hrrr_temperature_2m` | °C | 2-metre air temperature |
| `hrrr_wind_speed_10m` | m/s | 10-metre wind speed |
| `hrrr_wind_direction_10m` | degrees | Wind direction (meteorological) |
| `hrrr_relative_humidity_2m` | % | Relative humidity |
| `hrrr_precipitation_rate` | mm/hr | Hourly precipitation |
| `hrrr_solar_irradiance_ghi` | W/m² | Downward shortwave radiation |
| `hrrr_snow_depth` | m | Snow depth |

**Best for:** High spatial detail within CONUS. Required if your predictive model needs to distinguish neighboring substations separated by less than 50 km. Requires `pip install "climagrid[noaa-nwp]"`.

---

## NOAA NCEI CDO

**Module:** `climagrid.sources.noaa_ncei.NoaaNceiAdapter`

NOAA's National Centers for Environmental Information Climate Data Online (CDO) API provides historical surface station observations from the Global Historical Climatology Network (GHCN) and Integrated Surface Database (ISD).

| Property | Value |
|----------|-------|
| Temporal resolution | Hourly or daily |
| Spatial resolution | Station-based (nearest station matched) |
| Latency | ~1 day |
| Coverage | Global |
| API key | Free, register at ncdc.noaa.gov/cdo-web/token |
| Env var | `NOAA_CDO_TOKEN` (set this and the adapter picks it up automatically) |

**Setup:**

```bash
export NOAA_CDO_TOKEN="your_token_here"
```

Or pass it directly:

```python
from climagrid.sources.noaa_ncei import NceiAdapter
adapter = NceiAdapter(api_token="your_token_here")
```

**Columns produced:**

| Column | Units | Description |
|--------|-------|-------------|
| `ncei_temperature_dry_bulb` | °C | Dry-bulb temperature |
| `ncei_wind_speed` | m/s | Wind speed |
| `ncei_precipitation_hourly` | mm | Hourly precipitation |
| `ncei_relative_humidity` | % | Relative humidity |

**Best for:** Ground truth validation. Actual station measurements vs. model-derived data from NASA POWER or HRRR. Use for bias correction or audit trails in regulatory filings.

---

## USDA NRCS AWDB

**Module:** `climagrid.sources.usda_nrcs.UsdaNrcsAdapter`

NOAA's Natural Resources Conservation Service (NRCS) operates the Soil Climate Analysis Network (SCAN) and SNOwpack TELemetry (SNOTEL) networks. The Automated Water Database (AWDB) REST API provides hourly soil moisture, soil temperature, and snow water equivalent from ~900 stations across the western U.S.

| Property | Value |
|----------|-------|
| Temporal resolution | Hourly |
| Spatial resolution | Station-based |
| Coverage | Primarily western CONUS |
| API key | Not required |
| Endpoint | `https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1` |

**Columns produced:**

| Column | Units | Description |
|--------|-------|-------------|
| `nrcs_soil_moisture_pct` | % | Volumetric soil moisture |
| `nrcs_soil_temperature` | °C | Soil temperature at 2-inch depth |
| `nrcs_snow_water_equivalent` | mm | Snow water equivalent |
| `nrcs_station_distance_km` | km | Distance to nearest NRCS station |

**Best for:** Soil saturation risk (ground stability for pole foundations, underground cable performance), spring snowmelt flood risk, cold-climate cooperative service territories.

---

## USFS NIFC WFIGS

**Module:** `climagrid.sources.usfs_wfigs.WfigsAdapter`

The National Interagency Fire Center (NIFC) Wildland Fire Incident Geospatial Service (WFIGS) provides near-real-time wildfire perimeters as GeoJSON polygons.

| Property | Value |
|----------|-------|
| Temporal resolution | Updated multiple times daily |
| Coverage | CONUS + Alaska + Hawaii |
| API key | Not required |
| Endpoint | `https://services3.arcgis.com/T4QMspbfLg3qTGWY/...` |

**Columns produced:**

| Column | Units | Description |
|--------|-------|-------------|
| `wfigs_nearest_fire_km` | km | Distance to nearest active fire perimeter edge |
| `wfigs_fire_active` | bool | Any active fire within 50 km |
| `wfigs_fire_area_ha` | ha | Area of nearest active fire |

**Best for:** Western U.S. cooperatives, transmission lines traversing WUI (wildland–urban interface) terrain, rapid situational awareness during red-flag conditions.

The `compute_proximity()` standalone function can be used independently of the adapter to compute fire risk scores from a pre-fetched perimeter DataFrame.
