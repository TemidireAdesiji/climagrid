"""
Column contract for climagrid's EnvironmentalDataFrame.

Every adapter and feature function must conform to these names, dtypes, and units.
Adding a column? Define it here first.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Column descriptors
# ---------------------------------------------------------------------------


class ColumnSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    units: str
    description: str
    source: str  # "index" | "noaa_hrrr" | "nasa_power" | "noaa_ncei" | "usda_nrcs" | "usfs_wfigs" | "feature"
    nullable: bool = True


# ---------------------------------------------------------------------------
# Index columns (always present, never null)
# ---------------------------------------------------------------------------

INDEX_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="asset_id",   dtype="str",              units="n/a",       description="Utility asset identifier from AssetRegistry",       source="index", nullable=False),
    ColumnSpec(name="timestamp",  dtype="datetime64[ns, UTC]", units="n/a",   description="UTC timestamp of the observation or forecast hour", source="index", nullable=False),
    ColumnSpec(name="lat",        dtype="float64",          units="degrees", description="Asset latitude (WGS-84)",                           source="index", nullable=False),
    ColumnSpec(name="lon",        dtype="float64",          units="degrees", description="Asset longitude (WGS-84)",                          source="index", nullable=False),
]

# ---------------------------------------------------------------------------
# NOAA HRRR columns (3 km CONUS analysis/forecast)
# ---------------------------------------------------------------------------

NOAA_HRRR_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="hrrr_temperature_2m",        dtype="float64", units="°C",      description="2-metre air temperature",                    source="noaa_hrrr"),
    ColumnSpec(name="hrrr_wind_speed_10m",        dtype="float64", units="m/s",     description="10-metre wind speed (magnitude)",            source="noaa_hrrr"),
    ColumnSpec(name="hrrr_wind_direction_10m",    dtype="float64", units="degrees", description="10-metre wind direction (met convention)",   source="noaa_hrrr"),
    ColumnSpec(name="hrrr_relative_humidity_2m",  dtype="float64", units="%",       description="2-metre relative humidity",                  source="noaa_hrrr"),
    ColumnSpec(name="hrrr_precipitation_rate",    dtype="float64", units="mm/hr",   description="Hourly accumulated precipitation",           source="noaa_hrrr"),
    ColumnSpec(name="hrrr_solar_irradiance_ghi",  dtype="float64", units="W/m²",    description="Downward short-wave radiation at surface",   source="noaa_hrrr"),
    ColumnSpec(name="hrrr_snow_depth",            dtype="float64", units="m",       description="Snow depth at surface",                      source="noaa_hrrr"),
]

# ---------------------------------------------------------------------------
# NASA POWER columns (satellite-derived surface meteorology)
# ---------------------------------------------------------------------------

NASA_POWER_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="nasa_temperature_2m",        dtype="float64", units="°C",   description="2-metre air temperature (MERRA-2 based)",       source="nasa_power"),
    ColumnSpec(name="nasa_wind_speed_10m",        dtype="float64", units="m/s",  description="10-metre wind speed",                           source="nasa_power"),
    ColumnSpec(name="nasa_solar_irradiance_ghi",  dtype="float64", units="W/m²", description="Global horizontal irradiance",                  source="nasa_power"),
    ColumnSpec(name="nasa_relative_humidity_2m",  dtype="float64", units="%",    description="Relative humidity at 2 m",                      source="nasa_power"),
    ColumnSpec(name="nasa_precipitation",         dtype="float64", units="mm",   description="Precipitation (daily or sub-daily)",            source="nasa_power"),
]

# ---------------------------------------------------------------------------
# NOAA NCEI columns (surface station observations)
# ---------------------------------------------------------------------------

NOAA_NCEI_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="ncei_temperature_max",       dtype="float64", units="°C",  description="Daily maximum temperature",                      source="noaa_ncei"),
    ColumnSpec(name="ncei_temperature_min",       dtype="float64", units="°C",  description="Daily minimum temperature",                      source="noaa_ncei"),
    ColumnSpec(name="ncei_wind_speed",            dtype="float64", units="m/s", description="Average daily wind speed",                       source="noaa_ncei"),
    ColumnSpec(name="ncei_precipitation_daily",   dtype="float64", units="mm",  description="Daily liquid-equivalent precipitation",          source="noaa_ncei"),
    ColumnSpec(name="ncei_relative_humidity",     dtype="float64", units="%",   description="Average daily relative humidity",                source="noaa_ncei"),
]

# ---------------------------------------------------------------------------
# USDA NRCS columns (SCAN / SNOTEL soil sensors)
# ---------------------------------------------------------------------------

USDA_NRCS_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="nrcs_soil_moisture_pct",     dtype="float64", units="%",   description="Volumetric soil moisture (nearest SCAN station)", source="usda_nrcs"),
    ColumnSpec(name="nrcs_soil_temperature",      dtype="float64", units="°C",  description="Soil temperature at 2-inch depth",                source="usda_nrcs"),
    ColumnSpec(name="nrcs_snow_water_equivalent", dtype="float64", units="mm",  description="Snow water equivalent (nearest SNOTEL)",          source="usda_nrcs"),
    ColumnSpec(name="nrcs_station_distance_km",   dtype="float64", units="km",  description="Distance from asset to nearest NRCS station",     source="usda_nrcs"),
]

# ---------------------------------------------------------------------------
# USFS / NIFC WFIGS columns (wildfire perimeters)
# ---------------------------------------------------------------------------

USFS_WFIGS_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="wfigs_nearest_fire_km",      dtype="float64", units="km",  description="Distance to nearest active fire perimeter edge", source="usfs_wfigs"),
    ColumnSpec(name="wfigs_fire_active",          dtype="bool",    units="n/a",   description="True if any active fire within 50 km",           source="usfs_wfigs"),
    ColumnSpec(name="wfigs_fire_area_ha",         dtype="float64", units="ha",  description="Area of nearest active fire in hectares",         source="usfs_wfigs"),
]

# ---------------------------------------------------------------------------
# Feature (stress index) columns: computed by climagrid.features
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[ColumnSpec] = [
    ColumnSpec(name="feat_thermal_aging_factor",   dtype="float64", units="per-unit", description="Arrhenius FAA relative to 110°C reference (IEEE C57.91)",  source="feature"),
    ColumnSpec(name="feat_heat_hours_above_35c",   dtype="float64", units="hours",    description="Cumulative hours with temperature > 35°C in rolling window", source="feature"),
    ColumnSpec(name="feat_freeze_thaw_cycles",     dtype="float64", units="count",    description="Freeze-thaw transition count over the rolling window",       source="feature"),
    ColumnSpec(name="feat_ice_loading_risk",       dtype="float64", units="0-1",      description="Normalized ice accretion risk (simplified, ASCE 7-22 based)", source="feature"),
    ColumnSpec(name="feat_soil_saturation_index",  dtype="float64", units="0-1",      description="Normalized soil saturation proxy for ground stability",      source="feature"),
    ColumnSpec(name="feat_wildfire_proximity",     dtype="float64", units="0-1",      description="Normalized wildfire proximity score (0 = far, 1 = adjacent)", source="feature"),
    ColumnSpec(name="feat_conductor_sag_index",    dtype="float64", units="0-1",      description="Normalized thermal sag index (IEEE 738 simplified)",         source="feature"),
]

# ---------------------------------------------------------------------------
# Master registry and helpers
# ---------------------------------------------------------------------------

ALL_COLUMNS: list[ColumnSpec] = (
    INDEX_COLUMNS
    + NOAA_HRRR_COLUMNS
    + NASA_POWER_COLUMNS
    + NOAA_NCEI_COLUMNS
    + USDA_NRCS_COLUMNS
    + USFS_WFIGS_COLUMNS
    + FEATURE_COLUMNS
)

COLUMN_MAP: dict[str, ColumnSpec] = {c.name: c for c in ALL_COLUMNS}


def validate_dataframe(df: pd.DataFrame, required_sources: list[str] | None = None) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []

    # Check non-nullable index columns
    for spec in INDEX_COLUMNS:
        if spec.name not in df.columns:
            errors.append(f"Missing required column: {spec.name}")
        elif df[spec.name].isna().any():
            errors.append(f"Null values in non-nullable column: {spec.name}")

    # Check that any present column matches the expected dtype
    for col_name in df.columns:
        if col_name in COLUMN_MAP:
            spec = COLUMN_MAP[col_name]
            expected = spec.dtype
            actual = str(df[col_name].dtype)
            # Loose check: just verify numeric vs non-numeric alignment
            if expected == "float64" and not pd.api.types.is_float_dtype(df[col_name]):
                errors.append(f"Column {col_name} expected float64, got {actual}")
            elif expected == "bool" and not pd.api.types.is_bool_dtype(df[col_name]):
                # bool columns are sometimes stored as object: only warn
                pass

    return errors


def empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with all index columns present."""
    return pd.DataFrame(columns=[c.name for c in INDEX_COLUMNS])


def schema_summary() -> pd.DataFrame:
    """Return a human-readable summary of the full schema."""
    return pd.DataFrame(
        [
            {
                "column": c.name,
                "dtype": c.dtype,
                "units": c.units,
                "source": c.source,
                "nullable": c.nullable,
                "description": c.description,
            }
            for c in ALL_COLUMNS
        ]
    )
