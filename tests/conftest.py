"""Shared pytest fixtures for climagrid tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climagrid.sources.base import BoundingBox

# ---------------------------------------------------------------------------
# Common time range fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def start_dt() -> datetime:
    return datetime(2024, 7, 15, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def end_dt() -> datetime:
    return datetime(2024, 7, 15, 6, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def central_texas_bbox() -> BoundingBox:
    """Small bounding box around Waco, TX — representative rural cooperative territory."""
    return BoundingBox(min_lat=31.3, max_lat=31.9, min_lon=-97.4, max_lon=-96.9)


@pytest.fixture
def single_point_bbox() -> BoundingBox:
    """Tiny bbox around a single transformer location for point queries."""
    return BoundingBox(min_lat=31.54, max_lat=31.56, min_lon=-97.16, max_lon=-97.14)


# ---------------------------------------------------------------------------
# Synthetic environmental DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_env_df() -> pd.DataFrame:
    """6-hour hourly synthetic environmental DataFrame with HRRR-like columns."""
    timestamps = pd.date_range("2024-07-15", periods=6, freq="h", tz="UTC")
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "lat": 31.55,
        "lon": -97.15,
        "timestamp": timestamps,
        "hrrr_temperature_2m": rng.uniform(28, 42, 6),       # Summer Texas heat
        "hrrr_wind_speed_10m": rng.uniform(1, 15, 6),
        "hrrr_wind_direction_10m": rng.uniform(0, 360, 6),
        "hrrr_relative_humidity_2m": rng.uniform(30, 85, 6),
        "hrrr_precipitation_rate": rng.uniform(0, 2, 6),
        "hrrr_solar_irradiance_ghi": rng.uniform(200, 900, 6),
        "hrrr_snow_depth": 0.0,
    })
    return df


@pytest.fixture
def freezing_env_df() -> pd.DataFrame:
    """48-hour dataset crossing the freezing point — for freeze-thaw tests."""
    timestamps = pd.date_range("2024-01-10", periods=48, freq="h", tz="UTC")
    # Temperature oscillates across 0°C
    temps = 2.0 * np.sin(np.linspace(0, 4 * np.pi, 48)) + 0.5  # crosses 0 multiple times
    return pd.DataFrame({
        "lat": 35.47,
        "lon": -97.52,
        "timestamp": timestamps,
        "hrrr_temperature_2m": temps,
        "hrrr_wind_speed_10m": 3.0,
        "hrrr_precipitation_rate": np.where(temps < 2, 0.5, 0.0),
        "hrrr_relative_humidity_2m": 85.0,
        "hrrr_solar_irradiance_ghi": 150.0,
    })


@pytest.fixture
def asset_env_df(synthetic_env_df) -> pd.DataFrame:
    """synthetic_env_df with asset_id column for per-asset feature tests."""
    df = synthetic_env_df.copy()
    df["asset_id"] = "TX-001"
    return df


@pytest.fixture
def multi_asset_env_df(synthetic_env_df) -> pd.DataFrame:
    """Two assets worth of hourly data stacked."""
    df1 = synthetic_env_df.copy()
    df1["asset_id"] = "TX-001"
    df2 = synthetic_env_df.copy()
    df2["asset_id"] = "TX-002"
    df2["lat"] = 31.76
    df2["lon"] = -97.05
    df2["hrrr_temperature_2m"] += 2.0   # slightly different temperatures
    return pd.concat([df1, df2], ignore_index=True)


# ---------------------------------------------------------------------------
# Sample asset registry path
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_assets_path() -> Path:
    return Path(__file__).parent.parent / "examples" / "data" / "sample_assets.csv"


@pytest.fixture
def tmp_assets_csv(tmp_path) -> Path:
    """Minimal 3-asset CSV for registry tests."""
    p = tmp_path / "test_assets.csv"
    p.write_text(
        "asset_id,lat,lon,asset_type\n"
        "A-001,31.55,-97.15,transformer\n"
        "A-002,31.76,-97.05,circuit_breaker\n"
        "A-003,35.47,-97.52,distribution_line\n"
    )
    return p


# ---------------------------------------------------------------------------
# Wildfire fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nearby_fires_df() -> pd.DataFrame:
    """Synthetic fire perimeter DataFrame simulating two active fires."""
    return pd.DataFrame([
        {
            "fire_centroid_lat": 31.65,
            "fire_centroid_lon": -97.10,
            "fire_area_ha": 500.0,
            "fire_active": True,
            "fire_discovery_dt": "2024-07-14T10:00:00Z",
            "fire_contained_dt": None,
        },
        {
            "fire_centroid_lat": 31.90,
            "fire_centroid_lon": -97.30,
            "fire_area_ha": 1200.0,
            "fire_active": False,
            "fire_discovery_dt": "2024-07-10T08:00:00Z",
            "fire_contained_dt": "2024-07-13T18:00:00Z",
        },
    ])


@pytest.fixture
def empty_fires_df() -> pd.DataFrame:
    from climagrid.sources.usfs_wfigs import WfigsAdapter
    return WfigsAdapter._empty_df()
