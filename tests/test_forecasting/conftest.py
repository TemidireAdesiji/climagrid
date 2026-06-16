"""Shared fixtures for forecast tests: synthetic daily panels and a mock run()."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from climagrid.assets.registry import AssetRegistry

_TARGET = "feat_thermal_aging_factor"


def _seasonal_daily_series(dates: pd.DatetimeIndex, seed: int) -> np.ndarray:
    """A positive, seasonal, autocorrelated daily stress series."""
    rng = np.random.default_rng(seed)
    doy = dates.dayofyear.to_numpy(dtype=float)
    seasonal = 1.0 + 0.6 * np.sin(2.0 * np.pi * (doy - 200.0) / 365.25)
    wiggle = rng.normal(0.0, 0.04, size=len(dates))
    return np.clip(seasonal + wiggle, 0.05, None)


@pytest.fixture
def daily_panel() -> pd.DataFrame:
    """Two assets, ~4 years of daily thermal-aging values (no fetching)."""
    dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    frames = []
    for i, (asset_id, lat, lon) in enumerate(
        [("TX-001", 31.55, -97.15), ("TX-002", 35.47, -97.52)]
    ):
        frames.append(
            pd.DataFrame(
                {
                    "asset_id": asset_id,
                    "date": dates,
                    "lat": lat,
                    "lon": lon,
                    _TARGET: _seasonal_daily_series(dates, seed=42 + i),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def two_asset_csv(tmp_path) -> object:
    """A minimal 2-asset registry CSV path."""
    path = tmp_path / "assets.csv"
    path.write_text(
        "asset_id,lat,lon\nTX-001,31.55,-97.15\nTX-002,35.47,-97.52\n"
    )
    return path


@pytest.fixture
def mock_run_fn():
    """A stand-in for climagrid.run that returns hourly synthetic data.

    Mirrors the orchestrator contract: given a single-asset registry plus a
    date range, return an hourly wide frame with asset_id, timestamp, lat, lon
    and the thermal-aging feature column.
    """

    def _run(
        assets: AssetRegistry,
        start_dt: datetime,
        end_dt: datetime,
        *,
        sources: list[str] | None = None,
        features: list[str] | str = "all",
    ) -> pd.DataFrame:
        registry = (
            assets if isinstance(assets, AssetRegistry) else AssetRegistry(assets)
        )
        row = registry.assets.iloc[0]
        timestamps = pd.date_range(start_dt, end_dt, freq="h", tz="UTC")
        if len(timestamps) == 0:
            return pd.DataFrame()
        rng = np.random.default_rng(7)
        doy = timestamps.dayofyear.to_numpy(dtype=float)
        values = 1.0 + 0.5 * np.sin(2.0 * np.pi * doy / 365.25)
        values = values + rng.normal(0.0, 0.03, size=len(timestamps))
        return pd.DataFrame(
            {
                "asset_id": row["asset_id"],
                "timestamp": timestamps,
                "lat": row["lat"],
                "lon": row["lon"],
                _TARGET: np.clip(values, 0.05, None),
            }
        )

    return _run
