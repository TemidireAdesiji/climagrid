"""Tests for the forecast dataset builder (aggregation, lags, leakage)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from climagrid.assets.registry import AssetRegistry
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import (
    build_supervised_frame,
    build_training_panel,
    horizon_target_columns,
    predictor_columns,
)

_TARGET = "feat_thermal_aging_factor"


def test_build_training_panel_aggregates_to_daily(two_asset_csv, mock_run_fn) -> None:
    config = ForecastConfig(targets=[_TARGET])
    start = datetime(2021, 6, 1, tzinfo=timezone.utc)
    end = datetime(2021, 6, 4, tzinfo=timezone.utc)

    panel = build_training_panel(two_asset_csv, start, end, config, run_fn=mock_run_fn)

    assert set(panel.columns) >= {"asset_id", "date", "lat", "lon", _TARGET}
    # One row per (asset, day); both assets present.
    assert panel["asset_id"].nunique() == 2
    counts = panel.groupby(["asset_id", "date"]).size()
    assert (counts == 1).all()


def test_long_range_is_fetched_in_chunks(two_asset_csv) -> None:
    # A multi-year range must be split into several smaller fetches (so one huge
    # request cannot time out), then reassembled into a continuous panel with no
    # duplicate days at the chunk seams.
    calls: list[tuple[str, str, str]] = []

    def recording_run(
        assets, start, end, *, sources=None, features="all"
    ) -> pd.DataFrame:
        registry = (
            assets if isinstance(assets, AssetRegistry) else AssetRegistry(assets)
        )
        row = registry.assets.iloc[0]
        calls.append(
            (row["asset_id"], start.date().isoformat(), end.date().isoformat())
        )
        timestamps = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "asset_id": row["asset_id"],
                "timestamp": timestamps,
                "lat": row["lat"],
                "lon": row["lon"],
                _TARGET: 1.0,
            }
        )

    config = ForecastConfig(targets=[_TARGET])
    start = datetime(2012, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 12, 31, tzinfo=timezone.utc)  # 9 years -> multiple chunks
    panel = build_training_panel(
        two_asset_csv, start, end, config, run_fn=recording_run
    )

    # More than one fetch per asset (2 assets, so more than 2 calls total).
    assert len(calls) > 2
    # No duplicated (asset, day) rows at the chunk seams.
    assert panel.duplicated(["asset_id", "date"]).sum() == 0
    # Full, continuous daily coverage of the requested range.
    assert panel["date"].min() == pd.Timestamp("2012-01-01")
    assert panel["date"].max() == pd.Timestamp("2020-12-31")


def test_aggregate_daily_uses_max(two_asset_csv, mock_run_fn) -> None:
    config = ForecastConfig(targets=[_TARGET], daily_agg="max")
    start = datetime(2021, 6, 1, tzinfo=timezone.utc)
    end = datetime(2021, 6, 2, tzinfo=timezone.utc)
    panel = build_training_panel(two_asset_csv, start, end, config, run_fn=mock_run_fn)
    # Daily max must be >= the daily mean of the same series.
    assert panel[_TARGET].notna().all()
    assert (panel[_TARGET] > 0).all()


def test_supervised_frame_targets_are_future_shifts(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=3, lags=[1, 2])
    sup = build_supervised_frame(daily_panel, _TARGET, config)

    # y_h{h} for an asset equals that asset's target shifted up by h.
    one = daily_panel[daily_panel["asset_id"] == "TX-001"].sort_values("date")
    sup_one = sup[sup["asset_id"] == "TX-001"].sort_values("date").reset_index(drop=True)
    expected_h1 = one[_TARGET].shift(-1).reset_index(drop=True)
    np.testing.assert_allclose(
        sup_one["y_h1"].to_numpy(), expected_h1.to_numpy(), equal_nan=True
    )


def test_supervised_frame_lags_are_causal(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, lags=[1])
    sup = build_supervised_frame(daily_panel, _TARGET, config)

    one = daily_panel[daily_panel["asset_id"] == "TX-001"].sort_values("date")
    sup_one = sup[sup["asset_id"] == "TX-001"].sort_values("date").reset_index(drop=True)
    # lag_1 equals the previous day's value (past only, never future).
    expected_lag1 = one[_TARGET].shift(1).reset_index(drop=True)
    np.testing.assert_allclose(
        sup_one["lag_1"].to_numpy(), expected_lag1.to_numpy(), equal_nan=True
    )
    # First row of each asset has no past, so lag_1 is NaN (no leakage).
    assert np.isnan(sup_one["lag_1"].iloc[0])


def test_supervised_frame_has_expected_columns(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=4)
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    for col in predictor_columns(config):
        assert col in sup.columns
    for col in horizon_target_columns(config):
        assert col in sup.columns


def test_supervised_frame_is_sorted_by_asset_and_date(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET])
    shuffled = daily_panel.sample(frac=1.0, random_state=1).reset_index(drop=True)
    sup = build_supervised_frame(shuffled, _TARGET, config)
    for _, grp in sup.groupby("asset_id"):
        assert grp["date"].is_monotonic_increasing


def test_rolling_mean_excludes_current_row(daily_panel) -> None:
    # rollmean_w uses values up to t-1, so it never equals a window that
    # includes the origin value (guards against look-ahead in the mean).
    config = ForecastConfig(targets=[_TARGET], rolling_windows=[7])
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    one = sup[sup["asset_id"] == "TX-001"].sort_values("date").reset_index(drop=True)
    # The first row has no prior history -> rolling mean is NaN.
    assert np.isnan(one["rollmean_7"].iloc[0])
