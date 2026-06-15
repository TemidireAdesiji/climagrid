"""End-to-end tests for climagrid.forecast with a mocked run()."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from climagrid.forecasting import forecast
from climagrid.forecasting.config import ForecastConfig

_TARGET = "feat_thermal_aging_factor"


def _range() -> tuple[datetime, datetime]:
    return (
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 4, 30, tzinfo=timezone.utc),
    )


def test_forecast_lightgbm_end_to_end(two_asset_csv, mock_run_fn) -> None:
    pytest.importorskip("lightgbm")
    config = ForecastConfig(targets=[_TARGET], horizon_days=3, lags=[1, 2, 7])
    start, end = _range()

    result = forecast(
        two_asset_csv,
        config=config,
        history_start=start,
        history_end=end,
        run_fn=mock_run_fn,
    )

    assert not result.empty
    assert {"asset_id", "forecast_date", "horizon_day", "p10", "p50", "p90"} <= set(
        result.columns
    )
    assert sorted(result["horizon_day"].unique()) == [1, 2, 3]
    assert (result["p10"] <= result["p50"] + 1e-9).all()
    assert (result["p50"] <= result["p90"] + 1e-9).all()


def test_forecast_persistence_collapses_interval(two_asset_csv, mock_run_fn) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, model="persistence")
    start, end = _range()
    result = forecast(
        two_asset_csv,
        config=config,
        history_start=start,
        history_end=end,
        run_fn=mock_run_fn,
    )
    assert not result.empty
    # A point baseline has no spread: p10 == p50 == p90.
    assert (result["p10"] == result["p50"]).all()
    assert (result["p50"] == result["p90"]).all()


def test_forecast_empty_panel_returns_empty(two_asset_csv) -> None:
    def _empty_run(*args: object, **kwargs: object):
        import pandas as pd

        return pd.DataFrame()

    config = ForecastConfig(targets=[_TARGET], horizon_days=2)
    start, end = _range()
    result = forecast(
        two_asset_csv,
        config=config,
        history_start=start,
        history_end=end,
        run_fn=_empty_run,
    )
    assert result.empty
