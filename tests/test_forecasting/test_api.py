"""Tests for climagrid.forecast (load-and-serve from saved models)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from climagrid.forecasting import forecast
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame
from climagrid.forecasting.models import LightGBMForecaster

_TARGET = "feat_thermal_aging_factor"
_END = datetime(2024, 12, 31, tzinfo=timezone.utc)


def _train_and_save(daily_panel, tmp_path, name="model_x.joblib"):
    """Train a model offline (no fetching) and save it; return its path."""
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, lags=[1, 2, 7])
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LightGBMForecaster(config).fit(sup, _TARGET)
    return model.save(tmp_path / name)


def test_forecast_from_single_model_file(
    daily_panel, two_asset_csv, mock_run_fn, tmp_path
) -> None:
    pytest.importorskip("lightgbm")
    path = _train_and_save(daily_panel, tmp_path)

    result = forecast(two_asset_csv, path, history_end=_END, run_fn=mock_run_fn)

    assert not result.empty
    assert {"asset_id", "forecast_date", "horizon_day", "p10", "p50", "p90"} <= set(
        result.columns
    )
    assert sorted(result["horizon_day"].unique()) == [1, 2]
    assert (result["p10"] <= result["p50"] + 1e-9).all()
    assert (result["p50"] <= result["p90"] + 1e-9).all()


def test_forecast_from_loaded_instance(
    daily_panel, two_asset_csv, mock_run_fn, tmp_path
) -> None:
    pytest.importorskip("lightgbm")
    model = LightGBMForecaster.load(_train_and_save(daily_panel, tmp_path))
    result = forecast(two_asset_csv, model, history_end=_END, run_fn=mock_run_fn)
    assert not result.empty


def test_forecast_from_manifest_dir_adds_recommendation(
    daily_panel, two_asset_csv, mock_run_fn, tmp_path
) -> None:
    pytest.importorskip("lightgbm")
    _train_and_save(daily_panel, tmp_path, name="model_x.joblib")
    manifest = {
        "factors": [_TARGET],
        "horizon_days": 2,
        "min_inference_history_days": 30,
        "per_factor": {
            _TARGET: {
                "history_years": 10,
                "model_file": "model_x.joblib",
                "recommendation": "lightgbm",
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    result = forecast(two_asset_csv, tmp_path, history_end=_END, run_fn=mock_run_fn)

    assert not result.empty
    assert "recommendation" in result.columns
    assert (result["recommendation"] == "lightgbm").all()


def test_forecast_empty_when_no_recent_data(
    daily_panel, two_asset_csv, tmp_path
) -> None:
    pytest.importorskip("lightgbm")
    path = _train_and_save(daily_panel, tmp_path)

    def _empty_run(*args: object, **kwargs: object):
        import pandas as pd

        return pd.DataFrame()

    result = forecast(two_asset_csv, path, history_end=_END, run_fn=_empty_run)
    assert result.empty


def test_forecast_missing_manifest_raises(two_asset_csv, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        forecast(two_asset_csv, tmp_path)  # empty dir, no manifest.json
