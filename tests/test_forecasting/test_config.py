"""Tests for ForecastConfig validation and helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from climagrid.forecasting.config import ForecastConfig


def test_default_target_and_embargo() -> None:
    config = ForecastConfig()
    assert config.targets == ["feat_thermal_aging_factor"]
    # Embargo defaults to the max horizon.
    assert config.effective_embargo_days == config.horizon_days


def test_required_features_maps_targets() -> None:
    config = ForecastConfig(
        targets=["feat_thermal_aging_factor", "feat_freeze_thaw_cycles"]
    )
    assert config.required_features() == ["freeze_thaw", "thermal"]


def test_unknown_target_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(targets=["feat_not_real"])


def test_quantiles_must_include_median() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(quantiles=[0.1, 0.9])


def test_quantiles_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(quantiles=[0.0, 0.5, 1.0])


def test_explicit_embargo_overrides_default() -> None:
    config = ForecastConfig(horizon_days=7, embargo_days=14)
    assert config.effective_embargo_days == 14


def test_min_inference_history_days() -> None:
    config = ForecastConfig(lags=[1, 2, 30], rolling_windows=[7, 60])
    # The longest predictor lookback (here the 60-day rolling window).
    assert config.min_inference_history_days == 60


def test_calibration_defaults() -> None:
    config = ForecastConfig()
    assert config.calibrate_intervals is False
    assert config.calibration_method == "mondrian"
    assert config.calibration_days == 365


def test_invalid_calibration_method_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastConfig(calibration_method="bogus")
