"""Tests for the rolling-origin backtest and skill scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climagrid.forecasting.backtest import (
    _pinball_loss,
    evaluate,
    rolling_origin_splits,
)
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame

_TARGET = "feat_thermal_aging_factor"


def test_splits_respect_embargo() -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=7)  # embargo defaults to 7
    dates = list(pd.date_range("2020-01-01", "2022-12-31", freq="D"))
    splits = rolling_origin_splits(dates, config, n_splits=2, test_size_days=60)

    assert len(splits) >= 1
    for train_dates, test_dates in splits:
        # No overlap between train and test origins.
        assert train_dates.isdisjoint(test_dates)
        gap = (min(test_dates) - max(train_dates)).days
        # Gap between last train origin and first test origin >= embargo.
        assert gap >= config.effective_embargo_days


def test_pinball_loss_nonnegative() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=100)
    y_pred = rng.normal(size=100)
    for q in (0.1, 0.5, 0.9):
        assert _pinball_loss(y_true, y_pred, q) >= 0.0


def test_evaluate_returns_metrics(daily_panel) -> None:
    pytest.importorskip("lightgbm")
    config = ForecastConfig(targets=[_TARGET], horizon_days=3, lags=[1, 2, 7])
    result = evaluate(daily_panel, config, n_splits=2, test_size_days=45)

    assert not result.empty
    expected = {
        "fold",
        "target",
        "horizon_day",
        "mae",
        "rmse",
        "skill_vs_persistence",
        "skill_vs_climatology",
        "interval_coverage",
        "pinball",
    }
    assert expected <= set(result.columns)
    # Coverage is a probability.
    assert ((result["interval_coverage"] >= 0) & (result["interval_coverage"] <= 1)).all()


def test_evaluate_supervised_frame_nonempty(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2)
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    assert not sup.empty
