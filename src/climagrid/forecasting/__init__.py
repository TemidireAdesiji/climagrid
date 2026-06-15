"""
climagrid.forecast: medium-range probabilistic forecasting of asset stress.

Forecasts each asset's daily-aggregated stress features N days ahead with
prediction intervals, benchmarked against persistence and climatology
baselines. The forecasts describe environmental STRESS for maintenance lead
time, not equipment failure (see docs/validation_notes.md).

LightGBM is an optional dependency (the ``[ml]`` extra); importing this package
does not require it, but training the ``lightgbm`` model does.
"""

from __future__ import annotations

from climagrid.forecasting.api import forecast
from climagrid.forecasting.backtest import (
    evaluate,
    history_ablation,
    rolling_origin_splits,
)
from climagrid.forecasting.baselines import (
    ClimatologyForecaster,
    PersistenceForecaster,
)
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame, build_training_panel
from climagrid.forecasting.models import LightGBMForecaster

__all__ = [
    "ClimatologyForecaster",
    "ForecastConfig",
    "LightGBMForecaster",
    "PersistenceForecaster",
    "build_supervised_frame",
    "build_training_panel",
    "evaluate",
    "forecast",
    "history_ablation",
    "rolling_origin_splits",
]
