"""Tests for the persistence and climatology baselines."""

from __future__ import annotations

import numpy as np

from climagrid.forecasting.baselines import ClimatologyForecaster, PersistenceForecaster
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame

_TARGET = "feat_thermal_aging_factor"


def test_persistence_predicts_origin_value(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=5)
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    forecaster = PersistenceForecaster(config).fit(sup, _TARGET)

    pred_h1 = forecaster.predict(sup, 1)
    pred_h5 = forecaster.predict(sup, 5)
    # Persistence is horizon-independent and equals y_t.
    np.testing.assert_allclose(pred_h1, sup["y_t"].to_numpy())
    np.testing.assert_allclose(pred_h5, sup["y_t"].to_numpy())


def test_climatology_predicts_seasonal_mean(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=3)
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    forecaster = ClimatologyForecaster(config).fit(sup, _TARGET)

    pred = forecaster.predict(sup, 1)
    assert len(pred) == len(sup)
    assert np.isfinite(pred).all()
    # On a strongly seasonal series the climatology mean tracks the target.
    actual = sup["y_t"].to_numpy()
    assert np.corrcoef(pred, actual)[0, 1] > 0.5


def test_climatology_window_limits_history(daily_panel) -> None:
    config = ForecastConfig(
        targets=[_TARGET], horizon_days=2, climatology_window_years=1
    )
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    forecaster = ClimatologyForecaster(config).fit(sup, _TARGET)
    pred = forecaster.predict(sup, 1)
    assert np.isfinite(pred).all()
