"""Tests for the LSTM quantile forecaster (the ``[dl]`` extra)."""

from __future__ import annotations

import numpy as np
import pytest

from climagrid.forecasting.backtest import evaluate
from climagrid.forecasting.config import ForecastConfig, LSTMParams
from climagrid.forecasting.dataset import build_supervised_frame

pytest.importorskip("torch")

from climagrid.forecasting.torch_models import LSTMForecaster  # noqa: E402

_TARGET = "feat_thermal_aging_factor"


def _fast_config(**overrides: object) -> ForecastConfig:
    """A small, contiguous-lag config that trains in a few epochs for tests."""
    params = LSTMParams(
        hidden_size=8,
        num_layers=1,
        epochs=4,
        batch_size=128,
        patience=2,
        seed=0,
    )
    base = {
        "targets": [_TARGET],
        "horizon_days": 2,
        "lags": [1, 2, 3, 4, 5],
        "quantiles": [0.1, 0.5, 0.9],
        "lstm": params,
    }
    base.update(overrides)
    return ForecastConfig(**base)  # type: ignore[arg-type]


def _latest_rows(sup):  # type: ignore[no-untyped-def]
    return sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)


def test_fit_predict_shapes_and_columns(daily_panel) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LSTMForecaster(config).fit(sup, _TARGET)
    latest = _latest_rows(sup)
    preds = model.predict(latest, _TARGET)

    expected_cols = {
        "asset_id",
        "origin_date",
        "forecast_date",
        "horizon_day",
        "target",
        "p10",
        "p50",
        "p90",
    }
    assert expected_cols <= set(preds.columns)
    assert len(preds) == latest["asset_id"].nunique() * config.horizon_days


def test_quantiles_are_monotonic(daily_panel) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LSTMForecaster(config).fit(sup, _TARGET)
    preds = model.predict(_latest_rows(sup), _TARGET)

    assert (preds["p10"] <= preds["p50"] + 1e-9).all()
    assert (preds["p50"] <= preds["p90"] + 1e-9).all()


def test_forecast_date_is_origin_plus_horizon(daily_panel) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LSTMForecaster(config).fit(sup, _TARGET)
    preds = model.predict(_latest_rows(sup), _TARGET)
    delta = (preds["forecast_date"] - preds["origin_date"]).dt.days
    assert (delta == preds["horizon_day"]).all()


def test_seed_determinism(daily_panel) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    latest = _latest_rows(sup)
    a = LSTMForecaster(config).fit(sup, _TARGET).predict(latest, _TARGET)
    b = LSTMForecaster(config).fit(sup, _TARGET).predict(latest, _TARGET)
    np.testing.assert_allclose(a["p50"].to_numpy(), b["p50"].to_numpy(), rtol=1e-5)


def test_calibration_widens_interval(daily_panel) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LSTMForecaster(config).fit(sup, _TARGET)
    latest = _latest_rows(sup)

    before = model.predict(latest, _TARGET)
    width_before = (before["p90"] - before["p10"]).to_numpy()

    model.calibrate(sup, _TARGET, method="constant")
    after = model.predict(latest, _TARGET)
    width_after = (after["p90"] - after["p10"]).to_numpy()

    # Conformal widening never shrinks the interval and keeps it monotone.
    assert (width_after >= width_before - 1e-9).all()
    assert (after["p10"] <= after["p50"] + 1e-9).all()
    assert (after["p50"] <= after["p90"] + 1e-9).all()


def test_save_load_round_trip(daily_panel, tmp_path) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LSTMForecaster(config).fit(sup, _TARGET).calibrate(sup, _TARGET, method="mondrian")
    latest = _latest_rows(sup)
    before = model.predict(latest, _TARGET)

    path = model.save(tmp_path / "lstm.pt")
    reloaded = LSTMForecaster.load(path)
    after = reloaded.predict(latest, _TARGET)

    np.testing.assert_allclose(
        before["p50"].to_numpy(), after["p50"].to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        before["p90"].to_numpy(), after["p90"].to_numpy(), rtol=1e-5
    )


def test_predict_before_fit_raises(daily_panel) -> None:
    config = _fast_config()
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    with pytest.raises(RuntimeError):
        LSTMForecaster(config).predict(_latest_rows(sup), _TARGET)


def test_evaluate_runs_with_lstm_factory(daily_panel) -> None:
    """The LSTM drops into the same backtest harness as LightGBM."""
    config = _fast_config()
    scores = evaluate(
        daily_panel,
        config,
        n_splits=1,
        test_size_days=60,
        model_factory=LSTMForecaster,
    )
    assert not scores.empty
    assert {"skill_vs_persistence", "interval_coverage", "pinball"} <= set(
        scores.columns
    )
    assert scores["horizon_day"].max() == config.horizon_days
