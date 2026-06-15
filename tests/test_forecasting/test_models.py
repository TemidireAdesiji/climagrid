"""Tests for the LightGBM quantile forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame
from climagrid.forecasting.models import LightGBMForecaster, quantile_column_names

pytest.importorskip("lightgbm")

_TARGET = "feat_thermal_aging_factor"


def test_quantile_column_names() -> None:
    config = ForecastConfig(quantiles=[0.1, 0.5, 0.9])
    assert quantile_column_names(config) == ["p10", "p50", "p90"]


def test_fit_predict_shapes_and_columns(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=3, lags=[1, 2, 7])
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LightGBMForecaster(config).fit(sup, _TARGET)

    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
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
    # One row per (asset, horizon).
    assert len(preds) == latest["asset_id"].nunique() * config.horizon_days


def test_quantiles_are_monotonic(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, quantiles=[0.1, 0.5, 0.9])
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LightGBMForecaster(config).fit(sup, _TARGET)
    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    preds = model.predict(latest, _TARGET)

    assert (preds["p10"] <= preds["p50"] + 1e-9).all()
    assert (preds["p50"] <= preds["p90"] + 1e-9).all()


def test_forecast_date_is_origin_plus_horizon(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=3)
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LightGBMForecaster(config).fit(sup, _TARGET)
    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    preds = model.predict(latest, _TARGET)

    delta = (preds["forecast_date"] - preds["origin_date"]).dt.days
    assert (delta == preds["horizon_day"]).all()


def test_predict_before_fit_raises() -> None:
    config = ForecastConfig(targets=[_TARGET])
    model = LightGBMForecaster(config)
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"asset_id": [], "date": []}), _TARGET)


def test_save_load_roundtrip(daily_panel, tmp_path) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, lags=[1, 2, 7])
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    model = LightGBMForecaster(config).fit(sup, _TARGET)
    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    before = model.predict(latest, _TARGET)

    path = model.save(tmp_path / "model.joblib")
    assert path.exists()

    reloaded = LightGBMForecaster.load(path)
    after = reloaded.predict(latest, _TARGET)
    pd.testing.assert_frame_equal(before, after)


def test_save_before_fit_raises() -> None:
    config = ForecastConfig(targets=[_TARGET])
    with pytest.raises(RuntimeError):
        LightGBMForecaster(config).save("unused.joblib")


def _train_calib_split(daily_panel, config):
    sup = build_supervised_frame(daily_panel, _TARGET, config)
    dates = sorted(sup["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    return sup, sup[sup["date"] <= cut], sup[sup["date"] > cut]


def test_calibrate_before_fit_raises() -> None:
    config = ForecastConfig(targets=[_TARGET])
    with pytest.raises(RuntimeError):
        LightGBMForecaster(config).calibrate(pd.DataFrame(), _TARGET)


def test_calibrate_keeps_median_and_monotonic(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=3, lags=[1, 2, 7])
    sup, train, calib = _train_calib_split(daily_panel, config)
    model = LightGBMForecaster(config).fit(train, _TARGET)
    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    before = model.predict(latest, _TARGET).sort_values(
        ["asset_id", "horizon_day"]
    ).reset_index(drop=True)

    model.calibrate(calib, _TARGET)
    # A conformal width is set for every horizon (keys are (horizon, bin)).
    horizons = {h for (h, _bin) in model._conformal}
    assert horizons == set(range(1, config.horizon_days + 1))
    assert all(np.isfinite(v) for v in model._conformal.values())

    after = model.predict(latest, _TARGET).sort_values(
        ["asset_id", "horizon_day"]
    ).reset_index(drop=True)
    # The median is untouched and quantiles stay non-crossing after widening.
    np.testing.assert_allclose(before["p50"].to_numpy(), after["p50"].to_numpy())
    assert (after["p10"] <= after["p50"] + 1e-9).all()
    assert (after["p50"] <= after["p90"] + 1e-9).all()


def test_save_load_preserves_calibration(daily_panel, tmp_path) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, lags=[1, 2, 7])
    sup, train, calib = _train_calib_split(daily_panel, config)
    model = LightGBMForecaster(config).fit(train, _TARGET).calibrate(calib, _TARGET)
    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    before = model.predict(latest, _TARGET)

    loaded = LightGBMForecaster.load(model.save(tmp_path / "cal.joblib"))
    assert loaded._conformal == model._conformal
    assert loaded._conformal_method == model._conformal_method
    pd.testing.assert_frame_equal(before, loaded.predict(latest, _TARGET))


def test_mondrian_calibration_per_season(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, lags=[1, 2, 7])
    sup, train, calib = _train_calib_split(daily_panel, config)
    model = LightGBMForecaster(config).fit(train, _TARGET).calibrate(
        calib, _TARGET, method="mondrian"
    )
    assert model._conformal_method == "mondrian"

    # Per (horizon, season) widths, with a -1 pooled fallback for each horizon.
    seasons_by_h: dict[int, set[int]] = {}
    for h, season in model._conformal:
        seasons_by_h.setdefault(h, set()).add(season)
    for h in range(1, config.horizon_days + 1):
        assert -1 in seasons_by_h[h]
        assert seasons_by_h[h] & {0, 1, 2, 3}

    latest = sup.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    out = model.predict(latest, _TARGET)
    assert (out["p10"] <= out["p50"] + 1e-9).all()
    assert (out["p50"] <= out["p90"] + 1e-9).all()


def test_unknown_calibration_method_raises(daily_panel) -> None:
    config = ForecastConfig(targets=[_TARGET], horizon_days=2, lags=[1, 2, 7])
    sup, train, calib = _train_calib_split(daily_panel, config)
    model = LightGBMForecaster(config).fit(train, _TARGET)
    with pytest.raises(ValueError):
        model.calibrate(calib, _TARGET, method="bogus")
