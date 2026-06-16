"""Tests for the shared conformal calibration logic.

These cover ``climagrid.forecasting.conformal`` directly, independently of any
model, so both forecasters inherit a verified calibration core.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climagrid.forecasting import conformal as cf


def test_season_index_maps_months() -> None:
    dates = pd.to_datetime(["2021-01-15", "2021-04-15", "2021-07-15", "2021-10-15"])
    assert list(cf.season_index(dates)) == [0, 1, 2, 3]


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        cf.fit_conformal("bogus", 0.8, {})


def _calib(n: int = 400) -> cf.HorizonCalib:
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=n)
    # A fixed narrow band around 0 that the spread-out targets often escape, so
    # the conformity scores are positive and the interval needs widening.
    lo = np.full(n, -0.5)
    hi = np.full(n, 0.5)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    return lo, hi, y, dates


def test_constant_width_is_positive_for_narrow_interval() -> None:
    conformal, scale = cf.fit_conformal("constant", 0.8, {1: _calib()})
    assert conformal[(1, 0)] > 0.0
    assert scale == {}


def test_normalized_stores_scale_and_delta_scales_with_width() -> None:
    conformal, scale = cf.fit_conformal("normalized", 0.8, {1: _calib()})
    assert (1, 0) in scale
    preds = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 3.0]])  # second row is wider
    delta = cf.conformal_delta("normalized", conformal, scale, 1, preds, None)
    assert isinstance(delta, np.ndarray)
    assert delta[1] > delta[0]


def test_mondrian_has_per_season_and_fallback() -> None:
    conformal, _ = cf.fit_conformal("mondrian", 0.8, {1: _calib()})
    assert (1, -1) in conformal  # pooled fallback
    forecast_dates = pd.to_datetime(["2021-07-15", "2021-01-15"])
    preds = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    delta = cf.conformal_delta("mondrian", conformal, {}, 1, preds, forecast_dates)
    assert isinstance(delta, np.ndarray)
    assert (delta >= 0.0).all()
