"""
Rolling-origin backtest and skill scoring.

The honest core of the forecasting module: a trained model is only worth
shipping if it beats persistence and climatology on out-of-sample data. This
module provides:

- ``rolling_origin_splits``: expanding-window train/test splits over the origin
  timeline with an embargo gap so a training target window never overlaps a
  test predictor window.
- ``evaluate``: fits the LightGBM model and both baselines on each fold and
  reports MAE, RMSE, skill scores, interval coverage and pinball loss per
  (fold, target, horizon).
- ``history_ablation``: reruns ``evaluate`` over several history lengths so the
  default history window is chosen on measured skill, not assumed.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd

from climagrid.forecasting.baselines import ClimatologyForecaster, PersistenceForecaster
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import build_supervised_frame
from climagrid.forecasting.models import LightGBMForecaster, quantile_column_names

logger = logging.getLogger(__name__)


class Forecaster(Protocol):
    """Minimal interface a model needs to be backtested by :func:`evaluate`."""

    def fit(self, frame: pd.DataFrame, target: str) -> Forecaster: ...

    def predict(
        self, frame: pd.DataFrame, target: str | None = ...
    ) -> pd.DataFrame: ...


class ModelFactory(Protocol):
    """Builds a fresh forecaster from a config (e.g. ``LightGBMForecaster``)."""

    def __call__(self, config: ForecastConfig) -> Forecaster: ...


def rolling_origin_splits(
    dates: list[pd.Timestamp],
    config: ForecastConfig,
    *,
    n_splits: int = 3,
    test_size_days: int = 90,
) -> list[tuple[set[pd.Timestamp], set[pd.Timestamp]]]:
    """
    Build expanding-window (train, test) origin-date splits.

    Test windows are placed back-to-back at the end of the timeline. Each
    fold's training set expands to include everything up to ``embargo`` days
    before its test window starts, where ``embargo`` defaults to the max
    horizon (so a training target at ``t + H`` never reaches a test origin).
    """
    unique_dates = sorted({pd.Timestamp(d) for d in dates})
    n = len(unique_dates)
    embargo = config.effective_embargo_days

    splits: list[tuple[set[pd.Timestamp], set[pd.Timestamp]]] = []
    for k in range(n_splits):
        test_end_idx = n - (n_splits - 1 - k) * test_size_days
        test_start_idx = test_end_idx - test_size_days
        train_end_idx = test_start_idx - embargo
        if test_start_idx <= 0 or train_end_idx <= 0:
            logger.warning(
                "Skipping fold %d: not enough history for the requested split.", k
            )
            continue
        train_dates = set(unique_dates[:train_end_idx])
        test_dates = set(unique_dates[test_start_idx:test_end_idx])
        if train_dates and test_dates:
            splits.append((train_dates, test_dates))
    return splits


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Mean pinball (quantile) loss for one quantile."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def _skill(mse_model: float, mse_baseline: float) -> float:
    """Skill score 1 - MSE_model / MSE_baseline (NaN if baseline MSE is zero)."""
    if mse_baseline <= 0.0:
        return float("nan")
    return 1.0 - mse_model / mse_baseline


def evaluate(
    panel: pd.DataFrame,
    config: ForecastConfig,
    *,
    n_splits: int = 3,
    test_size_days: int = 90,
    event_threshold: float = 0.0,
    model_factory: ModelFactory = LightGBMForecaster,
) -> pd.DataFrame:
    """
    Backtest a trained model against baselines on a daily panel.

    Returns one row per (fold, target, horizon) with point-accuracy metrics,
    skill scores versus both baselines, and interval calibration metrics.

    ``model_factory`` builds the model under test from the config; it defaults
    to ``LightGBMForecaster`` (the v1 baseline). Pass ``LSTMForecaster`` to run
    the deep model through the identical harness for an apples-to-apples
    comparison.

    ``skill_vs_persistence_events`` is the skill computed on only the active
    rows (where the actual value exceeds ``event_threshold``), and
    ``event_fraction`` is their share. For mostly-zero, sparse targets (e.g.
    ice loading) the all-rows skill is inflated by the easy zero stretches; the
    event-conditioned skill reflects how well the model predicts the rare
    events that matter, and is the honest basis for a deploy decision.
    """
    quantiles = config.quantiles
    q_cols = quantile_column_names(config)
    low_col, high_col = q_cols[0], q_cols[-1]

    rows: list[dict[str, float | int | str]] = []
    for target in config.targets:
        sup = build_supervised_frame(panel, target, config)
        if sup.empty:
            continue
        dates = [pd.Timestamp(d) for d in sup["date"].unique()]
        splits = rolling_origin_splits(
            dates, config, n_splits=n_splits, test_size_days=test_size_days
        )

        for fold, (train_dates, test_dates) in enumerate(splits):
            train = sup[sup["date"].isin(train_dates)]
            test = sup[sup["date"].isin(test_dates)]
            if train.empty or test.empty:
                continue

            model = model_factory(config).fit(train, target)
            persistence = PersistenceForecaster(config).fit(train, target)
            climatology = ClimatologyForecaster(config).fit(train, target)
            model_pred = model.predict(test, target)

            for h in range(1, config.horizon_days + 1):
                actual = test[f"y_h{h}"]
                valid = actual.notna().to_numpy()
                if valid.sum() == 0:
                    continue
                test_h = test[valid]
                y_true = test_h[f"y_h{h}"].to_numpy(dtype=float)

                y_pers = persistence.predict(test_h, h)
                y_clim = climatology.predict(test_h, h)

                pred_h = model_pred[model_pred["horizon_day"] == h]
                merged = test_h[["asset_id", "date"]].merge(
                    pred_h,
                    left_on=["asset_id", "date"],
                    right_on=["asset_id", "origin_date"],
                    how="left",
                )
                p50 = merged["p50"].to_numpy(dtype=float)
                p_low = merged[low_col].to_numpy(dtype=float)
                p_high = merged[high_col].to_numpy(dtype=float)

                mse_model = float(np.mean((p50 - y_true) ** 2))
                mse_pers = float(np.mean((y_pers - y_true) ** 2))
                mse_clim = float(np.mean((y_clim - y_true) ** 2))

                # Event-conditioned skill: only the active (above-threshold) rows.
                event = y_true > event_threshold
                if event.any():
                    skill_events = _skill(
                        float(np.mean((p50[event] - y_true[event]) ** 2)),
                        float(np.mean((y_pers[event] - y_true[event]) ** 2)),
                    )
                else:
                    skill_events = float("nan")
                coverage = float(np.mean((y_true >= p_low) & (y_true <= p_high)))
                pinball = float(
                    np.mean(
                        [
                            _pinball_loss(y_true, merged[col].to_numpy(dtype=float), q)
                            for col, q in zip(q_cols, quantiles, strict=True)
                        ]
                    )
                )

                rows.append(
                    {
                        "fold": fold,
                        "target": target,
                        "horizon_day": h,
                        "n": int(len(y_true)),
                        "mae": float(np.mean(np.abs(p50 - y_true))),
                        "rmse": float(np.sqrt(mse_model)),
                        "mae_persistence": float(np.mean(np.abs(y_pers - y_true))),
                        "mae_climatology": float(np.mean(np.abs(y_clim - y_true))),
                        "skill_vs_persistence": _skill(mse_model, mse_pers),
                        "skill_vs_persistence_events": skill_events,
                        "event_fraction": float(np.mean(event)),
                        "skill_vs_climatology": _skill(mse_model, mse_clim),
                        "interval_coverage": coverage,
                        "pinball": pinball,
                    }
                )

    return pd.DataFrame(rows)


def history_ablation(
    panel: pd.DataFrame,
    config: ForecastConfig,
    *,
    windows_years: list[int] | None = None,
    n_splits: int = 3,
    test_size_days: int = 90,
    model_factory: ModelFactory = LightGBMForecaster,
) -> pd.DataFrame:
    """
    Rerun ``evaluate`` across several history-window lengths.

    Lets the default history length be chosen on measured skill and calibration
    rather than assumed. Returns the concatenated per-fold metrics with a
    ``history_years`` column.
    """
    windows = windows_years or [10, 15, 25]
    if panel.empty:
        return pd.DataFrame()

    max_date = panel["date"].max()
    frames: list[pd.DataFrame] = []
    for window in windows:
        cutoff = max_date - pd.DateOffset(years=window)
        subset = panel[panel["date"] >= cutoff]
        result = evaluate(
            subset,
            config,
            n_splits=n_splits,
            test_size_days=test_size_days,
            model_factory=model_factory,
        )
        if not result.empty:
            result = result.assign(history_years=window)
            frames.append(result)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
