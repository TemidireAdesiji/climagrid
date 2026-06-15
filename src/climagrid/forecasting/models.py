"""
LightGBM quantile forecaster.

Trains one gradient-boosted quantile regressor per (horizon, quantile) using
the direct multi-horizon strategy: horizon ``h`` predicts ``y_h{h}`` directly
from the origin-time predictors, avoiding recursive error accumulation.

LightGBM is an optional dependency (the ``[ml]`` extra). It is imported lazily
so that importing ``climagrid.forecasting`` never hard-requires it, mirroring how
``outputs.report`` guards weasyprint and ``sources.noaa_hrrr`` guards herbie.

A fitted forecaster can be saved to disk and reloaded for fast inference
(``save`` / ``load``) so a model trained once on long history can be reused
without retraining or refetching that history.

Because each quantile is trained independently, predicted quantiles can cross;
they are sorted per row after prediction so that p10 <= p50 <= p90 always holds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import predictor_columns

logger = logging.getLogger(__name__)


def quantile_column_names(config: ForecastConfig) -> list[str]:
    """Map configured quantiles to column names, e.g. 0.1 -> ``p10``."""
    return [f"p{int(round(q * 100))}" for q in config.quantiles]


def _import_lightgbm() -> Any:
    """Import lightgbm or raise a helpful error pointing at the ``[ml]`` extra."""
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - exercised only without lightgbm
        raise ImportError(
            "LightGBM is required for the 'lightgbm' forecaster. "
            'Install it with: pip install "climagrid[ml]"'
        ) from exc
    return lightgbm


class LightGBMForecaster:
    """Direct multi-horizon quantile forecaster backed by LightGBM."""

    def __init__(self, config: ForecastConfig):
        self._config = config
        self._predictors = predictor_columns(config)
        self._models: dict[tuple[int, float], Any] = {}
        self._target: str | None = None

    def fit(self, frame: pd.DataFrame, target: str) -> LightGBMForecaster:
        """
        Train one LGBMRegressor per (horizon, quantile).

        Rows whose ``y_h{h}`` target is NaN (the tail of each asset's series)
        are dropped per horizon. NaN predictors in early rows are kept;
        LightGBM handles them natively.
        """
        lightgbm = _import_lightgbm()
        self._target = target
        x_all = frame[self._predictors]

        for h in range(1, self._config.horizon_days + 1):
            target_col = f"y_h{h}"
            y_all = frame[target_col]
            mask = y_all.notna().to_numpy()
            x_h = x_all[mask]
            y_h = y_all[mask]
            if len(y_h) == 0:
                logger.warning("No training rows for horizon %d; skipping.", h)
                continue
            for q in self._config.quantiles:
                model = lightgbm.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self._config.n_estimators,
                    num_leaves=self._config.num_leaves,
                    learning_rate=self._config.learning_rate,
                    n_jobs=self._config.n_jobs,
                    random_state=self._config.random_state,
                    verbose=-1,
                )
                model.fit(x_h, y_h)
                self._models[(h, q)] = model
        return self

    def predict(self, frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """
        Produce long-form forecasts for every (row, horizon).

        Returns one row per (asset_id, origin date, horizon) with columns:
        ``asset_id``, ``origin_date``, ``forecast_date``, ``horizon_day``,
        ``target`` and one column per quantile (``p10``, ``p50``, ``p90``),
        sorted so the quantile columns are non-decreasing.
        """
        if not self._models:
            raise RuntimeError("LightGBMForecaster.predict called before fit.")
        target_name = target or self._target or ""
        quantiles = self._config.quantiles
        q_cols = quantile_column_names(self._config)
        x_all = frame[self._predictors]

        records: list[pd.DataFrame] = []
        for h in range(1, self._config.horizon_days + 1):
            if (h, quantiles[0]) not in self._models:
                continue
            preds = np.column_stack(
                [self._models[(h, q)].predict(x_all) for q in quantiles]
            )
            # Enforce non-crossing quantiles: sort each row's predictions.
            preds = np.sort(preds, axis=1)
            block = pd.DataFrame(
                {
                    "asset_id": frame["asset_id"].to_numpy(),
                    "origin_date": frame["date"].to_numpy(),
                    "horizon_day": h,
                    "target": target_name,
                }
            )
            block["forecast_date"] = block["origin_date"] + pd.to_timedelta(h, unit="D")
            for i, col in enumerate(q_cols):
                block[col] = preds[:, i]
            records.append(block)

        if not records:
            return pd.DataFrame()

        ordered_cols = [
            "asset_id",
            "origin_date",
            "forecast_date",
            "horizon_day",
            "target",
            *q_cols,
        ]
        return pd.concat(records, ignore_index=True)[ordered_cols]  # type: ignore[no-any-return]

    def save(self, path: str | Path) -> Path:
        """Persist the fitted forecaster (config, predictors, per-quantile models).

        Uses joblib (a scikit-learn dependency, always present with the ``[ml]``
        extra). Reload with :meth:`load` for inference without retraining.
        """
        import joblib

        if not self._models:
            raise RuntimeError("Cannot save a LightGBMForecaster before fit.")
        path = Path(path)
        joblib.dump(
            {
                "config": self._config,
                "predictors": self._predictors,
                "models": self._models,
                "target": self._target,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LightGBMForecaster:
        """Reload a forecaster saved by :meth:`save`."""
        import joblib

        state = joblib.load(Path(path))
        forecaster = cls(state["config"])
        forecaster._predictors = state["predictors"]
        forecaster._models = state["models"]
        forecaster._target = state["target"]
        return forecaster
