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
        # Conformal calibration of the outer (p_lo, p_hi) interval; populated by
        # calibrate(), empty otherwise. Keyed by (horizon, season bin): bin 0 for
        # the global "constant"/"normalized" methods, bins 0-3 (DJF/MAM/JJA/SON)
        # plus a -1 fallback for the season-conditional "mondrian" method.
        self._conformal: dict[tuple[int, int], float] = {}
        self._conformal_scale: dict[tuple[int, int], float] = {}
        self._conformal_method: str = ""

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

    _MIN_SEASON_BIN = 50  # min calibration points to trust a per-season width

    @staticmethod
    def _season_index(dates: Any) -> np.ndarray:
        """Meteorological season per date: 0=DJF, 1=MAM, 2=JJA, 3=SON."""
        months = pd.DatetimeIndex(pd.to_datetime(dates)).month.to_numpy()
        return ((months % 12) // 3).astype(int)

    def calibrate(
        self,
        frame: pd.DataFrame,
        target: str | None = None,
        *,
        method: str = "normalized",
    ) -> LightGBMForecaster:
        """
        Conformalize the outer prediction interval (Romano et al. 2019, CQR).

        Uses a held-out calibration ``frame`` (rows NOT seen during fit) to
        adjust the lowest/highest quantile bounds per horizon so the interval
        attains its nominal coverage (``q_hi - q_lo``, e.g. 0.80 for p10-p90)
        out of sample. The calibration set should span a full seasonal cycle.
        The conformity score is ``E = max(p_lo - y, y - p_hi)`` and ``Q`` is its
        ``ceil((n + 1) * level) / n`` empirical quantile; predict() widens by Q.

        method:
            ``"normalized"`` (default) scales the score by the model's own
            interval width ``(p_hi - p_lo) + c`` so the widening adapts to local
            uncertainty. ``"constant"`` applies a single additive ``Q`` per
            horizon (marginal coverage only). ``"mondrian"`` fits a separate
            additive ``Q`` per (horizon, meteorological season) keyed on the
            forecast date, targeting per-season coverage (helps where a season,
            e.g. summer, is otherwise under-covered); seasons with too few
            calibration points fall back to the pooled per-horizon ``Q``.
        """
        if not self._models:
            raise RuntimeError("LightGBMForecaster.calibrate called before fit.")
        if method not in {"constant", "normalized", "mondrian"}:
            raise ValueError(f"Unknown calibration method: {method!r}")
        quantiles = self._config.quantiles
        q_lo, q_hi = quantiles[0], quantiles[-1]
        level = q_hi - q_lo
        x_all = frame[self._predictors]

        self._conformal = {}
        self._conformal_scale = {}
        self._conformal_method = method

        def _emp_quantile(scores: np.ndarray) -> float:
            n = len(scores)
            rank = min(max(int(np.ceil((n + 1) * level)), 1), n)
            return float(np.sort(scores)[rank - 1])

        for h in range(1, self._config.horizon_days + 1):
            if (h, q_lo) not in self._models or (h, q_hi) not in self._models:
                continue
            y_all = frame[f"y_h{h}"]
            mask = y_all.notna().to_numpy()
            if mask.sum() == 0:
                continue
            x_h = x_all[mask]
            y_h = y_all[mask].to_numpy(dtype=float)
            lo = self._models[(h, q_lo)].predict(x_h)
            hi = self._models[(h, q_hi)].predict(x_h)
            raw = np.maximum(lo - y_h, y_h - hi)

            if method == "normalized":
                width = hi - lo
                c = float(np.median(width))
                self._conformal_scale[(h, 0)] = c
                self._conformal[(h, 0)] = _emp_quantile(raw / (width + c))
            elif method == "constant":
                self._conformal[(h, 0)] = _emp_quantile(raw)
            else:  # mondrian: additive Q per season of the forecast (target) date
                target_dates = pd.to_datetime(frame.loc[mask, "date"]) + pd.to_timedelta(
                    h, unit="D"
                )
                seasons = self._season_index(target_dates)
                pooled = _emp_quantile(raw)
                self._conformal[(h, -1)] = pooled  # fallback for sparse seasons
                for season in np.unique(seasons):
                    in_season = raw[seasons == season]
                    self._conformal[(h, int(season))] = (
                        _emp_quantile(in_season)
                        if len(in_season) >= self._MIN_SEASON_BIN
                        else pooled
                    )
        return self

    def predict(self, frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """
        Produce long-form forecasts for every (row, horizon).

        Returns one row per (asset_id, origin date, horizon) with columns:
        ``asset_id``, ``origin_date``, ``forecast_date``, ``horizon_day``,
        ``target`` and one column per quantile (``p10``, ``p50``, ``p90``),
        sorted so the quantile columns are non-decreasing. If the model has
        been conformally calibrated, the outer interval is widened accordingly.
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
            # Conformal widening of the outer interval, if calibrated.
            if self._conformal_method:
                delta = self._conformal_delta(h, preds, frame)
                preds[:, 0] -= delta
                preds[:, -1] += delta
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

    def _conformal_delta(
        self, h: int, preds: np.ndarray, frame: pd.DataFrame
    ) -> float | np.ndarray:
        """Per-row interval widening for horizon ``h`` under the active method."""
        method = self._conformal_method
        if method == "normalized":
            q = self._conformal.get((h, 0))
            if q is None:
                return 0.0
            interval = preds[:, -1] - preds[:, 0]
            return q * (interval + self._conformal_scale.get((h, 0), 0.0))  # type: ignore[no-any-return]
        if method == "mondrian":
            forecast_dates = pd.to_datetime(frame["date"]) + pd.to_timedelta(
                h, unit="D"
            )
            seasons = self._season_index(forecast_dates)
            fallback = self._conformal.get((h, -1), 0.0)
            return np.array(
                [self._conformal.get((h, int(s)), fallback) for s in seasons],
                dtype=float,
            )
        # constant
        return self._conformal.get((h, 0), 0.0)

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
                "conformal": self._conformal,
                "conformal_scale": self._conformal_scale,
                "conformal_method": self._conformal_method,
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
        forecaster._conformal = state.get("conformal", {})
        forecaster._conformal_scale = state.get("conformal_scale", {})
        forecaster._conformal_method = state.get("conformal_method", "")
        return forecaster
