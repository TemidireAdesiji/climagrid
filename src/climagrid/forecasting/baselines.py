"""
Baseline forecasters.

Every ML forecast must beat these to justify itself; the backtest reports the
skill score of the trained model relative to both. They are deliberately
simple and parameter-free.

- ``PersistenceForecaster``: tomorrow looks like today. Predicts the origin
  value for every horizon. A strong baseline for smooth, autocorrelated series.
- ``ClimatologyForecaster``: predicts the historical day-of-year average of the
  target (optionally restricted to a recent window, a hedge against climate
  drift). Captures seasonality with no trend or autocorrelation.

Both expose ``fit(frame, target)`` and ``predict(frame, horizon)`` returning a
point forecast as a NumPy array aligned to the rows of ``frame``. ``frame`` is
a supervised frame from ``dataset.build_supervised_frame`` (it has ``asset_id``,
``date`` and ``y_t`` columns).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from climagrid.forecasting.config import ForecastConfig


class PersistenceForecaster:
    """Predicts the origin value ``y_t`` for every horizon."""

    def __init__(self, config: ForecastConfig):
        self._config = config

    def fit(self, frame: pd.DataFrame, target: str) -> PersistenceForecaster:
        """No-op: persistence has nothing to learn."""
        return self

    def predict(self, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        """Return the origin value for each row (horizon-independent)."""
        return frame["y_t"].to_numpy(dtype=float)


class ClimatologyForecaster:
    """Predicts the historical day-of-year mean of the target."""

    def __init__(self, config: ForecastConfig):
        self._config = config
        self._global_mean: float = float("nan")
        self._table: pd.Series = pd.Series(dtype=float)

    def fit(self, frame: pd.DataFrame, target: str) -> ClimatologyForecaster:
        """Build the day-of-year mean table from the training rows."""
        df = frame[["asset_id", "date", "y_t"]].dropna(subset=["y_t"]).copy()

        window = self._config.climatology_window_years
        if window is not None and not df.empty:
            cutoff = df["date"].max() - pd.DateOffset(years=window)
            df = df[df["date"] >= cutoff]

        if df.empty:
            self._global_mean = float("nan")
            self._table = pd.Series(dtype=float)
            return self

        df["doy"] = df["date"].dt.dayofyear
        self._global_mean = float(df["y_t"].mean())
        if self._config.per_asset:
            self._table = df.groupby(["asset_id", "doy"])["y_t"].mean()
        else:
            self._table = df.groupby("doy")["y_t"].mean()
        return self

    def predict(self, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        """Return the day-of-year mean for each row's target date (origin + h)."""
        target_dates = frame["date"] + pd.to_timedelta(horizon, unit="D")
        day_of_year = target_dates.dt.dayofyear

        if self._config.per_asset:
            values = [
                self._table.get((asset_id, doy), self._global_mean)
                for asset_id, doy in zip(frame["asset_id"], day_of_year, strict=True)
            ]
            return np.asarray(values, dtype=float)

        mapped = day_of_year.map(self._table).to_numpy(dtype=float)
        return np.where(np.isnan(mapped), self._global_mean, mapped)
