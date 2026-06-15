"""
High-level forecasting entry point.

``forecast`` mirrors the ergonomics of ``climagrid.run``: pass an asset
registry (or a path), get a tidy long-form DataFrame of per-asset, per-horizon
stress forecasts with prediction intervals. It builds the daily training panel,
fits the configured forecaster on the full history, and emits a forward
forecast from each asset's most recent origin.

The forecasts are of environmental STRESS, not equipment failure. See
docs/validation_notes.md for what climagrid does and does not establish.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from climagrid.assets.registry import AssetRegistry
from climagrid.forecasting.baselines import ClimatologyForecaster, PersistenceForecaster
from climagrid.forecasting.config import ForecastConfig
from climagrid.forecasting.dataset import (
    RunFn,
    build_supervised_frame,
    build_training_panel,
)
from climagrid.forecasting.models import LightGBMForecaster, quantile_column_names

logger = logging.getLogger(__name__)

_FORECAST_COLUMNS = [
    "asset_id",
    "origin_date",
    "forecast_date",
    "horizon_day",
    "target",
]


def _latest_origins(supervised: pd.DataFrame) -> pd.DataFrame:
    """Return the single most recent origin row per asset."""
    latest = supervised.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    return latest  # type: ignore[no-any-return]


def _baseline_forward(
    forecaster: PersistenceForecaster | ClimatologyForecaster,
    latest: pd.DataFrame,
    target: str,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Forward forecast from a point baseline (intervals collapse to the point)."""
    q_cols = quantile_column_names(config)
    blocks: list[pd.DataFrame] = []
    for h in range(1, config.horizon_days + 1):
        point = forecaster.predict(latest, h)
        block = pd.DataFrame(
            {
                "asset_id": latest["asset_id"].to_numpy(),
                "origin_date": latest["date"].to_numpy(),
                "horizon_day": h,
                "target": target,
            }
        )
        block["forecast_date"] = block["origin_date"] + pd.to_timedelta(h, unit="D")
        for col in q_cols:
            block[col] = point
        blocks.append(block)
    combined = pd.concat(blocks, ignore_index=True)
    return combined[[*_FORECAST_COLUMNS, *q_cols]]  # type: ignore[no-any-return]


def forecast(
    assets: AssetRegistry | str | Path,
    *,
    config: ForecastConfig | None = None,
    history_start: datetime | None = None,
    history_end: datetime | None = None,
    run_fn: RunFn | None = None,
) -> pd.DataFrame:
    """
    Forecast each asset's daily stress features forward, with intervals.

    Parameters
    ----------
    assets:
        AssetRegistry or path to an asset CSV/GeoJSON (asset_id, lat, lon).
    config:
        ForecastConfig. Defaults forecast ``feat_thermal_aging_factor`` 7 days
        ahead with LightGBM quantile intervals.
    history_start, history_end:
        UTC history range for training. Defaults to ``config.history_years``
        ending today (UTC).
    run_fn:
        Injection point for ``climagrid.run`` (used by tests).

    Returns
    -------
    pd.DataFrame
        Long form: one row per (asset_id, origin_date, target, horizon_day)
        with ``forecast_date`` and one column per quantile (``p10``/``p50``/
        ``p90``). Empty if no training data could be assembled.
    """
    config = config or ForecastConfig()

    if history_end is None:
        history_end = datetime.now(timezone.utc)
    if history_start is None:
        history_start = (
            pd.Timestamp(history_end) - pd.DateOffset(years=config.history_years)
        ).to_pydatetime()

    panel = build_training_panel(
        assets, history_start, history_end, config, run_fn=run_fn
    )
    if panel.empty:
        logger.warning("Empty training panel; returning an empty forecast.")
        return pd.DataFrame()

    q_cols = quantile_column_names(config)
    frames: list[pd.DataFrame] = []
    for target in config.targets:
        if target not in panel.columns:
            logger.warning("Target %s missing from panel; skipping.", target)
            continue
        supervised = build_supervised_frame(panel, target, config)
        if supervised.empty:
            continue
        latest = _latest_origins(supervised)

        if config.model == "lightgbm":
            model = LightGBMForecaster(config).fit(supervised, target)
            preds = model.predict(latest, target)
        elif config.model == "persistence":
            fitted = PersistenceForecaster(config).fit(supervised, target)
            preds = _baseline_forward(fitted, latest, target, config)
        else:
            fitted_clim = ClimatologyForecaster(config).fit(supervised, target)
            preds = _baseline_forward(fitted_clim, latest, target, config)

        if not preds.empty:
            frames.append(preds)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["asset_id", "target", "horizon_day"]).reset_index(
        drop=True
    )
    return result[[*_FORECAST_COLUMNS, *q_cols]]  # type: ignore[no-any-return]
