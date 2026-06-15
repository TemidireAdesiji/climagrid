"""
High-level forecasting entry point.

``forecast`` is **load-and-serve only**: it loads one or more previously trained
and saved models and produces a forward forecast. It never trains. Training
happens offline (the Kaggle training notebook via ``LightGBMForecaster.fit``,
or ``backtest.evaluate`` / ``history_ablation`` for evaluation), and the saved
artifacts are what production serves.

Because inference only needs each model's recent predictor window, ``forecast``
fetches just the recent ~``min_inference_history_days`` per asset, not the full
training history.

The forecasts are of environmental STRESS, not equipment failure. See
docs/validation_notes.md for what climagrid does and does not establish.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from climagrid.assets.registry import AssetRegistry
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

# What forecast() accepts to locate the trained model(s).
ModelSource = str | Path | LightGBMForecaster


def _latest_origins(supervised: pd.DataFrame) -> pd.DataFrame:
    """Return the single most recent origin row per asset."""
    latest = supervised.sort_values("date").groupby("asset_id", as_index=False).tail(1)
    return latest  # type: ignore[no-any-return]


def _resolve_models(model: ModelSource) -> list[tuple[LightGBMForecaster, str | None]]:
    """Resolve a model source into ``(forecaster, recommendation)`` pairs.

    - a loaded ``LightGBMForecaster`` -> itself,
    - a directory containing ``manifest.json`` -> every per-factor model it lists,
      each paired with its deploy ``recommendation``,
    - a path to a single ``.joblib`` -> that one model.
    """
    if isinstance(model, LightGBMForecaster):
        return [(model, None)]

    path = Path(model)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json found in {path}")
        manifest = json.loads(manifest_path.read_text())
        pairs: list[tuple[LightGBMForecaster, str | None]] = []
        for info in manifest.get("per_factor", {}).values():
            forecaster = LightGBMForecaster.load(path / info["model_file"])
            pairs.append((forecaster, info.get("recommendation")))
        return pairs

    return [(LightGBMForecaster.load(path), None)]


def forecast(
    assets: AssetRegistry | str | Path,
    model: ModelSource,
    *,
    history_end: datetime | None = None,
    run_fn: RunFn | None = None,
) -> pd.DataFrame:
    """
    Forecast asset stress from previously saved model(s). Never trains.

    Parameters
    ----------
    assets:
        AssetRegistry or path to an asset CSV/GeoJSON (asset_id, lat, lon).
    model:
        A directory containing ``manifest.json`` (and the per-factor model
        files), a path to a single saved ``.joblib``, or a loaded
        ``LightGBMForecaster``. Train and save these with the Kaggle training
        notebook (``LightGBMForecaster.fit(...).save(...)``).
    history_end:
        End of the recent window to fetch (UTC). Defaults to now. Only the
        recent ``min_inference_history_days`` (plus a small buffer) is fetched,
        since that is all the saved model needs to build its predictors.
    run_fn:
        Injection point for ``climagrid.run`` (used by tests).

    Returns
    -------
    pd.DataFrame
        Long form: one row per (asset_id, origin_date, target, horizon_day)
        with ``forecast_date``, the quantile columns (``p10``/``p50``/``p90``),
        and a ``recommendation`` confidence flag when served from a manifest.
        Empty if no recent data could be fetched.
    """
    registry = assets if isinstance(assets, AssetRegistry) else AssetRegistry(assets)
    loaded = _resolve_models(model)
    if not loaded:
        logger.warning("No models to serve; returning an empty forecast.")
        return pd.DataFrame()

    # Fetch only the recent window each model needs for its predictors.
    targets = list(dict.fromkeys(m._target for m, _ in loaded if m._target))
    lookback = max(m._config.min_inference_history_days for m, _ in loaded) + 15
    fetch_config = loaded[0][0]._config.model_copy(update={"targets": targets})

    if history_end is None:
        history_end = datetime.now(timezone.utc)
    history_start = (
        pd.Timestamp(history_end) - pd.Timedelta(days=lookback)
    ).to_pydatetime()

    panel = build_training_panel(
        registry, history_start, history_end, fetch_config, run_fn=run_fn
    )
    if panel.empty:
        logger.warning("No recent data fetched; returning an empty forecast.")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for forecaster, recommendation in loaded:
        target = forecaster._target
        if target is None or target not in panel.columns:
            logger.warning("Target %s missing from recent data; skipping.", target)
            continue
        supervised = build_supervised_frame(panel, target, forecaster._config)
        if supervised.empty:
            continue
        preds = forecaster.predict(_latest_origins(supervised), target)
        if preds.empty:
            continue
        if recommendation is not None:
            preds["recommendation"] = recommendation
        frames.append(preds)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True).sort_values(
        ["asset_id", "target", "horizon_day"]
    ).reset_index(drop=True)

    columns = [*_FORECAST_COLUMNS, *quantile_column_names(loaded[0][0]._config)]
    if "recommendation" in result.columns:
        columns.append("recommendation")
    return result[columns]  # type: ignore[no-any-return]
