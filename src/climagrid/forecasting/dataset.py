"""
Dataset construction for stress-feature forecasting.

Two stages:

1. ``build_training_panel`` turns an asset registry into a daily panel by
   fetching hourly climagrid features per asset (streaming one asset at a time
   so the full hourly panel is never held in memory) and aggregating to one
   row per (asset, day). The daily panel is optionally cached to parquet.

2. ``build_supervised_frame`` turns the daily panel into a supervised learning
   frame: backward-looking predictors (autoregressive lags, trailing rolling
   statistics, calendar harmonics, static location) plus the shifted target
   columns ``y_h1..y_hH``.

All predictors are strictly causal (computed from data available at or before
the forecast origin), and rows are sorted by (asset_id, date) before any lag
or rolling operation because the underlying climagrid feature functions assume
time-sorted input and do not sort internally.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from climagrid.assets.registry import AssetRegistry
from climagrid.forecasting.config import ForecastConfig
from climagrid.pipeline.orchestrator import run as default_run

logger = logging.getLogger(__name__)

RunFn = Callable[..., pd.DataFrame]


def predictor_columns(config: ForecastConfig) -> list[str]:
    """Return the ordered list of predictor column names for a config."""
    cols = ["y_t"]
    cols += [f"lag_{lag}" for lag in config.lags]
    for window in config.rolling_windows:
        cols += [f"rollmean_{window}", f"rollstd_{window}"]
    cols += ["doy_sin", "doy_cos", "lat", "lon"]
    return cols


def horizon_target_columns(config: ForecastConfig) -> list[str]:
    """Return the shifted-target column names ``y_h1..y_hH``."""
    return [f"y_h{h}" for h in range(1, config.horizon_days + 1)]


def _iter_single_asset_registries(registry: AssetRegistry) -> Iterator[AssetRegistry]:
    """Yield a one-asset AssetRegistry for each asset, via a temp CSV.

    Streaming one asset at a time bounds peak memory to a single asset's hourly
    series rather than the whole panel.
    """
    assets = registry.assets
    for _, row in assets.iterrows():
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as handle:
            pd.DataFrame(
                [{"asset_id": row["asset_id"], "lat": row["lat"], "lon": row["lon"]}]
            ).to_csv(handle, index=False)
            path = handle.name
        try:
            yield AssetRegistry(path)
        finally:
            os.unlink(path)


def _aggregate_daily(raw: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    """Aggregate an hourly climagrid frame to one row per (asset_id, date)."""
    df = raw.copy()
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = timestamps.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    present = [t for t in config.targets if t in df.columns]
    if not present:
        logger.warning(
            "None of the target columns %s are present in fetched data.",
            config.targets,
        )

    agg: dict[str, str] = {target: config.daily_agg for target in present}
    agg["lat"] = "first"
    agg["lon"] = "first"
    daily = df.groupby(["asset_id", "date"], as_index=False).agg(agg)
    return daily  # type: ignore[no-any-return]


def _cache_path(
    registry: AssetRegistry,
    history_start: datetime,
    history_end: datetime,
    config: ForecastConfig,
) -> Path | None:
    """Deterministic cache filename for a panel, or None when caching is off."""
    if config.cache_dir is None:
        return None
    key = "|".join(
        [
            str(registry.count),
            ",".join(sorted(str(a) for a in registry.assets["asset_id"])),
            history_start.isoformat(),
            history_end.isoformat(),
            ",".join(sorted(config.targets)),
            ",".join(sorted(config.sources)),
            config.daily_agg,
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return Path(config.cache_dir) / f"panel_{digest}.parquet"


def build_training_panel(
    assets: AssetRegistry | str | Path,
    history_start: datetime,
    history_end: datetime,
    config: ForecastConfig,
    *,
    run_fn: RunFn | None = None,
) -> pd.DataFrame:
    """
    Build the daily training panel for a set of assets.

    Parameters
    ----------
    assets:
        An AssetRegistry or a path to an asset CSV/GeoJSON.
    history_start, history_end:
        Inclusive UTC date range of history to fetch.
    config:
        Forecast configuration (targets, sources, daily aggregation, cache).
    run_fn:
        Injection point for ``climagrid.run`` (used by tests to mock fetching).

    Returns
    -------
    pd.DataFrame
        Columns: ``asset_id``, ``date``, ``lat``, ``lon`` and one column per
        present target, with one row per (asset, day), sorted by
        (asset_id, date). Empty if no data was returned for any asset.
    """
    resolved_run = run_fn or default_run
    registry = assets if isinstance(assets, AssetRegistry) else AssetRegistry(assets)

    cache_path = _cache_path(registry, history_start, history_end, config)
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached daily panel from %s", cache_path)
        return pd.read_parquet(cache_path)  # type: ignore[no-any-return]

    features = config.required_features()
    daily_frames: list[pd.DataFrame] = []
    for sub in _iter_single_asset_registries(registry):
        raw = resolved_run(
            sub,
            history_start,
            history_end,
            sources=config.sources,
            features=features,
        )
        if raw is None or raw.empty:
            logger.warning("No data returned for an asset; skipping it in the panel.")
            continue
        daily_frames.append(_aggregate_daily(raw, config))

    if not daily_frames:
        logger.warning("Training panel is empty: no asset returned any data.")
        return pd.DataFrame()

    panel = (
        pd.concat(daily_frames, ignore_index=True)
        .sort_values(["asset_id", "date"])
        .reset_index(drop=True)
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path, index=False)
        logger.info("Cached daily panel to %s", cache_path)

    return panel  # type: ignore[no-any-return]


def build_supervised_frame(
    panel: pd.DataFrame,
    target: str,
    config: ForecastConfig,
) -> pd.DataFrame:
    """
    Turn a daily panel into a supervised frame for one target.

    Predictors (all known at the forecast origin ``t``):
      - ``y_t``: the target value at the origin,
      - ``lag_k``: the target value at ``t - k`` for each configured lag,
      - ``rollmean_w`` / ``rollstd_w``: trailing rolling mean/std over the
        ``w`` days ending at ``t - 1`` (shifted by 1 so the origin is excluded),
      - ``doy_sin`` / ``doy_cos``: day-of-year harmonics of the origin date,
      - ``lat`` / ``lon``: static location.

    Targets: ``y_h{h}`` is the target value at ``t + h`` for each horizon.

    Returns one row per (asset, origin date). Early rows have NaN lag/rolling
    predictors and late rows have NaN ``y_h*`` targets; both are retained here
    (LightGBM tolerates NaN predictors; NaN targets are dropped at fit time).
    """
    if target not in panel.columns:
        raise KeyError(f"Target {target!r} not in panel columns: {list(panel.columns)}")
    if panel.empty:
        return panel.copy()  # type: ignore[no-any-return]

    df = panel.sort_values(["asset_id", "date"]).reset_index(drop=True).copy()
    grouped = df.groupby("asset_id", sort=False)[target]

    df["y_t"] = df[target].astype(float)
    for lag in config.lags:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in config.rolling_windows:
        df[f"rollmean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"rollstd_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
        )

    day_of_year = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    for h in range(1, config.horizon_days + 1):
        df[f"y_h{h}"] = grouped.shift(-h)

    keep: list[str] = ["asset_id", "date", target]
    keep += predictor_columns(config)
    keep += horizon_target_columns(config)
    ordered: list[str] = []
    for col in keep:
        if col not in ordered:
            ordered.append(col)
    return df[ordered]  # type: ignore[no-any-return]
