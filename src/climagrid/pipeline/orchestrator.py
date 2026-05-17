"""
High-level pipeline orchestrator — fetch, join, and featurize in one call.

This is the primary entry point for users who don't want to wire up
individual adapters manually.
"""

from __future__ import annotations

import importlib
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from climagrid.assets.joiner import AssetEnvironmentJoiner
from climagrid.assets.registry import AssetRegistry
from climagrid.sources.base import BoundingBox

_SOURCE_MAP = {
    "nasa_power": ("climagrid.sources.nasa_power", "NasaPowerAdapter"),
    "noaa_hrrr":  ("climagrid.sources.noaa_hrrr",  "HrrrAdapter"),
    "noaa_ncei":  ("climagrid.sources.noaa_ncei",  "NceiAdapter"),
    "usda_nrcs":  ("climagrid.sources.usda_nrcs",  "NrcsAdapter"),
    "usfs_wfigs": ("climagrid.sources.usfs_wfigs", "WfigsAdapter"),
}

_FEATURE_MAP = {
    "thermal":       ("climagrid.features.thermal",       "ThermalStressIndex"),
    "freeze_thaw":   ("climagrid.features.freeze_thaw",   "FreezeThawtCycleCounter"),
    "ice_loading":   ("climagrid.features.ice_loading",   "IceLoadingRisk"),
    "soil":          ("climagrid.features.soil",          "SoilSaturationIndex"),
    "wildfire":      ("climagrid.features.wildfire",      "WildfireProximityScore"),
    "conductor_sag": ("climagrid.features.conductor_sag", "ConductorSagIndex"),
}


def run(
    assets: AssetRegistry | str | Path,
    start_dt: datetime,
    end_dt: datetime,
    *,
    sources: list[str] | None = None,
    features: list[str] | str = "all",
    bbox_radius_km: float = 50.0,
    max_join_distance_km: float = 100.0,
    source_kwargs: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Fetch environmental data, join it to assets, and compute stress features.

    Parameters
    ----------
    assets:
        An AssetRegistry instance, or a path to an asset CSV/GeoJSON file.
    start_dt:
        Start of the time range (UTC-aware recommended).
    end_dt:
        End of the time range (UTC-aware recommended).
    sources:
        List of source names to fetch from. Valid values:
        ``"nasa_power"``, ``"noaa_hrrr"``, ``"noaa_ncei"``, ``"usda_nrcs"``,
        ``"usfs_wfigs"``. Defaults to ``["nasa_power"]``.
    features:
        List of feature names to compute, or ``"all"`` (default).
        Valid values: ``"thermal"``, ``"freeze_thaw"``, ``"ice_loading"``,
        ``"soil"``, ``"wildfire"``, ``"conductor_sag"``.
    bbox_radius_km:
        Radius in km to build a bounding box around the asset centroid
        for data fetching. Default 50 km.
    max_join_distance_km:
        Maximum distance for spatial join. Assets farther than this from
        any data point will have NaN environmental values. Default 100 km.
    source_kwargs:
        Optional dict of keyword arguments passed to each source adapter.
        E.g. ``{"noaa_ncei": {"token": "my-cdo-token"}}``.

    Returns
    -------
    pd.DataFrame
        Wide-form DataFrame: one row per (asset_id, timestamp), with all
        environmental and feature columns present.

    Example
    -------
    >>> import climagrid
    >>> from datetime import datetime, timezone
    >>> df = climagrid.run(
    ...     "assets.csv",
    ...     start_dt=datetime(2024, 7, 1, tzinfo=timezone.utc),
    ...     end_dt=datetime(2024, 7, 2, tzinfo=timezone.utc),
    ...     sources=["nasa_power"],
    ...     features="all",
    ... )
    >>> df.columns.tolist()
    ['asset_id', 'timestamp', 'lat', 'lon', 'nasa_temperature_2m', ...]
    """
    if sources is None:
        sources = ["nasa_power"]
    if source_kwargs is None:
        source_kwargs = {}

    # Validate inputs upfront before any I/O
    for source_name in sources:
        if source_name not in _SOURCE_MAP:
            raise ValueError(f"Unknown source '{source_name}'. Valid: {list(_SOURCE_MAP)}")

    feature_names: list[str]
    if features == "all":
        feature_names = list(_FEATURE_MAP.keys())
    else:
        feature_names = list(features)
        for feat_name in feature_names:
            if feat_name not in _FEATURE_MAP:
                raise ValueError(f"Unknown feature '{feat_name}'. Valid: {list(_FEATURE_MAP)}")

    # Resolve asset registry
    if not isinstance(assets, AssetRegistry):
        assets = AssetRegistry(assets)

    # Build bounding box around asset centroid
    asset_lats = assets.assets["lat"].values
    asset_lons = assets.assets["lon"].values
    centroid_lat = float(asset_lats.mean())
    centroid_lon = float(asset_lons.mean())
    bbox = BoundingBox.from_center(centroid_lat, centroid_lon, bbox_radius_km)

    # Fetch from each source and merge
    env_frames: list[pd.DataFrame] = []
    joiner = AssetEnvironmentJoiner(max_distance_km=max_join_distance_km)
    fires_raw: pd.DataFrame = pd.DataFrame()  # kept separate for wildfire feature

    for source_name in sources:
        module_path, class_name = _SOURCE_MAP[source_name]
        mod = importlib.import_module(module_path)
        adapter_cls = getattr(mod, class_name)
        kwargs = source_kwargs.get(source_name, {})
        adapter = adapter_cls(**kwargs)

        try:
            raw = adapter.fetch(bbox, start_dt, end_dt)
        except Exception as exc:
            warnings.warn(
                f"Source '{source_name}' failed: {exc}. Skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if raw.empty:
            continue

        # Keep WFIGS fire perimeters unjoined so the wildfire feature can use them
        if source_name == "usfs_wfigs":
            fires_raw = raw

        joined = joiner.join(assets, raw)
        env_frames.append(joined)

    if not env_frames:
        return pd.DataFrame()

    # Merge all sources on (asset_id, timestamp)
    result = env_frames[0]
    for frame in env_frames[1:]:
        merge_on = [c for c in ["asset_id", "timestamp", "lat", "lon"] if c in result.columns and c in frame.columns]
        result = result.merge(frame, on=merge_on, how="outer", suffixes=("", "_dup"))
        # Drop duplicate columns
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        result = result.drop(columns=dup_cols)

    # Apply feature computations
    for feat_name in feature_names:
        module_path, class_name = _FEATURE_MAP[feat_name]
        mod = importlib.import_module(module_path)
        feat_cls = getattr(mod, class_name)
        try:
            if feat_name == "wildfire":
                result = feat_cls().compute(result, fires_raw)
            else:
                result = feat_cls().compute(result)
        except Exception as exc:
            warnings.warn(
                f"Feature '{feat_name}' failed: {exc}. Skipping.",
                UserWarning,
                stacklevel=2,
            )

    return result.reset_index(drop=True)
