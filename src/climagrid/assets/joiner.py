"""
AssetEnvironmentJoiner — spatially joins environmental data to asset locations.

For each asset in the registry, finds the nearest data point (grid cell or
station) in the environmental DataFrame and extracts its time series.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from climagrid.assets.registry import AssetRegistry


class AssetEnvironmentJoiner:
    """
    Joins time-series environmental data to utility asset point locations.

    Strategy: nearest-neighbor match in Euclidean lat/lon space (valid for
    small regions, <500 km extents). For large extents consider haversine.

    Parameters
    ----------
    max_distance_km:
        Reject matches farther than this distance. Points beyond this
        threshold will have NaN environmental values. Default 100 km.

    Example
    -------
    >>> registry = AssetRegistry("assets.csv")
    >>> nasa = NasaPowerAdapter()
    >>> env_df = nasa.fetch(bbox, start_dt, end_dt)
    >>> joiner = AssetEnvironmentJoiner()
    >>> result = joiner.join(registry, env_df)
    >>> result.head()
    """

    def __init__(self, max_distance_km: float = 100.0):
        self._max_distance_km = max_distance_km

    def join(
        self,
        registry: AssetRegistry,
        env_df: pd.DataFrame,
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Join environmental observations to each asset for every timestamp.

        Parameters
        ----------
        registry:
            AssetRegistry with asset locations.
        env_df:
            DataFrame returned by any adapter's fetch() method.
            Must have 'lat', 'lon', and at least one timestamp.
        time_col:
            Name of the timestamp column in env_df.

        Returns
        -------
        pd.DataFrame
            One row per (asset_id, timestamp) with index columns and all
            environmental columns present in env_df.
        """
        assets = registry.assets

        if env_df.empty:
            return pd.DataFrame(
                {"asset_id": assets["asset_id"].values}
            )

        if "lat" not in env_df.columns or "lon" not in env_df.columns:
            raise ValueError("env_df must contain 'lat' and 'lon' columns")

        # Build KD-tree from unique environmental grid points
        env_points = env_df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
        tree = cKDTree(env_points[["lat", "lon"]].values)

        asset_lats = assets["lat"].values
        asset_lons = assets["lon"].values
        asset_coords = np.column_stack([asset_lats, asset_lons])

        # Query nearest grid point for each asset
        distances_deg, indices = tree.query(asset_coords, k=1)
        # Rough conversion: 1 degree ≈ 111 km
        distances_km = distances_deg * 111.0

        # Warn about far matches
        too_far = distances_km > self._max_distance_km
        if too_far.any():
            n_far = too_far.sum()
            warnings.warn(
                f"{n_far} asset(s) are more than {self._max_distance_km} km "
                "from any environmental data point. Those rows will have NaN values.",
                UserWarning,
                stacklevel=2,
            )

        # Map each asset to its nearest environmental grid point lat/lon
        nearest_lats = env_points.loc[indices, "lat"].values
        nearest_lons = env_points.loc[indices, "lon"].values

        # Build result: for each asset, extract the env time series at its nearest point
        result_frames: list[pd.DataFrame] = []

        env_value_cols = [
            c for c in env_df.columns
            if c not in {"lat", "lon", time_col}
        ]

        for i, row in assets.iterrows():
            asset_id = row["asset_id"]
            asset_lat = row["lat"]
            asset_lon = row["lon"]

            nn_lat = nearest_lats[list(assets.index).index(i) if i in assets.index else i]
            nn_lon = nearest_lons[list(assets.index).index(i) if i in assets.index else i]

            env_slice = env_df[
                (env_df["lat"] == nn_lat) & (env_df["lon"] == nn_lon)
            ][env_value_cols + ([time_col] if time_col in env_df.columns else [])].copy()

            env_slice["asset_id"] = asset_id
            env_slice["lat"] = asset_lat
            env_slice["lon"] = asset_lon

            if distances_km[list(assets.index).index(i) if i in assets.index else i] > self._max_distance_km:
                for col in env_value_cols:
                    env_slice[col] = float("nan")

            result_frames.append(env_slice)

        if not result_frames:
            return pd.DataFrame()

        result = pd.concat(result_frames, ignore_index=True)

        # Reorder columns: asset_id, timestamp, lat, lon, then env columns
        front_cols = ["asset_id"]
        if time_col in result.columns:
            front_cols.append(time_col)
        front_cols += ["lat", "lon"]
        remaining = [c for c in result.columns if c not in front_cols]
        return result[front_cols + remaining].reset_index(drop=True)

    def join_point(
        self,
        asset_lat: float,
        asset_lon: float,
        env_df: pd.DataFrame,
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Convenience method: join env data for a single lat/lon point."""
        import os
        import tempfile

        from climagrid.assets.registry import AssetRegistry

        tmp_data = pd.DataFrame([{
            "asset_id": "point",
            "lat": asset_lat,
            "lon": asset_lon,
        }])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmp_data.to_csv(f, index=False)
            tmp_path = f.name

        try:
            reg = AssetRegistry(tmp_path)
            return self.join(reg, env_df, time_col)
        finally:
            os.unlink(tmp_path)
