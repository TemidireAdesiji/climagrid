"""Tests for AssetEnvironmentJoiner."""

from __future__ import annotations

import pandas as pd
import pytest

from climagrid.assets.joiner import AssetEnvironmentJoiner
from climagrid.assets.registry import AssetRegistry


class TestAssetEnvironmentJoiner:
    def test_join_produces_one_row_per_asset_per_timestamp(
        self, tmp_assets_csv, synthetic_env_df
    ):
        registry = AssetRegistry(tmp_assets_csv)
        joiner = AssetEnvironmentJoiner()
        result = joiner.join(registry, synthetic_env_df)

        n_assets = registry.count
        n_timestamps = synthetic_env_df["timestamp"].nunique()
        assert len(result) == n_assets * n_timestamps

    def test_join_result_has_asset_id_column(self, tmp_assets_csv, synthetic_env_df):
        registry = AssetRegistry(tmp_assets_csv)
        joiner = AssetEnvironmentJoiner()
        result = joiner.join(registry, synthetic_env_df)
        assert "asset_id" in result.columns

    def test_join_result_has_lat_lon(self, tmp_assets_csv, synthetic_env_df):
        registry = AssetRegistry(tmp_assets_csv)
        joiner = AssetEnvironmentJoiner()
        result = joiner.join(registry, synthetic_env_df)
        assert "lat" in result.columns
        assert "lon" in result.columns

    def test_join_preserves_env_columns(self, tmp_assets_csv, synthetic_env_df):
        registry = AssetRegistry(tmp_assets_csv)
        joiner = AssetEnvironmentJoiner()
        result = joiner.join(registry, synthetic_env_df)
        assert "hrrr_temperature_2m" in result.columns

    def test_join_empty_env_df_returns_minimal(self, tmp_assets_csv):
        registry = AssetRegistry(tmp_assets_csv)
        joiner = AssetEnvironmentJoiner()
        result = joiner.join(registry, pd.DataFrame())
        assert "asset_id" in result.columns

    def test_join_missing_lat_lon_raises(self, tmp_assets_csv):
        registry = AssetRegistry(tmp_assets_csv)
        joiner = AssetEnvironmentJoiner()
        bad_df = pd.DataFrame({"temperature": [25.0]})
        with pytest.raises(ValueError, match="lat"):
            joiner.join(registry, bad_df)

    def test_join_point_returns_dataframe(self, synthetic_env_df):
        joiner = AssetEnvironmentJoiner()
        result = joiner.join_point(31.55, -97.15, synthetic_env_df)
        assert not result.empty
        assert "asset_id" in result.columns
        assert result["asset_id"].iloc[0] == "point"

    def test_join_point_preserves_env_columns(self, synthetic_env_df):
        joiner = AssetEnvironmentJoiner()
        result = joiner.join_point(31.55, -97.15, synthetic_env_df)
        assert "hrrr_temperature_2m" in result.columns
