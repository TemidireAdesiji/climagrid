"""Tests for WildfireProximityScore."""

import pandas as pd
import pytest

from climagrid.features.wildfire import WildfireProximityScore


class TestWildfireProximityScore:
    def test_adds_score_column(self, asset_env_df, nearby_fires_df):
        wps = WildfireProximityScore(use_fire_weather=False)
        result = wps.compute(asset_env_df, nearby_fires_df)
        assert "feat_wildfire_proximity" in result.columns

    def test_score_in_unit_interval(self, asset_env_df, nearby_fires_df):
        wps = WildfireProximityScore(use_fire_weather=False)
        result = wps.compute(asset_env_df, nearby_fires_df)
        assert (result["feat_wildfire_proximity"] >= 0).all()
        assert (result["feat_wildfire_proximity"] <= 1).all()

    def test_empty_fires_returns_zero_score(self, asset_env_df, empty_fires_df):
        wps = WildfireProximityScore()
        result = wps.compute(asset_env_df, empty_fires_df)
        assert (result["feat_wildfire_proximity"] == 0.0).all()

    def test_nearby_fire_produces_nonzero_score(self, asset_env_df, nearby_fires_df):
        wps = WildfireProximityScore(use_fire_weather=False)
        result = wps.compute(asset_env_df, nearby_fires_df)
        assert result["feat_wildfire_proximity"].max() > 0.0

    def test_adds_proximity_columns(self, asset_env_df, nearby_fires_df):
        wps = WildfireProximityScore()
        result = wps.compute(asset_env_df, nearby_fires_df)
        assert "wfigs_nearest_fire_km" in result.columns
        assert "wfigs_fire_active" in result.columns
        assert "wfigs_fire_area_ha" in result.columns

    def test_fire_weather_modulation_increases_risk(self, asset_env_df, nearby_fires_df):
        """Hot, dry, windy conditions should amplify the base proximity score."""
        # Extreme fire weather: 45°C, 10% RH, 20 m/s wind
        hot_df = asset_env_df.copy()
        hot_df["hrrr_temperature_2m"] = 45.0
        hot_df["hrrr_relative_humidity_2m"] = 10.0
        hot_df["hrrr_wind_speed_10m"] = 20.0

        wps_with = WildfireProximityScore(use_fire_weather=True)
        wps_without = WildfireProximityScore(use_fire_weather=False)

        score_with = wps_with.compute(hot_df, nearby_fires_df)["feat_wildfire_proximity"].mean()
        score_without = wps_without.compute(hot_df, nearby_fires_df)["feat_wildfire_proximity"].mean()
        assert score_with >= score_without
