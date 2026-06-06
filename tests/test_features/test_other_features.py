"""Tests for IceLoadingRisk, SoilSaturationIndex, ConductorSagIndex."""

import pandas as pd
import pytest

from climagrid.features.conductor_sag import ConductorSagIndex
from climagrid.features.ice_loading import IceLoadingRisk
from climagrid.features.soil import SoilSaturationIndex

# ---------------------------------------------------------------------------
# IceLoadingRisk
# ---------------------------------------------------------------------------

class TestIceLoadingRisk:
    def test_adds_column(self, asset_env_df):
        result = IceLoadingRisk().compute(asset_env_df)
        assert "feat_ice_loading_risk" in result.columns

    def test_summer_texas_zero_risk(self, asset_env_df):
        """Summer Texas (temps 28-42°C) is well above ice formation range."""
        result = IceLoadingRisk().compute(asset_env_df)
        assert (result["feat_ice_loading_risk"] == 0.0).all()

    def test_freezing_rain_conditions_high_risk(self):
        df = pd.DataFrame({
            "hrrr_temperature_2m": [-2.0],
            "hrrr_precipitation_rate": [3.0],
            "hrrr_wind_speed_10m": [12.0],
        })
        result = IceLoadingRisk().compute(df)
        assert result["feat_ice_loading_risk"].iloc[0] > 0.3

    def test_score_in_unit_interval(self, freezing_env_df):
        result = IceLoadingRisk().compute(freezing_env_df)
        assert (result["feat_ice_loading_risk"] >= 0).all()
        assert (result["feat_ice_loading_risk"] <= 1).all()

    def test_missing_temp_returns_nan(self):
        df = pd.DataFrame({"some_col": [1.0]})
        result = IceLoadingRisk().compute(df)
        assert result["feat_ice_loading_risk"].isna().all()


# ---------------------------------------------------------------------------
# SoilSaturationIndex
# ---------------------------------------------------------------------------

class TestSoilSaturationIndex:
    def test_adds_column(self, asset_env_df):
        result = SoilSaturationIndex().compute(asset_env_df)
        assert "feat_soil_saturation_index" in result.columns

    def test_nrcs_data_used_when_present(self):
        df = pd.DataFrame({
            "nrcs_soil_moisture_pct": [40.0],  # at field capacity
        })
        result = SoilSaturationIndex(field_capacity_pct=40.0, wilting_point_pct=12.0).compute(df)
        assert result["feat_soil_saturation_index"].iloc[0] == pytest.approx(1.0)

    def test_wilting_point_gives_zero(self):
        df = pd.DataFrame({"nrcs_soil_moisture_pct": [12.0]})  # at wilting point
        result = SoilSaturationIndex(field_capacity_pct=40.0, wilting_point_pct=12.0).compute(df)
        assert result["feat_soil_saturation_index"].iloc[0] == pytest.approx(0.0)

    def test_fallback_to_precipitation(self, asset_env_df):
        """Without NRCS data, precipitation rolling sum used as proxy."""
        result = SoilSaturationIndex().compute(asset_env_df)
        assert "feat_soil_saturation_index" in result.columns
        assert not result["feat_soil_saturation_index"].isna().all()

    def test_score_clamped_to_unit_interval(self):
        df = pd.DataFrame({"nrcs_soil_moisture_pct": [100.0]})  # above field capacity
        result = SoilSaturationIndex().compute(df)
        assert result["feat_soil_saturation_index"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# ConductorSagIndex
# ---------------------------------------------------------------------------

class TestConductorSagIndex:
    def test_adds_column(self, asset_env_df):
        result = ConductorSagIndex().compute(asset_env_df)
        assert "feat_conductor_sag_index" in result.columns

    def test_score_in_unit_interval(self, asset_env_df):
        result = ConductorSagIndex().compute(asset_env_df)
        assert (result["feat_conductor_sag_index"] >= 0).all()
        assert (result["feat_conductor_sag_index"] <= 1).all()

    def test_high_temp_high_solar_increases_sag(self):
        low = pd.DataFrame({
            "hrrr_temperature_2m": [15.0],
            "hrrr_solar_irradiance_ghi": [100.0],
            "hrrr_wind_speed_10m": [8.0],
        })
        high = pd.DataFrame({
            "hrrr_temperature_2m": [45.0],
            "hrrr_solar_irradiance_ghi": [900.0],
            "hrrr_wind_speed_10m": [1.0],
        })
        csi = ConductorSagIndex()
        assert (
            csi.compute(high)["feat_conductor_sag_index"].iloc[0]
            > csi.compute(low)["feat_conductor_sag_index"].iloc[0]
        )

    def test_more_wind_reduces_sag(self):
        """Physical-behavior check (simplified IEEE 738 model): wind cools the
        conductor, so more wind must lower the sag index, all else equal."""
        base = {"hrrr_temperature_2m": [30.0], "hrrr_solar_irradiance_ghi": [600.0]}
        calm = pd.DataFrame({**base, "hrrr_wind_speed_10m": [0.5]})
        windy = pd.DataFrame({**base, "hrrr_wind_speed_10m": [12.0]})
        csi = ConductorSagIndex()
        assert (
            csi.compute(windy)["feat_conductor_sag_index"].iloc[0]
            < csi.compute(calm)["feat_conductor_sag_index"].iloc[0]
        )

    def test_more_solar_increases_sag(self):
        """Physical-behavior check: more solar heating raises the sag index,
        all else equal."""
        base = {"hrrr_temperature_2m": [40.0], "hrrr_wind_speed_10m": [4.0]}
        shade = pd.DataFrame({**base, "hrrr_solar_irradiance_ghi": [50.0]})
        sun = pd.DataFrame({**base, "hrrr_solar_irradiance_ghi": [1000.0]})
        csi = ConductorSagIndex()
        assert (
            csi.compute(sun)["feat_conductor_sag_index"].iloc[0]
            > csi.compute(shade)["feat_conductor_sag_index"].iloc[0]
        )

    def test_missing_temp_returns_nan(self):
        df = pd.DataFrame({"some_col": [1.0]})
        result = ConductorSagIndex().compute(df)
        assert result["feat_conductor_sag_index"].isna().all()
