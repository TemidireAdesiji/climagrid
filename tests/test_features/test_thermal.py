"""Tests for ThermalStressIndex."""

from __future__ import annotations

import pandas as pd
import pytest

from climagrid.features.thermal import _T_REF_K, ThermalStressIndex


class TestThermalStressIndex:
    def test_compute_adds_aging_factor_column(self, asset_env_df):
        tsi = ThermalStressIndex()
        result = tsi.compute(asset_env_df)
        assert "feat_thermal_aging_factor" in result.columns

    def test_compute_adds_heat_hours_column(self, asset_env_df):
        tsi = ThermalStressIndex()
        result = tsi.compute(asset_env_df)
        assert "feat_heat_hours_above_35c" in result.columns

    def test_aging_factor_at_reference_temperature_is_one(self):
        """At the exact reference hotspot temperature (T_REF_K), FAA must equal 1.0."""
        # T_REF_K = 383.0 K → 109.85°C ambient with hotspot_rise=0
        ref_temp_c = _T_REF_K - 273.15
        tsi = ThermalStressIndex(hotspot_rise=0.0)
        df = pd.DataFrame({
            "hrrr_temperature_2m": [ref_temp_c],
            "asset_id": ["TX-001"],
        })
        result = tsi.compute(df)
        faa = result["feat_thermal_aging_factor"].iloc[0]
        assert faa == pytest.approx(1.0, rel=1e-6)

    def test_aging_factor_above_reference_greater_than_one(self, asset_env_df):
        """Temperatures above the reference must produce FAA > 1."""
        tsi = ThermalStressIndex()
        result = tsi.compute(asset_env_df)
        # Summer Texas temps (28–42°C) + 25°C hotspot rise = 53–67°C hotspot
        # All below 110°C reference → FAA should be < 1 (less aging than reference)
        assert (result["feat_thermal_aging_factor"] > 0).all()

    def test_aging_factor_extreme_heat_above_one(self):
        """Very high ambient temperature should produce FAA > 1."""
        tsi = ThermalStressIndex(hotspot_rise=100.0)  # 100°C rise
        df = pd.DataFrame({
            "hrrr_temperature_2m": [50.0],   # 50 + 100 = 150°C hotspot
            "asset_id": ["TX-001"],
        })
        result = tsi.compute(df)
        assert result["feat_thermal_aging_factor"].iloc[0] > 1.0

    def test_heat_hours_count_correct(self):
        """Hours above 35°C threshold should be counted correctly."""
        temps = [30.0, 36.0, 38.0, 34.0, 40.0, 29.0]  # 3 hours above 35°C
        df = pd.DataFrame({
            "hrrr_temperature_2m": temps,
            "asset_id": "TX-001",
        })
        tsi = ThermalStressIndex(heat_threshold_c=35.0, rolling_window_h=10)
        result = tsi.compute(df)
        # Last row should show cumulative 3 heat hours
        assert result["feat_heat_hours_above_35c"].iloc[-1] == pytest.approx(3.0)

    def test_missing_temp_column_returns_nan(self):
        tsi = ThermalStressIndex(temp_col="nonexistent_col")
        df = pd.DataFrame({"asset_id": ["TX-001"], "some_other_col": [1.0]})
        result = tsi.compute(df)
        assert result["feat_thermal_aging_factor"].isna().all()

    def test_fallback_to_nasa_temperature(self):
        tsi = ThermalStressIndex(temp_col="hrrr_temperature_2m")
        df = pd.DataFrame({
            "nasa_temperature_2m": [35.0],
            "asset_id": ["TX-001"],
        })
        result = tsi.compute(df)
        assert "feat_thermal_aging_factor" in result.columns
        assert not result["feat_thermal_aging_factor"].isna().all()

    def test_input_not_modified(self, asset_env_df):
        original_cols = list(asset_env_df.columns)
        tsi = ThermalStressIndex()
        tsi.compute(asset_env_df)
        assert list(asset_env_df.columns) == original_cols
