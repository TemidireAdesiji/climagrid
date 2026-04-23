"""Tests for FreezeThawtCycleCounter."""

import pandas as pd
import pytest

from climagrid.features.freeze_thaw import FreezeThawtCycleCounter


class TestFreezeThawtCycleCounter:
    def test_adds_cycle_column(self, freezing_env_df):
        ftc = FreezeThawtCycleCounter()
        result = ftc.compute(freezing_env_df)
        assert "feat_freeze_thaw_cycles" in result.columns

    def test_no_cycles_in_summer(self, asset_env_df):
        """Summer Texas dataset (all temps > 28°C) should have zero cycles."""
        ftc = FreezeThawtCycleCounter()
        result = ftc.compute(asset_env_df)
        assert result["feat_freeze_thaw_cycles"].max() == pytest.approx(0.0)

    def test_counts_transitions_correctly(self):
        # 4 clear transitions: below → above → below → above → below → above → below → above
        # frozen:              T T F F T T F F
        # diff abs:              0 1 0 1 0 1 0  = 3 transitions / 2 = 1.5 cycles
        temps = [-2, -1, 1, 2, -1, -2, 1, 2]
        df = pd.DataFrame({
            "hrrr_temperature_2m": temps,
            "asset_id": "TX-001",
        })
        ftc = FreezeThawtCycleCounter(rolling_window_h=10)
        result = ftc.compute(df)
        # 3 state changes in 8 hours = 1.5 cycles (transition counting / 2)
        assert result["feat_freeze_thaw_cycles"].iloc[-1] == pytest.approx(1.5)

    def test_missing_temp_returns_nan(self):
        df = pd.DataFrame({"asset_id": ["TX-001"]})
        ftc = FreezeThawtCycleCounter()
        result = ftc.compute(df)
        assert result["feat_freeze_thaw_cycles"].isna().all()

    def test_input_not_modified(self, freezing_env_df):
        original_cols = list(freezing_env_df.columns)
        FreezeThawtCycleCounter().compute(freezing_env_df)
        assert list(freezing_env_df.columns) == original_cols
