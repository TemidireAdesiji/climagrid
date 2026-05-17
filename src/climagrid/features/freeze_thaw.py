"""
FreezeThawtCycleCounter — counts freeze-thaw transitions in a rolling window.

Freeze-thaw cycling causes fatigue in:
- Overhead conductors (galloping, vibration)
- Insulator strings and polymer housings (moisture ingress, cracking)
- Wood and composite pole foundations (frost heave)
- Underground cable sheathing (ground movement)

A freeze-thaw cycle is counted each time temperature crosses 0°C in
either direction within a rolling time window.
"""

from __future__ import annotations

import pandas as pd


class FreezeThawtCycleCounter:
    """
    Counts freeze-thaw transitions from an hourly temperature time series.

    Parameters
    ----------
    temp_col:
        Column containing temperature in °C.
    freeze_threshold_c:
        Temperature below which the asset is considered "frozen".
        Default 0°C (water freezing point).
    rolling_window_h:
        Rolling window in hours over which to count cycles.
        Default 720 (30 days).

    Example
    -------
    >>> ftc = FreezeThawtCycleCounter()
    >>> df = ftc.compute(asset_env_df)
    >>> df["feat_freeze_thaw_cycles"]
    """

    def __init__(
        self,
        temp_col: str = "hrrr_temperature_2m",
        freeze_threshold_c: float = 0.0,
        rolling_window_h: int = 720,
    ):
        self._temp_col = temp_col
        self._freeze_threshold_c = freeze_threshold_c
        self._rolling_window_h = rolling_window_h

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feat_freeze_thaw_cycles column. Returns a copy."""
        df = df.copy()

        temp_col = self._temp_col
        if temp_col not in df.columns:
            for fallback in ["nasa_temperature_2m", "ncei_temperature_max"]:
                if fallback in df.columns:
                    temp_col = fallback
                    break
            else:
                df["feat_freeze_thaw_cycles"] = float("nan")
                return df

        def _count_transitions(series: pd.Series) -> pd.Series:
            frozen = series < self._freeze_threshold_c
            # A transition occurs wherever the frozen state changes
            transitions = frozen.astype(int).diff().abs().fillna(0)
            # Each complete cycle = 2 transitions (freeze + thaw); count transitions
            # and divide by 2 so we report full cycles
            cycles = transitions.rolling(self._rolling_window_h, min_periods=1).sum() / 2.0
            return cycles

        if "asset_id" in df.columns:
            result_parts = []
            for _, grp in df.groupby("asset_id", sort=False):
                part = _count_transitions(grp[temp_col])
                result_parts.append(part)
            df["feat_freeze_thaw_cycles"] = pd.concat(result_parts).reindex(df.index)
        else:
            df["feat_freeze_thaw_cycles"] = _count_transitions(df[temp_col])

        return df
