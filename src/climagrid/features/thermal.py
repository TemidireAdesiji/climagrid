"""
ThermalStressIndex — transformer thermal aging and heat stress features.

Implements two metrics per IEEE C57.91-2011 (Guide for Loading
Mineral-Oil-Immersed Transformers):

1. Functional Aging Acceleration factor (FAA) — the Arrhenius-based ratio
   of the insulation aging rate at observed temperature vs. the reference
   (110°C hotspot for normal aging).

2. Heat hours above threshold — cumulative hours in a rolling window where
   the ambient temperature exceeds a configurable threshold (default 35°C).

References
----------
IEEE C57.91-2011, Section 5.1 (Normal Insulation Life and Aging)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# IEEE C57.91 Arrhenius constants for normal aging (thermally upgraded paper)
_EA_OVER_K = 15000.0   # E_A / k_B in Kelvin  (derived from IEEE constants)
_T_REF_K = 383.0       # Reference hotspot temperature: 110°C = 383 K


class ThermalStressIndex:
    """
    Computes transformer thermal aging features from ambient temperature data.

    Parameters
    ----------
    temp_col:
        Column name for ambient temperature in °C.
    hotspot_rise:
        Hotspot temperature rise above ambient in °C. Default 25°C is a
        typical value for distribution transformers (IEEE C57.91 Table 2).
    heat_threshold_c:
        Temperature threshold in °C for counting heat-stress hours.
        Default 35°C (common threshold for transformer derating advisories).
    rolling_window_h:
        Rolling window size in hours for cumulative heat-hour calculation.
        Default 168 (one week).

    Example
    -------
    >>> tsi = ThermalStressIndex()
    >>> df = tsi.compute(asset_env_df)
    >>> df[["feat_thermal_aging_factor", "feat_heat_hours_above_35c"]]
    """

    def __init__(
        self,
        temp_col: str = "hrrr_temperature_2m",
        hotspot_rise: float = 25.0,
        heat_threshold_c: float = 35.0,
        rolling_window_h: int = 168,
    ):
        self._temp_col = temp_col
        self._hotspot_rise = hotspot_rise
        self._heat_threshold_c = heat_threshold_c
        self._rolling_window_h = rolling_window_h

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add thermal stress columns to df in-place (returns a copy).

        The DataFrame must contain a temperature column in °C and an
        'asset_id' column for grouping. Timestamp ordering is assumed.

        Parameters
        ----------
        df:
            Asset-level environmental DataFrame from AssetEnvironmentJoiner.

        Returns
        -------
        pd.DataFrame
            Input df with two new columns added:
            - feat_thermal_aging_factor
            - feat_heat_hours_above_35c
        """
        df = df.copy()

        if self._temp_col not in df.columns:
            # Try fallback columns
            for fallback in ["nasa_temperature_2m", "ncei_temperature_max"]:
                if fallback in df.columns:
                    self._temp_col = fallback
                    break
            else:
                df["feat_thermal_aging_factor"] = float("nan")
                df["feat_heat_hours_above_35c"] = float("nan")
                return df

        temp_c = df[self._temp_col]
        hotspot_c = temp_c + self._hotspot_rise
        hotspot_k = hotspot_c + 273.15

        # Arrhenius FAA: ratio of aging rate at observed vs. reference temperature
        df["feat_thermal_aging_factor"] = np.exp(
            _EA_OVER_K * (1.0 / _T_REF_K - 1.0 / hotspot_k)
        )

        # Heat hours above threshold — rolling count per asset
        above_threshold = (temp_c > self._heat_threshold_c).astype(float)

        if "asset_id" in df.columns:
            df["feat_heat_hours_above_35c"] = (
                df.groupby("asset_id")["feat_thermal_aging_factor"]
                .transform(lambda _: above_threshold.rolling(
                    self._rolling_window_h, min_periods=1
                ).sum())
            )
            # simpler: just apply rolling per group
            result_col = []
            for _, grp in df.groupby("asset_id", sort=False):
                above = (grp[self._temp_col] > self._heat_threshold_c).astype(float)
                rolled = above.rolling(self._rolling_window_h, min_periods=1).sum()
                result_col.append(rolled)
            df["feat_heat_hours_above_35c"] = pd.concat(result_col).reindex(df.index)
        else:
            df["feat_heat_hours_above_35c"] = above_threshold.rolling(
                self._rolling_window_h, min_periods=1
            ).sum()

        return df
