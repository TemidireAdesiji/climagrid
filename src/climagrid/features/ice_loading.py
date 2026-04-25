"""
IceLoadingRisk — ice accretion risk on overhead conductors.

Based on ASCE 7-22 (Minimum Design Loads and Associated Criteria for
Buildings and Other Structures) Chapter 10 (Ice Loads) conditions,
combined with the NOAA/NWS ice storm criteria.

Ice accretion risk is HIGH when:
- Air temperature is between -10°C and +2°C (supercooled liquid water zone)
- Precipitation is occurring (freezing rain)
- Wind speed > 5 m/s (driving freezing rain onto conductors)

The output is normalized to [0, 1] for use as an ML feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IceLoadingRisk:
    """
    Computes normalized ice loading risk for overhead T&D conductors.

    Parameters
    ----------
    temp_col:
        Temperature in °C.
    precip_col:
        Precipitation rate in mm/hr.
    wind_col:
        Wind speed in m/s.
    temp_ice_min_c:
        Lower temperature bound for ice accretion (default -10°C).
    temp_ice_max_c:
        Upper temperature bound for ice accretion (default +2°C).
    precip_threshold:
        Minimum precipitation rate (mm/hr) to trigger ice risk scoring.

    Example
    -------
    >>> ilr = IceLoadingRisk()
    >>> df = ilr.compute(env_df)
    >>> df["feat_ice_loading_risk"]
    """

    def __init__(
        self,
        temp_col: str = "hrrr_temperature_2m",
        precip_col: str = "hrrr_precipitation_rate",
        wind_col: str = "hrrr_wind_speed_10m",
        temp_ice_min_c: float = -10.0,
        temp_ice_max_c: float = 2.0,
        precip_threshold: float = 0.1,
    ):
        self._temp_col = temp_col
        self._precip_col = precip_col
        self._wind_col = wind_col
        self._temp_ice_min_c = temp_ice_min_c
        self._temp_ice_max_c = temp_ice_max_c
        self._precip_threshold = precip_threshold

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feat_ice_loading_risk column [0, 1]. Returns a copy."""
        df = df.copy()

        temp = self._get_col(df, self._temp_col, ["nasa_temperature_2m", "ncei_temperature_max"])
        precip = self._get_col(df, self._precip_col, ["nasa_precipitation", "ncei_precipitation_daily"])
        wind = self._get_col(df, self._wind_col, ["nasa_wind_speed_10m", "ncei_wind_speed"])

        if temp is None:
            df["feat_ice_loading_risk"] = float("nan")
            return df

        # Temperature factor: peaks at 0°C, zero outside [min, max]
        temp_range = self._temp_ice_max_c - self._temp_ice_min_c
        temp_factor = np.where(
            (temp >= self._temp_ice_min_c) & (temp <= self._temp_ice_max_c),
            1.0 - np.abs(temp) / max(abs(self._temp_ice_min_c), abs(self._temp_ice_max_c)),
            0.0,
        )
        temp_factor = np.clip(temp_factor, 0.0, 1.0)

        # Precipitation factor: normalized 0–1 capped at 10 mm/hr
        if precip is not None:
            precip_factor = np.where(
                precip >= self._precip_threshold,
                np.clip(precip / 10.0, 0.0, 1.0),
                0.0,
            )
        else:
            precip_factor = np.where(temp_factor > 0, 0.5, 0.0)

        # Wind factor: normalized 0–1, capped at 20 m/s
        if wind is not None:
            wind_factor = np.clip(wind / 20.0, 0.0, 1.0)
        else:
            wind_factor = 0.5  # assume moderate wind if unknown

        # Combined risk: geometric mean of factors (requires ALL conditions)
        risk = (temp_factor * precip_factor * wind_factor) ** (1.0 / 3.0)
        df["feat_ice_loading_risk"] = np.clip(risk, 0.0, 1.0)

        return df

    @staticmethod
    def _get_col(df: pd.DataFrame, primary: str, fallbacks: list[str]):
        if primary in df.columns:
            return df[primary].values
        for fb in fallbacks:
            if fb in df.columns:
                return df[fb].values
        return None
