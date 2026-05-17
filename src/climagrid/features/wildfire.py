"""
WildfireProximityScore — normalized wildfire risk for T&D infrastructure.

Wildfire is the fastest-growing cause of major U.S. transmission and
distribution outages. Key risk pathways:
- Direct flame contact with conductors (phase-to-phase faults)
- Smoke particles reducing air insulation strength (flashovers)
- Pole and structure ignition
- Post-fire debris flow and slope instability damaging underground lines

The score combines:
1. Distance to nearest active fire perimeter (from USFS WFIGS)
2. Optionally: NOAA HRRR-derived fire weather index (temperature, humidity, wind)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from climagrid.sources.usfs_wfigs import compute_proximity

# Distance decay: risk drops to ~5% at this distance (km)
_HALF_DISTANCE_KM = 10.0


class WildfireProximityScore:
    """
    Computes a normalized wildfire proximity score [0, 1] for asset locations.

    Parameters
    ----------
    max_risk_distance_km:
        Distance within which risk = 1.0 (fire perimeter adjacent). Default 1 km.
    zero_risk_distance_km:
        Distance beyond which risk ≈ 0. Default 200 km.
    use_fire_weather:
        If True and fire weather columns are present, modulate score by
        fire weather conditions (temperature, humidity, wind).
    temp_col:
        Temperature column for fire weather modulation.
    rh_col:
        Relative humidity column.
    wind_col:
        Wind speed column.

    Example
    -------
    >>> wfigs = WfigsAdapter()
    >>> fires_df = wfigs.fetch(bbox, start_dt, end_dt)
    >>> score = WildfireProximityScore()
    >>> df = score.compute(asset_df, fires_df)
    >>> df["feat_wildfire_proximity"]
    """

    def __init__(
        self,
        max_risk_distance_km: float = 1.0,
        zero_risk_distance_km: float = 200.0,
        use_fire_weather: bool = True,
        temp_col: str = "hrrr_temperature_2m",
        rh_col: str = "hrrr_relative_humidity_2m",
        wind_col: str = "hrrr_wind_speed_10m",
    ):
        self._max_risk_km = max_risk_distance_km
        self._zero_risk_km = zero_risk_distance_km
        self._use_fire_weather = use_fire_weather
        self._temp_col = temp_col
        self._rh_col = rh_col
        self._wind_col = wind_col

    def compute(self, df: pd.DataFrame, fires_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feat_wildfire_proximity column to df.

        Parameters
        ----------
        df:
            Asset-level environmental DataFrame with lat/lon columns.
        fires_df:
            DataFrame returned by WfigsAdapter.fetch() with fire perimeter data.

        Returns
        -------
        pd.DataFrame
            Input df with feat_wildfire_proximity [0,1] and
            wfigs_nearest_fire_km, wfigs_fire_active, wfigs_fire_area_ha added.
        """
        df = df.copy()

        if fires_df.empty or "lat" not in df.columns:
            df["feat_wildfire_proximity"] = 0.0
            df["wfigs_nearest_fire_km"] = float("inf")
            df["wfigs_fire_active"] = False
            df["wfigs_fire_area_ha"] = 0.0
            return df

        # For each unique asset location, compute proximity
        location_cache: dict[tuple[float, float], tuple[float, bool, float]] = {}

        def _proximity(row: pd.Series) -> tuple[float, bool, float]:
            key = (row["lat"], row["lon"])
            if key not in location_cache:
                location_cache[key] = compute_proximity(row["lat"], row["lon"], fires_df)
            return location_cache[key]

        proximities = df.apply(_proximity, axis=1)
        df["wfigs_nearest_fire_km"] = [p[0] for p in proximities]
        df["wfigs_fire_active"] = [p[1] for p in proximities]
        df["wfigs_fire_area_ha"] = [p[2] for p in proximities]

        # Distance-based score: exponential decay
        dist = df["wfigs_nearest_fire_km"]
        base_score = np.where(
            dist <= self._max_risk_km,
            1.0,
            np.exp(
                -math.log(20) * (dist - self._max_risk_km)
                / (self._zero_risk_km - self._max_risk_km)
            ),
        )
        base_score = np.clip(base_score, 0.0, 1.0)

        # Fire weather modulation (optional)
        if self._use_fire_weather:
            weather_factor = self._fire_weather_factor(df)
            base_score = np.clip(base_score * weather_factor, 0.0, 1.0)

        df["feat_wildfire_proximity"] = base_score
        return df

    def _fire_weather_factor(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a multiplicative modulation factor [0.5, 2.0] based on
        fire weather conditions. Values > 1 amplify risk under extreme
        conditions; values < 1 reduce risk under benign conditions.
        """
        factor = np.ones(len(df))

        # High temperature increases risk
        if self._temp_col in df.columns:
            temp = df[self._temp_col].fillna(20.0)
            factor *= np.clip(1.0 + (temp - 20.0) / 40.0, 0.5, 2.0)

        # Low humidity increases risk
        if self._rh_col in df.columns:
            rh = df[self._rh_col].fillna(50.0)
            factor *= np.clip(2.0 - rh / 50.0, 0.5, 2.0)

        # High wind increases risk
        if self._wind_col in df.columns:
            wind = df[self._wind_col].fillna(5.0)
            factor *= np.clip(1.0 + wind / 20.0, 0.5, 2.0)

        return np.clip(factor, 0.5, 2.0)
