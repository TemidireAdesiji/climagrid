"""
ConductorSagIndex — thermal sag estimation for overhead T&D lines.

When conductor temperature rises, the aluminum/ACSR strands expand and
the conductor sags downward, reducing ground clearance. Excessive sag
causes regulatory violations and phase-to-ground faults.

Thermal sag is governed by the IEEE 738-2012 (Standard for Calculating
the Current-Temperature Relationship of Bare Overhead Conductors). This
module implements a simplified version using ambient temperature and solar
irradiance as primary inputs.

The output is a normalized index [0, 1] representing sag relative to the
maximum allowable sag (configurable), suitable as an ML feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConductorSagIndex:
    """
    Computes normalized conductor thermal sag index from weather inputs.

    Simplified IEEE 738-2012 heat balance:
        T_conductor ≈ T_ambient + (I²R + Q_solar - Q_convective) / (thermal_capacity)

    For stress *feature* purposes (not full current rating), we approximate
    the conductor temperature as ambient + solar heating - convective cooling,
    then compute sag as a fraction of the maximum design sag.

    Parameters
    ----------
    temp_col:
        Ambient temperature column in °C.
    solar_col:
        Global horizontal irradiance column in W/m².
    wind_col:
        Wind speed column in m/s. Wind is the primary cooling mechanism.
    max_sag_temp_c:
        Conductor temperature at which sag reaches the design maximum.
        Default 75°C (typical for ACSR "Drake" conductor per IEEE 738).
    conductor_absorptivity:
        Solar absorptivity of the conductor surface (0–1). Default 0.5.
    conductor_emissivity:
        Emissivity for radiated cooling (0–1). Default 0.5.
    conductor_diameter_mm:
        Conductor outer diameter for convective heat loss. Default 28.1 mm (Drake ACSR).

    Example
    -------
    >>> csi = ConductorSagIndex()
    >>> df = csi.compute(env_df)
    >>> df["feat_conductor_sag_index"]
    """

    # Stefan-Boltzmann constant W/(m²·K⁴)
    _SIGMA = 5.6704e-8

    def __init__(
        self,
        temp_col: str = "hrrr_temperature_2m",
        solar_col: str = "hrrr_solar_irradiance_ghi",
        wind_col: str = "hrrr_wind_speed_10m",
        max_sag_temp_c: float = 75.0,
        conductor_absorptivity: float = 0.5,
        conductor_emissivity: float = 0.5,
        conductor_diameter_mm: float = 28.1,
    ):
        self._temp_col = temp_col
        self._solar_col = solar_col
        self._wind_col = wind_col
        self._max_sag_temp_c = max_sag_temp_c
        self._alpha = conductor_absorptivity
        self._eps = conductor_emissivity
        self._d = conductor_diameter_mm / 1000.0  # convert to metres

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feat_conductor_sag_index column [0, 1]. Returns a copy."""
        df = df.copy()

        temp = self._resolve_col(df, self._temp_col, ["nasa_temperature_2m", "ncei_temperature_max"])
        solar = self._resolve_col(df, self._solar_col, ["nasa_solar_irradiance_ghi"])
        wind = self._resolve_col(df, self._wind_col, ["nasa_wind_speed_10m", "ncei_wind_speed"])

        if temp is None:
            df["feat_conductor_sag_index"] = float("nan")
            return df

        # Defaults when data not available
        if solar is None:
            solar = np.full_like(temp, 400.0)   # moderate irradiance
        if wind is None:
            wind = np.full_like(temp, 1.0)       # near-calm (conservative)

        wind = np.maximum(wind, 0.5)  # prevent divide-by-zero in convection

        # Solar heat gain per unit length (W/m)
        q_solar = self._alpha * solar * self._d

        # Convective cooling (simplified Morgan formula per IEEE 738 Eq. 3a)
        # q_conv = (1.01 + 1.35 * Re^0.52) * k_f * (T_c - T_a)
        # Simplified for feature purposes: linear approximation
        # Full implementation would require air density, viscosity, thermal conductivity
        # We use: q_conv ≈ h_c * d * delta_T where h_c ≈ 10 * sqrt(wind) W/(m²·K) (typical)
        h_c = 10.0 * np.sqrt(wind)
        q_conv_per_k = h_c * self._d  # W/(m·K) per unit temperature rise

        # Radiative cooling (W/m) at ambient temperature (lower bound)
        t_amb_k = temp + 273.15
        q_rad = self._eps * self._sigma_val * np.pi * self._d * t_amb_k**4

        # Steady-state conductor temperature rise above ambient (°C)
        # Solving: q_solar = q_conv_per_k * delta_T + delta_radiation (ignoring radiation delta for simplicity)
        delta_t = np.maximum(q_solar / q_conv_per_k, 0.0)
        t_conductor = temp + delta_t

        # Sag index: ratio of conductor temperature to max allowable, clamped [0,1]
        # Using linear approximation: sag scales approximately linearly with temperature
        # (IEEE 738 Table B.1 confirms near-linear relationship for typical conductors)
        baseline_temp = 25.0  # °C design reference temperature
        sag_index = (t_conductor - baseline_temp) / (self._max_sag_temp_c - baseline_temp)
        df["feat_conductor_sag_index"] = np.clip(sag_index, 0.0, 1.0)

        return df

    @property
    def _sigma_val(self) -> float:
        return self._SIGMA

    @staticmethod
    def _resolve_col(df: pd.DataFrame, primary: str, fallbacks: list[str]):
        if primary in df.columns:
            return df[primary].values.astype(float)
        for fb in fallbacks:
            if fb in df.columns:
                return df[fb].values.astype(float)
        return None
