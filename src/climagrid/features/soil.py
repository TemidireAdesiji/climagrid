"""
SoilSaturationIndex: ground stability proxy for buried cables and pole foundations.

High soil saturation increases risk of:
- Pole foundation heave and subsidence (distribution poles)
- Underground cable sheath damage from soil movement
- Increased corrosion rate for grounding systems
- Reduced insulation resistance in underground duct banks

The index combines USDA NRCS soil moisture with precipitation accumulation
when direct soil moisture readings are unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SoilSaturationIndex:
    """
    Computes normalized soil saturation index [0, 1].

    Strategy
    --------
    1. If NRCS soil moisture % is available, normalize against field capacity
       (nominally 40% VWC for loam soils; configurable).
    2. If NRCS data is unavailable, estimate from cumulative precipitation
       using a simple linear proxy (saturates at 100mm cumulative precip
       over the rolling window).

    Parameters
    ----------
    soil_col:
        NRCS volumetric soil moisture column (%).
    precip_col:
        Precipitation column for fallback estimation.
    field_capacity_pct:
        Volumetric water content at field capacity (%). Default 40%.
    wilting_point_pct:
        Volumetric water content at wilting point (%). Default 12%.
    rolling_window_h:
        Rolling window for precipitation accumulation fallback.

    Example
    -------
    >>> ssi = SoilSaturationIndex()
    >>> df = ssi.compute(env_df)
    >>> df["feat_soil_saturation_index"]
    """

    def __init__(
        self,
        soil_col: str = "nrcs_soil_moisture_pct",
        precip_col: str = "hrrr_precipitation_rate",
        field_capacity_pct: float = 40.0,
        wilting_point_pct: float = 12.0,
        rolling_window_h: int = 72,
        precip_saturation_mm: float = 100.0,
    ):
        self._soil_col = soil_col
        self._precip_col = precip_col
        self._field_capacity_pct = field_capacity_pct
        self._wilting_point_pct = wilting_point_pct
        self._rolling_window_h = rolling_window_h
        self._precip_saturation_mm = precip_saturation_mm

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feat_soil_saturation_index column [0, 1]. Returns a copy."""
        df = df.copy()

        _PRECIP_FALLBACKS = ["nasa_precipitation", "ncei_precipitation_daily"]

        if self._soil_col in df.columns and df[self._soil_col].notna().any():
            moisture = df[self._soil_col]
            # Normalize between wilting point (0) and field capacity (1)
            available_water = self._field_capacity_pct - self._wilting_point_pct
            index = (moisture - self._wilting_point_pct) / available_water
            df["feat_soil_saturation_index"] = np.clip(index, 0.0, 1.0)

        else:
            precip_col = self._precip_col if self._precip_col in df.columns else next(
                (c for c in _PRECIP_FALLBACKS if c in df.columns), None
            )
            if precip_col is not None:
                precip = df[precip_col].fillna(0.0)
                cumulative = precip.rolling(self._rolling_window_h, min_periods=1).sum()
                df["feat_soil_saturation_index"] = np.clip(
                    cumulative / self._precip_saturation_mm, 0.0, 1.0
                )
            else:
                df["feat_soil_saturation_index"] = float("nan")

        return df
