"""
NASA POWER adapter — surface meteorology via NASA POWER REST API.

NASA POWER provides MERRA-2-based hourly surface meteorology at any
lat/lon point globally. No API key required.

Docs: https://power.larc.nasa.gov/docs/services/api/
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import requests

from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox

_BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# NASA POWER parameter names → climagrid column names
_PARAM_MAP = {
    "T2M": "nasa_temperature_2m",
    "WS10M": "nasa_wind_speed_10m",
    "ALLSKY_SFC_SW_DWN": "nasa_solar_irradiance_ghi",
    "RH2M": "nasa_relative_humidity_2m",
    "PRECTOTCORR": "nasa_precipitation",
}


class NasaPowerAdapter(BaseEnvironmentalSource):
    """
    Fetches hourly surface meteorology from NASA POWER for point locations.

    For a bounding box query the center point is used. For asset-level
    queries use AssetEnvironmentJoiner to call fetch_point() per asset.
    """

    def __init__(self, timeout: int = 60, session: requests.Session | None = None):
        self._timeout = timeout
        self._session = session or requests.Session()

    @property
    def source_name(self) -> str:
        return "nasa_power"

    def fetch(
        self,
        bbox: BoundingBox,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Fetch for the center point of the bounding box."""
        lat, lon = bbox.center
        return self.fetch_point(lat, lon, start_dt, end_dt)

    def fetch_point(
        self,
        lat: float,
        lon: float,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly NASA POWER data for a single lat/lon point."""
        start_dt = self._ensure_utc(start_dt)
        end_dt = self._ensure_utc(end_dt)
        self._validate_time_range(start_dt, end_dt)

        params = {
            "parameters": ",".join(_PARAM_MAP.keys()),
            "community": "RE",
            "longitude": round(lon, 4),
            "latitude": round(lat, 4),
            "start": start_dt.strftime("%Y%m%d"),
            "end": end_dt.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }

        resp = self._session.get(_BASE_URL, params=params, timeout=self._timeout)  # type: ignore[arg-type]
        resp.raise_for_status()
        payload = resp.json()

        return self._parse_response(payload, lat, lon)

    def _parse_response(
        self, payload: dict, lat: float, lon: float
    ) -> pd.DataFrame:
        """Convert NASA POWER JSON response to a climagrid-schema DataFrame."""
        if "properties" not in payload or "parameter" not in payload["properties"]:
            raise ValueError("Unexpected NASA POWER response structure")

        parameters = payload["properties"]["parameter"]

        # Build timestamp index from the first available parameter's keys
        first_key = next(iter(parameters))
        timestamps = pd.to_datetime(
            list(parameters[first_key].keys()), format="%Y%m%d%H", utc=True
        )

        df = pd.DataFrame(index=timestamps)
        df.index.name = "timestamp"

        for nasa_param, cg_col in _PARAM_MAP.items():
            if nasa_param in parameters:
                series = pd.Series(parameters[nasa_param], dtype=float)
                series.index = pd.to_datetime(
                    series.index, format="%Y%m%d%H", utc=True
                )
                # NASA POWER uses -999 as fill value
                series = series.replace(-999.0, float("nan"))
                df[cg_col] = series

        df["lat"] = lat
        df["lon"] = lon
        df = df.reset_index()

        return df  # type: ignore[no-any-return]
