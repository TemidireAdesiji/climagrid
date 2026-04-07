"""
NOAA NCEI ISD adapter — Integrated Surface Database hourly observations.

Fetches hourly surface station data (temperature, wind, precipitation,
humidity) from NOAA's Climate Data Online API. Free API key required:
https://www.ncdc.noaa.gov/cdo-web/token

Docs: https://www.ncdc.noaa.gov/cdo-web/webservices/v2
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox

_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# GHCND data type IDs available from official airport/ASOS stations → climagrid column names
# GHCND is a daily dataset; these are daily summary values (not hourly observations)
_DTYPE_MAP = {
    "TMAX": "ncei_temperature_max",    # daily max temperature (°C)
    "TMIN": "ncei_temperature_min",    # daily min temperature (°C)
    "AWND": "ncei_wind_speed",         # average daily wind speed (m/s)
    "PRCP": "ncei_precipitation_daily",# daily precipitation (mm)
    "RHAV": "ncei_relative_humidity",  # average daily relative humidity (%)
}

# CDO dataset — GHCND has reliable daily data from official ASOS/airport stations
_DATASET = "GHCND"


class NceiAdapter(BaseEnvironmentalSource):
    """
    Fetches hourly surface observations from NOAA NCEI CDO API.

    Finds the nearest NCEI station within the bounding box and returns
    its hourly observation record for the requested time range.

    Parameters
    ----------
    api_token:
        NOAA CDO API token. If None, reads from NOAA_CDO_TOKEN env var.
    radius_km:
        Search radius for finding the nearest station (default 50 km).
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        radius_km: float = 50.0,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ):
        self._token = api_token or os.environ.get("NOAA_CDO_TOKEN", "")
        self._radius_km = radius_km
        self._timeout = timeout
        self._session = session or requests.Session()
        if self._token:
            self._session.headers["token"] = self._token

    @property
    def source_name(self) -> str:
        return "noaa_ncei"

    def fetch(
        self,
        bbox: BoundingBox,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        start_dt = self._ensure_utc(start_dt)
        end_dt = self._ensure_utc(end_dt)
        self._validate_time_range(start_dt, end_dt)

        station_id = self._find_nearest_station(bbox, start_dt, end_dt)
        if station_id is None:
            lat, lon = bbox.center
            return pd.DataFrame({"lat": [lat], "lon": [lon], "timestamp": [start_dt]})

        return self._fetch_station_data(station_id, bbox.center, start_dt, end_dt)

    def _find_nearest_station(
        self, bbox: BoundingBox, start_dt: datetime, end_dt: datetime
    ) -> Optional[str]:
        lat, lon = bbox.center
        params = {
            "datasetid": _DATASET,
            "extent": f"{bbox.min_lat},{bbox.min_lon},{bbox.max_lat},{bbox.max_lon}",
            "startdate": start_dt.strftime("%Y-%m-%d"),
            "enddate": end_dt.strftime("%Y-%m-%d"),
            "datatypeid": "TMAX",
            "limit": 25,
        }
        try:
            resp = self._session.get(
                f"{_BASE_URL}/stations", params=params, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None

        results = data.get("results", [])
        if not results:
            return None

        # Pick the station with the highest data coverage
        best = max(results, key=lambda s: s.get("datacoverage", 0))
        return best["id"]

    def _fetch_station_data(
        self,
        station_id: str,
        center: tuple[float, float],
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        lat, lon = center
        params = {
            "datasetid": _DATASET,
            "stationid": station_id,
            "startdate": start_dt.strftime("%Y-%m-%d"),
            "enddate": end_dt.strftime("%Y-%m-%d"),
            "datatypeid": ",".join(_DTYPE_MAP.keys()),
            "limit": 1000,
            "units": "metric",
        }
        try:
            resp = self._session.get(
                f"{_BASE_URL}/data", params=params, timeout=self._timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return pd.DataFrame({"lat": [lat], "lon": [lon], "timestamp": [start_dt]})

        results = data.get("results", [])
        if not results:
            return pd.DataFrame({"lat": [lat], "lon": [lon], "timestamp": [start_dt]})

        raw = pd.DataFrame(results)
        raw["timestamp"] = pd.to_datetime(raw["date"], utc=True)

        # Pivot so each datatype becomes a column
        pivoted = raw.pivot_table(
            index="timestamp", columns="datatype", values="value", aggfunc="mean"
        ).reset_index()

        pivoted = pivoted.rename(columns=_DTYPE_MAP)
        pivoted["lat"] = lat
        pivoted["lon"] = lon

        return pivoted
