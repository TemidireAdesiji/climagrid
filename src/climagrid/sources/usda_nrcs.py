"""
USDA NRCS adapter — SCAN and SNOTEL soil sensor data.

Fetches soil moisture, soil temperature, and snow water equivalent from
the USDA Natural Resources Conservation Service AWDB REST API.

No API key required. Public data.

Docs: https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import requests

from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox

_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"

# NRCS element codes → climagrid column names
# Elements with depth/height sensors require the ':*' wildcard in API requests
_ELEMENT_MAP = {
    "SMS": "nrcs_soil_moisture_pct",       # Soil Moisture % (volumetric)
    "STO": "nrcs_soil_temperature",        # Soil Temperature °C
    "WTEQ": "nrcs_snow_water_equivalent",  # Snow Water Equivalent (mm)
}

# API element query string — wildcard depth for sensor arrays, bare code for point sensors
_ELEMENTS_QUERY = "SMS:*,STO:*,WTEQ"


class NrcsAdapter(BaseEnvironmentalSource):
    """
    Fetches soil and snow data from USDA NRCS SCAN/SNOTEL network.

    Finds the nearest active SCAN or SNOTEL station within the bounding box,
    fetches hourly readings, and returns a climagrid DataFrame.

    Parameters
    ----------
    max_distance_km:
        Maximum search distance for nearest station (default 200 km).
        SCAN/SNOTEL networks are sparse in some regions.
    """

    def __init__(
        self,
        max_distance_km: float = 200.0,
        timeout: int = 30,
        session: requests.Session | None = None,
    ):
        self._max_distance_km = max_distance_km
        self._timeout = timeout
        self._session = session or requests.Session()

    @property
    def source_name(self) -> str:
        return "usda_nrcs"

    def fetch(
        self,
        bbox: BoundingBox,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        start_dt = self._ensure_utc(start_dt)
        end_dt = self._ensure_utc(end_dt)
        self._validate_time_range(start_dt, end_dt)

        lat, lon = bbox.center
        station = self._find_nearest_station(lat, lon)
        if station is None:
            return self._empty_df(lat, lon, start_dt)

        return self._fetch_station_data(station, lat, lon, start_dt, end_dt)

    def _find_nearest_station(
        self, lat: float, lon: float
    ) -> dict | None:
        """Return the nearest active SCAN/SNOTEL station that has soil sensors."""
        params = {
            "activeOnly": "true",
            "networkCds": "SCAN,SNTL",
            "elements": _ELEMENTS_QUERY,
        }
        try:
            resp = self._session.get(
                f"{_BASE_URL}/stations",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            stations = resp.json()
        except Exception:
            return None

        if not stations:
            return None

        def _dist(s: dict) -> float:
            return _haversine(lat, lon, float(s["latitude"]), float(s["longitude"]))

        nearest = min(stations, key=_dist)
        dist_km = _dist(nearest)
        if dist_km > self._max_distance_km:
            return None

        nearest["_distance_km"] = dist_km
        return nearest  # type: ignore[return-value, no-any-return]

    def _fetch_station_data(
        self,
        station: dict,
        asset_lat: float,
        asset_lon: float,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        triplet = station["stationTriplet"]
        dist_km = station.get("_distance_km", float("nan"))

        params = {
            "stationTriplets": triplet,
            "elements": _ELEMENTS_QUERY,
            "beginDate": start_dt.strftime("%Y-%m-%d"),
            "endDate": end_dt.strftime("%Y-%m-%d"),
            "duration": "HOURLY",
        }
        try:
            resp = self._session.get(
                f"{_BASE_URL}/data",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return self._empty_df(asset_lat, asset_lon, start_dt)

        return self._parse_response(payload, asset_lat, asset_lon, dist_km)

    def _parse_response(
        self,
        payload: list,
        lat: float,
        lon: float,
        dist_km: float,
    ) -> pd.DataFrame:
        if not payload:
            return self._empty_df(lat, lon)

        station_data = payload[0] if isinstance(payload, list) else payload
        rows: list[dict] = []

        for element_data in station_data.get("data", []):
            elem = element_data.get("stationElement", {})
            # Live API uses "elementCode"; mock used "elementCd" — handle both
            element_cd = elem.get("elementCode") or elem.get("elementCd", "")
            col_name = _ELEMENT_MAP.get(element_cd)
            if col_name is None:
                continue

            for entry in element_data.get("values", []):
                date_str = entry.get("date") or entry.get("dateTime")
                value = entry.get("value")
                if date_str is None or value is None:
                    continue
                try:
                    ts = pd.to_datetime(date_str, utc=True)
                    rows.append({"timestamp": ts, col_name: float(value)})
                except (ValueError, TypeError):
                    continue

        if not rows:
            return self._empty_df(lat, lon)

        df = pd.DataFrame(rows)
        # Average across sensor depths for the same element/timestamp
        df = df.groupby("timestamp").mean().reset_index()
        df["lat"] = lat
        df["lon"] = lon
        df["nrcs_station_distance_km"] = dist_km
        return df  # type: ignore[no-any-return]

    @staticmethod
    def _empty_df(lat: float, lon: float, ts: datetime | None = None) -> pd.DataFrame:
        row: dict = {"lat": lat, "lon": lon}
        if ts is not None:
            row["timestamp"] = ts
        return pd.DataFrame([row])


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
