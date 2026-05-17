"""
USFS / NIFC WFIGS adapter — wildfire perimeter data.

Fetches active and year-to-date fire perimeters from the National
Interagency Fire Center (NIFC) Wildland Fire Interagency Geospatial
Services (WFIGS) ArcGIS REST API.

No API key required. Public data.

Docs: https://data-nifc.opendata.arcgis.com/
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import requests

from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox

# Current-year perimeters (active + contained fires for the current season)
_PERIMETERS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
    "/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
)

# Fields we want back
_OUT_FIELDS = "OBJECTID,poly_GISAcres,attr_FireDiscoveryDateTime,attr_ContainmentDateTime"


class WfigsAdapter(BaseEnvironmentalSource):
    """
    Fetches wildfire perimeter data from NIFC WFIGS for a bounding box.

    For each asset location, the joiner can use this data to compute:
    - Distance to the nearest fire perimeter edge
    - Whether any active fire is within a configurable radius
    - Area of the nearest fire
    """

    def __init__(
        self,
        timeout: int = 30,
        session: requests.Session | None = None,
    ):
        self._timeout = timeout
        self._session = session or requests.Session()

    @property
    def source_name(self) -> str:
        return "usfs_wfigs"

    def fetch(
        self,
        bbox: BoundingBox,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """
        Fetch current fire perimeters intersecting the bounding box.

        Returns a DataFrame with one row per fire, including centroid
        lat/lon and area. The joiner uses this to compute per-asset
        proximity scores.
        """
        start_dt = self._ensure_utc(start_dt)
        end_dt = self._ensure_utc(end_dt)

        geometry_filter = (
            f"{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}"
        )
        params = {
            "where": "1=1",
            "geometry": geometry_filter,
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": _OUT_FIELDS,
            "returnGeometry": "true",
            "geometryPrecision": "4",
            "outSR": "4326",
            "f": "geojson",
        }

        try:
            resp = self._session.get(
                _PERIMETERS_URL, params=params, timeout=self._timeout
            )
            resp.raise_for_status()
            geojson = resp.json()
        except Exception:
            return self._empty_df()

        return self._parse_geojson(geojson)

    def _parse_geojson(self, geojson: dict) -> pd.DataFrame:
        features = geojson.get("features", [])
        if not features:
            return self._empty_df()

        rows = []
        for feature in features:
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})

            centroid_lat, centroid_lon = _geojson_centroid(geom)

            rows.append(
                {
                    "fire_centroid_lat": centroid_lat,
                    "fire_centroid_lon": centroid_lon,
                    "fire_area_ha": (props.get("poly_GISAcres") or 0) * 0.404686,
                    "fire_discovery_dt": props.get("attr_FireDiscoveryDateTime"),
                    "fire_contained_dt": props.get("attr_ContainmentDateTime"),
                    "fire_active": props.get("attr_ContainmentDateTime") is None,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "fire_centroid_lat",
                "fire_centroid_lon",
                "fire_area_ha",
                "fire_discovery_dt",
                "fire_contained_dt",
                "fire_active",
            ]
        )


def _geojson_centroid(geometry: dict) -> tuple[float, float]:
    """Approximate centroid of a GeoJSON geometry."""
    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if geom_type == "Point":
        return float(coords[1]), float(coords[0])

    # Flatten all coordinate pairs for Polygon / MultiPolygon
    all_points: list[list[float]] = []

    def _flatten(c: list) -> None:
        if not c:
            return
        if isinstance(c[0], int | float):
            all_points.append(c)
        else:
            for sub in c:
                _flatten(sub)

    _flatten(coords)

    if not all_points:
        return 0.0, 0.0

    avg_lon = sum(p[0] for p in all_points) / len(all_points)
    avg_lat = sum(p[1] for p in all_points) / len(all_points)
    return avg_lat, avg_lon


def compute_proximity(
    asset_lat: float,
    asset_lon: float,
    fires_df: pd.DataFrame,
) -> tuple[float, bool, float]:
    """
    Compute wildfire proximity metrics for a single asset location.

    Returns
    -------
    (nearest_km, any_active, nearest_area_ha)
    """
    if fires_df.empty:
        return float("inf"), False, 0.0

    distances = fires_df.apply(
        lambda row: _haversine(
            asset_lat, asset_lon,
            row["fire_centroid_lat"], row["fire_centroid_lon"],
        ),
        axis=1,
    )
    idx_min = distances.idxmin()
    nearest_km = float(distances[idx_min])
    active = bool(fires_df.loc[idx_min, "fire_active"])
    area_ha = float(fires_df.loc[idx_min, "fire_area_ha"])  # type: ignore[arg-type]
    return nearest_km, active, area_ha


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
