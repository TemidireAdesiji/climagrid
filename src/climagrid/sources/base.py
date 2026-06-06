"""Abstract base for all environmental data source adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class BoundingBox(BaseModel):
    """Geographic bounding box in WGS-84 decimal degrees."""

    model_config = ConfigDict(frozen=True)

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __init__(
        self,
        min_lat: float | None = None,
        max_lat: float | None = None,
        min_lon: float | None = None,
        max_lon: float | None = None,
        **data: Any,
    ) -> None:
        # Support positional arguments in addition to keyword arguments
        if min_lat is not None:
            data["min_lat"] = min_lat
        if max_lat is not None:
            data["max_lat"] = max_lat
        if min_lon is not None:
            data["min_lon"] = min_lon
        if max_lon is not None:
            data["max_lon"] = max_lon
        super().__init__(**data)

    @model_validator(mode="after")
    def _check_bounds(self) -> BoundingBox:
        if self.min_lat >= self.max_lat:
            raise ValueError(f"min_lat ({self.min_lat}) must be < max_lat ({self.max_lat})")
        if self.min_lon >= self.max_lon:
            raise ValueError(f"min_lon ({self.min_lon}) must be < max_lon ({self.max_lon})")
        if not (-90 <= self.min_lat <= 90 and -90 <= self.max_lat <= 90):
            raise ValueError("Latitudes must be in [-90, 90]")
        if not (-180 <= self.min_lon <= 180 and -180 <= self.max_lon <= 180):
            raise ValueError("Longitudes must be in [-180, 180]")
        return self

    @classmethod
    def from_center(cls, lat: float, lon: float, radius_km: float) -> BoundingBox:
        """Create a bounding box centered on a point with a radius in km.

        Edges are clamped to the valid WGS-84 domain ([-90, 90] latitude,
        [-180, 180] longitude) so that centers near the poles or the
        antimeridian still yield a valid box rather than raising.
        """
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        return cls(
            min_lat=max(lat - lat_delta, -90.0),
            max_lat=min(lat + lat_delta, 90.0),
            min_lon=max(lon - lon_delta, -180.0),
            max_lon=min(lon + lon_delta, 180.0),
        )

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_lat + self.max_lat) / 2, (self.min_lon + self.max_lon) / 2)


class BaseEnvironmentalSource(ABC):
    """
    Common interface all data source adapters must implement.

    Each adapter fetches raw data for a geographic region and time window,
    returning a pandas DataFrame with columns conforming to climagrid.schema.
    """

    # Point-based sources (e.g. NASA POWER) return data for a single location
    # per request. They set this True and implement fetch_points() so the
    # orchestrator fetches one location per asset instead of a single point for
    # the whole, possibly geographically spread, asset set.
    point_based: bool = False

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier used as a column prefix (e.g. 'hrrr', 'nasa_power')."""

    @abstractmethod
    def fetch(
        self,
        bbox: BoundingBox,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """
        Fetch environmental data for a bounding box over a time range.

        Parameters
        ----------
        bbox:
            Geographic extent of the query.
        start_dt:
            Start of the time range (UTC-aware or naive UTC).
        end_dt:
            End of the time range (UTC-aware or naive UTC).

        Returns
        -------
        pd.DataFrame
            Rows indexed by (lat, lon, timestamp). Column names must be
            drawn from climagrid.schema.COLUMN_MAP.
        """

    def fetch_points(
        self,
        points: list[tuple[float, float]],
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        """
        Fetch data for multiple (lat, lon) point locations.

        Point-based sources (``point_based = True``) override this to return one
        block of rows per location, each tagged with its own ``lat``/``lon``, so
        every asset gets weather at its actual position rather than a single
        shared point. The default raises, since grid and station sources use
        :meth:`fetch` with a bounding box instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support per-point fetching"
        )

    def _fetch_points_via_bbox(
        self,
        points: list[tuple[float, float]],
        start_dt: datetime,
        end_dt: datetime,
        radius_km: float,
    ) -> pd.DataFrame:
        """
        Fetch per point by issuing a small bounding-box query around each one.

        Shared by station-based sources whose :meth:`fetch` resolves a bounding
        box to its nearest station: one small bbox per asset yields the nearest
        station to each asset rather than to the whole set's centroid.
        """
        frames: list[pd.DataFrame] = []
        for lat, lon in points:
            df = self.fetch(BoundingBox.from_center(lat, lon, radius_km), start_dt, end_dt)
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # Shared helpers available to all adapters
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _validate_time_range(start_dt: datetime, end_dt: datetime) -> None:
        if start_dt >= end_dt:
            raise ValueError(f"start_dt ({start_dt}) must be before end_dt ({end_dt})")
