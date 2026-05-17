"""Abstract base for all environmental data source adapters."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    import pandas as pd


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
        """Create a bounding box centered on a point with a radius in km."""
        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
        return cls(
            min_lat=lat - lat_delta,
            max_lat=lat + lat_delta,
            min_lon=lon - lon_delta,
            max_lon=lon + lon_delta,
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
