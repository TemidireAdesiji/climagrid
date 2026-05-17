"""
NOAA HRRR adapter — High-Resolution Rapid Refresh NWP model.

Fetches 3-km CONUS hourly analysis fields (temperature, wind, precipitation,
humidity, solar irradiance) via the Herbie package from NOAA NOMADS or
cloud archives (AWS NODD, Google, Azure).

Requires optional dependency: pip install climagrid[noaa-nwp]

Docs: https://herbie.readthedocs.io/
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox

try:
    from herbie import Herbie
    _HERBIE_AVAILABLE = True
except ImportError:
    _HERBIE_AVAILABLE = False


# HRRR searchString patterns for the variables we need
# These match GRIB2 level/parameter descriptions
_HRRR_FIELDS = {
    "TMP:2 m": "hrrr_temperature_2m",
    "UGRD:10 m": "_hrrr_u_wind",          # combined with VGRD for speed/direction
    "VGRD:10 m": "_hrrr_v_wind",
    "RH:2 m": "hrrr_relative_humidity_2m",
    "APCP:surface": "hrrr_precipitation_rate",
    "DSWRF:surface": "hrrr_solar_irradiance_ghi",
    "SNOD:surface": "hrrr_snow_depth",
}


class HrrrAdapter(BaseEnvironmentalSource):
    """
    Fetches NOAA HRRR NWP data at 3 km CONUS resolution.

    Each call fetches the analysis hour (fxx=0) for every UTC hour in
    [start_dt, end_dt), subsets to the bounding box, and returns a
    long-form DataFrame with one row per (lat, lon, timestamp).

    Parameters
    ----------
    product:
        HRRR product type. "sfc" (surface fields) covers all variables
        needed for grid asset stress analysis.
    fxx:
        Forecast hour. Use 0 for analysis (best accuracy for past dates),
        1–18 for near-real-time forecasting.
    save_dir:
        Local directory for GRIB2 file caching. Defaults to ~/data/hrrr.
    """

    def __init__(
        self,
        product: str = "sfc",
        fxx: int = 0,
        save_dir: str | None = None,
    ):
        if not _HERBIE_AVAILABLE:
            raise ImportError(
                "Herbie is required for HrrrAdapter. "
                "Install with: pip install climagrid[noaa-nwp]"
            )
        self._product = product
        self._fxx = fxx
        self._save_dir = save_dir

    @property
    def source_name(self) -> str:
        return "noaa_hrrr"

    def fetch(
        self,
        bbox: BoundingBox,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        start_dt = self._ensure_utc(start_dt)
        end_dt = self._ensure_utc(end_dt)
        self._validate_time_range(start_dt, end_dt)

        hours = self._hour_range(start_dt, end_dt)
        frames = []

        for dt in hours:
            try:
                df = self._fetch_one_hour(dt, bbox)
                frames.append(df)
            except Exception as exc:
                warnings.warn(f"HRRR fetch failed for {dt}: {exc}", RuntimeWarning, stacklevel=2)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _fetch_one_hour(self, dt: datetime, bbox: BoundingBox) -> pd.DataFrame:
        # Herbie requires a naive UTC datetime — strip tzinfo before passing.
        # Only pass save_dir when explicitly set; Path(None) raises TypeError.
        herbie_kwargs: dict = dict(
            model="hrrr",
            product=self._product,
            fxx=self._fxx,
            verbose=False,
        )
        if self._save_dir is not None:
            herbie_kwargs["save_dir"] = self._save_dir
        H = Herbie(dt.replace(tzinfo=None), **herbie_kwargs)

        # Fetch each variable separately and merge
        frames: list[pd.DataFrame] = []

        for search_str, col_name in _HRRR_FIELDS.items():
            try:
                ds = H.xarray(search_str, remove_grib=True)
                df_var = self._extract_bbox(ds, bbox, col_name, dt)
                frames.append(df_var)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        # Merge on lat/lon
        result = frames[0]
        for df_var in frames[1:]:
            merge_cols = [c for c in ["lat", "lon", "timestamp"] if c in df_var.columns]
            result = result.merge(df_var, on=merge_cols, how="outer")

        # Derive wind speed and direction from U/V components
        if "_hrrr_u_wind" in result.columns and "_hrrr_v_wind" in result.columns:
            u = result["_hrrr_u_wind"]
            v = result["_hrrr_v_wind"]
            result["hrrr_wind_speed_10m"] = np.sqrt(u**2 + v**2)
            result["hrrr_wind_direction_10m"] = (
                np.degrees(np.arctan2(-u, -v)) % 360
            )
            result = result.drop(columns=["_hrrr_u_wind", "_hrrr_v_wind"])

        # Convert temperature from K to °C
        if "hrrr_temperature_2m" in result.columns:
            result["hrrr_temperature_2m"] = result["hrrr_temperature_2m"] - 273.15

        return result

    @staticmethod
    def _extract_bbox(
        ds, bbox: BoundingBox, col_name: str, dt: datetime
    ) -> pd.DataFrame:
        """Subset an xarray Dataset to the bounding box and return a DataFrame."""
        # HRRR uses latitude/longitude coordinate names
        # Check for data variables before accessing coordinates
        data_vars = list(ds.data_vars)
        if not data_vars:
            return pd.DataFrame()

        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        # HRRR uses 0-360 longitude convention; normalize to -180 to 180
        import numpy as np
        lon = np.where(lon > 180, lon - 360, lon)

        # Spatial mask
        mask = (
            (lat >= bbox.min_lat) & (lat <= bbox.max_lat)
            & (lon >= bbox.min_lon) & (lon <= bbox.max_lon)
        )

        var = data_vars[0]
        values = ds[var].values

        flat_lat = lat[mask]
        flat_lon = lon[mask]
        flat_val = values[mask] if values.shape == lat.shape else values.flatten()[mask.flatten()]

        return pd.DataFrame(
            {
                "lat": flat_lat,
                "lon": flat_lon,
                col_name: flat_val,
                "timestamp": pd.Timestamp(dt).tz_convert("UTC") if dt.tzinfo else pd.Timestamp(dt, tz="UTC"),
            }
        )

    @staticmethod
    def _hour_range(start_dt: datetime, end_dt: datetime) -> list[datetime]:
        hours = []
        current = start_dt.replace(minute=0, second=0, microsecond=0)
        while current < end_dt:
            hours.append(current)
            current += timedelta(hours=1)
        return hours
