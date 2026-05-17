"""
AssetRegistry — loads utility asset records from CSV or GeoJSON.

Each asset must have at minimum: asset_id, lat, lon.
Optional fields: asset_type, voltage_kv, install_year, manufacturer.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

REQUIRED_COLUMNS = {"asset_id", "lat", "lon"}

ASSET_TYPE_VALUES = {
    "transformer",
    "circuit_breaker",
    "transmission_line",
    "distribution_line",
    "substation",
    "capacitor_bank",
    "recloser",
    "other",
}


class AssetRegistry:
    """
    Loads and validates a utility asset registry from CSV or GeoJSON.

    Parameters
    ----------
    path:
        Path to a CSV file (must have asset_id, lat, lon columns) or
        a GeoJSON file (must have asset_id and Point geometry).
    asset_type_filter:
        If provided, only include assets of these types.

    Example
    -------
    >>> registry = AssetRegistry("my_coop_assets.csv")
    >>> registry.assets.head()
    """

    def __init__(
        self,
        path: str | Path,
        asset_type_filter: list[str] | None = None,
    ):
        self._path = Path(path)
        self._gdf = self._load(self._path)

        if asset_type_filter:
            self._gdf = self._gdf[
                self._gdf["asset_type"].isin(asset_type_filter)
            ]

    @property
    def assets(self) -> gpd.GeoDataFrame:
        """GeoDataFrame with one row per asset, CRS=EPSG:4326."""
        return self._gdf

    @property
    def count(self) -> int:
        return len(self._gdf)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """(min_lat, max_lat, min_lon, max_lon) covering all assets."""
        bounds = self._gdf.total_bounds  # (minx, miny, maxx, maxy)
        return bounds[1], bounds[3], bounds[0], bounds[2]

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"AssetRegistry(n={self.count}, path={self._path.name!r})"

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> gpd.GeoDataFrame:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix in {".geojson", ".json"}:
            return self._load_geojson(path)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix!r}. "
                "Use .csv or .geojson."
            )

    def _load_csv(self, path: Path) -> gpd.GeoDataFrame:
        df = pd.read_csv(path, dtype={"asset_id": str})
        self._check_required_columns(df, path)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        n_before = len(df)
        df = df.dropna(subset=["lat", "lon"])
        if len(df) < n_before:
            import warnings
            warnings.warn(
                f"Dropped {n_before - len(df)} rows with null lat/lon",
                UserWarning,
                stacklevel=3,
            )
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326",
        )
        return gdf

    def _load_geojson(self, path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_file(path)
        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")

        # Extract lat/lon from geometry if not present
        if "lat" not in gdf.columns:
            gdf["lat"] = gdf.geometry.y
        if "lon" not in gdf.columns:
            gdf["lon"] = gdf.geometry.x

        df_check = pd.DataFrame(gdf.drop(columns="geometry"))
        self._check_required_columns(df_check, path)
        return gdf

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, path: Path) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Asset file {path.name!r} is missing required columns: "
                f"{sorted(missing)}. "
                f"Required: {sorted(REQUIRED_COLUMNS)}"
            )


def load_sample_assets() -> AssetRegistry:
    """Load the bundled 50-asset sample registry for testing and demos."""
    here = Path(__file__).parent.parent.parent.parent
    sample_path = here / "examples" / "data" / "sample_assets.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample asset file not found at {sample_path}. "
            "Has the repository been cloned fully?"
        )
    return AssetRegistry(sample_path)
