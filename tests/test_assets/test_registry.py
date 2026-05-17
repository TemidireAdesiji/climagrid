"""Tests for AssetRegistry."""

from __future__ import annotations

import pytest

from climagrid.assets.registry import AssetRegistry


class TestAssetRegistry:
    def test_load_csv(self, tmp_assets_csv):
        registry = AssetRegistry(tmp_assets_csv)
        assert registry.count == 3
        assert set(registry.assets["asset_id"]) == {"A-001", "A-002", "A-003"}

    def test_load_sample_assets(self, sample_assets_path):
        registry = AssetRegistry(sample_assets_path)
        assert registry.count >= 40  # sample file has 45 assets across 8 states
        assert "asset_type" in registry.assets.columns

    def test_geometry_is_point(self, tmp_assets_csv):
        registry = AssetRegistry(tmp_assets_csv)
        from shapely.geometry import Point
        assert all(isinstance(g, Point) for g in registry.assets.geometry)

    def test_crs_is_wgs84(self, tmp_assets_csv):
        registry = AssetRegistry(tmp_assets_csv)
        assert registry.assets.crs.to_epsg() == 4326

    def test_bounding_box(self, tmp_assets_csv):
        registry = AssetRegistry(tmp_assets_csv)
        min_lat, max_lat, min_lon, max_lon = registry.bounding_box
        assert min_lat < max_lat
        assert min_lon < max_lon

    def test_missing_required_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("asset_id,lat\nA-001,31.55\n")  # missing lon
        with pytest.raises(ValueError, match="lon"):
            AssetRegistry(bad_csv)

    def test_null_lat_lon_dropped(self, tmp_path):
        csv = tmp_path / "nulls.csv"
        csv.write_text(
            "asset_id,lat,lon\n"
            "A-001,31.55,-97.15\n"
            "A-002,,\n"  # null lat/lon
            "A-003,35.47,-97.52\n"
        )
        import warnings
        with warnings.catch_warnings(record=True):
            registry = AssetRegistry(csv)
        assert registry.count == 2  # null row dropped

    def test_asset_type_filter(self, sample_assets_path):
        registry = AssetRegistry(sample_assets_path, asset_type_filter=["transformer"])
        assert all(registry.assets["asset_type"] == "transformer")

    def test_unsupported_file_type_raises(self, tmp_path):
        bad = tmp_path / "assets.xlsx"
        bad.write_bytes(b"fake xlsx")
        with pytest.raises(ValueError, match="Unsupported"):
            AssetRegistry(bad)

    def test_repr(self, tmp_assets_csv):
        r = repr(AssetRegistry(tmp_assets_csv))
        assert "AssetRegistry" in r
        assert "n=3" in r

    def test_load_geojson(self, tmp_path):
        import json
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"asset_id": "TX-001", "asset_type": "transformer"},
                    "geometry": {"type": "Point", "coordinates": [-97.15, 31.55]},
                },
                {
                    "type": "Feature",
                    "properties": {"asset_id": "TX-002", "asset_type": "distribution_line"},
                    "geometry": {"type": "Point", "coordinates": [-97.05, 31.76]},
                },
            ],
        }
        gj_path = tmp_path / "assets.geojson"
        gj_path.write_text(json.dumps(geojson))
        registry = AssetRegistry(gj_path)
        assert registry.count == 2
        assert set(registry.assets["asset_id"]) == {"TX-001", "TX-002"}
        assert "lat" in registry.assets.columns
        assert "lon" in registry.assets.columns


def test_load_sample_assets_function():
    from climagrid.assets.registry import load_sample_assets
    registry = load_sample_assets()
    assert registry.count >= 40
