"""Tests for WfigsAdapter and proximity calculation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import responses as resp_mock

from climagrid.sources.base import BoundingBox
from climagrid.sources.usfs_wfigs import (
    _PERIMETERS_URL,
    WfigsAdapter,
    _haversine,
    compute_proximity,
)

_MOCK_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "OBJECTID": 1,
                "poly_GISAcres": 1234.5,
                "attr_FireDiscoveryDateTime": "2024-07-14T10:00:00Z",
                "attr_ContainmentDateTime": None,  # active fire
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-97.20, 31.60],
                    [-97.10, 31.60],
                    [-97.10, 31.70],
                    [-97.20, 31.70],
                    [-97.20, 31.60],
                ]],
            },
        }
    ],
}


@resp_mock.activate
def test_fetch_returns_fire_dataframe():
    adapter = WfigsAdapter()
    bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    resp_mock.add(
        resp_mock.GET,
        _PERIMETERS_URL,
        json=_MOCK_GEOJSON,
        status=200,
    )

    df = adapter.fetch(bbox, start, end)

    assert not df.empty
    assert "fire_centroid_lat" in df.columns
    assert "fire_area_ha" in df.columns
    assert df["fire_active"].iloc[0] == True  # noqa: E712 (numpy bool comparison)
    # 1234.5 acres * 0.404686 = ~499.9 ha
    assert df["fire_area_ha"].iloc[0] == pytest.approx(1234.5 * 0.404686, abs=1.0)


@resp_mock.activate
def test_fetch_empty_returns_empty_df():
    adapter = WfigsAdapter()
    bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)

    resp_mock.add(
        resp_mock.GET,
        _PERIMETERS_URL,
        json={"type": "FeatureCollection", "features": []},
        status=200,
    )

    df = adapter.fetch(
        bbox,
        datetime(2024, 7, 15, tzinfo=timezone.utc),
        datetime(2024, 7, 15, 6, tzinfo=timezone.utc),
    )
    assert df.empty


def test_compute_proximity_nearby(nearby_fires_df):
    # Asset at 31.55, -97.15; fire centroid at 31.65, -97.10 ≈ 12.7 km
    dist_km, active, area_ha = compute_proximity(31.55, -97.15, nearby_fires_df)
    assert 5 < dist_km < 25
    assert active is True
    assert area_ha > 0


def test_compute_proximity_empty(empty_fires_df):
    dist_km, active, area_ha = compute_proximity(31.55, -97.15, empty_fires_df)
    assert dist_km == float("inf")
    assert active is False
    assert area_ha == 0.0


def test_haversine_known_distance():
    # NYC (40.71, -74.01) to Boston (42.36, -71.06) ≈ 306 km
    dist = _haversine(40.71, -74.01, 42.36, -71.06)
    assert 280 < dist < 330


# ---------------------------------------------------------------------------
# Integration test — hits the real USFS NIFC WFIGS ArcGIS API (no credentials required)
# Run with: pytest -m integration
# Uses a California bbox during a historically active fire period
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_fetch_returns_fire_data():
    # Northern California bbox — high fire activity in summer 2024
    bbox = BoundingBox(38.0, 40.5, -122.5, -120.0)
    adapter = WfigsAdapter()
    df = adapter.fetch(
        bbox,
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 31, tzinfo=timezone.utc),
    )

    # The API should return a valid (possibly empty) DataFrame without error
    import pandas as pd
    assert isinstance(df, pd.DataFrame), "fetch() did not return a DataFrame"
    if not df.empty:
        assert "fire_centroid_lat" in df.columns
        assert "fire_area_ha" in df.columns
        assert "fire_active" in df.columns
        assert (df["fire_area_ha"] >= 0).all(), "Negative fire area found"
