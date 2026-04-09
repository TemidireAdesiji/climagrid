"""Tests for NrcsAdapter using mocked HTTP responses."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import responses as resp_mock

from climagrid.sources.usda_nrcs import NrcsAdapter, _BASE_URL
from climagrid.sources.base import BoundingBox


_MOCK_STATIONS = [
    {
        "stationTriplet": "2057:TX:SCAN",
        "name": "Riesel",
        "latitude": "31.47",
        "longitude": "-96.93",
        "elevation": "191",
        "networkCd": "SCAN",
    }
]

_MOCK_STATION_DATA = [
    {
        "stationTriplet": "2057:TX:SCAN",
        "data": [
            {
                "stationElement": {"elementCode": "SMS", "heightDepth": -2},
                "values": [
                    {"date": "2024-07-15 00:00", "value": 28.5},
                    {"date": "2024-07-15 01:00", "value": 27.9},
                ],
            },
            {
                "stationElement": {"elementCode": "STO", "heightDepth": -2},
                "values": [
                    {"date": "2024-07-15 00:00", "value": 24.1},
                    {"date": "2024-07-15 01:00", "value": 24.3},
                ],
            },
        ],
    }
]


@resp_mock.activate
def test_fetch_returns_soil_columns():
    adapter = NrcsAdapter()
    bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
    start = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 2, tzinfo=timezone.utc)

    resp_mock.add(
        resp_mock.GET,
        f"{_BASE_URL}/stations",
        json=_MOCK_STATIONS,
        status=200,
    )
    resp_mock.add(
        resp_mock.GET,
        f"{_BASE_URL}/data",
        json=_MOCK_STATION_DATA,
        status=200,
    )

    df = adapter.fetch(bbox, start, end)

    assert not df.empty
    assert "nrcs_soil_moisture_pct" in df.columns
    assert "nrcs_soil_temperature" in df.columns
    assert df["nrcs_soil_moisture_pct"].iloc[0] == pytest.approx(28.5)


@resp_mock.activate
def test_no_stations_returns_empty_df():
    adapter = NrcsAdapter()
    bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
    start = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 2, tzinfo=timezone.utc)

    resp_mock.add(
        resp_mock.GET,
        f"{_BASE_URL}/stations",
        json=[],
        status=200,
    )

    df = adapter.fetch(bbox, start, end)
    # Should return minimal DataFrame without crashing
    assert isinstance(df, __import__("pandas").DataFrame)


def test_haversine_distance():
    from climagrid.sources.usda_nrcs import _haversine
    # Waco TX to Austin TX ≈ 183 km
    dist = _haversine(31.55, -97.15, 30.27, -97.74)
    assert 150 < dist < 220


# ---------------------------------------------------------------------------
# Integration test — hits the real USDA NRCS AWDB API (no credentials required)
# Run with: pytest -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_fetch_returns_soil_data():
    # Riesel, TX SCAN station (2199:TX:SCAN) is ~27 km from this bbox centroid
    bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
    adapter = NrcsAdapter()
    df = adapter.fetch(
        bbox,
        datetime(2024, 7, 15, 0, tzinfo=timezone.utc),
        datetime(2024, 7, 15, 6, tzinfo=timezone.utc),
    )

    assert not df.empty, "Live fetch returned no data"
    assert "nrcs_soil_moisture_pct" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert df["nrcs_soil_moisture_pct"].notna().any(), "All soil moisture values are NaN"
