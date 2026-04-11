"""Tests for NasaPowerAdapter using mocked HTTP responses."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import responses as resp_mock

from climagrid.sources.nasa_power import NasaPowerAdapter, _BASE_URL
from climagrid.sources.base import BoundingBox


def _make_nasa_response(lat: float, lon: float) -> dict:
    """Minimal valid NASA POWER JSON response structure."""
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 0]},
        "properties": {
            "parameter": {
                "T2M": {
                    "2024071500": 35.42,
                    "2024071501": 36.10,
                    "2024071502": 37.05,
                },
                "WS10M": {
                    "2024071500": 4.2,
                    "2024071501": 5.1,
                    "2024071502": 3.8,
                },
                "ALLSKY_SFC_SW_DWN": {
                    "2024071500": 750.0,
                    "2024071501": 820.0,
                    "2024071502": 780.0,
                },
                "RH2M": {
                    "2024071500": 45.0,
                    "2024071501": 42.0,
                    "2024071502": 48.0,
                },
                "PRECTOTCORR": {
                    "2024071500": 0.0,
                    "2024071501": 0.1,
                    "2024071502": -999.0,  # fill value
                },
            }
        },
    }


@resp_mock.activate
def test_fetch_point_returns_dataframe():
    adapter = NasaPowerAdapter()
    lat, lon = 31.55, -97.15
    start = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 2, tzinfo=timezone.utc)

    resp_mock.add(
        resp_mock.GET,
        _BASE_URL,
        json=_make_nasa_response(lat, lon),
        status=200,
    )

    df = adapter.fetch_point(lat, lon, start, end)

    assert not df.empty
    assert "nasa_temperature_2m" in df.columns
    assert "nasa_wind_speed_10m" in df.columns
    assert "nasa_solar_irradiance_ghi" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert df["lat"].iloc[0] == lat
    assert df["lon"].iloc[0] == lon


@resp_mock.activate
def test_fill_value_replaced_with_nan():
    adapter = NasaPowerAdapter()
    lat, lon = 31.55, -97.15
    start = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 2, tzinfo=timezone.utc)

    resp_mock.add(
        resp_mock.GET,
        _BASE_URL,
        json=_make_nasa_response(lat, lon),
        status=200,
    )

    df = adapter.fetch_point(lat, lon, start, end)

    # The -999 fill value in PRECTOTCORR should become NaN
    precip = df["nasa_precipitation"]
    assert precip.isna().any(), "Expected NaN where NASA POWER returned -999"


@resp_mock.activate
def test_fetch_uses_bbox_center():
    adapter = NasaPowerAdapter()
    bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
    start = datetime(2024, 7, 15, 0, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 2, tzinfo=timezone.utc)

    resp_mock.add(
        resp_mock.GET,
        _BASE_URL,
        json=_make_nasa_response(*bbox.center),
        status=200,
    )

    df = adapter.fetch(bbox, start, end)
    assert not df.empty


def test_invalid_time_range_raises():
    adapter = NasaPowerAdapter()
    with pytest.raises(ValueError, match="start_dt"):
        adapter.fetch_point(
            31.55, -97.15,
            datetime(2024, 7, 15, 6, tzinfo=timezone.utc),
            datetime(2024, 7, 15, 0, tzinfo=timezone.utc),  # end before start
        )


@resp_mock.activate
def test_http_error_propagates():
    adapter = NasaPowerAdapter()
    resp_mock.add(resp_mock.GET, _BASE_URL, status=500)
    with pytest.raises(Exception):
        adapter.fetch_point(
            31.55, -97.15,
            datetime(2024, 7, 15, 0, tzinfo=timezone.utc),
            datetime(2024, 7, 15, 2, tzinfo=timezone.utc),
        )


# ---------------------------------------------------------------------------
# Integration test — hits the real NASA POWER API (no credentials required)
# Run with: pytest -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_fetch_returns_data():
    adapter = NasaPowerAdapter()
    df = adapter.fetch_point(
        31.55, -97.15,
        datetime(2024, 7, 15, 0, tzinfo=timezone.utc),
        datetime(2024, 7, 15, 6, tzinfo=timezone.utc),
    )

    assert not df.empty, "Live fetch returned no data"
    assert "nasa_temperature_2m" in df.columns
    assert "nasa_wind_speed_10m" in df.columns
    assert "nasa_solar_irradiance_ghi" in df.columns
    assert df["nasa_temperature_2m"].notna().any(), "All temperature values are NaN"
    assert (df["lat"] - 31.55).abs().lt(0.01).all()
    assert (df["lon"] - (-97.15)).abs().lt(0.01).all()
