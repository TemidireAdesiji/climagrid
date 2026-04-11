"""Tests for NceiAdapter using mocked HTTP responses."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
import responses as resp_mock

from climagrid.sources.noaa_ncei import NceiAdapter, _BASE_URL
from climagrid.sources.base import BoundingBox


_BBOX = BoundingBox(min_lat=31.3, max_lat=31.9, min_lon=-97.4, max_lon=-96.9)
_START = datetime(2024, 7, 15, tzinfo=timezone.utc)
_END   = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)


def _mock_stations_payload() -> dict:
    return {
        "results": [
            {"id": "GHCND:USW00013958", "name": "WACO REGIONAL AP", "datacoverage": 0.95},
            {"id": "GHCND:USW00013957", "name": "TEMPLE AIRPORT",   "datacoverage": 0.80},
        ]
    }


def _mock_data_payload() -> dict:
    return {
        "results": [
            {"date": "2024-07-15T00:00:00", "datatype": "TMAX", "value": 36.7},
            {"date": "2024-07-15T00:00:00", "datatype": "TMIN", "value": 24.4},
            {"date": "2024-07-15T00:00:00", "datatype": "AWND", "value": 5.7},
            {"date": "2024-07-15T00:00:00", "datatype": "PRCP", "value": 0.0},
            {"date": "2024-07-15T00:00:00", "datatype": "RHAV", "value": 61.0},
        ]
    }


@resp_mock.activate
def test_fetch_returns_dataframe_with_temperature():
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/stations", json=_mock_stations_payload(), status=200)
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/data",     json=_mock_data_payload(),     status=200)

    adapter = NceiAdapter(api_token="test-token")
    df = adapter.fetch(_BBOX, _START, _END)

    assert not df.empty
    assert "ncei_temperature_max" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns


@resp_mock.activate
def test_fetch_returns_wind_and_precip_columns():
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/stations", json=_mock_stations_payload(), status=200)
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/data",     json=_mock_data_payload(),     status=200)

    adapter = NceiAdapter(api_token="test-token")
    df = adapter.fetch(_BBOX, _START, _END)

    assert "ncei_wind_speed" in df.columns
    assert "ncei_precipitation_daily" in df.columns


@resp_mock.activate
def test_fetch_no_stations_returns_minimal_df():
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/stations",
                  json={"results": []}, status=200)

    adapter = NceiAdapter(api_token="test-token")
    df = adapter.fetch(_BBOX, _START, _END)

    # Should return a minimal one-row DataFrame, not raise
    assert "lat" in df.columns
    assert "lon" in df.columns


@resp_mock.activate
def test_fetch_station_http_error_returns_minimal_df():
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/stations", status=500)

    adapter = NceiAdapter(api_token="test-token")
    df = adapter.fetch(_BBOX, _START, _END)

    assert "lat" in df.columns


@resp_mock.activate
def test_fetch_data_http_error_returns_minimal_df():
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/stations", json=_mock_stations_payload(), status=200)
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/data",     status=500)

    adapter = NceiAdapter(api_token="test-token")
    df = adapter.fetch(_BBOX, _START, _END)

    assert "lat" in df.columns


@resp_mock.activate
def test_fetch_empty_data_returns_minimal_df():
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/stations", json=_mock_stations_payload(), status=200)
    resp_mock.add(resp_mock.GET, f"{_BASE_URL}/data",     json={"results": []}, status=200)

    adapter = NceiAdapter(api_token="test-token")
    df = adapter.fetch(_BBOX, _START, _END)

    assert "lat" in df.columns


def test_source_name():
    adapter = NceiAdapter(api_token="test-token")
    assert adapter.source_name == "noaa_ncei"


def test_token_from_env(monkeypatch):
    monkeypatch.setenv("NOAA_CDO_TOKEN", "env-token-123")
    adapter = NceiAdapter()
    assert adapter._token == "env-token-123"


def test_invalid_time_range_raises():
    adapter = NceiAdapter(api_token="test-token")
    with pytest.raises(ValueError, match="start_dt"):
        adapter.fetch(
            _BBOX,
            datetime(2024, 7, 15, 6, tzinfo=timezone.utc),
            datetime(2024, 7, 15, 0, tzinfo=timezone.utc),
        )


# ---------------------------------------------------------------------------
# Integration test — hits the real NOAA CDO API
# Run with: pytest -m integration
# Requires: NOAA_CDO_TOKEN env var set
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_fetch_returns_data():
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        pytest.skip("NOAA_CDO_TOKEN not set")

    adapter = NceiAdapter(api_token=token)
    df = adapter.fetch(
        _BBOX,
        datetime(2024, 7, 15, 0, tzinfo=timezone.utc),
        datetime(2024, 7, 15, 6, tzinfo=timezone.utc),
    )

    assert not df.empty, "Live fetch returned no data"
    assert "ncei_temperature_max" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert df["ncei_temperature_max"].notna().any(), "All temperature values are NaN"
    # Waco TX July: expect daily max between 30–45°C
    assert df["ncei_temperature_max"].dropna().between(20, 50).all()
