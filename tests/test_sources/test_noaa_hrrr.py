"""Tests for HrrrAdapter and its static helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climagrid.sources.base import BoundingBox

_BBOX  = BoundingBox(min_lat=31.3, max_lat=31.9, min_lon=-97.4, max_lon=-96.9)
_START = datetime(2024, 7, 15, 0,  tzinfo=timezone.utc)
_END   = datetime(2024, 7, 15, 3,  tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# ImportError path (herbie not installed in test env)
# ---------------------------------------------------------------------------

def test_hrrr_adapter_raises_import_error_without_herbie():
    """HrrrAdapter must raise ImportError clearly when herbie is not installed."""
    from climagrid.sources import noaa_hrrr
    if noaa_hrrr._HERBIE_AVAILABLE:
        pytest.skip("herbie is installed: skip ImportError test")

    from climagrid.sources.noaa_hrrr import HrrrAdapter
    with pytest.raises(ImportError, match="(?i)herbie"):
        HrrrAdapter()


# ---------------------------------------------------------------------------
# _hour_range static method: pure, no deps
# ---------------------------------------------------------------------------

def test_hour_range_returns_correct_count():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    start = datetime(2024, 7, 15, 0,  tzinfo=timezone.utc)
    end   = datetime(2024, 7, 15, 6,  tzinfo=timezone.utc)
    hours = HrrrAdapter._hour_range(start, end)
    assert len(hours) == 6


def test_hour_range_truncates_to_whole_hours():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    start = datetime(2024, 7, 15, 0, 30, tzinfo=timezone.utc)
    end   = datetime(2024, 7, 15, 3,  0, tzinfo=timezone.utc)
    hours = HrrrAdapter._hour_range(start, end)
    for h in hours:
        assert h.minute == 0
        assert h.second == 0


def test_hour_range_empty_when_start_equals_end():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
    assert HrrrAdapter._hour_range(dt, dt) == []


# ---------------------------------------------------------------------------
# _extract_bbox static method: uses real xarray, no herbie
# ---------------------------------------------------------------------------

def _make_test_dataset(var_name: str = "TMP", value: float = 300.0) -> xr.Dataset:
    """3×3 grid covering the test bbox."""
    lats = np.array([31.3, 31.6, 31.9])
    lons = np.array([-97.4, -97.15, -96.9])
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    values = np.full_like(lat2d, fill_value=value)
    return xr.Dataset(
        {var_name: (["y", "x"], values)},
        coords={
            "latitude":  (["y", "x"], lat2d),
            "longitude": (["y", "x"], lon2d),
        },
    )


def test_extract_bbox_returns_dataframe():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    ds = _make_test_dataset()
    dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
    df = HrrrAdapter._extract_bbox(ds, _BBOX, "hrrr_temperature_2m", dt)
    assert isinstance(df, pd.DataFrame)
    assert "hrrr_temperature_2m" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "timestamp" in df.columns


def test_extract_bbox_filters_to_bbox():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    ds = _make_test_dataset()
    dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
    df = HrrrAdapter._extract_bbox(ds, _BBOX, "hrrr_temperature_2m", dt)
    assert (df["lat"] >= _BBOX.min_lat).all()
    assert (df["lat"] <= _BBOX.max_lat).all()
    assert (df["lon"] >= _BBOX.min_lon).all()
    assert (df["lon"] <= _BBOX.max_lon).all()


def test_extract_bbox_preserves_values():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    ds = _make_test_dataset(value=300.0)
    dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
    df = HrrrAdapter._extract_bbox(ds, _BBOX, "hrrr_temperature_2m", dt)
    assert (df["hrrr_temperature_2m"] == 300.0).all()


def test_extract_bbox_returns_empty_on_no_data_vars():
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    ds = xr.Dataset()  # no data vars
    dt = datetime(2024, 7, 15, tzinfo=timezone.utc)
    df = HrrrAdapter._extract_bbox(ds, _BBOX, "hrrr_temperature_2m", dt)
    assert df.empty


# ---------------------------------------------------------------------------
# Full adapter with patched Herbie
# ---------------------------------------------------------------------------

def _make_mock_herbie_class(dataset: xr.Dataset):
    """Return a mock Herbie class whose instances yield the given dataset."""
    mock_instance = MagicMock()
    mock_instance.xarray.return_value = dataset
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


@patch("climagrid.sources.noaa_hrrr._HERBIE_AVAILABLE", True)
@patch("climagrid.sources.noaa_hrrr.Herbie", create=True)
def test_fetch_returns_dataframe_with_temperature(mock_herbie_cls):
    from climagrid.sources.noaa_hrrr import HrrrAdapter

    ds = _make_test_dataset("TMP", value=300.0)
    mock_herbie_cls.return_value.xarray.return_value = ds

    adapter = HrrrAdapter()
    df = adapter.fetch(_BBOX, _START, _END)

    assert not df.empty
    assert "hrrr_temperature_2m" in df.columns
    # Temperature converted from K to °C: 300.0 - 273.15 = 26.85
    assert (df["hrrr_temperature_2m"] < 50).all()


@patch("climagrid.sources.noaa_hrrr._HERBIE_AVAILABLE", True)
@patch("climagrid.sources.noaa_hrrr.Herbie", create=True)
def test_fetch_warns_and_continues_on_hour_failure(mock_herbie_cls):
    import warnings

    from climagrid.sources.noaa_hrrr import HrrrAdapter

    ds = _make_test_dataset("TMP", value=295.0)
    ok_instance = MagicMock()
    ok_instance.xarray.return_value = ds

    # Herbie() itself raises for the second hour: bubbles through _fetch_one_hour
    mock_herbie_cls.side_effect = [
        ok_instance,
        RuntimeError("GRIB download failed"),
        ok_instance,
    ]

    adapter = HrrrAdapter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = adapter.fetch(_BBOX, _START, _END)

    assert not df.empty
    assert any("HRRR fetch failed" in str(w.message) for w in caught)


@patch("climagrid.sources.noaa_hrrr._HERBIE_AVAILABLE", True)
@patch("climagrid.sources.noaa_hrrr.Herbie", create=True)
def test_fetch_returns_empty_when_all_hours_fail(mock_herbie_cls):
    from climagrid.sources.noaa_hrrr import HrrrAdapter

    mock_herbie_cls.return_value.xarray.side_effect = RuntimeError("all failed")

    adapter = HrrrAdapter()
    df = adapter.fetch(_BBOX, _START, _END)
    assert df.empty


@patch("climagrid.sources.noaa_hrrr._HERBIE_AVAILABLE", True)
@patch("climagrid.sources.noaa_hrrr.Herbie", create=True)
def test_source_name(mock_herbie_cls):
    from climagrid.sources.noaa_hrrr import HrrrAdapter
    assert HrrrAdapter().source_name == "noaa_hrrr"


# ---------------------------------------------------------------------------
# Integration test: downloads a real HRRR GRIB2 file via Herbie
# Run with: pytest -m integration
# Requires: pip install climagrid[noaa-nwp]
# Downloads ~5 MB from NOAA NOMADS or AWS NODD (cached after first run)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_live_fetch_returns_temperature():
    from climagrid.sources import noaa_hrrr
    if not noaa_hrrr._HERBIE_AVAILABLE:
        pytest.skip("herbie not installed: run: pip install climagrid[noaa-nwp]")

    from climagrid.sources.noaa_hrrr import HrrrAdapter

    bbox = BoundingBox(min_lat=31.3, max_lat=31.9, min_lon=-97.4, max_lon=-96.9)
    adapter = HrrrAdapter()
    df = adapter.fetch(
        bbox,
        datetime(2024, 7, 15, 0, tzinfo=timezone.utc),
        datetime(2024, 7, 15, 3, tzinfo=timezone.utc),
    )

    assert not df.empty, "Live HRRR fetch returned no data"
    assert "hrrr_temperature_2m" in df.columns, f"Expected hrrr_temperature_2m, got: {list(df.columns)}"
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "timestamp" in df.columns
    # Temperature should be in °C after conversion (Texas July: expect 20-45°C)
    temps = df["hrrr_temperature_2m"].dropna()
    assert not temps.empty, "All temperature values are NaN"
    assert temps.between(-10, 60).all(), f"Unexpected temperature range: {temps.min():.1f} to {temps.max():.1f}"
    # Coordinates should fall within the requested bbox
    assert df["lat"].between(31.3, 31.9).all()
    assert df["lon"].between(-97.4, -96.9).all()
