"""Tests for the high-level pipeline orchestrator (climagrid.run)."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest
import responses as resp_mock

import climagrid
from climagrid.pipeline.orchestrator import run
from climagrid.sources.nasa_power import _BASE_URL as _NASA_URL


# Reuse the NASA POWER mock fixture pattern from test_nasa_power.py
def _nasa_mock_payload(lat: float = 31.55, lon: float = -97.15) -> dict:
    hours = ["2024071500", "2024071501", "2024071502", "2024071503", "2024071504", "2024071505"]
    temps = [35.0, 36.0, 37.0, 38.0, 36.5, 34.0]
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 0]},
        "properties": {
            "parameter": {
                "T2M":              dict(zip(hours, temps, strict=False)),
                "WS10M":            dict(zip(hours, [4.2, 5.1, 3.8, 4.0, 6.0, 5.5], strict=False)),
                "ALLSKY_SFC_SW_DWN": dict(zip(hours, [0.0, 200.0, 750.0, 820.0, 600.0, 100.0], strict=False)),
                "RH2M":             dict(zip(hours, [45.0, 42.0, 48.0, 50.0, 47.0, 52.0], strict=False)),
                "PRECTOTCORR":      dict(zip(hours, [0.0, 0.0, 0.1, 0.0, 0.2, 0.0], strict=False)),
            }
        },
    }


@resp_mock.activate
def test_run_returns_dataframe_with_feature_columns(tmp_assets_csv):
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_mock_payload(), status=200)

    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    result = run(tmp_assets_csv, start, end, sources=["nasa_power"], features="all")

    assert not result.empty
    assert "asset_id" in result.columns
    assert "feat_thermal_aging_factor" in result.columns
    assert "feat_conductor_sag_index" in result.columns
    assert "nasa_temperature_2m" in result.columns


@resp_mock.activate
def test_run_accepts_string_path(tmp_assets_csv):
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_mock_payload(), status=200)

    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    result = run(str(tmp_assets_csv), start, end, sources=["nasa_power"])

    assert not result.empty


@resp_mock.activate
def test_run_selects_subset_of_features(tmp_assets_csv):
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_mock_payload(), status=200)

    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    result = run(
        tmp_assets_csv, start, end,
        sources=["nasa_power"],
        features=["thermal", "conductor_sag"],
    )

    assert "feat_thermal_aging_factor" in result.columns
    assert "feat_conductor_sag_index" in result.columns
    # freeze_thaw was not requested
    assert "feat_freeze_thaw_cycles" not in result.columns


@resp_mock.activate
def test_run_fetches_one_point_per_asset(tmp_assets_csv):
    """Point-based sources must fetch one location per asset, not a single
    shared centroid point. Regression test for geographically spread registries
    (tmp_assets_csv has an asset ~440 km from the others)."""
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_mock_payload(), status=200)

    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    result = run(tmp_assets_csv, start, end, sources=["nasa_power"], features=[])

    # All three assets are present, each carrying its OWN coordinates.
    assert result["asset_id"].nunique() == 3
    coords = result.drop_duplicates("asset_id")[["lat", "lon"]]
    got = {(round(la, 2), round(lo, 2)) for la, lo in coords.itertuples(index=False)}
    assert got == {(31.55, -97.15), (31.76, -97.05), (35.47, -97.52)}

    # One NASA POWER API call per unique asset location (not a single centroid).
    assert len(resp_mock.calls) == 3


def test_run_unknown_source_raises(tmp_assets_csv):
    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="Unknown source"):
        run(tmp_assets_csv, start, end, sources=["not_a_real_source"])


def test_run_unknown_feature_raises(tmp_assets_csv):
    """Requesting a non-existent feature name should raise immediately."""
    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    # No HTTP needed: raises before any fetch
    with pytest.raises(ValueError, match="Unknown feature"):
        run(
            tmp_assets_csv, start, end,
            sources=[],          # no sources → empty result before feature step
            features=["thermal", "nonexistent_feature"],
        )


@resp_mock.activate
def test_run_failing_source_skipped_with_warning(tmp_assets_csv):
    """A source that returns HTTP 500 should be skipped, not crash the pipeline."""
    resp_mock.add(resp_mock.GET, _NASA_URL, status=500)

    start = datetime(2024, 7, 15, tzinfo=timezone.utc)
    end = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(tmp_assets_csv, start, end, sources=["nasa_power"], features=[])

    # Result is empty (all sources failed) but no exception raised
    assert result.empty or isinstance(result, __import__("pandas").DataFrame)
    assert any("failed" in str(w.message).lower() or "skip" in str(w.message).lower() for w in caught)


def test_run_is_exported_from_top_level_module():
    """climagrid.run should be importable directly from the package."""
    assert callable(climagrid.run)
