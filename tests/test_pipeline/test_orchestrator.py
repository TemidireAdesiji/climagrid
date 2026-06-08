"""Tests for the high-level pipeline orchestrator (climagrid.run)."""

from __future__ import annotations

import sys
import types
import warnings
from datetime import datetime, timezone

import pandas as pd
import pytest
import responses as resp_mock

import climagrid
from climagrid.pipeline.orchestrator import _FEATURE_MAP, _SOURCE_MAP, run
from climagrid.sources.base import BaseEnvironmentalSource
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


# ---------------------------------------------------------------------------
# Dispatch coverage: non-point (bbox) sources and the multi-source merge.
# Fake adapters are injected via _SOURCE_MAP so run()'s dispatch is exercised
# without any real network calls.
# ---------------------------------------------------------------------------

# tmp_assets_csv carries assets at exactly these three coordinates.
_ASSET_COORDS = [(31.55, -97.15), (31.76, -97.05), (35.47, -97.52)]
_START = datetime(2024, 7, 15, tzinfo=timezone.utc)
_END = datetime(2024, 7, 15, 6, tzinfo=timezone.utc)


class _FakeGridAdapter(BaseEnvironmentalSource):
    """Non-point_based source: run() must dispatch to fetch(bbox)."""

    point_based = False

    @property
    def source_name(self) -> str:
        return "fakegrid"

    def fetch(self, bbox, start_dt, end_dt):
        return pd.DataFrame({
            "lat": [la for la, _ in _ASSET_COORDS],
            "lon": [lo for _, lo in _ASSET_COORDS],
            "timestamp": pd.to_datetime([start_dt] * 3, utc=True),
            "fakegrid_value": [1.0, 2.0, 3.0],
        })


class _FakePointAdapter(BaseEnvironmentalSource):
    """point_based source: run() must dispatch to fetch_points()."""

    point_based = True

    @property
    def source_name(self) -> str:
        return "fakepoint"

    def fetch(self, bbox, start_dt, end_dt):  # pragma: no cover - never dispatched
        raise AssertionError("point_based source must use fetch_points")

    def fetch_points(self, points, start_dt, end_dt):
        return pd.concat(
            [
                pd.DataFrame({
                    "lat": [lat],
                    "lon": [lon],
                    "timestamp": pd.to_datetime([start_dt], utc=True),
                    "fakepoint_value": [9.0],
                })
                for lat, lon in points
            ],
            ignore_index=True,
        )


class _FakeEmptyAdapter(BaseEnvironmentalSource):
    """Non-point source returning no data, to exercise the skip path."""

    point_based = False

    @property
    def source_name(self) -> str:
        return "fakeempty"

    def fetch(self, bbox, start_dt, end_dt):
        return pd.DataFrame()


def _register_fakes(monkeypatch):
    mod = types.ModuleType("climagrid_fake_sources")
    mod._FakeGridAdapter = _FakeGridAdapter
    mod._FakePointAdapter = _FakePointAdapter
    mod._FakeEmptyAdapter = _FakeEmptyAdapter
    monkeypatch.setitem(sys.modules, "climagrid_fake_sources", mod)
    for name, cls in (
        ("fakegrid", "_FakeGridAdapter"),
        ("fakepoint", "_FakePointAdapter"),
        ("fakeempty", "_FakeEmptyAdapter"),
    ):
        monkeypatch.setitem(_SOURCE_MAP, name, ("climagrid_fake_sources", cls))


class TestOrchestratorDispatch:
    def test_non_point_source_uses_bbox_fetch(self, monkeypatch, tmp_assets_csv):
        # fakegrid is point_based=False, so its column can only appear via run()'s
        # `else: adapter.fetch(bbox)` branch.
        _register_fakes(monkeypatch)
        result = run(tmp_assets_csv, _START, _END, sources=["fakegrid"], features=[])
        assert "fakegrid_value" in result.columns
        assert result["asset_id"].nunique() == 3

    def test_merges_multiple_sources(self, monkeypatch, tmp_assets_csv):
        # Two sources both return data, so run()'s multi-source merge loop runs
        # and the result carries columns from both.
        _register_fakes(monkeypatch)
        result = run(
            tmp_assets_csv, _START, _END,
            sources=["fakegrid", "fakepoint"], features=[],
        )
        assert "fakegrid_value" in result.columns
        assert "fakepoint_value" in result.columns
        assert result["asset_id"].nunique() == 3

    def test_empty_source_is_skipped(self, monkeypatch, tmp_assets_csv):
        _register_fakes(monkeypatch)
        result = run(tmp_assets_csv, _START, _END, sources=["fakeempty"], features=[])
        assert result.empty

    @resp_mock.activate
    def test_defaults_to_nasa_power_when_sources_omitted(self, tmp_assets_csv):
        # No sources= argument: run() falls back to ["nasa_power"] (line 101).
        resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_mock_payload(), status=200)
        result = run(tmp_assets_csv, _START, _END, features=[])
        assert not result.empty
        assert "nasa_temperature_2m" in result.columns

    def test_wfigs_source_is_passed_through(self, monkeypatch, tmp_assets_csv):
        # The usfs_wfigs branch stashes raw fire data separately (line 174).
        _register_fakes(monkeypatch)
        monkeypatch.setitem(
            _SOURCE_MAP, "usfs_wfigs", ("climagrid_fake_sources", "_FakeGridAdapter")
        )
        result = run(tmp_assets_csv, _START, _END, sources=["usfs_wfigs"], features=[])
        assert "fakegrid_value" in result.columns

    def test_failing_feature_is_skipped_with_warning(self, monkeypatch, tmp_assets_csv):
        # A feature whose compute() raises is warned and skipped (lines 201-202),
        # without crashing the pipeline.
        _register_fakes(monkeypatch)

        class _FailingFeature:
            def compute(self, df):
                raise RuntimeError("boom")

        fmod = types.ModuleType("climagrid_fake_feature")
        fmod._FailingFeature = _FailingFeature
        monkeypatch.setitem(sys.modules, "climagrid_fake_feature", fmod)
        monkeypatch.setitem(
            _FEATURE_MAP, "fakefail", ("climagrid_fake_feature", "_FailingFeature")
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run(
                tmp_assets_csv, _START, _END,
                sources=["fakegrid"], features=["fakefail"],
            )

        assert any("fakefail" in str(w.message) for w in caught)
        assert "fakegrid_value" in result.columns  # pipeline still returns data
