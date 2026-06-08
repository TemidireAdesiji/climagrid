"""Tests for BoundingBox and BaseEnvironmentalSource."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from climagrid.sources.base import BaseEnvironmentalSource, BoundingBox

_START = datetime(2024, 7, 1, tzinfo=timezone.utc)
_END = datetime(2024, 7, 2, tzinfo=timezone.utc)


class _DummySource(BaseEnvironmentalSource):
    """Concrete source whose fetch() returns one row tagged with the bbox center."""

    def __init__(self) -> None:
        self.fetched_boxes: list[BoundingBox] = []

    @property
    def source_name(self) -> str:
        return "dummy"

    def fetch(self, bbox: BoundingBox, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        self.fetched_boxes.append(bbox)
        lat, lon = bbox.center
        return pd.DataFrame({"lat": [lat], "lon": [lon], "dummy_value": [1.0]})


class TestBoundingBox:
    def test_valid_creation(self):
        bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
        assert bbox.min_lat == 31.3
        assert bbox.max_lat == 31.9
        assert bbox.min_lon == -97.4
        assert bbox.max_lon == -96.9

    def test_center(self):
        bbox = BoundingBox(30.0, 32.0, -98.0, -96.0)
        lat, lon = bbox.center
        assert lat == pytest.approx(31.0)
        assert lon == pytest.approx(-97.0)

    def test_from_center(self):
        bbox = BoundingBox.from_center(31.55, -97.15, 50.0)
        assert bbox.min_lat < 31.55 < bbox.max_lat
        assert bbox.min_lon < -97.15 < bbox.max_lon

    def test_invalid_lat_order(self):
        with pytest.raises(ValueError, match="min_lat"):
            BoundingBox(32.0, 31.0, -97.4, -96.9)

    def test_invalid_lon_order(self):
        with pytest.raises(ValueError, match="min_lon"):
            BoundingBox(31.3, 31.9, -96.9, -97.4)

    def test_lat_out_of_range(self):
        with pytest.raises(ValueError, match="Latitudes"):
            BoundingBox(-95.0, 31.9, -97.4, -96.9)

    def test_lon_out_of_range(self):
        with pytest.raises(ValueError, match="Longitudes"):
            BoundingBox(31.3, 31.9, -190.0, -96.9)

    def test_frozen(self):
        bbox = BoundingBox(31.3, 31.9, -97.4, -96.9)
        with pytest.raises(ValidationError):
            bbox.min_lat = 30.0  # type: ignore


class TestPerAssetFetching:
    """Per-asset fetching: one weather location per asset, not one for the set."""

    def test_fetch_points_not_implemented_by_default(self):
        # Grid/station sources that do not override fetch_points must raise,
        # so the orchestrator only calls it on genuine point_based adapters.
        with pytest.raises(NotImplementedError, match="per-point"):
            _DummySource().fetch_points([(31.0, -97.0)], _START, _END)

    def test_fetch_points_via_bbox_issues_one_query_per_point(self):
        src = _DummySource()
        points = [(31.0, -97.0), (40.0, -105.0), (47.6, -122.3)]
        out = src._fetch_points_via_bbox(points, _START, _END, radius_km=25.0)
        # one small-bbox fetch per asset, results concatenated
        assert len(src.fetched_boxes) == 3
        assert len(out) == 3
        # each returned row is centered on its requested point (not the centroid)
        assert out["lat"].tolist() == pytest.approx([31.0, 40.0, 47.6], abs=0.3)
        assert out["lon"].tolist() == pytest.approx([-97.0, -105.0, -122.3], abs=0.3)

    def test_fetch_points_via_bbox_skips_empty_and_returns_empty(self):
        class _EmptySource(_DummySource):
            def fetch(self, bbox, start_dt, end_dt):
                return pd.DataFrame()

        out = _EmptySource()._fetch_points_via_bbox([(1.0, 1.0)], _START, _END, 10.0)
        assert out.empty

    def test_ensure_utc_localizes_naive_datetime(self):
        localized = _DummySource()._ensure_utc(datetime(2024, 7, 1, 12, 0))
        assert localized.tzinfo == timezone.utc

    def test_ensure_utc_converts_aware_datetime(self):
        aware = datetime(2024, 7, 1, 12, 0, tzinfo=timezone(timedelta(hours=5)))
        out = _DummySource()._ensure_utc(aware)
        assert out.tzinfo == timezone.utc
        assert out.hour == 7  # 12:00 +05:00 -> 07:00 UTC
