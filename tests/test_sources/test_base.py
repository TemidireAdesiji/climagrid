"""Tests for BoundingBox and BaseEnvironmentalSource."""

import pytest
from pydantic import ValidationError

from climagrid.sources.base import BoundingBox


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
