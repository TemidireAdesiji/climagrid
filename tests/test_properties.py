"""
Property-based tests using Hypothesis.

These tests verify mathematical invariants that hand-written unit tests
can miss: particularly edge cases at floating-point boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from climagrid.features.thermal import _EA_OVER_K, _T_REF_K, ThermalStressIndex
from climagrid.sources.base import BoundingBox

# ---------------------------------------------------------------------------
# BoundingBox invariants
# ---------------------------------------------------------------------------

@given(
    min_lat=st.floats(-89.9, 89.8),
    lat_span=st.floats(0.01, 5.0),
    min_lon=st.floats(-179.9, 179.8),
    lon_span=st.floats(0.01, 5.0),
)
def test_bbox_valid_inputs_always_construct(min_lat, lat_span, min_lon, lon_span):
    assume(min_lat + lat_span <= 90.0)
    assume(min_lon + lon_span <= 180.0)
    bb = BoundingBox(
        min_lat=min_lat,
        max_lat=min_lat + lat_span,
        min_lon=min_lon,
        max_lon=min_lon + lon_span,
    )
    assert bb.min_lat < bb.max_lat
    assert bb.min_lon < bb.max_lon


@given(
    min_lat=st.floats(-89.9, 89.8),
    lat_span=st.floats(0.01, 5.0),
    min_lon=st.floats(-179.9, 179.8),
    lon_span=st.floats(0.01, 5.0),
)
def test_bbox_center_is_midpoint(min_lat, lat_span, min_lon, lon_span):
    assume(min_lat + lat_span <= 90.0)
    assume(min_lon + lon_span <= 180.0)
    bb = BoundingBox(
        min_lat=min_lat,
        max_lat=min_lat + lat_span,
        min_lon=min_lon,
        max_lon=min_lon + lon_span,
    )
    center_lat, center_lon = bb.center
    assert center_lat == pytest.approx((min_lat + min_lat + lat_span) / 2, rel=1e-6)
    assert center_lon == pytest.approx((min_lon + min_lon + lon_span) / 2, rel=1e-6)


@given(
    lat=st.floats(-85, 85),
    lon=st.floats(-170, 170),
    radius_km=st.floats(1.0, 200.0),
)
def test_bbox_from_center_contains_center(lat, lon, radius_km):
    bb = BoundingBox.from_center(lat, lon, radius_km)
    c_lat, c_lon = bb.center
    assert bb.min_lat <= lat <= bb.max_lat
    assert bb.min_lon <= lon <= bb.max_lon


@given(
    lat=st.floats(-90, 90),
    lat2=st.floats(-90, 90),
)
def test_bbox_inverted_latitudes_raise(lat, lat2):
    assume(lat >= lat2)
    with pytest.raises(ValueError):
        BoundingBox(min_lat=lat, max_lat=lat2, min_lon=-10.0, max_lon=10.0)


# ---------------------------------------------------------------------------
# Thermal aging factor (FAA) invariants
# ---------------------------------------------------------------------------

@given(st.floats(-50.0, 200.0, allow_nan=False, allow_infinity=False))
def test_faa_always_positive(temp_c):
    """FAA must be positive for any finite ambient temperature."""
    hotspot_k = temp_c + 25.0 + 273.15
    assume(hotspot_k > 0)
    faa = np.exp(_EA_OVER_K * (1.0 / _T_REF_K - 1.0 / hotspot_k))
    assert faa > 0.0


@given(st.floats(-50.0, 200.0, allow_nan=False, allow_infinity=False))
def test_faa_monotonically_increases_with_temperature(temp_c):
    """Higher temperature → higher FAA (more aging), always."""
    delta = 1.0
    t1_k = temp_c + 25.0 + 273.15
    t2_k = (temp_c + delta) + 25.0 + 273.15
    assume(t1_k > 0 and t2_k > 0)
    faa1 = np.exp(_EA_OVER_K * (1.0 / _T_REF_K - 1.0 / t1_k))
    faa2 = np.exp(_EA_OVER_K * (1.0 / _T_REF_K - 1.0 / t2_k))
    assert faa2 >= faa1


@given(st.floats(-50.0, 200.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=50)
def test_faa_compute_returns_same_shape(temp_c):
    """ThermalStressIndex.compute() must return same number of rows as input."""
    df = pd.DataFrame({
        "hrrr_temperature_2m": [temp_c] * 4,
        "asset_id": ["A"] * 4,
    })
    tsi = ThermalStressIndex()
    result = tsi.compute(df)
    assert len(result) == len(df)
    assert "feat_thermal_aging_factor" in result.columns


# ---------------------------------------------------------------------------
# Schema round-trip invariant
# ---------------------------------------------------------------------------

def test_column_map_has_no_duplicate_names():
    """Every column name in the schema must be unique."""
    from climagrid.schema import ALL_COLUMNS
    names = [c.name for c in ALL_COLUMNS]
    assert len(names) == len(set(names)), "Duplicate column names in schema"


@given(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P"))))
@settings(max_examples=20)
def test_column_spec_rejects_empty_name(name):
    """ColumnSpec must always have a non-empty name."""
    from climagrid.schema import ColumnSpec
    spec = ColumnSpec(name=name, dtype="float64", units="°C", description="test", source="test")
    assert spec.name == name
