"""Tests for the climagrid schema module."""

import pandas as pd
import pytest

from climagrid.schema import (
    ALL_COLUMNS,
    COLUMN_MAP,
    INDEX_COLUMNS,
    FEATURE_COLUMNS,
    validate_dataframe,
    empty_dataframe,
    schema_summary,
)


class TestSchema:
    def test_all_columns_have_unique_names(self):
        names = [c.name for c in ALL_COLUMNS]
        assert len(names) == len(set(names)), "Duplicate column names in schema"

    def test_column_map_matches_all_columns(self):
        assert set(COLUMN_MAP.keys()) == {c.name for c in ALL_COLUMNS}

    def test_index_columns_are_non_nullable(self):
        for spec in INDEX_COLUMNS:
            assert spec.nullable is False, f"{spec.name} should be non-nullable"

    def test_feature_columns_have_feature_source(self):
        for spec in FEATURE_COLUMNS:
            assert spec.source == "feature", f"{spec.name} source should be 'feature'"

    def test_validate_valid_dataframe(self):
        df = pd.DataFrame({
            "asset_id": ["TX-001"],
            "timestamp": pd.to_datetime(["2024-07-15"], utc=True),
            "lat": [31.55],
            "lon": [-97.15],
        })
        errors = validate_dataframe(df)
        assert errors == []

    def test_validate_missing_required_column(self):
        df = pd.DataFrame({
            "asset_id": ["TX-001"],
            "lat": [31.55],
        })  # missing timestamp and lon
        errors = validate_dataframe(df)
        assert any("timestamp" in e for e in errors)

    def test_validate_null_in_non_nullable(self):
        df = pd.DataFrame({
            "asset_id": [None],
            "timestamp": pd.to_datetime(["2024-07-15"], utc=True),
            "lat": [31.55],
            "lon": [-97.15],
        })
        errors = validate_dataframe(df)
        assert any("asset_id" in e for e in errors)

    def test_empty_dataframe_has_index_columns(self):
        df = empty_dataframe()
        for spec in INDEX_COLUMNS:
            assert spec.name in df.columns

    def test_schema_summary_is_dataframe(self):
        df = schema_summary()
        assert isinstance(df, pd.DataFrame)
        assert "column" in df.columns
        assert "units" in df.columns
        assert len(df) == len(ALL_COLUMNS)
