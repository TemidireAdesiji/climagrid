"""Tests for output exporters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from climagrid.outputs.exporters import to_csv, to_parquet, to_json_schema, to_dataframe, to_long_parquet
from climagrid.schema import INDEX_COLUMNS, FEATURE_COLUMNS


@pytest.fixture
def sample_output_df() -> pd.DataFrame:
    return pd.DataFrame({
        "asset_id": ["TX-001", "TX-001"],
        "timestamp": pd.to_datetime(["2024-07-15 00:00", "2024-07-15 01:00"], utc=True),
        "lat": [31.55, 31.55],
        "lon": [-97.15, -97.15],
        "feat_thermal_aging_factor": [0.45, 0.52],
        "feat_freeze_thaw_cycles": [0.0, 0.0],
    })


class TestToCsv:
    def test_writes_file(self, tmp_path, sample_output_df):
        path = tmp_path / "out.csv"
        result_path = to_csv(sample_output_df, path)
        assert result_path.exists()

    def test_readable_as_dataframe(self, tmp_path, sample_output_df):
        path = tmp_path / "out.csv"
        to_csv(sample_output_df, path)
        df = pd.read_csv(path)
        assert set(sample_output_df.columns).issubset(set(df.columns))

    def test_creates_parent_dirs(self, tmp_path, sample_output_df):
        path = tmp_path / "subdir" / "out.csv"
        to_csv(sample_output_df, path)
        assert path.exists()


class TestToParquet:
    def test_writes_file(self, tmp_path, sample_output_df):
        path = tmp_path / "out.parquet"
        result_path = to_parquet(sample_output_df, path)
        assert result_path.exists()

    def test_readable_as_dataframe(self, tmp_path, sample_output_df):
        path = tmp_path / "out.parquet"
        to_parquet(sample_output_df, path)
        df = pd.read_parquet(path)
        assert set(sample_output_df.columns).issubset(set(df.columns))
        assert len(df) == len(sample_output_df)


class TestToJsonSchema:
    def test_returns_dict(self):
        schema = to_json_schema()
        assert isinstance(schema, dict)
        assert "columns" in schema
        assert "climagrid_schema_version" in schema

    def test_all_columns_present(self):
        schema = to_json_schema()
        names = {c["name"] for c in schema["columns"]}
        for spec in INDEX_COLUMNS + FEATURE_COLUMNS:
            assert spec.name in names

    def test_writes_file_when_path_given(self, tmp_path):
        path = tmp_path / "schema.json"
        to_json_schema(path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert "columns" in loaded

    def test_column_has_required_fields(self):
        schema = to_json_schema()
        col = schema["columns"][0]
        for field in ("name", "dtype", "units", "source", "nullable", "description"):
            assert field in col


class TestToLongParquet:
    def test_writes_file(self, tmp_path, sample_output_df):
        path = tmp_path / "long.parquet"
        result_path = to_long_parquet(sample_output_df, path)
        assert result_path.exists()

    def test_has_feature_name_and_value_columns(self, tmp_path, sample_output_df):
        path = tmp_path / "long.parquet"
        to_long_parquet(sample_output_df, path)
        df = pd.read_parquet(path)
        assert "feature_name" in df.columns
        assert "feature_value" in df.columns

    def test_row_count_equals_rows_times_value_cols(self, tmp_path, sample_output_df):
        path = tmp_path / "long.parquet"
        to_long_parquet(sample_output_df, path)
        df = pd.read_parquet(path)
        id_cols = {"asset_id", "timestamp", "lat", "lon"}
        n_value_cols = len([c for c in sample_output_df.columns if c not in id_cols])
        assert len(df) == len(sample_output_df) * n_value_cols

    def test_id_columns_preserved(self, tmp_path, sample_output_df):
        path = tmp_path / "long.parquet"
        to_long_parquet(sample_output_df, path)
        df = pd.read_parquet(path)
        assert "asset_id" in df.columns
        assert "timestamp" in df.columns

    def test_creates_parent_dirs(self, tmp_path, sample_output_df):
        path = tmp_path / "nested" / "dir" / "long.parquet"
        to_long_parquet(sample_output_df, path)
        assert path.exists()


class TestToDataframe:
    def test_returns_same_dataframe(self, sample_output_df):
        result = to_dataframe(sample_output_df)
        assert len(result) == len(sample_output_df)
        assert list(result.columns) == list(sample_output_df.columns)

    def test_emits_warning_on_schema_violations(self):
        import warnings
        # DataFrame missing required index columns → validate_dataframe returns errors
        bad_df = pd.DataFrame({"some_random_col": [1.0, 2.0]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_dataframe(bad_df)
        assert any("Schema validation" in str(w.message) for w in caught)
