"""Tests for the climagrid CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import responses as resp_mock
from click.testing import CliRunner

from climagrid.cli import main
from climagrid.sources.nasa_power import _BASE_URL as _NASA_URL


def _nasa_payload(lat: float = 31.55, lon: float = -97.15) -> dict:
    hours = ["2024071500", "2024071501", "2024071502"]
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat, 0]},
        "properties": {
            "parameter": {
                "T2M":              dict(zip(hours, [35.0, 36.0, 37.0])),
                "WS10M":            dict(zip(hours, [4.2, 5.1, 3.8])),
                "ALLSKY_SFC_SW_DWN": dict(zip(hours, [750.0, 820.0, 780.0])),
                "RH2M":             dict(zip(hours, [45.0, 42.0, 48.0])),
                "PRECTOTCORR":      dict(zip(hours, [0.0, 0.1, 0.0])),
            }
        },
    }


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# --help and --version
# ---------------------------------------------------------------------------

def test_main_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "schema" in result.output


def test_fetch_help(runner):
    result = runner.invoke(main, ["fetch", "--help"])
    assert result.exit_code == 0
    assert "--assets" in result.output
    assert "--start" in result.output
    assert "--end" in result.output


# ---------------------------------------------------------------------------
# schema command
# ---------------------------------------------------------------------------

def test_schema_prints_columns(runner):
    result = runner.invoke(main, ["schema"])
    assert result.exit_code == 0
    assert "asset_id" in result.output
    assert "feat_thermal_aging_factor" in result.output


def test_schema_writes_json_file(runner, tmp_path):
    import json
    out = tmp_path / "schema.json"
    result = runner.invoke(main, ["schema", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "columns" in data


# ---------------------------------------------------------------------------
# fetch command — happy path
# ---------------------------------------------------------------------------

@resp_mock.activate
def test_fetch_produces_output_parquet(runner, tmp_assets_csv, tmp_path):
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_payload(), status=200)
    out = tmp_path / "out.parquet"
    result = runner.invoke(main, [
        "fetch",
        "--assets", str(tmp_assets_csv),
        "--start", "2024-07-15",
        "--end", "2024-07-16",
        "--sources", "nasa_power",
        "--features", "thermal",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


@resp_mock.activate
def test_fetch_produces_csv_when_extension_is_csv(runner, tmp_assets_csv, tmp_path):
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_payload(), status=200)
    out = tmp_path / "out.csv"
    result = runner.invoke(main, [
        "fetch",
        "--assets", str(tmp_assets_csv),
        "--start", "2024-07-15",
        "--end", "2024-07-16",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()


@resp_mock.activate
def test_fetch_long_form_flag(runner, tmp_assets_csv, tmp_path):
    resp_mock.add(resp_mock.GET, _NASA_URL, json=_nasa_payload(), status=200)
    import pandas as pd
    out = tmp_path / "out.parquet"
    result = runner.invoke(main, [
        "fetch",
        "--assets", str(tmp_assets_csv),
        "--start", "2024-07-15",
        "--end", "2024-07-16",
        "--long-form",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    if out.exists():
        df = pd.read_parquet(out)
        assert "feature_name" in df.columns
        assert "feature_value" in df.columns


# ---------------------------------------------------------------------------
# fetch command — error paths
# ---------------------------------------------------------------------------

def test_fetch_bad_date_format_exits_nonzero(runner, tmp_assets_csv):
    result = runner.invoke(main, [
        "fetch",
        "--assets", str(tmp_assets_csv),
        "--start", "not-a-date",
        "--end", "2024-07-16",
    ])
    assert result.exit_code != 0


def test_fetch_nonexistent_assets_file_exits_nonzero(runner, tmp_path):
    result = runner.invoke(main, [
        "fetch",
        "--assets", str(tmp_path / "missing.csv"),
        "--start", "2024-07-15",
        "--end", "2024-07-16",
    ])
    assert result.exit_code != 0


@resp_mock.activate
def test_fetch_unknown_source_exits_nonzero(runner, tmp_assets_csv, tmp_path):
    result = runner.invoke(main, [
        "fetch",
        "--assets", str(tmp_assets_csv),
        "--start", "2024-07-15",
        "--end", "2024-07-16",
        "--sources", "not_a_real_source",
        "--output", str(tmp_path / "out.parquet"),
    ])
    assert result.exit_code != 0
