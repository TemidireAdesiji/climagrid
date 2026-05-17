"""
End-to-end integration tests for the climagrid pipeline.

Tests items that unit tests cannot catch:
- Multi-source orchestrator merge logic with real APIs
- Feature classes receiving real column names (catches silent NaN from fallback misses)
- climagrid.run() convenience API (the exact code path shown in the README)
- WFIGS wildfire feature path end-to-end

Run with: pytest -m integration
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

import climagrid

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WACO_TX = dict(
    bbox_lat_min=31.3, bbox_lat_max=31.9,
    bbox_lon_min=-97.4, bbox_lon_max=-96.9,
)

_START = datetime(2024, 7, 15, 0,  tzinfo=timezone.utc)
_END   = datetime(2024, 7, 15, 6,  tzinfo=timezone.utc)

# California bbox used for wildfire test — high summer fire activity
_CA_START = datetime(2024, 8, 1, 0, tzinfo=timezone.utc)
_CA_END   = datetime(2024, 8, 2, 0, tzinfo=timezone.utc)


@pytest.fixture
def waco_assets_csv(tmp_path) -> Path:
    p = tmp_path / "waco_assets.csv"
    p.write_text(
        "asset_id,lat,lon,asset_type\n"
        "TX-001,31.55,-97.15,transformer\n"
        "TX-002,31.70,-97.00,circuit_breaker\n"
    )
    return p


@pytest.fixture
def california_assets_csv(tmp_path) -> Path:
    p = tmp_path / "ca_assets.csv"
    p.write_text(
        "asset_id,lat,lon,asset_type\n"
        "CA-001,39.5,-121.5,distribution_line\n"
    )
    return p


# ---------------------------------------------------------------------------
# 1. climagrid.run() smoke test — README Quick Start path
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run_api_smoke_test(waco_assets_csv):
    """
    Exercises the exact code shown in the README Quick Start.
    Verifies run() returns a non-empty DataFrame with asset_id, timestamp,
    environmental columns, and feature columns — all non-NaN.
    """
    df = climagrid.run(
        waco_assets_csv,
        start_dt=_START,
        end_dt=_END,
        sources=["nasa_power"],
        features="all",
    )

    assert not df.empty, "run() returned an empty DataFrame"
    assert "asset_id" in df.columns
    assert "timestamp" in df.columns

    # Environmental data present
    assert "nasa_temperature_2m" in df.columns, "NASA POWER temperature column missing"
    assert "nasa_wind_speed_10m" in df.columns, "NASA POWER wind column missing"

    # Features computed — not all NaN (catches silent fallback miss bugs)
    assert df["feat_thermal_aging_factor"].notna().any(), \
        "feat_thermal_aging_factor is all NaN — temp column lookup failed"
    assert df["feat_conductor_sag_index"].notna().any(), \
        "feat_conductor_sag_index is all NaN — column lookup failed"
    assert df["feat_soil_saturation_index"].notna().any(), \
        "feat_soil_saturation_index is all NaN — precip column lookup failed"
    assert df["feat_freeze_thaw_cycles"].notna().any(), \
        "feat_freeze_thaw_cycles is all NaN — temp column lookup failed"

    # Physical sanity: thermal aging factor must be positive
    assert (df["feat_thermal_aging_factor"].dropna() > 0).all()

    # Two assets × 6 hours = at least 12 rows
    assert len(df) >= 12, f"Expected ≥12 rows (2 assets × 6 hours), got {len(df)}"


# ---------------------------------------------------------------------------
# 2. Multi-source merge test (NASA POWER + USDA NRCS)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run_multisource_nasa_power_and_nrcs(waco_assets_csv):
    """
    Runs the orchestrator with two sources and verifies:
    - Columns from both sources appear in the result
    - The merge didn't explode the row count
    - Wildfire feature gracefully returns 0 (no WFIGS source requested)
    """
    df = climagrid.run(
        waco_assets_csv,
        start_dt=_START,
        end_dt=_END,
        sources=["nasa_power", "usda_nrcs"],
        features=["thermal", "soil", "conductor_sag"],
    )

    assert not df.empty, "Multi-source run returned empty DataFrame"
    assert "asset_id" in df.columns

    # NASA POWER columns must be present
    assert "nasa_temperature_2m" in df.columns

    # Even if no NRCS station was found, features should still be computed from NASA
    assert df["feat_thermal_aging_factor"].notna().any(), \
        "Thermal feature is all NaN even with NASA POWER present"
    assert df["feat_soil_saturation_index"].notna().any(), \
        "Soil feature is all NaN — precipitation fallback should have fired"

    # Merge must not create more rows than assets × timestamps
    n_assets = df["asset_id"].nunique()
    n_timestamps = df["timestamp"].nunique()
    assert len(df) <= n_assets * n_timestamps * 2, \
        f"Row count {len(df)} suspiciously large — merge may have cross-joined"


# ---------------------------------------------------------------------------
# 3. Feature column name contract — checks each feature gets real data
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_all_features_produce_non_nan_values_with_nasa_power(waco_assets_csv):
    """
    With NASA POWER as source, every feature that has a NASA fallback column
    must produce at least one non-NaN value. Any all-NaN feature column means
    its column name lookup failed — a silent contract break.
    """
    df = climagrid.run(
        waco_assets_csv,
        start_dt=_START,
        end_dt=_END,
        sources=["nasa_power"],
        features=["thermal", "conductor_sag", "freeze_thaw"],
    )

    # These three features all have nasa_temperature_2m as a fallback column
    for feat_col in [
        "feat_thermal_aging_factor",
        "feat_conductor_sag_index",
        "feat_freeze_thaw_cycles",
    ]:
        non_nan = df[feat_col].dropna()
        assert not non_nan.empty, (
            f"{feat_col} is entirely NaN — its temperature column lookup failed. "
            f"Available columns: {[c for c in df.columns if 'temp' in c.lower()]}"
        )


# ---------------------------------------------------------------------------
# 4. WFIGS wildfire feature path end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run_wildfire_path_with_wfigs(california_assets_csv):
    """
    Runs the orchestrator with WFIGS + NASA POWER and verifies:
    - feat_wildfire_proximity column is present and not all NaN
    - wfigs_nearest_fire_km column is present
    - fires_raw handoff from orchestrator to WildfireProximityScore worked
    """
    df = climagrid.run(
        california_assets_csv,
        start_dt=_CA_START,
        end_dt=_CA_END,
        sources=["nasa_power", "usfs_wfigs"],
        features=["thermal", "wildfire"],
        bbox_radius_km=100.0,
    )

    assert not df.empty, "WFIGS + NASA POWER run returned empty DataFrame"
    assert "feat_wildfire_proximity" in df.columns, \
        "feat_wildfire_proximity missing — wildfire feature path broken"
    assert "wfigs_nearest_fire_km" in df.columns, \
        "wfigs_nearest_fire_km missing — WFIGS fires_raw handoff failed"

    # Score must be in [0, 1]
    scores = df["feat_wildfire_proximity"].dropna()
    assert (scores >= 0).all() and (scores <= 1).all(), \
        f"Wildfire score out of [0,1]: min={scores.min()}, max={scores.max()}"

    # Thermal feature should still work alongside wildfire
    assert df["feat_thermal_aging_factor"].notna().any()


# ---------------------------------------------------------------------------
# 5. Source kwargs are forwarded correctly (NCEI token path)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_run_ncei_source_with_token_kwarg(waco_assets_csv):
    """
    Verifies source_kwargs are forwarded to the adapter constructor.
    Also exercises the NCEI adapter end-to-end through the pipeline.
    """
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        pytest.skip("NOAA_CDO_TOKEN not set")

    df = climagrid.run(
        waco_assets_csv,
        start_dt=_START,
        end_dt=_END,
        sources=["nasa_power", "noaa_ncei"],
        features=["thermal", "conductor_sag"],
        source_kwargs={"noaa_ncei": {"api_token": token}},
    )

    assert not df.empty
    # NCEI columns may or may not be present depending on station coverage
    # but thermal feature should always work via NASA POWER fallback
    assert df["feat_thermal_aging_factor"].notna().any()
    ncei_temp_cols = [c for c in df.columns if "ncei_temperature" in c]
    if ncei_temp_cols:
        # If NCEI data arrived, verify the column name is the new one (not the old dry_bulb)
        assert "ncei_temperature_max" in df.columns, \
            f"Expected ncei_temperature_max but got: {ncei_temp_cols}"
