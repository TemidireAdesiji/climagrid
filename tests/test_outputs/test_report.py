"""Tests for the inspection-report ranking (pure, no network or PDF)."""

from __future__ import annotations

import pandas as pd
import pytest

from climagrid.outputs import rank_assets
from climagrid.outputs.report import _build_html, _render_map_png, generate_report


def _sample_df() -> pd.DataFrame:
    rows = []
    # Asset A: high thermal aging. Asset B: high ice loading.
    spec = {"A": (0.020, 0.10), "B": (0.004, 0.85)}
    for aid, (thermal, ice) in spec.items():
        for _ in range(5):
            rows.append({
                "asset_id": aid, "lat": 31.0, "lon": -97.0, "asset_type": "transformer",
                "feat_thermal_aging_factor": thermal,
                "feat_ice_loading_risk": ice,
            })
    return pd.DataFrame(rows)


class TestRankAssets:
    def test_one_row_per_asset_with_rank(self):
        ranked = rank_assets(_sample_df())
        assert len(ranked) == 2
        assert ranked["rank"].tolist() == [1, 2]
        assert {"dominant_hazard", "dominant_standard", "dominant_value", "priority_score"} <= set(
            ranked.columns
        )

    def test_dominant_hazard_is_named_and_cited(self):
        ranked = rank_assets(_sample_df()).set_index("asset_id")
        assert ranked.loc["A", "dominant_hazard"] == "Transformer heat aging"
        assert ranked.loc["A", "dominant_standard"] == "IEEE C57.91"
        assert ranked.loc["B", "dominant_hazard"] == "Ice loading"
        assert ranked.loc["B", "dominant_standard"] == "ASCE 7-22 (simplified)"

    def test_priority_score_in_unit_interval(self):
        ranked = rank_assets(_sample_df())
        assert (ranked["priority_score"] >= 0).all()
        assert (ranked["priority_score"] <= 1).all()

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError):
            rank_assets(pd.DataFrame())

    def test_no_feature_columns_raises(self):
        with pytest.raises(ValueError, match="feature columns"):
            rank_assets(pd.DataFrame({"asset_id": ["A"], "lat": [1.0], "lon": [2.0]}))


class TestReportRendering:
    def test_build_html_contains_ranked_assets(self):
        """HTML builder (no third-party deps) renders the priority table."""
        ranked = rank_assets(_sample_df())
        html = _build_html(ranked, "Test Report", "2024-07", top_n=10, map_png=None)
        assert "Test Report" in html
        assert "Inspection priority" in html
        assert "A" in html and "B" in html
        assert "<table" in html

    def test_render_map_png_returns_png_bytes(self):
        pytest.importorskip("matplotlib")
        ranked = rank_assets(_sample_df())
        png = _render_map_png(ranked)
        assert png is not None
        assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number

    def test_render_map_png_none_without_coordinates(self):
        df = _sample_df().drop(columns=["lat", "lon"])
        assert _render_map_png(rank_assets(df)) is None

    def test_generate_report_writes_pdf(self, tmp_path):
        pytest.importorskip("weasyprint")
        out = tmp_path / "report.pdf"
        result = generate_report(_sample_df(), out, title="Smoke Test", top_n=5)
        assert result == out
        assert out.exists() and out.stat().st_size > 1000
