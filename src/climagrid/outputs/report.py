"""
Co-op-ready inspection report.

Ranks utility assets by environmental stress and renders a one-page PDF with a
priority list and a map, plus a machine-readable inspection-list DataFrame.

This is a maintenance-prioritization aid, not a failure prediction. It tells you
where the weather stress is highest and why; it does not assert which equipment
will fail. See docs/validation_notes.md for the boundary.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd

# feature column -> (plain-language hazard label, engineering standard)
_FEATURE_INFO: dict[str, tuple[str, str]] = {
    "feat_thermal_aging_factor": ("Transformer heat aging", "IEEE C57.91"),
    "feat_heat_hours_above_35c": ("Heat exposure hours", "IEEE C57.91"),
    "feat_conductor_sag_index": ("Conductor thermal sag", "IEEE 738 (simplified)"),
    "feat_ice_loading_risk": ("Ice loading", "ASCE 7-22 (simplified)"),
    "feat_freeze_thaw_cycles": ("Freeze-thaw fatigue", "n/a"),
    "feat_soil_saturation_index": ("Soil saturation", "n/a"),
    "feat_wildfire_proximity": ("Wildfire proximity", "n/a"),
}


def rank_assets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a climagrid feature DataFrame to a ranked, per-asset inspection list.

    Each asset is summarized by the peak (max) of every stress feature over the
    period. Features are percentile-ranked across assets so that differently
    scaled hazards are comparable; an asset's priority is its single worst hazard
    percentile, and the dominant hazard is the feature driving it.

    Returns one row per asset, sorted by priority (highest first), with columns:
    rank, asset_id, [asset_type], [lat, lon], dominant_hazard, dominant_standard,
    dominant_value, priority_score, plus the peak value of every feature.
    """
    if df.empty or "asset_id" not in df.columns:
        raise ValueError("rank_assets needs a non-empty DataFrame with an asset_id column")

    feat_cols = [c for c in df.columns if c.startswith("feat_") and c in _FEATURE_INFO]
    if not feat_cols:
        raise ValueError("no recognized feature columns (feat_*) found in the DataFrame")

    agg: dict[str, str] = {c: "max" for c in feat_cols}
    for c in ("lat", "lon", "asset_type"):
        if c in df.columns:
            agg[c] = "first"
    summary = df.groupby("asset_id", as_index=False).agg(agg)

    pct = summary[feat_cols].rank(pct=True)
    summary["priority_score"] = pct.max(axis=1).round(4)
    dominant = pct.idxmax(axis=1)
    summary["dominant_hazard"] = dominant.map(lambda c: _FEATURE_INFO[c][0])
    summary["dominant_standard"] = dominant.map(lambda c: _FEATURE_INFO[c][1])
    summary["dominant_value"] = [
        round(float(summary.loc[i, dominant.iloc[i]]), 4) for i in range(len(summary))
    ]

    summary = summary.sort_values("priority_score", ascending=False).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    front = ["rank", "asset_id"]
    for c in ("asset_type", "lat", "lon"):
        if c in summary.columns:
            front.append(c)
    front += ["dominant_hazard", "dominant_standard", "dominant_value", "priority_score"]
    rest = [c for c in summary.columns if c not in front]
    return summary[front + rest]  # type: ignore[no-any-return]


def _render_map_png(ranked: pd.DataFrame) -> bytes | None:
    """Render a stress-priority map to PNG bytes, or None if matplotlib is absent."""
    if not {"lat", "lon"}.issubset(ranked.columns):
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sc = ax.scatter(
        ranked["lon"], ranked["lat"], c=ranked["priority_score"], cmap="YlOrRd",
        s=90, edgecolor="black", linewidth=0.3, alpha=0.9,
    )
    plt.colorbar(sc, ax=ax, label="Stress priority (0 = low, 1 = highest in fleet)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Where the weather stress concentrates", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _build_html(
    ranked: pd.DataFrame, title: str, period: str | None, top_n: int, map_png: bytes | None
) -> str:
    n_assets = len(ranked)
    has_loc = {"lat", "lon"}.issubset(ranked.columns)
    has_type = "asset_type" in ranked.columns

    head = "<tr><th>#</th><th>Asset</th>"
    if has_type:
        head += "<th>Type</th>"
    if has_loc:
        head += "<th>Location</th>"
    head += "<th>Top hazard</th><th>Value</th></tr>"

    body = ""
    for _, r in ranked.head(top_n).iterrows():
        std = r["dominant_standard"]
        hazard = r["dominant_hazard"] + (f" <span class='std'>({std})</span>" if std != "n/a" else "")
        cells = f"<td class='n'>{int(r['rank'])}</td><td>{r['asset_id']}</td>"
        if has_type:
            cells += f"<td>{r.get('asset_type', '')}</td>"
        if has_loc:
            cells += f"<td class='loc'>{r['lat']:.3f}, {r['lon']:.3f}</td>"
        cells += f"<td>{hazard}</td><td class='n'>{r['dominant_value']:.3f}</td>"
        body += f"<tr>{cells}</tr>"

    map_block = ""
    if map_png is not None:
        b64 = base64.b64encode(map_png).decode("ascii")
        map_block = (
            "<h2>Map</h2>"
            f"<img class='map' src='data:image/png;base64,{b64}' alt='stress priority map' />"
        )

    period_line = f"Period: {period}<br/>" if period else ""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
      @page {{ size: A4; margin: 16mm 15mm; @bottom-center {{
        content: "climagrid inspection report  |  page " counter(page) " of " counter(pages);
        font-size: 8pt; color: #999; }} }}
      html {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10pt; color: #1a1a1a; }}
      h1 {{ font-size: 18pt; color: #0b3d2e; margin: 0 0 2pt 0; }}
      .meta {{ font-size: 9pt; color: #555; margin: 0 0 10pt 0; }}
      h2 {{ font-size: 12pt; color: #0b3d2e; border-bottom: 1.5px solid #1a6b3a;
            padding-bottom: 2pt; margin: 14pt 0 6pt 0; }}
      table {{ width: 100%; border-collapse: collapse; font-size: 9pt; }}
      th {{ background: #0b3d2e; color: #fff; text-align: left; padding: 4pt 6pt; }}
      td {{ padding: 3.5pt 6pt; border-bottom: 0.5px solid #ddd; }}
      tr:nth-child(even) td {{ background: #f4f7f5; }}
      td.n {{ text-align: right; }} td.loc {{ color: #555; font-size: 8.5pt; }}
      .std {{ color: #777; font-size: 8pt; }}
      img.map {{ width: 100%; max-width: 460px; display: block; margin: 4pt auto; }}
      .note {{ background: #fdf7ec; border-left: 3px solid #e0b449; padding: 7pt 9pt;
               font-size: 8.5pt; color: #5a4a1f; margin-top: 12pt; }}
    </style></head><body>
      <h1>{title}</h1>
      <p class="meta">{period_line}{n_assets} assets analyzed. Ranked by environmental
      stress, highest first. Showing the top {min(top_n, n_assets)}.</p>
      <h2>Inspection priority</h2>
      <table>{head}{body}</table>
      {map_block}
      <p class="note"><strong>How to read this:</strong> assets are ranked by the single
      weather hazard each is most exposed to, computed from public weather data using the
      cited IEEE and ASCE engineering standards. This is a prioritization aid for
      inspection and planning. It indicates where weather stress is highest, not which
      equipment will fail. Combine it with your own maintenance and failure records before
      acting.</p>
    </body></html>"""


def generate_report(
    df: pd.DataFrame,
    output: str | Path,
    *,
    title: str = "climagrid asset stress report",
    top_n: int = 20,
    period: str | None = None,
) -> Path:
    """
    Render a one-page PDF inspection report from a climagrid feature DataFrame.

    Requires the report extra: ``pip install "climagrid[report]"``.
    Returns the path to the written PDF.
    """
    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise ImportError(
            'PDF reports require the report extra: pip install "climagrid[report]"'
        ) from exc

    ranked = rank_assets(df)
    map_png = _render_map_png(ranked)
    html = _build_html(ranked, title, period, top_n, map_png)
    output = Path(output)
    HTML(string=html).write_pdf(str(output))
    return output
