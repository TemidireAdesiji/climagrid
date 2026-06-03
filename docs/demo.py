"""
Demo script — captures the output shown in the README animated gif.

Runs the real `climagrid fetch` CLI command against NASA POWER (no credentials
needed) and prints a richer feature summary below the CLI output so the gif
communicates what the output file actually contains.

Used by docs/make_cast.py — not intended to be run directly by users.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT     = Path(__file__).parent.parent
ASSETS   = "examples/data/tx_assets.csv"
START    = "2024-07-15"
END      = "2024-07-16"
OUTPUT   = "/tmp/stress_features.parquet"

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"


def main() -> None:
    # Run the real CLI — its output goes straight to stdout so it appears in the gif
    result = subprocess.run(
        [
            "climagrid", "fetch",
            "--assets",   ASSETS,
            "--start",    START,
            "--end",      END,
            "--sources",  "nasa_power",
            "--features", "all",
            "--output",   OUTPUT,
        ],
        cwd=str(ROOT),
        text=True,
    )

    if result.returncode != 0:
        return

    # Show what's inside the output file
    import pandas as pd
    df = pd.read_parquet(OUTPUT)

    print()
    print(f"{DIM}{'─' * 60}{RESET}")
    print(f"{DIM}  stress_features.parquet written — contents below:{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")
    print()
    print(f"{CYAN}Inside stress_features.parquet:{RESET}")
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns  "
          f"({df['asset_id'].nunique()} assets × {df['timestamp'].nunique()} hours)")
    print()

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"{CYAN}Stress features ({len(feat_cols)}):{RESET}")
    for fc in feat_cols:
        vals = df[fc].dropna()
        if len(vals):
            print(f"  {fc:<37}  min={vals.min():.3f}  max={vals.max():.3f}")
    print()

    import plotext as plt

    # Conductor sag varies by hour (solar + temperature drive it).
    asset0  = df["asset_id"].iloc[0]
    day_one = df["timestamp"].dt.normalize().iloc[0]
    hourly  = (
        df[(df["asset_id"] == asset0) & (df["timestamp"].dt.normalize() == day_one)]
        [["timestamp", "feat_conductor_sag_index"]]
        .sort_values("timestamp")
    )

    hours = hourly["timestamp"].dt.hour.tolist()
    sag   = hourly["feat_conductor_sag_index"].fillna(0).tolist()

    plt.theme("dark")
    plt.bar(hours, sag, width=0.7)
    plt.title(f"Conductor Sag Index — {asset0}  (July 15, UTC hours)")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Sag Index")
    plt.ylim(0, 1)
    plt.plotsize(80, 20)
    plt.show()

    print(f"{DIM}Join stress_features.parquet to your maintenance records.{RESET}")


if __name__ == "__main__":
    main()
