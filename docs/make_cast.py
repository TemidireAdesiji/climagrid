"""
Generates docs/assets/demo.cast (asciinema v2 format) from the output of
docs/demo.py.

Running the demo in a subprocess and writing its output as a cast file avoids
the TTY requirement of `asciinema rec`, so this works cleanly in CI.

After running this script, convert to gif with:
    agg docs/assets/demo.cast docs/assets/demo.gif --font-size 14 --speed 1.5

Or run everything via the GitHub Actions workflow (.github/workflows/media.yml).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAST_FILE = ROOT / "docs" / "assets" / "demo.cast"

# Terminal dimensions for the recording
WIDTH  = 100
HEIGHT = 32

# Simulated typing delay before the command appears (seconds)
PROMPT_PAUSE  = 0.8
TYPING_DELAY  = 0.05   # per character when "typing" the command
LINE_DELAY    = 0.06   # between output lines
FINAL_PAUSE   = 2.0    # hold at end so reader can see the result


def _run_demo() -> str:
    """Execute docs/demo.py and return its stdout."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "docs" / "demo.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("demo.py failed:", result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def _build_events(output: str) -> list[list]:
    events: list[list] = []
    t = 0.0

    # Fake prompt + command typing — shows the actual CLI command users run
    cmd = ("climagrid fetch --assets examples/data/tx_assets.csv"
           " --start 2024-07-15 --end 2024-07-16"
           " --sources nasa_power --features all --output stress_features.parquet")
    t += PROMPT_PAUSE
    events.append([t, "o", "$ "])
    for ch in cmd:
        t += TYPING_DELAY
        events.append([t, "o", ch])
    t += 0.3
    events.append([t, "o", "\r\n"])

    # Output lines
    for line in output.splitlines():
        t += LINE_DELAY
        events.append([t, "o", line + "\r\n"])

    # Hold at the end
    t += FINAL_PAUSE
    events.append([t, "o", ""])

    return events


def main() -> None:
    print("Running demo script...", flush=True)
    output = _run_demo()
    print(f"Captured {len(output.splitlines())} lines of output.", flush=True)

    events = _build_events(output)
    duration = events[-1][0]

    header = {
        "version":   2,
        "width":     WIDTH,
        "height":    HEIGHT,
        "duration":  round(duration, 3),
        "title":     "climagrid demo",
        "env":       {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
    }

    CAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CAST_FILE.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for event in events:
            fh.write(json.dumps(event) + "\n")

    print(f"Cast file written: {CAST_FILE}  ({duration:.1f}s)")
    print()
    print("Convert to gif with:")
    print(f"  agg {CAST_FILE} docs/assets/demo.gif --font-size 14 --speed 1.5")


if __name__ == "__main__":
    main()
