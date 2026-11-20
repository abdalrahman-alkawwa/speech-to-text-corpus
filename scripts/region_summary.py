"""
scripts/region_summary.py - rebuild the Stage 4 summary table from saved regions JSON.

    python scripts/region_summary.py            # every data/processed/*_regions.json
    python scripts/region_summary.py CvrNaPFIwcQ
"""
import json
import sys
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
LABELS = ["speech", "silence", "music", "mixed"]

files = sorted(PROCESSED.glob("*_regions.json"))
if len(sys.argv) > 1:
    files = [f for f in files if sys.argv[1] in f.name]
for f in files:
    regions = json.loads(f.read_text(encoding="utf-8"))
    total = max(r["end_sec"] for r in regions) if regions else 0.0
    print(f"\n{f.name}  ({len(regions)} regions, {total:.1f}s)")
    print(f"  {'label':<8} {'count':>5} {'seconds':>9} {'percent':>8}")
    for label in LABELS:
        rs = [r for r in regions if r["label"] == label]
        dur = sum(r["end_sec"] - r["start_sec"] for r in rs)
        print(f"  {label:<8} {len(rs):>5} {dur:>9.1f} {100 * dur / total if total else 0:>7.1f}%")
