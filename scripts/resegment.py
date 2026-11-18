"""
scripts/resegment.py - re-run ONLY Stage 5 (segmentation) from the saved regions JSON.

Reuses data/processed/<video_id>_regions.json, so the slow VAD step is skipped.
Deletes and rewrites that video's clips (data/segments/<video_id>_seg_*.wav) and
metadata/<video_id>_segments.json.  Do this BEFORE SNR/transcription: segment IDs
change, so existing *_snr.json / *_transcripts.json would no longer match.

    python scripts/resegment.py CvrNaPFIwcQ          # one video
    python scripts/resegment.py                      # every video that has a regions JSON
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmenter import export_segments

PROCESSED = PROJECT_ROOT / "data" / "processed"


def main(argv) -> int:
    ids = argv[1:] or sorted(p.name[: -len("_regions.json")] for p in PROCESSED.glob("*_regions.json"))
    if not ids:
        print(f"No *_regions.json files in {PROCESSED}")
        return 1
    for video_id in ids:
        regions_path = PROCESSED / f"{video_id}_regions.json"
        if not regions_path.exists():
            matches = sorted(PROCESSED.glob(f"*{video_id}*_regions.json"))
            regions_path = matches[0] if matches else regions_path
        wavs = [w for w in sorted(PROCESSED.glob(f"*{video_id}*.wav"))]
        if not regions_path.exists() or not wavs:
            print(f"[{video_id}] skipped: need a regions JSON and a processed WAV")
            continue
        with open(regions_path, "r", encoding="utf-8") as fh:
            regions = json.load(fh)
        export_segments(wavs[0], regions, video_id)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
