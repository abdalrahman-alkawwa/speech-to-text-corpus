"""
tests/test_export_corpus.py
Test suite for Stage 11: Dataset Manifest Export
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import json
import src.transcript_store as transcript_store
from src.export_corpus import export_corpus_manifests


def test_export_manifests():
    print("==============================================")
    print(" Running Stage 11: Export Manifests Tests     ")
    print("==============================================\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_transcripts = tmp_path / "transcripts"
        test_out = tmp_path / "metadata"
        original_transcripts_dir = transcript_store.TRANSCRIPTS_DIR
        transcript_store.TRANSCRIPTS_DIR = test_transcripts

        try:
            # Populate the JSON transcript store with 10 segments.
            print("[TEST 1] Populating JSON transcript store with 10 segments...")
            transcript_store.save_transcripts("vid_test", [
                {
                    "segment_id": f"vid_test_seg_{i:06d}",
                    "video_id": "vid_test",
                    "audio_path": f"data/segments/vid_test_seg_{i:06d}.wav",
                    "start_sec": (i - 1) * 10.0,
                    "end_sec": i * 10.0,
                    "duration_sec": 10.0,
                    "snr_wada_db": 25.0 + i,
                    "boundary_check": "auto_verified",
                    "transcript_a": f"نص تجريبي للقطعة رقم {i}",
                    "transcript_b": f"نص تجريبي للقطعة رقم {i}",
                    "ground_truth": f"نص تجريبي للقطعة رقم {i}",
                    "wer": 0.05 if i % 2 == 0 else 0.25,
                    "triage": "accepted" if i % 2 == 0 else "review_needed",
                    "status": "verified" if i <= 5 else "unreviewed",
                }
                for i in range(1, 11)
            ])
            print("  -> Populated 10 segments\n")

            print("[TEST 2] Running export_corpus_manifests()...")
            res = export_corpus_manifests(output_dir=test_out)

            assert Path(res["jsonl_path"]).exists()
            assert Path(res["csv_path"]).exists()
            assert Path(res["gold_path"]).exists()

            # Verify line count
            with open(res["jsonl_path"], "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 10
            assert "split" in lines[0]
            print(f"  -> Generated JSONL with {len(lines)} records")

            print("  -> PASSED\n")

        finally:
            transcript_store.TRANSCRIPTS_DIR = original_transcripts_dir

    print("==============================================")
    print(" ALL EXPORT MANIFESTS TESTS PASSED!           ")
    print("==============================================")


if __name__ == "__main__":
    test_export_manifests()
