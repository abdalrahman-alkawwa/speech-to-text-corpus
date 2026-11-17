"""
tests/test_transcript_store.py
Test suite for JSON-based transcript storage and audit trail
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import json
import src.transcript_store as ts


def test_transcript_store():
    print("==================================================")
    print(" Running Stage 10: JSON Transcript Store Tests    ")
    print("==================================================\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        orig_dir = ts.TRANSCRIPTS_DIR
        ts.TRANSCRIPTS_DIR = tmp_path

        try:
            # Test 1: Save transcripts
            print("[TEST 1] Saving segment transcripts to JSON...")
            test_segments = [
                {
                    "segment_id": "testvid_seg_000001",
                    "video_id": "testvid",
                    "audio_path": "data/segments/testvid_seg_000001.wav",
                    "duration_sec": 10.5,
                    "transcript_a": "هذا اختبار تجريبي",
                    "transcript_b": "هذا اختبار تجريبي",
                    "wer": 0.0,
                    "triage": "accepted",
                }
            ]
            json_file = ts.save_transcripts("testvid", test_segments)
            assert json_file.exists()
            print(f"  -> Created JSON: {json_file.name} (PASSED)\n")

            # Test 2: Load transcripts
            print("[TEST 2] Loading transcripts from JSON...")
            loaded = ts.load_transcripts("testvid")
            assert len(loaded) == 1
            assert loaded[0]["transcript_a"] == "هذا اختبار تجريبي"
            assert loaded[0]["status"] == "unreviewed"
            print("  -> PASSED\n")

            # Test 3: Update ground truth and verify embedded audit trail
            print("[TEST 3] Updating ground truth and checking embedded audit trail...")
            ts.update_ground_truth(
                segment_id="testvid_seg_000001",
                new_ground_truth="هذا اختبار تجريبي مدقق بالكامل",
                status="verified",
                changed_by="annotator_alice"
            )

            updated = ts.load_transcripts("testvid")
            assert updated[0]["ground_truth"] == "هذا اختبار تجريبي مدقق بالكامل"
            assert updated[0]["status"] == "verified"

            trail = ts.get_audit_trail("testvid_seg_000001")
            assert len(trail) >= 1
            assert trail[0]["changed_by"] == "annotator_alice"
            print(f"  -> Audit trail verified with {len(trail)} record(s) (PASSED)\n")

            # Test 4: Compute corpus statistics from JSON
            print("[TEST 4] Computing corpus statistics directly from JSON...")
            stats = ts.get_corpus_statistics()
            assert stats["total_segments"] == 1
            assert stats["total_duration_sec"] == 10.5
            assert stats["status_breakdown"]["verified"] == 1
            print(f"  -> Stats: {stats} (PASSED)\n")

        finally:
            ts.TRANSCRIPTS_DIR = orig_dir

    print("==================================================")
    print(" ALL JSON TRANSCRIPT STORE TESTS PASSED!          ")
    print("==================================================")


if __name__ == "__main__":
    test_transcript_store()
