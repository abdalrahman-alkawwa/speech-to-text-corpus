"""
tests/test_database.py
Test suite for Stage 10: Annotation Database & Audit Trail
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
from src.database import (
    init_db, upsert_segment, update_ground_truth,
    get_all_segments, get_audit_trail, get_corpus_statistics
)


def test_database():
    print("==============================================")
    print(" Running Stage 10: Database & Audit Tests     ")
    print("==============================================\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_corpus.db"

        # Test 1: Initialize
        print("[TEST 1] Initializing SQLite database schema...")
        init_db(test_db)
        assert test_db.exists()
        print("  -> PASSED\n")

        # Test 2: Upsert segment
        print("[TEST 2] Upserting segment records...")
        rec = {
            "segment_id": "test_video_seg_000001",
            "video_id": "test_video",
            "audio_path": "data/segments/test_video_seg_000001.wav",
            "start_sec": 0.0,
            "end_sec": 10.5,
            "duration_sec": 10.5,
            "snr_wada_db": 28.5,
            "snr_segmental_db": 22.1,
            "transcript_a": "هذا اختبار تجريبي",
            "transcript_b": "هذا اختبار تجريبي",
            "normalized_a": "هذا اختبار تجريبي",
            "normalized_b": "هذا اختبار تجريبي",
            "wer": 0.0,
            "cer": 0.0,
            "triage": "accepted",
        }
        upsert_segment(rec, db_path=test_db)

        rows = get_all_segments(db_path=test_db)
        assert len(rows) == 1
        assert rows[0]["segment_id"] == "test_video_seg_000001"
        assert rows[0]["status"] == "unreviewed"
        print("  -> PASSED\n")

        # Test 3: Update ground truth and verify audit trail
        print("[TEST 3] Updating ground truth and verifying audit logging...")
        update_ground_truth(
            segment_id="test_video_seg_000001",
            new_ground_truth="هذا اختبار تجريبي مدقق",
            status="verified",
            changed_by="tester",
            db_path=test_db
        )

        rows_updated = get_all_segments(db_path=test_db)
        assert rows_updated[0]["ground_truth"] == "هذا اختبار تجريبي مدقق"
        assert rows_updated[0]["status"] == "verified"

        audit = get_audit_trail("test_video_seg_000001", db_path=test_db)
        print(f"  -> Audit trail contains {len(audit)} entries")
        assert len(audit) >= 1
        print("  -> PASSED\n")

        # Test 4: Corpus statistics
        print("[TEST 4] Calculating corpus summary statistics...")
        stats = get_corpus_statistics(db_path=test_db)
        assert stats["total_segments"] == 1
        assert stats["total_duration_sec"] == 10.5
        print(f"  -> Stats: {stats}")
        print("  -> PASSED\n")

    print("==============================================")
    print(" ALL DATABASE & AUDIT TESTS PASSED!           ")
    print("==============================================")


if __name__ == "__main__":
    test_database()
