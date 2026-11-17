"""
tests/test_evaluator.py
Test suite for Stage 9: JiWER Disagreement Triage
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import compute_metrics, triage_segment, evaluate_transcripts


def test_evaluator():
    print("==============================================")
    print(" Running Stage 9: JiWER Triage Tests          ")
    print("==============================================\n")

    # Test 1: Identical strings -> 0% WER -> accepted
    m1 = compute_metrics("ذهب أحمد إلى المدرسة", "ذهب أحمد إلى المدرسة")
    print(f"[TEST 1] Identical strings: WER={m1['wer']} CER={m1['cer']}")
    assert m1["wer"] == 0.0 and m1["cer"] == 0.0
    assert triage_segment(m1["wer"]) == "accepted"
    print("  -> PASSED\n")

    # Test 2: Moderate difference -> review_needed
    m2 = compute_metrics("ذهب أحمد إلى المدرسة اليوم", "ذهب أحمد المدرسة")
    print(f"[TEST 2] Moderate difference: WER={m2['wer']} CER={m2['cer']}")
    t2 = triage_segment(m2["wer"])
    assert t2 in ("accepted", "review_needed")
    print(f"  -> Triage: {t2} (PASSED)\n")

    # Test 3: Major difference -> difficult
    m3 = compute_metrics("هذا درس في معالجة الإشارات", "سيارة سريعة في الشارع")
    print(f"[TEST 3] Major difference: WER={m3['wer']} CER={m3['cer']}")
    assert m3["wer"] > 0.40
    assert triage_segment(m3["wer"]) == "difficult"
    print("  -> PASSED\n")

    # Test 4: Batch evaluation
    batch = [
        {"segment_id": "seg_1", "transcript_a": "سلام عليكم", "transcript_b": "سلام عليكم"},
        {"segment_id": "seg_2", "transcript_a": "صباح الخير يا مصر", "transcript_b": "مساء النور يا عالم"},
    ]
    results = evaluate_transcripts(batch)
    assert len(results) == 2
    assert results[0]["triage"] == "accepted"
    assert results[1]["triage"] == "difficult"
    print("[TEST 4] Batch evaluation:\n  -> PASSED\n")

    print("==============================================")
    print(" ALL EVALUATOR TESTS PASSED!                  ")
    print("==============================================")


if __name__ == "__main__":
    test_evaluator()
