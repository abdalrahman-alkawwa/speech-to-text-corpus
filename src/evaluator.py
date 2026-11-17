"""
src/evaluator.py
================
Stage 9: JiWER Disagreement Triage

Computes WER and CER between two ASR engine transcripts (after normalization)
and assigns each segment to a triage bucket:
  - accepted:       WER < 15% — both engines agree, minimal review needed
  - review_needed:  15% <= WER <= 40% — moderate disagreement, human check
  - difficult:      WER > 40% — major disagreement, priority manual listening
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any
import jiwer
from src.normalizer import normalize_arabic


# Triage thresholds (WER)
ACCEPTED_THRESHOLD = 0.15
REVIEW_THRESHOLD = 0.40


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Computes WER and CER between normalized reference and hypothesis.
    If both strings are empty, returns 0.0 for both.
    """
    ref_norm = normalize_arabic(reference)
    hyp_norm = normalize_arabic(hypothesis)

    if not ref_norm and not hyp_norm:
        return {"wer": 0.0, "cer": 0.0}

    if not ref_norm or not hyp_norm:
        return {"wer": 1.0, "cer": 1.0}

    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)

    return {
        "wer": round(float(wer), 4),
        "cer": round(float(cer), 4),
    }


def triage_segment(wer: float) -> str:
    """Assigns triage bucket based on WER value."""
    if wer < ACCEPTED_THRESHOLD:
        return "accepted"
    elif wer <= REVIEW_THRESHOLD:
        return "review_needed"
    else:
        return "difficult"


def evaluate_transcripts(
    transcript_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluates a list of dual-transcript records. Each record must have
    'transcript_a' and 'transcript_b' fields.

    Returns enriched records with normalized texts, WER, CER, and triage status.
    """
    results = []
    triage_counts = {"accepted": 0, "review_needed": 0, "difficult": 0}

    for rec in transcript_records:
        text_a = rec.get("transcript_a", "")
        text_b = rec.get("transcript_b", "")

        norm_a = normalize_arabic(text_a)
        norm_b = normalize_arabic(text_b)

        # WER/CER: treat Engine A (Whisper) as reference
        metrics = compute_metrics(text_a, text_b)
        bucket = triage_segment(metrics["wer"])
        triage_counts[bucket] += 1

        results.append({
            **rec,
            "normalized_a": norm_a,
            "normalized_b": norm_b,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "triage": bucket,
        })

    # Print summary
    total = len(results)
    print(f"\n[evaluator] Triage Summary ({total} segments):")
    print(f"  • Accepted (WER < 15%):       {triage_counts['accepted']}")
    print(f"  • Review Needed (15-40%):      {triage_counts['review_needed']}")
    print(f"  • Difficult (WER > 40%):       {triage_counts['difficult']}")

    return results


if __name__ == "__main__":
    # Quick self-test with synthetic data
    test_data = [
        {
            "segment_id": "test_001",
            "transcript_a": "بسم الله الرحمن الرحيم",
            "transcript_b": "بسم الله الرحمن الرحيم",
        },
        {
            "segment_id": "test_002",
            "transcript_a": "أحمد ذهب إلى المدرسة اليوم",
            "transcript_b": "احمد ذهب المدرسة",
        },
        {
            "segment_id": "test_003",
            "transcript_a": "هذا درس في معالجة الإشارات الرقمية",
            "transcript_b": "في الدرس معالجة شاره رقمي",
        },
    ]

    results = evaluate_transcripts(test_data)
    print()
    for r in results:
        print(f"  {r['segment_id']}: WER={r['wer']:.1%} CER={r['cer']:.1%} → {r['triage']}")
