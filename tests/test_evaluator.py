"""Tests for transcript normalization freshness and disagreement triage."""

from src import evaluator
from src.normalizer import normalize_arabic


def test_stale_saved_normalized_text_is_recomputed_from_raw_text():
    record = {
        "asr_a_text": "السفر من الصفر",
        "asr_a_text_norm": "نص قديم",
        "asr_b_text": "الصفر من الصفر",
        "asr_b_text_norm": "الصفر من الصفر",
    }

    assert evaluator.get_normalized_text(record, "asr_a_text") == normalize_arabic(
        record["asr_a_text"]
    )
    assert record["asr_a_text"] == "السفر من الصفر"


def test_generation_failure_is_triaged_difficult():
    result = evaluator.triage_clip(
        {
            "asr_a_text": "ده كلام مفهوم وطبيعي",
            "asr_b_text": "ده كلام مفهوم وطبيعي نفسي نفسي نفسي",
            "asr_b_quality_flags": ["token_limit_reached"],
        }
    )

    assert result["triage_label"] == "difficult"
    assert "engine_output_hit_token_limit" in result["triage_reasons"]


def test_repetition_failure_is_triaged_difficult():
    result = evaluator.triage_clip(
        {
            "asr_a_text": "ده كلام مفهوم وطبيعي",
            "asr_b_text": "ده كلام مفهوم وطبيعي",
            "asr_b_quality_flags": ["repetition_loop"],
        }
    )

    assert result["triage_label"] == "difficult"
    assert "engine_output_repetition_suspected" in result["triage_reasons"]


def test_word_edit_counts_and_metrics_are_directionally_consistent():
    edits = evaluator.count_word_edits("a b c", "a x")
    wer, _ = evaluator.compute_wer_cer("a b c", "a x")

    assert edits == {
        "substitutions": 1,
        "deletions": 1,
        "insertions": 0,
        "total": 2,
    }
    assert wer == 2 / 3


def test_evaluation_does_not_mutate_raw_transcript_fields():
    record = {
        "asr_a_text": "السفر بتاعي",
        "asr_b_text": "الصفر بتاعي",
    }
    original = dict(record)

    evaluator.triage_clip(record)

    assert record == original
