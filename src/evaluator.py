
"""
Arabic Speech Corpus — Transcript Evaluator
============================================

Stage 9: Compare the two ASR transcripts using Stage 8 normalized text.

Decision rules:
1. Both normalized transcripts identical -> accepted.
2. Character Error Rate (CER) <= 2% -> accepted.
3. Relative word edit rate <= configured threshold -> human_choice.
4. Otherwise -> difficult.

Relative word edit rate:
    word edits / max(number of words in A, number of words in B)

The default human-choice threshold is 10%.
It can be changed from the command line.

Important:
- The normalized fields from normalizer.py are used.
- Raw ASR transcripts are never modified.
- Engine A is used as the WER/CER reference by convention only.
  It is NOT treated as ground truth.
- SNR and audio quality do not change the transcript-agreement decision.
"""

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path

try:
    import jiwer
except ImportError:
    jiwer = None


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from config import METADATA_DIR
    METADATA_DIR = Path(METADATA_DIR)
except ImportError:
    try:
        from src.config import METADATA_DIR
        METADATA_DIR = Path(METADATA_DIR)
    except ImportError:
        METADATA_DIR = PROJECT_ROOT / "metadata"


TRIAGE_VERSION = "triage_v3"

CER_ACCEPTANCE_THRESHOLD = 0.02

# Initial tunable threshold:
# If relative word disagreement is <= 10%, send to human choice.
HUMAN_CHOICE_MAX_EDIT_RATE = 0.10


# ---------------------------------------------------------
# Normalization access
# ---------------------------------------------------------

def get_normalized_text(record, field):
    """
    Read the normalized transcript created by Stage 8.

    Expected fields:
        asr_a_text_norm
        asr_b_text_norm

    Only fall back to normalizing the raw text if the
    normalized field is missing or unusable.
    """

    normalized_field = f"{field}_norm"
    normalized = record.get(normalized_field)

    if isinstance(normalized, str):
        if normalized.strip():
            return normalized.strip()

        # An empty normalized string is valid if the raw
        # transcript is also empty.
        raw = record.get(field) or ""
        if not raw.strip():
            return ""

    raw = record.get(field) or ""

    if not raw.strip():
        return ""

    # Fallback only. The normal pipeline should already
    # have produced the normalized field in Stage 8.
    try:
        from normalizer import normalize_arabic
    except ImportError:
        from src.normalizer import normalize_arabic

    return normalize_arabic(raw) or ""


# ---------------------------------------------------------
# Word-level edit counting
# ---------------------------------------------------------

def fallback_word_edit_counts(reference, hypothesis):
    """
    Calculate word-level substitutions, deletions and
    insertions using dynamic programming.

    Used only if jiwer is unavailable.
    """

    ref = reference.split()
    hyp = hypothesis.split()

    rows = len(ref) + 1
    cols = len(hyp) + 1

    # Each cell stores:
    # (total_cost, substitutions, deletions, insertions)
    dp = [[None] * cols for _ in range(rows)]

    dp[0][0] = (0, 0, 0, 0)

    for i in range(1, rows):
        dp[i][0] = (i, 0, i, 0)

    for j in range(1, cols):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, rows):
        for j in range(1, cols):

            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            previous = dp[i - 1][j - 1]
            substitution = (
                previous[0] + 1,
                previous[1] + 1,
                previous[2],
                previous[3],
            )

            previous = dp[i - 1][j]
            deletion = (
                previous[0] + 1,
                previous[1],
                previous[2] + 1,
                previous[3],
            )

            previous = dp[i][j - 1]
            insertion = (
                previous[0] + 1,
                previous[1],
                previous[2],
                previous[3] + 1,
            )

            dp[i][j] = min(
                substitution,
                deletion,
                insertion,
                key=lambda item: item[0],
            )

    _, substitutions, deletions, insertions = dp[-1][-1]

    return substitutions, deletions, insertions


def count_word_edits(reference, hypothesis):
    """Return substitutions, deletions, insertions and total edits."""

    if jiwer is not None:
        result = jiwer.process_words(reference, hypothesis)

        substitutions = result.substitutions
        deletions = result.deletions
        insertions = result.insertions
    else:
        substitutions, deletions, insertions = (
            fallback_word_edit_counts(reference, hypothesis)
        )

    total = substitutions + deletions + insertions

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total": total,
    }


# ---------------------------------------------------------
# WER and CER
# ---------------------------------------------------------

def levenshtein_distance(reference, hypothesis):
    """Basic character-level Levenshtein distance."""

    previous = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1

            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1]


def compute_wer_cer(reference, hypothesis):
    """
    Calculate WER and CER.

    Engine A is the reference for metric calculation.
    This is a calculation convention, not a claim that
    Engine A is correct.
    """

    reference = " ".join((reference or "").split())
    hypothesis = " ".join((hypothesis or "").split())

    if not reference:
        return None, None

    if jiwer is not None:
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
    else:
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        word_edits = count_word_edits(reference, hypothesis)["total"]
        wer = word_edits / len(ref_words)

        char_edits = levenshtein_distance(reference, hypothesis)
        cer = char_edits / len(reference)

    return wer, cer


# ---------------------------------------------------------
# Evaluate one segment
# ---------------------------------------------------------

def triage_clip(record, human_choice_max_edit_rate=None):
    """
    Compare the normalized transcripts of one segment.
    """

    if human_choice_max_edit_rate is None:
        human_choice_max_edit_rate = HUMAN_CHOICE_MAX_EDIT_RATE

    text_a = get_normalized_text(record, "asr_a_text")
    text_b = get_normalized_text(record, "asr_b_text")

    words_a = text_a.split()
    words_b = text_b.split()

    word_count_a = len(words_a)
    word_count_b = len(words_b)

    # Use the longer transcript as the denominator.
    comparison_word_count = max(word_count_a, word_count_b)

    edits = count_word_edits(text_a, text_b)

    word_edit_rate = (
        edits["total"] / comparison_word_count
        if comparison_word_count > 0
        else None
    )

    wer, cer = compute_wer_cer(text_a, text_b)

    reasons = []

    # Missing transcript: comparison cannot be trusted.
    if not text_a:
        label = "difficult"
        reasons.append("engine_a_empty")

    elif not text_b:
        label = "difficult"
        reasons.append("engine_b_empty")

    # Both transcripts are identical after normalization.
    elif text_a == text_b:
        label = "accepted"
        reasons.append("identical_normalized_text")

    # CER acceptance rule from the project specification.
    elif cer is not None and cer <= CER_ACCEPTANCE_THRESHOLD:
        label = "accepted"
        reasons.append("cer_at_or_below_2_percent")

    # Relative disagreement determines whether the
    # annotator should choose between the transcripts.
    elif (
        word_edit_rate is not None
        and word_edit_rate <= human_choice_max_edit_rate
    ):
        label = "human_choice"
        reasons.append("relative_word_edit_rate_within_threshold")

    else:
        label = "difficult"
        reasons.append("relative_word_edit_rate_above_threshold")

    return {
        "asr_a_eval_text": text_a,
        "asr_b_eval_text": text_b,

        "word_count_a": word_count_a,
        "word_count_b": word_count_b,
        "comparison_word_count": comparison_word_count,

        "wer": round(wer, 4) if wer is not None else None,
        "cer": round(cer, 4) if cer is not None else None,

        "word_substitutions": edits["substitutions"],
        "word_deletions": edits["deletions"],
        "word_insertions": edits["insertions"],
        "word_edit_count": edits["total"],

        "word_edit_rate": (
            round(word_edit_rate, 4)
            if word_edit_rate is not None
            else None
        ),

        "human_choice_max_edit_rate": human_choice_max_edit_rate,

        "triage_label": label,
        "automatic_status": label,
        "triage_reasons": reasons,
    }


# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

def calculate_stats(values):
    values = [value for value in values if value is not None]

    if not values:
        return {
            "n": 0,
            "median": None,
            "mean": None,
        }

    return {
        "n": len(values),
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
    }


def summarize(records, human_choice_max_edit_rate):
    labels = Counter(
        record.get("triage_label", "unknown")
        for record in records
    )

    reasons = Counter()

    for record in records:
        reasons.update(record.get("triage_reasons", []))

    wer_values = [record.get("wer") for record in records]
    cer_values = [record.get("cer") for record in records]
    edit_values = [record.get("word_edit_count") for record in records]
    rate_values = [record.get("word_edit_rate") for record in records]

    return {
        "triage_version": TRIAGE_VERSION,

        "thresholds": {
            "cer_acceptance": CER_ACCEPTANCE_THRESHOLD,
            "human_choice_max_edit_rate": human_choice_max_edit_rate,
            "relative_rate_denominator": "maximum_word_count_of_either_transcript",
        },

        "segments": len(records),

        "counts": {
            "accepted": labels["accepted"],
            "human_choice": labels["human_choice"],
            "difficult": labels["difficult"],
        },

        "wer": calculate_stats(wer_values),
        "cer": calculate_stats(cer_values),
        "word_edit_count": calculate_stats(edit_values),
        "word_edit_rate": calculate_stats(rate_values),

        "reasons": dict(reasons),
    }


# ---------------------------------------------------------
# File handling
# ---------------------------------------------------------

def atomic_write_json(path, data):
    """Write JSON safely using a temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as temp_file:

            json.dump(
                data,
                temp_file,
                ensure_ascii=False,
                indent=2,
            )

            temp_path = Path(temp_file.name)

        os.replace(temp_path, path)

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------
# Evaluate one video
# ---------------------------------------------------------

def evaluate_video(
    video_id,
    metadata_dir=METADATA_DIR,
    human_choice_max_edit_rate=None,
):
    if human_choice_max_edit_rate is None:
        human_choice_max_edit_rate = HUMAN_CHOICE_MAX_EDIT_RATE

    metadata_dir = Path(metadata_dir)

    transcripts_path = metadata_dir / f"{video_id}_transcripts.json"
    summary_path = metadata_dir / f"{video_id}_triage_summary.json"

    if not transcripts_path.exists():
        raise FileNotFoundError(
            f"Transcript metadata not found: {transcripts_path}"
        )

    with open(transcripts_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a list of transcript records in {transcripts_path}"
        )

    for record in records:

        # Calculate automatic agreement results.
        evaluation = triage_clip(
            record,
            human_choice_max_edit_rate=human_choice_max_edit_rate,
        )

        record.update(evaluation)
        record["triage_version"] = TRIAGE_VERSION

        # Preserve a transcript already completed by a human.
        if record.get("annotation_status") == "completed":
            record["status"] = "accepted"
            continue

        if evaluation["triage_label"] == "accepted":

            record["status"] = "accepted"
            record["annotation_status"] = "not_required"

            # The raw Engine A transcript is used as the
            # initial final transcript. Normalized text is
            # only for comparison, not for corpus output.
            record["final_transcript"] = (
                record.get("asr_a_text") or ""
            )

        elif evaluation["triage_label"] == "human_choice":

            record["status"] = "pending_annotation"
            record["annotation_status"] = "pending"

            # Clear a previous automatically selected
            # transcript if this segment now needs review.
            record["final_transcript"] = ""

        else:

            record["status"] = "difficult"
            record["annotation_status"] = "pending"
            record["final_transcript"] = ""

    summary = summarize(
        records,
        human_choice_max_edit_rate,
    )

    summary["video_id"] = video_id

    atomic_write_json(transcripts_path, records)
    atomic_write_json(summary_path, summary)

    print(f"\n[evaluator] {video_id}: {len(records)} clips")

    print("\n  Project outcomes:")

    total = len(records)

    for label in ("accepted", "human_choice", "difficult"):
        count = summary["counts"][label]
        percentage = (count / total * 100) if total else 0

        print(
            f"  {label:<18} {count:>5} "
            f"({percentage:.1f}%)"
        )

    print("\n  Agreement metrics:")

    for metric in ("wer", "cer", "word_edit_count", "word_edit_rate"):
        values = summary[metric]

        if values["n"]:
            print(
                f"  {metric:<18} "
                f"median {values['median']}, "
                f"mean {values['mean']} "
                f"({values['n']} clips)"
            )

    print("\n  Reasons:")

    if summary["reasons"]:
        for reason, count in summary["reasons"].items():
            print(f"    {reason}: {count}")
    else:
        print("    None")

    print(f"\n  Summary saved: {summary_path}")

    return summary


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agreement between two Arabic ASR engines."
    )

    parser.add_argument(
        "--video-id",
        action="append",
        help="Video ID to evaluate. Can be supplied more than once.",
    )

    parser.add_argument(
        "--human-choice-rate",
        type=float,
        default=HUMAN_CHOICE_MAX_EDIT_RATE,
        help=(
            "Maximum relative word edit rate for human_choice. "
            "Example: 0.10 means 10%%. Default: 0.10."
        ),
    )

    args = parser.parse_args()

    if not 0 <= args.human_choice_rate <= 1:
        parser.error("--human-choice-rate must be between 0 and 1.")

    if args.video_id:
        video_ids = args.video_id
    else:
        video_ids = [
            path.name.replace("_transcripts.json", "")
            for path in METADATA_DIR.glob("*_transcripts.json")
        ]

    if not video_ids:
        print("[evaluator] No transcript files found.")
        return

    if jiwer is None:
        print(
            "[evaluator] Warning: jiwer is not installed. "
            "Using the built-in edit-distance fallback."
        )

    print(
        f"[evaluator] Human-choice threshold: "
        f"{args.human_choice_rate * 100:.1f}%"
    )

    for video_id in video_ids:
        evaluate_video(
            video_id,
            human_choice_max_edit_rate=args.human_choice_rate,
        )


if __name__ == "__main__":
    main()
