"""
src/evaluator.py
================
Stage 9: How much do the two engines disagree?  (symmetric, length-aware) and triage.

The rule from the project brief, made length-aware
--------------------------------------------------
    Both transcripts agree            identical after normalization, or CER <= 2%   -> accepted
    Different in a few words          the annotator listens and chooses A, B or corrects -> review_needed
    Very different                    many words differ                              -> difficult

"A few words" is not a fixed 3.  It scales with the clip: up to `few_ratio` of the words
(default 15%), but at least `min_words` and at most `max_words`:

        allowed = min(max_words, max(min_words, ceil(few_ratio * words_in_clip)))

    10-word clip -> 2 words    25-word clip -> 4 words    60-word clip -> 6 words (the cap)

How "different words" is counted
--------------------------------
* SYMMETRIC: the number of word edits (substitutions + insertions + deletions) between the
  two transcripts, relative to the LONGER of the two.  Neither engine is the reference.
* TOLERANT of spelling: Egyptian Arabic has no standard spelling, so two words that are
  almost the same (at most `near_ratio` of their letters differ, words of 5+ letters) count
  as the same word.  Set --near-ratio 0 to require exact words.
* CER is symmetric too (character edits / the longer text).

A clip is NOT accepted, even when the engines agree, if a warning sign is present (agreement
is not proof).  Warning signs: region "mixed"/"music", a flagged cut, engine B text missing,
Latin letters / digits left in engine A's text, Whisper compression_ratio > 2.4 or
avg_logprob < -1.0, WADA SNR below min_snr_db.  Engine A text empty or Whisper
no_speech_prob > 0.6 makes a clip "difficult".

All numbers are starting values to TUNE: the summary shows how the labels would change under
other settings, and src/engine_report.py measures how accurate the "accepted" clips really are
once you have reviewed some clips.

Output
------
Adds to each record in metadata/<video_id>_transcripts.json:
    diff_words, diff_words_strict, n_words_max, diff_ratio, allowed_diff_words, cer_sym,
    wer, cer (engine A as reference - informational only), triage_label, triage_reasons,
    triage_version
and writes metadata/<video_id>_triage_summary.json.  Raw and normalized texts are never changed.

Usage (PowerShell, repository root):
    python src/evaluator.py --video-id CvrNaPFIwcQ
    python src/evaluator.py --few-ratio 0.25 --max-words 8 --near-ratio 0.2
"""

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalizer import normalize_with_flags

METADATA_DIR = PROJECT_ROOT / "metadata"
TRIAGE_VERSION = "triage_v3_waw_spacing"
MIN_NEAR_LENGTH = 5                      # words shorter than this must match exactly

try:                                    # optional dependency
    import jiwer as _jiwer
except ImportError:                     # pragma: no cover - exercised only without jiwer
    _jiwer = None


@dataclass
class TriageConfig:
    agree_cer: float = 0.02              # CER at or below this = "the engines agree"
    few_ratio: float = 0.15              # a few words = up to this share of the clip's words...
    min_words: int = 1                   # ...but always allow at least this many
    max_words: int = 6                   # ...and never more than this many
    near_ratio: float = 0.20             # words this close in spelling count as equal (0 = exact only)
    max_compression_ratio: float = 2.4   # Whisper repetition-loop warning
    min_avg_logprob: float = -1.0        # Whisper "unsure" warning
    max_no_speech_prob: float = 0.6      # Whisper "probably not speech"
    min_snr_db: Optional[float] = 10.0   # WADA SNR below this -> review (None disables)


def allowed_words(n_words: int, cfg: TriageConfig) -> int:
    """How many differing words still count as 'a few' for a clip of this many words."""
    scaled = math.ceil(round(cfg.few_ratio * n_words, 9))
    return min(cfg.max_words, max(cfg.min_words, scaled))


# --------------------------------------------------------------------------- #
# Edit distances
# --------------------------------------------------------------------------- #
def _edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance (substitution, insertion, deletion all cost 1)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1,              # deletion
                               current[j - 1] + 1,           # insertion
                               previous[j - 1] + (x != y)))  # substitution / match
        previous = current
    return previous[-1]


@lru_cache(maxsize=None)
def words_near(a: str, b: str, near_ratio: float) -> bool:
    """Same word, allowing a small spelling difference (words of 5+ letters only)."""
    if a == b:
        return True
    if near_ratio <= 0 or min(len(a), len(b)) < MIN_NEAR_LENGTH:
        return False
    return _edit_distance(a, b) / max(len(a), len(b)) <= near_ratio


def word_difference(words_a: Sequence[str], words_b: Sequence[str], near_ratio: float = 0.0) -> int:
    """Word-level edit distance in which a substitution between 'near' words is free."""
    previous = list(range(len(words_b) + 1))
    for i, x in enumerate(words_a, start=1):
        current = [i]
        for j, y in enumerate(words_b, start=1):
            substitution = 0 if words_near(x, y, near_ratio) else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + substitution))
        previous = current
    return previous[-1]


def compare_texts(text_a: str, text_b: str, near_ratio: float = 0.0) -> Dict[str, Any]:
    """Symmetric comparison of two normalized transcripts."""
    words_a, words_b = text_a.split(), text_b.split()
    n_max = max(len(words_a), len(words_b))
    if n_max == 0:
        return {"n_words_max": 0, "diff_words": 0, "diff_words_strict": 0, "diff_ratio": 0.0, "cer_sym": 0.0}
    strict = word_difference(words_a, words_b, 0.0)
    soft = strict if near_ratio <= 0 else word_difference(words_a, words_b, near_ratio)
    chars = max(len(text_a), len(text_b))
    return {
        "n_words_max": n_max,
        "diff_words": soft,
        "diff_words_strict": strict,
        "diff_ratio": soft / n_max,
        "cer_sym": _edit_distance(text_a, text_b) / chars,
    }


def compute_wer_cer(ref: Optional[str], hyp: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Classic WER/CER of hyp against ref (ref = engine A).  Informational only."""
    ref = " ".join((ref or "").split())
    hyp = " ".join((hyp or "").split())
    if not ref:
        return None, None
    if _jiwer is not None:
        return float(_jiwer.wer(ref, hyp)), float(_jiwer.cer(ref, hyp))
    return _edit_distance(ref.split(), hyp.split()) / len(ref.split()), _edit_distance(ref, hyp) / len(ref)


def canonicalize_arabic_waw_spacing(text: Optional[str]) -> Optional[str]:
    """Join a standalone Arabic conjunction و to the following word for comparison.

    ASR systems differ in whether they emit the conjunction attached ("ومصر")
    or as a separate token ("و مصر"). This comparison-only rule gives both
    spellings one form without changing either saved raw transcript.
    """
    if text is None:
        return None
    words = text.split()
    canonical = []
    index = 0
    while index < len(words):
        next_is_arabic = (
            index + 1 < len(words)
            and words[index + 1]
            and "\u0600" <= words[index + 1][0] <= "\u06ff"
        )
        if words[index] == "و" and next_is_arabic:
            canonical.append("و" + words[index + 1])
            index += 2
        else:
            canonical.append(words[index])
            index += 1
    return " ".join(canonical)


# --------------------------------------------------------------------------- #
# Triage
# --------------------------------------------------------------------------- #
def agreement_label(diff_words: int, cer_sym: float, n_words: int, cfg: TriageConfig) -> str:
    """accepted / review_needed / difficult from the disagreement alone."""
    if diff_words == 0 or cer_sym <= cfg.agree_cer:
        return "accepted"
    return "review_needed" if diff_words <= allowed_words(n_words, cfg) else "difficult"


def triage_clip(
    rec: Dict[str, Any],
    cfg: TriageConfig,
    region_label: Optional[str] = None,
    boundary_check: Optional[str] = None,
    snr_wada_db: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one clip: the disagreement metrics, a triage label and the reasons for it."""
    # Always normalize from the RAW text.  Stored *_text_norm fields can be stale (for example
    # after engine B was re-run), and normalizing is instant and deterministic anyway.
    raw_a = rec.get("asr_a_text")
    raw_b = rec.get("asr_b_text")
    text_a, flags_a = normalize_with_flags(raw_a if raw_a is not None else rec.get("asr_a_text_norm"))
    text_b, _ = normalize_with_flags(raw_b if raw_b is not None else rec.get("asr_b_text_norm"))
    text_a = canonicalize_arabic_waw_spacing(text_a)
    text_b = canonicalize_arabic_waw_spacing(text_b)

    difficult: List[str] = []
    review: List[str] = []
    result: Dict[str, Any] = {"asr_a_eval_text": text_a, "asr_b_eval_text": text_b,
                              "n_words_max": None, "diff_words": None, "diff_words_strict": None,
                              "diff_ratio": None, "allowed_diff_words": None, "cer_sym": None,
                              "wer": None, "cer": None}

    # --- Whisper's own warning signs ---
    if not text_a:
        difficult.append("engine_a_empty")
    no_speech = rec.get("asr_a_no_speech_prob")
    if no_speech is not None and no_speech > cfg.max_no_speech_prob:
        difficult.append("no_speech_prob_high")
    compression = rec.get("asr_a_compression_ratio")
    if compression is not None and compression > cfg.max_compression_ratio:
        review.append("compression_ratio_high")
    logprob = rec.get("asr_a_avg_logprob")
    if logprob is not None and logprob < cfg.min_avg_logprob:
        review.append("avg_logprob_low")

    # --- agreement between the engines ---
    if text_b is None:
        review.append("engine_b_missing")
    elif text_a:
        metrics = compare_texts(text_a, text_b, cfg.near_ratio)
        allowed = allowed_words(metrics["n_words_max"], cfg)
        wer, cer = compute_wer_cer(text_a, text_b)
        result.update({**metrics, "allowed_diff_words": allowed,
                       "diff_ratio": round(metrics["diff_ratio"], 4), "cer_sym": round(metrics["cer_sym"], 4),
                       "wer": None if wer is None else round(wer, 4),
                       "cer": None if cer is None else round(cer, 4)})
        verdict = agreement_label(metrics["diff_words"], metrics["cer_sym"], metrics["n_words_max"], cfg)
        if verdict == "difficult":
            difficult.append("many_words_differ")
        elif verdict == "review_needed":
            review.append("few_words_differ")

    # --- context: where the clip came from ---
    if region_label in ("mixed", "music"):
        review.append(f"region_{region_label}")
    if boundary_check == "flagged_manual_review":
        review.append("boundary_flagged")
    if snr_wada_db is not None and cfg.min_snr_db is not None and snr_wada_db < cfg.min_snr_db:
        review.append("low_snr")
    # Flags describe engine A's text, the one reviewers start from.  Engine B may use another
    # convention (for example English words in Latin letters); that already shows up in the difference.
    for flag in sorted(set(flags_a)):
        if flag != "empty":
            review.append(f"norm_{flag}")

    label = "difficult" if difficult else ("review_needed" if review else "accepted")
    result.update({"triage_label": label, "triage_reasons": difficult + review})
    return result


def summarize(records: List[Dict[str, Any]], cfg: TriageConfig) -> Dict[str, Any]:
    """Counts per label, the disagreement distribution, and how the labels change under other settings."""
    labels = [r["triage_label"] for r in records]
    ratios = sorted(r["diff_ratio"] for r in records if r.get("diff_ratio") is not None)
    reasons: Dict[str, int] = {}
    for r in records:
        for reason in r["triage_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    def quantile(q: float) -> Optional[float]:
        return None if not ratios else round(ratios[min(len(ratios) - 1, int(q * len(ratios)))], 4)

    measured = [r for r in records if r.get("diff_words") is not None]

    def counts_for(few_ratio: float, strict: bool) -> Dict[str, int]:
        alt = TriageConfig(**{**asdict(cfg), "few_ratio": few_ratio})
        out = {"accepted": 0, "review_needed": 0, "difficult": 0}
        for r in measured:
            d = r["diff_words_strict"] if strict else r["diff_words"]
            out[agreement_label(d, r["cer_sym"], r["n_words_max"], alt)] += 1
        return out

    sensitivity = [{"few_ratio": f, "tolerant_spelling": counts_for(f, False), "exact_words": counts_for(f, True)}
                   for f in (0.10, 0.15, 0.20, 0.30, 0.40)]
    return {
        "triage_version": TRIAGE_VERSION,
        "settings": asdict(cfg),
        "segments": len(records),
        "counts": {k: labels.count(k) for k in ("accepted", "review_needed", "difficult")},
        "difference_ratio": {"n": len(ratios), "median": quantile(0.5), "p75": quantile(0.75),
                             "p90": quantile(0.9),
                             "mean": None if not ratios else round(statistics.mean(ratios), 4)},
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "agreement_only_sensitivity": sensitivity,
    }


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def evaluate_video(
    video_id: str,
    metadata_dir: Path = METADATA_DIR,
    cfg: Optional[TriageConfig] = None,
) -> Dict[str, Any]:
    """Evaluate every clip of one video; update its transcripts file; return the summary."""
    cfg = cfg or TriageConfig()
    metadata_dir = Path(metadata_dir)
    path = metadata_dir / f"{video_id}_transcripts.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run Stage 7 (transcriber) first.")
    records = _load_json(path)

    manifest, snr = {}, {}
    if (metadata_dir / f"{video_id}_segments.json").exists():
        manifest = {r["segment_id"]: r for r in _load_json(metadata_dir / f"{video_id}_segments.json")}
    if (metadata_dir / f"{video_id}_snr.json").exists():
        snr = {r["segment_id"]: r for r in _load_json(metadata_dir / f"{video_id}_snr.json")}

    for rec in records:
        seg = manifest.get(rec["segment_id"], {})
        rec.update(triage_clip(
            rec, cfg,
            region_label=seg.get("region_label"),
            boundary_check=seg.get("boundary_check"),
            snr_wada_db=snr.get(rec["segment_id"], {}).get("snr_wada_db"),
        ))
        rec["triage_version"] = TRIAGE_VERSION

    summary = summarize(records, cfg)
    summary["video_id"] = video_id
    _save_json(path, records)
    _save_json(metadata_dir / f"{video_id}_triage_summary.json", summary)
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    c, d, s = summary["counts"], summary["difference_ratio"], summary["settings"]
    n = summary["segments"] or 1
    print(f"\n[evaluator] {summary['video_id']}: {summary['segments']} clips")
    for label in ("accepted", "review_needed", "difficult"):
        print(f"  {label:<14} {c[label]:>4}  ({100 * c[label] / n:.0f}%)")
    print(f"  share of words that differ (A vs B, symmetric): median {d['median']}, p75 {d['p75']}, p90 {d['p90']}"
          f"  (over {d['n']} clips)")
    print(f"  settings: a few words = up to {100 * s['few_ratio']:.0f}% of the clip "
          f"(min {s['min_words']}, max {s['max_words']}), near-spelling tolerance {s['near_ratio']}")
    if summary["reasons"]:
        top = ", ".join(f"{k} x{v}" for k, v in list(summary["reasons"].items())[:6])
        print(f"  top reasons: {top}")
    print("  agreement alone, under other 'a few words' limits (accepted / review / difficult):")
    for row in summary["agreement_only_sensitivity"]:
        t, e = row["tolerant_spelling"], row["exact_words"]
        print(f"    {100 * row['few_ratio']:.0f}% of the words:  tolerant spelling "
              f"{t['accepted']} / {t['review_needed']} / {t['difficult']}    exact words "
              f"{e['accepted']} / {e['review_needed']} / {e['difficult']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 9: symmetric engine disagreement and triage.")
    parser.add_argument("--video-id", action="append", help="Limit to this video id (repeatable).")
    parser.add_argument("--agree-cer", type=float, default=TriageConfig.agree_cer,
                        help="CER at or below this counts as agreement (default 0.02).")
    parser.add_argument("--few-ratio", type=float, default=TriageConfig.few_ratio,
                        help="'a few words' = up to this share of the clip's words (default 0.15).")
    parser.add_argument("--min-words", type=int, default=TriageConfig.min_words)
    parser.add_argument("--max-words", type=int, default=TriageConfig.max_words)
    parser.add_argument("--near-ratio", type=float, default=TriageConfig.near_ratio,
                        help="words this close in spelling count as equal; 0 = exact (default 0.2).")
    parser.add_argument("--min-snr-db", type=float, default=TriageConfig.min_snr_db,
                        help="WADA SNR below this sends a clip to review (default 10).")
    args = parser.parse_args(argv)

    cfg = TriageConfig(agree_cer=args.agree_cer, few_ratio=args.few_ratio, min_words=args.min_words,
                       max_words=args.max_words, near_ratio=args.near_ratio, min_snr_db=args.min_snr_db)
    if args.video_id:
        video_ids = args.video_id
    else:
        video_ids = sorted(p.name[: -len("_transcripts.json")] for p in METADATA_DIR.glob("*_transcripts.json"))
    if not video_ids:
        print(f"No *_transcripts.json files in {METADATA_DIR}. Run Stage 7 first.")
        return 1
    for video_id in video_ids:
        print_summary(evaluate_video(video_id, cfg=cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
