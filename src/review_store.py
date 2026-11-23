
"""
src/review_store.py
===================
Stage 10: CSV-based review data layer.

The per-video CSV is the single source of truth for human annotations.
The JSON files provide the original segment, SNR, and ASR information.

CSV columns:
    start, end, label, snr, quality,
    model_A_transcript, model_B_transcript,
    final_transcript, status, segment_id
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "metadata"

CSV_COLUMNS = [
    "start",
    "end",
    "label",
    "snr",
    "quality",
    "model_A_transcript",
    "model_B_transcript",
    "final_transcript",
    "status",
    "segment_id",
]

STATUSES = ("pending", "accepted", "corrected", "rejected")
TRIAGE_ORDER = {"difficult": 0, "review_needed": 1, "accepted": 2}


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def clean_text(text: Optional[str]) -> Optional[str]:
    """Collapse runs of whitespace."""
    return None if text is None else " ".join(str(text).split())


_clean = clean_text


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def csv_path(video_id: str, metadata_dir: Path = METADATA_DIR) -> Path:
    return Path(metadata_dir) / f"{video_id}_review.csv"


def _initial_status(record: Dict[str, Any]) -> str:
    """
    Convert the existing pipeline annotation fields into review status.

    A segment marked not_required and automatically accepted is accepted.
    Everything else starts pending.
    """
    annotation_status = record.get("annotation_status")
    automatic_status = record.get("automatic_status")

    if annotation_status == "not_required" and automatic_status == "accepted":
        return "accepted"

    if annotation_status in STATUSES:
        return annotation_status

    return "pending"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write the CSV through a temporary file to reduce partial-write risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    with open(temporary_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    temporary_path.replace(path)


# --------------------------------------------------------------------------- #
# CSV generation and loading
# --------------------------------------------------------------------------- #
def list_videos(metadata_dir: Path = METADATA_DIR) -> List[str]:
    """Find video IDs from the existing transcript JSON files."""
    return sorted(
        p.name[:-len("_transcripts.json")]
        for p in Path(metadata_dir).glob("*_transcripts.json")
    )


def ensure_csv(
    video_id: str,
    metadata_dir: Path = METADATA_DIR,
) -> Path:
    """
    Create the per-video review CSV if it does not exist.

    If it already exists, preserve reviewer-controlled fields:
    label, quality, final_transcript, and status.

    Original timing, SNR, and model transcripts are refreshed from JSON.
    """
    metadata_dir = Path(metadata_dir)
    output_path = csv_path(video_id, metadata_dir)

    transcripts_path = metadata_dir / f"{video_id}_transcripts.json"
    segments_path = metadata_dir / f"{video_id}_segments.json"
    snr_path = metadata_dir / f"{video_id}_snr.json"

    if not transcripts_path.exists():
        raise FileNotFoundError(f"Transcript JSON not found: {transcripts_path}")

    transcripts = _load_json(transcripts_path)
    segments = _load_json(segments_path) if segments_path.exists() else []
    snr_records = _load_json(snr_path) if snr_path.exists() else []

    segment_by_id = {r["segment_id"]: r for r in segments}
    snr_by_id = {r["segment_id"]: r for r in snr_records}

    # Read previous annotations, if a CSV already exists.
    old_rows = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("segment_id"):
                    old_rows[row["segment_id"]] = row

    new_rows = []

    for record in transcripts:
        sid = record["segment_id"]
        segment = segment_by_id.get(sid, {})
        snr = snr_by_id.get(sid, {})
        previous = old_rows.get(sid, {})

        # Segment manifest is the preferred source for timing and region label.
        start = segment.get("start_sec", record.get("start_sec", ""))
        end = segment.get("end_sec", record.get("end_sec", ""))
        original_label = segment.get("region_label", record.get("region_label", ""))

        row = {
            "start": start,
            "end": end,
            "label": original_label,
            "snr": snr.get("snr_wada_db", ""),
            "quality": "",
            "model_A_transcript": record.get("asr_a_text", ""),
            "model_B_transcript": record.get("asr_b_text", ""),
            "final_transcript": record.get("final_transcript", ""),
            "status": _initial_status(record),
            "segment_id": sid,
        }

        # Preserve fields that a reviewer may already have edited.
        if previous:
            for field in ("label", "quality", "final_transcript", "status"):
                if field in previous:
                    row[field] = previous[field]

        new_rows.append(row)

    new_rows.sort(key=lambda r: r["segment_id"])
    _write_csv(output_path, new_rows)
    return output_path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_clips(
    video_id: str,
    metadata_dir: Path = METADATA_DIR,
    root: Path = PROJECT_ROOT,
) -> List[Dict[str, Any]]:
    """
    Load review rows from CSV and enrich them with supplementary JSON metadata.

    CSV values are authoritative for the ten review columns.
    """
    metadata_dir = Path(metadata_dir)
    ensure_csv(video_id, metadata_dir)

    rows = _read_csv(csv_path(video_id, metadata_dir))

    transcripts_path = metadata_dir / f"{video_id}_transcripts.json"
    segments_path = metadata_dir / f"{video_id}_segments.json"
    snr_path = metadata_dir / f"{video_id}_snr.json"

    transcripts = _load_json(transcripts_path)
    segments = _load_json(segments_path) if segments_path.exists() else []
    snr_records = _load_json(snr_path) if snr_path.exists() else []

    transcript_by_id = {r["segment_id"]: r for r in transcripts}
    segment_by_id = {r["segment_id"]: r for r in segments}
    snr_by_id = {r["segment_id"]: r for r in snr_records}

    clips = []

    for row in rows:
        sid = row["segment_id"]
        transcript = transcript_by_id.get(sid, {})
        segment = segment_by_id.get(sid, {})
        snr_record = snr_by_id.get(sid, {})

        # Start with supplementary transcript metadata.
        clip = dict(transcript)
        clip.update(segment)

        # CSV is authoritative for review fields.
        clip.update(row)

        clip["video_id"] = video_id
        clip["start_sec"] = _as_float(row.get("start"))
        clip["end_sec"] = _as_float(row.get("end"))
        clip["region_label"] = segment.get(
            "region_label", transcript.get("region_label")
        )
        clip["snr_wada_db"] = _as_float(row.get("snr"))
        clip["snr_segmental_db"] = snr_record.get("snr_segmental_db")

        if clip["start_sec"] is not None and clip["end_sec"] is not None:
            clip["duration_sec"] = clip["end_sec"] - clip["start_sec"]

        audio_path = transcript.get("audio_path") or segment.get("audio_path")
        clip["audio_path"] = audio_path
        clip["audio_abspath"] = (
            str(Path(root) / audio_path) if audio_path else None
        )

        clips.append(clip)

    return sorted(clips, key=lambda c: c["segment_id"])


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Current decisions and progress
# --------------------------------------------------------------------------- #
def latest_decisions(
    clips: Sequence[Dict[str, Any]],
    video_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build the current decision lookup from the CSV-loaded clips.

    The function name is retained to minimize changes in the review app.
    There is no historical decision log in the CSV-only design.
    """
    return {
        clip["segment_id"]: clip
        for clip in clips
        if video_id is None or clip.get("video_id") == video_id
    }


def status_of(decision: Optional[Dict[str, Any]]) -> str:
    if decision is None:
        return "pending"
    status = decision.get("status", "pending")
    return status if status in STATUSES else "pending"


def progress(
    clips: Sequence[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    counts = {status: 0 for status in STATUSES}

    for clip in clips:
        counts[status_of(decisions.get(clip["segment_id"]))] += 1

    counts["total"] = len(clips)
    return counts


def update_review(
    video_id: str,
    segment_id: str,
    status: str,
    final_transcript: Optional[str] = None,
    label: Optional[str] = None,
    quality: Optional[str] = None,
    metadata_dir: Path = METADATA_DIR,
) -> None:
    """Update one segment's annotation fields directly in its CSV row."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")

    if status in ("accepted", "corrected") and not clean_text(final_transcript):
        raise ValueError("An accepted or corrected clip needs a final transcript.")

    path = csv_path(video_id, metadata_dir)
    if not path.exists():
        ensure_csv(video_id, metadata_dir)

    rows = _read_csv(path)
    found = False

    for row in rows:
        if row.get("segment_id") == segment_id:
            row["status"] = status

            if final_transcript is not None:
                row["final_transcript"] = clean_text(final_transcript) or ""

            if label is not None:
                row["label"] = label

            if quality is not None:
                row["quality"] = quality

            found = True
            break

    if not found:
        raise KeyError(f"Segment {segment_id} was not found in {path.name}")

    _write_csv(path, rows)


# --------------------------------------------------------------------------- #
# Review queue
# --------------------------------------------------------------------------- #
def build_queue(
    clips: Sequence[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
    labels: Optional[Sequence[str]] = None,
    statuses: Optional[Sequence[str]] = ("pending",),
    order: str = "priority",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Filter clips by triage label and CSV review status.

    order:
        priority  - difficult first, then highest WER
        recording - segment ID order
        random    - deterministic shuffle
    """
    selected = []

    for clip in clips:
        triage_label = clip.get("triage_label") or "review_needed"

        if labels is not None and triage_label not in labels:
            continue

        if statuses is not None:
            current_status = status_of(decisions.get(clip["segment_id"]))
            if current_status not in statuses:
                continue

        selected.append(clip)

    if order == "recording":
        return sorted(selected, key=lambda c: c["segment_id"])

    if order == "random":
        shuffled = sorted(selected, key=lambda c: c["segment_id"])
        random.Random(seed).shuffle(shuffled)
        return shuffled

    if order != "priority":
        raise ValueError('order must be "priority", "recording" or "random"')

    def priority(clip: Dict[str, Any]):
        triage_label = clip.get("triage_label") or "review_needed"
        wer = clip.get("wer")
        return (
            TRIAGE_ORDER.get(triage_label, 1),
            -(wer if wer is not None else -1.0),
            clip["segment_id"],
        )

    return sorted(selected, key=priority)


def starting_text(
    clip: Dict[str, Any],
    decision: Optional[Dict[str, Any]],
) -> str:
    """Prefer a previously saved final transcript, otherwise start with Engine A."""
    if decision is not None and decision.get("final_transcript"):
        return decision["final_transcript"]

    return clip.get("asr_a_text") or clip.get("asr_b_text") or ""
