
"""
src/review_store.py
===================
Stage 10: CSV-based review data layer.

The per-video CSV is the single source of truth for human annotations.
The JSON files provide the original segment, SNR, and ASR information.

CSV columns:
    start, end, label, snr, quality,
    model_A_transcript, model_B_transcript,
    final_transcript, status, segment_id, plus review/audit fields.
"""

import csv
import json
import random
from datetime import datetime, timezone
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
    "model_A_language",
    "model_B_transcript",
    "model_B_language",
    "final_transcript",
    "status",
    "review_status",
    "automatic_status",
    "segment_id",
    "audio_file",
    "changed_by",
    "changed_at",
    "change_log",
]

STATUSES = ("pending", "accepted", "corrected", "rejected")
CORPUS_STATUSES = ("accepted", "human_choice", "difficult", "deleted")
TRIAGE_ORDER = {"difficult": 0, "human_choice": 1, "review_needed": 1, "accepted": 2}
SNR_CLEAN_THRESHOLD_DB = 20.0


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

    if record.get("triage_label") == "accepted":
        return "accepted"

    if annotation_status == "not_required" and automatic_status == "accepted":
        return "accepted"

    if annotation_status in STATUSES:
        return annotation_status

    return "pending"


def _corpus_status(triage_label: Optional[str]) -> str:
    """Map evaluator triage labels to the project CSV's three outcomes."""
    if triage_label == "accepted":
        return "accepted"
    if triage_label in ("human_choice", "review_needed"):
        return "human_choice"
    return "difficult"


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

    If it already exists, preserve reviewer-controlled fields, timing,
    attribution, and manual segment rows.

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
        previous = old_rows.get(sid, {})
        has_model_transcript = bool(
            clean_text(record.get("asr_a_text"))
            or clean_text(record.get("asr_b_text"))
        )
        has_saved_human_work = bool(
            clean_text(previous.get("final_transcript"))
            or clean_text(previous.get("changed_by"))
            or previous.get("review_status") in ("accepted", "corrected", "rejected")
        )

        # Segmentation can produce many more clips than the user selects for
        # transcription. Keep those untouched clips out of review/export until
        # at least one engine has supplied text. Retain rows with prior human
        # work so refreshing the CSV never discards an annotation.
        if not has_model_transcript and not has_saved_human_work:
            continue

        segment = segment_by_id.get(sid, {})
        snr = snr_by_id.get(sid, {})
        snr_value = snr.get("snr_wada_db", "")
        try:
            initial_quality = (
                "clean" if float(snr_value) >= SNR_CLEAN_THRESHOLD_DB else "noisy"
            ) if snr_value not in (None, "") else ""
        except (TypeError, ValueError):
            initial_quality = ""

        # Segment manifest is the preferred source for timing and region label.
        start = segment.get("start_sec", record.get("start_sec", ""))
        end = segment.get("end_sec", record.get("end_sec", ""))
        original_label = segment.get("region_label", record.get("region_label", ""))

        row = {
            "start": start,
            "end": end,
            "label": original_label,
            "snr": snr_value,
            "quality": initial_quality,
            "model_A_transcript": record.get("asr_a_text", ""),
            "model_A_language": record.get("asr_a_language", ""),
            "model_B_transcript": record.get("asr_b_text", ""),
            "model_B_language": record.get("asr_b_language", ""),
            "final_transcript": (
                record.get("asr_b_text", "")
                if _corpus_status(record.get("triage_label")) == "accepted"
                else record.get("final_transcript", "")
            ),
            "status": _corpus_status(record.get("triage_label")),
            "review_status": _initial_status(record),
            "automatic_status": _corpus_status(record.get("triage_label")),
            "segment_id": sid,
            "audio_file": record.get("audio_path") or segment.get("audio_path", ""),
            "changed_by": "",
            "changed_at": "",
            "change_log": "[]",
        }

        # Preserve fields that a reviewer may already have edited.
        if previous:
            prior_workflow = previous.get("review_status") or previous.get("status", "")
            if prior_workflow not in STATUSES and prior_workflow != "deleted":
                prior_workflow = (
                    prior_workflow if prior_workflow in STATUSES
                    else _initial_status(record)
                )
            prior_corpus = previous.get("status", "")
            has_human_decision = bool(previous.get("changed_by")) or prior_workflow in ("corrected", "rejected")
            if prior_workflow == "deleted" or prior_corpus == "deleted":
                row["review_status"] = "deleted"
                row["status"] = "deleted"
            elif has_human_decision:
                row["review_status"] = prior_workflow
                if prior_corpus in CORPUS_STATUSES:
                    row["status"] = prior_corpus
                elif prior_workflow in ("accepted", "corrected"):
                    row["status"] = "accepted"
                elif prior_workflow == "rejected":
                    row["status"] = "difficult"
                else:
                    row["status"] = row["automatic_status"]
            else:
                row["review_status"] = _initial_status(record)
            for field in (
                "start", "end", "label", "quality", "final_transcript", "audio_file",
                "changed_by", "changed_at", "change_log",
            ):
                if field in previous:
                    if (
                        field == "final_transcript"
                        and not has_human_decision
                        and row["automatic_status"] == "accepted"
                    ):
                        # Keep automatic acceptance consistent with the
                        # selected Engine B transcript, including older CSVs
                        # that were initialized from Engine A.
                        continue
                    row[field] = previous[field]

        new_rows.append(row)

    new_rows.sort(key=lambda r: r["segment_id"])
    # Keep manually added/split/merged rows, which do not exist in the
    # machine-generated transcript JSON. Their CSV record is authoritative.
    known_ids = {row["segment_id"] for row in new_rows}
    generated_ids = {record["segment_id"] for record in transcripts}
    for sid, previous in old_rows.items():
        if sid not in known_ids:
            if sid in generated_ids:
                # This is a machine-generated segment that has not been
                # transcribed yet, rather than a manually added CSV row.
                # Leave it out until an engine transcript or human annotation
                # exists.
                continue
            manual = {column: previous.get(column, "") for column in CSV_COLUMNS}
            new_rows.append(manual)
    def _time_order(row):
        try:
            start_time = float(row.get("start", ""))
        except (TypeError, ValueError):
            start_time = float("inf")
        return start_time, row["segment_id"]

    new_rows.sort(key=_time_order)
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

        # Manual and structurally edited segments can receive fresh ASR
        # results directly in the review CSV, without a transcript JSON row.
        clip["asr_a_text"] = row.get("model_A_transcript") or clip.get("asr_a_text", "")
        clip["asr_a_language"] = row.get("model_A_language") or clip.get("asr_a_language")
        clip["asr_b_text"] = row.get("model_B_transcript") or clip.get("asr_b_text", "")
        clip["asr_b_language"] = row.get("model_B_language") or clip.get("asr_b_language")

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

        audio_path = row.get("audio_file") or transcript.get("audio_path") or segment.get("audio_path")
        clip["audio_path"] = audio_path
        clip["audio_abspath"] = (
            str(Path(root) / audio_path) if audio_path else None
        )

        clips.append(clip)

    return sorted(
        clips,
        key=lambda c: (
            c["start_sec"] if c.get("start_sec") is not None else float("inf"),
            c["segment_id"],
        ),
    )


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
    status = decision.get("review_status", decision.get("status", "pending"))
    if status == "deleted":
        return "deleted"
    return status if status in STATUSES else "pending"


def progress(
    clips: Sequence[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    counts = {status: 0 for status in STATUSES}

    for clip in clips:
        status = status_of(decisions.get(clip["segment_id"]))
        if status != "deleted":
            counts[status] += 1

    counts["total"] = sum(counts.values())
    return counts


def update_review(
    video_id: str,
    segment_id: str,
    status: str,
    final_transcript: Optional[str] = None,
    label: Optional[str] = None,
    quality: Optional[str] = None,
    changed_by: Optional[str] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    snr: Optional[float] = None,
    audio_file: Optional[str] = None,
    metadata_dir: Path = METADATA_DIR,
) -> None:
    """Update one segment's annotation fields directly in its CSV row."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")

    if status in ("accepted", "corrected") and not clean_text(final_transcript):
        raise ValueError("An accepted or corrected clip needs a final transcript.")
    reviewer = clean_text(changed_by)
    if not reviewer:
        raise ValueError("Enter your name before saving a review change.")
    if start is not None and end is not None and float(end) <= float(start):
        raise ValueError("Segment end must be later than segment start.")

    path = csv_path(video_id, metadata_dir)
    if not path.exists():
        ensure_csv(video_id, metadata_dir)

    rows = _read_csv(path)
    found = False

    for row in rows:
        if row.get("segment_id") == segment_id:
            before = {
                "status": row.get("status", "difficult"),
                "review_status": row.get("review_status", row.get("status", "pending")),
                "final_transcript": row.get("final_transcript", ""),
                "label": row.get("label", ""),
                "quality": row.get("quality", ""),
                "snr": row.get("snr", ""),
                "audio_file": row.get("audio_file", ""),
            }
            before.update({
                key: row.get(key, "")
                for key in ("start", "end")
            })
            row["review_status"] = status
            if status in ("accepted", "corrected"):
                row["status"] = "accepted"
            elif status == "rejected":
                row["status"] = "difficult"
            elif status == "pending":
                row["status"] = row.get("automatic_status") or "human_choice"

            if final_transcript is not None:
                row["final_transcript"] = clean_text(final_transcript) or ""

            if label is not None:
                row["label"] = label

            if quality is not None:
                row["quality"] = quality
            if start is not None:
                row["start"] = round(float(start), 3)
            if end is not None:
                row["end"] = round(float(end), 3)
            if snr is not None:
                row["snr"] = round(float(snr), 2)
            if audio_file is not None:
                row["audio_file"] = audio_file

            after = {
                "status": row.get("status", "difficult"),
                "review_status": row.get("review_status", "pending"),
                "final_transcript": row.get("final_transcript", ""),
                "label": row.get("label", ""),
                "quality": row.get("quality", ""),
                "snr": row.get("snr", ""),
                "audio_file": row.get("audio_file", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
            }
            if before != after:
                stamp = datetime.now(timezone.utc).isoformat()
                try:
                    history = json.loads(row.get("change_log") or "[]")
                except (TypeError, json.JSONDecodeError):
                    history = []
                history.append({
                    "changed_at": stamp,
                    "changed_by": reviewer,
                    "before": before,
                    "after": after,
                })
                row["changed_by"] = reviewer
                row["changed_at"] = stamp
                row["change_log"] = json.dumps(history, ensure_ascii=False)

            found = True
            break

    if not found:
        raise KeyError(f"Segment {segment_id} was not found in {path.name}")

    _write_csv(path, rows)


def save_asr_results(
    video_id: str,
    segment_id: str,
    results: Dict[str, Any],
    reviewer: str,
    metadata_dir: Path = METADATA_DIR,
) -> None:
    """Save model transcripts for a segment without changing its review decision."""
    reviewer = clean_text(reviewer)
    if not reviewer:
        raise ValueError("Enter your name before saving ASR results.")

    field_map = {
        "asr_a_text": "model_A_transcript",
        "asr_a_language": "model_A_language",
        "asr_b_text": "model_B_transcript",
        "asr_b_language": "model_B_language",
    }
    updates = {
        target: clean_text(results[source]) or ""
        for source, target in field_map.items()
        if source in results
    }
    if not updates:
        raise ValueError("No model transcript fields were provided.")

    path = csv_path(video_id, metadata_dir)
    if not path.exists():
        raise FileNotFoundError(f"Review CSV not found: {path}")

    rows = _read_csv(path)
    for row in rows:
        if row.get("segment_id") != segment_id:
            continue

        before = {field: row.get(field, "") for field in updates}
        row.update(updates)
        after = {field: row.get(field, "") for field in updates}
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            history = json.loads(row.get("change_log") or "[]")
        except (TypeError, json.JSONDecodeError):
            history = []
        history.append({
            "changed_at": stamp,
            "changed_by": reviewer,
            "operation": "asr_refresh",
            "before": before,
            "after": after,
        })
        row["changed_by"] = reviewer
        row["changed_at"] = stamp
        row["change_log"] = json.dumps(history, ensure_ascii=False)
        _write_csv(path, rows)
        return

    raise KeyError(f"Segment {segment_id} was not found in {path.name}")


def _manual_id(video_id: str, rows: Sequence[Dict[str, str]]) -> str:
    """Allocate a stable-looking ID that cannot collide with generated IDs."""
    existing = {row.get("segment_id", "") for row in rows}
    index = 1
    while f"{video_id}_manual_{index:04d}" in existing:
        index += 1
    return f"{video_id}_manual_{index:04d}"


def _append_manual_row(
    video_id: str,
    rows: List[Dict[str, str]],
    start: float,
    end: float,
    label: str,
    quality: str,
    reviewer: str,
    operation: str,
    snr: Optional[float] = None,
) -> str:
    sid = _manual_id(video_id, rows)
    stamp = datetime.now(timezone.utc).isoformat()
    event = {
        "changed_at": stamp,
        "changed_by": reviewer,
        "operation": operation,
        "before": None,
        "after": {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "label": label,
            "quality": quality,
            "snr": round(float(snr), 2) if snr is not None else "",
            "status": "pending",
        },
    }
    rows.append({
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "label": label,
        "snr": round(float(snr), 2) if snr is not None else "",
        "quality": quality or (
            "clean" if snr is not None and float(snr) >= SNR_CLEAN_THRESHOLD_DB
            else "noisy" if snr is not None else ""
        ),
        "model_A_transcript": "",
        "model_B_transcript": "",
        "final_transcript": "",
        "status": "human_choice",
        "review_status": "pending",
        "automatic_status": "human_choice",
        "segment_id": sid,
        "changed_by": reviewer,
        "changed_at": stamp,
        "change_log": json.dumps([event], ensure_ascii=False),
    })
    return sid


def add_segment(
    video_id: str,
    start: float,
    end: float,
    reviewer: str,
    label: str = "speech",
    quality: str = "",
    snr: Optional[float] = None,
    metadata_dir: Path = METADATA_DIR,
) -> str:
    """Add a manually timed segment to the per-video CSV."""
    reviewer = clean_text(reviewer)
    if not reviewer:
        raise ValueError("Enter your name before changing segment structure.")
    if label not in ("speech", "music", "mixed"):
        raise ValueError("Segment label must be speech, music, or mixed.")
    if float(start) < 0 or float(end) <= float(start):
        raise ValueError("Segment times must satisfy 0 <= start < end.")
    path = csv_path(video_id, metadata_dir)
    if not path.exists():
        ensure_csv(video_id, metadata_dir)
    rows = _read_csv(path)
    sid = _append_manual_row(video_id, rows, start, end, label, quality, reviewer, "add", snr)
    _write_csv(path, rows)
    return sid


def delete_segment(
    video_id: str,
    segment_id: str,
    reviewer: str,
    metadata_dir: Path = METADATA_DIR,
) -> None:
    """Mark a segment deleted while retaining its row and audit trail."""
    reviewer = clean_text(reviewer)
    if not reviewer:
        raise ValueError("Enter your name before changing segment structure.")
    path = csv_path(video_id, metadata_dir)
    rows = _read_csv(path)
    for row in rows:
        if row.get("segment_id") == segment_id:
            if row.get("status") == "deleted":
                return
            before = {key: row.get(key, "") for key in ("start", "end", "label", "status", "review_status")}
            row["status"] = "deleted"
            row["review_status"] = "deleted"
            stamp = datetime.now(timezone.utc).isoformat()
            try:
                history = json.loads(row.get("change_log") or "[]")
            except (TypeError, json.JSONDecodeError):
                history = []
            history.append({"changed_at": stamp, "changed_by": reviewer,
                            "operation": "delete", "before": before,
                            "after": {**before, "status": "deleted", "review_status": "deleted"}})
            row["changed_by"] = reviewer
            row["changed_at"] = stamp
            row["change_log"] = json.dumps(history, ensure_ascii=False)
            _write_csv(path, rows)
            return
    raise KeyError(f"Segment {segment_id} was not found in {path.name}")


def split_segment(
    video_id: str,
    segment_id: str,
    split_at: float,
    reviewer: str,
    metadata_dir: Path = METADATA_DIR,
    snrs: Optional[Sequence[Optional[float]]] = None,
) -> List[str]:
    """Replace one active row with two manually timed child rows."""
    path = csv_path(video_id, metadata_dir)
    rows = _read_csv(path)
    row = next((r for r in rows if r.get("segment_id") == segment_id), None)
    if row is None:
        raise KeyError(f"Segment {segment_id} was not found in {path.name}")
    start, end = float(row["start"]), float(row["end"])
    split_at = float(split_at)
    if not start < split_at < end:
        raise ValueError("Split point must be inside the segment.")
    reviewer = clean_text(reviewer)
    if not reviewer:
        raise ValueError("Enter your name before changing segment structure.")
    child_snrs = list(snrs or [None, None])
    if len(child_snrs) != 2:
        raise ValueError("A split needs one SNR value for each new segment.")
    label, quality = row.get("label") or "speech", row.get("quality") or ""
    delete_segment(video_id, segment_id, reviewer, metadata_dir)
    rows = _read_csv(path)
    first = _append_manual_row(video_id, rows, start, split_at, label, quality, reviewer, f"split_from:{segment_id}", child_snrs[0])
    second = _append_manual_row(video_id, rows, split_at, end, label, quality, reviewer, f"split_from:{segment_id}", child_snrs[1])
    _write_csv(path, rows)
    return [first, second]


def merge_segments(
    video_id: str,
    segment_ids: Sequence[str],
    reviewer: str,
    metadata_dir: Path = METADATA_DIR,
    snr: Optional[float] = None,
) -> str:
    """Replace two selected active segments with one manually merged row."""
    if len(segment_ids) != 2 or segment_ids[0] == segment_ids[1]:
        raise ValueError("Choose exactly two different segments to merge.")
    path = csv_path(video_id, metadata_dir)
    rows = _read_csv(path)
    chosen = [next((r for r in rows if r.get("segment_id") == sid), None) for sid in segment_ids]
    if any(row is None for row in chosen):
        raise KeyError("One of the selected segments is missing from the CSV.")
    if any(row.get("status") == "deleted" for row in chosen):
        raise ValueError("Deleted segments cannot be merged.")
    reviewer = clean_text(reviewer)
    if not reviewer:
        raise ValueError("Enter your name before changing segment structure.")
    ordered = sorted(chosen, key=lambda r: float(r["start"]))
    start = float(ordered[0]["start"])
    end = max(float(row["end"]) for row in ordered)
    label = ordered[0].get("label") or "speech"
    quality = ordered[0].get("quality") or ""
    for row in ordered:
        delete_segment(video_id, row["segment_id"], reviewer, metadata_dir)
    rows = _read_csv(path)
    sid = _append_manual_row(video_id, rows, start, end, label, quality, reviewer,
                             f"merge:{segment_ids[0]}+{segment_ids[1]}", snr)
    _write_csv(path, rows)
    return sid


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

        if status_of(decisions.get(clip["segment_id"])) == "deleted":
            continue

        if labels is not None and triage_label not in labels:
            continue

        if statuses is not None:
            current_status = status_of(decisions.get(clip["segment_id"]))
            if current_status not in statuses:
                continue

        selected.append(clip)

    if order == "recording":
        return sorted(
            selected,
            key=lambda c: (
                c.get("start_sec") if c.get("start_sec") is not None else float("inf"),
                c["segment_id"],
            ),
        )

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
