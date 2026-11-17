"""
src/transcript_store.py
=======================
JSON-Based Transcript Storage & Audit Trail

Replaces the database with transparent, human-readable JSON files stored in:
  data/transcripts/<video_id>_transcripts.json

Each JSON file contains a list of segment objects with transcripts, metrics,
ground truth, and an embedded audit trail of human edits.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"


def _ensure_dir():
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


def get_transcript_file(video_id: str) -> Path:
    """Returns the JSON file path for a specific video's transcripts."""
    _ensure_dir()
    return TRANSCRIPTS_DIR / f"{video_id}_transcripts.json"


def save_transcripts(video_id: str, segments: List[Dict[str, Any]]) -> Path:
    """Saves or updates a list of segment records into <video_id>_transcripts.json."""
    _ensure_dir()
    json_path = get_transcript_file(video_id)

    # Merge with existing file if it already exists to preserve manual ground truth & audit trail
    existing_map = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for s in json.load(f):
                    existing_map[s["segment_id"]] = s
        except Exception:
            pass

    merged_list = []
    for s in segments:
        seg_id = s["segment_id"]
        if seg_id in existing_map:
            old = existing_map[seg_id]
            # Preserve user verified ground truth and audit trail
            s["ground_truth"] = old.get("ground_truth") or s.get("ground_truth") or s.get("transcript_a") or ""
            s["status"] = old.get("status") or s.get("status", "unreviewed")
            s["audit_trail"] = old.get("audit_trail", [])
        else:
            s.setdefault("ground_truth", s.get("transcript_a", ""))
            s.setdefault("status", "unreviewed")
            s.setdefault("audit_trail", [])

        merged_list.append(s)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)

    print(f"[transcript_store] Saved {len(merged_list)} segments to: {json_path.name}")
    return json_path


def load_transcripts(video_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads segments from a specific video's JSON or across all JSON files."""
    _ensure_dir()
    all_segments = []

    if video_id:
        target_file = get_transcript_file(video_id)
        if target_file.exists():
            with open(target_file, "r", encoding="utf-8") as f:
                all_segments = json.load(f)
    else:
        for f in TRANSCRIPTS_DIR.glob("*_transcripts.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    all_segments.extend(json.load(jf))
            except Exception:
                pass

    return all_segments


def upsert_segment(rec: Dict[str, Any]):
    """Upserts a single segment into its respective video's JSON file."""
    video_id = rec.get("video_id")
    if not video_id:
        seg_id = rec.get("segment_id", "")
        video_id = seg_id.split("_seg_")[0] if "_seg_" in seg_id else "unknown"

    json_path = get_transcript_file(video_id)
    existing_segments = []
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_segments = json.load(f)
        except Exception:
            existing_segments = []

    found = False
    for i, s in enumerate(existing_segments):
        if s["segment_id"] == rec["segment_id"]:
            # Update fields
            s.update({k: v for k, v in rec.items() if v is not None})
            found = True
            break

    if not found:
        rec.setdefault("ground_truth", rec.get("transcript_a", ""))
        rec.setdefault("status", "unreviewed")
        rec.setdefault("audit_trail", [])
        existing_segments.append(rec)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_segments, f, indent=2, ensure_ascii=False)


def update_ground_truth(
    segment_id: str,
    new_ground_truth: str,
    status: str = "verified",
    changed_by: str = "annotator"
):
    """
    Updates ground_truth text and status for a segment,
    appending the modification to the segment's embedded audit_trail in its JSON file.
    """
    _ensure_dir()
    now = datetime.now(timezone.utc).isoformat()
    updated = False

    for json_file in TRANSCRIPTS_DIR.glob("*_transcripts.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            segments = json.load(f)

        for s in segments:
            if s["segment_id"] == segment_id:
                old_gt = s.get("ground_truth", "")
                old_status = s.get("status", "")

                s["ground_truth"] = new_ground_truth
                s["status"] = status
                s["reviewed_by"] = changed_by
                s["updated_at"] = now

                # Initialize audit_trail if missing
                if "audit_trail" not in s or not isinstance(s["audit_trail"], list):
                    s["audit_trail"] = []

                # Append audit record if text or status changed
                if old_gt != new_ground_truth:
                    s["audit_trail"].append({
                        "field": "ground_truth",
                        "old_value": old_gt,
                        "new_value": new_ground_truth,
                        "changed_by": changed_by,
                        "timestamp": now
                    })
                if old_status != status:
                    s["audit_trail"].append({
                        "field": "status",
                        "old_value": old_status,
                        "new_value": status,
                        "changed_by": changed_by,
                        "timestamp": now
                    })

                updated = True
                break

        if updated:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(segments, f, indent=2, ensure_ascii=False)
            return

    if not updated:
        raise KeyError(f"Segment {segment_id} not found in any transcript JSON file.")


def get_all_segments(
    video_id: Optional[str] = None,
    triage_filter: Optional[str] = None,
    status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Loads and filters segments from the JSON files."""
    segments = load_transcripts(video_id=video_id)

    filtered = []
    for s in segments:
        if triage_filter and s.get("triage") != triage_filter:
            continue
        if status_filter and s.get("status") != status_filter:
            continue
        filtered.append(s)

    return filtered


def get_audit_trail(segment_id: str) -> List[Dict[str, Any]]:
    """Returns the chronological audit trail list for a segment from JSON."""
    for s in load_transcripts():
        if s["segment_id"] == segment_id:
            return s.get("audit_trail", [])
    return []


def get_corpus_statistics() -> Dict[str, Any]:
    """Computes summary statistics directly from all transcript JSON files."""
    segments = load_transcripts()
    total = len(segments)

    if total == 0:
        return {
            "total_segments": 0,
            "total_duration_sec": 0.0,
            "status_breakdown": {},
            "triage_breakdown": {},
            "avg_wer": 0.0,
            "avg_cer": 0.0,
            "avg_snr_db": 0.0,
        }

    total_duration = sum(s.get("duration_sec", 0.0) or 0.0 for s in segments)

    status_counts = {}
    triage_counts = {}
    wer_vals = []
    cer_vals = []
    snr_vals = []

    for s in segments:
        st = s.get("status", "unreviewed")
        status_counts[st] = status_counts.get(st, 0) + 1

        tr = s.get("triage", "unreviewed")
        triage_counts[tr] = triage_counts.get(tr, 0) + 1

        if s.get("wer") is not None:
            wer_vals.append(s["wer"])
        if s.get("cer") is not None:
            cer_vals.append(s["cer"])
        if s.get("snr_wada_db") is not None:
            snr_vals.append(s["snr_wada_db"])

    return {
        "total_segments": total,
        "total_duration_sec": round(total_duration, 2),
        "status_breakdown": status_counts,
        "triage_breakdown": triage_counts,
        "avg_wer": round(sum(wer_vals) / len(wer_vals), 3) if wer_vals else 0.0,
        "avg_cer": round(sum(cer_vals) / len(cer_vals), 3) if cer_vals else 0.0,
        "avg_snr_db": round(sum(snr_vals) / len(snr_vals), 1) if snr_vals else 0.0,
    }
