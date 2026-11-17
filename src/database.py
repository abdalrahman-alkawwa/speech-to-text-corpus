"""
src/database.py
===============
Stage 10: Corpus Annotation Database & Audit Trail

Provides a persistent SQLite storage layer with an append-only audit trail
to track human review, ground truth transcriptions, and triage statuses.
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "data" / "corpus.db"


@contextmanager
def get_db(db_path: Optional[Path] = None):
    """Context manager that yields an active connection and guarantees closing it."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None):
    """Initializes SQLite schema for segments and audit trail."""
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Master segments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                start_sec REAL,
                end_sec REAL,
                duration_sec REAL,
                snr_wada_db REAL,
                snr_segmental_db REAL,
                boundary_check TEXT DEFAULT 'auto_verified',
                transcript_a TEXT,
                engine_a TEXT,
                transcript_b TEXT,
                engine_b TEXT,
                normalized_a TEXT,
                normalized_b TEXT,
                wer REAL,
                cer REAL,
                triage TEXT DEFAULT 'unreviewed',
                ground_truth TEXT,
                status TEXT DEFAULT 'unreviewed',
                reviewed_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)

        # Append-only audit log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_by TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
            );
        """)


def upsert_segment(rec: Dict[str, Any], db_path: Optional[Path] = None):
    """Inserts or updates a segment record from pipeline outputs."""
    now = datetime.now(timezone.utc).isoformat()
    init_db(db_path)

    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO segments (
                segment_id, video_id, audio_path, start_sec, end_sec, duration_sec,
                snr_wada_db, snr_segmental_db, boundary_check, transcript_a, engine_a,
                transcript_b, engine_b, normalized_a, normalized_b, wer, cer,
                triage, ground_truth, status, created_at, updated_at
            ) VALUES (
                :segment_id, :video_id, :audio_path, :start_sec, :end_sec, :duration_sec,
                :snr_wada_db, :snr_segmental_db, :boundary_check, :transcript_a, :engine_a,
                :transcript_b, :engine_b, :normalized_a, :normalized_b, :wer, :cer,
                :triage, :ground_truth, :status, :created_at, :updated_at
            )
            ON CONFLICT(segment_id) DO UPDATE SET
                snr_wada_db = COALESCE(:snr_wada_db, segments.snr_wada_db),
                snr_segmental_db = COALESCE(:snr_segmental_db, segments.snr_segmental_db),
                transcript_a = COALESCE(:transcript_a, segments.transcript_a),
                transcript_b = COALESCE(:transcript_b, segments.transcript_b),
                normalized_a = COALESCE(:normalized_a, segments.normalized_a),
                normalized_b = COALESCE(:normalized_b, segments.normalized_b),
                wer = COALESCE(:wer, segments.wer),
                cer = COALESCE(:cer, segments.cer),
                triage = COALESCE(:triage, segments.triage),
                updated_at = :updated_at;
        """, {
            "segment_id": rec["segment_id"],
            "video_id": rec.get("video_id", ""),
            "audio_path": rec.get("audio_path", ""),
            "start_sec": rec.get("start_sec", 0.0),
            "end_sec": rec.get("end_sec", 0.0),
            "duration_sec": rec.get("duration_sec", 0.0),
            "snr_wada_db": rec.get("snr_wada_db"),
            "snr_segmental_db": rec.get("snr_segmental_db"),
            "boundary_check": rec.get("boundary_check", "auto_verified"),
            "transcript_a": rec.get("transcript_a"),
            "engine_a": rec.get("engine_a", "whisper-large-v3-turbo"),
            "transcript_b": rec.get("transcript_b"),
            "engine_b": rec.get("engine_b", "mms-1b-all"),
            "normalized_a": rec.get("normalized_a"),
            "normalized_b": rec.get("normalized_b"),
            "wer": rec.get("wer"),
            "cer": rec.get("cer"),
            "triage": rec.get("triage", "unreviewed"),
            "ground_truth": rec.get("ground_truth", rec.get("transcript_a")),
            "status": rec.get("status", "unreviewed"),
            "created_at": now,
            "updated_at": now,
        })


def update_ground_truth(
    segment_id: str,
    new_ground_truth: str,
    status: str = "verified",
    changed_by: str = "annotator",
    db_path: Optional[Path] = None
):
    """
    Updates ground_truth text and status for a segment,
    recording the change in the append-only audit log.
    """
    now = datetime.now(timezone.utc).isoformat()
    init_db(db_path)

    with get_db(db_path) as conn:
        cursor = conn.cursor()

        # Fetch old record
        cursor.execute("SELECT ground_truth, status FROM segments WHERE segment_id = ?", (segment_id,))
        row = cursor.fetchone()
        if not row:
            raise KeyError(f"Segment not found: {segment_id}")

        old_gt = row["ground_truth"] or ""
        old_status = row["status"] or ""

        # Update segment
        cursor.execute("""
            UPDATE segments
            SET ground_truth = ?, status = ?, reviewed_by = ?, updated_at = ?
            WHERE segment_id = ?
        """, (new_ground_truth, status, changed_by, now, segment_id))

        # Record in audit log if text changed
        if old_gt != new_ground_truth:
            cursor.execute("""
                INSERT INTO audit_log (segment_id, field_name, old_value, new_value, changed_by, timestamp)
                VALUES (?, 'ground_truth', ?, ?, ?, ?)
            """, (segment_id, old_gt, new_ground_truth, changed_by, now))

        # Record in audit log if status changed
        if old_status != status:
            cursor.execute("""
                INSERT INTO audit_log (segment_id, field_name, old_value, new_value, changed_by, timestamp)
                VALUES (?, 'status', ?, ?, ?, ?)
            """, (segment_id, old_status, status, changed_by, now))


def get_all_segments(
    video_id: Optional[str] = None,
    triage_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Fetches segment rows with optional filtering."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM segments WHERE 1=1"
        params = []

        if video_id:
            query += " AND video_id = ?"
            params.append(video_id)
        if triage_filter:
            query += " AND triage = ?"
            params.append(triage_filter)
        if status_filter:
            query += " AND status = ?"
            params.append(status_filter)

        query += " ORDER BY start_sec ASC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]


def get_audit_trail(segment_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Returns chronological audit log records for a segment."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log WHERE segment_id = ? ORDER BY timestamp DESC",
            (segment_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_corpus_statistics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Returns overview statistics of the corpus dataset."""
    init_db(db_path)
    with get_db(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total, SUM(duration_sec) as total_duration FROM segments")
        overview = cursor.fetchone()

        cursor.execute("SELECT status, COUNT(*) as count FROM segments GROUP BY status")
        status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT triage, COUNT(*) as count FROM segments GROUP BY triage")
        triage_counts = {row["triage"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT AVG(wer) as avg_wer, AVG(cer) as avg_cer, AVG(snr_wada_db) as avg_snr_db FROM segments")
        avgs = cursor.fetchone()

        return {
            "total_segments": overview["total"] or 0,
            "total_duration_sec": round(overview["total_duration"] or 0.0, 2),
            "status_breakdown": status_counts,
            "triage_breakdown": triage_counts,
            "avg_wer": round(avgs["avg_wer"] or 0.0, 3),
            "avg_cer": round(avgs["avg_cer"] or 0.0, 3),
            "avg_snr_db": round(avgs["avg_snr_db"] or 0.0, 1),
        }
