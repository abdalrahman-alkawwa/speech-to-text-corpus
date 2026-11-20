"""
src/local_ingest.py - Stage 1 for audio/video files that are already on your laptop.

Counterpart of ingest.ingest_video(url): instead of downloading from YouTube it
takes a local file, then runs the same Stage 2 extraction and validation, and
writes the same kind of metadata record.  Everything after that (classifier,
segmenter, SNR, transcriber) is unchanged because it only needs a video_id and
data/processed/<name>.wav.

What it does
------------
1. Checks the file exists, is non-trivial in size and has a supported extension.
2. video_id = "local_" + first 12 hex characters of the file's SHA-256.
   * stable: the same file always gets the same id (re-uploading reuses the work)
   * ASCII-only and filename-independent: safe for Windows, FFmpeg, Arabic names
3. COPIES the file to data/raw/<safe_name>_<video_id><ext>.  Your original is never
   moved, modified or deleted.
4. Extracts 16 kHz mono PCM_16 WAV to data/processed/ and validates it.
5. Writes metadata/<video_id>.json (UTF-8, Arabic-safe).  Same fields as the URL
   flow, plus source_type, original_filename and sha256; source_url is None.

If extraction fails (for example a corrupt file renamed to .mp3), the raw copy made
by this call is removed and no metadata file is written.

Usage (PowerShell, repository root):
    python src/local_ingest.py "C:\\Users\\me\\Desktop\\lecture 1.mp3"
    python src/local_ingest.py "D:\\audio\\interview.m4a" "recorded with permission"
"""

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_extractor import extract_audio, validate_audio

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"

# Anything FFmpeg can decode is fine; this list just rejects obvious non-audio files.
ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".webm", ".mp4", ".mkv", ".mov", ".avi",
}
MIN_FILE_BYTES = 1000


def sanitize_stem(stem: str, max_length: int = 50) -> str:
    """File-name-safe version of a title: keeps Arabic/Latin letters and digits,
    drops emoji and symbols (same rule as ingest.sanitize_title)."""
    cleaned = re.sub(r"[^\w\s-]", "_", stem or "")
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("._ -")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._ -")
    return cleaned or "audio"


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """SHA-256 of a file, read in 1 MB chunks (safe for multi-GB files)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    """Path relative to the project root, with forward slashes.

    Both sides are resolved first: on Windows the same folder can be spelled two ways
    (long "C:\\Users\\Abdalrahman Khaled\\..." and short "C:\\Users\\ABDALR~1\\..."),
    and FFmpeg-facing code resolves paths while other code may not.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _find_existing(directory: Path, video_id: str, extensions) -> Optional[Path]:
    for candidate in sorted(directory.glob(f"*{video_id}*")):
        if candidate.suffix.lower() in extensions and candidate.stat().st_size > MIN_FILE_BYTES:
            return candidate
    return None


def ingest_local_file(
    path,
    notes: Optional[str] = None,
    custom_license: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a local audio/video file and prepare its 16 kHz mono WAV.

    Returns the metadata dict that was saved to metadata/<video_id>.json.
    Raises FileNotFoundError, ValueError (bad/unsupported/empty file) or RuntimeError
    (FFmpeg could not decode it).
    """
    source = Path(str(path).strip().strip('"')).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    if not source.is_file():
        raise ValueError(f"Not a file: {source}")
    extension = source.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    size_bytes = source.stat().st_size
    if size_bytes < MIN_FILE_BYTES:
        raise ValueError(f"{source.name} is empty or too small to be audio ({size_bytes} bytes).")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    digest = file_sha256(source)
    video_id = f"local_{digest[:12]}"
    file_stem = f"{sanitize_stem(source.stem)}_{video_id}"
    print(f"\n[local_ingest] {source.name} -> video_id {video_id} ({size_bytes / (1024 * 1024):.1f} MB)")

    # --- raw copy (never modify the original) ---
    created_raw = False
    if source.parent == RAW_DIR.resolve():
        raw_path = source                               # already in data/raw/: use in place
    else:
        raw_path = _find_existing(RAW_DIR, video_id, ALLOWED_EXTENSIONS)
        if raw_path:
            print(f"[local_ingest] Reusing existing raw copy: {raw_path.name}")
        else:
            raw_path = RAW_DIR / f"{file_stem}{extension}"
            shutil.copy2(source, raw_path)
            created_raw = True
            print(f"[local_ingest] Copied to {_rel(raw_path)}")

    # --- Stage 2: 16 kHz mono PCM_16 WAV ---
    existing_wav = _find_existing(PROCESSED_DIR, video_id, {".wav"})
    processed_path = existing_wav if existing_wav else PROCESSED_DIR / f"{file_stem}.wav"
    try:
        extracted = extract_audio(
            input_path=raw_path, output_path=processed_path,
            sample_rate=16000, channels=1, overwrite=False,
        )
        audio = validate_audio(extracted)
    except Exception:
        if created_raw:
            raw_path.unlink(missing_ok=True)            # leave no junk behind
        raise
    print(f"[local_ingest] Audio validated: {audio['duration_seconds']}s, 16kHz, mono, PCM_16")

    record = {
        "video_id": video_id,
        "source_type": "local_upload",
        "source_url": None,
        "original_filename": source.name,
        "sha256": digest,
        "title": title or source.stem,
        "channel": None,
        "channel_id": None,
        "upload_date": None,
        "duration_seconds": audio["duration_seconds"],
        "description": None,
        "language_reported": "ar",
        "license_reported": custom_license or "NA",
        "notes": notes,
        "ingestion_info": {
            "tool": "local_ingest",
            "tool_version": None,
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_file": _rel(raw_path),
            "raw_size_bytes": raw_path.stat().st_size,
        },
        "audio_properties": {
            "processed_file": _rel(extracted),
            "sample_rate": audio["samplerate"],
            "channels": audio["channels"],
            "subtype": audio["subtype"],
            "format": audio["format"],
            "duration_seconds": audio["duration_seconds"],
            "total_frames": audio["frames"],
        },
    }
    metadata_path = METADATA_DIR / f"{video_id}.json"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    print(f"[local_ingest] Metadata recorded to: {metadata_path.name}")
    return record


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/local_ingest.py "<path to audio or video file>" [notes]')
        sys.exit(1)
    ingest_local_file(sys.argv[1], notes=sys.argv[2] if len(sys.argv) > 2 else None)
