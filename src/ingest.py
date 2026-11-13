"""
src/ingest.py
Stage 1: Video Ingestion and Metadata Recording

Downloads online audio/video sources using yt-dlp, saves comprehensive
metadata JSON records for corpus reproducibility, and automatically triggers
Stage 2 audio extraction and validation.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'from src.xxx import ...' works reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import yt_dlp

from src.audio_extractor import extract_audio, validate_audio


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "metadata"


def get_ytdlp_version() -> str:
    """Returns the installed yt-dlp version."""
    return getattr(yt_dlp.version, "__version__", "unknown")


def fetch_video_metadata(url: str) -> Dict[str, Any]:
    """
    Extracts complete metadata dictionary from video URL without downloading.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return ydl.sanitize_info(info)


def find_existing_raw_file(video_id: str) -> Optional[Path]:
    """
    Checks if raw media for this video ID already exists on disk under any extension.
    Ignores incomplete .part / .ytdl files.
    """
    valid_exts = {".m4a", ".webm", ".mp4", ".mkv", ".wav", ".opus", ".mp3"}
    for candidate in RAW_DIR.glob(f"{video_id}.*"):
        if candidate.suffix.lower() in valid_exts and candidate.stat().st_size > 1000:
            return candidate
    return None


def ingest_video(
    url: str,
    notes: Optional[str] = None,
    custom_license: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main ingestion pipeline:
    1. Fetches metadata and identifies video ID.
    2. Uses existing raw media if present; otherwise downloads best audio stream.
    3. Triggers Stage 2 audio extraction to data/processed/<video_id>.wav.
    4. Validates extracted WAV with SoundFile.
    5. Saves full metadata record to metadata/<video_id>.json.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[ingest] Fetching metadata for: {url}")
    info = fetch_video_metadata(url)
    video_id = info.get("id")
    if not video_id:
        raise ValueError(f"Could not determine video ID from URL: {url}")

    # Check if we already have raw media on disk for this video
    raw_path = find_existing_raw_file(video_id)

    if raw_path:
        print(f"[ingest] Found existing raw media on disk: {raw_path.name} ({raw_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"[ingest] No local file found. Downloading raw audio for video ID: {video_id}")
        ydl_download_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(RAW_DIR / f"{video_id}.%(ext)s"),
            "quiet": False,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
            ydl.download([url])

        raw_path = find_existing_raw_file(video_id)
        if not raw_path:
            raise FileNotFoundError(f"Failed to find downloaded file for video ID: {video_id}")

    # Stage 2: Extract standardized 16 kHz mono 16-bit PCM WAV
    processed_wav_path = PROCESSED_DIR / f"{video_id}.wav"
    extracted_wav = extract_audio(
        input_path=raw_path,
        output_path=processed_wav_path,
        sample_rate=16000,
        channels=1,
        overwrite=False
    )

    # Validate the extracted audio
    audio_validation = validate_audio(extracted_wav)
    print(f"[ingest] Audio validated: {audio_validation['duration_seconds']}s, 16kHz, mono, PCM_16")

    # Construct the formal metadata record
    metadata_record = {
        "video_id": video_id,
        "source_url": url,
        "title": info.get("title"),
        "channel": info.get("uploader") or info.get("channel"),
        "channel_id": info.get("channel_id") or info.get("uploader_id"),
        "upload_date": info.get("upload_date"),
        "duration_seconds": info.get("duration"),
        "description": info.get("description", "")[:500] if info.get("description") else None,
        "language_reported": info.get("language") or "ar",
        "license_reported": custom_license or info.get("license") or "NA",
        "notes": notes,
        "ingestion_info": {
            "tool": "yt-dlp",
            "tool_version": get_ytdlp_version(),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_file": str(raw_path.relative_to(PROJECT_ROOT)),
            "raw_size_bytes": raw_path.stat().st_size,
        },
        "audio_properties": {
            "processed_file": str(extracted_wav.relative_to(PROJECT_ROOT)),
            "sample_rate": audio_validation["samplerate"],
            "channels": audio_validation["channels"],
            "subtype": audio_validation["subtype"],
            "format": audio_validation["format"],
            "duration_seconds": audio_validation["duration_seconds"],
            "total_frames": audio_validation["frames"]
        }
    }

    # Save metadata JSON
    metadata_json_path = METADATA_DIR / f"{video_id}.json"
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_record, f, indent=2, ensure_ascii=False)

    print(f"[ingest] Metadata successfully recorded to: {metadata_json_path.name}")
    return metadata_record


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <VIDEO_URL> [OPTIONAL_NOTES]")
    else:
        url_arg = sys.argv[1]
        notes_arg = sys.argv[2] if len(sys.argv) > 2 else None
        ingest_video(url_arg, notes=notes_arg)
