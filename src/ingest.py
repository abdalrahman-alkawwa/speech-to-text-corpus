"""
src/ingest.py
=============
Stage 1: Video Ingestion and Metadata Recording

Purpose:
--------
In speech corpus creation, "provenance" (the recorded history of who created
the content, where it was acquired, under what license, and using what tools)
is as crucial as the audio data itself.

This module provides the automated ingestion pipeline:
  1. Accepts an online video URL (e.g. YouTube).
  2. Queries the YouTube API via yt-dlp to extract metadata without heavy downloading.
  3. Checks data/raw/ to see if the file already exists on disk (preventing redundant downloads).
  4. If not present, downloads the highest-quality audio stream to data/raw/<video_id>.<ext>.
  5. Automatically triggers Stage 2 (audio_extractor.py) to produce a standardized 16 kHz WAV.
  6. Validates the extracted WAV with SoundFile.
  7. Writes a permanent, reproducible metadata record to metadata/<video_id>.json.
"""


import sys


from pathlib import Path

# --- Python Path Configuration ---
# Resolve the project root folder (C:\Speech_2_Text\speech-to-text-corpus)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# If the project root is not in sys.path, insert it at index 0
# This guarantees that 'from src.xxx import ...' works whether the script is invoked
# directly (python src/ingest.py) or as a module (python -m src.ingest).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# json: Standard library for reading and writing structured JSON metadata files
import json
import re

# datetime, timezone: For generating standard ISO 8601 UTC timestamps of when downloads occurred
from datetime import datetime, timezone

# Typing hints for clarity and IDE auto-completion
from typing import Dict, Any, Optional

# yt_dlp: Open-source command-line and Python library for downloading video/audio from YouTube
import yt_dlp

# Import our Stage 2 extraction and validation functions directly
from src.audio_extractor import extract_audio, validate_audio


# --- Standard Directory Definitions ---
# data/raw/: Stores original, untouched media files downloaded from the web
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# data/processed/: Stores standardized, converted 16 kHz mono 16-bit PCM WAV files
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# metadata/: Stores reproducible JSON provenance records for every recording
METADATA_DIR = PROJECT_ROOT / "metadata"


def get_ytdlp_version() -> str:
    """
    Returns the currently installed yt-dlp version as a string.
    Recording the tool version in metadata is required for scientific reproducibility.
    """
    return yt_dlp.version.__version__


def fetch_video_metadata(url: str) -> Dict[str, Any]:
    """
    Extracts complete metadata dictionary from a video URL without downloading the media.

    Parameters:
    -----------
    url : str
        The full YouTube video URL.

    Returns:
    --------
    dict : Sanitized metadata dictionary containing title, uploader, duration, etc.
    """
    # Options passed to yt-dlp:
    # - quiet: suppress command-line noise
    # - no_warnings: suppress non-critical warnings
    
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
      
    }

    # Initialize yt-dlp with our options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # extract_info queries the platform API and returns raw metadata
        info = ydl.extract_info(url, download=False)
        # sanitize_info removes un-serializable objects (e.g. callables, file handles)
        return ydl.sanitize_info(info)


def sanitize_title(title: str, max_length: int = 50) -> str:
    """
    Sanitizes a video title so it can be safely used in Windows/Linux file paths.
    Replaces illegal characters (\\ / : * ? " < > |) and spaces with clean underscores.
    Preserves Arabic and alphanumeric text, truncating to max_length to avoid Windows PATH limits.
    """
    if not title:
        return "audio"
    # Replace characters not allowed in Windows filenames with an underscore
    cleaned = re.sub(r'[\\/*?:"<>|#%&{}\\<>*?/$!\'":@+`|=]', "_", title)
    # Collapse multiple consecutive underscores or whitespace into a single underscore
    cleaned = re.sub(r"[\s_]+", "_", cleaned).strip("._ ")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._ ")
    return cleaned or "audio"


def find_existing_raw_file(video_id: str) -> Optional[Path]:
    """
    Checks if a raw media file matching this video_id already exists in data/raw/.
    Uses glob matching '*{video_id}*' so it supports both pure ID names
    (e.g. gK017cCH36k.m4a) and title+ID names (e.g. DSP_Lecture_1_gK017cCH36k.m4a).
    """
    valid_exts = {".m4a", ".webm", ".mp4", ".mkv", ".wav", ".opus", ".mp3"}

    # Search for any file containing the video_id in data/raw/
    for candidate in RAW_DIR.glob(f"*{video_id}*"):
        if candidate.suffix.lower() in valid_exts and candidate.stat().st_size > 1000:
            return candidate
    return None


def find_existing_processed_wav(video_id: str) -> Optional[Path]:
    """
    Checks if a processed 16 kHz WAV file matching this video_id already exists in data/processed/.
    """
    for candidate in PROCESSED_DIR.glob(f"*{video_id}*.wav"):
        if candidate.stat().st_size > 1000:
            return candidate
    return None


def ingest_video(
    url: str,
    notes: Optional[str] = None,
    custom_license: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the end-to-end ingestion and registration pipeline:
      1. Fetches metadata and parses the unique video ID.
      2. Detects local media in data/raw/ or downloads best available audio.
      3. Invokes Stage 2 audio extraction to create data/processed/<video_id>.wav.
      4. Programmatically validates the extracted audio.
      5. Constructs and writes metadata/<video_id>.json.

    Parameters:
    -----------
    url : str
        The video URL to process.
    notes : str, optional
        Custom context note (e.g. lecturer permission, academic recording details).
    custom_license : str, optional
        Explicit license override if the YouTube API reports 'NA'.

    Returns:
    --------
    dict : The complete saved metadata dictionary.
    """
    # Ensure the destination directories exist on disk
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[ingest] Fetching metadata for: {url}")
    info = fetch_video_metadata(url)
    video_id = info.get("id")
    title = info.get("title", "")

    # Safety check: every online video must have a unique identifier
    if not video_id:
        raise ValueError(f"Could not determine video ID from URL: {url}")

    # Build human-readable file stem: <Sanitized_Title>_<video_id>
    # e.g., DSP_Elective_2_Lecture_1_gK017cCH36k
    safe_title = sanitize_title(title)
    file_stem = f"{safe_title}_{video_id}"

    # Check if we already have the raw recording saved on disk
    raw_path = find_existing_raw_file(video_id)

    if raw_path:
        # File is already on disk: reuse it to avoid bandwidth and throttling
        print(f"[ingest] Found existing raw media on disk: {raw_path.name} ({raw_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        # File is not on disk: download best available audio stream
        print(f"[ingest] No local file found. Downloading raw audio: {file_stem}")
        ydl_download_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(RAW_DIR / f"{file_stem}.%(ext)s"),
            "quiet": False,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
            ydl.download([url])

        # Verify that the download successfully produced a valid file
        raw_path = find_existing_raw_file(video_id)
        if not raw_path:
            raise FileNotFoundError(f"Failed to find downloaded file for video ID: {video_id}")

    # --- Stage 2 Execution: Audio Extraction ---
    # Check if a 16 kHz WAV already exists for this video (matches either old ID or new title+ID)
    existing_wav = find_existing_processed_wav(video_id)
    processed_wav_path = existing_wav if existing_wav else (PROCESSED_DIR / f"{file_stem}.wav")

    # Extract 16 kHz mono 16-bit PCM WAV using our audio_extractor module
    extracted_wav = extract_audio(
        input_path=raw_path,
        output_path=processed_wav_path,
        sample_rate=16000,
        channels=1,
        overwrite=False
    )

    # Programmatically validate the extracted audio file using SoundFile
    audio_validation = validate_audio(extracted_wav)
    print(f"[ingest] Audio validated: {audio_validation['duration_seconds']}s, 16kHz, mono, PCM_16")

    # --- Stage 1 Output: Construct the Formal Metadata Record ---
    metadata_record = {
        "video_id": video_id,
        "source_url": url,
        "title": info.get("title"),
        "channel": info.get("uploader") or info.get("channel"),
        "channel_id": info.get("channel_id") or info.get("uploader_id"),
        "upload_date": info.get("upload_date"),
        "duration_seconds": info.get("duration"),
        # Preserve first 500 characters of description if available
        "description": info.get("description", "")[:500] if info.get("description") else None,
        "language_reported": info.get("language") or "ar",
        "license_reported": custom_license or info.get("license") or "NA",
        "notes": notes,
        "ingestion_info": {
            "tool": "yt-dlp",
            "tool_version": get_ytdlp_version(),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
            # Record relative paths so the metadata remains portable across machines
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

    # Write the formatted metadata to metadata/<video_id>.json
    metadata_json_path = METADATA_DIR / f"{video_id}.json"
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        # indent=2 formats with clean indentation; ensure_ascii=False preserves Arabic text
        json.dump(metadata_record, f, indent=2, ensure_ascii=False)

    print(f"[ingest] Metadata successfully recorded to: {metadata_json_path.name}")
    return metadata_record


# --- Command-Line Entry Point ---
if __name__ == "__main__":
    # If run from the command line without arguments, display usage instructions
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py <VIDEO_URL> [OPTIONAL_NOTES]")
    else:
        # First command-line argument: Video URL
        url_arg = sys.argv[1]
        # Optional second command-line argument: Notes
        notes_arg = sys.argv[2] if len(sys.argv) > 2 else None
        # Run the ingestion pipeline
        ingest_video(url_arg, notes=notes_arg)
