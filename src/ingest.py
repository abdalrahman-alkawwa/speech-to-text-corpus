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
        "noplaylist": True,
        "no_warnings": True,
      
    }

    # Initialize yt-dlp with our options
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # extract_info queries the platform API and returns raw metadata
        info = ydl.extract_info(url, download=False)
        # sanitize_info removes un-serializable objects (e.g. callables, file handles)
        return ydl.sanitize_info(info)


def sanitize_title(title: str, max_length: int = 50) -> str:
    if not title:
        return "audio"
    # Replace characters not allowed in Windows filenames with an underscore
    cleaned = re.sub(r'[\\/*?:"<>|#%&{}\\<>*?/$!\'":@+`|=]', "_", title)
    # Drop emoji/symbols; \w keeps Arabic and Latin letters and digits.
    cleaned = re.sub(r"[^\w\s-]", "_", cleaned)  # drop emoji/symbols
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

def validate_directs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    return None


def download_audio(url: str, stem: str, id: str) -> Path:

    raw_path = find_existing_raw_file(id)

    if raw_path:
        print(
            f"Found existing raw media on disk: {raw_path.name} "
            f"({raw_path.stat().st_size / (1024 * 1024):.1f} MB)"
        )

    else:
        print(f"No local file found. Downloading raw audio: {stem}")

        ydl_download_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(RAW_DIR / f"{stem}.%(ext)s"),
            "quiet": False,
            "noplaylist": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
            ydl.download([url])

        raw_path = find_existing_raw_file(id)

        if not raw_path:
            raise FileNotFoundError(
                f"Failed to find downloaded file for video ID: {id}"
            )

    return raw_path



# ============================================================
# INGESTION HELPER FUNCTIONS
# ============================================================

def get_video_id(info: Dict[str, Any], url: str) -> str:
    

    video_id = info.get("id")

    if not video_id:
        raise ValueError(
            f"Could not determine video ID from URL: {url}"
        )

    return video_id





def prepare_processed_audio(
    raw_path: Path,
    video_id: str,
    file_stem: str
) -> Path:
    """
    Find an existing processed WAV or generate a new one.

    The output is standardized to:
        - 16 kHz sample rate
        - Mono (1 channel)
        - 16-bit PCM

    Reuses an existing WAV to avoid unnecessary processing.
    """

    # Check whether this video already has a processed WAV.
    existing_wav = find_existing_processed_wav(video_id)

    if existing_wav:
        print(
            f"[ingest] Reusing processed audio: "
            f"{existing_wav.name}"
        )
        return existing_wav

    # Define where the newly extracted WAV will be stored.
    output_path = PROCESSED_DIR / f"{file_stem}.wav"

    print("[ingest] Extracting standardized audio...")

    # Convert the original media into our standardized WAV.
    return extract_audio(
        input_path=raw_path,
        output_path=output_path,
        sample_rate=16000,
        channels=1,
        overwrite=False
    )




def build_metadata_record(
    url: str,
    info: Dict[str, Any],
    video_id: str,
    notes: Optional[str],
    custom_license: Optional[str],
    raw_path: Path,
    extracted_wav: Path,
    audio_validation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Construct the complete metadata dictionary.

    Combines:
        - Original video information from yt-dlp
        - Raw media file information
        - Processed audio properties
        - Ingestion history and timestamps

    Does not write anything to disk.
    """

    metadata_record = {
        # Basic source identification.
        "video_id": video_id,
        "source_url": url,
        "title": info.get("title"),

        # Information about the original uploader.
        "channel": info.get("uploader") or info.get("channel"),
        "channel_id": info.get("channel_id") or info.get("uploader_id"),
        "upload_date": info.get("upload_date"),

        # Original video properties.
        "duration_seconds": info.get("duration"),
        "description": (
            info.get("description", "")[:500]
            if info.get("description") else None
        ),

        # Language and licensing information.
        "language_reported": info.get("language") or "ar",
        "license_reported": (
            custom_license or info.get("license") or "NA"
        ),

        # User-provided notes and permissions.
        "notes": notes,

        # Record how and when the original media was acquired.
        "ingestion_info": {
            "tool": "yt-dlp",
            "tool_version": get_ytdlp_version(),
            "ingested_at_utc": datetime.now(timezone.utc).isoformat(),

            # Store relative paths to keep metadata portable.
            "raw_file": raw_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),

            "raw_size_bytes": raw_path.stat().st_size,
        },

        # Record the properties of the standardized WAV.
        "audio_properties": {
            "processed_file": extracted_wav.relative_to(
                PROJECT_ROOT
            ).as_posix(),

            "sample_rate": audio_validation["samplerate"],
            "channels": audio_validation["channels"],
            "subtype": audio_validation["subtype"],
            "format": audio_validation["format"],
            "duration_seconds": audio_validation["duration_seconds"],
            "total_frames": audio_validation["frames"],
        }
    }

    return metadata_record


def save_metadata(
    metadata_record: Dict[str, Any]
) -> Path:
    """
    Save the metadata dictionary as a JSON file.

    Each video gets its own metadata file named after
    its unique video ID.

    Returns the path of the saved JSON file.
    """

    video_id = metadata_record["video_id"]
    

    metadata_json_path = METADATA_DIR / f"{video_id}.json"

    # Write JSON using UTF-8 to preserve Arabic characters.
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata_record,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"[ingest] Metadata successfully recorded to: "
        f"{metadata_json_path.name}"
    )

    return metadata_json_path


# ============================================================
# MAIN INGESTION ORCHESTRATOR
# ============================================================

def ingest_video(
    url: str,
    notes: Optional[str] = None,
    custom_license: Optional[str] = None
) -> Dict[str, Any]:
    """
    Coordinate the complete video ingestion workflow.

    This function does not implement the individual operations.
    Instead, it calls specialized helper functions in the
    correct order and passes data between them.

    Workflow:
        1. Prepare project directories.
        2. Fetch video metadata.
        3. Identify the video.
        4. Find or download original media.
        5. Find or extract standardized audio.
        6. Validate processed audio.
        7. Build metadata record.
        8. Save metadata record.

    Returns the complete metadata dictionary.
    """

    # --------------------------------------------------------
    # STEP 1: Ensure all required project directories exist.
    # --------------------------------------------------------

    validate_directs()

    # --------------------------------------------------------
    # STEP 2: Retrieve video metadata without downloading it.
    # --------------------------------------------------------

    print(f"\n[ingest] Fetching metadata for: {url}")

    info = fetch_video_metadata(url)

    # --------------------------------------------------------
    # STEP 3: Extract the video ID and prepare its file name.
    # --------------------------------------------------------

    video_id = get_video_id(info, url)

    safe_title = sanitize_title(info.get("title", ""))

    file_stem = f"{safe_title}_{video_id}"

    # --------------------------------------------------------
    # STEP 4: Acquire the original raw media file.
    # Reuses an existing file or downloads it if necessary.
    # --------------------------------------------------------

    raw_path = download_audio(
    url=url,
    stem=file_stem,
    id=video_id
)

    # --------------------------------------------------------
    # STEP 5: Prepare the standardized 16 kHz WAV file.
    # Reuses an existing processed file when available.
    # --------------------------------------------------------

    extracted_wav = prepare_processed_audio(
        raw_path=raw_path,
        video_id=video_id,
        file_stem=file_stem
    )

    # --------------------------------------------------------
    # STEP 6: Validate the processed audio properties.
    # --------------------------------------------------------
    audio_validation = validate_audio(extracted_wav)
    print(
    f"[ingest] Audio validated: "
    f"{audio_validation['duration_seconds']}s, "
    f"16kHz, mono, PCM_16"
)

    # --------------------------------------------------------
    # STEP 7: Construct the complete metadata dictionary.
    # --------------------------------------------------------

    metadata_record = build_metadata_record(
        url=url,
        info=info,
        video_id=video_id,
        notes=notes,
        custom_license=custom_license,
        raw_path=raw_path,
        extracted_wav=extracted_wav,
        audio_validation=audio_validation
    )

    # --------------------------------------------------------
    # STEP 8: Save the metadata record as a JSON file.
    # --------------------------------------------------------

    save_metadata(metadata_record)

    # Return the metadata so other modules can use it.
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
