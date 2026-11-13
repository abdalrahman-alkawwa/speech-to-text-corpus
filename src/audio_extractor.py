"""
src/audio_extractor.py
Stage 2: Audio Extraction and Format Validation

Extracts 16 kHz, mono, 16-bit PCM WAV audio from source media using FFmpeg
and validates properties programmatically with SoundFile.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Union
import soundfile as sf


def check_ffmpeg_available() -> str:
    """Verifies that ffmpeg executable is reachable in PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg executable not found in PATH. "
            "Please ensure FFmpeg is installed and added to your environment Path."
        )
    return ffmpeg_path


def extract_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = False
) -> Path:
    """
    Extracts and standardizes audio to 16-bit PCM WAV using FFmpeg.

    Command equivalent:
    ffmpeg -i <input> -vn -ac <channels> -ar <sample_rate> -sample_fmt s16 <output>
    """
    check_ffmpeg_available()

    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input media file not found: {input_file}")

    if input_file == output_file:
        raise ValueError("Input and output paths cannot be the same. Never overwrite original source.")

    if output_file.exists() and not overwrite:
        print(f"[audio_extractor] Output already exists, skipping re-extraction: {output_file.name}")
        return output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(input_file),
        "-vn",                      # Disable video stream
        "-ac", str(channels),       # Channel count (1 = mono)
        "-ar", str(sample_rate),    # Sample rate (16000 Hz)
        "-sample_fmt", "s16",       # Signed 16-bit PCM
        str(output_file)
    ]

    print(f"[audio_extractor] Extracting: {input_file.name} -> {output_file.name}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg extraction failed with return code {result.returncode}:\n{result.stderr}"
        )

    return output_file


def validate_audio(
    wav_path: Union[str, Path],
    expected_sr: int = 16000,
    expected_channels: int = 1,
    expected_subtype: str = "PCM_16"
) -> Dict[str, Any]:
    """
    Validates audio format properties using SoundFile.
    Raises ValueError if requirements are not met.
    """
    path = Path(wav_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found for validation: {path}")

    info = sf.info(str(path))

    errors = []
    if info.samplerate != expected_sr:
        errors.append(f"Sample rate mismatch: expected {expected_sr} Hz, got {info.samplerate} Hz")
    if info.channels != expected_channels:
        errors.append(f"Channel mismatch: expected {expected_channels} (mono), got {info.channels}")
    if info.subtype != expected_subtype:
        errors.append(f"Subtype mismatch: expected {expected_subtype}, got {info.subtype}")
    if info.format != "WAV":
        errors.append(f"Format mismatch: expected WAV, got {info.format}")

    if errors:
        raise ValueError(f"Audio validation failed for {path.name}:\n" + "\n".join(errors))

    return {
        "valid": True,
        "file": str(path),
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "format": info.format,
        "duration_seconds": round(info.duration, 2),
        "frames": info.frames
    }
