"""
src/audio_extractor.py
======================
Stage 2: Audio Extraction and Format Validation

Purpose:
--------
Speech recognition models (such as Whisper and Meta MMS) and Voice Activity
Detectors (like Silero VAD) strictly expect audio in a standardized format:
  1. Sample Rate: 16,000 Hz (16 kHz).
  2. Channels: 1 (Mono).
  3. Bit Depth / Encoding: 16-bit linear PCM (signed integer, subtype PCM_16).
  4. Container: WAV.

This module provides two core functions:
  - extract_audio(): Uses FFmpeg via Python's subprocess module to perform
    high-speed, memory-efficient conversion directly on disk.
  - validate_audio(): Uses SoundFile to programmatically inspect the WAV file's
    header and frames, ensuring it complies with the required specification.
"""

# sys: Provides access to system-specific parameters and functions (e.g., detecting OS platform)
import sys

# os: Provides operating system interaction (e.g., environment variables, PATH management)
import os

# shutil: High-level file operations; shutil.which checks if an executable exists in PATH
import shutil

# subprocess: Spawns external system processes (used here to invoke the FFmpeg CLI tool)
import subprocess

# Path: Object-oriented filesystem path handling from Python's standard library
from pathlib import Path

# Typing helpers for function signatures and documentation
from typing import Dict, Any, Union

# soundfile: C-library wrapper (libsndfile) for reading/writing audio metadata and audio samples
import soundfile as sf


def check_ffmpeg_available() -> str:
    """
    Verifies that the FFmpeg executable is accessible on the system.

    Why this function is necessary:
    -------------------------------
    FFmpeg is a system-level binary, not a Python package. When running inside
    IDEs (like VS Code) on Windows, environment variables (like PATH) are sometimes
    inherited from a parent process that started before FFmpeg was installed.
    
    If shutil.which("ffmpeg") fails to find FFmpeg in the active terminal PATH,
    this function performs resilient fallbacks:
      1. Checks the Windows Registry for the current user's PATH variable.
      2. Directly scans the standard Windows WinGet package installation directory.
      
    If found via a fallback, it dynamically adds the directory to os.environ["PATH"]
    so that subsequent subprocess calls succeed without restarting the IDE.
    """
    # 1. Standard check: look for 'ffmpeg' in the current process's PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        # Found in standard PATH; return the path to the executable
        return ffmpeg_path

    # 2. Resilient fallback logic for Windows systems
    if sys.platform == "win32":
        # Fallback A: Query the Windows Registry for User Environment Variables
        try:
            import winreg  # Windows-only built-in module for reading the Windows Registry
            
            # Open the registry key where User Environment variables are stored
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                # Read the 'Path' value stored in the registry
                user_path, _ = winreg.QueryValueEx(key, "Path")
                
                # Split the PATH string into individual folder directories (separated by ';')
                for item in user_path.split(";"):
                    item = item.strip()
                    # If this directory exists and contains ffmpeg.exe, use it
                    if item and Path(item).is_dir():
                        candidate = Path(item) / "ffmpeg.exe"
                        if candidate.exists():
                            # Dynamically inject this directory into the running Python process's PATH
                            os.environ["PATH"] = str(item) + os.pathsep + os.environ.get("PATH", "")
                            return str(candidate)
        except Exception:
            # If registry lookup fails or permissions are restricted, silently continue to Fallback B
            pass

        # Fallback B: Directly scan the default WinGet Packages directory
        # WinGet installs packages to: %LOCALAPPDATA%\Microsoft\WinGet\Packages
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_pkgs = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_pkgs.exists():
                # Recursively search for any ffmpeg.exe inside the WinGet packages folder
                for candidate in winget_pkgs.glob("**/ffmpeg.exe"):
                    bin_dir = str(candidate.parent)
                    # Inject the discovered bin folder into the active process's PATH
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    return str(candidate)

    # If all detection methods fail, raise an explicit error with troubleshooting instructions
    raise RuntimeError(
        "FFmpeg executable not found in PATH or standard installation locations. "
        "Please ensure FFmpeg is installed and added to your environment Path."
    )


def extract_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = False
) -> Path:
    """
    Extracts and standardizes audio from a media container into a 16-bit PCM WAV file.

    Parameters:
    -----------
    input_path : str or Path
        Path to the source media file (e.g. .mp4, .m4a, .webm, .mkv).
    output_path : str or Path
        Target path for the extracted .wav file.
    sample_rate : int, default=16000
        Target sample rate in Hz. We use 16,000 Hz because standard speech models
        (Whisper, Silero, Wav2Vec2) are trained on 16 kHz audio.
    channels : int, default=1
        Number of audio channels. 1 = Mono. Downmixing to mono saves 50% RAM and compute.
    overwrite : bool, default=False
        If False and output_path already exists, skips extraction to avoid redundant disk I/O.

    Returns:
    --------
    Path: Absolute Path object pointing to the extracted WAV file.
    """
    # Ensure FFmpeg is accessible before doing any work
    check_ffmpeg_available()

    # Convert paths to absolute Path objects to avoid issues with relative directories
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    # Safety Check 1: The input file must actually exist on disk
    if not input_file.exists():
        raise FileNotFoundError(f"Input media file not found: {input_file}")

    # Safety Check 2: Never overwrite the raw source file
    if input_file == output_file:
        raise ValueError("Input and output paths cannot be the same. Never overwrite original source.")

    # Idempotency Check: If output already exists and overwrite is False, reuse it
    if output_file.exists() and not overwrite:
        print(f"[audio_extractor] Output already exists, skipping re-extraction: {output_file.name}")
        return output_file

    # Ensure the parent folder (e.g., data/processed/) exists before writing into it
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build the FFmpeg command line arguments:
    # ffmpeg -y/-n -i <input> -vn -ac 1 -ar 16000 -sample_fmt s16 <output>
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",  # -y forces overwrite; -n cancels if file exists
        "-i", str(input_file),        # -i specifies the input media file
        "-vn",                        # -vn disables video recording (strips any video streams)
        "-ac", str(channels),         # -ac sets the number of audio channels (1 = mono)
        "-ar", str(sample_rate),      # -ar sets audio sampling rate in Hz (16000)
        "-sample_fmt", "s16",         # -sample_fmt sets audio sample format to signed 16-bit PCM
        str(output_file)              # Target output WAV destination
    ]

    print(f"[audio_extractor] Extracting: {input_file.name} -> {output_file.name}")

    # Execute the command via subprocess
    # stdout and stderr are captured as text pipes for clean error reporting
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # If FFmpeg exited with any code other than 0, an error occurred
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
    Validates that a WAV file strictly adheres to the required audio specifications.

    Why this validation step is mandatory:
    --------------------------------------
    Silent errors in audio formats (e.g. 44.1 kHz instead of 16 kHz, or stereo instead
    of mono) cause AI models to either crash or produce hallucinated/garbage transcripts.
    This function acts as a strict quality gate before any file enters the corpus pipeline.

    Parameters:
    -----------
    wav_path : str or Path
        Path to the WAV file to inspect.
    expected_sr : int, default=16000
        Expected sample rate in Hz.
    expected_channels : int, default=1
        Expected channel count (1 = mono).
    expected_subtype : str, default='PCM_16'
        Expected encoding subtype (16-bit signed pulse-code modulation).

    Returns:
    --------
    dict: Metadata dictionary containing audio duration, frames, sample rate, etc.
    """
    path = Path(wav_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found for validation: {path}")

    # Read audio metadata from the WAV header without decoding all samples into memory
    info = sf.info(str(path))

    # Collect any discrepancies between expected and actual properties
    errors = []
    if info.samplerate != expected_sr:
        errors.append(f"Sample rate mismatch: expected {expected_sr} Hz, got {info.samplerate} Hz")
    if info.channels != expected_channels:
        errors.append(f"Channel mismatch: expected {expected_channels} (mono), got {info.channels}")
    if info.subtype != expected_subtype:
        errors.append(f"Subtype mismatch: expected {expected_subtype}, got {info.subtype}")
    if info.format != "WAV":
        errors.append(f"Format mismatch: expected WAV, got {info.format}")

    # If any check failed, raise an explicit error stopping downstream processing
    if errors:
        raise ValueError(f"Audio validation failed for {path.name}:\n" + "\n".join(errors))

    # Return a validated summary dictionary
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
