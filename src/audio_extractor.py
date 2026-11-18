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
                    item = os.path.expandvars(item.strip())
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


def _prepare_audio_paths(
    input_path: str|Path,
    output_path: str|Path
) -> tuple[Path, Path]:
    """
    Convert paths to absolute paths and validate them.
    """

    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()

    # The source media file must exist.
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input media file not found: {input_file}"
        )

    # Never allow the output to overwrite the source.
    if input_file == output_file:
        raise ValueError(
            "Input and output paths cannot be the same."
        )

    # Ensure the destination directory exists.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    return input_file, output_file




def _run_ffmpeg(
    cmd: list[str],
    temporary_file: Path,
    input_file: Path,
    output_file: Path
) -> None:
    """
    Execute FFmpeg and handle extraction errors.

    If extraction fails, remove the temporary file so that
    an incomplete WAV cannot accidentally be reused.
    """

    print(
        f"[audio_extractor] Extracting: "
        f"{input_file.name} -> {output_file.name}"
    )

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg extraction failed with return code "
                f"{result.returncode}:\n{result.stderr}"
            )

    except Exception:
        # Clean up any incomplete temporary output.
        temporary_file.unlink(missing_ok=True)
        raise


def _finalize_audio(
    temporary_file: Path,
    output_file: Path
) -> Path:
    """
    Move the completed temporary WAV to its final destination.

    os.replace() ensures that the final output filename is only
    updated after FFmpeg has successfully completed extraction.
    """

    try:
        os.replace(temporary_file, output_file)
    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise

    return output_file


def extract_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = False
) -> Path:
    """
    Extract and standardize audio into a 16-bit PCM WAV file.

    This function coordinates the extraction workflow:
        1. Check FFmpeg availability.
        2. Prepare and validate file paths.
        3. Reuse an existing output when possible.
        4. Build the FFmpeg command.
        5. Execute extraction.
        6. Finalize the output file.
    """

    # STEP 1: Make sure FFmpeg is available.
    ffmpeg_exe = check_ffmpeg_available()

    # STEP 2: Prepare and validate input/output paths.
    input_file, output_file = _prepare_audio_paths(
        input_path,
        output_path
    )

    # STEP 3: Avoid unnecessary extraction if output already exists.
    if output_file.exists() and not overwrite:
        print(
            f"[audio_extractor] Output already exists, "
            f"skipping re-extraction: {output_file.name}"
        )
        return output_file

    # STEP 4: Create a temporary output filename.
    temporary_file = output_file.with_name(
        output_file.stem + ".part.wav"
    )

    # Remove any leftover temporary file from a previous failed run.
    temporary_file.unlink(missing_ok=True)

    # STEP 5: Construct the FFmpeg command.
    cmd = [
    ffmpeg_exe,
    "-nostdin",
    "-y" if overwrite else "-n",
    "-i", str(input_file),
    "-vn",
    "-ac", str(channels),
    "-ar", str(sample_rate),
    "-sample_fmt", "s16",
    str(temporary_file)
    ]

    # STEP 6: Execute FFmpeg.
    _run_ffmpeg(
        cmd=cmd,
        temporary_file=temporary_file,
        input_file=input_file,
        output_file=output_file
    )

    # STEP 7: Move the completed WAV to its final location.
    return _finalize_audio(
        temporary_file,
        output_file
    )



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
    if info.frames == 0:
        errors.append("File contains no audio frames")
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
