"""
tests/test_audio_extractor.py
Test suite for Stage 2 audio extraction and validation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio_extractor import check_ffmpeg_available, validate_audio, extract_audio


def test_ffmpeg_available():
    print("[TEST 1] Checking FFmpeg availability...")
    ffmpeg_bin = check_ffmpeg_available()
    print(f"  -> Found FFmpeg at: {ffmpeg_bin}")
    assert Path(ffmpeg_bin).exists()
    print("  -> PASSED\n")


def test_validate_existing_processed_wav():
    print("[TEST 2] Validating existing processed audio (gK017cCH36k.wav)...")
    wav_path = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"

    if not wav_path.exists():
        print(f"  -> Skipping (file not found: {wav_path})")
        return

    result = validate_audio(wav_path)
    print(f"  -> File: {Path(result['file']).name}")
    print(f"  -> Sample Rate: {result['samplerate']} Hz")
    print(f"  -> Channels: {result['channels']} (mono)")
    print(f"  -> Subtype: {result['subtype']}")
    print(f"  -> Duration: {result['duration_seconds']} seconds ({result['duration_seconds']/60:.1f} minutes)")
    assert result["valid"] is True
    print("  -> PASSED\n")


def test_small_extraction():
    print("[TEST 3] Testing extraction pipeline on sample...")
    raw_m4a = PROJECT_ROOT / "data" / "raw" / "gK017cCH36k.m4a"
    test_out = PROJECT_ROOT / "data" / "processed" / "test_sample_output.wav"

    if not raw_m4a.exists():
        print(f"  -> Skipping (raw source not found: {raw_m4a})")
        return

    try:
        # Extract audio using our module
        extracted = extract_audio(raw_m4a, test_out, overwrite=True)
        assert extracted.exists()

        # Validate extracted audio
        result = validate_audio(extracted)
        assert result["valid"] is True
        print(f"  -> Extracted & validated successfully: {extracted.name}")
        print("  -> PASSED\n")
    finally:
        # Cleanup test artifact
        if test_out.exists():
            test_out.unlink()
            print("  -> Cleaned up test artifact.")


if __name__ == "__main__":
    print("========================================")
    print(" Running Stage 2: Audio Extractor Tests ")
    print("========================================\n")
    test_ffmpeg_available()
    test_validate_existing_processed_wav()
    test_small_extraction()
    print("ALL TESTS PASSED SUCCESSFULLY!")
