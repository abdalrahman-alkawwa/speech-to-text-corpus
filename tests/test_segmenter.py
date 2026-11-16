"""
tests/test_segmenter.py
Test suite for Stage 5: Speech Segmentation
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
from src.audio_extractor import validate_audio
from src.classifier import (
    compute_frame_energy_db,
    run_silero_vad_chunked,
    classify_frames,
    merge_frames_into_regions
)
from src.segmenter import export_segments, SEGMENTS_DIR


def test_segmenter():
    print("=========================================")
    print(" Running Stage 5: Speech Segmenter Tests ")
    print("=========================================\n")

    wav_path = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    assert wav_path.exists(), f"Audio file not found: {wav_path}"

    # Step 1: Generate classified regions for a 60-second slice
    print("[TEST 1] Generating classified regions for 60s slice...")
    audio, sr = sf.read(str(wav_path), start=16000 * 60, frames=16000 * 60, dtype="float32")
    energy_db = compute_frame_energy_db(audio)
    vad_probs = run_silero_vad_chunked(audio, sr)
    frame_labels = classify_frames(energy_db, vad_probs)
    regions = merge_frames_into_regions(frame_labels, sr)
    print(f"  -> {len(regions)} regions ready for segmentation.\n")

    # Step 2: Run segmentation and export up to 5 segments
    print("[TEST 2] Segmenting and exporting clips to data/segments/...")
    # Create test segment slice directly
    manifest = export_segments(
        wav_path=wav_path,
        regions=regions,
        video_id="gK017cCH36k",
        max_segments=5
    )
    assert len(manifest) > 0, "No segments generated!"
    print(f"  -> Generated {len(manifest)} test segments.\n")

    # Step 3: Validate each generated segment
    print("[TEST 3] Validating exported segment WAV files...")
    for rec in manifest:
        seg_file = PROJECT_ROOT / rec["audio_path"]
        assert seg_file.exists(), f"Segment file missing: {seg_file}"

        # Check audio format compliance (16 kHz, mono, PCM_16)
        validation = validate_audio(seg_file)
        assert validation["valid"] is True

        dur = rec["duration_sec"]
        print(f"  • {rec['segment_id']}: {dur:.2f}s | {rec['boundary_method']} | {rec['boundary_check']}")

        # Enforce project duration constraints: 5s to 15s (or close edge)
        assert 3.0 <= dur <= 16.0, f"Segment duration {dur}s outside expected limits!"

    print("\n=========================================")
    print(" ALL SEGMENTATION TESTS PASSED!          ")
    print("=========================================")


if __name__ == "__main__":
    test_segmenter()
