"""
tests/test_classifier.py
Test for Stage 4: Region Classification (on a 60-second slice)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
from src.classifier import (
    compute_frame_energy_db,
    compute_frame_spectral_flatness,
    run_silero_vad_chunked,
    classify_frames,
    merge_frames_into_regions,
)


def test_classifier_on_60s_slice():
    wav_path = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"

    print("==============================================")
    print(" Stage 4 Classifier Test (60-second slice)    ")
    print("==============================================\n")

    # Load 60 seconds of audio starting at the 1-minute mark
    print("[TEST] Loading 60-second audio slice (1:00 to 2:00)...")
    audio, sr = sf.read(str(wav_path), start=16000 * 60, frames=16000 * 60, dtype="float32")
    print(f"  -> Loaded {len(audio)} samples at {sr} Hz ({len(audio)/sr:.1f}s)\n")

    # Signal A: Energy analysis
    print("[TEST] Computing frame-level RMS energy...")
    energy_db = compute_frame_energy_db(audio)
    print(f"  -> {len(energy_db)} frames computed (50 ms each)\n")

    # Signal B: Spectral Flatness (Wiener entropy) to distinguish fan noise from music
    print("[TEST] Computing frame-level Spectral Flatness...")
    flatness = compute_frame_spectral_flatness(audio)
    print(f"  -> {len(flatness)} flatness values computed (avg: {flatness.mean():.3f})\n")

    # Signal C: Silero VAD
    print("[TEST] Running Silero VAD...")
    vad_probs = run_silero_vad_chunked(audio, sr)
    print(f"  -> {len(vad_probs)} frames with speech probabilities\n")

    # Decision fusion
    print("[TEST] Applying classification rules with Spectral Flatness...")
    frame_labels = classify_frames(energy_db, vad_probs, spectral_flatness=flatness)
    print(f"  -> {len(frame_labels)} frames classified\n")

    # Merge into regions
    print("[TEST] Merging frames into continuous regions...")
    regions = merge_frames_into_regions(frame_labels, sr)
    print(f"  -> {len(regions)} regions created\n")

    # Display results
    print("=" * 55)
    print(f" {'Start':>8}   {'End':>8}   {'Label':<10}  {'Confidence'}")
    print("-" * 55)
    for r in regions:
        print(f" {r['start_sec']:>7.1f}s  {r['end_sec']:>7.1f}s   {r['label']:<10}  {r['confidence']}")
    print("=" * 55)
    print("\nTEST PASSED!")


if __name__ == "__main__":
    test_classifier_on_60s_slice()
