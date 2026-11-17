"""
tests/test_snr_estimator.py
Test for Stage 6: SNR Estimation (both methods)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import soundfile as sf
from src.snr_estimator import compute_wada_snr, compute_segmental_snr, estimate_segment_snr


def test_snr_estimation():
    print("==========================================")
    print(" Running Stage 6: SNR Estimation Tests    ")
    print("==========================================\n")

    wav_path = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    assert wav_path.exists(), f"Audio not found: {wav_path}"

    # Load 60s slice (1:00 to 2:00)
    print("[TEST 1] Loading 60-second audio slice...")
    audio, sr = sf.read(str(wav_path), start=16000 * 60, frames=16000 * 60, dtype="float32")
    print(f"  -> {len(audio)} samples at {sr} Hz ({len(audio)/sr:.1f}s)\n")

    # Test WADA-SNR on the slice
    print("[TEST 2] Computing WADA-SNR (blind estimation)...")
    wada_snr = compute_wada_snr(audio)
    print(f"  -> WADA-SNR: {wada_snr:.1f} dB")
    assert not (wada_snr != wada_snr), "WADA-SNR returned NaN on valid audio!"
    assert -10.0 <= wada_snr <= 60.0, f"WADA-SNR out of range: {wada_snr}"
    print("  -> PASSED\n")

    # Build quick regions from classifier for the slice
    print("[TEST 3] Running classifier to get regions for Segmental SNR...")
    from src.classifier import (
        compute_frame_energy_db, run_silero_vad_chunked,
        classify_frames, merge_frames_into_regions
    )
    energy_db = compute_frame_energy_db(audio)
    vad_probs = run_silero_vad_chunked(audio, sr)
    frame_labels = classify_frames(energy_db, vad_probs)
    regions = merge_frames_into_regions(frame_labels, sr)
    print(f"  -> {len(regions)} regions classified\n")

    # Test Segmental SNR
    print("[TEST 4] Computing Segmental SNR (region-based)...")
    seg_snr = compute_segmental_snr(audio, sr, regions)
    print(f"  -> Segmental SNR: {seg_snr:.1f} dB")
    assert not (seg_snr != seg_snr), "Segmental SNR returned NaN!"
    assert -10.0 <= seg_snr <= 80.0, f"Segmental SNR out of range: {seg_snr}"
    print("  -> PASSED\n")

    # Test per-segment estimation
    print("[TEST 5] Estimating SNR for a single segment clip...")
    snr_result = estimate_segment_snr(audio, sr, 0.0, 10.0, regions)
    print(f"  -> WADA: {snr_result['snr_wada_db']:.1f} dB | Segmental: {snr_result['snr_segmental_db']:.1f} dB")
    print("  -> PASSED\n")

    print("==========================================")
    print(" ALL SNR ESTIMATION TESTS PASSED!         ")
    print("==========================================")


if __name__ == "__main__":
    test_snr_estimation()
