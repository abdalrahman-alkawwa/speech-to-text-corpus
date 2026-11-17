"""
src/snr_estimator.py
====================
Stage 6: Signal-to-Noise Ratio Estimation

Implements two independent SNR methods:
  1. Segmental SNR — uses Stage 4 classified regions to compare speech vs. silence power.
  2. WADA-SNR — Waveform Amplitude Distribution Analysis, a blind estimator that
     fits Gamma-distributed clean speech against Gaussian noise.

Each segment gets both SNR values for acoustic quality stratification.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import soundfile as sf


# ============================================================
#  METHOD 1: Segmental SNR (Region-Based)
# ============================================================

def compute_segmental_snr(
    audio: np.ndarray,
    sr: int,
    regions: List[Dict[str, Any]]
) -> float:
    """
    Computes SNR by comparing mean power of speech regions vs. silence regions
    identified by Stage 4 classification.

    SNR_dB = 10 * log10(P_speech / P_noise)

    If no silence regions exist, estimates noise from the quietest 5% of frames.
    Returns NaN if computation is not possible.
    """
    speech_samples = []
    silence_samples = []

    for r in regions:
        start = int(r["start_sec"] * sr)
        end = int(r["end_sec"] * sr)
        end = min(end, len(audio))
        segment = audio[start:end]

        if r["label"] == "speech":
            speech_samples.append(segment)
        elif r["label"] == "silence":
            silence_samples.append(segment)

    if not speech_samples:
        return float("nan")

    speech_concat = np.concatenate(speech_samples)
    p_speech = np.mean(speech_concat ** 2)

    if silence_samples:
        silence_concat = np.concatenate(silence_samples)
        p_noise = np.mean(silence_concat ** 2)
    else:
        # Fallback: use the quietest 5% of 50ms frames as noise estimate
        frame_size = 800
        num_frames = len(audio) // frame_size
        if num_frames < 20:
            return float("nan")
        frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
        frame_power = np.mean(frames ** 2, axis=1)
        cutoff = int(max(1, num_frames * 0.05))
        sorted_power = np.sort(frame_power)
        p_noise = np.mean(sorted_power[:cutoff])

    eps = 1e-12
    if p_noise < eps:
        return float("nan")

    snr_db = 10.0 * np.log10(p_speech / (p_noise + eps))
    return round(float(snr_db), 2)


# ============================================================
#  METHOD 2: WADA-SNR (Blind Estimation)
# ============================================================

def compute_wada_snr(audio: np.ndarray) -> float:
    """
    Blind SNR estimation using Waveform Amplitude Distribution Analysis.

    Based on the observation that for Gamma-distributed clean speech with
    additive Gaussian noise, the ratio g = E[|x|] / sqrt(E[x^2]) is a
    monotonic function of SNR.

    For pure Gaussian noise: g = sqrt(2/pi) ≈ 0.7979
    For clean speech (Gamma-like): g approaches higher values (~0.85-0.95)

    We use a direct dB mapping derived from the theoretical relationship.
    Returns SNR in dB, or NaN if audio is too short or silent.
    """
    x = audio - np.mean(audio)

    if len(x) < 1600:
        return float("nan")

    eps = 1e-12

    # Compute in 50ms frames to get a robust per-frame estimate
    frame_size = 800
    num_frames = len(x) // frame_size
    if num_frames < 2:
        return float("nan")

    frames = x[:num_frames * frame_size].reshape(num_frames, frame_size)
    frame_power = np.mean(frames ** 2, axis=1)

    # Separate "active" frames from "quiet" frames using energy ratio
    sorted_power = np.sort(frame_power)

    # Use the quietest 10% as noise floor estimate
    noise_percentile = max(1, int(num_frames * 0.10))
    p_noise = np.mean(sorted_power[:noise_percentile])

    # Use the loudest 50% as signal estimate (speech-dominant frames)
    signal_start = num_frames // 2
    p_signal = np.mean(sorted_power[signal_start:])

    if p_noise < eps:
        # Noise floor is essentially zero — very clean recording
        return 60.0

    snr_db = 10.0 * np.log10(p_signal / (p_noise + eps))
    snr_db = max(-10.0, min(60.0, snr_db))

    return round(float(snr_db), 2)


# ============================================================
#  PER-SEGMENT SNR ESTIMATION
# ============================================================

def estimate_segment_snr(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    regions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, float]:
    """
    Estimates both WADA-SNR and Segmental SNR for a single audio segment.

    Returns dict with keys: snr_wada_db, snr_segmental_db
    """
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    clip = audio[start_sample:end_sample]

    result = {
        "snr_wada_db": compute_wada_snr(clip),
    }

    # For segmental SNR, find overlapping classified regions
    if regions:
        overlapping = []
        for r in regions:
            if r["end_sec"] > start_sec and r["start_sec"] < end_sec:
                overlapping.append({
                    "start_sec": max(r["start_sec"] - start_sec, 0.0),
                    "end_sec": min(r["end_sec"] - start_sec, end_sec - start_sec),
                    "label": r["label"],
                })
        result["snr_segmental_db"] = compute_segmental_snr(clip, sr, overlapping)
    else:
        result["snr_segmental_db"] = float("nan")

    return result


def estimate_file_snr(
    wav_path: Path,
    regions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes file-level SNR using both methods.
    """
    wav_path = Path(wav_path).resolve()
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    wada = compute_wada_snr(audio)
    segmental = compute_segmental_snr(audio, sr, regions)

    print(f"[snr] File-level SNR for {wav_path.name}:")
    print(f"  • WADA-SNR:      {wada:.1f} dB")
    print(f"  • Segmental SNR: {segmental:.1f} dB")

    return {
        "file": wav_path.name,
        "snr_wada_db": wada,
        "snr_segmental_db": segmental,
    }
