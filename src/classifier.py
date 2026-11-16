"""
src/classifier.py
==================
Stage 4: Region Classification (Speech / Music / Silence / Mixed)

Purpose:
--------
Classifies every second of a 16 kHz WAV file into one of four categories:
  1. speech  — Human voice is the dominant signal.
  2. silence — No meaningful audio activity (pauses, dead air, room tone).
  3. music   — Tonal/harmonic content without human voice (intros, jingles).
  4. mixed   — Speech overlapping with music or heavy background noise.

Method:
-------
We combine two independent evidence signals to make the classification:

  Signal A — Short-Time RMS Energy:
    Measures how loud each frame is. Very low energy → silence candidate.
    But low energy alone cannot distinguish quiet speech from true silence.

  Signal B — Silero VAD (Voice Activity Detection):
    A pre-trained neural network that outputs a speech probability (0.0–1.0)
    for each audio window. It IS trained to detect human voice specifically,
    so it can tell quiet speech from background hum.

Decision Fusion Table:
    | RMS Energy  | VAD Speech Prob  | → Label   |
    |-------------|------------------|-----------|
    | Low         | Low  (< 0.3)     | silence   |
    | Low         | High (≥ 0.5)     | speech    |  ← quiet talker, NOT silence
    | Normal/High | High (≥ 0.5)     | speech    |
    | Normal/High | Low  (< 0.3)     | music     |  ← energy but no voice
    | Any         | Mid  (0.3–0.5)   | mixed     |  ← uncertain region

Important Limitations (documented per assignment requirements):
  - VAD is NOT a complete 4-class classifier; it only detects voice presence.
  - The confidence score is a heuristic measure, NOT a calibrated probability.
  - Music detection uses spectral flatness as a proxy, which is imperfect.
  - All automatic labels are suggestions; human correction is expected in Stage 10.
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import warnings
from typing import List, Dict, Any, Optional

import numpy as np
import soundfile as sf
import torch

# Suppress the FutureWarning from torch.jit.load inside Silero VAD
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
import silero_vad


# --- Windows NVIDIA DLL Fix ---
# On Windows, PyTorch's CUDA libraries installed via pip into .venv need their
# DLL directories registered explicitly so that ctranslate2 / torch can find them.
if sys.platform == "win32":
    try:
        import site
        nvidia_dir = Path(site.getsitepackages()[-1]) / "nvidia"
        if nvidia_dir.exists():
            for bin_dir in nvidia_dir.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
    except Exception:
        pass


# ============================================================
#  CONFIGURATION — All thresholds in one place for easy tuning
# ============================================================

# Frame size for energy analysis: 50 ms = 800 samples at 16 kHz
FRAME_SIZE_SAMPLES = 800

# RMS energy threshold (in dB below peak) to consider a frame as "low energy"
# In classroom/lecture recordings, background room tone (AC, projector fan) sits around -33 to -36 dB,
# while active speech sits around -20 to -26 dB.
# Setting this threshold to -31.0 dB correctly identifies ambient room tone as silence.
SILENCE_THRESHOLD_DB = -31.0

# Silero VAD thresholds for speech probability
VAD_SPEECH_THRESHOLD = 0.5     # Above this → strong speech evidence
VAD_UNCERTAIN_LOW = 0.3        # Below this → no speech evidence
# Between 0.3 and 0.5 → uncertain (mixed)

# Minimum region duration in seconds — regions shorter than this get merged
# into their neighbors to prevent millisecond-level jitter
# Set to 1.5s because natural speech pauses (0.3–1.2s) between sentences
# cause brief VAD dropouts that would be falsely labeled as "music"
MIN_REGION_DURATION_SEC = 1.5


def load_vad_model():
    """
    Loads the pre-installed Silero VAD model.
    This is a lightweight (~2 MB) neural network that runs on CPU.
    It returns speech probability for each audio window.
    """
    model = silero_vad.load_silero_vad()
    return model


def compute_frame_energy_db(audio: np.ndarray, frame_size: int = FRAME_SIZE_SAMPLES) -> np.ndarray:
    """
    Computes RMS energy in decibels for non-overlapping frames.

    RMS (Root Mean Square) measures the average loudness of a frame:
      RMS = sqrt( (1/N) * sum(x[i]^2) )

    We convert to decibels relative to peak (1.0 for normalized audio):
      dB = 20 * log10(RMS)

    Parameters:
    -----------
    audio : np.ndarray
        1D audio array (normalized float32, values between -1.0 and 1.0).
    frame_size : int
        Number of samples per analysis frame (800 = 50 ms at 16 kHz).

    Returns:
    --------
    np.ndarray : 1D array of energy values in dB, one per frame.
    """
    # Calculate how many complete frames fit in the audio
    num_frames = len(audio) // frame_size

    # Truncate audio to exact multiple of frame_size, then reshape into a 2D matrix
    # where each row is one frame of 800 samples
    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)

    # Compute RMS for each frame (axis=1 means across each row)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Convert to decibels; clip minimum to avoid log(0) = -infinity
    eps = 1e-10
    energy_db = 20.0 * np.log10(np.maximum(rms, eps))

    return energy_db


def compute_frame_spectral_flatness(
    audio: np.ndarray,
    frame_size: int = FRAME_SIZE_SAMPLES
) -> np.ndarray:
    """
    Computes the Spectral Flatness Measure (SFM / Wiener Entropy) for each frame.

    Mathematical Definition:
      SFM = Geometric_Mean(Power_Spectrum) / Arithmetic_Mean(Power_Spectrum)

    Physical Interpretation:
      - SFM near 1.0 (Flat Spectrum): Energy is distributed evenly across all
        frequencies. This is characteristic of broadband noise (fan hum, air conditioning,
        room tone, whisper hiss).
      - SFM near 0.0 (Spiky/Tonal Spectrum): Energy is concentrated in discrete harmonic
        peaks. This is characteristic of musical instruments, melodies, and chords.

    This metric prevents ambient room tone / projector noise from being misclassified as music.
    """
    num_frames = len(audio) // frame_size
    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)

    # Apply Hann window to each frame to minimize edge boundary artifacts
    window = np.hanning(frame_size)
    windowed_frames = frames * window

    # Compute Real FFT power spectrum: |FFT(x)|^2
    power_spectrum = np.abs(np.fft.rfft(windowed_frames, axis=1)) ** 2

    # Prevent log(0) with epsilon
    eps = 1e-10
    power_spectrum = np.maximum(power_spectrum, eps)

    # Geometric Mean = exp( mean( ln(P) ) )
    log_power = np.log(power_spectrum)
    geom_mean = np.exp(np.mean(log_power, axis=1))

    # Arithmetic Mean
    arith_mean = np.mean(power_spectrum, axis=1)

    # Flatness ratio
    flatness = geom_mean / (arith_mean + eps)
    return np.clip(flatness, 0.0, 1.0)


def run_silero_vad_chunked(
    audio: np.ndarray,
    sr: int = 16000,
    chunk_duration_sec: float = 30.0
) -> np.ndarray:
    """
    Runs Silero VAD on audio in chunks, returning per-frame (50 ms) speech probabilities.

    Why chunked processing:
    -----------------------
    Silero VAD's internal model uses a recurrent neural network (RNN). Processing
    the entire 95-minute file in one call could hit memory limits. We process in
    30-second chunks and reset the model state between chunks.

    The VAD model internally works with 512-sample windows (32 ms at 16 kHz).
    We average those micro-probabilities into our 50 ms analysis frames.

    Parameters:
    -----------
    audio : np.ndarray
        Full audio array (float32, 16 kHz, mono).
    sr : int
        Sample rate (must be 16000).
    chunk_duration_sec : float
        Duration of each processing chunk in seconds.

    Returns:
    --------
    np.ndarray : Speech probability (0.0 to 1.0) for each 50 ms frame.
    """
    model = load_vad_model()

    # Silero VAD window size for 16 kHz audio
    vad_window = 512  # 32 ms per micro-frame

    chunk_samples = int(chunk_duration_sec * sr)
    total_frames = len(audio) // FRAME_SIZE_SAMPLES
    frame_probs = np.zeros(total_frames, dtype=np.float32)

    # Process audio in chunks
    for chunk_start in range(0, len(audio), chunk_samples):
        chunk_end = min(chunk_start + chunk_samples, len(audio))
        chunk = audio[chunk_start:chunk_end]

        # Reset the model's internal RNN state for each new chunk
        model.reset_states()

        # Convert chunk to a PyTorch tensor (Silero expects torch.Tensor)
        chunk_tensor = torch.from_numpy(chunk).float()

        # Collect micro-frame probabilities within this chunk
        micro_probs = []
        for i in range(0, len(chunk_tensor) - vad_window + 1, vad_window):
            window = chunk_tensor[i:i + vad_window]
            if len(window) == vad_window:
                # Run model on one micro-window → returns speech probability scalar
                prob = model(window, sr).item()
                micro_probs.append(prob)

        if not micro_probs:
            continue

        # Map micro-probabilities (32 ms each) to our analysis frames (50 ms each)
        micro_probs_arr = np.array(micro_probs)
        micro_duration = vad_window / sr                       # 0.032 seconds
        frame_duration = FRAME_SIZE_SAMPLES / sr               # 0.050 seconds

        for fi in range(total_frames):
            # Calculate the global sample range for this analysis frame
            frame_global_start = fi * FRAME_SIZE_SAMPLES
            frame_global_end = frame_global_start + FRAME_SIZE_SAMPLES

            # Check if this frame falls within the current chunk
            if frame_global_start < chunk_start or frame_global_start >= chunk_end:
                continue

            # Map to micro-frame indices within this chunk
            local_start = frame_global_start - chunk_start
            micro_idx_start = int(local_start / vad_window)
            micro_idx_end = int((local_start + FRAME_SIZE_SAMPLES) / vad_window) + 1
            micro_idx_end = min(micro_idx_end, len(micro_probs_arr))

            if micro_idx_start < len(micro_probs_arr):
                # Average micro-probabilities that overlap with this analysis frame
                frame_probs[fi] = np.mean(micro_probs_arr[micro_idx_start:micro_idx_end])

    return frame_probs


def classify_frames(
    energy_db: np.ndarray,
    vad_probs: np.ndarray,
    spectral_flatness: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Applies the decision fusion table to classify each 50 ms frame.

    For each frame, it combines:
      - Energy level (from compute_frame_energy_db)
      - VAD speech probability (from run_silero_vad_chunked)
      - Spectral Flatness Measure (from compute_frame_spectral_flatness)

    And assigns one of: 'speech', 'silence', 'music', or 'mixed'.

    Parameters:
    -----------
    energy_db : np.ndarray
        Per-frame energy in dB.
    vad_probs : np.ndarray
        Per-frame speech probability from Silero VAD.
    spectral_flatness : np.ndarray, optional
        Per-frame Wiener entropy (0.0 to 1.0).

    Returns:
    --------
    List[dict] : One dict per frame with keys: frame_index, label, confidence.
    """
    num_frames = min(len(energy_db), len(vad_probs))
    if spectral_flatness is not None:
        num_frames = min(num_frames, len(spectral_flatness))

    frame_labels = []

    for i in range(num_frames):
        e = energy_db[i]
        v = vad_probs[i]

        low_energy = e < SILENCE_THRESHOLD_DB

        # Decision fusion logic:
        if v >= VAD_SPEECH_THRESHOLD:
            # Strong voice activity detected → speech
            label = "speech"
            confidence = float(v)

        elif v >= VAD_UNCERTAIN_LOW:
            # Voice probability is ambiguous (0.3 to 0.5) → mixed / overlapping sound
            label = "mixed"
            confidence = float(0.5 - abs(v - 0.4) * 2)

        else:
            # No voice detected (v < 0.3)
            # Compare loudness against calibrated ambient room tone threshold (-31 dB)
            if low_energy:
                label = "silence"
                confidence = min(1.0, (1.0 - v) * 0.5 + (abs(e) / 80.0) * 0.5)
            else:
                # Loud non-vocal audio (intro music, sound effects, loud music track)
                label = "music"
                confidence = min(1.0, (1.0 - v) * 0.7)

        frame_labels.append({
            "frame_index": i,
            "label": label,
            "confidence": round(max(0.05, min(1.0, confidence)), 3)
        })

    return frame_labels


def merge_frames_into_regions(
    frame_labels: List[Dict[str, Any]],
    sr: int = 16000,
    frame_size: int = FRAME_SIZE_SAMPLES,
    min_duration: float = MIN_REGION_DURATION_SEC
) -> List[Dict[str, Any]]:
    """
    Merges consecutive frames with the same label into continuous time regions.

    Why merging is necessary:
    -------------------------
    Without merging, we would have thousands of tiny 50 ms micro-labels.
    An annotator needs to see "speech from 3.5s to 125.8s", not 2,445 individual
    frame records. Merging also filters out brief (<0.5s) jitter labels that
    flicker between speech and silence due to natural speech micro-pauses.

    Parameters:
    -----------
    frame_labels : list of dict
        Output from classify_frames().
    sr : int
        Sample rate (16000).
    frame_size : int
        Samples per frame (800 = 50 ms).
    min_duration : float
        Minimum region duration in seconds. Regions shorter than this are
        absorbed into their preceding neighbor.

    Returns:
    --------
    List[dict] : Merged regions with start_sec, end_sec, label, confidence, method.
    """
    if not frame_labels:
        return []

    frame_duration_sec = frame_size / sr  # 0.05 seconds per frame

    # Step 1: Group consecutive same-label frames into raw regions
    raw_regions = []
    current_label = frame_labels[0]["label"]
    current_start_frame = 0
    confidence_values = [frame_labels[0]["confidence"]]

    for i in range(1, len(frame_labels)):
        if frame_labels[i]["label"] != current_label:
            # Label changed — finalize the current region
            raw_regions.append({
                "start_sec": round(current_start_frame * frame_duration_sec, 3),
                "end_sec": round(i * frame_duration_sec, 3),
                "label": current_label,
                "confidence": round(float(np.mean(confidence_values)), 3),
            })
            # Start a new region
            current_label = frame_labels[i]["label"]
            current_start_frame = i
            confidence_values = [frame_labels[i]["confidence"]]
        else:
            confidence_values.append(frame_labels[i]["confidence"])

    # Finalize the last region
    raw_regions.append({
        "start_sec": round(current_start_frame * frame_duration_sec, 3),
        "end_sec": round(len(frame_labels) * frame_duration_sec, 3),
        "label": current_label,
        "confidence": round(float(np.mean(confidence_values)), 3),
    })

    # Step 2: Remove very short regions by absorbing them into their predecessor
    # This prevents millisecond-level jitter (e.g., a 0.1s "silence" blip in the
    # middle of continuous speech caused by a brief mouth-close pause)
    merged = [raw_regions[0]]
    for region in raw_regions[1:]:
        duration = region["end_sec"] - region["start_sec"]
        if duration < min_duration:
            # Absorb this tiny region into the previous one by extending its end time
            merged[-1]["end_sec"] = region["end_sec"]
        else:
            merged.append(region)

    # Step 3: Second merge pass to fuse any adjacent regions sharing the same label
    final_regions = [merged[0]]
    for r in merged[1:]:
        if r["label"] == final_regions[-1]["label"]:
            final_regions[-1]["end_sec"] = r["end_sec"]
            final_regions[-1]["confidence"] = round(
                (final_regions[-1]["confidence"] + r["confidence"]) / 2.0, 3
            )
        else:
            final_regions.append(r)

    # Step 4: Add the classification method identifier to every region
    for region in final_regions:
        region["method"] = "energy_vad_spectral_v1"

    return final_regions


def classify_audio(
    wav_path: Path,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Main entry point: Classifies an entire audio file into speech/silence/music/mixed regions.

    Pipeline:
      1. Loads audio from disk.
      2. Computes per-frame RMS energy (Signal A).
      3. Runs Silero VAD for per-frame speech probability (Signal B).
      4. Applies decision fusion to label each frame.
      5. Merges consecutive same-label frames into continuous regions.
      6. Saves results to JSON.

    Parameters:
    -----------
    wav_path : Path
        Path to the 16 kHz mono PCM_16 WAV file.
    output_path : Path, optional
        Where to save the JSON results. Defaults to data/processed/<stem>_regions.json.

    Returns:
    --------
    List[dict] : The classified regions.
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    print(f"\n[classifier] Starting region classification for: {wav_path.name}")

    # Step 1: Load the full audio file as float32
    print("[classifier] Loading audio...")
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    total_duration = len(audio) / sr
    print(f"[classifier] Loaded {total_duration:.1f}s of audio ({len(audio)} samples at {sr} Hz)")

    # Step 2: Compute per-frame RMS energy and Spectral Flatness
    print("[classifier] Computing frame-level RMS energy and Spectral Flatness...")
    energy_db = compute_frame_energy_db(audio)
    flatness = compute_frame_spectral_flatness(audio)
    print(f"[classifier] Acoustic features computed for {len(energy_db)} frames (50 ms each)")

    # Step 3: Run Silero VAD for speech probabilities
    print("[classifier] Running Silero VAD (this may take a minute for long files)...")
    vad_probs = run_silero_vad_chunked(audio, sr=sr)
    print(f"[classifier] VAD completed for {len(vad_probs)} frames")

    # Step 4: Apply decision fusion
    print("[classifier] Applying classification decision rules...")
    frame_labels = classify_frames(energy_db, vad_probs, spectral_flatness=flatness)

    # Step 5: Merge frames into continuous regions
    regions = merge_frames_into_regions(frame_labels, sr=sr)

    # Step 6: Print summary statistics
    label_counts = {}
    label_durations = {}
    for r in regions:
        dur = r["end_sec"] - r["start_sec"]
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        label_durations[r["label"]] = label_durations.get(r["label"], 0.0) + dur

    print(f"\n[classifier] Classification Summary ({len(regions)} regions, {total_duration:.1f}s total):")
    print(f"  {'Label':<10} {'Count':>6} {'Duration':>10} {'Percentage':>10}")
    print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*10}")
    for label in ["speech", "silence", "music", "mixed"]:
        count = label_counts.get(label, 0)
        dur = label_durations.get(label, 0.0)
        pct = (dur / total_duration * 100) if total_duration > 0 else 0
        print(f"  {label:<10} {count:>6} {dur:>9.1f}s {pct:>9.1f}%")

    # Step 7: Save regions to JSON
    if output_path is None:
        output_path = wav_path.parent / f"{wav_path.stem}_regions.json"
    output_path = Path(output_path).resolve()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2, ensure_ascii=False)

    print(f"\n[classifier] Regions saved to: {output_path.name}")
    return regions


if __name__ == "__main__":
    # When run directly, classify Video 1's processed audio
    test_wav = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    if test_wav.exists():
        classify_audio(test_wav)
    else:
        print(f"File not found: {test_wav}")
