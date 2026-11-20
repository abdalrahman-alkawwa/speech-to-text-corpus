"""
src/classifier.py
==================
Stage 4: Region Classification (Speech / Music / Silence / Mixed)

This module classifies a 16 kHz WAV recording into frame-level labels and
merges those labels into time regions.

Important limitations:
- Silero VAD detects voice activity; it is not a four-class classifier.
- "music" means energetic audio with low VAD confidence. Sound effects and
  noise can receive this label too.
- "mixed" means uncertain VAD output (0.3 <= score < 0.5); it does not
  prove that speech and music overlap.
- Confidence values are heuristic, not calibrated probabilities.
- Automatic labels are suggestions and should be reviewed by an annotator.
"""

import sys
import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
import silero_vad


# --- Windows NVIDIA DLL setup ---
# Register NVIDIA DLL folders when they exist. This is harmless on CPU-only
# installations and can help PyTorch locate its bundled CUDA libraries.
if sys.platform == "win32":
    try:
        import site

        site_packages = site.getsitepackages()
        if site_packages:
            nvidia_dir = Path(site_packages[-1]) / "nvidia"
            if nvidia_dir.exists():
                for bin_dir in nvidia_dir.glob("*/bin"):
                    os.add_dll_directory(str(bin_dir))
    except (OSError, AttributeError, IndexError):
        pass


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
FRAME_SIZE_SAMPLES = 800             # 50 ms at 16 kHz
VAD_WINDOW_SAMPLES = 512             # 32 ms at 16 kHz

SILENCE_THRESHOLD_DB = -31.0
VAD_SPEECH_THRESHOLD = 0.5
VAD_UNCERTAIN_LOW = 0.3

# Keep this fairly small so short pauses remain visible to the annotator.
# This is a cleanup setting, not the final 5–15 second ASR segmentation rule.
MIN_REGION_DURATION_SEC = 1.5


def load_vad_model():
    """Load the lightweight Silero Voice Activity Detection model."""
    return silero_vad.load_silero_vad()


def _validate_audio_array(audio: np.ndarray) -> np.ndarray:
    """Validate and normalize an audio array to one-dimensional float32."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        # soundfile returns stereo as (samples, channels).
        audio = np.mean(audio, axis=1)
    elif audio.ndim != 1:
        raise ValueError(
            f"Expected mono or stereo audio, received array with shape {audio.shape}."
        )

    if audio.size == 0:
        raise ValueError("Audio file contains no samples.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite sample values.")

    return np.ascontiguousarray(audio, dtype=np.float32)


def compute_frame_energy_db(
    audio: np.ndarray,
    frame_size: int = FRAME_SIZE_SAMPLES
) -> np.ndarray:
    """
    Compute RMS energy in dBFS for non-overlapping frames.

    RMS = sqrt(mean(samples ** 2))
    dBFS = 20 * log10(RMS)

    Only complete frames are included. Any trailing samples shorter than one
    frame are ignored consistently by the frame-based classifier.
    """
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than zero.")

    num_frames = len(audio) // frame_size
    if num_frames == 0:
        return np.empty(0, dtype=np.float32)

    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    energy_db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    return energy_db.astype(np.float32)


def compute_frame_spectral_flatness(
    audio: np.ndarray,
    frame_size: int = FRAME_SIZE_SAMPLES
) -> np.ndarray:
    """
    Compute spectral flatness for diagnostics.

    """
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than zero.")

    num_frames = len(audio) // frame_size
    if num_frames == 0:
        return np.empty(0, dtype=np.float32)

    frames = audio[:num_frames * frame_size].reshape(num_frames, frame_size)
    windowed_frames = frames * np.hanning(frame_size)
    power_spectrum = np.abs(np.fft.rfft(windowed_frames, axis=1)) ** 2
    power_spectrum = np.maximum(power_spectrum, 1e-10)

    geometric_mean = np.exp(np.mean(np.log(power_spectrum), axis=1))
    arithmetic_mean = np.mean(power_spectrum, axis=1)
    flatness = geometric_mean / (arithmetic_mean + 1e-10)

    return np.clip(flatness, 0.0, 1.0).astype(np.float32)


def run_silero_vad_chunked(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    chunk_duration_sec: float = 30.0,
    model=None
) -> np.ndarray:
    """
    Run Silero VAD on audio in chunks and map its predictions
    onto the 50 ms analysis frames using overlap-weighted averaging.

    Supplying an already-loaded model allows reuse across recordings.
    """
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Silero VAD expects {SAMPLE_RATE} Hz audio; got {sr} Hz."
        )

    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be greater than zero.")

    audio = _validate_audio_array(audio)

    # Align chunk boundaries with analysis-frame boundaries.
    requested_samples = int(round(chunk_duration_sec * sr))
    chunk_samples = max(
        FRAME_SIZE_SAMPLES,
        (requested_samples // FRAME_SIZE_SAMPLES) * FRAME_SIZE_SAMPLES
    )

    total_frames = len(audio) // FRAME_SIZE_SAMPLES
    frame_probs = np.zeros(total_frames, dtype=np.float32)

    if total_frames == 0:
        return frame_probs

    if model is None:
        model = load_vad_model()

    def get_vad_probabilities(chunk: np.ndarray) -> np.ndarray:
        """Run Silero on all complete 512-sample windows in a chunk."""
        chunk_tensor = torch.from_numpy(chunk)

        probabilities = []

        for start in range(
            0,
            len(chunk_tensor) - VAD_WINDOW_SAMPLES + 1,
            VAD_WINDOW_SAMPLES
        ):
            window = chunk_tensor[start:start + VAD_WINDOW_SAMPLES]
            probability = model(window, sr).item()
            probabilities.append(probability)

        return np.asarray(probabilities, dtype=np.float32)

    def map_vad_to_frames(
        probabilities: np.ndarray,
        chunk_start: int,
        chunk_end: int
    ) -> None:
        """Map VAD window probabilities onto overlapping analysis frames."""
        first_frame = chunk_start // FRAME_SIZE_SAMPLES
        last_frame = min(
            total_frames,
            chunk_end // FRAME_SIZE_SAMPLES
        )

        for frame_index in range(first_frame, last_frame):
            frame_start = (
                frame_index * FRAME_SIZE_SAMPLES - chunk_start
            )
            frame_end = frame_start + FRAME_SIZE_SAMPLES

            first_window = frame_start // VAD_WINDOW_SAMPLES
            last_window = (
                frame_end + VAD_WINDOW_SAMPLES - 1
            ) // VAD_WINDOW_SAMPLES

            weighted_sum = 0.0
            covered_samples = 0

            for window_index in range(
                first_window,
                min(last_window, len(probabilities))
            ):
                window_start = window_index * VAD_WINDOW_SAMPLES
                window_end = window_start + VAD_WINDOW_SAMPLES

                overlap = max(
                    0,
                    min(frame_end, window_end)
                    - max(frame_start, window_start)
                )

                if overlap:
                    weighted_sum += (
                        float(probabilities[window_index]) * overlap
                    )
                    covered_samples += overlap

            if covered_samples:
                frame_probs[frame_index] = (
                    weighted_sum / covered_samples
                )

    with torch.inference_mode():
        for chunk_start in range(0, len(audio), chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, len(audio))
            chunk = audio[chunk_start:chunk_end]

            model.reset_states()

            probabilities = get_vad_probabilities(chunk)

            if len(probabilities) == 0:
                continue

            map_vad_to_frames(
                probabilities,
                chunk_start,
                chunk_end
            )

    return frame_probs


def classify_frames(
    energy_db: np.ndarray,
    vad_probs: np.ndarray,
    spectral_flatness: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Fuse frame energy and VAD scores into speech/silence/music/mixed labels.

    `spectral_flatness` is retained as an optional argument for compatibility
    with existing callers, but it is diagnostic only and does not determine
    how many frames are classified or affect the labels.
    """
    if len(energy_db) != len(vad_probs):
        raise ValueError(
            "energy_db and vad_probs must contain the same number of frames "
            f"(got {len(energy_db)} and {len(vad_probs)})."
        )

    frame_labels = []

    for i, (energy, vad_probability) in enumerate(zip(energy_db, vad_probs)):
        energy = float(energy)
        vad_probability = float(vad_probability)
        low_energy = energy < SILENCE_THRESHOLD_DB

        if vad_probability >= VAD_SPEECH_THRESHOLD:
            label = "speech"
            confidence = vad_probability

        elif vad_probability >= VAD_UNCERTAIN_LOW:
            label = "mixed"
            confidence = 0.5 - abs(vad_probability - 0.4) * 2

        elif low_energy:
            label = "silence"
            confidence = min(
                1.0,
                (1.0 - vad_probability) * 0.5
                + (abs(energy) / 80.0) * 0.5
            )

        else:
            label = "music"
            confidence = min(1.0, (1.0 - vad_probability) * 0.7)

        frame_labels.append({
            "frame_index": i,
            "label": label,
            "confidence": round(
                max(0.05, min(1.0, confidence)), 3
            )
        })

    return frame_labels


def _merge_two_regions(
    left: Dict[str, Any],
    right: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge adjacent regions and calculate duration-weighted confidence."""
    left_duration = left["end_sec"] - left["start_sec"]
    right_duration = right["end_sec"] - right["start_sec"]
    total_duration = left_duration + right_duration

    if total_duration > 0:
        confidence = (
            left["confidence"] * left_duration
            + right["confidence"] * right_duration
        ) / total_duration
    else:
        confidence = left["confidence"]

    return {
        "start_sec": left["start_sec"],
        "end_sec": right["end_sec"],
        "label": left["label"],
        "confidence": round(float(confidence), 3)
    }


def merge_frames_into_regions(
    frame_labels: List[Dict[str, Any]],
    sr: int = SAMPLE_RATE,
    frame_size: int = FRAME_SIZE_SAMPLES,
    min_duration: float = MIN_REGION_DURATION_SEC
) -> List[Dict[str, Any]]:
    """
    Merge consecutive frame labels into timestamped regions.

    Regions shorter than `min_duration` are absorbed into a neighboring
    region. A short first region is absorbed forward; later short regions are
    absorbed backward. This is only light smoothing: it is not the separate
    5–15 second segmentation stage required for ASR.
    """
    if sr <= 0 or frame_size <= 0:
        raise ValueError("sr and frame_size must be greater than zero.")
    if min_duration < 0:
        raise ValueError("min_duration cannot be negative.")
    if not frame_labels:
        return []

    frame_duration = frame_size / sr

    # First, group consecutive frames that have the same label.
    raw_regions = []
    current_label = frame_labels[0]["label"]
    current_start = 0
    confidence_values = [float(frame_labels[0]["confidence"])]

    for i in range(1, len(frame_labels)):
        label = frame_labels[i]["label"]

        if label != current_label:
            raw_regions.append({
                "start_sec": round(current_start * frame_duration, 3),
                "end_sec": round(i * frame_duration, 3),
                "label": current_label,
                "confidence": round(float(np.mean(confidence_values)), 3)
            })
            current_label = label
            current_start = i
            confidence_values = [float(frame_labels[i]["confidence"])]
        else:
            confidence_values.append(float(frame_labels[i]["confidence"]))

    raw_regions.append({
        "start_sec": round(current_start * frame_duration, 3),
        "end_sec": round(len(frame_labels) * frame_duration, 3),
        "label": current_label,
        "confidence": round(float(np.mean(confidence_values)), 3)
    })

    if min_duration == 0 or len(raw_regions) == 1:
        final_regions = raw_regions
    else:
        # Repeatedly absorb short regions until all remaining regions meet
        # the duration threshold, or only one region remains.
        regions = [dict(region) for region in raw_regions]

        index = 0
        while index < len(regions) and len(regions) > 1:
            duration = regions[index]["end_sec"] - regions[index]["start_sec"]

            if duration >= min_duration:
                index += 1
                continue

            if index == 0:
                # No preceding region exists: merge the first region forward.
                left = regions[0]
                right = regions[1]
                merged = _merge_two_regions(
                    {**left, "label": right["label"]},
                    right
                )
                # Preserve the right region's label while spanning both.
                merged["label"] = right["label"]
                regions[1] = merged
                del regions[0]
                index = 0
            else:
                # Merge this short region backward, preserving the preceding
                # region's label.
                left = regions[index - 1]
                right = regions[index]
                merged = _merge_two_regions(left, right)
                merged["label"] = left["label"]
                regions[index - 1] = merged
                del regions[index]
                index = max(0, index - 1)

        final_regions = regions

    # Adjacent regions can now share a label after short-region absorption.
    compacted = []
    for region in final_regions:
        if compacted and compacted[-1]["label"] == region["label"]:
            compacted[-1] = _merge_two_regions(compacted[-1], region)
        else:
            compacted.append(region.copy())

    for region in compacted:
        region["method"] = "energy_vad_v2"

    return compacted


def classify_audio(
    wav_path: Path,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Classify a 16 kHz WAV file and save its regions as JSON.

    Stereo input is downmixed to mono. Files with any other sample rate are
    rejected rather than silently being analyzed with incorrect timing/VAD
    assumptions.
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    if not wav_path.is_file():
        raise ValueError(f"Audio path is not a file: {wav_path}")

    print(f"\n[classifier] Starting region classification for: {wav_path.name}")
    print("[classifier] Loading audio...")

    audio, sr = sf.read(str(wav_path), dtype="float32")

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected a {SAMPLE_RATE} Hz WAV file, but '{wav_path.name}' "
            f"has a sample rate of {sr} Hz. Resample it before classification."
        )

    audio = _validate_audio_array(audio)
    total_duration = len(audio) / sr

    print(
        f"[classifier] Loaded {total_duration:.1f}s of audio "
        f"({len(audio)} samples at {sr} Hz, mono)"
    )

    # Compute frame-level acoustic features.
    print("[classifier] Computing frame-level RMS energy and spectral flatness...")
    energy_db = compute_frame_energy_db(audio)
    flatness = compute_frame_spectral_flatness(audio)
    print(
        f"[classifier] Acoustic features computed for "
        f"{len(energy_db)} frames (50 ms each)"
    )
    if len(flatness) != len(energy_db):
        raise RuntimeError("Energy and spectral-flatness frame counts do not match.")

    # Run voice activity detection.
    print("[classifier] Running Silero VAD (this may take a minute for long files)...")
    vad_probs = run_silero_vad_chunked(audio, sr=sr)
    print(f"[classifier] VAD completed for {len(vad_probs)} frames")

    # Fuse the two signals into frame labels.
    print("[classifier] Applying classification decision rules...")
    frame_labels = classify_frames(
        energy_db,
        vad_probs,
        spectral_flatness=flatness
    )

    # Merge frame labels into continuous regions.
    regions = merge_frames_into_regions(frame_labels, sr=sr)

    # Print a summary.
    label_counts = {}
    label_durations = {}

    for region in regions:
        duration = region["end_sec"] - region["start_sec"]
        label = region["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        label_durations[label] = label_durations.get(label, 0.0) + duration

    classified_duration = (
        len(frame_labels) * FRAME_SIZE_SAMPLES / sr
    )

    print(
        f"\n[classifier] Classification Summary "
        f"({len(regions)} regions, {classified_duration:.1f}s classified):"
    )
    print(f"  {'Label':<10} {'Count':>6} {'Duration':>10} {'Percentage':>10}")
    print(f"  {'-' * 10} {'-' * 6} {'-' * 10} {'-' * 10}")

    for label in ["speech", "silence", "music", "mixed"]:
        count = label_counts.get(label, 0)
        duration = label_durations.get(label, 0.0)
        percentage = (
            duration / classified_duration * 100
            if classified_duration > 0 else 0
        )
        print(
            f"  {label:<10} {count:>6} "
            f"{duration:>9.1f}s {percentage:>9.1f}%"
        )

    # Save the region list.
    if output_path is None:
        output_path = wav_path.parent / f"{wav_path.stem}_regions.json"

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(regions, file, indent=2, ensure_ascii=False)

    print(f"\n[classifier] Regions saved to: {output_path}")
    return regions


if __name__ == "__main__":
    test_wav = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"

    if test_wav.exists():
        classify_audio(test_wav)
    else:
        print(f"File not found: {test_wav}")
