"""
src/visualizer.py
Stage 3: Audio Visualization (Waveform and Spectrogram)

Generates memory-efficient waveforms and log-frequency spectrograms using
SoundFile, SciPy, Librosa, NumPy, and Matplotlib. Supports both full-file overview
envelope plots and high-resolution segment analysis windows.
"""

import sys
import os
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure matplotlib for headless terminal execution before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Optional, Tuple
import numpy as np
import soundfile as sf
from scipy import signal


PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def load_audio_window(
    wav_path: Path,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Reads an audio slice directly from disk without loading the entire multi-hour file.
    Returns: (audio_array, sample_rate, actual_start_sec)
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    info = sf.info(str(wav_path))
    sr = info.samplerate
    start_frame = int(start_sec * sr)
    frames_to_read = int(duration_sec * sr) if duration_sec is not None else -1

    audio, _ = sf.read(str(wav_path), start=start_frame, frames=frames_to_read, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr, start_sec


def plot_waveform_and_spectrogram(
    wav_path: Path,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    output_path: Optional[Path] = None,
    title_prefix: str = "Audio Segment",
    nperseg: int = 2048,
    noverlap: int = 1536
) -> Path:
    """
    Renders a synchronized dual-panel plot:
    - Top: Time-domain Waveform (Amplitude vs Time in seconds)
    - Bottom: Log-Frequency Spectrogram (Frequency in Hz vs Time in seconds, dB colorbar)
    """
    audio, sr, actual_start = load_audio_window(wav_path, start_sec, duration_sec)
    num_samples = len(audio)
    time_axis = actual_start + np.arange(num_samples) / sr

    # Compute Spectrogram using SciPy (deterministic, high-performance C STFT)
    # nperseg=2048 (~128ms at 16kHz) for fine harmonic frequency resolution
    # noverlap=1536 (75% overlap) for smooth temporal tracking
    freqs, seg_times, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum"
    )

    # Convert power spectrum to decibels relative to peak
    eps = 1e-10
    Sxx_db = 10 * np.log10(np.maximum(Sxx, eps))
    Sxx_db -= np.max(Sxx_db)  # Normalize peak to 0 dB

    # Offset spectrogram time axis to match audio start time
    spec_time_axis = actual_start + seg_times

    fig, (ax_wave, ax_spec) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.5]}
    )

    # 1. Waveform Plot
    ax_wave.plot(time_axis, audio, color="#1f77b4", linewidth=0.8, alpha=0.9)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.grid(True, linestyle="--", alpha=0.5)
    ax_wave.set_title(
        f"{title_prefix}: {Path(wav_path).name} [{actual_start:.2f}s - {actual_start + (num_samples/sr):.2f}s]",
        fontsize=12,
        fontweight="bold"
    )

    # 2. Spectrogram Plot
    # Display up to 8000 Hz (Nyquist frequency for 16 kHz audio)
    mesh = ax_spec.pcolormesh(
        spec_time_axis,
        freqs,
        Sxx_db,
        shading="gouraud",
        cmap="magma",
        vmin=-60,  # 60 dB dynamic range shows speech formants cleanly while suppressing noise floor
        vmax=0
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (seconds)")
    ax_spec.set_ylim(0, 8000)

    cbar = fig.colorbar(mesh, ax=ax_spec, format="%+2.0f dB")
    cbar.set_label("Relative Energy (dB)")

    plt.tight_layout()

    if output_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLOTS_DIR / f"{Path(wav_path).stem}_spec_{int(actual_start)}s_{int(duration_sec)}s.png"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    print(f"[visualizer] Plot successfully generated: {output_path.name}")
    return output_path


def plot_full_audio_envelope(
    wav_path: Path,
    downsample_factor: int = 2000,
    output_path: Optional[Path] = None
) -> Path:
    """
    Renders an amplitude envelope for multi-hour recordings using min/max decimation.
    Processes in chunks to maintain strict memory efficiency.
    """
    wav_path = Path(wav_path).resolve()
    info = sf.info(str(wav_path))
    sr = info.samplerate
    total_frames = info.frames

    print(f"[visualizer] Computing decimated envelope for {wav_path.name} ({total_frames} frames)...")

    # Read audio in 10-minute chunks
    chunk_size = sr * 60 * 10
    min_peaks = []
    max_peaks = []

    with sf.SoundFile(str(wav_path)) as f:
        while f.tell() < total_frames:
            chunk = f.read(frames=chunk_size, dtype="float32")
            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)

            num_bins = len(chunk) // downsample_factor
            if num_bins == 0:
                continue
            truncated = chunk[: num_bins * downsample_factor].reshape(num_bins, downsample_factor)
            min_peaks.append(np.min(truncated, axis=1))
            max_peaks.append(np.max(truncated, axis=1))

    all_min = np.concatenate(min_peaks)
    all_max = np.concatenate(max_peaks)
    time_axis_min = np.linspace(0, total_frames / sr / 60.0, len(all_min))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(time_axis_min, all_min, all_max, color="#2ca02c", alpha=0.7)
    ax.set_title(f"Full Recording Amplitude Envelope: {wav_path.name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLOTS_DIR / f"{wav_path.stem}_full_envelope.png"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    print(f"[visualizer] Full envelope plot generated: {output_path.name}")
    return output_path


if __name__ == "__main__":
    test_wav = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    if test_wav.exists():
        print(f"Generating sample plots for: {test_wav.name}")
        # 10-second slice starting at 60s
        plot_waveform_and_spectrogram(test_wav, start_sec=60.0, duration_sec=10.0)
        # Full recording envelope
        plot_full_audio_envelope(test_wav)
    else:
        print(f"File not found: {test_wav}")
