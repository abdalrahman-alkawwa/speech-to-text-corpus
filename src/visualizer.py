"""
src/visualizer.py
=================
Stage 3: Audio Visualization (Waveform and Spectrogram)

Purpose:
--------
Human annotators (and researchers reviewing the dataset) need intuitive,
high-precision visual inspection of audio files to:
  1. Identify where speech begins and pauses occur.
  2. Visually distinguish between speech, music, silence, and mixed audio.
  3. Verify segmentation boundaries without having to listen to entire recordings.

This module provides two visualization workflows:
  - plot_waveform_and_spectrogram(): Renders a synchronized dual-panel plot
    for any chosen time window (e.g. 5–15 seconds).
      * Top Panel: Time-domain Waveform (Amplitude vs Time).
      * Bottom Panel: Log-frequency Spectrogram computed via STFT (Frequency vs Time in dB).
  - plot_full_audio_envelope(): Renders an overview amplitude envelope of the
    entire multi-hour recording using peak decimation, ensuring zero RAM bloat.
"""

# sys: System parameters and module search path
import sys

# os: Operating system interface for environment variables
import os

# Path: Object-oriented path handling
from pathlib import Path

# --- Python Path Configuration ---
# Locate the project root folder (C:\Speech_2_Text\speech-to-text-corpus)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Guarantee project root is in sys.path for direct script execution
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Matplotlib Headless Configuration ---
# 1. Direct Matplotlib's cache to our project .cache folder so it doesn't fail
#    on restricted Windows user appdata permissions.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

# 2. Select the 'Agg' non-interactive backend BEFORE importing pyplot.
#    'Agg' renders directly to image files (PNG/PDF) in memory without trying to
#    connect to a Windows GUI display server or window manager. This prevents hangs.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Typing hints for clear function signatures
from typing import Optional, Tuple

# numpy: Fast, vectorised array operations and math
import numpy as np

# soundfile: Direct, low-overhead C-library audio file I/O
import soundfile as sf

# scipy.signal: Advanced signal-processing library (used for STFT spectrogram computation)
from scipy import signal


# Destination folder for generated plots: outputs/plots/
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def load_audio_window(
    wav_path: Path,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None
) -> Tuple[np.ndarray, int, float]:
    """
    Selectively reads an audio time window directly from disk.

    Why this function is critical for memory management:
    ----------------------------------------------------
    A 95-minute 16 kHz recording contains ~91.8 million audio samples. Loading
    the entire array into memory requires ~367 MB of raw floats, and performing
    operations on it will quickly exhaust system RAM.

    By using soundfile's start frame seeking, we read ONLY the requested frames
    (e.g., 10 seconds = 160,000 samples) directly off disk in milliseconds.

    Parameters:
    -----------
    wav_path : Path
        Path to the 16 kHz WAV file.
    start_sec : float, default=0.0
        Starting offset in seconds.
    duration_sec : float, optional
        Duration to read in seconds. If None, reads from start_sec to the end of the file.

    Returns:
    --------
    tuple : (audio_array: np.ndarray, sample_rate: int, actual_start_sec: float)
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Inspect WAV header to find the sampling rate without reading the audio data
    info = sf.info(str(wav_path))
    sr = info.samplerate

    # Convert start time in seconds to the exact audio frame number (frame = time * sample_rate)
    start_frame = int(start_sec * sr)

    # Convert duration in seconds to total frames to read (-1 means read until end of file)
    frames_to_read = int(duration_sec * sr) if duration_sec is not None else -1

    # Read strictly the requested slice into a 32-bit floating point NumPy array (values between -1.0 and +1.0)
    audio, _ = sf.read(str(wav_path), start=start_frame, frames=frames_to_read, dtype="float32")

    # Safety: If audio has multiple channels (stereo), average across channels to obtain mono
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
      - Top: Time-domain Waveform (Amplitude vs Time in seconds).
      - Bottom: Log-Frequency Spectrogram (Frequency in Hz vs Time in seconds, dB colormap).

    Signal Processing Parameters Explained:
    ---------------------------------------
    - nperseg = 2048:
        The length of each Fourier transform window (2048 samples = 128 ms at 16 kHz).
        A 128 ms window provides fine frequency resolution (~7.8 Hz per bin), allowing
        individual vowel harmonic formants and musical pitch tracks to be clearly visible.
    - noverlap = 1536:
        The number of overlapping samples between successive FFT windows (75% overlap).
        High overlap produces smooth temporal transitions along the horizontal time axis.
    - window = "hann":
        A Hann tapering window applied to each frame to minimize spectral leakage (frequency
        smearing at window edges).

    Parameters:
    -----------
    wav_path : Path
        Path to the WAV file.
    start_sec : float
        Start timestamp in seconds.
    duration_sec : float
        Duration of the slice to plot.
    output_path : Path, optional
        Target destination for the saved PNG.
    title_prefix : str
        Title displayed on the waveform header.

    Returns:
    --------
    Path: Path to the generated PNG image.
    """
    # 1. Load the requested audio window from disk
    audio, sr, actual_start = load_audio_window(wav_path, start_sec, duration_sec)
    num_samples = len(audio)

    # Construct the time array (in absolute seconds) for the time-domain waveform
    time_axis = actual_start + np.arange(num_samples) / sr

    # 2. Compute the Short-Time Fourier Transform (STFT) Spectrogram using SciPy
    # freqs: 1D array of frequency bin centers (from 0 Hz up to Nyquist: 8000 Hz)
    # seg_times: 1D array of relative time centers for each FFT window
    # Sxx: 2D array of raw spectral power [frequency_bins, time_windows]
    freqs, seg_times, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum"
    )

    # 3. Convert Power Spectrum to Decibels (dB)
    # Human loudness perception is logarithmic, not linear.
    # eps (1e-10) prevents log(0) which would yield negative infinity.
    eps = 1e-10
    Sxx_db = 10 * np.log10(np.maximum(Sxx, eps))

    # Normalize relative to the peak power in this window so peak is 0 dB
    Sxx_db -= np.max(Sxx_db)

    # Align spectrogram time axis with the actual audio start time
    spec_time_axis = actual_start + seg_times

    # 4. Create the Matplotlib Figure with two vertically stacked subplots
    fig, (ax_wave, ax_spec) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 6),
        sharex=True,                                  # Synchronize horizontal zoom/pan between both plots
        gridspec_kw={"height_ratios": [1, 1.5]}      # Allocate 40% height to waveform, 60% to spectrogram
    )

    # --- Upper Subplot: Waveform ---
    ax_wave.plot(time_axis, audio, color="#1f77b4", linewidth=0.8, alpha=0.9)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_ylim(-1.05, 1.05)                    # Audio samples are normalized between -1.0 and +1.0
    ax_wave.grid(True, linestyle="--", alpha=0.5)
    ax_wave.set_title(
        f"{title_prefix}: {Path(wav_path).name} [{actual_start:.2f}s - {actual_start + (num_samples/sr):.2f}s]",
        fontsize=12,
        fontweight="bold"
    )

    # --- Lower Subplot: Spectrogram ---
    # pcolormesh renders the 2D matrix as a colored heatmap
    # cmap="magma": dark purple for low energy, bright orange/yellow for high energy
    # vmin=-60, vmax=0: a 60 dB dynamic range cleanly reveals speech formants while suppressing background floor
    mesh = ax_spec.pcolormesh(
        spec_time_axis,
        freqs,
        Sxx_db,
        shading="gouraud",
        cmap="magma",
        vmin=-60,
        vmax=0
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (seconds)")
    ax_spec.set_ylim(0, 8000)                        # 8000 Hz is the Nyquist limit for 16 kHz audio

    # Add a colorbar indicating the Decibel scale
    cbar = fig.colorbar(mesh, ax=ax_spec, format="%+2.0f dB")
    cbar.set_label("Relative Energy (dB)")

    # Adjust layout to eliminate whitespace
    plt.tight_layout()

    # Determine output file path
    if output_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLOTS_DIR / f"{Path(wav_path).stem}_spec_{int(actual_start)}s_{int(duration_sec)}s.png"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure to PNG at 150 DPI for crisp resolution
    fig.savefig(str(output_path), dpi=150)

    # Close figure to free Matplotlib memory
    plt.close(fig)

    print(f"[visualizer] Plot successfully generated: {output_path.name}")
    return output_path


def plot_full_audio_envelope(
    wav_path: Path,
    downsample_factor: int = 2000,
    output_path: Optional[Path] = None
) -> Path:
    """
    Renders an amplitude envelope of a multi-hour recording using peak decimation.

    How Peak Decimation Works:
    --------------------------
    Instead of plotting all 91,852,939 samples, audio is divided into bins of 2,000 samples.
    For each bin, we compute the maximum positive peak and the minimum negative peak.
    This reduces the data size by 2,000x (~46,000 points), rendering instantly with zero lag.

    Audio is processed in 10-minute chunks to keep RAM usage constant and minimal.
    """
    wav_path = Path(wav_path).resolve()
    info = sf.info(str(wav_path))
    sr = info.samplerate
    total_frames = info.frames

    print(f"[visualizer] Computing decimated envelope for {wav_path.name} ({total_frames} frames)...")

    # Read in 10-minute chunks (16000 samples/sec * 60 sec * 10 min = 9,600,000 samples per chunk)
    chunk_size = sr * 60 * 10
    min_peaks = []
    max_peaks = []

    # Stream through the file sequentially
    with sf.SoundFile(str(wav_path)) as f:
        while f.tell() < total_frames:
            chunk = f.read(frames=chunk_size, dtype="float32")
            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)

            # Reshape chunk into bins of length downsample_factor
            num_bins = len(chunk) // downsample_factor
            if num_bins == 0:
                continue

            truncated = chunk[: num_bins * downsample_factor].reshape(num_bins, downsample_factor)
            # Find the minimum and maximum sample in each bin
            min_peaks.append(np.min(truncated, axis=1))
            max_peaks.append(np.max(truncated, axis=1))

    # Combine all chunk peaks
    all_min = np.concatenate(min_peaks)
    all_max = np.concatenate(max_peaks)

    # Build time axis in minutes
    time_axis_min = np.linspace(0, total_frames / sr / 60.0, len(all_min))

    # Render envelope plot
    fig, ax = plt.subplots(figsize=(14, 4))
    # fill_between shades the area between the negative peaks and positive peaks
    ax.fill_between(time_axis_min, all_min, all_max, color="#2ca02c", alpha=0.7)
    ax.set_title(f"Full Recording Amplitude Envelope: {wav_path.name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Determine destination path
    if output_path is None:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PLOTS_DIR / f"{wav_path.stem}_full_envelope.png"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to disk
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    print(f"[visualizer] Full envelope plot generated: {output_path.name}")
    return output_path


# --- Standalone Execution Example ---
if __name__ == "__main__":
    test_wav = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    if test_wav.exists():
        print(f"Generating sample plots for: {test_wav.name}")
        # 1. Detailed 10-second slice starting at second 60.0
        plot_waveform_and_spectrogram(test_wav, start_sec=60.0, duration_sec=10.0)
        # 2. Complete 95-minute overview envelope
        plot_full_audio_envelope(test_wav)
    else:
        print(f"File not found: {test_wav}")
