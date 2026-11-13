"""
tests/test_visualizer.py
Test suite for Stage 3 audio visualizer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualizer import load_audio_window, plot_waveform_and_spectrogram, plot_full_audio_envelope


def test_load_audio_window():
    print("[TEST 1] Testing selective audio window loading from disk...")
    wav_path = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    audio, sr, start = load_audio_window(wav_path, start_sec=30.0, duration_sec=5.0)

    print(f"  -> Requested: 5.0 seconds starting at 30.0s")
    print(f"  -> Loaded samples: {len(audio)} at {sr} Hz ({len(audio)/sr:.2f} seconds)")
    assert sr == 16000
    assert len(audio) == 16000 * 5
    print("  -> PASSED\n")


def test_plot_segment_waveform_and_spectrogram():
    print("[TEST 2] Testing waveform and spectrogram generation...")
    wav_path = PROJECT_ROOT / "data" / "processed" / "gK017cCH36k.wav"
    out_img = PROJECT_ROOT / "outputs" / "plots" / "test_spec_30s_10s.png"

    plot_path = plot_waveform_and_spectrogram(
        wav_path=wav_path,
        start_sec=30.0,
        duration_sec=10.0,
        output_path=out_img,
        title_prefix="Unit Test Speech Window"
    )

    print(f"  -> Generated plot at: {plot_path.name} ({plot_path.stat().st_size / 1024:.1f} KB)")
    assert plot_path.exists()
    assert plot_path.stat().st_size > 5000  # Non-trivial image
    print("  -> PASSED\n")


if __name__ == "__main__":
    print("=========================================")
    print(" Running Stage 3: Audio Visualizer Tests ")
    print("=========================================\n")
    test_load_audio_window()
    test_plot_segment_waveform_and_spectrogram()
    print("ALL VISUALIZER TESTS PASSED SUCCESSFULLY!")
