"""
Tests for Stage 6 (src/snr_estimator.py).

Run directly:   python tests/test_snr_estimator.py
Or with pytest: pytest tests/test_snr_estimator.py

All tests use synthetic audio with a KNOWN SNR, so they check the maths
without needing your real recordings.  A final optional check runs on your
real segments if any exist.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import snr_estimator as snr  # noqa: E402

SR = 16_000


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _gamma_speech(n: int, rng: np.random.Generator) -> np.ndarray:
    """Samples from the speech model WADA assumes: symmetric Gamma(0.4)."""
    return rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)


def _mix_at_snr(speech: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(speech.size)
    noise *= np.sqrt(np.mean(speech ** 2) / 10 ** (snr_db / 10) / np.mean(noise ** 2))
    return speech + noise


def _bursty_speech(duration_sec: float, noise_rms: float, snr_db: float, rng) -> np.ndarray:
    """Syllable-like bursts (0.3 s on / 0.2 s off) over a constant noise floor.

    The active (on) frames have power = noise power * 10^(snr/10) + noise power,
    so the 'active speech SNR' is exactly snr_db.
    """
    n = int(duration_sec * SR)
    t = np.arange(n) / SR
    on = ((t % 0.5) < 0.3).astype(float)
    sig_rms = noise_rms * 10 ** (snr_db / 20)
    speech = on * rng.standard_normal(n) * sig_rms
    noise = rng.standard_normal(n) * noise_rms
    return speech + noise


# ----------------------------- WADA ------------------------------------------
def test_wada_pure_noise_hits_floor():
    x = _rng(1).standard_normal(SR * 10)
    est = snr.estimate_wada_snr(x)
    assert est is not None and est < -15.0, est


def test_wada_tracks_known_snr():
    for target in (0.0, 10.0, 20.0):
        rng = _rng(int(target) + 7)
        x = _mix_at_snr(_gamma_speech(SR * 10, rng), target, rng)
        est = snr.estimate_wada_snr(x)
        assert est is not None and abs(est - target) < 2.0, (target, est)


def test_wada_is_scale_invariant():
    rng = _rng(3)
    x = _mix_at_snr(_gamma_speech(SR * 10, rng), 10.0, rng)
    a = snr.estimate_wada_snr(x)
    b = snr.estimate_wada_snr(0.05 * x)
    assert abs(a - b) < 1e-6, (a, b)


def test_wada_silence_and_empty_return_none():
    assert snr.estimate_wada_snr(np.zeros(SR)) is None
    assert snr.estimate_wada_snr(np.array([])) is None


# ------------------------- Segmental / frame-energy ---------------------------
def test_segmental_tracks_known_snr_in_clip():
    for target in (5.0, 15.0, 25.0):
        x = _bursty_speech(10.0, noise_rms=0.001, snr_db=target, rng=_rng(int(target)))
        est = snr.estimate_segmental_snr(x)
        assert est is not None and abs(est - target) < 2.0, (target, est)


def test_segmental_with_explicit_noise_reference():
    x = _bursty_speech(10.0, noise_rms=0.001, snr_db=20.0, rng=_rng(5))
    est = snr.estimate_segmental_snr(x, noise_power=0.001 ** 2)
    assert est is not None and abs(est - 20.0) < 1.5, est


def test_segmental_none_cases():
    assert snr.estimate_segmental_snr(np.zeros(SR)) is None                       # digital silence
    assert snr.estimate_segmental_snr(np.ones(100)) is None                       # shorter than 4 frames
    flat = _rng(9).standard_normal(SR * 5) * 0.01                                 # noise only
    assert snr.estimate_segmental_snr(flat) is None                               # nothing 10 dB above floor


# ------------------------------ Regions ---------------------------------------
def test_noise_power_from_regions_recovers_room_tone():
    rng = _rng(11)
    noise_rms = 0.002
    audio = np.concatenate([
        rng.standard_normal(SR * 4) * noise_rms,         # 0-4 s   silence
        rng.standard_normal(SR * 6) * 0.1,               # 4-10 s  loud (speech)
        rng.standard_normal(SR * 3) * noise_rms,         # 10-13 s silence
    ])
    regions = [
        {"start_sec": 0.0, "end_sec": 4.0, "label": "silence"},
        {"start_sec": 4.0, "end_sec": 10.0, "label": "speech"},
        {"start_sec": 10.0, "end_sec": 13.0, "label": "silence"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "full.wav"
        sf.write(str(path), audio, SR, subtype="PCM_16")
        est = snr.noise_power_from_regions(path, regions)
        assert est is not None and abs(est / noise_rms ** 2 - 1.0) < 0.1, est
        assert snr.noise_power_from_regions(path, [regions[1]]) is None


# --------------------------- File-level behaviour -----------------------------
def test_segment_record_roundtrip_and_validation():
    x = _bursty_speech(10.0, noise_rms=0.001, snr_db=20.0, rng=_rng(13))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = tmp / "abc_123_seg_000001.wav"
        sf.write(str(good), x, SR, subtype="PCM_16")
        rec = snr.estimate_snr_for_segment(good)
        assert rec["segment_id"] == "abc_123_seg_000001"
        assert abs(rec["snr_segmental_db"] - 20.0) < 2.0, rec
        assert rec["snr_noise_ref"] == "in_clip_quietest_frames"
        assert snr.video_id_from_segment(good) == "abc_123"

        bad_sr = tmp / "bad_seg_000001.wav"
        sf.write(str(bad_sr), x[:SR], 8_000, subtype="PCM_16")
        try:
            snr.estimate_snr_for_segment(bad_sr)
            raise AssertionError("expected ValueError for wrong sample rate")
        except ValueError:
            pass

        out = snr.save_snr_records("vid", [rec], metadata_dir=tmp / "meta")
        assert json.loads(out.read_text(encoding="utf-8"))[0]["segment_id"] == rec["segment_id"]


def test_compute_snr_for_video_uses_regions_when_available():
    rng = _rng(17)
    noise_rms = 0.001
    full = np.concatenate([rng.standard_normal(SR * 3) * noise_rms,
                           _bursty_speech(10.0, noise_rms, 20.0, rng)])
    seg = _bursty_speech(10.0, noise_rms, 20.0, rng)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "seg").mkdir(); (tmp / "proc").mkdir()
        sf.write(str(tmp / "seg" / "v1_seg_000001.wav"), seg, SR, subtype="PCM_16")
        sf.write(str(tmp / "proc" / "Some Title_v1.wav"), full, SR, subtype="PCM_16")
        (tmp / "proc" / "Some Title_v1_regions.json").write_text(
            json.dumps([{"start_sec": 0.0, "end_sec": 3.0, "label": "silence"},
                        {"start_sec": 3.0, "end_sec": 13.0, "label": "speech"}]), encoding="utf-8")
        recs = snr.compute_snr_for_video("v1", tmp / "seg", tmp / "proc")
        assert len(recs) == 1 and recs[0]["snr_noise_ref"] == "recording_silence_regions"
        assert abs(recs[0]["snr_segmental_db"] - 20.0) < 2.0, recs[0]


def test_real_segments_if_present():
    """Informational: runs on your real clips if data/segments has any."""
    wavs = sorted(snr.SEGMENTS_DIR.glob("*_seg_*.wav"))[:5]
    if not wavs:
        print("  (no real segments found - skipped)")
        return
    for w in wavs:
        rec = snr.estimate_snr_for_segment(w)
        print(f"  {rec['segment_id']}: WADA={rec['snr_wada_db']}  segmental={rec['snr_segmental_db']}")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
