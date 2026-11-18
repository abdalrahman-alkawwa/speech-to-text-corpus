"""
scripts/check_boundaries.py - audit every internal clip boundary of a video.

For each place where clip k is followed by clip k+1 (a cut made INSIDE a speech
region) it measures, from the audio itself, how long the pause at the cut is and
how much speech leaked into the padding on either side.  This is how you find
"cut mid-word" boundaries without listening to every clip.

    python scripts/check_boundaries.py CvrNaPFIwcQ
    python scripts/check_boundaries.py CvrNaPFIwcQ --wav data/processed/some_file.wav

Read-only.  Needs metadata/<video_id>_segments.json (written by Stage 5).
Pure numpy/soundfile: does not import torch, so it starts instantly.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAME = 320                    # 20 ms at 16 kHz
PAUSE_DEPTH_DB = 15.0          # pause = frames this far below the local speech level
LEAK_MS = 40                   # more speech than this inside the padding = suspicious


def energy_db(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    if n == 0:
        return np.empty(0)
    f = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return 20 * np.log10(np.maximum(np.sqrt((f ** 2).mean(axis=1)), 1e-10))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--wav", help="processed WAV (default: data/processed/*<video_id>*.wav)")
    ap.add_argument("--quiet", action="store_true", help="print only problem boundaries")
    a = ap.parse_args()

    man_path = PROJECT_ROOT / "metadata" / f"{a.video_id}_segments.json"
    if not man_path.exists():
        print(f"Missing {man_path}"); return 1
    recs = json.loads(man_path.read_text(encoding="utf-8"))
    wav = Path(a.wav) if a.wav else next(iter(sorted((PROJECT_ROOT / "data" / "processed").glob(f"*{a.video_id}*.wav"))), None)
    if wav is None or not wav.exists():
        print("Processed WAV not found (use --wav)"); return 1
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    fs = FRAME / sr

    e_all = energy_db(audio)
    print(f"File level: p5 {np.percentile(e_all, 5):.1f} dB | median {np.median(e_all):.1f} dB | p95 {np.percentile(e_all, 95):.1f} dB (dBFS, 20 ms frames)")
    print(f"Frames below -28 dB (the old 'verified' rule): {100 * (e_all < -28).mean():.0f}%\n")

    print(" cut      time(s)   pause_ms  tail_ms  head_ms  verdict")
    counts = {"clean": 0, "leak": 0, "short_pause": 0}
    for k in range(len(recs) - 1):
        end_k, start_n = recs[k]["end_sec"], recs[k + 1]["start_sec"]
        if start_n > end_k + 0.05:
            continue                                   # a region edge, not an internal cut
        cut = 0.5 * (end_k + start_n)
        lo, hi = max(0.0, cut - 1.0), min(len(audio) / sr, cut + 1.0)
        local = energy_db(audio[int(lo * sr):int(hi * sr)])
        if local.size < 10:
            continue
        thr = np.percentile(local, 75) - PAUSE_DEPTH_DB
        ci = int((cut - lo) / fs)
        ci = min(max(ci, 0), local.size - 1)
        # length of the low-energy run that contains (or is nearest to) the cut
        left = right = ci
        if local[ci] < thr:
            while left > 0 and local[left - 1] < thr:
                left -= 1
            while right < local.size - 1 and local[right + 1] < thr:
                right += 1
            pause_ms = (right - left + 1) * fs * 1000
        else:
            pause_ms = 0.0
        t_idx = lambda t: int((t - lo) / fs)
        tail = local[max(ci, 0):max(t_idx(end_k), ci)]
        head = local[max(t_idx(start_n), 0):ci]
        tail_ms = (tail >= thr).sum() * fs * 1000
        head_ms = (head >= thr).sum() * fs * 1000
        if tail_ms > LEAK_MS or head_ms > LEAK_MS:
            verdict, key = "SPEECH IN PADDING (likely mid-word)", "leak"
        elif pause_ms < 150:
            verdict, key = "no real pause at cut", "short_pause"
        else:
            verdict, key = "clean", "clean"
        counts[key] += 1
        if not a.quiet or key != "clean":
            print(f" {k + 1:>3}->{k + 2:<4} {cut:9.2f}  {pause_ms:8.0f} {tail_ms:8.0f} {head_ms:8.0f}  {verdict}")
    total = sum(counts.values())
    print(f"\n{total} internal cuts: {counts['clean']} clean, {counts['leak']} with speech in the padding, "
          f"{counts['short_pause']} with no real pause")
    return 0


if __name__ == "__main__":
    sys.exit(main())
