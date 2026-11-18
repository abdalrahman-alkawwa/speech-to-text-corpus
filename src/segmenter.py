"""
src/segmenter.py  -  Stage 5: Speech Segmentation (v2: pause-based cuts)
=========================================================================

Turns Stage 4 speech/mixed regions into 5-15 s clips (target 8-12 s) that do not
start or end in the middle of a word.

What changed from v1 (and why)
------------------------------
v1 cut at the single quietest 50 ms frame and then added 0.15 s of padding on
BOTH sides of EVERY cut.  That is right for the outer edges of a region (silence
follows) but wrong for a cut through continuous speech: when the pause at the
cut is shorter than 0.3 s, each clip's padding reaches into the neighbouring
word, so a clip ends with the first syllable of the next word and the next clip
starts with the tail of the previous one.  v1 also judged "is this a clean cut?"
with a fixed -28 dBFS, which says nothing about how loud THIS recording is.

v2:
 1. Finds real PAUSES: runs of 20 ms frames that are at least PAUSE_DEPTH_DB below
    the local speech level and last at least MIN_PAUSE_SEC.  Closures inside words
    (stops such as b/t/k) are shorter than that and are ignored.
 2. Cuts at the CENTRE of the longest such pause in the 8-12 s window; if none,
    it widens the window to 6-14 s; if still none it falls back to the smoothed
    energy minimum and flags the cut for manual review.
 3. At an internal cut, padding is limited to 40 % of the pause length (so it stays
    inside the pause).  The full 0.15 s cushion is kept only at the outer edges of a
    region.
 4. "auto_verified" now means "cut sits in a real pause"; each segment also records
    pause_ms and pause_depth_db so you can audit it.

Public names kept from v1: find_best_cut_point (2-tuple), find_best_cut_point_with_energy
(3-tuple), segment_speech_region, export_segments (same signature and record fields,
plus cut_energy_db, pause_ms, pause_depth_db).  boundary_method is now "energy_pause_v2".
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from src.classifier import compute_frame_energy_db, FRAME_SIZE_SAMPLES  # noqa: F401  (FRAME_SIZE_SAMPLES re-exported)

SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments"

# Clip length limits (seconds, including padding)
MIN_SEGMENT_DURATION_SEC = 5.0
MAX_SEGMENT_DURATION_SEC = 15.0
TARGET_DURATION_MIN_SEC = 8.0
TARGET_DURATION_MAX_SEC = 12.0

# Cushion at the OUTER edges of a region
PADDING_SEC = 0.15

# --- v2 pause detection ---
CUT_FRAME_SAMPLES = 320            # 20 ms frames at 16 kHz
MIN_PAUSE_SEC = 0.15               # shorter dips are word-internal closures, not pauses
PAUSE_DEPTH_DB = 15.0              # a pause frame is >= 15 dB below the local speech level
SPEECH_LEVEL_PERCENTILE = 75.0     # "local speech level" = 75th percentile of frame energy
PRIMARY_WINDOW_SEC = (TARGET_DURATION_MIN_SEC, TARGET_DURATION_MAX_SEC)
WIDE_WINDOW_SEC = (6.0, 14.0)      # used only when the primary window has no real pause
PAD_FRACTION_OF_PAUSE = 0.4        # internal-cut padding = 40 % of the pause (<= PADDING_SEC)

BOUNDARY_METHOD = "energy_pause_v2"


# --------------------------------------------------------------------------- #
# Cut-point search
# --------------------------------------------------------------------------- #
def _frame_energy(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    """20 ms frame energies (dBFS) for audio[start_sec:end_sec]; empty if too short."""
    s = max(0, int(start_sec * sr))
    e = min(len(audio), int(end_sec * sr))
    chunk = audio[s:e]
    if len(chunk) < CUT_FRAME_SAMPLES:
        return np.empty(0, dtype=np.float64)
    return compute_frame_energy_db(chunk, frame_size=CUT_FRAME_SAMPLES)


def _pause_runs(is_pause: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
    """(start, end_exclusive) of every run of True values at least min_frames long."""
    runs: List[Tuple[int, int]] = []
    i, n = 0, len(is_pause)
    while i < n:
        if is_pause[i]:
            j = i
            while j < n and is_pause[j]:
                j += 1
            if j - i >= min_frames:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def find_cut_point_details(
    audio: np.ndarray,
    sr: int,
    region_start_sec: float,
    current_cut_sec: float,
    region_end_sec: float,
) -> Dict[str, Any]:
    """Choose where the clip that starts at current_cut_sec should end.

    Returns a dict: cut_sec, status ('auto_verified' | 'flagged_manual_review'),
    cut_energy_db, pause_sec, pause_depth_db (None when not applicable).
    """
    remaining_sec = region_end_sec - current_cut_sec
    if remaining_sec <= MAX_SEGMENT_DURATION_SEC:          # the rest fits in one clip
        return {"cut_sec": region_end_sec, "status": "auto_verified",
                "cut_energy_db": None, "pause_sec": None, "pause_depth_db": None}

    frame_sec = CUT_FRAME_SAMPLES / sr
    min_frames = int(round(MIN_PAUSE_SEC / frame_sec))

    # Local speech level from the stretch this clip could cover
    span = _frame_energy(audio, sr, current_cut_sec,
                         min(current_cut_sec + MAX_SEGMENT_DURATION_SEC, region_end_sec))
    speech_db: Optional[float] = None
    if span.size >= 10:
        speech_db = float(np.percentile(span, SPEECH_LEVEL_PERCENTILE))
        threshold = speech_db - PAUSE_DEPTH_DB
        target_time = current_cut_sec + 0.5 * (PRIMARY_WINDOW_SEC[0] + PRIMARY_WINDOW_SEC[1])

        for w_lo, w_hi in (PRIMARY_WINDOW_SEC, WIDE_WINDOW_SEC):
            win_start = current_cut_sec + w_lo
            win_end = min(current_cut_sec + w_hi, region_end_sec - MIN_SEGMENT_DURATION_SEC)
            if win_end - win_start < 2 * frame_sec:
                continue
            energy = _frame_energy(audio, sr, win_start, win_end)
            if energy.size == 0:
                continue
            runs = _pause_runs(energy < threshold, min_frames)
            if not runs:
                continue

            def score(run: Tuple[int, int]) -> Tuple[int, float]:
                centre_t = win_start + 0.5 * (run[0] + run[1]) * frame_sec
                return (run[1] - run[0], -abs(centre_t - target_time))   # longest, then nearest 10 s

            lo, hi = max(runs, key=score)
            mean_db = float(np.mean(energy[lo:hi]))
            return {"cut_sec": win_start + 0.5 * (lo + hi) * frame_sec,
                    "status": "auto_verified",
                    "cut_energy_db": mean_db,
                    "pause_sec": (hi - lo) * frame_sec,
                    "pause_depth_db": speech_db - mean_db}

    # Fallback: no real pause anywhere -> smoothed energy minimum, flagged for review
    win_start = current_cut_sec + PRIMARY_WINDOW_SEC[0]
    win_end = min(current_cut_sec + PRIMARY_WINDOW_SEC[1], region_end_sec - MIN_SEGMENT_DURATION_SEC)
    energy = _frame_energy(audio, sr, win_start, win_end) if win_end > win_start else np.empty(0)
    if energy.size < 3:
        return {"cut_sec": min(current_cut_sec + 10.0, region_end_sec),
                "status": "flagged_manual_review",
                "cut_energy_db": None, "pause_sec": 0.0, "pause_depth_db": None}
    smooth = np.convolve(energy, np.ones(3) / 3.0, mode="same")
    idx = int(np.argmin(smooth))
    return {"cut_sec": win_start + (idx + 0.5) * frame_sec,
            "status": "flagged_manual_review",
            "cut_energy_db": float(energy[idx]),
            "pause_sec": 0.0,
            "pause_depth_db": None if speech_db is None else speech_db - float(energy[idx])}


def find_best_cut_point_with_energy(
    audio: np.ndarray, sr: int, region_start_sec: float,
    current_cut_sec: float, region_end_sec: float,
) -> Tuple[float, str, Optional[float]]:
    """(cut_sec, status, cut_energy_db) - same shape as the v1 patched function."""
    d = find_cut_point_details(audio, sr, region_start_sec, current_cut_sec, region_end_sec)
    return d["cut_sec"], d["status"], d["cut_energy_db"]


def find_best_cut_point(
    audio: np.ndarray, sr: int, region_start_sec: float,
    current_cut_sec: float, region_end_sec: float,
) -> Tuple[float, str]:
    """Backward-compatible 2-tuple: (cut_sec, status)."""
    cut, status, _ = find_best_cut_point_with_energy(
        audio, sr, region_start_sec, current_cut_sec, region_end_sec)
    return cut, status


# --------------------------------------------------------------------------- #
# Region -> segment definitions
# --------------------------------------------------------------------------- #
def _internal_pad(pause_sec: Optional[float], padding_sec: float) -> float:
    """Padding allowed at a cut THROUGH speech: stays inside the pause."""
    if not pause_sec:
        return 0.0
    return min(padding_sec, PAD_FRACTION_OF_PAUSE * pause_sec)


def segment_speech_region(
    audio: np.ndarray,
    sr: int,
    region: Dict[str, Any],
    padding_sec: float = PADDING_SEC,
) -> List[Dict[str, Any]]:
    """Split one speech/mixed region into 5-15 s segments (3-5 s fragments are kept whole)."""
    reg_start = region["start_sec"]
    reg_end = region["end_sec"]
    total_audio_sec = len(audio) / sr

    if region["label"] not in ("speech", "mixed"):
        return []

    duration = reg_end - reg_start

    def whole(method: str) -> List[Dict[str, Any]]:
        padded_start = max(0.0, reg_start - padding_sec)
        padded_end = min(total_audio_sec, reg_end + padding_sec)
        return [{
            "start_sec": round(padded_start, 3),
            "end_sec": round(padded_end, 3),
            "duration_sec": round(padded_end - padded_start, 3),
            "region_label": region["label"],
            "confidence_score": region.get("confidence", 1.0),
            "boundary_method": method,
            "boundary_check": "auto_verified",
            "cut_energy_db": None, "pause_ms": None, "pause_depth_db": None,
        }]

    if duration < MIN_SEGMENT_DURATION_SEC:
        return whole("whole_short_region") if duration >= 3.0 else []
    if duration <= MAX_SEGMENT_DURATION_SEC:
        return whole("whole_region_fit")

    segments: List[Dict[str, Any]] = []
    current_start = reg_start
    start_pad = padding_sec            # outer edge of the region: full cushion
    while current_start < reg_end:
        d = find_cut_point_details(audio, sr, reg_start, current_start, reg_end)
        cut = d["cut_sec"]
        is_last = cut >= reg_end - 1e-6
        end_pad = padding_sec if is_last else _internal_pad(d["pause_sec"], padding_sec)

        padded_start = max(0.0, current_start - start_pad)
        padded_end = min(total_audio_sec, cut + end_pad)
        seg_duration = padded_end - padded_start

        if seg_duration >= MIN_SEGMENT_DURATION_SEC:
            segments.append({
                "start_sec": round(padded_start, 3),
                "end_sec": round(padded_end, 3),
                "duration_sec": round(seg_duration, 3),
                "region_label": region["label"],
                "confidence_score": region.get("confidence", 1.0),
                "boundary_method": BOUNDARY_METHOD,
                "boundary_check": d["status"],
                "cut_energy_db": None if d["cut_energy_db"] is None else round(d["cut_energy_db"], 2),
                "pause_ms": None if d["pause_sec"] is None else int(round(d["pause_sec"] * 1000)),
                "pause_depth_db": None if d["pause_depth_db"] is None else round(d["pause_depth_db"], 1),
            })
        current_start = cut
        start_pad = end_pad            # the next clip starts where this one stopped padding
    return segments


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def export_segments(
    wav_path: Path,
    regions: List[Dict[str, Any]],
    video_id: str,
    output_dir: Optional[Path] = None,
    max_segments: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Cut the WAV into segment files, save the manifest, return the records."""
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    output_dir = Path(output_dir).resolve() if output_dir is not None else SEGMENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[segmenter] Loading audio from {wav_path.name}...")
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    print("[segmenter] Segmenting speech regions (pause-based cuts, 5-15 s clips)...")
    all_defs: List[Dict[str, Any]] = []
    for reg in regions:
        if reg["label"] in ("speech", "mixed"):
            all_defs.extend(segment_speech_region(audio, sr, reg))
    if max_segments is not None:
        all_defs = all_defs[:max_segments]
    print(f"[segmenter] Identified {len(all_defs)} valid speech segments.")

    # Remove clips from a previous run of this video so stale files never reach Stage 6+
    for stale in output_dir.glob(f"{video_id}_seg_*.wav"):
        stale.unlink()

    records: List[Dict[str, Any]] = []
    for idx, d in enumerate(all_defs, start=1):
        segment_id = f"{video_id}_seg_{idx:06d}"
        segment_path = output_dir / f"{segment_id}.wav"
        clip = audio[int(d["start_sec"] * sr):int(d["end_sec"] * sr)]
        sf.write(str(segment_path), clip, sr, subtype="PCM_16")
        records.append({
            "video_id": video_id,
            "segment_id": segment_id,
            "audio_path": _rel(segment_path),
            "start_sec": d["start_sec"],
            "end_sec": d["end_sec"],
            "duration_sec": d["duration_sec"],
            "region_label": d["region_label"],
            "confidence_score": d["confidence_score"],
            "boundary_method": d["boundary_method"],
            "boundary_check": d["boundary_check"],
            "cut_energy_db": d.get("cut_energy_db"),
            "pause_ms": d.get("pause_ms"),
            "pause_depth_db": d.get("pause_depth_db"),
        })

    durations = [r["duration_sec"] for r in records]
    pauses = [r["pause_ms"] for r in records if r["pause_ms"]]
    verified = sum(1 for r in records if r["boundary_check"] == "auto_verified")
    flagged = sum(1 for r in records if r["boundary_check"] == "flagged_manual_review")
    print("\n[segmenter] Export Complete:")
    print(f"  • Total Segments: {len(records)}")
    if durations:
        print(f"  • Average Duration: {np.mean(durations):.2f}s "
              f"(Min: {min(durations):.2f}s, Max: {max(durations):.2f}s)")
    print(f"  • Cuts in a real pause: {verified}   Flagged for review: {flagged}")
    if pauses:
        print(f"  • Pause at internal cuts: median {int(np.median(pauses))} ms, min {min(pauses)} ms")
    print(f"  • Destination: {_rel(output_dir)}/")

    manifest_path = PROJECT_ROOT / "metadata" / f"{video_id}_segments.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    print(f"  • Manifest: {_rel(manifest_path)}")
    return records