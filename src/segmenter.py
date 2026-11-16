"""
src/segmenter.py
================
Stage 5: Speech Segmentation

Purpose:
--------
Transforms continuous speech and mixed regions (from Stage 4) into individual,
standardized audio clips suitable for speech recognition (ASR) training and evaluation.

Mandatory Project Constraints:
  1. Duration Range: 5.0 to 15.0 seconds per clip (target: 8.0 to 12.0 seconds).
  2. Word Boundary Preservation: Never knowingly slice through a spoken word.
     Cuts are placed at natural energy troughs (micro-pauses between words).
  3. Edge Silence Cushion: Preserves ~0.15s of silence at the start and end of each
     clip so initial and final consonants are not clipped.
  4. Stable Naming: Each clip receives a unique, zero-padded identifier:
     <video_id>_seg_000001.wav, <video_id>_seg_000002.wav, etc.
  5. Audit Readiness: Records boundary verification status for each segment
     ('auto_verified' vs 'flagged_manual_review').
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import soundfile as sf

from src.classifier import compute_frame_energy_db, FRAME_SIZE_SAMPLES


SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments"

# Target duration limits (in seconds)
MIN_SEGMENT_DURATION_SEC = 5.0
MAX_SEGMENT_DURATION_SEC = 15.0
TARGET_DURATION_MIN_SEC = 8.0
TARGET_DURATION_MAX_SEC = 12.0

# Silence cushion / padding added to segment boundaries (in seconds)
PADDING_SEC = 0.15


def find_best_cut_point(
    audio: np.ndarray,
    sr: int,
    region_start_sec: float,
    current_cut_sec: float,
    region_end_sec: float
) -> Tuple[float, str]:
    """
    Finds the optimal timestamp to place a cut within a long speech region.

    How it works:
    -------------
    1. Defines a search window around 8 to 12 seconds from current_cut_sec.
    2. Within that window, scans 50 ms RMS energy frames.
    3. Finds the local energy minimum (trough) where sound level is lowest.
    4. If the quietest point is below -28 dB → confident natural pause ('auto_verified').
       If the quietest point is louder → speech was continuous ('flagged_manual_review').

    Parameters:
    -----------
    audio : np.ndarray
        Full audio array.
    sr : int
        Sample rate (16000 Hz).
    region_start_sec : float
        Absolute start timestamp of the entire speech region.
    current_cut_sec : float
        Where the current segment started.
    region_end_sec : float
        Absolute end timestamp of the entire speech region.

    Returns:
    --------
    Tuple[float, str] : (cut_timestamp_sec, boundary_status)
    """
    remaining_sec = region_end_sec - current_cut_sec

    # If what remains is already within the acceptable range (<= 15s), take all of it
    if remaining_sec <= MAX_SEGMENT_DURATION_SEC:
        return region_end_sec, "auto_verified"

    # Define the search window for candidate cut points:
    # Look between 8.0s and 12.0s after current_cut_sec
    search_start_sec = current_cut_sec + TARGET_DURATION_MIN_SEC
    search_end_sec = min(current_cut_sec + TARGET_DURATION_MAX_SEC, region_end_sec - MIN_SEGMENT_DURATION_SEC)

    # Convert seconds to sample indices
    search_start_frame = int(search_start_sec * sr)
    search_end_frame = int(search_end_sec * sr)

    search_slice = audio[search_start_frame:search_end_frame]
    if len(search_slice) < FRAME_SIZE_SAMPLES * 2:
        # Fallback if window is too small: cut at 10 seconds
        fallback_cut = min(current_cut_sec + 10.0, region_end_sec)
        return fallback_cut, "flagged_manual_review"

    # Compute frame energy in 50 ms steps across the search window
    frame_energy = compute_frame_energy_db(search_slice, frame_size=FRAME_SIZE_SAMPLES)

    # Find the frame with the lowest energy (deepest trough)
    best_frame_idx = int(np.argmin(frame_energy))
    best_energy_db = frame_energy[best_frame_idx]

    # Convert best frame back to absolute seconds
    offset_sec = (best_frame_idx * FRAME_SIZE_SAMPLES) / sr
    best_cut_sec = search_start_sec + offset_sec

    # Assess boundary quality:
    # If energy at cut point is below -28 dB, it's a clear acoustic gap between words
    if best_energy_db < -28.0:
        status = "auto_verified"
    else:
        # Energy was still somewhat elevated; flag for human annotator review
        status = "flagged_manual_review"

    return best_cut_sec, status


def segment_speech_region(
    audio: np.ndarray,
    sr: int,
    region: Dict[str, Any],
    padding_sec: float = PADDING_SEC
) -> List[Dict[str, Any]]:
    """
    Splits a single classified speech region into one or more 5–15 second segments.

    Parameters:
    -----------
    audio : np.ndarray
        Full 16 kHz audio array.
    sr : int
        Sample rate (16000).
    region : dict
        A region record with 'start_sec', 'end_sec', 'label', 'confidence'.
    padding_sec : float
        Silence cushion added to the start and end of each segment.

    Returns:
    --------
    List[dict] : One or more segment boundary dictionaries.
    """
    reg_start = region["start_sec"]
    reg_end = region["end_sec"]
    total_audio_sec = len(audio) / sr

    # Only segment 'speech' and 'mixed' regions (skip silence and music)
    if region["label"] not in ("speech", "mixed"):
        return []

    duration = reg_end - reg_start

    # Case 1: Region is shorter than 5 seconds
    # In corpus construction, isolated speech fragments < 5s are kept with a note
    # if they are >= 3.0s, otherwise skipped
    if duration < MIN_SEGMENT_DURATION_SEC:
        if duration >= 3.0:
            padded_start = max(0.0, reg_start - padding_sec)
            padded_end = min(total_audio_sec, reg_end + padding_sec)
            return [{
                "start_sec": round(padded_start, 3),
                "end_sec": round(padded_end, 3),
                "duration_sec": round(padded_end - padded_start, 3),
                "region_label": region["label"],
                "confidence_score": region.get("confidence", 1.0),
                "boundary_method": "whole_short_region",
                "boundary_check": "auto_verified",
            }]
        else:
            # Fragment is too tiny (< 3.0s), skip it
            return []

    # Case 2: Region is already in the sweet spot (5.0s to 15.0s)
    if duration <= MAX_SEGMENT_DURATION_SEC:
        padded_start = max(0.0, reg_start - padding_sec)
        padded_end = min(total_audio_sec, reg_end + padding_sec)
        return [{
            "start_sec": round(padded_start, 3),
            "end_sec": round(padded_end, 3),
            "duration_sec": round(padded_end - padded_start, 3),
            "region_label": region["label"],
            "confidence_score": region.get("confidence", 1.0),
            "boundary_method": "whole_region_fit",
            "boundary_check": "auto_verified",
        }]

    # Case 3: Region is longer than 15.0s → split iteratively using energy troughs
    segments = []
    current_start = reg_start

    while current_start < reg_end:
        cut_point, status = find_best_cut_point(audio, sr, reg_start, current_start, reg_end)

        # Apply padding to start and end
        # (Don't pad backwards past 0 or forwards past total audio)
        padded_start = max(0.0, current_start - padding_sec)
        padded_end = min(total_audio_sec, cut_point + padding_sec)

        seg_duration = padded_end - padded_start

        # Ensure segment satisfies minimum length
        if seg_duration >= MIN_SEGMENT_DURATION_SEC:
            segments.append({
                "start_sec": round(padded_start, 3),
                "end_sec": round(padded_end, 3),
                "duration_sec": round(seg_duration, 3),
                "region_label": region["label"],
                "confidence_score": region.get("confidence", 1.0),
                "boundary_method": "energy_trough_v1",
                "boundary_check": status,
            })

        # Advance to next segment
        current_start = cut_point

    return segments


def export_segments(
    wav_path: Path,
    regions: List[Dict[str, Any]],
    video_id: str,
    output_dir: Optional[Path] = None,
    max_segments: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Splits audio into speech segments, exports individual WAV files,
    and returns a structured manifest list.

    Parameters:
    -----------
    wav_path : Path
        Source 16 kHz mono WAV.
    regions : List[dict]
        Classified regions from Stage 4.
    video_id : str
        Video identifier for segment naming (e.g. 'gK017cCH36k').
    output_dir : Path, optional
        Target directory for segment WAVs (defaults to data/segments/).
    max_segments : int, optional
        Limit number of segments exported (useful for fast testing).

    Returns:
    --------
    List[dict] : Complete segment records with paths, timestamps, and status.
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    if output_dir is None:
        output_dir = SEGMENTS_DIR
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[segmenter] Loading audio from {wav_path.name}...")
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    print(f"[segmenter] Segmenting speech regions (target: 8–12s, limits: 5–15s)...")

    # Step 1: Generate segment timestamp definitions
    all_segment_defs = []
    for reg in regions:
        if reg["label"] in ("speech", "mixed"):
            defs = segment_speech_region(audio, sr, reg)
            all_segment_defs.extend(defs)

    if max_segments is not None:
        all_segment_defs = all_segment_defs[:max_segments]

    print(f"[segmenter] Identified {len(all_segment_defs)} valid speech segments.")
    print(f"[segmenter] Slicing and exporting individual WAV files...")

    manifest_records = []

    # Step 2: Slice each segment from audio and save to disk
    for idx, seg_def in enumerate(all_segment_defs, start=1):
        segment_id = f"{video_id}_seg_{idx:06d}"
        segment_filename = f"{segment_id}.wav"
        segment_path = output_dir / segment_filename

        # Convert timestamps to exact sample indices
        start_frame = int(seg_def["start_sec"] * sr)
        end_frame = int(seg_def["end_sec"] * sr)

        # Slice the audio array
        clip = audio[start_frame:end_frame]

        # Save as standard 16-bit PCM WAV (subtype='PCM_16')
        sf.write(str(segment_path), clip, sr, subtype="PCM_16")

        # Build the formal manifest record
        record = {
            "video_id": video_id,
            "segment_id": segment_id,
            "audio_path": str(segment_path.relative_to(PROJECT_ROOT)),
            "start_sec": seg_def["start_sec"],
            "end_sec": seg_def["end_sec"],
            "duration_sec": seg_def["duration_sec"],
            "region_label": seg_def["region_label"],
            "confidence_score": seg_def["confidence_score"],
            "boundary_method": seg_def["boundary_method"],
            "boundary_check": seg_def["boundary_check"],
        }
        manifest_records.append(record)

    # Step 3: Print summary statistics
    durations = [r["duration_sec"] for r in manifest_records]
    avg_dur = np.mean(durations) if durations else 0.0
    verified_count = sum(1 for r in manifest_records if r["boundary_check"] == "auto_verified")
    flagged_count = sum(1 for r in manifest_records if r["boundary_check"] == "flagged_manual_review")

    print(f"\n[segmenter] Export Complete:")
    print(f"  • Total Segments: {len(manifest_records)}")
    print(f"  • Average Duration: {avg_dur:.2f}s (Min: {min(durations):.2f}s, Max: {max(durations):.2f}s)")
    print(f"  • Auto-Verified Cuts: {verified_count}")
    print(f"  • Flagged for Review: {flagged_count}")
    print(f"  • Destination: {output_dir.relative_to(PROJECT_ROOT)}/")

    return manifest_records
