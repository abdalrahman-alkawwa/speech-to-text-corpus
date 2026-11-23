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

# Import sys so the project root can be added to Python's module search path.
import sys
# Import Path for safe, cross-platform filesystem path handling.
from pathlib import Path

# Resolve this file's location, then move up two folders to identify the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Check whether the project root is already in Python's module search path.
if str(PROJECT_ROOT) not in sys.path:
    # If it is missing, add it at the beginning so project modules can be imported.
    sys.path.insert(0, str(PROJECT_ROOT))

# Import json for writing the segment manifest as a JSON file.
import json
# Import type-hint helpers used to document expected argument, return, and data types.
from typing import Any, Dict, List, Optional, Tuple

# Import NumPy for audio arrays and numerical calculations.
import numpy as np
# Import SoundFile for reading and writing WAV audio.
import soundfile as sf

# Import the shared frame-energy function and frame-size constant from the classifier module.
from src.classifier import compute_frame_energy_db, FRAME_SIZE_SAMPLES  # noqa: F401  (FRAME_SIZE_SAMPLES re-exported)

# Build the default output folder path for exported audio segments.
SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments"

# Clip length limits (seconds, including padding)
# Set the minimum allowed exported segment duration in seconds.
MIN_SEGMENT_DURATION_SEC = 5.0
# Set the maximum allowed exported segment duration in seconds.
MAX_SEGMENT_DURATION_SEC = 15.0
# Set the lower end of the preferred segment-duration window.
TARGET_DURATION_MIN_SEC = 8.0
# Set the upper end of the preferred segment-duration window.
TARGET_DURATION_MAX_SEC = 12.0

# Cushion at the OUTER edges of a region
# Set the amount of extra audio retained at the outer edges of a region.
PADDING_SEC = 0.15

# --- v2 pause detection ---
# Set each pause-detection frame to 320 samples, which is 20 ms at 16 kHz.
CUT_FRAME_SAMPLES = 320            # 20 ms frames at 16 kHz
# Require a quiet run to last at least 0.15 seconds before treating it as a pause.
MIN_PAUSE_SEC = 0.15               # shorter dips are word-internal closures, not pauses
# Require pause frames to be at least 15 dB quieter than the estimated speech level.
PAUSE_DEPTH_DB = 15.0              # a pause frame is >= 15 dB below the local speech level
# Estimate local speech loudness using the 75th percentile of frame energies.
SPEECH_LEVEL_PERCENTILE = 75.0     # "local speech level" = 75th percentile of frame energy
# Define the preferred cut-search window using the 8- and 12-second limits above.
PRIMARY_WINDOW_SEC = (TARGET_DURATION_MIN_SEC, TARGET_DURATION_MAX_SEC)
# Define a wider fallback search window used if the preferred window has no pause.
WIDE_WINDOW_SEC = (6.0, 14.0)      # used only when the primary window has no real pause
# Limit padding around an internal cut to 40 percent of the detected pause duration.
PAD_FRACTION_OF_PAUSE = 0.4        # internal-cut padding = 40 % of the pause (<= PADDING_SEC)

# Label segments produced by this algorithm version for later auditing.
BOUNDARY_METHOD = "energy_pause_v2"


# --------------------------------------------------------------------------- #
# Cut-point search
# --------------------------------------------------------------------------- #
# Define a helper that calculates frame energies for a selected time interval.
def _frame_energy(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    
    """20 ms frame energies (dBFS) for audio[start_sec:end_sec]; empty if too short."""
    # Convert the requested start time from seconds to a sample index and prevent negative indexing.
    s = max(0, int(start_sec * sr))
    # Convert the requested end time to a sample index and prevent it exceeding the audio length.
    e = min(len(audio), int(end_sec * sr))
    # Extract only the requested audio samples from the full audio array.
    chunk = audio[s:e]
    # Check whether the extracted audio is long enough to contain one complete analysis frame.
    if len(chunk) < CUT_FRAME_SAMPLES:
        # Return an empty float array when there is not enough audio to calculate frame energy.
        return np.empty(0, dtype=np.float64)
    # Calculate and return the dBFS energy of each frame using the configured frame size.
    return compute_frame_energy_db(chunk, frame_size=CUT_FRAME_SAMPLES)


# Define a helper that groups consecutive True values into pause intervals.
def _pause_runs(is_pause: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
   
    """(start, end_exclusive) of every run of True values at least min_frames long."""
    # Create a list that will hold the qualifying pause intervals.
    runs: List[Tuple[int, int]] = []
    # Initialize the scan index at zero and store the total number of frames.
    i, n = 0, len(is_pause)
    # Continue scanning while the index is within the array.
    while i < n:
        # Check whether the current frame has been classified as part of a pause.
        if is_pause[i]:
            # Start a second index at the beginning of this possible pause run.
            j = i
            # Advance through consecutive pause frames until the run ends or the array ends.
            while j < n and is_pause[j]:
                # Move the second index forward by one frame.
                j += 1
            # Check whether this run is at least the required minimum number of frames long.
            if j - i >= min_frames:
                # Store the run's start and exclusive end indices if it is long enough.
                runs.append((i, j))
            # Move the main scan index to the end of the run so those frames are not scanned twice.
            i = j
        # Handle the case where the current frame is not part of a pause.
        else:
            # Advance by one frame and continue searching.
            i += 1
    # Return all pause runs that met the minimum duration requirement.
    return runs


# --------------------------------------------------------------------------- #
# Small helpers used by find_cut_point_details
# --------------------------------------------------------------------------- #

def _estimate_local_speech_level(
    audio: np.ndarray,
    sr: int,
    current_cut_sec: float,
    region_end_sec: float,
) -> Optional[float]:
    """Estimate the typical speech level in the audio this segment could cover.

    We inspect at most MAX_SEGMENT_DURATION_SEC seconds from the current segment
    start. _frame_energy converts that stretch into 20 ms dBFS measurements.
    The 75th percentile is used as the local speech-level estimate: it is based
    on this recording's actual loudness, rather than a fixed dBFS threshold.

    Fewer than 10 frames is not enough data for this estimate, so return None.
    """
    span_end_sec = min(
        current_cut_sec + MAX_SEGMENT_DURATION_SEC,
        region_end_sec,
    )
    span_energy = _frame_energy(audio, sr, current_cut_sec, span_end_sec)

    if span_energy.size < 10:
        return None

    return float(np.percentile(span_energy, SPEECH_LEVEL_PERCENTILE))


def _find_pause_cut_in_window(
    audio: np.ndarray,
    sr: int,
    current_cut_sec: float,
    region_end_sec: float,
    window_sec: Tuple[float, float],
    frame_sec: float,
    min_frames: int,
    threshold: float,
    target_time: float,
) -> Optional[Dict[str, Any]]:
    """Find the best qualifying pause inside one search window.

    A frame is considered quiet when its energy is below threshold. _pause_runs
    groups consecutive quiet frames and keeps only runs lasting at least
    min_frames. The chosen run is the longest; equal-length runs are resolved
    in favour of the one nearest target_time.

    Frame indices in a run are relative to this window's energy array, not to
    the full audio. We convert the winning run's midpoint back to absolute
    seconds before returning it. None means this window had no usable pause.
    """
    window_lo_sec, window_hi_sec = window_sec
    win_start = current_cut_sec + window_lo_sec
    # Reserve MIN_SEGMENT_DURATION_SEC after the cut so the final clip can
    # still meet its minimum duration. This may shorten the search window.
    win_end = min(
        current_cut_sec + window_hi_sec,
        region_end_sec - MIN_SEGMENT_DURATION_SEC,
    )

    # Two frame durations is the existing minimum window-size check.
    if win_end - win_start < 2 * frame_sec:
        return None

    energy = _frame_energy(audio, sr, win_start, win_end)
    if energy.size == 0:
        return None

    # True marks one frame below the quietness threshold. A run of True values
    # must last long enough to count as a pause (not just a brief consonant dip).
    runs = _pause_runs(energy < threshold, min_frames)
    if not runs:
        return None

    def score(run: Tuple[int, int]) -> Tuple[int, float]:
        # run[1] is exclusive, so averaging the two boundaries gives the
        # centre position in frame units. Multiplying by frame_sec converts it
        # to seconds; win_start makes it an absolute audio timestamp.
        centre_t = win_start + 0.5 * (run[0] + run[1]) * frame_sec
        # max() prefers the longest run; for a tie, the negative distance
        # makes the pause nearest target_time receive the higher score.
        return (run[1] - run[0], -abs(centre_t - target_time))

    lo, hi = max(runs, key=score)
    # Average the ORIGINAL frame energies across the winning pause.
    mean_db = float(np.mean(energy[lo:hi]))

    return {
        # Cut halfway through the pause, not halfway through the whole window.
        "cut_sec": win_start + 0.5 * (lo + hi) * frame_sec,
        "status": "auto_verified",
        "cut_energy_db": mean_db,
        "pause_sec": (hi - lo) * frame_sec,
        "pause_depth_db": None,
    }


def _fallback_cut_details(
    audio: np.ndarray,
    sr: int,
    current_cut_sec: float,
    region_end_sec: float,
    frame_sec: float,
    speech_db: Optional[float],
) -> Dict[str, Any]:
    """Choose the quietest smoothed point in the primary window if no pause exists.

    This is deliberately a fallback, not a verified pause: it finds a low-energy
    location but cannot prove that the location is a natural speech break.
    Therefore every result from this helper is flagged for manual review.
    """
    win_start = current_cut_sec + PRIMARY_WINDOW_SEC[0]
    win_end = min(
        current_cut_sec + PRIMARY_WINDOW_SEC[1],
        region_end_sec - MIN_SEGMENT_DURATION_SEC,
    )
    energy = (
        _frame_energy(audio, sr, win_start, win_end)
        if win_end > win_start
        else np.empty(0)
    )

    # A three-frame moving average needs at least three energy values to be
    # useful. If there are fewer, preserve the existing approximate 10 s cut.
    if energy.size < 3:
        return {
            "cut_sec": min(current_cut_sec + 10.0, region_end_sec),
            "status": "flagged_manual_review",
            "cut_energy_db": None,
            "pause_sec": 0.0,
            "pause_depth_db": None,
        }

    # Average each frame with its neighbours (3 * 20 ms = 60 ms at 16 kHz).
    # Smoothing reduces the influence of a single unusually low frame. It does
    # NOT establish a minimum pause duration; this branch remains unverified.
    smooth = np.convolve(energy, np.ones(3) / 3.0, mode="same")
    idx = int(np.argmin(smooth))

    # idx identifies a frame within the search window. Add half a frame so the
    # timestamp points to that frame's centre, then offset by win_start.
    return {
        "cut_sec": win_start + (idx + 0.5) * frame_sec,
        "status": "flagged_manual_review",
        # Store the selected frame's original energy, not the smoothed value.
        "cut_energy_db": float(energy[idx]),
        # No qualifying pause was detected, so its verified duration is zero.
        "pause_sec": 0.0,
        # If local speech level could not be estimated, pause depth is unknown.
        "pause_depth_db": None if speech_db is None else speech_db - float(energy[idx]),
    }


# Define the detailed cut-search function. Its job is now orchestration:
# handle the short-remainder case, prepare shared measurements, try pause
# windows in order, and delegate the no-pause case to the fallback helper.
def find_cut_point_details(
    audio: np.ndarray,
    sr: int,
    region_start_sec: float,
    current_cut_sec: float,
    region_end_sec: float,
) -> Dict[str, Any]:
    """Choose a cut and return its status plus energy/pause audit measurements.

    The decision order is:
      1. If the remaining audio already fits, use the region end.
      2. Estimate the local speech level from this recording.
      3. Search the primary window, then the wider window, for a real pause.
      4. If no qualifying pause is found, use the smoothed-energy fallback and
         flag the proposed cut for manual review.

    region_start_sec remains in the signature for compatibility with callers;
    the existing cut-selection logic does not use it.
    """
    remaining_sec = region_end_sec - current_cut_sec

    # No internal cut is needed when all remaining audio fits in one segment.
    # This existing duration shortcut returns the region end without pause search.
    if remaining_sec <= MAX_SEGMENT_DURATION_SEC:
        return {
            "cut_sec": region_end_sec,
            "status": "auto_verified",
            "cut_energy_db": None,
            "pause_sec": None,
            "pause_depth_db": None,
        }

    # Convert the 320-sample frame size into seconds (0.02 s at 16 kHz).
    frame_sec = CUT_FRAME_SAMPLES / sr
    # Convert the minimum pause length into a count of consecutive frames.
    min_frames = int(round(MIN_PAUSE_SEC / frame_sec))

    speech_db = _estimate_local_speech_level(
        audio, sr, current_cut_sec, region_end_sec
    )

    # Keep the same existing behaviour: without a speech-level estimate there
    # is no threshold, so the pause-search loop is skipped and fallback is used.
    if speech_db is not None:
        threshold = speech_db - PAUSE_DEPTH_DB
        # This is a preferred location for tie-breaking, not a forced cut time.
        target_time = current_cut_sec + 0.5 * (
            PRIMARY_WINDOW_SEC[0] + PRIMARY_WINDOW_SEC[1]
        )

        # Try the preferred 8-12 s window first. Only if it has no qualifying
        # pause do we try the wider 6-14 s window.
        for window_sec in (PRIMARY_WINDOW_SEC, WIDE_WINDOW_SEC):
            pause_details = _find_pause_cut_in_window(
                audio=audio,
                sr=sr,
                current_cut_sec=current_cut_sec,
                region_end_sec=region_end_sec,
                window_sec=window_sec,
                frame_sec=frame_sec,
                min_frames=min_frames,
                threshold=threshold,
                target_time=target_time,
            )
            if pause_details is not None:
                # Attach pause depth here because this helper knows the local
                # speech level; the window helper measures the pause itself.
                pause_details["pause_depth_db"] = (
                    speech_db - pause_details["cut_energy_db"]
                )
                return pause_details

    # No qualifying pause in either window (or no speech-level estimate):
    # choose a smoothed low-energy point, explicitly marked for review.
    return _fallback_cut_details(
        audio, sr, current_cut_sec, region_end_sec, frame_sec, speech_db
    )


# Define a compatibility wrapper that returns cut time, status, and cut energy.
def find_best_cut_point_with_energy(
    # Accept the waveform, sample rate, and region start as inputs.
    audio: np.ndarray, sr: int, region_start_sec: float,
    # Accept the current segment start and region end as inputs.
    current_cut_sec: float, region_end_sec: float,
# Specify the three-element return tuple and its value types.
) -> Tuple[float, str, Optional[float]]:
    # Document that this preserves the three-value interface used by the earlier version.
    """(cut_sec, status, cut_energy_db) - same shape as the v1 patched function."""
    # Call the detailed function to perform the actual cut search and collect its result dictionary.
    d = find_cut_point_details(audio, sr, region_start_sec, current_cut_sec, region_end_sec)
    # Return only the three fields promised by this wrapper.
    return d["cut_sec"], d["status"], d["cut_energy_db"]


# --------------------------------------------------------------------------- #
# Region -> segment definitions
# --------------------------------------------------------------------------- #
# Define a helper that calculates safe padding around a cut inside a detected pause.
def _internal_pad(pause_sec: Optional[float], padding_sec: float) -> float:
    # Document that this padding must remain within the pause rather than reaching neighbouring words.
    """Padding allowed at a cut THROUGH speech: stays inside the pause."""
    # Check whether a pause duration is available and non-zero.
    if not pause_sec:
        # Use no padding when no pause was detected.
        return 0.0
    # Otherwise use the smaller of the normal padding and 40 percent of the pause duration.
    return min(padding_sec, PAD_FRACTION_OF_PAUSE * pause_sec)


# Define the function that converts one classified speech/mixed region into segment definitions.
def segment_speech_region(
    # Accept the complete audio waveform.
    audio: np.ndarray,
    # Accept the audio sample rate.
    sr: int,
    # Accept the region dictionary produced by the classifier.
    region: Dict[str, Any],
    # Allow callers to override the default edge-padding duration.
    padding_sec: float = PADDING_SEC,
# Specify that the function returns a list of segment dictionaries.
) -> List[Dict[str, Any]]:
   
    """Split one speech/mixed region into 5-15 s segments (3-5 s fragments are kept whole)."""
    # Read the region's start time from its metadata.
    reg_start = region["start_sec"]
    # Read the region's end time from its metadata.
    reg_end = region["end_sec"]
    # Calculate the total audio duration from sample count divided by sample rate.
    total_audio_sec = len(audio) / sr

    # Process only regions labelled speech or mixed.
    if region["label"] not in ("speech", "mixed"):
        # Return an empty list for music, silence, or any other label.
        return []

    # Calculate the duration of the region before adding padding.
    duration = reg_end - reg_start

    # Define a nested helper for returning the region as one unsplit segment.
    def whole(method: str) -> List[Dict[str, Any]]:
        # Calculate the padded start while preventing it from going before time zero.
        padded_start = max(0.0, reg_start - padding_sec)
        # Calculate the padded end while preventing it from exceeding the audio duration.
        padded_end = min(total_audio_sec, reg_end + padding_sec)
        # Begin returning a one-item list containing the segment metadata dictionary.
        return [{
            # Store the padded segment start time rounded to milliseconds.
            "start_sec": round(padded_start, 3),
            # Store the padded segment end time rounded to milliseconds.
            "end_sec": round(padded_end, 3),
            # Store the resulting segment duration rounded to milliseconds.
            "duration_sec": round(padded_end - padded_start, 3),
            # Preserve the original classifier label, such as speech or mixed.
            "region_label": region["label"],
            # Copy the classifier confidence, defaulting to 1.0 if it was not supplied.
            "confidence_score": region.get("confidence", 1.0),
            # Record the reason or method used to keep this region as one whole segment.
            "boundary_method": method,
            # Mark the boundary check as verified for this unsplit region.
            "boundary_check": "auto_verified",
            # Set pause-cut measurements to None because this segment has no internal cut.
            "cut_energy_db": None, "pause_ms": None, "pause_depth_db": None,
        }]

    # Check whether the original region is shorter than the normal minimum segment duration.
    if duration < MIN_SEGMENT_DURATION_SEC:
        # Keep regions of at least three seconds whole; discard anything shorter.
        return whole("whole_short_region") if duration >= 3.0 else []
    # Check whether the region already fits within the maximum segment duration.
    if duration <= MAX_SEGMENT_DURATION_SEC:
        # Return a single whole-region segment when splitting is unnecessary.
        return whole("whole_region_fit")

    # Create an empty list that will collect the definitions of all generated segments.
    segments: List[Dict[str, Any]] = []
    # Set the first segment's starting point to the beginning of the classified region.
    current_start = reg_start
    # Apply full padding at the region's outer starting edge.
    start_pad = padding_sec            # outer edge of the region: full cushion
    # Continue creating segments until the current start reaches the region end.
    while current_start < reg_end:
        # Find the best next cut and its pause/energy audit details.
        d = find_cut_point_details(audio, sr, reg_start, current_start, reg_end)
        # Read the selected cut time from the detailed result.
        cut = d["cut_sec"]
        # Determine whether this cut reaches the region end, allowing a tiny floating-point tolerance.
        is_last = cut >= reg_end - 1e-6
        # Use full padding at the final outer edge; for internal cuts, use pause-limited padding.
        end_pad = padding_sec if is_last else _internal_pad(d["pause_sec"], padding_sec)

        # Calculate the padded segment start without allowing it before the audio begins.
        padded_start = max(0.0, current_start - start_pad)
        # Calculate the padded segment end without allowing it past the audio's end.
        padded_end = min(total_audio_sec, cut + end_pad)
        # Calculate the final clip duration after padding has been applied.
        seg_duration = padded_end - padded_start

        # Only append a segment if its final duration meets the configured minimum.
        if seg_duration >= MIN_SEGMENT_DURATION_SEC:
            # Begin a dictionary describing this segment.
            segments.append({
                # Store the segment's padded start time rounded to three decimal places.
                "start_sec": round(padded_start, 3),
                # Store the segment's padded end time rounded to three decimal places.
                "end_sec": round(padded_end, 3),
                # Store the padded duration rounded to three decimal places.
                "duration_sec": round(seg_duration, 3),
                # Store the speech/mixed label inherited from the classified region.
                "region_label": region["label"],
                # Store the region confidence, defaulting to 1.0 when missing.
                "confidence_score": region.get("confidence", 1.0),
                # Record the algorithm version used to choose this segment's boundaries.
                "boundary_method": BOUNDARY_METHOD,
                # Record whether the cut was automatically verified or flagged for manual review.
                "boundary_check": d["status"],
                # Store cut energy rounded to two decimals, preserving None when unavailable.
                "cut_energy_db": None if d["cut_energy_db"] is None else round(d["cut_energy_db"], 2),
                # Convert pause duration from seconds to milliseconds and store it as an integer.
                "pause_ms": None if d["pause_sec"] is None else int(round(d["pause_sec"] * 1000)),
                # Store pause depth rounded to one decimal place, preserving None when unavailable.
                "pause_depth_db": None if d["pause_depth_db"] is None else round(d["pause_depth_db"], 1),
            # Finish adding this segment dictionary to the output list.
            })
        # Move the next segment's unpadded start to the cut selected for this segment.
        current_start = cut
        # Carry the previous segment's end padding forward so adjacent clips meet consistently.
        start_pad = end_pad            # the next clip starts where this one stopped padding
    # Return all segment definitions created for this region.
    return segments


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
# Define a helper that converts an absolute path into a project-relative path when possible.
def _rel(path: Path) -> str:
    # Begin a try block because the path may not be located inside the project directory.
    try:
        # Convert the path relative to the project root and use forward slashes for consistency.
        return path.relative_to(PROJECT_ROOT).as_posix()
    # Handle the case where the path is outside the project root.
    except ValueError:
        # Return the original path as a string when it cannot be made project-relative.
        return str(path)


# Define the function that writes segment WAV files and the JSON manifest.
def export_segments(
    # Accept the source WAV file path.
    wav_path: Path,
    # Accept the classified speech/mixed region dictionaries.
    regions: List[Dict[str, Any]],
    # Accept the source video's identifier for naming output files.
    video_id: str,
    # Allow an optional custom output directory; None means use the default segments folder.
    output_dir: Optional[Path] = None,
    # Allow an optional limit on how many segments to export.
    max_segments: Optional[int] = None,
# Specify that the function returns the exported segment records.
) -> List[Dict[str, Any]]:
    # Document that this function cuts audio files, writes a manifest, and returns metadata.
    """Cut the WAV into segment files, save the manifest, return the records."""
    # Normalize the input WAV path into an absolute Path object.
    wav_path = Path(wav_path).resolve()
    # Check that the source audio file exists before attempting to read it.
    if not wav_path.exists():
        # Raise a clear error immediately if the source WAV file is missing.
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Use the caller's output directory when supplied; otherwise use the default segments directory.
    output_dir = Path(output_dir).resolve() if output_dir is not None else SEGMENTS_DIR
    # Create the output directory and any missing parent folders; do not fail if it already exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print which source audio file is about to be loaded.
    print(f"\n[segmenter] Loading audio from {wav_path.name}...")
    # Read the WAV into a float32 NumPy array and capture its sample rate.
    audio, sr = sf.read(str(wav_path), dtype="float32")
    # Check whether the loaded audio has more than one channel.
    if audio.ndim > 1:
        # Convert multichannel audio to mono by averaging samples across the channel axis.
        audio = np.mean(audio, axis=1)

    # Print a message describing the pause-based segmentation operation.
    print("[segmenter] Segmenting speech regions (pause-based cuts, 5-15 s clips)...")
    # Create a list to collect segment definitions from all eligible regions.
    all_defs: List[Dict[str, Any]] = []
    # Loop through every classified region supplied to the exporter.
    for reg in regions:
        # Check whether the current region is speech or mixed.
        if reg["label"] in ("speech", "mixed"):
            # Generate segment definitions for this region and append them to the combined list.
            all_defs.extend(segment_speech_region(audio, sr, reg))
    # Check whether the caller supplied a maximum number of segments.
    if max_segments is not None:
        # Keep only the first requested number of segment definitions.
        all_defs = all_defs[:max_segments]
    # Print how many valid segment definitions were identified.
    print(f"[segmenter] Identified {len(all_defs)} valid speech segments.")

    # This cleanup prevents WAV files left by an earlier run from being mistaken for current output.
    # Remove clips from a previous run of this video so stale files never reach Stage 6+
    # Find existing segment WAV files whose names begin with this video's identifier.
    for stale in output_dir.glob(f"{video_id}_seg_*.wav"):
        # Delete each matching stale file from the output directory.
        stale.unlink()

    # Create a list that will hold the final metadata records for exported segments.
    records: List[Dict[str, Any]] = []
    # Loop over every segment definition, numbering the output segments from one.
    for idx, d in enumerate(all_defs, start=1):
        # Build a zero-padded segment ID using the video ID and a six-digit sequence number.
        segment_id = f"{video_id}_seg_{idx:06d}"
        # Build the output WAV path from the output directory and segment ID.
        segment_path = output_dir / f"{segment_id}.wav"
        # Slice the original waveform using the segment's start/end seconds converted to sample indices.
        clip = audio[int(d["start_sec"] * sr):int(d["end_sec"] * sr)]
        # Write the extracted clip as a 16-bit PCM WAV file at the original sample rate.
        sf.write(str(segment_path), clip, sr, subtype="PCM_16")
        # Begin a metadata dictionary for the newly exported clip.
        records.append({
            # Store the source video identifier.
            "video_id": video_id,
            # Store the unique segment identifier.
            "segment_id": segment_id,
            # Store the segment audio path relative to the project when possible.
            "audio_path": _rel(segment_path),
            # Store the segment's start time in seconds.
            "start_sec": d["start_sec"],
            # Store the segment's end time in seconds.
            "end_sec": d["end_sec"],
            # Store the segment's duration in seconds.
            "duration_sec": d["duration_sec"],
            # Store the original region label, such as speech or mixed.
            "region_label": d["region_label"],
            # Store the classifier confidence score for the source region.
            "confidence_score": d["confidence_score"],
            # Store the method used to determine the segment boundaries.
            "boundary_method": d["boundary_method"],
            # Store whether the boundary was verified or requires manual review.
            "boundary_check": d["boundary_check"],
            # Store cut energy when available; use None when this field was not provided.
            "cut_energy_db": d.get("cut_energy_db"),
            # Store the detected pause duration in milliseconds when available.
            "pause_ms": d.get("pause_ms"),
            # Store the detected pause depth in decibels when available.
            "pause_depth_db": d.get("pause_depth_db"),
        # Finish adding this segment's record to the records list.
        })

    # Collect all exported segment durations for summary statistics.
    durations = [r["duration_sec"] for r in records]
    # Collect non-zero pause durations so pause statistics exclude missing/zero values.
    pauses = [r["pause_ms"] for r in records if r["pause_ms"]]
    # Count records whose boundary check says the cut was automatically verified.
    verified = sum(1 for r in records if r["boundary_check"] == "auto_verified")
    # Count records whose boundary check says manual review is required.
    flagged = sum(1 for r in records if r["boundary_check"] == "flagged_manual_review")
    # Print a heading for the export summary.
    print("\n[segmenter] Export Complete:")
    # Print the total number of exported segments.
    print(f"  • Total Segments: {len(records)}")
    # Only calculate duration statistics when at least one segment exists.
    if durations:
        # Print the average duration, formatted to two decimal places.
        print(f"  • Average Duration: {np.mean(durations):.2f}s "
              # Continue the same output line with the minimum and maximum durations.
              f"(Min: {min(durations):.2f}s, Max: {max(durations):.2f}s)")
    # Print the counts of pause-verified and manually flagged cuts.
    print(f"  • Cuts in a real pause: {verified}   Flagged for review: {flagged}")
    # Only print pause statistics if at least one non-zero pause duration was recorded.
    if pauses:
        # Print the median and minimum internal-pause durations in milliseconds.
        print(f"  • Pause at internal cuts: median {int(np.median(pauses))} ms, min {min(pauses)} ms")
    # Print the output directory using a project-relative path where possible.
    print(f"  • Destination: {_rel(output_dir)}/")

    # Build the JSON manifest path under the project's metadata folder.
    manifest_path = PROJECT_ROOT / "metadata" / f"{video_id}_segments.json"
    # Create the metadata folder and any missing parent folders.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Open the manifest file for UTF-8 text writing, binding the open file to fh.
    with open(manifest_path, "w", encoding="utf-8") as fh:
        # Serialize all segment records as readable, indented JSON while preserving Unicode characters.
        json.dump(records, fh, ensure_ascii=False, indent=2)
    # Print the location of the saved manifest.
    print(f"  • Manifest: {_rel(manifest_path)}")
    # Return all exported segment metadata records to the caller.
    return records
