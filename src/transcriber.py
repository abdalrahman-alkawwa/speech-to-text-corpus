
"""
Stage 7: ASR transcription of segment clips -> JSON.

Engine A: faster-whisper, CTranslate2, CUDA fp16.
Engine B will later add its output to the same JSON records.

Output: metadata/<video_id>_transcripts.json

Features:
- Raw transcription text (normalization happens in Stage 8).
- Resumable: completed segments are skipped unless --force is used.
- Atomic JSON saving.
- Periodic saving and saving on errors / Ctrl+C.
- CUDA DLL registration for Windows.
- GPU memory cleanup after transcription.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import site
import sys
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import soundfile as sf


# --------------------------------------------------------------------------- #
# Project configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments"
METADATA_DIR = PROJECT_ROOT / "metadata"
DEFAULT_DOWNLOAD_ROOT = PROJECT_ROOT / ".cache" / "whisper"

EXPECTED_SAMPLE_RATE = 16_000
SAVE_EVERY_N = 10

WHISPER_MODEL = "large-v3-turbo"
WHISPER_ENGINE_TAG = "faster-whisper-large-v3-turbo"
WHISPER_LANGUAGE = "ar"
WHISPER_BEAM_SIZE = 5

# Keep Windows DLL directory handles alive after load_whisper() returns.
_DLL_HANDLES = []


# --------------------------------------------------------------------------- #
# JSON storage
# --------------------------------------------------------------------------- #

def transcripts_path(
    video_id: str,
    metadata_dir: Path = METADATA_DIR
) -> Path:
    return Path(metadata_dir) / f"{video_id}_transcripts.json"


def load_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load existing transcription records, keyed by segment_id."""

    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as fh:
        return {
            rec["segment_id"]: rec
            for rec in json.load(fh)
        }



def save_records(
    path: Path,
    records: Dict[str, Dict[str, Any]]
) -> None:
    """Atomically save records to JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(
            [records[k] for k in sorted(records)],
            fh,
            ensure_ascii=False,
            indent=2
        )

    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Engine A: faster-whisper
# --------------------------------------------------------------------------- #

def load_whisper(
    model_name: str = WHISPER_MODEL,
    device: str = "cuda",
    compute_type: str = "float16",
    download_root: Path = DEFAULT_DOWNLOAD_ROOT,
):
    """Load faster-whisper and configure NVIDIA CUDA DLL paths."""

    global _DLL_HANDLES

    # Locate NVIDIA packages in the active virtual environment.
    nvidia_path = Path(site.getsitepackages()[-1]) / "nvidia"

    # Find all NVIDIA package bin directories.
    dll_dirs = [
        path for path in nvidia_path.glob("*/bin")
        if path.is_dir()
    ]

    # Register directories with the Windows DLL loader.
    _DLL_HANDLES = [
        os.add_dll_directory(str(path))
        for path in dll_dirs
    ]

    # Also add them to this process's PATH.
    os.environ["PATH"] = (
        os.pathsep.join(str(path) for path in dll_dirs)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    # Import after configuring DLL search paths.
    from faster_whisper import WhisperModel

    Path(download_root).mkdir(parents=True, exist_ok=True)

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(download_root)
    )


def unload_model(model: Any) -> None:
    """
    Release GPU memory after transcription.

    The caller must also remove its own model reference.
    """

    del model
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except ImportError:
        pass


def transcribe_whisper(
    model: Any,
    wav_path: Path,
    language: Optional[str] = WHISPER_LANGUAGE
) -> Dict[str, Any]:
    """Transcribe one audio clip and return its ASR-A fields."""

    chunks_iter, info = model.transcribe(
        str(wav_path),
        language=language,   # None = auto-detect per clip
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    # faster-whisper returns a generator.
    # Materialize it to execute transcription.
    chunks = list(chunks_iter)

    # Combine the text produced by all chunks.
    text = " ".join(
        c.text.strip()
        for c in chunks
        if c.text.strip()
    ).strip()

    # Weight each chunk by its duration.
    weights = [
        max(float(c.end) - float(c.start), 1e-3)
        for c in chunks
    ]

    if chunks:

        # Duration-weighted average log probability.
        avg_logprob = (
            sum(
                float(c.avg_logprob) * w
                for c, w in zip(chunks, weights)
            )
            / sum(weights)
        )

        # Worst-case diagnostic values across chunks.
        no_speech = max(
            float(c.no_speech_prob)
            for c in chunks
        )

        compression = max(
            float(c.compression_ratio)
            for c in chunks
        )

    else:
        # Model returned no chunks.
        avg_logprob = None
        no_speech = None
        compression = None

    return {
        "asr_a_engine": WHISPER_ENGINE_TAG,
        "asr_a_text": text,

        "asr_a_avg_logprob": (
            None if avg_logprob is None
            else round(avg_logprob, 4)
        ),

        "asr_a_no_speech_prob": (
            None if no_speech is None
            else round(no_speech, 4)
        ),

        "asr_a_compression_ratio": (
            None if compression is None
            else round(compression, 3)
        ),

        "asr_a_language": info.language,
        "asr_a_language_prob": round(
            float(info.language_probability), 4
        ),

        "asr_a_n_chunks": len(chunks),
    }


# --------------------------------------------------------------------------- #
# Driver helpers
# --------------------------------------------------------------------------- #

def _check_wav(wav_path: Path) -> float:
    """Validate audio format and return duration in seconds."""

    info = sf.info(str(wav_path))

    problems = []

    if info.samplerate != EXPECTED_SAMPLE_RATE:
        problems.append(f"sample rate {info.samplerate}")

    if info.channels != 1:
        problems.append(f"{info.channels} channels")

    if info.subtype != "PCM_16":
        problems.append(f"subtype {info.subtype}")

    if problems:
        raise ValueError(
            f"{wav_path.name}: not 16 kHz mono PCM_16 "
            f"({', '.join(problems)})"
        )

    return float(info.duration)


def video_id_from_segment(wav_path: Path) -> str:
    """Extract video ID from a segment filename."""

    return wav_path.stem.rsplit("_seg_", 1)[0]


# --------------------------------------------------------------------------- #
# Video transcription
# --------------------------------------------------------------------------- #

def transcribe_video(
    video_id: str,
    model: Any,
    segments_dir: Path = SEGMENTS_DIR,
    metadata_dir: Path = METADATA_DIR,
    force: bool = False,
    limit: Optional[int] = None,
    language: Optional[str] = WHISPER_LANGUAGE,
) -> Path:
    """Transcribe all pending segments belonging to one video."""

    wavs = sorted(
        Path(segments_dir).glob(f"{video_id}_seg_*.wav")
    )

    out_path = transcripts_path(video_id, metadata_dir)

    records = load_records(out_path)

    # Find segments that have not already been transcribed.
    pending = [
        w for w in wavs
        if force or records.get(w.stem, {}).get("asr_a_text") is None
    ]

    # Apply optional limit for testing.
    todo = pending[:limit] if limit is not None else pending

    print(
        f"[{video_id}] {len(wavs)} segments, "
        f"{len(wavs) - len(pending)} already done, "
        f"{len(todo)} to transcribe now"
    )

    done = 0
    audio_sec = 0.0
    start = time.time()

    try:

        for wav in todo:

            # Validate the segment before transcription.
            duration = _check_wav(wav)

            rec = records.get(wav.stem, {})

            # Preserve existing fields, including future Engine B fields.
            rec.update({
                "segment_id": wav.stem,

                "audio_path": (
                    str(wav.relative_to(PROJECT_ROOT)).replace("\\", "/")
                    if PROJECT_ROOT in wav.parents
                    else str(wav)
                ),

                "duration_sec": round(duration, 3),
            })

            # Run Engine A transcription.
            rec.update(
                transcribe_whisper(model, wav, language=language)
            )

            records[wav.stem] = rec

            done += 1
            audio_sec += duration

            # Save periodically.
            if done % SAVE_EVERY_N == 0:

                save_records(out_path, records)

                rtf = (
                    (time.time() - start)
                    / max(audio_sec, 1e-6)
                )

                print(
                    f"  {done}/{len(todo)} done "
                    f"(processing time = {rtf:.2f}x audio duration)"
                )

    finally:

        # Save progress even if an exception or Ctrl+C occurs.
        if records:
            save_records(out_path, records)

    if done:
        elapsed = time.time() - start
        print(
            f"[{video_id}] transcribed {done} segments ({audio_sec:.0f}s of audio) "
            f"in {elapsed:.1f}s -> processing time = "
            f"{elapsed / max(audio_sec, 1e-6):.2f}x audio duration"
        )

    print(f"[{video_id}] saved -> {out_path}")

    return out_path


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #

def main(
    argv: Optional[Sequence[str]] = None
) -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Stage 7 (engine A): "
            "transcribe segments with faster-whisper."
        )
    )

    parser.add_argument(
        "--video-id",
        action="append",
        help="Limit to this video ID (repeatable)."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe segments that already have text."
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Transcribe at most N segments per video."
    )

    parser.add_argument(
        "--language",
        default=WHISPER_LANGUAGE,
        help='Whisper language code (default "ar"); "auto" detects per clip.'
    )

    parser.add_argument("--model", default=WHISPER_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")

    parser.add_argument(
        "--download-root",
        default=str(DEFAULT_DOWNLOAD_ROOT)
    )

    args = parser.parse_args(argv)

    # Determine which videos to process.
    video_ids = (
        args.video_id
        or sorted({
            video_id_from_segment(p)
            for p in SEGMENTS_DIR.glob("*_seg_*.wav")
        })
    )

    if not video_ids:
        print(
            f"No segment WAVs found in {SEGMENTS_DIR}. "
            "Run Stage 5 first."
        )
        return 1

    # Load the model.
    model = load_whisper(
        args.model,
        args.device,
        args.compute_type,
        Path(args.download_root)
    )

    try:

        for video_id in video_ids:

            transcribe_video(
                video_id,
                model,
                force=args.force,
                limit=args.limit,
                language=None if args.language == "auto" else args.language
            )

    finally:

        # Release model and GPU memory.
        unload_model(model)
        model = None

    return 0


if __name__ == "__main__":
    sys.exit(main())
