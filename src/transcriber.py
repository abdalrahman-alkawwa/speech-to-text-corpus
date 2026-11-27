
"""
Arabic Speech Corpus — Transcription Stage (Stage 7)

Engine A: faster-whisper large-v3-turbo
Engine B: Cohere Transcribe Arabic (local Hugging Face model)

Both engines process the same segment WAV files independently.

Engine B runs locally; no Cohere API key or internet connection
is needed after the model has been downloaded and cached.


Output:
    metadata/<video_id>_transcripts.json

The existing transcript records are preserved. Each engine updates
only its own fields, so running one engine does not overwrite the
other engine's results.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
_DLL_HANDLES = []

if sys.platform == "win32":
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"

    cuda_dll_dirs = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
    ]

    for dll_dir in cuda_dll_dirs:
        if dll_dir.is_dir():
            _DLL_HANDLES.append(
                os.add_dll_directory(str(dll_dir))
            )

    # Also expose the DLL directories through PATH.
    os.environ["PATH"] = (
        os.pathsep.join(str(p) for p in cuda_dll_dirs if p.is_dir())
        + os.pathsep
        + os.environ["PATH"]
    )
import time


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEGMENTS_DIR = PROJECT_ROOT / "data" / "segments"
METADATA_DIR = PROJECT_ROOT / "metadata"

DEFAULT_DOWNLOAD_ROOT = PROJECT_ROOT / ".cache" / "whisper"

EXPECTED_SAMPLE_RATE = 16000
SAVE_EVERY_N = 10


# ============================================================
# 2. ENGINE A — FASTER-WHISPER
# ============================================================

WHISPER_MODEL = "large-v3-turbo"
WHISPER_ENGINE_TAG = "faster-whisper-large-v3-turbo"

WHISPER_LANGUAGE = "auto"
WHISPER_BEAM_SIZE = 5


# ============================================================
# 3. ENGINE B — COHERE TRANSCRIBE ARABIC (LOCAL)
# ============================================================

COHERE_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
COHERE_ENGINE_TAG = "cohere-transcribe-arabic-07-2026-local"

COHERE_LANGUAGE = "ar"
SUPPORTED_COHERE_LANGUAGES = {"ar", "en"}
# The Cohere model card's offline quick start uses 256. 512 allowed pathological
# repetition loops to consume the entire generation budget on short clips.
COHERE_MAX_NEW_TOKENS = 256


# ============================================================
# 4. TRANSCRIPT FILE HELPERS
# ============================================================

def transcripts_path(video_id: str) -> Path:
    """Return the JSON file containing all transcripts for a video."""
    return METADATA_DIR / f"{video_id}_transcripts.json"


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load transcript records, returning an empty list if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Transcript metadata not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON list in {path}, got {type(records).__name__}"
        )

    return records


def save_records(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Save transcript records safely.

    A temporary file is written first. os.replace() then replaces the
    original file, reducing the risk of losing metadata if interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    os.replace(temporary_path, path)


# ============================================================
# 5. FIND THE WAV FILE ASSOCIATED WITH A SEGMENT
# ============================================================

def find_segment_audio(
    video_id: str,
    record: Dict[str, Any],
) -> Path:
    """
    Locate a segment WAV file.

    First, try any audio path already stored in the record.
    Then try the common segment directory layouts.
    Finally, search under data/segments for the segment ID.
    """

    segment_id = str(record.get("segment_id", "")).strip()

    if not segment_id:
        raise ValueError("A transcript record is missing segment_id.")

    # Try paths stored in the metadata record.
    possible_path_fields = (
        "wav_path",
        "audio_path",
        "audio_file",
        "segment_path",
        "file_path",
    )

    for field in possible_path_fields:
        stored_path = record.get(field)

        if not stored_path:
            continue

        candidate = Path(str(stored_path))

        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

        if candidate.is_file():
            return candidate

    # Try common directory layouts.
    candidates = [
        SEGMENTS_DIR / video_id / f"{segment_id}.wav",
        SEGMENTS_DIR / f"{video_id}_{segment_id}.wav",
        SEGMENTS_DIR / f"{segment_id}.wav",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Fallback: search for a WAV whose filename matches the segment ID.
    if SEGMENTS_DIR.exists():
        matches = list(SEGMENTS_DIR.rglob(f"{segment_id}.wav"))

        if matches:
            # Prefer a match inside a directory named after this video.
            for match in matches:
                if video_id in str(match):
                    return match

            return matches[0]

    raise FileNotFoundError(
        f"Could not find WAV for segment '{segment_id}' "
        f"belonging to video '{video_id}'."
    )


# ============================================================
# 6. MODEL MEMORY CLEANUP
# ============================================================

def unload_model() -> None:
    """
    Release unused Python objects and GPU memory.

    This is particularly important because Engine A and Engine B
    should not occupy GPU memory at the same time.
    """
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    except Exception:
        pass


# ============================================================
# 7. LOAD ENGINE A — FASTER-WHISPER
# ============================================================

def load_whisper():
    """Load faster-whisper, with Windows DLL setup when necessary."""

    if os.name == "nt":
        try:
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"

            if torch_lib.is_dir() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))

        except Exception:
            pass

    from faster_whisper import WhisperModel

    print(f"\nLoading Engine A: {WHISPER_MODEL}")

    model = WhisperModel(
        WHISPER_MODEL,
        device="cuda",
        compute_type="float16",
        download_root=str(DEFAULT_DOWNLOAD_ROOT),
    )

    print("Engine A loaded.")

    return model


# ============================================================
# 8. ENGINE A — TRANSCRIBE ONE SEGMENT
# ============================================================

def transcribe_whisper(
    model,
    wav_path: Path,
    language: str = "auto",
) -> Dict[str, Any]:
    """
    Transcribe one WAV file using faster-whisper.

    The returned metadata includes the same diagnostic fields
    used by the evaluator.
    """

    segments, info = model.transcribe(
        str(wav_path),
        language=None if language == "auto" else language,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=False,
        condition_on_previous_text=False,
        word_timestamps=False,
    )

    text_parts = []

    weighted_logprob_sum = 0.0
    total_duration = 0.0

    no_speech_values = []
    compression_values = []

    for segment in segments:
        text_parts.append(segment.text.strip())

        duration = max(
            0.0,
            float(segment.end) - float(segment.start),
        )

        if segment.avg_logprob is not None and duration > 0:
            weighted_logprob_sum += (
                float(segment.avg_logprob) * duration
            )
            total_duration += duration

        if segment.no_speech_prob is not None:
            no_speech_values.append(float(segment.no_speech_prob))

        if segment.compression_ratio is not None:
            compression_values.append(float(segment.compression_ratio))

    transcript = " ".join(part for part in text_parts if part).strip()

    average_logprob = (
        weighted_logprob_sum / total_duration
        if total_duration > 0
        else None
    )

    return {
        "asr_a_engine": WHISPER_ENGINE_TAG,
        "asr_a_text": transcript,
        "asr_a_language": getattr(info, "language", None),
        "asr_a_language_mode": language,
        "asr_a_language_probability": getattr(
            info, "language_probability", None
        ),
        "asr_a_avg_logprob": average_logprob,
        "asr_a_no_speech_prob": (
            max(no_speech_values) if no_speech_values else None
        ),
        "asr_a_compression_ratio": (
            max(compression_values) if compression_values else None
        ),
    }


# ============================================================
# 9. LOAD ENGINE B — LOCAL COHERE TRANSCRIBE ARABIC
# ============================================================

def load_cohere():
    """
    Load the official Cohere Transcribe Arabic model locally.

    On an NVIDIA GPU, use FP16 to reduce memory consumption.
    If CUDA is unavailable, fall back to CPU inference.
    """

    import torch

    try:
        from transformers import (
            AutoProcessor,
            CohereAsrForConditionalGeneration,
        )
    except ImportError as exc:
        raise ImportError(
            "Your Transformers installation does not include "
            "CohereAsrForConditionalGeneration.\n"
            "Install Transformers 5.4.0 or newer."
        ) from exc

    if torch.cuda.is_available():
        device = "cuda"
        model_dtype = torch.float16

        print("\nCUDA detected.")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        free_memory, total_memory = torch.cuda.mem_get_info()

        print(
            "GPU memory: "
            f"{free_memory / (1024 ** 3):.2f} GB free / "
            f"{total_memory / (1024 ** 3):.2f} GB total"
        )

    else:
        device = "cpu"
        model_dtype = torch.float32

        print("\nCUDA not detected. Cohere will run on CPU.")
        print("CPU inference will be considerably slower.")

    print(f"\nLoading Engine B: {COHERE_MODEL}")
    print("The model will be downloaded the first time it is used.")

    processor = AutoProcessor.from_pretrained(COHERE_MODEL)

    model = CohereAsrForConditionalGeneration.from_pretrained(
        COHERE_MODEL,
        torch_dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    print("Engine B loaded.")

    return processor, model


# ============================================================
# 10. ENGINE B — TRANSCRIBE ONE SEGMENT
# ============================================================

def transcribe_cohere(
    processor,
    model,
    wav_path: Path,
    language: str = COHERE_LANGUAGE,
    language_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe one WAV file using local Cohere Transcribe Arabic.

    The corpus segmenter is expected to produce 16 kHz WAV files.
    Cohere's processor receives the waveform and language setting.
    """

    import torch

    audio, sample_rate = sf.read(
        str(wav_path),
        dtype="float32",
    )

    # Convert stereo audio to mono if necessary.
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio after conversion: {wav_path}"
        )

    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_RATE} Hz audio, "
            f"but found {sample_rate} Hz in {wav_path}."
        )

    # Convert to the format expected by the Cohere processor.
    inputs = processor(
        audio,
        sampling_rate=EXPECTED_SAMPLE_RATE,
        return_tensors="pt",
        language=language,
    )

    # Move input tensors to the same device as the model.
    inputs.to(
        model.device,
        dtype=model.dtype,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=COHERE_MAX_NEW_TOKENS,
        )

    generated_tokens, hit_token_limit = _generation_diagnostics(
        outputs=outputs,
        inputs=inputs,
        model=model,
        max_new_tokens=COHERE_MAX_NEW_TOKENS,
    )

    # Decode the generated tokens into text.
    decoded = processor.decode(
        outputs,
        skip_special_tokens=True,
    )

    # Some processor versions return a list; others return a string.
    if isinstance(decoded, (list, tuple)):
        transcript = decoded[0] if decoded else ""
    else:
        transcript = decoded

    transcript = str(transcript).strip()

    quality_flags = _transcript_quality_flags(transcript)
    if hit_token_limit:
        quality_flags.insert(0, "token_limit_reached")

    return {
        "asr_b_engine": COHERE_ENGINE_TAG,
        "asr_b_text": transcript,
        "asr_b_confidence": None,
        "asr_b_language": language,
        "asr_b_language_mode": language_mode or language,
        "asr_b_status": "transcribed",
        "asr_b_generated_tokens": generated_tokens,
        "asr_b_hit_token_limit": hit_token_limit,
        "asr_b_quality_flags": quality_flags,
    }


def _generation_diagnostics(outputs, inputs, model, max_new_tokens: int) -> Tuple[int, bool]:
    """Return generated-token count and whether generation stopped at its cap."""
    sequences = getattr(outputs, "sequences", outputs)
    shape = getattr(sequences, "shape", ())
    if not shape:
        return 0, False

    sequence_length = int(shape[-1])
    decoder_inputs = inputs.get("decoder_input_ids")
    if decoder_inputs is not None:
        prompt_length = int(decoder_inputs.shape[-1])
    else:
        # Encoder-decoder generation normally begins with one decoder-start token.
        prompt_length = 1 if getattr(model, "generation_config", None) is not None else 0

    generated_tokens = max(0, sequence_length - prompt_length)
    if generated_tokens < max_new_tokens:
        return generated_tokens, False

    # EOS at the final position means the model ended normally on the last
    # permitted token. Otherwise the transcript may have been cut off.
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos_token_id, (tuple, list, set)):
        eos_ids = {int(token_id) for token_id in eos_token_id}
    elif eos_token_id is None:
        eos_ids = set()
    else:
        eos_ids = {int(eos_token_id)}

    try:
        last_token = int(sequences[0, -1])
    except (IndexError, TypeError, ValueError):
        last_token = None
    return generated_tokens, last_token not in eos_ids


def _transcript_quality_flags(text: str) -> List[str]:
    """Flag obvious decoder degeneration without changing the raw transcript."""
    flags = []
    words = text.split()

    longest_word_run = 1
    current_word_run = 1
    for previous, current in zip(words, words[1:]):
        current_word_run = current_word_run + 1 if current == previous else 1
        longest_word_run = max(longest_word_run, current_word_run)

    if longest_word_run >= 8:
        flags.append("repetition_loop")

    visible = [char for char in text if not char.isspace()]
    if len(visible) >= 16 and visible:
        most_common_fraction = max(text.count(char) for char in set(visible)) / len(visible)
        if most_common_fraction >= 0.8:
            flags.append("character_loop")

    if any(len(word) >= 40 and len(set(word)) <= 3 for word in words):
        flags.append("character_loop")

    return sorted(set(flags))


# ============================================================
# 11. TRANSCRIBE ALL SEGMENTS — ENGINE A
# ============================================================

def transcribe_video(
    video_id: str,
    model=None,
    limit: Optional[int] = None,
    force: bool = False,
    language: str = WHISPER_LANGUAGE,
) -> None:
    """
    Run Engine A on every segment belonging to a video.

    If force=False, segments that already have a non-empty
    Engine A transcript are skipped.
    """

    path = transcripts_path(video_id)
    records = load_records(path)

    own_model = model is None

    if own_model:
        model = load_whisper()

    processed = 0
    skipped = 0

    try:
        for record in records:

            if limit is not None and processed >= limit:
                break

            existing_text = record.get("asr_a_text")

            if not force and existing_text:
                skipped += 1
                continue

            segment_id = record.get("segment_id", "unknown")
            wav_path = find_segment_audio(video_id, record)

            print(f"\n[Engine A] {segment_id}")

            result = transcribe_whisper(model, wav_path, language=language)

            record.update(result)

            processed += 1

            print(f"Transcript: {result['asr_a_text']}")

            if processed % SAVE_EVERY_N == 0:
                save_records(path, records)
                print("Progress saved.")

        save_records(path, records)

    finally:
        if own_model:
            del model
            unload_model()

    print(
        f"\nEngine A finished for {video_id}. "
        f"Processed: {processed}, skipped: {skipped}"
    )


# ============================================================
# 12. TRANSCRIBE ALL SEGMENTS — ENGINE B
# ============================================================

def transcribe_video_b(
    video_id: str,
    processor=None,
    model=None,
    limit: Optional[int] = None,
    force: bool = False,
    language: str = WHISPER_LANGUAGE,
) -> None:
    """
    Run local Cohere Transcribe Arabic on every segment.

    If force=False, segments that already have a non-empty
    Engine B transcript are skipped.

    Engine A fields are preserved.
    """

    path = transcripts_path(video_id)
    records = load_records(path)

    own_model = processor is None or model is None
    loaded_model = False

    processed = 0
    skipped = 0

    try:
        for record in records:

            if limit is not None and processed >= limit:
                break

            existing_text = record.get("asr_b_text")

            if not force and existing_text:
                skipped += 1
                continue

            segment_id = record.get("segment_id", "unknown")
            wav_path = find_segment_audio(video_id, record)

            print(f"\n[Engine B] {segment_id}")

            detected_language = str(record.get("asr_a_language") or "").lower().split("-")[0]
            target_language = (
                language if language in SUPPORTED_COHERE_LANGUAGES
                else detected_language
            )
            if target_language not in SUPPORTED_COHERE_LANGUAGES:
                result = {
                    "asr_b_engine": COHERE_ENGINE_TAG,
                    "asr_b_text": "",
                    "asr_b_confidence": None,
                    "asr_b_language": None,
                    "asr_b_language_mode": language,
                    "asr_b_status": "unsupported_language",
                    "asr_b_generated_tokens": 0,
                    "asr_b_hit_token_limit": False,
                    "asr_b_quality_flags": ["unsupported_language"],
                }
                print(
                    "Cohere skipped: Whisper detected an unsupported language "
                    f"({detected_language or 'unknown'})."
                )
            else:
                if own_model and not loaded_model:
                    processor, model = load_cohere()
                    loaded_model = True
                result = transcribe_cohere(
                    processor,
                    model,
                    wav_path,
                    language=target_language,
                    language_mode=language,
                )

            record.update(result)

            processed += 1

            print(f"Transcript: {result['asr_b_text']}")

            if processed % SAVE_EVERY_N == 0:
                save_records(path, records)
                print("Progress saved.")

        save_records(path, records)

    finally:
        if loaded_model:
            del model
            del processor
            unload_model()

    print(
        f"\nEngine B finished for {video_id}. "
        f"Processed: {processed}, skipped: {skipped}"
    )


# ============================================================
# 13. RUN THE SELECTED ENGINE(S)
# ============================================================

def run_engines(
    video_ids: List[str],
    engines: str = "a",
    limit: Optional[int] = None,
    force: bool = False,
    language: str = WHISPER_LANGUAGE,
) -> None:
    """
    Run Engine A, Engine B, or both.

    When both are selected, Engine A is unloaded before
    Engine B is loaded to reduce GPU memory pressure.
    """

    if engines not in ("a", "b", "both"):
        raise ValueError("engines must be 'a', 'b', or 'both'")
    if language not in ("auto", "ar", "en"):
        raise ValueError("language must be 'auto', 'ar', or 'en'")

    for video_id in video_ids:

        print("\n" + "=" * 65)
        print(f"PROCESSING VIDEO: {video_id}")
        print("=" * 65)

        if engines in ("a", "both"):

            print("\nStarting Engine A...")

            transcribe_video(
                video_id,
                limit=limit,
                force=force,
                language=language,
            )

            unload_model()

        if engines in ("b", "both"):

            print("\nStarting Engine B...")

            transcribe_video_b(
                video_id,
                limit=limit,
                force=force,
                language=language,
            )

            unload_model()


# ============================================================
# 14. FIND VIDEO IDS
# ============================================================

def discover_video_ids() -> List[str]:
    """Find all transcript metadata files and extract video IDs."""

    if not METADATA_DIR.exists():
        return []

    suffix = "_transcripts.json"

    video_ids = []

    for path in sorted(METADATA_DIR.glob(f"*{suffix}")):
        video_ids.append(path.name[:-len(suffix)])

    return video_ids


# ============================================================
# 15. COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Arabic speech corpus transcription"
    )

    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Process one video ID. If omitted, process all videos.",
    )

    parser.add_argument(
        "--engine",
        choices=["a", "b", "both"],
        default="a",
        help="Select Engine A, Engine B, or both.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of segments to process per video.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe segments that already have transcripts.",
    )

    parser.add_argument(
        "--language",
        choices=["auto", "ar", "en"],
        default="auto",
        help="Auto-detect Arabic/English, or force Arabic or English.",
    )

    args = parser.parse_args()

    if args.video_id:
        video_ids = [args.video_id]
    else:
        video_ids = discover_video_ids()

    if not video_ids:
        print("No transcript metadata files were found.")
        return

    print("=" * 65)
    print("ARABIC SPEECH CORPUS — TRANSCRIPTION")
    print("=" * 65)

    print(f"Videos found: {len(video_ids)}")
    print(f"Selected engine: {args.engine}")
    print(f"Segment limit: {args.limit or 'No limit'}")
    print(f"Force re-transcription: {args.force}")

    start_time = time.time()

    run_engines(
        video_ids=video_ids,
        engines=args.engine,
        limit=args.limit,
        force=args.force,
        language=args.language,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("TRANSCRIPTION COMPLETE")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print("=" * 65)


if __name__ == "__main__":
    main()
