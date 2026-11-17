"""
src/transcriber.py
==================
Stage 7: Dual-Engine ASR Transcription

Runs two independent ASR engines on each speech segment:
  Engine A: faster-whisper (large-v3-turbo) via CTranslate2 — CUDA fp16
  Engine B: facebook/mms-1b-all via HuggingFace Transformers — Arabic target

Models are loaded sequentially and VRAM is released between them to stay
within the 8 GB budget.
"""

import sys
import os
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register NVIDIA CUDA DLLs on Windows into PATH and DLL directory
if sys.platform == "win32":
    try:
        import site
        for site_pkg in site.getsitepackages():
            nvidia_base = Path(site_pkg) / "nvidia"
            if nvidia_base.exists():
                for bin_dir in nvidia_base.glob("*/bin"):
                    bin_str = str(bin_dir)
                    os.add_dll_directory(bin_str)
                    if bin_str not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

from typing import List, Dict, Any, Optional
import numpy as np
import soundfile as sf
import torch


def _release_gpu():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
#  ENGINE A: faster-whisper (large-v3-turbo)
# ============================================================

def transcribe_whisper(
    wav_paths: List[Path],
    model_size: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    language: str = "ar",
    device: str = "cuda",
    compute_type: str = "float16"
) -> List[Dict[str, Any]]:
    """
    Transcribes a batch of WAV files using faster-whisper on GPU.
    Returns list of dicts with segment_id, text, and per-segment info.
    """
    from faster_whisper import WhisperModel

    print(f"\n[transcriber] Loading Whisper model: {model_size}")
    print(f"[transcriber] Device: {device}, Compute: {compute_type}")

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    results = []

    for i, wav_path in enumerate(wav_paths, 1):
        wav_path = Path(wav_path).resolve()
        print(f"[transcriber] Whisper [{i}/{len(wav_paths)}]: {wav_path.name}")

        # Load audio directly via SoundFile as float32 to bypass PyAV metadata bugs
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        segments_iter, info = model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=False,  # We already did VAD in Stage 4
        )

        # Collect all text segments
        text_parts = []
        word_data = []
        for seg in segments_iter:
            text_parts.append(seg.text.strip())
            word_data.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            })

        full_text = " ".join(text_parts).strip()

        results.append({
            "file": wav_path.name,
            "engine": "whisper-large-v3-turbo",
            "language_detected": info.language,
            "language_probability": round(info.language_probability, 3),
            "text": full_text,
            "segments": word_data,
        })

    # Release model from VRAM
    del model
    _release_gpu()
    print("[transcriber] Whisper model released from VRAM.")

    return results


# ============================================================
#  ENGINE B: facebook/mms-1b-all (Arabic)
# ============================================================

def transcribe_mms(
    wav_paths: List[Path],
    model_name: str = "facebook/mms-1b-all",
    language: str = "ara",
    device: str = "cuda"
) -> List[Dict[str, Any]]:
    """
    Transcribes WAV files using Meta MMS-1B via HuggingFace Transformers.
    Uses the 'ara' (Arabic) language adapter.
    """
    from transformers import Wav2Vec2ForCTC, AutoProcessor

    print(f"\n[transcriber] Loading MMS model: {model_name}")
    print(f"[transcriber] Language adapter: {language}")

    import socket

    def _is_online(host="huggingface.co", port=443, timeout=1.5) -> bool:
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            s.close()
            return True
        except Exception:
            return False

    cache_hub = Path.home() / ".cache" / "huggingface" / "hub"
    is_cached = cache_hub.exists() and any("mms-1b-all" in p.name for p in cache_hub.glob("models--*") if p.is_dir())

    if not is_cached and not _is_online():
        print(f"[transcriber] MMS model is not cached and system is offline. Using local cached Whisper-tiny as Engine B...")
        from faster_whisper import WhisperModel
        tiny_model = WhisperModel("Systran/faster-whisper-tiny", device=device, compute_type="float16")
        results = []
        for i, wav_path in enumerate(wav_paths, 1):
            wav_path = Path(wav_path).resolve()
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            segments_iter, _ = tiny_model.transcribe(audio, language="ar")
            text = " ".join([s.text.strip() for s in segments_iter]).strip()
            results.append({
                "file": wav_path.name,
                "engine": "whisper-tiny-local",
                "language_adapter": "ar",
                "text": text,
            })
        del tiny_model
        _release_gpu()
        return results

    try:
        processor = AutoProcessor.from_pretrained(model_name)
        processor.tokenizer.set_target_lang(language)

        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        model.load_adapter(language)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"[transcriber] Could not load MMS ({e}). Falling back to cached local Whisper-tiny...")
        from faster_whisper import WhisperModel
        tiny_model = WhisperModel("Systran/faster-whisper-tiny", device=device, compute_type="float16")
        results = []
        for i, wav_path in enumerate(wav_paths, 1):
            wav_path = Path(wav_path).resolve()
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            segments_iter, _ = tiny_model.transcribe(audio, language="ar")
            text = " ".join([s.text.strip() for s in segments_iter]).strip()
            results.append({
                "file": wav_path.name,
                "engine": "whisper-tiny-local",
                "language_adapter": "ar",
                "text": text,
            })
        del tiny_model
        _release_gpu()
        return results

    results = []

    for i, wav_path in enumerate(wav_paths, 1):
        wav_path = Path(wav_path).resolve()
        print(f"[transcriber] MMS [{i}/{len(wav_paths)}]: {wav_path.name}")

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # MMS expects 16kHz — our segments are already 16kHz
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0].strip()

        results.append({
            "file": wav_path.name,
            "engine": "mms-1b-all",
            "language_adapter": language,
            "text": text,
        })

    # Release model from VRAM
    del model, processor
    _release_gpu()
    print("[transcriber] MMS model released from VRAM.")

    return results


# ============================================================
#  DUAL-ENGINE ORCHESTRATOR
# ============================================================

def transcribe_segments(
    segment_dir: Path,
    video_id: str,
    max_segments: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Runs both ASR engines sequentially on all segments for a given video_id.
    Returns a merged list of records with both transcripts.
    """
    segment_dir = Path(segment_dir).resolve()
    wav_files = sorted(segment_dir.glob(f"{video_id}_seg_*.wav"))

    if max_segments is not None:
        wav_files = wav_files[:max_segments]

    if not wav_files:
        print(f"[transcriber] No segments found for {video_id} in {segment_dir}")
        return []

    print(f"\n[transcriber] Found {len(wav_files)} segments for {video_id}")

    # Engine A: faster-whisper
    print("\n" + "=" * 50)
    print(">>> ENGINE A: faster-whisper (large-v3-turbo)")
    print("=" * 50)
    whisper_results = transcribe_whisper(wav_files)

    # Engine B: MMS-1B
    print("\n" + "=" * 50)
    print(">>> ENGINE B: facebook/mms-1b-all (Arabic)")
    print("=" * 50)
    mms_results = transcribe_mms(wav_files)

    # Merge results by filename
    merged = []
    mms_lookup = {r["file"]: r for r in mms_results}

    for wr in whisper_results:
        fname = wr["file"]
        mr = mms_lookup.get(fname, {})
        merged.append({
            "segment_id": fname.replace(".wav", ""),
            "audio_file": fname,
            "transcript_a": wr.get("text", ""),
            "engine_a": wr.get("engine", "whisper-large-v3-turbo"),
            "language_detected": wr.get("language_detected", "ar"),
            "language_probability": wr.get("language_probability", 0.0),
            "transcript_b": mr.get("text", ""),
            "engine_b": mr.get("engine", "mms-1b-all"),
        })

    print(f"\n[transcriber] Dual transcription complete: {len(merged)} segments processed.")
    return merged


if __name__ == "__main__":
    segments_dir = PROJECT_ROOT / "data" / "segments"
    results = transcribe_segments(segments_dir, "gK017cCH36k", max_segments=2)
    for r in results:
        print(f"\n--- {r['segment_id']} ---")
        print(f"  Whisper: {r['transcript_a'][:80]}")
        print(f"  MMS:     {r['transcript_b'][:80]}")
