"""
src/pipeline.py
===============
End-to-End Ingestion, Extraction, Classification, Segmentation, SNR, and DB Ingestion

Covers Stages 1 through 10:
  Stage 1: Ingestion & Metadata (src.ingest)
  Stage 2: Audio Extraction & Validation (src.audio_extractor)
  Stage 3: Waveform & Spectrogram Visualization (src.visualizer)
  Stage 4: 4-Class Region Classification (src.classifier)
  Stage 5: Speech Segmentation into 8–12s Clips (src.segmenter)
  Stage 6: SNR Estimation (src.snr_estimator)
  Stage 7-9: Dual ASR, Normalization, & Triage (optional via run_asr=True)
  Stage 10: SQLite Corpus Persistence (src.database)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf
import numpy as np

from typing import Optional

from src.ingest import ingest_video, ingest_local_file
from src.visualizer import plot_waveform_and_spectrogram, plot_full_audio_envelope
from src.classifier import classify_audio
from src.segmenter import export_segments
from src.snr_estimator import estimate_file_snr, estimate_segment_snr
from src.transcript_store import upsert_segment


def run_pipeline(
    url: Optional[str] = None,
    local_file_path: Optional[Path] = None,
    notes: str = None,
    max_segments: int = None,
    run_asr: bool = False
) -> dict:
    """
    Executes the end-to-end pipeline across Stages 1 through 6 (+ optional 7-9) for a given URL or local file.
    Saves all segments and transcripts into data/transcripts/<id>_transcripts.json for review.
    """
    print("\n" + "=" * 65)
    print(">>> STAGE 1 & 2: INGESTION, EXTRACTION, AND VALIDATION")
    print("=" * 65)
    if local_file_path:
        metadata = ingest_local_file(file_path=Path(local_file_path), notes=notes)
    elif url:
        metadata = ingest_video(url=url, notes=notes)
    else:
        raise ValueError("Must provide either a video URL or a local file path.")

    video_id = metadata["video_id"]

    processed_rel_path = metadata["audio_properties"]["processed_file"]
    wav_path = PROJECT_ROOT / processed_rel_path
    total_duration = metadata["audio_properties"]["duration_seconds"]

    print("\n" + "=" * 65)
    print(">>> STAGE 3: AUDIO VISUALIZATION")
    print("=" * 65)
    slice_start = 60.0 if total_duration > 70.0 else 0.0
    slice_duration = min(10.0, total_duration)

    spec_plot = plot_waveform_and_spectrogram(
        wav_path=wav_path,
        start_sec=slice_start,
        duration_sec=slice_duration,
        title_prefix=f"Preview: {metadata['title'][:25]}"
    )
    env_plot = plot_full_audio_envelope(wav_path=wav_path)

    print("\n" + "=" * 65)
    print(">>> STAGE 4: REGION CLASSIFICATION (Speech / Silence / Music / Mixed)")
    print("=" * 65)
    regions_file = wav_path.parent / f"{video_id}_regions.json"
    regions = classify_audio(wav_path=wav_path, output_path=regions_file)

    print("\n" + "=" * 65)
    print(">>> STAGE 5: SPEECH SEGMENTATION (8–12s Clips)")
    print("=" * 65)
    segments = export_segments(
        wav_path=wav_path,
        regions=regions,
        video_id=video_id,
        max_segments=max_segments
    )

    print("\n" + "=" * 65)
    print(">>> STAGE 6: SNR ESTIMATION (WADA + Segmental)")
    print("=" * 65)
    snr_info = estimate_file_snr(wav_path=wav_path, regions=regions)

    # Calculate per-segment SNR and persist into JSON transcript store
    print("\n[pipeline] Calculating per-segment SNR and saving to data/transcripts/...")
    audio_full, sr = sf.read(str(wav_path), dtype="float32")
    if audio_full.ndim > 1:
        audio_full = np.mean(audio_full, axis=1)

    for seg in segments:
        seg_snr = estimate_segment_snr(
            audio=audio_full,
            sr=sr,
            start_sec=seg["start_sec"],
            end_sec=seg["end_sec"],
            regions=regions
        )
        seg["snr_wada_db"] = seg_snr["snr_wada_db"]
        seg["snr_segmental_db"] = seg_snr["snr_segmental_db"]

        # Persist segment to JSON store
        upsert_segment({
            "segment_id": seg["segment_id"],
            "video_id": video_id,
            "audio_path": seg["audio_path"],
            "start_sec": seg["start_sec"],
            "end_sec": seg["end_sec"],
            "duration_sec": seg["duration_sec"],
            "snr_wada_db": seg["snr_wada_db"],
            "snr_segmental_db": seg["snr_segmental_db"],
            "boundary_check": seg.get("boundary_check", "auto_verified"),
            "status": "unreviewed"
        })

    # Optional Stage 7-9: Dual ASR Transcription & Triage
    if run_asr and segments:
        print("\n" + "=" * 65)
        print(">>> STAGES 7–9: DUAL ASR, NORMALIZATION & TRIAGE")
        print("=" * 65)
        from src.transcriber import transcribe_segments
        from src.evaluator import evaluate_transcripts

        seg_dir = PROJECT_ROOT / "data" / "segments"
        dual_results = transcribe_segments(
            segment_dir=seg_dir,
            video_id=video_id,
            max_segments=max_segments
        )
        evaluated = evaluate_transcripts(dual_results)

        for ev in evaluated:
            upsert_segment({
                "segment_id": ev["segment_id"],
                "video_id": video_id,
                "transcript_a": ev.get("transcript_a"),
                "engine_a": ev.get("engine_a"),
                "transcript_b": ev.get("transcript_b"),
                "engine_b": ev.get("engine_b"),
                "normalized_a": ev.get("normalized_a"),
                "normalized_b": ev.get("normalized_b"),
                "wer": ev.get("wer"),
                "cer": ev.get("cer"),
                "triage": ev.get("triage"),
                "ground_truth": ev.get("transcript_a"),
            })

    # Summary
    print("\n" + "=" * 65)
    print("   PIPELINE EXECUTION COMPLETE — SUMMARY OF ASSETS CREATED   ")
    print("=" * 65)
    print(f"  • Video ID:       {video_id}")
    print(f"  • Title:          {metadata['title']}")
    print(f"  • Duration:       {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  • Processed WAV:  {processed_rel_path}")
    print(f"  • Metadata JSON:  metadata/{video_id}.json")
    print(f"  • Classified JSON:data/processed/{video_id}_regions.json ({len(regions)} regions)")
    print(f"  • Audio Segments: {len(segments)} clips in data/segments/")
    print(f"  • Transcripts:    data/transcripts/{video_id}_transcripts.json")
    print(f"  • Inspection Plot:outputs/plots/{spec_plot.name}")
    print(f"  • Envelope Plot:  outputs/plots/{env_plot.name}")
    print(f"  • WADA-SNR:       {snr_info['snr_wada_db']:.1f} dB")
    print(f"  • Segmental SNR:  {snr_info['snr_segmental_db']:.1f} dB")
    print("=" * 65)

    return {
        "metadata": metadata,
        "wav_path": wav_path,
        "spec_plot": spec_plot,
        "env_plot": env_plot,
        "regions": regions,
        "segments": segments,
        "snr": snr_info
    }


def run_interactive_pipeline():
    """Interactive terminal prompt."""
    print("=" * 65)
    print("   ARABIC SPEECH CORPUS — MASTER PROCESSING PIPELINE         ")
    print("=" * 65)

    url = input("\n[?] Enter YouTube video URL: ").strip()
    if not url:
        print("[!] No URL entered. Exiting.")
        return

    notes = input("[?] Enter notes/permissions [press Enter for default]: ").strip()
    if not notes:
        notes = "Downloaded for Arabic labelled speech corpus research."

    run_pipeline(url=url, notes=notes)


if __name__ == "__main__":
    run_interactive_pipeline()
