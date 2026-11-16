"""
src/pipeline.py
===============
End-to-End Ingestion, Extraction, Classification, and Segmentation Pipeline

Purpose:
--------
This is the master orchestrator script. It coordinates the first five stages
of the Arabic Labelled Speech Corpus project:

  Stage 1: Ingestion & Metadata (src.ingest)
  Stage 2: Audio Extraction & Validation (src.audio_extractor)
  Stage 3: Waveform & Spectrogram Visualization (src.visualizer)
  Stage 4: 4-Class Region Classification (src.classifier)
  Stage 5: Speech Segmentation into 8–12s Clips (src.segmenter)

It can be run interactively via terminal or invoked programmatically by a UI.
"""

import sys
from pathlib import Path

# Ensure project root (speech-to-text-corpus) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_video
from src.visualizer import plot_waveform_and_spectrogram, plot_full_audio_envelope
from src.classifier import classify_audio
from src.segmenter import export_segments


def run_pipeline(url: str, notes: str = None, max_segments: int = None) -> dict:
    """
    Executes the end-to-end pipeline across Stages 1 through 5 for a given URL.
    Returns a dictionary of all created metadata, plots, regions, and segments.
    """
    print("\n" + "=" * 65)
    print(">>> STAGE 1 & 2: INGESTION, EXTRACTION, AND VALIDATION")
    print("=" * 65)
    metadata = ingest_video(url=url, notes=notes)
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
    print(f"  • Inspection Plot:outputs/plots/{spec_plot.name}")
    print(f"  • Envelope Plot:  outputs/plots/{env_plot.name}")
    print("=" * 65)

    return {
        "metadata": metadata,
        "wav_path": wav_path,
        "spec_plot": spec_plot,
        "env_plot": env_plot,
        "regions": regions,
        "segments": segments
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
