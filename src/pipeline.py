"""
src/pipeline.py
===============
End-to-End Ingestion, Extraction, Classification, Segmentation, SNR and Transcription Pipeline

Purpose:
--------
This is the master orchestrator script. It coordinates the stages of the
Arabic Labelled Speech Corpus project that exist so far:

  Stage 1: Ingestion & Metadata (src.ingest for links, src.local_ingest for files)
  Stage 2: Audio Extraction & Validation (src.audio_extractor)
  Stage 3: Waveform & Spectrogram Visualization (src.visualizer)
  Stage 4: 4-Class Region Classification (src.classifier)
  Stage 5: Speech Segmentation into 5-15s Clips (src.segmenter)
  Stage 6: Signal-to-Noise Ratio per clip (src.snr_estimator)
  Stage 7: Whisper transcription of every clip, engine A (src.transcriber)

The input can be either a video link (YouTube etc.) or the path of an audio/video
file on your computer.

It can be run interactively, from the command line, or invoked by a UI:

    python src/pipeline.py                                   # asks you questions
    python src/pipeline.py "https://www.youtube.com/watch?v=..."
    python src/pipeline.py "C:\\Users\\me\\Desktop\\lecture.mp3" --language auto
    python src/pipeline.py "C:\\audio\\talk.m4a" --no-transcribe
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

# Ensure project root (speech-to-text-corpus) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import ingest_video
from src.local_ingest import ingest_local_file
from src.visualizer import plot_waveform_and_spectrogram, plot_full_audio_envelope
from src.classifier import classify_audio
from src.segmenter import export_segments
from src.snr_estimator import compute_snr_for_video, save_snr_records
from src import transcriber

METADATA_DIR = PROJECT_ROOT / "metadata"

DEFAULT_NOTES_URL = "Downloaded for Arabic labelled speech corpus research."
DEFAULT_NOTES_FILE = "Local file provided for Arabic labelled speech corpus research."


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _banner(text: str) -> None:
    print("\n" + "=" * 65)
    print(text)
    print("=" * 65)


def _looks_like_url(text: str) -> bool:
    text = text.strip().lower()
    return bool(re.match(r"^[a-z][a-z0-9+.-]*://", text)) or text.startswith(
        ("www.", "youtu.be/", "youtube.com/", "m.youtube.com/")
    )


def resolve_source(text: str) -> Tuple[str, Union[str, Path]]:
    """Decide whether the user typed a link or a local file.

    Returns ("url", <url string>) or ("file", <Path>).  Quotes around a Windows
    path ("Copy as path" adds them) are removed.  Raises ValueError if the text is
    neither a link nor an existing file.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Nothing entered.")
    if _looks_like_url(raw):
        return "url", raw
    candidate = Path(raw.strip('"').strip("'")).expanduser()
    if candidate.is_file():
        return "file", candidate
    raise ValueError(
        f"'{raw}' is neither a link nor an existing file. "
        "For a file, give the full path, e.g. C:\\Users\\you\\Desktop\\lecture.mp3"
    )


def _archive_stale_outputs(video_id: str, metadata_dir: Path = METADATA_DIR) -> list:
    """Set aside SNR and transcript files from an earlier run of this video.

    Stage 5 deletes and renumbers a video's clips, so seg_000007 may now be different
    audio.  The transcriber skips segments that already have text, so leaving the old
    transcripts in place would silently attach them to the wrong audio.  Instead of
    deleting them, they are renamed to <name>.stale_<timestamp>.json.
    """
    archived = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in ("_snr.json", "_transcripts.json"):
        path = Path(metadata_dir) / f"{video_id}{suffix}"
        if path.exists():
            target = path.with_name(f"{path.stem}.stale_{stamp}.json")
            path.rename(target)
            archived.append(target.name)
    if archived:
        print(f"[pipeline] Clips were re-segmented, so earlier results were set aside: {', '.join(archived)}")
    return archived


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    url: str,
    notes: str = None,
    max_segments: int = None,
    run_snr: bool = True,
    run_transcription: bool = False,
    language: Optional[str] = "ar",
) -> dict:
    """
    Executes the end-to-end pipeline for a link OR a local audio/video file.

    The first argument keeps its old name (``url``) so existing callers still work,
    but it may also be the path of a file on your computer.

    run_snr            -- Stage 6 (cheap; on by default)
    run_transcription  -- Stage 7, Whisper on the GPU (off by default for callers such
                          as a UI; the command line turns it on)
    language           -- Whisper language: "ar" (default), another code, or None/"auto"
                          to detect per clip

    Returns a dictionary of all created metadata, plots, regions, and segments.
    """
    kind, source = resolve_source(url)

    _banner(">>> STAGE 1 & 2: INGESTION, EXTRACTION, AND VALIDATION"
            + (" (LOCAL FILE)" if kind == "file" else " (LINK)"))
    if kind == "url":
        metadata = ingest_video(url=source, notes=notes)
    else:
        metadata = ingest_local_file(source, notes=notes)
    video_id = metadata["video_id"]

    processed_rel_path = metadata["audio_properties"]["processed_file"]
    wav_path = PROJECT_ROOT / processed_rel_path
    total_duration = metadata["audio_properties"]["duration_seconds"]

    _banner(">>> STAGE 3: AUDIO VISUALIZATION")
    slice_start = 60.0 if total_duration > 70.0 else 0.0
    slice_duration = min(10.0, total_duration)

    spec_plot = plot_waveform_and_spectrogram(
        wav_path=wav_path,
        start_sec=slice_start,
        duration_sec=slice_duration,
        title_prefix=f"Preview: {metadata['title'][:25]}"
    )
    env_plot = plot_full_audio_envelope(wav_path=wav_path)

    _banner(">>> STAGE 4: REGION CLASSIFICATION (Speech / Silence / Music / Mixed)")
    regions_file = wav_path.parent / f"{video_id}_regions.json"
    regions = classify_audio(wav_path=wav_path, output_path=regions_file)

    _banner(">>> STAGE 5: SPEECH SEGMENTATION (5-15s Clips)")
    segments = export_segments(
        wav_path=wav_path,
        regions=regions,
        video_id=video_id,
        max_segments=max_segments
    )

    # Clips were just rewritten: results computed on the old clips are no longer valid.
    archived = _archive_stale_outputs(video_id)

    snr_records = None
    if run_snr:
        _banner(">>> STAGE 6: SIGNAL-TO-NOISE RATIO (WADA + frame-energy)")
        snr_records = compute_snr_for_video(video_id)
        if snr_records:
            save_snr_records(video_id, snr_records)
            wada = sorted(r["snr_wada_db"] for r in snr_records if r["snr_wada_db"] is not None)
            if wada:
                print(f"[pipeline] WADA SNR over {len(wada)} clips: "
                      f"min {wada[0]:.1f} / median {wada[len(wada) // 2]:.1f} / max {wada[-1]:.1f} dB")
        else:
            print("[pipeline] No segments to measure.")

    transcripts_path = None
    if run_transcription:
        _banner(">>> STAGE 7: WHISPER TRANSCRIPTION (engine A)")
        if not segments:
            print("[pipeline] No segments to transcribe.")
        else:
            model = transcriber.load_whisper()
            try:
                transcripts_path = transcriber.transcribe_video(
                    video_id, model,
                    language=None if language in (None, "auto") else language,
                )
            finally:
                transcriber.unload_model(model)   # free the GPU for engine B
                model = None

    # Summary
    _banner("   PIPELINE EXECUTION COMPLETE — SUMMARY OF ASSETS CREATED   ")
    print(f"  • Source:         {'local file' if kind == 'file' else 'link'}")
    print(f"  • Video ID:       {video_id}")
    print(f"  • Title:          {metadata['title']}")
    print(f"  • Duration:       {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  • Processed WAV:  {processed_rel_path}")
    print(f"  • Metadata JSON:  metadata/{video_id}.json")
    print(f"  • Classified JSON:data/processed/{video_id}_regions.json ({len(regions)} regions)")
    print(f"  • Audio Segments: {len(segments)} clips in data/segments/")
    if snr_records is not None:
        print(f"  • SNR JSON:       metadata/{video_id}_snr.json ({len(snr_records)} clips)")
    if transcripts_path is not None:
        print(f"  • Transcripts:    metadata/{video_id}_transcripts.json")
    print(f"  • Inspection Plot:outputs/plots/{spec_plot.name}")
    print(f"  • Envelope Plot:  outputs/plots/{env_plot.name}")
    print("=" * 65)

    return {
        "metadata": metadata,
        "source_kind": kind,
        "wav_path": wav_path,
        "spec_plot": spec_plot,
        "env_plot": env_plot,
        "regions": regions,
        "segments": segments,
        "archived_results": archived,
        "snr_records": snr_records,
        "transcripts_path": transcripts_path,
    }


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def run_interactive_pipeline():
    """Interactive terminal prompt."""
    print("=" * 65)
    print("   ARABIC SPEECH CORPUS — MASTER PROCESSING PIPELINE         ")
    print("=" * 65)

    entered = input("\n[?] Enter a YouTube URL, or the path of a local audio/video file: ").strip()
    if not entered:
        print("[!] Nothing entered. Exiting.")
        return
    try:
        kind, _ = resolve_source(entered)
    except ValueError as exc:
        print(f"[!] {exc}")
        return

    notes = input("[?] Enter notes/permissions [press Enter for default]: ").strip()
    if not notes:
        notes = DEFAULT_NOTES_FILE if kind == "file" else DEFAULT_NOTES_URL

    answer = input("[?] Transcribe with Whisper after segmentation? [Y/n]: ").strip().lower()
    run_pipeline(url=entered, notes=notes, run_transcription=answer in ("", "y", "yes"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the corpus pipeline on a video link or a local audio/video file."
    )
    parser.add_argument("source", nargs="?",
                        help="video link or path of a local file (omit to be asked interactively)")
    parser.add_argument("--notes", help="notes/permissions to store in the metadata")
    parser.add_argument("--max-segments", type=int, help="export at most N clips (quick test)")
    parser.add_argument("--no-snr", action="store_true", help="skip Stage 6")
    parser.add_argument("--no-transcribe", action="store_true", help="skip Stage 7 (Whisper)")
    parser.add_argument("--language", default="ar",
                        help='Whisper language (default "ar"); "auto" detects per clip')
    args = parser.parse_args(argv)

    if args.source is None:
        run_interactive_pipeline()
        return 0
    try:
        kind, _ = resolve_source(args.source)
    except ValueError as exc:
        print(f"[!] {exc}")
        return 1
    run_pipeline(
        url=args.source,
        notes=args.notes or (DEFAULT_NOTES_FILE if kind == "file" else DEFAULT_NOTES_URL),
        max_segments=args.max_segments,
        run_snr=not args.no_snr,
        run_transcription=not args.no_transcribe,
        language=args.language,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
