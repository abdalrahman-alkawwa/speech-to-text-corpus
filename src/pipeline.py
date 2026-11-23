
"""
src/pipeline.py
===============
End-to-End Arabic Speech Corpus Pipeline

Stages:
-------
Stage 1: Ingestion & Metadata
Stage 2: Audio Extraction & Validation
Stage 3: Waveform & Spectrogram Visualization
Stage 4: 4-Class Region Classification
Stage 5: Speech Segmentation into 5-15s Clips
Stage 6: Signal-to-Noise Ratio per clip
Stage 7: ASR Transcription (Whisper + Cohere)
Stage 8: Arabic Transcript Normalization
Stage 9: ASR Evaluation and Human-Review Triage

Usage:
------
python src/pipeline.py
python src/pipeline.py "https://www.youtube.com/watch?v=..."
python src/pipeline.py "C:\\audio\\talk.mp3"
python src/pipeline.py "C:\\audio\\talk.mp3" --no-transcribe
python src/pipeline.py "C:\\audio\\talk.mp3" --max-segments 3
"""

import argparse
import json
import re
import sys

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METADATA_DIR = PROJECT_ROOT / "metadata"


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from src.ingest import ingest_video
from src.local_ingest import ingest_local_file

from src.visualizer import (
    plot_waveform_and_spectrogram,
    plot_full_audio_envelope,
)

from src.classifier import classify_audio
from src.segmenter import export_segments

from src.snr_estimator import (
    compute_snr_for_video,
    save_snr_records,
)

from src import transcriber
from src import normalizer
from src import evaluator


# ============================================================
# 3. DEFAULT NOTES
# ============================================================

DEFAULT_NOTES_URL = (
    "Downloaded for Arabic labelled speech corpus research."
)

DEFAULT_NOTES_FILE = (
    "Local file provided for Arabic labelled speech corpus research."
)


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def _banner(text: str) -> None:
    """Print a section heading."""

    print("\n" + "=" * 65)
    print(text)
    print("=" * 65)


def _looks_like_url(text: str) -> bool:
    """Check whether the input looks like a URL."""

    text = text.strip().lower()

    return (
        bool(re.match(r"^[a-z][a-z0-9+.-]*://", text))
        or text.startswith(
            (
                "www.",
                "youtu.be/",
                "youtube.com/",
                "m.youtube.com/",
            )
        )
    )


def resolve_source(
    text: str,
) -> Tuple[str, Union[str, Path]]:
    """
    Determine whether the user supplied a URL or local file.

    Returns:
        ("url", URL string)
        ("file", Path object)
    """

    raw = (text or "").strip()

    if not raw:
        raise ValueError("Nothing entered.")

    if _looks_like_url(raw):
        return "url", raw

    candidate = Path(
        raw.strip('"').strip("'")
    ).expanduser()

    if candidate.is_file():
        return "file", candidate

    raise ValueError(
        f"'{raw}' is neither a link nor an existing file. "
        "For a file, provide its full path."
    )


# ============================================================
# 5. ARCHIVE OLD OUTPUTS
# ============================================================

def _archive_stale_outputs(
    video_id: str,
    metadata_dir: Path = METADATA_DIR,
) -> list:
    """
    Archive old SNR and transcript outputs.

    Stage 5 may recreate and renumber audio clips.

    Therefore, old transcripts and SNR measurements must
    not accidentally be attached to newly generated audio.

    Files are archived rather than permanently deleted.
    """

    archived = []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for suffix in (
        "_snr.json",
        "_transcripts.json",
    ):

        path = Path(metadata_dir) / f"{video_id}{suffix}"

        if path.exists():

            target = path.with_name(
                f"{path.stem}.stale_{stamp}.json"
            )

            path.rename(target)

            archived.append(target.name)

    if archived:

        print(
            "[pipeline] Previous outputs archived: "
            + ", ".join(archived)
        )

    return archived


# ============================================================
# 6. INITIALIZE TRANSCRIPT RECORDS
# ============================================================

def _initialize_transcripts(
    video_id: str,
    segments: list,
    metadata_dir: Path = METADATA_DIR,
) -> Path:
    """
    Create a fresh transcript JSON file from the new segments.

    export_segments() creates:
        <video_id>_segments.json

    transcriber.py expects:
        <video_id>_transcripts.json

    Therefore, we copy the segment records into a new
    transcript JSON file before running either ASR engine.

    The records already contain segment information such as:
        segment_id
        audio_path
        start_sec
        end_sec
        duration_sec
        region_label
        boundary information

    The ASR engines subsequently add their own fields:
        asr_a_text
        asr_b_text
        and associated metadata.
    """

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transcripts_path = (
        metadata_dir / f"{video_id}_transcripts.json"
    )

    print(
        "[pipeline] Creating transcript metadata from "
        f"{len(segments)} exported segments..."
    )

    with transcripts_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            segments,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Verify that initialization actually succeeded.
    if not transcripts_path.is_file():

        raise FileNotFoundError(
            "Failed to initialize transcript metadata: "
            f"{transcripts_path}"
        )

    print(
        f"[pipeline] Transcript metadata initialized: "
        f"{transcripts_path}"
    )

    print(
        f"[pipeline] Initialized records: {len(segments)}"
    )

    return transcripts_path


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def run_pipeline(
    url: str,
    notes: str = None,
    max_segments: int = None,
    run_snr: bool = True,
    run_transcription: bool = False,
) -> dict:
    """
    Execute the complete Arabic speech corpus pipeline.

    Parameters
    ----------
    url:
        YouTube URL or local audio/video file path.

    notes:
        Metadata notes or permissions information.

    max_segments:
        Optional limit on the number of clips exported.

    run_snr:
        Run Stage 6 if True.

    run_transcription:
        Run Stages 7, 8 and 9 if True.

    Returns
    -------
    Dictionary containing generated pipeline assets.
    """

    # --------------------------------------------------------
    # Resolve input source
    # --------------------------------------------------------

    kind, source = resolve_source(url)

    # ========================================================
    # STAGE 1 & 2: INGESTION AND EXTRACTION
    # ========================================================

    _banner(
        ">>> STAGE 1 & 2: INGESTION, EXTRACTION, AND VALIDATION"
        + (" (LOCAL FILE)" if kind == "file" else " (LINK)")
    )

    if kind == "url":

        metadata = ingest_video(
            url=source,
            notes=notes,
        )

    else:

        metadata = ingest_local_file(
            source,
            notes=notes,
        )

    video_id = metadata["video_id"]

    processed_rel_path = (
        metadata["audio_properties"]["processed_file"]
    )

    wav_path = PROJECT_ROOT / processed_rel_path

    total_duration = (
        metadata["audio_properties"]["duration_seconds"]
    )

    # ========================================================
    # STAGE 3: AUDIO VISUALIZATION
    # ========================================================

    _banner(">>> STAGE 3: AUDIO VISUALIZATION")

    slice_start = (
        60.0 if total_duration > 70.0 else 0.0
    )

    slice_duration = min(10.0, total_duration)

    spec_plot = plot_waveform_and_spectrogram(
        wav_path=wav_path,
        start_sec=slice_start,
        duration_sec=slice_duration,
        title_prefix=f"Preview: {metadata['title'][:25]}",
    )

    env_plot = plot_full_audio_envelope(
        wav_path=wav_path,
    )

    # ========================================================
    # STAGE 4: REGION CLASSIFICATION
    # ========================================================

    _banner(
        ">>> STAGE 4: REGION CLASSIFICATION "
        "(Speech / Silence / Music / Mixed)"
    )

    regions_file = (
        wav_path.parent / f"{video_id}_regions.json"
    )

    regions = classify_audio(
        wav_path=wav_path,
        output_path=regions_file,
    )

    # ========================================================
    # STAGE 5: SPEECH SEGMENTATION
    # ========================================================

    _banner(
        ">>> STAGE 5: SPEECH SEGMENTATION (5-15s CLIPS)"
    )

    segments = export_segments(
        wav_path=wav_path,
        regions=regions,
        video_id=video_id,
        max_segments=max_segments,
    )

    # Validate segmentation output.
    if segments is None:

        raise RuntimeError(
            "export_segments() returned None. "
            "Check the return statement in segmenter.py."
        )

    if not isinstance(segments, list):

        raise TypeError(
            "export_segments() must return a list of segment "
            f"records, but returned {type(segments).__name__}."
        )

    # The clips may have been rewritten and renumbered.
    # Old transcript and SNR results are no longer reliable.

    archived = _archive_stale_outputs(video_id)

    # ========================================================
    # STAGE 6: SIGNAL-TO-NOISE RATIO
    # ========================================================

    snr_records = None

    if run_snr:

        _banner(
            ">>> STAGE 6: SIGNAL-TO-NOISE RATIO "
            "(WADA + FRAME ENERGY)"
        )

        snr_records = compute_snr_for_video(
            video_id
        )

        if snr_records:

            save_snr_records(
                video_id,
                snr_records,
            )

            wada = sorted(
                r["snr_wada_db"]
                for r in snr_records
                if r["snr_wada_db"] is not None
            )

            if wada:

                median_wada = wada[len(wada) // 2]

                print(
                    f"[pipeline] WADA SNR over {len(wada)} clips: "
                    f"min {wada[0]:.1f} / "
                    f"median {median_wada:.1f} / "
                    f"max {wada[-1]:.1f} dB"
                )

        else:

            print("[pipeline] No segments to measure.")

    # ========================================================
    # STAGES 7, 8 & 9: ASR, NORMALIZATION AND EVALUATION
    # ========================================================

    transcripts_path = None
    triage_summary_path = None

    if run_transcription:

        if not segments:

            print(
                "[pipeline] No segments available "
                "for transcription."
            )

        else:

            # ------------------------------------------------
            # PREPARE TRANSCRIPT RECORDS
            # ------------------------------------------------

            # IMPORTANT:
            # This MUST happen before run_engines().
            #
            # The transcriber loads the existing transcript
            # JSON file. It does not initialize it itself.
            #
            # We create it from the newly exported segments.

            transcripts_path = _initialize_transcripts(
                video_id=video_id,
                segments=segments,
            )

            # Confirm the file exists before loading ASR models.
            if not transcripts_path.exists():

                raise FileNotFoundError(
                    "Transcript JSON is missing before ASR: "
                    f"{transcripts_path}"
                )

            # ------------------------------------------------
            # STAGE 7: BOTH ASR ENGINES
            # ------------------------------------------------

            _banner(
                ">>> STAGE 7: ASR TRANSCRIPTION "
                "(WHISPER + COHERE)"
            )

            print(
                "[pipeline] Starting both transcription engines."
            )

            print(
                "[pipeline] Engine A: faster-whisper large-v3-turbo"
            )

            print(
                "[pipeline] Engine B: Cohere Transcribe Arabic"
            )

            print(
                f"[pipeline] Transcript file: {transcripts_path}"
            )

            # run_engines() handles the complete model lifecycle:
            #
            # 1. Load Whisper.
            # 2. Transcribe all segments.
            # 3. Unload Whisper and release GPU memory.
            # 4. Load Cohere.
            # 5. Transcribe all segments.
            # 6. Unload Cohere and release GPU memory.
            # TEMPORARY CUDA DIAGNOSTICS
            import ctranslate2
            import os

            print("\n[CUDA DIAGNOSTICS]")
            print("CUDA devices:", ctranslate2.get_cuda_device_count())
            print("DLL handles:", len(transcriber._DLL_HANDLES))
            print("PATH contains cuBLAS:", "cublas" in os.environ.get("PATH", "").lower())
            print("[END CUDA DIAGNOSTICS]\n")

            transcriber.run_engines(
                video_ids=[video_id],
                engines="both",
                force=False,
            )

            # Confirm the transcript file still exists.
            if not transcripts_path.is_file():

                raise FileNotFoundError(
                    "Transcription did not produce the expected "
                    f"file: {transcripts_path}"
                )

            print(
                "[pipeline] Both ASR engines completed."
            )

            # ------------------------------------------------
            # STAGE 8: NORMALIZATION
            # ------------------------------------------------

            _banner(
                ">>> STAGE 8: ARABIC TRANSCRIPT NORMALIZATION"
            )

            print(
                "[pipeline] Normalizing Engine A and Engine B transcripts."
            )

            normalizer.normalize_video(
                video_id
            )

            print(
                "[pipeline] Transcript normalization completed."
            )

            # ------------------------------------------------
            # STAGE 9: EVALUATION AND TRIAGE
            # ------------------------------------------------

            _banner(
                ">>> STAGE 9: ASR EVALUATION AND HUMAN-REVIEW TRIAGE"
            )

            print(
                "[pipeline] Calculating WER, CER and triage labels."
            )

            evaluator.evaluate_video(
                video_id
            )

            triage_summary_path = (
                METADATA_DIR
                / f"{video_id}_triage_summary.json"
            )

            print(
                "[pipeline] Evaluation and triage completed."
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    _banner(
        "PIPELINE EXECUTION COMPLETE — SUMMARY OF ASSETS"
    )

    print(
        f"  • Source:          "
        f"{'local file' if kind == 'file' else 'link'}"
    )

    print(
        f"  • Video ID:        {video_id}"
    )

    print(
        f"  • Title:           {metadata['title']}"
    )

    print(
        f"  • Duration:        "
        f"{total_duration:.1f}s "
        f"({total_duration / 60:.1f} min)"
    )

    print(
        f"  • Processed WAV:   {processed_rel_path}"
    )

    print(
        f"  • Metadata JSON:   metadata/{video_id}.json"
    )

    print(
        f"  • Classified JSON: "
        f"data/processed/{video_id}_regions.json "
        f"({len(regions)} regions)"
    )

    print(
        f"  • Audio Segments:  "
        f"{len(segments)} clips in data/segments/"
    )

    if snr_records is not None:

        print(
            f"  • SNR JSON:        "
            f"metadata/{video_id}_snr.json "
            f"({len(snr_records)} clips)"
        )

    if transcripts_path is not None:

        print(
            f"  • Transcripts:     "
            f"metadata/{video_id}_transcripts.json"
        )

        print(
            "  • Normalization:   Completed for both engines"
        )

    if triage_summary_path is not None:

        print(
            f"  • Evaluation:      "
            f"metadata/{video_id}_triage_summary.json"
        )

    print(
        f"  • Inspection Plot: outputs/plots/{spec_plot.name}"
    )

    print(
        f"  • Envelope Plot:   outputs/plots/{env_plot.name}"
    )

    print("=" * 65)

    # ========================================================
    # RETURN PIPELINE RESULTS
    # ========================================================

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
        "triage_summary_path": triage_summary_path,
    }


# ============================================================
# 8. INTERACTIVE ENTRY POINT
# ============================================================

def run_interactive_pipeline(
    max_segments: int = None,
    run_snr: bool = True,
):
    """
    Run the pipeline through interactive terminal prompts.

    CLI options such as --max-segments and --no-snr
    are passed into this function by main().
    """

    print("=" * 65)

    print(
        "   ARABIC SPEECH CORPUS — MASTER PROCESSING PIPELINE"
    )

    print("=" * 65)

    entered = input(
        "\n[?] Enter a YouTube URL, or the path "
        "of a local audio/video file: "
    ).strip()

    if not entered:

        print("[!] Nothing entered. Exiting.")

        return

    try:

        kind, _ = resolve_source(entered)

    except ValueError as exc:

        print(f"[!] {exc}")

        return

    notes = input(
        "[?] Enter notes/permissions [press Enter for default]: "
    ).strip()

    if not notes:

        notes = (
            DEFAULT_NOTES_FILE
            if kind == "file"
            else DEFAULT_NOTES_URL
        )

    answer = input(
        "[?] Run both ASR engines, normalization "
        "and evaluation after segmentation? [Y/n]: "
    ).strip().lower()

    run_pipeline(
        url=entered,
        notes=notes,
        max_segments=max_segments,
        run_snr=run_snr,
        run_transcription=answer in ("", "y", "yes"),
    )


# ============================================================
# 9. COMMAND-LINE ENTRY POINT
# ============================================================

def main(
    argv: Optional[Sequence[str]] = None,
) -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Arabic speech corpus pipeline "
            "on a video link or local audio/video file."
        )
    )

    parser.add_argument(
        "source",
        nargs="?",
        help=(
            "Video URL or path of a local file. "
            "Omit to use interactive mode."
        ),
    )

    parser.add_argument(
        "--notes",
        help="Notes/permissions to store in metadata.",
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        help="Export at most N clips (useful for testing).",
    )

    parser.add_argument(
        "--no-snr",
        action="store_true",
        help="Skip Stage 6.",
    )

    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help=(
            "Skip both ASR engines, normalization "
            "and evaluation."
        ),
    )

    args = parser.parse_args(argv)

    # --------------------------------------------------------
    # Interactive mode
    # --------------------------------------------------------

    if args.source is None:

        run_interactive_pipeline(
            max_segments=args.max_segments,
            run_snr=not args.no_snr,
        )

        return 0

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    try:

        kind, _ = resolve_source(args.source)

    except ValueError as exc:

        print(f"[!] {exc}")

        return 1

    # --------------------------------------------------------
    # Execute pipeline
    # --------------------------------------------------------

    run_pipeline(
        url=args.source,

        notes=args.notes or (
            DEFAULT_NOTES_FILE
            if kind == "file"
            else DEFAULT_NOTES_URL
        ),

        max_segments=args.max_segments,

        run_snr=not args.no_snr,

        run_transcription=not args.no_transcribe,
    )

    return 0


# ============================================================
# 10. SCRIPT EXECUTION
# ============================================================

if __name__ == "__main__":

    sys.exit(main())
