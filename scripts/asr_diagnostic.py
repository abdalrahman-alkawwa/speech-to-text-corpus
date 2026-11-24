"""Compare saved and optional fresh ASR outputs on selected corpus segments.

This diagnostic writes a standalone JSON report. It does not update transcript
metadata, annotation status, or final transcripts.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import evaluator, transcriber


DEFAULT_VIDEO_ID = "Izdf8guwjFs"
DEFAULT_SEGMENTS = {
    "agreement": "113",
    "moderate_disagreement": "061",
    "generation_cap_suspect": "260",
    "code_switch_low_snr": "006",
    "low_snr": "011",
    "user_audited": "004",
}


def segment_number(segment_id):
    return str(segment_id).rsplit("_", 1)[-1].lstrip("0") or "0"


def audio_details(path):
    info = sf.info(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": round(info.duration, 4),
        "subtype": info.subtype,
    }


def summarize_pair(text_a, text_b):
    a = evaluator.get_normalized_text({"asr_a_text": text_a or ""}, "asr_a_text")
    b = evaluator.get_normalized_text({"asr_b_text": text_b or ""}, "asr_b_text")
    edits = evaluator.count_word_edits(a, b)
    comparison_words = max(len(a.split()), len(b.split()))
    wer, cer = evaluator.compute_wer_cer(a, b)
    return {
        "normalized_a": a,
        "normalized_b": b,
        "word_count_a": len(a.split()),
        "word_count_b": len(b.split()),
        "word_edits": edits,
        "relative_word_edit_rate": (
            edits["total"] / comparison_words if comparison_words else None
        ),
        "wer_a_reference": wer,
        "cer_a_reference": cer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument(
        "--segment", action="append", dest="segments",
        help="Segment number (e.g. 260); repeat to select several. Defaults to a diagnostic set.",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Run both ASR engines on only the selected clips and save outputs separately.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "asr_diagnostics",
    )
    args = parser.parse_args()

    selected = set(args.segments or DEFAULT_SEGMENTS.values())
    metadata_path = transcriber.transcripts_path(args.video_id)
    records = transcriber.load_records(metadata_path)
    selected_records = [
        r for r in records if segment_number(r.get("segment_id", "")) in selected
    ]
    found = {segment_number(r.get("segment_id", "")) for r in selected_records}
    missing = sorted(selected - found)
    if missing:
        parser.error(f"Segments not found for {args.video_id}: {', '.join(missing)}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_id": args.video_id,
        "rerun": args.rerun,
        "cohere_max_new_tokens": transcriber.COHERE_MAX_NEW_TOKENS,
        "segments": [],
    }
    for record in selected_records:
        path = transcriber.find_segment_audio(args.video_id, record)
        entry = {
            "segment_id": record["segment_id"],
            "audio": audio_details(path),
            "saved": {
                "asr_a_text": record.get("asr_a_text", ""),
                "asr_b_text": record.get("asr_b_text", ""),
            },
        }
        entry["saved"]["comparison"] = summarize_pair(
            entry["saved"]["asr_a_text"], entry["saved"]["asr_b_text"]
        )
        if "asr_b_generated_tokens" in record:
            entry["saved"]["asr_b_generated_tokens"] = record["asr_b_generated_tokens"]
        if "asr_b_hit_token_limit" in record:
            entry["saved"]["asr_b_hit_token_limit"] = record["asr_b_hit_token_limit"]
        entry["saved"]["asr_b_quality_flags"] = record.get("asr_b_quality_flags", [])
        report["segments"].append(entry)

    if args.rerun:
        engine_a = transcriber.load_whisper()
        try:
            for entry, record in zip(report["segments"], selected_records):
                entry["rerun"] = {"asr_a": transcriber.transcribe_whisper(
                    engine_a, transcriber.find_segment_audio(args.video_id, record)
                )}
        finally:
            del engine_a
            transcriber.unload_model()

        processor, engine_b = transcriber.load_cohere()
        try:
            for entry, record in zip(report["segments"], selected_records):
                result = transcriber.transcribe_cohere(
                    processor, engine_b,
                    transcriber.find_segment_audio(args.video_id, record),
                )
                entry["rerun"]["asr_b"] = result
                entry["rerun"]["comparison"] = summarize_pair(
                    entry["rerun"]["asr_a"]["asr_a_text"], result["asr_b_text"]
                )
        finally:
            del processor, engine_b
            transcriber.unload_model()

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.video_id}_selected_segments.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(selected_records)} selected segment diagnostics to {output_path}")
    if args.rerun:
        print("Fresh ASR inference completed for selected segments only; corpus metadata was not changed.")


if __name__ == "__main__":
    main()
