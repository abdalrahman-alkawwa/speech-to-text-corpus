"""
src/export_corpus.py
====================
Stage 11: Corpus Export & Dataset Packaging

Generates standardized dataset manifests for ASR model training/evaluation:
  - HuggingFace-compatible JSON Lines manifest (corpus_manifest.jsonl)
  - Tabular CSV manifest (corpus_manifest.csv)
  - Train / Validation / Test stratified splits (80 / 10 / 10)
  - 10-minute gold-standard evaluation split with verified audio & ground truth
"""

import sys
import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transcript_store import get_all_segments, get_corpus_statistics

MANIFEST_DIR = PROJECT_ROOT / "metadata"


def export_corpus_manifests(
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Exports all segments into train/val/test splits and saves JSONL + CSV manifests.
    """
    out_dir = Path(output_dir or MANIFEST_DIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = get_all_segments()
    if not segments:
        print("[export] No segments found in database to export!")
        return {}

    # Deterministic split shuffling
    random.seed(seed)
    shuffled = list(segments)
    random.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    # Assign split attribute
    for s in train_set:
        s["split"] = "train"
    for s in val_set:
        s["split"] = "validation"
    for s in test_set:
        s["split"] = "test"

    all_split_segments = train_set + val_set + test_set

    # 1. Export JSON Lines (HuggingFace format)
    jsonl_path = out_dir / "corpus_manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in all_split_segments:
            record = {
                "audio": s["audio_path"],
                "sentence": s["ground_truth"] or s["transcript_a"] or "",
                "segment_id": s["segment_id"],
                "video_id": s["video_id"],
                "duration": s["duration_sec"],
                "snr_db": s["snr_wada_db"],
                "wer": s["wer"],
                "triage": s["triage"],
                "status": s["status"],
                "split": s["split"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Export CSV Manifest
    csv_path = out_dir / "corpus_manifest.csv"
    fieldnames = [
        "segment_id", "video_id", "audio_path", "split", "duration_sec",
        "snr_wada_db", "ground_truth", "transcript_a", "transcript_b",
        "wer", "cer", "triage", "status", "boundary_check"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for s in all_split_segments:
            writer.writerow(s)

    # 3. Dedicated 10-minute gold evaluation sample (up to 600 seconds)
    gold_target_sec = 600.0
    accumulated_sec = 0.0
    gold_records = []

    # Prioritize 'accepted' and 'verified' segments for the gold sample
    candidate_gold = sorted(
        all_split_segments,
        key=lambda x: (0 if x["status"] == "verified" else 1, x["wer"] if x["wer"] is not None else 1.0)
    )

    for s in candidate_gold:
        dur = s.get("duration_sec", 0.0) or 0.0
        if accumulated_sec + dur <= gold_target_sec:
            gold_records.append(s)
            accumulated_sec += dur

    gold_path = out_dir / "gold_evaluation_sample.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for g in gold_records:
            item = {
                "audio": g["audio_path"],
                "sentence": g["ground_truth"] or g["transcript_a"] or "",
                "segment_id": g["segment_id"],
                "video_id": g["video_id"],
                "duration": g["duration_sec"],
                "snr_db": g["snr_wada_db"],
                "boundary_check": g["boundary_check"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[export] Corpus Manifest Export Complete:")
    print(f"  • JSONL Manifest: {jsonl_path.name} ({len(all_split_segments)} segments)")
    print(f"  • CSV Manifest:   {csv_path.name}")
    print(f"  • Splits:         Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"  • Gold Sample:    {gold_path.name} ({len(gold_records)} segments, {accumulated_sec:.1f}s / {accumulated_sec/60:.1f} min)")

    return {
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "gold_path": str(gold_path),
        "total_segments": total,
        "splits": {"train": len(train_set), "val": len(val_set), "test": len(test_set)},
        "gold_sample_duration_sec": accumulated_sec,
    }


if __name__ == "__main__":
    export_corpus_manifests()
