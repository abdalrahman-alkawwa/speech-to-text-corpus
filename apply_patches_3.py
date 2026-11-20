"""
apply_patches_3.py - adds to src/transcriber.py:
  1. --language option ("ar" by default, "auto" = let Whisper detect per clip)
  2. a one-line timing summary at the end of every run

    python apply_patches_3.py --dry-run
    python apply_patches_3.py          # backup saved as src/transcriber.py.bak3

Every edit must match exactly one place, otherwise the file is left untouched.
Your CUDA DLL registration code is not touched.
"""
import shutil
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent / "src" / "transcriber.py"
MARKER = "language: Optional[str] = WHISPER_LANGUAGE"

EDITS = [
    ("transcribe_whisper signature",
     "def transcribe_whisper(\n    model: Any,\n    wav_path: Path\n) -> Dict[str, Any]:",
     "def transcribe_whisper(\n    model: Any,\n    wav_path: Path,\n    language: Optional[str] = WHISPER_LANGUAGE\n) -> Dict[str, Any]:"),
    ("pass language to faster-whisper",
     "        language=WHISPER_LANGUAGE,\n        beam_size=WHISPER_BEAM_SIZE,",
     "        language=language,   # None = auto-detect per clip\n        beam_size=WHISPER_BEAM_SIZE,"),
    ("transcribe_video signature",
     "    force: bool = False,\n    limit: Optional[int] = None,\n) -> Path:",
     "    force: bool = False,\n    limit: Optional[int] = None,\n    language: Optional[str] = WHISPER_LANGUAGE,\n) -> Path:"),
    ("pass language in the loop",
     "            rec.update(\n                transcribe_whisper(model, wav)\n            )",
     "            rec.update(\n                transcribe_whisper(model, wav, language=language)\n            )"),
    ("end-of-run timing summary",
     '    print(f"[{video_id}] saved -> {out_path}")',
     "    if done:\n"
     "        elapsed = time.time() - start\n"
     "        print(\n"
     '            f"[{video_id}] transcribed {done} segments ({audio_sec:.0f}s of audio) "\n'
     '            f"in {elapsed:.1f}s -> processing time = "\n'
     '            f"{elapsed / max(audio_sec, 1e-6):.2f}x audio duration"\n'
     "        )\n\n"
     '    print(f"[{video_id}] saved -> {out_path}")'),
    ("--language command-line option",
     '    parser.add_argument("--model", default=WHISPER_MODEL)',
     "    parser.add_argument(\n"
     '        "--language",\n'
     "        default=WHISPER_LANGUAGE,\n"
     '        help=\'Whisper language code (default "ar"); "auto" detects per clip.\'\n'
     "    )\n\n"
     '    parser.add_argument("--model", default=WHISPER_MODEL)'),
    ("use --language in main()",
     "                force=args.force,\n                limit=args.limit\n            )",
     "                force=args.force,\n                limit=args.limit,\n"
     '                language=None if args.language == "auto" else args.language\n            )'),
]


def main() -> int:
    dry = "--dry-run" in sys.argv
    if not PATH.exists():
        print(f"[skip] {PATH} not found"); return 1
    text = PATH.read_text(encoding="utf-8")
    if MARKER in text:
        print("[ok]   transcriber.py: already patched"); return 0
    bad = [d for d, old, _ in EDITS if text.count(old) != 1]
    if bad:
        print(f"[FAIL] transcriber.py NOT modified; no unique match for: {bad}"); return 1
    for _, old, new in EDITS:
        text = text.replace(old, new)
    if dry:
        print(f"[ok]   transcriber.py: {len(EDITS)} edits would apply"); return 0
    backup = PATH.with_suffix(PATH.suffix + ".bak3")
    if not backup.exists():
        shutil.copy2(PATH, backup)
    PATH.write_text(text, encoding="utf-8")
    print(f"[done] transcriber.py: {len(EDITS)} edits applied (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
