"""
Tests for the small helpers in src/pipeline.py (link-vs-file detection and the
set-aside of stale results).  No audio, GPU or network needed.

    python tests/test_pipeline_helpers.py        (or: pytest tests/test_pipeline_helpers.py)
"""
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import pipeline  # noqa: E402


def test_urls_are_recognised():
    for text in ("https://www.youtube.com/watch?v=CvrNaPFIwcQ&t=1s", "http://x.org/a.mp3",
                 "youtu.be/CvrNaPFIwcQ", "www.youtube.com/watch?v=abc", "  https://youtu.be/abc  "):
        assert pipeline.resolve_source(text)[0] == "url", text


def test_existing_file_is_recognised_even_with_quotes():
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "محاضرة 1.mp3"
        f.write_bytes(b"x" * 2000)
        for text in (str(f), f'"{f}"', f"'{f}'"):
            kind, value = pipeline.resolve_source(text)
            assert kind == "file" and Path(value) == f, text


def test_missing_file_and_empty_input_are_rejected():
    for text in ("", "   ", "C:\\no\\such\\file.mp3", "just some words"):
        try:
            pipeline.resolve_source(text)
            raise AssertionError(f"expected ValueError for {text!r}")
        except ValueError:
            pass


def test_stale_results_are_renamed_not_deleted():
    with tempfile.TemporaryDirectory() as t:
        meta = Path(t)
        (meta / "v1_snr.json").write_text("[1]", encoding="utf-8")
        (meta / "v1_transcripts.json").write_text(json.dumps([{"asr_a_text": "نص"}], ensure_ascii=False), encoding="utf-8")
        (meta / "v1.json").write_text("{}", encoding="utf-8")                 # provenance record: untouched
        (meta / "other_transcripts.json").write_text("[]", encoding="utf-8")  # another video: untouched
        archived = pipeline._archive_stale_outputs("v1", metadata_dir=meta)
        names = {p.name for p in meta.iterdir()}
        assert len(archived) == 2 and not (meta / "v1_snr.json").exists()
        assert "v1.json" in names and "other_transcripts.json" in names
        stale = [p for p in meta.iterdir() if ".stale_" in p.name and "transcripts" in p.name][0]
        assert "نص" in stale.read_text(encoding="utf-8")                      # content preserved
        assert pipeline._archive_stale_outputs("v1", metadata_dir=meta) == []  # nothing left to set aside


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1; print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
