"""
Tests for src/local_ingest.py (needs FFmpeg on PATH, like the real pipeline).

    python tests/test_local_ingest.py        (or: pytest tests/test_local_ingest.py)

Everything runs in a temporary project folder: your real data/ is never touched.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import local_ingest as li  # noqa: E402


def _sandbox(tmp: Path):
    """Point the module's folders at a throw-away project tree."""
    tmp = tmp.resolve()
    li.RAW_DIR = tmp / "data" / "raw"
    li.PROCESSED_DIR = tmp / "data" / "processed"
    li.METADATA_DIR = tmp / "metadata"
    li.PROJECT_ROOT = tmp


def _make_audio(path: Path, seconds=3.0, sr=44100, channels=2):
    """Write a real stereo 44.1 kHz file in the format implied by the extension."""
    t = np.arange(int(seconds * sr)) / sr
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = np.stack([tone] * channels, axis=1)
    wav = path.with_suffix(".src.wav")
    sf.write(str(wav), data, sr)
    if path.suffix.lower() == ".wav":
        wav.replace(path)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), str(path)], check=True)
        wav.unlink()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ingest_arabic_emoji_filename_mp3():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _sandbox(tmp)
        src = tmp / "uploads" / "⚽ محاضرة 1 👏.mp3"
        src.parent.mkdir(); _make_audio(src)
        before = _sha(src)

        rec = li.ingest_local_file(src, notes="unit test")

        assert rec["video_id"].startswith("local_") and len(rec["video_id"]) == len("local_") + 12
        assert rec["source_type"] == "local_upload" and rec["source_url"] is None
        assert rec["original_filename"] == src.name and rec["notes"] == "unit test"
        ap = rec["audio_properties"]
        assert (ap["sample_rate"], ap["channels"], ap["subtype"]) == (16000, 1, "PCM_16")
        assert 2.5 < rec["duration_seconds"] < 3.5
        assert (tmp / ap["processed_file"]).exists()
        assert (tmp / rec["ingestion_info"]["raw_file"]).exists()
        assert _sha(src) == before                                   # original untouched
        names = [p.name for p in li.RAW_DIR.iterdir()] + [p.name for p in li.PROCESSED_DIR.iterdir()]
        assert all(n.isascii() or "محاضرة" in n for n in names) and not any("⚽" in n or "👏" in n for n in names)
        saved = json.loads((li.METADATA_DIR / f"{rec['video_id']}.json").read_text(encoding="utf-8"))
        assert saved["original_filename"] == src.name                # Arabic survives in the JSON


def test_same_content_same_id_and_no_duplicates():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _sandbox(tmp)
        a = tmp / "up" / "a.flac"; a.parent.mkdir(); _make_audio(a)
        b = tmp / "up" / "renamed copy.flac"; shutil.copy2(a, b)
        r1 = li.ingest_local_file(a)
        r2 = li.ingest_local_file(b)
        assert r1["video_id"] == r2["video_id"]
        assert len(list(li.RAW_DIR.iterdir())) == 1                  # reused, not copied again
        assert len(list(li.PROCESSED_DIR.glob("*.wav"))) == 1


def test_wav_m4a_and_ogg_inputs():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _sandbox(tmp)
        for name in ("x.wav", "y.m4a", "z.ogg"):
            f = tmp / "up" / name; f.parent.mkdir(exist_ok=True); _make_audio(f, seconds=2.0 + len(name) * 0.1)
            rec = li.ingest_local_file(f)
            assert rec["audio_properties"]["sample_rate"] == 16000, name


def test_bad_inputs_are_rejected():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _sandbox(tmp)
        for bad, exc in [(tmp / "nope.mp3", FileNotFoundError), (tmp, ValueError)]:
            try:
                li.ingest_local_file(bad); raise AssertionError(f"expected {exc.__name__}")
            except exc:
                pass
        txt = tmp / "notes.txt"; txt.write_text("hello" * 500)
        empty = tmp / "empty.mp3"; empty.write_bytes(b"")
        for f in (txt, empty):
            try:
                li.ingest_local_file(f); raise AssertionError("expected ValueError")
            except ValueError:
                pass


def test_corrupt_audio_leaves_no_junk():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _sandbox(tmp)
        fake = tmp / "up" / "fake.mp3"; fake.parent.mkdir()
        fake.write_bytes(b"this is not audio at all " * 200)
        try:
            li.ingest_local_file(fake); raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
        assert not list(li.RAW_DIR.iterdir()) and not list(li.METADATA_DIR.iterdir())
        assert not list(li.PROCESSED_DIR.glob("*.wav"))


def test_project_root_spelled_two_ways():
    """Regression: Windows temp folders have a long and a short (ABDALR~1) spelling."""
    with tempfile.TemporaryDirectory() as t:
        real = Path(t).resolve()
        alias = real.parent / (real.name + "_alias")
        try:
            os.symlink(real, alias, target_is_directory=True)
        except (OSError, NotImplementedError):
            print("  (cannot create a symlink here - skipped)")
            return
        try:
            li.RAW_DIR, li.PROCESSED_DIR = alias / "data" / "raw", alias / "data" / "processed"
            li.METADATA_DIR, li.PROJECT_ROOT = alias / "metadata", alias
            f = real / "up" / "s.mp3"; f.parent.mkdir(); _make_audio(f)
            rec = li.ingest_local_file(f)                       # used to raise ValueError
            assert rec["audio_properties"]["processed_file"].startswith("data/processed/")
        finally:
            alias.unlink()


def test_quoted_path_from_a_text_box():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); _sandbox(tmp)
        f = tmp / "up" / "q.mp3"; f.parent.mkdir(); _make_audio(f)
        rec = li.ingest_local_file(f'"{f}"')                           # Windows "Copy as path" adds quotes
        assert rec["video_id"].startswith("local_")


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
