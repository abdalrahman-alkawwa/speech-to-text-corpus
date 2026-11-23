"""
Tests for Stage 7 engine B (MMS) in src/transcriber.py, using a FAKE engine so they
run without torch, a GPU or the 4 GB download.

    python tests/test_transcriber_b.py        (or: pytest tests/test_transcriber_b.py)

The real model is only exercised by the smoke test:
    python src/transcriber.py --engine b --video-id <id> --limit 5
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import transcriber as tr  # noqa: E402

SR = 16_000
VOCAB = {1: "م", 2: "ر", 3: "ح", 4: "ب", 5: "ا"}


class FakeMms:
    """Same interface as MmsEngine: logits(), decode(), blank_id, language, close()."""

    blank_id = 0
    language = "ara"

    def __init__(self, frame_ids, top=0.9, nan=False):
        self.frame_ids, self.top, self.nan, self.calls = frame_ids, top, nan, 0

    def logits(self, audio):
        self.calls += 1
        n, v = len(self.frame_ids), 6
        out = np.full((n, v), np.log((1 - self.top) / (v - 1)), dtype=np.float32)
        for i, k in enumerate(self.frame_ids):
            out[i, k] = np.log(self.top)
        if self.nan:
            out[0, 0] = np.nan
        return out

    def decode(self, ids):                      # CTC rule: collapse repeats, drop blanks
        text, prev = [], None
        for k in ids:
            if k != prev and k != self.blank_id:
                text.append(VOCAB[int(k)])
            prev = k
        return "".join(text)

    def close(self):
        pass


def _make_segments(folder: Path, video_id: str, n: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        sf.write(str(folder / f"{video_id}_seg_{i:06d}.wav"),
                 np.zeros(SR * 8, dtype=np.float32) + 0.01, SR, subtype="PCM_16")


def test_ctc_confidence_ignores_blank_frames():
    logits = np.log(np.array([
        [0.05, 0.90, 0.05],     # letter 1, top prob 0.90
        [0.60, 0.20, 0.20],     # blank (id 0) - must be ignored
        [0.10, 0.10, 0.80],     # letter 2, top prob 0.80
    ], dtype=np.float32))
    assert abs(tr.ctc_confidence(logits, blank_id=0) - 0.85) < 1e-5
    only_blank = np.log(np.array([[0.9, 0.05, 0.05]] * 4, dtype=np.float32))
    assert tr.ctc_confidence(only_blank, blank_id=0) is None


def test_transcribe_mms_collapses_repeats_and_blanks():
    with tempfile.TemporaryDirectory() as t:
        _make_segments(Path(t), "v", 1)
        out = tr.transcribe_mms(FakeMms([1, 1, 0, 2, 2, 0, 0, 3, 3, 3]), Path(t) / "v_seg_000001.wav")
    assert out["asr_b_text"] == "مرح"                       # م م _ ر ر _ _ ح ح ح
    assert out["asr_b_engine"] == tr.MMS_ENGINE_TAG and out["asr_b_language"] == "ara"
    assert abs(out["asr_b_confidence"] - 0.9) < 1e-3


def test_nan_scores_raise_a_helpful_error():
    with tempfile.TemporaryDirectory() as t:
        _make_segments(Path(t), "v", 1)
        try:
            tr.transcribe_mms(FakeMms([1, 2, 3], nan=True), Path(t) / "v_seg_000001.wav")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert "--mms-fp32" in str(exc)


def test_engine_b_merges_into_engine_a_records_and_resumes():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        seg_dir, meta = tmp / "seg", tmp / "meta"
        _make_segments(seg_dir, "vid", 3)
        path = tr.transcripts_path("vid", meta)
        tr.save_records(path, {                              # as if engine A had already run
            f"vid_seg_{i:06d}": {"segment_id": f"vid_seg_{i:06d}", "asr_a_text": f"نص {i}", "asr_a_avg_logprob": -0.1}
            for i in (1, 2, 3)})
        eng = FakeMms([1, 0, 2, 3])

        out = tr.transcribe_video_b("vid", eng, seg_dir, meta, limit=2)
        recs = json.loads(out.read_text(encoding="utf-8"))
        assert [bool(r.get("asr_b_text")) for r in recs] == [True, True, False]
        assert recs[0]["asr_a_text"] == "نص 1" and recs[0]["asr_a_avg_logprob"] == -0.1   # A untouched
        assert recs[0]["duration_sec"] == 8.0 and eng.calls == 2

        tr.transcribe_video_b("vid", eng, seg_dir, meta)       # resume: only seg 3 is new
        assert eng.calls == 3
        tr.transcribe_video_b("vid", eng, seg_dir, meta, force=True)
        assert eng.calls == 6
        raw = path.read_text(encoding="utf-8")
        assert "\\u0645" not in raw                            # Arabic stored as real characters


def test_engine_a_rerun_keeps_engine_b_fields():
    class FakeWhisper:
        def transcribe(self, path, **kw):
            chunk = SimpleNamespace(start=0.0, end=8.0, text="نص أ", avg_logprob=-0.2, no_speech_prob=0.0, compression_ratio=1.0)
            return iter([chunk]), SimpleNamespace(language="ar", language_probability=1.0)

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        seg_dir, meta = tmp / "seg", tmp / "meta"
        _make_segments(seg_dir, "vid", 1)
        tr.transcribe_video_b("vid", FakeMms([1, 2]), seg_dir, meta)
        tr.transcribe_video("vid", FakeWhisper(), seg_dir, meta)
        rec = json.loads(tr.transcripts_path("vid", meta).read_text(encoding="utf-8"))[0]
        assert rec["asr_a_text"] == "نص أ" and rec["asr_b_text"] == "مر"


def test_crash_midway_still_saves_finished_work():
    class Boom(FakeMms):
        def logits(self, audio):
            if self.calls == 2:
                raise RuntimeError("CUDA out of memory")
            return super().logits(audio)

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        seg_dir, meta = tmp / "seg", tmp / "meta"
        _make_segments(seg_dir, "vid", 4)
        try:
            tr.transcribe_video_b("vid", Boom([1, 2]), seg_dir, meta)
        except RuntimeError:
            pass
        assert len(json.loads(tr.transcripts_path("vid", meta).read_text(encoding="utf-8"))) == 2


def test_run_engines_order_and_gpu_release():
    """With 'both', engine A must be unloaded BEFORE engine B is loaded."""
    events = []
    orig = (tr.load_whisper, tr.unload_model, tr.release_gpu, tr.load_mms,
            tr.transcribe_video, tr.transcribe_video_b)
    tr.load_whisper = lambda *a, **k: events.append("load A") or object()
    tr.load_mms = lambda *a, **k: events.append("load B") or FakeMms([1])
    tr.unload_model = lambda m: events.append("unload")
    tr.release_gpu = lambda: events.append("release")
    tr.transcribe_video = lambda *a, **k: events.append("run A")
    tr.transcribe_video_b = lambda *a, **k: events.append("run B")
    try:
        tr.run_engines(["v1", "v2"], engines="both")
        assert events == ["load A", "run A", "run A", "unload", "release",
                          "load B", "run B", "run B", "unload", "release"], events
        events.clear(); tr.run_engines(["v1"], engines="b")
        assert events == ["load B", "run B", "unload", "release"], events
        events.clear(); tr.run_engines(["v1"], engines="a")
        assert events == ["load A", "run A", "unload", "release"], events
        try:
            tr.run_engines(["v1"], engines="c"); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        (tr.load_whisper, tr.unload_model, tr.release_gpu, tr.load_mms,
         tr.transcribe_video, tr.transcribe_video_b) = orig


class FakeWhisperB:
    """Stands in for a faster-whisper model used as engine B."""

    def transcribe(self, path, **kwargs):
        chunk = SimpleNamespace(start=0.0, end=8.0, text=" نص من النموذج الثاني ", avg_logprob=-0.2231436,
                                no_speech_prob=0.01, compression_ratio=1.3)
        return iter([chunk]), SimpleNamespace(language="ar", language_probability=0.99)


def test_whisper_as_engine_b_uses_asr_b_names():
    with tempfile.TemporaryDirectory() as t:
        _make_segments(Path(t), "v", 1)
        out = tr.transcribe_whisper_as_b(FakeWhisperB(), Path(t) / "v_seg_000001.wav", "faster-whisper:test-model")
    assert out["asr_b_text"] == "نص من النموذج الثاني"
    assert out["asr_b_engine"] == "faster-whisper:test-model"
    assert abs(out["asr_b_confidence"] - 0.80) < 1e-3                 # exp(-0.2231) = 0.8
    assert out["asr_b_no_speech_prob"] == 0.01 and out["asr_b_language"] == "ar"
    assert not any(k.startswith("asr_a_") for k in out)                # nothing leaks into engine A's names


def test_whisper_backend_merges_into_engine_a_records():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        seg_dir, meta = tmp / "seg", tmp / "meta"
        _make_segments(seg_dir, "vid", 2)
        path = tr.transcripts_path("vid", meta)
        tr.save_records(path, {f"vid_seg_00000{i}": {"segment_id": f"vid_seg_00000{i}", "asr_a_text": f"نص {i}"} for i in (1, 2)})
        tr.transcribe_video_b("vid", FakeWhisperB(), seg_dir, meta,
                              transcribe_fn=lambda m, w: tr.transcribe_whisper_as_b(m, w, "tag"))
        recs = json.loads(path.read_text(encoding="utf-8"))
        assert [r["asr_a_text"] for r in recs] == ["نص 1", "نص 2"]                 # engine A untouched
        assert all(r["asr_b_text"] == "نص من النموذج الثاني" and r["asr_b_engine"] == "tag" for r in recs)


def test_run_engines_with_whisper_backend():
    calls, events = [], []
    orig = (tr.load_whisper, tr.unload_model, tr.release_gpu, tr.transcribe_video, tr.transcribe_video_b)
    tr.load_whisper = lambda name, device, compute, root: calls.append((name, compute)) or events.append("load " + name) or object()
    tr.unload_model = lambda m: events.append("unload")
    tr.release_gpu = lambda: events.append("release")
    tr.transcribe_video = lambda *a, **k: events.append("run A")
    tr.transcribe_video_b = lambda *a, **k: events.append("run B")
    try:
        tr.run_engines(["v1"], engines="both", b_backend="whisper", b_model="org/second-model")
        assert events == ["load large-v3-turbo", "run A", "unload", "release",
                          "load org/second-model", "run B", "unload", "release"], events
        assert calls[1] == ("org/second-model", tr.DEFAULT_B_COMPUTE_TYPE)
        events.clear(); calls.clear()
        tr.run_engines(["v1"], engines="b", b_backend="whisper")                       # default second model
        assert calls == [(tr.DEFAULT_B_WHISPER_MODEL, tr.DEFAULT_B_COMPUTE_TYPE)]
        try:
            tr.run_engines(["v1"], engines="b", b_backend="nonsense"); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        (tr.load_whisper, tr.unload_model, tr.release_gpu, tr.transcribe_video, tr.transcribe_video_b) = orig


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