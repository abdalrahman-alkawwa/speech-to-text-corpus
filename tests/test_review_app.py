"""
Drives the real Streamlit screen (review_app.py) headlessly with Streamlit's AppTest and checks
what ends up in the SQLite audit log.  No browser needed.

    python tests/test_review_app.py        (or: pytest tests/test_review_app.py)
"""
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SR = 16_000


def _make_project(tmp: Path) -> Path:
    """A throw-away copy of the project: the app, the data layer, 4 clips with transcripts."""
    (tmp / "src").mkdir()
    (tmp / "data" / "segments").mkdir(parents=True)
    (tmp / "metadata").mkdir()
    shutil.copy(PROJECT_ROOT / "review_app.py", tmp / "review_app.py")
    shutil.copy(PROJECT_ROOT / "src" / "review_store.py", tmp / "src" / "review_store.py")
    (tmp / "src" / "__init__.py").write_text("", encoding="utf-8")

    rows = [  # (n, label, wer)
        (1, "accepted", 0.02), (2, "difficult", 0.70), (3, "review_needed", 0.25), (4, "difficult", 0.90)]
    records = []
    for n, label, wer in rows:
        sid = f"vid_seg_{n:06d}"
        sf.write(str(tmp / "data" / "segments" / f"{sid}.wav"), np.zeros(SR * 2, dtype=np.float32) + 0.01, SR, subtype="PCM_16")
        records.append({"segment_id": sid, "audio_path": f"data/segments/{sid}.wav", "duration_sec": 2.0,
                        "asr_a_text": f"نص المحرك أ {n}", "asr_b_text": f"نص المحرك ب {n}", "asr_b_engine": "test-b",
                        "wer": wer, "cer": wer / 2, "triage_label": label, "triage_reasons": ["wer_test"]})
    (tmp / "metadata" / "vid_transcripts.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return tmp


def _rows(tmp: Path):
    conn = sqlite3.connect(str(tmp / "metadata" / "review.db"))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM decisions ORDER BY id")]
    finally:
        conn.close()


def _open(tmp: Path) -> AppTest:
    at = AppTest.from_file(str(tmp / "review_app.py"), default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _current_clip(at: AppTest) -> str:
    return at.subheader[0].value.split("·")[1].strip()


def test_full_review_session():
    with tempfile.TemporaryDirectory() as t:
        tmp = _make_project(Path(t))
        at = _open(tmp)

        # default view: pending difficult + review_needed clips, hardest first
        assert _current_clip(at) == "vid_seg_000004"
        assert at.text_area(key="text_vid_seg_000004").value == "نص المحرك أ 4"          # starts from engine A

        # saving without a name is refused and nothing is logged
        at.button(key="save").click().run()
        assert any("name" in e.value for e in at.error) and _rows(tmp) == []

        # accept unchanged -> logged "accept"; the queue moves on to the next hardest clip
        at.sidebar.text_input(key="reviewer").set_value("sara").run()
        at.button(key="save").click().run()
        rows = _rows(tmp)
        assert [(r["segment_id"], r["action"], r["reviewer"]) for r in rows] == [("vid_seg_000004", "accept", "sara")]
        assert rows[0]["final_text"] == "نص المحرك أ 4" and rows[0]["triage_label"] == "difficult"
        assert _current_clip(at) == "vid_seg_000002"

        # edit the text -> logged "correct", with the machine text kept as source_text
        at.text_area(key="text_vid_seg_000002").set_value("نص صحيح بعد التصحيح").run()
        at.button(key="save").click().run()
        row = _rows(tmp)[-1]
        assert (row["action"], row["final_text"], row["source_text"]) == ("correct", "نص صحيح بعد التصحيح", "نص المحرك أ 2")
        assert _current_clip(at) == "vid_seg_000003"

        # "Use B text" copies engine B's transcript into the box
        at.button(key="use_b").click().run()
        assert at.text_area(key="text_vid_seg_000003").value == "نص المحرك ب 3"

        # reject -> logged with no text; queue is now empty
        at.button(key="reject").click().run()
        row = _rows(tmp)[-1]
        assert (row["segment_id"], row["action"], row["final_text"]) == ("vid_seg_000003", "reject", None)
        assert any("Nothing left" in s.value for s in at.success)

        # show everything, reopen a decided clip: it becomes pending again and a new row is appended
        at.sidebar.radio(key="status").set_value("all").run()
        at.sidebar.multiselect(key="labels").set_value(["difficult", "review_needed", "accepted"]).run()
        assert _current_clip(at) == "vid_seg_000004"
        at.button(key="reopen").click().run()
        assert _rows(tmp)[-1]["action"] == "reopen" and len(_rows(tmp)) == 4              # history only grows

        # the audit trail is intact and in order
        assert [r["action"] for r in _rows(tmp)] == ["accept", "correct", "reject", "reopen"]
        assert all(r["decided_at"].endswith("+00:00") for r in _rows(tmp))


def test_skip_and_back_do_not_log_anything():
    with tempfile.TemporaryDirectory() as t:
        tmp = _make_project(Path(t))
        at = _open(tmp)
        first = _current_clip(at)
        at.button(key="skip").click().run()
        assert _current_clip(at) != first
        at.button(key="prev").click().run()
        assert _current_clip(at) == first
        assert _rows(tmp) == []


def test_empty_text_cannot_be_saved():
    with tempfile.TemporaryDirectory() as t:
        tmp = _make_project(Path(t))
        at = _open(tmp)
        at.sidebar.text_input(key="reviewer").set_value("sara").run()
        sid = _current_clip(at)
        at.text_area(key=f"text_{sid}").set_value("   ").run()
        at.button(key="save").click().run()
        assert any("needs text" in e.value for e in at.error) and _rows(tmp) == []


def test_no_transcripts_shows_a_clear_message():
    with tempfile.TemporaryDirectory() as t:
        tmp = _make_project(Path(t))
        (tmp / "metadata" / "vid_transcripts.json").unlink()
        at = AppTest.from_file(str(tmp / "review_app.py"), default_timeout=30).run()
        assert not at.exception and any("No *_transcripts.json" in w.value for w in at.warning)


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
