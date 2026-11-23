"""
Tests for Stage 10's data layer (src/review_store.py): the SQLite audit log and the queue.

    python tests/test_review_store.py        (or: pytest tests/test_review_store.py)
"""
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import review_store as rs  # noqa: E402


def _db(tmp: Path):
    return closing(rs.connect(tmp / "review.db"))


def _clips():
    def c(n, label, wer):
        return {"segment_id": f"v_seg_{n:06d}", "video_id": "v", "triage_label": label, "wer": wer,
                "asr_a_text": f"نص {n}", "asr_b_text": f"نص ب {n}"}
    return [c(1, "accepted", 0.05), c(2, "difficult", 0.60), c(3, "review_needed", 0.20),
            c(4, "difficult", 0.95), c(5, "review_needed", None), c(6, "review_needed", 0.35)]


def test_log_validation():
    with tempfile.TemporaryDirectory() as t, _db(Path(t)) as conn:
        for kwargs in [
            dict(action="accept", reviewer="ali", final_text=""),            # accepted clip needs text
            dict(action="correct", reviewer="ali", final_text="   "),
            dict(action="accept", reviewer="  ", final_text="نص"),           # reviewer required
            dict(action="delete", reviewer="ali", final_text="نص"),          # unknown action
        ]:
            try:
                rs.log_decision(conn, "s1", "v", **kwargs); raise AssertionError(kwargs)
            except ValueError:
                pass
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0     # nothing half-written
        rs.log_decision(conn, "s1", "v", "reject", "ali", final_text="should be dropped")
        row = conn.execute("SELECT * FROM decisions").fetchone()
        assert row["final_text"] is None and row["action"] == "reject"
        rs.log_decision(conn, "s2", "v", "accept", "ali", final_text="  كلمة   اخرى \n")
        assert conn.execute("SELECT final_text FROM decisions WHERE segment_id='s2'").fetchone()[0] == "كلمة اخرى"


def test_log_is_append_only_and_latest_wins():
    with tempfile.TemporaryDirectory() as t, _db(Path(t)) as conn:
        rs.log_decision(conn, "s1", "v", "accept", "ali", "نص اول", "نص اول", "difficult")
        rs.log_decision(conn, "s1", "v", "correct", "sara", "نص مصحح", "نص اول", "difficult")
        assert rs.status_of(rs.latest_decisions(conn)["s1"]) == "corrected"
        rs.log_decision(conn, "s1", "v", "reopen", "ali")
        assert rs.status_of(rs.latest_decisions(conn).get("s1")) == "pending"
        past = rs.history(conn, "s1")
        assert [r["action"] for r in past] == ["accept", "correct", "reopen"]              # nothing overwritten
        assert [r["reviewer"] for r in past] == ["ali", "sara", "ali"]
        assert past[1]["source_text"] == "نص اول" and past[1]["final_text"] == "نص مصحح"
        assert all(r["decided_at"].endswith("+00:00") for r in past)                       # UTC time stamps
        rs.log_decision(conn, "other", "w", "accept", "ali", "x")
        assert set(rs.latest_decisions(conn, "v")) == {"s1"}                               # per-video view


def test_progress_counts():
    clips = _clips()
    with tempfile.TemporaryDirectory() as t, _db(Path(t)) as conn:
        rs.log_decision(conn, clips[0]["segment_id"], "v", "accept", "a", "x")
        rs.log_decision(conn, clips[1]["segment_id"], "v", "correct", "a", "y")
        rs.log_decision(conn, clips[2]["segment_id"], "v", "reject", "a")
        counts = rs.progress(clips, rs.latest_decisions(conn))
    assert counts == {"pending": 3, "accepted": 1, "corrected": 1, "rejected": 1, "total": 6}


def test_queue_priority_order_and_filters():
    clips = _clips()
    ids = lambda q: [c["segment_id"][-1] for c in q]
    q = rs.build_queue(clips, {}, labels=None, statuses=None)
    assert ids(q) == ["4", "2", "6", "3", "5", "1"]          # difficult (0.95, 0.60), review (0.35, 0.20, none), accepted
    assert ids(rs.build_queue(clips, {}, labels=["difficult"], statuses=None)) == ["4", "2"]
    assert ids(rs.build_queue(clips, {}, labels=["accepted"], statuses=None)) == ["1"]
    assert ids(rs.build_queue(clips, {}, order="recording", labels=None, statuses=None)) == list("123456")
    no_label = [{"segment_id": "v_seg_000009", "video_id": "v"}]
    assert rs.build_queue(no_label, {}, labels=["review_needed"], statuses=None)                # unlabeled = review_needed
    done = {clips[3]["segment_id"]: {"action": "accept"}}
    assert ids(rs.build_queue(clips, done, labels=None, statuses=("pending",)))[0] == "2"       # decided clip leaves the queue
    assert ids(rs.build_queue(clips, done, labels=None, statuses=("accepted",))) == ["4"]
    try:
        rs.build_queue(clips, {}, order="sideways"); raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_random_order_is_reproducible():
    clips = [{"segment_id": f"v_seg_{n:06d}", "video_id": "v", "triage_label": "accepted"} for n in range(1, 31)]
    a = rs.build_queue(clips, {}, labels=None, statuses=None, order="random", seed=1)
    b = rs.build_queue(clips, {}, labels=None, statuses=None, order="random", seed=1)
    c = rs.build_queue(clips, {}, labels=None, statuses=None, order="random", seed=2)
    assert a == b and a != c and sorted(x["segment_id"] for x in a) == sorted(x["segment_id"] for x in clips)


def test_starting_text():
    clip = {"asr_a_text": "من المحرك أ", "asr_b_text": "من المحرك ب"}
    assert rs.starting_text(clip, None) == "من المحرك أ"
    assert rs.starting_text(clip, {"final_text": "نص المراجع"}) == "نص المراجع"
    assert rs.starting_text(clip, {"final_text": None}) == "من المحرك أ"             # after a reject/reopen
    assert rs.starting_text({"asr_b_text": "ب فقط"}, None) == "ب فقط"
    assert rs.starting_text({}, None) == ""


def test_load_clips_joins_files_and_lists_videos():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); meta = root / "metadata"; meta.mkdir()
        (meta / "v_transcripts.json").write_text(json.dumps([
            {"segment_id": "v_seg_000002", "audio_path": "data/segments/v_seg_000002.wav", "asr_a_text": "ب", "wer": 0.1},
            {"segment_id": "v_seg_000001", "audio_path": "data/segments/v_seg_000001.wav", "asr_a_text": "أ"}],
            ensure_ascii=False), encoding="utf-8")
        (meta / "v_segments.json").write_text(json.dumps(
            [{"segment_id": "v_seg_000001", "start_sec": 1.5, "end_sec": 11.5, "region_label": "speech",
              "boundary_check": "auto_verified", "pause_ms": 480}]), encoding="utf-8")
        (meta / "v_snr.json").write_text(json.dumps([{"segment_id": "v_seg_000002", "snr_wada_db": 33.0}]), encoding="utf-8")
        (meta / "w_transcripts.json").write_text("[]", encoding="utf-8")
        (meta / "w_transcripts.stale_20261121_1.json").write_text("[]", encoding="utf-8")   # archived: not a video
        clips = rs.load_clips("v", meta, root)
        assert [c["segment_id"] for c in clips] == ["v_seg_000001", "v_seg_000002"]
        assert clips[0]["region_label"] == "speech" and clips[0]["pause_ms"] == 480
        assert clips[1]["snr_wada_db"] == 33.0 and clips[1]["region_label"] is None
        assert clips[0]["audio_abspath"] == str(root / "data/segments/v_seg_000001.wav")
        assert rs.list_videos(meta) == ["v", "w"]


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
