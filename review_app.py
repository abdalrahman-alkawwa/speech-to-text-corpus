
"""
review_app.py - Stage 10: CSV-based human review.

Run from the repository root:

    streamlit run review_app.py

The per-video CSV is the single source of truth for review annotations.
"""

import html
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import review_store as rs  # noqa: E402

META = PROJECT_ROOT / "metadata"

BADGE = {
    "difficult": "🔴",
    "review_needed": "🟠",
    "accepted": "🟢",
}

ORDER_NAMES = {
    "priority": "Hardest first",
    "recording": "Recording order",
    "random": "Random (spot-check)",
}

LABELS = ["speech", "music", "mixed"]
QUALITIES = ["", "clean", "noisy"]

st.set_page_config(page_title="Corpus review", layout="wide")

st.markdown(
    "<style>textarea{direction:rtl;text-align:right;font-size:1.2rem;}"
    ".rtl{direction:rtl;text-align:right;font-size:1.25rem;line-height:1.9;"
    "padding:0.4rem 0.6rem;border:1px solid rgba(128,128,128,.35);border-radius:6px;}"
    "</style>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Navigation and CSV updates
# --------------------------------------------------------------------------- #
def _reset_position():
    st.session_state["pos"] = 0


def _move(step):
    st.session_state["pos"] += step


def _use_b(clip):
    """Copy Model B's transcript into the editable text box."""
    st.session_state[f"text_{clip['segment_id']}"] = (
        clip.get("asr_b_text") or ""
    )


def _record(clip, action, text=None, source_text=None):
    """
    Save a review action directly to the CSV.

    action:
        accept -> accepted
        correct -> corrected
        reject -> rejected
        reopen -> pending
    """
    action_to_status = {
        "accept": "accepted",
        "correct": "corrected",
        "reject": "rejected",
        "reopen": "pending",
    }

    if action not in action_to_status:
        st.session_state["flash"] = ("error", f"Unknown action: {action}")
        return

    sid = clip["segment_id"]
    label = st.session_state.get(f"label_{sid}", clip.get("label", "speech"))
    quality = st.session_state.get(
        f"quality_{sid}", clip.get("quality", "")
    )

    try:
        rs.update_review(
            video_id=clip["video_id"],
            segment_id=sid,
            status=action_to_status[action],
            final_transcript=text,
            label=label,
            quality=quality,
            changed_by=st.session_state.get("reviewer", "").strip(),
            metadata_dir=META,
        )
    except (ValueError, KeyError, OSError) as exc:
        st.session_state["flash"] = ("error", str(exc))
        return

    st.session_state["flash"] = (
        "success",
        f"{action_to_status[action]}: {sid}",
    )

    # In the "all" view, the current clip stays in the queue.
    # Move forward after a decision so the reviewer sees the next clip.
    if st.session_state.get("status") == "all" and action != "reopen":
        st.session_state["pos"] += 1


def _save(clip, source_text):
    """Decide whether the reviewer accepted or corrected Model A."""
    text = st.session_state.get(f"text_{clip['segment_id']}", "")

    if not rs.clean_text(text):
        st.session_state["flash"] = (
            "error",
            "The final transcript cannot be empty. Use Reject if the clip is unusable.",
        )
        return

    action = (
        "accept"
        if rs.clean_text(text) == rs.clean_text(source_text)
        else "correct"
    )

    _record(clip, action, text=text, source_text=source_text)


# --------------------------------------------------------------------------- #
# Sidebar: recording and review filters
# --------------------------------------------------------------------------- #
st.session_state.setdefault("pos", 0)

videos = rs.list_videos(META)

if not videos:
    st.warning(
        f"No *_transcripts.json files found in {META}. "
        "Run the pipeline first."
    )
    st.stop()

with st.sidebar:
    st.header("Review settings")

    st.text_input("Your name (saved with changes)", key="reviewer")

    st.selectbox(
        "Recording",
        videos,
        key="video",
        on_change=_reset_position,
    )

    st.multiselect(
        "Triage labels to show",
        ["difficult", "human_choice", "review_needed", "accepted"],
        default=["difficult", "human_choice", "review_needed"],
        key="labels",
        on_change=_reset_position,
    )

    st.radio(
        "Show",
        ["pending", "all"],
        key="status",
        on_change=_reset_position,
        format_func=lambda value: (
            "Only clips not yet reviewed"
            if value == "pending"
            else "All clips"
        ),
    )

    st.radio(
        "Order",
        list(ORDER_NAMES),
        key="order",
        format_func=ORDER_NAMES.get,
    )

    with st.expander("Annotation guide"):
        st.markdown(
            "- Write **only what is spoken**, spelled the way you hear it "
            "(dialect spelling is fine).\n"
            "- English words: write them in **Arabic letters** "
            "(for example بودكاست).\n"
            "- Non-speech sounds go in square brackets: **[ضحك]** laughter, "
            "**[موسيقى]** music, **[ضوضاء]** noise.\n"
            "- Punctuation is optional and ignored when scoring; numbers "
            "are scored as words.\n"
            "- **Reject** a clip if the speech cannot be made out or the "
            "cut is through a word."
        )


# --------------------------------------------------------------------------- #
# Load CSV records and build the review queue
# --------------------------------------------------------------------------- #
video_id = st.session_state["video"]

try:
    clips = rs.load_clips(video_id, META, PROJECT_ROOT)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load recording data: {exc}")
    st.stop()

decisions = rs.latest_decisions(clips)
counts = rs.progress(clips, decisions)

queue = rs.build_queue(
    clips,
    decisions,
    labels=st.session_state["labels"] or None,
    statuses=("pending",) if st.session_state["status"] == "pending" else None,
    order=st.session_state["order"],
    seed=1,
)

with st.sidebar:
    reviewed = counts["total"] - counts["pending"]

    st.progress(
        reviewed / counts["total"] if counts["total"] else 0.0
    )

    st.caption(
        f"{reviewed} of {counts['total']} clips reviewed\n\n"
        f"accepted {counts['accepted']} · "
        f"corrected {counts['corrected']} · "
        f"rejected {counts['rejected']} · "
        f"pending {counts['pending']}"
    )


# --------------------------------------------------------------------------- #
# Main panel: one clip
# --------------------------------------------------------------------------- #
st.title(f"Review · {video_id}")

flash = st.session_state.pop("flash", None)

if flash:
    (st.success if flash[0] == "success" else st.error)(flash[1])

if not queue:
    st.success(
        "Nothing left in this view. Change the filters in the sidebar "
        "to see other clips."
    )
    st.stop()

pos = min(max(st.session_state["pos"], 0), len(queue) - 1)
st.session_state["pos"] = pos

clip = queue[pos]
sid = clip["segment_id"]

decision = decisions.get(sid)
status = rs.status_of(decision)

triage_label = clip.get("triage_label") or "review_needed"
source_text = clip.get("asr_a_text") or ""

st.subheader(f"Clip {pos + 1} of {len(queue)} · {sid}")


# --------------------------------------------------------------------------- #
# Audio player
# --------------------------------------------------------------------------- #
audio_path = (
    Path(clip["audio_abspath"])
    if clip.get("audio_abspath")
    else None
)

if audio_path and audio_path.exists():
    st.audio(audio_path.read_bytes(), format="audio/wav")
else:
    st.error(f"Audio file not found: {clip.get('audio_path')}")


# --------------------------------------------------------------------------- #
# Segment information and model transcripts
# --------------------------------------------------------------------------- #
info, texts = st.columns([1, 2])

with info:
    st.markdown(
        f"**{BADGE.get(triage_label, '')} {triage_label}**"
    )

    if clip.get("triage_reasons"):
        st.caption(
            "Why: " + ", ".join(clip["triage_reasons"])
        )

    metrics = []

    if clip.get("start_sec") is not None and clip.get("end_sec") is not None:
        metrics.append(
            f"Time: {clip['start_sec']:.2f}–{clip['end_sec']:.2f} s"
        )

    if clip.get("wer") is not None:
        metrics.append(
            f"WER {clip['wer']:.2f} · CER {clip.get('cer', 0):.2f}"
        )

    if clip.get("boundary_check"):
        metrics.append(f"Cut: {clip['boundary_check']}")

    if clip.get("pause_ms") is not None:
        metrics.append(f"Pause at cut: {clip['pause_ms']} ms")

    if clip.get("snr_wada_db") is not None:
        metrics.append(
            f"SNR (WADA): {clip['snr_wada_db']:.2f} dB"
        )

    if clip.get("snr_segmental_db") is not None:
        metrics.append(
            f"SNR (segmental): {clip['snr_segmental_db']} dB"
        )

    if clip.get("duration_sec") is not None:
        metrics.append(f"Length: {clip['duration_sec']:.2f} s")

    for line in metrics:
        st.caption(line)

    st.caption(f"Current status: **{status}**")

with texts:
    st.markdown("**Model A**")

    st.markdown(
        f"<div class='rtl'>{html.escape(clip.get('asr_a_text') or '—')}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Model B** · {clip.get('asr_b_engine') or 'not run'}"
    )

    st.markdown(
        f"<div class='rtl'>{html.escape(clip.get('asr_b_text') or '—')}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Human annotation fields
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("Human annotation")

label_key = f"label_{sid}"
quality_key = f"quality_{sid}"
text_key = f"text_{sid}"

# Initialize each widget from its current CSV value.
if label_key not in st.session_state:
    current_label = clip.get("label") or "speech"
    st.session_state[label_key] = (
        current_label if current_label in LABELS else "speech"
    )

if quality_key not in st.session_state:
    current_quality = clip.get("quality") or ""
    st.session_state[quality_key] = (
        current_quality if current_quality in QUALITIES else ""
    )

if text_key not in st.session_state:
    st.session_state[text_key] = rs.starting_text(clip, decision)

c1, c2 = st.columns(2)

with c1:
    st.selectbox(
        "Audio label",
        LABELS,
        key=label_key,
    )

with c2:
    st.selectbox(
        "Audio quality",
        QUALITIES,
        key=quality_key,
        format_func=lambda value: value if value else "Not classified",
    )

st.text_area(
    "Final transcript",
    key=text_key,
    height=150,
)


# --------------------------------------------------------------------------- #
# Review actions
# --------------------------------------------------------------------------- #
b1, b2, b3, b4, b5, b6 = st.columns(6)

b1.button(
    "Save & next",
    key="save",
    type="primary",
    on_click=_save,
    args=(clip, source_text),
)

b2.button(
    "Reject clip",
    key="reject",
    on_click=_record,
    args=(clip, "reject"),
)

b3.button(
    "Use B text",
    key="use_b",
    on_click=_use_b,
    args=(clip,),
    disabled=not clip.get("asr_b_text"),
)

b4.button(
    "Skip",
    key="skip",
    on_click=_move,
    args=(1,),
    disabled=pos >= len(queue) - 1,
)

b5.button(
    "Back",
    key="prev",
    on_click=_move,
    args=(-1,),
    disabled=pos == 0,
)

b6.button(
    "Reopen",
    key="reopen",
    on_click=_record,
    args=(clip, "reopen"),
    disabled=status == "pending",
)
