"""Arabic Speech Corpus Studio: pipeline, segment review, and CSV export."""

import io
import html
import json
import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from src import review_store as rs
from src.pipeline import run_pipeline
from src.snr_estimator import estimate_wada_snr

META = PROJECT_ROOT / "metadata"
LABELS = ["speech", "music", "mixed"]
QUALITIES = ["", "clean", "noisy"]
TRIAGE_LABELS = ["difficult", "human_choice", "review_needed", "accepted"]

st.set_page_config(
    page_title="Arabic Speech Corpus Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem;}
    .rtl {direction:rtl;text-align:right;font-size:1.2rem;line-height:1.8;
          padding:.55rem .7rem;border:1px solid rgba(128,128,128,.35);
          border-radius:8px;white-space:pre-wrap;}
    textarea {direction:rtl;text-align:right;font-size:1.15rem;}
    </style>""",
    unsafe_allow_html=True,
)


def _videos():
    return rs.list_videos(META)


def _reset_review_position():
    st.session_state["review_pos"] = 0


def _audio_bytes(path: Path, start: float = None, end: float = None):
    # Seek into the source WAV so reviewing one short cut never loads a
    # multi-hour recording into memory.
    with sf.SoundFile(str(path), mode="r") as source:
        sr = source.samplerate
        left = max(0, int((start or 0.0) * sr))
        right = source.frames if end is None else min(source.frames, int(end * sr))
        source.seek(left)
        audio = source.read(max(0, right - left), dtype="float32", always_2d=False)
    if not len(audio):
        return None
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _wada_snr(path: Path, start: float, end: float):
    """Estimate WADA SNR for a short source-audio span without writing a WAV."""
    with sf.SoundFile(str(path), mode="r") as source:
        sr = source.samplerate
        left = max(0, int(start * sr))
        right = min(source.frames, int(end * sr))
        source.seek(left)
        samples = source.read(max(0, right - left), dtype="float64", always_2d=False)
    value = estimate_wada_snr(samples)
    return None if value is None else round(float(value), 2)


def _save_reviewed_audio(video_id: str, segment_id: str, source_path: Path, start: float, end: float):
    """Write a corrected segment copy without changing the original WAV."""
    with sf.SoundFile(str(source_path), mode="r") as source:
        sr = source.samplerate
        first = max(0, int(start * sr))
        last = min(source.frames, int(end * sr))
        if last <= first:
            raise ValueError("The corrected segment contains no audio frames.")
        source.seek(first)
        samples = source.read(last - first, dtype="float32", always_2d=False)
    destination = PROJECT_ROOT / "data" / "reviewed_segments" / video_id / f"{segment_id}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), samples, sr, subtype="PCM_16")
    return destination.relative_to(PROJECT_ROOT).as_posix()


def _region_spectrogram(path: Path, regions, start: float, duration: float = 10.0):
    """Draw one short spectrogram with the saved region labels overlaid."""
    with sf.SoundFile(str(path), mode="r") as source:
        sr = source.samplerate
        first = max(0, int(start * sr))
        source.seek(first)
        audio = source.read(int(duration * sr), dtype="float32", always_2d=False)
    if len(audio) < 32:
        return None
    nperseg = min(1024, len(audio))
    frequencies, times, power = spectrogram(audio, fs=sr, nperseg=nperseg)
    db = 10 * np.log10(np.maximum(power, 1e-12))
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.pcolormesh(times + start, frequencies, db, shading="auto", cmap="magma")
    colors = {"speech": "#22c55e", "silence": "#94a3b8", "music": "#a78bfa", "mixed": "#f59e0b"}
    finish = start + len(audio) / sr
    for region in regions:
        left = max(start, float(region.get("start_sec", 0)))
        right = min(finish, float(region.get("end_sec", 0)))
        if left >= right:
            continue
        label = str(region.get("label", "mixed"))
        axis.axvspan(left, right, color=colors.get(label, "#38bdf8"), alpha=0.15)
        axis.text((left + right) / 2, sr / 2, label, color="white", ha="center", va="top", fontsize=8)
    axis.set_ylim(0, min(8000, sr / 2))
    axis.set_xlabel("Time in recording (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title("Spectrogram with classified region spans")
    figure.tight_layout()
    return figure


def _source_wav(video_id: str):
    metadata_path = META / f"{video_id}.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        relative = metadata.get("audio_properties", {}).get("processed_file")
        path = PROJECT_ROOT / relative if relative else None
        return path if path and path.is_file() else None
    except (OSError, ValueError, TypeError):
        return None


def _render_pipeline():
    st.title("Video to labelled speech corpus")
    st.caption(
        "Run ingestion, audio preparation, classification, segmentation, SNR, "
        "both ASR systems, normalization, and disagreement triage."
    )

    with st.form("pipeline_form"):
        source = st.text_input(
            "Video link or local audio/video path",
            placeholder="https://… or C:\\path\\to\\recording.mp4",
            help="Paste a supported video URL or the full path to a local media file.",
        )
        notes = st.text_input(
            "Source / permissions notes",
            placeholder="Document permission, license, dialect, or context",
        )
        language_choice = st.selectbox(
            "Speech language",
            ["Auto-detect Arabic/English", "Arabic (including Arabic-English mix)", "English"],
            help=(
                "Auto mode detects each segment with Whisper, then selects "
                "Arabic or English for Cohere. Mixed Arabic-English speech "
                "can use the Arabic option."
            ),
        )
        transcription_language = {
            "Auto-detect Arabic/English": "auto",
            "Arabic (including Arabic-English mix)": "ar",
            "English": "en",
        }[language_choice]
        choice = st.radio(
            "Transcription scope",
            ["Transcribe every segment", "Transcribe the first N segments"],
            horizontal=True,
        )
        limit = None
        if choice == "Transcribe the first N segments":
            limit = st.number_input(
                "Number of segments to transcribe",
                min_value=1,
                max_value=10000,
                value=10,
                step=1,
            )
        submitted = st.form_submit_button(
            "Run complete pipeline", type="primary", use_container_width=True
        )

    if submitted:
        if not source.strip():
            st.error("Enter a video link or local file path.")
            return
        with st.status("Running the corpus pipeline…", expanded=True) as status:
            try:
                result = run_pipeline(
                    url=source.strip(),
                    notes=notes.strip() or None,
                    max_segments=None,
                    run_snr=True,
                    run_transcription=True,
                    transcription_limit=int(limit) if limit is not None else None,
                    transcription_language=transcription_language,
                    progress_callback=status.write,
                )
                video_id = result["metadata"]["video_id"]
                status.update(label="Pipeline complete", state="complete")
                if result["segments"]:
                    csv_path = rs.ensure_csv(video_id, META)
                    st.session_state["active_video"] = video_id
                    st.success(
                        f"Processed **{result['metadata']['title']}**. "
                        f"Created {len(result['segments'])} segments."
                    )
                    st.info(f"Review CSV ready: `{csv_path.name}`")
                else:
                    st.warning(
                        f"Processed **{result['metadata']['title']}**, but no speech "
                        "segments were exported. Transcription and the review CSV "
                        "were skipped. Check the classified regions and speech "
                        "detection settings, then rerun the pipeline."
                    )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Stage": "Classified regions", "Result": len(result["regions"])},
                            {"Stage": "Exported segments", "Result": len(result["segments"])},
                            {"Stage": "Segments selected for ASR", "Result": min(len(result["segments"]), int(limit)) if limit is not None else len(result["segments"])},
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as exc:  # show pipeline failures in the UI
                status.update(label="Pipeline stopped with an error", state="error")
                st.exception(exc)


def _render_review():
    videos = _videos()
    st.title("Review transcripts and segments")
    st.caption(
        "Listen to the audio, compare both model outputs, edit the final "
        "annotation, and save each decision to the per-video CSV."
    )
    if not videos:
        st.info("No transcript records yet. Run the pipeline first.")
        return

    with st.sidebar:
        st.subheader("Review settings")
        video_id = st.selectbox(
            "Recording", videos,
            index=videos.index(st.session_state["active_video"])
            if st.session_state.get("active_video") in videos else 0,
            key="review_video",
            on_change=_reset_review_position,
        )
        reviewer = st.text_input(
            "Your name (saved with changes)", key="reviewer_name"
        ).strip()
        shown_triage = st.multiselect(
            "Triage groups", TRIAGE_LABELS,
            default=["difficult", "human_choice", "review_needed"],
            key="review_triage",
            on_change=_reset_review_position,
        )
        show_status = st.radio(
            "Review queue", ["Pending only", "All segments"],
            key="review_status",
            on_change=_reset_review_position,
        )
        order = st.selectbox(
            "Order", ["Hardest first", "Recording order", "Random spot-check"],
            key="review_order",
        )

    try:
        clips = rs.load_clips(video_id, META, PROJECT_ROOT)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load this recording: {exc}")
        return

    decisions = rs.latest_decisions(clips)
    counts = rs.progress(clips, decisions)
    order_key = {
        "Hardest first": "priority",
        "Recording order": "recording",
        "Random spot-check": "random",
    }[order]
    queue = rs.build_queue(
        clips,
        decisions,
        labels=shown_triage or None,
        statuses=("pending",) if show_status == "Pending only" else None,
        order=order_key,
        seed=1,
    )

    reviewed = counts["total"] - counts["pending"]
    st.progress(reviewed / counts["total"] if counts["total"] else 0.0)
    st.caption(
        f"{reviewed}/{counts['total']} decided · {counts['accepted']} accepted · "
        f"{counts['corrected']} corrected · {counts['rejected']} rejected · "
        f"{counts['pending']} pending"
    )
    notice = st.session_state.pop("review_notice", None)
    if notice:
        st.success(notice)

    output_csv = rs.csv_path(video_id, META)
    if output_csv.exists():
        st.download_button(
            "Download per-video CSV",
            data=output_csv.read_bytes(),
            file_name=output_csv.name,
            mime="text/csv",
            key=f"download_{video_id}",
        )

    if not queue:
        st.success("No segments match these filters.")
        return

    pos = min(max(st.session_state.get("review_pos", 0), 0), len(queue) - 1)
    st.session_state["review_pos"] = pos
    clip = queue[pos]
    sid = clip["segment_id"]
    st.subheader(f"Segment {pos + 1} of {len(queue)} · {sid}")

    start_key, end_key = f"start_{sid}", f"end_{sid}"
    label_key, quality_key, text_key = f"label_{sid}", f"quality_{sid}", f"final_{sid}"
    if start_key not in st.session_state:
        st.session_state[start_key] = float(clip.get("start_sec") or 0.0)
    if end_key not in st.session_state:
        st.session_state[end_key] = float(clip.get("end_sec") or 0.0)
    if label_key not in st.session_state:
        prior_label = clip.get("label") or clip.get("region_label") or "speech"
        st.session_state[label_key] = prior_label if prior_label in LABELS else "speech"
    if quality_key not in st.session_state:
        prior_quality = clip.get("quality") or ""
        st.session_state[quality_key] = prior_quality if prior_quality in QUALITIES else ""
    if text_key not in st.session_state:
        decision = decisions.get(sid)
        st.session_state[text_key] = rs.starting_text(clip, decision)

    audio_source = _source_wav(video_id)
    if audio_source:
        source_duration = sf.info(str(audio_source)).duration
        bounds = st.columns(2)
        with bounds[0]:
            st.number_input("Start (seconds)", min_value=0.0, max_value=float(source_duration), step=0.05, key=start_key)
        with bounds[1]:
            st.number_input("End (seconds)", min_value=0.0, max_value=float(source_duration), step=0.05, key=end_key)
        if st.session_state[end_key] > st.session_state[start_key]:
            payload = _audio_bytes(audio_source, st.session_state[start_key], st.session_state[end_key])
            if payload:
                st.audio(payload, format="audio/wav")
            context_start = max(0.0, st.session_state[start_key] - 2.0)
            context_end = st.session_state[end_key] + 2.0
            context = _audio_bytes(audio_source, context_start, context_end)
            if context:
                with st.expander("Listen around the segment cuts (2 seconds of context)"):
                    st.audio(context, format="audio/wav")
    else:
        audio_path = Path(clip["audio_abspath"]) if clip.get("audio_abspath") else None
        if audio_path and audio_path.is_file():
            st.audio(audio_path.read_bytes(), format="audio/wav")
        else:
            st.warning("Audio file is unavailable for this segment.")

    with st.expander("Add, split, merge, or delete segments"):
        st.caption(
            "Structural edits and their audit history are stored in the review CSV. "
            "WADA SNR is recalculated when source audio is available; new or "
            "changed segments need fresh ASR transcripts."
        )
        if not reviewer:
            st.info("Enter your name in Review settings to enable structural edits.")
        split_key = f"split_point_{sid}"
        if split_key not in st.session_state:
            st.session_state[split_key] = (
                st.session_state[start_key] + st.session_state[end_key]
            ) / 2
        st.number_input(
            "Split this segment at (seconds)",
            min_value=0.0,
            step=0.05,
            key=split_key,
            disabled=not reviewer,
        )
        split_col, delete_col = st.columns(2)
        if split_col.button("Split segment", key=f"split_{sid}", disabled=not reviewer):
            try:
                split_at = st.session_state[split_key]
                child_snrs = (
                    [_wada_snr(audio_source, st.session_state[start_key], split_at),
                     _wada_snr(audio_source, split_at, st.session_state[end_key])]
                    if audio_source else [None, None]
                )
                child_ids = rs.split_segment(video_id, sid, split_at, reviewer, META, snrs=child_snrs)
                for child_id, (child_start, child_end), child_snr in zip(
                    child_ids,
                    [(st.session_state[start_key], split_at), (split_at, st.session_state[end_key])],
                    child_snrs,
                ):
                    if audio_source:
                        audio_file = _save_reviewed_audio(video_id, child_id, audio_source, child_start, child_end)
                        rs.update_review(
                            video_id, child_id, "pending", changed_by=reviewer,
                            start=child_start, end=child_end, snr=child_snr,
                            audio_file=audio_file, metadata_dir=META,
                        )
                st.session_state["review_notice"] = "Segment split. The original row remains marked deleted for audit."
                st.rerun()
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))
        if delete_col.button("Delete segment", key=f"delete_{sid}", disabled=not reviewer):
            try:
                rs.delete_segment(video_id, sid, reviewer, META)
                st.session_state["review_notice"] = "Segment marked deleted; its row and change history remain in the CSV."
                st.rerun()
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))

        active_ids = [
            c["segment_id"] for c in clips
            if rs.status_of(decisions.get(c["segment_id"])) != "deleted"
        ]
        merge_choices = [candidate for candidate in active_ids if candidate != sid]
        if merge_choices:
            selected_merge = st.multiselect(
                "Choose one other segment to merge with this one",
                merge_choices,
                max_selections=1,
                key=f"merge_with_{sid}",
                disabled=not reviewer,
            )
            if st.button(
                "Merge selected segments",
                key=f"merge_{sid}",
                disabled=not reviewer or not selected_merge,
            ):
                try:
                    other = next(c for c in clips if c["segment_id"] == selected_merge[0])
                    merge_start = min(float(clip["start_sec"] or 0), float(other["start_sec"] or 0))
                    merge_end = max(float(clip["end_sec"] or 0), float(other["end_sec"] or 0))
                    merged_snr = _wada_snr(audio_source, merge_start, merge_end) if audio_source else None
                    merged_id = rs.merge_segments(video_id, [sid, selected_merge[0]], reviewer, META, snr=merged_snr)
                    if audio_source:
                        merged_audio = _save_reviewed_audio(video_id, merged_id, audio_source, merge_start, merge_end)
                        rs.update_review(
                            video_id, merged_id, "pending", changed_by=reviewer,
                            start=merge_start, end=merge_end, snr=merged_snr,
                            audio_file=merged_audio, metadata_dir=META,
                        )
                    st.session_state["review_notice"] = "Segments merged. Source rows remain marked deleted for audit."
                    st.rerun()
                except (ValueError, KeyError, OSError) as exc:
                    st.error(str(exc))

        st.markdown("**Add a segment from the source audio**")
        add_cols = st.columns(2)
        add_start = add_cols[0].number_input(
            "New segment start (seconds)", min_value=0.0,
            max_value=float(source_duration) if audio_source else None, value=0.0,
            step=0.05, key=f"add_start_{video_id}", disabled=not reviewer,
        )
        add_end = add_cols[1].number_input(
            "New segment end (seconds)", min_value=0.0,
            max_value=float(source_duration) if audio_source else None, value=1.0,
            step=0.05, key=f"add_end_{video_id}", disabled=not reviewer,
        )
        add_label = st.selectbox(
            "New segment label", LABELS,
            key=f"add_label_{video_id}", disabled=not reviewer,
        )
        if st.button("Add segment", key=f"add_segment_{video_id}", disabled=not reviewer):
            try:
                new_id = rs.add_segment(
                    video_id, add_start, add_end, reviewer,
                    label=add_label,
                    snr=_wada_snr(audio_source, add_start, add_end) if audio_source else None,
                    metadata_dir=META,
                )
                if audio_source:
                    new_audio = _save_reviewed_audio(video_id, new_id, audio_source, add_start, add_end)
                    rs.update_review(
                        video_id, new_id, "pending", changed_by=reviewer,
                        start=add_start, end=add_end,
                        snr=_wada_snr(audio_source, add_start, add_end),
                        audio_file=new_audio, metadata_dir=META,
                    )
                st.session_state["review_notice"] = f"Added {new_id}; it is pending transcript and SNR review."
                st.rerun()
            except (ValueError, KeyError, OSError) as exc:
                st.error(str(exc))

    triage = clip.get("triage_label") or "review_needed"
    info, transcripts = st.columns([1, 2])
    with info:
        st.markdown(f"**Triage:** {triage}")
        st.markdown(f"**Current review status:** {rs.status_of(decisions.get(sid))}")
        if clip.get("wer") is not None:
            st.caption(f"Model disagreement · WER {clip['wer']:.3f} · CER {clip.get('cer', 0):.3f}")
        if clip.get("snr_wada_db") is not None:
            st.caption(f"WADA SNR: {clip['snr_wada_db']:.2f} dB")
        if clip.get("triage_reasons"):
            st.caption("Review flags: " + ", ".join(clip["triage_reasons"]))
    with transcripts:
        language_a = clip.get("asr_a_language") or "unknown"
        language_b = clip.get("asr_b_language") or "unknown"
        direction_a = "rtl" if language_a == "ar" else "ltr"
        direction_b = "rtl" if language_b == "ar" else "ltr"
        alignment_a = "right" if language_a == "ar" else "left"
        alignment_b = "right" if language_b == "ar" else "left"
        st.markdown(f"**Model A · faster-whisper · {html.escape(language_a)}**")
        st.markdown(
            f"<div style='direction:{direction_a};text-align:{alignment_a};font-size:1.2rem;line-height:1.8;padding:.55rem .7rem;border:1px solid rgba(128,128,128,.35);border-radius:8px;white-space:pre-wrap'>{html.escape(rs.clean_text(clip.get('asr_a_text')) or '—')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Model B · {html.escape(clip.get('asr_b_engine') or 'Cohere')} · {html.escape(language_b)}**")
        st.markdown(
            f"<div style='direction:{direction_b};text-align:{alignment_b};font-size:1.2rem;line-height:1.8;padding:.55rem .7rem;border:1px solid rgba(128,128,128,.35);border-radius:8px;white-space:pre-wrap'>{html.escape(rs.clean_text(clip.get('asr_b_text')) or '—')}</div>",
            unsafe_allow_html=True,
        )
        asr_audio = Path(clip["audio_abspath"]) if clip.get("audio_abspath") else None
        if st.button(
            "Transcribe this segment with both models",
            key=f"transcribe_review_{sid}",
            disabled=not reviewer or not (asr_audio and asr_audio.is_file()),
            help="Runs faster-whisper, then Cohere, and saves each result to the review CSV.",
        ):
            from src import transcriber

            saved_a = False
            try:
                with st.spinner("Running Model A (faster-whisper)…"):
                    model_a = transcriber.load_whisper()
                    try:
                        result_a = transcriber.transcribe_whisper(
                            model_a, asr_audio, language="auto"
                        )
                    finally:
                        del model_a
                        transcriber.unload_model()
                    rs.save_asr_results(
                        video_id, sid, result_a, reviewer, metadata_dir=META
                    )
                    saved_a = True
                    if not st.session_state.get(text_key):
                        st.session_state[text_key] = result_a.get("asr_a_text", "")

                detected_language = str(
                    result_a.get("asr_a_language") or ""
                ).lower().split("-")[0]
                if detected_language not in transcriber.SUPPORTED_COHERE_LANGUAGES:
                    st.session_state["review_notice"] = (
                        f"Model A transcript saved. Model B was skipped because "
                        f"Whisper detected unsupported language "
                        f"'{detected_language or 'unknown'}'."
                    )
                    st.rerun()

                with st.spinner("Running Model B (Cohere)…"):
                    processor_b, model_b = transcriber.load_cohere()
                    try:
                        result_b = transcriber.transcribe_cohere(
                            processor_b,
                            model_b,
                            asr_audio,
                            language=detected_language,
                            language_mode="auto",
                        )
                    finally:
                        del model_b, processor_b
                        transcriber.unload_model()
                    rs.save_asr_results(
                        video_id, sid, result_b, reviewer, metadata_dir=META
                    )

                st.session_state["review_notice"] = (
                    f"Both model transcripts saved for {sid}."
                )
                st.rerun()
            except Exception as exc:
                if saved_a:
                    st.session_state["review_notice"] = (
                        f"Model A transcript saved for {sid}; Model B failed: {exc}"
                    )
                    st.rerun()
                st.error(f"Could not transcribe {sid}: {exc}")

    picks = st.columns(2)
    if picks[0].button("Use Model A", key=f"use_a_{sid}", use_container_width=True):
        st.session_state[text_key] = clip.get("asr_a_text") or ""
        st.rerun()
    if picks[1].button("Use Model B", key=f"use_b_{sid}", disabled=not clip.get("asr_b_text"), use_container_width=True):
        st.session_state[text_key] = clip.get("asr_b_text") or ""
        st.rerun()

    st.markdown("#### Human annotation")
    edit_cols = st.columns(2)
    with edit_cols[0]:
        st.selectbox("Segment label", LABELS, key=label_key)
    with edit_cols[1]:
        st.selectbox("Audio quality", QUALITIES, key=quality_key,
                     format_func=lambda value: value or "Choose clean/noisy")
        st.caption("Initial quality is clean at WADA SNR ≥20 dB; change it when listening shows otherwise.")
    st.text_area("Final transcript", key=text_key, height=150)

    action_cols = st.columns([2, 1, 1, 1, 1])
    save = action_cols[0].button("Save decision & next", type="primary", key=f"save_{sid}", use_container_width=True)
    reject = action_cols[1].button("Mark difficult", key=f"reject_{sid}", use_container_width=True)
    back = action_cols[2].button("Previous", key=f"back_{sid}", disabled=pos == 0, use_container_width=True)
    skip = action_cols[3].button("Skip", key=f"skip_{sid}", disabled=pos >= len(queue) - 1, use_container_width=True)
    reopen = action_cols[4].button("Reopen", key=f"reopen_{sid}", disabled=rs.status_of(decisions.get(sid)) == "pending", use_container_width=True)

    if back:
        st.session_state["review_pos"] = max(0, pos - 1)
        st.rerun()
    if skip:
        st.session_state["review_pos"] = min(len(queue) - 1, pos + 1)
        st.rerun()
    if save or reject or reopen:
        if not reviewer:
            st.error("Enter your name in Review settings before saving changes.")
        else:
            final_text = "" if reject else st.session_state[text_key]
            if not reject and not rs.clean_text(final_text):
                st.error("Final transcript cannot be empty. Use Mark difficult if it is unrecoverable.")
            elif st.session_state[end_key] <= st.session_state[start_key]:
                st.error("Segment end must be later than its start.")
            else:
                if reopen:
                    status = "pending"
                elif reject:
                    status = "rejected"
                elif rs.clean_text(final_text) in {
                    rs.clean_text(clip.get("asr_a_text")),
                    rs.clean_text(clip.get("asr_b_text")),
                }:
                    status = "accepted"
                else:
                    status = "corrected"
                try:
                    bounds_changed = (
                        abs(st.session_state[start_key] - float(clip.get("start_sec") or 0.0)) > 0.001
                        or abs(st.session_state[end_key] - float(clip.get("end_sec") or 0.0)) > 0.001
                    )
                    audio_file = clip.get("audio_path") or ""
                    if audio_source and bounds_changed:
                        audio_file = _save_reviewed_audio(
                            video_id, sid, audio_source,
                            st.session_state[start_key], st.session_state[end_key],
                        )
                    rs.update_review(
                        video_id=video_id,
                        segment_id=sid,
                        status=status,
                        final_transcript=final_text,
                        label=st.session_state[label_key],
                        quality=st.session_state[quality_key],
                        changed_by=reviewer,
                        start=st.session_state[start_key],
                        end=st.session_state[end_key],
                        snr=(
                            _wada_snr(audio_source, st.session_state[start_key], st.session_state[end_key])
                            if audio_source else None
                        ),
                        audio_file=audio_file,
                        metadata_dir=META,
                    )
                    if not (show_status == "All segments" and reopen):
                        st.session_state["review_pos"] = min(pos + 1, len(queue) - 1)
                    st.session_state["review_notice"] = f"Saved {status} for {sid}; change history is in the CSV."
                    st.rerun()
                except (ValueError, KeyError, OSError) as exc:
                    st.error(str(exc))


def _render_segments():
    st.title("Browse segments")
    videos = _videos()
    if not videos:
        st.info("No transcribed recordings yet.")
        return
    video_id = st.selectbox("Recording", videos, key="browse_video")
    try:
        clips = rs.load_clips(video_id, META, PROJECT_ROOT)
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    query = st.text_input("Filter by segment ID")
    rows = [
        {
            "segment_id": c["segment_id"],
            "start": c.get("start"),
            "end": c.get("end"),
            "label": c.get("label") or c.get("region_label"),
            "status": c.get("status"),
            "Model A": c.get("asr_a_text"),
            "Model B": c.get("asr_b_text"),
            "Final": c.get("final_transcript"),
        }
        for c in clips if query.casefold() in c["segment_id"].casefold()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    csv_file = rs.csv_path(video_id, META)
    if csv_file.exists():
        st.download_button("Download CSV", csv_file.read_bytes(), csv_file.name, "text/csv")


def _render_analysis():
    st.title("Audio and region analysis")
    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    regions_dir = PROJECT_ROOT / "data" / "processed"
    region_files = sorted(regions_dir.glob("*_regions.json")) if regions_dir.exists() else []
    if not region_files:
        st.info("Run the pipeline to generate spectrograms and classified regions.")
        return

    path = st.selectbox(
        "Classified recording",
        region_files,
        format_func=lambda value: value.name.removesuffix("_regions.json"),
    )
    video_id = path.name.removesuffix("_regions.json")
    try:
        regions = json.loads(path.read_text(encoding="utf-8"))
        if not regions:
            st.info("No regions were classified for this recording.")
            return

        audio_source = _source_wav(video_id)
        left, right = st.columns(2)
        with left:
            st.subheader("Spectrogram and region labels")
            if audio_source:
                audio_duration = sf.info(str(audio_source)).duration
                preview_start = st.number_input(
                    "Preview start (seconds)", min_value=0.0,
                    max_value=max(0.0, float(audio_duration) - 1.0),
                    value=min(60.0, max(0.0, float(audio_duration) - 10.0)),
                    step=1.0,
                    key=f"region_preview_start_{video_id}",
                )
                figure = _region_spectrogram(audio_source, regions, preview_start)
                if figure is not None:
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)
            else:
                st.info("Source audio is unavailable for this recording.")
        with right:
            st.subheader("Amplitude envelope")
            envelopes = sorted(plots_dir.glob(f"*{video_id}*_envelope.png")) if plots_dir.exists() else []
            if envelopes:
                st.image(str(envelopes[-1]), caption=envelopes[-1].name, use_container_width=True)
            else:
                st.info("No envelope plot is available for this recording.")

        st.markdown("#### Correct region labels and boundaries")
        st.caption("Changes update the saved region manifest and an append-only region audit CSV. Existing segment audio is not regenerated by this editor.")
        editable = pd.DataFrame([
            {"start_sec": row.get("start_sec", 0.0),
             "end_sec": row.get("end_sec", 0.0),
             "label": row.get("label", "mixed")}
            for row in regions
        ])
        changed = st.data_editor(
            editable,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key=f"regions_editor_{video_id}",
            column_config={
                "start_sec": st.column_config.NumberColumn("Start (s)", min_value=0.0, step=0.05, format="%.3f"),
                "end_sec": st.column_config.NumberColumn("End (s)", min_value=0.0, step=0.05, format="%.3f"),
                "label": st.column_config.SelectboxColumn("Region label", options=["speech", "silence", "music", "mixed"], required=True),
            },
        )
        editor_reviewer = st.text_input("Your name (region audit)", key="region_reviewer").strip()
        if st.button("Save region corrections", key=f"save_regions_{video_id}", type="primary"):
            if not editor_reviewer:
                st.error("Enter your name so region changes are attributable.")
                return
            updated = []
            events = []
            previous_end = -1.0
            duration = sf.info(str(audio_source)).duration if audio_source else None
            for index, row in enumerate(changed.to_dict("records")):
                start, end = float(row["start_sec"]), float(row["end_sec"])
                label = str(row["label"])
                if start < 0 or end <= start or start < previous_end:
                    st.error("Region times must be valid, ordered, and non-overlapping.")
                    return
                if duration is not None and end > duration:
                    st.error("A region boundary extends past the source audio duration.")
                    return
                previous_end = end
                original = regions[index]
                revised = dict(original)
                revised.update({"start_sec": round(start, 3), "end_sec": round(end, 3), "label": label})
                updated.append(revised)
                if (float(original.get("start_sec", 0)) != start
                        or float(original.get("end_sec", 0)) != end
                        or original.get("label") != label):
                    stamp = datetime.now(timezone.utc).isoformat()
                    event = {
                        "changed_at": stamp,
                        "changed_by": editor_reviewer,
                        "video_id": video_id,
                        "region_index": index,
                        "before_start": original.get("start_sec", ""),
                        "before_end": original.get("end_sec", ""),
                        "before_label": original.get("label", ""),
                        "after_start": round(start, 3),
                        "after_end": round(end, 3),
                        "after_label": label,
                    }
                    revised.setdefault("human_changes", []).append(event)
                    events.append(event)
            if not events:
                st.info("No region values changed.")
                return
            try:
                audit_path = META / f"{video_id}_region_changes.csv"
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                fields = list(events[0])
                is_new = not audit_path.exists()
                with audit_path.open("a", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                    if is_new:
                        writer.writeheader()
                    writer.writerows(events)
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(path)
                updated_segments = 0
                if rs.csv_path(video_id, META).exists():
                    clips = rs.load_clips(video_id, META, PROJECT_ROOT)
                    for clip in clips:
                        if rs.status_of(clip) == "deleted":
                            continue
                        midpoint = ((clip.get("start_sec") or 0) + (clip.get("end_sec") or 0)) / 2
                        matching = next((r for r in updated if r["start_sec"] <= midpoint < r["end_sec"]), None)
                        if not matching or matching.get("label") not in LABELS:
                            continue
                        if clip.get("label") == matching.get("label"):
                            continue
                        rs.update_review(
                            video_id=video_id,
                            segment_id=clip["segment_id"],
                            status=rs.status_of(clip),
                            final_transcript=clip.get("final_transcript"),
                            label=matching["label"],
                            quality=clip.get("quality"),
                            changed_by=editor_reviewer,
                            start=clip.get("start_sec"),
                            end=clip.get("end_sec"),
                            snr=clip.get("snr_wada_db"),
                            audio_file=clip.get("audio_path") or "",
                            metadata_dir=META,
                        )
                        updated_segments += 1
                st.session_state["region_notice"] = (
                    f"Saved {len(events)} region change(s) and appended the CSV audit. "
                    f"Updated {updated_segments} overlapping segment label(s) in the review CSV."
                )
                st.rerun()
            except OSError as exc:
                st.error(f"Could not save region corrections: {exc}")
        notice = st.session_state.pop("region_notice", None)
        if notice:
            st.success(notice)
        full_frame = pd.DataFrame(regions)
        if not full_frame.empty:
            full_frame["duration_sec"] = (full_frame["end_sec"] - full_frame["start_sec"]).round(3)
            with st.expander("Classifier confidence and method"):
                columns = [c for c in ("start_sec", "end_sec", "duration_sec", "label", "confidence", "method") if c in full_frame]
                st.dataframe(full_frame[columns], use_container_width=True, hide_index=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        st.error(f"Could not read or edit region records: {exc}")


with st.sidebar:
    st.title("🎙️ Corpus Studio")
    page = st.radio(
        "Workspace",
        ["Run pipeline", "Review and export", "Browse segments", "Audio analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.metric("Videos with transcripts", len(_videos()))
    segment_dir = PROJECT_ROOT / "data" / "segments"
    st.metric("Segment audio files", len(list(segment_dir.glob("*.wav"))) if segment_dir.exists() else 0)

if page == "Run pipeline":
    _render_pipeline()
elif page == "Review and export":
    _render_review()
elif page == "Browse segments":
    _render_segments()
else:
    _render_analysis()
