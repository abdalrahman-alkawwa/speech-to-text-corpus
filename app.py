"""
app.py
======
Arabic Labelled Speech Corpus Studio — Interactive Web UI (Streamlit)

Features:
  1. Ingest & Process Video or Local Audio/Video Files (Stages 1–6)
  2. Audio & Region Analysis (Stages 3 & 4)
  3. Browse & Play Segments (Stage 5 & 6)
  4. Human Annotation Workbench (Stages 7–10) using JSON files with Audit Trail
  5. Corpus Export & Manifests (Stage 11)
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import soundfile as sf

from src.pipeline import run_pipeline
from src.transcript_store import (
    get_all_segments, update_ground_truth,
    get_audit_trail, get_corpus_statistics, TRANSCRIPTS_DIR
)
from src.export_corpus import export_corpus_manifests


# Page Configuration
st.set_page_config(
    page_title="Arabic Speech Corpus Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .badge-verified { background-color: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-review { background-color: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-difficult { background-color: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .arabic-text { font-family: 'Segoe UI', Tahoma, Arial, sans-serif; font-size: 1.15rem; line-height: 1.8; direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)


def list_available_recordings():
    """Finds all recordings with metadata JSON in the metadata/ folder."""
    meta_dir = PROJECT_ROOT / "metadata"
    records = []
    if meta_dir.exists():
        for f in meta_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    records.append(data)
            except Exception:
                pass
    return records


# ============================================================
#  SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/audio-wave.png", width=64)
    st.markdown("### 🎙️ Corpus Navigation")

    menu_option = st.radio(
        "Select Workspace:",
        [
            "🚀 Ingest & Process Audio",
            "📊 Audio & Region Analysis",
            "🎧 Browse & Play Segments",
            "✍️ Annotation Workbench (JSON)",
            "📦 Corpus Export & Manifests"
        ]
    )

    st.markdown("---")
    st.markdown("#### 📂 Corpus Overview")
    existing_recordings = list_available_recordings()
    st.metric("Registered Media", len(existing_recordings))

    stats = get_corpus_statistics()
    st.metric("Total Segments", stats["total_segments"])
    st.metric("Speech Duration", f"{stats['total_duration_sec']/60:.1f} min")

    st.markdown("---")
    st.caption("📁 Transcripts stored in `data/transcripts/*.json`")


# ============================================================
#  VIEW 1: INGEST & PROCESS AUDIO (URL OR LOCAL UPLOAD)
# ============================================================
if menu_option == "🚀 Ingest & Process Audio":
    st.markdown('<div class="main-title">🚀 Audio Ingestion & Pipeline Runner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Acquire audio from YouTube or upload local files from your laptop to run extraction, segmentation, and ASR transcription.</div>', unsafe_allow_html=True)

    tab_url, tab_upload = st.tabs(["🌐 From YouTube URL", "💻 Upload from Laptop"])

    # --- TAB 1: YOUTUBE URL ---
    with tab_url:
        col1, col2 = st.columns([2, 1])
        with col1:
            default_url = "https://www.youtube.com/watch?v=xotB2qn9tDQ"
            url_input = st.text_input(
                "YouTube Video URL:",
                value=default_url,
                help="Paste any YouTube video or lecture link."
            )
            notes_input = st.text_input(
                "Provenance Notes / Source Info:",
                value="Arabic lecture/vlog collected for academic speech corpus research.",
                key="notes_url"
            )

        with col2:
            st.markdown("#### ⚙️ Pipeline Options")
            quick_test = st.checkbox("Quick Test Mode (Limit to 10 segments)", value=True, key="qt_url")
            enable_asr = st.checkbox("Run Dual ASR (Whisper + MMS/Tiny)", value=True, key="asr_url", help="Transcribes segments on GPU and saves to JSON")
            max_segs = 10 if quick_test else None

        if st.button("▶️ Run Pipeline on URL", type="primary", use_container_width=True):
            if not url_input.strip():
                st.error("Please enter a valid video URL.")
            else:
                with st.status("Executing Pipeline Stages...", expanded=True) as status:
                    try:
                        st.write("📥 **Stage 1 & 2:** Querying metadata and preparing 16 kHz mono PCM_16 WAV...")
                        results = run_pipeline(
                            url=url_input.strip(),
                            notes=notes_input.strip(),
                            max_segments=max_segs,
                            run_asr=enable_asr
                        )

                        st.write("🎨 **Stage 3:** Generated waveform and spectrogram inspection plots.")
                        st.write("🧠 **Stage 4:** Classified speech, silence, music, and mixed regions.")
                        st.write(f"✂️ **Stage 5:** Exported {len(results['segments'])} clips to `data/segments/`.")
                        st.write(f"📡 **Stage 6:** Estimated SNR (WADA: {results['snr']['snr_wada_db']:.1f} dB).")
                        st.write(f"📄 **Stage 7–10:** Transcripts saved to `data/transcripts/{results['metadata']['video_id']}_transcripts.json`.")

                        status.update(label="✅ Pipeline Completed Successfully!", state="complete", expanded=False)
                        st.success(f"Processing complete for: **{results['metadata']['title']}**")
                        st.balloons()
                    except Exception as e:
                        status.update(label="❌ Pipeline Failed", state="error")
                        st.error(f"Error during execution: {str(e)}")

    # --- TAB 2: LOCAL FILE UPLOAD FROM LAPTOP ---
    with tab_upload:
        col_up1, col_up2 = st.columns([2, 1])

        with col_up1:
            uploaded_file = st.file_uploader(
                "Choose an audio or video file from your computer:",
                type=["wav", "mp3", "m4a", "webm", "mp4", "ogg", "flac"],
                help="Upload any audio or video file from your local storage."
            )
            local_title = st.text_input("Custom Title (optional):", placeholder="e.g. My_Speech_Recording")
            local_notes = st.text_input("Notes / Description:", value="Uploaded from local computer for speech corpus.")

        with col_up2:
            st.markdown("#### ⚙️ Pipeline Options")
            quick_test_local = st.checkbox("Quick Test Mode (Limit to 10 segments)", value=True, key="qt_local")
            enable_asr_local = st.checkbox("Run Dual ASR (Whisper + MMS/Tiny)", value=True, key="asr_local")
            max_segs_local = 10 if quick_test_local else None

        if st.button("🚀 Process Uploaded File", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.error("Please select a file to upload first.")
            else:
                with st.status("Processing Uploaded Media...", expanded=True) as status:
                    try:
                        # Save uploaded bytes to data/raw/
                        raw_dir = PROJECT_ROOT / "data" / "raw"
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        temp_save_path = raw_dir / uploaded_file.name

                        with open(temp_save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        st.write(f"📁 **Saved file to disk:** `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.2f} MB)")
                        st.write("⚙️ **Running end-to-end extraction and segmentation...**")

                        results = run_pipeline(
                            local_file_path=temp_save_path,
                            notes=local_notes,
                            max_segments=max_segs_local,
                            run_asr=enable_asr_local
                        )

                        vid_id = results['metadata']['video_id']
                        st.write(f"✂️ **Stage 5:** Exported {len(results['segments'])} clips to `data/segments/`.")
                        st.write(f"📄 **Stage 7–10:** Transcripts saved to `data/transcripts/{vid_id}_transcripts.json`.")

                        status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                        st.success(f"Successfully processed uploaded file: **{uploaded_file.name}**")
                        st.balloons()
                    except Exception as e:
                        status.update(label="❌ Upload Processing Failed", state="error")
                        st.error(f"Error: {str(e)}")


# ============================================================
#  VIEW 2: AUDIO & REGION ANALYSIS
# ============================================================
elif menu_option == "📊 Audio & Region Analysis":
    st.markdown('<div class="main-title">📊 Visual Inspection & Region Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect waveform envelopes, STFT spectrograms, and classification coverage.</div>', unsafe_allow_html=True)

    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    st.markdown("### 🎨 Visual Inspection Plots")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Waveform & Spectrogram Window")
        spec_plots = list(plots_dir.glob("*_spec_*.png")) if plots_dir.exists() else []
        if spec_plots:
            selected_spec = st.selectbox("Select Spectrogram Plot:", [p.name for p in spec_plots])
            st.image(str(plots_dir / selected_spec), caption=selected_spec, use_container_width=True)
        else:
            st.info("No spectrogram plot found. Run the pipeline to generate one.")

    with col2:
        st.markdown("#### 📈 Full Amplitude Envelope")
        env_plots = list(plots_dir.glob("*_envelope.png")) if plots_dir.exists() else []
        if env_plots:
            selected_env = st.selectbox("Select Envelope Plot:", [p.name for p in env_plots])
            st.image(str(plots_dir / selected_env), caption=selected_env, use_container_width=True)
        else:
            st.info("No envelope plot found. Run the pipeline to generate one.")

    st.markdown("---")
    st.markdown("### 🧠 4-Class Region Classifications")

    region_files = list(processed_dir.glob("*_regions.json")) if processed_dir.exists() else []
    if region_files:
        selected_file = st.selectbox("Select regions file:", [f.name for f in region_files])
        file_path = processed_dir / selected_file

        with open(file_path, "r", encoding="utf-8") as rf:
            regions_data = json.load(rf)

        df = pd.DataFrame(regions_data)
        df["duration"] = (df["end_sec"] - df["start_sec"]).round(2)

        total_dur = df["duration"].sum()
        speech_dur = df[df["label"] == "speech"]["duration"].sum()
        silence_dur = df[df["label"] == "silence"]["duration"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Regions", len(df))
        m2.metric("Total Duration", f"{total_dur/60:.1f} min")
        m3.metric("Speech %", f"{(speech_dur/total_dur*100):.1f}%" if total_dur > 0 else "0%")
        m4.metric("Silence %", f"{(silence_dur/total_dur*100):.1f}%" if total_dur > 0 else "0%")

        st.dataframe(
            df[["start_sec", "end_sec", "duration", "label", "confidence", "method"]],
            use_container_width=True,
            height=300
        )


# ============================================================
#  VIEW 3: BROWSE & PLAY SEGMENTS
# ============================================================
elif menu_option == "🎧 Browse & Play Segments":
    st.markdown('<div class="main-title">🎧 Speech Segment Audio Browser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Listen to individual 8–12s speech segments and verify acoustic boundaries.</div>', unsafe_allow_html=True)

    segments_dir = PROJECT_ROOT / "data" / "segments"
    if not segments_dir.exists() or not list(segments_dir.glob("*.wav")):
        st.warning("No audio segments found yet! Run the pipeline on a video or audio file first.")
    else:
        seg_files = sorted(list(segments_dir.glob("*.wav")))

        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            search_query = st.text_input("🔍 Search Segment ID:", placeholder="e.g. xotB2qn9tDQ")
        with filter_col2:
            st.markdown(f"**Total Audio Clips:** `{len(seg_files)}`")

        filtered_files = [f for f in seg_files if search_query.lower() in f.name.lower()]

        for f in filtered_files:
            info = sf.info(str(f))
            duration = info.duration

            with st.container():
                c1, c2, c3 = st.columns([1.5, 2.5, 1])
                with c1:
                    st.markdown(f"**`{f.name}`**")
                    st.caption(f"⏱️ Duration: **{duration:.2f}s** | 16 kHz Mono PCM_16")
                with c2:
                    st.audio(str(f), format="audio/wav")
                with c3:
                    st.markdown('<span class="badge-verified">Auto-Verified Cut</span>', unsafe_allow_html=True)
                st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)


# ============================================================
#  VIEW 4: HUMAN ANNOTATION WORKBENCH (JSON STORE)
# ============================================================
elif menu_option == "✍️ Annotation Workbench (JSON)":
    st.markdown('<div class="main-title">✍️ Human Annotation Workbench (JSON-Based)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review dual ASR transcripts, edit text, and verify segments. All changes are saved directly into <code>data/transcripts/&lt;video_id&gt;_transcripts.json</code>.</div>', unsafe_allow_html=True)

    # Filter controls
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        triage_filter = st.selectbox(
            "Filter by Triage:",
            ["All", "accepted", "review_needed", "difficult", "unreviewed"]
        )
    with f_col2:
        status_filter = st.selectbox(
            "Filter by Status:",
            ["All", "unreviewed", "verified", "flagged"]
        )
    with f_col3:
        user_name = st.text_input("Annotator Name:", value="Annotator_1")

    t_arg = None if triage_filter == "All" else triage_filter
    s_arg = None if status_filter == "All" else status_filter

    segments = get_all_segments(triage_filter=t_arg, status_filter=s_arg)

    if not segments:
        st.info("No segments match the selected filters. Process a video with Dual ASR enabled to generate transcripts.")
    else:
        st.markdown(f"**Displaying `{len(segments)}` segments loaded from JSON:**")

        for seg in segments:
            seg_id = seg["segment_id"]
            wav_path = PROJECT_ROOT / seg.get("audio_path", "")

            status_color = "badge-verified" if seg["status"] == "verified" else ("badge-difficult" if seg["status"] == "flagged" else "badge-review")
            wer_display = f"{seg['wer']:.1%}" if seg.get("wer") is not None else "N/A"

            with st.expander(f"🎙️ {seg_id} — [{seg['status'].upper()}] (WER: {wer_display})", expanded=False):
                col_audio, col_meta = st.columns([1.5, 1])

                with col_audio:
                    if wav_path.exists():
                        st.audio(str(wav_path), format="audio/wav")
                    else:
                        st.warning(f"Audio file missing: {seg.get('audio_path')}")

                with col_meta:
                    st.markdown(f"**Duration:** `{seg['duration_sec']:.2f}s` | **WADA-SNR:** `{seg.get('snr_wada_db') or 0.0:.1f} dB`")
                    st.markdown(f"**Cut Boundary:** `{seg.get('boundary_check', 'auto_verified')}`")
                    st.markdown(f"**Disagreement Triage:** `{seg.get('triage', 'unreviewed')}`")

                # Dual ASR hypotheses
                st.markdown("---")
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown(f"##### 🤖 ASR A ({seg.get('engine_a', 'Whisper')})")
                    st.markdown(f'<div class="arabic-text">{seg.get("transcript_a") or "<i>No transcript available</i>"}</div>', unsafe_allow_html=True)
                with c_b:
                    st.markdown(f"##### 🤖 ASR B ({seg.get('engine_b', 'MMS/Tiny')})")
                    st.markdown(f'<div class="arabic-text">{seg.get("transcript_b") or "<i>No transcript available</i>"}</div>', unsafe_allow_html=True)

                # Ground Truth Editing
                st.markdown("---")
                st.markdown("##### ✍️ Verified Ground Truth Transcription")
                current_gt = seg.get("ground_truth") or seg.get("transcript_a") or ""
                edited_text = st.text_area(
                    label="Edit and verify text for this clip:",
                    value=current_gt,
                    key=f"gt_{seg_id}",
                    height=80
                )

                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                with btn_col1:
                    if st.button("💾 Save & Verify", key=f"btn_v_{seg_id}", type="primary"):
                        update_ground_truth(
                            segment_id=seg_id,
                            new_ground_truth=edited_text.strip(),
                            status="verified",
                            changed_by=user_name
                        )
                        st.success(f"Saved into JSON for {seg_id}!")
                        st.rerun()

                with btn_col2:
                    if st.button("🚩 Flag for Review", key=f"btn_f_{seg_id}"):
                        update_ground_truth(
                            segment_id=seg_id,
                            new_ground_truth=edited_text.strip(),
                            status="flagged",
                            changed_by=user_name
                        )
                        st.warning(f"Flagged {seg_id} in JSON!")
                        st.rerun()

                # Audit Trail Section
                with st.popover("📜 View Revision Audit Trail"):
                    audit_records = get_audit_trail(seg_id)
                    if audit_records:
                        for a in audit_records:
                            st.caption(f"**{a.get('timestamp', '')[:19]}** by `{a.get('changed_by')}` — changed `{a.get('field')}`:")
                            st.text(f"Old: {a.get('old_value')}")
                            st.text(f"New: {a.get('new_value')}")
                            st.divider()
                    else:
                        st.write("No revisions recorded yet. Edits will be recorded here.")


# ============================================================
#  VIEW 5: CORPUS EXPORT & MANIFESTS
# ============================================================
elif menu_option == "📦 Corpus Export & Manifests":
    st.markdown('<div class="main-title">📦 Dataset Manifests & Corpus Packaging</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Package all segments from <code>data/transcripts/</code> into HuggingFace JSONL manifests, CSV summaries, and train/val/test splits.</div>', unsafe_allow_html=True)

    stats = get_corpus_statistics()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Clips", stats["total_segments"])
    m2.metric("Total Audio Hours", f"{stats['total_duration_sec']/3600:.2f} hrs")
    m3.metric("Verified Count", stats["status_breakdown"].get("verified", 0))
    m4.metric("Average SNR", f"{stats['avg_snr_db']:.1f} dB")

    st.markdown("---")
    st.markdown("### 📤 Export Distribution Manifests")
    st.markdown("""
    Generates three distribution files in `metadata/`:
    - **`corpus_manifest.jsonl`** — HuggingFace audio dataset format with audio paths, ground truth sentences, duration, and split tags.
    - **`corpus_manifest.csv`** — Complete spreadsheet table of all clips, SNR, WER, and boundaries.
    - **`gold_evaluation_sample.jsonl`** — 10-minute gold standard test split prioritizing human-verified clips.
    """)

    split_col1, split_col2, split_col3 = st.columns(3)
    with split_col1:
        tr = st.slider("Train Split %", 50, 90, 80)
    with split_col2:
        va = st.slider("Validation Split %", 5, 25, 10)
    with split_col3:
        te = 100 - tr - va
        st.metric("Test Split %", f"{te}%")

    if st.button("🚀 Export Manifests Now", type="primary", use_container_width=True):
        if stats["total_segments"] == 0:
            st.error("Cannot export: no segments found in transcripts folder. Process audio first.")
        else:
            with st.spinner("Packaging dataset manifests..."):
                res = export_corpus_manifests(
                    train_ratio=tr / 100.0,
                    val_ratio=va / 100.0,
                    test_ratio=te / 100.0
                )
                st.success("✅ Manifests exported successfully to `metadata/`!")
                st.json(res)
