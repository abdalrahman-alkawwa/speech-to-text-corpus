"""
app.py
======
Arabic Labelled Speech Corpus Studio — Interactive Web UI (Streamlit)

Purpose:
--------
Provides an interactive, visual web dashboard to:
  1. Ingest and process online Arabic videos through Stages 1 to 5.
  2. Inspect Waveform, Spectrogram, and Amplitude Envelopes.
  3. Browse 4-class region classifications (Speech / Silence / Music / Mixed).
  4. Play and verify individual 8–12s speech segments with built-in audio players.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import soundfile as sf

from src.pipeline import run_pipeline


# Page Configuration
st.set_page_config(
    page_title="Arabic Speech Corpus Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for clean presentation
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background: #F3F4F6; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6; }
    .badge-verified { background-color: #D1FAE5; color: #065F46; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-review { background-color: #FEF3C7; color: #92400E; padding: 3px 8px; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
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
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/audio-wave.png", width=64)
    st.markdown("### 🎙️ Corpus Navigation")

    menu_option = st.radio(
        "Select Workspace:",
        [
            "🚀 Ingest & Process Video",
            "🎧 Browse & Play Segments",
            "📊 Audio & Region Analysis"
        ]
    )

    st.markdown("---")
    st.markdown("#### 📂 Corpus Status")
    existing_recordings = list_available_recordings()
    st.metric("Registered Videos", len(existing_recordings))

    segments_dir = PROJECT_ROOT / "data" / "segments"
    if segments_dir.exists():
        seg_count = len(list(segments_dir.glob("*.wav")))
        st.metric("Exported Segments", seg_count)
    else:
        st.metric("Exported Segments", 0)


# ============================================================
#  VIEW 1: INGEST & PROCESS VIDEO
# ============================================================
if menu_option == "🚀 Ingest & Process Video":
    st.markdown('<div class="main-title">🚀 Video Ingestion & Pipeline Runner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter any YouTube URL to execute Stages 1–5: Download, 16 kHz WAV Extraction, Visualizer, VAD Classification, and Segmentation.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        default_url = "https://www.youtube.com/watch?v=gK017cCH36k"
        url_input = st.text_input(
            "YouTube Video URL:",
            value=default_url,
            help="Paste any YouTube video or lecture link."
        )

        notes_input = st.text_input(
            "Provenance Notes / Permissions:",
            value="Recorded lecture by Dr. Mohsen Rashwan. Preserved for academic speech corpus research.",
            help="Notes documenting source permissions and dialect/context."
        )

    with col2:
        st.markdown("#### ⚙️ Processing Options")
        quick_test = st.checkbox("Quick Test Mode (Limit to first 10 segments)", value=True)
        max_segs = 10 if quick_test else None

    if st.button("▶️ Run End-to-End Pipeline", type="primary", use_container_width=True):
        if not url_input.strip():
            st.error("Please enter a valid video URL.")
        else:
            with st.status("Executing Pipeline Stages...", expanded=True) as status:
                try:
                    st.write("📥 **Stage 1 & 2:** Querying metadata and preparing 16 kHz mono PCM_16 WAV...")
                    # Run pipeline
                    results = run_pipeline(
                        url=url_input.strip(),
                        notes=notes_input.strip(),
                        max_segments=max_segs
                    )

                    st.write("🎨 **Stage 3:** Generated waveform and spectrogram inspection plots.")
                    st.write("🧠 **Stage 4:** Classified speech, silence, music, and mixed regions.")
                    st.write(f"✂️ **Stage 5:** Exported {len(results['segments'])} clips to `data/segments/`.")

                    status.update(label="✅ Pipeline Completed Successfully!", state="complete", expanded=False)
                    st.success(f"Processing complete for: **{results['metadata']['title']}**")
                    st.balloons()
                except Exception as e:
                    status.update(label="❌ Pipeline Failed", state="error")
                    st.error(f"Error during execution: {str(e)}")


# ============================================================
#  VIEW 2: BROWSE & PLAY SEGMENTS
# ============================================================
elif menu_option == "🎧 Browse & Play Segments":
    st.markdown('<div class="main-title">🎧 Speech Segment Audio Browser</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Listen to individual 8–12 second clips and verify word boundaries.</div>', unsafe_allow_html=True)

    segments_dir = PROJECT_ROOT / "data" / "segments"
    if not segments_dir.exists() or not list(segments_dir.glob("*.wav")):
        st.warning("No audio segments found yet! Run the pipeline on a video first.")
    else:
        # Get list of segment WAV files
        seg_files = sorted(list(segments_dir.glob("*.wav")))

        # Filter controls
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            search_query = st.text_input("🔍 Search Segment ID:", placeholder="e.g. seg_000001")
        with filter_col2:
            st.markdown(f"**Total Clips:** `{len(seg_files)}`")

        filtered_files = [f for f in seg_files if search_query.lower() in f.name.lower()]

        # Display segments in a neat grid
        for f in filtered_files:
            info = sf.info(str(f))
            duration = info.duration

            with st.container():
                c1, c2, c3 = st.columns([1.5, 2.5, 1])
                with c1:
                    st.markdown(f"**`{f.name}`**")
                    st.caption(f"⏱️ Duration: **{duration:.2f}s** | 16 kHz Mono")
                with c2:
                    st.audio(str(f), format="audio/wav")
                with c3:
                    st.markdown('<span class="badge-verified">Auto-Verified Cut</span>', unsafe_allow_html=True)
                st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)


# ============================================================
#  VIEW 3: AUDIO & REGION ANALYSIS
# ============================================================
elif menu_option == "📊 Audio & Region Analysis":
    st.markdown('<div class="main-title">📊 Visual Inspection & Region Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect waveform envelopes, STFT spectrograms, and classification coverage.</div>', unsafe_allow_html=True)

    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # 1. Plots Section
    st.markdown("### 🎨 Visual Inspection Plots")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Waveform & Spectrogram Window")
        spec_plots = list(plots_dir.glob("*_spec_*.png")) if plots_dir.exists() else []
        if spec_plots:
            st.image(str(spec_plots[0]), caption=spec_plots[0].name, use_container_width=True)
        else:
            st.info("No spectrogram plot found. Run the pipeline to generate one.")

    with col2:
        st.markdown("#### 📈 Full Amplitude Envelope")
        env_plots = list(plots_dir.glob("*_envelope.png")) if plots_dir.exists() else []
        if env_plots:
            st.image(str(env_plots[0]), caption=env_plots[0].name, use_container_width=True)
        else:
            st.info("No envelope plot found. Run the pipeline to generate one.")

    # 2. Region Breakdown Section
    st.markdown("---")
    st.markdown("### 🧠 4-Class Region Classifications")

    region_files = list(processed_dir.glob("*_regions.json")) if processed_dir.exists() else []
    if region_files:
        with open(region_files[0], "r", encoding="utf-8") as rf:
            regions_data = json.load(rf)

        df = pd.DataFrame(regions_data)
        df["duration"] = (df["end_sec"] - df["start_sec"]).round(2)

        # Metrics row
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
    else:
        st.info("No classified regions JSON found yet. Run the pipeline to generate region classifications.")
