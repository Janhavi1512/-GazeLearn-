import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from datetime import datetime
from utils.attention_detector import AttentionDetector
from utils.session_tracker import SessionTracker

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GazeLearn",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0d0d0d; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
}

.status-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 20px;
}

.focused   { background: #0a2e1a; color: #00ff88; border: 2px solid #00ff88; }
.distracted{ background: #2e1a0a; color: #ff8800; border: 2px solid #ff8800; }
.drowsy    { background: #2e0a0a; color: #ff3333; border: 2px solid #ff3333; }
.unknown   { background: #1a1a1a; color: #888;    border: 2px solid #444;    }

.metric-card {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}

.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00ff88;
}

.metric-card .label {
    color: #888;
    font-size: 0.85rem;
    margin-top: 4px;
}

.stButton > button {
    background: #00ff88 !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}

.stButton > button:hover {
    background: #00cc6a !important;
}

.sidebar-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00ff88;
    margin-bottom: 6px;
}

.sidebar-sub {
    color: #888;
    font-size: 0.8rem;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────
if "running"         not in st.session_state: st.session_state.running         = False
if "session_tracker" not in st.session_state: st.session_state.session_tracker = SessionTracker()
if "detector"        not in st.session_state: st.session_state.detector        = AttentionDetector()
if "current_state"   not in st.session_state: st.session_state.current_state   = "unknown"
if "session_started" not in st.session_state: st.session_state.session_started = None


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">👁️ GazeLearn</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Real-Time Attention Detection</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    ear_thresh = st.slider("EAR Drowsiness Threshold", 0.15, 0.35, 0.22, 0.01,
                           help="Eye Aspect Ratio below this = drowsy")
    consec_frames = st.slider("Consecutive Frames for Drowsiness", 10, 40, 20,
                              help="Frames eyes must be closed to trigger drowsy alert")
    gaze_thresh = st.slider("Gaze Deviation Threshold", 0.3, 0.8, 0.5, 0.05,
                            help="Gaze ratio above this = looking away")

    st.markdown("---")
    st.markdown("### 📋 How it works")
    st.info("""
**FOCUSED** – Eyes open, gaze forward, head upright

**DISTRACTED** – Gaze deviated left/right, head turned

**DROWSY** – Eyes closing, low EAR score detected
    """)

    st.markdown("---")
    st.markdown("### 🔬 Tech Stack")
    st.markdown("""
- `OpenCV` – Video capture  
- `MediaPipe` – Face mesh (468 landmarks)  
- `Scikit-learn` – State classifier  
- `Streamlit` – Dashboard  
- `Pandas` – Session analytics  
    """)


# ── Main layout ───────────────────────────────────────────────
st.markdown("# 👁️ GazeLearn")
st.markdown("##### Real-Time Student Attention Detection System")
st.markdown("---")

col_feed, col_info = st.columns([3, 2], gap="large")

with col_feed:
    st.markdown("### 📹 Live Feed")
    frame_placeholder = st.empty()

with col_info:
    st.markdown("### 🧠 Attention State")
    state_placeholder = st.empty()

    st.markdown("### 📊 Live Metrics")
    m1, m2, m3 = st.columns(3)
    ear_placeholder   = m1.empty()
    gaze_placeholder  = m2.empty()
    score_placeholder = m3.empty()

# ── Controls ──────────────────────────────────────────────────
st.markdown("---")
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 4])

with ctrl1:
    start_btn = st.button("▶ Start Session", use_container_width=True)
with ctrl2:
    stop_btn  = st.button("⏹ Stop Session",  use_container_width=True)

if start_btn:
    st.session_state.running         = True
    st.session_state.session_started = datetime.now()
    st.session_state.session_tracker = SessionTracker()

if stop_btn:
    st.session_state.running = False

# ── Analytics section ─────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Session Report")

report_placeholder = st.empty()

# ── Main loop ─────────────────────────────────────────────────
if st.session_state.running:
    cap      = cv2.VideoCapture(0)
    detector = st.session_state.detector
    tracker  = st.session_state.session_tracker

    detector.update_thresholds(ear_thresh, consec_frames, gaze_thresh)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access webcam. Check camera permissions.")
            break

        frame = cv2.flip(frame, 1)
        result = detector.process_frame(frame)

        state = result["state"]
        ear   = result["ear"]
        gaze  = result["gaze_ratio"]
        score = result["attention_score"]

        tracker.log(state, ear, gaze, score)

        # ── Draw on frame ──
        annotated = result["annotated_frame"]
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # ── State badge ──
        css_class = {"focused": "focused", "distracted": "distracted",
                     "drowsy": "drowsy"}.get(state, "unknown")
        emoji     = {"focused": "✅", "distracted": "⚠️",
                     "drowsy": "😴"}.get(state, "❓")
        state_placeholder.markdown(
            f'<div class="status-box {css_class}">{emoji} {state.upper()}</div>',
            unsafe_allow_html=True
        )

        # ── Metrics ──
        ear_placeholder.markdown(
            f'<div class="metric-card"><div class="value">{ear:.2f}</div>'
            f'<div class="label">EAR Score</div></div>', unsafe_allow_html=True)
        gaze_placeholder.markdown(
            f'<div class="metric-card"><div class="value">{gaze:.2f}</div>'
            f'<div class="label">Gaze Ratio</div></div>', unsafe_allow_html=True)
        score_placeholder.markdown(
            f'<div class="metric-card"><div class="value">{score}%</div>'
            f'<div class="label">Attention %</div></div>', unsafe_allow_html=True)

        # ── Session report ──
        report = tracker.get_report()
        df     = pd.DataFrame([report])
        report_placeholder.dataframe(df, use_container_width=True, hide_index=True)

        time.sleep(0.03)

    cap.release()

else:
    # ── Idle placeholders ──
    frame_placeholder.markdown(
        """<div style="background:#111;border:2px dashed #333;border-radius:12px;
        height:360px;display:flex;align-items:center;justify-content:center;
        color:#444;font-family:'Space Mono',monospace;font-size:1rem;">
        Press ▶ Start Session to begin</div>""",
        unsafe_allow_html=True
    )
    state_placeholder.markdown(
        '<div class="status-box unknown">❓ WAITING</div>', unsafe_allow_html=True)

    ear_placeholder.markdown(
        '<div class="metric-card"><div class="value">—</div>'
        '<div class="label">EAR Score</div></div>', unsafe_allow_html=True)
    gaze_placeholder.markdown(
        '<div class="metric-card"><div class="value">—</div>'
        '<div class="label">Gaze Ratio</div></div>', unsafe_allow_html=True)
    score_placeholder.markdown(
        '<div class="metric-card"><div class="value">—</div>'
        '<div class="label">Attention %</div></div>', unsafe_allow_html=True)

    # show last session report if available
    report = st.session_state.session_tracker.get_report()
    if report["total_frames"] > 0:
        df = pd.DataFrame([report])
        report_placeholder.dataframe(df, use_container_width=True, hide_index=True)
    else:
        report_placeholder.info("No session data yet. Start a session to see your report.")
