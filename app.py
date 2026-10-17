import os
import sqlite3
import urllib.request

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from database import init_db, log_mood

# Initialize SQLite Database
init_db()

# Page Setup
st.set_page_config(page_title="AI Mood Tracker", page_icon="🎭", layout="wide")
st.title("🎭 AI Mood Tracker — Unified Dashboard")


# Load Vision Transformer Model (Cached so it loads only once)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device


processor, model, device = load_model()

# Load Haar Cascade
CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)


# Helper function to read database logs
def fetch_logs():
    try:
        conn = sqlite3.connect("mood_tracker.db")
        df = pd.read_sql_query("SELECT * FROM mood_logs ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# Sidebar Controls
st.sidebar.header("🕹️ Controls")
run_camera = st.sidebar.checkbox("Start Live Camera Detection", value=False)

# Main App Layout (2 Columns)
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📹 Live Camera Feed & Detection")
    frame_placeholder = st.empty()

with col_right:
    st.subheader("📊 Real-Time Metrics")
    kpi1, kpi2 = st.columns(2)
    metric_mood = kpi1.empty()
    metric_conf = kpi2.empty()

    chart_placeholder = st.empty()

st.divider()

# Historical Analytics Section
st.subheader("📈 Historical Mood Telemetry")
hist_col1, hist_col2 = st.columns(2)
line_chart_placeholder = hist_col1.empty()
pie_chart_placeholder = hist_col2.empty()

# Unified Loop: Camera Processing + Live Dashboard Updates
if run_camera:
    cap = cv2.VideoCapture(0)

    # Session state to smooth predictions
    if "smoothed_scores" not in st.session_state:
        st.session_state.smoothed_scores = {}

    ALPHA = 0.4
    frame_count = 0

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access camera feed.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(120, 120)
        )

        if len(faces) > 0:
            # Pick largest face
            primary_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = primary_face

            top = max(0, y - 40)
            bottom = min(frame_h, y + h + 40)
            left = max(0, x - 40)
            right = min(frame_w, x + w + 40)

            face_roi = rgb_frame[top:bottom, left:right]

            # Run PyTorch Model Inference every 2 frames
            if frame_count % 2 == 0:
                try:
                    pil_image = Image.fromarray(face_roi)
                    inputs = processor(images=pil_image, return_tensors="pt").to(device)

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)[0]

                    raw_scores = {}
                    for idx, prob in enumerate(probs):
                        label = model.config.id2label[idx].capitalize()
                        raw_scores[label] = float(prob.item() * 100.0)

                    # Smoothing
                    if not st.session_state.smoothed_scores:
                        st.session_state.smoothed_scores = raw_scores
                    else:
                        for k, v in raw_scores.items():
                            st.session_state.smoothed_scores[k] = (ALPHA * v) + (
                                (1 - ALPHA) * st.session_state.smoothed_scores.get(k, v)
                            )

                    sorted_emotions = sorted(
                        st.session_state.smoothed_scores.items(),
                        key=lambda i: i[1],
                        reverse=True,
                    )
                    dom_emo, conf = sorted_emotions[0]

                    # Log to Database
                    if conf > 25.0:
                        log_mood(dom_emo, round(conf, 2))

                    # Update Live KPI Display
                    metric_mood.metric("Current Mood", dom_emo)
                    metric_conf.metric("Confidence", f"{conf:.1f}%")

                except Exception:
                    pass

            # Draw Bounding Box on Frame
            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 128), 2)
            cv2.putText(
                rgb_frame,
                f"{dom_emo} ({conf:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 128),
                2,
            )

        # Render Live Video Frame directly in Streamlit
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

        # Refresh Charts every 10 frames (~0.3 seconds)
        if frame_count % 10 == 0:
            df = fetch_logs()
            if not df.empty:
                # Top emotions breakdown bar chart
                top_3 = sorted_emotions[:3]
                df_top3 = pd.DataFrame(top_3, columns=["Emotion", "Confidence (%)"])
                fig_bar = px.bar(
                    df_top3,
                    x="Emotion",
                    y="Confidence (%)",
                    color="Emotion",
                    height=250,
                )
                chart_placeholder.plotly_chart(fig_bar, use_container_width=True)

                # Historical Line Chart
                fig_line = px.line(
                    df.head(50),
                    x="timestamp",
                    y="confidence",
                    color="emotion",
                    title="Recent Telemetry",
                )
                line_chart_placeholder.plotly_chart(fig_line, use_container_width=True)

                # Overall Pie Chart
                mood_counts = df["emotion"].value_counts().reset_index()
                mood_counts.columns = ["Emotion", "Count"]
                fig_pie = px.pie(
                    mood_counts,
                    names="Emotion",
                    values="Count",
                    hole=0.4,
                    title="Overall Distribution",
                )
                pie_chart_placeholder.plotly_chart(fig_pie, use_container_width=True)

        frame_count += 1

    cap.release()
else:
    frame_placeholder.info(
        "Click **Start Live Camera Detection** in the sidebar to launch the camera feed."
    )

    # Display static historical charts when camera is off
    df = fetch_logs()
    if not df.empty:
        fig_line = px.line(
            df.head(100),
            x="timestamp",
            y="confidence",
            color="emotion",
            title="Recent Telemetry Stream",
        )
        line_chart_placeholder.plotly_chart(fig_line, use_container_width=True)

        mood_counts = df["emotion"].value_counts().reset_index()
        mood_counts.columns = ["Emotion", "Count"]
        fig_pie = px.pie(
            mood_counts,
            names="Emotion",
            values="Count",
            hole=0.4,
            title="Overall Distribution",
        )
        pie_chart_placeholder.plotly_chart(fig_pie, use_container_width=True)
