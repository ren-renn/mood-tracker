import os
import time
import urllib.request
import sqlite3
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

# Load Vision Transformer Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "dima806/facial_emotions_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model, device

processor, model, device = load_model()

# Load Haar Cascade Face Detector
CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

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

# Main App Layout
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

# Camera Loop & Real-Time Analytics
if run_camera:
    cap = cv2.VideoCapture(0)

    if "smoothed_scores" not in st.session_state:
        st.session_state.smoothed_scores = {}

    ALPHA = 0.4
    last_inference_time = 0
    INFERENCE_INTERVAL = 0.35

    last_chart_time = 0
    CHART_INTERVAL = 1.0
    chart_counter = 0  # Dynamic counter prevents Streamlit duplicate key crashes

    dom_emo = "Detecting..."
    conf = 0.0
    sorted_emotions = []

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access camera feed.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(120, 120))

        if len(faces) > 0:
            primary_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = primary_face

            top = max(0, y - 40)
            bottom = min(frame_h, y + h + 40)
            left = max(0, x - 40)
            right = min(frame_w, x + w + 40)

            face_roi = rgb_frame[top:bottom, left:right]

            # Inference pass
            current_time = time.time()
            if current_time - last_inference_time > INFERENCE_INTERVAL:
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

                    if not st.session_state.smoothed_scores:
                        st.session_state.smoothed_scores = raw_scores
                    else:
                        for k, v in raw_scores.items():
                            st.session_state.smoothed_scores[k] = (ALPHA * v) + ((1 - ALPHA) * st.session_state.smoothed_scores.get(k, v))

                    sorted_emotions = sorted(st.session_state.smoothed_scores.items(), key=lambda i: i[1], reverse=True)
                    dom_emo, conf = sorted_emotions[0]

                    if conf > 25.0:
                        log_mood(dom_emo, round(conf, 2))

                    metric_mood.metric("Current Mood", dom_emo)
                    metric_conf.metric("Confidence", f"{conf:.1f}%")

                    last_inference_time = current_time

                except Exception:
                    pass

            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 128), 2)
            cv2.putText(rgb_frame, f"{dom_emo} ({conf:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)

        # Frame continuously renders without crashing
        frame_placeholder.image(rgb_frame, channels="RGB", width="stretch")

        # Dynamic UI Chart Refresh
        current_time = time.time()
        if current_time - last_chart_time > CHART_INTERVAL:
            df = fetch_logs()
            if not df.empty:
                chart_counter += 1  # Ensures keys are 100% unique every loop iteration

                if sorted_emotions:
                    top_3 = sorted_emotions[:3]
                    df_top3 = pd.DataFrame(top_3, columns=["Emotion", "Confidence (%)"])
                    fig_bar = px.bar(df_top3, x="Emotion", y="Confidence (%)", color="Emotion", height=250)
                    chart_placeholder.plotly_chart(fig_bar, width="stretch", key=f"bar_{chart_counter}")

                fig_line = px.line(df.head(50), x="timestamp", y="confidence", color="emotion", title="Recent Telemetry Stream")
                line_chart_placeholder.plotly_chart(fig_line, width="stretch", key=f"line_{chart_counter}")

                mood_counts = df["emotion"].value_counts().reset_index()
                mood_counts.columns = ["Emotion", "Count"]
                fig_pie = px.pie(mood_counts, names="Emotion", values="Count", hole=0.4, title="Overall Distribution")
                pie_chart_placeholder.plotly_chart(fig_pie, width="stretch", key=f"pie_{chart_counter}")

            last_chart_time = current_time

        time.sleep(0.01)

    cap.release()
else:
    frame_placeholder.info("Click **Start Live Camera Detection** in the sidebar to launch the camera feed.")

    df = fetch_logs()
    if not df.empty:
        fig_line = px.line(df.head(100), x="timestamp", y="confidence", color="emotion", title="Recent Telemetry Stream")
        line_chart_placeholder.plotly_chart(fig_line, width="stretch", key="static_line_chart")

        mood_counts = df["emotion"].value_counts().reset_index()
        mood_counts.columns = ["Emotion", "Count"]
        fig_pie = px.pie(mood_counts, names="Emotion", values="Count", hole=0.4, title="Overall Distribution")
        pie_chart_placeholder.plotly_chart(fig_pie, width="stretch", key="static_pie_chart")
