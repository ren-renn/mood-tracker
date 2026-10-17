import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Mood Detection Dashboard",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 AI Mood Detection - Analytics Dashboard")
st.caption("Real-time telemetry and mood trend analytics extracted from mood_tracker.db")

def load_data():
    try:
        conn = sqlite3.connect("mood_tracker.db")
        df = pd.read_sql_query("SELECT * FROM mood_logs", conn)
        conn.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()

df = load_data()

# Auto-refresh button
if st.button("🔄 Refresh Data"):
    st.rerun()

if df.empty:
    st.warning("No logs found in `mood_tracker.db` yet. Run `python app.py` and sit in front of the camera to generate records!")
else:
    # 1. KPI Metric Summary Cards
    col1, col2, col3, col4 = st.columns(4)

    total_logs = len(df)
    dominant_mood = df["emotion"].mode()[0] if not df.empty else "N/A"
    avg_confidence = df["confidence"].mean() if not df.empty else 0.0
    latest_entry = df.iloc[-1] if not df.empty else None

st.divider()

# Graphical Analytics Layout
col_left, col_right = st.columns(2)

with col_left:
        st.subheader("Overall Mood Distribution")
        mood_counts = df["emotion"].value_counts().reset_index()
        mood_counts.columns = ["Emotion", "Count"]

        fig_pie = px.pie(
            mood_counts,
            names="Emotion",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
        st.subheader("Average Confidence Score Per Mood")
        avg_conf_df = df.groupby("emotion")["confidence"].mean().reset_index()

        fig_bar = px.bar(
            avg_conf_df,
            x="emotion",
            y="confidence",
            color="emotion",
            labels={"emotion": "Emotion", "confidence": "Avg Confidence (%)"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# Time Series Detection Plot
st.subheader("Mood Telemetry Timeline")
fig_line = px.line(
    df,
    x="timestamp",
    y="confidence",
    color="emotion",
    markers=True,
    labels={"timestamp": "Time", "confidence": "Confidence (%)", "emotion": "Detected Mood"},
    title="Recorded Mood Detection Stream Over Time"
)
st.plotly_chart(fig_line, use_container_width=True)

# Raw Records Table
with st.expander("📁 View Raw Database Logs"):
        st.dataframe(
            df.sort_values(by="timestamp", ascending=False),
            use_container_width=True
        )
