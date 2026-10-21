# Real-Time AI Mood Tracker & Telemetry Dashboard

An end-to-end computer vision and machine learning web application that performs real-time facial emotion recognition using a Vision Transformer (ViT) model. It stores time-stamped emotion telemetry in an SQLite database and visualizes live and historical mood trends through an interactive, unified Streamlit web dashboard.

---

## Features

* **Real-Time Video Inference:** Asynchronous facial detection and emotion classification from a live webcam feed.
* **State-of-the-Art AI:** Powered by a Hugging Face Vision Transformer (`dima806/facial_emotions_image_detection`) capable of classifying 7 distinct facial emotions with high accuracy.
* **Unified Web Dashboard:** A single-page Streamlit application that renders live video alongside real-time analytical charts and metrics.
* **Telemetry Persistence:** Continuous background logging of dominant emotions and confidence scores into a local SQLite database for historical tracking.

---

## Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch & Hugging Face Transformers
* **Computer Vision:** OpenCV (Haar Cascade face detection & video frame ingestion)
* **Database:** SQLite3
* **Dashboard & Visualization:** Streamlit, Pandas, Plotly Express

---

## Project Structure

```text
mood-detection/
├── app.py              # Unified multi-threaded CV video stream & analytics dashboard
├── database.py         # SQLite connection & telemetry logging module
├── mood_tracker.db     # Local SQLite database (Auto-generated on first run)
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusion rules for large weights and local logs
└── README.md           # Project documentation
