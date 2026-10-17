# 🎭 Real-Time AI Mood Tracker & Telemetry Dashboard

An end-to-end computer vision and deep learning application that performs real-time facial emotion recognition using a Vision Transformer (ViT) model, stores time-stamped telemetry in an SQLite database, and visualizes mood trends through an interactive Streamlit web dashboard.

---

## ⚡ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch & Hugging Face Transformers (`dima806/facial_emotions_image_detection`)
* **Computer Vision:** OpenCV (Asynchronous multithreaded video ingestion & face tracking)
* **Database:** SQLite3
* **Dashboard & Visualization:** Streamlit, Pandas, Plotly Express

---

## 📁 Project Structure

```text
mood-detection/
├── app.py              # Main multi-threaded CV video stream & ViT inference engine
├── dashboard.py        # Streamlit web interface for mood analytics
├── database.py         # SQLite connection & logging module
├── mood_tracker.db     # Local SQLite database (generated on first run)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone & Setup Environment
```bash
git clone [https://github.com/your-username/ai-mood-tracker.git](https://github.com/your-username/ai-mood-tracker.git)
cd ai-mood-tracker

# Create & activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Real-Time AI Camera App
```bash
python app.py
```
*Press `q` to safely exit the camera feed.*

### 3. Launch Analytics Dashboard
```bash
streamlit run dashboard.py
```
*Access the dashboard at `http://localhost:8501`.*
