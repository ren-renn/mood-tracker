import os
import threading
import time
import urllib.request

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from database import init_db, log_mood

# Initialize Database
init_db()

# Select Computing Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on device: {device}")

# Load Vision Transformer
MODEL_NAME = "dima806/facial_emotions_image_detection"
print("Loading Vision Transformer model weights...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Face Detector Cascade
CASCADE_FILE = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_FILE):
    print("Downloading face detection model...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, CASCADE_FILE)

face_cascade = cv2.CascadeClassifier(CASCADE_FILE)

# Shared Thread Data
latest_face_roi = None
is_analyzing = False
lock = threading.Lock()

current_emotion = "Detecting..."
emotion_confidence = 0.0
top_emotions = []

ALPHA = 0.4
smoothed_scores = {}


# Background Worker Thread for Heavy PyTorch Inference
def run_inference_worker():
    global \
        latest_face_roi, \
        is_analyzing, \
        current_emotion, \
        emotion_confidence, \
        top_emotions, \
        smoothed_scores

    while True:
        roi_to_process = None

        with lock:
            if latest_face_roi is not None and not is_analyzing:
                roi_to_process = latest_face_roi.copy()
                is_analyzing = True

        if roi_to_process is not None:
            try:
                pil_image = Image.fromarray(roi_to_process)
                inputs = processor(images=pil_image, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

                raw_scores = {}
                for idx, prob in enumerate(probabilities):
                    label = model.config.id2label[idx].capitalize()
                    raw_scores[label] = float(prob.item() * 100.0)

                # Exponential Moving Average Smoothing
                if not smoothed_scores:
                    smoothed_scores = raw_scores
                else:
                    for k, v in raw_scores.items():
                        smoothed_scores[k] = (ALPHA * v) + (
                            (1 - ALPHA) * smoothed_scores.get(k, v)
                        )

                sorted_emotions = sorted(
                    smoothed_scores.items(), key=lambda item: item[1], reverse=True
                )
                dom_emo, conf = sorted_emotions[0]

                with lock:
                    current_emotion = dom_emo
                    emotion_confidence = conf
                    top_emotions = [(e, f"{s:.1f}") for e, s in sorted_emotions[:3]]

                if conf > 25.0:
                    log_mood(dom_emo, round(conf, 2))
            except Exception:
                pass
            finally:
                with lock:
                    is_analyzing = False

        time.sleep(0.03)  # Yield CPU to main rendering thread


# Start Inference Thread in Background
worker = threading.Thread(target=run_inference_worker, daemon=True)
worker.start()

# Main Video Capture Loop (30+ FPS Smooth Video)
cap = cv2.VideoCapture(0)

print("High-Accuracy Async Mood Tracker running at 30+ FPS! Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=6, minSize=(120, 120)
    )

    if len(faces) > 0:
        primary_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = primary_face

        top = max(0, y - 40)
        bottom = min(frame_h, y + h + 40)
        left = max(0, x - 40)
        right = min(frame_w, x + w + 40)

        face_roi = rgb_frame[top:bottom, left:right]

        # Send cropped face to background worker thread if free
        with lock:
            if not is_analyzing:
                latest_face_roi = face_roi

        # Render bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 128), 2)
        cv2.putText(
            frame,
            f"{current_emotion} ({emotion_confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 128),
            2,
        )

    # Render HUD Card
    cv2.rectangle(frame, (20, 20), (360, 140), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (360, 140), (0, 255, 128), 1)

    with lock:
        disp_emotion = current_emotion
        disp_conf = emotion_confidence
        disp_top = list(top_emotions)

    cv2.putText(
        frame,
        f"Primary: {disp_emotion} ({disp_conf:.1f}%)",
        (35, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    y_offset = 75
    for emo_name, emo_score_str in disp_top:
        cv2.putText(
            frame,
            f"- {emo_name}: {emo_score_str}%",
            (35, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        y_offset += 20

    cv2.imshow("High-Accuracy Async Mood Tracker - PyTorch ViT", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
