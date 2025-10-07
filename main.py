import os
import time
import threading
import signal
from collections import deque
from flask import Flask, Response, jsonify
import cv2
import numpy as np

# ========== Config ==========
VIDEO_SOURCE = "rtsp://admin:admin@192.168.1.4:8554/live"  # Use webcam or replace with RTSP
FRAME_WIDTH = 320
FRAME_SKIP = 2
CONF_THR = 0.4
ABSENCE_SECONDS = 10.0
JPG_QUALITY = 50
ALERT_DIR = "alerts"
MODEL_PATH = "models/yolov8n.onnx"
INPUT_SIZE = (320, 320)
CLASS_ID_PERSON = 0
STREAM_MAX_FPS = 10

# ========== State ==========
raw_lock = threading.Lock()
raw_frame = None
raw_ts = 0.0
proc_lock = threading.Lock()
latest_jpeg = None
status_lock = threading.Lock()
status = {"persons": 0, "absent_s": 0.0, "last_alert": ""}
stop_event = threading.Event()

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def preprocess(image):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
    return blob

def postprocess(outputs, image_shape, conf_threshold=CONF_THR):
    h, w = image_shape
    boxes = []
    for det in outputs[0]:
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold and class_id == CLASS_ID_PERSON:
            cx, cy, bw, bh = det[0:4]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            boxes.append((x, y, x2, y2, float(confidence)))
    return boxes

def capture_thread():
    global raw_frame, raw_ts
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        h, w = frame.shape[:2]
        if w != FRAME_WIDTH:
            scale = FRAME_WIDTH / float(w)
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)))
        with raw_lock:
            raw_frame = frame.copy()
            raw_ts = time.monotonic()

def process_thread():
    global latest_jpeg
    ensure_dir(ALERT_DIR)

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    last_seen = time.monotonic()
    frame_count = 0

    while not stop_event.is_set():
        with raw_lock:
            frame = None if raw_frame is None else raw_frame.copy()
            ts = raw_ts
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            ts_text = time.strftime("%H:%M:%S")
            cv2.putText(frame, ts_text, (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
            if ok:
                with proc_lock:
                    latest_jpeg = jpg.tobytes()
            time.sleep(0.05)
            continue

        # Preprocess and inference
        blob = preprocess(frame)
        net.setInput(blob)
        outputs = net.forward()
        boxes = postprocess(outputs, INPUT_SIZE)

        persons = len(boxes)
        now = time.monotonic()
        if persons > 0:
            last_seen = now

        absent = now - last_seen

        # Annotate
        for (x1, y1, x2, y2, c) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        if absent >= ABSENCE_SECONDS:
            h, w2 = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w2, 25), (0,0,255), -1)
            cv2.putText(frame, "NO PERSON ALERT", (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            fname = time.strftime("alert_%Y%m%d_%H%M%S.jpg")
            p = os.path.join(ALERT_DIR, fname)
            cv2.imwrite(p, frame)
            with status_lock:
                status["last_alert"] = p

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
        if ok:
            with proc_lock:
                latest_jpeg = jpg.tobytes()

        with status_lock:
            status["persons"] = persons
            status["absent_s"] = round(absent, 1)

        time.sleep(0.05)

app = Flask(__name__)

def mjpeg_gen():
    boundary = b"--frame"
    min_interval = 1.0 / float(STREAM_MAX_FPS)
    next_t = 0.0
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_t:
            time.sleep(next_t - now)
        with proc_lock:
            fb = latest_jpeg
        if fb:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + fb + b"\r\n"
        next_t = time.monotonic() + min_interval

@app.route("/")
def index(): return "<html><img src='/stream'></html>"

@app.route("/stream")
def stream():
    return Response(mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def stat():
    with status_lock:
        return jsonify(status)

def main():
    signal.signal(signal.SIGINT, lambda *args: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *args: stop_event.set())
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=process_thread, daemon=True)
    t1.start(); t2.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    main()

