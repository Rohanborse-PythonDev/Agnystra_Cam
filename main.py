import os
import time
import threading
import signal
from collections import deque
from flask import Flask, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

# ========== Configuration ==========

VIDEO_SOURCE = "rtsp://admin:admin@192.168.1.106:8554/live"  # or "0" for webcam
FRAME_WIDTH = 320
FRAME_SKIP = 2
CONF_THR = 0.45
IOU_THR = 0.5
ABSENCE_SECONDS = 10.0
JPG_QUALITY = 50
ALERT_DIR = "alerts"
STREAM_MAX_FPS = 10

# Shared state
raw_lock = threading.Lock()
raw_frame = None
raw_ts = 0.0

proc_lock = threading.Lock()
latest_jpeg = None

status_lock = threading.Lock()
status = {
    "persons": 0,
    "absent_s": 0.0,
    "last_alert": ""
}

stop_event = threading.Event()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

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

    # Try to load the NCNN model (Ultralytics wrapper should pick NCNN backend)
    try:
        model = YOLO("models/your_model_ncnn_model")
    except Exception as e:
        print("Failed to load NCNN model via YOLO:", e)
        print("Falling back to default YOLO model (slower).")
        model = None

    last_seen = time.monotonic()
    frame_count = 0

    while not stop_event.is_set():
        with raw_lock:
            if raw_frame is None:
                frame = None
            else:
                frame = raw_frame.copy()
                ts = raw_ts
        if frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        # Skip detection on some frames to reduce load
        if frame_count % FRAME_SKIP != 0:
            ts_text = time.strftime("%H:%M:%S")
            cv2.putText(frame, ts_text, (5, frame.shape[0] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
            if ok:
                with proc_lock:
                    latest_jpeg = jpg.tobytes()
            time.sleep(0.05)
            continue

        # Run detection
        persons = 0
        dets = []
        if model is not None:
            try:
                results = model.predict(source=frame, conf=CONF_THR, iou=IOU_THR, verbose=False)
                r = results[0]
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                else:
                    boxes = []
                    confs = []
                for (xy, c) in zip(boxes, confs):
                    if c >= CONF_THR:
                        persons += 1
                        x1, y1, x2, y2 = map(int, xy)
                        dets.append((x1, y1, x2, y2, float(c)))
            except Exception as e:
                print("Detection error:", e)

        now = time.monotonic()
        if persons > 0:
            last_seen = now

        absent = now - last_seen

        # Annotate frame
        for (x1, y1, x2, y2, c) in dets:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 1)
        ts_text = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts_text, (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        if absent >= ABSENCE_SECONDS:
            h, w2 = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w2, 25), (0,0,255), -1)
            cv2.putText(frame, "NO PERSON ALERT", (3, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
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
    min_interval = 1.0 / float(max(1, STREAM_MAX_FPS))
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
def index():
    return "<html><img src='/stream'></html>"

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
    t1.start()
    t2.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    main()
