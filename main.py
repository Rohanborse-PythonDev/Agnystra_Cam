import os
import time
import threading
import signal
from collections import deque
from flask import Flask, Response, jsonify
import cv2
import numpy as np

# Import your inference wrapper (NCNN or whichever)
# For example, if using Ultralytics wrapper for NCNN:
from ultralytics import YOLO

# -------- Config parameters --------
VIDEO_SOURCE = "rtsp://admin:admin@192.168.1.106:8554/live"
FRAME_WIDTH = 320
FRAME_SKIP = 2            # skip every 2nd frame (i.e., do inference only on alternate)
ABSENCE_SECONDS = 10.0
JPG_QUALITY = 50
ALERT_DIR = "alerts"
STREAM_FPS = 10           # target streaming fps
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Shared state
raw_lock = threading.Lock()
raw_frame = None
raw_ts = 0.0

proc_lock = threading.Lock()
latest_jpeg = None

status_lock = threading.Lock()
status = {"persons": 0, "absent_s": 0.0, "last_alert": ""}

stop_event = threading.Event()

def capture_thread():
    global raw_frame, raw_ts
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # reconnect logic or wait
            time.sleep(0.1)
            continue
        h, w = frame.shape[:2]
        if w != FRAME_WIDTH:
            scale = FRAME_WIDTH / float(w)
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h*scale)))
        with raw_lock:
            raw_frame = frame
            raw_ts = time.monotonic()

def process_thread():
    global latest_jpeg
    os.makedirs(ALERT_DIR, exist_ok=True)

    # Load NCNN or optimized model
    model = YOLO("yolov8n_ncnn_model")  # replace with your exported model path

    last_seen = time.monotonic()
    inference_count = 0

    while not stop_event.is_set():
        with raw_lock:
            if raw_frame is None:
                frame = None
                ts = None
            else:
                frame = raw_frame.copy()
                ts = raw_ts
        if frame is None:
            time.sleep(0.05)
            continue

        inference_count += 1
        persons = 0
        boxes = []

        # Skip some frames to reduce load
        if (inference_count % FRAME_SKIP) == 0:
            # run detection
            results = model.predict(source=frame,
                                    conf=CONF_THRESHOLD,
                                    iou=IOU_THRESHOLD,
                                    verbose=False)[0]
            # parse results
            if results.boxes is not None and results.boxes.xyxy.numel() > 0:
                arr = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                for (x1,y1,x2,y2), c in zip(arr, confs):
                    if c >= CONF_THRESHOLD:
                        persons += 1
                        boxes.append((int(x1), int(y1), int(x2), int(y2), float(c))

            if persons > 0:
                last_seen = time.monotonic()

        # overlay minimal
        for (x1,y1,x2,y2, c) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

        ts_text = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts_text, (5, frame.shape[0]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        absent = time.monotonic() - last_seen
        if absent >= ABSENCE_SECONDS:
            # alert
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0,0), (w, 25), (0,0,255), -1)
            cv2.putText(frame, "NO PERSON ALERT", (5,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            fname = time.strftime("alert_%Y%m%d_%H%M%S.jpg")
            fp = os.path.join(ALERT_DIR, fname)
            # could offload to another thread if disk I/O is heavy
            cv2.imwrite(fp, frame)
            with status_lock:
                status["last_alert"] = fp

        # encode jpeg
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
        if ok:
            with proc_lock:
                latest_jpeg = jpg.tobytes()

        with status_lock:
            status["persons"] = persons
            status["absent_s"] = round(absent, 1)

        # throttle loop
        time.sleep(1.0 / STREAM_FPS)

app = Flask(__name__)

def mjpeg_gen():
    boundary = b"--frame"
    while not stop_event.is_set():
        with proc_lock:
            data = latest_jpeg
        if data:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(1.0 / STREAM_FPS)

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

    t0 = threading.Thread(target=capture_thread, daemon=True)
    t1 = threading.Thread(target=process_thread, daemon=True)
    t0.start(); t1.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)

if __name__ == "__main__":
    main()
