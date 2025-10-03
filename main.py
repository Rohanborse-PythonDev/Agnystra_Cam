import os, time, platform, threading, signal
from collections import deque
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, make_response

# =========================
# Config (ENV-first)
# =========================
def _get(k, default):
    v = os.getenv(k)
    if v is None: return default
    if isinstance(default, bool):  return v.lower() in ("1","true","yes","on")
    if isinstance(default, int):   return int(v)
    if isinstance(default, float): return float(v)
    return v

MODEL_NAME       = _get("MODEL_NAME", "yolov8n.pt")
# VIDEO_SOURCE = "rtsp://admin:Digineous%40123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1"
VIDEO_SOURCE = "rtsp://admin:admin@192.168.1.106:8554/live" # "0" for webcam, or RTSP/file URL
# VIDEO_SOURCE = "0"
USE_GST          = _get("USE_GST", False)       # 1 to use GStreamer pipeline
GST_DECODER      = _get("GST_DECODER", "avdec_h264")  # e.g., avdec_h264 | nvh264dec | d3d11h264dec | msdkh264dec
GST_PROTOCOLS    = _get("GST_PROTOCOLS", "udp") # udp or tcp
GST_LATENCY_MS   = _get("GST_LATENCY_MS", 50)   # small jitterbuffer (0–100)
CONF_THR         = _get("CONF_THR", 0.45)
IOU_THR          = _get("IOU_THR", 0.50)
IMG_SIZE         = _get("IMG_SIZE", 416)        # lower than 640 for lower latency
FRAME_WIDTH      = _get("FRAME_WIDTH", 640)
FRAME_SKIP       = _get("FRAME_SKIP", 1)
ABSENCE_SECONDS  = _get("ABSENCE_SECONDS", 10.0)
DRAW_BOXES       = _get("DRAW_BOXES", True)
LIGHT_OVERLAY    = _get("LIGHT_OVERLAY", True)  # reduce text overlays to save time
JPG_QUALITY      = _get("JPG_QUALITY", 60)      # lower = faster/smaller

ALERT_DIR        = _get("ALERT_DIR", "alerts")
ALERT_ALPHA      = _get("ALERT_BANNER_ALPHA", 0.75)
EVENT_VIDEO_FPS  = _get("EVENT_VIDEO_FPS", 20)
EVENT_PRE_SEC    = _get("EVENT_PRE_SEC", 6)
EVENT_POST_SEC   = _get("EVENT_POST_SEC", 3)
FRAME_BUFFER_SEC = _get("FRAME_BUFFER_SEC", 12)

STREAM_MAX_FPS   = _get("STREAM_MAX_FPS", 25)
STREAM_SOURCE    = _get("STREAM_SOURCE", "processed")  # processed | raw
BASIC_AUTH_USER  = _get("BASIC_AUTH_USER", "")
BASIC_AUTH_PASS  = _get("BASIC_AUTH_PASS", "")
PORT             = _get("PORT", 5000)

# =========================
# Utils
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_alert_banner(frame, text):
    h, w = frame.shape[:2]
    bh = int(0.12 * h)
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, bh), (0, 0, 255), -1)
    cv2.addWeighted(ov, ALERT_ALPHA, frame, 1 - ALERT_ALPHA, 0, frame)
    f = cv2.FONT_HERSHEY_SIMPLEX; s = 0.8; t = 2
    (tw, th), _ = cv2.getTextSize(text, f, s, t)
    x = max(10, (w - tw) // 2); y = bh//2 + th//2
    cv2.putText(frame, text, (x, y), f, s, (0, 0, 0), t+2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), f, s, (255,255,255), t, cv2.LINE_AA)

def put_timestamp(frame, prefix=""):
    if LIGHT_OVERLAY:  # smaller, faster overlay
        ts = time.strftime("%H:%M:%S")
        msg = ts if not prefix else f"{prefix}{ts}"
        h = frame.shape[0]; f = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, msg, (8, h-8), f, 0.55, (255,255,255), 1, cv2.LINE_AA)
        return ts
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = ts if not prefix else f"{prefix}{ts}"
    h = frame.shape[0]; f = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, msg, (10, h-10), f, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, msg, (10, h-10), f, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return ts

def save_event_video(frames_with_ts, out_path, frame_size, fps, start_ts, end_ts):
    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    selected = [f for (f, t) in frames_with_ts if start_ts <= t <= end_ts]
    if not selected:
        vw.release(); return False
    duration = max(0.1, end_ts - start_ts)
    target = int(max(1, duration * fps))
    if len(selected) > target:
        step = len(selected) / float(target)
        idxs = [int(i * step) for i in range(target)]
        selected = [selected[i] for i in idxs]
    for fr in selected:
        if fr.shape[1] != w or fr.shape[0] != h:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        vw.write(fr)
    vw.release(); return True

def drain_latest(cap, max_grabs=5) -> Optional[np.ndarray]:
    grabbed = False
    for _ in range(max_grabs - 1):
        ok = cap.grab()
        if not ok: break
        grabbed = True
    if grabbed:
        ok, frame = cap.retrieve()
        return frame if ok else None
    return None

# =========================
# Camera (low-latency)
# =========================
class Camera:
    def __init__(self, src: Union[int,str], width: int):
        self.src = int(src) if (isinstance(src,str) and src.isdigit()) else src
        self.width = width
        self.cap = None
        self._open()

    def _gst_pipeline(self, url: str) -> str:
        # Hardware decoders (set via GST_DECODER env) if installed:
        #   - nvh264dec (NVIDIA), d3d11h264dec (Windows DXVA), msdkh264dec (Intel QSV)
        # Fallback: avdec_h264 (software)
        return (
            f'rtspsrc location="{url}" latency={int(GST_LATENCY_MS)} protocols={GST_PROTOCOLS} ! '
            f'rtph264depay ! h264parse ! {GST_DECODER} ! videoconvert ! '
            f'appsink drop=1 max-buffers=1 sync=false'
        )

    def _open(self):
        if USE_GST and not isinstance(self.src, int):
            pipeline = self._gst_pipeline(self.src)
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            if isinstance(self.src, int):
                backend = cv2.CAP_DSHOW if platform.system().lower().startswith('win') else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.src, backend)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            else:
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[np.ndarray]:
        if not self.cap or not self.cap.isOpened(): return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def read_low_latency(self):
        """Use appsink/drop (GST) or drain frames (FFmpeg) to reduce lag."""
        if not self.cap or not self.cap.isOpened():
            return None

        if USE_GST and not isinstance(self.src, int):
            # For GST appsink, .read() already gives the latest frame (drop=1)
            ok, frame = self.cap.read()
            return frame if ok else None

        # FFmpeg fallback: drain a few and return the freshest
        frame = drain_latest(self.cap, max_grabs=5)
        if frame is None:
            frame = self.read()
        return frame

    def reconnect(self, backoff_s=0.3, max_s=4.0):
        try:
            if self.cap: self.cap.release()
        except: pass
        t = backoff_s
        while True:
            try:
                self._open()
                if self.cap.isOpened(): return
            except: pass
            time.sleep(t); t = min(max_s, t*1.7)

    def release(self):
        try:
            if self.cap: self.cap.release()
        except: pass

# =========================
# Shared state (thread-safe)
# =========================
stop_event = threading.Event()

raw_lock = threading.Lock()
raw_frame: Optional[np.ndarray] = None
raw_ts: float = 0.0

proc_lock = threading.Lock()
latest_jpeg: Optional[bytes] = None

status_lock = threading.Lock()
status_dict = {
    "device": "", "fps": 0.0, "absent_s": 0.0, "persons": 0,
    "model": MODEL_NAME, "last_alert": "", "stream_source": STREAM_SOURCE
}

# =========================
# Threads: capture, process
# =========================
def capture_thread(cam: Camera):
    global raw_frame, raw_ts
    while not stop_event.is_set():
        frame = cam.read_low_latency()
        if frame is None:
            cam.reconnect(); continue
        # Resize here once
        h, w = frame.shape[:2]
        if w != FRAME_WIDTH:
            scale = FRAME_WIDTH / float(w)
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h*scale)), interpolation=cv2.INTER_LINEAR)
        with raw_lock:
            raw_frame = frame
            raw_ts = time.monotonic()

def process_thread():
    global latest_jpeg
    ensure_dir(ALERT_DIR)
    cv2.setUseOptimized(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = (device == "cuda")
    model = YOLO(MODEL_NAME); model.fuse()

    last_seen = time.monotonic()
    alert_fired = False
    frame_idx = 0

    fps_t0 = time.monotonic()
    fps_counter = 0
    fps_value = 0.0

    # Keep raw frames for event video (lighter than annotated)
    frame_buffer: deque[Tuple[np.ndarray, float]] = deque()
    post_collect_until: Optional[float] = None

    while not stop_event.is_set():
        with raw_lock:
            frame = None if raw_frame is None else raw_frame.copy()
            ts_mono = raw_ts
        if frame is None:
            time.sleep(0.005); continue

        frame_idx += 1
        persons = 0
        det_boxes = []

        if frame_idx % FRAME_SKIP == 0:
            try:
                r0 = model.predict(
                    source=frame,
                    imgsz=IMG_SIZE,
                    conf=CONF_THR,
                    iou=IOU_THR,
                    device=device,
                    half=use_half,
                    classes=[0],
                    verbose=False
                )[0]
                if r0.boxes is not None and r0.boxes.xyxy.numel() > 0:
                    cls = r0.boxes.cls.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    for (x1,y1,x2,y2), c, p in zip(xyxy, conf, cls):
                        if c >= CONF_THR:
                            persons += 1
                            x1 = int(max(0, min(x1, frame.shape[1]-1)))
                            y1 = int(max(0, min(y1, frame.shape[0]-1)))
                            x2 = int(max(0, min(x2, frame.shape[1]-1)))
                            y2 = int(max(0, min(y2, frame.shape[0]-1)))
                            det_boxes.append((x1,y1,x2,y2,float(c)))
            except Exception:
                persons = 0; det_boxes = []

            now = time.monotonic()
            if persons > 0:
                last_seen = now
                alert_fired = False

        # annotate minimally (faster)
        if DRAW_BOXES and det_boxes:
            for (x1,y1,x2,y2,c) in det_boxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # FPS calc
        fps_counter += 1
        now = time.monotonic()
        if now - fps_t0 >= 1.0:
            fps_value = fps_counter / (now - fps_t0)
            fps_counter = 0; fps_t0 = now

        absent = now - last_seen
        if not LIGHT_OVERLAY:
            status = f"P:{persons} Absent:{absent:0.1f}s FPS:{fps_value:0.1f} {MODEL_NAME}"
            cv2.putText(frame, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

        # Alert & snapshot
        triggered = (absent >= ABSENCE_SECONDS)
        if triggered:
            draw_alert_banner(frame, f"NO PERSON {int(ABSENCE_SECONDS)}s – ALERT")
            if not alert_fired:
                post_collect_until = now + EVENT_POST_SEC
                alert_fired = True
                put_timestamp(frame, "Captured: ")
                fname_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                snap_path = os.path.join(ALERT_DIR, f"alert_{fname_ts}.jpg")
                cv2.imwrite(snap_path, frame)
                with status_lock: status_dict["last_alert"] = snap_path
        else:
            put_timestamp(frame)

        # buffer raw frames for event video
        frame_buffer.append((frame.copy(), ts_mono))
        cutoff = now - FRAME_BUFFER_SEC
        while frame_buffer and frame_buffer[0][1] < cutoff:
            frame_buffer.popleft()

        if post_collect_until and now >= post_collect_until:
            start_ts = max(0, now - (EVENT_PRE_SEC + EVENT_POST_SEC))
            end_ts   = now
            out_w, out_h = frame.shape[1], frame.shape[0]
            fname_ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            vid_path = os.path.join(ALERT_DIR, f"alert_{fname_ts}.mp4")
            saved = save_event_video(list(frame_buffer), vid_path, (out_w,out_h),
                                     EVENT_VIDEO_FPS or fps_value or 20, start_ts, end_ts)
            print(f"[ALERT] Event video {'saved' if saved else 'not saved'}: {vid_path}")
            post_collect_until = None

        # produce JPEG for streaming (processed path)
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
        if ok:
            with proc_lock:
                latest_jpeg = jpg.tobytes()

        with status_lock:
            status_dict.update({
                "device": device, "fps": round(fps_value,1),
                "absent_s": round(absent,1), "persons": persons
            })

# =========================
# Flask app (stream/status)
# =========================
app = Flask(__name__)

INDEX_HTML = """<!doctype html>
<html><head><title>No Person in Frame for 10Sec Alert </title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
body{background:#111;color:#eee;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:960px;margin:24px auto;padding:0 12px}
img{width:100%;height:auto;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.5)}
code{background:#222;padding:2px 6px;border-radius:6px}
</style></head>
<body><div class="wrap">
<h2>No Person in Frame for 10Sec Alert</h2>

<img src="/stream" alt="Live stream">
</div></body></html>
"""

def _auth_ok():
    if not BASIC_AUTH_USER: return True
    a = request.authorization
    return a and a.username==BASIC_AUTH_USER and a.password==BASIC_AUTH_PASS

def _no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

def _encode_raw_to_jpeg() -> Optional[bytes]:
    with raw_lock:
        rf = None if raw_frame is None else raw_frame.copy()
    if rf is None: return None
    ok, jpg = cv2.imencode(".jpg", rf, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    return jpg.tobytes() if ok else None

def mjpeg_generator():
    boundary = b"--frame"
    min_interval = 1.0 / max(1, STREAM_MAX_FPS)
    next_t = 0.0
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_t:
            time.sleep(max(0, next_t - now)); continue

        if STREAM_SOURCE == "raw":
            frame_bytes = _encode_raw_to_jpeg()
        else:
            with proc_lock:
                frame_bytes = None if latest_jpeg is None else bytes(latest_jpeg)

        if frame_bytes is None:
            time.sleep(0.005); continue

        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        next_t = time.monotonic() + min_interval

@app.route("/")
def index():
    if not _auth_ok():
        return _no_cache(Response("Auth required", 401, {"WWW-Authenticate":"Basic"}))
    return _no_cache(make_response(INDEX_HTML))

@app.route("/stream")
def stream():
    if not _auth_ok():
        return _no_cache(Response("Auth required", 401, {"WWW-Authenticate":"Basic"}))
    resp = Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame", direct_passthrough=True)
    return _no_cache(resp)

@app.route("/status")
def status():
    if not _auth_ok():
        return _no_cache(Response("Auth required", 401, {"WWW-Authenticate":"Basic"}))
    with status_lock:
        return _no_cache(jsonify(status_dict))

@app.route("/healthz")
def healthz():
    return _no_cache(Response("ok", 200))

def _handle_stop(*_): stop_event.set()

def main():
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    cam = Camera(VIDEO_SOURCE, FRAME_WIDTH)
    t_cap = threading.Thread(target=capture_thread, args=(cam,), daemon=True)
    t_cap.start()

    t_proc = threading.Thread(target=process_thread, daemon=True)
    t_proc.start()

    try:
        app.run(host="0.0.0.0", port=int(PORT), threaded=True)
    finally:
        stop_event.set()
        t_cap.join(timeout=2)
        t_proc.join(timeout=2)
        cam.release()

if __name__ == "__main__":
    main()
