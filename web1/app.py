import os
import sys
import cv2
import time
import threading
import sqlite3
import torch
import numpy as np
from datetime import datetime
from collections import deque
from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from ultralytics import YOLO

# =============================================================
# 0. CONFIGURATION & SETUP
# =============================================================

# Suppress FFmpeg mjpeg warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

app = Flask(__name__)

# Base project directory (parent of web1/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTION_FOLDER = os.path.join(BASE_DIR, "detections")
DB_PATH = os.path.join(DETECTION_FOLDER, "alerts.db")
WEAPON_MODEL_PATH = os.path.join(BASE_DIR, "models/thebest.onnx")

os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Add parent directory to sys.path so we can import from inference.py
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import (
    load_models as load_anomaly_models,
    frames_to_tensor,
    get_human_probability_from_cache,
    FRAME_SIZE,
    NUM_FRAMES,
    USE_FP16,
    LABEL_NORMAL,
    LABEL_ANOMALY,
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME,
                  video_path TEXT,
                  image_path TEXT,
                  confidence FLOAT,
                  type TEXT,
                  location TEXT,
                  status TEXT)''')
    conn.commit()
    conn.close()

init_db()


class GlobalState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        # Weapon detection state
        self.weapon_detected = False
        self.weapon_boxes = []  # List of (x1,y1,x2,y2,conf) for overlays
        self.consecutive_frames = 0
        self.is_recording = False
        self.last_alert_time = 0
        self.current_video_path = ""
        self.fps = 0.0
        self.trigger_frames = 6
        self.cooldown = 60
        self.cooldown_enabled = True  # Toggle for alert cooldown
        self.weapon_threshold = 0.65
        self.best_weapon_snapshot = None  # Frame with bounding boxes drawn by weapon worker
        # Anomaly detection state
        self.anomaly_detected = False
        self.anomaly_score = 0.0
        self.anomaly_window_id = 0
        self.anomaly_threshold = 0.9
        self.anomaly_enabled = True  # Toggle for anomaly detection
        self.anomaly_verifier_state = "IDLE"
        self.anomaly_verifier_progress = ""
        # General
        self.url = "0"
        self.location = "Main Entrance"
        self.max_conf_seen = 0.0
        # Combined threat flag
        self.threat_active = False
        self.threat_type = ""
        self.frame_counter = 0

state = GlobalState()


# =============================================================
# 1. ZERO-LATENCY FRAME READER
# =============================================================

class ZeroLatencyReader:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

        if self.cap.isOpened():
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def isOpened(self):
        return self.cap.isOpened()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[VideoStream] Stream dead, reconnecting...")
                self.cap.release()
                time.sleep(2.0)
                self.cap = cv2.VideoCapture(self.url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            with self.lock:
                self.ret = ret
                self.frame = frame

            self.new_frame_event.set()

    def read(self):
        self.new_frame_event.wait(timeout=2.0)
        with self.lock:
            self.new_frame_event.clear()
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def stop(self):
        self.running = False


# =============================================================
# 2. ANOMALY SLIDING WINDOW ENGINE (matches live_inference.py)
# =============================================================

def _compute_motion_energy(frames):
    """Measure how much rapid motion exists across a window.
    Low motion = people standing/walking normally.
    High motion = fighting, running, violent gestures.
    Returns a value between 0.0 and 1.0."""
    if len(frames) < 4:
        return 0.0
    # Sample ~8 evenly spaced pairs for efficiency
    sample_ids = np.linspace(0, len(frames) - 2, min(8, len(frames) - 1), dtype=int)
    diffs = []
    for i in sample_ids:
        f1 = frames[i].astype(np.float32)
        f2 = frames[i + 1].astype(np.float32)
        diff = np.abs(f2 - f1).mean() / 255.0  # normalised 0..1
        diffs.append(diff)
    return float(np.mean(diffs))


class SlidingWindowEngine:
    """Accumulates frames and runs anomaly inference in a background thread."""

    def __init__(self, video_model, human_model, device,
                 threshold=0.9, window_sec=4.0, stride_sec=1.0,
                 num_clips=10, stream_fps=30.0):
        self.video_model = video_model
        self.human_model = human_model
        self.device = device
        self.threshold = threshold
        self.num_clips = num_clips
        self.use_half = USE_FP16 and device.type == "cuda"

        self.window_frames = int(window_sec * stream_fps)
        self.stride_frames = int(stride_sec * stream_fps)

        self.buffer = deque(maxlen=self.window_frames)
        self.new_frame_count = 0

        self.prediction = LABEL_NORMAL
        self.score = 0.0
        self.window_id = 0
        self.result_lock = threading.Lock()
        self._busy = False

    def feed_frame(self, bgr_frame):
        small = cv2.resize(bgr_frame, FRAME_SIZE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.buffer.append(rgb)
        self.new_frame_count += 1

    def should_process(self):
        if self._busy or len(self.buffer) < self.window_frames:
            return False
        return self.new_frame_count >= self.stride_frames

    def start_processing(self):
        if self._busy:
            return
        self._busy = True
        self.new_frame_count = 0
        window_frames = list(self.buffer)
        wid = self.window_id + 1
        t = threading.Thread(target=self._process_window,
                             args=(window_frames, wid), daemon=True)
        t.start()

    @torch.no_grad()
    def _process_window(self, frames, window_id):
        try:
            n = len(frames)

            # ── 1. Human Presence Gate ────────────────────────
            p_human = 1.0
            if self.human_model is not None:
                ids = np.linspace(0, n - 1, min(5, n), dtype=int)
                human_frames = [frames[i] for i in ids]
                p_human = get_human_probability_from_cache(
                    human_frames, self.human_model, self.device)

            # ── 2. Motion Energy Check ────────────────────────
            # Prevents false positives when people just stand/walk/sit.
            # Only violent or rapid motion should trigger anomaly.
            motion = _compute_motion_energy(frames)
            # motion < 0.02 = near-static scene (standing, talking)
            # motion 0.02-0.05 = normal walking / gentle movement
            # motion > 0.05 = rapid / violent motion
            MOTION_FLOOR = 0.025

            if motion < MOTION_FLOOR:
                # Near-static or gentle movement — skip expensive CNN
                with self.result_lock:
                    self.prediction = LABEL_NORMAL
                    self.score = 0.0
                    self.window_id = window_id
                print(f"  [Anomaly] 🟢 Window #{window_id:>3d}  NORMAL    "
                      f"(low motion={motion:.4f}, human={p_human:.2f})")
                return

            # ── 3. Violence Detection (dense 1.2s subclips) ────────────────────────
            window_frac = 0.3
            clip_stride_frac = 0.7 / max(1, self.num_clips - 1)
            clip_tensors = []

            for i in range(self.num_clips):
                s_frac = i * clip_stride_frac
                e_frac = min(s_frac + window_frac, 1.0)
                s_idx = int(s_frac * (n - 1))
                e_idx = int(e_frac * (n - 1))
                if e_idx <= s_idx:
                    e_idx = min(s_idx + NUM_FRAMES, n - 1)
                ids = np.linspace(s_idx, e_idx, NUM_FRAMES, dtype=int)
                clip = [frames[j] for j in ids]
                tensor = frames_to_tensor(clip, use_half=self.use_half)
                clip_tensors.append(tensor)

            batch = torch.cat(clip_tensors, dim=0).to(self.device)
            scores = self.video_model(batch).squeeze(-1)
            clip_scores = scores.cpu().tolist()
            p_video = float(np.median(clip_scores))

            # ── 4. Fusion Logic ────────────────────────
            # If no human detected at all, suppress the score
            if p_human < 0.2:
                p_video *= p_human

            final_score = p_video
            prediction = LABEL_ANOMALY if final_score > self.threshold else LABEL_NORMAL

            with self.result_lock:
                self.prediction = prediction
                self.score = final_score
                self.window_id = window_id

            icon = "🔴" if prediction == LABEL_ANOMALY else "🟢"
            print(f"  [Anomaly] {icon} Window #{window_id:>3d}  {prediction:8s}  "
                  f"score={final_score:.4f}  (motion={motion:.4f}, human={p_human:.2f})")

        except Exception as e:
            print(f"  [Anomaly] ⚠️ Window #{window_id} error: {e}")
        finally:
            self._busy = False

    def get_result(self):
        with self.result_lock:
            return self.prediction, self.score, self.window_id


# =============================================================
# 3. ANOMALY VERIFIER (from live_inference.py)
# =============================================================

class AnomalyVerifier:
    """Collects multiple window scores to verify anomaly before saving."""

    def __init__(self, fps, window_frames, num_windows=5, threshold=0.9):
        self.fps = fps
        self.num_windows = num_windows
        self.threshold = threshold
        self.state = "IDLE"
        self.collected_scores = []
        self.recorded_frames = []
        self.target_windows_left = 0
        self.last_processed_wid = -1
        self.rolling_frames = deque(maxlen=int(window_frames))

    def process_frame(self, frame):
        self.rolling_frames.append(frame)
        if self.state == "COLLECTING":
            self.recorded_frames.append(frame)

    def update_result(self, wid, score):
        if wid <= self.last_processed_wid:
            return
        self.last_processed_wid = wid

        if self.state == "IDLE":
            if score > self.threshold:
                self.state = "COLLECTING"
                self.collected_scores = [score]
                self.recorded_frames = list(self.rolling_frames)
                self.target_windows_left = self.num_windows
                print(f"\n  [Verifier] ⚠️ Anomaly trigger at window {wid} "
                      f"(score {score:.4f}), verifying over next "
                      f"{self.num_windows} windows...")

        elif self.state == "COLLECTING":
            self.collected_scores.append(score)
            self.target_windows_left -= 1
            print(f"  [Verifier] Collected window {wid} score: {score:.4f} "
                  f"({self.target_windows_left} left)")

            if self.target_windows_left <= 0:
                median_score = float(np.median(self.collected_scores))
                if median_score > self.threshold:
                    print(f"  [Verifier] 🚨 VERIFIED! Median {median_score:.4f}")
                    t = threading.Thread(
                        target=self._save_and_record,
                        args=(list(self.recorded_frames), median_score),
                        daemon=True)
                    t.start()
                else:
                    print(f"  [Verifier] ❌ FALSE ALARM. Median {median_score:.4f}")
                self.state = "IDLE"
                self.recorded_frames.clear()
                self.collected_scores.clear()

    def _save_and_record(self, frames_to_save, score):
        if not frames_to_save:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vname = f"anomaly_{ts}.mp4"
        vpath = os.path.join(DETECTION_FOLDER, vname)

        h, w = frames_to_save[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(vpath, fourcc, float(self.fps), (w, h))
        for f in frames_to_save:
            out.write(f)
        out.release()

        # Save snapshot
        img_name = vname.replace(".mp4", ".jpg")
        img_path = os.path.join(DETECTION_FOLDER, img_name)
        cv2.imwrite(img_path, frames_to_save[len(frames_to_save) // 2])

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO alerts (timestamp, video_path, image_path, "
            "confidence, type, location, status) VALUES (?,?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             vname, img_name, round(score, 2),
             "ANOMALY", state.location, "New"))
        conn.commit()
        conn.close()
        print(f"  [Verifier] 💾 Anomaly saved: {vpath}")


# =============================================================
# 4. CAMERA THREAD (Weapon + Anomaly Detection)
# =============================================================

class CameraThread(threading.Thread):
    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.running = True

        # Load weapon model (YOLO)
        print(f"[Models] Loading YOLO weapon model from {WEAPON_MODEL_PATH}")
        self.weapon_model = YOLO(WEAPON_MODEL_PATH, task='detect')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load anomaly models (video violence + human detector)
        anomaly_video_path = os.path.join(BASE_DIR, "models/video_violence_detector.pt")
        anomaly_human_path = os.path.join(BASE_DIR, "models/human_detector.pt")
        print(f"[Models] Loading anomaly models...")
        self.anomaly_video_model, self.anomaly_human_model = load_anomaly_models(
            anomaly_video_path, anomaly_human_path,
            torch.device(self.device))

    def run(self):
        reader = ZeroLatencyReader(self.url)
        if not reader.isOpened():
            print("[Error] Failed to open camera.")
            return

        w, h = reader.w, reader.h
        fps_cap = reader.fps

        # Weapon recording state
        pre_roll_buffer = deque(maxlen=int(fps_cap * 2))
        video_writer = None
        max_event_conf = 0.0
        weapon_alert_type = ""

        # Anomaly engine
        # instantiate anomaly engine and verifier; processing will occur in background worker
        anomaly_engine = SlidingWindowEngine(
            self.anomaly_video_model, self.anomaly_human_model,
            torch.device(self.device),
            threshold=state.anomaly_threshold,
            window_sec=4.0, stride_sec=1.0, num_clips=10,
            stream_fps=fps_cap)

        anomaly_verifier = AnomalyVerifier(
            fps=fps_cap,
            window_frames=int(fps_cap * 4.0),
            num_windows=5,
            threshold=state.anomaly_threshold)

        # buffer for frames to analyze asynchronously
        anomaly_queue = deque()
        weapon_queue = deque(maxlen=1)  # Only process latest frame to reduce lag

        def anomaly_worker():
            while self.running:
                # Skip anomaly processing if disabled
                if not state.anomaly_enabled:
                    anomaly_queue.clear()
                    state.anomaly_detected = False
                    state.anomaly_score = 0.0
                    state.anomaly_verifier_state = "IDLE"
                    state.anomaly_verifier_progress = ""
                    time.sleep(0.1)
                    continue

                if anomaly_queue:
                    f = anomaly_queue.popleft()
                    # feed frames into engine without blocking camera thread
                    anomaly_engine.feed_frame(f)
                    anomaly_verifier.process_frame(f)
                    if anomaly_engine.should_process():
                        anomaly_engine.start_processing()
                    a_pred, a_score, a_wid = anomaly_engine.get_result()
                    anomaly_is_active = (a_pred == LABEL_ANOMALY)
                    if a_wid > anomaly_verifier.last_processed_wid:
                        anomaly_verifier.update_result(a_wid, a_score)
                    
                    # update state for backend logs
                    # Keep threat active on UI while verifying
                    is_verifying = (anomaly_verifier.state == "COLLECTING")
                    state.anomaly_detected = anomaly_is_active or is_verifying
                    state.anomaly_score = a_score
                    state.anomaly_window_id = a_wid
                    state.anomaly_verifier_state = anomaly_verifier.state
                    
                    if is_verifying:
                        done = anomaly_verifier.num_windows - anomaly_verifier.target_windows_left
                        state.anomaly_verifier_progress = f"{done}/{anomaly_verifier.num_windows}"
                    else:
                        state.anomaly_verifier_progress = ""
                else:
                    time.sleep(0.01)

        def weapon_worker():
            while self.running:
                if weapon_queue:
                    frame = weapon_queue.popleft()
                    # Perform YOLO detection
                    resized = cv2.resize(frame, (768, 768))
                    try:
                        results = self.weapon_model.predict(
                            source=resized, device=self.device,
                            verbose=False, imgsz=(768, 768))
                    except Exception as e:
                        print(f"[Weapon Inference Error] {e}")
                        results = []

                    weapon_found = False
                    frame_max_conf = 0.0
                    boxes = []

                    for r in results:
                        for box in r.boxes:
                            conf = float(box.conf[0])
                            frame_max_conf = max(frame_max_conf, conf)
                            if conf < state.weapon_threshold:
                                continue
                            weapon_found = True
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1 = int(x1 * w / 768)
                            y1 = int(y1 * h / 768)
                            x2 = int(x2 * w / 768)
                            y2 = int(y2 * h / 768)
                            boxes.append((x1, y1, x2, y2, conf))

                    # If weapon found, draw boxes on THIS analyzed frame
                    # and keep it as the best snapshot candidate
                    if weapon_found:
                        annotated = frame.copy()
                        for bx1, by1, bx2, by2, bconf in boxes:
                            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 70, 255), 3)
                            lbl = f"WEAPON {bconf:.2f}"
                            (tw2, th2), bl2 = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(annotated, (bx1, by1 - th2 - bl2 - 10), (bx1 + tw2, by1), (0, 70, 255), -1)
                            cv2.putText(annotated, lbl, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        with state.lock:
                            if frame_max_conf >= state.max_conf_seen or state.best_weapon_snapshot is None:
                                state.best_weapon_snapshot = annotated

                    # Update state
                    with state.lock:
                        state.weapon_detected = weapon_found
                        state.weapon_boxes = boxes
                        state.max_conf_seen = frame_max_conf
                else:
                    time.sleep(0.01)

        worker_thread = threading.Thread(target=anomaly_worker, daemon=True)
        worker_thread.start()
        weapon_thread = threading.Thread(target=weapon_worker, daemon=True)
        weapon_thread.start()

        frame_count = 0
        timer = time.time()
        weapon_frame_count = 0

        while self.running:
            ret, frame = reader.read()
            if not ret or frame is None:
                continue

            clean_frame = frame.copy()
            now = time.time()

            # Queue frame for weapon detection (every 3rd frame to reduce load)
            weapon_frame_count += 1
            if weapon_frame_count % 3 == 0 and len(weapon_queue) < 1:
                weapon_queue.append(clean_frame)

            # Queue frame for anomaly processing
            if len(anomaly_queue) < 200:
                anomaly_queue.append(clean_frame)

            # Draw weapon overlays using latest detection results
            with state.lock:
                weapon_found = state.weapon_detected
                boxes = state.weapon_boxes.copy()
                frame_max_conf = state.max_conf_seen

            for x1, y1, x2, y2, conf in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 70, 255), 3)
                label = f"WEAPON {conf:.2f}"
                (tw, th_t), bl = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th_t - bl - 10),
                              (x1 + tw, y1), (0, 70, 255), -1)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            # ── COMBINED THREAT LOGIC ────────────────────────
            # Use the latest anomaly state (updated asynchronously) to avoid blocking the camera loop.
            anomaly_is_active = state.anomaly_detected and state.anomaly_enabled
            threat_active = weapon_found or anomaly_is_active
            if weapon_found and anomaly_is_active:
                threat_type = "WEAPON + ANOMALY"
            elif weapon_found:
                threat_type = "WEAPON"
            elif anomaly_is_active:
                threat_type = "ANOMALY"
            else:
                threat_type = ""

            state.threat_active = threat_active
            state.threat_type = threat_type

            # ── WEAPON ALERT & RECORDING ─────────────────────
            if weapon_found:
                state.consecutive_frames += 1
                max_event_conf = max(max_event_conf, frame_max_conf)
                if (state.consecutive_frames >= state.trigger_frames
                        and not state.is_recording):
                    if (not state.cooldown_enabled) or (now - state.last_alert_time) >= state.cooldown:
                        state.is_recording = True
                        state.last_alert_time = now
                        weapon_alert_type = threat_type
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        vname = f"alert_{ts}.mp4"
                        vpath = os.path.join(DETECTION_FOLDER, vname)
                        state.current_video_path = vpath
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(
                            vpath, fourcc, fps_cap, (w, h))
                        for f in pre_roll_buffer:
                            video_writer.write(f)

            if state.is_recording:
                if video_writer:
                    video_writer.write(frame)
                if not weapon_found and (now - state.last_alert_time) >= 5.0:
                    video_writer.release()
                    video_writer = None
                    state.is_recording = False
                    
                    # Use the snapshot from weapon_worker (has correct bounding boxes)
                    with state.lock:
                        frame_to_save = state.best_weapon_snapshot if state.best_weapon_snapshot is not None else frame
                        state.best_weapon_snapshot = None
                    save_alert_to_db(
                        os.path.basename(state.current_video_path),
                        max_event_conf, frame_to_save, weapon_alert_type or "WEAPON")
                        
                    max_event_conf = 0.0
                    weapon_alert_type = ""

            if not weapon_found:
                state.consecutive_frames = 0
                pre_roll_buffer.append(clean_frame)

            # FPS tracking
            frame_count += 1
            if now - timer >= 1.0:
                state.fps = frame_count / (now - timer)
                print(f"[Camera] FPS: {state.fps:.1f}")
                frame_count = 0
                timer = now

            with state.lock:
                state.frame = frame
                state.frame_counter += 1


def save_alert_to_db(vname, conf, frame, alert_type="WEAPON"):
    img_name = vname.replace(".mp4", ".jpg")
    img_path = os.path.join(DETECTION_FOLDER, img_name)
    cv2.imwrite(img_path, frame)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO alerts (timestamp, video_path, image_path, "
        "confidence, type, location, status) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         vname, img_name, round(conf, 2),
         alert_type, state.location, "New"))
    conn.commit()
    conn.close()


# =============================================================
# 5. FLASK ROUTES
# =============================================================

@app.route('/')
def dashboard():
    alerts = get_recent_alerts(5)
    return render_template('dashboard.html', alerts=alerts, stats=get_stats())

@app.route('/alerts')
def alerts_history():
    alerts = get_all_alerts()
    return render_template('alerts.html', alerts=alerts)

@app.route('/alert/<int:id>')
def alert_detail(id):
    alert = get_alert_by_id(id)
    if alert:
        return render_template('detail.html', alert=alert)
    return "Not found", 404

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections/<path:filename>')
def serve_detections(filename):
    return send_from_directory(DETECTION_FOLDER, filename)

@app.route('/api/stats')
def api_stats():
    stats = get_stats()
    return jsonify({
        'status': 'ONLINE',
        'weapon_detected': state.weapon_detected,
        'anomaly_detected': state.anomaly_detected,
        'anomaly_score': round(state.anomaly_score, 4),
        'anomaly_window': state.anomaly_window_id,
        'anomaly_verifier': state.anomaly_verifier_state,
        'anomaly_progress': state.anomaly_verifier_progress,
        'threat_active': state.threat_active,
        'threat_type': state.threat_type,
        'fps': round(state.fps, 1),
        'recording': state.is_recording,
        'total_alerts': stats['total_alerts'],
        'unresolved': stats['unresolved'],
        'weapon_alerts': stats.get('weapon_alerts', 0),
        'anomaly_alerts': stats.get('anomaly_alerts', 0),
        'location': state.location,
        'max_conf': round(state.max_conf_seen, 2),
        'anomaly_enabled': state.anomaly_enabled,
        'cooldown_enabled': state.cooldown_enabled,
    })

@app.route('/api/resolve/<int:id>', methods=['POST'])
def resolve_alert(id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE alerts SET status = 'Resolved' WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/delete/<int:id>', methods=['POST'])
def delete_alert(id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM alerts WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/toggle_anomaly', methods=['POST'])
def toggle_anomaly():
    state.anomaly_enabled = not state.anomaly_enabled
    status = 'ON' if state.anomaly_enabled else 'OFF'
    print(f"[Toggle] Anomaly detection is now {status}")
    return jsonify({'success': True, 'anomaly_enabled': state.anomaly_enabled})

@app.route('/api/toggle_cooldown', methods=['POST'])
def toggle_cooldown():
    state.cooldown_enabled = not state.cooldown_enabled
    status = 'ON' if state.cooldown_enabled else 'OFF'
    print(f"[Toggle] Alert cooldown is now {status}")
    return jsonify({'success': True, 'cooldown_enabled': state.cooldown_enabled})

@app.route('/api/delete_all', methods=['POST'])
def delete_all_alerts():
    try:
        # 1. Clear database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM alerts")
        conn.commit()
        conn.close()

        # 2. Clear physical files (.mp4 and .jpg) to save space
        import glob
        for ext in ['*.mp4', '*.jpg']:
            files = glob.glob(os.path.join(DETECTION_FOLDER, ext))
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error removing {f}: {e}")

        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting all alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recent_alerts')
def api_recent_alerts():
    """Return the 5 most recent alerts as JSON for live dashboard refresh."""
    alerts = get_recent_alerts(5)
    return jsonify(alerts)


def generate_frames():
    last_frame_counter = -1
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    while True:
        with state.lock:
            frame = state.frame
            counter = state.frame_counter

        if frame is None:
            time.sleep(0.05)
            continue
            
        # Avoid re-encoding the exact same frame if the camera is slower than the stream
        if counter == last_frame_counter:
            time.sleep(0.01)
            continue
        last_frame_counter = counter

        # Encode OUTSIDE the lock! This frees up the camera thread instantly.
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



# =============================================================
# 6. DATABASE HELPERS & STATS
# =============================================================

def get_recent_alerts(limit=5):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def get_all_alerts():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def get_alert_by_id(id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM alerts WHERE id = ?", (id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM alerts")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts WHERE status = 'New'")
        unresolved = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts WHERE timestamp > datetime('now','-1 day')")
        last_24h = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts WHERE type LIKE '%WEAPON%'")
        weapon_alerts = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts WHERE type LIKE '%ANOMALY%'")
        anomaly_alerts = c.fetchone()[0]
        conn.close()
    except Exception:
        total, unresolved, last_24h = 0, 0, 0
        weapon_alerts, anomaly_alerts = 0, 0

    return {
        'total_alerts': total,
        'alerts_24h': last_24h,
        'unresolved': unresolved,
        'cameras': 1,
        'weapon_alerts': weapon_alerts,
        'anomaly_alerts': anomaly_alerts,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
