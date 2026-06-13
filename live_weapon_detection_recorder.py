"""
============================================================
 Live Weapon Detection with 5-Second Trigger Recording
============================================================
Processes the live camera stream for weapon detection.
If a weapon is detected for 5 consecutive seconds:
    1. Saves the triggering frame as an image.
    2. Logs the detection (date, time, image path) to a CSV.

Usage:
    python live_weapon_detection_recorder.py --url http://192.168.1.50:8080/video
"""

import os
import sys
import cv2
import time
import argparse
import threading
import torch
import numpy as np
import csv
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# ── Auto-detect display for Jetson SSH sessions ────────────────
if not os.environ.get("DISPLAY"):
    x11_dir = "/tmp/.X11-unix"
    if os.path.isdir(x11_dir):
        displays = sorted(os.listdir(x11_dir))
        if displays:
            disp = ":" + displays[-1].replace("X", "")
            os.environ["DISPLAY"] = disp
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

# Suppress FFmpeg mjpeg warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# =============================================================
# 0. ZERO-LATENCY FRAME READER
# =============================================================

class ZeroLatencyReader:
    def __init__(self, url):
        self.url = url
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
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
        return getattr(self, 'thread', None) is not None

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1.0)
                self.cap.release()
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

# =============================================================
# 1. LOGGING UTILITY
# =============================================================

def log_detection(video_path, max_confidence):
    csv_file = "detections/log.csv"
    os.makedirs("detections", exist_ok=True)
    
    file_exists = os.path.isfile(csv_file)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Time", "Video Path", "Max Confidence"])
        writer.writerow([date_str, time_str, video_path, f"{max_confidence:.2f}"])
    print(f"📄 Logged: {date_str} {time_str} -> {video_path}")

# =============================================================
# 2. OVERLAY DRAWING
# =============================================================

def draw_overlay(frame, results, fps, conf_threshold, consecutive_frames, recording_status, cooldown_remaining):
    h, w = frame.shape[:2]
    weapon_detected = False
    max_conf = 0.0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            
            weapon_detected = True
            max_conf = max(max_conf, conf)
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(x1 * w / 768)
            y1 = int(y1 * h / 768)
            x2 = int(x2 * w / 768)
            y2 = int(y2 * h / 768)
            
            label = f"weapon {conf:.2f}"
            color = (0, 70, 255) # Bright orange-red
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Banner logic
    if weapon_detected:
        bg = (0, 0, 180) # Red banner for detection
        label_text = "WEAPON DETECTED"
        if recording_status:
            label_text += " [ALERTING]"
    else:
        bg = (0, 120, 0) # Green banner for normal
        label_text = "NORMAL"

    banner = frame[0:70, :].copy()
    cv2.rectangle(banner, (0, 0), (w, 70), bg, -1)
    cv2.addWeighted(banner, 0.75, frame[0:70, :], 0.25, 0, frame[0:70, :])
    cv2.putText(frame, label_text, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # Bottom info
    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    status_msg = f"Inference FPS: {fps:.1f}"
    if weapon_detected:
        status_msg += f" | Consecutive Frames: {consecutive_frames}"
    
    cv2.putText(frame, status_msg, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if weapon_detected:
        thick = 3 + int(2 * abs(np.sin(time.time() * 4)))
        cv2.rectangle(frame, (0, 70), (w - 1, h - 35), (0, 0, 255), thick)

    return weapon_detected, max_conf

# =============================================================
# 3. MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="Camera URL (e.g. rtsp://... or 0)")
    parser.add_argument("--model", type=str, default="models/thebest.onnx")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--trigger_frames", type=int, default=6, help="Consecutive frames to trigger alert (5-7 recommended)")
    parser.add_argument("--cooldown", type=int, default=60, help="Seconds to wait between alerts")
    parser.add_argument("--no_display", action="store_true")

    args = parser.parse_args()
    url = args.url
    if url.isdigit():
        url = int(url)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    print(f"\n📦 Loading model from {args.model}...")
    try:
        model = YOLO(args.model, task='detect')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    cap = ZeroLatencyReader(url)
    if not cap.isOpened():
        print("❌ Cannot connect! Check URL.")
        return

    show = not args.no_display
    if show:
        win = "Live Weapon Detection Recorder"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

    print(f"\n🔴 VIDEO RECORDER ACTIVE — Trigger: {args.trigger_frames} frames @ {args.threshold} conf\n")

    consecutive_frames = 0
    is_recording = False
    video_writer = None
    max_event_conf = 0.0
    last_alert_time = 0
    
    # Buffer for pre-roll (e.g., last 30 frames)
    pre_roll_buffer = deque(maxlen=int(cap.fps * 2)) # 2 seconds pre-roll
    
    current_video_path = ""
    frame_count = 0
    fps_timer = time.time()
    display_fps = 0.0
    os.makedirs("detections", exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Keep a copy of the clean frame for video saving
            clean_frame = frame.copy()
            
            now = time.time()
            resized_frame = cv2.resize(frame, (768, 768))
            results = model.predict(source=resized_frame, device=device, verbose=False, imgsz=768)

            # Performance stats
            frame_count += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = now

            cooldown_remaining = max(0, args.cooldown - (now - last_alert_time))
            weapon_detected, frame_max_conf = draw_overlay(frame, results, display_fps, args.threshold, consecutive_frames, is_recording, cooldown_remaining)

            # ── TRIGGER & RECORDING LOGIC ─────────────────────────
            if weapon_detected:
                consecutive_frames += 1
                max_event_conf = max(max_event_conf, frame_max_conf)
                
                # Check if we hit the trigger (5-7 frames) AND not in cooldown
                if consecutive_frames >= args.trigger_frames and not is_recording:
                    if (now - last_alert_time) >= args.cooldown:
                        print(f"🔥 ALERT TRIGGERED: ({consecutive_frames} frames). Starting recording...")
                        is_recording = True
                        last_alert_time = now # Set alert start time for cooldown
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_path = f"detections/alert_{timestamp}.mp4"
                        
                        # Initialize VideoWriter
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, cap.fps, (cap.w, cap.h))
                        
                        # Write pre-roll buffer to video
                        for f in pre_roll_buffer:
                            video_writer.write(f)
                        
                        current_video_path = video_path
            
            # Stop condition for recording
            if is_recording:
                # Keep recording if detection continues OR if we haven't reached 5s yet
                # This ensures the "sent" video has enough context.
                recording_elapsed = now - last_alert_time
                if not weapon_detected and recording_elapsed >= 5.0:
                    print(f"✅ Alert recording finished. Saving video and logging to CSV...")
                    video_writer.release()
                    video_writer = None
                    is_recording = False
                    log_detection(current_video_path, max_event_conf)
                    max_event_conf = 0.0
            
            if not weapon_detected:
                consecutive_frames = 0

            # Write to active video or store in pre-roll
            if is_recording and video_writer:
                video_writer.write(frame) 
            else:
                pre_roll_buffer.append(clean_frame)
            # ──────────────────────────────────────────────────────

            if show:
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.001)

    except Exception as e:
        print(f"❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")

    if video_writer:
        video_writer.release()
    cap.running = False
    if show:
        cv2.destroyAllWindows()
    os._exit(0)

if __name__ == "__main__":
    main()
