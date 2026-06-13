"""
============================================================
 Live Anomaly Detection — Sliding Window (Zero Lag)
============================================================
Processes the live camera stream in sliding windows, e.g.
3-second window, sliding forward by 1 second.

Features a Zero-Latency Background Reader that ensures the
stream never lags or builds up a buffer, and paces the display
perfectly to match the camera's natural frame rate (no jitter).

Usage:
    python live_inference.py --url http://192.168.1.50:8080/video
    python live_inference.py --url rtsp://192.168.1.50:8554/stream
    python live_inference.py --url 0
"""

from __future__ import annotations

import os
import sys
import queue

# ── Auto-detect display for Jetson SSH sessions ────────────────
if not os.environ.get("DISPLAY"):
    x11_dir = "/tmp/.X11-unix"
    if os.path.isdir(x11_dir):
        displays = sorted(os.listdir(x11_dir))
        if displays:
            disp = ":" + displays[-1].replace("X", "")
            os.environ["DISPLAY"] = disp
            print(f"  ℹ️  Auto-set DISPLAY={disp}")
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print("  ℹ️  Defaulting DISPLAY=:0")

# Suppress FFmpeg mjpeg warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import time
import argparse
import threading
import numpy as np
from collections import deque

import torch
import torch.nn as nn

# ── Import from inference.py ──────────────────────────────────
from inference import (
    load_models,
    frames_to_tensor,
    get_human_probability_from_cache,
    HUMAN_GATE_THRESHOLD,
    FRAME_SIZE,
    NUM_FRAMES,
    DEFAULT_THRESHOLD,
    USE_FP16,
    LABEL_NORMAL,
    LABEL_ANOMALY,
)

# =============================================================
# 0. ZERO-LATENCY FRAME READER
# =============================================================

class ZeroLatencyReader:
    """Runs cap.read() endlessly in the background to drain the
    buffer instantly. The main thread receives only the absolute
    newest frame. This prevents ANY video lag from accumulating,
    and perfectly smooths out the display pacing."""

    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

            # Signal that a fresh frame arrived
            self.new_frame_event.set()

    def read(self):
        # Block until the background thread catches a new frame.
        # This paces the main loop perfectly to the camera's natural FPS.
        self.new_frame_event.wait(timeout=2.0)
        
        with self.lock:
            # Clear event so we wait for the NEXT frame next time
            self.new_frame_event.clear()
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None


# =============================================================
# 1. SLIDING WINDOW INFERENCE ENGINE
# =============================================================

class SlidingWindowEngine:
    """Accumulates frames. Once `stride_sec` worth of new frames
    is received, snapshots the last `window_sec` block and runs
    inference in a background thread."""

    def __init__(self, video_model, human_model, device,
                 threshold=DEFAULT_THRESHOLD,
                 window_sec=4.0,
                 stride_sec=1.0,
                 num_clips=5,
                 stream_fps=30.0):
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
        t = threading.Thread(
            target=self._process_window,
            args=(window_frames, wid),
            daemon=True
        )
        t.start()

    @torch.no_grad()
    def _process_window(self, frames, window_id):
        try:
            n = len(frames)
            # ── 1. Human Presence Gate ────────────────────────
            p_human = 1.0
            if self.human_model is not None:
                human_sample_ids = np.linspace(0, n - 1, min(5, n), dtype=int)
                human_frames = [frames[i] for i in human_sample_ids]
                p_human = get_human_probability_from_cache(human_frames, self.human_model, self.device)

            # ── 2. Violence Detection (matches inference.py 1.2s dense subclips) ────────────────────────
            window_frac = 0.3
            clip_stride = 0.7 / max(1, self.num_clips - 1)
            clip_tensors = []
            
            for i in range(self.num_clips):
                s_frac = i * clip_stride
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

            # ── 3. Multimodal Fusion Logic ───────────────────
            # "change the value of human detection as less and anomali detection as more"
            # We use it strictly as a gate to avoid artificially inflating normal human activities
            if p_human < 0.2:
                p_video *= p_human
                
            final_score = p_video
            
            prediction = (
                LABEL_ANOMALY if final_score > self.threshold
                else LABEL_NORMAL
            )

            with self.result_lock:
                self.prediction = prediction
                self.score = final_score
                self.window_id = window_id

            icon = "🔴" if prediction == LABEL_ANOMALY else "🟢"
            print(f"  {icon} Window #{window_id:>3d}  {prediction:8s}  "
                  f"score={final_score:.4f}  (video_raw={clip_scores[0]:.2f}, human={p_human:.2f})")

        except Exception as e:
            print(f"  ⚠️  Window #{window_id} error: {e}")
        finally:
            self._busy = False

    def get_result(self):
        with self.result_lock:
            return self.prediction, self.score, self.window_id


# =============================================================
# 2. OVERLAY DRAWING
# =============================================================

def draw_overlay(frame, prediction, score, fps, window_id):
    h, w = frame.shape[:2]
    is_anomaly = prediction == LABEL_ANOMALY

    bg = (0, 0, 180) if is_anomaly else (0, 120, 0)
    banner = frame[0:70, :].copy()
    cv2.rectangle(banner, (0, 0), (w, 70), bg, -1)
    cv2.addWeighted(banner, 0.75, frame[0:70, :], 0.25, 0, frame[0:70, :])

    label = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    cv2.putText(frame, label, (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, f"Score: {score:.3f}", (w - 280, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h - 35), (w, h), (30, 30, 30), -1)
    info = f"FPS: {fps:.1f}  |  Window: #{window_id}  |  'q' to quit"
    cv2.putText(frame, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    if is_anomaly:
        thick = 3 + int(2 * abs(np.sin(time.time() * 4)))
        cv2.rectangle(frame, (0, 70), (w - 1, h - 35), (0, 0, 255), thick)

    return frame


# =============================================================
# 3. VERIFICATION AND VIDEO RECORDING
# =============================================================

class AnomalyVerifier:
    def __init__(self, fps, window_frames, num_windows=5, threshold=0.5):
        self.fps = fps
        self.num_windows = num_windows
        self.threshold = threshold
        self.state = "IDLE" # "IDLE", "COLLECTING"
        self.collected_scores = []
        self.recorded_frames = []
        self.target_windows_left = 0
        self.last_processed_wid = -1
        
        self.rolling_frames = deque(maxlen=int(window_frames))
        
        # Ensure output dir exists
        self.out_dir = "anomaly_videos"
        os.makedirs(self.out_dir, exist_ok=True)

    def process_frame(self, frame):
        # Always keep rolling buffer updated
        self.rolling_frames.append(frame)
        
        if self.state == "COLLECTING":
            self.recorded_frames.append(frame)

    def update_result(self, wid, score):
        if wid <= self.last_processed_wid:
            return  # duplicate or old
        self.last_processed_wid = wid
        
        if self.state == "IDLE":
            if score > self.threshold:
                # Anomaly detected! Start collecting next N windows
                self.state = "COLLECTING"
                # Inclusion: include the trigger score in the average
                self.collected_scores = [score]
                self.recorded_frames = list(self.rolling_frames)
                self.target_windows_left = self.num_windows
                print(f"\n  [Verifier] ⚠️ Anomaly trigger at window {wid} (score {score:.4f}), starting verification over next {self.num_windows} windows...")
        elif self.state == "COLLECTING":
            self.collected_scores.append(score)
            self.target_windows_left -= 1
            print(f"  [Verifier] Collected window {wid} score: {score:.4f} ({self.target_windows_left} left)")
            
            if self.target_windows_left <= 0:
                median_score = float(np.median(self.collected_scores))
                if median_score > self.threshold:
                    print(f"  [Verifier] 🚨 VERIFIED! Median score {median_score:.4f} > {self.threshold}. Saving video...")
                    # Offload saving to a background thread to prevent pausing the main loop
                    t = threading.Thread(
                        target=self.save_video_background,
                        args=(list(self.recorded_frames),),
                        daemon=True
                    )
                    t.start()
                else:
                    print(f"  [Verifier] ❌ FALSE ALARM. Median score {median_score:.4f} <= {self.threshold}.")
                
                # Reset
                self.state = "IDLE"
                self.recorded_frames.clear()
                self.collected_scores.clear()

    def save_video_background(self, frames_to_save):
        if not frames_to_save:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.out_dir, f"anomaly_{timestamp}.mp4")
        
        h, w = frames_to_save[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, float(self.fps), (w, h))
        for f in frames_to_save:
            out.write(f)
        out.release()
        print(f"  [Verifier] 💾 Video successfully saved to {filepath}\n")


# =============================================================
# 4. MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--video_model", type=str, default="models/video_violence_detector.pt")
    parser.add_argument("--human_model", type=str, default="models/human_detector.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--num_clips", type=int, default=10)
    parser.add_argument("--no_display", action="store_true")

    args = parser.parse_args()

    print(f"  Threshold: {args.threshold}  (Detection boundary)")

    url = args.url
    if url.isdigit():
        url = int(url)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    print("\n📦 Loading models...")
    video_model, human_model = load_models(args.video_model, args.human_model, device)

    print(f"\n📡 Connecting to: {url}")
    
    # Use ZERO-LATENCY READER instead of basic VideoCapture
    cap = ZeroLatencyReader(url)

    if not cap.isOpened():
        print("❌ Cannot connect! Check URL and WiFi network.")
        return

    print(f"  ✅ Connected — {cap.w}x{cap.h} @ {cap.fps:.0f} FPS\n")

    engine = SlidingWindowEngine(
        video_model, human_model, device,
        threshold=args.threshold,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        num_clips=args.num_clips,
        stream_fps=cap.fps,
    )

    window_frames = cap.fps * args.window_sec
    verifier = AnomalyVerifier(
        fps=cap.fps, 
        window_frames=window_frames, 
        num_windows=5, 
        threshold=args.threshold
    )

    show = not args.no_display
    if show:
        win = "Live Anomaly Detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(cap.w, 1280), min(cap.h, 720))

    print(f"\n🔴 LIVE DETECTION STARTED — Press 'q' to quit\n")

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0.0

    try:
        while True:
            # Reads EXACTLY in pace with the camera (waits for fresh frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            now = time.time()

            engine.feed_frame(frame)
            verifier.process_frame(frame)

            if engine.should_process():
                engine.start_processing()

            frame_count += 1
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_timer = now

            prediction, score, wid = engine.get_result()
            
            # The engine runs in a background thread. When a result arrives,
            # we pass it to the verifier.
            if wid > verifier.last_processed_wid:
                verifier.update_result(wid, score)

            if show:
                draw_overlay(frame, prediction, score, display_fps, wid)
                
                # Show verifying status
                if verifier.state == "COLLECTING":
                    cv2.putText(frame, f"VERIFYING: {verifier.num_windows - verifier.target_windows_left}/{verifier.num_windows}", (15, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                                
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")

    print("  Shutting down...")
    if show:
        cv2.destroyAllWindows()
    
    # FAST EXIT: Prevent cv2 C++ segfault/bus error when closing threads on Jetson
    os._exit(0)


if __name__ == "__main__":
    main()
