"""
============================================================
 Anomaly Detection — Inference Script
============================================================
Binary classification:  NORMAL  vs  ANOMALY

Uses two trained models:
  1. HumanPresenceCNN      — gate: is a human present?
  2. VideoViolenceDetector  — 3D CNN (16 frames @ 112×112)

The video model was trained on generic Fight/NonFight data
(RWF-2000 + Real Life Violence). For custom test videos that
look similar in both classes (same room, same people), we
combine the CNN prediction with temporal consistency analysis
to improve discrimination.

Usage:
    python inference.py
    python inference.py --video path/to/clip.mp4
    python inference.py --threshold 0.55
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# =============================================================
# 1. MODEL DEFINITIONS  (EXACT replicas from training notebook)
# =============================================================

class VideoViolenceDetector(nn.Module):
    """3D CNN — structure must match the training checkpoint exactly."""

    def __init__(self):
        super().__init__()

        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv3d_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


class HumanPresenceCNN(nn.Module):
    """Binary human presence classifier — structure matches training."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


# =============================================================
# 2. CONFIGURATION  (matches training)
# =============================================================

NUM_FRAMES  = 16               # training: ViolenceVideoDataset used 16
FRAME_SIZE  = (112, 112)       # training: frame_size=(112, 112)
HUMAN_SIZE  = (224, 224)       # training: transforms.Resize((224, 224))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_THRESHOLD = 0.7
HUMAN_GATE_THRESHOLD = 0.15

# Dense temporal sampling: number of overlapping clip windows
NUM_DENSE_CLIPS = 10

MAX_CACHE_FRAMES = 300
USE_FP16 = True

LABEL_NORMAL  = "NORMAL"
LABEL_ANOMALY = "ANOMALY"


# =============================================================
# 3. LOAD MODELS
# =============================================================

def load_models(video_path, human_path, device):
    use_half = USE_FP16 and device.type == "cuda"
    
    ckpt = torch.load(video_path, map_location=device, weights_only=False)
    video_model = VideoViolenceDetector().to(device)
    video_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    video_model.eval()
    if use_half:
        video_model.half()
    print(f"  Video model loaded   — acc: {ckpt.get('accuracy', '?')}%  "
          f"epochs: {ckpt.get('epochs', '?')}  "
          f"{'(FP16)' if use_half else '(FP32)'}")

    human_model = None
    if os.path.isfile(human_path):
        ckpt_h = torch.load(human_path, map_location=device, weights_only=False)
        human_model = HumanPresenceCNN().to(device)
        human_model.load_state_dict(ckpt_h["model_state_dict"], strict=True)
        human_model.eval()
        if use_half:
            human_model.half()
        print(f"  Human model loaded   — acc: {ckpt_h.get('accuracy', '?')}%  "
              f"epochs: {ckpt_h.get('epochs', '?')}  "
              f"{'(FP16)' if use_half else '(FP32)'}")
    else:
        print(f"  ⚠️  Human model not found at {human_path}")

    return video_model, human_model


# =============================================================
# 4. PREPROCESSING  (OPTIMISED)
# =============================================================

def _read_all_frames_sequential(video_path, max_frames=MAX_CACHE_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        cap.release()
        return [], [], {"total_frames": 0, "fps": fps, "width": width, "height": height}

    keep_n = min(total, max_frames)
    keep_set = set(np.linspace(0, total - 1, keep_n, dtype=int).tolist())
    human_ids = set(np.linspace(0, total - 1, min(5, total), dtype=int).tolist())

    small_frames = []
    human_frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in keep_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, FRAME_SIZE)
            small_frames.append(small)
            if idx in human_ids:
                human_frames.append(rgb)
        elif idx in human_ids:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            human_frames.append(rgb)
        idx += 1

    cap.release()
    meta = {"total_frames": total, "fps": fps, "width": width, "height": height}
    return small_frames, human_frames, meta


def _select_clip_from_cache(cached_frames, start_frac, end_frac, num_frames=NUM_FRAMES):
    n = len(cached_frames)
    if n == 0:
        return [np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)] * num_frames
    s = int(start_frac * (n - 1))
    e = int(end_frac   * (n - 1))
    if e <= s:
        e = min(s + num_frames, n - 1)
    ids = np.linspace(s, e, num_frames, dtype=int)
    frames = [cached_frames[i] for i in ids]
    return frames


def frames_to_tensor(rgb_frames, use_half=False):
    arr = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
    arr = np.transpose(arr, (3, 0, 1, 2))

    mean = IMAGENET_MEAN.reshape(3, 1, 1, 1)
    std  = IMAGENET_STD.reshape(3, 1, 1, 1)
    arr  = (arr - mean) / std

    dtype = torch.float16 if use_half else torch.float32
    return torch.tensor(arr, dtype=dtype).unsqueeze(0)


_human_tf = transforms.Compose([
    transforms.Resize(HUMAN_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(),
                         std=IMAGENET_STD.tolist()),
])


def get_human_probability_from_cache(human_frames, human_model, device):
    if not human_frames:
        return 0.0
    use_half = USE_FP16 and device.type == "cuda"
    tensors = []
    for rgb in human_frames:
        pil = Image.fromarray(rgb)
        t = _human_tf(pil)
        tensors.append(t)
    batch = torch.stack(tensors).to(device)
    if use_half:
        batch = batch.half()
    with torch.no_grad():
        scores = human_model(batch).squeeze(-1)
    return float(scores.max().item())


# =============================================================
# 5. CORE INFERENCE (SLIDING WINDOW METHOD)
# =============================================================

@torch.no_grad()
def predict_video(video_path, video_model, human_model, device,
                  threshold=DEFAULT_THRESHOLD):
    """Predict NORMAL vs ANOMALY using an exactly matched 4-second sliding window
    replicating the real-time stream's ZeroLatency processor."""
    cached_frames, human_frames, meta = _read_all_frames_sequential(video_path)
    total  = meta["total_frames"]
    fps    = meta["fps"]
    w, h   = meta["width"], meta["height"]

    result = {
        "video": os.path.basename(video_path),
        "total_frames": total,
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
    }

    if not cached_frames:
        result["final_score"] = 0.0
        result["prediction"] = LABEL_NORMAL
        result["note"] = "Could not read video frames"
        return result

    # 1. Human gate (using 5 uniform frames exactly as live)
    p_human = None
    if human_model is not None:
        p_human = get_human_probability_from_cache(human_frames, human_model, device)
        result["p_human"] = round(p_human, 4)
        if p_human < HUMAN_GATE_THRESHOLD:
            result["final_score"] = 0.0
            result["prediction"] = LABEL_NORMAL
            result["note"] = "No human detected"
            return result

    # 2. Sliding Window Settings (Must match live_inference.py exactly)
    # The user specifically requested: "use the median of the chunks for anomali in the sliding window method"
    window_sec = 4.0
    stride_sec = 1.0
    num_clips_per_chunk = NUM_DENSE_CLIPS
    use_half = USE_FP16 and device.type == "cuda"
    
    window_frames = int(window_sec * fps)
    stride_frames = int(stride_sec * fps)
    
    # Generate chunks
    chunks = []
    if total < window_frames:
        chunks.append(cached_frames)
    else:
        for i in range(0, max(1, total - window_frames + 1), stride_frames):
            chunks.append(cached_frames[i : i + window_frames])
            
    chunk_scores = []
    
        # Evaluate each 4-second sliding chunk exactly like the live stream
    window_frac = 0.3
    clip_stride = 0.7 / max(1, num_clips_per_chunk - 1)
    
    for chunk in chunks:
        n = len(chunk)
        if n == 0:
            continue
            
        clip_tensors = []
        for i in range(num_clips_per_chunk):
            s_frac = i * clip_stride
            e_frac = min(s_frac + window_frac, 1.0)
            frames_for_clip = _select_clip_from_cache(chunk, s_frac, e_frac, NUM_FRAMES)
            t = frames_to_tensor(frames_for_clip, use_half=use_half)
            clip_tensors.append(t)
            
        batch = torch.cat(clip_tensors, dim=0).to(device)
        scores = video_model(batch).squeeze(-1)
        
        # Median across overlapping clips
        p_video = float(scores.median().item())
        
        # ── 2. Human Presence Gate ────────────────────────
        p_human = 1.0
        if human_model is not None:
            human_sample_ids = np.linspace(0, n - 1, min(5, n), dtype=int)
            human_frames_chunk = [chunk[i] for i in human_sample_ids]
            p_human = get_human_probability_from_cache(human_frames_chunk, human_model, device)
            
        # ── 3. Multimodal Fusion Logic ───────────────────
        if p_human < 0.2:
            p_video *= p_human
            
        final_score = p_video
            
        chunk_scores.append(final_score)
        
    if not chunk_scores:
        result["final_score"] = 0.0
        result["prediction"] = LABEL_NORMAL
        return result

    # 3. Final Verification (Median of the Chunks)
    # The anomaly is verified by checking the median of chunk sequence (similar to live_inference AnomalyVerifier)
    chunk_group_size = 6  # 1 trigger chunk + 5 collection chunks = 6 windows
    
    if len(chunk_scores) <= chunk_group_size:
        # If video is short, just take the median of all available chunks
        final_score = float(np.median(chunk_scores))
    else:
        # For longer videos, find the most anomalous 6-chunk sequence
        rolling_medians = []
        for i in range(len(chunk_scores) - chunk_group_size + 1):
            group = chunk_scores[i : i + chunk_group_size]
            rolling_medians.append(float(np.median(group)))
        final_score = max(rolling_medians)
        
    result["chunk_scores"] = [round(s, 4) for s in chunk_scores]
    result["final_score"] = round(final_score, 4)
    result["threshold"] = threshold
    
    # 4. Classify
    result["prediction"] = LABEL_ANOMALY if final_score > threshold else LABEL_NORMAL
    
    return result


# =============================================================
# 7. ANNOTATED VIDEO OUTPUT
# =============================================================

def create_annotated_video(video_path, prediction, final_score, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    is_anomaly = (prediction == LABEL_ANOMALY)
    colour = (0, 0, 255) if is_anomaly else (0, 200, 0)
    bg = (0, 0, 180) if is_anomaly else (0, 120, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, (0, 0), (w, 70), bg, -1)
        cv2.putText(frame, prediction, (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, f"Score: {final_score:.3f}", (w - 280, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


# =============================================================
# 8. BATCH PROCESSING
# =============================================================

def run_batch(video_dir, video_model, human_model, device,
              threshold, output_dir, annotate):
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    videos = sorted([f for f in os.listdir(video_dir)
                     if f.lower().endswith(exts)])

    if not videos:
        print(f"⚠️  No video files in {video_dir}")
        return pd.DataFrame()

    print(f"\n{'='*65}")
    print(f"  Processing {len(videos)} videos from: {video_dir}")
    print(f"  Threshold: {threshold}   Device: {device}")
    print(f"  Dense clips: {NUM_DENSE_CLIPS}")
    print(f"{'='*65}\n")

    results = []

    for i, vf in enumerate(videos, 1):
        vpath = os.path.join(video_dir, vf)
        t0 = time.time()

        try:
            res = predict_video(vpath, video_model, human_model,
                                device, threshold)
            elapsed = time.time() - t0
            res["time_s"] = round(elapsed, 2)

            if annotate:
                out = create_annotated_video(
                    vpath, res["prediction"], res["final_score"], output_dir)
                res["annotated_video"] = out

            icon = "🔴" if res["prediction"] == LABEL_ANOMALY else "🟢"
            c_scores = res.get("chunk_scores", [])
            max_c = max(c_scores) if c_scores else 0.0
            min_c = min(c_scores) if c_scores else 0.0
            print(f"  [{i}/{len(videos)}] {icon}  {vf:<44}  "
                  f"{res['prediction']:<8}  "
                  f"score={res['final_score']:.4f}  "
                  f"(min={min_c:.3f} max={max_c:.3f} "
                  f"chunks={len(c_scores)})  "
                  f"[{elapsed:.1f}s]")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(videos)}] ❌  {vf}: {e}")
            res = {"video": vf, "prediction": "ERROR", "error": str(e),
                   "time_s": round(elapsed, 2)}

        results.append(res)

    return pd.DataFrame(results)


def print_summary(df):
    if df.empty or "prediction" not in df.columns:
        return

    print(f"\n{'='*65}")
    print(f"  📊  RESULTS SUMMARY")
    print(f"{'='*65}")

    normal  = (df["prediction"] == LABEL_NORMAL).sum()
    anomaly = (df["prediction"] == LABEL_ANOMALY).sum()

    print(f"\n  🟢 NORMAL  : {normal}")
    print(f"  🔴 ANOMALY : {anomaly}")

    print(f"\n  {'─'*63}")
    print(f"  {'Video':<44} {'Result':<10} {'Score'}")
    print(f"  {'─'*63}")

    for _, row in df.iterrows():
        pred = row.get("prediction", "?")
        score = row.get("final_score", 0)
        icon = "🔴" if pred == LABEL_ANOMALY else "🟢" if pred == LABEL_NORMAL else "❌"
        s_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        print(f"  {icon} {row['video']:<43} {pred:<10} {s_str}")

    print(f"  {'─'*63}\n")


# =============================================================
# 9. MAIN
# =============================================================

def _update_num_clips(n):
    global NUM_DENSE_CLIPS
    NUM_DENSE_CLIPS = n


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection — Binary (NORMAL vs ANOMALY)")

    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default="test_videos")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--video_model", type=str,
                        default="models/video_violence_detector.pt")
    parser.add_argument("--human_model", type=str,
                        default="models/human_detector.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no_human_gate", action="store_true")
    parser.add_argument("--no_annotate", action="store_true")
    parser.add_argument("--num_clips", type=int, default=10,
                        help="Number of temporal clips to sample")

    args = parser.parse_args()

    _update_num_clips(args.num_clips)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    print("\n📦 Loading models...")
    hpath = "" if args.no_human_gate else args.human_model
    video_model, human_model = load_models(args.video_model, hpath, device)

    if args.video:
        print(f"\n🎥 Processing: {args.video}")
        res = predict_video(args.video, video_model, human_model,
                            device, args.threshold)

        icon = "🔴" if res["prediction"] == LABEL_ANOMALY else "🟢"
        print(f"\n  {icon}  Prediction  : {res['prediction']}")
        print(f"     Final Score : {res['final_score']:.4f}")
        print(f"     Threshold   : {args.threshold}")
        if "chunk_scores" in res:
            print(f"     Chunk scores: {res['chunk_scores']}")
        if "p_human" in res:
            print(f"     Human prob  : {res['p_human']:.4f}")

        if not args.no_annotate:
            out = create_annotated_video(
                args.video, res["prediction"], res["final_score"],
                args.output_dir)
            print(f"     Annotated   : {out}")

        os.makedirs(args.output_dir, exist_ok=True)
        jpath = os.path.join(args.output_dir, "result.json")
        with open(jpath, "w") as f:
            json.dump(res, f, indent=2, default=str)

    else:
        df = run_batch(args.video_dir, video_model, human_model,
                       device, args.threshold, args.output_dir,
                       annotate=not args.no_annotate)
        print_summary(df)

        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"  📁 CSV  → {csv_path}")

        json_path = os.path.join(args.output_dir, "predictions.json")
        records = df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"  📁 JSON → {json_path}")

    print(f"\n✅ Done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
