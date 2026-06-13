# Video Anomaly & Weapon Detection System

This project is an advanced computer vision system designed to detect anomalies (such as violence/fights) and weapons in both pre-recorded videos and live camera streams. It leverages PyTorch-based 3D/2D CNNs for temporal action detection and Ultralytics YOLOv11 for real-time object detection.

## 🌟 Features

### 1. Video Anomaly (Violence) Detection
- **Dual-Model Architecture**: Uses a 2D CNN (`human_detector.pt`) to gate processing (only checks for violence if a human is present) and a 3D CNN (`video_violence_detector.pt`) to analyze temporal sequences of 16 frames.
- **Batch Processing**: Process entire directories of videos with detailed output reports (`inference.py`).
- **Zero-Latency Live Stream**: Processes RTSP/HTTP streams or webcams in real-time using a sliding window and background frame draining to ensure the display never lags (`live_inference.py`).
- **Automated Recording**: Automatically records and saves verified incidents of violence to disk.

### 2. Weapon Detection
- **Real-Time Object Detection**: Uses a YOLOv11 ONNX model (`thebest.onnx`) optimized for speed and accuracy.
- **Video File Analysis**: Analyzes pre-recorded videos, drawing bounding boxes and labels around detected weapons (`weapon_detection.py`).
- **Live Alert System & Recording**: Monitors live camera feeds. If a weapon is detected consistently (e.g., 5-7 consecutive frames), it triggers an alert, records a video snippet including pre-roll footage, and logs the event to a CSV file (`live_weapon_detection_recorder.py`).

## 📁 Project Structure

```text
├── models/
│   ├── video_violence_detector.pt    # 3D CNN model for violence
│   ├── human_detector.pt             # 2D CNN model for human presence
│   └── thebest.onnx                  # YOLOv11 ONNX model for weapon detection
├── inference.py                      # Batch anomaly detection for video files
├── live_inference.py                 # Live real-time anomaly detection
├── weapon_detection.py               # Batch weapon detection for video files
├── live_weapon_detection_recorder.py # Live weapon detection with auto-recording
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```
*(Note: Output directories like `outputs/`, `anomaly_videos/`, and `detections/` are created automatically when running the scripts).*

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Models:**
   The trained models are hosted on Hugging Face. You must download them and place them in the `models/` directory.

   **Method 1: Manual Download**
   Download the following files from [Suhas1805/Live_Anomaly_and_Weapon_Detection](https://huggingface.co/Suhas1805/Live_Anomaly_and_Weapon_Detection/tree/main) and place them in the `models/` folder:
   - `video_violence_detector.pt`
   - `human_detector.pt`
   - `thebest.onnx`

   **Method 2: Using Python**
   You can install `huggingface_hub` (`pip install huggingface_hub`) and download them via command line:
   ```bash
   huggingface-cli download Suhas1805/Live_Anomaly_and_Weapon_Detection video_violence_detector.pt --local-dir models/
   huggingface-cli download Suhas1805/Live_Anomaly_and_Weapon_Detection human_detector.pt --local-dir models/
   huggingface-cli download Suhas1805/Live_Anomaly_and_Weapon_Detection thebest.onnx --local-dir models/
   ```

## 🚀 Usage

### Anomaly Detection (Violence)

**Batch Processing Video Files:**
```bash
# Process a single video
python inference.py --video path/to/video.mp4

# Process a directory of videos (default is test_videos/)
python inference.py --video_dir path/to/videos/
```

**Live Stream Processing:**
```bash
# Monitor a webcam (device 0)
python live_inference.py --url 0

# Monitor an IP Camera stream
python live_inference.py --url http://192.168.1.50:8080/video
python live_inference.py --url rtsp://192.168.1.50:8554/stream
```

### Web Application Dashboard

The `web1/` directory contains a full-featured Flask web application that serves as the main dashboard for monitoring and managing alerts.

**Running the Web App:**
Navigate into the `web1` directory and run the `app.py` script:
```bash
cd web1
python app.py
```
After starting the server, open your web browser and go to `http://localhost:5000` to view the dashboard.

### Weapon Detection

**Processing Video Files:**
By default, the script looks for videos in a folder named `tv1/`.
```bash
python weapon_detection.py
```

**Live Stream & Auto-Recording:**
Monitors a live stream. If a weapon is detected, it logs the event to `detections/log.csv` and saves a video clip to `detections/`.
```bash
# Monitor a webcam (device 0)
python live_weapon_detection_recorder.py --url 0

# Monitor an IP Camera stream
python live_weapon_detection_recorder.py --url rtsp://192.168.1.50:8554/stream
```
*Optional Arguments for live weapon detection:*
- `--threshold`: Confidence threshold (default `0.72`)
- `--trigger_frames`: Consecutive frames to trigger an alert (default `6`)
- `--cooldown`: Seconds to wait between alerts (default `60`)

## 📄 Output Locations
- **Anomaly Video Batch**: `outputs/`
- **Live Anomaly Recordings**: `anomaly_videos/`
- **Live Weapon Recordings & Logs**: `detections/`
- **Weapon Video Batch**: `outputs/tv1/`
