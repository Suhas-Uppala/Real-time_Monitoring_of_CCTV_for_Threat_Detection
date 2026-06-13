import sys
import os
import argparse

# Add parent directory to path so inference.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, state, CameraThread

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard — Weapon + Anomaly Detection")
    parser.add_argument("--url", type=str, default="http://10.100.7.94:8080/video",
                        help="Camera URL (RTSP/HTTP/device index)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--weapon_threshold", type=float, default=0.72,
                        help="Weapon detection threshold")
    parser.add_argument("--anomaly_threshold", type=float, default=0.9,
                        help="Anomaly detection threshold")
    parser.add_argument("--cooldown", type=int, default=60,
                        help="Cooldown between alerts (seconds)")

    args = parser.parse_args()

    # Update global state from CLI
    state.url = args.url
    state.weapon_threshold = args.weapon_threshold
    state.anomaly_threshold = args.anomaly_threshold
    state.cooldown = args.cooldown

    print(f"\n🚀 VISIONGUARD — Weapon + Anomaly Detection")
    print(f"🔗 Camera: {args.url}")
    print(f"🔫 Weapon threshold: {args.weapon_threshold}")
    print(f"🔴 Anomaly threshold: {args.anomaly_threshold}")
    print(f"🌐 Access at: http://0.0.0.0:{args.port}")

    # Start detection background thread
    cam_thread = CameraThread(args.url)
    cam_thread.start()

    try:
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nStopping...")
        cam_thread.running = False
        sys.exit(0)
