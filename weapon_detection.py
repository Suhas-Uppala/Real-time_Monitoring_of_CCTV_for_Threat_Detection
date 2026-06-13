import os
import cv2
import time
from ultralytics import YOLO
import torch

def process_videos():
    # Paths
    model_path = "models/thebest.onnx"
    input_dir = "tv1"
    output_dir = "outputs/tv1"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load YOLOv11 ONNX model
    # The Ultralytics library will use onnxruntime and can leverage tensorrt/cuda execution providers
    print(f"Loading model {model_path}...")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Get all videos in tv1
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found")
        return
        
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No videos found in {input_dir}")
        return
        
    print(f"Found {len(video_files)} videos to process in {input_dir}")
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)
        
        print(f"\nProcessing {video_file}...")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video {input_path}")
            continue
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Resize
            resized_frame = cv2.resize(frame, (768, 768))

            # Run inference
            # We set device to use GPU if available, and verbose=False to keep logs clean
            results = model.predict(source=resized_frame, device=device, verbose=False, imgsz=768)
            
            # Process results
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Confidence and class
                    conf = float(box.conf[0])
                    
                    # Enforce the label "weapon" correctly
                    label = f"weapon {conf:.2f}"
                    
                    # Draw bounding box (BGR color: Red for weapon)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Draw label background
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_height - baseline - 5), (x1 + label_width, y1), (0, 0, 255), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
                
            # Write out frame
            out.write(frame)
            
        cap.release()
        out.release()
        
        process_time = time.time() - start_time
        print(f"Finished {video_file} in {process_time:.2f}s ({(frame_count/process_time):.1f} FPS processing)")
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    process_videos()
    print("\nAll processing complete!")
