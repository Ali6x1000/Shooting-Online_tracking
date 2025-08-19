import cv2
import numpy as np
import os
import time
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ====================== CONFIGURATION ======================
# YOLO model settings
MODEL_PATH = 'best_yolov11lv2_puck.pt'  # Path to YOLO model file
CONF_THRESHOLD = 0.25      # Confidence threshold for detections
IOU_THRESHOLD = 0.45       # IOU threshold for NMS
USE_MPS = True             # Use Metal Performance Shaders on compatible Mac devices

# Video settings
INPUT_VIDEO_PATH = 'shooting2.mp4'            # Path to input video
OUTPUT_VIDEO_PATH = 'shooting2_annotated.mp4'  # Path to output video
RESIZE_WIDTH = None        # Resize width (None for original size)
RESIZE_HEIGHT = None       # Resize height (None for original size)
PROCESS_EVERY_N_FRAME = 1  # Process every Nth frame (1 = process all frames)

# Visualization settings
DRAW_LABELS = True         # Draw class labels
DRAW_CONFIDENCE = True     # Draw confidence scores
BOX_THICKNESS = 2          # Bounding box thickness
TEXT_THICKNESS = 2         # Text thickness
TEXT_FONT_SCALE = 0.8      # Text font scale
# ===========================================================

def get_device():
    """Get the appropriate device (MPS, CUDA, or CPU) for processing."""
    if USE_MPS and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device for acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU for processing (no GPU acceleration available)")
    return device

def process_video(model_path, input_video_path, output_video_path):
    """Process video with YOLO model and create annotated output."""
    # Get appropriate device
    device = get_device()
    
    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
        if str(device) != "cpu":
            # Set model to the selected device
            model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to CPU")
        device = torch.device("cpu")
        model = YOLO(model_path)
    
    # Open video file
    print(f"Opening video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Apply resize if specified
    if RESIZE_WIDTH and RESIZE_HEIGHT:
        frame_width = RESIZE_WIDTH
        frame_height = RESIZE_HEIGHT
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Process video frames
    frame_count = 0
    processed_count = 0
    
    print(f"Processing video with {total_frames} frames...")
    progress_bar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        progress_bar.update(1)
        
        # Process every Nth frame
        if frame_count % PROCESS_EVERY_N_FRAME != 0:
            # Write original frame to output
            if RESIZE_WIDTH and RESIZE_HEIGHT:
                frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
            out.write(frame)
            continue
        
        processed_count += 1
        
        # Resize frame if needed
        if RESIZE_WIDTH and RESIZE_HEIGHT:
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        
        # Run YOLO detection
        try:
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=device)
        except Exception as e:
            print(f"Error during inference: {e}")
            print("Falling back to CPU for this frame")
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu")
        
        # Draw detections on frame
        annotated_frame = draw_detections(frame, results)
        
        # Write frame to output video
        out.write(annotated_frame)
    
    # Release resources
    progress_bar.close()
    cap.release()
    out.release()
    print(f"Video processing complete. Processed {processed_count} frames.")
    print(f"Output saved to: {output_video_path}")

def draw_detections(frame, results):
    """Draw detection boxes and labels on the frame."""
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Get detection results from first (and only) image
    result = results[0]
    
    if result.boxes is not None:
        boxes = result.boxes.cpu().numpy()
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence score
            conf = float(box.conf[0])
            
            # Get class ID and name
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Create color for this class (consistent color per class)
            color = (int(hash(class_name) % 256), 
                     int(hash(class_name + "salt") % 256),
                     int(hash(class_name + "pepper") % 256))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            
            # Prepare label text
            label_text = ""
            if DRAW_LABELS:
                label_text += f"{class_name}"
            if DRAW_CONFIDENCE:
                if label_text:
                    label_text += f" {conf:.2f}"
                else:
                    label_text += f"{conf:.2f}"
            
            # Draw label background
            if label_text:
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, TEXT_FONT_SCALE, TEXT_THICKNESS)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            TEXT_FONT_SCALE, (255, 255, 255), TEXT_THICKNESS)
    
    return annotated_frame

if __name__ == "__main__":
    # Check system compatibility
    if USE_MPS:
        mps_available = torch.backends.mps.is_available()
        if mps_available:
            mps_version = torch.backends.mps.is_built()
            print(f"MPS is available. Built: {mps_version}")
        else:
            print("MPS is not available on this system")
    
    # Check if input video exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at {INPUT_VIDEO_PATH}")
        exit(1)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: YOLO model not found at {MODEL_PATH}")
        exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_VIDEO_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the video
    start_time = time.time()
    process_video(MODEL_PATH, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)
    end_time = time.time()
    
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
