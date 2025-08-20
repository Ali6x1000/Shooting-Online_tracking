# test_ocsort.py

import sys
import cv2
import numpy as np


# Import the OCSort class
from trackers.ocsort_tracker.ocsort import OCSort
from ultralytics import YOLO




class OnlinePuckTracker: 
    def __init__(self, model_path):
        # Initialize the OnlinePuckTracker with the given model path
        self.model_path = model_path
        self.Yolo_Model = YOLO(model_path)
        self.OC = OCSort(
            det_thresh=0.4,        # Lower threshold to catch more detections
            max_age=10,            # Shorter age for fast scenarios
            min_hits=2,            # Fewer hits needed for confirmation
            iou_threshold=0.2,     # Lower IoU for fast movement
            delta_t=2,             # Shorter look-back for velocity
            asso_func="giou",      # GIoU often better for fast objects
            inertia=0.3,           # Higher inertia for smoother velocity
            use_byte=True          # Enable for better association
        )


    def Start (detections):
                # Initialize video capture
        cap = cv2.VideoCapture('your_video.mp4')
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame dimensions
            img_height, img_width = frame.shape[:2]
            img_info = [img_height, img_width]
            img_size = [1920, 1080]  # Your YOLO input size
            
            # Run YOLO detection
            yolo_results = your_yolo_model.predict(frame,img_size=1080)
            
            # Convert to OCSort format
            detections = process_yolo_detections(yolo_results, img_info, img_size)
            
            # Update tracker
            tracked_objects = tracker.update(detections, img_info, img_size)
            
            # Draw tracking results
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw track ID
                cv2.putText(frame, f'ID: {int(track_id)}', 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
            


    def process_yolo_detections(yolo_results, img_info, img_size):
        """
        Convert YOLO results to OCSort format
        
        Args:
            yolo_results: YOLO detection results
            img_info: [img_height, img_width] 
            img_size: [input_height, input_width] used for YOLO
        """
        if len(yolo_results) == 0:
            return np.empty((0, 5))
        
        # Extract bounding boxes and scores from YOLO
        # Assuming yolo_results format: [x1, y1, x2, y2, confidence, class_id]
        detections = []
        
        for detection in yolo_results:
            x1, y1, x2, y2, conf, class_id = detection
            # OCSort expects [x1, y1, x2, y2, score]
            detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections)

        




def main():
    model_path = "/Users/alinawaf/Desktop/Projects/shooting/DBSCAN/Shooting-Online_tracking/best_yolov11lv2_puck.pt"
    Model = YOLO(model_path)
    print(Model)
    # Initialize tracker with a detection threshold
    tracker = OCSort(det_thresh=0.3)
    print("OCSort imported and initialized successfully!")
    print(tracker)

if __name__ == "__main__":
    main()
