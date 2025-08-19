def annotate_frame_with_predictions(self, frame, detections, timestamp):
        """Enhanced annotation showing predicted vs real detections"""
        annotated = frame.copy()
        
        # Draw current detections
        if len(detections) > 0 and hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            for i in range(len(detections)):
                track_id = detections.tracker_id[i]
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                
                x1, y1, x2, y2 = bbox.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                color = self.trajectory_metadata[track_id]['color']
                
                # Check if this is a predicted detection
                is_predicted = False
                for detection in self.all_detections:
                    if (detection['frame_num'] == self.frame_count and 
                        detection.get('predicted', False) and
                        detection.get('track_id') == track_id):
                        is_predicted = True
                        break
                
                # Draw bounding box (dashed for predictions)
                if is_predicted:
                    # Draw dashed rectangle for predictions
                    self._draw_dashed_rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} PRED ({confidence:.2f})"
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} ({confidence:.2f})"
                
                # Add tracker type indicator
                tracker_type = "HS" if self.use_high_speed_tracker else "BT"
                label += f" [{tracker_type}]"
                
                # Draw center point
                cv2.circle(annotated, (center_x, center_y), 4, color, -1)
                
                # Draw label
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectories with prediction indicators
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            color = self.trajectory_metadata[track_id]['color']
            
            # Separate real and predicted points
            real_points = [(int(p['x']), int(p['y'])) for p in trajectory if not p.get('predicted', False)]
            pred_points = [(int(p['x']), int(p['y'])) for p in trajectory if p.get('predicted', False)]
            
            # Draw real trajectory (solid lines)
            if len(real_points) > 1:
                for i in range(1, len(real_points)):
                    cv2.line(annotated, real_points[i-1], real_points[i], color, 2)
            
            # Draw predicted trajectory (dashed lines)
            if len(pred_points) > 1:
                for i in range(1, len(pred_points)):
                    self._draw_dashed_line(annotated, pred_points[i-1], pred_points[i], color, 2)
            
            # Draw ALL points in trajectory as a continuous line (mixed real and predicted)
            all_points = [(int(p['x']), int(p['y'])) for p in trajectory]
            if len(all_points) > 1:
                # Draw start point (green circle)
                cv2.circle(annotated, all_points[0], 6, (0, 255, 0), -1)
                cv2.circle(annotated, all_points[0], 6, (0, 0, 0), 1)
                
                # Draw end point (red circle) 
                cv2.circle(annotated, all_points[-1], 6, (0, 0, 255), -1)
                cv2.circle(annotated, all_points[-1], 6, (0, 0, 0), 1)
                
                # Draw direction arrow at midpoint
                if len(all_points) > 2:
                    mid_idx = len(all_points) // 2
                    if mid_idx < len(all_points) - 1:
                        dx = all_points[mid_idx + 1][0] - all_points[mid_idx][0]
                        dy = all_points[mid_idx + 1][1] - all_points[mid_idx][1]
                        if dx != 0 or dy != 0:  # Avoid zero-length arrow
                            length = np.sqrt(dx*dx + dy*dy)
                            if length > 0:
                                dx = dx / length * 20  # Normalize and scale
                                dy = dy / length * 20
                                cv2.arrowedLine(annotated, all_points[mid_idx], 
                                              (int(all_points[mid_idx][0] + dx), int(all_points[mid_idx][1] + dy)), 
                                              color, 2, tipLength=0.3)
            
            # Draw prediction indicators
            if self.enable_prediction and track_id in self.last_positions:
                last_pos = self.last_positions[track_id]
                
                # Draw prediction cone/uncertainty for future position
                pred_x, pred_y, confidence = self.predictor.predict_position(track_id, timestamp + 0.1)
                if pred_x is not None:
                    future_point = (int(pred_x), int(pred_y))
                    current_point = (int(last_pos['x']), int(last_pos['y']))
                    
                    # Draw prediction vector
                    self._draw_dashed_line(annotated, current_point, future_point, 
                                         (255, 255, 0), 1)  # Yellow for prediction
                    cv2.circle(annotated, future_point, 3, (255, 255, 0), -1)
        
        # Enhanced frame info with tracker type
        total_trajectories = len(self.trajectories)
        active_trajectories = len([t for t in self.trajectories.values() 
                                 if len(t) > 0 and abs(t[-1]['timestamp'] - timestamp) < 1.0])
        predicted_count = len([d for d in self.all_detections 
                             if d['frame_num'] == self.frame_count and d.get('predicted', False)])
        
        tracker_name = "High-Speed" if self.use_high_speed_tracker else "ByteTracker"
        info_text = f"Frame: {self.frame_count} | Time: {timestamp:.2f}s | {tracker_name} | Current: {len(detections)} | Predicted: {predicted_count} | Total Tracks: {total_trajectories} | Active: {active_trajectories}"
        
        # Draw info with background
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Enhanced legend
        legend_y = 60
        legend_text = "Legend: Green=Start, Red=End, Solid=Real, Dashed=Predicted, Yellow=Future, HS=High-Speed, BT=ByteTracker"
        cv2.putText(annotated, legend_text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(annotated, legend_text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return annotated
        
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import os
from pathlib import Path
import json
from scipy import interpolate
from sklearn.linear_model import LinearRegression

# You'll need to install these:
# pip install ultralytics supervision opencv-python matplotlib scipy scikit-learn filterpy

try:
    from ultralytics import YOLO
    import supervision as sv
    from filterpy.kalman import KalmanFilter
except ImportError:
    print("Missing required packages. Install with:")
    print("pip install ultralytics supervision filterpy scipy scikit-learn")
    raise

class PuckPredictor:
    """
    Predicts puck positions based on trajectory history
    Supports multiple prediction methods: linear, polynomial, and Kalman filter
    """
    
    def __init__(self, method='kalman', history_window=10):
        """
        Initialize predictor
        
        Args:
            method (str): Prediction method ('linear', 'polynomial', 'kalman')
            history_window (int): Number of past positions to use for prediction
        """
        self.method = method
        self.history_window = history_window
        self.track_histories = defaultdict(deque)  # track_id: deque of positions
        self.kalman_filters = {}  # track_id: KalmanFilter
        
    def update_history(self, track_id, x, y, timestamp):
        """Update position history for a track"""
        history = self.track_histories[track_id]
        history.append({
            'x': x, 'y': y, 'timestamp': timestamp
        })
        
        # Keep only recent history
        while len(history) > self.history_window:
            history.popleft()
    
    def predict_position(self, track_id, future_timestamp):
        """
        Predict position at future timestamp
        
        Args:
            track_id: Track identifier
            future_timestamp: Time to predict position for
            
        Returns:
            tuple: (predicted_x, predicted_y, confidence)
        """
        if track_id not in self.track_histories:
            return None, None, 0.0
            
        history = list(self.track_histories[track_id])
        if len(history) < 2:
            return None, None, 0.0
        
        if self.method == 'linear':
            return self._predict_linear(history, future_timestamp)
        elif self.method == 'polynomial':
            return self._predict_polynomial(history, future_timestamp)
        elif self.method == 'kalman':
            return self._predict_kalman(track_id, history, future_timestamp)
        else:
            raise ValueError(f"Unknown prediction method: {self.method}")
    
    def _predict_linear(self, history, future_timestamp):
        """Linear extrapolation based on velocity"""
        if len(history) < 2:
            return None, None, 0.0
        
        # Use last two points to calculate velocity
        p1, p2 = history[-2], history[-1]
        dt = p2['timestamp'] - p1['timestamp']
        
        if dt <= 0:
            return p2['x'], p2['y'], 0.5
        
        vx = (p2['x'] - p1['x']) / dt
        vy = (p2['y'] - p1['y']) / dt
        
        # Predict future position
        dt_future = future_timestamp - p2['timestamp']
        pred_x = p2['x'] + vx * dt_future
        pred_y = p2['y'] + vy * dt_future
        
        # Confidence decreases with prediction distance
        confidence = max(0.1, 0.9 - abs(dt_future) * 0.5)
        
        return pred_x, pred_y, confidence
    
    def _predict_polynomial(self, history, future_timestamp):
        """Polynomial fitting for curved trajectories"""
        if len(history) < 3:
            return self._predict_linear(history, future_timestamp)
        
        # Extract time and position data
        times = np.array([p['timestamp'] for p in history])
        x_coords = np.array([p['x'] for p in history])
        y_coords = np.array([p['y'] for p in history])
        
        # Fit polynomial (degree 2 for parabolic motion)
        try:
            # Normalize time to avoid numerical issues
            t_norm = times - times[0]
            t_future_norm = future_timestamp - times[0]
            
            # Fit polynomials
            degree = min(2, len(history) - 1)
            px = np.polyfit(t_norm, x_coords, degree)
            py = np.polyfit(t_norm, y_coords, degree)
            
            # Predict
            pred_x = np.polyval(px, t_future_norm)
            pred_y = np.polyval(py, t_future_norm)
            
            # Calculate confidence based on fit quality
            x_fit = np.polyval(px, t_norm)
            y_fit = np.polyval(py, t_norm)
            x_error = np.mean((x_coords - x_fit) ** 2)
            y_error = np.mean((y_coords - y_fit) ** 2)
            confidence = max(0.1, 0.9 - (x_error + y_error) / 1000)
            
            return pred_x, pred_y, confidence
            
        except np.linalg.LinAlgError:
            return self._predict_linear(history, future_timestamp)
    
    def _predict_kalman(self, track_id, history, future_timestamp):
        """Kalman filter prediction for smooth tracking"""
        # Initialize Kalman filter if not exists
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._create_kalman_filter()
        
        kf = self.kalman_filters[track_id]
        
        # Update Kalman filter with recent measurements
        latest = history[-1]
        measurement = np.array([[latest['x']], [latest['y']]])
        
        # Predict step
        kf.predict()
        
        # Update step
        kf.update(measurement)
        
        # Predict future position
        current_time = latest['timestamp']
        dt = future_timestamp - current_time
        
        # Create prediction transition matrix
        F_pred = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Predict state
        pred_state = F_pred @ kf.x
        pred_x, pred_y = pred_state[0, 0], pred_state[1, 0]
        
        # Confidence based on uncertainty
        pred_cov = F_pred @ kf.P @ F_pred.T
        uncertainty = np.trace(pred_cov[:2, :2])  # Position uncertainty
        confidence = max(0.1, 0.9 - uncertainty / 10000)
        
        return pred_x, pred_y, confidence
    
    def _create_kalman_filter(self):
        """Create a Kalman filter for constant velocity model"""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        dt = 1/30  # Assume 30 FPS
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R *= 10  # Measurement uncertainty
        
        # Process noise
        kf.Q = np.eye(4) * 0.1
        
        # Initial covariance
        kf.P *= 100
        
        return kf
    
    def get_interpolated_trajectory(self, track_id, start_time, end_time, num_points=10):
        """Generate interpolated trajectory between two timestamps"""
        if track_id not in self.track_histories:
            return []
        
        history = list(self.track_histories[track_id])
        if len(history) < 2:
            return []
        
        # Generate time points
        time_points = np.linspace(start_time, end_time, num_points)
        interpolated = []
        
        for t in time_points:
            pred_x, pred_y, confidence = self.predict_position(track_id, t)
            if pred_x is not None:
                interpolated.append({
                    'x': pred_x,
                    'y': pred_y,
                    'timestamp': t,
                    'confidence': confidence,
                    'predicted': True
                })
        
        return interpolated


class HighSpeedTracker:
    """
    Custom high-speed tracker that works better than ByteTracker for fast objects
    Uses prediction-assisted association and larger search windows
    """
    
    def __init__(self, max_disappeared=30, max_distance=100, prediction_weight=0.7):
        """
        Initialize high-speed tracker
        
        Args:
            max_disappeared (int): Max frames a track can be missing before deletion
            max_distance (float): Maximum distance for association
            prediction_weight (float): Weight given to predicted positions vs raw IoU
        """
        self.next_id = 1
        self.tracks = {}  # track_id: track_info
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.prediction_weight = prediction_weight
        
    def update(self, detections, predictor=None, timestamp=None):
        """
        Update tracks with new detections
        
        Args:
            detections: sv.Detections object
            predictor: PuckPredictor for prediction-assisted tracking
            timestamp: Current timestamp
            
        Returns:
            sv.Detections with tracker_id assigned
        """
        if len(detections) == 0:
            # Age all tracks
            self._age_tracks()
            return sv.Detections.empty()
        
        # Convert detections to format we can work with
        detection_centers = []
        detection_bboxes = []
        detection_confidences = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detection_centers.append((center_x, center_y))
            detection_bboxes.append(bbox)
            detection_confidences.append(
                detections.confidence[i] if detections.confidence is not None else 0.5
            )
        
        # Associate detections with existing tracks
        matched_tracks, unmatched_detections = self._associate_detections(
            detection_centers, detection_bboxes, detection_confidences, 
            predictor, timestamp
        )
        
        # Update matched tracks
        for track_id, detection_idx in matched_tracks:
            self.tracks[track_id]['center'] = detection_centers[detection_idx]
            self.tracks[track_id]['bbox'] = detection_bboxes[detection_idx]
            self.tracks[track_id]['confidence'] = detection_confidences[detection_idx]
            self.tracks[track_id]['disappeared'] = 0
            self.tracks[track_id]['last_seen'] = timestamp
            
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            
            self.tracks[track_id] = {
                'center': detection_centers[detection_idx],
                'bbox': detection_bboxes[detection_idx],
                'confidence': detection_confidences[detection_idx],
                'disappeared': 0,
                'last_seen': timestamp,
                'created': timestamp
            }
            print(f"Created new high-speed track: {track_id}")
        
        # Age unmatched tracks
        self._age_tracks()
        
        # Create output detections with track IDs
        return self._create_output_detections(matched_tracks, unmatched_detections, 
                                            detection_bboxes, detection_confidences)
    
    def _associate_detections(self, detection_centers, detection_bboxes, detection_confidences,
                            predictor, timestamp):
        """Associate detections with tracks using prediction-assisted matching"""
        
        if len(self.tracks) == 0:
            # No existing tracks, all detections are unmatched
            return [], list(range(len(detection_centers)))
        
        # Calculate cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detection_centers)), np.inf)
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_center = track['center']
            
            # Get predicted position if predictor available
            predicted_center = None
            if predictor and timestamp:
                pred_x, pred_y, confidence = predictor.predict_position(track_id, timestamp)
                if pred_x is not None and confidence > 0.3:
                    predicted_center = (pred_x, pred_y)
            
            for j, det_center in enumerate(detection_centers):
                # Calculate distance cost
                if predicted_center:
                    # Use prediction-weighted distance
                    raw_distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                                         (track_center[1] - det_center[1])**2)
                    pred_distance = np.sqrt((predicted_center[0] - det_center[0])**2 + 
                                          (predicted_center[1] - det_center[1])**2)
                    
                    # Weighted combination
                    distance = (1 - self.prediction_weight) * raw_distance + \
                              self.prediction_weight * pred_distance
                else:
                    # Just use raw distance
                    distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                                     (track_center[1] - det_center[1])**2)
                
                # Additional costs for better matching
                bbox_cost = self._calculate_bbox_cost(track['bbox'], detection_bboxes[j])
                confidence_cost = abs(track['confidence'] - detection_confidences[j]) * 50
                
                # Combined cost
                total_cost = distance + bbox_cost + confidence_cost
                
                if total_cost < self.max_distance:
                    cost_matrix[i, j] = total_cost
        
        # Solve assignment problem using simple greedy approach (faster than Hungarian)
        matched_tracks = []
        unmatched_detections = list(range(len(detection_centers)))
        unmatched_tracks = list(range(len(track_ids)))
        
        # Greedy matching - assign lowest cost first
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            # Find minimum cost
            min_cost = np.inf
            min_track_idx = -1
            min_det_idx = -1
            
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_track_idx = i
                        min_det_idx = j
            
            # If we found a valid match
            if min_cost < self.max_distance:
                track_id = track_ids[min_track_idx]
                matched_tracks.append((track_id, min_det_idx))
                unmatched_tracks.remove(min_track_idx)
                unmatched_detections.remove(min_det_idx)
            else:
                break
        
        return matched_tracks, unmatched_detections
    
    def _calculate_bbox_cost(self, bbox1, bbox2):
        """Calculate cost based on bounding box similarity"""
        # Calculate IoU
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 100  # No overlap, high cost
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return (1 - iou) * 50  # Convert IoU to cost
    
    def _age_tracks(self):
        """Age tracks and remove old ones"""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            track['disappeared'] += 1
            if track['disappeared'] > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            print(f"Removing aged track: {track_id}")
            del self.tracks[track_id]
    
    def _create_output_detections(self, matched_tracks, unmatched_detections, 
                                detection_bboxes, detection_confidences):
        """Create output detections with track IDs"""
        if len(matched_tracks) == 0 and len(unmatched_detections) == 0:
            return sv.Detections.empty()
        
        # Collect all detections (matched + unmatched)
        output_bboxes = []
        output_confidences = []
        output_track_ids = []
        
        # Add matched detections
        for track_id, detection_idx in matched_tracks:
            output_bboxes.append(detection_bboxes[detection_idx])
            output_confidences.append(detection_confidences[detection_idx])
            output_track_ids.append(track_id)
        
        # Add unmatched detections (new tracks)
        for detection_idx in unmatched_detections:
            # Find the track ID we just created for this detection
            # This is a bit hacky but works for our purposes
            for track_id, track in self.tracks.items():
                if (track['bbox'] == detection_bboxes[detection_idx]).all():
                    output_bboxes.append(detection_bboxes[detection_idx])
                    output_confidences.append(detection_confidences[detection_idx])
                    output_track_ids.append(track_id)
                    break
        
        if len(output_bboxes) == 0:
            return sv.Detections.empty()
        
        # Create detections object
        detections = sv.Detections(
            xyxy=np.array(output_bboxes),
            confidence=np.array(output_confidences)
        )
        detections.tracker_id = np.array(output_track_ids)
        
        return detections


class OnlineHockeyPuckTracker:
    """
    Online hockey puck tracker using YOLO detection + ByteTracker
    Processes video frame-by-frame and maintains real-time trajectory tracking
    """
    
    def __init__(self, yolo_model_path, confidence_threshold=0.1, 
                 enable_prediction=True, prediction_method='kalman', use_high_speed_tracker=True):
        """
        Initialize the online tracker with prediction capabilities
        
        Args:
            yolo_model_path (str): Path to YOLO model file (.pt)
            confidence_threshold (float): Minimum confidence for detections
            enable_prediction (bool): Enable position prediction
            prediction_method (str): Method for prediction ('linear', 'polynomial', 'kalman')
            use_high_speed_tracker (bool): Use custom high-speed tracker instead of ByteTracker
        """
        print(f"🚀 Initializing Online Hockey Puck Tracker with Prediction")
        print(f"   YOLO Model: {yolo_model_path}")
        print(f"   Confidence Threshold: {confidence_threshold}")
        print(f"   Prediction Enabled: {enable_prediction}")
        print(f"   Prediction Method: {prediction_method if enable_prediction else 'N/A'}")
        print(f"   High-Speed Tracker: {use_high_speed_tracker}")
        
        # Load YOLO model
        try:
            self.model = YOLO(yolo_model_path)
            print(f"✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
        
        # Initialize prediction system
        self.enable_prediction = enable_prediction
        self.predictor = PuckPredictor(method=prediction_method) if enable_prediction else None
        self.use_high_speed_tracker = use_high_speed_tracker
        
        # Initialize tracker based on user preference
        if use_high_speed_tracker:
            # Use our custom high-speed tracker
            self.tracker = HighSpeedTracker(
                max_disappeared=60,  # 2 seconds at 30 FPS
                max_distance=150,    # Larger search radius for high-speed objects
                prediction_weight=0.8 if enable_prediction else 0.0
            )
            print("✅ Using custom high-speed tracker")
        else:
            # Use ByteTracker with enhanced settings for prediction
            if enable_prediction:
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=confidence_threshold * 0.3,
                    lost_track_buffer=180,
                    minimum_matching_threshold=0.4,
                    frame_rate=30,
                    minimum_consecutive_frames=1
                )
            else:
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=confidence_threshold * 0.5,
                    lost_track_buffer=120,
                    minimum_matching_threshold=0.6,
                    frame_rate=30,
                    minimum_consecutive_frames=1
                )
            print("✅ Using ByteTracker")
        
        self.confidence_threshold = confidence_threshold
        
        # Trajectory storage
        self.trajectories = defaultdict(list)  # track_id: list of trajectory points
        self.trajectory_metadata = defaultdict(dict)  # track_id: metadata
        
        # Prediction-specific storage
        self.last_positions = {}  # track_id: last known position for prediction
        self.prediction_cache = {}  # frame_num: predicted detections
        
        # Track all detections for debugging
        self.all_detections = []  # For debugging purposes
        
        # Video processing state
        self.frame_count = 0
        self.video_fps = 30
        self.video_width = 0
        self.video_height = 0
        
        # Visualization storage
        self.annotated_frames = []
        self.save_annotated_video = True
        
        print(f"✅ Enhanced tracker initialized successfully")
    
    def process_video(self, video_path, output_dir="hockey_tracking_results", 
                     save_frames=True, save_video=True, max_frames=None):
        """
        Process entire video and track pucks online
        
        Args:
            video_path (str): Path to input MP4 video
            output_dir (str): Directory to save results
            save_frames (bool): Whether to save annotated frames
            save_video (bool): Whether to save annotated video
            max_frames (int): Maximum frames to process (None = all)
        
        Returns:
            dict: Complete trajectory data
        """
        print(f"\n🎬 Processing Video: {video_path}")
        print("=" * 60)
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Properties:")
        print(f"   Resolution: {self.video_width}x{self.video_height}")
        print(f"   FPS: {self.video_fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Duration: {total_frames/self.video_fps:.1f}s")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            print(f"   Processing: {total_frames} frames (limited)")
        
        # Setup video writer for annotated output
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video_path = output_path / "annotated_hockey_tracking.mp4"
            video_writer = cv2.VideoWriter(
                str(out_video_path), fourcc, self.video_fps,
                (self.video_width, self.video_height)
            )
        
        # Process frames
        start_time = time.time()
        last_print_time = start_time
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and self.frame_count >= max_frames:
                    break
                
                # Calculate timestamp
                timestamp = self.frame_count / self.video_fps
                
                # Process frame
                annotated_frame = self.process_frame(frame, timestamp)
                
                # Save annotated frame
                if save_video and video_writer:
                    video_writer.write(annotated_frame)
                
                if save_frames and self.frame_count % 30 == 0:  # Save every 30th frame
                    frame_path = output_path / f"frame_{self.frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                
                self.frame_count += 1
                
                # Progress update
                current_time = time.time()
                if current_time - last_print_time > 2.0:  # Update every 2 seconds
                    progress = (self.frame_count / total_frames) * 100
                    fps_processing = self.frame_count / (current_time - start_time)
                    active_tracks = len(self.get_active_trajectories(timestamp))
                    total_detections = len([d for d in self.all_detections if d['frame_num'] == self.frame_count - 1])
                    
                    print(f"   Progress: {progress:.1f}% | Frame {self.frame_count}/{total_frames} | "
                          f"Processing FPS: {fps_processing:.1f} | Active Tracks: {active_tracks} | "
                          f"Current Detections: {total_detections}")
                    last_print_time = current_time
        
        finally:
            cap.release()
            if save_video and video_writer:
                video_writer.release()
        
        processing_time = time.time() - start_time
        print(f"\n✅ Video Processing Complete!")
        print(f"   Total Frames Processed: {self.frame_count}")
        print(f"   Processing Time: {processing_time:.1f}s")
        print(f"   Average FPS: {self.frame_count/processing_time:.1f}")
        print(f"   Total Trajectories Found: {len(self.trajectories)}")
        print(f"   Total Detections: {len(self.all_detections)}")
        
        # Save trajectory data
        self.save_trajectory_data(output_path)
        
        return self.trajectories
    
    def process_frame(self, frame, timestamp):
        """
        Process a single frame with enhanced prediction capabilities
        
        Args:
            frame (np.ndarray): Current video frame
            timestamp (float): Frame timestamp in seconds
        
        Returns:
            np.ndarray: Annotated frame
        """
        # YOLO detection
        results = self.model.predict(
            frame, 
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Convert to supervision format and check for detections
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Debug: Store all raw detections
        for i in range(len(detections)):
            detection_data = {
                'frame_num': self.frame_count,
                'timestamp': timestamp,
                'bbox': detections.xyxy[i],
                'confidence': detections.confidence[i] if detections.confidence is not None else 0.0,
                'class_id': detections.class_id[i] if detections.class_id is not None else 0,
                'predicted': False
            }
            self.all_detections.append(detection_data)
        
        print(f"Frame {self.frame_count}: Found {len(detections)} raw detections")
        
        # Add predicted detections when no real detections found
        if self.enable_prediction and len(detections) == 0:
            predicted_detections = self._generate_predicted_detections(timestamp)
            if len(predicted_detections) > 0:
                detections = predicted_detections
                print(f"Frame {self.frame_count}: Using {len(predicted_detections)} predicted detections")
        
        # Enhance low-confidence real detections with predictions
        elif self.enable_prediction and len(detections) > 0:
            enhanced_detections = self._enhance_detections_with_predictions(detections, timestamp)
            if len(enhanced_detections) > len(detections):
                detections = enhanced_detections
                print(f"Frame {self.frame_count}: Enhanced with predictions: {len(enhanced_detections)} total")
        
        # Update tracker based on type
        if len(detections) > 0:
            try:
                if self.use_high_speed_tracker:
                    # Use custom high-speed tracker with prediction support
                    detections = self.tracker.update(detections, self.predictor, timestamp)
                else:
                    # Use standard ByteTracker
                    detections = self.tracker.update_with_detections(detections)
                
                print(f"Frame {self.frame_count}: Tracker returned {len(detections)} tracked objects")
                
                # Check if tracker_id exists
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    for i, track_id in enumerate(detections.tracker_id):
                        print(f"  Track ID: {track_id}")
                else:
                    print("  No tracker IDs assigned")
                    
            except Exception as e:
                print(f"Tracking error on frame {self.frame_count}: {e}")
                import traceback
                traceback.print_exc()
                # Create empty detections to continue processing
                detections = sv.Detections.empty()
        else:
            # No detections, update tracker with empty detections
            if self.use_high_speed_tracker:
                detections = self.tracker.update(sv.Detections.empty(), self.predictor, timestamp)
            else:
                detections = self.tracker.update_with_detections(sv.Detections.empty())
        
        # Update trajectories with prediction support
        self.update_trajectories_with_prediction(detections, timestamp)
        
        # Create annotated frame with prediction visualization
        annotated_frame = self.annotate_frame_with_predictions(frame, detections, timestamp)
        
        return annotated_frame
    
    def _generate_predicted_detections(self, timestamp):
        """Generate predicted detections for tracks that might be temporarily lost"""
        if not self.enable_prediction:
            return sv.Detections.empty()
        
        predicted_bboxes = []
        predicted_confidences = []
        predicted_class_ids = []
        
        # Look for recently active tracks
        for track_id, last_pos in self.last_positions.items():
            time_since_last = timestamp - last_pos['timestamp']
            
            # Only predict for recently seen tracks (within 0.5 seconds)
            if time_since_last > 0.5:
                continue
            
            # Get prediction
            pred_x, pred_y, confidence = self.predictor.predict_position(track_id, timestamp)
            
            if pred_x is not None and confidence > 0.3:
                # Create bounding box around predicted center
                bbox_size = last_pos.get('bbox_size', 20)  # Default size
                x1 = pred_x - bbox_size // 2
                y1 = pred_y - bbox_size // 2
                x2 = pred_x + bbox_size // 2
                y2 = pred_y + bbox_size // 2
                
                # Ensure bbox is within frame bounds
                x1 = max(0, min(x1, self.video_width - 1))
                y1 = max(0, min(y1, self.video_height - 1))
                x2 = max(1, min(x2, self.video_width))
                y2 = max(1, min(y2, self.video_height))
                
                predicted_bboxes.append([x1, y1, x2, y2])
                predicted_confidences.append(confidence)
                predicted_class_ids.append(0)  # Assume class 0 for puck
                
                # Store predicted detection for debugging
                prediction_data = {
                    'frame_num': self.frame_count,
                    'timestamp': timestamp,
                    'bbox': np.array([x1, y1, x2, y2]),
                    'confidence': confidence,
                    'class_id': 0,
                    'predicted': True,
                    'track_id': track_id
                }
                self.all_detections.append(prediction_data)
                
                print(f"  Generated prediction for track {track_id}: ({pred_x:.1f}, {pred_y:.1f}) conf={confidence:.2f}")
        
        if len(predicted_bboxes) == 0:
            return sv.Detections.empty()
        
        # Create detections object
        detections = sv.Detections(
            xyxy=np.array(predicted_bboxes),
            confidence=np.array(predicted_confidences),
            class_id=np.array(predicted_class_ids)
        )
        
        return detections
    
    def _enhance_detections_with_predictions(self, detections, timestamp):
        """Enhance existing detections with predictions for missing tracks"""
        if not self.enable_prediction or len(detections) == 0:
            return detections
        
        # Get current detection centers for comparison
        current_centers = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            current_centers.append((center_x, center_y))
        
        # Check for missing tracks that should be predicted
        predicted_bboxes = []
        predicted_confidences = []
        predicted_class_ids = []
        
        for track_id, last_pos in self.last_positions.items():
            time_since_last = timestamp - last_pos['timestamp']
            
            # Only predict for recently seen tracks
            if time_since_last > 0.3:
                continue
            
            # Get prediction
            pred_x, pred_y, confidence = self.predictor.predict_position(track_id, timestamp)
            
            if pred_x is not None and confidence > 0.4:
                # Check if this prediction is far from existing detections
                min_distance = float('inf')
                for center in current_centers:
                    distance = np.sqrt((pred_x - center[0])**2 + (pred_y - center[1])**2)
                    min_distance = min(min_distance, distance)
                
                # Only add prediction if it's not too close to existing detections
                if min_distance > 30:  # pixels
                    bbox_size = last_pos.get('bbox_size', 20)
                    x1 = max(0, pred_x - bbox_size // 2)
                    y1 = max(0, pred_y - bbox_size // 2)
                    x2 = min(self.video_width, pred_x + bbox_size // 2)
                    y2 = min(self.video_height, pred_y + bbox_size // 2)
                    
                    predicted_bboxes.append([x1, y1, x2, y2])
                    predicted_confidences.append(confidence * 0.8)  # Reduce confidence for predictions
                    predicted_class_ids.append(0)
                    
                    print(f"  Enhanced with prediction for track {track_id}: ({pred_x:.1f}, {pred_y:.1f})")
        
        # Combine original and predicted detections
        if len(predicted_bboxes) > 0:
            all_bboxes = np.vstack([detections.xyxy, np.array(predicted_bboxes)])
            all_confidences = np.concatenate([
                detections.confidence if detections.confidence is not None else np.ones(len(detections)) * 0.5,
                np.array(predicted_confidences)
            ])
            all_class_ids = np.concatenate([
                detections.class_id if detections.class_id is not None else np.zeros(len(detections)),
                np.array(predicted_class_ids)
            ])
            
            enhanced_detections = sv.Detections(
                xyxy=all_bboxes,
                confidence=all_confidences,
                class_id=all_class_ids
            )
            
            return enhanced_detections
        
        return detections
    
    def update_trajectories_with_prediction(self, detections, timestamp):
        """Enhanced trajectory update with prediction history"""
        if not hasattr(detections, 'tracker_id') or detections.tracker_id is None or len(detections) == 0:
            return
            
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            bbox = detections.xyxy[i]
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            
            # Calculate center point
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_size = max(x2 - x1, y2 - y1)
            
            # Update predictor history
            if self.enable_prediction:
                self.predictor.update_history(track_id, center_x, center_y, timestamp)
            
            # Store last known position for future predictions
            self.last_positions[track_id] = {
                'x': center_x,
                'y': center_y,
                'timestamp': timestamp,
                'bbox_size': bbox_size
            }
            
            # Determine if this was a predicted detection
            is_predicted = False
            # Check if this detection matches any of our predictions
            for detection_data in self.all_detections:
                if (detection_data['frame_num'] == self.frame_count and 
                    detection_data.get('predicted', False) and
                    detection_data.get('track_id') == track_id):
                    is_predicted = True
                    break
            
            # Add to trajectory
            trajectory_point = {
                'x': float(center_x),
                'y': float(center_y),
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'timestamp': timestamp,
                'frame_num': self.frame_count,
                'predicted': is_predicted
            }
            
            self.trajectories[track_id].append(trajectory_point)
            
            # Update metadata
            if track_id not in self.trajectory_metadata:
                self.trajectory_metadata[track_id] = {
                    'start_time': timestamp,
                    'start_frame': self.frame_count,
                    'color': self.generate_color(track_id)
                }
                print(f"New track started: ID {track_id}")
            
            self.trajectory_metadata[track_id]['end_time'] = timestamp
            self.trajectory_metadata[track_id]['end_frame'] = self.frame_count
            
            # Calculate center point
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Add to trajectory
            self.trajectories[track_id].append({
                'x': float(center_x),
                'y': float(center_y),
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'timestamp': timestamp,
                'frame_num': self.frame_count
            })
            
            # Update metadata
            if track_id not in self.trajectory_metadata:
                self.trajectory_metadata[track_id] = {
                    'start_time': timestamp,
                    'start_frame': self.frame_count,
                    'color': self.generate_color(track_id)
                }
                print(f"New track started: ID {track_id}")
            
            self.trajectory_metadata[track_id]['end_time'] = timestamp
            self.trajectory_metadata[track_id]['end_frame'] = self.frame_count
    
    def annotate_frame_with_predictions(self, frame, detections, timestamp):
        """Enhanced annotation showing predicted vs real detections"""
        annotated = frame.copy()
        
        # Draw current detections
        if len(detections) > 0 and detections.tracker_id is not None:
            for i in range(len(detections)):
                track_id = detections.tracker_id[i]
                bbox = detections.xyxy[i]
                confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                
                x1, y1, x2, y2 = bbox.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                color = self.trajectory_metadata[track_id]['color']
                
                # Check if this is a predicted detection
                is_predicted = False
                for detection in self.all_detections:
                    if (detection['frame_num'] == self.frame_count and 
                        detection.get('predicted', False) and
                        detection.get('track_id') == track_id):
                        is_predicted = True
                        break
                
                # Draw bounding box (dashed for predictions)
                if is_predicted:
                    # Draw dashed rectangle for predictions
                    self._draw_dashed_rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} PRED ({confidence:.2f})"
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} ({confidence:.2f})"
                
                # Draw center point
                cv2.circle(annotated, (center_x, center_y), 4, color, -1)
                
                # Draw label
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectories with prediction indicators
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            color = self.trajectory_metadata[track_id]['color']
            
            # Separate real and predicted points
            real_points = [(int(p['x']), int(p['y'])) for p in trajectory if not p.get('predicted', False)]
            pred_points = [(int(p['x']), int(p['y'])) for p in trajectory if p.get('predicted', False)]
            
            # Draw real trajectory (solid lines)
            if len(real_points) > 1:
                for i in range(1, len(real_points)):
                    cv2.line(annotated, real_points[i-1], real_points[i], color, 2)
            
            # Draw predicted trajectory (dashed lines)
            if len(pred_points) > 1:
                for i in range(1, len(pred_points)):
                    self._draw_dashed_line(annotated, pred_points[i-1], pred_points[i], color, 2)
            
            # Draw ALL points in trajectory as a continuous line (mixed real and predicted)
            all_points = [(int(p['x']), int(p['y'])) for p in trajectory]
            if len(all_points) > 1:
                # Draw start point (green circle)
                cv2.circle(annotated, all_points[0], 6, (0, 255, 0), -1)
                cv2.circle(annotated, all_points[0], 6, (0, 0, 0), 1)
                
                # Draw end point (red circle) 
                cv2.circle(annotated, all_points[-1], 6, (0, 0, 255), -1)
                cv2.circle(annotated, all_points[-1], 6, (0, 0, 0), 1)
                
                # Draw direction arrow at midpoint
                if len(all_points) > 2:
                    mid_idx = len(all_points) // 2
                    if mid_idx < len(all_points) - 1:
                        dx = all_points[mid_idx + 1][0] - all_points[mid_idx][0]
                        dy = all_points[mid_idx + 1][1] - all_points[mid_idx][1]
                        if dx != 0 or dy != 0:  # Avoid zero-length arrow
                            length = np.sqrt(dx*dx + dy*dy)
                            if length > 0:
                                dx = dx / length * 20  # Normalize and scale
                                dy = dy / length * 20
                                cv2.arrowedLine(annotated, all_points[mid_idx], 
                                              (int(all_points[mid_idx][0] + dx), int(all_points[mid_idx][1] + dy)), 
                                              color, 2, tipLength=0.3)
            
            # Draw prediction indicators
            if self.enable_prediction and track_id in self.last_positions:
                last_pos = self.last_positions[track_id]
                
                # Draw prediction cone/uncertainty for future position
                pred_x, pred_y, confidence = self.predictor.predict_position(track_id, timestamp + 0.1)
                if pred_x is not None:
                    future_point = (int(pred_x), int(pred_y))
                    current_point = (int(last_pos['x']), int(last_pos['y']))
                    
                    # Draw prediction vector
                    self._draw_dashed_line(annotated, current_point, future_point, 
                                         (255, 255, 0), 1)  # Yellow for prediction
                    cv2.circle(annotated, future_point, 3, (255, 255, 0), -1)
        
        # Enhanced frame info
        total_trajectories = len(self.trajectories)
        active_trajectories = len([t for t in self.trajectories.values() 
                                 if len(t) > 0 and abs(t[-1]['timestamp'] - timestamp) < 1.0])
        predicted_count = len([d for d in self.all_detections 
                             if d['frame_num'] == self.frame_count and d.get('predicted', False)])
        
        info_text = f"Frame: {self.frame_count} | Time: {timestamp:.2f}s | Current: {len(detections)} | Predicted: {predicted_count} | Total Tracks: {total_trajectories} | Active: {active_trajectories}"
        
        # Draw info with background
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Enhanced legend
        legend_y = 60
        legend_text = "Legend: Green=Start, Red=End, Solid=Real, Dashed=Predicted, Yellow=Future"
        cv2.putText(annotated, legend_text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated, legend_text, (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
    
    def _draw_dashed_line(self, img, pt1, pt2, color, thickness):
        """Draw a dashed line"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist == 0:
            return
        pts = []
        for i in np.arange(0, dist, 5):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
            p = (x, y)
            pts.append(p)
        
        for p in pts[::2]:  # Draw every other point for dashed effect
            cv2.circle(img, p, thickness, color, -1)
    
    def _draw_dashed_rectangle(self, img, pt1, pt2, color, thickness):
        """Draw a dashed rectangle"""
        # Top line
        self._draw_dashed_line(img, pt1, (pt2[0], pt1[1]), color, thickness)
        # Right line
        self._draw_dashed_line(img, (pt2[0], pt1[1]), pt2, color, thickness)
        # Bottom line
        self._draw_dashed_line(img, pt2, (pt1[0], pt2[1]), color, thickness)
        # Left line
        self._draw_dashed_line(img, (pt1[0], pt2[1]), pt1, color, thickness)
    
    def fill_trajectory_gaps(self, max_gap_duration=0.5):
        """Fill gaps in trajectories using prediction"""
        if not self.enable_prediction:
            return
        
        print(f"\n🔧 Filling trajectory gaps using prediction")
        
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            filled_trajectory = []
            
            for i in range(len(trajectory)):
                filled_trajectory.append(trajectory[i])
                
                # Check for gaps to next point
                if i < len(trajectory) - 1:
                    current_time = trajectory[i]['timestamp']
                    next_time = trajectory[i + 1]['timestamp']
                    gap_duration = next_time - current_time
                    
                    # Fill significant gaps
                    if gap_duration > max_gap_duration:
                        print(f"   Filling {gap_duration:.2f}s gap in track {track_id}")
                        
                        # Generate interpolated points
                        num_points = max(2, int(gap_duration * self.video_fps))
                        interpolated = self.predictor.get_interpolated_trajectory(
                            track_id, current_time, next_time, num_points
                        )
                        
                        # Add interpolated points (excluding endpoints)
                        for interp_point in interpolated[1:-1]:
                            interp_point['frame_num'] = int(interp_point['timestamp'] * self.video_fps)
                            interp_point['bbox'] = trajectory[i]['bbox']  # Use previous bbox
                            filled_trajectory.append(interp_point)
            
            # Update trajectory
            self.trajectories[track_id] = filled_trajectory
        
        print(f"✅ Trajectory gap filling complete")
    
    def generate_color(self, track_id):
        """Generate consistent color for track ID"""
        np.random.seed(int(track_id) % 2147483647)  # Ensure it's within int32 range
        color = tuple(map(int, np.random.randint(50, 255, 3)))
        return color
    
    def get_active_trajectories(self, current_time, timeout=3.0):
        """Get trajectories that are currently active"""
        active = {}
        for track_id, trajectory in self.trajectories.items():
            if trajectory and (current_time - trajectory[-1]['timestamp']) <= timeout:
                active[track_id] = trajectory
        return active
    
    def filter_trajectories(self, min_length=5, min_duration=0.2, min_movement=10):
        """Filter trajectories based on quality criteria - more permissive defaults"""
        print(f"\n🔍 Filtering Trajectories")
        print(f"   Min Length: {min_length} points")
        print(f"   Min Duration: {min_duration}s")
        print(f"   Min Movement: {min_movement} pixels")
        
        filtered = {}
        rejected = 0
        
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < min_length:
                print(f"   Rejected Track {track_id}: Too short ({len(trajectory)} < {min_length})")
                rejected += 1
                continue
            
            duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
            if duration < min_duration:
                print(f"   Rejected Track {track_id}: Too brief ({duration:.2f}s < {min_duration}s)")
                rejected += 1
                continue
            
            # Calculate total movement
            start_x, start_y = trajectory[0]['x'], trajectory[0]['y']
            end_x, end_y = trajectory[-1]['x'], trajectory[-1]['y']
            movement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            if movement < min_movement:
                print(f"   Rejected Track {track_id}: Too little movement ({movement:.1f} < {min_movement})")
                rejected += 1
                continue
            
            # Calculate additional metrics
            avg_confidence = np.mean([p['confidence'] for p in trajectory])
            velocity = movement / duration if duration > 0 else 0
            
            # Update metadata
            self.trajectory_metadata[track_id].update({
                'length': len(trajectory),
                'duration': duration,
                'movement': movement,
                'avg_confidence': avg_confidence,
                'velocity': velocity
            })
            
            print(f"   Kept Track {track_id}: {duration:.2f}s, {len(trajectory)} pts, {velocity:.1f} px/s")
            filtered[track_id] = trajectory
        
        print(f"✅ Kept {len(filtered)} trajectories")
        print(f"❌ Rejected {rejected} trajectories")
        
        self.trajectories = filtered
        return filtered
    
    def create_trajectory_plots(self, output_dir):
        """Create comprehensive trajectory visualizations"""
        output_path = Path(output_dir)
        
        print(f"\n📊 Creating Trajectory Visualizations")
        print(f"   Output Directory: {output_path}")
        
        if not self.trajectories:
            print("❌ No trajectories to plot!")
            return
        
        # 1. Overview plot - all trajectories
        self.plot_all_trajectories(output_path)
        
        # 2. Individual trajectory plots
        self.plot_individual_trajectories(output_path)
        
        # 3. Trajectory statistics
        self.plot_trajectory_statistics(output_path)
        
        # 4. Time-based analysis
        self.plot_temporal_analysis(output_path)
        
        # 5. Detection analysis
        self.plot_detection_analysis(output_path)
    
    def plot_detection_analysis(self, output_path):
        """Plot detection vs tracking analysis"""
        if not self.all_detections:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract detection data
        frames = [d['frame_num'] for d in self.all_detections]
        confidences = [d['confidence'] for d in self.all_detections]
        timestamps = [d['timestamp'] for d in self.all_detections]
        
        # Plot 1: Detections per frame
        frame_counts = {}
        for frame in frames:
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        
        frame_nums = list(frame_counts.keys())
        detection_counts = list(frame_counts.values())
        
        ax1.plot(frame_nums, detection_counts, 'b-', alpha=0.7)
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detections per Frame')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence distribution
        ax2.hist(confidences, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.confidence_threshold})')
        ax2.set_xlabel('Detection Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Detection Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Detections vs Trajectories over time
        # Group detections by time bins
        time_bins = np.arange(0, max(timestamps) + 1, 1.0)  # 1-second bins
        detection_timeline = np.histogram(timestamps, bins=time_bins)[0]
        
        # Count trajectory points in same time bins
        trajectory_timeline = np.zeros_like(detection_timeline)
        for trajectory in self.trajectories.values():
            traj_times = [p['timestamp'] for p in trajectory]
            traj_counts = np.histogram(traj_times, bins=time_bins)[0]
            trajectory_timeline += traj_counts
        
        time_centers = time_bins[:-1] + 0.5
        ax3.plot(time_centers, detection_timeline, 'b-', label='Detections', linewidth=2)
        ax3.plot(time_centers, trajectory_timeline, 'r-', label='Tracked Points', linewidth=2)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Count per Second')
        ax3.set_title('Detections vs Tracked Points Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Tracking efficiency
        efficiency = []
        for i in range(len(detection_timeline)):
            if detection_timeline[i] > 0:
                eff = trajectory_timeline[i] / detection_timeline[i]
                efficiency.append(min(eff, 1.0))  # Cap at 100%
            else:
                efficiency.append(0)
        
        ax4.plot(time_centers, efficiency, 'purple', linewidth=2)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Tracking Efficiency (Tracked/Detected)')
        ax4.set_title('Tracking Efficiency Over Time')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        # Add summary statistics
        total_detections = len(self.all_detections)
        total_tracked_points = sum(len(traj) for traj in self.trajectories.values())
        overall_efficiency = total_tracked_points / total_detections if total_detections > 0 else 0
        
        summary_text = f"""Detection Analysis:
        Total Detections: {total_detections}
        Total Tracked Points: {total_tracked_points}
        Overall Efficiency: {overall_efficiency:.2%}
        Trajectories Created: {len(self.trajectories)}
        Avg Confidence: {np.mean(confidences):.3f}
        Confidence Range: {np.min(confidences):.3f} - {np.max(confidences):.3f}"""
        
        fig.text(0.02, 0.98, summary_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.15)
        plt.savefig(output_path / 'detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved detection analysis plot")
    
    def plot_all_trajectories(self, output_path):
        """Plot all trajectories on hockey rink overview"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot 1: Spatial trajectories
        ax1.set_xlim(0, self.video_width)
        ax1.set_ylim(self.video_height, 0)  # Invert Y to match video coordinates
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.set_title(f'All Puck Trajectories ({len(self.trajectories)} tracks)')
        ax1.grid(True, alpha=0.3)
        
        for track_id, trajectory in self.trajectories.items():
            x_coords = [p['x'] for p in trajectory]
            y_coords = [p['y'] for p in trajectory]
            
            color = [c/255.0 for c in self.trajectory_metadata[track_id]['color']]
            
            # Plot trajectory line
            ax1.plot(x_coords, y_coords, '-', color=color, linewidth=3, alpha=0.8, 
                    label=f'Track {track_id}')
            
            # Mark start and end
            ax1.plot(x_coords[0], y_coords[0], 'o', color='green', markersize=10, 
                    markeredgecolor='black', markeredgewidth=2)
            ax1.plot(x_coords[-1], y_coords[-1], 's', color='red', markersize=10,
                    markeredgecolor='black', markeredgewidth=2)
            
            # Add track ID at midpoint
            if len(trajectory) > 1:
                mid_idx = len(trajectory) // 2
                ax1.text(x_coords[mid_idx], y_coords[mid_idx], str(track_id), 
                        fontsize=12, fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add legend for start/end markers
        ax1.plot([], [], 'o', color='green', markersize=8, label='Start')
        ax1.plot([], [], 's', color='red', markersize=8, label='End')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Temporal view
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Track ID')
        ax2.set_title('Trajectory Timeline')
        
        for track_id, trajectory in self.trajectories.items():
            times = [p['timestamp'] for p in trajectory]
            track_ids = [track_id] * len(times)
            
            color = [c/255.0 for c in self.trajectory_metadata[track_id]['color']]
            ax2.scatter(times, track_ids, c=[color], s=20, alpha=0.6)
            
            # Draw duration line
            start_time = times[0]
            end_time = times[-1]
            ax2.plot([start_time, end_time], [track_id, track_id], 
                    color=color, linewidth=4, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'all_trajectories_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved overview plot: all_trajectories_overview.png")
    
    def plot_individual_trajectories(self, output_path):
        """Create separate detailed plots for each trajectory"""
        individual_dir = output_path / "individual_trajectories"
        individual_dir.mkdir(exist_ok=True)
        
        for track_id, trajectory in self.trajectories.items():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract data
            times = [p['timestamp'] for p in trajectory]
            x_coords = [p['x'] for p in trajectory]
            y_coords = [p['y'] for p in trajectory]
            confidences = [p['confidence'] for p in trajectory]
            
            color = [c/255.0 for c in self.trajectory_metadata[track_id]['color']]
            metadata = self.trajectory_metadata[track_id]
            
            # Plot 1: Spatial trajectory
            ax1.set_xlim(0, self.video_width)
            ax1.set_ylim(self.video_height, 0)
            ax1.set_xlabel('X Position (pixels)')
            ax1.set_ylabel('Y Position (pixels)')
            ax1.set_title(f'Track {track_id}: Spatial Trajectory')
            ax1.grid(True, alpha=0.3)
            
            # Color trajectory by time
            scatter = ax1.scatter(x_coords, y_coords, c=times, cmap='viridis', s=30, alpha=0.8)
            ax1.plot(x_coords, y_coords, '-', color=color, linewidth=2, alpha=0.6)
            
            # Mark start and end
            ax1.plot(x_coords[0], y_coords[0], 'go', markersize=10, label='Start')
            ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label='End')
            ax1.legend()
            
            plt.colorbar(scatter, ax=ax1, label='Time (s)')
            
            # Plot 2: X position over time
            ax2.plot(times, x_coords, 'b-', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('X Position (pixels)')
            ax2.set_title(f'Track {track_id}: X Position vs Time')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Y position over time
            ax3.plot(times, y_coords, 'r-', linewidth=2)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Y Position (pixels)')
            ax3.set_title(f'Track {track_id}: Y Position vs Time')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Confidence over time
            ax4.plot(times, confidences, 'g-', linewidth=2)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Detection Confidence')
            ax4.set_title(f'Track {track_id}: Confidence vs Time')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            # Add summary text
            summary_text = f"""Track {track_id} Summary:
            Duration: {metadata['duration']:.2f}s
            Length: {metadata['length']} detections
            Movement: {metadata['movement']:.1f} pixels
            Avg Velocity: {metadata['velocity']:.1f} px/s
            Avg Confidence: {metadata['avg_confidence']:.3f}
            Frames: {metadata['start_frame']}-{metadata['end_frame']}"""
            
            fig.text(0.02, 0.98, summary_text, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(left=0.15)  # Make room for summary text
            
            filename = f'trajectory_track_{track_id:03d}.png'
            plt.savefig(individual_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Saved {len(self.trajectories)} individual trajectory plots")
    
    def plot_trajectory_statistics(self, output_path):
        """Create statistical analysis plots"""
        if not self.trajectories:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract statistics
        durations = [self.trajectory_metadata[tid]['duration'] for tid in self.trajectories.keys()]
        lengths = [self.trajectory_metadata[tid]['length'] for tid in self.trajectories.keys()]
        movements = [self.trajectory_metadata[tid]['movement'] for tid in self.trajectories.keys()]
        velocities = [self.trajectory_metadata[tid]['velocity'] for tid in self.trajectories.keys()]
        
        # Plot 1: Duration histogram
        ax1.hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Number of Trajectories')
        ax1.set_title('Trajectory Duration Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Length histogram
        ax2.hist(lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Number of Detections')
        ax2.set_ylabel('Number of Trajectories')
        ax2.set_title('Trajectory Length Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Movement histogram
        ax3.hist(movements, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax3.set_xlabel('Total Movement (pixels)')
        ax3.set_ylabel('Number of Trajectories')
        ax3.set_title('Trajectory Movement Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Velocity histogram
        ax4.hist(velocities, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Average Velocity (pixels/second)')
        ax4.set_ylabel('Number of Trajectories')
        ax4.set_title('Trajectory Velocity Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Trajectory Statistics (n={len(self.trajectories)}):
        
        Duration:
          Mean: {np.mean(durations):.2f}s
          Median: {np.median(durations):.2f}s
          Range: {np.min(durations):.2f}-{np.max(durations):.2f}s
        
        Length:
          Mean: {np.mean(lengths):.1f} detections
          Median: {np.median(lengths):.1f} detections
          Range: {np.min(lengths)}-{np.max(lengths)} detections
        
        Movement:
          Mean: {np.mean(movements):.1f} pixels
          Median: {np.median(movements):.1f} pixels
          Range: {np.min(movements):.1f}-{np.max(movements):.1f} pixels
        
        Velocity:
          Mean: {np.mean(velocities):.1f} px/s
          Median: {np.median(velocities):.1f} px/s
          Range: {np.min(velocities):.1f}-{np.max(velocities):.1f} px/s"""
        
        fig.text(0.02, 0.98, stats_text, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.25)
        plt.savefig(output_path / 'trajectory_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved trajectory statistics plot")
    
    def plot_temporal_analysis(self, output_path):
        """Create temporal analysis plots"""
        if not self.trajectories:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Trajectories over time (Gantt chart style)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Track ID')
        ax1.set_title('Trajectory Timeline (Gantt Chart)')
        
        for i, (track_id, trajectory) in enumerate(self.trajectories.items()):
            start_time = trajectory[0]['timestamp']
            end_time = trajectory[-1]['timestamp']
            duration = end_time - start_time
            
            color = [c/255.0 for c in self.trajectory_metadata[track_id]['color']]
            
            # Draw timeline bar
            ax1.barh(track_id, duration, left=start_time, height=0.8, 
                    color=color, alpha=0.7, edgecolor='black')
            
            # Add duration text
            ax1.text(start_time + duration/2, track_id, f'{duration:.1f}s', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detection density over time
        all_times = []
        for trajectory in self.trajectories.values():
            all_times.extend([p['timestamp'] for p in trajectory])
        
        if all_times:
            ax2.hist(all_times, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Number of Detections')
            ax2.set_title('Detection Density Over Time')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved temporal analysis plot")
    
    def save_trajectory_data(self, output_path):
        """Save trajectory data to files"""
        # Save as JSON
        trajectory_data = {
            'video_info': {
                'width': self.video_width,
                'height': self.video_height,
                'fps': self.video_fps,
                'total_frames': self.frame_count
            },
            'tracking_stats': {
                'total_detections': len(self.all_detections),
                'total_trajectories': len(self.trajectories),
                'total_tracked_points': sum(len(traj) for traj in self.trajectories.values())
            },
            'trajectories': {},
            'metadata': dict(self.trajectory_metadata)
        }
        
        for track_id, trajectory in self.trajectories.items():
            # Convert all numpy types to Python types for JSON serialization
            clean_trajectory = []
            for point in trajectory:
                clean_point = {}
                for key, value in point.items():
                    if hasattr(value, 'item'):  # NumPy scalar
                        clean_point[key] = value.item()
                    elif isinstance(value, np.ndarray):  # NumPy array
                        clean_point[key] = value.tolist()
                    else:
                        clean_point[key] = value
                clean_trajectory.append(clean_point)
            trajectory_data['trajectories'][str(track_id)] = clean_trajectory
        
        json_path = output_path / 'trajectory_data.json'
        with open(json_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2, default=str)
        
        # Save detection data as well
        detection_data = {
            'detections': self.all_detections,
            'summary': {
                'total_detections': len(self.all_detections),
                'frames_with_detections': len(set(d['frame_num'] for d in self.all_detections)),
                'avg_detections_per_frame': len(self.all_detections) / self.frame_count if self.frame_count > 0 else 0
            }
        }
        
        detection_path = output_path / 'detection_data.json'
        with open(detection_path, 'w') as f:
            json.dump(detection_data, f, indent=2, default=str)
        
        print(f"✅ Saved trajectory data: {json_path}")
        print(f"✅ Saved detection data: {detection_path}")
    
    def print_summary(self):
        """Print comprehensive tracking summary"""
        print(f"\n" + "="*80)
        print(f"🏒 ONLINE HOCKEY PUCK TRACKING SUMMARY")
        print("="*80)
        
        print(f"📊 PROCESSING RESULTS:")
        print(f"   Video Resolution: {self.video_width}x{self.video_height}")
        print(f"   Total Frames Processed: {self.frame_count}")
        print(f"   Video Duration: {self.frame_count/self.video_fps:.1f} seconds")
        print(f"   Total Detections: {len(self.all_detections)}")
        print(f"   Total Trajectories: {len(self.trajectories)}")
        
        if self.all_detections:
            frames_with_detections = len(set(d['frame_num'] for d in self.all_detections))
            avg_detections_per_frame = len(self.all_detections) / self.frame_count if self.frame_count > 0 else 0
            print(f"   Frames with Detections: {frames_with_detections}/{self.frame_count} ({frames_with_detections/self.frame_count*100:.1f}%)")
            print(f"   Average Detections per Frame: {avg_detections_per_frame:.2f}")
        
        if not self.trajectories:
            print("❌ No trajectories found!")
            print("\n🔍 DEBUGGING SUGGESTIONS:")
            print("   1. Check if YOLO model is detecting objects (confidence threshold too high?)")
            print("   2. Verify ByteTracker settings (thresholds too strict?)")
            print("   3. Check video quality and puck visibility")
            print("   4. Try lowering confidence_threshold in constructor")
            return
        
        if self.trajectories:
            durations = [self.trajectory_metadata[tid]['duration'] for tid in self.trajectories.keys()]
            lengths = [self.trajectory_metadata[tid]['length'] for tid in self.trajectories.keys()]
            velocities = [self.trajectory_metadata[tid]['velocity'] for tid in self.trajectories.keys()]
            movements = [self.trajectory_metadata[tid]['movement'] for tid in self.trajectories.keys()]
            
            total_tracked_points = sum(len(traj) for traj in self.trajectories.values())
            tracking_efficiency = total_tracked_points / len(self.all_detections) if self.all_detections else 0
            
            print(f"\n🎯 TRAJECTORY STATISTICS:")
            print(f"   Duration - Min: {min(durations):.2f}s, Max: {max(durations):.2f}s, Avg: {np.mean(durations):.2f}s")
            print(f"   Length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f} detections")
            print(f"   Velocity - Min: {min(velocities):.1f}, Max: {max(velocities):.1f}, Avg: {np.mean(velocities):.1f} px/s")
            print(f"   Movement - Min: {min(movements):.1f}, Max: {max(movements):.1f}, Avg: {np.mean(movements):.1f} pixels")
            print(f"   Tracking Efficiency: {tracking_efficiency:.2%} (tracked points / total detections)")
            
            print(f"\n🏒 INDIVIDUAL TRAJECTORIES:")
            for track_id in sorted(self.trajectories.keys()):
                meta = self.trajectory_metadata[track_id]
                print(f"   Track {track_id:2d}: {meta['duration']:.2f}s, {meta['length']:3d} pts, "
                      f"{meta['velocity']:6.1f} px/s, frames {meta['start_frame']}-{meta['end_frame']}")


def main():
    """
    Enhanced main function with high-speed tracking capabilities
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python hockey_tracker.py <video_path> <yolo_model_path> [prediction_method] [tracker_type]")
        print("Prediction methods: linear, polynomial, kalman (default)")
        print("Tracker types: high_speed (default), bytetrack")
        print("Example: python hockey_tracker.py hockey_video.mp4 yolov8n.pt kalman high_speed")
        sys.exit(1)
    
    video_path = sys.argv[1]
    yolo_model_path = sys.argv[2]
    prediction_method = sys.argv[3] if len(sys.argv) > 3 else 'kalman'
    tracker_type = sys.argv[4] if len(sys.argv) > 4 else 'high_speed'
    
    # Verify files exist
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO model file not found: {yolo_model_path}")
        print("You can download a model with: from ultralytics import YOLO; YOLO('yolov8n.pt')")
        sys.exit(1)
    
    use_high_speed = tracker_type.lower() == 'high_speed'
    
    print("🚀 Starting Enhanced Hockey Puck Tracking with High-Speed Support")
    print("="*70)
    print(f"📹 Video: {video_path}")
    print(f"🤖 Model: {yolo_model_path}")
    print(f"🔮 Prediction: {prediction_method}")
    print(f"🏃 Tracker: {'High-Speed Custom' if use_high_speed else 'ByteTracker'}")
    print("="*70)
    
    # Create enhanced tracker with high-speed support
    tracker = OnlineHockeyPuckTracker(
        yolo_model_path=yolo_model_path,
        confidence_threshold=0.05,  # Lower threshold for better detection
        enable_prediction=True,
        prediction_method=prediction_method,
        use_high_speed_tracker=use_high_speed
    )
    
    # Process video
    trajectories = tracker.process_video(
        video_path=video_path,
        output_dir="high_speed_hockey_tracking_results",
        save_frames=True,
        save_video=True,
        max_frames=None  # Process entire video
    )
    
    # Fill gaps using prediction
    tracker.fill_trajectory_gaps(max_gap_duration=0.3)
    
    # Filter trajectories with more permissive settings
    filtered_trajectories = tracker.filter_trajectories(
        min_length=3,       # Even lower for high-speed tracking
        min_duration=0.1,   # Very short duration threshold
        min_movement=5      # Lower movement threshold
    )
    
    # Create visualizations
    tracker.create_trajectory_plots("high_speed_hockey_tracking_results")
    
    # Print summary
    tracker.print_summary()
    
    print(f"\n🎉 HIGH-SPEED TRACKING COMPLETE!")
    print(f"📁 Results saved to: high_speed_hockey_tracking_results/")
    print(f"📊 Check the following outputs:")
    print(f"   - annotated_hockey_tracking.mp4 (video with high-speed trajectories)")
    print(f"   - all_trajectories_overview.png (overview plot)")
    print(f"   - individual_trajectories/ (detailed plots for each track)")
    print(f"   - trajectory_statistics.png (statistical analysis)")
    print(f"   - temporal_analysis.png (time-based analysis)")
    print(f"   - detection_analysis.png (detection vs tracking analysis)")
    print(f"   - trajectory_data.json (raw trajectory data)")
    print(f"   - detection_data.json (raw detection data)")
    
    # Performance comparison if both trackers were tested
    if use_high_speed:
        print(f"\n🚀 HIGH-SPEED TRACKER BENEFITS:")
        print(f"   ✅ Better association for fast-moving objects")
        print(f"   ✅ Larger search radius (150px vs ByteTracker's ~50px)")
        print(f"   ✅ Prediction-assisted association")
        print(f"   ✅ Custom cost function optimized for pucks")
        print(f"   ✅ Reduced track fragmentation")
    
    return tracker, trajectories


# Standalone usage example with high-speed support
def example_usage():
    """
    Example of how to use the enhanced high-speed tracker programmatically
    """
    # Initialize enhanced tracker with high-speed tracking
    tracker = OnlineHockeyPuckTracker(
        yolo_model_path="yolov8n.pt",  # Download automatically if not present
        confidence_threshold=0.05,  # Lower threshold
        enable_prediction=True,
        prediction_method='kalman',  # Best prediction method
        use_high_speed_tracker=True  # Use high-speed tracker
    )
    
    # Process video
    trajectories = tracker.process_video(
        video_path="hockey_video.mp4",
        output_dir="my_high_speed_hockey_results",
        save_frames=True,
        save_video=True
    )
    
    # Fill trajectory gaps using prediction
    tracker.fill_trajectory_gaps(max_gap_duration=0.3)
    
    # Filter trajectories with high-speed friendly settings
    good_trajectories = tracker.filter_trajectories(
        min_length=5,       # Lower threshold for high-speed
        min_duration=0.2,   # Shorter duration
        min_movement=10     # Lower movement threshold
    )
    
    # Create all visualizations
    tracker.create_trajectory_plots("my_high_speed_hockey_results")
    
    # Access trajectory data programmatically
    for track_id, trajectory in good_trajectories.items():
        real_points = [p for p in trajectory if not p.get('predicted', False)]
        pred_points = [p for p in trajectory if p.get('predicted', False)]
        
        # Calculate velocity statistics
        if len(real_points) > 1:
            total_distance = 0
            for i in range(1, len(real_points)):
                dx = real_points[i]['x'] - real_points[i-1]['x']
                dy = real_points[i]['y'] - real_points[i-1]['y']
                total_distance += np.sqrt(dx*dx + dy*dy)
            
            duration = real_points[-1]['timestamp'] - real_points[0]['timestamp']
            avg_velocity = total_distance / duration if duration > 0 else 0
            
            print(f"Track {track_id}:")
            print(f"  Duration: {tracker.trajectory_metadata[track_id]['duration']:.2f}s")
            print(f"  Average Velocity: {avg_velocity:.1f} px/s")
            print(f"  Total Points: {len(trajectory)}")
            print(f"  Real Points: {len(real_points)}")
            print(f"  Predicted Points: {len(pred_points)}")
            print(f"  Prediction Ratio: {len(pred_points)/len(trajectory)*100:.1f}%")
    
    return tracker, trajectories


if __name__ == "__main__":
    main()