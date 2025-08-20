#!/usr/bin/env python3
"""
Complete OC-SORT Hockey Puck Tracker with Shot Detection
Tracks puck trajectories that start in lower third and move toward net

# Basic usage (puck tracking only)
python test_ocsort.py shooting2.mp4 best_yolov11lv2_puck.pt

# With net detection
python test_ocsort.py shooting2.mp4 best_yolov11lv2_puck.pt --net-model Nets_n.pt

# Custom confidence and output
python test_ocsort.py video.mp4 puck_model.pt --confidence 0.4 --output tracked_video.mp4


"""
#!/usr/bin/env python3
"""
Complete OC-SORT Hockey Puck Tracker with Shot Detection
Tracks puck trajectories that start in lower third and move toward net
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# Import OCSort from the provided code structure
from trackers.ocsort_tracker.ocsort import OCSort

class HockeyPuckTracker:
    def __init__(self, puck_model_path, net_model_path=None, confidence_threshold=0.3):
        """
        Initialize the hockey puck tracker
        
        Args:
            puck_model_path (str): Path to puck detection YOLO model
            net_model_path (str): Path to net detection YOLO model
            confidence_threshold (float): Detection confidence threshold
        """
        print(f"Initializing Hockey Puck Tracker")
        print(f"Puck Model: {puck_model_path}")
        print(f"Net Model: {net_model_path if net_model_path else 'Not provided'}")
        
        # Check for MPS availability
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS (Apple Silicon GPU) for acceleration")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Using CUDA GPU for acceleration")
        else:
            device = 'cpu'
            print("Using CPU (no GPU acceleration)")
        
        # Load YOLO models with device specification
        self.puck_model = YOLO(puck_model_path)
        self.puck_model.to(device)
        
        self.net_model = None
        if net_model_path:
            self.net_model = YOLO(net_model_path)
            self.net_model.to(device)
        
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize OC-SORT tracker with optimized parameters for hockey

        # Play with these parameters  (ALI)
        self.tracker = OCSort(
            det_thresh=confidence_threshold,
            max_age=30,          # Shorter for fast hockey action
            min_hits=5,          # Quick confirmation
            iou_threshold=0.25,  # Lower for fast movement
            delta_t=2,           # Short look-back
            asso_func="giou",    # Better for fast objects
            inertia=0.4,         # Higher inertia for smoother velocity
            use_byte=False        # Enable ByteTrack-style matching
        )
        
        # Tracking storage
        self.all_trajectories = defaultdict(list)
        self.active_trajectories = {}  # Currently active tracks
        self.selected_shots = []       # Final selected shot trajectories
        self.frame_count = 0
        self.video_width = 0
        self.video_height = 0
        self.net_positions = []
        
        # Shot detection parameters
        self.max_active_tracks = 3
        self.angle_tolerance = 30  # degrees
        
        print("Hockey Puck Tracker initialized successfully")
    
    def detect_nets(self, frame):
        """Detect hockey nets in the frame"""
        if self.net_model is None:
            return []
        
        results = self.net_model.predict(frame, conf=0.5, verbose=False)
        nets = []
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Calculate net center and dimensions
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    nets.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'center': (center_x, center_y),
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
        
        return nets
    def print_selected_trajectories(self, output_file="selected_trajectories.txt"):
        """Save detailed information about all selected trajectories to a text file"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SELECTED TRAJECTORIES SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Total selected shots: {len(self.selected_shots)}\n")
            
            if not self.selected_shots:
                f.write("No trajectories were selected.\n")
                return
            
            for i, shot in enumerate(self.selected_shots):
                trajectory = shot['trajectory']
                track_id = shot['track_id']
                start_frame = shot['start_frame']
                end_frame = shot['end_frame']
                color = shot['color']
                
                f.write(f"\n--- Shot {i+1} (Track ID: {track_id}) ---\n")
                f.write(f"Frames: {start_frame} to {end_frame} (Duration: {end_frame - start_frame + 1} frames)\n")
                f.write(f"Total points: {len(trajectory)}\n")
                f.write(f"Color (BGR): {color}\n")
                
                if trajectory:
                    start_point = trajectory[0]
                    end_point = trajectory[-1]
                    f.write(f"Start position: ({start_point['x']:.1f}, {start_point['y']:.1f})\n")
                    f.write(f"End position: ({end_point['x']:.1f}, {end_point['y']:.1f})\n")
                    
                    # Calculate trajectory stats
                    total_distance = 0
                    for j in range(1, len(trajectory)):
                        prev = trajectory[j-1]
                        curr = trajectory[j]
                        dx = curr['x'] - prev['x']
                        dy = curr['y'] - prev['y']
                        total_distance += math.sqrt(dx*dx + dy*dy)
                    
                    f.write(f"Total distance traveled: {total_distance:.1f} pixels\n")
                    
                    # Write all trajectory points
                    f.write(f"\nAll trajectory points:\n")
                    f.write(f"{'Point':<6} {'Frame':<6} {'X':<8} {'Y':<8} {'Time(s)':<8}\n")
                    f.write("-" * 40 + "\n")
                    
                    for j, point in enumerate(trajectory):
                        f.write(f"{j+1:<6} {point['frame_num']:<6} {point['x']:<8.1f} "
                            f"{point['y']:<8.1f} {point['timestamp']:<8.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
    print(f"Selected trajectories saved")
    def detect_pucks(self, frame):
        """Detect pucks in the frame and convert to OCSort format"""
        results = self.puck_model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, confidence])
        
        return np.array(detections) if detections else np.empty((0, 5))
    
    def is_in_lower_third(self, y_position):
        """Check if position is in lower third of frame"""
        lower_third_start = self.video_height * (2/3)
        return y_position >= lower_third_start
    
    def calculate_movement_angle(self, trajectory):
        """Calculate the angle of movement for a trajectory"""
        if len(trajectory) < 2:
            return None
        
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        dx = end_point['x'] - start_point['x']
        dy = end_point['y'] - start_point['y']  # Note: y increases downward
        
        # Calculate angle in degrees (0 = horizontal right, 90 = downward)
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def is_moving_toward_net(self, trajectory, nets):
        """Check if trajectory is moving toward any net within angle tolerance"""
        if len(trajectory) < 3 or not nets:
            return False
        
        # Get recent movement direction
        recent_points = trajectory[-min(5, len(trajectory)):]
        if len(recent_points) < 2:
            return False
        
        # Calculate movement vector
        start = recent_points[0]
        end = recent_points[-1]
        movement_dx = end['x'] - start['x']
        movement_dy = end['y'] - start['y']
        
        if movement_dx == 0 and movement_dy == 0:
            return False
        
        # Check angle to each net
        current_pos = (end['x'], end['y'])
        
        for net in nets:
            net_center = net['center']
            
            # Vector to net
            to_net_dx = net_center[0] - current_pos[0]
            to_net_dy = net_center[1] - current_pos[1]
            
            if to_net_dx == 0 and to_net_dy == 0:
                continue
            
            # Calculate angle between movement and direction to net
            movement_angle = math.atan2(movement_dy, movement_dx)
            to_net_angle = math.atan2(to_net_dy, to_net_dx)
            
            angle_diff = abs(math.degrees(movement_angle - to_net_angle))
            angle_diff = min(angle_diff, 360 - angle_diff)  # Normalize to 0-180
            
            if angle_diff <= self.angle_tolerance:
                return True
        
        return False
    
    def is_trajectory_in_net(self, trajectory, nets):
        """Check if trajectory ends inside or very close to a net"""
        if not trajectory or not nets:
            return False
        
        end_point = trajectory[-1]
        end_x, end_y = end_point['x'], end_point['y']
        
        for net in nets:
            x1, y1, x2, y2 = net['bbox']
            # Add some tolerance around the net
            tolerance = 20
            if (x1 - tolerance <= end_x <= x2 + tolerance and 
                y1 - tolerance <= end_y <= y2 + tolerance):
                return True
        
        return False
    
    def manage_active_trajectories(self, current_tracks, nets):
        """Manage active trajectories based on shot criteria"""
        # Update active trajectories with new track data
        for track_id in list(self.active_trajectories.keys()):
            if track_id not in current_tracks:
                # Track is lost, check if it's a valid shot
                trajectory = self.all_trajectories[track_id]
                if self.is_valid_shot_trajectory(trajectory, nets):
                    self.active_trajectories[track_id]['is_valid_shot'] = True
                else:
                    # Remove invalid trajectory
                    del self.active_trajectories[track_id]
        
        # Add new tracks that start in lower third
        for track_id, track_data in current_tracks.items():
            if track_id not in self.active_trajectories:
                trajectory = self.all_trajectories[track_id]
                if len(trajectory) > 0:
                    first_point = trajectory[0]
                    if self.is_in_lower_third(first_point['y']):
                        self.active_trajectories[track_id] = {
                            'start_frame': self.frame_count,
                            'is_valid_shot': False
                        }
        
        # Limit to max active trajectories, keep the most recent ones
        if len(self.active_trajectories) > self.max_active_tracks:
            # Sort by start frame and keep the most recent
            sorted_tracks = sorted(self.active_trajectories.items(), 
                                 key=lambda x: x[1]['start_frame'], reverse=True)
            
            tracks_to_remove = sorted_tracks[self.max_active_tracks:]
            for track_id, _ in tracks_to_remove:
                del self.active_trajectories[track_id]
    
    def is_valid_shot_trajectory(self, trajectory, nets):
        """Check if a trajectory meets shot criteria"""
        if len(trajectory) < 5:  # Minimum length
            return False
        
        # Must start in lower third
        if not self.is_in_lower_third(trajectory[0]['y']):
            return False
        
        # Must move toward net
        if not self.is_moving_toward_net(trajectory, nets):
            return False
        
        return True
    
    def select_best_shot_from_active(self, nets):
        """Select the best shot from currently active trajectories"""
        valid_shots = []
        
        for track_id in self.active_trajectories:
            trajectory = self.all_trajectories[track_id]
            if self.is_valid_shot_trajectory(trajectory, nets):
                score = len(trajectory)  # Longer trajectories preferred
                
                # Bonus points if trajectory ends in net
                if self.is_trajectory_in_net(trajectory, nets):
                    score += 50
                
                valid_shots.append((track_id, trajectory, score))
        
        if not valid_shots:
            return None
        
        # Select the highest scoring trajectory
        best_shot = max(valid_shots, key=lambda x: x[2])
        return best_shot[0], best_shot[1]  # track_id, trajectory
    
    def reset_for_next_shot(self):
        """Reset tracking state for next shot detection"""
        # Select best shot from current active trajectories
        if self.active_trajectories:
            best_shot = self.select_best_shot_from_active(self.net_positions)
            if best_shot:
                track_id, trajectory = best_shot
                self.selected_shots.append({
                    'track_id': track_id,
                    'trajectory': trajectory,
                    'start_frame': trajectory[0]['frame_num'] if trajectory else self.frame_count,
                    'end_frame': trajectory[-1]['frame_num'] if trajectory else self.frame_count,
                    'color': self.generate_color(len(self.selected_shots))
                })
                print(f"Selected shot: Track {track_id} with {len(trajectory)} points")
        
        # Clear active trajectories for next shot
        self.active_trajectories.clear()
    
    def generate_color(self, index):
        """Generate a distinct color for each trajectory"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]
        return colors[index % len(colors)]
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Detect nets
        nets = self.detect_nets(frame)
        if nets:
            self.net_positions = nets
        
        # Detect pucks
        detections = self.detect_pucks(frame)
        
        # Update tracker
        if len(detections) > 0:
            # Convert detections for OCSort
            img_info = [self.video_height, self.video_width]
            img_size = [self.video_width, self.video_height]
            tracked_objects = self.tracker.update(detections, img_info, img_size)
        else:
            # No detections, update with empty
            tracked_objects = np.empty((0, 5))
        
        # Update trajectories
        current_tracks = {}
        if len(tracked_objects) > 0:
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Store trajectory point
                trajectory_point = {
                    'x': float(center_x),
                    'y': float(center_y),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'frame_num': self.frame_count,
                    'timestamp': self.frame_count / 30.0  # Assuming 30 FPS
                }
                
                self.all_trajectories[track_id].append(trajectory_point)
                current_tracks[track_id] = trajectory_point
        
        # Manage active trajectories
        self.manage_active_trajectories(current_tracks, nets)
        
        # Check if we should reset for next shot (simple heuristic)
        # Reset when no active tracks for several frames
        if not current_tracks and self.active_trajectories:
            self.reset_for_next_shot()
        
        return self.annotate_frame(frame, tracked_objects, nets)
    
    def annotate_frame(self, frame, tracked_objects, nets):
        """Annotate frame with tracking results"""
        annotated = frame.copy()
        
        # Draw nets
        for net in nets:
            bbox = net['bbox'].astype(int)
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                        (0, 255, 255), 3)
            cv2.putText(annotated, f"NET ({net['confidence']:.2f})", 
                    (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 255), 2)
        
        # Draw lower third line
        lower_third_y = int(self.video_height * (2/3))
        cv2.line(annotated, (0, lower_third_y), (self.video_width, lower_third_y), 
                (255, 255, 255), 2)
        cv2.putText(annotated, "Lower Third", (10, lower_third_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ===== ALWAYS DRAW ALL SELECTED SHOTS FIRST (so they appear in background) =====
        for i, shot in enumerate(self.selected_shots):
            trajectory = shot['trajectory']
            color = shot['color']
            track_id = shot['track_id']
            
            if len(trajectory) > 1:
                points = [(int(p['x']), int(p['y'])) for p in trajectory]
                
                # Draw the complete trajectory path
                for j in range(1, len(points)):
                    cv2.line(annotated, points[j-1], points[j], color, 4)
                
                # Mark start and end points
                cv2.circle(annotated, points[0], 8, (0, 255, 0), -1)   # Green start
                cv2.circle(annotated, points[-1], 8, (0, 0, 255), -1) # Red end
                
                # Add shot label near the start
                label_pos = points[min(3, len(points)-1)]  # Position label a few points in
                cv2.putText(annotated, f"Shot {i+1} (ID:{track_id})", 
                        (label_pos[0] + 10, label_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current detections
        if len(tracked_objects) > 0:
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Color based on whether it's an active trajectory
                if track_id in self.active_trajectories:
                    color = (0, 255, 0)  # Green for active
                    thickness = 3
                else:
                    color = (128, 128, 128)  # Gray for inactive
                    thickness = 2
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                label = f"ID:{track_id}"
                if track_id in self.active_trajectories:
                    label += " ACTIVE"
                
                cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectories for currently active tracks (in progress)
        for track_id in self.active_trajectories:
            if track_id in self.all_trajectories:
                trajectory = self.all_trajectories[track_id]
                if len(trajectory) > 1:
                    points = [(int(p['x']), int(p['y'])) for p in trajectory]
                    # Draw with dashed/dotted effect for active tracks
                    for i in range(1, len(points)):
                        if i % 2 == 0:  # Create dashed effect
                            cv2.line(annotated, points[i-1], points[i], (0, 255, 0), 2)
        
        # Enhanced info text with shot summary
        info_text = (f"Frame: {self.frame_count} | Active: {len(self.active_trajectories)} | "
                    f"Selected Shots: {len(self.selected_shots)}")
        
        # Draw text with background for better visibility
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(annotated, (5, 5), (text_size[0] + 15, 40), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
        
        # Add legend for selected shots
        if self.selected_shots:
            legend_y_start = 60
            cv2.putText(annotated, "Selected Shots:", (10, legend_y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, shot in enumerate(self.selected_shots):
                y_pos = legend_y_start + 25 + (i * 25)
                color = shot['color']
                track_id = shot['track_id']
                # Draw a small line sample of the shot color
                cv2.line(annotated, (10, y_pos), (40, y_pos), color, 4)
                cv2.putText(annotated, f"Shot {i+1} (Track {track_id})", 
                        (50, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def process_video(self, video_path, output_path="output_tracking.mp4"):
        """Process entire video"""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.video_width}x{self.video_height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (self.video_width, self.video_height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                annotated_frame = self.process_frame(frame)
                out.write(annotated_frame)
                
                self.frame_count += 1
                
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        finally:
            # Final shot selection
            if self.active_trajectories:
                self.reset_for_next_shot()
            
            cap.release()
            out.release()
        
        print(f"Video processing complete. Output: {output_path}")
        print(f"Total trajectories: {len(self.all_trajectories)}")
        print(f"Selected shots: {len(self.selected_shots)}")
        
        return annotated_frame  # Return last frame for final annotation
    
    def create_final_frame_with_all_shots(self, last_frame):
        """Create final frame showing all selected shot trajectories"""
        final_frame = last_frame.copy()
        
        # Draw all selected shots
        for i, shot in enumerate(self.selected_shots):
            trajectory = shot['trajectory']
            color = shot['color']
            
            if len(trajectory) > 1:
                points = [(int(p['x']), int(p['y'])) for p in trajectory]
                
                # Draw trajectory
                for j in range(1, len(points)):
                    cv2.line(final_frame, points[j-1], points[j], color, 4)
                
                # Mark start and end
                cv2.circle(final_frame, points[0], 10, (0, 255, 0), -1)   # Green start
                cv2.circle(final_frame, points[-1], 10, (0, 0, 255), -1) # Red end
                
                # Add shot number
                mid_idx = len(points) // 2
                cv2.putText(final_frame, f"Shot {i+1}", points[mid_idx], 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add title
        cv2.putText(final_frame, f"Selected Shot Trajectories ({len(self.selected_shots)} shots)", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(final_frame, f"Selected Shot Trajectories ({len(self.selected_shots)} shots)", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return final_frame
    
    def create_trajectory_plots(self, output_dir="output"):
        """Create trajectory visualization plots"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot 1: All trajectories
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.set_xlim(0, self.video_width)
        ax.set_ylim(self.video_height, 0)  # Flip Y axis
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'All Detected Trajectories ({len(self.all_trajectories)} tracks)')
        ax.grid(True, alpha=0.3)
        
        # Draw lower third line
        lower_third_y = self.video_height * (2/3)
        ax.axhline(y=lower_third_y, color='white', linestyle='--', linewidth=2, 
                   label='Lower Third')
        
        # Draw nets
        for net in self.net_positions:
            x1, y1, x2, y2 = net['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='yellow', 
                               facecolor='none', alpha=0.8)
            ax.add_patch(rect)
        
        # Draw all trajectories
        for track_id, trajectory in self.all_trajectories.items():
            if len(trajectory) > 1:
                x_coords = [p['x'] for p in trajectory]
                y_coords = [p['y'] for p in trajectory]
                ax.plot(x_coords, y_coords, '-', alpha=0.6, linewidth=1)
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'all_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Selected shots only
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.set_xlim(0, self.video_width)
        ax.set_ylim(self.video_height, 0)  # Flip Y axis
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'Selected Shot Trajectories ({len(self.selected_shots)} shots)')
        ax.grid(True, alpha=0.3)
        
        # Draw lower third line
        ax.axhline(y=lower_third_y, color='white', linestyle='--', linewidth=2, 
                   label='Lower Third')
        
        # Draw nets
        for net in self.net_positions:
            x1, y1, x2, y2 = net['bbox']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=3, edgecolor='yellow', 
                               facecolor='none', alpha=0.8, label='Net')
            ax.add_patch(rect)
        
        # Draw selected shots
        for i, shot in enumerate(self.selected_shots):
            trajectory = shot['trajectory']
            color = [c/255.0 for c in shot['color']]
            
            x_coords = [p['x'] for p in trajectory]
            y_coords = [p['y'] for p in trajectory]
            
            ax.plot(x_coords, y_coords, '-', color=color, linewidth=3, 
                   label=f'Shot {i+1}')
            ax.plot(x_coords[0], y_coords[0], 'o', color='green', markersize=8)
            ax.plot(x_coords[-1], y_coords[-1], 's', color='red', markersize=8)
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'selected_shots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory plots saved to {output_path}/")
        return str(output_path / 'all_trajectories.png'), str(output_path / 'selected_shots.png')


def main():
    parser = argparse.ArgumentParser(description='Hockey Puck Tracker with Shot Detection')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('puck_model', help='Path to puck detection YOLO model')
    parser.add_argument('--net-model', help='Path to net detection YOLO model')
    parser.add_argument('--output', default='output_tracking.mp4', help='Output video path')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Set output directory to current script directory if not specified
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        args.output_dir = str(script_dir)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create tracker
    tracker = HockeyPuckTracker(
        puck_model_path=args.puck_model,
        net_model_path=args.net_model,
        confidence_threshold=args.confidence
    )
    
    # Process video
    last_frame = tracker.process_video(args.video_path, args.output)
    
    # Save all selected trajectories to text file in the script directory
    trajectory_file = f"{args.output_dir}/selected_trajectories.txt"
    tracker.print_selected_trajectories(trajectory_file)
    
    # Create final frame with all shots
    final_frame = tracker.create_final_frame_with_all_shots(last_frame)
    cv2.imwrite(f"{args.output_dir}/final_frame_all_shots.jpg", final_frame)
    
    # Create trajectory plots
    tracker.create_trajectory_plots(args.output_dir)
    
    print("\nProcessing Complete!")
    print(f"Output video: {args.output}")
    print(f"Final frame: {args.output_dir}/final_frame_all_shots.jpg")
    print(f"Trajectory data: {trajectory_file}")
    print(f"Trajectory plots: {args.output_dir}/")


if __name__ == "__main__":
    main()