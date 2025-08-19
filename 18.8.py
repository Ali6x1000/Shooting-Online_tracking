import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import os
from pathlib import Path
import json

# You'll need to install these:
# pip install ultralytics supervision opencv-python matplotlib

try:
    from ultralytics import YOLO
    import supervision as sv
except ImportError:
    print("Missing required packages. Install with:")
    print("pip install ultralytics supervision")
    raise

class OnlineHockeyPuckTracker:
    """
    Online hockey puck tracker using YOLO detection + ByteTracker
    Processes video frame-by-frame and maintains real-time trajectory tracking
    """
    
    def __init__(self, yolo_model_path, confidence_threshold=0.1):
        """
        Initialize the online tracker
        
        Args:
            yolo_model_path (str): Path to YOLO model file (.pt)
            confidence_threshold (float): Minimum confidence for detections
        """
        print(f"🚀 Initializing Online Hockey Puck Tracker")
        print(f"   YOLO Model: {yolo_model_path}")
        print(f"   Confidence Threshold: {confidence_threshold}")
        
        # Load YOLO model
        try:
            self.model = YOLO(yolo_model_path)
            print(f"✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
        
        # Initialize ByteTracker with more permissive settings
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold * 0.5,  # Lower threshold for activation
            lost_track_buffer=30,  # 4 seconds at 30 FPS - longer buffer
            minimum_matching_threshold=0.6,  # Lower threshold for matching
            frame_rate=30,
            minimum_consecutive_frames=1  # Allow tracking after just 1 frame
        )
        
        self.confidence_threshold = confidence_threshold
        
        # Trajectory storage
        self.trajectories = defaultdict(list)  # track_id: list of (x, y, t, conf, frame_num)
        self.trajectory_metadata = defaultdict(dict)  # track_id: metadata
        
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
        
        print(f"✅ Tracker initialized successfully")
    
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
        Process a single frame and update trajectories
        
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
        
        # Debug: Store all detections
        for i in range(len(detections)):
            detection_data = {
                'frame_num': self.frame_count,
                'timestamp': timestamp,
                'bbox': detections.xyxy[i],
                'confidence': detections.confidence[i] if detections.confidence is not None else 0.0,
                'class_id': detections.class_id[i] if detections.class_id is not None else 0
            }
            self.all_detections.append(detection_data)
        
        print(f"Frame {self.frame_count}: Found {len(detections)} detections")
        
        # Update ByteTracker only if we have detections
        if len(detections) > 0:
            try:
                detections = self.tracker.update_with_detections(detections)
                print(f"Frame {self.frame_count}: Tracker returned {len(detections)} tracked objects")
                
                # Check if tracker_id exists
                if detections.tracker_id is not None:
                    for i, track_id in enumerate(detections.tracker_id):
                        print(f"  Track ID: {track_id}")
                else:
                    print("  No tracker IDs assigned")
                    
            except Exception as e:
                print(f"Tracking error on frame {self.frame_count}: {e}")
                # Create empty detections to continue processing
                detections = sv.Detections.empty()
        else:
            # No detections, update tracker with empty detections
            detections = self.tracker.update_with_detections(sv.Detections.empty())
        
        # Update trajectories
        self.update_trajectories(detections, timestamp)
        
        # Create annotated frame
        annotated_frame = self.annotate_frame(frame, detections, timestamp)
        
        return annotated_frame
    
    def update_trajectories(self, detections, timestamp):
        """Update trajectory storage with new detections"""
        if detections.tracker_id is None or len(detections) == 0:
            return
            
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            bbox = detections.xyxy[i]
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            
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
    
    def annotate_frame(self, frame, detections, timestamp):
        """Create annotated frame with detections and trajectories"""
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
                
                # Get color for this track
                color = self.trajectory_metadata[track_id]['color']
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw center point
                cv2.circle(annotated, (center_x, center_y), 4, color, -1)
                
                # Draw track ID and confidence
                label = f"ID:{track_id} ({confidence:.2f})"
                cv2.putText(annotated, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw COMPLETE trajectory lines for each track
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            color = self.trajectory_metadata[track_id]['color']
            
            # Draw ALL points in trajectory as a continuous line
            points = [(int(p['x']), int(p['y'])) for p in trajectory]
            
            if len(points) > 1:
                # Draw the complete trajectory as connected lines
                for i in range(1, len(points)):
                    cv2.line(annotated, points[i-1], points[i], color, 2)
                
                # Draw start point (green circle)
                cv2.circle(annotated, points[0], 6, (0, 255, 0), -1)
                cv2.circle(annotated, points[0], 6, (0, 0, 0), 1)
                
                # Draw end point (red circle) 
                cv2.circle(annotated, points[-1], 6, (0, 0, 255), -1)
                cv2.circle(annotated, points[-1], 6, (0, 0, 0), 1)
                
                # Draw direction arrow at midpoint
                if len(points) > 2:
                    mid_idx = len(points) // 2
                    if mid_idx < len(points) - 1:
                        dx = points[mid_idx + 1][0] - points[mid_idx][0]
                        dy = points[mid_idx + 1][1] - points[mid_idx][1]
                        if dx != 0 or dy != 0:  # Avoid zero-length arrow
                            length = np.sqrt(dx*dx + dy*dy)
                            if length > 0:
                                dx = dx / length * 20  # Normalize and scale
                                dy = dy / length * 20
                                cv2.arrowedLine(annotated, points[mid_idx], 
                                              (int(points[mid_idx][0] + dx), int(points[mid_idx][1] + dy)), 
                                              color, 2, tipLength=0.3)
        
        # Add frame info
        total_trajectories = len(self.trajectories)
        active_trajectories = len([t for t in self.trajectories.values() 
                                 if len(t) > 0 and abs(t[-1]['timestamp'] - timestamp) < 1.0])
        
        info_text = f"Frame: {self.frame_count} | Time: {timestamp:.2f}s | Current: {len(detections)} | Total Tracks: {total_trajectories} | Active: {active_trajectories}"
        
        # White text with black outline for visibility
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add legend
        legend_y = 60
        cv2.putText(annotated, "Legend: Green=Start, Red=End, Colored Line=Path", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated, "Legend: Green=Start, Red=End, Colored Line=Path", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
    
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
    Example usage of the OnlineHockeyPuckTracker
    
    Usage:
        python hockey_tracker.py <video_path> <yolo_model_path>
    
    Example:
        python hockey_tracker.py hockey_video.mp4 yolov8n.pt
    """
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python hockey_tracker.py <video_path> <yolo_model_path>")
        print("Example: python hockey_tracker.py hockey_video.mp4 yolov8n.pt")
        sys.exit(1)
    
    video_path = sys.argv[1]
    yolo_model_path = sys.argv[2]
    
    # Verify files exist
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(yolo_model_path):
        print(f"❌ YOLO model file not found: {yolo_model_path}")
        print("You can download a model with: from ultralytics import YOLO; YOLO('yolov8n.pt')")
        sys.exit(1)
    
    print("🚀 Starting Online Hockey Puck Tracking")
    print("="*60)
    
    # Create tracker with more permissive settings
    tracker = OnlineHockeyPuckTracker(
        yolo_model_path=yolo_model_path,
        confidence_threshold=0.05  # Lower threshold for better detection
    )
    
    # Process video
    trajectories = tracker.process_video(
        video_path=video_path,
        output_dir="hockey_tracking_results",
        save_frames=True,
        save_video=True,
        max_frames=None  # Process entire video
    )
    
    # Filter trajectories with more permissive settings
    filtered_trajectories = tracker.filter_trajectories(
        min_length=5,       # At least 5 detections (lower threshold)
        min_duration=0.2,   # At least 0.2 seconds (lower threshold)
        min_movement=10     # At least 10 pixels movement (lower threshold)
    )
    
    # Create visualizations
    tracker.create_trajectory_plots("hockey_tracking_results")
    
    # Print summary
    tracker.print_summary()
    
    print(f"\n🎉 TRACKING COMPLETE!")
    print(f"📁 Results saved to: hockey_tracking_results/")
    print(f"📊 Check the following outputs:")
    print(f"   - annotated_hockey_tracking.mp4 (video with trajectories)")
    print(f"   - all_trajectories_overview.png (overview plot)")
    print(f"   - individual_trajectories/ (detailed plots for each track)")
    print(f"   - trajectory_statistics.png (statistical analysis)")
    print(f"   - temporal_analysis.png (time-based analysis)")
    print(f"   - detection_analysis.png (detection vs tracking analysis)")
    print(f"   - trajectory_data.json (raw trajectory data)")
    print(f"   - detection_data.json (raw detection data)")


# Standalone usage example
def example_usage():
    """
    Example of how to use the tracker programmatically
    """
    # Initialize tracker with lower thresholds for better detection
    tracker = OnlineHockeyPuckTracker(
        yolo_model_path="yolov8n.pt",  # Download automatically if not present
        confidence_threshold=0.05  # Lower threshold
    )
    
    # Process video
    trajectories = tracker.process_video(
        video_path="hockey_video.mp4",
        output_dir="my_hockey_results",
        save_frames=True,
        save_video=True
    )
    
    # Filter high-quality trajectories with more permissive settings
    good_trajectories = tracker.filter_trajectories(
        min_length=10,      # At least 10 detections
        min_duration=0.5,   # At least 0.5 seconds
        min_movement=20     # At least 20 pixels movement
    )
    
    # Create all visualizations
    tracker.create_trajectory_plots("my_hockey_results")
    
    # Access trajectory data programmatically
    for track_id, trajectory in good_trajectories.items():
        print(f"Track {track_id}:")
        print(f"  Duration: {tracker.trajectory_metadata[track_id]['duration']:.2f}s")
        print(f"  Velocity: {tracker.trajectory_metadata[track_id]['velocity']:.1f} px/s")
        print(f"  Points: {len(trajectory)}")
    
    return tracker, trajectories


if __name__ == "__main__":
    main()


