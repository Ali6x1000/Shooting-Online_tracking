#!/usr/bin/env python3
"""
OC-SORT Hockey Tracker Integration
Combines OC-SORT tracker with hockey-specific enhancements
# Simple OC-SORT tracking
python OC-SORT.py shooting2.mp4 best_yolov11lv2_puck.pt

# With net detection
python OC-SORT.py shooting2.mp4 best_yolov11lv2_puck.pt --net-model Nets_n.pt

# With shot analysis
python OC-SORT.py shooting2.mp4 best_yolov11lv2_puck.pt --net-model Nets_n.pt --shot-analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import os
from pathlib import Path
import json
import argparse
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

# Import from the main hockey tracker
try:
    from hockey_tracker import (
        PuckPredictor, 
        HockeyRinkAnalyzer, 
        ShotAnalyzer,
        YOLO,
        sv
    )
except ImportError:
    print("❌ Could not import from main hockey tracker file")
    print("   Make sure 'hockey_tracker_fixed.py' is in the same directory")
    raise

# ----------------- OC-SORT Implementation -----------------
def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    return np.array([x1 + w/2.0, y1 + h/2.0, w, h], dtype=float)

def cxcywh_to_xyxy(b):
    cx, cy, w, h = b
    return np.array([cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0], dtype=float)

def iou_xyxy(a, b):
    # a: (N,4), b: (M,4)
    N, M = a.shape[0], b.shape[0]
    iou = np.zeros((N, M), dtype=float)
    for i in range(N):
        xx1 = np.maximum(a[i,0], b[:,0])
        yy1 = np.maximum(a[i,1], b[:,1])
        xx2 = np.minimum(a[i,2], b[:,2])
        yy2 = np.minimum(a[i,3], b[:,3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_a = (a[i,2] - a[i,0]) * (a[i,3] - a[i,1])
        area_b = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
        union = np.maximum(area_a + area_b - inter, 1e-6)
        iou[i] = inter / union
    return iou

@dataclass
class OCTrack:
    tid: int
    last_obs_xyxy: np.ndarray                 # last observed bbox (x1,y1,x2,y2)
    score: float
    obs_history: deque = field(default_factory=lambda: deque(maxlen=5))  # cx,cy,w,h
    v_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))        # observation-based velocity (px/frame)
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    lost: bool = False

    def observe(self, det_xyxy, det_score, ema=0.7):
        """Update with a fresh detection (observation-centric)."""
        cur = xyxy_to_cxcywh(det_xyxy)
        if len(self.obs_history) > 0:
            prev = self.obs_history[-1]
            obs_v = cur[:2] - prev[:2]  # dx, dy per frame
            self.v_xy = ema * self.v_xy + (1 - ema) * obs_v
        self.obs_history.append(cur)
        self.last_obs_xyxy = det_xyxy.astype(float)
        self.score = det_score
        self.hits += 1
        self.time_since_update = 0
        self.lost = False

    def project_from_observation(self):
        """Project next position using observation-based velocity only."""
        if len(self.obs_history) == 0:
            return self.last_obs_xyxy.copy()
        cur = self.obs_history[-1].copy()           # cx, cy, w, h
        cur[:2] = cur[:2] + self.v_xy               # cx+vx, cy+vy
        return cxcywh_to_xyxy(cur)

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1
        if self.time_since_update > 0:
            self.lost = True

class OCSortTracker:
    def __init__(
        self,
        iou_thresh_active=0.3,
        iou_thresh_lost=0.2,
        max_age=30,
        min_hits=2,
        det_score_thresh=0.3,
        motion_lambda=0.5
    ):
        self.iou_thresh_active = iou_thresh_active
        self.iou_thresh_lost = iou_thresh_lost
        self.max_age = max_age
        self.min_hits = min_hits
        self.det_score_thresh = det_score_thresh
        self.motion_lambda = motion_lambda

        self.tracks = []        # active tracks
        self.lost_pool = []     # recently lost tracks
        self._next_tid = 1
        self.frame_count = 0

    def _build_cost(self, tracks, dets_xyxy):
        if len(tracks) == 0 or dets_xyxy.shape[0] == 0:
            return np.zeros((len(tracks), dets_xyxy.shape[0]))
        proj_boxes = np.stack([t.project_from_observation() for t in tracks], axis=0)
        iou = iou_xyxy(proj_boxes, dets_xyxy)
        # motion penalty
        proj_c = np.array([xyxy_to_cxcywh(b)[:2] for b in proj_boxes])  # (T,2)
        det_c = np.array([xyxy_to_cxcywh(b)[:2] for b in dets_xyxy])    # (D,2)
        # normalize by track box size to be scale-invariant
        sizes = np.array([xyxy_to_cxcywh(t.last_obs_xyxy)[2:4].mean() for t in tracks])  # (T,)
        sizes = np.maximum(sizes, 1.0)
        dist = np.sqrt(((proj_c[:, None, :] - det_c[None, :, :]) ** 2).sum(-1)) / sizes[:, None]
        cost = (1.0 - iou) + self.motion_lambda * dist
        return cost

    def _hungarian_match(self, tracks, dets_xyxy, iou_thresh):
        if len(tracks) == 0 or dets_xyxy.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(dets_xyxy.shape[0]))
        cost = self._build_cost(tracks, dets_xyxy)
        r_idx, c_idx = linear_sum_assignment(cost)
        matches, u_t, u_d = [], [], []
        # filter by IoU threshold using projected boxes
        proj_boxes = np.stack([t.project_from_observation() for t in tracks], axis=0)
        iou = iou_xyxy(proj_boxes, dets_xyxy)
        assigned_t = set()
        assigned_d = set()
        for ti, di in zip(r_idx, c_idx):
            if iou[ti, di] >= iou_thresh:
                matches.append((ti, di))
                assigned_t.add(ti)
                assigned_d.add(di)
        for i in range(len(tracks)):
            if i not in assigned_t:
                u_t.append(i)
        for j in range(dets_xyxy.shape[0]):
            if j not in assigned_d:
                u_d.append(j)
        return matches, u_t, u_d

    def _age_and_sweep(self):
        # move expired lost tracks out; age active tracks
        keep_tracks = []
        for t in self.tracks:
            if t.time_since_update > 0:
                t.mark_missed()
            else:
                t.age += 1
        # move newly-lost to lost_pool
        for t in self.tracks:
            if t.lost and t.time_since_update <= self.max_age:
                self.lost_pool.append(t)
        # keep only tracks that are not lost this frame
        keep_tracks = [t for t in self.tracks if not t.lost]
        self.tracks = keep_tracks
        # purge very old from lost pool
        self.lost_pool = [t for t in self.lost_pool if t.time_since_update <= self.max_age]

    def update(self, detections):
        """Update with new detections, return tracked objects with supervision format"""
        self.frame_count += 1
        if len(detections) == 0:
            for t in self.tracks: 
                t.mark_missed()
            for t in self.lost_pool: 
                t.time_since_update += 1
            self._age_and_sweep()
            return sv.Detections.empty()

        # Convert supervision detections to OC-SORT format
        if hasattr(detections, 'xyxy'):
            dets_xyxy = detections.xyxy
            scores = detections.confidence if detections.confidence is not None else np.ones(len(detections)) * 0.5
        else:
            # Assume it's already in the right format
            dets_xyxy = np.array([d[:4] for d in detections], dtype=float)
            scores = np.array([d[4] for d in detections], dtype=float)

        # Filter by score threshold
        valid = scores >= self.det_score_thresh
        dets_xyxy = dets_xyxy[valid]
        scores = scores[valid]

        if len(dets_xyxy) == 0:
            for t in self.tracks: 
                t.mark_missed()
            for t in self.lost_pool: 
                t.time_since_update += 1
            self._age_and_sweep()
            return sv.Detections.empty()

        # 1) match active tracks
        matches, u_t, u_d = self._hungarian_match(self.tracks, dets_xyxy, self.iou_thresh_active)

        # apply matches to active tracks
        for ti, di in matches:
            self.tracks[ti].observe(dets_xyxy[di], scores[di])

        # unmatched active -> mark missed
        for idx in u_t:
            self.tracks[idx].mark_missed()

        # 2) try to re-associate with lost tracks
        re_dets = dets_xyxy[u_d]
        re_scores = scores[u_d]
        if len(self.lost_pool) > 0 and re_dets.shape[0] > 0:
            lost_boxes = np.stack([t.last_obs_xyxy for t in self.lost_pool], axis=0)
            iou_lost = iou_xyxy(lost_boxes, re_dets)
            # greedy match by IoU
            used_lost, used_det = set(), set()
            pairs = []
            flat = [(-iou_lost[i, j], i, j) for i in range(iou_lost.shape[0]) for j in range(iou_lost.shape[1])]
            for _, i, j in sorted(flat):
                if i in used_lost or j in used_det: 
                    continue
                if iou_lost[i, j] >= self.iou_thresh_lost:
                    pairs.append((i, j))
                    used_lost.add(i)
                    used_det.add(j)
            
            # apply re-association
            for li, dj in pairs:
                lost_track = self.lost_pool[li]
                det_idx = u_d[dj]
                lost_track.observe(dets_xyxy[det_idx], scores[det_idx])
                self.tracks.append(lost_track)
            
            # remove revived from lost_pool
            self.lost_pool = [t for k, t in enumerate(self.lost_pool) if k not in used_lost]
            # remaining unmatched detections after re-assoc
            u_d = [ud for k, ud in enumerate(u_d) if k not in used_det]

        # 3) create new tracks for remaining unmatched detections
        for di in u_d:
            t = OCTrack(
                tid=self._next_tid,
                last_obs_xyxy=dets_xyxy[di].astype(float),
                score=float(scores[di]),
            )
            t.observe(dets_xyxy[di], float(scores[di]))
            self.tracks.append(t)
            self._next_tid += 1

        # 4) age bookkeeping
        for t in self.lost_pool:
            t.time_since_update += 1
        self._age_and_sweep()

        # 5) Convert back to supervision format
        if len(self.tracks) == 0:
            return sv.Detections.empty()

        # Only return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits or self.frame_count <= self.min_hits]
        
        if len(confirmed_tracks) == 0:
            return sv.Detections.empty()

        output_boxes = np.array([t.last_obs_xyxy for t in confirmed_tracks])
        output_scores = np.array([t.score for t in confirmed_tracks])
        output_ids = np.array([t.tid for t in confirmed_tracks])

        result = sv.Detections(
            xyxy=output_boxes,
            confidence=output_scores
        )
        result.tracker_id = output_ids

        return result

# ----------------- Enhanced Hockey Tracker with OC-SORT -----------------
class OCSortHockeyTracker:
    """Hockey tracker using OC-SORT algorithm with hockey-specific enhancements"""
    
    def __init__(self, yolo_model_path, confidence_threshold=0.1, 
                 enable_prediction=True, prediction_method='kalman',
                 net_model_path=None, oc_sort_config=None):
        """
        Initialize OC-SORT hockey tracker
        
        Args:
            yolo_model_path (str): Path to puck detection model
            confidence_threshold (float): YOLO detection threshold
            enable_prediction (bool): Enable prediction system
            prediction_method (str): Prediction method
            net_model_path (str): Path to net detection model
            oc_sort_config (dict): OC-SORT configuration parameters
        """
        print(f"🚀 Initializing OC-SORT Hockey Tracker")
        print(f"   Puck Model: {yolo_model_path}")
        print(f"   Net Model: {net_model_path if net_model_path else 'Not provided'}")
        print(f"   Prediction: {prediction_method if enable_prediction else 'Disabled'}")
        
        # Load YOLO model for puck detection
        self.model = YOLO(yolo_model_path)
        self.confidence_threshold = confidence_threshold
        
        # Initialize rink analyzer
        self.rink_analyzer = HockeyRinkAnalyzer()
        self.net_detection_enabled = False
        if net_model_path:
            if self.rink_analyzer.load_net_model(net_model_path):
                self.net_detection_enabled = True
        
        # Initialize prediction system
        self.enable_prediction = enable_prediction
        self.predictor = PuckPredictor(method=prediction_method) if enable_prediction else None
        
        # Initialize OC-SORT tracker
        if oc_sort_config is None:
            oc_sort_config = {
                'iou_thresh_active': 0.3,
                'iou_thresh_lost': 0.2,
                'max_age': 30,
                'min_hits': 2,
                'det_score_thresh': confidence_threshold,
                'motion_lambda': 0.5
            }
        
        self.tracker = OCSortTracker(**oc_sort_config)
        print(f"✅ OC-SORT configured: {oc_sort_config}")
        
        # Initialize shot analyzer
        self.shot_analyzer = ShotAnalyzer(self.rink_analyzer) if net_model_path else None
        
        # Storage
        self.trajectories = defaultdict(list)
        self.trajectory_metadata = defaultdict(dict)
        self.last_positions = {}
        self.all_detections = []
        self.net_detections = []
        
        # Video state
        self.frame_count = 0
        self.video_fps = 30
        self.video_width = 0
        self.video_height = 0
        
        print(f"✅ OC-SORT Hockey Tracker initialized successfully")
    
    def process_video(self, video_path, output_dir="oc_sort_hockey_results", 
                     save_frames=True, save_video=True, max_frames=None):
        """Process video using OC-SORT tracking"""
        print(f"\n🎬 Processing Video with OC-SORT: {video_path}")
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
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Setup video writer
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video_path = output_path / "oc_sort_hockey_tracking.mp4"
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
                if not ret or (max_frames and self.frame_count >= max_frames):
                    break
                
                timestamp = self.frame_count / self.video_fps
                annotated_frame = self.process_frame(frame, timestamp)
                
                if save_video and video_writer:
                    video_writer.write(annotated_frame)
                
                if save_frames and self.frame_count % 30 == 0:
                    frame_path = output_path / f"oc_sort_frame_{self.frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
                
                self.frame_count += 1
                
                # Progress update
                current_time = time.time()
                if current_time - last_print_time > 2.0:
                    progress = (self.frame_count / total_frames) * 100
                    fps_processing = self.frame_count / (current_time - start_time)
                    active_tracks = len(self.tracker.tracks)
                    
                    print(f"   Progress: {progress:.1f}% | Frame {self.frame_count}/{total_frames} | "
                          f"Processing FPS: {fps_processing:.1f} | OC-SORT Tracks: {active_tracks}")
                    last_print_time = current_time
        
        finally:
            cap.release()
            if save_video and video_writer:
                video_writer.release()
        
        processing_time = time.time() - start_time
        print(f"\n✅ OC-SORT Video Processing Complete!")
        print(f"   Processing Time: {processing_time:.1f}s")
        print(f"   Average FPS: {self.frame_count/processing_time:.1f}")
        print(f"   Total Trajectories: {len(self.trajectories)}")
        
        return self.trajectories
    
    def process_frame(self, frame, timestamp):
        """Process single frame with OC-SORT"""
        # Detect nets for context
        nets = []
        if self.net_detection_enabled:
            nets = self.rink_analyzer.detect_nets(frame)
            self.rink_analyzer.update_rink_analysis(nets)
        
        # YOLO puck detection
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter detections using rink context
        if self.net_detection_enabled and len(detections) > 0:
            detections = self._filter_detections_by_rink_context(detections, nets)
        
        # Store raw detections
        for i in range(len(detections)):
            detection_data = {
                'frame_num': self.frame_count,
                'timestamp': timestamp,
                'bbox': detections.xyxy[i],
                'confidence': detections.confidence[i] if detections.confidence is not None else 0.0,
                'predicted': False
            }
            if self.net_detection_enabled:
                center_x = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
                center_y = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
                detection_data['zone'] = self.rink_analyzer.get_zone(center_x, center_y)
            self.all_detections.append(detection_data)
        
        # OC-SORT tracking
        tracked_detections = self.tracker.update(detections)
        
        print(f"Frame {self.frame_count}: {len(detections)} detections → {len(tracked_detections)} tracks")
        
        # Update trajectories
        self.update_trajectories(tracked_detections, timestamp)
        
        # Create annotated frame
        annotated_frame = self.annotate_frame(frame, tracked_detections, nets, timestamp)
        
        return annotated_frame
    
    def _filter_detections_by_rink_context(self, detections, nets):
        """Filter detections using rink context - same as main tracker"""
        if len(detections) == 0 or len(nets) == 0:
            return detections
        
        valid_indices = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Filter out detections too close to nets
            min_net_distance = float('inf')
            for net in nets:
                net_x, net_y = net['center']
                distance = np.sqrt((center_x - net_x)**2 + (center_y - net_y)**2)
                min_net_distance = min(min_net_distance, distance)
            
            if min_net_distance >= 30:  # Must be at least 30px from nets
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return sv.Detections.empty()
        
        return sv.Detections(
            xyxy=detections.xyxy[valid_indices],
            confidence=detections.confidence[valid_indices] if detections.confidence is not None else None,
            class_id=detections.class_id[valid_indices] if detections.class_id is not None else None
        )
    
    def update_trajectories(self, detections, timestamp):
        """Update trajectory storage"""
        if not hasattr(detections, 'tracker_id') or detections.tracker_id is None or len(detections) == 0:
            return
        
        for i in range(len(detections)):
            track_id = detections.tracker_id[i]
            bbox = detections.xyxy[i]
            confidence = detections.confidence[i] if detections.confidence is not None else 0.0
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Update predictor
            if self.enable_prediction:
                self.predictor.update_history(track_id, center_x, center_y, timestamp)
            
            # Store position
            self.last_positions[track_id] = {
                'x': center_x, 'y': center_y, 'timestamp': timestamp
            }
            
            # Add rink context
            rink_context = {}
            if self.net_detection_enabled:
                rink_context['zone'] = self.rink_analyzer.get_zone(center_x, center_y)
                rink_context['in_goal_area'] = self.rink_analyzer.is_goal_area(center_x, center_y)
            
            # Create trajectory point
            trajectory_point = {
                'x': float(center_x),
                'y': float(center_y),
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'timestamp': timestamp,
                'frame_num': self.frame_count,
                'predicted': False,
                **rink_context
            }
            
            self.trajectories[track_id].append(trajectory_point)
            
            # Update metadata
            if track_id not in self.trajectory_metadata:
                self.trajectory_metadata[track_id] = {
                    'start_time': timestamp,
                    'start_frame': self.frame_count,
                    'color': self.generate_color(track_id)
                }
            
            self.trajectory_metadata[track_id]['end_time'] = timestamp
            self.trajectory_metadata[track_id]['end_frame'] = self.frame_count
    
    def annotate_frame(self, frame, detections, nets, timestamp):
        """Create annotated frame"""
        annotated = frame.copy()
        
        # Draw nets
        if self.net_detection_enabled and len(nets) > 0:
            for net in nets:
                bbox = net['bbox'].astype(int)
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 3)
                cv2.putText(annotated, f"NET ({net['confidence']:.2f})", 
                           (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw current detections
        if len(detections) > 0 and hasattr(detections, 'tracker_id'):
            for i in range(len(detections)):
                track_id = detections.tracker_id[i]
                bbox = detections.xyxy[i].astype(int)
                confidence = detections.confidence[i] if detections.confidence is not None else 0.0
                
                color = self.trajectory_metadata[track_id]['color']
                
                # Draw bounding box
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw center point
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                cv2.circle(annotated, (center_x, center_y), 4, color, -1)
                
                # Label with OC-SORT indicator
                label = f"ID:{track_id} OC ({confidence:.2f})"
                cv2.putText(annotated, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw trajectories
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            color = self.trajectory_metadata[track_id]['color']
            points = [(int(p['x']), int(p['y'])) for p in trajectory]
            
            # Draw trajectory lines
            for i in range(1, len(points)):
                cv2.line(annotated, points[i-1], points[i], color, 2)
            
            # Draw start/end markers
            if len(points) > 0:
                cv2.circle(annotated, points[0], 6, (0, 255, 0), -1)   # Start
                cv2.circle(annotated, points[-1], 6, (0, 0, 255), -1) # End
        
        # Frame info
        active_tracks = len([t for t in self.trajectories.values() 
                           if len(t) > 0 and abs(t[-1]['timestamp'] - timestamp) < 1.0])
        info_text = f"Frame: {self.frame_count} | OC-SORT | Nets: {len(nets)} | Tracks: {len(self.trajectories)}/{active_tracks}"
        
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return annotated
    
    def generate_color(self, track_id):
        """Generate consistent color for track"""
        np.random.seed(int(track_id) % 2147483647)
        color = tuple(map(int, np.random.randint(50, 255, 3)))
        return color
    
    def merge_and_filter_shots(self, min_shot_distance=50):
        """Merge and filter for shots using shot analyzer"""
        if not self.shot_analyzer:
            print("⚠️ Shot analysis requires net detection model")
            return self.trajectories
        
        return self.shot_analyzer.merge_trajectory_fragments(self.trajectories)
    
    def create_trajectory_plots(self, output_dir):
        """Create trajectory visualizations"""
        output_path = Path(output_dir)
        
        if not self.trajectories:
            print("❌ No trajectories to plot!")
            return
        
        # Create basic overview plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        ax.set_xlim(0, self.video_width)
        ax.set_ylim(self.video_height, 0)
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'OC-SORT Hockey Trajectories ({len(self.trajectories)} tracks)')
        ax.grid(True, alpha=0.3)
        
        # Draw nets if available
        if self.net_detection_enabled and self.rink_analyzer.nets:
            for net in self.rink_analyzer.nets:
                net_rect = plt.Rectangle(
                    (net['center'][0] - net['width']/2, net['center'][1] - net['height']/2),
                    net['width'], net['height'],
                    linewidth=3, edgecolor='yellow', facecolor='none', alpha=0.8
                )
                ax.add_patch(net_rect)
        
        # Draw trajectories
        for track_id, trajectory in self.trajectories.items():
            x_coords = [p['x'] for p in trajectory]
            y_coords = [p['y'] for p in trajectory]
            
            color = [c/255.0 for c in self.trajectory_metadata[track_id]['color']]
            
            ax.plot(x_coords, y_coords, '-', color=color, linewidth=2, alpha=0.8)
            ax.plot(x_coords[0], y_coords[0], 'o', color='green', markersize=8)
            ax.plot(x_coords[-1], y_coords[-1], 's', color='red', markersize=8)
            
            # Add track label
            mid_idx = len(trajectory) // 2
            ax.text(x_coords[mid_idx], y_coords[mid_idx], str(track_id), 
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(output_path / 'oc_sort_trajectories.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved OC-SORT trajectory plot: oc_sort_trajectories.png")
    
    def print_summary(self):
        """Print tracking summary"""
        print(f"\n" + "="*80)
        print(f"🏒 OC-SORT HOCKEY TRACKING SUMMARY")
        print("="*80)
        
        print(f"📊 PROCESSING RESULTS:")
        print(f"   Video Resolution: {self.video_width}x{self.video_height}")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Video Duration: {self.frame_count/self.video_fps:.1f} seconds")
        print(f"   Total Detections: {len(self.all_detections)}")
        print(f"   Total Trajectories: {len(self.trajectories)}")
        
        if self.trajectories:
            durations = []
            lengths = []
            for track_id, trajectory in self.trajectories.items():
                duration = trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
                durations.append(duration)
                lengths.append(len(trajectory))
            
            print(f"\n🎯 OC-SORT TRAJECTORY STATISTICS:")
            print(f"   Duration - Min: {min(durations):.2f}s, Max: {max(durations):.2f}s, Avg: {np.mean(durations):.2f}s")
            print(f"   Length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f} points")
            
            print(f"\n🏒 INDIVIDUAL TRAJECTORIES:")
            for track_id in sorted(self.trajectories.keys()):
                traj = self.trajectories[track_id]
                duration = traj[-1]['timestamp'] - traj[0]['timestamp']
                print(f"   Track {track_id:2d}: {duration:.2f}s, {len(traj):3d} points")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='OC-SORT Hockey Puck Tracker')
    
    # Required arguments
    parser.add_argument('video_path', help='Path to hockey video')
    parser.add_argument('puck_model', help='Path to puck detection YOLO model')
    
    # Optional arguments
    parser.add_argument('--net-model', help='Path to net detection YOLO model')
    parser.add_argument('--output-dir', default='oc_sort_hockey_results', help='Output directory')
    parser.add_argument('--confidence', type=float, default=0.05, help='Detection confidence threshold')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--prediction-method', choices=['linear', 'polynomial', 'kalman'], 
                       default='kalman', help='Prediction method')
    parser.add_argument('--no-prediction', action='store_true', help='Disable prediction system')
    parser.add_argument('--no-video', action='store_true', help='Skip video output')
    parser.add_argument('--no-frames', action='store_true', help='Skip frame saving')
    
    # OC-SORT specific parameters
    parser.add_argument('--iou-active', type=float, default=0.3, 
                       help='IoU threshold for active track matching')
    parser.add_argument('--iou-lost', type=float, default=0.2, 
                       help='IoU threshold for lost track re-association')
    parser.add_argument('--max-age', type=int, default=30, 
                       help='Maximum frames to keep lost tracks')
    parser.add_argument('--min-hits', type=int, default=2, 
                       help='Minimum hits before track confirmation')
    parser.add_argument('--motion-lambda', type=float, default=0.5, 
                       help='Weight for motion penalty in matching')
    
    # Shot analysis
    parser.add_argument('--shot-analysis', action='store_true', 
                       help='Enable shot analysis and filtering')
    parser.add_argument('--min-shot-distance', type=float, default=50, 
                       help='Minimum distance to net for shot detection')
    
    return parser.parse_args()


def main():
    """Main function with OC-SORT integration"""
    args = parse_arguments()
    
    # Verify input files
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    if not os.path.exists(args.puck_model):
        print(f"❌ Puck model not found: {args.puck_model}")
        return
    
    if args.net_model and not os.path.exists(args.net_model):
        print(f"⚠️ Net model not found: {args.net_model}")
        args.net_model = None
    
    print("🚀 Starting OC-SORT Hockey Puck Tracking")
    print("="*70)
    print(f"📹 Video: {args.video_path}")
    print(f"🏒 Puck Model: {args.puck_model}")
    print(f"🥅 Net Model: {args.net_model if args.net_model else 'Not provided'}")
    print(f"🔮 Prediction: {args.prediction_method if not args.no_prediction else 'Disabled'}")
    print("="*70)
    
    # Configure OC-SORT parameters
    oc_sort_config = {
        'iou_thresh_active': args.iou_active,
        'iou_thresh_lost': args.iou_lost,
        'max_age': args.max_age,
        'min_hits': args.min_hits,
        'det_score_thresh': args.confidence,
        'motion_lambda': args.motion_lambda
    }
    
    print(f"⚙️ OC-SORT Configuration:")
    for key, value in oc_sort_config.items():
        print(f"   {key}: {value}")
    
    # Create tracker
    tracker = OCSortHockeyTracker(
        yolo_model_path=args.puck_model,
        confidence_threshold=args.confidence,
        enable_prediction=not args.no_prediction,
        prediction_method=args.prediction_method,
        net_model_path=args.net_model,
        oc_sort_config=oc_sort_config
    )
    
    # Process video
    trajectories = tracker.process_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        save_frames=not args.no_frames,
        save_video=not args.no_video,
        max_frames=args.max_frames
    )
    
    # Shot analysis if enabled
    if args.shot_analysis and args.net_model:
        print(f"\n🎯 Performing shot analysis...")
        shot_trajectories = tracker.merge_and_filter_shots(args.min_shot_distance)
        
        if shot_trajectories:
            print(f"✅ Found {len(shot_trajectories)} valid shots")
        else:
            print("❌ No valid shots detected")
    
    # Create visualizations
    tracker.create_trajectory_plots(args.output_dir)
    
    # Print summary
    tracker.print_summary()
    
    print(f"\n🎉 OC-SORT TRACKING COMPLETE!")
    print(f"📁 Results saved to: {args.output_dir}/")
    print(f"📊 Outputs:")
    print(f"   - oc_sort_hockey_tracking.mp4 (annotated video)")
    print(f"   - oc_sort_trajectories.png (trajectory overview)")
    print(f"   - oc_sort_frame_*.jpg (sample frames)")
    
    # Performance comparison
    print(f"\n📈 OC-SORT ADVANTAGES:")
    print(f"   ✅ Observation-centric tracking (handles occlusion well)")
    print(f"   ✅ Two-stage association (active + lost track recovery)")
    print(f"   ✅ Motion-aware matching (velocity + IoU)")
    print(f"   ✅ Configurable parameters for hockey optimization")
    print(f"   ✅ Robust track lifecycle management")


if __name__ == "__main__":
    main()