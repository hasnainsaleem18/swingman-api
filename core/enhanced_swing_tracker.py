"""
Enhanced Swing Tracker - Integration with YOLO Detector and Pose Analyzer
"""

import cv2
import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# Import your modules
from .yolo_detector import YoloDetector
from .pose_analyzer import PoseAnalyzer
from .swing_analyzer import SwingAnalyzer
from .impact_detector import ImpactDetector
from .heatmap_generator import HeatmapGenerator
from .bat_visualizer import BatVisualizer

@dataclass
class SwingMetrics:
    """Complete swing metrics"""
    efficiency_score: int = 0
    swing_plane: str = "Unknown"  # Upward, Level, Downward
    swing_path: str = "Unknown"   # Inside-Out, Straight, Outside-In
    swing_speed: float = 0.0
    sweet_spot_contact: bool = False
    impact_point: Optional[Tuple[int, int]] = None
    path_consistency: int = 0
    power_score: int = 0
    follow_through: int = 0
    pose_stability: int = 0

class EnhancedSwingTracker:
    def __init__(self, custom_bat_model_path=None, enable_pose=True):
        """Initialize the Enhanced Swing Tracker"""
        print("🚀 Initializing Enhanced Swing Tracker...")
        
        # Initialize all detection systems
        self.yolo_detector = YoloDetector(custom_bat_model_path)
        self.pose_analyzer = PoseAnalyzer() if enable_pose else None
        self.swing_analyzer = SwingAnalyzer()
        self.impact_detector = ImpactDetector()
        
        # Tracking state
        self.is_tracking = False
        self.swing_in_progress = False
        
        # Data storage with reduced thresholds
        self.swing_path_points = deque(maxlen=50)
        self.bat_positions = deque(maxlen=25)
        self.ball_path_points = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)
        self.pose_history = deque(maxlen=15)
        
        # Current swing data
        self.current_swing = SwingMetrics()
        self.best_bat_detection = None
        self.best_ball_detection = None
        self.last_impact_point = None
        
        print("✅ Enhanced Swing Tracker initialized!")

    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        current_time = time.time()
        frame_analyzed = False
        
        # Run YOLO detection
        detections = self.yolo_detector.detect_objects(frame)
        
        # Run pose analysis
        pose_data = None
        if self.pose_analyzer:
            pose_data = self.pose_analyzer.analyze_pose(frame)
            if pose_data['is_detected']:
                self.pose_history.append(pose_data)
                frame = self.pose_analyzer.draw_pose(frame, pose_data)
        
        # Get best bat and ball detections
        best_bat = self.yolo_detector.get_best_bat_detection(detections, min_confidence=0.01)
        best_ball = self.yolo_detector.get_best_ball_detection(detections, min_confidence=0.01)
        
        # Track movement if tracking is active
        if self.is_tracking:
            tracking_point = None
            
            # Try to get tracking point from different sources
            if hasattr(self, 'current_position'):
                # Mouse tracking
                tracking_point = self.current_position
            elif best_bat:
                # Bat detection tracking
                tracking_point = best_bat['center']
            elif pose_data and pose_data['is_detected'] and len(pose_data['landmarks']) > 16:
                # Use wrist position from pose as fallback
                right_wrist = pose_data['landmarks'][16]  # Right wrist landmark
                tracking_point = right_wrist
            
            if tracking_point:
                # Always add first point
                if len(self.swing_path_points) == 0:
                    self.swing_path_points.append(tracking_point)
                    self.timestamps.append(current_time)
                    print("Started tracking swing")
                # Add subsequent points with minimal movement check
                elif self._has_significant_movement(tracking_point, self.swing_path_points[-1], min_distance=1):
                    self.swing_path_points.append(tracking_point)
                    self.timestamps.append(current_time)
                    
                    # Calculate metrics in real-time
                    if len(self.swing_path_points) >= 2:
                        self._update_metrics_realtime()
                        print(f"Swing progress: {len(self.swing_path_points)} points")
            
            # Check for impact
        if best_ball and len(self.swing_path_points) > 0:
            ball_center = best_ball['center']
            last_point = self.swing_path_points[-1]
            distance = np.sqrt((ball_center[0] - last_point[0])**2 + 
                             (ball_center[1] - last_point[1])**2)
            if distance < 100:  # Impact detection radius
                self.last_impact_point = ball_center
                frame_analyzed = True
        
        # Auto-complete swing if enough movement
        if self.is_tracking and len(self.swing_path_points) >= 2:
            total_distance = self._calculate_path_distance(self.swing_path_points)
            if total_distance > 30:  # Very lenient threshold
                frame_analyzed = True
                self._update_metrics_realtime()
                if not self.last_impact_point and len(self.swing_path_points) > 0:
                    self.last_impact_point = self.swing_path_points[-1]
        
        # Draw detections on frame
        if best_bat or best_ball:
            frame = self.yolo_detector.draw_detections(frame.copy(), detections)
        
        # Return all detection data for UI to handle
        return {
            'frame': frame,
            'pose_data': pose_data,
            'best_bat': best_bat,
            'best_ball': best_ball,
            'swing_path': list(self.swing_path_points),
            'metrics': self.get_current_metrics(),
            'impact_point': self.last_impact_point
        }

    def update_current_position(self, x, y):
        """Update the current tracking position"""
        self.current_position = (x, y)
        
    def _has_significant_movement(self, current_pos, last_pos, min_distance=2):
        """Check if there's significant movement between positions"""
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance >= min_distance

    def _calculate_path_distance(self, points):
        """Calculate total distance of the path"""
        total_distance = 0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        return total_distance

    def complete_swing_analysis(self):
        """Analyze completed swing"""
        if len(self.swing_path_points) < 5:  # Reduced from 10
            return
        
        print("📊 Analyzing swing...")
        
        path_points = list(self.swing_path_points)
        timestamps = list(self.timestamps)
        
        # Calculate swing speed and movement
        total_distance = self._calculate_path_distance(path_points)
        time_diff = timestamps[-1] - timestamps[0]
        swing_speed = total_distance / time_diff if time_diff > 0 else 0
        
        # Run swing analysis
        efficiency = self.swing_analyzer.analyze_swing(
            path_points, timestamps, self.last_impact_point
        )
        
        # Update metrics
        self.current_swing.efficiency_score = efficiency
        self.current_swing.swing_plane = self.swing_analyzer.swing_plane
        self.current_swing.swing_path = self.swing_analyzer.swing_path
        self.current_swing.swing_speed = swing_speed
        self.current_swing.impact_point = self.last_impact_point
        
        # Calculate additional metrics
        self._calculate_advanced_metrics(path_points)
        
        # Add to heatmap
        if self.last_impact_point and self.best_bat_detection:
            bat_center = self.best_bat_detection['center']
            bat_angle = self._calculate_bat_angle()
            
            self.heatmap_generator.add_impact_point(
                point=self.last_impact_point,
                bat_center=bat_center,
                bat_angle=bat_angle,
                efficiency_score=self.current_swing.efficiency_score
            )
        
        print(f"✅ Swing analyzed - Efficiency: {self.current_swing.efficiency_score}%")
        self.save_swing_session()  # Save before clearing
        self.clear_current_swing()

    def _calculate_advanced_metrics(self, path_points):
        """Calculate additional swing metrics"""
        # Path consistency
        if len(path_points) >= 5:
            deviations = []
            for i in range(2, len(path_points)):
                p1, p2, p3 = path_points[i-2], path_points[i-1], path_points[i]
                expected_x = p1[0] + (p3[0] - p1[0]) * 0.5
                expected_y = p1[1] + (p3[1] - p1[1]) * 0.5
                deviation = math.sqrt((p2[0] - expected_x)**2 + (p2[1] - expected_y)**2)
                deviations.append(deviation)
            
            if deviations:
                avg_deviation = sum(deviations) / len(deviations)
                self.current_swing.path_consistency = max(0, min(100, int(100 - avg_deviation * 2)))
        
        # Power score
        speed_score = min(100, self.current_swing.swing_speed * 2)
        self.current_swing.power_score = int((speed_score + self.current_swing.efficiency_score) / 2)
        
        # Follow through
        if self.last_impact_point and len(path_points) > 10:
            impact_index = len(path_points) // 2
            follow_points = path_points[impact_index:]
            
            if len(follow_points) > 3:
                follow_distance = sum(
                    math.sqrt((follow_points[i][0] - follow_points[i-1][0])**2 + 
                             (follow_points[i][1] - follow_points[i-1][1])**2)
                    for i in range(1, len(follow_points))
                )
                self.current_swing.follow_through = min(100, int(follow_distance / 2))
        
        # Pose stability
        if self.pose_history:
            stability_scores = [pose['stability_score'] for pose in self.pose_history]
            self.current_swing.pose_stability = int(sum(stability_scores) / len(stability_scores))
        
        # Sweet spot detection
        self.current_swing.sweet_spot_contact = self.current_swing.efficiency_score >= 70

    def _calculate_bat_angle(self):
        """Calculate bat angle from recent positions"""
        if len(self.bat_positions) < 2:
            return 0.0
        p1, p2 = self.bat_positions[-2], self.bat_positions[-1]
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    def draw_analysis_overlays(self, frame, detections, pose_data):
        """Draw all analysis overlays"""
        # 1. YOLO detections
        if self.show_yolo_detections:
            frame = self.yolo_detector.draw_detections(frame, detections)
        
        # 2. Ball path
        if len(self.ball_path_points) > 1:
            self._draw_ball_path(frame)
        
        # 3. Swing path
        if len(self.swing_path_points) > 1:
            self._draw_swing_path(frame)
        
        # 4. Bat visualization
        if self.best_bat_detection:
            self._draw_enhanced_bat_overlay(frame)
        
        # 5. Impact detection
        if self.last_impact_point:
            frame = self.impact_detector.draw_impact(frame)
        
        # 6. Draw pose analysis in left corner
        if self.show_pose_overlay and pose_data and self.pose_analyzer:
            # Left corner panel for pose
            panel_height = 150
            panel_width = 200
            padding = 10
            
            # Create semi-transparent overlay for left panel
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (padding, padding), 
                         (panel_width + padding, panel_height + padding), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame,
                         (padding, padding),
                         (panel_width + padding, panel_height + padding),
                         (255, 255, 255), 1)
            
            # Draw pose metrics
            cv2.putText(frame, "POSE ANALYSIS", (padding + 10, padding + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_pos = padding + 60
            if self.pose_history:
                stability = sum(pose['stability_score'] for pose in self.pose_history) / len(self.pose_history)
                cv2.putText(frame, f"Stability: {stability:.0f}%", (padding + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._get_score_color(stability), 1)
        
        # 7. Draw swing analysis in right corner
        if self.show_swing_analysis:
            # Right corner panel for swing
            panel_height = 200
            panel_width = 200
            padding = 10
            x_start = frame.shape[1] - panel_width - padding
            
            # Create semi-transparent overlay for right panel
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (x_start, padding), 
                         (x_start + panel_width, panel_height + padding), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame,
                         (x_start, padding),
                         (x_start + panel_width, panel_height + padding),
                         (255, 255, 255), 1)
            
            # Draw swing metrics
            cv2.putText(frame, "SWING ANALYSIS", (x_start + 10, padding + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_pos = padding + 60
            metrics = [
                ("Efficiency", f"{self.current_swing.efficiency_score}%"),
                ("Power", f"{self.current_swing.power_score}%"),
                ("Speed", f"{self.current_swing.swing_speed:.1f}"),
                ("Consistency", f"{self.current_swing.path_consistency}%"),
                ("Follow-Through", f"{self.current_swing.follow_through}%")
            ]
            
            for label, value in metrics:
                color = (255, 255, 255)
                if "%" in str(value) and value != "0%":
                    try:
                        percent_val = int(value.replace("%", ""))
                        color = self._get_score_color(percent_val)
                    except:
                        pass
                
                cv2.putText(frame, f"{label}: {value}", (x_start + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 25
            
            # Sweet spot indicator
            if self.current_swing.sweet_spot_contact:
                cv2.putText(frame, "✓ SWEET SPOT!", (x_start + 10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def _draw_ball_path(self, frame):
        """Draw ball movement path with bright green trail"""
        points = list(self.ball_path_points)
        
        # Draw path with increasing thickness and brightness
        for i in range(1, len(points)):
            progress = i / len(points)
            
            # Bright green color scheme with increasing intensity
            green_intensity = int(155 + 100 * progress)  # 155-255 range
            color = (0, green_intensity, 0)
            
            # Increased line thickness for better visibility
            thickness = max(2, int(3 * progress))
            
            # Draw main line
            cv2.line(frame, points[i-1], points[i], color, thickness)
            
            # Add glow effect for recent positions
            if progress > 0.7:
                cv2.line(frame, points[i-1], points[i], (255, 255, 255), 1)
        
        # Draw current ball position with emphasis
        if points:
            current_point = points[-1]
            # Outer glow
            cv2.circle(frame, current_point, 8, (0, 255, 0), 2)
            # Inner bright point
            cv2.circle(frame, current_point, 4, (255, 255, 255), -1)

    def _draw_swing_path(self, frame):
        """Draw swing path with enhanced visibility"""
        points = list(self.swing_path_points)
        
        # Draw path with increasing thickness and brightness
        for i in range(1, len(points)):
            progress = i / len(points)
            
            # Enhanced color scheme
            if progress < 0.3:
                color = (200, 0, 0)      # Blue start
            elif progress < 0.6:
                color = (0, 200, 0)      # Green middle
            else:
                color = (0, 200, 255)    # Bright yellow end
            
            # Increased line thickness
            thickness = max(2, int(4 * progress))
            
            # Draw main line
            cv2.line(frame, points[i-1], points[i], color, thickness)
            
            # Draw glow effect
            if progress > 0.5:
                cv2.line(frame, points[i-1], points[i], (255, 255, 255), 1)
        
        # Draw current point with emphasis
        if points:
            current_point = points[-1]
            cv2.circle(frame, current_point, 6, (0, 255, 255), -1)
            cv2.circle(frame, current_point, 8, (0, 255, 255), 1)
        
        # Impact point
        if self.last_impact_point:
            cv2.circle(frame, self.last_impact_point, 12, (0, 0, 255), 3)
            cv2.circle(frame, self.last_impact_point, 6, (255, 255, 255), -1)

    def _draw_enhanced_bat_overlay(self, frame):
        """Draw bat with efficiency indicator"""
        if not self.best_bat_detection:
            return
        
        center = self.best_bat_detection['center']
        angle = self._calculate_bat_angle()
        
        # Realistic bat visualization
        frame = self.bat_visualizer.draw_realistic_bat(frame, center, angle, True)
        
        # Efficiency indicator
        if self.current_swing.efficiency_score > 0:
            score = self.current_swing.efficiency_score
            
            # Color based on efficiency
            if score >= 80:
                color = (0, 255, 0)
            elif score >= 60:
                color = (0, 255, 255)
            elif score >= 40:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            # Efficiency ring
            cv2.circle(frame, center, 30, color, 5)
            
            # Score text
            text = f"{score}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _get_score_color(self, score):
        """Get color based on score"""
        if score >= 80:
            return (0, 255, 0)      # Green
        elif score >= 60:
            return (0, 255, 255)    # Yellow
        elif score >= 40:
            return (0, 165, 255)    # Orange
        else:
            return (0, 0, 255)      # Red

    def toggle_visualization(self, viz_type):
        """Toggle visualization overlays"""
        if viz_type == "yolo":
            self.show_yolo_detections = not self.show_yolo_detections
        elif viz_type == "pose":
            self.show_pose_overlay = not self.show_pose_overlay
        elif viz_type == "analysis":
            self.show_swing_analysis = not self.show_swing_analysis
        elif viz_type == "heatmap":
            self.show_heatmap = not self.show_heatmap

    def save_swing_session(self):
        """Save session data"""
        session_dir = self.heatmap_generator.save_session(include_heatmap=True)
        if session_dir:
            print(f"💾 Session saved to: {session_dir}")

    def get_current_metrics(self):
        """Get current swing metrics"""
        # Update metrics in real-time if we have points
        if len(self.swing_path_points) >= 2:
            self._update_metrics_realtime()
        
        # Return all metrics
        return {
            'efficiency_score': self.current_swing.efficiency_score,
            'power_score': self.current_swing.power_score,
            'swing_speed': self.current_swing.swing_speed,
            'path_consistency': self.current_swing.path_consistency,
            'follow_through': self.current_swing.follow_through,
            'pose_stability': self.current_swing.pose_stability,
            'sweet_spot_contact': self.current_swing.sweet_spot_contact,
            'impact_point': self.last_impact_point
        }

    def _update_metrics_realtime(self):
        """Update metrics in real-time during swing"""
        if len(self.swing_path_points) < 2:
            return
            
        # Calculate basic metrics
        total_distance = self._calculate_path_distance(self.swing_path_points)
        time_diff = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.1
        swing_speed = total_distance / time_diff if time_diff > 0 else 0
        
        # Get the last 5 points for recent movement analysis
        recent_points = list(self.swing_path_points)[-5:]
        recent_distance = self._calculate_path_distance(recent_points)
        
        # Update current swing metrics with minimum values
        self.current_swing.swing_speed = max(swing_speed, 1.0)
        self.current_swing.efficiency_score = max(30, min(100, int(swing_speed * 2)))
        self.current_swing.power_score = max(30, min(100, int(swing_speed * 1.5)))
        
        # Calculate path consistency based on recent points
        if len(recent_points) >= 3:
            deviations = []
            for i in range(1, len(recent_points)-1):
                p1, p2, p3 = recent_points[i-1], recent_points[i], recent_points[i+1]
                expected_x = p1[0] + (p3[0] - p1[0]) * 0.5
                expected_y = p1[1] + (p3[1] - p1[1]) * 0.5
                deviation = math.sqrt((p2[0] - expected_x)**2 + (p2[1] - expected_y)**2)
                deviations.append(deviation)
            
            if deviations:
                avg_deviation = sum(deviations) / len(deviations)
                self.current_swing.path_consistency = max(40, min(100, int(100 - avg_deviation)))
            else:
                self.current_swing.path_consistency = 40
        else:
            self.current_swing.path_consistency = 40
        
        # Calculate follow through based on total distance
        self.current_swing.follow_through = max(30, min(100, int(total_distance / 2)))
        
        # Update pose stability if available
        if self.pose_history:
            recent_poses = list(self.pose_history)[-5:]  # Get last 5 poses
            stability_scores = [pose['stability_score'] for pose in recent_poses]
            self.current_swing.pose_stability = max(30, int(sum(stability_scores) / len(stability_scores)))
        else:
            self.current_swing.pose_stability = 50  # Default stability
        
        # Set sweet spot for significant swings
        self.current_swing.sweet_spot_contact = total_distance > 100

    def start_tracking_session(self):
        """Start tracking session"""
        print("🎯 Starting tracking session...")
        self.is_tracking = True
        self.clear_current_swing()
        # Initialize timestamps with current time
        self.timestamps.append(time.time())
        print("✅ Tracking session started")

    def stop_tracking_session(self):
        """Stop tracking session and analyze final swing"""
        print("🛑 Stopping tracking session...")
        print(f"Total tracking points: {len(self.swing_path_points)}")
        
        if len(self.swing_path_points) >= 2:
            total_distance = self._calculate_path_distance(self.swing_path_points)
            print(f"Total swing distance: {total_distance:.1f}")
            
            if total_distance > 30:
                self.is_tracking = False
                self._update_metrics_realtime()
                print("✅ Swing analyzed")
                return True
            else:
                print(f"Swing distance ({total_distance:.1f}) too short - minimum 30 pixels required")
        else:
            print(f"Not enough points ({len(self.swing_path_points)}) - minimum 2 required")
        
        self.clear_current_swing()
        return False

    def clear_current_swing(self):
        """Clear current swing data"""
        self.swing_path_points.clear()
        self.bat_positions.clear()
        self.ball_path_points.clear()
        self.timestamps.clear()
        self.pose_history.clear()
        self.current_swing = SwingMetrics()
        self.best_bat_detection = None
        self.best_ball_detection = None
        self.last_impact_point = None
