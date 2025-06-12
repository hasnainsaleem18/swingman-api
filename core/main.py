#!/usr/bin/env python3
"""
Swingman - Enhanced Baseball Swing Analysis Tool
A clean, modular implementation for easy porting to iOS/Swift
"""

import os
import sys
import cv2
import argparse
from datetime import datetime
import numpy as np
import json
from collections import deque


# Force OpenCV to use xcb backend for Wayland compatibility
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from core.enhanced_swing_tracker import EnhancedSwingTracker
from core.swing_data_manager import SwingDataManager
from core.heatmap_generator import HeatmapGenerator
from core.pose_analyzer import PoseAnalyzer
from utils.drawing import (
    draw_logo, draw_instructions, draw_statistics,
    draw_tracking_box, draw_pose_info
)
from utils.json_encoder import NumpyEncoder, convert_numpy_types

class SwingmanApp:
    """Main application class for Swingman"""
    
    def __init__(self, args):
        """Initialize the Swingman application"""
        self.args = args
        self.setup_components()
        self.setup_window()
        self.setup_state()

    def setup_components(self):
        """Initialize all core components"""
        # Core tracking and analysis
        self.tracker = EnhancedSwingTracker(enable_pose=True)
        self.pose_analyzer = PoseAnalyzer()
        
        # Data management
        self.data_manager = SwingDataManager(base_dir=self.args.output_dir)
        self.session_id = self.data_manager.start_new_session(self.args.session_name)
        
        # Heatmap generation
        self.heatmap_generator = HeatmapGenerator(output_dir=self.args.output_dir)
        
        print(f"Started new session: {self.session_id}")

    def setup_window(self):
        """Setup OpenCV window and camera"""
        self.window_name = "Swingman - Bat Tracker"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Parse window size
        try:
            width, height = map(int, self.args.window_size.split("x"))
            cv2.resizeWindow(self.window_name, width, height)
        except:
            cv2.resizeWindow(self.window_name, 1280, 720)
        
        # Setup camera
        self.setup_camera()
        
        # Mouse callback
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def setup_camera(self):
        """Initialize camera capture"""
        self.capture = None
        for i in range(3):  # Try camera indices 0, 1, 2
            if i == self.args.camera or i == 0:
                capture = cv2.VideoCapture(i)
                if capture.isOpened():
                    self.capture = capture
                    print(f"Successfully opened camera index {i}")
                    
                    # Set camera properties
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    break
        
        if not self.capture:
            raise RuntimeError("Failed to open any camera")

    def setup_state(self):
        """Initialize application state"""
        self.running = True
        self.frame_count = 0
        self.fps = 0
        self.start_time = cv2.getTickCount()
        self.mouse_position = (0, 0)

    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_position = (x, y)
        
        # Start tracking on left click
        if event == cv2.EVENT_LBUTTONDOWN and not self.tracker.is_tracking:
            print(f"Starting tracking at ({x}, {y})")
            self.tracker.start_tracking_session()
        
        # Update current position while tracking
        if self.tracker.is_tracking:
            self.tracker.update_current_position(x, y)
            
        # Stop tracking on right click
        if event == cv2.EVENT_RBUTTONDOWN and self.tracker.is_tracking:
            self.stop_tracking()

    def update_fps(self):
        """Calculate FPS"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            end_time = cv2.getTickCount()
            self.fps = self.frame_count * cv2.getTickFrequency() / (end_time - self.start_time)
            self.start_time = end_time
            self.frame_count = 0

    def process_frame(self, frame):
        """Process a single frame"""
        # Get all detection and tracking data
        results = self.tracker.process_frame(frame.copy())
        
        # Start with the frame that has detections
        processed_frame = results['frame'].copy()
        
        # 1. Draw swing path
        if results['swing_path']:
            self._draw_swing_path(processed_frame, results['swing_path'], results['impact_point'])
        
        # 2. Draw bat visualization if detected
        if results['best_bat']:
            self._draw_bat_overlay(processed_frame, results['best_bat'], results['metrics'])
        
        # 3. Draw metrics panels
        metrics = results['metrics']
        pose_data = results['pose_data']
        
        # Create semi-transparent overlay for pose panel
        if pose_data and pose_data.get('is_detected', False):
            panel_height = 150
            panel_width = 200
            padding = 10
            
            # Create semi-transparent overlay
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, 
                         (padding, padding), 
                         (panel_width + padding, panel_height + padding), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
            cv2.rectangle(processed_frame,
                         (padding, padding),
                         (panel_width + padding, panel_height + padding),
                         (255, 255, 255), 1)
            
            # Draw pose metrics
            cv2.putText(processed_frame, "POSE ANALYSIS", (padding + 10, padding + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            y_pos = padding + 60
            stability = pose_data.get('stability_score', 0)
            cv2.putText(processed_frame, f"Stability: {stability}%", (padding + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._get_score_color(stability), 1)
        
        # Create semi-transparent overlay for swing panel
        panel_height = 200
        panel_width = 200
        padding = 10
        x_start = processed_frame.shape[1] - panel_width - padding
        
        # Create semi-transparent overlay
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, 
                     (x_start, padding), 
                     (x_start + panel_width, panel_height + padding), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
        cv2.rectangle(processed_frame,
                     (x_start, padding),
                     (x_start + panel_width, panel_height + padding),
                     (255, 255, 255), 1)
        
        # Draw swing metrics
        cv2.putText(processed_frame, "SWING ANALYSIS", (x_start + 10, padding + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y_pos = padding + 60
        metric_items = [
            ("Efficiency", f"{metrics.get('efficiency_score', 0)}%"),
            ("Power", f"{metrics.get('power_score', 0)}%"),
            ("Speed", f"{metrics.get('swing_speed', 0.0):.1f}"),
            ("Consistency", f"{metrics.get('path_consistency', 0)}%"),
            ("Follow-Through", f"{metrics.get('follow_through', 0)}%")
        ]
        
        for label, value in metric_items:
            color = (255, 255, 255)
            if "%" in str(value) and value != "0%":
                try:
                    percent_val = int(value.replace("%", ""))
                    color = self._get_score_color(percent_val)
                except:
                    pass
            
            cv2.putText(processed_frame, f"{label}: {value}", (x_start + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 25
        
        # Sweet spot indicator
        if metrics.get('sweet_spot_contact', False):
            cv2.putText(processed_frame, "✓ SWEET SPOT!", (x_start + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add FPS counter
        if self.fps > 0:
            cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return processed_frame

    def _draw_swing_path(self, frame, swing_path, impact_point):
        """Draw swing path with enhanced visibility"""
        for i in range(1, len(swing_path)):
            progress = i / len(swing_path)
            
            # Enhanced color scheme
            if progress < 0.3:
                color = (200, 0, 0)      # Blue start
            elif progress < 0.6:
                color = (0, 200, 0)      # Green middle
            else:
                color = (0, 200, 255)    # Bright yellow end
            
            thickness = max(2, int(4 * progress))
            pt1 = (int(swing_path[i-1][0]), int(swing_path[i-1][1]))
            pt2 = (int(swing_path[i][0]), int(swing_path[i][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
            
            # Draw glow effect
            if progress > 0.5:
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)
        
        # Draw current point with emphasis
        if swing_path:
            current_point = (int(swing_path[-1][0]), int(swing_path[-1][1]))
            cv2.circle(frame, current_point, 6, (0, 255, 255), -1)
            cv2.circle(frame, current_point, 8, (0, 255, 255), 1)
        
        # Impact point
        if impact_point:
            impact = (int(impact_point[0]), int(impact_point[1]))
            cv2.circle(frame, impact, 12, (0, 0, 255), 3)
            cv2.circle(frame, impact, 6, (255, 255, 255), -1)

    def _draw_bat_overlay(self, frame, bat_detection, metrics):
        """Draw bat with efficiency indicator"""
        center = (int(bat_detection['center'][0]), int(bat_detection['center'][1]))
        
        # Draw bat rectangle
        x1, y1, x2, y2 = map(int, bat_detection['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Efficiency indicator
        if metrics.get('efficiency_score', 0) > 0:
            score = metrics['efficiency_score']
            
            # Color based on efficiency
            color = self._get_score_color(score)
            
            # Efficiency ring
            cv2.circle(frame, center, 30, color, 2)
            
            # Score text
            text = f"{score}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

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

    def draw_ui(self, frame):
        """Draw UI elements"""
        # Only draw FPS counter
        if self.fps > 0:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def handle_keypress(self, key):
        """Handle keyboard input"""
        key = chr(key & 0xFF)
        
        if key == 'q':
            self.running = False
        elif key == 's' and self.tracker.is_tracking:
            self.stop_tracking()
        elif key == 'r':
            self.reset_tracking()
        elif key == 'h':
            self.show_heatmap()
        elif key == 'n':
            self.start_new_session()
        elif key == 'e':
            self.export_session()

    def stop_tracking(self):
        """Stop tracking and analyze swing"""
        # Get final frame detections
        ret, frame = self.capture.read()
        if not ret:
            return
            
        # Force swing analysis if we have enough points
        if len(self.tracker.swing_path_points) >= 2:  # Use very lenient minimum
            # Calculate path distance to validate swing
            total_distance = 0
            points = list(self.tracker.swing_path_points)
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                total_distance += (dx*dx + dy*dy)**0.5
            
            if total_distance > 30:  # Very lenient threshold
                # Stop tracking and analyze
                self.tracker.stop_tracking_session()
                metrics = self.tracker.get_current_metrics()
                
                if metrics:
                    # Convert numpy types to Python native types
                    swing_data = {
                        "efficiency_score": int(metrics['efficiency_score']),
                        "power_score": int(metrics['power_score']),
                        "swing_speed": float(metrics['swing_speed']),
                        "path_consistency": int(metrics['path_consistency']),
                        "follow_through": int(metrics['follow_through']),
                        "pose_stability": int(metrics['pose_stability']),
                        "sweet_spot_contact": bool(metrics['sweet_spot_contact']),
                        "impact_point": tuple(map(int, metrics['impact_point'])) if metrics['impact_point'] else None
                    }
                    
                    # Convert path points to tuples of integers
                    path_points = [tuple(map(int, point)) for point in points]
                    
                    # Create a copy of the frame for visualization
                    analyzed_frame = frame.copy()
                    
                    # Draw swing path
                    if path_points:
                        # Draw path line
                        for i in range(1, len(path_points)):
                            cv2.line(analyzed_frame, path_points[i-1], path_points[i], (0, 255, 0), 2)
                        
                        # Draw impact point if exists
                        if swing_data["impact_point"]:
                            cv2.circle(analyzed_frame, swing_data["impact_point"], 5, (0, 0, 255), -1)
                            cv2.circle(analyzed_frame, swing_data["impact_point"], 8, (0, 0, 255), 2)
                    
                    # Draw metrics panel
                    metrics_panel = {
                        "Efficiency": f"{swing_data['efficiency_score']}%",
                        "Power": f"{swing_data['power_score']}%",
                        "Speed": f"{swing_data['swing_speed']:.1f}",
                        "Consistency": f"{swing_data['path_consistency']}%",
                        "Follow Through": f"{swing_data['follow_through']}%",
                        "Pose Stability": f"{swing_data['pose_stability']}%"
                    }
                    
                    # Draw logo and metrics
                    draw_logo(analyzed_frame)
                    draw_statistics(analyzed_frame, metrics_panel)
                    
                    # Add timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(analyzed_frame, timestamp, (10, analyzed_frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                    # Add to data manager with the analyzed frame
                    self.data_manager.add_swing_to_session(swing_data, analyzed_frame, path_points)
                    
                    # Update heatmap if impact point exists
                    if swing_data["impact_point"]:
                        # Calculate bat angle from last few points
                        bat_angle = 0
                        if len(points) >= 2:
                            p1, p2 = points[-2:]
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            bat_angle = np.degrees(np.arctan2(dy, dx))
                        
                        self.heatmap_generator.add_impact_point(
                            point=swing_data["impact_point"],
                            bat_center=tuple(map(int, points[-1])) if points else None,
                            bat_angle=bat_angle,
                            efficiency_score=swing_data["efficiency_score"]
                        )
                    
                    # Print analysis results
                    print("\nSwing Analysis Results:")
                    print("----------------------")
                    print(f"Efficiency Score: {swing_data['efficiency_score']}%")
                    print(f"Power Score: {swing_data['power_score']}%")
                    print(f"Swing Speed: {swing_data['swing_speed']:.1f}")
                    print(f"Path Consistency: {swing_data['path_consistency']}%")
                    print(f"Follow Through: {swing_data['follow_through']}%")
                    print(f"Pose Stability: {swing_data['pose_stability']}%")
                    if swing_data['sweet_spot_contact']:
                        print("✓ Sweet Spot Contact!")
            else:
                print(f"Swing distance ({total_distance:.1f}) too short - minimum 30 pixels required")
                self.tracker.clear_current_swing()
        else:
            print(f"Not enough points ({len(self.tracker.swing_path_points)}) - minimum 2 required")
            self.tracker.clear_current_swing()

    def reset_tracking(self):
        """Reset tracking state"""
        self.tracker.clear_current_swing()
        print("Tracking reset")
        
    def show_heatmap(self):
        """Generate and display heatmap"""
        if len(self.heatmap_generator.normalized_impacts) > 0:
            heatmap_img = self.heatmap_generator.generate_heatmap_image()
            cv2.imshow("Swing Impact Heatmap", heatmap_img)
            
            # Save to session directory
            session_dir = os.path.join(self.args.output_dir, self.data_manager.current_session["id"])
            os.makedirs(session_dir, exist_ok=True)
            cv2.imwrite(os.path.join(session_dir, "impact_heatmap.png"), heatmap_img)
        else:
            print("No impact data available for heatmap")

    def start_new_session(self):
        """Start a new tracking session"""
        self.data_manager.save_current_session()
        new_name = f"{self.args.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if self.args.session_name else None
        self.session_id = self.data_manager.start_new_session(new_name)
        self.heatmap_generator.start_new_session()
        print(f"Started new session: {self.session_id}")

    def export_session(self):
        """Export current session data"""
        if self.data_manager.current_session["swings"]:
            session_dir = self.data_manager.save_current_session()
            self.data_manager.export_data(self.data_manager.current_session["id"], "json")
            self.data_manager.export_data(self.data_manager.current_session["id"], "csv")
            
            if self.heatmap_generator.normalized_impacts:
                self.heatmap_generator.save_session()
            print(f"Exported session data to {session_dir}")
        else:
            print("No swings to export")

    def cleanup(self):
        """Clean up resources"""
        self.data_manager.save_current_session()
        if self.heatmap_generator.normalized_impacts:
            self.heatmap_generator.save_session()
        
        # Close camera
        self.capture.release()
        
        # Close all windows
        cv2.destroyWindow(self.window_name)
        cv2.destroyAllWindows()
        
        print("Application closed")

    def run(self):
        """Main application loop"""
        print("\nSwingman - Baseball Swing Analysis Tool")
        print("=======================================")
        print("Controls:")
        print("  q - Quit application")
        print("  s - Stop tracking and analyze swing")
        print("  r - Reset tracking")
        print("  h - Generate and show heatmap")
        print("  n - Start new session")
        print("  e - Export session data")
        print("=======================================\n")
        
        while self.running:
            # Read frame
            ret, frame = self.capture.read()
            if not ret:
                print("Error reading frame")
                continue
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display frame
            cv2.imshow(self.window_name, processed_frame)
            
            # Update FPS counter
            self.update_fps()
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key != -1:
                self.handle_keypress(key)
        
        self.cleanup()

def main():
    """Entry point for the application"""
    parser = argparse.ArgumentParser(description="Swingman - Baseball Swing Analysis Tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--window-size", type=str, default="1280x720", help="Window size (WxH)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output files")
    parser.add_argument("--session-name", type=str, help="Optional name for the session")
    
    args = parser.parse_args()
    
    try:
        app = SwingmanApp(args)
        app.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())