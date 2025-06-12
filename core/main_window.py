"""
Main window controller for Swingman CV Test
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from utils import draw_path, draw_status_bar, draw_logo, get_fps, draw_instructions

class MainWindow:
    """Main window controller class"""
    
    def __init__(self, tracker, camera_index=0, video_path=None, demo_mode=False, display_backend='cv2'):
        # Initialize parameters
        self.display_backend = display_backend
        
        # Components
        self.tracker = tracker
        self.bat_grid = None
        self.impact_detector = None
        self.swing_analyzer = None
        
        # Camera or video
        self.camera_index = camera_index
        self.video_path = video_path
        self.capture = None
        self.frame_size = (640, 480)
        
        # State
        self.running = False
        self.demo_mode = demo_mode
        self.analysis_mode = False
        self.prev_frame = None
        self.start_time = 0
        self.frame_count = 0
        self.mouse_position = (0, 0)  # Initialize mouse position
        
        # Recording
        self.recording = False
        self.output_video = None
        self.record_frames = []
        
        # Import components here to avoid circular imports
        from core.bat_grid import BatGrid
        from core.impact_detector import ImpactDetector
        from core.swing_analyzer import SwingAnalyzer
        
        self.bat_grid = BatGrid()
        self.impact_detector = ImpactDetector()
        self.swing_analyzer = SwingAnalyzer()
    
    def setup_video_source(self):
        """Setup the video source (camera or file)"""
        if self.video_path:
            self.capture = cv2.VideoCapture(self.video_path)
        else:
            # Try multiple camera indices if the first one fails
            for i in range(3):  # Try camera indices 0, 1, 2
                self.capture = cv2.VideoCapture(i)
                if self.capture.isOpened():
                    self.camera_index = i
                    print(f"Successfully opened camera index {i}")
                    break
            
            # Set camera properties for best results
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if opened successfully
        if not self.capture.isOpened():
            print("Error: Could not open video source")
            return False
        
        # Get actual frame size
        ret, test_frame = self.capture.read()
        if ret:
            self.frame_size = (test_frame.shape[1], test_frame.shape[0])
            print(f"Video source size: {self.frame_size}")
        else:
            print("Warning: Could not read initial frame")
        
        return True
    
    def run(self):
        """Run the main application loop"""
        # Setup video source
        if not self.setup_video_source():
            print("Failed to setup video source. Exiting.")
            return
        
        # Select display backend
        if self.display_backend == 'cv2':
            self.run_opencv_display()
        elif self.display_backend == 'matplotlib':
            self.run_matplotlib_display()
        else:
            print(f"Unsupported display backend: {self.display_backend}")
            print("Falling back to OpenCV display")
            self.run_opencv_display()
    
    def run_opencv_display(self):
        """Run the application with OpenCV display"""
        try:
            # Create window
            window_name = "Swingman Bat Tracker"
            cv2.namedWindow(window_name)
            
            # Set mouse callback
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            # Main loop
            self.running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            print("Click on the bat to start tracking")
            print("Press 'q' to quit, 's' to stop tracking, 'r' to reset, 'v' to save video")
            
            while self.running:
                # Read frame
                ret, frame = self.capture.read()
                if not ret:
                    if self.video_path:  # If video file ended, loop it
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Error reading from camera, retrying...")
                        # Try to reopen the camera
                        self.capture.release()
                        self.capture = cv2.VideoCapture(self.camera_index)
                        if not self.capture.isOpened():
                            print("Could not reopen camera. Exiting.")
                            break
                        continue
                
                # Update frame count for FPS calculation
                self.frame_count += 1
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Show FPS
                fps = get_fps(self.start_time, self.frame_count)
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Record frame if recording is active
                if self.recording and self.output_video is not None:
                    self.output_video.write(processed_frame)
                elif self.recording and self.output_video is None:
                    self.record_frames.append(processed_frame.copy())
                
                # Show frame
                cv2.imshow(window_name, processed_frame)
                
                # Process keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('s') and self.tracker.is_tracking:
                    self.complete_tracking()
                elif key == ord('r'):
                    self.reset_tracking()
                elif key == ord('v'):
                    self.toggle_recording()
                elif key == ord('d'):
                    # Toggle demo mode
                    self.demo_mode = not self.demo_mode
                    print(f"Demo mode: {'ON' if self.demo_mode else 'OFF'}")
            
            # Clean up
            if self.output_video is not None:
                self.output_video.release()
            self.capture.release()
            cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"Error in OpenCV display: {e}")
            import traceback
            traceback.print_exc()
    
    def run_matplotlib_display(self):
        """Run the application with matplotlib display"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import tkinter as tk
            
            # Create Tkinter window
            self.root = tk.Tk()
            self.root.title("Swingman Bat Tracker")
            self.root.geometry("800x600")
            self.root.protocol("WM_DELETE_WINDOW", self.quit)
            
            # Create matplotlib figure
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Bind mouse events
            self.canvas.mpl_connect('button_press_event', self._mpl_mouse_callback)
            
            # Add controls
            control_frame = tk.Frame(self.root)
            control_frame.pack(fill=tk.X)
            
            tk.Button(control_frame, text="Stop Tracking", command=self.complete_tracking).pack(side=tk.LEFT, padx=5)
            tk.Button(control_frame, text="Reset", command=self.reset_tracking).pack(side=tk.LEFT, padx=5)
            tk.Button(control_frame, text="Record", command=self.toggle_recording).pack(side=tk.LEFT, padx=5)
            tk.Button(control_frame, text="Quit", command=self.quit).pack(side=tk.RIGHT, padx=5)
            
            # Status label
            self.status_label = tk.Label(self.root, text="Click on the bat to start tracking")
            self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Start update loop
            self.running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            # Schedule first update
            self.root.after(100, self._update_mpl)
            
            # Start Tkinter mainloop
            self.root.mainloop()
        
        except Exception as e:
            print(f"Error in matplotlib display: {e}")
            print("Make sure matplotlib and tkinter are installed:")
            print("pip install matplotlib")
            print("sudo apt-get install python3-tk")
            import traceback
            traceback.print_exc()
    
    def _mpl_mouse_callback(self, event):
        """Handle matplotlib mouse events"""
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.mouse_position = (int(event.xdata), int(event.ydata))
            if not self.tracker.is_tracking and not self.analysis_mode:
                print(f"Starting tracking at ({self.mouse_position[0]}, {self.mouse_position[1]})")
                # Get current frame
                ret, frame = self.capture.read()
                if ret:
                    # Start tracking
                    success = self.tracker.start_tracking(frame, self.mouse_position)
                    if success:
                        # Start impact detection
                        self.impact_detector.start_monitoring()
                        self.status_label.config(text="Tracking started")
                        print("Tracking started")
    
    def _update_mpl(self):
        """Update matplotlib display"""
        if not self.running:
            return
        
        # Read frame
        ret, frame = self.capture.read()
        if not ret:
            if self.video_path:  # If video file ended, loop it
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                print("Error reading from camera, retrying...")
                # Try to reopen the camera
                self.capture.release()
                self.capture = cv2.VideoCapture(self.camera_index)
                if not self.capture.isOpened():
                    print("Could not reopen camera. Exiting.")
                    self.quit()
                    return
            
            self.root.after(100, self._update_mpl)
            return
        
        # Update frame count
        self.frame_count += 1
        
        # Process frame
        processed_frame = self.process_frame(frame)
        
        # Display frame in matplotlib
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update status
        fps = get_fps(self.start_time, self.frame_count)
        status_text = f"FPS: {fps:.1f} | "
        
        if self.tracker.is_tracking:
            status_text += "Tracking active - Press 'Stop Tracking'"
        elif self.analysis_mode:
            status_text += "Analysis mode - Press 'Reset' to start over"
        else:
            status_text += "Click on the bat to start tracking"
        
        self.status_label.config(text=status_text)
        
        # Record frame if recording is active
        if self.recording and self.output_video is not None:
            self.output_video.write(processed_frame)
        elif self.recording and self.output_video is None:
            self.record_frames.append(processed_frame.copy())
        
        # Schedule next update
        self.root.after(30, self._update_mpl)
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Store original frame
        original_frame = frame.copy()
        
        # Add mouse position tracking for better interaction
        if hasattr(self, 'mouse_position'):
            cv2.circle(frame, self.mouse_position, 5, (0, 255, 255), -1)
        
        # Update tracking
        if self.tracker.is_tracking:
            success, box = self.tracker.update_tracking(frame)
            
            if success:
                # Get current tracking point
                if len(self.tracker.path_points) > 0:
                    current_point = self.tracker.path_points[-1]
                    
                    # Get bat angle
                    angle = self.tracker.get_bat_angle()
                    
                    # Draw tracking box
                    try:
                        x, y, w, h = [int(v) for v in box]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except:
                        # Handle case where box might not be properly formatted
                        pass
                    
                    # Draw bat grid
                    self.bat_grid.draw_grid(frame, current_point, angle)
                    
                    # Check for impact if not in analysis mode
                    if not self.analysis_mode:
                        has_impact, impact_point = self.impact_detector.detect_impact(
                            frame, current_point)
                        
                        if has_impact:
                            print("Impact detected!")
                            self.analysis_mode = True
                            self.complete_tracking()
                            
                            # Demo mode: stop after impact
                            if self.demo_mode:
                                self.tracker.is_tracking = False
                        
                        # In demo mode, simulate impact after certain number of frames
                        elif self.demo_mode and len(self.tracker.path_points) > 30 and not self.impact_detector.has_detected_impact:
                            print("Demo mode: Simulating impact")
                            self.impact_detector.has_detected_impact = True
                            self.impact_detector.impact_point = current_point
                            self.analysis_mode = True
                            self.complete_tracking()
            else:
                # Tracking lost
                self.tracker.is_tracking = False
                if len(self.tracker.path_points) > 10:
                    self.analysis_mode = True
                    self.complete_tracking()
        
        # Draw impact visualization
        self.impact_detector.draw_impact(frame)
        
        # Draw path
        if len(self.tracker.path_points) > 1:
            draw_path(frame, self.tracker.path_points, glow=True)
        
        # Draw analysis in analysis mode
        if self.analysis_mode:
            self.swing_analyzer.draw_analysis(frame, self.tracker.path_points)
        
        # Add logo
        draw_logo(frame)
        
        # Add instructions
        if not self.tracker.is_tracking and not self.analysis_mode:
            draw_instructions(frame, "Click on the bat to start tracking")
        elif self.tracker.is_tracking:
            draw_instructions(frame, "Tracking active - Press 's' to stop")
        
        # Add recording indicator
        if self.recording:
            # Red circle in corner
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 80, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add demo mode indicator
        if self.demo_mode:
            cv2.putText(frame, "DEMO MODE", (frame.shape[1] - 150, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Store frame for next iteration
        self.prev_frame = original_frame
        
        return frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        # Store current mouse position for interaction
        self.mouse_position = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.tracker.is_tracking and not self.analysis_mode:
                print(f"Starting tracking at ({x}, {y})")
                
                # Get current frame
                ret, frame = self.capture.read()
                if ret:
                    # Start tracking
                    success = self.tracker.start_tracking(frame, (x, y))
                    
                    if success:
                        # Start impact detection
                        self.impact_detector.start_monitoring()
                        print("Tracking started")
    
    def complete_tracking(self):
        """Complete the tracking and analyze results"""
        if not self.tracker.is_tracking:
            return
            
        # Stop tracking
        path_points = self.tracker.stop_tracking()
        
        # Stop impact detection
        has_impact, impact_point = self.impact_detector.stop_monitoring()
        
        # Analyze swing
        efficiency = self.swing_analyzer.analyze_swing(
            path_points, 
            impact_point=impact_point
        )
        
        print(f"Tracking completed. Efficiency score: {efficiency}%")
        print(f"Swing plane: {self.swing_analyzer.swing_plane}")
        print(f"Swing path: {self.swing_analyzer.swing_path}")
        
        # Enter analysis mode
        self.analysis_mode = True
    
    def reset_tracking(self):
        """Reset tracking and analysis state"""
        self.tracker.stop_tracking()
        self.impact_detector.stop_monitoring()
        self.analysis_mode = False
        print("Tracking reset")
    
    def toggle_recording(self):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            self.recording = True
            
            # Create output directory if it doesn't exist
            os.makedirs('recordings', exist_ok=True)
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recordings/swing_{timestamp}.mp4"
            
            # Create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec
            self.output_video = cv2.VideoWriter(
                output_path, fourcc, 30.0, self.frame_size)
            
            print(f"Started recording to {output_path}")
        else:
            # Stop recording
            self.recording = False
            if self.output_video is not None:
                self.output_video.release()
                self.output_video = None
                print("Recording saved")
            elif len(self.record_frames) > 0:
                # If we've been collecting frames but haven't created a writer yet
                os.makedirs('recordings', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"recordings/swing_{timestamp}.mp4"
                
                if len(self.record_frames) > 0 and self.record_frames[0] is not None:
                    h, w = self.record_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
                    
                    for frame in self.record_frames:
                        out.write(frame)
                    
                    out.release()
                    print(f"Recording saved to {output_path}")
                
                self.record_frames = []
    
    def quit(self):
        """Quit the application"""
        self.running = False
        if hasattr(self, 'root'):
            self.root.quit()