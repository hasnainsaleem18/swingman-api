"""
Impact detection system
"""

import cv2
import numpy as np
from collections import deque

class ImpactDetector:
    """Detects the moment of impact between bat and ball"""
    
    def __init__(self, buffer_size=5):
        # Detection state
        self.is_monitoring = False
        self.has_detected_impact = False
        self.impact_point = None
        self.impact_frame = None
        
        # Analysis buffers
        self.brightness_values = deque(maxlen=buffer_size)
        self.frame_diffs = deque(maxlen=buffer_size)
        self.last_frame = None
        
        # Detection parameters
        self.brightness_threshold = 15.0
        self.motion_threshold = 10000
        self.buffer_size = buffer_size
    
    def start_monitoring(self):
        """Start monitoring for impact"""
        self.is_monitoring = True
        self.has_detected_impact = False
        self.impact_point = None
        self.impact_frame = None
        self.brightness_values.clear()
        self.frame_diffs.clear()
        self.last_frame = None
    
    def stop_monitoring(self):
        """Stop monitoring for impact"""
        self.is_monitoring = False
        return self.has_detected_impact, self.impact_point
    
    def detect_impact(self, frame, tracking_point=None):
        """
        Analyze frame for potential impact
        Returns: (has_impact, impact_point)
        """
        if not self.is_monitoring or self.has_detected_impact:
            return False, None
        
        # Store a copy of the frame for visualization if impact is detected
        current_frame = frame.copy()
        
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = np.mean(gray)
        self.brightness_values.append(current_brightness)
        
        # Calculate frame difference if we have a previous frame
        if self.last_frame is not None:
            diff = cv2.absdiff(self.last_frame, frame)
            diff_value = np.sum(diff) / (diff.shape[0] * diff.shape[1])
            self.frame_diffs.append(diff_value)
            
            # Focus analysis near tracking point if provided
            if tracking_point is not None:
                x, y = tracking_point
                # Create a region of interest around the tracking point
                roi_size = 100
                roi_x1 = max(0, x - roi_size//2)
                roi_y1 = max(0, y - roi_size//2)
                roi_x2 = min(frame.shape[1], x + roi_size//2)
                roi_y2 = min(frame.shape[0], y + roi_size//2)
                
                if roi_x1 < roi_x2 and roi_y1 < roi_y2:
                    roi = diff[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi_diff_value = np.sum(roi) / (roi.shape[0] * roi.shape[1])
                    
                    # ROI difference is weighted higher
                    diff_value = (diff_value + roi_diff_value * 3) / 4
            
            # Check for impact using both brightness and motion
            if len(self.brightness_values) >= self.buffer_size and len(self.frame_diffs) >= self.buffer_size:
                # Calculate recent averages
                avg_brightness = sum(list(self.brightness_values)[:-1]) / (len(self.brightness_values) - 1)
                avg_diff = sum(list(self.frame_diffs)[:-1]) / (len(self.frame_diffs) - 1)
                
                # Current values
                current_diff = self.frame_diffs[-1]
                
                # Check for sudden changes
                brightness_delta = abs(current_brightness - avg_brightness)
                diff_delta = current_diff - avg_diff
                
                # Impact detected if both brightness and motion change significantly
                if (brightness_delta > self.brightness_threshold and 
                    diff_delta > self.motion_threshold):
                    self.has_detected_impact = True
                    self.impact_point = tracking_point if tracking_point else (frame.shape[1]//2, frame.shape[0]//2)
                    self.impact_frame = current_frame
                    return True, self.impact_point
        
        # Update last frame
        self.last_frame = frame.copy()
        
        return False, None
    
    def draw_impact(self, frame):
        """Draw impact visualization on the frame"""
        if not self.has_detected_impact or self.impact_point is None:
            return frame
        
        # Draw impact marker
        x, y = self.impact_point
        radius = 20
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw filled circle
        cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1)
        
        # Draw outer ring
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
        cv2.circle(frame, (x, y), radius+5, (0, 0, 255), 1)
        
        # Apply transparency
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Add impact text
        cv2.putText(frame, "IMPACT", (x - 40, y - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame