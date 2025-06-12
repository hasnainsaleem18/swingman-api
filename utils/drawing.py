"""
Drawing utilities for Swingman application
"""

import cv2
import numpy as np

def draw_logo(frame):
    """Draw Swingman logo on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for logo in top-left corner
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (130, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Logo text
    cv2.putText(frame, "Swingman", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_instructions(frame, text):
    """Draw instruction text on frame"""
    h, w = frame.shape[:2]
    
    # Position at bottom of frame
    position = (w//2 - 100, h - 20)
    
    # Semi-transparent background
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (position[0]-5, position[1]-15),
                 (position[0] + text_size[0] + 5, position[1] + 5),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Text
    cv2.putText(frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_statistics(frame, metrics):
    """Draw statistics on frame with clean layout"""
    if not metrics:
        return
    
    h, w = frame.shape[:2]
    
    # Position in top-right corner with padding
    padding = 5
    start_x = w - 120  # Reduced width for stats
    start_y = 25       # Start from top
    
    # Calculate background size
    line_height = 20   # Reduced line height
    total_height = len(metrics) * line_height
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (start_x - padding, start_y - padding),
                 (w - padding, start_y + total_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw metrics
    y = start_y
    for key, value in metrics.items():
        # Use different colors for different types of metrics
        if 'FPS' in key:
            color = (0, 255, 0)  # Green for FPS
        elif 'Speed' in key:
            color = (255, 255, 0)  # Yellow for speed
        elif 'Power' in key:
            color = (0, 255, 255)  # Cyan for power
        else:
            color = (255, 255, 255)  # White for others
            
        text = f"{key}: {value}"
        cv2.putText(frame, text, (start_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += line_height

def draw_tracking_box(frame):
    """Draw tracking zone box"""
    h, w = frame.shape[:2]
    
    # Calculate tracking zone dimensions (centered, 70% of frame)
    zone_width = int(w * 0.7)
    zone_height = int(h * 0.7)
    x1 = (w - zone_width) // 2
    y1 = (h - zone_height) // 2
    x2 = x1 + zone_width
    y2 = y1 + zone_height
    
    # Draw semi-transparent overlay for tracking zone
    overlay = frame.copy()
    # Draw darker overlay outside tracking zone
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    
    # Draw tracking zone border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

def draw_pose_info(frame, metrics):
    """Draw pose analysis information"""
    h, w = frame.shape[:2]
    
    # Position in top-left, below logo
    start_x = 10
    start_y = 50
    
    # Background
    text_height = len(metrics) * 20 + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (start_x, start_y),
                 (start_x + 150, start_y + text_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw metrics
    y = start_y + 15
    for key, value in metrics.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (start_x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 20 