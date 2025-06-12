"""
Bat grid overlay system
"""

import cv2
import numpy as np
import math

class BatGrid:
    """Creates and manages a virtual grid overlay for the bat"""
    
    def __init__(self):
        # Grid properties
        self.bat_length = 200  # Default bat length in pixels
        self.bat_width = 30    # Default bat width in pixels
        self.grid_cells = 7    # Number of grid cells
        
        # Colors
        self.grid_color = (255, 255, 255)
        self.sweet_spot_color = (0, 255, 0)
        self.handle_color = (0, 165, 255)
        self.alpha = 0.4  # Transparency for overlays
    
    def configure_dimensions(self, length, width):
        """Configure the bat dimensions"""
        self.bat_length = length
        self.bat_width = width
    
    def draw_grid(self, frame, center_point, angle):
        """Draw the bat grid at the specified position and angle"""
        # Create rotation matrix
        x, y = center_point
        rot_mat = cv2.getRotationMatrix2D((x, y), math.degrees(angle), 1.0)
        
        # Create bat shape (rectangle)
        bat_points = np.array([
            [x - self.bat_length/2, y - self.bat_width/2],  # Top left
            [x + self.bat_length/2, y - self.bat_width/2],  # Top right
            [x + self.bat_length/2, y + self.bat_width/2],  # Bottom right
            [x - self.bat_length/2, y + self.bat_width/2]   # Bottom left
        ], dtype=np.float32)
        
        # Apply rotation
        bat_points = np.array([
            np.dot(rot_mat, np.array([p[0], p[1], 1])) for p in bat_points
        ])
        
        # Convert to integer points for drawing
        bat_points = bat_points.astype(np.int32)
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw bat outline
        cv2.polylines(frame, [bat_points], True, self.grid_color, 2)
        
        # Draw grid lines
        cell_length = self.bat_length / self.grid_cells
        
        for i in range(1, self.grid_cells):
            # Vertical grid lines
            offset = i * cell_length
            p1 = np.array([x - self.bat_length/2 + offset, y - self.bat_width/2, 1])
            p2 = np.array([x - self.bat_length/2 + offset, y + self.bat_width/2, 1])
            
            p1_rot = np.dot(rot_mat, p1).astype(np.int32)
            p2_rot = np.dot(rot_mat, p2).astype(np.int32)
            
            cv2.line(frame, (p1_rot[0], p1_rot[1]), (p2_rot[0], p2_rot[1]), 
                     self.grid_color, 1)
        
        # Sweet spot (approximately 1/3 from the barrel end)
        sweet_spot_points = np.array([
            [x + self.bat_length/6, y - self.bat_width/2],
            [x + self.bat_length/3, y - self.bat_width/2],
            [x + self.bat_length/3, y + self.bat_width/2],
            [x + self.bat_length/6, y + self.bat_width/2]
        ], dtype=np.float32)
        
        sweet_spot_points = np.array([
            np.dot(rot_mat, np.array([p[0], p[1], 1])) for p in sweet_spot_points
        ])
        sweet_spot_points = sweet_spot_points.astype(np.int32)
        
        # Draw sweet spot
        cv2.fillPoly(overlay, [sweet_spot_points], self.sweet_spot_color)
        
        # Handle area (last 1/5 of the bat)
        handle_points = np.array([
            [x - self.bat_length/2, y - self.bat_width/2],
            [x - self.bat_length/2 + self.bat_length/5, y - self.bat_width/2],
            [x - self.bat_length/2 + self.bat_length/5, y + self.bat_width/2],
            [x - self.bat_length/2, y + self.bat_width/2]
        ], dtype=np.float32)
        
        handle_points = np.array([
            np.dot(rot_mat, np.array([p[0], p[1], 1])) for p in handle_points
        ])
        handle_points = handle_points.astype(np.int32)
        
        # Draw handle
        cv2.fillPoly(overlay, [handle_points], self.handle_color)
        
        # Apply transparency
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)
        
        return frame