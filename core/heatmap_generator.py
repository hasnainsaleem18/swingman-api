"""
Heatmap Generator - Track and visualize bat impact points
"""

import os
import cv2
import numpy as np
from datetime import datetime

class HeatmapGenerator:
    def __init__(self, output_dir="output", resolution=(640, 480)):
        """Initialize the heatmap generator"""
        self.output_dir = output_dir
        self.resolution = resolution
        self.normalized_impacts = []  # [(x, y, efficiency), ...]
        self.sweet_spot_radius = 0.15  # 15% of bat length for sweet spot
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize heatmap parameters with consistent dimensions
        self.heatmap_resolution = (150, 150)  # Square resolution for consistency
        self.gaussian_sigma = 5.0  # Increased spread for better visibility
        
        print("✅ Heatmap Generator initialized")

    def add_impact_point(self, point, bat_center, bat_angle, efficiency_score):
        """Add a new impact point to the heatmap"""
        if not point or not bat_center:
            return False
            
        try:
            # Convert points to integers
            x, y = map(int, point)
            bat_x, bat_y = map(int, bat_center)
            
            # Normalize impact point relative to bat position and angle
            dx = x - bat_x
            dy = y - bat_y
            
            # Rotate point to align with bat angle
            angle_rad = np.radians(bat_angle)
            rot_x = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
            rot_y = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
            
            # Scale normalization to better fit the visualization
            norm_x = 2.0 * rot_x / self.resolution[0]  # Scale factor of 2.0 for better spread
            norm_y = 2.0 * rot_y / self.resolution[1]
            
            # Add to impacts list with efficiency
            self.normalized_impacts.append((norm_x, norm_y, efficiency_score))
            
            print(f"Successfully added impact point - Normalized to bat coords: ({norm_x:.2f}, {norm_y:.2f})")
            print(f"Current impact count: {len(self.normalized_impacts)}")
            return True
            
        except Exception as e:
            print(f"Error adding impact point: {e}")
            return False

    def is_sweet_spot_contact(self, impact_point, bat_center, bat_length):
        """Check if the impact point is within the sweet spot zone"""
        if not impact_point or not bat_center:
            return False
            
        # Calculate distance from impact to bat center
        dx = impact_point[0] - bat_center[0]
        dy = impact_point[1] - bat_center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Sweet spot is typically around 60-70% of the way from handle to barrel
        optimal_distance = bat_length * 0.65
        sweet_spot_range = bat_length * self.sweet_spot_radius
        
        return abs(distance - optimal_distance) <= sweet_spot_range

    def generate_heatmap_image(self, width=800, height=600):
        """Generate a simplified but reliable heatmap visualization"""
        if not self.normalized_impacts:
            return np.zeros((height, width, 3), dtype=np.uint8)
            
        try:
            # Create base image
            heatmap = np.zeros((height, width), dtype=np.float32)
            
            # Add each impact point with a simple circular pattern
            radius = 30  # Fixed radius for impact points
            for x, y, efficiency in self.normalized_impacts:
                # Convert normalized coordinates to image coordinates
                ix = int((x + 1) * width / 2)
                iy = int((y + 1) * height / 2)
                
                # Ensure coordinates are within bounds
                ix = max(radius, min(ix, width - radius))
                iy = max(radius, min(iy, height - radius))
                
                # Draw filled circle for each impact
                cv2.circle(heatmap, (ix, iy), radius, efficiency/100.0, -1)
            
            # Normalize and apply color
            heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)  # Smooth the heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            # Convert to color image
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Add dark background
            result = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.addWeighted(heatmap_color, 0.7, result, 0.3, 0, result)
            
            # Add overlay elements
            self._add_heatmap_overlay(result)
            
            print("Heatmap generated successfully")
            return result
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            # Return a blank image in case of error
            return np.zeros((height, width, 3), dtype=np.uint8)

    def _add_heatmap_overlay(self, image):
        """Add overlay elements to the heatmap"""
        # Draw bat reference outline
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        bat_length = min(image.shape) // 3
        
        # Draw bat outline
        cv2.line(image, 
                 (center_x - bat_length, center_y),
                 (center_x + bat_length, center_y),
                 (255, 255, 255), 2)
        
        # Draw sweet spot zone
        sweet_spot_start = int(center_x + bat_length * 0.5)
        sweet_spot_end = int(center_x + bat_length * 0.8)
        cv2.line(image,
                 (sweet_spot_start, center_y),
                 (sweet_spot_end, center_y),
                 (0, 255, 0), 3)
        
        # Add legend
        legend_x = 20
        legend_y = image.shape[0] - 60
        cv2.putText(image, "Impact Intensity", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw color scale
        scale_width = 100
        scale_height = 20
        for i in range(scale_width):
            color = cv2.applyColorMap(
                np.array([[int(255 * i / scale_width)]], dtype=np.uint8),
                cv2.COLORMAP_JET
            )[0][0]
            cv2.line(image,
                    (legend_x + i, legend_y + 10),
                    (legend_x + i, legend_y + 10 + scale_height),
                    tuple(map(int, color)), 1)
    
    def start_new_session(self):
        """Clear current session data"""
        self.normalized_impacts = []
    
    def save_session(self, include_heatmap=True):
        """Save current session data"""
        if not self.normalized_impacts:
            return None
            
        # Create session directory
        session_dir = os.path.join(self.output_dir, 
                                 f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Save impact data
        impact_file = os.path.join(session_dir, "impact_data.txt")
        with open(impact_file, "w") as f:
            for x, y, efficiency in self.normalized_impacts:
                f.write(f"{x:.4f},{y:.4f},{efficiency}\n")
        
        # Generate and save heatmap
        if include_heatmap:
            heatmap_img = self.generate_heatmap_image()
            cv2.imwrite(os.path.join(session_dir, "heatmap.png"), heatmap_img)
        
        return session_dir