"""
Swing analysis system
"""

import cv2
import numpy as np
import math

class SwingAnalyzer:
    """Analyzes bat swing mechanics and efficiency"""
    
    def __init__(self):
        # Analysis results
        self.efficiency_score = 0
        self.swing_speed = 0
        self.swing_plane = "Unknown"
        self.swing_path = "Unknown"
        
        # Timestamps for speed calculation
        self.timestamps = []
    
    def analyze_swing(self, path_points, timestamps=None, impact_point=None):
        """
        Analyze the swing path and calculate metrics
        Returns: efficiency_score (0-100)
        """
        if len(path_points) < 10:
            self.efficiency_score = 0
            return 0
        
        # Store timestamps if provided
        if timestamps is not None:
            self.timestamps = timestamps
        
        # Calculate base metrics
        self._calculate_speed(path_points)
        self._analyze_swing_plane(path_points)
        self._analyze_swing_path(path_points)
        
        # Calculate efficiency score components
        speed_score = self._calculate_speed_score()
        plane_score = self._calculate_plane_score()
        path_score = self._calculate_path_score()
        impact_score = self._calculate_impact_score(path_points, impact_point)
        
        # Calculate overall efficiency (weighted components)
        self.efficiency_score = int(
            0.25 * speed_score + 
            0.30 * plane_score + 
            0.30 * path_score + 
            0.15 * impact_score
        )
        
        # Ensure score is within range
        self.efficiency_score = max(0, min(100, self.efficiency_score))
        
        return self.efficiency_score
    
    def _calculate_speed(self, path_points):
        """Calculate swing speed from path points"""
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(path_points)):
            p1 = path_points[i-1]
            p2 = path_points[i]
            distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_distance += distance
            
        # Calculate speed (pixels per point)
        if len(path_points) > 1:
            self.swing_speed = total_distance / (len(path_points) - 1)
        else:
            self.swing_speed = 0
            
        # If we have timestamps, calculate actual speed (pixels per second)
        if hasattr(self, 'timestamps') and len(self.timestamps) >= 2:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            if time_diff > 0:
                self.swing_speed = total_distance / time_diff
                # Convert to mph or appropriate unit if desired
                # self.swing_speed_mph = self.swing_speed * conversion_factor
    
        return self.swing_speed
        
    def _analyze_swing_plane(self, path_points):
        """Analyze the swing plane (level, upward, downward)"""
        # Divide path into sections
        n = len(path_points)
        start_section = path_points[:n//3]
        mid_section = path_points[n//3:2*n//3]
        end_section = path_points[2*n//3:]
        
        # Calculate average Y positions
        start_y = sum(p[1] for p in start_section) / len(start_section)
        mid_y = sum(p[1] for p in mid_section) / len(mid_section)
        end_y = sum(p[1] for p in end_section) / len(end_section)
        
        # Determine swing plane
        y_diff = end_y - start_y
        if abs(y_diff) < 15:
            self.swing_plane = "Level"
        elif y_diff < 0:  # Remember, Y increases downward in images
            self.swing_plane = "Upward"
        else:
            self.swing_plane = "Downward"
    
    def _analyze_swing_path(self, path_points):
        """Analyze the swing path (inside-out, straight, outside-in)"""
        # For simplicity, we'll use X coordinate progression
        # In a more complete implementation, this would account for camera angle
        
        n = len(path_points)
        start_section = path_points[:n//3]
        mid_section = path_points[n//3:2*n//3]
        end_section = path_points[2*n//3:]
        
        # Calculate average X positions
        start_x = sum(p[0] for p in start_section) / len(start_section)
        mid_x = sum(p[0] for p in mid_section) / len(mid_section)
        end_x = sum(p[0] for p in end_section) / len(end_section)
        
        # Calculate mid-point deviation
        expected_mid_x = (start_x + end_x) / 2
        deviation = mid_x - expected_mid_x
        
        # Determine swing path
        if abs(deviation) < 15:
            self.swing_path = "Straight"
        elif deviation > 0:
            self.swing_path = "Inside-Out"
        else:
            self.swing_path = "Outside-In"
    
    def _calculate_speed_score(self):
        """Calculate score component for swing speed"""
        # This is a simplified scoring that rewards consistent, smooth speed
        # In a real implementation, this would compare to player norms
        
        # Base score for speed
        if self.swing_speed < 10:
            return 50  # Too slow
        elif self.swing_speed < 20:
            return 75  # Medium speed
        else:
            return 90  # Good speed
    
    def _calculate_plane_score(self):
        """Calculate score component for swing plane"""
        # Reward level swings most
        if self.swing_plane == "Level":
            return 95
        elif self.swing_plane == "Upward":
            return 75
        else:  # Downward
            return 60
    
    def _calculate_path_score(self):
        """Calculate score component for swing path"""
        # Straight or inside-out paths are preferred
        if self.swing_path == "Straight":
            return 90
        elif self.swing_path == "Inside-Out":
            return 85
        else:  # Outside-In
            return 70
    
    def _calculate_impact_score(self, path_points, impact_point):
        """Calculate score component for impact location"""
        if impact_point is None:
            return 75  # No impact detected, neutral score
        
        # Find the closest point on the path to the impact
        min_distance = float('inf')
        closest_point = None
        
        for point in path_points:
            distance = math.sqrt((point[0] - impact_point[0])**2 + 
                                 (point[1] - impact_point[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        
        # If impact is very close to path, high score
        if min_distance < 10:
            return 95
        elif min_distance < 30:
            return 80
        else:
            return 60
    
    def draw_analysis(self, frame, path_points):
        """Draw swing analysis visualization on the frame"""
        if len(path_points) < 10:
            return frame
        
        # Draw a simplified swing plane indicator
        if len(path_points) >= 3:
            start_point = path_points[0]
            end_point = path_points[-1]
            
            # Draw line from start to end
            cv2.line(frame, start_point, end_point, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Add plane annotation
            mid_x = (start_point[0] + end_point[0]) // 2
            mid_y = (start_point[1] + end_point[1]) // 2
            
            cv2.putText(frame, self.swing_plane, (mid_x, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw efficiency score
        cv2.putText(frame, f"Efficiency: {self.efficiency_score}%", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw swing details
        details = [
            f"Speed: {int(self.swing_speed)}",
            f"Plane: {self.swing_plane}",
            f"Path: {self.swing_path}"
        ]
        
        for i, detail in enumerate(details):
            cv2.putText(frame, detail, (20, 90 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame