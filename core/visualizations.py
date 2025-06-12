"""
Visual effects and animations for UI
"""

import cv2
import numpy as np

class SwingPathVisualizer:
    """Creates enhanced visuals for swing paths"""
    
    def __init__(self, thickness=2, color=(0, 255, 0)):
        self.thickness = thickness
        self.color = color
        self.glow_radius = 10
        self.trail_length = 20  # Number of points to show in motion trail
    
    def draw_path(self, frame, points, glow=True):
        """Draw a path with glow effect"""
        if len(points) < 2:
            return frame
        
        # Create a copy for overlay if using glow
        if glow:
            glow_layer = np.zeros_like(frame)
        
        # Draw the main path
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            cv2.line(frame, p1, p2, self.color, self.thickness, cv2.LINE_AA)
        
        # Add glow effect
        if glow:
            # Draw wider lines for the glow
            for i in range(1, len(points)):
                p1 = points[i-1]
                p2 = points[i]
                cv2.line(glow_layer, p1, p2, self.color, 
                        self.thickness + self.glow_radius, cv2.LINE_AA)
            
            # Blur the glow layer
            glow_layer = cv2.GaussianBlur(glow_layer, 
                                         (self.glow_radius*2+1, self.glow_radius*2+1), 
                                         0)
            
            # Add glow to original frame
            frame = cv2.addWeighted(frame, 1.0, glow_layer, 0.5, 0)
        
        return frame
    
    def draw_motion_trail(self, frame, points):
        """Draw a motion trail with fading effect"""
        if len(points) < 2:
            return frame
        
        # Get the trail points (last N points)
        trail_points = points[-min(len(points), self.trail_length):]
        
        # Draw with varying opacity
        for i in range(1, len(trail_points)):
            p1 = trail_points[i-1]
            p2 = trail_points[i]
            
            # Calculate opacity based on position in trail
            opacity = i / len(trail_points)
            
            # Adjust color based on opacity
            color = tuple([int(c * opacity) for c in self.color])
            
            # Draw line segment
            cv2.line(frame, p1, p2, color, self.thickness, cv2.LINE_AA)
        
        return frame


class ImpactVisualizer:
    """Creates enhanced impact visualizations"""
    
    def __init__(self, color=(255, 0, 0)):
        self.color = color
        self.animation_frames = 15
        self.current_frame = 0
        self.is_animating = False
        self.impact_point = None
    
    def start_animation(self, point):
        """Start impact animation at the given point"""
        self.impact_point = point
        self.current_frame = 0
        self.is_animating = True
    
    def draw(self, frame):
        """Draw impact visualization"""
        if not self.is_animating or self.impact_point is None:
            return frame
        
        # Calculate animation parameters
        max_radius = 40
        radius = int((self.current_frame / self.animation_frames) * max_radius)
        opacity = 1.0 - (self.current_frame / self.animation_frames)
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw circle
        cv2.circle(overlay, self.impact_point, radius, self.color, -1)
        
        # Draw outer ring
        cv2.circle(frame, self.impact_point, radius, (255, 255, 255), 2)
        
        # Apply transparency
        cv2.addWeighted(overlay, opacity, frame, 1-opacity, 0, frame)
        
        # Update animation state
        self.current_frame += 1
        if self.current_frame >= self.animation_frames:
            self.is_animating = False
        
        return frame


class BatGridVisualizer:
    """Enhanced bat grid visualization"""
    
    def __init__(self):
        self.bat_length = 200
        self.bat_width = 30
        self.grid_color = (255, 255, 255)
        self.sweet_spot_color = (0, 255, 0)
        self.handle_color = (0, 165, 255)
    
    def draw_bat_grid_3d(self, frame, center_point, angle, perspective=0.2):
        """Draw bat grid with 3D perspective effect"""
        x, y = center_point
        
        # Create rotation matrix
        rot_mat = cv2.getRotationMatrix2D((0, 0), np.degrees(angle), 1.0)
        
        # Create 3D bat shape (approximated as rectangular prism)
        thickness = 10  # Z-depth of bat
        
        # Define 8 corners of rectangular prism in 3D
        # Front face
        front_tl = np.array([-self.bat_length/2, -self.bat_width/2, 0])
        front_tr = np.array([self.bat_length/2, -self.bat_width/2, 0])
        front_br = np.array([self.bat_length/2, self.bat_width/2, 0])
        front_bl = np.array([-self.bat_length/2, self.bat_width/2, 0])
        
        # Back face (with perspective)
        back_tl = np.array([-self.bat_length/2, -self.bat_width/2, -thickness])
        back_tr = np.array([self.bat_length/2, -self.bat_width/2, -thickness])
        back_br = np.array([self.bat_length/2, self.bat_width/2, -thickness])
        back_bl = np.array([-self.bat_length/2, self.bat_width/2, -thickness])
        
        # Apply perspective (simple approximation)
        def apply_perspective(point):
            z_factor = 1 + (point[2] * perspective / 100)
            return [point[0] * z_factor, point[1] * z_factor]
        
        # Apply perspective to all points
        front_tl_2d = apply_perspective(front_tl)
        front_tr_2d = apply_perspective(front_tr)
        front_br_2d = apply_perspective(front_br)
        front_bl_2d = apply_perspective(front_bl)
        
        back_tl_2d = apply_perspective(back_tl)
        back_tr_2d = apply_perspective(back_tr)
        back_br_2d = apply_perspective(back_br)
        back_bl_2d = apply_perspective(back_bl)
        
        # Apply rotation and translation to all points
        def transform_point(point):
            rotated = np.dot(rot_mat, [point[0], point[1], 1])
            return (int(rotated[0] + x), int(rotated[1] + y))
        
        # Transform all points
        front_tl_t = transform_point(front_tl_2d)
        front_tr_t = transform_point(front_tr_2d)
        front_br_t = transform_point(front_br_2d)
        front_bl_t = transform_point(front_bl_2d)
        
        back_tl_t = transform_point(back_tl_2d)
        back_tr_t = transform_point(back_tr_2d)
        back_br_t = transform_point(back_br_2d)
        back_bl_t = transform_point(back_bl_2d)
        
        # Draw front face
        front_face = np.array([front_tl_t, front_tr_t, front_br_t, front_bl_t], dtype=np.int32)
        cv2.polylines(frame, [front_face], True, self.grid_color, 2)
        
        # Draw back face
        back_face = np.array([back_tl_t, back_tr_t, back_br_t, back_bl_t], dtype=np.int32)
        cv2.polylines(frame, [back_face], True, self.grid_color, 1)
        
        # Draw connecting lines
        cv2.line(frame, front_tl_t, back_tl_t, self.grid_color, 1)
        cv2.line(frame, front_tr_t, back_tr_t, self.grid_color, 1)
        cv2.line(frame, front_br_t, back_br_t, self.grid_color, 1)
        cv2.line(frame, front_bl_t, back_bl_t, self.grid_color, 1)
        
        # Add sweet spot
        sweet_spot_tl = transform_point([self.bat_length/6, -self.bat_width/2])
        sweet_spot_tr = transform_point([self.bat_length/3, -self.bat_width/2])
        sweet_spot_br = transform_point([self.bat_length/3, self.bat_width/2])
        sweet_spot_bl = transform_point([self.bat_length/6, self.bat_width/2])
        
        sweet_spot = np.array([sweet_spot_tl, sweet_spot_tr, sweet_spot_br, sweet_spot_bl], dtype=np.int32)
        
        # Create overlay for transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [sweet_spot], self.sweet_spot_color)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        return frame