"""
Enhanced bat visualization module
Provides realistic bat visualization during tracking
"""

import cv2
import numpy as np
import math

class BatVisualizer:
    """Provides enhanced visualization of baseball bat during tracking"""
    
    def __init__(self):
        # Default bat dimensions
        self.bat_length = 200  # pixels
        self.bat_width = 30    # pixels
        self.handle_length = 60  # pixels
        self.handle_width = 15   # pixels
        
        # Visualization settings
        self.bat_color = (50, 50, 50)      # Dark gray for bat barrel
        self.handle_color = (139, 69, 19)  # Brown for handle
        self.outline_color = (255, 255, 255)  # White outline
        self.sweet_spot_color = (0, 255, 0)  # Green for sweet spot
        self.grip_color = (80, 30, 20)       # Dark brown for handle grip
        
        # Effects
        self.use_3d_effect = True
        self.use_gradient = True
        self.use_texture = True
        
        # Texture patterns
        self.wood_texture = None
        self.grip_texture = None
        self.generate_textures()
    
    def generate_textures(self):
        """Generate wood grain and grip textures"""
        # Wood grain texture (procedural)
        wood_size = (256, 64)
        self.wood_texture = np.zeros((*wood_size, 3), dtype=np.uint8)
        
        # Base wood color
        self.wood_texture[:] = (210, 180, 140)  # Tan color
        
        # Add grain lines
        for i in range(20):  # Add some grain lines
            thickness = np.random.randint(1, 3)
            darkness = np.random.randint(30, 80)
            x = np.random.randint(0, wood_size[0])
            cv2.line(self.wood_texture, 
                    (x, 0), 
                    (x + np.random.randint(-30, 30), wood_size[1]), 
                    (210-darkness, 180-darkness, 140-darkness), 
                    thickness)
        
        # Add some knots
        for _ in range(2):
            x = np.random.randint(0, wood_size[0])
            y = np.random.randint(0, wood_size[1])
            radius = np.random.randint(3, 8)
            color = (120, 100, 80)  # Darker for knots
            cv2.circle(self.wood_texture, (x, y), radius, color, -1)
            # Add ring around knot
            cv2.circle(self.wood_texture, (x, y), radius+1, (100, 80, 60), 1)
        
        # Apply slight blur for realism
        self.wood_texture = cv2.GaussianBlur(self.wood_texture, (3, 3), 0)
        
        # Grip texture
        grip_size = (64, 64)
        self.grip_texture = np.zeros((*grip_size, 3), dtype=np.uint8)
        
        # Base grip color
        self.grip_texture[:] = (60, 30, 20)  # Dark brown
        
        # Add grip pattern (diagonal lines)
        for i in range(0, grip_size[0]+grip_size[1], 8):
            thickness = 1
            color_var = np.random.randint(-10, 10)
            color = (70+color_var, 40+color_var, 30+color_var)
            cv2.line(self.grip_texture, 
                    (0, i), 
                    (i, 0), 
                    color, 
                    thickness)
    
    def draw_realistic_bat(self, frame, center_point, angle, with_sweet_spot=True):
        """
        Draw a realistic baseball bat visualization
        
        Parameters:
            frame: The frame to draw on
            center_point: (x, y) coordinates of bat center
            angle: Angle of bat in radians
            with_sweet_spot: Whether to highlight the sweet spot
        """
        x, y = center_point
        
        # Create rotation matrix
        rot_mat = cv2.getRotationMatrix2D((0, 0), math.degrees(angle), 1.0)
        
        # Create bat shape
        # We'll create a more complex bat shape with barrel, taper, and handle
        
        # Create a mask for the bat shape
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Barrel points (main part of the bat)
        barrel_length = self.bat_length - self.handle_length
        barrel_points = np.array([
            [-barrel_length/2, -self.bat_width/2],  # Top left
            [barrel_length/2, -self.bat_width/2],   # Top right
            [barrel_length/2, self.bat_width/2],    # Bottom right
            [-barrel_length/2, self.bat_width/2]    # Bottom left
        ], dtype=np.float32)
        
        # Rotate barrel points
        barrel_points_rot = np.array([
            np.dot(rot_mat, np.append(p, 1))[:2] for p in barrel_points
        ])
        
        # Translate to center position
        barrel_points_pos = barrel_points_rot + [x, y]
        
        # Draw barrel
        barrel_points_draw = barrel_points_pos.astype(np.int32)
        
        # Handle points (thinner part)
        handle_points = np.array([
            [-barrel_length/2, -self.handle_width/2],  # Join to barrel
            [-barrel_length/2 - self.handle_length, -self.handle_width/2],  # End of handle
            [-barrel_length/2 - self.handle_length, self.handle_width/2],
            [-barrel_length/2, self.handle_width/2]
        ], dtype=np.float32)
        
        # Rotate handle points
        handle_points_rot = np.array([
            np.dot(rot_mat, np.append(p, 1))[:2] for p in handle_points
        ])
        
        # Translate to center position
        handle_points_pos = handle_points_rot + [x, y]
        
        # Draw handle
        handle_points_draw = handle_points_pos.astype(np.int32)
        
        # Create a copy for drawing
        overlay = frame.copy()
        
        # Draw bat parts to mask
        cv2.fillPoly(mask, [barrel_points_draw], 255)
        cv2.fillPoly(mask, [handle_points_draw], 255)
        
        # Draw the bat with texture
        if self.use_texture:
            # Create a clean canvas for the bat texture
            bat_canvas = np.zeros_like(frame)
            
            # Fill barrel with wood texture
            barrel_mask = np.zeros_like(mask)
            cv2.fillPoly(barrel_mask, [barrel_points_draw], 255)
            
            # Apply wood texture to barrel
            # We need to rotate and scale the texture to match the bat orientation
            h, w = frame.shape[:2]
            texture_rotated = cv2.warpAffine(
                self.wood_texture, 
                cv2.getRotationMatrix2D((self.wood_texture.shape[1]//2, self.wood_texture.shape[0]//2), 
                                      math.degrees(angle), 1.0),
                (w, h)
            )
            
            # Position the texture over the barrel
            texture_mask = cv2.bitwise_and(texture_rotated, texture_rotated, mask=barrel_mask)
            bat_canvas = cv2.add(bat_canvas, texture_mask)
            
            # Apply grip texture to handle
            handle_mask = np.zeros_like(mask)
            cv2.fillPoly(handle_mask, [handle_points_draw], 255)
            
            # Create a handle-sized grip texture
            grip_rotated = cv2.warpAffine(
                self.grip_texture,
                cv2.getRotationMatrix2D((self.grip_texture.shape[1]//2, self.grip_texture.shape[0]//2),
                                      math.degrees(angle), 1.0),
                (w, h)
            )
            
            # Position grip texture over handle
            grip_mask = cv2.bitwise_and(grip_rotated, grip_rotated, mask=handle_mask)
            bat_canvas = cv2.add(bat_canvas, grip_mask)
            
            # Apply bat canvas to frame
            frame_without_bat = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame_without_bat, bat_canvas)
        else:
            # Simple colored version without texture
            color_barrel = np.zeros_like(frame)
            color_barrel[:] = (*self.bat_color, 0)  # BGR format
            barrel_colored = cv2.bitwise_and(color_barrel, color_barrel, mask=cv2.fillPoly(np.zeros_like(mask), [barrel_points_draw], 255))
            
            color_handle = np.zeros_like(frame)
            color_handle[:] = (*self.handle_color, 0)  # BGR format
            handle_colored = cv2.bitwise_and(color_handle, color_handle, mask=cv2.fillPoly(np.zeros_like(mask), [handle_points_draw], 255))
            
            # Combine colored parts
            bat_colored = cv2.add(barrel_colored, handle_colored)
            
            # Apply to frame
            frame_without_bat = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
            frame = cv2.add(frame_without_bat, bat_colored)
        
        # Add 3D effect with shading
        if self.use_3d_effect:
            # Add a highlight along the top edge
            highlight_points = np.array([
                barrel_points_draw[0],  # Top left of barrel
                barrel_points_draw[1],  # Top right of barrel
                handle_points_draw[1],  # End of handle (top)
                handle_points_draw[0]   # Join to barrel (top)
            ])
            
            cv2.polylines(frame, [highlight_points], False, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add shadow along the bottom edge
            shadow_points = np.array([
                barrel_points_draw[3],  # Bottom left of barrel
                barrel_points_draw[2],  # Bottom right of barrel
                handle_points_draw[2],  # End of handle (bottom)
                handle_points_draw[3]   # Join to barrel (bottom)
            ])
            
            cv2.polylines(frame, [shadow_points], False, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Add subtle edge highlight using a thin white outline
            cv2.polylines(frame, [barrel_points_draw], True, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.polylines(frame, [handle_points_draw], True, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add sweet spot if requested
        if with_sweet_spot:
            # The sweet spot is approximately 5-7 inches from the barrel end
            # For our visualization, we'll put it around 25-30% from the barrel end
            sweet_spot_center = barrel_points[1] * 0.7 + barrel_points[0] * 0.3
            sweet_spot_radius = self.bat_width * 0.4
            
            # Rotate and position sweet spot
            sweet_spot_pos = np.dot(rot_mat, np.append(sweet_spot_center, 1))[:2] + [x, y]
            sweet_spot_pos = sweet_spot_pos.astype(np.int32)
            
            # Draw sweet spot highlight
            overlay = frame.copy()
            cv2.circle(overlay, tuple(sweet_spot_pos), int(sweet_spot_radius), self.sweet_spot_color, -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Add sweet spot label
            label_pos = sweet_spot_pos + np.array([0, -int(self.bat_width * 0.7)])
            cv2.putText(frame, "Sweet Spot", tuple(label_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame
    
    def draw_bat_with_grid(self, frame, center_point, angle, with_sweet_spot=True):
        """
        Draw realistic bat with grid overlay for impact tracking
        """
        # First draw the realistic bat
        frame = self.draw_realistic_bat(frame, center_point, angle, with_sweet_spot)
        
        # Then add grid overlay
        x, y = center_point
        rot_mat = cv2.getRotationMatrix2D((0, 0), math.degrees(angle), 1.0)
        
        # Only draw grid on barrel part
        barrel_length = self.bat_length - self.handle_length
        
        # Draw vertical grid lines
        grid_spacing = barrel_length / 7  # 7 sections on the barrel
        for i in range(1, 7):
            # Position relative to bat center
            start_x = -barrel_length/2 + i * grid_spacing
            start_y = -self.bat_width/2
            end_y = self.bat_width/2
            
            # Rotate and translate
            start_point = np.dot(rot_mat, [start_x, start_y, 1])[:2] + [x, y]
            end_point = np.dot(rot_mat, [start_x, end_y, 1])[:2] + [x, y]
            
            # Draw line
            cv2.line(frame, 
                    tuple(start_point.astype(np.int32)), 
                    tuple(end_point.astype(np.int32)), 
                    (255, 255, 255, 128), 1, cv2.LINE_AA)
        
        # Draw horizontal grid line (center line)
        start_x = -barrel_length/2
        end_x = barrel_length/2
        center_y = 0
        
        start_point = np.dot(rot_mat, [start_x, center_y, 1])[:2] + [x, y]
        end_point = np.dot(rot_mat, [end_x, center_y, 1])[:2] + [x, y]
        
        cv2.line(frame, 
                tuple(start_point.astype(np.int32)), 
                tuple(end_point.astype(np.int32)), 
                (255, 255, 255, 128), 1, cv2.LINE_AA)
        
        return frame

    def draw_bat_with_impact(self, frame, center_point, angle, impact_point, efficiency_score):
        """
        Draw bat with impact point visualization
        
        Parameters:
            frame: The frame to draw on
            center_point: (x, y) coordinates of bat center
            angle: Angle of bat in radians
            impact_point: (x, y) coordinates of impact point on screen
            efficiency_score: Efficiency score (0-100)
        """
        # Draw the bat first
        frame = self.draw_bat_with_grid(frame, center_point, angle, True)
        
        if impact_point is None:
            return frame
            
        # Draw impact marker
        marker_size = 15
        marker_thickness = 2
        
        # Determine color based on efficiency score
        if efficiency_score >= 80:
            color = (0, 255, 0)  # Green for great hits
        elif efficiency_score >= 60:
            color = (0, 255, 255)  # Yellow for good hits
        elif efficiency_score >= 40:
            color = (0, 165, 255)  # Orange for okay hits
        else:
            color = (0, 0, 255)  # Red for poor hits
        
        # Draw outer circle
        cv2.circle(frame, impact_point, marker_size, color, marker_thickness, cv2.LINE_AA)
        
        # Draw inner circle
        cv2.circle(frame, impact_point, marker_size//2, color, -1, cv2.LINE_AA)
        
        # Add pulse effect
        pulse_size = marker_size + 5
        pulse_alpha = 0.5
        overlay = frame.copy()
        cv2.circle(overlay, impact_point, pulse_size, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, pulse_alpha, frame, 1 - pulse_alpha, 0, frame)
        
        # Add efficiency score text
        text_pos = (impact_point[0] - 20, impact_point[1] - marker_size - 10)
        cv2.putText(frame, f"{efficiency_score}%", text_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        return frame
    
    def draw_follow_through_guide(self, frame, path_points, target_angle=None):
        """
        Draw a follow-through guide to help with swing training
        
        Parameters:
            frame: The frame to draw on
            path_points: List of path points from tracker
            target_angle: Target angle for ideal follow-through (None to auto-detect)
        """
        if len(path_points) < 10:
            return frame
        
        # Get the last segment of the path (follow-through)
        follow_points = path_points[-min(10, len(path_points)):]
        
        # Calculate current follow-through angle
        if len(follow_points) >= 2:
            end_p = follow_points[-1]
            start_p = follow_points[0]
            current_angle = math.atan2(end_p[1] - start_p[1], end_p[0] - start_p[0])
            
            # If no target angle provided, use a slight upward angle as ideal
            if target_angle is None:
                target_angle = -0.1  # Slightly upward
            
            # Determine how close current angle is to target
            angle_diff = abs(current_angle - target_angle)
            
            # Normalize to 0-1 where 0 is perfect
            angle_quality = max(0, 1 - angle_diff / math.pi)
            
            # Draw follow-through indicator
            last_point = path_points[-1]
            
            # Determine color based on quality
            if angle_quality > 0.8:
                color = (0, 255, 0)  # Green for good follow-through
                message = "Great Follow-Through!"
            elif angle_quality > 0.5:
                color = (0, 165, 255)  # Orange for okay follow-through
                message = "Good Follow-Through"
            else:
                color = (0, 0, 255)  # Red for poor follow-through
                message = "Improve Follow-Through"
            
            # Draw message
            cv2.putText(frame, message, 
                       (last_point[0] - 70, last_point[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            # Draw ideal follow-through line
            ideal_length = 100
            ideal_end = (
                int(last_point[0] + ideal_length * math.cos(target_angle)),
                int(last_point[1] + ideal_length * math.sin(target_angle))
            )
            
            # Draw ideal line as dashed
            overlay = frame.copy()
            cv2.line(overlay, last_point, ideal_end, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Draw small arrows along ideal line
            arrow_len = 10
            for i in range(1, 4):
                pos = (
                    int(last_point[0] + (i * ideal_length / 4) * math.cos(target_angle)),
                    int(last_point[1] + (i * ideal_length / 4) * math.sin(target_angle))
                )
                
                # Draw small arrow
                cv2.arrowedLine(frame, 
                               (int(pos[0] - arrow_len * math.cos(target_angle)), 
                                int(pos[1] - arrow_len * math.sin(target_angle))),
                               pos, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
        
        return frame