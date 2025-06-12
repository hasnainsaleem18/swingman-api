"""
MediaPipe Pose integration for body posture analysis
"""
import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque

class PoseAnalyzer:
    def __init__(self):
        """Initialize MediaPipe Pose detector"""
        print("Initializing MediaPipe Pose...")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure pose detection
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize pose history for stability tracking
        self.pose_history = deque(maxlen=30)
        
        # Key body landmarks for baseball swing analysis
        self.key_landmarks = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        print("MediaPipe Pose initialized successfully!")
    
    def analyze_pose(self, frame):
        """Analyze pose in frame and return pose data"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {
                'is_detected': False,
                'landmarks': [],
                'stability_score': 0
            }
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            landmarks.append((x, y))
        
        # Create pose data
        pose_data = {
            'is_detected': True,
            'landmarks': landmarks,
            'stability_score': 0
        }
        
        # Add to history and calculate stability
        self.pose_history.append(pose_data)
        pose_data['stability_score'] = self._calculate_stability_score(landmarks)
        
        return pose_data
    
    def draw_pose(self, frame, pose_data):
        """Draw pose landmarks and connections on frame"""
        if not pose_data or not pose_data['is_detected']:
            return frame
        
        # Draw landmarks
        landmarks = pose_data['landmarks']
        for landmark in landmarks:
            x, y = landmark
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green dots
            cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)  # White outline
        
        # Draw connections between landmarks
        connections = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11),  # Shoulders to hips
            # Arms
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            # Legs
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)  # Yellow lines
        
        return frame
    
    def _extract_normalized_landmarks(self, landmarks):
        """Extract normalized landmark coordinates"""
        normalized = {}
        
        for name, idx in self.key_landmarks.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                normalized[name] = {
                    'x': lm[0] / frame.shape[1],
                    'y': lm[1] / frame.shape[0],
                    'z': 0,  # Assuming z is not available in the landmarks
                    'visibility': 1.0  # Assuming visibility is always 1.0
                }
        
        return normalized
    
    def _calculate_swing_angles(self, landmarks):
        """Calculate important angles for baseball swing analysis"""
        angles = {}
        
        try:
            lm = landmarks
            
            # 1. Shoulder angle (shoulder tilt/alignment)
            left_shoulder = [lm[self.key_landmarks['left_shoulder']][0], 
                           lm[self.key_landmarks['left_shoulder']][1]]
            right_shoulder = [lm[self.key_landmarks['right_shoulder']][0], 
                            lm[self.key_landmarks['right_shoulder']][1]]
            
            shoulder_angle = np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ) * 180 / np.pi
            
            angles['shoulder_tilt'] = abs(shoulder_angle)
            
            # 2. Hip angle (hip rotation indicator)
            left_hip = [lm[self.key_landmarks['left_hip']][0], 
                       lm[self.key_landmarks['left_hip']][1]]
            right_hip = [lm[self.key_landmarks['right_hip']][0], 
                        lm[self.key_landmarks['right_hip']][1]]
            
            hip_angle = np.arctan2(
                right_hip[1] - left_hip[1],
                right_hip[0] - left_hip[0]
            ) * 180 / np.pi
            
            angles['hip_rotation'] = abs(hip_angle)
            
            # 3. Torso rotation (difference between shoulders and hips)
            angles['torso_rotation'] = abs(angles['shoulder_tilt'] - angles['hip_rotation'])
            
            # 4. Batting stance width (distance between feet)
            left_ankle = [lm[self.key_landmarks['left_ankle']][0], 
                         lm[self.key_landmarks['left_ankle']][1]]
            right_ankle = [lm[self.key_landmarks['right_ankle']][0], 
                          lm[self.key_landmarks['right_ankle']][1]]
            
            stance_width = np.sqrt(
                (right_ankle[0] - left_ankle[0])**2 + 
                (right_ankle[1] - left_ankle[1])**2
            )
            
            angles['stance_width'] = stance_width
            
        except Exception as e:
            print(f"Error calculating angles: {e}")
            # Return default angles if calculation fails
            angles = {
                'shoulder_tilt': 0,
                'hip_rotation': 0,
                'torso_rotation': 0,
                'stance_width': 0
            }
        
        return angles
    
    def _calculate_stability_score(self, landmarks):
        """Calculate pose stability score"""
        if len(self.pose_history) < 2:
            return 50  # Default to medium stability with not enough history
        
        # Key points for stability (shoulders, hips)
        key_indices = [11, 12, 23, 24]  # MediaPipe pose indices
        
        # Calculate movement of key points
        total_movement = 0
        num_valid_points = 0
        
        for idx in key_indices:
            if idx < len(landmarks):
                prev_poses = [pose['landmarks'][idx] for pose in list(self.pose_history)[-3:] 
                            if pose['landmarks'] and idx < len(pose['landmarks'])]
                
                if len(prev_poses) > 1:
                    movements = []
                    for i in range(1, len(prev_poses)):
                        dx = prev_poses[i][0] - prev_poses[i-1][0]
                        dy = prev_poses[i][1] - prev_poses[i-1][1]
                        movement = math.sqrt(dx*dx + dy*dy)
                        movements.append(movement)
                    
                    if movements:
                        avg_movement = sum(movements) / len(movements)
                        total_movement += avg_movement
                        num_valid_points += 1
        
        # Convert movement to stability score (inverse relationship)
        if num_valid_points == 0:
            return 50  # Default to medium stability if no valid points
            
        avg_movement = total_movement / num_valid_points
        max_movement = 50  # Lower threshold for maximum movement
        stability = max(0, min(100, 100 - (avg_movement / max_movement * 100)))
        
        # Smooth the stability score
        return int((stability + 50) / 2)  # Blend with medium stability to avoid extremes
    
    def _determine_swing_phase(self, landmarks):
        """Determine basic swing phase based on arm positions"""
        try:
            lm = landmarks
            
            # Get wrist and shoulder positions
            left_wrist_y = lm[self.key_landmarks['left_wrist']][1]
            right_wrist_y = lm[self.key_landmarks['right_wrist']][1]
            left_shoulder_y = lm[self.key_landmarks['left_shoulder']][1]
            right_shoulder_y = lm[self.key_landmarks['right_shoulder']][1]
            
            avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
            
            # Simple phase detection based on hand height relative to shoulders
            if avg_wrist_y < avg_shoulder_y - 0.1:
                return 'load_position'  # Hands high, loading
            elif avg_wrist_y > avg_shoulder_y + 0.1:
                return 'follow_through'  # Hands low, follow through
            else:
                return 'contact_zone'  # Hands near shoulder level, contact zone
                
        except:
            return 'unknown'
    
    def _draw_swing_specific_info(self, frame, pose_data):
        """Draw swing-specific information on frame"""
        if not pose_data['is_detected']:
            return
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Draw pose metrics in top-left corner
        y_offset = 50
        line_height = 25
        
        # Background for text
        cv2.rectangle(frame, (10, 30), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 30), (350, 180), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "POSE ANALYSIS", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += line_height
        
        # Stability score
        stability = pose_data['stability_score']
        color = (0, 255, 0) if stability > 70 else (0, 255, 255) if stability > 50 else (0, 0, 255)
        cv2.putText(frame, f"Stability: {stability}%", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += line_height
        
        # Angles
        angles = pose_data['angles']
        if angles:
            cv2.putText(frame, f"Shoulder Tilt: {angles.get('shoulder_tilt', 0):.1f}°", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Hip Rotation: {angles.get('hip_rotation', 0):.1f}°", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            cv2.putText(frame, f"Torso Twist: {angles.get('torso_rotation', 0):.1f}°", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Swing phase
        phase = pose_data['swing_phase']
        phase_color = {
            'load_position': (0, 255, 255),  # Yellow
            'contact_zone': (0, 255, 0),     # Green
            'follow_through': (255, 0, 0),   # Blue
            'unknown': (128, 128, 128)       # Gray
        }.get(phase, (255, 255, 255))
        
        cv2.putText(frame, f"Phase: {phase.replace('_', ' ').title()}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 2)

# Test function for pose_analyzer.py
def test_pose_analyzer():
    """Test the PoseAnalyzer with webcam"""
    print("Testing PoseAnalyzer...")
    
    # Initialize analyzer
    analyzer = PoseAnalyzer()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Pose Analyzer test running. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Analyze pose
        pose_data = analyzer.analyze_pose(frame)
        
        # Draw results
        frame = analyzer.draw_pose(frame, pose_data)
        
        # Show frame
        cv2.imshow("Pose Analyzer Test", frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Pose Analyzer test completed!")

if __name__ == "__main__":
    test_pose_analyzer()