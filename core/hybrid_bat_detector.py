import cv2
import numpy as np
import math

class AdvancedBatDetector:
    """Advanced CV-based bat detector using line detection and morphology"""
    
    def __init__(self):
        print("Initializing Advanced CV Bat Detector...")
        
        # Bat physical properties (in pixels)
        self.min_bat_length = 120
        self.max_bat_length = 400
        self.min_bat_width = 8
        self.max_bat_width = 50
        
        # Detection parameters
        self.line_threshold = 80      # Hough line detection threshold
        self.min_line_length = 100    # Minimum line length for bat
        self.max_line_gap = 20        # Maximum gap in line
        
        print("✅ Advanced CV Bat Detector ready!")
    
    def detect_bat_advanced_cv(self, frame):
        """Detect bat using advanced computer vision techniques"""
        detections = []
        
        # Method 1: Hough Line Transform (most effective for bats)
        line_detections = self._detect_with_hough_lines(frame)
        detections.extend(line_detections)
        
        # Method 2: Morphological operations for elongated objects
        morph_detections = self._detect_with_morphology(frame)
        detections.extend(morph_detections)
        
        # Method 3: Edge density analysis along lines
        edge_detections = self._detect_with_edge_lines(frame)
        detections.extend(edge_detections)
        
        # Filter and combine detections
        filtered_detections = self._filter_and_rank_detections(detections)
        
        return filtered_detections
    
    def _detect_with_hough_lines(self, frame):
        """Detect bats using Hough Line Transform - very effective for straight objects"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection with optimal parameters for bat detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Hough Line Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,                    # Distance resolution
                theta=np.pi/180,          # Angle resolution
                threshold=self.line_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    # Filter for bat-like lines
                    if self.min_bat_length <= length <= self.max_bat_length:
                        # Create bounding box around line
                        margin = 20
                        x_min = min(x1, x2) - margin
                        y_min = min(y1, y2) - margin
                        x_max = max(x1, x2) + margin
                        y_max = max(y1, y2) + margin
                        
                        # Ensure bounds are within frame
                        h, w = frame.shape[:2]
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(w, x_max)
                        y_max = min(h, y_max)
                        
                        # Calculate confidence based on line quality
                        confidence = min(0.9, (length / self.max_bat_length) * 0.7 + 0.2)
                        
                        detections.append({
                            'bbox': (x_min, y_min, x_max, y_max),
                            'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                            'confidence': confidence,
                            'source': 'hough_line',
                            'length': length,
                            'angle': angle,
                            'line_points': (x1, y1, x2, y2)
                        })
        
        except Exception as e:
            print(f"Hough line detection error: {e}")
        
        return detections
    
    def _detect_with_morphology(self, frame):
        """Detect elongated objects using morphological operations"""
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Create elongated kernel for morphological operations
            kernel_length = 40
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 3))
            
            # Apply morphological operations to enhance elongated objects
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Find contours of elongated objects
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Check if it matches bat dimensions
                length = max(w, h)
                width = min(w, h)
                aspect_ratio = length / width if width > 0 else 0
                
                if (self.min_bat_length <= length <= self.max_bat_length and
                    self.min_bat_width <= width <= self.max_bat_width and
                    aspect_ratio > 3 and area > 800):
                    
                    confidence = min(0.8, (aspect_ratio - 3) * 0.1 + (area / 5000) * 0.3 + 0.4)
                    
                    detections.append({
                        'bbox': (x, y, x + w, y + h),
                        'center': (x + w // 2, y + h // 2),
                        'confidence': confidence,
                        'source': 'morphology',
                        'aspect_ratio': aspect_ratio,
                        'area': area
                    })
        
        except Exception as e:
            print(f"Morphology detection error: {e}")
        
        return detections
    
    def _detect_with_edge_lines(self, frame):
        """Detect bats by analyzing edge density along potential bat lines"""
        detections = []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            
            h, w = frame.shape[:2]
            
            # Test different angles that a bat might be at
            test_angles = range(-60, 61, 15)  # -60 to 60 degrees in 15-degree steps
            
            for angle_deg in test_angles:
                angle_rad = np.radians(angle_deg)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                
                # Slide along potential bat positions
                for y_start in range(20, h - 20, 30):
                    for x_start in range(20, w - 20, 30):
                        
                        # Calculate line of potential bat
                        length = 150  # Test length
                        x_end = int(x_start + length * cos_a)
                        y_end = int(y_start + length * sin_a)
                        
                        # Check if end point is within frame
                        if 0 <= x_end < w and 0 <= y_end < h:
                            
                            # Sample points along this line
                            line_points = self._get_line_points(x_start, y_start, x_end, y_end)
                            
                            # Calculate edge density along this line
                            edge_density = 0
                            valid_points = 0
                            
                            for px, py in line_points:
                                if 0 <= px < w and 0 <= py < h:
                                    edge_density += edges[py, px]
                                    valid_points += 1
                            
                            if valid_points > 0:
                                avg_edge_density = edge_density / (valid_points * 255)
                                
                                # If this line has good edge density, it might be a bat
                                if avg_edge_density > 0.15:  # Threshold for bat-like edge density
                                    
                                    # Create bounding box
                                    margin = 25
                                    x_min = min(x_start, x_end) - margin
                                    y_min = min(y_start, y_end) - margin
                                    x_max = max(x_start, x_end) + margin
                                    y_max = max(y_start, y_end) + margin
                                    
                                    # Clamp to frame
                                    x_min = max(0, x_min)
                                    y_min = max(0, y_min)
                                    x_max = min(w, x_max)
                                    y_max = min(h, y_max)
                                    
                                    confidence = min(0.7, avg_edge_density * 2)
                                    
                                    detections.append({
                                        'bbox': (x_min, y_min, x_max, y_max),
                                        'center': ((x_min + x_max) // 2, (y_min + y_max) // 2),
                                        'confidence': confidence,
                                        'source': 'edge_line',
                                        'angle': angle_deg,
                                        'edge_density': avg_edge_density
                                    })
        
        except Exception as e:
            print(f"Edge line detection error: {e}")
        
        return detections
    
    def _get_line_points(self, x1, y1, x2, y2):
        """Get points along a line using Bresenham's algorithm"""
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def _filter_and_rank_detections(self, detections):
        """Filter overlapping detections and rank by quality"""
        if not detections:
            return []
        
        # Remove low-confidence detections
        filtered = [d for d in detections if d['confidence'] > 0.3]
        
        # Sort by confidence
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove overlapping detections
        final_detections = []
        for detection in filtered:
            is_duplicate = False
            for existing in final_detections:
                if self._calculate_overlap(detection['bbox'], existing['bbox']) > 0.4:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
            
            # Limit to top 3 detections
            if len(final_detections) >= 3:
                break
        
        return final_detections
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate IoU overlap between bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def draw_detections(self, frame, detections):
        """Draw detection results with line overlays"""
        if not detections:
            cv2.putText(frame, "No bat detected - try different angle", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return frame
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            source = detection['source']
            
            # Color based on source and confidence
            if confidence > 0.7:
                color = (0, 255, 0)      # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)    # Yellow for medium
            else:
                color = (0, 165, 255)    # Orange for lower
            
            # Draw bounding box
            thickness = 3 if i == 0 else 2  # Thicker for best detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            center = detection['center']
            cv2.circle(frame, center, 5, color, -1)
            
            # Label
            label = f"BAT {confidence:.2f} ({source})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw line if available (for Hough line detections)
            if 'line_points' in detection:
                lx1, ly1, lx2, ly2 = detection['line_points']
                cv2.line(frame, (lx1, ly1), (lx2, ly2), color, 2)
        
        # Mark best detection
        if detections:
            best_center = detections[0]['center']
            cv2.putText(frame, "BEST BAT", (best_center[0] + 10, best_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def get_best_bat(self, detections):
        """Get the best bat detection for grid overlay"""
        return detections[0] if detections else None

# Test function
def test_advanced_bat_detector():
    """Test the advanced CV bat detector"""
    print("Testing Advanced CV Bat Detector...")
    
    detector = AdvancedBatDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\n🎯 ADVANCED CV BAT DETECTOR")
    print("=" * 35)
    print("✅ Automatic bat detection")
    print("✅ No user assistance needed")
    print("✅ Line-based detection")
    print("=" * 35)
    print("Controls: 'q' to quit, 's' for stats")
    print("=" * 35)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run advanced detection
        detections = detector.detect_bat_advanced_cv(frame)
        
        # Draw results
        frame = detector.draw_detections(frame, detections)
        
        # Add title
        cv2.putText(frame, "Advanced CV Bat Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow("Advanced CV Bat Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n📊 Found {len(detections)} bat detections")
            for i, det in enumerate(detections):
                print(f"   {i+1}: {det['source']} - {det['confidence']:.3f}")
            print()
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Advanced CV test completed!")

if __name__ == "__main__":
    test_advanced_bat_detector()