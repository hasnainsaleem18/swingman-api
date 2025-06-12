"""
Fast Optimized YOLOv8 Detector - FIXED VERSION with Person & Bat Detection
Handles bat detection at various distances with SPEED priority
"""

import cv2
import numpy as np
import os
import time

class YoloDetector:
    def __init__(self, custom_bat_model_path=None):
        """Initialize FAST detector optimized for real-time performance"""
        # Initialize storage
        self.last_detections = {
            'bats': [],
            'balls': [],
            'persons': [],
            'raw_detections': [],
            'model_info': {
                'bat_model': 'none',
                'ball_model': 'none'
            }
        }
        
        # Performance tracking
        self.detection_times = []
        self.avg_fps = 0
        
        # Try to import YOLO
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
            self.model_available = True
        except ImportError:
            self.model_available = False
            return
        
        # Initialize model variables
        self.bat_model = None
        self.ball_model = None
        self.bat_model_available = False
        self.ball_model_available = False
        
        # Detection thresholds - Lower thresholds for more sensitivity
        self.bat_confidence_threshold = 0.1   # Reduced from 0.15
        self.ball_confidence_threshold = 0.15  # Reduced from 0.2
        self.person_confidence_threshold = 0.2 # Reduced from 0.25
        self.iou_threshold = 0.4              # Reduced from 0.5
        
        # Validation parameters - Reduced for better detection
        self.min_bat_area = 20               # Reduced from 30
        self.max_area_ratio = 0.99           # Increased from 0.98
        self.min_bat_dimension = 6           # Reduced from 8
        
        # COCO class IDs
        self.coco_person_id = 0
        self.coco_sports_ball_id = 32
        
        # Load models
        if self.model_available:
            self._load_bat_model(custom_bat_model_path)
            self._load_ball_model()
        
    def _load_bat_model(self, custom_bat_model_path):
        """Load custom bat model with speed optimization"""
        try:
            if custom_bat_model_path is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                possible_locations = [
                    os.path.join(script_dir, "..", "Models", "swingman_bat_detector.pt"),
                    os.path.join(script_dir, "Models", "swingman_bat_detector.pt"),
                    os.path.join(script_dir, "..", "models", "swingman_bat_detector.pt"),
                    os.path.join(script_dir, "models", "swingman_bat_detector.pt"),
                    "swingman_bat_detector.pt",
                    "best.pt"
                ]
                
                for location in possible_locations:
                    if os.path.exists(location):
                        custom_bat_model_path = location
                        break
            
            if custom_bat_model_path and os.path.exists(custom_bat_model_path):
                # Load with performance optimization
                self.bat_model = self.YOLO(custom_bat_model_path)
                
                # Warm up the model
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _ = self.bat_model(dummy_frame, conf=0.5, verbose=False)
                
                self.bat_model_available = True
                self.bat_model_path = custom_bat_model_path
                self.bat_model_classes = self.bat_model.names
                self.last_detections['model_info']['bat_model'] = os.path.basename(custom_bat_model_path)
            else:
                self.bat_model_available = False
                
        except Exception:
            self.bat_model_available = False
    
    def _load_ball_model(self):
        """Load COCO model (optional for speed)"""
        try:
            self.ball_model = self.YOLO("yolov8n.pt")
            
            # Warm up COCO model too
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.ball_model(dummy_frame, conf=0.5, verbose=False)
            
            self.ball_model_available = True
            self.ball_model_classes = self.ball_model.names
            self.last_detections['model_info']['ball_model'] = "yolov8n.pt"
            
        except Exception:
            self.ball_model_available = False
    
    def detect_objects(self, frame):
        """FAST single-pass detection optimized for speed"""
        start_time = time.time()
        
        if not self.model_available:
            return self._empty_detections()
        
        detections = {
            'bats': [],
            'balls': [],
            'persons': [],
            'raw_detections': [],
            'frame_info': {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'total_detections': 0,
                'bat_detections': 0,
                'ball_detections': 0,
                'person_detections': 0,
                'detection_time_ms': 0,
                'fps': 0
            },
            'model_info': self.last_detections['model_info'].copy()
        }
        
        # FAST bat detection - single pass only
        if self.bat_model_available:
            try:
                bat_detections = self._detect_bats_fast(frame)
                detections['bats'].extend(bat_detections)
                detections['raw_detections'].extend(bat_detections)
                detections['frame_info']['bat_detections'] = len(bat_detections)
                
            except Exception:
                pass
        
        # Ball and person detection with COCO model
        if self.ball_model_available:
            try:
                ball_person_detections = self._detect_balls_and_persons_fast(frame)
                detections['balls'].extend(ball_person_detections['balls'])
                detections['persons'].extend(ball_person_detections['persons'])
                detections['raw_detections'].extend(ball_person_detections['balls'] + ball_person_detections['persons'])
                detections['frame_info']['ball_detections'] = len(ball_person_detections['balls'])
                detections['frame_info']['person_detections'] = len(ball_person_detections['persons'])
                
            except Exception:
                pass
        
        # Performance tracking
        detection_time = (time.time() - start_time) * 1000  # Convert to ms
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 30:  # Keep last 30 measurements
            self.detection_times.pop(0)
        
        avg_detection_time = sum(self.detection_times) / len(self.detection_times)
        self.avg_fps = 1000 / avg_detection_time if avg_detection_time > 0 else 0
        
        detections['frame_info']['detection_time_ms'] = detection_time
        detections['frame_info']['fps'] = self.avg_fps
        detections['frame_info']['total_detections'] = len(detections['raw_detections'])
        
        self.last_detections = detections
        return detections
    
    def _detect_bats_fast(self, frame):
        """FAST single-pass bat detection with smart validation"""
        bat_detections = []
        frame_area = frame.shape[0] * frame.shape[1]
        
        try:
            results = self.bat_model(
                frame,
                conf=self.bat_confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w = frame.shape[:2]
                        x1 = max(0, min(w-1, x1))
                        y1 = max(0, min(h-1, y1))
                        x2 = max(x1+1, min(w, x2))
                        y2 = max(y1+1, min(h, y2))
                        
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': 'bat',
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'width': width,
                            'height': height,
                            'area': area,
                            'aspect_ratio': height / width if width > 0 else 0,
                            'detection_type': 'bat_fast',
                            'model_source': 'custom',
                            'frame_area': frame_area,
                            'area_ratio': area / frame_area
                        }
                        
                        if self._is_valid_bat_fast(detection):
                            bat_detections.append(detection)
        
        except Exception:
            pass
        
        return bat_detections
    
    def _is_valid_bat_fast(self, detection):
        """ULTRA FAST validation - minimal checks for speed"""
        confidence = detection['confidence']
        area = detection['area']
        area_ratio = detection['area_ratio']
        width = detection['width']
        height = detection['height']
        
        if (confidence >= 0.08 and           # Reduced from 0.1
            area >= self.min_bat_area and 
            area_ratio <= self.max_area_ratio and 
            width >= self.min_bat_dimension and 
            height >= self.min_bat_dimension):
            return True
        return False
    
    def _detect_balls_and_persons_fast(self, frame):
        """FAST ball and person detection - single pass"""
        ball_detections = []
        person_detections = []
        
        try:
            results = self.ball_model(
                frame,
                conf=0.1,
                iou=self.iou_threshold,
                verbose=False
            )
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.ball_model_classes.get(class_id, f'class_{class_id}')
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'width': width,
                            'height': height,
                            'area': width * height,
                            'aspect_ratio': height / width if width > 0 else 0,
                            'model_source': 'coco'
                        }
                        
                        # Check for ball
                        is_ball = (class_id == self.coco_sports_ball_id or 
                                'ball' in class_name.lower() or
                                'sports ball' in class_name.lower())
                            
                        # Check for person
                        is_person = (class_id == self.coco_person_id or 
                                   class_name.lower() == 'person')
                        
                        if is_ball and confidence >= self.ball_confidence_threshold:
                            detection['detection_type'] = 'coco_ball'
                            ball_detections.append(detection)
                        elif is_person and confidence >= self.person_confidence_threshold:
                            detection['detection_type'] = 'coco_person'
                            person_detections.append(detection)
        
        except Exception:
            pass
        
        return {'balls': ball_detections, 'persons': person_detections}
    
    def get_best_bat_detection(self, detections=None, min_confidence=0.08):  # Reduced from 0.1
        """FAST best bat selection"""
        if detections is None:
            detections = self.last_detections
        
        if not detections or not detections['bats']:
            return None
        
        # Simple max confidence selection for speed
        valid_bats = [bat for bat in detections['bats'] 
                     if bat['confidence'] >= min_confidence]
        
        return max(valid_bats, key=lambda x: x['confidence']) if valid_bats else None
    
    def get_best_ball_detection(self, detections=None, min_confidence=0.1):  # Reduced from 0.15
        """FAST best ball selection"""
        if detections is None:
            detections = self.last_detections
        
        if not detections or not detections['balls']:
            return None
        
        valid_balls = [ball for ball in detections['balls'] 
                      if ball['confidence'] >= min_confidence]
        
        return max(valid_balls, key=lambda x: x['confidence']) if valid_balls else None
    
    def get_best_person_detection(self, detections=None, min_confidence=0.2):
        """FAST best person selection"""
        if detections is None:
            detections = self.last_detections
        
        if not detections or not detections['persons']:
            return None
        
        valid_persons = [person for person in detections['persons'] 
                        if person['confidence'] >= min_confidence]
        
        return max(valid_persons, key=lambda x: x['confidence']) if valid_persons else None
    
    def draw_detections(self, frame, detections):
        """Draw detections with clean layout"""
        if not detections:
            return frame
            
        # Draw tracking zone first
        from utils import draw_tracking_box
        draw_tracking_box(frame)
        
        # Draw detections with minimal labels
        for bat in detections['bats']:
            self._draw_fast_detection(frame, bat, (0, 255, 0), "B")
        
        for ball in detections['balls']:
            self._draw_fast_detection(frame, ball, (0, 0, 255), "B")
        
        for person in detections['persons']:
            self._draw_fast_detection(frame, person, (255, 0, 0), "P")
        
        return frame
        
    def _draw_fast_detection(self, frame, detection, color, label_prefix):
        """Draw a minimal detection box"""
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Just draw box with thin line
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        
        # Minimal label
        if confidence > 0.3:  # Only show label for confident detections
            label = f"{label_prefix}{confidence:.1f}"
            cv2.putText(frame, label, (x1, y1-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _empty_detections(self):
        """Empty detection structure"""
        return {
            'bats': [], 'balls': [], 'persons': [], 'raw_detections': [],
            'frame_info': {
                'width': 0, 'height': 0, 'total_detections': 0,
                'bat_detections': 0, 'ball_detections': 0, 'person_detections': 0,
                'detection_time_ms': 0, 'fps': 0
            },
            'model_info': {'bat_model': 'none', 'ball_model': 'none'}
        }
    
    def is_available(self):
        return self.bat_model_available or self.ball_model_available
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        return {
            'avg_fps': self.avg_fps,
            'avg_detection_time_ms': sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0,
            'recent_detection_times': self.detection_times[-10:],  # Last 10 measurements
            'bat_model_available': self.bat_model_available,
            'ball_model_available': self.ball_model_available
        }

def test_fast_detector():
    """Test the FAST detector with minimal output"""
    detector = YoloDetector()
    if not detector.is_available():
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect_objects(frame)
        frame = detector.draw_detections(frame, detections)
        
        cv2.imshow("Swingman Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fast_detector()