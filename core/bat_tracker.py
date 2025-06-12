"""
Bat tracking functionality - core CV component
"""

import cv2
import numpy as np
from collections import deque
import sys
import traceback

class BatTracker:
    """Tracks a baseball bat using OpenCV"""
    
    def __init__(self, debug=False):
        # Tracking state
        self.is_tracking = False
        self.track_box = None
        self.tracker = None
        self.path_points = []
        self.max_points = 100  # Maximum points to store
        self.last_points = deque(maxlen=10)  # Recent points for angle calculation
        
        # Debug mode
        self.debug = debug
        
        # Initialize object detector
        try:
            self.object_detector = cv2.createBackgroundSubtractorMOG2(
                history=100, 
                varThreshold=50
            )
        except Exception as e:
            if debug:
                print(f"Error initializing background subtractor: {e}")
            self.object_detector = None
        
        # Check OpenCV version
        self.opencv_version = cv2.__version__
        if debug:
            print(f"Using OpenCV version: {self.opencv_version}")
            
            # Check available trackers
            self._check_available_trackers()
    
    def _check_available_trackers(self):
        """Check which trackers are available in this OpenCV installation"""
        trackers_to_check = [
            ("Legacy CSRT", lambda: cv2.legacy.TrackerCSRT_create() if hasattr(cv2, 'legacy') else None),
            ("Legacy KCF", lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') else None),
            ("Standard CSRT", lambda: cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else None),
            ("Standard KCF", lambda: cv2.TrackerKCF_create() if hasattr(cv2, 'TrackerKCF_create') else None),
        ]
        
        for name, create_func in trackers_to_check:
            try:
                tracker = create_func()
                if tracker is not None:
                    print(f"Tracker {name} is available")
            except Exception as e:
                print(f"Tracker {name} is NOT available: {e}")
    
    def create_tracker(self):
        """Create a tracker based on available OpenCV version"""
        # Try multiple tracker types in order of preference
        tracker_creators = [
            # Legacy module trackers (OpenCV 4.5.1+)
            lambda: cv2.legacy.TrackerCSRT_create() if hasattr(cv2, 'legacy') else None,
            lambda: cv2.legacy.TrackerKCF_create() if hasattr(cv2, 'legacy') else None,
            
            # Standard tracker API
            lambda: cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else None,
            lambda: cv2.TrackerKCF_create() if hasattr(cv2, 'TrackerKCF_create') else None,
        ]
        
        # Try each tracker creator in order
        for creator in tracker_creators:
            try:
                tracker = creator()
                if tracker is not None:
                    if self.debug:
                        print(f"Created tracker: {tracker}")
                    return tracker
            except Exception as e:
                if self.debug:
                    print(f"Failed to create tracker: {e}")
        
        # If all trackers fail, return None for manual tracking
        print("No OpenCV trackers available, using manual tracking")
        return None
    
    def start_tracking(self, frame, point):
        """Start tracking at the specified point"""
        if frame is None:
            print("Cannot start tracking with None frame")
            return False
            
        # Create initial bounding box around the point
        try:
            x, y = int(point[0]), int(point[1])
            width, height = 60, 30  # Approximate bat width/height
            
            # Ensure box is within frame bounds
            frame_height, frame_width = frame.shape[:2]
            x = max(width//2, min(frame_width - width//2, x))
            y = max(height//2, min(frame_height - height//2, y))
            
            self.track_box = (x - width//2, y - height//2, width, height)
            
            # Initialize tracker
            self.tracker = self.create_tracker()
            
            success = False
            if self.tracker is not None:
                # Initialize with OpenCV tracker
                try:
                    # Make sure box coordinates are integers
                    box = tuple(int(v) for v in self.track_box)
                    
                    # Box format: (x, y, width, height)
                    # Ensure width and height are positive and non-zero
                    x, y, w, h = box
                    if w <= 0 or h <= 0:
                        w = max(1, w)
                        h = max(1, h)
                        box = (x, y, w, h)
                    
                    success = self.tracker.init(frame, box)
                    if not success and self.debug:
                        print("OpenCV tracker initialization failed")
                except Exception as e:
                    if self.debug:
                        print(f"Error initializing tracker: {str(e)}")
                        traceback.print_exc()
                    self.tracker = None
            
            # If tracker initialization failed, fall back to manual tracking
            if self.tracker is None or not success:
                if self.debug:
                    print("Using manual tracking fallback")
                self.tracker = None
                success = True  # Manual tracking always succeeds initially
            
            if success:
                self.is_tracking = True
                self.path_points = [(x, y)]
                self.last_points.clear()
                self.last_points.append((x, y))
                print("Tracking started")
                return True
            else:
                print("Failed to start tracking")
                return False
                
        except Exception as e:
            print(f"Error in start_tracking: {str(e)}")
            traceback.print_exc()
            return False
    
    def update_tracking(self, frame):
        """Update tracking with new frame"""
        if not self.is_tracking:
            return False, None
        
        if frame is None:
            print("Cannot update tracking with None frame")
            return False, None
        
        try:
            if self.tracker is not None:
                # Update using OpenCV tracker
                try:
                    success, box = self.tracker.update(frame)
                except Exception as e:
                    if self.debug:
                        print(f"Error updating tracker: {str(e)}")
                    success = False
                    box = self.track_box
            else:
                # Manual tracking fallback
                success, box = self._manual_tracking_update(frame)
            
            if success:
                try:
                    # Convert box to int values
                    box = tuple(map(int, box))
                    x, y = box[0] + box[2]//2, box[1] + box[3]//2  # Center point
                    
                    # Ensure point is within frame bounds
                    height, width = frame.shape[:2]
                    x = max(0, min(width - 1, x))
                    y = max(0, min(height - 1, y))
                    
                    # Add point to path (if it moved enough to avoid duplicates)
                    if len(self.path_points) == 0 or self._distance(self.path_points[-1], (x, y)) > 2:
                        self.path_points.append((x, y))
                        self.last_points.append((x, y))
                        
                        # Limit the number of points
                        if len(self.path_points) > self.max_points:
                            self.path_points.pop(0)
                    
                    self.track_box = box
                    return True, box
                except Exception as e:
                    if self.debug:
                        print(f"Error processing tracking box: {str(e)}")
                    return False, None
            else:
                if self.debug:
                    print("Tracking lost")
                return False, None
        except Exception as e:
            print(f"Unexpected error in update_tracking: {str(e)}")
            return False, None
    
    def _manual_tracking_update(self, frame):
        """Fallback tracking method when OpenCV trackers are unavailable"""
        if len(self.path_points) < 1:
            return False, None
        
        try:
            # Get last known position
            last_x, last_y = self.path_points[-1]
            last_box = self.track_box if self.track_box else (last_x - 30, last_y - 15, 60, 30)
            
            # Define search region
            search_size = 100
            frame_height, frame_width = frame.shape[:2]
            
            x1 = max(0, last_x - search_size//2)
            y1 = max(0, last_y - search_size//2)
            x2 = min(frame_width - 1, last_x + search_size//2)
            y2 = min(frame_height - 1, last_y + search_size//2)
            
            # Skip if search region is invalid
            if x1 >= x2 or y1 >= y2:
                return False, last_box
            
            # Get search region
            roi = frame[y1:y2, x1:x2]
            
            # Detect motion in the ROI
            motion_regions = []
            if self.object_detector is not None:
                try:
                    motion_regions = self.detect_motion_areas(roi)
                except Exception as e:
                    if self.debug:
                        print(f"Error detecting motion: {str(e)}")
            
            if motion_regions:
                # Get largest motion region
                largest_area = 0
                largest_region = None
                
                for region in motion_regions:
                    x, y, w, h = region
                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_region = region
                
                if largest_region:
                    x, y, w, h = largest_region
                    # Adjust to frame coordinates
                    x += x1
                    y += y1
                    
                    # Update box - ensure width/height are positive
                    w = max(1, w)
                    h = max(1, h)
                    new_box = (x, y, w, h)
                    return True, new_box
                    
            # If no motion detected, keep the previous box but update slightly based on momentum
            if len(self.path_points) >= 2:
                # Calculate momentum from last two points
                prev_x, prev_y = self.path_points[-2]
                dx = last_x - prev_x
                dy = last_y - prev_y
                
                # Apply momentum (with damping)
                damping = 0.8
                new_x = int(last_x + dx * damping)
                new_y = int(last_y + dy * damping)
                
                # Constrain to frame boundaries
                new_x = max(0, min(frame_width - 1, new_x))
                new_y = max(0, min(frame_height - 1, new_y))
                
                # Update box
                w, h = last_box[2], last_box[3]
                new_box = (new_x - w//2, new_y - h//2, w, h)
                
                return True, new_box
            
            # If all else fails, keep the last box
            return True, last_box
            
        except Exception as e:
            print(f"Error in manual tracking: {str(e)}")
            # If we have a last_box, return it as fallback
            if hasattr(self, 'track_box') and self.track_box is not None:
                return True, self.track_box
            return False, None
    
    def detect_motion_areas(self, frame, threshold=20):
        """Detect areas with motion"""
        try:
            if frame is None or frame.size == 0:
                return []
                
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply background subtraction if available
            if self.object_detector is not None:
                fg_mask = self.object_detector.apply(frame)
                
                # Threshold to get binary image
                _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by size
                motion_areas = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Min area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        motion_areas.append((x, y, w, h))
                        
                return motion_areas
            else:
                # Simple frame differencing if no background subtractor
                if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                    frame_diff = cv2.absdiff(gray, self.prev_gray)
                    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours by size
                    motion_areas = []
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # Min area threshold
                            x, y, w, h = cv2.boundingRect(contour)
                            motion_areas.append((x, y, w, h))
                    
                    self.prev_gray = gray.copy()
                    return motion_areas
                else:
                    # First frame, just store it
                    self.prev_gray = gray.copy()
                    return []
        except Exception as e:
            print(f"Error detecting motion: {str(e)}")
            return []
    
    def stop_tracking(self):
        """Stop tracking and return the path points"""
        self.is_tracking = False
        self.tracker = None
        return self.path_points.copy()
    
    def get_bat_angle(self):
        """Calculate the current angle of the bat based on recent movement"""
        if len(self.last_points) < 2:
            return 0.0
            
        try:
            # Get last points
            p1 = self.last_points[0]
            p2 = self.last_points[-1]
            
            # Calculate angle
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.arctan2(dy, dx)
            
            return angle
        except Exception as e:
            print(f"Error calculating angle: {str(e)}")
            return 0.0
    
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        try:
            return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        except Exception as e:
            print(f"Error calculating distance: {str(e)}")
            return 0.0