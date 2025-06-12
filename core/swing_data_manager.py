"""
Swing data storage and management module
Handles saving, loading, and organizing swing analysis results
"""

import os
import json
import datetime
import cv2
from collections import defaultdict
from utils.json_encoder import NumpyEncoder
import csv

class SwingDataManager:
    """
    Manages swing data storage and retrieval
    """
    
    def __init__(self, base_dir="output"):
        """Initialize the data manager"""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.current_session = self._create_new_session()
        
        print("✅ Swing Data Manager initialized")
    
    def _create_new_session(self, name=None):
        """Create a new session structure"""
        timestamp = datetime.datetime.now()
        session_id = f"session_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        if not name:
            name = f"Session {timestamp.strftime('%Y-%m-%d %H:%M')}"
            
        return {
            "id": session_id,
            "name": name,
            "timestamp": timestamp.isoformat(),
            "swings": [],
            "stats": {
                "num_swings": 0,
                "avg_efficiency": 0.0,
                "max_efficiency": 0,
                "min_efficiency": 100,
                "last_updated": timestamp.isoformat()
            },
            "metadata": {
                "created_at": timestamp.isoformat(),
                "source": "Swingman CV"
            }
        }
    
    def start_new_session(self, name=None):
        """Start a new tracking session"""
        self.current_session = self._create_new_session(name)
        return self.current_session["id"]
    
    def add_swing_to_session(self, metrics, frame, path_points):
        """Add a swing to the current session"""
        if not self.current_session:
            return False

        timestamp = datetime.datetime.now()
        swing_id = f"swing_{len(self.current_session['swings']) + 1}_{timestamp.strftime('%H%M%S')}"
        
        # Prepare swing data
        swing_data = {
            "id": swing_id,
            "timestamp": timestamp.isoformat(),
            "data": {
                "efficiency_score": metrics["efficiency_score"],
                "power_score": metrics["power_score"],
                "swing_speed": metrics["swing_speed"],
                "path_consistency": metrics["path_consistency"],
                "follow_through": metrics["follow_through"],
                "pose_stability": metrics["pose_stability"],
                "sweet_spot_contact": metrics["sweet_spot_contact"],
                "impact_point": metrics["impact_point"],
                "path_length": len(path_points)
            },
            "has_image": True,
            "has_path": bool(path_points)
        }

        # Save swing image
        session_dir = os.path.join(self.base_dir, self.current_session["id"])
        os.makedirs(session_dir, exist_ok=True)
        
        image_path = os.path.join(session_dir, f"{swing_id}.png")
        cv2.imwrite(image_path, frame)

        # Save path points if available
        if path_points:
            path_file = os.path.join(session_dir, f"{swing_id}_path.json")
            with open(path_file, "w") as f:
                json.dump({"points": path_points}, f, cls=NumpyEncoder)

        # Add to session
        self.current_session["swings"].append(swing_data)
        
        # Update session stats
        stats = self.current_session["stats"]
        stats["num_swings"] = len(self.current_session["swings"])
        
        # Update efficiency stats
        efficiencies = [s["data"]["efficiency_score"] for s in self.current_session["swings"]]
        stats["avg_efficiency"] = sum(efficiencies) / len(efficiencies)
        stats["max_efficiency"] = max(efficiencies)
        stats["min_efficiency"] = min(efficiencies)
        stats["last_updated"] = timestamp.isoformat()

        return True
    
    def save_current_session(self):
        """Save current session data"""
        if not self.current_session:
            return None
            
        session_dir = os.path.join(self.base_dir, self.current_session["id"])
        os.makedirs(session_dir, exist_ok=True)
        
        # Save session data
        session_file = os.path.join(session_dir, "session.json")
        with open(session_file, "w") as f:
            json.dump(self.current_session, f, indent=2, cls=NumpyEncoder)
            
        return session_dir
    
    def load_session(self, session_id):
        """
        Load a session from disk
        
        Parameters:
            session_id: ID of the session to load
        
        Returns:
            session_data: Dictionary containing session data or None if not found
        """
        # Check if directory exists
        session_dir = os.path.join(self.base_dir, session_id)
        if not os.path.isdir(session_dir):
            print(f"Session directory not found: {session_dir}")
            return None
        
        # Check if metadata file exists
        metadata_file = os.path.join(session_dir, "session.json")
        if not os.path.isfile(metadata_file):
            print(f"Session metadata file not found: {metadata_file}")
            return None
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                session_data = json.load(f)
            
            return session_data
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def list_sessions(self):
        """
        List all available sessions
        
        Returns:
            sessions: List of session metadata
        """
        sessions = []
        
        if not os.path.isdir(self.base_dir):
            return sessions
        
        # Check each directory
        for item in os.listdir(self.base_dir):
            session_dir = os.path.join(self.base_dir, item)
            
            # Skip if not a directory
            if not os.path.isdir(session_dir):
                continue
            
            # Check for session metadata file
            metadata_file = os.path.join(session_dir, "session.json")
            if not os.path.isfile(metadata_file):
                continue
            
            # Load basic metadata
            try:
                with open(metadata_file, 'r') as f:
                    session_data = json.load(f)
                
                # Add to list
                sessions.append({
                    "id": session_data.get("id", item),
                    "name": session_data.get("name", item),
                    "timestamp": session_data.get("timestamp", ""),
                    "num_swings": len(session_data.get("swings", [])),
                    "avg_efficiency": session_data.get("stats", {}).get("avg_efficiency", 0)
                })
            except Exception as e:
                print(f"Error loading session metadata for {item}: {e}")
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return sessions
    
    def get_swing_image(self, session_id, swing_id):
        """
        Get image for a specific swing
        
        Parameters:
            session_id: ID of the session
            swing_id: ID of the swing
        
        Returns:
            image: OpenCV image or None if not found
        """
        image_path = os.path.join(self.base_dir, session_id, f"{swing_id}.png")
        
        if not os.path.isfile(image_path):
            print(f"Swing image not found: {image_path}")
            return None
        
        # Load image
        try:
            image = cv2.imread(image_path)
            return image
        except Exception as e:
            print(f"Error loading swing image: {e}")
            return None
    
    def get_swing_path_points(self, session_id, swing_id):
        """
        Get path points for a specific swing
        
        Parameters:
            session_id: ID of the session
            swing_id: ID of the swing
        
        Returns:
            path_points: List of path points or None if not found
        """
        path_file = os.path.join(self.base_dir, session_id, f"{swing_id}_path.json")
        
        if not os.path.isfile(path_file):
            print(f"Swing path file not found: {path_file}")
            return None
        
        # Load path points
        try:
            with open(path_file, 'r') as f:
                serialized_points = json.load(f)
            
            # Convert to (x, y) tuples
            path_points = serialized_points["points"]
            
            return path_points
        except Exception as e:
            print(f"Error loading swing path points: {e}")
            return None
    
    def export_data(self, session_id, format="json"):
        """Export session data in specified format"""
        session_dir = os.path.join(self.base_dir, session_id)
        if not os.path.exists(session_dir):
            return False
            
        # Load session data
        session_file = os.path.join(session_dir, "session.json")
        with open(session_file, "r") as f:
            session_data = json.load(f)
            
        if format.lower() == "json":
            # Already in JSON format
            return True
            
        elif format.lower() == "csv":
            # Export swing data to CSV
            csv_file = os.path.join(session_dir, "swings.csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "Swing ID", "Timestamp", "Efficiency Score", "Power Score",
                    "Swing Speed", "Path Consistency", "Follow Through",
                    "Pose Stability", "Sweet Spot Contact", "Impact Point X",
                    "Impact Point Y", "Path Length"
                ])
                
                # Write data
                for swing in session_data["swings"]:
                    data = swing["data"]
                    impact_x, impact_y = data["impact_point"] if data["impact_point"] else (None, None)
                    writer.writerow([
                        swing["id"],
                        swing["timestamp"],
                        data["efficiency_score"],
                        data.get("power_score", ""),
                        data.get("swing_speed", ""),
                        data.get("path_consistency", ""),
                        data.get("follow_through", ""),
                        data.get("pose_stability", ""),
                        data.get("sweet_spot_contact", ""),
                        impact_x,
                        impact_y,
                        data["path_length"]
                    ])
            return True
            
        return False