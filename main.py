#!/usr/bin/env python3
"""
Swingman API Server - Render Deployment Version
Optimized FastAPI backend for iOS app integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import base64
import time
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Swingman API", 
    description="Baseball Swing Analysis API for iOS", 
    version="1.0.0"
)

# Add CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_sessions = {}
FRAME_COUNTER = 0

# === DATA MODELS ===
class SwingMetrics(BaseModel):
    efficiency_score: int = 0
    power_score: int = 0
    swing_speed: float = 0.0
    path_consistency: int = 0
    follow_through: int = 0
    pose_stability: int = 0
    sweet_spot_contact: bool = False
    impact_point: Optional[List[float]] = None

class SwingAnalysisResponse(BaseModel):
    success: bool
    metrics: SwingMetrics
    swing_path: List[List[float]]
    message: str
    session_id: Optional[str] = None

class FrameData(BaseModel):
    frame_base64: str
    session_id: str
    timestamp: float

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str

class StartTrackingRequest(BaseModel):
    session_id: str
    x: float
    y: float

class StopTrackingRequest(BaseModel):
    session_id: str

# === CORE ANALYSIS CLASSES ===
class SimpleSwingTracker:
    """Simplified swing tracker for Render deployment"""
    
    def __init__(self):
        self.is_tracking = False
        self.swing_path_points = []
        self.timestamps = []
        self.current_position = None
        
    def start_tracking_session(self):
        self.is_tracking = True
        self.swing_path_points = []
        self.timestamps = []
        logger.info("Tracking session started")
        
    def stop_tracking_session(self):
        self.is_tracking = False
        success = len(self.swing_path_points) >= 2
        logger.info(f"Tracking session stopped. Success: {success}")
        return success
        
    def update_current_position(self, x, y):
        self.current_position = (int(x), int(y))
        if self.is_tracking:
            self.swing_path_points.append(self.current_position)
            self.timestamps.append(time.time())
            
    def process_frame(self, frame):
        """Process frame and return basic results"""
        # Simple frame processing - just return current state
        return {
            'swing_path': self.swing_path_points,
            'metrics': self.get_current_metrics(),
            'is_tracking': self.is_tracking
        }
        
    def get_current_metrics(self):
        """Calculate simple metrics based on swing path"""
        if len(self.swing_path_points) < 2:
            return {
                'efficiency_score': 0,
                'power_score': 0,
                'swing_speed': 0.0,
                'path_consistency': 0,
                'follow_through': 0,
                'pose_stability': 50,
                'sweet_spot_contact': False,
                'impact_point': None
            }
            
        # Calculate basic metrics
        total_distance = self._calculate_path_distance()
        time_diff = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.1
        swing_speed = total_distance / time_diff if time_diff > 0 else 0
        
        # Generate metrics based on movement
        efficiency_score = min(100, max(10, int(total_distance / 3)))
        power_score = min(100, max(10, int(swing_speed * 2)))
        path_consistency = min(100, max(20, int(80 - (len(self.swing_path_points) * 0.5))))
        follow_through = min(100, max(20, int(total_distance / 4)))
        
        return {
            'efficiency_score': efficiency_score,
            'power_score': power_score,
            'swing_speed': round(swing_speed, 1),
            'path_consistency': path_consistency,
            'follow_through': follow_through,
            'pose_stability': 50,  # Default
            'sweet_spot_contact': total_distance > 150,
            'impact_point': self.swing_path_points[-1] if self.swing_path_points else None
        }
        
    def _calculate_path_distance(self):
        """Calculate total distance of swing path"""
        if len(self.swing_path_points) < 2:
            return 0
            
        total = 0
        for i in range(1, len(self.swing_path_points)):
            p1 = self.swing_path_points[i-1]
            p2 = self.swing_path_points[i]
            distance = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            total += distance
        return total

# === API ENDPOINTS ===

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Swingman API Server is running", 
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "server_info": {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "environment": "production" if os.getenv("RENDER") else "development"
        }
    }

@app.post("/api/session/start")
async def start_session(user_id: str = "default_user"):
    """Start a new swing analysis session"""
    try:
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create new session with simple tracker
        active_sessions[session_id] = {
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "tracker": SimpleSwingTracker(),
            "swing_data": [],
            "status": "active"
        }
        
        logger.info(f"Started new session: {session_id}")
        
        return SessionResponse(
            session_id=session_id,
            status="started",
            message="Session started successfully"
        )
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/{session_id}/stop")
async def stop_session(session_id: str):
    """Stop a swing analysis session"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        session["status"] = "completed"
        session["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Stopped session: {session_id}")
        
        return SessionResponse(
            session_id=session_id,
            status="stopped",
            message="Session stopped successfully"
        )
    except Exception as e:
        logger.error(f"Error stopping session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/frame")
async def analyze_frame(frame_data: FrameData):
    """Analyze a single frame from iOS app"""
    global FRAME_COUNTER
    FRAME_COUNTER += 1
    
    logger.info(f"Processing frame #{FRAME_COUNTER} for session: {frame_data.session_id}")
    
    try:
        session_id = frame_data.session_id
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        tracker = session["tracker"]
        
        # Decode frame
        try:
            frame_bytes = base64.b64decode(frame_data.frame_base64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("Failed to decode frame")
                
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Process frame
        results = tracker.process_frame(frame)
        
        # Get swing path and metrics
        swing_path = []
        if results.get('swing_path'):
            swing_path = [[float(p[0]), float(p[1])] for p in results['swing_path']]
        
        metrics_data = results.get('metrics', {})
        
        metrics = SwingMetrics(
            efficiency_score=int(metrics_data.get('efficiency_score', 0)),
            power_score=int(metrics_data.get('power_score', 0)),
            swing_speed=float(metrics_data.get('swing_speed', 0.0)),
            path_consistency=int(metrics_data.get('path_consistency', 0)),
            follow_through=int(metrics_data.get('follow_through', 0)),
            pose_stability=int(metrics_data.get('pose_stability', 50)),
            sweet_spot_contact=bool(metrics_data.get('sweet_spot_contact', False)),
            impact_point=metrics_data.get('impact_point')
        )
        
        # Store frame data
        session["swing_data"].append({
            "timestamp": frame_data.timestamp,
            "metrics": metrics.dict(),
            "swing_path": swing_path,
            "frame_number": FRAME_COUNTER
        })
        
        return SwingAnalysisResponse(
            success=True,
            metrics=metrics,
            swing_path=swing_path,
            message=f"Frame #{FRAME_COUNTER} processed successfully",
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/swing/start_tracking")
async def start_tracking(request: StartTrackingRequest):
    """Start tracking at specific coordinates"""
    try:
        session_id = request.session_id
        x, y = request.x, request.y
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        tracker = session["tracker"]
        
        # Start tracking
        tracker.start_tracking_session()
        tracker.update_current_position(int(x), int(y))
        
        logger.info(f"Started tracking for session {session_id} at ({x}, {y})")
        
        return {"success": True, "message": "Tracking started", "coordinates": [x, y]}
        
    except Exception as e:
        logger.error(f"Error starting tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/swing/stop_tracking")
async def stop_tracking(request: StopTrackingRequest):
    """Stop tracking and analyze swing"""
    try:
        session_id = request.session_id
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        tracker = session["tracker"]
        
        # Stop tracking and get final analysis
        success = tracker.stop_tracking_session()
        
        logger.info(f"Stopped tracking for session {session_id}, success: {success}")
        
        if success:
            final_metrics = tracker.get_current_metrics()
            
            metrics = SwingMetrics(
                efficiency_score=int(final_metrics.get('efficiency_score', 0)),
                power_score=int(final_metrics.get('power_score', 0)),
                swing_speed=float(final_metrics.get('swing_speed', 0.0)),
                path_consistency=int(final_metrics.get('path_consistency', 0)),
                follow_through=int(final_metrics.get('follow_through', 0)),
                pose_stability=int(final_metrics.get('pose_stability', 50)),
                sweet_spot_contact=bool(final_metrics.get('sweet_spot_contact', False)),
                impact_point=final_metrics.get('impact_point')
            )
            
            return SwingAnalysisResponse(
                success=True,
                metrics=metrics,
                swing_path=[[float(p[0]), float(p[1])] for p in tracker.swing_path_points],
                message="Swing analysis completed",
                session_id=session_id
            )
        else:
            return SwingAnalysisResponse(
                success=False,
                metrics=SwingMetrics(),
                swing_path=[],
                message="Insufficient swing data for analysis"
            )
        
    except Exception as e:
        logger.error(f"Error stopping tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/results")
async def get_session_results(session_id: str):
    """Get complete session results"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        swing_data = session["swing_data"]
        
        if not swing_data:
            return {
                "session_id": session_id,
                "status": session["status"],
                "summary": {"total_swings": 0},
                "swings": []
            }
        
        # Calculate summary statistics
        total_swings = len(swing_data)
        efficiencies = [s["metrics"]["efficiency_score"] for s in swing_data]
        power_scores = [s["metrics"]["power_score"] for s in swing_data]
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "started_at": session["started_at"],
            "summary": {
                "total_swings": total_swings,
                "avg_efficiency": round(sum(efficiencies) / total_swings, 1),
                "avg_power": round(sum(power_scores) / total_swings, 1),
                "max_efficiency": max(efficiencies),
                "best_swing": max(swing_data, key=lambda x: x["metrics"]["efficiency_score"])
            },
            "swings": swing_data[-10:]  # Return last 10 swings
        }
        
    except Exception as e:
        logger.error(f"Error getting session results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "status": session["status"],
                "started_at": session["started_at"],
                "user_id": session["user_id"],
                "swing_count": len(session["swing_data"])
            }
            for sid, session in active_sessions.items()
        ]
    }

# === STARTUP ===
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Swingman API Server starting up...")
    logger.info("✅ Server initialized successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False
    )