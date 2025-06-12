#!/usr/bin/env python3
"""
Swingman API Server
FastAPI backend for iOS app integration
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import base64
import json
import asyncio
import uvicorn
from datetime import datetime
import logging
import io
from PIL import Image

# Import your swing analysis modules
from core.enhanced_swing_tracker import EnhancedSwingTracker
from core.swing_data_manager import SwingDataManager
from core.heatmap_generator import HeatmapGenerator

import time
import os
print(f"🐍 PYTHON SERVER STARTING - PID: {os.getpid()} at {time.strftime('%H:%M:%S')}")

# Add this global counter
FRAME_COUNTER = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Swingman API", description="Baseball Swing Analysis API", version="1.0.0")

# Add CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your iOS app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
tracker = None
data_manager = None
heatmap_generator = None
active_sessions = {}

# Data models
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

# Add these new request models after the existing ones
class StartTrackingRequest(BaseModel):
    session_id: str
    x: float
    y: float

class StopTrackingRequest(BaseModel):
    session_id: str

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global tracker, data_manager, heatmap_generator
    
    try:
        # Initialize components
        tracker = EnhancedSwingTracker(enable_pose=True)
        data_manager = SwingDataManager(base_dir="api_output")
        heatmap_generator = HeatmapGenerator(output_dir="api_output")
        
        logger.info("✅ Swingman API Server initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Swingman API Server is running", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "tracker": tracker is not None,
            "data_manager": data_manager is not None,
            "heatmap_generator": heatmap_generator is not None
        }
    }

@app.post("/api/session/start")
async def start_session(user_id: str = "default_user"):
    """Start a new swing analysis session"""
    try:
        session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create new session
        active_sessions[session_id] = {
            "user_id": user_id,
            "started_at": datetime.now().isoformat(),
            "tracker": EnhancedSwingTracker(enable_pose=True),
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
    global FRAME_COUNTER
    FRAME_COUNTER += 1
    
    print(f"🔥 PYTHON FRAME #{FRAME_COUNTER} - Session: {frame_data.session_id}")
    print(f"🔥 Frame size: {len(frame_data.frame_base64)} bytes")
    
    try:
        session_id = frame_data.session_id
        
        if session_id not in active_sessions:
            print(f"❌ Session {session_id} not found!")
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        session_tracker = session["tracker"]
        
        print(f"🎯 TRACKING STATUS: {session_tracker.is_tracking}")
        
        # Decode frame
        frame_bytes = base64.b64decode(frame_data.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("❌ Frame decode failed!")
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        print(f"✅ Frame decoded: {frame.shape}")
        
        # FORCE UPDATE POSITION IF TRACKING
        if session_tracker.is_tracking:
            # Force some movement to test
            session_tracker.update_current_position(320 + FRAME_COUNTER, 240 + FRAME_COUNTER)
            print(f"🎯 FORCED POSITION UPDATE: ({320 + FRAME_COUNTER}, {240 + FRAME_COUNTER})")
        
        # Process frame
        results = session_tracker.process_frame(frame)
        
        print(f"📊 RESULTS: swing_path={len(results.get('swing_path', []))}")
        print(f"📊 METRICS: {results.get('metrics', {})}")
        
        # Rest of your existing code...
        swing_path = []
        if results.get('swing_path'):
            swing_path = [[float(p[0]), float(p[1])] for p in results['swing_path']]
            print(f"✅ Swing path points: {len(swing_path)}")
        
        metrics_data = results.get('metrics', {})
        
        # FORCE SOME METRICS FOR TESTING
        efficiency_score = max(metrics_data.get('efficiency_score', 0), 10 + FRAME_COUNTER % 50)
        power_score = max(metrics_data.get('power_score', 0), 20 + FRAME_COUNTER % 30)
        
        print(f"🎯 FINAL METRICS: Efficiency={efficiency_score}, Power={power_score}")
        
        metrics = SwingMetrics(
            efficiency_score=int(efficiency_score),
            power_score=int(power_score),
            swing_speed=float(metrics_data.get('swing_speed', 1.0 + FRAME_COUNTER * 0.1)),
            path_consistency=int(metrics_data.get('path_consistency', 30)),
            follow_through=int(metrics_data.get('follow_through', 40)),
            pose_stability=int(metrics_data.get('pose_stability', 50)),
            sweet_spot_contact=bool(metrics_data.get('sweet_spot_contact', False)),
            impact_point=None
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
            message=f"🐍 PYTHON FRAME #{FRAME_COUNTER} PROCESSED! PID:{os.getpid()}",
            session_id=session_id
        )
        
    except Exception as e:
        print(f"🚨 PYTHON ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.post("/api/swing/start_tracking")
async def start_tracking(request: StartTrackingRequest):
    """Start tracking at specific coordinates"""
    try:
        session_id = request.session_id
        x = request.x
        y = request.y
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        session_tracker = session["tracker"]
        
        # Start tracking session
        session_tracker.start_tracking_session()
        session_tracker.update_current_position(int(x), int(y))
        
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
        session_tracker = session["tracker"]
        
        # Stop tracking and get final analysis
        success = session_tracker.stop_tracking_session()
        
        logger.info(f"Stopped tracking for session {session_id}, success: {success}")
        
        if success:
            final_metrics = session_tracker.get_current_metrics()
            
            # Convert metrics to response format
            impact_point = None
            if final_metrics.get('impact_point'):
                impact_point = [float(final_metrics['impact_point'][0]), float(final_metrics['impact_point'][1])]
            
            metrics = SwingMetrics(
                efficiency_score=int(final_metrics.get('efficiency_score', 0)),
                power_score=int(final_metrics.get('power_score', 0)),
                swing_speed=float(final_metrics.get('swing_speed', 0.0)),
                path_consistency=int(final_metrics.get('path_consistency', 0)),
                follow_through=int(final_metrics.get('follow_through', 0)),
                pose_stability=int(final_metrics.get('pose_stability', 0)),
                sweet_spot_contact=bool(final_metrics.get('sweet_spot_contact', False)),
                impact_point=impact_point
            )
            
            return SwingAnalysisResponse(
                success=True,
                metrics=metrics,
                swing_path=[],
                message="Swing analysis completed",
                session_id=session_id
            )
        else:
            return SwingAnalysisResponse(
                success=False,
                metrics=SwingMetrics(
                    efficiency_score=0, power_score=0, swing_speed=0.0,
                    path_consistency=0, follow_through=0, pose_stability=0,
                    sweet_spot_contact=False, impact_point=None
                ),
                swing_path=[],
                message="Insufficient swing data for analysis"
            )
        
    except Exception as e:
        logger.error(f"Error stopping tracking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test/force_swing")
async def force_swing_test(session_id: str):
    """Force a swing for testing"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    session_tracker = session["tracker"]
    
    # Force swing path points
    fake_swing_path = [
        (100, 300), (150, 280), (200, 260), (250, 240), (300, 220),
        (350, 200), (400, 180), (450, 160), (500, 140)
    ]
    
    for point in fake_swing_path:
        session_tracker.swing_path_points.append(point)
    
    # Force tracking state
    session_tracker.is_tracking = True
    
    return {
        "success": True,
        "message": "Forced swing data added",
        "swing_points": len(session_tracker.swing_path_points)
    }

@app.get("/api/session/{session_id}/results")
async def get_session_results(session_id: str):
    """Get complete session results"""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        
        # Calculate summary statistics
        swing_data = session["swing_data"]
        if not swing_data:
            return {
                "session_id": session_id,
                "status": session["status"],
                "summary": {"total_swings": 0},
                "swings": []
            }
        
        # Calculate averages
        total_swings = len(swing_data)
        avg_efficiency = sum(s["metrics"]["efficiency_score"] for s in swing_data) / total_swings
        avg_power = sum(s["metrics"]["power_score"] for s in swing_data) / total_swings
        max_efficiency = max(s["metrics"]["efficiency_score"] for s in swing_data)
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "started_at": session["started_at"],
            "summary": {
                "total_swings": total_swings,
                "avg_efficiency": round(avg_efficiency, 1),
                "avg_power": round(avg_power, 1),
                "max_efficiency": max_efficiency
            },
            "swings": swing_data[-10:]  # Return last 10 swings
        }
        
    except Exception as e:
        logger.error(f"Error getting session results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time swing analysis"""
    await websocket.accept()
    
    try:
        if session_id not in active_sessions:
            await websocket.send_json({"error": "Session not found"})
            await websocket.close()
            return
        
        session = active_sessions[session_id]
        session_tracker = session["tracker"]
        
        logger.info(f"WebSocket connected for session: {session_id}")
        
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Decode and process frame
            frame_bytes = base64.b64decode(frame_data["frame"])
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Process frame
                results = session_tracker.process_frame(frame)
                
                # Send results back
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": results.get('metrics', {}),
                    "swing_path_length": len(results.get('swing_path', [])),
                    "tracking_active": session_tracker.is_tracking
                }
                
                await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})

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
                "user_id": session["user_id"]
            }
            for sid, session in active_sessions.items()
        ]
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

