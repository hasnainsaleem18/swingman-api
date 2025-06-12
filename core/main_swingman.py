"""
Main Swingman Enhanced Tracker
Run this file to start the complete system
"""

import cv2
import sys
import os

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core import create_enhanced_tracker

def main():
    print("🚀 Swingman Enhanced Tracker")
    print("=" * 50)
    
    # Configure your model path
    custom_bat_model_path = "Models/swingman_bat_detector.pt"
    
    # Check if model exists
    if not os.path.exists(custom_bat_model_path):
        print(f"⚠️  Custom bat model not found: {custom_bat_model_path}")
        print("   Using fallback detection...")
        custom_bat_model_path = None
    else:
        print(f"✅ Using custom model: {custom_bat_model_path}")
    
    # Create tracker
    try:
        tracker = create_enhanced_tracker(
            custom_bat_model_path=custom_bat_model_path,
            enable_pose=True
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Install dependencies: pip install ultralytics mediapipe")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n🎯 Controls:")
    print("  SPACE - Start/Stop tracking")
    print("  'y' - Toggle YOLO detections")
    print("  'p' - Toggle pose overlay")
    print("  'a' - Toggle analysis overlay")
    print("  'h' - Toggle heatmap")
    print("  'r' - Reset swing")
    print("  's' - Save session")
    print("  'q' - Quit")
    print("=" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            analyzed_frame = tracker.process_frame(frame)
            
            # Status indicator
            status = "🔴 TRACKING" if tracker.is_tracking else "⚪ READY"
            cv2.putText(analyzed_frame, status, (10, analyzed_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255) if tracker.is_tracking else (255, 255, 255), 2)
            
            cv2.imshow("Swingman Enhanced Tracker", analyzed_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if tracker.is_tracking:
                    tracker.stop_tracking_session()
                else:
                    tracker.start_tracking_session()
            elif key == ord('y'):
                tracker.toggle_visualization("yolo")
            elif key == ord('p'):
                tracker.toggle_visualization("pose")
            elif key == ord('a'):
                tracker.toggle_visualization("analysis")
            elif key == ord('h'):
                tracker.toggle_visualization("heatmap")
            elif key == ord('r'):
                tracker.clear_current_swing()
            elif key == ord('s'):
                tracker.save_swing_session()
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    
    finally:
        if tracker.is_tracking:
            tracker.stop_tracking_session()
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Finished!")

if __name__ == "__main__":
    main()
