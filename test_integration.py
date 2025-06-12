#!/usr/bin/env python3
"""
Swingman Integration Test Script
Tests the complete API integration and functionality
"""

import requests
import base64
import json
import time
import cv2
import numpy as np
from pathlib import Path
import sys

class SwingmanIntegrationTest:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.session_id = None
        
    def test_health_check(self):
        """Test basic health check endpoint"""
        print("🔍 Testing health check...")
        
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Components: {data.get('components')}")
                return True
            else:
                print(f"❌ Health check failed: Status {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_session_management(self):
        """Test session creation and management"""
        print("\n🔍 Testing session management...")
        
        # Start session
        try:
            response = requests.post(f"{self.api_url}/session/start", 
                                   json={"user_id": "test_user"})
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                print(f"✅ Session created: {self.session_id}")
            else:
                print(f"❌ Session creation failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Session creation failed: {e}")
            return False
        
        # Stop session
        try:
            response = requests.post(f"{self.api_url}/session/{self.session_id}/stop")
            
            if response.status_code == 200:
                print("✅ Session stopped successfully")
                return True
            else:
                print(f"❌ Session stop failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Session stop failed: {e}")
            return False
    
    def test_frame_analysis(self):
        """Test frame analysis with synthetic image"""
        print("\n🔍 Testing frame analysis...")
        
        # Create test session
        try:
            response = requests.post(f"{self.api_url}/session/start", 
                                   json={"user_id": "test_user"})
            if response.status_code == 200:
                self.session_id = response.json()["session_id"]
            else:
                print("❌ Could not create test session")
                return False
        except:
            print("❌ Could not create test session")
            return False
        
        # Create synthetic test image
        test_image = self.create_test_image()
        
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', test_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare frame data
        frame_data = {
            "frame_base64": image_base64,
            "session_id": self.session_id,
            "timestamp": time.time()
        }
        
        try:
            response = requests.post(f"{self.api_url}/analyze/frame", 
                                   json=frame_data,
                                   timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    metrics = data.get("metrics", {})
                    print("✅ Frame analysis successful")
                    print(f"   Efficiency Score: {metrics.get('efficiency_score', 0)}%")
                    print(f"   Power Score: {metrics.get('power_score', 0)}%")
                    print(f"   Swing Speed: {metrics.get('swing_speed', 0.0)}")
                    return True
                else:
                    print(f"❌ Frame analysis failed: {data.get('message')}")
                    return False
            else:
                print(f"❌ Frame analysis request failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Frame analysis failed: {e}")
            return False
    
    def test_tracking_workflow(self):
        """Test complete tracking workflow"""
        print("\n🔍 Testing tracking workflow...")
        
        # Create test session
        try:
            response = requests.post(f"{self.api_url}/session/start", 
                                   json={"user_id": "test_user"})
            if response.status_code == 200:
                self.session_id = response.json()["session_id"]
            else:
                print("❌ Could not create test session")
                return False
        except:
            print("❌ Could not create test session")
            return False
        
        # Start tracking
        try:
            response = requests.post(f"{self.api_url}/swing/start_tracking",
                                   json={
                                       "session_id": self.session_id,
                                       "x": 320,
                                       "y": 240
                                   })
            
            if response.status_code == 200:
                print("✅ Tracking started")
            else:
                print(f"❌ Tracking start failed: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"❌ Tracking start failed: {e}")
            return False
        
        # Simulate some tracking time
        time.sleep(1)
        
        # Stop tracking
        try:
            response = requests.post(f"{self.api_url}/swing/stop_tracking",
                                   json={"session_id": self.session_id})
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    metrics = data.get("metrics", {})
                    print("✅ Tracking workflow completed")
                    print(f"   Final Efficiency: {metrics.get('efficiency_score', 0)}%")
                    return True
                else:
                    print(f"❌ Tracking completion failed: {data.get('message')}")
                    return False
            else:
                print(f"❌ Tracking stop failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Tracking stop failed: {e}")
            return False
    
    def test_session_results(self):
        """Test session results retrieval"""
        print("\n🔍 Testing session results...")
        
        if not self.session_id:
            print("❌ No active session for results test")
            return False
        
        try:
            response = requests.get(f"{self.api_url}/session/{self.session_id}/results")
            
            if response.status_code == 200:
                data = response.json()
                summary = data.get("summary", {})
                print("✅ Session results retrieved")
                print(f"   Total Swings: {summary.get('total_swings', 0)}")
                print(f"   Avg Efficiency: {summary.get('avg_efficiency', 0.0)}%")
                return True
            else:
                print(f"❌ Results retrieval failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Results retrieval failed: {e}")
            return False
    
    def create_test_image(self):
        """Create a synthetic test image for analysis"""
        # Create a 640x480 test image with some basic shapes
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add background
        image[:] = (50, 50, 50)  # Dark gray background
        
        # Add some shapes to simulate a scene
        # Rectangle (simulating a bat)
        cv2.rectangle(image, (200, 200), (400, 220), (139, 69, 19), -1)
        
        # Circle (simulating a ball)
        cv2.circle(image, (350, 300), 10, (255, 255, 255), -1)
        
        # Add some noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def run_all_tests(self):
        """Run complete integration test suite"""
        print("🚀 Starting Swingman Integration Tests")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Session Management", self.test_session_management),
            ("Frame Analysis", self.test_frame_analysis),
            ("Tracking Workflow", self.test_tracking_workflow),
            ("Session Results", self.test_session_results)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                time.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
        
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed_tests}/{total_tests} passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! Integration is working correctly.")
            return True
        else:
            print(f"⚠️  {total_tests - passed_tests} test(s) failed. Check server setup.")
            return False

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Swingman Integration Test")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the Swingman API server")
    parser.add_argument("--quick", action="store_true",
                       help="Run only basic health check")
    
    args = parser.parse_args()
    
    # Create test instance
    tester = SwingmanIntegrationTest(args.url)
    
    if args.quick:
        # Quick test mode
        print("🔍 Running quick health check...")
        if tester.test_health_check():
            print("✅ Quick test passed!")
            return 0
        else:
            print("❌ Quick test failed!")
            return 1
    else:
        # Full test suite
        if tester.run_all_tests():
            return 0
        else:
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)