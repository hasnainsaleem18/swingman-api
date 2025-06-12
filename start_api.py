#!/usr/bin/env python3
"""
Swingman API Startup Script
Easy startup script for the Swingman API server
"""

import subprocess
import sys
import os
import socket
import time
import requests
from pathlib import Path

def check_port_available(port):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'opencv-python',
        'ultralytics',
        'mediapipe',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_api.txt'
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def find_api_file():
    """Find the main API file"""
    possible_locations = [
        'main_api.py',
        'api/main_api.py',
        'src/main_api.py',
        'swingman_api.py'
    ]
    
    for location in possible_locations:
        if Path(location).exists():
            return location
    
    return None

def wait_for_server(host='localhost', port=8000, timeout=30):
    """Wait for the server to be ready"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/api/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        
        time.sleep(1)
    
    return False

def main():
    """Main startup function"""
    print("🚀 Starting Swingman API Server")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        
        if input("Install missing dependencies? (y/n): ").lower() == 'y':
            if install_dependencies():
                print("✅ Dependencies installed successfully")
            else:
                print("❌ Failed to install dependencies")
                return 1
        else:
            print("❌ Cannot start without required dependencies")
            return 1
    else:
        print("✅ All dependencies are installed")
    
    # Find API file
    api_file = find_api_file()
    if not api_file:
        print("❌ Could not find main_api.py file")
        print("Please ensure main_api.py is in the current directory")
        return 1
    
    print(f"✅ Found API file: {api_file}")
    
    # Check port availability
    port = 8000
    if not check_port_available(port):
        print(f"⚠️  Port {port} is already in use")
        
        # Try alternative ports
        for alt_port in [8001, 8002, 8003]:
            if check_port_available(alt_port):
                port = alt_port
                print(f"Using alternative port: {port}")
                break
        else:
            print("❌ No available ports found")
            return 1
    
    # Start the server
    print(f"Starting server on port {port}...")
    print("Server will be available at:")
    print(f"  Local:   http://localhost:{port}")
    print(f"  Network: http://0.0.0.0:{port}")
    print("\nAPI Documentation:")
    print(f"  Swagger UI: http://localhost:{port}/docs")
    print(f"  ReDoc:      http://localhost:{port}/redoc")
    print("\n" + "=" * 50)
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, '-m', 'uvicorn',
            f'{api_file.replace(".py", "").replace("/", ".")}:app',
            '--host', '0.0.0.0',
            '--port', str(port),
            '--reload',
            '--log-level', 'info'
        ]
        
        # Start server in background for health check
        import threading
        server_process = subprocess.Popen(cmd)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if wait_for_server('localhost', port, timeout=10):
            print("✅ Server is running successfully!")
            print("\nPress Ctrl+C to stop the server")
            
            # Keep the server running
            server_process.wait()
        else:
            print("❌ Server failed to start properly")
            server_process.terminate()
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)