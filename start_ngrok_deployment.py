
#!/usr/bin/env python3
"""
Script to start FastAPI server and ngrok tunnel for remote access
"""

import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path

def check_ngrok_installed():
    """Check if ngrok is installed"""
    try:
        result = subprocess.run(['ngrok', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ ngrok is installed: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ ngrok is not installed or not in PATH")
        print("Please download ngrok from: https://ngrok.com/download")
        return False

def check_port_available(port):
    """Check if port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
        return True
    except OSError:
        return False

def start_fastapi_server():
    """Start FastAPI server"""
    print("🚀 Starting FastAPI server...")
    
    # Check if port 8000 is available
    if not check_port_available(8000):
        print("❌ Port 8000 is already in use. Please stop the service using it.")
        return None
    
    # Start FastAPI server
    process = subprocess.Popen([
        sys.executable, 'main_api.py'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Check if server is running
    try:
        response = requests.get('http://127.0.0.1:8000/', timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running on http://127.0.0.1:8000")
            return process
    except requests.exceptions.RequestException:
        print("❌ Failed to start FastAPI server")
        return None

def start_ngrok_tunnel():
    """Start ngrok tunnel"""
    print("🌐 Starting ngrok tunnel...")
    
    try:
        # Start ngrok
        process = subprocess.Popen([
            'ngrok', 'http', '8000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for ngrok to start
        time.sleep(5)
        
        # Get ngrok URL
        try:
            response = requests.get('http://127.0.0.1:4040/api/tunnels', timeout=5)
            tunnels = response.json()
            public_url = tunnels['tunnels'][0]['public_url']
            print(f"✅ Ngrok tunnel started: {public_url}")
            return process, public_url
        except (requests.exceptions.RequestException, KeyError, IndexError):
            print("❌ Failed to get ngrok URL")
            return None, None
            
    except subprocess.SubprocessError as e:
        print(f"❌ Failed to start ngrok: {e}")
        return None, None

def test_api_connection(public_url):
    """Test API connection through ngrok"""
    print("\n🧪 Testing API connection...")
    
    try:
        # Test root endpoint
        response = requests.get(f"{public_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint is working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint returned status: {response.status_code}")
        
        # Test CORS headers
        cors_response = requests.options(f"{public_url}/", 
                                       headers={
                                           "Origin": "http://localhost:3000",
                                           "Access-Control-Request-Method": "POST",
                                           "Access-Control-Request-Headers": "Content-Type"
                                       }, timeout=10)
        
        if 'access-control-allow-origin' in cors_response.headers:
            print("✅ CORS headers are properly configured")
            print(f"   CORS Headers: {dict(cors_response.headers)}")
        else:
            print("❌ CORS headers are missing")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to test API connection: {e}")

def cleanup(signum, frame, processes):
    """Clean up processes on exit"""
    print("\n🛑 Shutting down services...")
    
    # Terminate all processes
    for process in processes:
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    print("✅ All services stopped")

def main():
    """Main function to start the deployment"""
    print("🚀 Starting ngrok deployment for 3D Pose Estimation API")
    print("=" * 60)
    
    # Check prerequisites
    if not check_ngrok_installed():
        return
    
    # Start FastAPI server
    fastapi_process = start_fastapi_server()
    if not fastapi_process:
        return
    
    # Start ngrok tunnel
    ngrok_process, public_url = start_ngrok_tunnel()
    if not ngrok_process or not public_url:
        fastapi_process.terminate()
        return
    
    # Test API connection
    test_api_connection(public_url)
    
    print("\n🎉 Deployment completed successfully!")
    print(f"📖 API Documentation: {public_url}/docs")
    print(f"🔗 API Root: {public_url}/")
    print(f"📱 Pose Comparison: {public_url}/api/compare_pose")
    print("\nPress Ctrl+C to stop all services")
    
    # Set up signal handlers for cleanup
    processes = [fastapi_process, ngrok_process]
    signal.signal(signal.SIGINT, lambda s, f: cleanup(s, f, processes))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup(s, f, processes))
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(None, None, processes)

if __name__ == "__main__":
    main()