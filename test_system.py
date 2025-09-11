#!/usr/bin/env python3
"""
Quick system test script for the Returns Classifier.
Run this to test the frontend-backend integration with RealSense camera.
"""

import asyncio
import sys
import os

# Add backend src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def test_backend():
    """Test backend server startup."""
    print("Testing backend server...")
    
    try:
        # Import the FastAPI app
        from api.main import app
        print("✓ Backend modules loaded successfully")
        
        # Check if Ollama is available
        import subprocess
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                print("✓ Ollama is available")
            else:
                print("⚠ Ollama not running (model inference will not work)")
        except:
            print("⚠ Ollama not installed (model inference will not work)")
        
        # Check RealSense availability
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                print(f"✓ RealSense camera detected: {devices[0].get_info(rs.camera_info.name)}")
            else:
                print("⚠ No RealSense camera detected (will fallback to webcam)")
        except ImportError:
            print("⚠ pyrealsense2 not installed (will use webcam)")
        except:
            print("⚠ RealSense not available (will use webcam)")
        
        # Check webcam availability
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Webcam available as fallback")
            cap.release()
        else:
            print("⚠ No webcam detected")
        
        return True
        
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        return False


def test_frontend():
    """Test frontend files exist."""
    print("\nTesting frontend...")
    
    frontend_files = [
        'frontend/src/index.html',
        'frontend/src/styles.css',
        'frontend/src/app.js',
        'frontend/src/services/webrtc-client.js',
        'frontend/src/services/session-manager.js',
        'frontend/src/components/video-display.js',
        'frontend/src/components/progress-panel.js',
        'frontend/src/components/control-panel.js'
    ]
    
    all_exist = True
    for file in frontend_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} not found")
            all_exist = False
    
    return all_exist


def print_instructions():
    """Print instructions for running the system."""
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Install dependencies:")
    print("   cd backend")
    print("   pip install -r requirements.txt")
    
    print("\n2. Start the backend server:")
    print("   python src/api/main.py")
    print("   (Server will run on http://localhost:8000)")
    
    print("\n3. Open the frontend in a browser:")
    print("   Open frontend/src/index.html in Chrome/Firefox")
    print("   Or serve it with: python -m http.server 8080 --directory frontend/src")
    print("   Then navigate to: http://localhost:8080")
    
    print("\n4. Using the application:")
    print("   - Click 'Start Monitoring' to begin")
    print("   - The system will attempt to use RealSense camera")
    print("   - If RealSense is not available, it will use webcam")
    print("   - Toggle 'Manual Mode' for operator control")
    print("   - Watch progressive results appear as agents process")
    
    print("\n" + "="*60)
    print("TROUBLESHOOTING")
    print("="*60)
    
    print("\nIf RealSense camera is not detected:")
    print("   - Ensure Intel RealSense SDK is installed")
    print("   - Check USB 3.0 connection")
    print("   - Run: rs-enumerate-devices (from RealSense SDK)")
    
    print("\nIf WebRTC connection fails:")
    print("   - Ensure backend is running on port 8000")
    print("   - Check browser console for errors")
    print("   - Try using localhost instead of file://")
    
    print("\nFor CORS issues:")
    print("   - Serve frontend via HTTP server (not file://)")
    print("   - Or disable CORS in browser for testing")


async def main():
    """Main test function."""
    print("Returns Classifier System Test")
    print("="*60)
    
    # Test backend
    backend_ok = await test_backend()
    
    # Test frontend
    frontend_ok = test_frontend()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    if backend_ok and frontend_ok:
        print("✓ All components ready!")
        print_instructions()
    else:
        print("✗ Some components missing or failed")
        print("\nPlease fix the issues above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())