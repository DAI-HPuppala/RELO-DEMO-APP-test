#!/usr/bin/env python3
"""
Quick integration test for frontend-backend WebRTC streaming.
"""

import asyncio
import httpx
import json
import sys
import os

# Add backend src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def test_backend_health():
    """Test backend health endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Backend health check: {data['status']}")
                print(f"  Camera available: {data['camera_available']}")
                return True
            else:
                print(f"✗ Backend health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to backend: {e}")
            print("  Make sure to run ./run.sh first")
            return False

async def test_session_creation():
    """Test creating a new session."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8000/api/session/start",
                json={
                    "mode": "automatic",
                    "camera_source": "webcam"  # Use webcam for testing
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Session created: {data['session_id']}")
                return data['session_id']
            else:
                print(f"✗ Session creation failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Cannot create session: {e}")
            return None

async def test_websocket():
    """Test WebSocket connection."""
    try:
        import websockets
        
        uri = "ws://localhost:8000/ws/stream"
        async with websockets.connect(uri) as websocket:
            # Send test message
            test_msg = json.dumps({"type": "associate_session", "session_id": "test"})
            await websocket.send(test_msg)
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print("✓ WebSocket connection successful")
                return True
            except asyncio.TimeoutError:
                print("✓ WebSocket connected (no response expected for associate)")
                return True
                
    except Exception as e:
        print(f"✗ WebSocket connection failed: {e}")
        return False

async def main():
    """Run integration tests."""
    print("="*60)
    print("Frontend-Backend Integration Test")
    print("="*60)
    
    # Test backend health
    print("\n1. Testing backend connectivity...")
    backend_ok = await test_backend_health()
    
    if not backend_ok:
        print("\nBackend is not running. Please run:")
        print("  ./run.sh")
        return False
    
    # Test session creation
    print("\n2. Testing session creation...")
    session_id = await test_session_creation()
    
    # Test WebSocket
    print("\n3. Testing WebSocket connection...")
    ws_ok = await test_websocket()
    
    # Summary
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    
    if backend_ok and session_id and ws_ok:
        print("✓ All integration tests passed!")
        print("\nYou can now:")
        print("1. Open http://localhost:8080 in your browser")
        print("2. Click 'Start Monitoring' to begin streaming")
        print("3. The camera feed should appear automatically")
        return True
    else:
        print("✗ Some tests failed")
        print("\nTroubleshooting:")
        print("1. Make sure backend is running: ./run.sh")
        print("2. Check logs for errors")
        print("3. Ensure camera permissions are granted")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)