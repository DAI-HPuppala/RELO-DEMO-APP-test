#!/usr/bin/env python3
"""Test the application's camera initialization logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

import asyncio
import logging
import time

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

async def test_app_camera():
    """Test the application's camera initialization."""
    print("=" * 60)
    print("Testing Application Camera Initialization")
    print("=" * 60)
    
    # Import the RealSense camera service
    from services.realsense_camera import RealSenseVideoTrack
    
    print("\n1. Creating RealSenseVideoTrack...")
    track = RealSenseVideoTrack(session_id="test_session")
    
    print(f"\n2. Camera initialized: {track.is_initialized}")
    print(f"   Pipeline: {track.pipeline is not None}")
    
    if not track.is_initialized:
        print("\n   ERROR: Camera failed to initialize!")
        print("   Application will fall back to test pattern")
        return False
    
    print("\n3. Testing frame capture...")
    for i in range(5):
        try:
            frame = await track.recv()
            print(f"   Frame {i+1}: {frame.width}x{frame.height}, pts={frame.pts}")
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"   Frame {i+1}: ERROR - {e}")
    
    print("\n4. Stopping track...")
    track.stop()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_app_camera())
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Application camera working!")
    else:
        print("FAILURE: Application falls back to test mode")
    print("=" * 60)