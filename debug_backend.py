#!/usr/bin/env python3
"""
Debug script to test backend initialization.
"""

import sys
import os
from pathlib import Path

# Add backend src to path
backend_src = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_src))

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    # Test imports
    print("\n1. Testing imports...")
    from models import SessionMode, ClassificationSession
    print("✓ Models imported successfully")
    
    from services import SessionManager, WebRTCManager
    print("✓ Services imported successfully")
    
    # Test initialization
    print("\n2. Testing initialization...")
    session_manager = SessionManager()
    print("✓ SessionManager initialized")
    
    webrtc_manager = WebRTCManager()
    print("✓ WebRTCManager initialized")
    
    # Test session creation
    print("\n3. Testing session creation...")
    import asyncio
    
    async def test_create_session():
        session = await session_manager.create_session(
            mode=SessionMode.AUTOMATIC,
            camera_source="webcam"
        )
        return session
    
    session = asyncio.run(test_create_session())
    print(f"✓ Session created: {session.session_id}")
    
    print("\n✅ All tests passed! Backend should work correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()