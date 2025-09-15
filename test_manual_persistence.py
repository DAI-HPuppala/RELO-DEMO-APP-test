#!/usr/bin/env python3
"""Test script to verify manual mode session persistence fix."""

import asyncio
import json
import logging
from websockets import connect
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_manual_mode_persistence():
    """Test that manual mode sessions persist correctly."""
    
    ws_url = "ws://localhost:8000/ws/stream"
    
    async with connect(ws_url) as websocket:
        # 1. Start a session
        session_id = str(uuid.uuid4())
        start_msg = {
            "type": "start_session",
            "mode": "manual",
            "camera_config": {"source": "webcam"}
        }
        
        await websocket.send(json.dumps(start_msg))
        response = json.loads(await websocket.recv())
        session_id = response.get("session_id")
        logger.info(f"Session started: {session_id}")
        
        # 2. Start manual mode 
        manual_start_msg = {
            "type": "start_automatic",
            "session_id": session_id,
            "manual_mode": True
        }
        
        await websocket.send(json.dumps(manual_start_msg))
        response = json.loads(await websocket.recv())
        logger.info(f"Manual mode started: {response}")
        
        # 3. Simulate manual navigation command
        await asyncio.sleep(2)
        
        manual_next_msg = {
            "type": "manual_next",
            "session_id": session_id
        }
        
        logger.info("Sending manual_next command...")
        await websocket.send(json.dumps(manual_next_msg))
        response = json.loads(await websocket.recv())
        logger.info(f"Manual next response: {response}")
        
        # 4. Stop the session (simulating disconnect)
        stop_msg = {
            "type": "stop_session",
            "session_id": session_id
        }
        
        await websocket.send(json.dumps(stop_msg))
        response = json.loads(await websocket.recv())
        logger.info(f"Session stopped: {response}")
        
        # 5. Try to send another manual command (should recreate handler)
        await asyncio.sleep(1)
        
        manual_next_msg2 = {
            "type": "manual_next",
            "session_id": session_id
        }
        
        logger.info("Sending manual_next after stop (testing persistence)...")
        await websocket.send(json.dumps(manual_next_msg2))
        response = json.loads(await websocket.recv())
        logger.info(f"Manual next after stop response: {response}")
        
        # Check if we got an error or successful response
        if response.get("type") == "error":
            logger.error(f"❌ Test failed: {response.get('message')}")
            return False
        else:
            logger.info("✅ Test passed: Manual mode persisted correctly!")
            return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_manual_mode_persistence())
        if result:
            print("\n✅ Manual mode persistence test PASSED")
        else:
            print("\n❌ Manual mode persistence test FAILED")
    except Exception as e:
        logger.error(f"Test error: {e}")
        print("\n❌ Test failed with exception")