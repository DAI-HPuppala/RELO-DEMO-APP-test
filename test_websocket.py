#!/usr/bin/env python3
"""Test WebSocket connection and automatic mode"""

import asyncio
import websockets
import json
import time

async def test_websocket():
    """Test WebSocket connection and automatic mode"""
    uri = "ws://localhost:8000/ws/stream"
    
    async with websockets.connect(uri) as websocket:
        print("✅ Connected to WebSocket")
        
        # First, start a session
        print("\n1. Starting session...")
        start_session = {
            "type": "start_session",
            "mode": "automatic",
            "camera_index": 0
        }
        await websocket.send(json.dumps(start_session))
        response = await websocket.recv()
        session_data = json.loads(response)
        print(f"Session response: {json.dumps(session_data, indent=2)}")
        
        if "session_id" not in session_data:
            print("❌ Failed to create session")
            return
        
        session_id = session_data["session_id"]
        print(f"✅ Session created: {session_id}")
        
        # Now try to start automatic mode
        print("\n2. Starting automatic mode...")
        start_auto = {
            "type": "start_automatic",
            "session_id": session_id
        }
        await websocket.send(json.dumps(start_auto))
        
        # Listen for responses
        print("\n3. Listening for messages...")
        try:
            for i in range(10):  # Listen for 10 messages
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"Message {i+1}: {data.get('type', 'unknown')}")
                
                if data.get('type') == 'error':
                    print(f"  Error: {data.get('message', 'Unknown error')}")
                    print(f"  Full error: {json.dumps(data, indent=2)}")
                elif data.get('type') == 'status':
                    print(f"  Status: {data.get('message', '')}")
                elif data.get('type') == 'agent_result':
                    print(f"  Agent: {data.get('agent_name', '')} - Result received")
                    
        except asyncio.TimeoutError:
            print("⏱️ Timeout - no more messages")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n✅ Test complete")

if __name__ == "__main__":
    asyncio.run(test_websocket())
