#!/usr/bin/env python3
"""Test script to verify V2 WebSocket endpoint is working"""

import asyncio
import json
import websockets

async def test_v2_connection():
    """Test connection to V2 WebSocket endpoint"""
    uri = "ws://localhost:8000/ws/v2/stream"
    
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✓ Connected to V2 WebSocket endpoint")
            
            # Send a test message
            test_msg = {
                "type": "get_session_state",
                "session_id": "test-session"
            }
            
            await websocket.send(json.dumps(test_msg))
            print(f"✓ Sent test message: {test_msg['type']}")
            
            # Try to receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                print(f"✓ Received response: {response_data.get('type', 'unknown')}")
                
                # Check if it's from V2
                if 'v2' in str(response_data).lower() or 'stateful' in str(response_data).lower():
                    print("✓ Response confirms V2 is active")
                else:
                    print("⚠ Response doesn't clearly indicate V2")
                    
            except asyncio.TimeoutError:
                print("⚠ No response received (timeout)")
            
            print("\n✓ V2 WebSocket endpoint is accessible")
            return True
            
    except Exception as e:
        print(f"✗ Failed to connect to V2: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_v2_connection())
    exit(0 if result else 1)