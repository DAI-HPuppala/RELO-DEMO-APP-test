#!/usr/bin/env python3
"""Test WebRTC video streaming with RealSense camera."""

import asyncio
import json
import time
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

async def test_video_stream():
    print("Testing WebRTC video streaming...")
    print("-" * 40)
    
    # Connect to WebSocket
    uri = "ws://localhost:8000/ws/stream"
    async with websockets.connect(uri) as websocket:
        print("✓ WebSocket connected")
        
        # Start session
        await websocket.send(json.dumps({
            "type": "start_session",
            "mode": "automatic",
            "camera_config": {
                "source": "realsense"
            }
        }))
        
        # Get session response
        response = await websocket.recv()
        session_data = json.loads(response)
        session_id = session_data.get("session_id")
        print(f"✓ Session created: {session_id}")
        
        # Create peer connection
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        )
        pc = RTCPeerConnection(configuration=config)
        
        # Track if we receive video
        video_received = False
        frame_count = 0
        
        @pc.on("track")
        def on_track(track):
            nonlocal video_received, frame_count
            print(f"✓ Received track: {track.kind}")
            if track.kind == "video":
                video_received = True
                
                @track.on("frame")
                async def on_frame(frame):
                    nonlocal frame_count
                    frame_count += 1
                    if frame_count == 1:
                        print(f"✓ First video frame received!")
                    elif frame_count % 30 == 0:
                        print(f"  Frames received: {frame_count}")
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"  Connection state: {pc.connectionState}")
            if pc.connectionState == "connected":
                print("✓ WebRTC connected!")
        
        # Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        # Send offer
        await websocket.send(json.dumps({
            "type": "offer",
            "sdp": offer.sdp,
            "session_id": session_id
        }))
        
        # Get answer
        response = await websocket.recv()
        answer_data = json.loads(response)
        
        if answer_data.get("type") == "answer":
            print("✓ Received WebRTC answer")
            
            # Set remote description
            answer = RTCSessionDescription(
                sdp=answer_data["sdp"],
                type="answer"
            )
            await pc.setRemoteDescription(answer)
            print("✓ Remote description set")
            
            # Wait for connection and frames
            print("\n⏳ Waiting for video stream...")
            await asyncio.sleep(3)
            
            if pc.connectionState == "connected":
                print("\n✅ WebRTC connection established!")
                
                if video_received:
                    print(f"✅ Video streaming working! Received {frame_count} frames")
                else:
                    print("⚠️  Connection established but no video track received")
            else:
                print(f"⚠️  Connection state: {pc.connectionState}")
                
            # Keep connection alive for a bit to test stability
            print("\n⏳ Testing connection stability...")
            await asyncio.sleep(2)
            
            if pc.connectionState == "connected":
                print("✅ Connection stable!")
            else:
                print(f"⚠️  Connection dropped to: {pc.connectionState}")
                
        else:
            print(f"❌ Unexpected response: {answer_data}")
        
        # Clean up
        await pc.close()
        print("\n✓ Test completed")

if __name__ == "__main__":
    try:
        asyncio.run(test_video_stream())
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"❌ Test failed: {e}")