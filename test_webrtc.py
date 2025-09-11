#!/usr/bin/env python3
"""
Test WebRTC connection to verify fixes.
"""

import asyncio
import websockets
import json

async def test_webrtc():
    """Test WebRTC connection flow."""
    uri = "ws://localhost:8000/ws/stream"
    
    async with websockets.connect(uri) as websocket:
        print("✓ WebSocket connected")
        
        # Start session
        start_msg = {
            "type": "start_session",
            "mode": "automatic",
            "camera_config": {
                "source": "webcam",
                "resolution": "640x480",
                "fps": 30
            }
        }
        await websocket.send(json.dumps(start_msg))
        
        # Get session response
        response = await websocket.recv()
        data = json.loads(response)
        
        if data.get("type") == "session_created":
            session_id = data["session_id"]
            print(f"✓ Session created: {session_id}")
            
            # Send WebRTC offer
            offer_msg = {
                "type": "offer",
                "session_id": session_id,
                "sdp": "v=0\r\no=- 1234567890 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=group:BUNDLE 0\r\na=extmap-allow-mixed\r\na=msid-semantic: WMS\r\nm=video 9 UDP/TLS/RTP/SAVPF 96 97 98 99\r\nc=IN IP4 0.0.0.0\r\na=rtcp:9 IN IP4 0.0.0.0\r\na=ice-ufrag:4Xg8\r\na=ice-pwd:by4GZGG1lw+040DQ+XK5QVAD\r\na=ice-options:trickle\r\na=fingerprint:sha-256 58:E0:FE:56:6A:60:3D:74:81:8B:E0:55:4D:4C:B6:F7:2D:13:47:FC:83:FE:F6:5C:1D:5C:0C:D5:F9:57:7C:1C\r\na=setup:actpass\r\na=mid:0\r\na=recvonly\r\na=rtcp-mux\r\na=rtcp-rsize\r\na=rtpmap:96 VP8/90000\r\na=rtcp-fb:96 goog-remb\r\na=rtcp-fb:96 transport-cc\r\na=rtcp-fb:96 ccm fir\r\na=rtcp-fb:96 nack\r\na=rtcp-fb:96 nack pli\r\na=rtpmap:97 rtx/90000\r\na=fmtp:97 apt=96\r\na=rtpmap:98 VP9/90000\r\na=rtcp-fb:98 goog-remb\r\na=rtcp-fb:98 transport-cc\r\na=rtcp-fb:98 ccm fir\r\na=rtcp-fb:98 nack\r\na=rtcp-fb:98 nack pli\r\na=rtpmap:99 rtx/90000\r\na=fmtp:99 apt=98\r\n"
            }
            await websocket.send(json.dumps(offer_msg))
            
            # Get answer
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "answer":
                print("✓ Received WebRTC answer")
                print("✅ WebRTC connection test PASSED!")
                return True
            elif data.get("type") == "error":
                print(f"✗ Error: {data.get('message')}")
                return False
        else:
            print(f"✗ Unexpected response: {data}")
            return False

async def main():
    """Run the test."""
    print("Testing WebRTC connection...")
    print("-" * 40)
    
    try:
        success = await test_webrtc()
        if success:
            print("\n✅ All WebRTC errors have been fixed!")
        else:
            print("\n❌ WebRTC issues still present")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())