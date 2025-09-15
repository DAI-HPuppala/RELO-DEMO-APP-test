"""Contract test for WebRTC signaling via WebSocket."""
import pytest
import json
from websockets import connect
import uuid


@pytest.mark.contract
@pytest.mark.asyncio
class TestWebRTCSignaling:
    """Test WebRTC signaling contract."""
    
    async def test_webrtc_offer_answer_exchange(self, websocket_url: str):
        """Test WebRTC offer/answer exchange."""
        async with connect(websocket_url) as websocket:
            # First create a session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Send WebRTC offer
            offer_message = {
                "type": "offer",
                "sdp": "v=0\r\no=- 123456 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n",
                "session_id": session_id
            }
            await websocket.send(json.dumps(offer_message))
            
            # Receive answer
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "answer"
            assert "sdp" in data
            assert data["session_id"] == session_id
            assert data["sdp"].startswith("v=0")
    
    async def test_ice_candidate_exchange(self, websocket_url: str):
        """Test ICE candidate exchange."""
        async with connect(websocket_url) as websocket:
            # Create session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Send ICE candidate
            ice_message = {
                "type": "ice_candidate",
                "candidate": {
                    "candidate": "candidate:1 1 UDP 2130706431 192.168.1.1 54321 typ host",
                    "sdpMLineIndex": 0,
                    "sdpMid": "0"
                },
                "session_id": session_id
            }
            await websocket.send(json.dumps(ice_message))
            
            # Should receive acknowledgment or server's ICE candidate
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] in ["ice_candidate", "ice_candidate_received"]
            if data["type"] == "ice_candidate":
                assert "candidate" in data
                assert data["session_id"] == session_id
    
    async def test_webrtc_offer_without_session(self, websocket_url: str):
        """Test sending WebRTC offer without active session."""
        async with connect(websocket_url) as websocket:
            # Send offer without creating session first
            offer_message = {
                "type": "offer",
                "sdp": "v=0\r\no=- 123456 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n",
                "session_id": str(uuid.uuid4())
            }
            await websocket.send(json.dumps(offer_message))
            
            # Should receive error
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "error"
            assert data["error_code"] == "SESSION_NOT_FOUND"
    
    async def test_webrtc_invalid_sdp_format(self, websocket_url: str):
        """Test sending invalid SDP format."""
        async with connect(websocket_url) as websocket:
            # Create session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Send invalid SDP
            offer_message = {
                "type": "offer",
                "sdp": "invalid_sdp_format",
                "session_id": session_id
            }
            await websocket.send(json.dumps(offer_message))
            
            # Should receive error
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "error"
            assert "invalid" in data["message"].lower()
    
    async def test_webrtc_data_channel_creation(self, websocket_url: str):
        """Test WebRTC data channel is created for results."""
        async with connect(websocket_url) as websocket:
            # Create session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Send offer with data channel
            offer_message = {
                "type": "offer",
                "sdp": "v=0\r\no=- 123456 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n",
                "session_id": session_id
            }
            await websocket.send(json.dumps(offer_message))
            
            # Receive answer
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "answer"
            # Answer should include data channel support
            assert "webrtc-datachannel" in data["sdp"] or "application" in data["sdp"]