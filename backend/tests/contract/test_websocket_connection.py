"""Contract test for WebSocket connection."""
import pytest
import asyncio
import json
from websockets import connect
import uuid


@pytest.mark.contract
@pytest.mark.asyncio
class TestWebSocketConnection:
    """Test WebSocket connection contract."""
    
    async def test_websocket_connection_established(self, websocket_url: str):
        """Test establishing WebSocket connection."""
        async with connect(websocket_url) as websocket:
            assert websocket.open
            await websocket.close()
    
    async def test_websocket_start_session_message(self, websocket_url: str):
        """Test sending start session message via WebSocket."""
        async with connect(websocket_url) as websocket:
            # Send start session message
            message = {
                "type": "start_session",
                "mode": "automatic",
                "camera_config": {
                    "source": "webcam",
                    "resolution": "640x480",
                    "fps": 30
                }
            }
            await websocket.send(json.dumps(message))
            
            # Receive response
            response = await websocket.recv()
            data = json.loads(response)
            
            # Verify session created response
            assert data["type"] == "session_created"
            assert "session_id" in data
            assert data["mode"] == "automatic"
            assert data["status"] == "initializing"
            
            # Verify UUID format
            assert uuid.UUID(data["session_id"])
    
    async def test_websocket_stop_session_message(self, websocket_url: str):
        """Test sending stop session message via WebSocket."""
        async with connect(websocket_url) as websocket:
            # First create a session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Then stop it
            stop_message = {
                "type": "stop_session",
                "session_id": session_id
            }
            await websocket.send(json.dumps(stop_message))
            
            # Receive confirmation
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "session_stopped"
            assert data["session_id"] == session_id
    
    async def test_websocket_invalid_message_type(self, websocket_url: str):
        """Test sending invalid message type."""
        async with connect(websocket_url) as websocket:
            # Send invalid message type
            message = {
                "type": "invalid_type",
                "data": "test"
            }
            await websocket.send(json.dumps(message))
            
            # Should receive error response
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "error"
            assert "error_code" in data
            assert "message" in data
    
    async def test_websocket_mode_switch_message(self, websocket_url: str):
        """Test switching mode via WebSocket."""
        async with connect(websocket_url) as websocket:
            # Create session in automatic mode
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Switch to manual mode
            switch_message = {
                "type": "switch_mode",
                "session_id": session_id,
                "mode": "manual"
            }
            await websocket.send(json.dumps(switch_message))
            
            # Receive confirmation
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "mode_switched"
            assert data["session_id"] == session_id
            assert data["mode"] == "manual"
    
    async def test_websocket_multiple_connections(self, websocket_url: str):
        """Test multiple concurrent WebSocket connections."""
        # Create multiple connections
        websockets = []
        for _ in range(3):
            ws = await connect(websocket_url)
            websockets.append(ws)
        
        # All should be open
        for ws in websockets:
            assert ws.open
        
        # Clean up
        for ws in websockets:
            await ws.close()