"""Contract test for WebSocket message flow."""
import pytest
import json
import asyncio
from websockets import connect


@pytest.mark.contract
@pytest.mark.asyncio
class TestWebSocketMessages:
    """Test WebSocket message flow contract."""
    
    async def test_progressive_update_messages(self, websocket_url: str):
        """Test receiving progressive update messages."""
        async with connect(websocket_url) as websocket:
            # Start session in automatic mode
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            
            # Collect messages for a few seconds
            messages = []
            try:
                while len(messages) < 5:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    messages.append(json.loads(message))
            except asyncio.TimeoutError:
                pass
            
            # Should have received session_created and some updates
            message_types = [msg["type"] for msg in messages]
            assert "session_created" in message_types
            
            # Check for expected message types
            expected_types = ["agent_started", "progressive_update", "status_update"]
            for msg in messages:
                if msg["type"] == "progressive_update":
                    assert "session_id" in msg
                    assert "agent" in msg
                    assert "attributes" in msg
                    assert "timestamp" in msg
    
    async def test_agent_status_messages(self, websocket_url: str):
        """Test agent status update messages."""
        async with connect(websocket_url) as websocket:
            # Start session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Wait for agent started message
            agent_started = False
            for _ in range(10):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data["type"] == "agent_started":
                        agent_started = True
                        assert data["session_id"] == session_id
                        assert "agent" in data
                        assert "timer_seconds" in data
                        assert data["mode"] == "automatic"
                        break
                except asyncio.TimeoutError:
                    continue
            
            assert agent_started, "Did not receive agent_started message"
    
    async def test_status_update_messages(self, websocket_url: str):
        """Test system status update messages."""
        async with connect(websocket_url) as websocket:
            # Start session
            start_message = {
                "type": "start_session",
                "mode": "automatic"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Wait for status update
            status_received = False
            for _ in range(10):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data["type"] == "status_update":
                        status_received = True
                        assert data["session_id"] == session_id
                        assert "current_agent" in data
                        assert "progress" in data
                        assert "fps" in data
                        assert "connection_quality" in data
                        assert 0 <= data["progress"] <= 100
                        break
                except asyncio.TimeoutError:
                    continue
            
            assert status_received, "Did not receive status_update message"
    
    async def test_error_message_format(self, websocket_url: str):
        """Test error message format."""
        async with connect(websocket_url) as websocket:
            # Send malformed message
            await websocket.send("invalid json")
            
            # Should receive error message
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "error"
            assert "error_code" in data
            assert "message" in data
            assert "recoverable" in data
            assert isinstance(data["recoverable"], bool)
    
    async def test_manual_trigger_response(self, websocket_url: str):
        """Test manual trigger message in manual mode."""
        async with connect(websocket_url) as websocket:
            # Start session in manual mode
            start_message = {
                "type": "start_session",
                "mode": "manual"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Send manual trigger
            trigger_message = {
                "type": "manual_trigger",
                "session_id": session_id,
                "agent": "initial_classifier"
            }
            await websocket.send(json.dumps(trigger_message))
            
            # Should receive agent started message
            triggered = False
            for _ in range(5):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data["type"] == "agent_started":
                        triggered = True
                        assert data["agent"] == "initial_classifier"
                        assert data["mode"] == "manual"
                        break
                except asyncio.TimeoutError:
                    continue
            
            assert triggered, "Agent was not triggered"
    
    async def test_export_ready_message(self, websocket_url: str):
        """Test export ready message."""
        async with connect(websocket_url) as websocket:
            # This would normally happen after session completes
            # For testing, we'll request export via WebSocket
            start_message = {
                "type": "start_session",
                "mode": "manual"
            }
            await websocket.send(json.dumps(start_message))
            response = await websocket.recv()
            session_id = json.loads(response)["session_id"]
            
            # Request export
            export_message = {
                "type": "export_results",
                "session_id": session_id,
                "format": "json"
            }
            await websocket.send(json.dumps(export_message))
            
            # Look for export ready or error (session not complete)
            response = await websocket.recv()
            data = json.loads(response)
            
            # Either export_ready or error is acceptable
            assert data["type"] in ["export_ready", "error"]
            if data["type"] == "export_ready":
                assert "download_url" in data
                assert "format" in data
                assert "size_bytes" in data