"""Integration test for session recovery in manual mode."""
import pytest
import asyncio
import json
from httpx import AsyncClient
from websockets import connect


@pytest.mark.integration
@pytest.mark.asyncio
class TestSessionRecovery:
    """Test session recovery functionality."""
    
    async def test_manual_mode_session_recovery(self, async_client: AsyncClient):
        """Test recovering an interrupted manual mode session."""
        # Start session in manual mode
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Trigger some agents
        await async_client.post(f"/api/session/{session_id}/trigger", json={})
        await asyncio.sleep(0.5)
        await async_client.post(f"/api/session/{session_id}/trigger", json={})
        
        # Simulate interruption by stopping
        await async_client.post(f"/api/session/{session_id}/stop")
        
        # Verify session is marked as resumable
        status_response = await async_client.get(f"/api/session/{session_id}/status")
        status = status_response.json()
        assert status["status"] == "interrupted"
        
        # Resume the session
        resume_response = await async_client.post(f"/api/session/{session_id}/resume")
        assert resume_response.status_code == 200
        resume_data = resume_response.json()
        
        assert resume_data["session_id"] == session_id
        assert resume_data["status"] == "processing"
        assert "last_checkpoint" in resume_data
        
        # Verify we can continue triggering agents
        trigger_response = await async_client.post(
            f"/api/session/{session_id}/trigger",
            json={}
        )
        assert trigger_response.status_code == 200
    
    async def test_automatic_mode_no_recovery(self, async_client: AsyncClient):
        """Test that automatic mode sessions cannot be recovered."""
        # Start session in automatic mode
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "automatic"}
        )
        session_id = response.json()["session_id"]
        
        # Let it run briefly
        await asyncio.sleep(1)
        
        # Stop the session
        await async_client.post(f"/api/session/{session_id}/stop")
        
        # Try to resume - should fail
        resume_response = await async_client.post(f"/api/session/{session_id}/resume")
        assert resume_response.status_code == 400
        
        error_data = resume_response.json()
        assert "not resumable" in error_data["detail"].lower()
    
    async def test_websocket_reconnection_recovery(self, async_client: AsyncClient, websocket_url: str):
        """Test WebSocket reconnection with session recovery."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # First WebSocket connection
        async with connect(websocket_url) as ws1:
            await ws1.send(json.dumps({
                "type": "associate_session",
                "session_id": session_id
            }))
            
            # Trigger an agent
            await async_client.post(f"/api/session/{session_id}/trigger", json={})
            
            # Receive some messages
            messages_1 = []
            for _ in range(3):
                try:
                    msg = await asyncio.wait_for(ws1.recv(), timeout=0.5)
                    messages_1.append(json.loads(msg))
                except asyncio.TimeoutError:
                    break
        
        # Simulate disconnection
        await asyncio.sleep(0.5)
        
        # Reconnect with new WebSocket
        async with connect(websocket_url) as ws2:
            # Send reconnection with session ID
            await ws2.send(json.dumps({
                "type": "reconnect",
                "session_id": session_id
            }))
            
            # Should receive session state
            response = await ws2.recv()
            data = json.loads(response)
            
            assert data["type"] in ["session_state", "reconnected"]
            if data["type"] == "session_state":
                assert data["session_id"] == session_id
                assert "agents_completed" in data
    
    async def test_session_checkpoint_persistence(self, async_client: AsyncClient):
        """Test that session checkpoints are persisted."""
        # Start session in manual mode
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Process some agents and track state
        agents_triggered = []
        for i in range(2):
            trigger_response = await async_client.post(
                f"/api/session/{session_id}/trigger",
                json={}
            )
            agents_triggered.append(trigger_response.json()["triggered_agent"])
            await asyncio.sleep(0.5)
        
        # Get current status before interruption
        status_before = await async_client.get(f"/api/session/{session_id}/status")
        agents_completed_before = status_before.json()["agents_completed"]
        
        # Interrupt session
        await async_client.post(f"/api/session/{session_id}/stop")
        
        # Resume session
        resume_response = await async_client.post(f"/api/session/{session_id}/resume")
        checkpoint = resume_response.json()["last_checkpoint"]
        
        # Verify checkpoint contains completed agents
        assert "agents_completed" in checkpoint
        assert checkpoint["agents_completed"] == agents_completed_before
    
    async def test_completed_session_no_recovery(self, async_client: AsyncClient):
        """Test that completed sessions cannot be resumed."""
        # Create and complete a manual session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Trigger all agents to complete
        agents = ["initial_classifier", "detail_extractor", "damage_detector"]
        for _ in agents:
            await async_client.post(f"/api/session/{session_id}/trigger", json={})
            await asyncio.sleep(0.3)
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Check if completed
        status_response = await async_client.get(f"/api/session/{session_id}/status")
        if status_response.json()["status"] == "completed":
            # Try to resume - should fail
            resume_response = await async_client.post(f"/api/session/{session_id}/resume")
            assert resume_response.status_code == 400
            
            error_data = resume_response.json()
            assert "completed" in error_data["detail"].lower()
    
    async def test_session_data_persistence_after_recovery(self, async_client: AsyncClient):
        """Test that session data is preserved after recovery."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Trigger first agent and wait for results
        await async_client.post(f"/api/session/{session_id}/trigger", json={})
        await asyncio.sleep(1)
        
        # Get status before interruption
        status_before = await async_client.get(f"/api/session/{session_id}/status")
        frames_before = status_before.json()["frames_analyzed"]
        
        # Interrupt and resume
        await async_client.post(f"/api/session/{session_id}/stop")
        await async_client.post(f"/api/session/{session_id}/resume")
        
        # Continue processing
        await async_client.post(f"/api/session/{session_id}/trigger", json={})
        await asyncio.sleep(1)
        
        # Get final status
        status_after = await async_client.get(f"/api/session/{session_id}/status")
        frames_after = status_after.json()["frames_analyzed"]
        
        # Frame count should be cumulative
        assert frames_after >= frames_before