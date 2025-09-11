"""Integration test for manual mode with triggers."""
import pytest
import asyncio
import json
from httpx import AsyncClient
from websockets import connect


@pytest.mark.integration
@pytest.mark.asyncio
class TestManualMode:
    """Test manual mode with operator triggers."""
    
    async def test_manual_mode_trigger_sequence(self, async_client: AsyncClient):
        """Test manual triggering of agents in sequence."""
        # Start session in manual mode
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        
        # Define agent sequence
        agents = ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]
        
        for agent in agents[:3]:  # Trigger first 3 agents
            # Trigger agent
            trigger_response = await async_client.post(
                f"/api/session/{session_id}/trigger",
                json={}  # Trigger next agent
            )
            assert trigger_response.status_code == 200
            assert trigger_response.json()["triggered_agent"] == agent
            
            # Wait briefly for processing
            await asyncio.sleep(0.5)
            
            # Check status
            status_response = await async_client.get(f"/api/session/{session_id}/status")
            status = status_response.json()
            assert agent in status["agents_completed"] or status["current_agent"] == agent
        
        # Final compilation should happen automatically
        await asyncio.sleep(1)
        
        # Check final results
        results_response = await async_client.get(f"/api/session/{session_id}/results")
        if results_response.status_code == 200:
            results = results_response.json()
            assert results["final_classification"] is not None
    
    async def test_manual_mode_specific_agent_trigger(self, async_client: AsyncClient):
        """Test triggering specific agent out of order."""
        # Start session in manual mode
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Trigger damage detector directly (skipping others)
        trigger_response = await async_client.post(
            f"/api/session/{session_id}/trigger",
            json={"agent": "damage_detector"}
        )
        assert trigger_response.status_code == 200
        assert trigger_response.json()["triggered_agent"] == "damage_detector"
        
        # Verify agent is running
        await asyncio.sleep(0.5)
        status_response = await async_client.get(f"/api/session/{session_id}/status")
        status = status_response.json()
        assert status["current_agent"] == "damage_detector" or "damage_detector" in status["agents_completed"]
    
    async def test_manual_mode_skip_agent(self, async_client: AsyncClient):
        """Test skipping current agent in manual mode."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Trigger first agent
        trigger_response = await async_client.post(
            f"/api/session/{session_id}/trigger",
            json={}
        )
        first_agent = trigger_response.json()["triggered_agent"]
        
        # Skip current agent
        skip_response = await async_client.post(
            f"/api/session/{session_id}/trigger",
            json={"skip_current": True}
        )
        assert skip_response.status_code == 200
        second_agent = skip_response.json()["triggered_agent"]
        
        # Should have moved to next agent
        assert first_agent != second_agent
    
    async def test_manual_mode_with_websocket_updates(self, async_client: AsyncClient, websocket_url: str):
        """Test manual mode with WebSocket updates."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Connect WebSocket
        async with connect(websocket_url) as websocket:
            await websocket.send(json.dumps({
                "type": "associate_session",
                "session_id": session_id
            }))
            
            # Trigger agent via REST
            await async_client.post(
                f"/api/session/{session_id}/trigger",
                json={}
            )
            
            # Should receive agent_started via WebSocket
            agent_started = False
            for _ in range(10):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(message)
                    
                    if data["type"] == "agent_started":
                        agent_started = True
                        assert data["mode"] == "manual"
                        assert "agent" in data
                        break
                except asyncio.TimeoutError:
                    continue
            
            assert agent_started
    
    async def test_manual_mode_no_auto_progression(self, async_client: AsyncClient):
        """Test that agents don't auto-progress in manual mode."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Trigger first agent
        await async_client.post(
            f"/api/session/{session_id}/trigger",
            json={}
        )
        
        # Wait for what would be auto-progression time
        await asyncio.sleep(3)
        
        # Check status - should still be on first agent or waiting
        status_response = await async_client.get(f"/api/session/{session_id}/status")
        status = status_response.json()
        
        # Should have only completed the triggered agent
        assert len(status["agents_completed"]) <= 1
        
        # Should not have auto-progressed to final results
        results_response = await async_client.get(f"/api/session/{session_id}/results")
        assert results_response.status_code == 425  # Not ready
    
    async def test_manual_mode_switch_to_automatic(self, async_client: AsyncClient, websocket_url: str):
        """Test switching from manual to automatic mode."""
        # Start in manual mode
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "manual"}
        )
        session_id = response.json()["session_id"]
        
        # Switch to automatic mode via WebSocket
        async with connect(websocket_url) as websocket:
            switch_message = {
                "type": "switch_mode",
                "session_id": session_id,
                "mode": "automatic"
            }
            await websocket.send(json.dumps(switch_message))
            
            # Confirm mode switch
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "mode_switched"
            assert data["mode"] == "automatic"
        
        # Agents should now auto-progress
        await asyncio.sleep(5)
        
        # Check that progression happened
        status_response = await async_client.get(f"/api/session/{session_id}/status")
        status = status_response.json()
        assert len(status["agents_completed"]) > 0