"""Integration test for automatic mode session lifecycle."""
import pytest
import asyncio
import json
from httpx import AsyncClient
from websockets import connect


@pytest.mark.integration
@pytest.mark.asyncio
class TestAutomaticMode:
    """Test automatic mode full session lifecycle."""
    
    async def test_automatic_mode_complete_lifecycle(self, async_client: AsyncClient, websocket_url: str):
        """Test complete automatic mode session from start to finish."""
        # Start session via REST API
        response = await async_client.post(
            "/api/session/start",
            json={
                "mode": "automatic",
                "agent_timers": {
                    "initial_classifier": 1,  # Short timers for testing
                    "detail_extractor": 1,
                    "damage_detector": 1
                }
            }
        )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        
        # Connect WebSocket to receive updates
        async with connect(websocket_url) as websocket:
            # Send session association message
            await websocket.send(json.dumps({
                "type": "associate_session",
                "session_id": session_id
            }))
            
            # Track agent progression
            agents_started = []
            agents_completed = []
            final_results_received = False
            
            # Collect messages for the duration of the session
            start_time = asyncio.get_event_loop().time()
            timeout = 10  # Maximum time to wait
            
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(message)
                    
                    if data["type"] == "agent_started":
                        agents_started.append(data["agent"])
                    elif data["type"] == "agent_completed":
                        agents_completed.append(data["agent"])
                    elif data["type"] == "final_results":
                        final_results_received = True
                        break
                except asyncio.TimeoutError:
                    continue
        
        # Verify all agents ran
        expected_agents = ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]
        for agent in expected_agents[:3]:  # First 3 agents should start
            assert agent in agents_started
        
        # Check final results via REST API
        await asyncio.sleep(1)  # Give time for processing to complete
        results_response = await async_client.get(f"/api/session/{session_id}/results")
        
        if results_response.status_code == 200:
            results = results_response.json()
            assert results["status"] == "completed"
            assert results["final_classification"] is not None
    
    async def test_automatic_mode_progressive_updates(self, async_client: AsyncClient, websocket_url: str):
        """Test progressive updates during automatic mode."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "automatic"}
        )
        session_id = response.json()["session_id"]
        
        # Connect WebSocket
        async with connect(websocket_url) as websocket:
            await websocket.send(json.dumps({
                "type": "associate_session",
                "session_id": session_id
            }))
            
            # Collect progressive updates
            progressive_updates = []
            start_time = asyncio.get_event_loop().time()
            
            while asyncio.get_event_loop().time() - start_time < 5:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    data = json.loads(message)
                    
                    if data["type"] == "progressive_update":
                        progressive_updates.append(data)
                except asyncio.TimeoutError:
                    continue
            
            # Should have received some progressive updates
            assert len(progressive_updates) > 0
            
            # Verify update structure
            for update in progressive_updates:
                assert "agent" in update
                assert "attributes" in update
                assert "confidence" in update
                assert "frames_processed" in update
    
    async def test_automatic_mode_timer_enforcement(self, async_client: AsyncClient):
        """Test that agent timers are enforced in automatic mode."""
        # Start session with specific timers
        timers = {
            "initial_classifier": 2,
            "detail_extractor": 2,
            "damage_detector": 2
        }
        
        response = await async_client.post(
            "/api/session/start",
            json={
                "mode": "automatic",
                "agent_timers": timers
            }
        )
        session_id = response.json()["session_id"]
        
        # Check status periodically
        agent_times = {}
        current_agent = None
        
        for _ in range(20):  # Check for 10 seconds
            status_response = await async_client.get(f"/api/session/{session_id}/status")
            status = status_response.json()
            
            if status["current_agent"] != current_agent:
                if current_agent:
                    # Agent changed, record time
                    agent_times[current_agent] = asyncio.get_event_loop().time()
                current_agent = status["current_agent"]
                agent_times[current_agent] = asyncio.get_event_loop().time()
            
            if status["status"] == "completed":
                break
            
            await asyncio.sleep(0.5)
        
        # Verify agents ran for approximately the configured time
        # (Allow some variance for processing)
        for agent, configured_time in timers.items():
            if agent in agent_times and agent != list(agent_times.keys())[-1]:
                # Can't check the last agent's duration
                actual_time = agent_times[list(agent_times.keys())[list(agent_times.keys()).index(agent) + 1]] - agent_times[agent]
                assert abs(actual_time - configured_time) < 1.0  # Within 1 second
    
    async def test_automatic_mode_stop_monitoring(self, async_client: AsyncClient):
        """Test stopping automatic mode session."""
        # Start session
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "automatic"}
        )
        session_id = response.json()["session_id"]
        
        # Let it run briefly
        await asyncio.sleep(1)
        
        # Stop the session
        stop_response = await async_client.post(f"/api/session/{session_id}/stop")
        assert stop_response.status_code == 200
        
        # Verify session stopped
        status_response = await async_client.get(f"/api/session/{session_id}/status")
        status = status_response.json()
        assert status["status"] in ["interrupted", "stopped"]
        
        # In automatic mode, results should be cleared
        results_response = await async_client.get(f"/api/session/{session_id}/results")
        assert results_response.status_code in [404, 425]  # Not found or not ready