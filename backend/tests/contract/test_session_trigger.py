"""Contract test for POST /api/session/{id}/trigger endpoint."""
import pytest
from httpx import AsyncClient
import uuid


@pytest.mark.contract
class TestSessionTrigger:
    """Test POST /api/session/{id}/trigger endpoint contract."""
    
    async def test_trigger_next_agent_manual_mode(self, async_client: AsyncClient, manual_session_id: str):
        """Test triggering next agent in manual mode."""
        response = await async_client.post(
            f"/api/session/{manual_session_id}/trigger",
            json={}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "triggered_agent" in data
        assert "status" in data
        
        # Verify agent name
        assert data["triggered_agent"] in ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]
        assert data["status"] == "processing"
    
    async def test_trigger_specific_agent_manual_mode(self, async_client: AsyncClient, manual_session_id: str):
        """Test triggering specific agent in manual mode."""
        response = await async_client.post(
            f"/api/session/{manual_session_id}/trigger",
            json={"agent": "damage_detector"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["triggered_agent"] == "damage_detector"
        assert data["status"] == "processing"
    
    async def test_trigger_skip_current_agent(self, async_client: AsyncClient, manual_session_id: str):
        """Test skipping current agent and moving to next."""
        response = await async_client.post(
            f"/api/session/{manual_session_id}/trigger",
            json={"skip_current": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "triggered_agent" in data
        assert data["status"] == "processing"
    
    async def test_trigger_automatic_mode_error(self, async_client: AsyncClient, auto_session_id: str):
        """Test triggering agent in automatic mode returns error."""
        response = await async_client.post(
            f"/api/session/{auto_session_id}/trigger",
            json={}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "automatic mode" in data["detail"].lower()
    
    async def test_trigger_session_not_found(self, async_client: AsyncClient):
        """Test triggering agent for non-existent session."""
        fake_id = str(uuid.uuid4())
        response = await async_client.post(
            f"/api/session/{fake_id}/trigger",
            json={}
        )
        
        assert response.status_code == 404
    
    async def test_trigger_invalid_agent_name(self, async_client: AsyncClient, manual_session_id: str):
        """Test triggering with invalid agent name."""
        response = await async_client.post(
            f"/api/session/{manual_session_id}/trigger",
            json={"agent": "invalid_agent"}
        )
        
        assert response.status_code == 422  # Validation error
    
    async def test_trigger_completed_session_error(self, async_client: AsyncClient, completed_session_id: str):
        """Test triggering agent on completed session."""
        response = await async_client.post(
            f"/api/session/{completed_session_id}/trigger",
            json={}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "completed" in data["detail"].lower()