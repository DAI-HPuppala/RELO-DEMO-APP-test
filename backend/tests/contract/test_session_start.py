"""Contract test for POST /api/session/start endpoint."""
import pytest
from httpx import AsyncClient
import uuid


@pytest.mark.contract
class TestSessionStart:
    """Test POST /api/session/start endpoint contract."""
    
    async def test_create_session_automatic_mode(self, async_client: AsyncClient):
        """Test creating a session in automatic mode."""
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "automatic"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "session_id" in data
        assert "status" in data
        assert "mode" in data
        assert "websocket_url" in data
        
        # Verify UUID format
        assert uuid.UUID(data["session_id"])
        
        # Verify values
        assert data["status"] == "initializing"
        assert data["mode"] == "automatic"
        assert data["websocket_url"] == "ws://localhost:8000/ws/stream"
    
    async def test_create_session_manual_mode(self, async_client: AsyncClient):
        """Test creating a session in manual mode."""
        response = await async_client.post(
            "/api/session/start",
            json={
                "mode": "manual",
                "camera_source": "realsense"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "manual"
    
    async def test_create_session_with_custom_timers(self, async_client: AsyncClient):
        """Test creating a session with custom agent timers."""
        response = await async_client.post(
            "/api/session/start",
            json={
                "mode": "automatic",
                "agent_timers": {
                    "initial_classifier": 5,
                    "detail_extractor": 4,
                    "damage_detector": 6
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    async def test_create_session_missing_mode(self, async_client: AsyncClient):
        """Test creating a session without required mode field."""
        response = await async_client.post(
            "/api/session/start",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
    
    async def test_create_session_invalid_mode(self, async_client: AsyncClient):
        """Test creating a session with invalid mode."""
        response = await async_client.post(
            "/api/session/start",
            json={"mode": "invalid"}
        )
        
        assert response.status_code == 422  # Validation error
    
    async def test_create_session_invalid_camera_source(self, async_client: AsyncClient):
        """Test creating a session with invalid camera source."""
        response = await async_client.post(
            "/api/session/start",
            json={
                "mode": "automatic",
                "camera_source": "invalid_camera"
            }
        )
        
        assert response.status_code == 422  # Validation error