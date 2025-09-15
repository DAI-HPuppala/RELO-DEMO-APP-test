"""Contract test for GET /api/session/{id}/status endpoint."""
import pytest
from httpx import AsyncClient
import uuid


@pytest.mark.contract
class TestSessionStatus:
    """Test GET /api/session/{id}/status endpoint contract."""
    
    async def test_get_session_status_active(self, async_client: AsyncClient, active_session_id: str):
        """Test getting status of an active session."""
        response = await async_client.get(f"/api/session/{active_session_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "session_id" in data
        assert "status" in data
        assert "current_agent" in data
        assert "progress_percentage" in data
        assert "agents_completed" in data
        assert "timer_remaining" in data
        assert "frames_analyzed" in data
        
        # Verify data types
        assert data["session_id"] == active_session_id
        assert data["status"] in ["initializing", "processing", "completed", "error", "interrupted"]
        assert isinstance(data["progress_percentage"], int)
        assert 0 <= data["progress_percentage"] <= 100
        assert isinstance(data["agents_completed"], list)
        assert isinstance(data["frames_analyzed"], int)
    
    async def test_get_session_status_processing(self, async_client: AsyncClient, processing_session_id: str):
        """Test getting status of a processing session."""
        response = await async_client.get(f"/api/session/{processing_session_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "processing"
        assert data["current_agent"] is not None
        assert data["current_agent"] in ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]
    
    async def test_get_session_status_completed(self, async_client: AsyncClient, completed_session_id: str):
        """Test getting status of a completed session."""
        response = await async_client.get(f"/api/session/{completed_session_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "completed"
        assert data["progress_percentage"] == 100
        assert len(data["agents_completed"]) == 4
    
    async def test_get_session_status_not_found(self, async_client: AsyncClient):
        """Test getting status of non-existent session."""
        fake_id = str(uuid.uuid4())
        response = await async_client.get(f"/api/session/{fake_id}/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    async def test_get_session_status_invalid_id(self, async_client: AsyncClient):
        """Test getting status with invalid UUID."""
        response = await async_client.get("/api/session/invalid-uuid/status")
        
        assert response.status_code == 422  # Validation error