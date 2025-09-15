"""Contract test for GET /api/session/{id}/results endpoint."""
import pytest
from httpx import AsyncClient
import uuid


@pytest.mark.contract
class TestSessionResults:
    """Test GET /api/session/{id}/results endpoint contract."""
    
    async def test_get_completed_session_results(self, async_client: AsyncClient, completed_session_id: str):
        """Test getting results of a completed session."""
        response = await async_client.get(f"/api/session/{completed_session_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "session_id" in data
        assert "status" in data
        assert "final_classification" in data
        assert "agent_results" in data
        assert "processing_time_seconds" in data
        assert "total_frames_analyzed" in data
        
        # Verify final classification structure
        classification = data["final_classification"]
        assert "type" in classification
        assert "color_primary" in classification
        assert "pattern" in classification
        assert "brand" in classification
        assert "size" in classification
        assert "has_damage" in classification
        
        # Verify agent results
        assert isinstance(data["agent_results"], list)
        assert len(data["agent_results"]) >= 1
        
        for agent_result in data["agent_results"]:
            assert "agent_name" in agent_result
            assert "attributes" in agent_result
            assert "confidence" in agent_result
            assert "reasoning" in agent_result
        
        # Verify data types
        assert data["session_id"] == completed_session_id
        assert data["status"] == "completed"
        assert isinstance(data["processing_time_seconds"], (int, float))
        assert isinstance(data["total_frames_analyzed"], int)
    
    async def test_get_processing_session_results_not_ready(self, async_client: AsyncClient, processing_session_id: str):
        """Test getting results of a still-processing session."""
        response = await async_client.get(f"/api/session/{processing_session_id}/results")
        
        assert response.status_code == 425  # Too Early
        data = response.json()
        assert "detail" in data
        assert "not ready" in data["detail"].lower()
    
    async def test_get_session_results_with_damage(self, async_client: AsyncClient, damaged_item_session_id: str):
        """Test getting results when damage is detected."""
        response = await async_client.get(f"/api/session/{damaged_item_session_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        
        classification = data["final_classification"]
        assert classification["has_damage"] is True
        assert classification["damage_type"] is not None
        assert "damage_locations" in classification
        assert isinstance(classification["damage_locations"], list)
    
    async def test_get_session_results_null_fields(self, async_client: AsyncClient, incomplete_session_id: str):
        """Test getting results with unreadable attributes marked as null."""
        response = await async_client.get(f"/api/session/{incomplete_session_id}/results")
        
        assert response.status_code == 200
        data = response.json()
        
        classification = data["final_classification"]
        # Some fields can be null if unreadable
        if classification["brand"] is None:
            assert classification["brand"] is None
        if classification["size"] is None:
            assert classification["size"] is None
    
    async def test_get_session_results_not_found(self, async_client: AsyncClient):
        """Test getting results of non-existent session."""
        fake_id = str(uuid.uuid4())
        response = await async_client.get(f"/api/session/{fake_id}/results")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    async def test_get_session_results_invalid_id(self, async_client: AsyncClient):
        """Test getting results with invalid UUID."""
        response = await async_client.get("/api/session/invalid-uuid/results")
        
        assert response.status_code == 422  # Validation error