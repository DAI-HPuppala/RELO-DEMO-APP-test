"""Contract test for GET /api/health endpoint."""
import pytest
from httpx import AsyncClient


@pytest.mark.contract
class TestHealth:
    """Test GET /api/health endpoint contract."""
    
    async def test_health_check_success(self, async_client: AsyncClient):
        """Test successful health check."""
        response = await async_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "gpu_available" in data
        assert "model_loaded" in data
        assert "camera_connected" in data
        assert "version" in data
        
        # Verify data types
        assert data["status"] == "healthy"
        assert isinstance(data["gpu_available"], bool)
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["camera_connected"], bool)
        assert isinstance(data["version"], str)
    
    async def test_health_check_headers(self, async_client: AsyncClient):
        """Test health check response headers."""
        response = await async_client.get("/api/health")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        # Health check should not be cached
        assert "no-cache" in response.headers.get("cache-control", "").lower()
    
    async def test_health_check_no_auth_required(self, async_client: AsyncClient):
        """Test health check doesn't require authentication."""
        # Send request without any auth headers
        response = await async_client.get(
            "/api/health",
            headers={}
        )
        
        assert response.status_code == 200