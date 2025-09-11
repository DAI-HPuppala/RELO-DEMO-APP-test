"""Contract test for POST /api/session/{id}/export endpoint."""
import pytest
from httpx import AsyncClient
import uuid


@pytest.mark.contract
class TestSessionExport:
    """Test POST /api/session/{id}/export endpoint contract."""
    
    async def test_export_completed_session_json(self, async_client: AsyncClient, completed_session_id: str):
        """Test exporting completed session results as JSON."""
        response = await async_client.post(
            f"/api/session/{completed_session_id}/export",
            json={"format": "json"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "download_url" in data
        assert "format" in data
        assert "size_bytes" in data
        assert "expires_at" in data
        
        # Verify values
        assert data["format"] == "json"
        assert data["download_url"] == f"/api/session/{completed_session_id}/export/download"
        assert isinstance(data["size_bytes"], int)
        assert data["size_bytes"] > 0
    
    async def test_export_default_format(self, async_client: AsyncClient, completed_session_id: str):
        """Test exporting with default format (JSON)."""
        response = await async_client.post(
            f"/api/session/{completed_session_id}/export",
            json={}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["format"] == "json"
    
    async def test_export_processing_session_error(self, async_client: AsyncClient, processing_session_id: str):
        """Test exporting a still-processing session."""
        response = await async_client.post(
            f"/api/session/{processing_session_id}/export",
            json={"format": "json"}
        )
        
        assert response.status_code == 425  # Too Early
        data = response.json()
        assert "detail" in data
        assert "not complete" in data["detail"].lower()
    
    async def test_export_session_not_found(self, async_client: AsyncClient):
        """Test exporting non-existent session."""
        fake_id = str(uuid.uuid4())
        response = await async_client.post(
            f"/api/session/{fake_id}/export",
            json={"format": "json"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    async def test_export_invalid_format(self, async_client: AsyncClient, completed_session_id: str):
        """Test exporting with invalid format."""
        response = await async_client.post(
            f"/api/session/{completed_session_id}/export",
            json={"format": "invalid"}
        )
        
        assert response.status_code == 422  # Validation error
    
    async def test_download_exported_file(self, async_client: AsyncClient, completed_session_id: str):
        """Test downloading the exported file."""
        # First export
        export_response = await async_client.post(
            f"/api/session/{completed_session_id}/export",
            json={"format": "json"}
        )
        assert export_response.status_code == 200
        
        # Then download
        download_url = export_response.json()["download_url"]
        download_response = await async_client.get(download_url)
        
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "application/json"
        
        # Verify JSON content
        content = download_response.json()
        assert "session_id" in content
        assert "final_classification" in content
        assert content["session_id"] == completed_session_id