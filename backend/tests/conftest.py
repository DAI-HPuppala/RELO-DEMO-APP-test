"""Pytest configuration and shared fixtures."""
import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
import uuid
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    from api.main import app
    
    async with AsyncClient(app=app, base_url="http://localhost:8000") as client:
        yield client


@pytest.fixture
def websocket_url() -> str:
    """WebSocket URL for testing."""
    return "ws://localhost:8000/ws/stream"


@pytest.fixture
async def active_session_id(async_client: AsyncClient) -> str:
    """Create an active session and return its ID."""
    response = await async_client.post(
        "/api/session/start",
        json={"mode": "automatic"}
    )
    return response.json()["session_id"]


@pytest.fixture
async def manual_session_id(async_client: AsyncClient) -> str:
    """Create a manual mode session and return its ID."""
    response = await async_client.post(
        "/api/session/start",
        json={"mode": "manual"}
    )
    return response.json()["session_id"]


@pytest.fixture
async def auto_session_id(async_client: AsyncClient) -> str:
    """Create an automatic mode session and return its ID."""
    response = await async_client.post(
        "/api/session/start",
        json={"mode": "automatic"}
    )
    return response.json()["session_id"]


@pytest.fixture
async def processing_session_id(async_client: AsyncClient) -> str:
    """Create a session that's currently processing."""
    response = await async_client.post(
        "/api/session/start",
        json={"mode": "automatic"}
    )
    session_id = response.json()["session_id"]
    
    # Let it start processing
    await asyncio.sleep(0.5)
    
    return session_id


@pytest.fixture
async def completed_session_id(async_client: AsyncClient) -> str:
    """Create a completed session (mock for testing)."""
    # In real implementation, this would complete all agents
    # For now, return a mock completed session ID
    response = await async_client.post(
        "/api/session/start",
        json={
            "mode": "automatic",
            "agent_timers": {
                "initial_classifier": 0.1,
                "detail_extractor": 0.1,
                "damage_detector": 0.1
            }
        }
    )
    session_id = response.json()["session_id"]
    
    # Wait for completion
    await asyncio.sleep(1)
    
    return session_id


@pytest.fixture
async def damaged_item_session_id(async_client: AsyncClient) -> str:
    """Create a session with damage detected."""
    # This would be a session that detected damage
    response = await async_client.post(
        "/api/session/start",
        json={"mode": "manual"}
    )
    return response.json()["session_id"]


@pytest.fixture
async def incomplete_session_id(async_client: AsyncClient) -> str:
    """Create a session with some null fields."""
    response = await async_client.post(
        "/api/session/start",
        json={"mode": "manual"}
    )
    return response.json()["session_id"]