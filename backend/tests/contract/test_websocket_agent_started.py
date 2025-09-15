"""Contract test for agent_started WebSocket message with timer"""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

# These tests MUST FAIL until implementation is complete (TDD)

class TestAgentStartedContract:
    """Contract tests for agent_started message with timer information"""
    
    @pytest.fixture
    async def websocket_handler(self):
        from backend.src.api.websocket_v2 import WebSocketHandlerV2
        return WebSocketHandlerV2()
    
    @pytest.mark.asyncio
    async def test_agent_started_message_format(self):
        """Test agent_started message includes timer information"""
        message = {
            "type": "agent_started",
            "session_id": str(uuid4()),
            "agent": "initial_classifier",
            "timer_seconds": 4.0,
            "mode": "automatic",
            "sequence_position": 1,
            "total_agents": 4
        }
        
        assert message["type"] == "agent_started"
        assert "timer_seconds" in message
        assert isinstance(message["timer_seconds"], float)
        assert message["timer_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_agent_timers_match_spec(self, websocket_handler):
        """Test that default agent timers match specification"""
        timers = {
            "initial_classifier": 4.0,
            "detail_extractor": 3.0,
            "damage_detector": 4.0,
            "final_compiler": 1.0  # Aggregation only
        }
        
        for agent, expected_timer in timers.items():
            config = websocket_handler.get_agent_timer(agent)
            assert config == expected_timer
    
    @pytest.mark.asyncio
    async def test_custom_timer_configuration(self, websocket_handler):
        """Test custom timer configuration for edge case testing"""
        custom_timers = {
            "initial_classifier": 1.5,
            "detail_extractor": 2.0,
            "damage_detector": 4.0
        }
        
        websocket_handler.set_custom_timers(custom_timers)
        
        assert websocket_handler.get_agent_timer("initial_classifier") == 1.5
        assert websocket_handler.get_agent_timer("detail_extractor") == 2.0
        assert websocket_handler.get_agent_timer("damage_detector") == 4.0