"""Contract test for resume_flow WebSocket message"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

# These tests MUST FAIL until implementation is complete (TDD)

class TestResumeFlowContract:
    """Contract tests for resume_flow message according to WebSocket API v2"""
    
    @pytest.fixture
    async def websocket_handler(self):
        """Mock WebSocket handler that should handle resume_flow"""
        from backend.src.api.websocket_v2 import WebSocketHandlerV2
        return WebSocketHandlerV2()
    
    @pytest.fixture
    def valid_resume_message(self):
        """Valid resume_flow message format"""
        return {
            "type": "resume_flow",
            "session_id": str(uuid4()),
            "restart_agent": True
        }
    
    @pytest.mark.asyncio
    async def test_resume_flow_message_format(self, valid_resume_message):
        """Test that resume_flow message follows contract schema"""
        assert "type" in valid_resume_message
        assert valid_resume_message["type"] == "resume_flow"
        assert "session_id" in valid_resume_message
        assert "restart_agent" in valid_resume_message
        assert isinstance(valid_resume_message["restart_agent"], bool)
    
    @pytest.mark.asyncio
    async def test_resume_paused_agent(self, websocket_handler):
        """Test resuming a paused agent with timer reset"""
        session_id = str(uuid4())
        
        # Setup: Mock paused session
        mock_agent_state = Mock()
        mock_agent_state.status = "PAUSED"
        mock_agent_state.timer_seconds = 3.0
        mock_agent_state.inference_count = 1
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.status = "PAUSED"
        mock_session.paused_agent = "detail_extractor"
        mock_session.agent_states = {"detail_extractor": mock_agent_state}
        
        websocket_handler.sessions = {session_id: mock_session}
        
        # Send resume message
        resume_msg = {
            "type": "resume_flow",
            "session_id": session_id,
            "restart_agent": True
        }
        
        response = await websocket_handler.handle_resume_flow(resume_msg)
        
        # Verify response
        assert response["type"] == "flow_resumed"
        assert response["resuming_agent"] == "detail_extractor"
        assert response["timer_seconds"] == 3.0
        assert response["restarted"] == True
    
    @pytest.mark.asyncio
    async def test_resume_response_format(self, websocket_handler):
        """Test that resume response follows contract"""
        session_id = str(uuid4())
        
        expected_response = {
            "type": "flow_resumed",
            "session_id": session_id,
            "resuming_agent": "initial_classifier",
            "timer_seconds": 4.0,
            "restarted": True,
            "inference_count_before_pause": 2
        }
        
        # Verify all required fields
        assert expected_response["type"] == "flow_resumed"
        assert "session_id" in expected_response
        assert "resuming_agent" in expected_response
        assert "timer_seconds" in expected_response
        assert isinstance(expected_response["timer_seconds"], float)
    
    @pytest.mark.asyncio
    async def test_resume_with_timer_reset(self, websocket_handler):
        """Test that restart_agent=true resets timer to full duration"""
        session_id = str(uuid4())
        
        mock_agent_state = Mock()
        mock_agent_state.status = "PAUSED"
        mock_agent_state.timer_seconds = 4.0  # Original timer
        mock_agent_state.timer_remaining = 1.5  # Was paused with 1.5s left
        mock_agent_state.inference_results = [Mock(), Mock()]  # Had 2 inferences
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.paused_agent = "initial_classifier"
        mock_session.agent_states = {"initial_classifier": mock_agent_state}
        
        websocket_handler.sessions = {session_id: mock_session}
        
        await websocket_handler.handle_resume_flow({
            "type": "resume_flow",
            "session_id": session_id,
            "restart_agent": True
        })
        
        # Timer should reset to full duration
        assert mock_agent_state.timer_remaining == 4.0
        # Inference results should be cleared
        assert len(mock_agent_state.inference_results) == 0
    
    @pytest.mark.asyncio
    async def test_resume_without_restart(self, websocket_handler):
        """Test resume without restarting agent (continue from pause point)"""
        session_id = str(uuid4())
        
        mock_agent_state = Mock()
        mock_agent_state.status = "PAUSED"
        mock_agent_state.timer_remaining = 2.3
        mock_agent_state.inference_results = [Mock(), Mock()]
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.paused_agent = "damage_detector"
        mock_session.agent_states = {"damage_detector": mock_agent_state}
        
        websocket_handler.sessions = {session_id: mock_session}
        
        await websocket_handler.handle_resume_flow({
            "type": "resume_flow",
            "session_id": session_id,
            "restart_agent": False
        })
        
        # Timer should continue from where it was
        assert mock_agent_state.timer_remaining == 2.3
        # Previous inferences should be kept
        assert len(mock_agent_state.inference_results) == 2
    
    @pytest.mark.asyncio
    async def test_resume_triggers_agent_started(self, websocket_handler):
        """Test that resuming sends agent_started message"""
        session_id = str(uuid4())
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.paused_agent = "initial_classifier"
        mock_session.agent_states = {
            "initial_classifier": Mock(timer_seconds=4.0)
        }
        
        websocket_handler.sessions = {session_id: mock_session}
        websocket_handler.send_message = AsyncMock()
        
        await websocket_handler.handle_resume_flow({
            "type": "resume_flow",
            "session_id": session_id,
            "restart_agent": True
        })
        
        # Should send agent_started message
        websocket_handler.send_message.assert_called()
        sent_msg = websocket_handler.send_message.call_args[0][0]
        assert sent_msg["type"] == "agent_started"
        assert sent_msg["agent"] == "initial_classifier"
        assert sent_msg["timer_seconds"] == 4.0
    
    @pytest.mark.asyncio
    async def test_resume_non_paused_session_error(self, websocket_handler):
        """Test resuming a non-paused session returns error"""
        session_id = str(uuid4())
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.status = "ACTIVE"
        
        websocket_handler.sessions = {session_id: mock_session}
        
        response = await websocket_handler.handle_resume_flow({
            "type": "resume_flow",
            "session_id": session_id,
            "restart_agent": True
        })
        
        assert response["type"] == "error"
        assert "not paused" in response["message"].lower()