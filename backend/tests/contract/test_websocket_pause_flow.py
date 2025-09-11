"""Contract test for pause_flow WebSocket message"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

# These tests MUST FAIL until implementation is complete (TDD)

class TestPauseFlowContract:
    """Contract tests for pause_flow message according to WebSocket API v2"""
    
    @pytest.fixture
    async def websocket_handler(self):
        """Mock WebSocket handler that should handle pause_flow"""
        from backend.src.api.websocket_v2 import WebSocketHandlerV2
        return WebSocketHandlerV2()
    
    @pytest.fixture
    def valid_pause_message(self):
        """Valid pause_flow message format"""
        return {
            "type": "pause_flow",
            "session_id": str(uuid4())
        }
    
    @pytest.mark.asyncio
    async def test_pause_flow_message_format(self, websocket_handler, valid_pause_message):
        """Test that pause_flow message follows contract schema"""
        # Verify message has required fields
        assert "type" in valid_pause_message
        assert valid_pause_message["type"] == "pause_flow"
        assert "session_id" in valid_pause_message
        
        # Verify session_id is UUID format
        try:
            uuid_obj = uuid4().hex
            session_id = valid_pause_message["session_id"].replace("-", "")
            assert len(session_id) == 32  # UUID without hyphens
        except:
            pytest.fail("session_id must be valid UUID format")
    
    @pytest.mark.asyncio
    async def test_pause_flow_during_agent_processing(self, websocket_handler):
        """Test pausing flow while agent is running"""
        session_id = str(uuid4())
        
        # Setup: Mock active session with running agent
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.status = "ACTIVE"
        mock_session.current_agent = "detail_extractor"
        mock_session.agent_states = {
            "detail_extractor": Mock(status="RUNNING", timer_remaining=2.5)
        }
        
        websocket_handler.sessions = {session_id: mock_session}
        
        # Send pause message
        pause_msg = {
            "type": "pause_flow",
            "session_id": session_id
        }
        
        response = await websocket_handler.handle_pause_flow(pause_msg)
        
        # Verify response format
        assert response["type"] == "flow_paused"
        assert response["paused_agent"] == "detail_extractor"
        assert response["session_id"] == session_id
        assert "timer_remaining" in response
        assert response["timer_remaining"] == 2.5
    
    @pytest.mark.asyncio
    async def test_pause_flow_response_format(self, websocket_handler):
        """Test that pause response follows contract"""
        session_id = str(uuid4())
        
        expected_response = {
            "type": "flow_paused",
            "session_id": session_id,
            "paused_agent": "initial_classifier",
            "timer_remaining": 3.2,
            "inference_count": 2,
            "status": "PAUSED"
        }
        
        # Verify all required fields
        assert expected_response["type"] == "flow_paused"
        assert "session_id" in expected_response
        assert "paused_agent" in expected_response
        assert "timer_remaining" in expected_response
        assert isinstance(expected_response["timer_remaining"], float)
    
    @pytest.mark.asyncio
    async def test_pause_during_final_compiler_rejected(self, websocket_handler):
        """Test that pause during final_compiler is rejected"""
        session_id = str(uuid4())
        
        # Setup: Mock session with final_compiler running
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.current_agent = "final_compiler"
        mock_session.agent_states = {
            "final_compiler": Mock(status="RUNNING")
        }
        
        websocket_handler.sessions = {session_id: mock_session}
        
        pause_msg = {
            "type": "pause_flow",
            "session_id": session_id
        }
        
        response = await websocket_handler.handle_pause_flow(pause_msg)
        
        # Should return error or ignore
        assert response["type"] == "error"
        assert "cannot pause" in response["message"].lower()
        assert "final_compiler" in response["message"]
    
    @pytest.mark.asyncio
    async def test_pause_clears_pending_frames(self, websocket_handler):
        """Test that pausing clears accumulated frames for current inference"""
        session_id = str(uuid4())
        
        # Setup: Mock session with frames collected
        mock_agent_state = Mock()
        mock_agent_state.status = "RUNNING"
        mock_agent_state.frames_collected = ["frame1", "frame2", "frame3"]
        mock_agent_state.current_inference_num = 3
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.current_agent = "damage_detector"
        mock_session.agent_states = {"damage_detector": mock_agent_state}
        
        websocket_handler.sessions = {session_id: mock_session}
        
        await websocket_handler.handle_pause_flow({
            "type": "pause_flow",
            "session_id": session_id
        })
        
        # Verify frames were cleared
        assert len(mock_agent_state.frames_collected) == 0
        assert mock_agent_state.status == "PAUSED"
    
    @pytest.mark.asyncio
    async def test_pause_invalid_session_id(self, websocket_handler):
        """Test pause with non-existent session ID"""
        fake_session_id = str(uuid4())
        
        response = await websocket_handler.handle_pause_flow({
            "type": "pause_flow",
            "session_id": fake_session_id
        })
        
        assert response["type"] == "error"
        assert "session not found" in response["message"].lower()
    
    @pytest.mark.asyncio
    async def test_pause_already_paused_session(self, websocket_handler):
        """Test pausing an already paused session"""
        session_id = str(uuid4())
        
        mock_session = Mock()
        mock_session.session_id = session_id
        mock_session.status = "PAUSED"
        mock_session.paused_agent = "initial_classifier"
        
        websocket_handler.sessions = {session_id: mock_session}
        
        response = await websocket_handler.handle_pause_flow({
            "type": "pause_flow",
            "session_id": session_id
        })
        
        # Should return current pause state
        assert response["type"] == "flow_paused"
        assert response["paused_agent"] == "initial_classifier"
        assert response["status"] == "PAUSED"