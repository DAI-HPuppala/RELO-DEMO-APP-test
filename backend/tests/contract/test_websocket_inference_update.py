"""Contract test for inference_update WebSocket message"""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

# These tests MUST FAIL until implementation is complete (TDD)

class TestInferenceUpdateContract:
    """Contract tests for inference_update message according to WebSocket API v2"""
    
    @pytest.fixture
    async def websocket_handler(self):
        """Mock WebSocket handler that should send inference updates"""
        from backend.src.api.websocket_v2 import WebSocketHandlerV2
        return WebSocketHandlerV2()
    
    @pytest.fixture
    def valid_inference_update(self):
        """Valid inference_update message format"""
        return {
            "type": "inference_update",
            "session_id": str(uuid4()),
            "agent": "initial_classifier",
            "inference_num": 2,
            "frames_used": 3,
            "timer_remaining": 2.1,
            "inference_type": "multi_image",
            "status": "processing"
        }
    
    @pytest.mark.asyncio
    async def test_inference_update_message_format(self, valid_inference_update):
        """Test that inference_update message follows contract schema"""
        # Required fields
        assert valid_inference_update["type"] == "inference_update"
        assert "session_id" in valid_inference_update
        assert "agent" in valid_inference_update
        assert "inference_num" in valid_inference_update
        assert "frames_used" in valid_inference_update
        assert "timer_remaining" in valid_inference_update
        
        # Data types
        assert isinstance(valid_inference_update["inference_num"], int)
        assert isinstance(valid_inference_update["frames_used"], int)
        assert isinstance(valid_inference_update["timer_remaining"], float)
        assert valid_inference_update["inference_type"] in ["single_image", "multi_image"]
    
    @pytest.mark.asyncio
    async def test_first_inference_single_image(self, websocket_handler):
        """Test that first inference is always single image"""
        session_id = str(uuid4())
        
        # Mock first inference
        update = await websocket_handler.create_inference_update(
            session_id=session_id,
            agent="initial_classifier",
            inference_num=1,
            frames=["frame1"],
            timer_remaining=3.5
        )
        
        assert update["inference_num"] == 1
        assert update["frames_used"] == 1
        assert update["inference_type"] == "single_image"
    
    @pytest.mark.asyncio
    async def test_subsequent_inference_multi_image(self, websocket_handler):
        """Test that subsequent inferences are multi-image"""
        session_id = str(uuid4())
        
        # Mock second inference with multiple frames
        update = await websocket_handler.create_inference_update(
            session_id=session_id,
            agent="damage_detector",
            inference_num=2,
            frames=["frame1", "frame2", "frame3"],
            timer_remaining=2.0
        )
        
        assert update["inference_num"] == 2
        assert update["frames_used"] == 3
        assert update["inference_type"] == "multi_image"
    
    @pytest.mark.asyncio
    async def test_progressive_updates_sent(self, websocket_handler):
        """Test that updates are sent progressively during agent processing"""
        session_id = str(uuid4())
        
        # Mock agent performing multiple inferences
        mock_agent = Mock()
        mock_agent.agent_name = "detail_extractor"
        mock_agent.timer_seconds = 3.0
        
        updates_sent = []
        websocket_handler.send_message = AsyncMock(side_effect=lambda msg: updates_sent.append(msg))
        
        # Simulate 3 inferences within timer
        for i in range(1, 4):
            await websocket_handler.send_inference_update(
                session_id=session_id,
                agent=mock_agent.agent_name,
                inference_num=i,
                frames_used=1 if i == 1 else i,
                timer_remaining=3.0 - (i * 0.9)
            )
        
        # Verify all updates sent
        assert len(updates_sent) == 3
        assert updates_sent[0]["inference_num"] == 1
        assert updates_sent[1]["inference_num"] == 2
        assert updates_sent[2]["inference_num"] == 3
        
        # Verify timer countdown
        assert updates_sent[0]["timer_remaining"] > updates_sent[1]["timer_remaining"]
        assert updates_sent[1]["timer_remaining"] > updates_sent[2]["timer_remaining"]
    
    @pytest.mark.asyncio
    async def test_max_frames_per_inference(self, websocket_handler):
        """Test that frames_used doesn't exceed max batch size (5)"""
        session_id = str(uuid4())
        
        # Try to create update with too many frames
        frames = [f"frame{i}" for i in range(7)]  # 7 frames
        
        update = await websocket_handler.create_inference_update(
            session_id=session_id,
            agent="damage_detector",
            inference_num=3,
            frames=frames,
            timer_remaining=1.5
        )
        
        # Should cap at 5 frames
        assert update["frames_used"] <= 5
    
    @pytest.mark.asyncio
    async def test_update_includes_confidence(self, websocket_handler):
        """Test that update can include confidence score"""
        update = {
            "type": "inference_update",
            "session_id": str(uuid4()),
            "agent": "initial_classifier",
            "inference_num": 2,
            "frames_used": 2,
            "timer_remaining": 2.5,
            "inference_type": "multi_image",
            "confidence": 0.85,  # Optional confidence
            "status": "completed"
        }
        
        assert "confidence" in update
        assert 0.0 <= update["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_final_compiler_no_inference_updates(self, websocket_handler):
        """Test that final_compiler doesn't send inference updates (aggregation only)"""
        session_id = str(uuid4())
        
        updates_sent = []
        websocket_handler.send_message = AsyncMock(side_effect=lambda msg: updates_sent.append(msg))
        
        # Final compiler should not send inference updates
        await websocket_handler.process_final_compiler(session_id)
        
        # No inference_update messages should be sent
        inference_updates = [u for u in updates_sent if u.get("type") == "inference_update"]
        assert len(inference_updates) == 0