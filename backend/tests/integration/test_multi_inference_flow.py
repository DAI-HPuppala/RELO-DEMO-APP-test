"""Integration test for multi-inference flow with timers"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4

# These tests MUST FAIL until implementation is complete (TDD)

class TestMultiInferenceFlow:
    """Integration tests for multi-inference flow within timer windows"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Mock stateful orchestrator"""
        from backend.src.v2.services.stateful_orchestrator import StatefulOrchestrator
        return StatefulOrchestrator()
    
    @pytest.fixture
    def mock_frame_provider(self):
        """Mock frame provider that returns test frames"""
        frames = []
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[0, 0] = i  # Mark frame number
            frames.append(frame)
        
        async def provider():
            if frames:
                await asyncio.sleep(0.1)  # Simulate capture time
                return frames.pop(0)
            return None
        
        return provider
    
    @pytest.fixture
    def mock_vlm_inference(self):
        """Mock VLM inference with timing"""
        async def inference(frames, prompt):
            # Simulate inference time based on batch size
            batch_size = len(frames) if isinstance(frames, list) else 1
            if batch_size == 1:
                await asyncio.sleep(0.8)
            elif batch_size <= 3:
                await asyncio.sleep(1.5)
            else:
                await asyncio.sleep(2.2)
            
            return {
                "attributes": {
                    "item_type": "shirt",
                    "color": "blue",
                    "pattern": "solid"
                },
                "confidence": 0.85,
                "reasoning": f"Analyzed {batch_size} frame(s)"
            }
        
        return inference
    
    @pytest.mark.asyncio
    async def test_agent_multiple_inferences_within_timer(self, orchestrator, mock_frame_provider, mock_vlm_inference):
        """Test that agents perform multiple inferences within timer window"""
        orchestrator.frame_provider = mock_frame_provider
        orchestrator.vlm_inference = mock_vlm_inference
        
        # Run initial_classifier with 4-second timer
        agent_state = await orchestrator.run_agent_with_timer(
            agent_name="initial_classifier",
            timer_seconds=4.0
        )
        
        # Should have 3-4 inferences
        assert 3 <= agent_state.inference_count <= 4
        
        # First inference should be single image
        assert agent_state.inference_results[0].inference_type == "single_image"
        assert agent_state.inference_results[0].frame_count == 1
        
        # Subsequent inferences should be multi-image
        if agent_state.inference_count > 1:
            assert agent_state.inference_results[1].inference_type == "multi_image"
            assert agent_state.inference_results[1].frame_count > 1
    
    @pytest.mark.asyncio
    async def test_progressive_frame_accumulation(self, orchestrator, mock_frame_provider):
        """Test frame accumulation pattern: 1 → 2 → 3 → 4 → 5 (max)"""
        orchestrator.frame_provider = mock_frame_provider
        
        frames_collected = []
        
        async def track_frames(frames, prompt):
            frames_collected.append(len(frames) if isinstance(frames, list) else 1)
            return {"attributes": {}, "confidence": 0.8}
        
        orchestrator.vlm_inference = track_frames
        
        await orchestrator.run_agent_with_timer(
            agent_name="detail_extractor",
            timer_seconds=3.0
        )
        
        # Verify progressive accumulation
        assert frames_collected[0] == 1  # First inference: single
        if len(frames_collected) > 1:
            assert frames_collected[1] >= 2  # Second: multi
        if len(frames_collected) > 2:
            assert frames_collected[2] <= 5  # Cap at 5
    
    @pytest.mark.asyncio
    async def test_timer_expiry_completes_ongoing_inference(self, orchestrator):
        """Test that ongoing inference completes even if timer expires"""
        
        inference_started = False
        inference_completed = False
        
        async def slow_inference(frames, prompt):
            nonlocal inference_started, inference_completed
            inference_started = True
            await asyncio.sleep(1.5)  # Longer than remaining time
            inference_completed = True
            return {"attributes": {}, "confidence": 0.9}
        
        orchestrator.vlm_inference = slow_inference
        orchestrator.frame_provider = AsyncMock(return_value=np.zeros((480, 640, 3)))
        
        # Run with short timer
        start_time = time.time()
        await orchestrator.run_agent_with_timer(
            agent_name="damage_detector",
            timer_seconds=2.0  # Timer shorter than inference time
        )
        elapsed = time.time() - start_time
        
        # Should complete the ongoing inference
        assert inference_started
        assert inference_completed
        assert elapsed >= 1.5  # Waited for inference to complete
    
    @pytest.mark.asyncio
    async def test_aggregation_buffer_time(self, orchestrator):
        """Test 1-second buffer reserved for aggregation"""
        
        inference_times = []
        
        async def track_timing(frames, prompt):
            inference_times.append(time.time())
            await asyncio.sleep(0.5)
            return {"attributes": {}, "confidence": 0.85}
        
        orchestrator.vlm_inference = track_timing
        orchestrator.frame_provider = AsyncMock(return_value=np.zeros((480, 640, 3)))
        
        start_time = time.time()
        await orchestrator.run_agent_with_timer(
            agent_name="initial_classifier",
            timer_seconds=3.0
        )
        
        # Last inference should be before (timer - 1.0) seconds
        if len(inference_times) > 0:
            last_inference_time = inference_times[-1] - start_time
            assert last_inference_time < 2.0  # 3.0 - 1.0 buffer
    
    @pytest.mark.asyncio
    async def test_damage_detector_starts_with_batch(self, orchestrator):
        """Test damage detector always starts with multi-image (initial's frame + own)"""
        
        # Setup shared frame from initial classifier
        shared_frame = np.ones((480, 640, 3), dtype=np.uint8)
        orchestrator.frame_registry.register_initial_frame("session123", shared_frame, "frame001")
        
        first_inference_frames = None
        
        async def capture_first(frames, prompt):
            nonlocal first_inference_frames
            if first_inference_frames is None:
                first_inference_frames = len(frames) if isinstance(frames, list) else 1
            return {"attributes": {"is_damaged": False}, "confidence": 0.9}
        
        orchestrator.vlm_inference = capture_first
        orchestrator.frame_provider = AsyncMock(return_value=np.zeros((480, 640, 3)))
        orchestrator.session_id = "session123"
        
        await orchestrator.run_agent_with_timer(
            agent_name="damage_detector",
            timer_seconds=4.0
        )
        
        # First inference should be multi-image (at least 2 frames)
        assert first_inference_frames >= 2