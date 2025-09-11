"""Integration test for timer expiry handling"""

import pytest
import asyncio
import time

class TestTimerExpiry:
    """Integration tests for timer expiry scenarios"""
    
    @pytest.mark.asyncio
    async def test_timer_expiry_mid_inference_completes(self):
        """Test that ongoing inference completes when timer expires"""
        from backend.src.v2.services.stateful_orchestrator import StatefulOrchestrator
        
        orchestrator = StatefulOrchestrator()
        inference_completed = False
        
        async def slow_inference(frames, prompt):
            nonlocal inference_completed
            await asyncio.sleep(2.0)  # Longer than remaining time
            inference_completed = True
            return {"attributes": {}}
        
        orchestrator.vlm_inference = slow_inference
        
        # Start with short timer
        await orchestrator.run_agent_with_timer("test_agent", 1.5)
        
        # Inference should complete despite timer expiry
        assert inference_completed
    
    @pytest.mark.asyncio
    async def test_no_new_inference_near_expiry(self):
        """Test no new inference starts with <1s remaining"""
        from backend.src.v2.services.stateful_orchestrator import StatefulOrchestrator
        
        orchestrator = StatefulOrchestrator()
        inference_count = 0
        
        async def count_inferences(frames, prompt):
            nonlocal inference_count
            inference_count += 1
            await asyncio.sleep(0.5)
            return {"attributes": {}}
        
        orchestrator.vlm_inference = count_inferences
        
        # Run with timer that allows ~2 inferences
        await orchestrator.run_agent_with_timer("test_agent", 1.8)
        
        # Should not start inference in last second
        assert inference_count <= 2
