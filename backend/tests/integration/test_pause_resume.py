"""Integration test for pause/resume during processing"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

class TestPauseResume:
    """Integration tests for pause/resume functionality"""
    
    @pytest.mark.asyncio
    async def test_pause_during_agent_processing(self):
        """Test pausing flow while agent is running"""
        from backend.src.v2.services.stateful_orchestrator import StatefulOrchestrator
        orchestrator = StatefulOrchestrator()
        
        # Start agent processing
        task = asyncio.create_task(orchestrator.run_agent_with_timer(
            "initial_classifier", 4.0
        ))
        
        await asyncio.sleep(1.5)  # Let it run for a bit
        
        # Pause the flow
        await orchestrator.pause_flow()
        
        # Agent should be paused
        agent_state = orchestrator.get_agent_state("initial_classifier")
        assert agent_state.status == "PAUSED"
        assert agent_state.timer_remaining < 4.0
    
    @pytest.mark.asyncio
    async def test_resume_with_timer_reset(self):
        """Test resuming with timer reset"""
        from backend.src.v2.services.stateful_orchestrator import StatefulOrchestrator
        orchestrator = StatefulOrchestrator()
        
        # Setup paused state
        agent_state = Mock(status="PAUSED", timer_seconds=3.0)
        orchestrator.agent_states = {"detail_extractor": agent_state}
        
        # Resume with reset
        await orchestrator.resume_flow(restart_agent=True)
        
        # Timer should be reset
        assert agent_state.timer_remaining == 3.0
        assert agent_state.status == "RUNNING"
