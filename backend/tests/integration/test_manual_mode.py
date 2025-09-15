"""Integration tests for manual mode functionality"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime

from services.manual_mode_handler import ManualModeHandler, ManualCommand
from models.manual_session import ManualSessionState, NavigationDirection, AgentExecution


class TestManualModeHandler:
    """Test suite for ManualModeHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create a manual mode handler instance"""
        return ManualModeHandler("test_session_123")
    
    @pytest.fixture
    def session_state(self):
        """Create a manual session state instance"""
        return ManualSessionState(session_id="test_session_123")
    
    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.session_id == "test_session_123"
        assert handler.current_agent == "initial_classifier"
        assert handler.current_agent_index == 0
        assert not handler.manual_mode_active
        assert len(handler.agent_sequence) == 3
    
    def test_activate_manual_mode(self, handler):
        """Test activating manual mode"""
        handler.activate_manual_mode()
        assert handler.manual_mode_active
        assert handler.waiting_for_command
    
    def test_deactivate_manual_mode(self, handler):
        """Test deactivating manual mode"""
        handler.activate_manual_mode()
        handler.deactivate_manual_mode()
        assert not handler.manual_mode_active
        assert not handler.waiting_for_command
    
    @pytest.mark.asyncio
    async def test_handle_next_command(self, handler):
        """Test handling next navigation command"""
        handler.activate_manual_mode()
        
        # Navigate from initial to detail
        result = await handler.handle_command({
            "type": "manual_next",
            "timestamp": datetime.now().timestamp()
        })
        
        assert result["success"]
        assert result["action"] == "navigate_next"
        assert handler.current_agent == "detail_extractor"
        assert handler.current_agent_index == 1
    
    @pytest.mark.asyncio
    async def test_handle_previous_command(self, handler):
        """Test handling previous navigation command"""
        handler.activate_manual_mode()
        handler.current_agent_index = 1
        handler.current_agent = "detail_extractor"
        
        # Navigate back to initial
        result = await handler.handle_command({
            "type": "manual_previous",
            "timestamp": datetime.now().timestamp()
        })
        
        assert result["success"]
        assert result["action"] == "navigate_previous"
        assert handler.current_agent == "initial_classifier"
        assert handler.current_agent_index == 0
    
    @pytest.mark.asyncio
    async def test_handle_redo_command(self, handler):
        """Test handling redo command"""
        handler.activate_manual_mode()
        handler.agent_results["initial_classifier"] = {"test": "data"}
        handler.completed_agents["initial_classifier"] = True
        
        result = await handler.handle_command({
            "type": "manual_redo",
            "agent": "initial_classifier",
            "timestamp": datetime.now().timestamp()
        })
        
        assert result["success"]
        assert result["action"] == "redo_agent"
        assert "initial_classifier" not in handler.agent_results
        assert not handler.completed_agents.get("initial_classifier", True)
    
    @pytest.mark.asyncio
    async def test_handle_select_agent_command(self, handler):
        """Test handling select agent command"""
        handler.activate_manual_mode()
        
        result = await handler.handle_command({
            "type": "manual_select_agent",
            "agent": "damage_detector",
            "timestamp": datetime.now().timestamp()
        })
        
        assert result["success"]
        assert result["action"] == "select_agent"
        assert handler.current_agent == "damage_detector"
        assert handler.current_agent_index == 2
    
    @pytest.mark.asyncio
    async def test_handle_confirm_final_all_complete(self, handler):
        """Test confirming final compilation when all agents complete"""
        handler.activate_manual_mode()
        
        # Mark all agents as completed
        for agent in handler.agent_sequence:
            handler.completed_agents[agent] = True
            handler.agent_results[agent] = {"test": f"{agent}_data"}
        
        result = await handler.handle_command({
            "type": "manual_confirm_final",
            "timestamp": datetime.now().timestamp()
        })
        
        assert result["success"]
        assert result["action"] == "run_final_compiler"
        assert result["next_cycle"] == 2
        assert handler.current_cycle == 2
        assert handler.current_agent == "initial_classifier"
        assert len(handler.completed_agents) == 0
    
    @pytest.mark.asyncio
    async def test_handle_confirm_final_incomplete(self, handler):
        """Test confirming final compilation when agents incomplete"""
        handler.activate_manual_mode()
        
        # Only mark first agent as completed
        handler.completed_agents["initial_classifier"] = True
        
        result = await handler.handle_command({
            "type": "manual_confirm_final",
            "timestamp": datetime.now().timestamp()
        })
        
        assert not result["success"]
        assert result["error"] == "Not all agents completed"
        assert "detail_extractor" in result["incomplete_agents"]
        assert "damage_detector" in result["incomplete_agents"]
    
    def test_mark_agent_completed(self, handler):
        """Test marking an agent as completed"""
        results = {"category": "shirt", "confidence": 0.95}
        handler.mark_agent_completed("initial_classifier", results)
        
        assert handler.completed_agents["initial_classifier"]
        assert handler.agent_results["initial_classifier"] == results
        assert handler.waiting_for_command
    
    def test_can_compile(self, handler):
        """Test checking if ready for compilation"""
        assert not handler.can_compile()
        
        # Mark all agents complete
        for agent in handler.agent_sequence:
            handler.completed_agents[agent] = True
        
        assert handler.can_compile()
    
    def test_get_session_state(self, handler):
        """Test getting session state for persistence"""
        handler.activate_manual_mode()
        handler.completed_agents["initial_classifier"] = True
        handler.agent_results["initial_classifier"] = {"test": "data"}
        
        state = handler.get_session_state()
        
        assert state["session_id"] == "test_session_123"
        assert state["current_agent"] == "initial_classifier"
        assert state["completed_agents"]["initial_classifier"]
        assert state["manual_mode_active"]
    
    def test_restore_session_state(self, handler):
        """Test restoring session state"""
        state = {
            "current_agent": "detail_extractor",
            "current_agent_index": 1,
            "completed_agents": {"initial_classifier": True},
            "agent_results": {"initial_classifier": {"test": "data"}},
            "current_cycle": 3,
            "manual_mode_active": True
        }
        
        handler.restore_session_state(state)
        
        assert handler.current_agent == "detail_extractor"
        assert handler.current_agent_index == 1
        assert handler.completed_agents["initial_classifier"]
        assert handler.current_cycle == 3
        assert handler.manual_mode_active


class TestManualSessionState:
    """Test suite for ManualSessionState"""
    
    @pytest.fixture
    def session_state(self):
        """Create a manual session state instance"""
        return ManualSessionState(session_id="test_session_123")
    
    def test_initialization(self, session_state):
        """Test session state initialization"""
        assert session_state.session_id == "test_session_123"
        assert session_state.current_agent == "initial_classifier"
        assert session_state.current_cycle == 1
        assert session_state.is_manual_mode
    
    def test_record_agent_start(self, session_state):
        """Test recording agent start"""
        session_state.record_agent_start("initial_classifier", 4.0)
        
        assert "initial_classifier" in session_state.agent_executions
        assert len(session_state.agent_executions["initial_classifier"]) == 1
        execution = session_state.agent_executions["initial_classifier"][0]
        assert execution.timer_seconds == 4.0
        assert not session_state.waiting_for_command
    
    def test_record_agent_completion(self, session_state):
        """Test recording agent completion"""
        session_state.record_agent_start("initial_classifier", 4.0)
        results = {"category": "shirt", "confidence": 0.95}
        session_state.record_agent_completion("initial_classifier", results)
        
        assert session_state.completed_agents["initial_classifier"]
        assert session_state.agent_results["initial_classifier"] == results
        assert session_state.waiting_for_command
        
        execution = session_state.agent_executions["initial_classifier"][0]
        assert execution.completed_at is not None
        assert execution.results == results
    
    def test_record_navigation(self, session_state):
        """Test recording navigation events"""
        session_state.record_navigation(
            "initial_classifier",
            "detail_extractor",
            NavigationDirection.FORWARD,
            "manual_next"
        )
        
        assert len(session_state.navigation_history) == 1
        event = session_state.navigation_history[0]
        assert event.from_agent == "initial_classifier"
        assert event.to_agent == "detail_extractor"
        assert event.direction == NavigationDirection.FORWARD
        assert session_state.current_agent == "detail_extractor"
    
    def test_mark_agent_for_override(self, session_state):
        """Test marking agent for override"""
        # First execution
        session_state.record_agent_start("initial_classifier", 4.0)
        session_state.record_agent_completion("initial_classifier", {"v": 1})
        
        # Mark for override
        session_state.mark_agent_for_override("initial_classifier")
        
        assert "initial_classifier" not in session_state.completed_agents
        assert "initial_classifier" not in session_state.agent_results
        assert session_state.agent_executions["initial_classifier"][0].was_overridden
    
    def test_start_new_cycle(self, session_state):
        """Test starting a new cycle"""
        session_state.completed_agents["initial_classifier"] = True
        session_state.current_cycle = 2
        session_state.current_agent_index = 2
        
        session_state.start_new_cycle()
        
        assert session_state.current_cycle == 3
        assert session_state.current_agent == "initial_classifier"
        assert session_state.current_agent_index == 0
        assert len(session_state.completed_agents) == 0
    
    def test_get_incomplete_agents(self, session_state):
        """Test getting incomplete agents"""
        session_state.completed_agents["initial_classifier"] = True
        
        incomplete = session_state.get_incomplete_agents()
        
        assert "detail_extractor" in incomplete
        assert "damage_detector" in incomplete
        assert "initial_classifier" not in incomplete
    
    def test_get_navigation_summary(self, session_state):
        """Test getting navigation summary"""
        # Add some navigation events
        session_state.record_navigation("a", "b", NavigationDirection.FORWARD, "next")
        session_state.record_navigation("b", "a", NavigationDirection.BACKWARD, "prev")
        session_state.record_navigation("a", "c", NavigationDirection.JUMP, "select")
        
        summary = session_state.get_navigation_summary()
        
        assert summary["total_navigations"] == 3
        assert summary["forward_navigations"] == 1
        assert summary["backward_navigations"] == 1
        assert summary["jump_navigations"] == 1
        assert summary["last_navigation"] == "c"
    
    def test_to_dict(self, session_state):
        """Test converting to dictionary"""
        session_state.completed_agents["initial_classifier"] = True
        
        data = session_state.to_dict()
        
        assert data["session_id"] == "test_session_123"
        assert data["current_agent"] == "initial_classifier"
        assert data["is_manual_mode"]
        assert "initial_classifier" in data["completed_agents"]
    
    def test_from_dict(self):
        """Test creating from dictionary"""
        data = {
            "session_id": "test_456",
            "current_agent": "detail_extractor",
            "current_agent_index": 1,
            "current_cycle": 2,
            "is_manual_mode": True,
            "completed_agents": {"initial_classifier": True}
        }
        
        state = ManualSessionState.from_dict(data)
        
        assert state.session_id == "test_456"
        assert state.current_agent == "detail_extractor"
        assert state.current_agent_index == 1
        assert state.current_cycle == 2
        assert state.completed_agents["initial_classifier"]


@pytest.mark.asyncio
class TestManualModeIntegration:
    """Integration tests for manual mode with mocked dependencies"""
    
    async def test_manual_mode_full_cycle(self):
        """Test a full manual mode cycle"""
        handler = ManualModeHandler("test_session")
        handler.activate_manual_mode()
        
        # Mock callbacks
        status_updates = []
        async def mock_send_status(status):
            status_updates.append(status)
        
        handler.set_callbacks(send_status=mock_send_status)
        
        # Navigate through all agents
        agents = ["initial_classifier", "detail_extractor", "damage_detector"]
        
        for i, agent in enumerate(agents):
            # Simulate agent completion
            handler.mark_agent_completed(agent, {
                "category": f"result_{agent}",
                "confidence": 0.9 + i * 0.01
            })
            
            if i < len(agents) - 1:
                # Navigate to next
                result = await handler.handle_command({
                    "type": "manual_next",
                    "timestamp": datetime.now().timestamp()
                })
                assert result["success"]
        
        # Verify all agents completed
        assert handler.can_compile()
        
        # Confirm final compilation
        result = await handler.handle_command({
            "type": "manual_confirm_final",
            "timestamp": datetime.now().timestamp()
        })
        
        assert result["success"]
        assert result["action"] == "run_final_compiler"
        assert handler.current_cycle == 2
        
        # Check status updates were sent
        assert len(status_updates) > 0
    
    async def test_manual_mode_with_redo(self):
        """Test manual mode with redo operations"""
        handler = ManualModeHandler("test_session")
        handler.activate_manual_mode()
        
        # Complete first agent
        handler.mark_agent_completed("initial_classifier", {"v": 1})
        
        # Navigate to next
        await handler.handle_command({"type": "manual_next"})
        assert handler.current_agent == "detail_extractor"
        
        # Go back and redo
        await handler.handle_command({"type": "manual_previous"})
        result = await handler.handle_command({
            "type": "manual_redo",
            "agent": "initial_classifier"
        })
        
        assert result["success"]
        assert "initial_classifier" not in handler.agent_results
        
        # Complete with new results
        handler.mark_agent_completed("initial_classifier", {"v": 2})
        assert handler.agent_results["initial_classifier"]["v"] == 2
    
    async def test_manual_mode_jump_navigation(self):
        """Test jumping between agents"""
        handler = ManualModeHandler("test_session")
        handler.activate_manual_mode()
        
        # Jump directly to damage detector
        result = await handler.handle_command({
            "type": "manual_select_agent",
            "agent": "damage_detector"
        })
        
        assert result["success"]
        assert handler.current_agent == "damage_detector"
        
        # Complete it
        handler.mark_agent_completed("damage_detector", {"damage": "none"})
        
        # Jump back to initial
        result = await handler.handle_command({
            "type": "manual_select_agent",
            "agent": "initial_classifier"
        })
        
        assert result["success"]
        assert handler.current_agent == "initial_classifier"
        
        # Complete remaining agents
        handler.mark_agent_completed("initial_classifier", {"cat": "shirt"})
        handler.mark_agent_completed("detail_extractor", {"color": "blue"})
        
        assert handler.can_compile()