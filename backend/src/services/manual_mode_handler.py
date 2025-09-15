"""Manual Mode Handler Service for controlling agent navigation through WebRTC commands"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ManualCommand(Enum):
    """Manual mode command types"""
    PREVIOUS = "manual_previous"
    NEXT = "manual_next"
    REDO = "manual_redo"
    SELECT_AGENT = "manual_select_agent"


class ManualModeHandler:
    """Handles manual mode operations for agent orchestration"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Agent sequence configuration
        self.agent_sequence = [
            "initial_classifier",
            "detail_extractor", 
            "damage_detector"
        ]
        
        # Current state
        self.current_agent_index = 0
        self.current_agent = self.agent_sequence[0]
        self.waiting_for_command = False
        self.manual_mode_active = False
        
        # Track completion status per cycle
        self.completed_agents: Dict[str, bool] = {}
        self.completed_agents_set: set = set()  # Track agents completed at least once across all cycles
        self.agent_results: Dict[str, Any] = {}
        self.current_cycle = 1
        
        # Navigation history for undo/redo
        self.navigation_history: List[str] = []
        
        # Command handlers registry
        self.command_handlers = {
            ManualCommand.PREVIOUS: self._handle_previous,
            ManualCommand.NEXT: self._handle_next,
            ManualCommand.REDO: self._handle_redo,
            ManualCommand.SELECT_AGENT: self._handle_select_agent
        }
        
        # Callbacks for external communication
        self.send_status_callback: Optional[Callable] = None
        self.run_agent_callback: Optional[Callable] = None
        self.run_final_compiler_callback: Optional[Callable] = None
        
        logger.info(f"ManualModeHandler initialized for session {session_id}")
    
    def activate_manual_mode(self):
        """Activate manual mode"""
        self.manual_mode_active = True
        self.waiting_for_command = True
        logger.info(f"Manual mode activated for session {self.session_id}")
    
    def deactivate_manual_mode(self):
        """Deactivate manual mode"""
        self.manual_mode_active = False
        self.waiting_for_command = False
        logger.info(f"Manual mode deactivated for session {self.session_id}")
    
    def is_active(self) -> bool:
        """Check if manual mode is active"""
        return self.manual_mode_active
    
    def set_callbacks(self, 
                     send_status: Optional[Callable] = None,
                     run_agent: Optional[Callable] = None,
                     run_final_compiler: Optional[Callable] = None):
        """Set callback functions for external communication"""
        if send_status:
            self.send_status_callback = send_status
        if run_agent:
            self.run_agent_callback = run_agent
        if run_final_compiler:
            self.run_final_compiler_callback = run_final_compiler
    
    async def handle_command(self, command_data: Dict) -> Dict[str, Any]:
        """Handle incoming manual command"""
        command_type = command_data.get("type")
        
        # Parse command type
        try:
            command = ManualCommand(command_type)
        except ValueError:
            logger.error(f"Unknown manual command type: {command_type}")
            return {
                "success": False,
                "error": f"Unknown command type: {command_type}"
            }
        
        # Execute command handler
        handler = self.command_handlers.get(command)
        if handler:
            result = await handler(command_data)
            
            # Send status update after command execution
            await self._send_status_update()
            
            return result
        else:
            return {
                "success": False,
                "error": f"No handler for command: {command_type}"
            }
    
    async def _handle_previous(self, command_data: Dict) -> Dict[str, Any]:
        """Navigate to previous agent"""
        if self.current_agent_index > 0:
            self.current_agent_index -= 1
            self.current_agent = self.agent_sequence[self.current_agent_index]
            self.navigation_history.append(self.current_agent)
            
            logger.info(f"Navigated to previous agent: {self.current_agent}")
            
            return {
                "success": True,
                "action": "navigate_previous",
                "current_agent": self.current_agent,
                "agent_index": self.current_agent_index
            }
        else:
            logger.warning("Already at first agent, cannot go previous")
            return {
                "success": False,
                "error": "Already at first agent"
            }
    
    async def _handle_next(self, command_data: Dict) -> Dict[str, Any]:
        """Navigate to next agent or trigger final compilation"""
        
        # First check if all agents have been completed at least once
        if self._all_agents_completed():
            # All agents completed - trigger final compiler from ANY position
            logger.info("All agents completed, triggering final compilation")
            
            # This will trigger final_compiler and then reset for next cycle
            return {
                "success": True,
                "action": "run_final_compiler",
                "current_agent": self.current_agent,
                "message": "Running final compiler"
            }
        
        # Not all agents completed - smart navigation
        if self.current_agent_index == len(self.agent_sequence) - 1:
            # We're at damage_detector, need to find first incomplete agent
            for i, agent in enumerate(self.agent_sequence):
                if not self.completed_agents.get(agent, False):
                    # Jump to first incomplete agent
                    self.current_agent_index = i
                    self.current_agent = agent
                    self.navigation_history.append(agent)
                    logger.info(f"Jumping to incomplete agent: {agent}")
                    
                    return {
                        "success": True,
                        "action": "select_agent",  # Using select_agent action for jump
                        "current_agent": agent,
                        "agent_index": i,
                        "message": f"Jumping to incomplete agent: {agent}"
                    }
            
            # This shouldn't happen if logic is correct
            logger.error("At last agent but not all completed - logic error")
            return {
                "success": False,
                "error": "Navigation logic error"
            }
        
        # Normal forward navigation
        self.current_agent_index += 1
        self.current_agent = self.agent_sequence[self.current_agent_index]
        self.navigation_history.append(self.current_agent)
        
        logger.info(f"Navigated to next agent: {self.current_agent}")
        
        return {
            "success": True,
            "action": "navigate_next",
            "current_agent": self.current_agent,
            "agent_index": self.current_agent_index
        }
    
    async def _handle_redo(self, command_data: Dict) -> Dict[str, Any]:
        """Re-run the current agent"""
        agent = command_data.get("agent", self.current_agent)
        
        logger.info(f"Redo requested for agent: {agent}")
        
        # Clear previous results for this agent
        if agent in self.agent_results:
            del self.agent_results[agent]
        
        # Mark as not completed to allow re-running
        self.completed_agents[agent] = False
        
        return {
            "success": True,
            "action": "redo_agent",
            "agent": agent,
            "message": f"Ready to re-run {agent}"
        }
    
    async def _handle_select_agent(self, command_data: Dict) -> Dict[str, Any]:
        """Jump directly to a specific agent"""
        target_agent = command_data.get("agent")
        
        if target_agent not in self.agent_sequence:
            return {
                "success": False,
                "error": f"Invalid agent: {target_agent}"
            }
        
        # Find agent index
        try:
            agent_index = self.agent_sequence.index(target_agent)
            self.current_agent_index = agent_index
            self.current_agent = target_agent
            self.navigation_history.append(target_agent)
            
            # If jumping to an already completed agent, mark for override
            if self.completed_agents.get(target_agent, False):
                logger.info(f"Jumping to completed agent {target_agent}, will override results")
                # Clear previous results
                if target_agent in self.agent_results:
                    del self.agent_results[target_agent]
                self.completed_agents[target_agent] = False
            
            logger.info(f"Jumped to agent: {target_agent} at index {agent_index}")
            
            return {
                "success": True,
                "action": "select_agent",
                "agent": target_agent,
                "agent_index": agent_index,
                "will_override": target_agent in self.completed_agents
            }
        except ValueError:
            return {
                "success": False,
                "error": f"Agent {target_agent} not found in sequence"
            }
    
    
    def mark_agent_completed(self, agent_name: str, results: Dict[str, Any]):
        """Mark an agent as completed with its results"""
        self.completed_agents[agent_name] = True
        self.agent_results[agent_name] = results
        
        # Add to completed_agents_set if it's one of the main classification agents
        if agent_name in self.agent_sequence:
            self.completed_agents_set.add(agent_name)
            logger.info(f"Agent {agent_name} marked as completed, total unique completions: {len(self.completed_agents_set)}")
        
        # Update waiting status
        self.waiting_for_command = True
    
    def get_current_agent(self) -> str:
        """Get the current agent"""
        return self.current_agent
    
    def get_next_agent(self) -> Optional[str]:
        """Get the next agent in sequence"""
        if self.current_agent_index < len(self.agent_sequence) - 1:
            return self.agent_sequence[self.current_agent_index + 1]
        return None
    
    def get_previous_agent(self) -> Optional[str]:
        """Get the previous agent in sequence"""
        if self.current_agent_index > 0:
            return self.agent_sequence[self.current_agent_index - 1]
        return None
    
    def _all_agents_completed(self) -> bool:
        """Check if all classification agents have completed"""
        return all(self.completed_agents.get(agent, False) for agent in self.agent_sequence)
    
    def reset_after_final_compiler(self):
        """Reset state after final compiler completes"""
        self.current_cycle += 1
        self.current_agent_index = 0
        self.current_agent = self.agent_sequence[0]
        self.completed_agents = {}
        self.agent_results = {}
        self.navigation_history = [self.current_agent]
        self.waiting_for_command = True
        # Note: completed_agents_set is NOT cleared - it persists across cycles
        logger.info(f"Reset after final compiler, starting cycle {self.current_cycle}, completed agents set: {self.completed_agents_set}")
    
    def can_compile(self) -> bool:
        """Check if ready for final compilation"""
        return self._all_agents_completed()
    
    async def _send_status_update(self):
        """Send comprehensive status update"""
        if not self.send_status_callback:
            return
        
        all_completed = self._all_agents_completed()
        # Check if all three main agents have been completed at least once
        all_agents_completed_once = len(self.completed_agents_set) >= len(self.agent_sequence)
        
        logger.info(f"Manual mode status: completed_agents={list(self.completed_agents.keys())}, all_completed={all_completed}, completed_set={self.completed_agents_set}")
        
        status = {
            "type": "manual_mode_status",
            "session_id": self.session_id,
            "current_agent": self.current_agent,
            "agent_status": "completed" if self.completed_agents.get(self.current_agent) else "pending",
            "waiting_for_command": self.waiting_for_command,
            "completed_agents": list(self.completed_agents.keys()),
            "completed_agents_set": list(self.completed_agents_set),  # Include the set for frontend tracking
            "pending_agents": [a for a in self.agent_sequence if not self.completed_agents.get(a, False)],
            "can_compile": self.can_compile(),
            "node_clicking_enabled": all_agents_completed_once,  # Enable only when all agents completed at least once
            "cycle_number": self.current_cycle,
            "results": self.agent_results,
            "manual_mode_active": self.manual_mode_active
        }
        
        logger.debug(f"Sending manual_mode_status with node_clicking_enabled={all_agents_completed_once} (set size: {len(self.completed_agents_set)})")
        await self.send_status_callback(status)
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get complete session state for persistence"""
        return {
            "session_id": self.session_id,
            "current_agent": self.current_agent,
            "current_agent_index": self.current_agent_index,
            "completed_agents": self.completed_agents,
            "completed_agents_set": list(self.completed_agents_set),  # Convert set to list for JSON serialization
            "agent_results": self.agent_results,
            "current_cycle": self.current_cycle,
            "navigation_history": self.navigation_history[-10:],  # Keep last 10 navigations
            "waiting_for_command": self.waiting_for_command,
            "manual_mode_active": self.manual_mode_active
        }
    
    def restore_session_state(self, state: Dict[str, Any]):
        """Restore session state from persistence"""
        self.current_agent = state.get("current_agent", self.agent_sequence[0])
        self.current_agent_index = state.get("current_agent_index", 0)
        self.completed_agents = state.get("completed_agents", {})
        self.completed_agents_set = set(state.get("completed_agents_set", []))  # Convert list back to set
        self.agent_results = state.get("agent_results", {})
        self.current_cycle = state.get("current_cycle", 1)
        self.navigation_history = state.get("navigation_history", [])
        self.waiting_for_command = state.get("waiting_for_command", False)
        self.manual_mode_active = state.get("manual_mode_active", False)
        
        logger.info(f"Session state restored for {self.session_id}, current agent: {self.current_agent}, completed_set: {self.completed_agents_set}")
    
    def reset(self):
        """Reset manual mode state for new session (clears everything including completed_agents_set)"""
        self.current_agent_index = 0
        self.current_agent = self.agent_sequence[0]
        self.completed_agents = {}
        self.completed_agents_set = set()  # Clear the set when starting fresh session
        self.agent_results = {}
        self.waiting_for_command = True
        self.navigation_history = []
        logger.info(f"Manual mode state reset for session {self.session_id} (cleared completed_agents_set)")