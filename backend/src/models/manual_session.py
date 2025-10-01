"""Manual Mode Session State model for tracking manual navigation state"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class NavigationDirection(Enum):
    """Navigation direction for manual mode"""
    FORWARD = "forward"
    BACKWARD = "backward"
    JUMP = "jump"
    REDO = "redo"


@dataclass
class AgentExecution:
    """Record of a single agent execution"""
    agent_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    timer_seconds: float = 0.0
    results: Dict[str, Any] = field(default_factory=dict)
    was_interrupted: bool = False
    was_overridden: bool = False
    execution_number: int = 1  # Track multiple executions of same agent


@dataclass
class NavigationEvent:
    """Record of a navigation event in manual mode"""
    timestamp: datetime
    from_agent: str
    to_agent: str
    direction: NavigationDirection
    command_type: str
    cycle_number: int


@dataclass
class ManualSessionState:
    """Complete state for a manual mode session"""
    
    # Session identification
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Current state
    current_agent: str = "initial_classifier"
    current_agent_index: int = 0
    current_cycle: int = 1
    is_manual_mode: bool = True
    waiting_for_command: bool = True
    
    # Agent sequence configuration
    agent_sequence: List[str] = field(default_factory=lambda: [
        "initial_classifier",
        "detail_extractor",
        "damage_detector"
    ])
    
    # Execution tracking
    agent_executions: Dict[str, List[AgentExecution]] = field(default_factory=dict)
    completed_agents: Dict[str, bool] = field(default_factory=dict)
    completed_agents_set: set = field(default_factory=set)  # Track agents completed at least once across all cycles
    
    # Results storage
    agent_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    final_compilation_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Navigation history
    navigation_history: List[NavigationEvent] = field(default_factory=list)
    
    # Timer tracking
    agent_timers: Dict[str, float] = field(default_factory=lambda: {
        "initial_classifier": 4.0,
        "detail_extractor": 3.0,
        "damage_detector": 4.0,
        "final_compiler": 1.0
    })
    
    # State flags
    can_compile: bool = False
    compilation_pending: bool = False
    session_active: bool = True
    
    def record_agent_start(self, agent_name: str, timer_seconds: float):
        """Record the start of an agent execution"""
        if agent_name not in self.agent_executions:
            self.agent_executions[agent_name] = []
        
        # Count previous executions
        execution_number = len(self.agent_executions[agent_name]) + 1
        
        execution = AgentExecution(
            agent_name=agent_name,
            started_at=datetime.now(),
            timer_seconds=timer_seconds,
            execution_number=execution_number
        )
        
        self.agent_executions[agent_name].append(execution)
        self.waiting_for_command = False
    
    def record_agent_completion(self, agent_name: str, results: Dict[str, Any]):
        """Record the completion of an agent execution"""
        if agent_name in self.agent_executions and self.agent_executions[agent_name]:
            # Get the latest execution
            execution = self.agent_executions[agent_name][-1]
            execution.completed_at = datetime.now()
            execution.results = results
            
            # Mark agent as completed
            self.completed_agents[agent_name] = True
            self.agent_results[agent_name] = results
            
            # Add to completed_agents_set if it's one of the main classification agents
            if agent_name in self.agent_sequence:
                self.completed_agents_set.add(agent_name)
            
            # Check if we can compile
            self.can_compile = self._check_can_compile()
            
            # Set waiting for command
            self.waiting_for_command = True
    
    def record_agent_interruption(self, agent_name: str):
        """Record that an agent was interrupted"""
        if agent_name in self.agent_executions and self.agent_executions[agent_name]:
            execution = self.agent_executions[agent_name][-1]
            execution.was_interrupted = True
            execution.completed_at = datetime.now()
    
    def record_navigation(self, from_agent: str, to_agent: str, 
                         direction: NavigationDirection, command_type: str):
        """Record a navigation event"""
        event = NavigationEvent(
            timestamp=datetime.now(),
            from_agent=from_agent,
            to_agent=to_agent,
            direction=direction,
            command_type=command_type,
            cycle_number=self.current_cycle
        )
        self.navigation_history.append(event)
        
        # Update current agent
        self.current_agent = to_agent
        if to_agent in self.agent_sequence:
            self.current_agent_index = self.agent_sequence.index(to_agent)
    
    def mark_agent_for_override(self, agent_name: str):
        """Mark that an agent's results will be overridden"""
        if agent_name in self.agent_executions and self.agent_executions[agent_name]:
            # Mark all previous executions as overridden
            for execution in self.agent_executions[agent_name]:
                execution.was_overridden = True
        
        # Clear completion status
        if agent_name in self.completed_agents:
            del self.completed_agents[agent_name]
        
        # Clear results
        if agent_name in self.agent_results:
            del self.agent_results[agent_name]
    
    def start_new_cycle(self):
        """Start a new classification cycle"""
        self.current_cycle += 1
        self.current_agent = self.agent_sequence[0]
        self.current_agent_index = 0
        self.completed_agents = {}
        self.agent_results = {}
        self.can_compile = False
        self.compilation_pending = False
        self.waiting_for_command = True
        # Note: completed_agents_set is NOT cleared - it persists across cycles
    
    def record_final_compilation(self, results: Dict[str, Any]):
        """Record final compilation results"""
        self.final_compilation_results.append({
            "cycle": self.current_cycle,
            "timestamp": datetime.now().isoformat(),
            "results": results
        })
        
        # Start new cycle after compilation
        self.start_new_cycle()
    
    def _check_can_compile(self) -> bool:
        """Check if all classification agents are completed"""
        return all(self.completed_agents.get(agent, False) 
                  for agent in self.agent_sequence)
    
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
    
    def get_incomplete_agents(self) -> List[str]:
        """Get list of incomplete agents"""
        return [agent for agent in self.agent_sequence 
                if not self.completed_agents.get(agent, False)]
    
    def get_execution_history(self, agent_name: str) -> List[AgentExecution]:
        """Get execution history for an agent"""
        return self.agent_executions.get(agent_name, [])
    
    def get_latest_execution(self, agent_name: str) -> Optional[AgentExecution]:
        """Get the latest execution for an agent"""
        executions = self.agent_executions.get(agent_name, [])
        return executions[-1] if executions else None
    
    def get_navigation_summary(self) -> Dict[str, Any]:
        """Get summary of navigation events"""
        if not self.navigation_history:
            return {"total_navigations": 0}
        
        return {
            "total_navigations": len(self.navigation_history),
            "forward_navigations": sum(1 for e in self.navigation_history 
                                     if e.direction == NavigationDirection.FORWARD),
            "backward_navigations": sum(1 for e in self.navigation_history 
                                      if e.direction == NavigationDirection.BACKWARD),
            "jump_navigations": sum(1 for e in self.navigation_history 
                                  if e.direction == NavigationDirection.JUMP),
            "redo_navigations": sum(1 for e in self.navigation_history 
                                  if e.direction == NavigationDirection.REDO),
            "last_navigation": self.navigation_history[-1].to_agent if self.navigation_history else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "current_agent": self.current_agent,
            "current_agent_index": self.current_agent_index,
            "current_cycle": self.current_cycle,
            "is_manual_mode": self.is_manual_mode,
            "waiting_for_command": self.waiting_for_command,
            "agent_sequence": self.agent_sequence,
            "completed_agents": self.completed_agents,
            "completed_agents_set": list(self.completed_agents_set),  # Convert set to list for JSON serialization
            "agent_results": self.agent_results,
            "can_compile": self.can_compile,
            "compilation_pending": self.compilation_pending,
            "session_active": self.session_active,
            "navigation_summary": self.get_navigation_summary(),
            "incomplete_agents": self.get_incomplete_agents()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ManualSessionState":
        """Create instance from dictionary"""
        state = cls(
            session_id=data["session_id"],
            current_agent=data.get("current_agent", "initial_classifier"),
            current_agent_index=data.get("current_agent_index", 0),
            current_cycle=data.get("current_cycle", 1),
            is_manual_mode=data.get("is_manual_mode", True),
            waiting_for_command=data.get("waiting_for_command", True)
        )
        
        # Restore other fields
        state.completed_agents = data.get("completed_agents", {})
        state.completed_agents_set = set(data.get("completed_agents_set", []))  # Convert list back to set
        state.agent_results = data.get("agent_results", {})
        state.can_compile = data.get("can_compile", False)
        state.compilation_pending = data.get("compilation_pending", False)
        state.session_active = data.get("session_active", True)
        
        return state