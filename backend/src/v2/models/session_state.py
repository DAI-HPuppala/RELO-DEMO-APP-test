"""Session State model for overall orchestration tracking"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
from uuid import uuid4


class SessionMode(Enum):
    """Session operation modes"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class SessionStatus(Enum):
    """Overall session status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SessionState:
    """Overall session state tracking all agents and orchestration"""
    
    # Session identification
    session_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Session configuration
    mode: SessionMode = SessionMode.AUTOMATIC
    status: SessionStatus = SessionStatus.ACTIVE
    
    # Agent tracking
    agents_sequence: List[str] = field(default_factory=lambda: [
        "initial_classifier", "detail_extractor", "damage_detector", "final_compiler"
    ])
    current_agent_index: int = 0  # Index in sequence (0-based)
    agents_completed: List[str] = field(default_factory=list)
    agent_states: Dict[str, Any] = field(default_factory=dict)  # AgentState objects
    
    # Pause/Resume tracking
    paused_at: Optional[datetime] = None
    paused_agent: Optional[str] = None
    resume_from_agent: Optional[str] = None
    
    # Frame sharing
    shared_frames: Dict[str, Any] = field(default_factory=dict)  # Special frames
    frame_registry: Optional[Any] = None  # FrameRegistry instance
    
    # Results
    final_classification: Optional[Any] = None  # FinalClassification object
    
    # Lifecycle timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @property
    def current_agent(self) -> Optional[str]:
        """Get the currently active agent name"""
        if 0 <= self.current_agent_index < len(self.agents_sequence):
            return self.agents_sequence[self.current_agent_index]
        return None
    
    def advance_to_next_agent(self) -> Optional[str]:
        """Move to the next agent in sequence"""
        if self.current_agent:
            self.agents_completed.append(self.current_agent)
        
        self.current_agent_index += 1
        self.updated_at = datetime.now()
        
        if self.current_agent_index >= len(self.agents_sequence):
            self.complete()
            return None
        
        return self.current_agent
    
    def pause(self, agent_name: str) -> None:
        """Pause the session"""
        self.status = SessionStatus.PAUSED
        self.paused_at = datetime.now()
        self.paused_agent = agent_name
        self.updated_at = datetime.now()
    
    def resume(self, restart_agent: bool = False) -> None:
        """Resume the session"""
        self.status = SessionStatus.ACTIVE
        if restart_agent and self.paused_agent:
            self.resume_from_agent = self.paused_agent
        self.paused_at = None
        self.updated_at = datetime.now()
    
    def complete(self) -> None:
        """Mark session as completed"""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def error(self, error_msg: str = "") -> None:
        """Mark session as errored"""
        self.status = SessionStatus.ERROR
        self.updated_at = datetime.now()
