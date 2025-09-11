"""Agent State model for tracking individual agent processing"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


class AgentStatus(Enum):
    """Possible states for an agent during processing"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentState:
    """Represents the current state of an individual agent during processing"""
    
    # Agent identification
    agent_name: str  # "initial_classifier", "detail_extractor", etc.
    
    # Status tracking
    status: AgentStatus = AgentStatus.IDLE
    
    # Timer configuration
    timer_seconds: float = 4.0  # Configured timer duration
    timer_remaining: float = 4.0  # Seconds remaining on timer
    timer_started_at: Optional[float] = None  # Unix timestamp when timer started
    
    # Inference tracking
    inference_count: int = 0  # Number of inferences completed
    current_inference_num: int = 0  # Current inference in progress (1-based)
    frames_collected: List[str] = field(default_factory=list)  # Frame IDs for next inference
    
    # Results storage
    inference_results: List[Any] = field(default_factory=list)  # InferenceResult objects
    finalized_attributes: Dict[str, Any] = field(default_factory=dict)  # Aggregated attributes
    aggregation_method: str = ""  # "majority", "last_with_fallback", "single"
    
    # Error tracking
    error_message: Optional[str] = None  # Error message if status is ERROR
    
    # State persistence timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checkpointed_at: Optional[datetime] = None
    
    def start_timer(self) -> None:
        """Start the agent's timer"""
        from time import time
        self.timer_started_at = time()
        self.timer_remaining = self.timer_seconds
        self.status = AgentStatus.RUNNING
        self.updated_at = datetime.now()
    
    def update_timer(self) -> None:
        """Update remaining time on timer"""
        if self.timer_started_at and self.status == AgentStatus.RUNNING:
            from time import time
            elapsed = time() - self.timer_started_at
            self.timer_remaining = max(0, self.timer_seconds - elapsed)
            self.updated_at = datetime.now()
    
    def pause(self) -> None:
        """Pause the agent and save timer state"""
        if self.status == AgentStatus.RUNNING:
            self.update_timer()
            self.status = AgentStatus.PAUSED
            self.frames_collected.clear()  # Clear pending frames
            self.updated_at = datetime.now()
    
    def resume(self, restart: bool = False) -> None:
        """Resume agent processing"""
        if self.status == AgentStatus.PAUSED:
            from time import time
            if restart:
                # Reset timer and clear previous inferences
                self.timer_remaining = self.timer_seconds
                self.inference_results.clear()
                self.inference_count = 0
                self.current_inference_num = 0
            
            self.timer_started_at = time()
            self.status = AgentStatus.RUNNING
            self.updated_at = datetime.now()
    
    def add_inference_result(self, result: Any) -> None:
        """Add a completed inference result"""
        self.inference_results.append(result)
        self.inference_count += 1
        self.frames_collected.clear()
        self.updated_at = datetime.now()
    
    def complete(self) -> None:
        """Mark agent as completed"""
        self.status = AgentStatus.COMPLETED
        self.timer_remaining = 0
        self.updated_at = datetime.now()
    
    def error(self, error_msg: str = "") -> None:
        """Mark agent as errored"""
        self.status = AgentStatus.ERROR
        self.updated_at = datetime.now()
    
    def reset(self) -> None:
        """Reset agent to initial state"""
        self.status = AgentStatus.IDLE
        self.timer_remaining = self.timer_seconds
        self.timer_started_at = None
        self.inference_count = 0
        self.current_inference_num = 0
        self.frames_collected.clear()
        self.inference_results.clear()
        self.finalized_attributes.clear()
        self.aggregation_method = ""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "timer_seconds": self.timer_seconds,
            "timer_remaining": self.timer_remaining,
            "timer_started_at": self.timer_started_at,
            "inference_count": self.inference_count,
            "current_inference_num": self.current_inference_num,
            "frames_collected": self.frames_collected,
            "finalized_attributes": self.finalized_attributes,
            "aggregation_method": self.aggregation_method,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checkpointed_at": self.checkpointed_at.isoformat() if self.checkpointed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create from dictionary"""
        state = cls(
            agent_name=data["agent_name"],
            status=AgentStatus(data["status"]),
            timer_seconds=data["timer_seconds"],
            timer_remaining=data["timer_remaining"],
            timer_started_at=data.get("timer_started_at"),
            inference_count=data["inference_count"],
            current_inference_num=data["current_inference_num"],
            frames_collected=data.get("frames_collected", []),
            finalized_attributes=data.get("finalized_attributes", {}),
            aggregation_method=data.get("aggregation_method", ""),
        )
        
        if data.get("created_at"):
            state.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            state.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("checkpointed_at"):
            state.checkpointed_at = datetime.fromisoformat(data["checkpointed_at"])
        
        return state