"""ClassificationSession data model."""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class SessionStatus(str, Enum):
    """Session status enumeration."""
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    STOPPED = "stopped"


class SessionMode(str, Enum):
    """Session mode enumeration."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class ClassificationSession(BaseModel):
    """Represents one complete classification cycle through all agents."""
    
    # Identification
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    apparel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Session Configuration
    mode: SessionMode = SessionMode.AUTOMATIC
    camera_source: str = "webcam"
    agent_timers: Dict[str, float] = Field(default_factory=lambda: {
        "initial_classifier": 4.0,
        "detail_extractor": 3.0,
        "damage_detector": 4.0
    })
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    
    # Status
    status: SessionStatus = SessionStatus.INITIALIZING
    current_agent: Optional[str] = None
    progress_percentage: int = 0
    
    # Results
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    final_classification: Optional[Dict[str, Any]] = None
    
    # Frame Processing
    total_frames_analyzed: int = 0
    frames_per_agent: Dict[str, int] = Field(default_factory=dict)
    
    # Recovery (Manual Mode Only)
    is_resumable: bool = False
    last_checkpoint: Optional[Dict[str, Any]] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    def update_progress(self):
        """Update progress percentage based on completed agents."""
        total_agents = 4  # initial_classifier, detail_extractor, damage_detector, final_compiler
        completed = len(self.agent_results)
        self.progress_percentage = int((completed / total_agents) * 100)
    
    def complete_session(self):
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
        self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds()
        self.progress_percentage = 100
        self.current_agent = None
    
    def interrupt_session(self):
        """Mark session as interrupted."""
        self.status = SessionStatus.INTERRUPTED
        if self.mode == SessionMode.MANUAL:
            self.is_resumable = True
            self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save current state as checkpoint for recovery."""
        self.last_checkpoint = {
            "agents_completed": [r["agent_name"] for r in self.agent_results],
            "frames_analyzed": self.total_frames_analyzed,
            "current_agent": self.current_agent,
            "timestamp": datetime.now().isoformat()
        }
    
    def resume_from_checkpoint(self):
        """Resume session from last checkpoint."""
        if not self.is_resumable or not self.last_checkpoint:
            raise ValueError("Session is not resumable")
        
        self.status = SessionStatus.PROCESSING
        # Restore state from checkpoint
        return self.last_checkpoint
    
    def to_status_dict(self) -> dict:
        """Convert to status response dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "current_agent": self.current_agent,
            "progress_percentage": self.progress_percentage,
            "agents_completed": [r["agent_name"] for r in self.agent_results],
            "timer_remaining": None,  # Will be calculated by service
            "frames_analyzed": self.total_frames_analyzed
        }
    
    def to_results_dict(self) -> dict:
        """Convert to results response dictionary."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "final_classification": self.final_classification,
            "agent_results": self.agent_results,
            "processing_time_seconds": self.total_duration_seconds,
            "total_frames_analyzed": self.total_frames_analyzed
        }