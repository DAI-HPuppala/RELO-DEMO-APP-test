"""ProcessingStatus data model."""
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field


class ProcessingStatus(BaseModel):
    """Real-time status of the classification pipeline."""
    
    # Current State
    session_id: str
    current_agent: Optional[str] = None
    agent_status: str = "waiting"  # waiting, processing, completed
    
    # Timing
    agent_timer_remaining: float = 0.0  # Seconds left in auto mode
    pipeline_elapsed_time: float = 0.0  # Total time since start
    estimated_completion: Optional[datetime] = None
    
    # Progress
    agents_completed: List[str] = Field(default_factory=list)
    agents_pending: List[str] = Field(default_factory=lambda: [
        "initial_classifier",
        "detail_extractor", 
        "damage_detector",
        "final_compiler"
    ])
    overall_progress: int = 0  # 0-100 percentage
    
    # Frame Processing
    current_frame_id: Optional[str] = None
    frames_in_queue: int = 0
    processing_fps: float = 0.0  # Actual processing rate
    
    # System Resources
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: int = 0
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def start_agent(self, agent_name: str, timer_seconds: Optional[float] = None):
        """Mark agent as started."""
        self.current_agent = agent_name
        self.agent_status = "processing"
        
        if timer_seconds:
            self.agent_timer_remaining = timer_seconds
            self.estimated_completion = datetime.now() + timedelta(seconds=timer_seconds)
        
        # Move from pending to processing
        if agent_name in self.agents_pending:
            self.agents_pending.remove(agent_name)
    
    def complete_agent(self, agent_name: str):
        """Mark agent as completed."""
        if agent_name not in self.agents_completed:
            self.agents_completed.append(agent_name)
        
        if agent_name in self.agents_pending:
            self.agents_pending.remove(agent_name)
        
        self.agent_status = "waiting"
        self.agent_timer_remaining = 0.0
        
        # Update overall progress
        total_agents = 4
        self.overall_progress = int((len(self.agents_completed) / total_agents) * 100)
        
        # Move to next agent if any
        if self.agents_pending:
            self.current_agent = self.agents_pending[0]
        else:
            self.current_agent = None
    
    def update_timer(self, remaining_seconds: float):
        """Update remaining timer."""
        self.agent_timer_remaining = max(0.0, remaining_seconds)
        
        if self.agent_timer_remaining > 0:
            self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
        else:
            self.estimated_completion = None
    
    def update_resources(self, gpu_usage: float, gpu_memory: int, 
                        cpu_usage: float, memory: int):
        """Update system resource usage."""
        self.gpu_usage_percent = min(100.0, max(0.0, gpu_usage))
        self.gpu_memory_mb = gpu_memory
        self.cpu_usage_percent = min(100.0, max(0.0, cpu_usage))
        self.memory_usage_mb = memory
    
    def to_websocket_message(self) -> dict:
        """Convert to WebSocket status update message."""
        return {
            "type": "status_update",
            "session_id": self.session_id,
            "current_agent": self.current_agent,
            "progress": self.overall_progress,
            "timer_remaining": self.agent_timer_remaining,
            "fps": self.processing_fps,
            "gpu_usage": self.gpu_usage_percent,
            "connection_quality": "excellent"  # Will be updated from CameraFeed
        }