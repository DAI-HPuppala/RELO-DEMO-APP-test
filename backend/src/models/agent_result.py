"""AgentResult data model."""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class TriggerType(str, Enum):
    """Agent trigger type enumeration."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class AgentResult(BaseModel):
    """Individual analysis output from each specialized agent."""
    
    # Identification
    agent_name: str
    session_id: str
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    processing_time_seconds: float = 0.0
    
    # Configuration
    timer_seconds: Optional[float] = None
    trigger_type: TriggerType = TriggerType.AUTOMATIC
    
    # Results
    detected_attributes: Dict[str, Any] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""
    
    # Frame Data
    frames_processed: int = 0
    last_frame_id: Optional[str] = None
    
    # Progressive Updates
    partial_results: List[Dict[str, Any]] = Field(default_factory=list)
    update_timestamps: List[datetime] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True
    
    def complete_processing(self):
        """Mark agent processing as complete."""
        self.completed_at = datetime.now()
        self.processing_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def add_partial_result(self, attributes: Dict[str, Any], confidence: float = 0.0):
        """Add a partial result during processing."""
        self.partial_results.append({
            "attributes": attributes,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "frame_number": self.frames_processed
        })
        self.update_timestamps.append(datetime.now())
    
    def to_response_dict(self) -> dict:
        """Convert to API response dictionary."""
        return {
            "agent_name": self.agent_name,
            "attributes": self.detected_attributes,
            "confidence": sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            "reasoning": self.reasoning,
            "frames_processed": self.frames_processed,
            "processing_time": self.processing_time_seconds
        }
    
    def to_progressive_update(self) -> dict:
        """Convert to progressive update message."""
        avg_confidence = sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0
        
        return {
            "type": "progressive_update",
            "session_id": self.session_id,
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "attributes": self.detected_attributes,
            "confidence": avg_confidence,
            "frames_processed": self.frames_processed
        }