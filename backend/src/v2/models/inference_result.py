"""Inference Result model for single inference cycle"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4


@dataclass
class InferenceResult:
    """Result from a single inference cycle (single or multi-image)"""
    
    # Identification
    inference_id: str = field(default_factory=lambda: str(uuid4()))  # Unique identifier
    agent_name: str = ""  # Which agent produced this
    inference_num: int = 0  # Sequential number (1, 2, 3...)
    inference_type: str = "single_image"  # "single_image" or "multi_image"
    
    # Frame data
    frame_ids: List[str] = field(default_factory=list)  # IDs of frames used
    frame_count: int = 0  # Number of frames in this inference
    
    # VLM results
    attributes: Dict[str, Any] = field(default_factory=dict)  # Agent-specific attributes
    confidence: float = 0.0  # 0.0 to 1.0
    reasoning: str = ""  # VLM reasoning/explanation
    raw_response: str = ""  # Complete VLM response
    
    # Timing information
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0  # Processing time in milliseconds
    
    # Prompt information
    prompt_version: str = "base"  # "base" or "multi_angle"
    prompt_text: str = ""  # Actual prompt sent to VLM
    
    def __post_init__(self):
        """Validate and compute derived fields"""
        # Set frame count from frame_ids if not set
        if self.frame_ids and not self.frame_count:
            self.frame_count = len(self.frame_ids)
        
        # Determine inference type from frame count
        if self.frame_count == 1:
            self.inference_type = "single_image"
        elif self.frame_count > 1:
            self.inference_type = "multi_image"
        
        # Calculate duration if timestamps are set
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            self.duration_ms = int(duration * 1000)
        
        # Validate confidence range
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def add_frame(self, frame_id: str) -> None:
        """Add a frame to this inference"""
        if frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)
            self.frame_count = len(self.frame_ids)
            self.inference_type = "single_image" if self.frame_count == 1 else "multi_image"
    
    def set_prompt(self, prompt_text: str, is_multi_angle: bool = False) -> None:
        """Set the prompt used for this inference"""
        self.prompt_text = prompt_text
        self.prompt_version = "multi_angle" if is_multi_angle else "base"
    
    def complete(self, attributes: Dict[str, Any], confidence: float, reasoning: str, raw_response: str = "") -> None:
        """Mark inference as complete with results"""
        self.attributes = attributes
        self.confidence = max(0.0, min(1.0, confidence))
        self.reasoning = reasoning
        self.raw_response = raw_response or reasoning
        self.completed_at = datetime.now()
        
        # Recalculate duration
        duration = (self.completed_at - self.started_at).total_seconds()
        self.duration_ms = int(duration * 1000)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "inference_id": self.inference_id,
            "agent_name": self.agent_name,
            "inference_num": self.inference_num,
            "inference_type": self.inference_type,
            "frame_ids": self.frame_ids,
            "frame_count": self.frame_count,
            "attributes": self.attributes,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_ms": self.duration_ms,
            "prompt_version": self.prompt_version,
            "prompt_text": self.prompt_text
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceResult':
        """Create from dictionary"""
        result = cls(
            inference_id=data.get("inference_id", str(uuid4())),
            agent_name=data.get("agent_name", ""),
            inference_num=data.get("inference_num", 0),
            inference_type=data.get("inference_type", "single_image"),
            frame_ids=data.get("frame_ids", []),
            frame_count=data.get("frame_count", 0),
            attributes=data.get("attributes", {}),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            raw_response=data.get("raw_response", ""),
            duration_ms=data.get("duration_ms", 0),
            prompt_version=data.get("prompt_version", "base"),
            prompt_text=data.get("prompt_text", "")
        )
        
        # Parse timestamps
        if data.get("started_at"):
            result.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            result.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return result