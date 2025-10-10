"""Final Classification model for compiled results"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ReasoningEntry:
    """Single entry in the reasoning sequence"""
    
    sequence_num: int  # 1, 2, 3...
    agent_name: str
    inference_num: int
    reasoning_text: str
    
    def to_dict(self) -> dict:
        return {
            "sequence_num": self.sequence_num,
            "agent_name": self.agent_name,
            "inference_num": self.inference_num,
            "reasoning_text": self.reasoning_text
        }


@dataclass
class FinalClassification:
    """Final compiled result from all agents"""
    
    # Core attributes from all agents
    item_type: str = ""  # From initial
    color: str = ""  # From initial
    pattern: Optional[str] = None  # From initial
    neckline: Optional[str] = None  # From initial
    sleeve_type: Optional[str] = None  # From initial
    closure_type: Optional[str] = None  # From initial
    
    brand: Optional[str] = None  # From detail
    size: Optional[str] = None  # From detail
    
    is_damaged: bool = False  # From damage
    damage_type: Optional[str] = None  # From damage
    damage_severity: Optional[str] = None  # From damage
    
    # Aggregated reasoning
    reasoning_sequence: List[ReasoningEntry] = field(default_factory=list)
    
    # Metadata
    overall_confidence: float = 0.0
    processing_time_ms: int = 0
    total_inferences: int = 0
    agents_used: List[str] = field(default_factory=list)
    
    def add_reasoning(self, agent: str, inference_num: int, text: str) -> None:
        """Add a reasoning entry to the sequence"""
        entry = ReasoningEntry(
            sequence_num=len(self.reasoning_sequence) + 1,
            agent_name=agent,
            inference_num=inference_num,
            reasoning_text=text
        )
        self.reasoning_sequence.append(entry)
    
    def to_dict(self) -> dict:
        return {
            "item_type": self.item_type,
            "color": self.color,
            "pattern": self.pattern,
            "neckline": self.neckline,
            "sleeve_type": self.sleeve_type,
            "closure_type": self.closure_type,
            "brand": self.brand,
            "size": self.size,
            "is_damaged": self.is_damaged,
            "damage_type": self.damage_type,
            "damage_severity": self.damage_severity,
            "reasoning_sequence": [r.to_dict() for r in self.reasoning_sequence],
            "overall_confidence": self.overall_confidence,
            "processing_time_ms": self.processing_time_ms,
            "total_inferences": self.total_inferences,
            "agents_used": self.agents_used
        }
