"""Aggregated Result model for combining multiple inferences"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class ConflictResolution:
    """Details about how a conflict between inferences was resolved"""
    
    attribute: str
    values_considered: List[Any]
    resolution_method: str  # "majority", "last_with_fallback", "single"
    final_value: Any
    reason: str
    
    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            "values_considered": self.values_considered,
            "resolution_method": self.resolution_method,
            "final_value": self.final_value,
            "reason": self.reason
        }


@dataclass
class AggregatedResult:
    """Result of aggregating multiple inferences for an agent"""
    
    agent_name: str
    total_inferences: int
    aggregation_method: str  # Method used for aggregation
    
    # Finalized attributes (agent-specific)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregation details
    attribute_votes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    conflicts_resolved: List[ConflictResolution] = field(default_factory=list)
    
    # Confidence metrics
    overall_confidence: float = 0.0
    per_attribute_confidence: Dict[str, float] = field(default_factory=dict)
    
    def add_conflict_resolution(self, attribute: str, values: List[Any], 
                              method: str, final_value: Any, reason: str) -> None:
        """Record how a conflict was resolved"""
        resolution = ConflictResolution(
            attribute=attribute,
            values_considered=values,
            resolution_method=method,
            final_value=final_value,
            reason=reason
        )
        self.conflicts_resolved.append(resolution)
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "total_inferences": self.total_inferences,
            "aggregation_method": self.aggregation_method,
            "attributes": self.attributes,
            "attribute_votes": self.attribute_votes,
            "conflicts_resolved": [c.to_dict() for c in self.conflicts_resolved],
            "overall_confidence": self.overall_confidence,
            "per_attribute_confidence": self.per_attribute_confidence
        }
