"""Frame Aggregator service for combining multiple inference results"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

from ..models.aggregated_result import AggregatedResult, ConflictResolution
from ..models.inference_result import InferenceResult

logger = logging.getLogger(__name__)


class FrameAggregator:
    """Aggregates multiple inference results based on inference count"""
    
    async def aggregate(self, agent_name: str, 
                       inferences: List[InferenceResult]) -> AggregatedResult:
        """Aggregate inference results based on count-specific rules"""
        
        result = AggregatedResult(
            agent_name=agent_name,
            total_inferences=len(inferences)
        )
        
        if not inferences:
            logger.warning(f"No inferences to aggregate for {agent_name}")
            return result
        
        # Determine aggregation method
        if len(inferences) == 1:
            result = self._aggregate_single(agent_name, inferences[0])
        elif len(inferences) == 2:
            result = self._aggregate_two(agent_name, inferences)
        else:
            result = self._aggregate_multiple(agent_name, inferences)
        
        # Calculate overall confidence
        result.overall_confidence = self._calculate_confidence(inferences)
        
        logger.info(f"Aggregated {len(inferences)} inferences for {agent_name} "
                   f"using {result.aggregation_method}")
        
        return result
    
    def _aggregate_single(self, agent_name: str, 
                         inference: InferenceResult) -> AggregatedResult:
        """Handle single inference - use as-is"""
        result = AggregatedResult(
            agent_name=agent_name,
            total_inferences=1,
            aggregation_method="single"
        )
        
        result.attributes = inference.attributes.copy()
        result.overall_confidence = inference.confidence
        
        for attr, value in inference.attributes.items():
            result.per_attribute_confidence[attr] = inference.confidence
        
        return result
    
    def _aggregate_two(self, agent_name: str, 
                      inferences: List[InferenceResult]) -> AggregatedResult:
        """Handle two inferences - last with first fallback for nulls"""
        result = AggregatedResult(
            agent_name=agent_name,
            total_inferences=2,
            aggregation_method="last_with_fallback"
        )
        
        first = inferences[0].attributes
        second = inferences[1].attributes
        
        # Get all unique attributes
        all_attrs = set(first.keys()) | set(second.keys())
        
        for attr in all_attrs:
            first_val = first.get(attr)
            second_val = second.get(attr)
            
            if second_val is not None:
                # Use second inference value
                final_val = second_val
                reason = "Using latest inference value"
            elif first_val is not None:
                # Fallback to first if second is null
                final_val = first_val
                reason = "Fallback to first inference (second was null)"
            else:
                # Both null
                final_val = None
                reason = "Both inferences returned null"
            
            result.attributes[attr] = final_val
            
            # Record resolution
            if first_val != second_val:
                result.add_conflict_resolution(
                    attr, [first_val, second_val],
                    "last_with_fallback", final_val, reason
                )
        
        return result
    
    def _aggregate_multiple(self, agent_name: str, 
                           inferences: List[InferenceResult]) -> AggregatedResult:
        """Handle 3+ inferences - majority voting"""
        result = AggregatedResult(
            agent_name=agent_name,
            total_inferences=len(inferences),
            aggregation_method="majority"
        )
        
        # Collect all attributes
        all_attrs = set()
        for inf in inferences:
            all_attrs.update(inf.attributes.keys())
        
        for attr in all_attrs:
            # Count occurrences of each value
            values = []
            for inf in inferences:
                if attr in inf.attributes:
                    values.append(inf.attributes[attr])
            
            if not values:
                result.attributes[attr] = None
                continue
            
            # Find most common value
            value_counts = Counter(values)
            result.attribute_votes[attr] = dict(value_counts)
            
            most_common = value_counts.most_common(1)[0]
            final_value = most_common[0]
            vote_count = most_common[1]
            
            result.attributes[attr] = final_value
            
            # Record if there was disagreement
            if len(value_counts) > 1:
                result.add_conflict_resolution(
                    attr, list(value_counts.keys()),
                    "majority", final_value,
                    f"Majority vote: {vote_count}/{len(values)} votes"
                )
            
            # Calculate per-attribute confidence
            result.per_attribute_confidence[attr] = vote_count / len(values)
        
        return result
    
    def _calculate_confidence(self, inferences: List[InferenceResult]) -> float:
        """Calculate overall confidence from multiple inferences"""
        if not inferences:
            return 0.0
        
        # Weighted average based on inference number (later = higher weight)
        total_weight = 0
        weighted_sum = 0
        
        for i, inf in enumerate(inferences):
            weight = i + 1  # Later inferences get more weight
            weighted_sum += inf.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
