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
        
        if not inferences:
            logger.warning(f"No inferences to aggregate for {agent_name}")
            # Return empty result with proper initialization
            result = AggregatedResult(
                agent_name=agent_name,
                total_inferences=0,
                aggregation_method="none"
            )
            return result
        
        # Determine aggregation method and delegate
        if len(inferences) == 1:
            result = self._aggregate_single(agent_name, inferences[0])
        elif len(inferences) == 2:
            result = self._aggregate_two(agent_name, inferences)
        else:
            result = self._aggregate_multiple(agent_name, inferences)
        
        # Calculate overall confidence
        result.overall_confidence = self._calculate_confidence(inferences)

        # Apply damage consistency validation for damage_detector
        if agent_name == "damage_detector":
            result.attributes = self._validate_damage_consistency(result.attributes)

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

    def _validate_damage_consistency(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate damage detection consistency and apply overrides"""
        if not attributes:
            return attributes

        is_damaged = attributes.get("is_damaged")
        damage_type = attributes.get("damage_type")

        # Override logic: if is_damaged="yes" but damage_type indicates no damage
        if self._is_positive_damage(is_damaged) and self._is_no_damage_type(damage_type):
            logger.info(f"FrameAggregator: Overriding is_damaged from '{is_damaged}' to 'No' "
                       f"due to damage_type: '{damage_type}'")
            attributes = attributes.copy()  # Don't modify original
            attributes["is_damaged"] = "No"
            # Also clear damage-related fields for consistency
            attributes["damage_severity"] = None
            attributes["damage_location"] = None
            attributes["repair_feasibility"] = None

        return attributes

    def _is_positive_damage(self, value) -> bool:
        """Check if value indicates damage"""
        if not value:
            return False
        return str(value).lower() in ['yes', 'true', '1', 'damaged']

    def _is_no_damage_type(self, value) -> bool:
        """Check if damage_type indicates no damage"""
        if not value:
            return True
        value_lower = str(value).lower()
        return (value_lower in ['null', 'undefined', 'none', 'no', 'no damage', 'clean'] or
                'no' in value_lower or 'none' in value_lower)
