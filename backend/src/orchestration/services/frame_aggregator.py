"""Frame Aggregator service for combining multiple inference results"""

import logging
import time
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

from ..models.aggregated_result import AggregatedResult, ConflictResolution
from ..models.inference_result import InferenceResult
from ..utils.key_normalizer import KeyNormalizer

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
        """Handle single inference - use as-is with normalization"""
        result = AggregatedResult(
            agent_name=agent_name,
            total_inferences=1,
            aggregation_method="single"
        )

        # Normalize the attributes for this agent before using them
        # Note: normalize_agent_attributes() creates a new dict, so no .copy() needed (O(1) optimization)
        normalized = KeyNormalizer.normalize_agent_attributes(agent_name, inference.attributes)
        result.attributes = normalized
        result.overall_confidence = inference.confidence

        for attr, value in normalized.items():
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

        # Normalize both inference results
        # Note: normalize_agent_attributes() creates new dicts, so no .copy() needed (O(1) optimization)
        first = KeyNormalizer.normalize_agent_attributes(agent_name, inferences[0].attributes)
        second = KeyNormalizer.normalize_agent_attributes(agent_name, inferences[1].attributes)

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
        """Handle 3+ inferences - majority voting (optimized single-pass)"""
        start_time = time.perf_counter()

        result = AggregatedResult(
            agent_name=agent_name,
            total_inferences=len(inferences),
            aggregation_method="majority"
        )

        # Single-pass collection: build attribute votes in one iteration
        attribute_values = defaultdict(list)

        # Collect all attribute values in a single pass (with normalization)
        for inf in inferences:
            # Normalize each inference result before aggregating
            # Note: normalize_agent_attributes() creates a new dict, so no .copy() needed (O(1) optimization)
            normalized = KeyNormalizer.normalize_agent_attributes(agent_name, inf.attributes)
            for attr, value in normalized.items():
                attribute_values[attr].append(value)

        # Process each attribute's votes
        for attr, values in attribute_values.items():
            if not values:
                result.attributes[attr] = None
                continue

            # Count votes efficiently
            value_counts = Counter(values)
            result.attribute_votes[attr] = dict(value_counts)

            # Get the most common value
            most_common = value_counts.most_common(1)[0]
            final_value = most_common[0]
            vote_count = most_common[1]

            result.attributes[attr] = final_value

            # Early exit optimization: skip conflict recording if unanimous
            total_votes = len(values)
            if vote_count == total_votes:
                # Unanimous agreement - no conflict to record
                result.per_attribute_confidence[attr] = 1.0
            else:
                # Record disagreement
                result.add_conflict_resolution(
                    attr, list(value_counts.keys()),
                    "majority", final_value,
                    f"Majority vote: {vote_count}/{total_votes} votes"
                )
                result.per_attribute_confidence[attr] = vote_count / total_votes

        # Performance logging
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 10:  # Only log if aggregation took more than 10ms
            logger.debug(f"Aggregation for {agent_name} took {elapsed_ms:.2f}ms for {len(inferences)} inferences")

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
        """Validate damage detection consistency and derive is_damaged from damage_type"""
        if not attributes:
            return attributes

        damage_type = attributes.get("damage_type")

        # Derive is_damaged from damage_type if not already set
        if "is_damaged" not in attributes:
            if self._indicates_no_damage(damage_type):
                attributes["is_damaged"] = False
                # Clear damage-related fields for consistency
                attributes["damage_severity"] = None
                attributes["damage_location"] = None
            else:
                attributes["is_damaged"] = True
                # Optionally set severity based on damage_type
                if damage_type and isinstance(damage_type, str):
                    damage_lower = damage_type.lower()
                    if any(word in damage_lower for word in ["major", "large", "severe", "significant"]):
                        attributes["damage_severity"] = "major"
                    elif any(word in damage_lower for word in ["minor", "small", "slight", "tiny"]):
                        attributes["damage_severity"] = "minor"

        return attributes

    def _indicates_no_damage(self, value) -> bool:
        """Check if damage_type value indicates no damage"""
        if not value:
            return True

        value_str = str(value).lower().strip()

        # Common no-damage indicators
        no_damage_indicators = [
            'null', 'none', 'nil', 'nill', 'undefined',
            'no damage', 'not damaged', 'no', 'clean',
            'good condition', 'perfect', 'pristine', 'n/a',
            'not applicable', 'na', 'nothing', 'empty', ''
        ]

        # Check exact matches
        if value_str in no_damage_indicators:
            return True

        # Check if starts with "no" or "not"
        if value_str.startswith(('no ', 'not ', 'no-', 'not-')):
            return True

        return False
