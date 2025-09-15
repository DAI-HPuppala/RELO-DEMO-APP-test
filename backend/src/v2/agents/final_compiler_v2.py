"""Final Compiler V2 - Aggregates all agent results with reasoning"""

import logging
from typing import List, Dict, Any

from ..models.final_classification import FinalClassification
from ..utils.key_normalizer import KeyNormalizer

logger = logging.getLogger(__name__)


class FinalCompilerV2:
    """Compile and aggregate results from all agents"""
    
    def __init__(self):
        self.agent_name = "final_compiler"
        self.timer_seconds = 1.0  # Quick aggregation only
        
        # Removed verbose initialization logging
    
    async def compile_results(self, agent_results: Dict[str, Dict[str, Any]]) -> FinalClassification:
        """Compile results from all agents into final classification"""
        
        logger.info("FinalCompilerV2 compiling results from all agents")
        
        final = FinalClassification()
        
        # Extract and normalize attributes from each agent
        if "initial_classifier" in agent_results:
            raw_initial = agent_results["initial_classifier"].get("attributes", {})
            # Normalize keys for initial classifier
            initial = KeyNormalizer.normalize_agent_attributes("initial_classifier", raw_initial)
            
            final.item_type = initial.get("item_type", "")
            final.color = initial.get("color", "")
            final.pattern = initial.get("pattern")
            final.neckline = initial.get("neckline")
            final.sleeve_type = initial.get("sleeve_type")
            final.closure_type = initial.get("closure_type")
            
            # Add reasoning
            if initial:
                final.add_reasoning(
                    "initial_classifier", 
                    agent_results["initial_classifier"].get("total_inferences", 1),
                    f"Identified {final.item_type} in {final.color} color"
                )
        
        if "detail_extractor" in agent_results:
            raw_detail = agent_results["detail_extractor"].get("attributes", {})
            # Normalize keys for detail extractor
            detail = KeyNormalizer.normalize_agent_attributes("detail_extractor", raw_detail)
            
            final.brand = detail.get("brand")
            final.size = detail.get("size")
            
            # Add reasoning
            if detail:
                final.add_reasoning(
                    "detail_extractor",
                    agent_results["detail_extractor"].get("total_inferences", 1),
                    f"Extracted brand: {final.brand or 'unknown'}, size: {final.size or 'unknown'}"
                )
        
        if "damage_detector" in agent_results:
            raw_damage = agent_results["damage_detector"].get("attributes", {})
            # Normalize keys for damage detector
            damage = KeyNormalizer.normalize_agent_attributes("damage_detector", raw_damage)
            
            final.is_damaged = damage.get("is_damaged", False)
            final.damage_type = damage.get("damage_type")
            final.damage_severity = damage.get("damage_severity")
            
            # Add reasoning
            if damage:
                damage_status = "damaged" if final.is_damaged else "undamaged"
                final.add_reasoning(
                    "damage_detector",
                    agent_results["damage_detector"].get("total_inferences", 1),
                    f"Item is {damage_status}" + 
                    (f" ({final.damage_type}, {final.damage_severity})" if final.is_damaged else "")
                )
        
        # Calculate overall confidence
        confidences = []
        total_inferences = 0
        
        for agent_name, result in agent_results.items():
            if "confidence" in result:
                confidences.append(result["confidence"])
            if "total_inferences" in result:
                total_inferences += result["total_inferences"]
        
        if confidences:
            final.overall_confidence = sum(confidences) / len(confidences)
        
        final.total_inferences = total_inferences
        final.agents_used = list(agent_results.keys())
        
        # Add final aggregated reasoning
        final.add_reasoning(
            "final_compiler",
            1,
            f"Compiled classification: {final.item_type} ({final.color}) - " +
            f"{'Damaged' if final.is_damaged else 'Good condition'}"
        )
        
        logger.info(f"FinalCompilerV2 completed: {final.item_type} - "
                   f"Total inferences: {final.total_inferences}, "
                   f"Confidence: {final.overall_confidence:.2f}")
        
        return final
    
    async def process(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process and return compiled results"""
        final = await self.compile_results(agent_results)
        return final.to_dict()
