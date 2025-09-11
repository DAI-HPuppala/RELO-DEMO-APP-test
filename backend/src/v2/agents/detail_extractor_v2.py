"""Detail Extractor V2 - Stateful implementation with multi-inference"""

import logging
from typing import List, Dict, Any

from .stateful_base_agent import StatefulBaseAgent

logger = logging.getLogger(__name__)


class DetailExtractorV2(StatefulBaseAgent):
    """Extract garment details with timer-based multi-inference"""
    
    def __init__(self):
        super().__init__("detail_extractor", timer_seconds=3.0)
        
        self.expected_attributes = [
            "brand",
            "size",
            "material",
            "care_instructions",
            "style_details",
            "condition_notes"
        ]
        
        logger.info("DetailExtractorV2 initialized with 3s timer")
    
    def get_expected_attributes(self) -> List[str]:
        """Get list of attributes this agent should detect"""
        return self.expected_attributes
    
    def validate_result(self, attributes: Dict[str, Any]) -> bool:
        """Validate that result contains expected attributes"""
        # At least brand or size should be detected
        if not attributes.get("brand") and not attributes.get("size"):
            logger.warning("DetailExtractorV2: Neither brand nor size detected")
            return False
        
        # Material is important
        if not attributes.get("material"):
            logger.warning("DetailExtractorV2: No material information detected")
        
        return True
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method"""
        logger.info("DetailExtractorV2 starting multi-inference processing")
        
        # Run timer-based inference loop
        result = await self.run_with_timer()
        
        # Log summary
        if "attributes" in result:
            attrs = result["attributes"]
            logger.info(f"DetailExtractorV2 extracted: Brand={attrs.get('brand', 'N/A')}, "
                       f"Size={attrs.get('size', 'N/A')}, Material={attrs.get('material', 'N/A')}")
        
        return result
