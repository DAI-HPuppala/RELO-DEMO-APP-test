"""Damage Detector V2 - Stateful implementation with frame sharing"""

import logging
from typing import List, Dict, Any

from .stateful_base_agent import StatefulBaseAgent

logger = logging.getLogger(__name__)


class DamageDetectorV2(StatefulBaseAgent):
    """Detect garment damage with frame sharing from initial classifier"""
    
    def __init__(self):
        super().__init__("damage_detector", timer_seconds=4.0)
        
        self.expected_attributes = [
            "is_damaged",
            "damage_type",
            "damage_severity",
            "damage_location",
            "repair_feasibility"
        ]
        
        # Removed verbose initialization logging
    
    def get_expected_attributes(self) -> List[str]:
        """Get list of attributes this agent should detect"""
        return self.expected_attributes
    
    def validate_result(self, attributes: Dict[str, Any]) -> bool:
        """Validate that result contains expected attributes"""
        # Must have is_damaged determination
        if "is_damaged" not in attributes:
            logger.warning("DamageDetectorV2: Missing is_damaged determination")
            return False
        
        # If damaged, should have type and severity
        if attributes.get("is_damaged"):
            if not attributes.get("damage_type"):
                logger.warning("DamageDetectorV2: Damage detected but no type specified")
                return False
            if not attributes.get("damage_severity"):
                logger.warning("DamageDetectorV2: Damage detected but no severity specified")
        
        return True
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method"""
        logger.info("DamageDetectorV2 starting multi-inference processing with frame sharing")
        
        # Note: First inference will automatically include shared frame from initial_classifier
        # This is handled by the base class's _capture_frames_with_retry method
        
        # Run timer-based inference loop
        result = await self.run_with_timer()
        
        # Log summary
        if "attributes" in result:
            attrs = result["attributes"]
            if attrs.get("is_damaged"):
                logger.info(f"DamageDetectorV2 found damage: Type={attrs.get('damage_type', 'N/A')}, "
                           f"Severity={attrs.get('damage_severity', 'N/A')}")
            else:
                logger.info("DamageDetectorV2: No damage detected")
        
        return result
