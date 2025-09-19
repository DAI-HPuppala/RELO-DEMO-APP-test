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

        # Apply damage consistency override before validation
        self._apply_damage_override(attributes)

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

    def _apply_damage_override(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Apply damage detection override logic"""
        if not attributes:
            return attributes

        is_damaged = attributes.get("is_damaged")
        damage_type = attributes.get("damage_type")

        # Override logic: if is_damaged="yes" but damage_type indicates no damage
        if self._is_positive_damage(is_damaged) and self._is_no_damage_type(damage_type):
            logger.info(f"DamageDetectorV2: Overriding is_damaged from '{is_damaged}' to 'No' "
                       f"due to damage_type: '{damage_type}'")
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
