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
            "damage_type"  # Only damage_type is returned by the model
        ]
        
        # Removed verbose initialization logging
    
    def get_expected_attributes(self) -> List[str]:
        """Get list of attributes this agent should detect"""
        return self.expected_attributes
    
    def validate_result(self, attributes: Dict[str, Any]) -> bool:
        """Validate that result contains expected attributes"""
        # Must have damage_type (can be null)
        if "damage_type" not in attributes:
            logger.warning("DamageDetectorV2: Missing damage_type")
            return False

        # Derive is_damaged from damage_type
        self._derive_is_damaged(attributes)

        return True
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method"""
        logger.info("DamageDetectorV2 starting multi-inference processing with frame sharing")

        # Note: First inference will automatically include shared frame from initial_classifier
        # This is handled by the base class's _capture_frames_with_retry method

        # Run timer-based inference loop
        result = await self.run_with_timer()

        # Derive is_damaged from damage_type for all results
        if "attributes" in result:
            attrs = result["attributes"]

            # Check if attributes are empty
            if not attrs:
                logger.warning("DamageDetectorV2: No attributes returned from model!")
                # Initialize with default values
                attrs["damage_type"] = None
                attrs["is_damaged"] = False
                result["attributes"] = attrs
            else:
                # Log raw output from model (before derivation)
                logger.info("="*60)
                logger.info("DamageDetectorV2 RAW OUTPUT FROM MODEL:")
                logger.info(f"  - damage_type (from model): {attrs.get('damage_type', 'NOT FOUND')}")
                logger.info("="*60)

                # Derive is_damaged from damage_type
                self._derive_is_damaged(attrs)

                # Log derived attributes after processing
                logger.info("DamageDetectorV2 DERIVED ATTRIBUTES:")
                logger.info(f"  - damaged (derived): {attrs.get('damaged', False)}")
                logger.info(f"  - is_damaged (derived): {attrs.get('is_damaged', False)}")
                logger.info(f"  - damage_type (final): {attrs.get('damage_type', None)}")

                # Log final summary
                if attrs.get("is_damaged"):
                    logger.info(f"DamageDetectorV2 RESULT: ✅ Damage found - Type={attrs.get('damage_type', 'N/A')}")
                else:
                    logger.info("DamageDetectorV2 RESULT: ✅ No damage detected")
        else:
            logger.warning("DamageDetectorV2: No 'attributes' key in result!")

        return result

    def _derive_is_damaged(self, attributes: Dict[str, Any]) -> None:
        """Derive is_damaged/damaged attribute from damage_type value"""
        if not attributes:
            return

        damage_type = attributes.get("damage_type")

        # Check if damage_type indicates no damage
        if self._indicates_no_damage(damage_type):
            attributes["is_damaged"] = False
            attributes["damaged"] = False  # Also set 'damaged' for consistency
        else:
            # There is damage
            attributes["is_damaged"] = True
            attributes["damaged"] = True  # Also set 'damaged' for consistency
            # Extract location from damage_type if present (e.g., "stain on front")
            if damage_type and isinstance(damage_type, str):
                # Try to extract location from damage_type description
                damage_parts = damage_type.lower()
                if " on " in damage_parts or " at " in damage_parts or " in " in damage_parts:
                    # Has location info
                    attributes["damage_location"] = damage_type

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
