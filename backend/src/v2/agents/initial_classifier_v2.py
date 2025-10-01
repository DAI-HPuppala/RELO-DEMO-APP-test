"""Initial Classifier V2 - Stateful implementation with multi-inference"""

import logging
from typing import List, Dict, Any

from .stateful_base_agent import StatefulBaseAgent

logger = logging.getLogger(__name__)


class InitialClassifierV2(StatefulBaseAgent):
    """Initial garment classifier with timer-based multi-inference"""
    
    def __init__(self):
        super().__init__("initial_classifier", timer_seconds=4.0)
        
        self.expected_attributes = [
            "item_type",
            "color", 
            "pattern",
            "neckline",
            "sleeve_type",
            "closure_type"
        ]
        
        # Removed verbose initialization logging
    
    def get_expected_attributes(self) -> List[str]:
        """Get list of attributes this agent should detect"""
        return self.expected_attributes
    
    def validate_result(self, attributes: Dict[str, Any]) -> bool:
        """Validate that result contains expected attributes"""
        # Required attributes
        required = ["item_type", "color"]
        
        for attr in required:
            if attr not in attributes or attributes[attr] is None:
                logger.warning(f"InitialClassifierV2: Missing required attribute '{attr}'")
                return False
        
        # Check that at least some optional attributes are present
        optional_found = 0
        optional = ["pattern", "neckline", "sleeve_type", "closure_type"]
        
        for attr in optional:
            if attr in attributes and attributes[attr] is not None:
                optional_found += 1
        
        if optional_found < 2:
            logger.warning(f"InitialClassifierV2: Too few optional attributes found ({optional_found}/4)")
        
        return True
    
    async def process(self) -> Dict[str, Any]:
        """Main processing method"""
        logger.info("InitialClassifierV2 starting multi-inference processing")
        
        # Run timer-based inference loop
        result = await self.run_with_timer()
        
        # Log summary
        if "attributes" in result:
            attrs = result["attributes"]
            logger.info(f"InitialClassifierV2 detected: {attrs.get('item_type', 'unknown')} "
                       f"in {attrs.get('color', 'unknown')} color")
        
        return result
    
    async def capture_and_register_first_frame(self) -> bool:
        """Override to register first frame for damage detector sharing"""
        try:
            # Capture first frame
            if self.frame_provider:
                frame = await self.frame_provider()
                if frame is not None and self.frame_registry:
                    # Register as initial_first for sharing
                    frame_id = await self.frame_registry.register_initial_first_frame(frame)
                    logger.info(f"InitialClassifierV2 registered first frame for sharing: {frame_id}")
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to register first frame for sharing: {e}")
            return False
