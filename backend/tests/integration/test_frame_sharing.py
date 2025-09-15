"""Integration test for frame sharing between agents"""

import pytest
import numpy as np
from unittest.mock import Mock

class TestFrameSharing:
    """Integration tests for frame sharing protocol"""
    
    @pytest.mark.asyncio
    async def test_initial_first_frame_shared_with_damage(self):
        """Test initial's first frame is shared with damage detector"""
        from backend.src.v2.services.frame_registry import FrameRegistry
        
        registry = FrameRegistry("session123")
        
        # Initial classifier captures first frame
        first_frame = np.ones((480, 640, 3), dtype=np.uint8)
        frame_id = registry.register_frame(first_frame, "initial_classifier")
        registry.share_frame(frame_id, "initial_first")
        
        # Damage detector retrieves shared frame
        shared = registry.get_shared_frame("initial_first")
        
        assert shared is not None
        assert np.array_equal(shared.data, first_frame)
        assert shared.shared_with == ["damage_detector"]
