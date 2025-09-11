"""Integration test for single inference fallback"""

import pytest
from unittest.mock import Mock

class TestSingleInferenceFallback:
    """Integration tests for aggregation with minimal inferences"""
    
    @pytest.mark.asyncio
    async def test_single_inference_uses_as_final(self):
        """Test that single inference result is used as-is"""
        from backend.src.v2.services.aggregation import aggregate_results
        
        single_inference = Mock(
            attributes={"item_type": "shirt", "color": "red"},
            confidence=0.75
        )
        
        result = aggregate_results([single_inference])
        
        assert result.aggregation_method == "single"
        assert result.attributes == {"item_type": "shirt", "color": "red"}
    
    @pytest.mark.asyncio
    async def test_two_inferences_last_with_fallback(self):
        """Test two inferences use second with first fallback for nulls"""
        from backend.src.v2.services.aggregation import aggregate_results
        
        first = Mock(attributes={"item_type": "shirt", "color": "blue", "pattern": "striped"})
        second = Mock(attributes={"item_type": "shirt", "color": "red", "pattern": None})
        
        result = aggregate_results([first, second])
        
        assert result.aggregation_method == "last_with_fallback"
        assert result.attributes["color"] == "red"  # Use second
        assert result.attributes["pattern"] == "striped"  # Fallback to first
