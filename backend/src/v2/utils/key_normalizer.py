"""Key normalizer for VLM response consistency"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class KeyNormalizer:
    """Normalizes keys from VLM responses to ensure consistency"""
    
    # Key mapping from various formats to standard format
    KEY_MAPPINGS = {
        # Initial classifier variations
        'neckline style': 'neckline',
        'neckline_style': 'neckline',
        'sleeve length': 'sleeve_type',
        'sleeve_length': 'sleeve_type',
        'closure type': 'closure_type',
        'closure_type': 'closure_type',
        
        # Common variations
        'item_type': 'item_type',
        'type': 'item_type',
        'garment_type': 'item_type',
        
        'colour': 'color',
        'color': 'color',
        
        'damage': 'is_damaged',
        'damaged': 'is_damaged',
        'has_damage': 'is_damaged',
        
        'damage_type': 'damage_type',
        'damage type': 'damage_type',
        'damage-type': 'damage_type',
        
        'pattern': 'pattern',
        'print': 'pattern',
        
        'brand': 'brand',
        'label': 'brand',
        'brand_name': 'brand',
        
        'size': 'size',
        'size_label': 'size',
        
        # Keep standard keys as-is
        'reasoning': 'reasoning',
        'confidence': 'confidence',
    }
    
    @classmethod
    def normalize_keys(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize keys in a dictionary to standard format
        
        Args:
            data: Dictionary with potentially inconsistent keys
            
        Returns:
            Dictionary with normalized keys
        """
        if not isinstance(data, dict):
            return data
            
        normalized = {}
        
        for key, value in data.items():
            # Convert to lowercase and strip whitespace
            clean_key = key.lower().strip()
            
            # Check if we have a mapping for this key
            normalized_key = cls.KEY_MAPPINGS.get(clean_key, clean_key)
            
            # Handle nested dictionaries recursively
            if isinstance(value, dict):
                normalized[normalized_key] = cls.normalize_keys(value)
            elif isinstance(value, list):
                # Handle lists of dictionaries
                normalized[normalized_key] = [
                    cls.normalize_keys(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # Handle boolean conversion for damage field
                if normalized_key == 'is_damaged' and isinstance(value, str):
                    normalized[normalized_key] = value.lower() == 'yes'
                else:
                    normalized[normalized_key] = value
                    
        return normalized
    
    @classmethod
    def normalize_agent_attributes(cls, agent_name: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize attributes based on the specific agent
        
        Args:
            agent_name: Name of the agent (initial_classifier, detail_extractor, etc.)
            attributes: Raw attributes from the agent
            
        Returns:
            Normalized attributes
        """
        # First do general normalization
        normalized = cls.normalize_keys(attributes)
        
        # Agent-specific processing
        if agent_name == 'initial_classifier':
            # Ensure all expected fields are present
            expected_fields = ['item_type', 'color', 'pattern', 'neckline', 'sleeve_type', 'closure_type']
            for field in expected_fields:
                if field not in normalized:
                    # Check for variations
                    for key in normalized:
                        if field in key or key in field:
                            normalized[field] = normalized.pop(key)
                            break
                            
        elif agent_name == 'damage_detector':
            # Normalize damage fields
            if 'damage' in normalized and 'is_damaged' not in normalized:
                damage_val = normalized.pop('damage')
                if isinstance(damage_val, str):
                    normalized['is_damaged'] = damage_val.lower() == 'yes'
                else:
                    normalized['is_damaged'] = bool(damage_val)
                    
        elif agent_name == 'detail_extractor':
            # Ensure brand and size are present
            if 'brand' not in normalized:
                normalized['brand'] = None
            if 'size' not in normalized:
                normalized['size'] = None
                
        return normalized
    
    @classmethod
    def merge_normalized_results(cls, *results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple normalized result dictionaries
        
        Args:
            *results: Variable number of result dictionaries
            
        Returns:
            Merged dictionary with all non-None values
        """
        merged = {}
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            for key, value in result.items():
                # Only update if value is not None and not empty
                if value is not None and value != "" and value != {}:
                    merged[key] = value
                elif key not in merged:
                    # Add the key with None if it doesn't exist
                    merged[key] = value
                    
        return merged