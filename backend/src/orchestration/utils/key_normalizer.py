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

        # NOTE: 'damage' is NOT mapped here - it can be either a dict (new format) or string/bool (old format)
        # Agent-specific processing in normalize_agent_attributes handles the conversion
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
                damage_val = normalized['damage']

                # NEW FORMAT: If damage is a dict (e.g., {"Hole": [x,y,x2,y2]}), parse it
                if isinstance(damage_val, dict):
                    logger.debug(f"KeyNormalizer: Parsing new damage format (dict)")

                    # EDGE CASE: Handle VLM returning {"type1": "Color fade", "bbox": [x,y,x2,y2]}
                    # O(1) check for this pattern
                    if 'bbox' in damage_val and len(damage_val) == 2:
                        damage_type_key = next((k for k in damage_val.keys() if k != 'bbox'), None)
                        if damage_type_key and isinstance(damage_val[damage_type_key], str):
                            actual_damage_type = damage_val[damage_type_key]
                            bbox = damage_val['bbox']
                            damage_val = {actual_damage_type: bbox}
                            logger.debug(f"KeyNormalizer: Restructured edge case format -> {damage_val}")

                    # Filter out null/empty values
                    valid_damages = {k: v for k, v in damage_val.items() if v is not None and k.lower() not in ['null', 'none']}

                    # Deduplicate based on bbox coordinates with O(1) complexity
                    if valid_damages:
                        seen_coords = set()  # O(1) lookup for coordinate tuples
                        unique_damages = {}

                        for damage_key, bbox in valid_damages.items():
                            # Convert bbox to tuple for hashable set lookup (O(1))
                            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                coord_tuple = tuple(float(x) for x in bbox[:4])

                                # Only add if coordinates are unique (O(1) set lookup)
                                if coord_tuple not in seen_coords:
                                    seen_coords.add(coord_tuple)
                                    unique_damages[damage_key] = bbox
                                else:
                                    logger.debug(f"KeyNormalizer: Skipping duplicate damage location {damage_key}: {bbox}")
                            else:
                                # Keep non-bbox format damages
                                unique_damages[damage_key] = bbox

                        valid_damages = unique_damages
                        logger.debug(f"KeyNormalizer: Deduplicated to {len(valid_damages)} unique damage locations")

                    if valid_damages:
                        # Group damages by base type (e.g., "Stain", "Stain_2", "Stain_3" -> "Stain": 3 instances)
                        # O(1) complexity per damage instance
                        damage_type_counts = {}

                        for damage_key in valid_damages.keys():
                            # Extract base type by removing "_N" suffix for indexed keys
                            if '_' in damage_key:
                                # Check if it's an indexed key (ends with _digit)
                                parts = damage_key.rsplit('_', 1)
                                if len(parts) == 2 and parts[1].isdigit():
                                    base_type = parts[0]
                                else:
                                    base_type = damage_key
                            else:
                                base_type = damage_key

                            # Count occurrences (O(1) dict operation)
                            damage_type_counts[base_type] = damage_type_counts.get(base_type, 0) + 1

                        # Create display format: "Stain (3 locations), Hole (1 location)"
                        display_parts = []
                        for dtype, count in damage_type_counts.items():
                            if count > 1:
                                display_parts.append(f"{dtype} ({count} locations)")
                            else:
                                display_parts.append(dtype)

                        normalized['damage_type'] = ', '.join(display_parts)

                        # Keep damage_locations with indexed keys for annotator (preserves all individual damages)
                        normalized['damage_locations'] = valid_damages

                        # Set is_damaged to True since there's valid damage
                        normalized['is_damaged'] = True
                        normalized['damaged'] = True

                        logger.debug(f"KeyNormalizer: Extracted {len(valid_damages)} damage instance(s) across {len(damage_type_counts)} type(s): {list(damage_type_counts.keys())}")
                    else:
                        # No valid damage
                        normalized['damage_type'] = None
                        normalized['damage_locations'] = None
                        normalized['is_damaged'] = False
                        normalized['damaged'] = False

                    # Remove the original 'damage' key
                    normalized.pop('damage')

                # OLD FORMAT: If damage is a string/bool, convert to is_damaged
                elif isinstance(damage_val, str):
                    normalized.pop('damage')
                    normalized['is_damaged'] = damage_val.lower() == 'yes'
                else:
                    normalized.pop('damage')
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