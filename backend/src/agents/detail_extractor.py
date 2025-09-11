"""Detail Extractor Agent - Extracts size and brand information from clothing items."""
import json
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DetailExtractor(BaseAgent):
    """Agent responsible for extracting size and brand information."""
    
    def __init__(self):
        """Initialize the Detail Extractor agent."""
        super().__init__(name="DetailExtractor")
    
    def get_prompt(self) -> str:
        """Get the prompt for detail extraction.
        
        Returns:
            Prompt string for the VLM
        """
        return """You are analyzing a returned clothing item to extract size and brand information.

Look for and provide:
1. SIZE: Look carefully for size tags or labels (S, M, L, XL, XXL, numeric sizes like 32, 34, etc.)
2. BRAND: Identify any visible brand names, logos, or labels
3. REASONING: Explain what you can see regarding size and brand - where are the labels, what do they show, etc.

Respond in JSON format like this:
{
    "size": "L",
    "brand": "Nike",
    "reasoning": "I can see a size tag on the inside collar showing 'L' and the Nike swoosh logo is visible on the chest area."
}

If you cannot determine something, use "unknown" and explain why in the reasoning."""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the VLM response for size and brand.
        
        Args:
            response: Raw text response from VLM
            
        Returns:
            Structured attributes dictionary
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Process and validate the response
                attributes = {
                    "size": self._normalize_size(data.get("size")),
                    "brand": self._clean_text(data.get("brand")),
                    "reasoning": data.get("reasoning", "No reasoning provided")
                }
                
                return attributes
            else:
                logger.warning("No JSON found in response, using defaults")
                return self._default_attributes()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._default_attributes()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._default_attributes()
    
    def _normalize_size(self, size: Any) -> str:
        """Normalize size value.
        
        Args:
            size: Raw size value
            
        Returns:
            Normalized size string
        """
        if not size:
            return "unknown"
        
        size_str = str(size).upper().strip()
        
        # Handle "unknown" or similar
        if size_str.lower() in ["unknown", "null", "none", ""]:
            return "unknown"
        
        # Standard size mappings
        size_map = {
            "EXTRA SMALL": "XS",
            "SMALL": "S",
            "MEDIUM": "M",
            "LARGE": "L",
            "EXTRA LARGE": "XL",
            "XXL": "2XL",
            "XXXL": "3XL"
        }
        
        # Check for standard sizes
        for full, abbr in size_map.items():
            if full in size_str:
                return abbr
        
        # Return as-is if already in standard format or numeric
        if size_str in ["XS", "S", "M", "L", "XL", "2XL", "3XL"] or size_str.isdigit():
            return size_str
        
        return size_str if size_str else "unknown"
    
    def _clean_text(self, text: Any) -> str:
        """Clean and normalize text values.
        
        Args:
            text: Raw text value
            
        Returns:
            Cleaned text or 'unknown'
        """
        if not text:
            return "unknown"
        
        cleaned = str(text).strip().lower()
        if cleaned in ["unknown", "null", "none", ""]:
            return "unknown"
        
        return cleaned
    
    def _default_attributes(self) -> Dict[str, Any]:
        """Return default attributes when parsing fails.
        
        Returns:
            Dictionary with default values
        """
        return {
            "size": "unknown",
            "brand": "unknown",
            "reasoning": "Unable to parse response"
        }
    
    def _calculate_confidence(self, attributes: Dict[str, Any]) -> float:
        """Calculate confidence based on size and brand detection.
        
        Args:
            attributes: Parsed attributes
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        
        # Size is most important for returns processing
        if attributes.get("size") and attributes["size"] != "unknown":
            score += 0.5
        
        # Brand is also valuable
        if attributes.get("brand") and attributes["brand"] != "unknown":
            score += 0.4
        
        # Bonus if reasoning is comprehensive
        if attributes.get("reasoning") and len(attributes.get("reasoning", "")) > 50:
            score = min(score + 0.1, 1.0)
        
        return min(score, 1.0)