"""Initial Classifier Agent - First pass classification of clothing items."""
import json
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class InitialClassifier(BaseAgent):
    """Agent responsible for initial classification of clothing type and basic attributes."""
    
    def __init__(self):
        """Initialize the Initial Classifier agent."""
        super().__init__(name="InitialClassifier")
    
    def get_prompt(self) -> str:
        """Get the prompt for initial classification.
        
        Returns:
            Prompt string for the VLM
        """
        return """You are analyzing a returned clothing item. Provide a detailed analysis with reasoning for each attribute:

Analyze and provide:
1. TYPE: What type of clothing is this? (shirt, pants, dress, jacket, etc.)
2. COLOR: What color(s) do you see?
3. PATTERN: What pattern does it have? (solid, striped, checkered, floral, etc.)
4. NECKLINE: What type of neckline? (round, v-neck, collar, turtle, etc.)
5. CLOSURE: How does it close? (buttons, zipper, pullover, snap, etc.)
6. SLEEVE: What type of sleeves? (long, short, sleeveless, 3/4, etc.)
7. REASONING: Explain what you see and why you made these determinations

Respond in JSON format like this:
{
    "type": "shirt",
    "color": "blue with white stripes",
    "pattern": "striped",
    "neckline": "collar",
    "closure": "buttons",
    "sleeve": "long",
    "reasoning": "I can see a blue and white striped shirt with a collar, button-down front closure, and long sleeves. The stripes are vertical and evenly spaced."
}

If you cannot determine something, use "unknown" and explain why in the reasoning."""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the VLM response for initial classification.
        
        Args:
            response: Raw text response from VLM
            
        Returns:
            Structured attributes dictionary
        """
        try:
            # Try to extract JSON from response
            # VLM might include extra text, so we look for JSON structure
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                # Extract the new attributes
                attributes = {
                    "type": self._clean_text(data.get("type", "unknown")),
                    "color": self._clean_text(data.get("color", "unknown")),
                    "pattern": self._clean_text(data.get("pattern", "unknown")),
                    "neckline": self._clean_text(data.get("neckline", "unknown")),
                    "closure": self._clean_text(data.get("closure", "unknown")),
                    "sleeve": self._clean_text(data.get("sleeve", "unknown")),
                    "reasoning": data.get("reasoning", "No reasoning provided")
                }
                
                return attributes
            else:
                # Fallback parsing if JSON not found
                logger.warning("No JSON found in response, using defaults")
                return self._default_attributes()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._default_attributes()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._default_attributes()
    
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
        if cleaned in ["null", "none", ""]:
            return "unknown"
        
        return cleaned
    
    def _default_attributes(self) -> Dict[str, Any]:
        """Return default attributes when parsing fails.
        
        Returns:
            Dictionary with default values
        """
        return {
            "type": "unknown",
            "color": "unknown",
            "pattern": "unknown",
            "neckline": "unknown",
            "closure": "unknown",
            "sleeve": "unknown",
            "reasoning": "Unable to parse response"
        }
    
    def _calculate_confidence(self, attributes: Dict[str, Any]) -> float:
        """Calculate confidence based on initial classification completeness.
        
        Args:
            attributes: Parsed attributes
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        weights = {
            "type": 0.25,  # Most important
            "color": 0.15,
            "pattern": 0.15,
            "neckline": 0.15,
            "closure": 0.15,
            "sleeve": 0.15
        }
        
        for key, weight in weights.items():
            value = attributes.get(key)
            if value and value != "unknown":
                score += weight
        
        # Bonus if reasoning is comprehensive
        if attributes.get("reasoning") and len(attributes.get("reasoning", "")) > 50:
            score = min(score + 0.1, 1.0)
        
        return min(score, 1.0)