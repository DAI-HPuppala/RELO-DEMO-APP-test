"""Damage Detector Agent - Identifies if item is damaged and explains the damage."""
import json
import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DamageDetector(BaseAgent):
    """Agent responsible for detecting damage and providing damage explanation."""
    
    def __init__(self):
        """Initialize the Damage Detector agent."""
        super().__init__(name="DamageDetector")
    
    def get_prompt(self) -> str:
        """Get the prompt for damage detection.
        
        Returns:
            Prompt string for the VLM
        """
        return """You are analyzing a returned clothing item to detect any damage or defects.

Carefully examine the item and provide:
1. IS_DAMAGED: Is the item damaged in any way? (yes/no)
2. DAMAGE_TYPE_EXPLANATION: If damaged, describe the type and location of damage (tears, stains, holes, broken parts, etc.). If not damaged, state "No damage detected"
3. REASONING: Explain your assessment - what did you look for, what did you find or not find

Respond in JSON format like this:
{
    "is_damaged": "yes",
    "damage_type_explanation": "There is a small tear on the left sleeve near the elbow, approximately 2 inches long. Also visible coffee stain on the front.",
    "reasoning": "I examined the entire garment and found two issues: a tear on the left sleeve and a stain on the front. The tear appears to be from wear or snagging, and the stain looks like a beverage spill."
}

Or if no damage:
{
    "is_damaged": "no",
    "damage_type_explanation": "No damage detected",
    "reasoning": "I thoroughly examined the garment including seams, fabric, and all visible areas. The item appears to be in good condition with no tears, stains, or structural damage."
}"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the VLM response for damage detection.
        
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
                is_damaged = str(data.get("is_damaged", "unknown")).lower()
                if is_damaged in ["yes", "true", "1"]:
                    is_damaged = "yes"
                elif is_damaged in ["no", "false", "0"]:
                    is_damaged = "no"
                else:
                    is_damaged = "unknown"
                
                attributes = {
                    "is_damaged": is_damaged,
                    "damage_type_explanation": data.get("damage_type_explanation", "Unable to determine"),
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
    
    def _default_attributes(self) -> Dict[str, Any]:
        """Return default attributes when parsing fails.
        
        Returns:
            Dictionary with default values
        """
        return {
            "is_damaged": "unknown",
            "damage_type_explanation": "Unable to determine",
            "reasoning": "Unable to parse response"
        }
    
    def _calculate_confidence(self, attributes: Dict[str, Any]) -> float:
        """Calculate confidence based on damage detection completeness.
        
        Args:
            attributes: Parsed attributes
            
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        
        # Clear damage determination is most important
        if attributes.get("is_damaged") in ["yes", "no"]:
            score += 0.5
        
        # Good explanation adds confidence
        explanation = attributes.get("damage_type_explanation", "")
        if explanation and len(explanation) > 20:
            score += 0.3
        
        # Comprehensive reasoning adds confidence
        reasoning = attributes.get("reasoning", "")
        if reasoning and len(reasoning) > 50:
            score += 0.2
        
        return min(score, 1.0)