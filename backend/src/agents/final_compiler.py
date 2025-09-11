"""Final Compiler Agent - Synthesizes all agent results into final classification."""
import json
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FinalCompiler(BaseAgent):
    """Agent responsible for compiling and reconciling results from all other agents."""
    
    def __init__(self):
        """Initialize the Final Compiler agent."""
        super().__init__(name="FinalCompiler")
        self.previous_results = []
    
    def set_previous_results(self, results: List[Dict[str, Any]]):
        """Set the results from previous agents for compilation.
        
        Args:
            results: List of results from other agents
        """
        self.previous_results = results
    
    def get_prompt(self) -> str:
        """Get the prompt for final compilation.
        
        Returns:
            Prompt string for the VLM
        """
        # Build context from previous results
        context = self._build_context()
        
        return f"""You are the final reviewer for a returned clothing item classification. 
Based on the image and the following analysis from other agents, provide a final comprehensive classification:

PREVIOUS ANALYSIS:
{context}

Now provide your FINAL CLASSIFICATION with:
1. FINAL_TYPE: Confirmed clothing type
2. FINAL_CATEGORY: Confirmed category
3. FINAL_CONDITION: Overall condition (excellent/good/fair/poor)
4. RECOMMENDED_ACTION: What to do with the item (resell/repair-and-resell/discount/recycle/return-to-vendor)
5. PRICING_TIER: Suggested pricing tier (full-price/discounted-25/discounted-50/clearance)
6. CONFIDENCE_LEVEL: Your confidence in this classification (high/medium/low)
7. NOTES: Any important observations or conflicts in the data

Respond in JSON format like this:
{{
    "final_type": "shirt",
    "final_category": "casual",
    "final_condition": "good",
    "recommended_action": "resell",
    "pricing_tier": "full-price",
    "confidence_level": "high",
    "notes": "Item in good condition with minor wear"
}}

Resolve any conflicts between agents using your visual analysis."""
    
    def _build_context(self) -> str:
        """Build context string from previous agent results.
        
        Returns:
            Formatted context string
        """
        if not self.previous_results:
            return "No previous analysis available."
        
        context_parts = []
        for result in self.previous_results:
            agent_name = result.get("agent_name", "Unknown")
            attributes = result.get("attributes", {})
            confidence = result.get("confidence", 0)
            
            context_parts.append(f"\n{agent_name} (confidence: {confidence:.2f}):")
            for key, value in attributes.items():
                if value is not None and value != [] and value != "unknown":
                    context_parts.append(f"  - {key}: {value}")
        
        return "\n".join(context_parts) if context_parts else "No valid analysis available."
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the VLM response for final compilation.
        
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
                    "final_type": self._clean_text(data.get("final_type", "unknown")),
                    "final_category": self._clean_text(data.get("final_category", "unknown")),
                    "final_condition": self._normalize_condition(data.get("final_condition")),
                    "recommended_action": self._normalize_action(data.get("recommended_action")),
                    "pricing_tier": self._normalize_pricing(data.get("pricing_tier")),
                    "confidence_level": self._normalize_confidence_level(data.get("confidence_level")),
                    "notes": self._clean_text(data.get("notes", ""))
                }
                
                # Add reconciliation data
                attributes["reconciliation"] = self._reconcile_conflicts()
                
                return attributes
            else:
                logger.warning("No JSON found in response, using synthesis")
                return self._synthesize_from_previous()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return self._synthesize_from_previous()
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return self._synthesize_from_previous()
    
    def _clean_text(self, text: Any) -> str:
        """Clean and normalize text values.
        
        Args:
            text: Raw text value
            
        Returns:
            Cleaned text
        """
        if not text:
            return "unknown"
        
        cleaned = str(text).strip().lower()
        return cleaned if cleaned else "unknown"
    
    def _normalize_condition(self, condition: str) -> str:
        """Normalize condition to standard categories.
        
        Args:
            condition: Raw condition
            
        Returns:
            Normalized condition
        """
        if not condition:
            return "unknown"
        
        cond = str(condition).lower().strip()
        
        if any(word in cond for word in ["excellent", "perfect", "new"]):
            return "excellent"
        elif any(word in cond for word in ["good", "fine"]):
            return "good"
        elif any(word in cond for word in ["fair", "okay", "moderate"]):
            return "fair"
        elif any(word in cond for word in ["poor", "bad", "damaged"]):
            return "poor"
        else:
            return cond if cond else "unknown"
    
    def _normalize_action(self, action: str) -> str:
        """Normalize recommended action.
        
        Args:
            action: Raw action
            
        Returns:
            Normalized action
        """
        if not action:
            return "review"
        
        act = str(action).lower().strip()
        
        if "resell" in act or "sell" in act:
            if "repair" in act:
                return "repair-and-resell"
            else:
                return "resell"
        elif "discount" in act:
            return "discount"
        elif "recycle" in act or "dispose" in act:
            return "recycle"
        elif "return" in act or "vendor" in act:
            return "return-to-vendor"
        else:
            return act if act else "review"
    
    def _normalize_pricing(self, pricing: str) -> str:
        """Normalize pricing tier.
        
        Args:
            pricing: Raw pricing tier
            
        Returns:
            Normalized pricing tier
        """
        if not pricing:
            return "review"
        
        price = str(pricing).lower().strip()
        
        if "full" in price:
            return "full-price"
        elif "25" in price or "quarter" in price:
            return "discounted-25"
        elif "50" in price or "half" in price:
            return "discounted-50"
        elif "clearance" in price or "final" in price:
            return "clearance"
        else:
            return price if price else "review"
    
    def _normalize_confidence_level(self, level: str) -> str:
        """Normalize confidence level.
        
        Args:
            level: Raw confidence level
            
        Returns:
            Normalized level
        """
        if not level:
            return "medium"
        
        lvl = str(level).lower().strip()
        
        if "high" in lvl:
            return "high"
        elif "low" in lvl:
            return "low"
        else:
            return "medium"
    
    def _reconcile_conflicts(self) -> Dict[str, Any]:
        """Reconcile conflicts between agent results.
        
        Returns:
            Reconciliation summary
        """
        if not self.previous_results:
            return {"conflicts": [], "agreements": []}
        
        conflicts = []
        agreements = []
        
        # Check for conflicts in key attributes
        types = []
        categories = []
        conditions = []
        
        for result in self.previous_results:
            attrs = result.get("attributes", {})
            
            # Collect values
            if "clothing_type" in attrs:
                types.append(attrs["clothing_type"])
            if "category" in attrs:
                categories.append(attrs["category"])
            if "wear_level" in attrs or "condition_score" in attrs:
                conditions.append(attrs.get("wear_level") or attrs.get("condition_score"))
        
        # Check for conflicts
        if len(set(types)) > 1:
            conflicts.append(f"Type conflict: {types}")
        elif types:
            agreements.append(f"Type agreed: {types[0]}")
        
        if len(set(categories)) > 1:
            conflicts.append(f"Category conflict: {categories}")
        elif categories:
            agreements.append(f"Category agreed: {categories[0]}")
        
        return {
            "conflicts": conflicts,
            "agreements": agreements,
            "resolution_method": "visual_verification" if conflicts else "consensus"
        }
    
    def _synthesize_from_previous(self) -> Dict[str, Any]:
        """Synthesize final result from previous agent results when VLM fails.
        
        Returns:
            Synthesized attributes
        """
        if not self.previous_results:
            return self._default_attributes()
        
        # Aggregate from previous results
        final_type = "unknown"
        final_category = "unknown"
        condition_scores = []
        resellable_votes = []
        
        for result in self.previous_results:
            attrs = result.get("attributes", {})
            
            # Get clothing type (prefer InitialClassifier)
            if result.get("agent_name") == "InitialClassifier" and attrs.get("clothing_type"):
                final_type = attrs["clothing_type"]
            
            # Get category
            if attrs.get("category"):
                final_category = attrs["category"]
            
            # Collect condition data
            if "condition_score" in attrs:
                condition_scores.append(attrs["condition_score"])
            
            # Collect resellable votes
            if "resellable" in attrs:
                resellable_votes.append(attrs["resellable"])
        
        # Determine final condition
        avg_condition = sum(condition_scores) / len(condition_scores) if condition_scores else 50
        if avg_condition >= 80:
            final_condition = "excellent"
        elif avg_condition >= 60:
            final_condition = "good"
        elif avg_condition >= 40:
            final_condition = "fair"
        else:
            final_condition = "poor"
        
        # Determine action based on condition and resellable votes
        if "no" in resellable_votes:
            recommended_action = "recycle"
        elif "yes-with-repair" in resellable_votes:
            recommended_action = "repair-and-resell"
        elif final_condition in ["excellent", "good"]:
            recommended_action = "resell"
        else:
            recommended_action = "discount"
        
        # Determine pricing
        if final_condition == "excellent":
            pricing_tier = "full-price"
        elif final_condition == "good":
            pricing_tier = "discounted-25"
        elif final_condition == "fair":
            pricing_tier = "discounted-50"
        else:
            pricing_tier = "clearance"
        
        return {
            "final_type": final_type,
            "final_category": final_category,
            "final_condition": final_condition,
            "recommended_action": recommended_action,
            "pricing_tier": pricing_tier,
            "confidence_level": "low",  # Low confidence since synthesized
            "notes": "Synthesized from agent results without visual confirmation",
            "reconciliation": self._reconcile_conflicts()
        }
    
    def _default_attributes(self) -> Dict[str, Any]:
        """Return default attributes when everything fails.
        
        Returns:
            Dictionary with default values
        """
        return {
            "final_type": "unknown",
            "final_category": "unknown",
            "final_condition": "unknown",
            "recommended_action": "review",
            "pricing_tier": "review",
            "confidence_level": "low",
            "notes": "Unable to classify - manual review required",
            "reconciliation": {"conflicts": [], "agreements": []}
        }
    
    def _calculate_confidence(self, attributes: Dict[str, Any]) -> float:
        """Calculate confidence based on final compilation completeness.
        
        Args:
            attributes: Parsed attributes
            
        Returns:
            Confidence score between 0 and 1
        """
        # Map confidence levels to scores
        confidence_map = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3
        }
        
        base_confidence = confidence_map.get(attributes.get("confidence_level", "low"), 0.3)
        
        # Adjust based on completeness
        if attributes.get("final_type") != "unknown":
            base_confidence += 0.05
        if attributes.get("recommended_action") != "review":
            base_confidence += 0.05
        
        # Penalize for conflicts
        reconciliation = attributes.get("reconciliation", {})
        if reconciliation.get("conflicts"):
            base_confidence -= 0.1 * len(reconciliation["conflicts"])
        
        return max(0.1, min(1.0, base_confidence))