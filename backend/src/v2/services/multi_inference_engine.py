"""Multi-Inference Engine for running VLM inferences with GPU serialization"""

import asyncio
import logging
import time
from typing import List, Any, Dict, Optional
import numpy as np

from ..models.inference_result import InferenceResult
from config.gpu_config import gpu_config

logger = logging.getLogger(__name__)


class MultiInferenceEngine:
    """Manages multi-image inference with GPU serialization"""
    
    def __init__(self):
        self.gpu_config = gpu_config
        
        # VLM inference callback (to be set by external system)
        self.vlm_inference: Optional[Any] = None
        
        # Prompt templates
        self.prompts = {
            "initial_classifier": {
                "base": "Analyze this garment image and identify: item_type, color, pattern, neckline, sleeve_type, closure_type",
                "multi_angle": "These {count} images show the same garment from different angles. Analyze and identify: item_type, color, pattern, neckline, sleeve_type, closure_type"
            },
            "detail_extractor": {
                "base": "Extract details from this garment: brand, size, material, care instructions",
                "multi_angle": "These {count} images show the same garment. Extract: brand, size, material, care instructions"
            },
            "damage_detector": {
                "base": "Inspect this garment for damage. Identify: is_damaged (yes/no), damage_type, damage_severity",
                "multi_angle": "These {count} images show the same garment. One is the initial view, others are current. Check for: is_damaged, damage_type, damage_severity"
            }
        }
        
        logger.info("MultiInferenceEngine initialized with GPU serialization")
    
    async def run_inference(self, agent_name: str, frames: List[np.ndarray], 
                          inference_num: int) -> InferenceResult:
        """Run inference on single or multiple frames with GPU serialization"""
        
        start_time = time.time()
        result = InferenceResult(
            agent_name=agent_name,
            inference_num=inference_num,
            frame_count=len(frames)
        )
        
        # Build prompt
        prompt = self._build_prompt(agent_name, len(frames), inference_num)
        result.set_prompt(prompt, is_multi_angle=(len(frames) > 1))
        
        # Run inference with GPU lock
        async with self.gpu_config.gpu_semaphore:
            logger.debug(f"GPU acquired for {agent_name} inference #{inference_num}")
            
            try:
                # Call VLM inference
                if self.vlm_inference:
                    vlm_response = await self.vlm_inference(frames, prompt)
                else:
                    # Mock response for testing
                    vlm_response = await self._mock_inference(frames, prompt)
                
                # Process response
                attributes = vlm_response.get("attributes", {})
                confidence = vlm_response.get("confidence", 0.85)
                reasoning = vlm_response.get("reasoning", "")
                
                result.complete(attributes, confidence, reasoning, str(vlm_response))
                
            except Exception as e:
                logger.error(f"Inference failed for {agent_name}: {e}")
                result.complete({}, 0.0, f"Error: {str(e)}")
            
            finally:
                logger.debug(f"GPU released for {agent_name} inference #{inference_num}")
        
        duration_ms = int((time.time() - start_time) * 1000)
        result.duration_ms = duration_ms
        
        logger.info(f"{agent_name} inference #{inference_num} completed in {duration_ms}ms "
                   f"({len(frames)} frames, {result.inference_type})")
        
        return result
    
    def _build_prompt(self, agent_name: str, frame_count: int, inference_num: int) -> str:
        """Build appropriate prompt based on agent and frame count"""
        agent_prompts = self.prompts.get(agent_name, self.prompts["initial_classifier"])
        
        if frame_count == 1:
            prompt = agent_prompts["base"]
        else:
            prompt = agent_prompts["multi_angle"].format(count=frame_count)
        
        # Add inference context
        if inference_num > 1:
            prompt += f" (Inference #{inference_num} - refine previous analysis)"
        
        return prompt
    
    async def _mock_inference(self, frames: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """Mock inference for testing"""
        # Simulate inference time based on batch size
        batch_size = len(frames)
        inference_time = self.gpu_config.estimate_inference_time(batch_size)
        await asyncio.sleep(inference_time)
        
        # Generate mock attributes based on prompt
        if "initial_classifier" in prompt or "item_type" in prompt:
            attributes = {
                "item_type": "shirt",
                "color": "blue",
                "pattern": "solid",
                "neckline": "crew",
                "sleeve_type": "short",
                "closure_type": "buttons"
            }
        elif "detail_extractor" in prompt or "brand" in prompt:
            attributes = {
                "brand": "TestBrand",
                "size": "M",
                "material": "cotton",
                "care_instructions": "Machine wash cold"
            }
        elif "damage_detector" in prompt or "damage" in prompt:
            attributes = {
                "is_damaged": False,
                "damage_type": None,
                "damage_severity": None
            }
        else:
            attributes = {}
        
        return {
            "attributes": attributes,
            "confidence": 0.85 + (0.05 * batch_size),  # Higher confidence with more frames
            "reasoning": f"Analyzed {batch_size} frame(s) with prompt: {prompt[:50]}..."
        }
    
    def set_vlm_inference(self, inference_func: Any) -> None:
        """Set the VLM inference function"""
        self.vlm_inference = inference_func
        logger.info("VLM inference function configured")
