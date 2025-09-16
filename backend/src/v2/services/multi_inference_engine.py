"""
Multi-Inference Engine - GPU Only Version
Streamlined GPU-accelerated multi-image inference without fallbacks
"""

import asyncio
import logging
import time
from typing import List, Any, Dict, Optional
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path

from ..models.inference_result import InferenceResult
from config.gpu_config import gpu_config
from services.vlm_singleton import vlm_singleton

logger = logging.getLogger(__name__)

class MultiInferenceEngine:
    """GPU-accelerated multi-image inference engine using VLM Singleton"""
    
    def __init__(self):
        self.gpu_config = gpu_config
        
        # Use VLM singleton instead of creating new instance
        self.vlm_singleton = vlm_singleton
        
        # Error handling settings
        self.max_retries = 2
        self.retry_delay = 1.0  # seconds
        
        # Performance tracking
        self.inference_history = []
        self.total_inferences = 0
        self.successful_inferences = 0
    
    async def initialize_gpu_optimization(self, progress_callback=None) -> bool:
        """Initialize GPU optimization for VLM inference"""
        logger.info("🚀 Checking VLM singleton initialization status...")
        
        try:
            # Check if VLM singleton is already initialized
            if self.vlm_singleton.is_ready():
                logger.info("✅ VLM singleton already initialized - no warmup needed")
                return True
            
            # Initialize VLM singleton (will only initialize once)
            result = await self.vlm_singleton.initialize_once(progress_callback)
            
            if result.get("status") in ["success", "already_initialized"]:
                logger.info("✅ VLM singleton ready for multi-inference")
                return True
            else:
                raise RuntimeError(f"VLM initialization failed: {result}")
                
        except Exception as e:
            logger.error(f"VLM singleton initialization failed: {e}")
            raise RuntimeError(f"VLM singleton initialization failed: {e}")
    
    def is_gpu_optimized(self) -> bool:
        """Check if GPU optimization is active and ready"""
        is_ready = self.vlm_singleton.is_ready()
        logger.debug(f"VLM singleton ready status: {is_ready}")
        return is_ready
    
    async def run_inference(self, agent_name: str, frames: List[np.ndarray], inference_num: int, previous_context: Dict[str, Any] = None) -> InferenceResult:
        """Run GPU-accelerated inference for multiple frames"""
        
        if not self.vlm_singleton.is_ready():
            logger.warning(f"VLM singleton not ready for {agent_name}, attempting initialization...")
            # Initialize VLM singleton if not ready (will only init once)
            try:
                result = await self.vlm_singleton.initialize_once()
                if result.get("status") not in ["success", "already_initialized"]:
                    raise RuntimeError(f"VLM initialization failed: {result}")
                logger.info("VLM singleton initialized successfully on demand")
            except Exception as e:
                logger.error(f"VLM initialization error: {e}")
                raise RuntimeError(f"VLM singleton not initialized: {str(e)}")
        
        # Save frames for debugging
        saved_frame_paths = await self._save_debug_frames(agent_name, frames, inference_num)

        # Also save the previous frame if being used for batch processing
        if inference_num > 1 and previous_context and previous_context.get('frame_data') is not None:
            prev_frame_paths = await self._save_debug_frames(
                agent_name,
                [previous_context.get('frame_data')],
                inference_num,
                is_previous=True
            )

        # Prepare frames for batch processing
        frames_to_process = frames.copy()

        # For inference #2 and beyond, include previous frame for batch processing
        if inference_num > 1 and previous_context and previous_context.get('frame_data') is not None:
            # Insert previous frame at the beginning for batch processing
            prev_frame = previous_context.get('frame_data')
            frames_to_process = [prev_frame] + frames_to_process

        # Clear logging for debugging
        logger.info("="*80)
        logger.info(f"🎯 INFERENCE START: {agent_name.upper()} - Inference #{inference_num}")
        logger.info("="*80)
        if inference_num == 1:
            logger.info(f"📸 FRAME CAPTURED: {len(frames)} {'frame' if len(frames) == 1 else 'frames'} captured from camera (first inference)")
        else:
            logger.info(f"📸 FRAME CAPTURED: 1 new frame captured for inference #{inference_num}")
            if previous_context and previous_context.get('frame_data') is not None:
                logger.info(f"🖼️ BATCH PROCESSING: Using 2 images (1 previous + 1 new) for context-aware inference")
                logger.info(f"📚 CONTEXT: Using results and image from inference #{previous_context.get('inference_num', inference_num-1)}")
                # Log the actual context being used
                logger.info("*" * 30 + " CONTEXT " + "*" * 30)
                prev_attrs = previous_context.get('attributes', {})
                if prev_attrs:
                    logger.info(f"Previous Attributes: {prev_attrs}")
                logger.info(f"Previous Confidence: {previous_context.get('confidence', 0.0):.1%}")
                if previous_context.get('reasoning'):
                    logger.info(f"Previous Reasoning: {previous_context.get('reasoning')}")
                logger.info("*" * 69)

        for i, path in enumerate(saved_frame_paths, 1):
            logger.info(f"  📁 NEW Frame SAVED TO: {path}")

        if inference_num > 1 and previous_context and previous_context.get('frame_data') is not None:
            logger.info(f"  📁 PREVIOUS Frame: Used from inference #{previous_context.get('inference_num')} (in memory)")
            logger.info(f"  🔍 Batch processing with 2 images (previous + new) for {agent_name} inference #{inference_num}")
            logger.info("*" * 30 + f" BATCH DETAILS FOR {agent_name.upper()} " + "*" * 30)
            logger.info(f"  Image 1: Previous frame from inference #{previous_context.get('inference_num')}")
            logger.info(f"  Image 2: New frame just captured")
            logger.info(f"  Previous Results: {previous_context.get('attributes')}")
            logger.info("*" * (60 + len(agent_name) + 14))
        else:
            logger.info(f"  🔍 This frame will be used for {agent_name} inference #{inference_num}")
        logger.info("-"*80)

        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                # Get VLM GPU loader from singleton
                vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()

                # Create prompt based on agent name and context
                prompt = self._get_agent_prompt(agent_name, inference_num, previous_context)

                # Use GPU-optimized inference with agent name for better detection
                # Use frames_to_process which includes previous frame for batch processing
                result_data = await vlm_loader.infer_optimized(frames_to_process, prompt, agent_name)
                
                # Create InferenceResult object
                result = InferenceResult(
                    agent_name=agent_name,
                    inference_num=inference_num,
                    frame_count=len(frames)
                )
                
                # Extract attributes and confidence from result
                attributes = result_data.get("attributes", {})
                confidence = result_data.get("confidence", 0.8)
                reasoning = result_data.get("reasoning", "")
                
                result.complete(attributes, confidence, reasoning)
                
                # Track performance
                inference_time = time.time() - start_time
                self.total_inferences += 1
                self.successful_inferences += 1
                self._record_performance(agent_name, inference_time, "success")
                
                logger.info(f"✅ GPU INFERENCE COMPLETED for {agent_name} in {inference_time:.2f}s")
                logger.info(f"📊 Results: {len(attributes)} attributes detected")
                logger.info("="*80)
                return result
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"GPU inference attempt {attempt + 1} failed for {agent_name}: {e}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Final attempt failed
                    inference_time = time.time() - start_time
                    self.total_inferences += 1
                    self._record_performance(agent_name, inference_time, "failed")
                    
                    logger.error(f"❌ GPU inference failed for {agent_name} after {self.max_retries} retries: {e}")
                    
                    # Return error result
                    error_result = InferenceResult(
                        agent_name=agent_name,
                        inference_num=inference_num,
                        frame_count=len(frames)
                    )
                    error_result.complete({}, 0.0, f"GPU inference failed: {str(e)}")
                    error_result.duration_ms = int(inference_time * 1000)
                    return error_result

    async def _save_debug_frames(self, agent_name: str, frames: List[np.ndarray],
                                inference_num: int, is_previous: bool = False) -> List[str]:
        """Save frames to disk for debugging"""
        saved_paths = []

        try:
            # Create directory for this agent
            base_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/captured_frames")
            agent_dir = base_dir / agent_name
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for unique naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for idx, frame in enumerate(frames):
                # Generate filename with prefix for previous frames
                prefix = "prev_" if is_previous else ""
                filename = f"{agent_name}_inf{inference_num}_{prefix}frame{idx+1}_{timestamp}.jpg"
                filepath = agent_dir / filename

                # Save frame
                success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if success:
                    saved_paths.append(str(filepath))
                else:
                    logger.error(f"❌ Failed to save frame: {filepath}")

        except Exception as e:
            logger.error(f"Error saving debug frames for {agent_name}: {e}")

        return saved_paths

    def _get_agent_prompt(self, agent_name: str, inference_num: int, previous_context: Dict[str, Any] = None) -> str:
        """Generate appropriate prompt based on agent name and context"""
        prompts = {
            "initial_classifier": "Analyze this garment and identify: type (e.g., T-shirt, Dress, Pants, Shoes, Shirt, Shorts, Jacket, Sweatshirt, Sweater, Hoodie), color, pattern, neckline style, sleeve length, and closure type. Neckline, closure type, and sleeve length are optional or could be null for Shoes. Return null for unrecognizable attributes.",
            "detail_extractor": "Look for brand name and size label on this garment. Return brand and size, or null if not visible.",
            "damage_detector": "Check this garment for damage. Identify if damaged (yes/no), Damage-type with location, and provide reasoning. Return null if no damage found.",
            "final_compiler": "Compile final classification based on all attributes."
        }

        base_prompt = prompts.get(agent_name, "Analyze this clothing item")

        # Add context for inference #2 and beyond (not for final_compiler)
        if inference_num > 1 and previous_context and agent_name != "final_compiler":
            prev_attrs = previous_context.get('attributes', {})
            prev_confidence = previous_context.get('confidence', 0.0)

            # Build context string based on agent type
            context_str = "\n\nContext from your previous analysis:"

            if agent_name == "initial_classifier":
                # For initial classifier, mention previous garment attributes
                if prev_attrs:
                    context_str += f"\n- Previously identified: {', '.join([f'{k}={v}' for k, v in prev_attrs.items() if v])}"
                context_str += "\n\nYou are now analyzing TWO images in batch: your previous inference image AND a new image of the same garment. Compare both views to confirm or refine your findings."

            elif agent_name == "detail_extractor":
                # For detail extractor, mention if brand/size was found before
                brand = prev_attrs.get('brand', 'not found')
                size = prev_attrs.get('size', 'not found')
                context_str += f"\n- Previous brand detection: {brand}"
                context_str += f"\n- Previous size detection: {size}"
                context_str += "\n\nYou are now analyzing TWO images in batch: your previous inference image AND a new image. Check both images for brand and size labels - they may be clearer in one view."

            elif agent_name == "damage_detector":
                # For damage detector, mention previous damage findings
                damaged = prev_attrs.get('damaged', 'unknown')
                damage_type = prev_attrs.get('damage_type', 'none')
                context_str += f"\n- Previous damage detection: {damaged}"
                if damaged == 'yes':
                    context_str += f"\n- Damage type found: {damage_type}"
                context_str += "\n\nYou are now analyzing TWO images in batch: your previous inference image AND a new image. Examine both views for damage to confirm findings or detect additional issues."

            base_prompt += context_str

        return f"{base_prompt} (Inference #{inference_num})"
    
    def _record_performance(self, agent_name: str, inference_time: float, status: str):
        """Record performance metrics"""
        performance_record = {
            "timestamp": time.time(),
            "agent_name": agent_name,
            "inference_time": inference_time,
            "status": status
        }
        
        self.inference_history.append(performance_record)
        
        # Keep only last 100 records
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-100:]
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get GPU optimization status"""
        
        vlm_status = self.vlm_singleton.get_status()
        
        base_status = {
            "gpu_optimized": True,
            "engine_ready": self.vlm_singleton.is_ready(),
            "vlm_state": vlm_status.get("state"),
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "success_rate": (self.successful_inferences / max(1, self.total_inferences)) * 100
        }
        
        # Add VLM singleton metrics
        if self.vlm_singleton.is_ready():
            vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()
            vlm_metrics = vlm_loader.get_performance_stats()
            base_status.update({
                "vlm_metrics": vlm_metrics,
                "optimization_level": vlm_metrics.get("optimization_level", "unknown"),
                "avg_inference_time_ms": vlm_metrics.get("avg_inference_time", 0)
            })
        
        return base_status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of GPU-optimized inference engine"""
        
        try:
            # Check VLM singleton health
            if self.vlm_singleton.is_ready():
                vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()
                vlm_health = await vlm_loader.health_check()
                engine_healthy = vlm_health.get("status") == "healthy"
            else:
                vlm_health = {"status": "error", "message": "VLM singleton not initialized"}
                engine_healthy = False
            
            # Calculate performance health
            success_rate = (self.successful_inferences / max(1, self.total_inferences)) * 100
            performance_healthy = success_rate > 80  # 80% success rate threshold
            
            overall_status = "healthy" if (engine_healthy and performance_healthy) else "degraded"
            
            return {
                "status": overall_status,
                "engine_healthy": engine_healthy,
                "performance_healthy": performance_healthy,
                "success_rate": success_rate,
                "total_inferences": self.total_inferences,
                "vlm_health": vlm_health
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Shutdown GPU-optimized inference engine"""
        logger.info("Shutting down GPU-optimized multi-inference engine...")
        
        try:
            # VLM singleton manages its own lifecycle
            # No need to shutdown as it's shared across the application
            logger.info("✅ GPU-optimized multi-inference engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last period"""
        
        if not self.inference_history:
            return {"message": "No inference history available"}
        
        # Calculate metrics from recent history
        recent_inferences = self.inference_history[-20:]  # Last 20 inferences
        
        successful = [r for r in recent_inferences if r["status"] == "success"]
        total_time = sum(r["inference_time"] for r in successful)
        avg_time = total_time / len(successful) if successful else 0
        
        # Group by agent
        agent_stats = {}
        for record in recent_inferences:
            agent = record["agent_name"]
            if agent not in agent_stats:
                agent_stats[agent] = {"count": 0, "avg_time": 0, "success_count": 0}
            
            agent_stats[agent]["count"] += 1
            if record["status"] == "success":
                agent_stats[agent]["success_count"] += 1
                agent_stats[agent]["avg_time"] = (
                    (agent_stats[agent]["avg_time"] * (agent_stats[agent]["success_count"] - 1) + 
                     record["inference_time"]) / agent_stats[agent]["success_count"]
                )
        
        return {
            "total_inferences": len(recent_inferences),
            "successful_inferences": len(successful),
            "success_rate": (len(successful) / len(recent_inferences)) * 100,
            "avg_inference_time": avg_time,
            "agent_statistics": agent_stats,
            "gpu_optimized": True
        }