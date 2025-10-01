"""
VLM Inference Engine with production-level GPU optimizations
Integrates with existing stateful orchestrator for WebRTC streaming
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
import numpy as np

from services.vlm_gpu_loader import VLMGPULoader
from ..models.frame import Frame
from ..models.inference_result import InferenceResult

logger = logging.getLogger(__name__)

@dataclass
class VLMInferenceConfig:
    """Configuration for VLM inference engine"""
    max_concurrent_inferences: int = 2
    target_fps: int = 10
    batch_size: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.3
    warmup_on_init: bool = True
    performance_monitoring: bool = True

class VLMInferenceEngine:
    """Optimized VLM inference engine for real-time processing"""
    
    def __init__(self, 
                 config: VLMInferenceConfig = None,
                 ollama_host: str = "http://localhost:11434",
                 model_name: str = "qwen2.5vl:3b"):
        self.config = config or VLMInferenceConfig()
        self.vlm_loader = VLMGPULoader(ollama_host, model_name)
        self.is_ready = False
        self._inference_semaphore = None
        self._initialization_lock = asyncio.Lock()
        
        # Performance tracking
        self.total_inferences = 0
        self.total_inference_time = 0
        self.failed_inferences = 0
        self.last_performance_check = 0
        
    async def initialize(self, progress_callback: Optional[Callable] = None) -> bool:
        """Initialize VLM engine with GPU optimizations"""
        async with self._initialization_lock:
            if self.is_ready:
                logger.info("VLM Inference Engine already initialized")
                return True
                
            logger.info("Initializing VLM Inference Engine with GPU optimizations...")
            
            try:
                # Initialize GPU loader with optimizations
                result = await self.vlm_loader.initialize_and_optimize(
                    target_fps=self.config.target_fps,
                    progress_callback=progress_callback
                )
                
                if result["status"] == "success" or result["status"] == "already_optimized":
                    self._inference_semaphore = asyncio.Semaphore(self.config.max_concurrent_inferences)
                    self.is_ready = True
                    
                    # Log initialization results
                    stats = self.vlm_loader.get_performance_stats()
                    logger.info(f"VLM Engine ready - GPU optimized: {stats.get('is_optimized', False)}")
                    logger.info(f"Optimization level: {stats.get('optimization_level', 'unknown')}")
                    logger.info(f"Average inference time: {stats.get('avg_inference_time', 0):.0f}ms")
                    
                    return True
                else:
                    logger.error(f"VLM initialization failed: {result}")
                    return False
                    
            except Exception as e:
                logger.error(f"VLM engine initialization failed: {e}")
                return False
    
    async def process_frames(self, frames: List[Frame], prompt: str) -> List[InferenceResult]:
        """Process multiple frames with optimized batching"""
        if not self.is_ready:
            raise RuntimeError("VLM engine not initialized")
        
        results = []
        
        # Process frames in batches for optimal GPU utilization
        for i in range(0, len(frames), self.config.batch_size):
            batch_frames = frames[i:i + self.config.batch_size]
            batch_results = await self._process_frame_batch(batch_frames, prompt)
            results.extend(batch_results)
        
        return results
    
    async def _process_frame_batch(self, frames: List[Frame], prompt: str) -> List[InferenceResult]:
        """Process a batch of frames with concurrent execution control"""
        async with self._inference_semaphore:
            # Convert frames to numpy arrays
            np_frames = []
            for frame in frames:
                try:
                    # Convert frame data to numpy array
                    frame_array = self._frame_to_numpy(frame)
                    np_frames.append(frame_array)
                except Exception as e:
                    logger.error(f"Failed to convert frame {frame.frame_id} to numpy: {e}")
                    # Create error result for failed frame
                    error_result = InferenceResult(
                        agent_name="vlm_engine",
                        inference_num=1,
                        frame_count=1
                    )
                    error_result.complete({}, 0.0, f"Frame conversion error: {str(e)}")
                    return [error_result]
            
            if not np_frames:
                logger.error("No valid frames to process")
                return []
            
            # Process batch with GPU optimization
            return await self._process_numpy_batch(np_frames, frames, prompt)
    
    def _frame_to_numpy(self, frame: Frame) -> np.ndarray:
        """Convert Frame object to numpy array"""
        if hasattr(frame, 'data'):
            if isinstance(frame.data, np.ndarray):
                return frame.data
            elif isinstance(frame.data, bytes):
                # Convert bytes to numpy array via PIL
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(frame.data)).convert("RGB")
                return np.array(image)
            else:
                raise ValueError(f"Unsupported frame data type: {type(frame.data)}")
        else:
            raise ValueError("Frame object missing data attribute")
    
    async def _process_numpy_batch(self, np_frames: List[np.ndarray], 
                                  original_frames: List[Frame], prompt: str) -> List[InferenceResult]:
        """Process numpy frame batch with optimized VLM inference"""
        start_time = time.time()
        
        try:
            # Use GPU-optimized inference
            async with self.vlm_loader.optimized_inference():
                result = await self.vlm_loader.infer_optimized(np_frames, prompt)
            
            inference_time = int((time.time() - start_time) * 1000)
            
            # Update performance tracking
            self.total_inferences += 1
            self.total_inference_time += inference_time
            
            # Parse result and create InferenceResult objects
            inference_results = []
            
            for i, frame in enumerate(original_frames):
                # Create result for each frame (even though we process as batch)
                inference_result = InferenceResult(
                    agent_name="vlm_engine",
                    inference_num=1,
                    frame_count=len(np_frames)
                )
                
                if result.get("error"):
                    # Handle error case
                    self.failed_inferences += 1
                    inference_result.complete({}, 0.0, f"VLM inference error: {result['error']}")
                else:
                    # Success case
                    attributes = result.get("attributes", {})
                    confidence = result.get("confidence", 0.8)
                    reasoning = result.get("reasoning", "")
                    raw_response = result.get("raw_response", "")
                    
                    # Adjust confidence based on response quality
                    if not attributes or all(v is None or v == "" for v in attributes.values()):
                        confidence = max(0.1, confidence * 0.5)
                        reasoning += " (Warning: Limited attribute extraction)"
                    
                    inference_result.complete(attributes, confidence, reasoning, raw_response)
                
                # Add GPU optimization metadata
                inference_result.duration_ms = inference_time
                if hasattr(inference_result, 'metadata'):
                    inference_result.metadata.update(result.get("inference_metadata", {}))
                
                inference_results.append(inference_result)
            
            # Log performance
            success_indicator = "✓" if not result.get("error") else "✗"
            opt_level = result.get("inference_metadata", {}).get("optimization_level", "unknown")
            logger.info(f"{success_indicator} VLM batch inference completed in {inference_time}ms "
                       f"({len(np_frames)} frames, {opt_level} optimization)")
            
            # Performance monitoring
            if self.config.performance_monitoring:
                await self._monitor_performance(inference_time)
            
            return inference_results
            
        except Exception as e:
            inference_time = int((time.time() - start_time) * 1000)
            logger.error(f"VLM batch inference failed: {e}")
            self.failed_inferences += 1
            
            # Return error results for all frames
            error_results = []
            for frame in original_frames:
                error_result = InferenceResult(
                    agent_name="vlm_engine",
                    inference_num=1,
                    frame_count=len(np_frames)
                )
                error_result.complete({}, 0.0, f"Batch inference error: {str(e)}")
                error_result.duration_ms = inference_time
                error_results.append(error_result)
            
            return error_results
    
    async def _monitor_performance(self, inference_time: int):
        """Monitor and report performance metrics"""
        current_time = time.time()
        
        # Report performance every 30 seconds
        if current_time - self.last_performance_check > 30:
            avg_inference_time = self.total_inference_time / max(1, self.total_inferences)
            success_rate = ((self.total_inferences - self.failed_inferences) / 
                           max(1, self.total_inferences)) * 100
            
            logger.info(f"📊 VLM Performance Report:")
            logger.info(f"  - Total inferences: {self.total_inferences}")
            logger.info(f"  - Average time: {avg_inference_time:.0f}ms")
            logger.info(f"  - Success rate: {success_rate:.1f}%")
            logger.info(f"  - Last inference: {inference_time}ms")
            
            # Get GPU loader stats
            gpu_stats = self.vlm_loader.get_performance_stats()
            if gpu_stats.get("is_optimized"):
                logger.info(f"  - GPU optimization: {gpu_stats.get('optimization_level', 'unknown')}")
                logger.info(f"  - GPU memory: {gpu_stats.get('available_memory_mb', 0)}MB available")
            
            self.last_performance_check = current_time
            
            # Performance warnings
            if avg_inference_time > 2000:  # > 2 seconds
                logger.warning("⚠️ VLM inference performance degraded - consider GPU optimization")
            
            if success_rate < 90:
                logger.warning(f"⚠️ VLM success rate low: {success_rate:.1f}% - check model health")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.is_ready:
            return {"status": "not_ready"}
        
        avg_inference_time = (self.total_inference_time / max(1, self.total_inferences))
        success_rate = ((self.total_inferences - self.failed_inferences) / 
                       max(1, self.total_inferences)) * 100
        
        base_stats = {
            "engine_ready": self.is_ready,
            "total_inferences": self.total_inferences,
            "avg_inference_time_ms": avg_inference_time,
            "success_rate_percent": success_rate,
            "failed_inferences": self.failed_inferences,
            "max_concurrent_inferences": self.config.max_concurrent_inferences,
            "target_fps": self.config.target_fps,
            "batch_size": self.config.batch_size,
        }
        
        # Add GPU loader stats
        gpu_stats = self.vlm_loader.get_performance_stats()
        base_stats.update({
            "gpu_optimized": gpu_stats.get("is_optimized", False),
            "optimization_level": gpu_stats.get("optimization_level", "unknown"),
            "warmup_completed": gpu_stats.get("warmup_completed", False),
            "gpu_avg_inference_time": gpu_stats.get("avg_inference_time", 0),
            "gpu_available_memory_mb": gpu_stats.get("available_memory_mb", 0)
        })
        
        return base_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Check GPU loader health
            gpu_health = await self.vlm_loader.health_check()
            
            # Engine-specific checks
            engine_healthy = (
                self.is_ready and 
                gpu_health.get("status") == "healthy" and
                self._inference_semaphore is not None
            )
            
            # Performance health check
            performance_healthy = True
            if self.total_inferences > 0:
                avg_time = self.total_inference_time / self.total_inferences
                success_rate = ((self.total_inferences - self.failed_inferences) / 
                               self.total_inferences) * 100
                
                performance_healthy = avg_time < 3000 and success_rate > 80  # 3s max, 80% success
            
            overall_status = "healthy" if (engine_healthy and performance_healthy) else "degraded"
            
            return {
                "status": overall_status,
                "engine_ready": self.is_ready,
                "gpu_loader_health": gpu_health,
                "performance_healthy": performance_healthy,
                "metrics": self.get_performance_metrics()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Gracefully shutdown VLM engine"""
        logger.info("Shutting down VLM Inference Engine...")
        
        try:
            self.is_ready = False
            
            # Cleanup GPU loader
            await self.vlm_loader.cleanup()
            
            # Reset semaphore
            self._inference_semaphore = None
            
            logger.info("VLM Inference Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    # Backward compatibility methods for existing multi_inference_engine
    
    async def run_inference_compat(self, agent_name: str, frames: List[np.ndarray], 
                                  inference_num: int) -> InferenceResult:
        """Backward compatibility method for existing multi_inference_engine"""
        
        # Convert numpy arrays to Frame objects
        frame_objects = []
        for i, frame_array in enumerate(frames):
            # Note: Frame expects data as bytes, but for numpy arrays we'll store directly
            # The _frame_to_numpy method already handles this
            frame_obj = Frame(
                frame_id=f"{agent_name}_{inference_num}_{i}",
                agent_name=agent_name,
                data=frame_array,  # Store numpy array directly
                frame_number=i
            )
            frame_objects.append(frame_obj)
        
        # Build appropriate prompt for agent
        prompt = self._build_agent_prompt(agent_name, len(frames), inference_num)
        
        # Process frames
        results = await self.process_frames(frame_objects, prompt)
        
        # Return first result for compatibility (existing code expects single result)
        if results:
            result = results[0]
            # Update agent name and inference number for compatibility
            result.agent_name = agent_name
            result.inference_num = inference_num
            return result
        else:
            # Return error result
            error_result = InferenceResult(
                agent_name=agent_name,
                inference_num=inference_num,
                frame_count=len(frames)
            )
            error_result.complete({}, 0.0, "No results returned from VLM engine")
            return error_result
    
    def _build_agent_prompt(self, agent_name: str, frame_count: int, inference_num: int) -> str:
        """Build appropriate prompt based on agent and frame count (for backward compatibility)"""
        
        prompts = {
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
        
        agent_prompts = prompts.get(agent_name, prompts["initial_classifier"])
        
        if frame_count == 1:
            prompt = agent_prompts["base"]
        else:
            prompt = agent_prompts["multi_angle"].format(count=frame_count)
        
        # Add inference context
        if inference_num > 1:
            prompt += f" (Inference #{inference_num} - refine previous analysis)"
        
        return prompt