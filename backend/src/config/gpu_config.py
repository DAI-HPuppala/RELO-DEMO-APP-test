"""GPU Configuration for Serialized Access"""

import asyncio
import os
from typing import Optional

class GPUConfig:
    """Configuration for GPU resource management and serialization"""
    
    def __init__(self):
        # Semaphore for exclusive GPU access (one inference at a time)
        self.gpu_semaphore = asyncio.Semaphore(1)
        
        # GPU device configuration
        self.cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        
        # Memory limits
        self.max_vram_mb = 8192  # 8GB max VRAM usage
        self.inference_vram_mb = 7168  # ~7GB for inference (leaving buffer)
        
        # Batch size limits for multi-image inference
        self.max_batch_size = 5  # Maximum frames per inference
        self.optimal_batch_size = 3  # Optimal for performance/memory balance
        
        # Timeout settings
        self.inference_timeout_seconds = 30.0  # Max time for single inference
        self.gpu_warmup_timeout = 10.0  # Max time for model loading
        
    async def acquire_gpu(self) -> None:
        """Acquire exclusive GPU access for inference"""
        await self.gpu_semaphore.acquire()
        
    def release_gpu(self) -> None:
        """Release GPU access after inference"""
        self.gpu_semaphore.release()
        
    async def with_gpu(self, func, *args, **kwargs):
        """Execute function with exclusive GPU access"""
        await self.acquire_gpu()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            self.release_gpu()
            
    def get_batch_size_for_timer(self, timer_remaining: float, inference_num: int) -> int:
        """Calculate optimal batch size based on timer and inference number"""
        if inference_num == 1:
            # First inference is always single image
            return 1
        elif timer_remaining < 2.0:
            # Not enough time for large batch
            return min(2, self.max_batch_size)
        elif timer_remaining < 3.0:
            # Medium batch
            return min(3, self.max_batch_size)
        else:
            # Full batch possible
            return self.optimal_batch_size
            
    def estimate_inference_time(self, batch_size: int) -> float:
        """Estimate inference time based on batch size (in seconds)"""
        base_time = 0.8  # Single image inference ~800ms
        if batch_size == 1:
            return base_time
        elif batch_size <= 3:
            return base_time * 1.9  # ~1.5s for 3 images
        else:
            return base_time * 2.75  # ~2.2s for 5 images

# Global GPU configuration instance
gpu_config = GPUConfig()