"""
Production GPU Optimization for Qwen2.5-VL 3B - Enhanced with SOTA Techniques
Optimized for RTX A1000 (8GB VRAM) with advanced memory management
"""

import subprocess
import logging
import time
import os
import json
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"  # Safe mode with margins
    BALANCED = "balanced"         # Balanced performance/memory
    AGGRESSIVE = "aggressive"     # Maximum performance (95% VRAM)

@dataclass
class GPUConfig:
    """GPU configuration for optimal VLM performance"""
    total_memory_mb: int
    available_memory_mb: int
    model_memory_requirement_mb: int
    optimization_level: OptimizationLevel
    num_inference_streams: int
    cuda_optimizations: Dict[str, Any]

class GPUOptimizer:
    """Advanced GPU optimization for VLM inference - Enhanced for RTX A1000"""
    
    def __init__(self, model_name: str = "qwen2.5vl:3b"):
        self.model_name = model_name
        self.model_memory_requirements = {
            "qwen2.5vl:3b": 2800,     # MB for FP16
            "qwen2.5-vl-3b": 2800,     # MB for FP16 (alternate name)
            "qwen2.5vl:3b-q4": 1400,   # MB for 4-bit quantized
            "qwen2.5-vl-7b": 6400,     # MB for FP16
        }
        self._cached_config: Optional[GPUConfig] = None
        self._last_memory_check = 0
        self._memory_history: List[Tuple[float, int, int, int]] = []  # timestamp, total, used, free
        
        # RTX A1000 specific optimizations
        self.is_rtx_a1000 = self._detect_rtx_a1000()
    
    def _detect_rtx_a1000(self) -> bool:
        """Detect if we're running on RTX A1000"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                return "A1000" in gpu_name or "RTX A1000" in gpu_name
        except:
            pass
        return False
        
    def get_gpu_memory_info(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Get current GPU memory usage"""
        current_time = time.time()
        if current_time - self._last_memory_check < 1.0:  # Cache for 1 second
            return self._cached_memory_info if hasattr(self, '_cached_memory_info') else (None, None, None)
            
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', 
                 '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=2
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                total, used, free = int(values[0]), int(values[1]), int(values[2])
                self._cached_memory_info = (total, used, free)
                self._last_memory_check = current_time
                
                # Track memory history
                self._memory_history.append((current_time, total, used, free))
                if len(self._memory_history) > 100:
                    self._memory_history = self._memory_history[-100:]
                
                return total, used, free
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
        
        raise RuntimeError("GPU not available or accessible - this system requires GPU")
    
    def calculate_optimal_config(self, target_fps: int = 10) -> GPUConfig:
        """Calculate optimal GPU configuration for target FPS - Enhanced for RTX A1000"""
        total, used, free = self.get_gpu_memory_info()
        
        model_memory_req = self.model_memory_requirements.get(self.model_name, 2800)
        
        # RTX A1000 specific optimizations (8GB VRAM)
        if self.is_rtx_a1000:
            logger.info("🎯 RTX A1000 detected - applying specific optimizations")
            # With 8GB total and ~2.8GB model, we can be aggressive
            safety_margin = 500  # Smaller margin for RTX A1000
            
            # Check current usage and adjust
            if used < 1000:  # GPU is mostly free
                opt_level = OptimizationLevel.AGGRESSIVE
                safety_margin = 300  # Even smaller margin
            elif used < 3000:  # Some usage but plenty of room
                opt_level = OptimizationLevel.BALANCED
                safety_margin = 500
            else:  # Heavy usage
                opt_level = OptimizationLevel.CONSERVATIVE
                safety_margin = 1000
        else:
            # Standard configuration for other GPUs
            safety_margin = 1000
            opt_level = OptimizationLevel.BALANCED
        
        available_for_model = free - safety_margin
        
        # Calculate number of inference streams
        if available_for_model >= model_memory_req * 2:  # Can fit 2+ models
            num_streams = min(3, max(1, available_for_model // model_memory_req))
        elif available_for_model >= model_memory_req + 500:  # Extra headroom
            num_streams = 2
        else:
            num_streams = 1
            
        # Generate CUDA optimizations for Ollama
        cuda_opts = self._generate_cuda_optimizations(opt_level, available_for_model)
        
        return GPUConfig(
            total_memory_mb=total,
            available_memory_mb=available_for_model,
            model_memory_requirement_mb=model_memory_req,
            optimization_level=opt_level,
            num_inference_streams=num_streams,
            cuda_optimizations=cuda_opts
        )
    
    def _generate_cuda_optimizations(self, opt_level: OptimizationLevel, available_memory: int) -> Dict[str, Any]:
        """Generate SOTA CUDA optimization parameters for Ollama - Enhanced for 2024"""
        
        # Base parameters optimized for VLM
        base_opts = {
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 512,
            "seed": -1,  # Deterministic for consistency
        }
        
        if opt_level == OptimizationLevel.AGGRESSIVE:
            # SOTA High-performance configuration for RTX A1000
            cuda_opts = {
                **base_opts,
                "num_gpu": -1,  # Use all available GPU layers
                "gpu_layers": 32,  # Full 32 layers for 3B model
                "main_gpu": 0,
                "gpu_memory_utilization": 0.95,  # Use 95% of VRAM
                "use_mmap": True,
                "use_mlock": True,
                "num_thread": min(32, os.cpu_count() or 16),
                "num_batch": 2048,  # Larger batch for throughput
                "num_ctx": 8192,    # Full context window
                "f16_kv": True,     # FP16 KV cache for memory efficiency
                # SOTA 2024 optimizations
                "flash_attn": True,  # Flash Attention if supported
                "use_flash_attn": True,
                "low_vram": False,   # We have enough VRAM
                # Performance optimizations
                "tfs_z": 1.0,
                "typical_p": 1.0,
                "mirostat": 0,
                # CUDA-specific optimizations
                "tensor_split": None,  # Auto-balance
                "rope_freq_base": 1000000.0,  # RoPE optimization
                "rope_freq_scale": 1.0,
                # Memory optimizations
                "offload_kqv": False,  # Keep everything on GPU
                "mul_mat_q": True,     # Quantized matrix multiplication
            }
        elif opt_level == OptimizationLevel.CONSERVATIVE:
            # Safe configuration with larger margins
            cuda_opts = {
                **base_opts,
                "num_gpu": 24,  # Use 75% of layers
                "gpu_layers": 24,
                "main_gpu": 0,
                "gpu_memory_utilization": 0.75,
                "use_mmap": True,
                "use_mlock": False,  # Don't lock in RAM
                "num_thread": min(8, os.cpu_count() or 4),
                "num_batch": 512,
                "num_ctx": 4096,
                "f16_kv": True,
                "low_vram": True,  # Conservative mode
            }
        else:
            # Balanced GPU configuration - SOTA optimized
            optimal_layers = min(32, max(20, int((available_memory / 2800) * 32 * 0.85)))
            cuda_opts = {
                **base_opts,
                "num_gpu": optimal_layers,
                "gpu_layers": optimal_layers,
                "main_gpu": 0,
                "gpu_memory_utilization": 0.85,
                "use_mmap": True,
                "use_mlock": True,
                "num_thread": min(16, os.cpu_count() or 8),
                "num_batch": 1024,
                "num_ctx": 4096,
                "f16_kv": True,
                # Additional balanced optimizations
                "flash_attn": True,
                "low_vram": False,
                "mul_mat_q": True,
            }
        
        return cuda_opts
    
    def get_optimization_summary(self, config: GPUConfig) -> Dict[str, Any]:
        """Get a summary of applied optimizations"""
        return {
            "optimization_level": config.optimization_level.value,
            "gpu_memory_available_mb": config.available_memory_mb,
            "model_memory_requirement_mb": config.model_memory_requirement_mb,
            "inference_streams": config.num_inference_streams,
            "cuda_optimizations_count": len(config.cuda_optimizations),
            "performance_mode": "GPU"
        }
    
    def monitor_gpu_memory_change(self):
        """Monitor GPU memory changes (useful for debugging)"""
        before = self.get_gpu_memory_info()
        logger.info(f"GPU Memory Before: Total={before[0]}MB, Used={before[1]}MB, Free={before[2]}MB")
        
        def check_after():
            after = self.get_gpu_memory_info()
            if after[0]:
                change = after[1] - before[1] if before[1] else 0
                logger.info(f"GPU Memory After: Used={after[1]}MB (Change: {change:+}MB), Free={after[2]}MB")
                return change
            return 0
        
        return check_after
    
    def get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 = no pressure, 1.0 = full)"""
        total, used, free = self.get_gpu_memory_info()
        if total:
            return used / total
        return 0.0
    
    def predict_batch_memory(self, batch_size: int, image_size: int = 448) -> int:
        """Predict memory requirement for batch inference"""
        # Base memory for model
        base_memory = self.model_memory_requirements.get(self.model_name, 2800)
        
        # Additional memory per image (rough estimate)
        # 448x448x3 channels x 4 bytes (float32) = ~2.4MB per image
        # With processing overhead, estimate 5MB per image
        per_image_memory = 5
        
        # KV cache memory (scales with batch size)
        kv_cache_per_batch = 50  # MB per item in batch
        
        total_memory = base_memory + (batch_size * (per_image_memory + kv_cache_per_batch))
        return total_memory
    
    def can_fit_batch(self, batch_size: int) -> bool:
        """Check if a batch of given size can fit in current GPU memory"""
        total, used, free = self.get_gpu_memory_info()
        required = self.predict_batch_memory(batch_size)
        
        # Add safety margin
        safety_margin = 500 if self.is_rtx_a1000 else 1000
        
        return free >= (required + safety_margin)
    
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current GPU memory"""
        total, used, free = self.get_gpu_memory_info()
        
        # Start with max batch size and reduce until it fits
        for batch_size in [5, 4, 3, 2, 1]:
            if self.can_fit_batch(batch_size):
                logger.info(f"Optimal batch size: {batch_size} (free memory: {free}MB)")
                return batch_size
        
        return 1  # Fallback to single image
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        total, used, free = self.get_gpu_memory_info()
        
        stats = {
            "total_mb": total,
            "used_mb": used,
            "free_mb": free,
            "pressure": self.get_memory_pressure(),
            "optimal_batch_size": self.get_optimal_batch_size(),
            "is_rtx_a1000": self.is_rtx_a1000,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add memory trend if we have history
        if len(self._memory_history) > 5:
            recent = self._memory_history[-5:]
            memory_trend = sum(h[2] for h in recent) / len(recent)  # Average used memory
            stats["memory_trend"] = "increasing" if memory_trend > used else "stable"
        
        return stats

# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()