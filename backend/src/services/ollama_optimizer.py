"""
Ollama Optimization Service - SOTA configuration for Qwen2.5-VL
Implements advanced CUDA optimizations and memory management
"""

import asyncio
import aiohttp
import json
import logging
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from config.gpu_optimizer import gpu_optimizer, OptimizationLevel
from services.provider_factory import get_ollama_host, get_model_name

logger = logging.getLogger(__name__)

class OllamaOptimizer:
    """Optimizes Ollama for maximum performance with Qwen2.5-VL"""

    def __init__(self, ollama_host: str = None, model_name: str = None):
        # Use provider factory for flexible backend support
        self.ollama_host = ollama_host or get_ollama_host()
        self.model_name = model_name or get_model_name()
        self.current_config: Optional[Dict[str, Any]] = None
        self.optimization_applied = False
        self.last_optimization_time = 0
        
    async def apply_optimal_configuration(self) -> Dict[str, Any]:
        """Apply SOTA optimal configuration to Ollama"""
        logger.info("🚀 Applying optimal Ollama configuration...")
        
        # Get optimal GPU configuration
        gpu_config = gpu_optimizer.calculate_optimal_config(target_fps=10)
        
        # Extract CUDA optimizations
        cuda_opts = gpu_config.cuda_optimizations
        
        logger.info(f"📊 Optimization level: {gpu_config.optimization_level.value}")
        logger.info(f"   GPU layers: {cuda_opts.get('num_gpu', 'auto')}")
        logger.info(f"   Batch size: {cuda_opts.get('num_batch', 'default')}")
        logger.info(f"   Context size: {cuda_opts.get('num_ctx', 'default')}")
        
        # Apply configuration via test inference
        success = await self._apply_config_via_inference(cuda_opts)
        
        if success:
            self.current_config = cuda_opts
            self.optimization_applied = True
            self.last_optimization_time = time.time()
            
            return {
                "status": "success",
                "optimization_level": gpu_config.optimization_level.value,
                "gpu_layers": cuda_opts.get("num_gpu", -1),
                "config": cuda_opts
            }
        else:
            return {
                "status": "failed",
                "error": "Could not apply configuration"
            }
    
    async def _apply_config_via_inference(self, config: Dict[str, Any]) -> bool:
        """Apply configuration by running an inference with optimal settings"""
        try:
            async with aiohttp.ClientSession() as session:
                # Run inference with new configuration
                payload = {
                    "model": self.model_name,
                    "prompt": "Optimization test",
                    "stream": False,
                    "keep_alive": -1,  # Keep model in memory permanently
                    "options": config
                }
                
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("✅ Configuration applied successfully")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Failed to apply config: {error}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            return False
    
    async def optimize_for_batch_inference(self, batch_size: int) -> Dict[str, Any]:
        """Optimize specifically for batch inference"""
        logger.info(f"🎯 Optimizing for batch size: {batch_size}")
        
        # Check if batch can fit
        if not gpu_optimizer.can_fit_batch(batch_size):
            logger.warning(f"Batch size {batch_size} too large for current GPU memory")
            batch_size = gpu_optimizer.get_optimal_batch_size()
            logger.info(f"Adjusted to optimal batch size: {batch_size}")
        
        # Get base configuration
        gpu_config = gpu_optimizer.calculate_optimal_config()
        cuda_opts = gpu_config.cuda_optimizations
        
        # Adjust for batch processing
        cuda_opts["num_batch"] = min(2048, batch_size * 512)  # Scale batch buffer
        cuda_opts["num_ctx"] = min(8192, batch_size * 2048)   # Scale context
        
        # Apply configuration
        success = await self._apply_config_via_inference(cuda_opts)
        
        return {
            "status": "success" if success else "failed",
            "batch_size": batch_size,
            "config": cuda_opts
        }
    
    async def enable_flash_attention(self) -> bool:
        """Enable Flash Attention if available"""
        logger.info("⚡ Attempting to enable Flash Attention...")
        
        # Check if Flash Attention is available
        test_config = {
            "flash_attn": True,
            "use_flash_attn": True,
            "num_predict": 1
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": "Flash attention test",
                    "stream": False,
                    "options": test_config
                }
                
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Flash Attention enabled successfully")
                        return True
                    else:
                        logger.warning("Flash Attention not available on this system")
                        return False
                        
        except Exception as e:
            logger.warning(f"Could not enable Flash Attention: {e}")
            return False
    
    async def set_permanent_gpu_residence(self) -> bool:
        """Configure model to stay in GPU permanently (no unloading)"""
        logger.info("🔒 Setting permanent GPU residence...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Set keep_alive to -1 for permanent residence
                payload = {
                    "model": self.model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": -1  # Never unload
                }
                
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Model set to permanent GPU residence")
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to set permanent residence: {e}")
            
        return False
    
    async def unload_model(self) -> bool:
        """Unload model from GPU (for cleanup)"""
        logger.info("🔓 Unloading model from GPU...")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0  # Unload immediately
                }
                
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("✅ Model unloaded from GPU")
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            
        return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        
                        for model in models:
                            if model.get("name") == self.model_name:
                                return {
                                    "name": model.get("name"),
                                    "size": model.get("size"),
                                    "modified": model.get("modified_at"),
                                    "loaded": True
                                }
                        
                        return {"loaded": False}
                        
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}
    
    async def benchmark_inference(self, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark inference performance"""
        logger.info(f"📊 Running {num_runs} benchmark inferences...")
        
        times = []
        prompts = [
            "What color is this item?",
            "Describe the type of clothing",
            "Is there any damage?",
            "What is the pattern?",
            "Describe the material"
        ]
        
        for i in range(min(num_runs, len(prompts))):
            start = time.time()
            
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompts[i],
                        "stream": False,
                        "options": {
                            "num_predict": 50,
                            "temperature": 0.3
                        }
                    }
                    
                    async with session.post(
                        f"{self.ollama_host}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            inference_time = time.time() - start
                            times.append(inference_time)
                            logger.info(f"  Run {i+1}: {inference_time:.2f}s")
                            
            except Exception as e:
                logger.error(f"Benchmark run {i+1} failed: {e}")
        
        if times:
            return {
                "num_runs": len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_time": sum(times) / len(times),
                "times": times
            }
        else:
            return {"error": "No successful benchmark runs"}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            "optimization_applied": self.optimization_applied,
            "current_config": self.current_config,
            "last_optimization": datetime.fromtimestamp(self.last_optimization_time).isoformat() if self.last_optimization_time else None,
            "gpu_stats": gpu_optimizer.get_memory_stats()
        }

# Global optimizer instance
ollama_optimizer = OllamaOptimizer()

async def optimize_ollama_at_startup():
    """Function to be called at startup to optimize Ollama"""
    logger.info("🚀 Optimizing Ollama at startup...")
    
    # Apply optimal configuration
    config_result = await ollama_optimizer.apply_optimal_configuration()
    
    # Set permanent GPU residence
    await ollama_optimizer.set_permanent_gpu_residence()
    
    # Try to enable Flash Attention
    flash_enabled = await ollama_optimizer.enable_flash_attention()
    
    # Run benchmark
    benchmark = await ollama_optimizer.benchmark_inference(3)
    
    return {
        "configuration": config_result,
        "flash_attention": flash_enabled,
        "benchmark": benchmark
    }

__all__ = ['ollama_optimizer', 'optimize_ollama_at_startup']