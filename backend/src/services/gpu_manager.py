"""
Advanced GPU Memory Management for Optimal Qwen2.5-VL Performance
Based on RELO-DEMO-APP optimization strategies with aggressive GPU utilization
"""

import subprocess
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Advanced GPU memory management for optimal VLM performance"""
    
    def __init__(self):
        self.gpu_info = None
        self.optimal_layers = None
        # Aggressive optimization - minimal safety margin for maximum GPU use
        self.memory_safety_margin = 0.05  # Only 5% safety margin (vs 15% default)
        self.model_size_mb = 3000  # Qwen2.5-VL 3B ~3GB
        self.base_memory_mb = 800   # Reduced base CUDA overhead estimate
        self.last_memory_check = 0
        self.memory_cache_duration = 30  # Cache memory info for 30 seconds
        
    def get_gpu_memory_info(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Get current GPU memory usage with caching
        Returns: (total_mb, used_mb, free_mb)
        """
        current_time = time.time()
        
        # Use cached data if recent
        if (current_time - self.last_memory_check) < self.memory_cache_duration and self.gpu_info:
            return self.gpu_info['total_memory_mb'], self.gpu_info['used_memory_mb'], self.gpu_info['free_memory_mb']
            
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu', 
                 '--format=csv,nounits,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                values = [v.strip() for v in result.stdout.strip().split(',')]
                if len(values) >= 3:
                    total = int(values[0])
                    used = int(values[1])
                    free = int(values[2])
                    gpu_util = int(values[3]) if len(values) > 3 else 0
                    gpu_temp = int(values[4]) if len(values) > 4 else 0
                    
                    self.gpu_info = {
                        'total_memory_mb': total,
                        'used_memory_mb': used,
                        'free_memory_mb': free,
                        'gpu_utilization': gpu_util,
                        'gpu_temperature': gpu_temp,
                        'memory_utilization': (used / total) * 100
                    }
                    self.last_memory_check = current_time
                    
                    return total, used, free
            
            logger.warning("nvidia-smi returned unexpected output")
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return None, None, None

    def calculate_aggressive_gpu_layers(self) -> int:
        """
        Calculate aggressive GPU layer allocation for maximum performance
        Based on RELO-DEMO-APP strategies but more aggressive for <1200ms target
        """
        total, used, free = self.get_gpu_memory_info()
        
        if free is None:
            logger.warning("Cannot determine GPU memory, defaulting to CPU")
            return 0
        
        logger.info(f"GPU Memory - Total: {total}MB, Used: {used}MB, Free: {free}MB")
        
        # Apply minimal safety margin for aggressive optimization
        usable_memory = free * (1 - self.memory_safety_margin)
        
        if usable_memory < 500:
            logger.info(f"Insufficient GPU memory ({usable_memory:.0f}MB), using CPU")
            return 0
        elif usable_memory < 1200:
            # Limited memory - use conservative layers
            gpu_layers = min(10, int(40 * 0.25))
            logger.info(f"Limited GPU memory: using {gpu_layers} layers")
            return gpu_layers
        elif usable_memory < 2500:
            # Partial GPU - aggressive but safe
            percentage = min(0.8, (usable_memory / self.model_size_mb) * 0.85)
            gpu_layers = min(32, int(40 * percentage))
            logger.info(f"Partial aggressive GPU: {gpu_layers} layers ({percentage:.1%})")
            return gpu_layers
        else:
            # Full aggressive GPU allocation - use all layers (Qwen2.5-VL has 40 layers)
            logger.info(f"Excellent GPU memory ({usable_memory:.0f}MB) - using ALL 40 layers")
            return 40  # Use ALL layers for maximum performance (Qwen2.5-VL 3B has ~40 layers)

    def get_aggressive_ollama_config(self) -> Dict[str, Any]:
        """
        Get aggressive GPU configuration for Ollama based on RELO-DEMO-APP strategies
        Optimized for <1200ms inference target
        """
        gpu_layers = self.calculate_aggressive_gpu_layers()
        
        # Aggressive base configuration for speed
        config = {
            "temperature": 0.2,  # Lower temperature for faster, more consistent inference
            "top_p": 0.85,      # Slightly lower for speed
            "top_k": 35,        # Reduced for faster sampling
            "num_predict": 300, # Optimized token count
            "num_ctx": 8192,    # Large context for VL tasks
            "repeat_penalty": 1.05,  # Minimal penalty for speed
            "tfs_z": 1.0,
            "mirostat": 0,      # Disable for speed
        }
        
        if gpu_layers == 0:
            # CPU-only - optimized for i9-13900TE
            config.update({
                "num_gpu": 0,
                "num_thread": 24,   # Use most CPU threads for speed
                "f16_kv": False,
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": False
            })
            logger.info("Aggressive CPU-only configuration (24 threads)")
            
        elif gpu_layers < 16:
            # Hybrid aggressive configuration
            config.update({
                "num_gpu": gpu_layers,
                "gpu_layers": gpu_layers,
                "num_thread": 12,   # Balance CPU threads
                "f16_kv": True,     # Enable for GPU speed
                "use_mmap": True,
                "use_mlock": True,  # Lock in memory for speed
                "low_vram": False,
                "main_gpu": 0,      # Use primary GPU
                "split_mode": 1     # Layer split mode
            })
            logger.info(f"Aggressive hybrid config: {gpu_layers} GPU layers + 12 CPU threads")
            
        else:
            # Full aggressive GPU configuration for maximum speed
            config.update({
                "num_gpu": 1,       # Use GPU
                "gpu_layers": gpu_layers,
                "num_thread": 6,    # Minimal CPU threads
                "f16_kv": True,     # Full FP16 for GPU speed
                "use_mmap": False,  # Direct GPU access
                "use_mlock": True,  # Lock model in GPU memory
                "low_vram": False,  # Disable VRAM optimization for speed
                "main_gpu": 0,
                "split_mode": 2,    # Row split for better performance
                "tensor_split": None, # Use single GPU fully
                "numa": False,      # Disable NUMA for speed
                # Aggressive GPU optimizations
                "flash_attn": True, # Enable flash attention if available
                "rope_freq_base": 10000,
                "rope_freq_scale": 1.0
            })
            logger.info(f"MAXIMUM AGGRESSIVE GPU: {gpu_layers}/40 layers, full optimization")
        
        return config

    def monitor_gpu_performance(self) -> Dict[str, Any]:
        """Monitor real-time GPU performance metrics"""
        total, used, free = self.get_gpu_memory_info()
        
        if total is None:
            return {"gpu_available": False, "error": "GPU monitoring unavailable"}
        
        performance_status = "unknown"
        gpu_util = self.gpu_info.get('gpu_utilization', 0) if self.gpu_info else 0
        memory_util = self.gpu_info.get('memory_utilization', 0) if self.gpu_info else 0
        
        # Performance assessment
        if gpu_util > 80 and memory_util > 60:
            performance_status = "excellent"
        elif gpu_util > 50 and memory_util > 40:
            performance_status = "good"
        elif gpu_util > 20:
            performance_status = "moderate"
        else:
            performance_status = "poor"
        
        return {
            "gpu_available": True,
            "total_memory_mb": total,
            "used_memory_mb": used,
            "free_memory_mb": free,
            "memory_utilization_percent": memory_util,
            "gpu_utilization_percent": gpu_util,
            "gpu_temperature": self.gpu_info.get('gpu_temperature', 0) if self.gpu_info else 0,
            "performance_status": performance_status,
            "optimal_gpu_layers": self.calculate_aggressive_gpu_layers(),
            "timestamp": time.time()
        }


class OllamaModelOptimizer:
    """Ollama model optimization for aggressive GPU utilization"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = "qwen2.5vl:3b"
        self.optimized_model_name = "qwen2.5vl:3b-optimized"
        self.gpu_manager = GPUMemoryManager()
        
    async def create_optimized_modelfile(self) -> str:
        """Create optimized Modelfile for maximum GPU performance"""
        
        gpu_config = self.gpu_manager.get_aggressive_ollama_config()
        
        # Build simplified aggressive Modelfile - only use supported parameters
        # Ensure FROM directive uses exact model name
        modelfile_content = f"""FROM {self.model_name}

# Core performance parameters for <800ms inference
PARAMETER num_ctx {gpu_config.get('num_ctx', 8192)}
PARAMETER temperature {gpu_config.get('temperature', 0.2)}
PARAMETER top_p {gpu_config.get('top_p', 0.85)}
PARAMETER top_k {gpu_config.get('top_k', 35)}
PARAMETER repeat_penalty {gpu_config.get('repeat_penalty', 1.05)}
PARAMETER num_predict {gpu_config.get('num_predict', 300)}
PARAMETER num_thread {gpu_config.get('num_thread', 6)}

# System message for VL tasks
SYSTEM You are an advanced vision-language AI optimized for rapid clothing analysis. Always respond with accurate JSON containing the requested attributes. Prioritize speed and accuracy.
"""
        
        logger.info(f"Generated aggressive Modelfile for GPU optimization (base: {self.model_name})")
        return modelfile_content
        
    async def create_optimized_model(self) -> bool:
        """Create optimized model with aggressive GPU settings"""
        try:
            # Get GPU configuration
            gpu_config = self.gpu_manager.get_aggressive_ollama_config()
            
            # Create the optimized model via Ollama API using 'from' parameter
            url = f"{self.ollama_host}/api/create"
            
            # Build parameters dict from gpu_config
            parameters = {
                "temperature": gpu_config.get('temperature', 0.2),
                "top_p": gpu_config.get('top_p', 0.85),
                "top_k": gpu_config.get('top_k', 35),
                "repeat_penalty": gpu_config.get('repeat_penalty', 1.05),
                "num_predict": gpu_config.get('num_predict', 300),
                "num_thread": gpu_config.get('num_thread', 6),
                "num_ctx": gpu_config.get('num_ctx', 8192)
            }
            
            payload = {
                "name": self.optimized_model_name,
                "from": self.model_name,  # Use 'from' to specify base model
                "parameters": parameters,  # Pass parameters as dict
                "stream": False
            }
            
            logger.info(f"Creating optimized model {self.optimized_model_name} from base: {self.model_name}")
            logger.info(f"GPU optimization parameters: {parameters}")
            
            timeout = aiohttp.ClientTimeout(total=120)  # Allow time for model creation
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        # Read the streaming response
                        async for line in response.content:
                            if line:
                                try:
                                    result = json.loads(line)
                                    if 'status' in result:
                                        logger.debug(f"Model creation: {result['status']}")
                                except json.JSONDecodeError:
                                    pass
                        logger.info(f"✅ Created optimized model: {self.optimized_model_name}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create optimized model: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error creating optimized model: {e}")
            return False
    
    async def ensure_optimized_model_loaded(self) -> str:
        """Ensure the optimized model is loaded and ready with aggressive GPU allocation
        Returns: actual model name that was loaded
        """
        try:
            # Try to create optimized model first
            model_created = await self.create_optimized_model()
            
            if not model_created:
                logger.warning("Using base model instead of optimized model")
                model_to_use = self.model_name
            else:
                model_to_use = self.optimized_model_name
            
            # Aggressive GPU pre-loading
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": model_to_use,
                "prompt": "GPU warmup and model loading test - respond with 'READY'",
                "stream": False,
                "keep_alive": -1,  # Keep in memory forever
                "options": {
                    "num_predict": 5,
                    "temperature": 0.1,
                    **self.gpu_manager.get_aggressive_ollama_config()
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                async with session.post(url, json=payload) as response:
                    load_time = int((time.time() - start_time) * 1000)
                    
                    if response.status == 200:
                        logger.info(f"🚀 Aggressive GPU model loaded in {load_time}ms: {model_to_use}")
                        return model_to_use
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to load model: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error ensuring optimized model loaded: {e}")
            return None
    
    def get_optimized_model_name(self) -> str:
        """Get the name of the optimized model to use"""
        return self.optimized_model_name