"""
VLM Model Initializer - Module-level initialization with warmup
Implements SOTA initialization strategy from RELO-DEMO-APP research
"""

import asyncio
import logging
import time
import os
import aiohttp
import json
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Global initialization state
_VLM_INITIALIZED = False
_INITIALIZATION_TIME = 0.0
_WARMUP_RESULTS = []
_MODEL_CONFIG = {}

class VLMInitializer:
    """Handles VLM model initialization and warmup at startup"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = "qwen2.5vl:3b"):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.initialization_complete = False
        self.warmup_complete = False
        
    async def initialize_models_at_startup(self) -> Dict[str, Any]:
        """
        Initialize all VLM models at module load time (not on first request)
        This is the SOTA approach from RELO-DEMO-APP
        """
        global _VLM_INITIALIZED, _INITIALIZATION_TIME, _MODEL_CONFIG
        
        if _VLM_INITIALIZED:
            logger.info("✅ VLM models already initialized")
            return {"status": "already_initialized", "time": _INITIALIZATION_TIME}
        
        logger.info("\n" + "="*60)
        logger.info("🚀 INITIALIZING VLM MODELS AT STARTUP")
        logger.info("="*60)
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Check Ollama availability
            logger.info("📡 Checking Ollama server availability...")
            if not await self._check_ollama_health():
                raise RuntimeError("Ollama server not available")
            
            # Step 2: Load model with optimal GPU configuration
            logger.info(f"🔄 Loading {self.model_name} with GPU optimization...")
            await self._load_model_with_gpu_optimization()
            
            # Step 3: Configure for permanent GPU residence
            logger.info("⚙️ Configuring model for permanent GPU residence...")
            await self._configure_permanent_gpu_residence()
            
            # Step 4: Run warmup sequences
            logger.info("🔥 Running warmup sequences...")
            warmup_results = await self._run_warmup_sequences()
            results["warmup"] = warmup_results
            
            # Step 5: Verify model is ready
            logger.info("✓ Verifying model readiness...")
            if await self._verify_model_ready():
                _VLM_INITIALIZED = True
                _INITIALIZATION_TIME = time.time() - start_time
                
                logger.info("="*60)
                logger.info(f"✅ VLM INITIALIZATION COMPLETE in {_INITIALIZATION_TIME:.2f}s")
                logger.info("="*60 + "\n")
                
                results["status"] = "success"
                results["initialization_time"] = _INITIALIZATION_TIME
                results["model"] = self.model_name
                
            else:
                raise RuntimeError("Model verification failed")
                
        except Exception as e:
            logger.error(f"❌ VLM initialization failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            
        return results
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama server is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags", 
                                      timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
                        
                        if self.model_name in models:
                            logger.info(f"✅ Model {self.model_name} is available")
                            return True
                        else:
                            logger.warning(f"⚠️ Model {self.model_name} not found. Available: {models}")
                            # Try to pull the model
                            return await self._pull_model()
                    return False
        except Exception as e:
            logger.error(f"Failed to check Ollama health: {e}")
            return False
    
    async def _pull_model(self) -> bool:
        """Pull the model if not available"""
        logger.info(f"📥 Pulling model {self.model_name}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/pull",
                                       json={"name": self.model_name},
                                       timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        # Read streaming response
                        async for line in response.content:
                            if line:
                                try:
                                    status = json.loads(line)
                                    if status.get("status") == "success":
                                        logger.info(f"✅ Model {self.model_name} pulled successfully")
                                        return True
                                except json.JSONDecodeError:
                                    continue
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
        return False
    
    async def _load_model_with_gpu_optimization(self):
        """Load model with optimal GPU configuration"""
        # Import GPU optimizer
        from config.gpu_optimizer import gpu_optimizer
        
        # Get optimal GPU configuration
        gpu_config = gpu_optimizer.calculate_optimal_config(target_fps=10)
        logger.info(f"📊 GPU Config: {gpu_optimizer.get_optimization_summary(gpu_config)}")
        
        # Store configuration
        global _MODEL_CONFIG
        _MODEL_CONFIG = {
            "gpu_layers": gpu_config.cuda_optimizations.get("num_gpu", -1),
            "optimization_level": gpu_config.optimization_level.value,
            "memory_available": gpu_config.available_memory_mb
        }
        
        # Load model with optimal settings
        load_payload = {
            "model": self.model_name,
            "prompt": "Initialize",
            "stream": False,
            "keep_alive": -1,  # Keep in memory permanently
            "options": gpu_config.cuda_optimizations
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # First load can take up to 60 seconds
                async with session.post(f"{self.ollama_host}/api/generate",
                                       json=load_payload,
                                       timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"✅ Model loaded with GPU optimization")
                        return True
                    else:
                        error = await response.text()
                        raise RuntimeError(f"Failed to load model: {error}")
        except asyncio.TimeoutError:
            logger.warning("Initial model load timed out - this is normal for first load")
            # Give it more time and check if it loaded
            await asyncio.sleep(5)
            return await self._check_ollama_health()
    
    async def _configure_permanent_gpu_residence(self):
        """Configure model to stay in GPU permanently"""
        config_payload = {
            "model": self.model_name,
            "keep_alive": -1  # Never unload
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/generate",
                                       json={**config_payload, "prompt": "", "stream": False},
                                       timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        logger.info("✅ Model configured for permanent GPU residence")
                        return True
        except Exception as e:
            logger.warning(f"Could not configure permanent residence: {e}")
            return False
    
    async def _run_warmup_sequences(self) -> Dict[str, Any]:
        """
        Run warmup sequences to pre-compile and optimize the model
        This is critical for performance - from RELO-DEMO-APP research
        """
        global _WARMUP_RESULTS
        
        warmup_prompts = [
            # Simple classification warmups
            {"prompt": "What color is this item?", "expected_time": 2.0},
            {"prompt": "Describe the clothing type", "expected_time": 1.5},
            {"prompt": "Is there any damage visible?", "expected_time": 1.5},
            # Multi-frame inference warmup
            {"prompt": "Analyze these frames and identify attributes", "expected_time": 3.0},
            # Complex aggregation warmup
            {"prompt": "Based on analysis: color=blue, type=shirt, damage=none. Summarize.", "expected_time": 2.0}
        ]
        
        results = []
        total_warmup_time = 0
        
        for i, warmup in enumerate(warmup_prompts, 1):
            logger.info(f"  Warmup {i}/{len(warmup_prompts)}: {warmup['prompt'][:50]}...")
            
            start = time.time()
            try:
                # Create simple test image
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Run inference
                async with aiohttp.ClientSession() as session:
                    # For now, just send text prompt (image support to be added)
                    async with session.post(f"{self.ollama_host}/api/generate",
                                           json={
                                               "model": self.model_name,
                                               "prompt": warmup["prompt"],
                                               "stream": False,
                                               "options": {"num_predict": 50}
                                           },
                                           timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            inference_time = time.time() - start
                            total_warmup_time += inference_time
                            
                            status = "✓" if inference_time <= warmup["expected_time"] else "⚠"
                            logger.info(f"    {status} Completed in {inference_time:.2f}s")
                            
                            results.append({
                                "prompt": warmup["prompt"],
                                "time": inference_time,
                                "expected": warmup["expected_time"],
                                "success": True
                            })
                        else:
                            results.append({
                                "prompt": warmup["prompt"],
                                "success": False,
                                "error": f"HTTP {response.status}"
                            })
                            
            except Exception as e:
                logger.warning(f"    ✗ Warmup failed: {e}")
                results.append({
                    "prompt": warmup["prompt"],
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate statistics
        successful = [r for r in results if r.get("success")]
        if successful:
            avg_time = sum(r["time"] for r in successful) / len(successful)
            logger.info(f"🎯 Warmup complete: {len(successful)}/{len(warmup_prompts)} successful, avg time: {avg_time:.2f}s")
        
        _WARMUP_RESULTS = results
        self.warmup_complete = True
        
        return {
            "total_warmups": len(warmup_prompts),
            "successful": len(successful),
            "total_time": total_warmup_time,
            "average_time": avg_time if successful else 0,
            "results": results
        }
    
    async def _verify_model_ready(self) -> bool:
        """Verify model is ready for production inference"""
        try:
            # Quick test inference
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/generate",
                                       json={
                                           "model": self.model_name,
                                           "prompt": "Ready check",
                                           "stream": False,
                                           "options": {"num_predict": 1}
                                       },
                                       timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.initialization_complete = True
                        return True
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
        return False
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        global _VLM_INITIALIZED, _INITIALIZATION_TIME, _MODEL_CONFIG, _WARMUP_RESULTS
        
        return {
            "initialized": _VLM_INITIALIZED,
            "initialization_time": _INITIALIZATION_TIME,
            "warmup_complete": self.warmup_complete,
            "warmup_results": _WARMUP_RESULTS,
            "model_config": _MODEL_CONFIG,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat()
        }

# Global initializer instance
vlm_initializer = VLMInitializer()

async def initialize_vlm_at_startup():
    """
    Function to be called at application startup
    This ensures models are loaded before any requests
    """
    logger.info("🚀 Starting VLM initialization at application startup...")
    result = await vlm_initializer.initialize_models_at_startup()
    return result

# Export for use in other modules
__all__ = ['vlm_initializer', 'initialize_vlm_at_startup']