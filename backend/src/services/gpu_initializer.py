"""
GPU Initialization and CUDA Context Management for Ollama
Handles CUDA initialization failures and ensures proper GPU setup
"""

import subprocess
import logging
import time
import os
import asyncio
import aiohttp
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUInitializer:
    """Handles GPU initialization and CUDA context management for Ollama"""
    
    def __init__(self):
        self.gpu_initialized = False
        self.cuda_available = False
        self.initialization_attempts = 0
        self.max_attempts = 3
        
    def check_cuda_availability(self) -> bool:
        """Check if CUDA is available and properly configured"""
        try:
            # Check CUDA environment variables
            cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
            if not cuda_path:
                # Common CUDA installation paths
                possible_paths = ['/usr/local/cuda', '/usr/cuda', '/opt/cuda']
                for path in possible_paths:
                    if os.path.exists(path):
                        os.environ['CUDA_PATH'] = path
                        logger.info(f"Set CUDA_PATH to {path}")
                        break
            
            # Check nvidia-smi availability
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                compute_cap = result.stdout.strip()
                logger.info(f"GPU Compute Capability: {compute_cap}")
                
                # RTX A1000 has compute capability 8.6
                if compute_cap:
                    major, minor = compute_cap.split('.')
                    if int(major) >= 5:  # Minimum compute capability for Ollama
                        self.cuda_available = True
                        return True
                        
            logger.warning("CUDA not properly configured or GPU compute capability too low")
            return False
            
        except Exception as e:
            logger.error(f"Failed to check CUDA availability: {e}")
            return False
    
    def reset_gpu_context(self) -> bool:
        """Reset GPU context to clear any stuck CUDA states"""
        try:
            logger.info("Resetting GPU context...")
            
            # Kill any existing Ollama processes that might hold GPU context
            subprocess.run(['pkill', '-f', 'ollama'], capture_output=True, timeout=5)
            time.sleep(2)
            
            # Check if nvidia-persistenced is running (helps with context issues)
            persist_result = subprocess.run(
                ['pgrep', 'nvidia-persistenced'], capture_output=True, timeout=5
            )
            
            if persist_result.returncode != 0:
                logger.info("nvidia-persistenced not running, attempting to start...")
                # Try to start nvidia-persistenced (may need sudo)
                subprocess.run(
                    ['nvidia-persistenced', '--user=ollama'], 
                    capture_output=True, timeout=5
                )
            
            # Clear any existing CUDA cache
            cuda_cache_path = os.path.expanduser('~/.nv/ComputeCache')
            if os.path.exists(cuda_cache_path):
                logger.info("Clearing CUDA compute cache...")
                subprocess.run(['rm', '-rf', cuda_cache_path], capture_output=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset GPU context: {e}")
            return False
    
    def set_cuda_environment(self) -> None:
        """Set optimal CUDA environment variables for Ollama"""
        # Set CUDA device explicitly
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Set compute capability for RTX A1000
        os.environ['OLLAMA_CUDA_COMPUTE_CAP'] = '8.6'
        
        # Enable GPU memory fraction
        os.environ['OLLAMA_GPU_MEMORY_FRACTION'] = '0.85'
        
        # Disable CUDA memory pooling to avoid fragmentation
        os.environ['CUDA_CACHE_DISABLE'] = '0'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        # Force CUDA to use the primary GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        logger.info("CUDA environment variables set:")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        logger.info(f"  OLLAMA_CUDA_COMPUTE_CAP: {os.environ.get('OLLAMA_CUDA_COMPUTE_CAP')}")
        logger.info(f"  OLLAMA_GPU_MEMORY_FRACTION: {os.environ.get('OLLAMA_GPU_MEMORY_FRACTION')}")
    
    async def test_ollama_gpu_inference(self, model_name: str = "qwen2.5vl:3b") -> bool:
        """Test if Ollama can perform GPU inference"""
        try:
            logger.info(f"Testing GPU inference with {model_name}...")
            
            async with aiohttp.ClientSession() as session:
                # First, ensure model is loaded with GPU layers
                payload = {
                    "model": model_name,
                    "prompt": "GPU test",
                    "stream": False,
                    "options": {
                        "num_gpu": -1,  # Use all available GPU layers
                        "num_predict": 1,
                        "temperature": 0.1
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Check if GPU is being used by monitoring nvidia-smi
                        gpu_check = subprocess.run(
                            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                             '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=5
                        )
                        
                        if gpu_check.returncode == 0 and 'ollama' in gpu_check.stdout.lower():
                            logger.info(" GPU inference confirmed - Ollama using GPU!")
                            return True
                        else:
                            logger.warning(" Inference completed but GPU usage not detected")
                            return False
                    else:
                        logger.error(f"Inference failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"GPU inference test failed: {e}")
            return False
    
    async def initialize_gpu_for_ollama(self) -> bool:
        """Complete GPU initialization process for Ollama"""
        logger.info("="*60)
        logger.info("Starting GPU Initialization for Ollama")
        logger.info("="*60)
        
        while self.initialization_attempts < self.max_attempts:
            self.initialization_attempts += 1
            logger.info(f"Initialization attempt {self.initialization_attempts}/{self.max_attempts}")
            
            # Step 1: Check CUDA availability
            if not self.check_cuda_availability():
                logger.error("CUDA not available, cannot use GPU")
                return False
            
            # Step 2: Reset GPU context if needed
            if self.initialization_attempts > 1:
                if not self.reset_gpu_context():
                    logger.warning("GPU context reset failed, continuing anyway...")
            
            # Step 3: Set CUDA environment
            self.set_cuda_environment()
            
            # Step 4: Restart Ollama service with new environment
            logger.info("Restarting Ollama service...")
            subprocess.run(['pkill', '-f', 'ollama serve'], capture_output=True)
            time.sleep(2)
            
            # Start Ollama in background with environment variables
            env = os.environ.copy()
            ollama_process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for Ollama to start
            logger.info("Waiting for Ollama to start...")
            for _ in range(10):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:11434/api/tags") as response:
                            if response.status == 200:
                                logger.info("Ollama service is ready")
                                break
                except:
                    await asyncio.sleep(1)
            
            # Step 5: Test GPU inference
            if await self.test_ollama_gpu_inference():
                self.gpu_initialized = True
                logger.info("="*60)
                logger.info(" GPU INITIALIZATION SUCCESSFUL!")
                logger.info("="*60)
                return True
            
            logger.warning(f"Attempt {self.initialization_attempts} failed, retrying...")
            await asyncio.sleep(2)
        
        logger.error("="*60)
        logger.error(" GPU INITIALIZATION FAILED AFTER ALL ATTEMPTS")
        logger.error("="*60)
        return False
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and memory usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                return {
                    "gpu_name": parts[0],
                    "total_memory_mb": int(parts[1].replace(' MiB', '')),
                    "used_memory_mb": int(parts[2].replace(' MiB', '')),
                    "free_memory_mb": int(parts[3].replace(' MiB', '')),
                    "gpu_utilization": int(parts[4].replace(' %', '')),
                    "cuda_available": self.cuda_available,
                    "gpu_initialized": self.gpu_initialized
                }
        except Exception as e:
            logger.error(f"Failed to get GPU status: {e}")
        
        return {
            "error": "Failed to get GPU status",
            "cuda_available": self.cuda_available,
            "gpu_initialized": self.gpu_initialized
        }


# Global GPU initializer instance
gpu_initializer = GPUInitializer()


async def ensure_gpu_ready() -> bool:
    """Ensure GPU is ready for Ollama inference"""
    if gpu_initializer.gpu_initialized:
        return True
    
    return await gpu_initializer.initialize_gpu_for_ollama()


async def main():
    """Test GPU initialization"""
    logging.basicConfig(level=logging.INFO)
    
    success = await ensure_gpu_ready()
    if success:
        status = gpu_initializer.get_gpu_status()
        logger.info(f"GPU Status: {status}")
    else:
        logger.error("GPU initialization failed")


if __name__ == "__main__":
    asyncio.run(main())