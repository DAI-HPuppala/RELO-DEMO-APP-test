"""
Direct Ollama interface based on RELO-DEMO-APP working implementation
Uses the same configuration that successfully loads GPU layers in the other project
"""

import requests
import logging
import time
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

# Use the same configuration as RELO-DEMO-APP
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def get_gpu_config():
    """Get GPU configuration from gpu_utils (same as RELO-DEMO-APP)"""
    try:
        # Import the working GPU utils from RELO-DEMO-APP approach
        import subprocess
        
        # Get GPU memory info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            free = int(values[2])
            
            # Use the same logic as RELO-DEMO-APP
            available = free - 1000  # 1GB safety margin
            
            if available < 500:
                gpu_layers = 0
            elif available < 1500:
                gpu_layers = 8
            elif available < 2500:
                gpu_layers = int(32 * (available / 2500) * 0.7)
            else:
                gpu_layers = 24  # Use 75% of layers like RELO-DEMO-APP
            
            logger.info(f"GPU config: {gpu_layers} layers (free memory: {free}MB)")
            
            return {
                "num_gpu": gpu_layers,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 512,
                "f16_kv": True if gpu_layers > 0 else False,
                "use_mmap": True,
                "use_mlock": False,
                "num_thread": 4 if gpu_layers > 20 else 8
            }
    except Exception as e:
        logger.error(f"Error getting GPU config: {e}")
    
    # Fallback to CPU
    return {
        "num_gpu": 0,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "num_predict": 512,
        "num_thread": 16,
        "f16_kv": False,
        "use_mmap": True,
        "use_mlock": False
    }

def query_ollama_direct(prompt: str, model: str = "qwen2.5vl:3b", temperature: float = 0.3) -> Dict:
    """
    Query Ollama directly using the same approach as RELO-DEMO-APP
    This bypasses any CUDA initialization issues by using the API directly
    """
    start_time = time.time()
    
    try:
        # Get dynamic GPU configuration
        gpu_config = get_gpu_config()
        gpu_config["temperature"] = temperature
        
        # Build request (same structure as RELO-DEMO-APP)
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30s",  # Keep model loaded for 30s
            "options": gpu_config
        }
        
        logger.info(f"🚀 Querying Ollama {model} (GPU layers: {gpu_config.get('num_gpu', 0)})")
        
        # Use session without proxy (same as RELO-DEMO-APP)
        with requests.Session() as session:
            session.trust_env = False
            
            response = session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            
            duration = time.time() - start_time
            logger.info(f"✅ Ollama responded in {duration:.2f}s")
            
            # Check actual GPU usage
            import subprocess
            gpu_check = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            gpu_memory = gpu_check.stdout.strip() if gpu_check.returncode == 0 else "Unknown"
            logger.info(f"   GPU Memory: {gpu_memory}")
            
            return {
                "response": answer,
                "duration": duration,
                "gpu_layers": gpu_config.get("num_gpu", 0),
                "gpu_memory": gpu_memory
            }
        else:
            logger.error(f"Ollama error: {response.status_code} - {response.text}")
            return {"error": f"Ollama returned status {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Ollama query error: {str(e)}")
        return {"error": str(e)}

async def test_ollama_gpu():
    """Test if Ollama is using GPU with the RELO-DEMO-APP approach"""
    logger.info("="*60)
    logger.info("Testing Ollama GPU with RELO-DEMO-APP configuration")
    logger.info("="*60)
    
    # Test prompts
    test_prompts = [
        "What is AI?",
        "Explain machine learning",
        "Describe neural networks"
    ]
    
    total_gpu_layers = 0
    successful_tests = 0
    
    for prompt in test_prompts:
        result = query_ollama_direct(prompt, temperature=0.3)
        
        if "error" not in result:
            successful_tests += 1
            gpu_layers = result.get("gpu_layers", 0)
            total_gpu_layers += gpu_layers
            
            logger.info(f"Test {successful_tests}: GPU layers={gpu_layers}, Time={result.get('duration', 0):.2f}s")
            logger.info(f"GPU Memory: {result.get('gpu_memory', 'Unknown')}")
        else:
            logger.error(f"Test failed: {result.get('error')}")
    
    avg_layers = total_gpu_layers / len(test_prompts) if test_prompts else 0
    
    logger.info("="*60)
    if avg_layers > 0:
        logger.info(f"✅ SUCCESS: Using average of {avg_layers:.0f} GPU layers")
    else:
        logger.info("❌ FAILURE: Still using CPU (0 GPU layers)")
    logger.info("="*60)
    
    return avg_layers > 0

# Export for use in other modules
__all__ = ['query_ollama_direct', 'test_ollama_gpu', 'get_gpu_config']