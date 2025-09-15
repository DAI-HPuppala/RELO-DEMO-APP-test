#!/usr/bin/env python3
"""
Workaround for Ollama GPU initialization error 999
Forces GPU usage through direct API calls with specific configuration
"""

import subprocess
import requests
import json
import time
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stop_ollama():
    """Stop existing Ollama service"""
    logger.info("Stopping Ollama service...")
    subprocess.run(["sudo", "systemctl", "stop", "ollama"], capture_output=True)
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(2)

def start_ollama_custom():
    """Start Ollama with custom environment"""
    logger.info("Starting Ollama with custom configuration...")
    
    # Set minimal environment to avoid conflicts
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/home/automation-dev",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_MODELS": "/usr/share/ollama/.ollama/models",
        # Try forcing CPU mode first, then migrate to GPU
        "OLLAMA_NUM_GPU": "0",  # Start with CPU
    }
    
    # Start Ollama
    process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    logger.info(f"Ollama started with PID: {process.pid}")
    
    # Wait for Ollama to be ready
    for _ in range(20):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=1)
            if response.status_code == 200:
                logger.info("Ollama is ready")
                return process
        except:
            time.sleep(1)
    
    logger.error("Ollama failed to start")
    return None

def load_model_with_gpu_layers(model_name="qwen2.5vl:3b"):
    """Load model with explicit GPU layer configuration"""
    logger.info(f"Loading model {model_name} with GPU layer migration...")
    
    # First, load model in CPU mode
    logger.info("Step 1: Loading model in CPU mode...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": "Init",
            "stream": False,
            "options": {
                "num_gpu": 0,  # Start with CPU
                "num_predict": 1,
                "num_thread": 8
            }
        },
        timeout=60
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to load model: {response.text}")
        return False
    
    logger.info("Model loaded in CPU mode")
    
    # Now try to migrate layers to GPU progressively
    for gpu_layers in [8, 16, 24, 32, 37]:
        logger.info(f"Step 2: Attempting to migrate {gpu_layers} layers to GPU...")
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"Testing {gpu_layers} GPU layers",
                    "stream": False,
                    "keep_alive": -1,  # Keep in memory
                    "options": {
                        "num_gpu": gpu_layers,
                        "gpu_layers": gpu_layers,
                        "num_predict": 5,
                        "num_thread": 4,
                        "f16_kv": True,
                        "use_mmap": True
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                total_duration = result.get('total_duration', 0) / 1e9  # Convert to seconds
                logger.info(f"✅ Successfully used {gpu_layers} GPU layers - Response time: {total_duration:.2f}s")
                
                # Check if actually using GPU
                gpu_check = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                    capture_output=True, text=True
                )
                gpu_memory = gpu_check.stdout.strip()
                logger.info(f"   GPU Memory Usage: {gpu_memory}")
                
                if int(gpu_memory.replace(" MiB", "")) > 1000:  # If using more than 1GB
                    logger.info(f"🎉 GPU is being used with {gpu_layers} layers!")
                    return True
            else:
                logger.warning(f"Failed with {gpu_layers} layers: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error with {gpu_layers} layers: {e}")
            break
    
    return False

def test_performance():
    """Test inference performance"""
    logger.info("\nTesting inference performance...")
    
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing",
        "Describe machine learning"
    ]
    
    total_time = 0
    for prompt in prompts:
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5vl:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 50,
                    "temperature": 0.3
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            elapsed = time.time() - start
            total_time += elapsed
            logger.info(f"   Prompt processed in {elapsed:.2f}s")
    
    avg_time = total_time / len(prompts)
    logger.info(f"\n📊 Average inference time: {avg_time:.2f}s")
    
    # Check final GPU usage
    gpu_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    logger.info(f"📊 GPU Status: {gpu_check.stdout.strip()}")

def main():
    """Main workaround process"""
    logger.info("="*70)
    logger.info("OLLAMA GPU WORKAROUND - CUDA ERROR 999 FIX")
    logger.info("="*70)
    
    # Check GPU availability
    gpu_check = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    logger.info(f"GPU: {gpu_check.stdout.strip()}")
    
    # Stop existing Ollama
    stop_ollama()
    
    # Start Ollama with custom config
    ollama_process = start_ollama_custom()
    if not ollama_process:
        logger.error("Failed to start Ollama")
        return 1
    
    # Try to load model with GPU layers
    success = load_model_with_gpu_layers()
    
    if success:
        # Test performance
        test_performance()
        logger.info("\n✅ WORKAROUND SUCCESSFUL - GPU is being used!")
        logger.info("Keep this process running or save the configuration")
    else:
        logger.error("\n❌ WORKAROUND FAILED - Still using CPU")
        logger.info("Consider:")
        logger.info("1. Installing CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
        logger.info("2. Downgrading NVIDIA driver to 535.x")
        logger.info("3. Using Docker with --gpus all flag")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())