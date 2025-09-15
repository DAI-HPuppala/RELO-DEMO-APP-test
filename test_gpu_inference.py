#!/usr/bin/env python3
"""Test script to verify GPU usage for Ollama inference"""

import asyncio
import aiohttp
import base64
import time
import json
import numpy as np
from PIL import Image
import io

async def test_gpu_inference():
    """Test Ollama inference with explicit GPU settings"""
    
    print("🚀 Testing Ollama GPU inference...")
    
    # Create a test image
    test_image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
    image = Image.fromarray(test_image)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Build payload with explicit GPU settings
    payload = {
        "model": "qwen2.5vl:3b",
        "prompt": "Analyze this image and identify the main objects visible. Respond in JSON format with keys: objects, colors, description",
        "images": [base64_img],
        "stream": False,
        "format": "json",
        "options": {
            "num_predict": 300,
            "temperature": 0.2,
            "num_ctx": 8192,
            "num_thread": 6,
            "top_p": 0.85,
            "top_k": 35,
            "repeat_penalty": 1.05,
            # CRITICAL GPU settings
            "num_gpu": 40,  # Force ALL layers to GPU
            "gpu_layers": 40,  # Ensure all 40 layers use GPU
            "main_gpu": 0,  # Use GPU 0
            "low_vram": False,  # Disable low VRAM mode
            "f16_kv": True,  # Use FP16 for GPU
        },
        "keep_alive": -1  # Keep in GPU memory
    }
    
    print("📊 Payload options:")
    for key, value in payload["options"].items():
        print(f"  - {key}: {value}")
    
    # Call Ollama API
    ollama_host = "http://localhost:11434"
    
    print("\n⏱️  Starting inference...")
    start_time = time.time()
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{ollama_host}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    inference_time = (time.time() - start_time) * 1000
                    
                    print(f"\n✅ Inference completed in {inference_time:.0f}ms")
                    
                    # Parse timing information
                    if 'total_duration' in result:
                        total_ms = result['total_duration'] / 1e6
                        load_ms = result.get('load_duration', 0) / 1e6
                        prompt_eval_ms = result.get('prompt_eval_duration', 0) / 1e6
                        eval_ms = result.get('eval_duration', 0) / 1e6
                        
                        print(f"\n📈 Performance metrics:")
                        print(f"  - Total duration: {total_ms:.0f}ms")
                        print(f"  - Model load: {load_ms:.0f}ms")
                        print(f"  - Prompt eval: {prompt_eval_ms:.0f}ms")
                        print(f"  - Generation: {eval_ms:.0f}ms")
                        
                        if result.get('eval_count', 0) > 0:
                            tokens_per_sec = result['eval_count'] / (eval_ms / 1000) if eval_ms > 0 else 0
                            print(f"  - Tokens generated: {result['eval_count']}")
                            print(f"  - Tokens/sec: {tokens_per_sec:.1f}")
                    
                    print(f"\n📝 Response:")
                    print(result.get('response', 'No response'))
                    
                    # Check if GPU was used
                    if inference_time < 5000:  # Less than 5 seconds is good
                        print(f"\n🎯 GPU acceleration appears to be working! ({inference_time:.0f}ms)")
                    else:
                        print(f"\n⚠️  Inference seems slow ({inference_time:.0f}ms), GPU might not be fully utilized")
                    
                else:
                    error_text = await response.text()
                    print(f"❌ Error {response.status}: {error_text}")
                    
    except Exception as e:
        print(f"❌ Failed: {e}")

async def check_gpu_status():
    """Check current GPU status"""
    import subprocess
    
    print("\n📊 Current GPU status:")
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', 
                           '--format=csv,noheader'], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Could not query GPU status")

async def main():
    """Main test function"""
    print("=" * 60)
    print("OLLAMA GPU INFERENCE TEST")
    print("=" * 60)
    
    # Check GPU status before
    await check_gpu_status()
    
    # Run multiple tests
    print("\n🔄 Running 3 inference tests...")
    for i in range(3):
        print(f"\n--- Test {i+1}/3 ---")
        await test_gpu_inference()
        
        # Check GPU status after each test
        if i == 0:  # Only after first test to see memory usage
            await check_gpu_status()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())