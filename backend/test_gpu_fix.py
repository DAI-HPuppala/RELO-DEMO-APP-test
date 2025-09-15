#!/usr/bin/env python3
"""
Test script to verify GPU initialization fix for Ollama
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.gpu_initializer import ensure_gpu_ready, gpu_initializer
from services.ollama_optimizer import ollama_optimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_gpu_initialization():
    """Test the GPU initialization and Ollama GPU inference"""
    
    print("="*70)
    print("TESTING GPU INITIALIZATION FIX FOR OLLAMA")
    print("="*70)
    
    # Step 1: Initialize GPU
    print("\n📍 Step 1: Initializing GPU and CUDA context...")
    gpu_ready = await ensure_gpu_ready()
    
    if not gpu_ready:
        print("❌ GPU initialization failed!")
        return False
    
    print("✅ GPU initialized successfully!")
    
    # Step 2: Check GPU status
    print("\n📍 Step 2: Checking GPU status...")
    gpu_status = gpu_initializer.get_gpu_status()
    print(f"   GPU Name: {gpu_status.get('gpu_name', 'Unknown')}")
    print(f"   Total Memory: {gpu_status.get('total_memory_mb', 0)} MB")
    print(f"   Used Memory: {gpu_status.get('used_memory_mb', 0)} MB")
    print(f"   Free Memory: {gpu_status.get('free_memory_mb', 0)} MB")
    print(f"   GPU Utilization: {gpu_status.get('gpu_utilization', 0)}%")
    print(f"   CUDA Available: {gpu_status.get('cuda_available', False)}")
    print(f"   GPU Initialized: {gpu_status.get('gpu_initialized', False)}")
    
    # Step 3: Apply Ollama optimizations
    print("\n📍 Step 3: Applying Ollama GPU optimizations...")
    opt_result = await ollama_optimizer.apply_optimal_configuration()
    
    if opt_result.get("status") == "success":
        print(f"✅ Ollama optimizations applied successfully!")
        print(f"   Optimization Level: {opt_result.get('optimization_level')}")
        print(f"   GPU Layers: {opt_result.get('gpu_layers')}")
    else:
        print(f"⚠️ Ollama optimization failed: {opt_result.get('error')}")
    
    # Step 4: Run benchmark
    print("\n📍 Step 4: Running GPU inference benchmark...")
    benchmark = await ollama_optimizer.benchmark_inference(3)
    
    if "error" not in benchmark:
        print(f"✅ Benchmark completed successfully!")
        print(f"   Runs: {benchmark.get('num_runs')}")
        print(f"   Average Time: {benchmark.get('avg_time', 0):.2f}s")
        print(f"   Min Time: {benchmark.get('min_time', 0):.2f}s")
        print(f"   Max Time: {benchmark.get('max_time', 0):.2f}s")
    else:
        print(f"❌ Benchmark failed: {benchmark.get('error')}")
    
    # Step 5: Final GPU status check
    print("\n📍 Step 5: Final GPU status check...")
    final_status = gpu_initializer.get_gpu_status()
    print(f"   Used Memory After: {final_status.get('used_memory_mb', 0)} MB")
    print(f"   Free Memory After: {final_status.get('free_memory_mb', 0)} MB")
    
    print("\n" + "="*70)
    if gpu_ready and opt_result.get("status") == "success":
        print("✅ GPU FIX SUCCESSFUL - OLLAMA IS USING GPU!")
    else:
        print("❌ GPU FIX INCOMPLETE - CHECK LOGS FOR DETAILS")
    print("="*70)
    
    return gpu_ready and opt_result.get("status") == "success"


async def main():
    """Main test function"""
    try:
        success = await test_gpu_initialization()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())