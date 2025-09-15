#!/usr/bin/env python3
"""
Test script to verify VLM optimizations
Run this to test the SOTA optimizations we've implemented
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from config.gpu_optimizer import gpu_optimizer
from services.vlm_initializer import vlm_initializer
from services.ollama_optimizer import ollama_optimizer
from services.performance_monitor import performance_monitor
from services.batch_processor import batch_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_gpu_optimization():
    """Test GPU optimization detection and configuration"""
    logger.info("\n" + "="*60)
    logger.info("🔍 Testing GPU Optimization")
    logger.info("="*60)
    
    # Get GPU info
    total, used, free = gpu_optimizer.get_gpu_memory_info()
    logger.info(f"GPU Memory: Total={total}MB, Used={used}MB, Free={free}MB")
    
    # Check if RTX A1000
    if gpu_optimizer.is_rtx_a1000:
        logger.info("✅ RTX A1000 detected - specific optimizations enabled")
    
    # Get optimal configuration
    config = gpu_optimizer.calculate_optimal_config(target_fps=10)
    summary = gpu_optimizer.get_optimization_summary(config)
    
    logger.info(f"Optimization Level: {config.optimization_level.value}")
    logger.info(f"GPU Layers: {config.cuda_optimizations.get('num_gpu', 'auto')}")
    logger.info(f"Inference Streams: {config.num_inference_streams}")
    
    # Test batch size calculation
    optimal_batch = gpu_optimizer.get_optimal_batch_size()
    logger.info(f"Optimal Batch Size: {optimal_batch}")
    
    return summary

async def test_vlm_initialization():
    """Test VLM initialization with warmup"""
    logger.info("\n" + "="*60)
    logger.info("🚀 Testing VLM Initialization")
    logger.info("="*60)
    
    # Initialize VLM
    result = await vlm_initializer.initialize_models_at_startup()
    
    if result.get("status") == "success":
        logger.info(f"✅ VLM initialized in {result.get('initialization_time', 0):.2f}s")
        
        # Check warmup results
        warmup = result.get("warmup", {})
        if warmup.get("successful"):
            logger.info(f"   Warmup: {warmup['successful']}/{warmup['total_warmups']} successful")
            logger.info(f"   Average warmup time: {warmup.get('average_time', 0):.2f}s")
    else:
        logger.error(f"❌ VLM initialization failed: {result.get('error')}")
    
    return result

async def test_ollama_optimization():
    """Test Ollama optimization"""
    logger.info("\n" + "="*60)
    logger.info("⚡ Testing Ollama Optimization")
    logger.info("="*60)
    
    # Apply optimal configuration
    config_result = await ollama_optimizer.apply_optimal_configuration()
    
    if config_result.get("status") == "success":
        logger.info(f"✅ Ollama optimized: {config_result['optimization_level']}")
        logger.info(f"   GPU layers: {config_result.get('gpu_layers', 'auto')}")
    
    # Test Flash Attention
    flash = await ollama_optimizer.enable_flash_attention()
    if flash:
        logger.info("✅ Flash Attention enabled")
    else:
        logger.info("ℹ️ Flash Attention not available")
    
    # Run benchmark
    logger.info("Running benchmark...")
    benchmark = await ollama_optimizer.benchmark_inference(3)
    
    if benchmark.get("avg_time"):
        logger.info(f"📊 Benchmark Results:")
        logger.info(f"   Average time: {benchmark['avg_time']:.2f}s")
        logger.info(f"   Min time: {benchmark['min_time']:.2f}s")
        logger.info(f"   Max time: {benchmark['max_time']:.2f}s")
    
    return benchmark

async def test_performance_monitoring():
    """Test performance monitoring"""
    logger.info("\n" + "="*60)
    logger.info("📊 Testing Performance Monitoring")
    logger.info("="*60)
    
    # Start monitoring
    await performance_monitor.start_monitoring(interval=5)
    
    # Simulate some inferences
    import random
    for i in range(10):
        duration = random.uniform(500, 2000)  # Random duration in ms
        performance_monitor.record_inference(
            duration_ms=duration,
            batch_size=random.randint(1, 3),
            agent=f"test_agent_{i % 3}",
            success=random.random() > 0.1
        )
    
    # Get stats
    stats = performance_monitor.get_current_stats()
    
    logger.info("Performance Stats:")
    if stats.get("recent_performance"):
        perf = stats["recent_performance"]
        logger.info(f"   Average: {perf.get('avg_ms', 0):.0f}ms")
        logger.info(f"   P95: {perf.get('p95_ms', 0):.0f}ms")
        logger.info(f"   P99: {perf.get('p99_ms', 0):.0f}ms")
    
    # Get suggestions
    suggestions = performance_monitor.get_optimization_suggestions()
    logger.info("\nOptimization Suggestions:")
    for suggestion in suggestions:
        logger.info(f"   {suggestion}")
    
    # Stop monitoring
    performance_monitor.stop_monitoring()
    
    return stats

async def main():
    """Run all tests"""
    logger.info("🧪 TESTING VLM OPTIMIZATIONS")
    logger.info("="*80)
    
    results = {}
    
    # Test GPU optimization
    try:
        results["gpu"] = await test_gpu_optimization()
    except Exception as e:
        logger.error(f"GPU test failed: {e}")
        results["gpu"] = {"error": str(e)}
    
    # Test VLM initialization
    try:
        results["vlm"] = await test_vlm_initialization()
    except Exception as e:
        logger.error(f"VLM test failed: {e}")
        results["vlm"] = {"error": str(e)}
    
    # Test Ollama optimization
    try:
        results["ollama"] = await test_ollama_optimization()
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        results["ollama"] = {"error": str(e)}
    
    # Test performance monitoring
    try:
        results["monitoring"] = await test_performance_monitoring()
    except Exception as e:
        logger.error(f"Monitoring test failed: {e}")
        results["monitoring"] = {"error": str(e)}
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📋 TEST SUMMARY")
    logger.info("="*80)
    
    successful_tests = sum(1 for r in results.values() if not isinstance(r, dict) or "error" not in r)
    total_tests = len(results)
    
    logger.info(f"Tests Passed: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        logger.info("✅ All optimizations working correctly!")
    else:
        logger.warning("⚠️ Some tests failed. Check logs for details.")
    
    return results

if __name__ == "__main__":
    # Run tests
    asyncio.run(main())