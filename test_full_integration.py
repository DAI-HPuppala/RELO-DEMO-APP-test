#!/usr/bin/env python3
"""
Comprehensive integration test for VLM optimizations
Tests all new modules and their integration with existing code
"""

import asyncio
import sys
import time
import numpy as np
from pathlib import Path
import logging
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from config.gpu_optimizer import gpu_optimizer, GPUConfig, OptimizationLevel
from services.vlm_initializer import vlm_initializer
from services.ollama_optimizer import ollama_optimizer
from services.batch_processor import batch_processor
from services.performance_monitor import performance_monitor, InferenceMetric
from services.vllm_server import vllm_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
    
    def add_pass(self, test_name):
        self.passed += 1
        logger.info(f"✅ PASS: {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        logger.error(f"❌ FAIL: {test_name} - {error}")
    
    def add_warning(self, message):
        self.warnings.append(message)
        logger.warning(f"⚠️ WARNING: {message}")
    
    def print_summary(self):
        total = self.passed + self.failed
        logger.info("\n" + "="*60)
        logger.info(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.failed > 0:
            logger.info(f"\nFailed tests:")
            for name, error in self.errors:
                logger.info(f"  - {name}: {error}")
        if self.warnings:
            logger.info(f"\nWarnings:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        logger.info("="*60)
        return self.failed == 0

results = TestResults()

# ========== GPU OPTIMIZER TESTS ==========

def test_gpu_optimizer():
    """Test GPU optimizer functionality"""
    logger.info("\n🔍 Testing GPU Optimizer...")
    
    try:
        # Test GPU memory info
        total, used, free = gpu_optimizer.get_gpu_memory_info()
        assert total is not None, "Could not get GPU memory info"
        assert total > 0, "Invalid GPU memory total"
        results.add_pass("GPU memory info retrieval")
        
        # Test RTX A1000 detection
        is_a1000 = gpu_optimizer.is_rtx_a1000
        logger.info(f"  RTX A1000 detected: {is_a1000}")
        results.add_pass("GPU model detection")
        
        # Test optimal config calculation
        config = gpu_optimizer.calculate_optimal_config(target_fps=10)
        assert isinstance(config, GPUConfig), "Invalid config type"
        assert config.optimization_level in OptimizationLevel, "Invalid optimization level"
        results.add_pass("Optimal config calculation")
        
        # Test memory pressure
        pressure = gpu_optimizer.get_memory_pressure()
        assert 0 <= pressure <= 1, f"Invalid memory pressure: {pressure}"
        results.add_pass("Memory pressure calculation")
        
        # Test batch size prediction
        predicted_mem = gpu_optimizer.predict_batch_memory(3)
        assert predicted_mem > 0, "Invalid batch memory prediction"
        results.add_pass("Batch memory prediction")
        
        # Test optimal batch size
        optimal_batch = gpu_optimizer.get_optimal_batch_size()
        assert 1 <= optimal_batch <= 5, f"Invalid optimal batch size: {optimal_batch}"
        results.add_pass("Optimal batch size calculation")
        
        # Test memory stats
        stats = gpu_optimizer.get_memory_stats()
        assert "total_mb" in stats, "Missing total memory in stats"
        assert "pressure" in stats, "Missing pressure in stats"
        results.add_pass("Memory stats generation")
        
    except Exception as e:
        results.add_fail("GPU optimizer tests", str(e))

# ========== VLM INITIALIZER TESTS ==========

async def test_vlm_initializer():
    """Test VLM initializer functionality"""
    logger.info("\n🚀 Testing VLM Initializer...")
    
    try:
        # Test initialization status check
        status = vlm_initializer.get_initialization_status()
        assert isinstance(status, dict), "Invalid status type"
        assert "initialized" in status, "Missing initialized field"
        results.add_pass("Initialization status check")
        
        # Test Ollama health check (don't actually initialize to save time)
        health = await vlm_initializer._check_ollama_health()
        if not health:
            results.add_warning("Ollama server not running - skipping full initialization test")
        else:
            results.add_pass("Ollama health check")
        
    except Exception as e:
        results.add_fail("VLM initializer tests", str(e))

# ========== OLLAMA OPTIMIZER TESTS ==========

async def test_ollama_optimizer():
    """Test Ollama optimizer functionality"""
    logger.info("\n⚡ Testing Ollama Optimizer...")
    
    try:
        # Test model info retrieval
        model_info = await ollama_optimizer.get_model_info()
        assert isinstance(model_info, dict), "Invalid model info type"
        results.add_pass("Model info retrieval")
        
        # Test optimization status
        status = ollama_optimizer.get_optimization_status()
        assert "optimization_applied" in status, "Missing optimization_applied field"
        assert "gpu_stats" in status, "Missing gpu_stats field"
        results.add_pass("Optimization status check")
        
        # Check if Ollama is running
        if model_info.get("loaded"):
            results.add_pass("Ollama model loaded")
        else:
            results.add_warning("Ollama model not loaded - skipping optimization tests")
        
    except Exception as e:
        results.add_fail("Ollama optimizer tests", str(e))

# ========== BATCH PROCESSOR TESTS ==========

async def test_batch_processor():
    """Test batch processor functionality"""
    logger.info("\n📦 Testing Batch Processor...")
    
    try:
        # Create test images
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        # Test single image preprocessing
        processed, cache_key = batch_processor.preprocess_image(test_images[0])
        assert processed.shape == (448, 448, 3), f"Invalid processed shape: {processed.shape}"
        assert processed.dtype == np.float32, f"Invalid dtype: {processed.dtype}"
        assert 0 <= processed.min() <= processed.max() <= 1, "Invalid normalization"
        results.add_pass("Single image preprocessing")
        
        # Test batch preprocessing
        processed_batch = await batch_processor.preprocess_batch(test_images)
        assert len(processed_batch) == len(test_images), "Batch size mismatch"
        for img in processed_batch:
            assert img.shape == (448, 448, 3), "Invalid batch image shape"
        results.add_pass("Batch preprocessing")
        
        # Test optimal batch creation
        batches = batch_processor.create_optimal_batches(test_images * 3)  # 9 images
        assert len(batches) > 0, "No batches created"
        total_images = sum(len(b) for b in batches)
        assert total_images == 9, f"Image count mismatch: {total_images} != 9"
        results.add_pass("Optimal batch creation")
        
        # Test base64 encoding
        encoded = batch_processor.encode_images_for_api(test_images[:2])
        assert len(encoded) == 2, "Encoding count mismatch"
        assert all(isinstance(e, str) for e in encoded), "Invalid encoding type"
        results.add_pass("Base64 image encoding")
        
        # Test cache clear
        batch_processor.clear_cache()
        results.add_pass("Cache clearing")
        
    except Exception as e:
        results.add_fail("Batch processor tests", str(e))

# ========== PERFORMANCE MONITOR TESTS ==========

def test_performance_monitor():
    """Test performance monitor functionality"""
    logger.info("\n📊 Testing Performance Monitor...")
    
    try:
        # Reset metrics for clean test
        performance_monitor.reset_metrics()
        results.add_pass("Metrics reset")
        
        # Record test metrics
        for i in range(5):
            performance_monitor.record_inference(
                duration_ms=500 + i * 100,
                batch_size=i % 3 + 1,
                model="test_model",
                agent=f"test_agent_{i % 2}",
                success=True,
                ttft_ms=100 + i * 20,
                tokens_per_second=50 - i * 5
            )
        
        # Record a failed inference
        performance_monitor.record_inference(
            duration_ms=5000,
            batch_size=1,
            agent="test_agent_fail",
            success=False,
            error="Test error"
        )
        results.add_pass("Metric recording")
        
        # Test current stats
        stats = performance_monitor.get_current_stats()
        assert "total_inferences" in stats, "Missing total_inferences"
        assert stats["total_inferences"] == 6, f"Wrong inference count: {stats['total_inferences']}"
        assert stats["successful_inferences"] == 5, f"Wrong success count: {stats['successful_inferences']}"
        results.add_pass("Current stats retrieval")
        
        # Test agent stats
        agent_stats = performance_monitor.get_agent_stats("test_agent_0")
        assert "total_inferences" in agent_stats, "Missing agent inferences"
        assert agent_stats["total_inferences"] > 0, "No agent inferences recorded"
        results.add_pass("Agent stats retrieval")
        
        # Test performance category
        category = performance_monitor.get_performance_category(450)
        assert category == "excellent", f"Wrong category: {category}"
        category = performance_monitor.get_performance_category(1500)
        assert category == "acceptable", f"Wrong category: {category}"
        results.add_pass("Performance categorization")
        
        # Test optimization suggestions
        suggestions = performance_monitor.get_optimization_suggestions()
        assert isinstance(suggestions, list), "Invalid suggestions type"
        assert len(suggestions) > 0, "No suggestions generated"
        results.add_pass("Optimization suggestions")
        
        # Test metrics export
        export_path = "/tmp/test_metrics.json"
        performance_monitor.export_metrics(export_path)
        
        # Verify export file
        with open(export_path, 'r') as f:
            exported = json.load(f)
            assert "metrics" in exported, "Missing metrics in export"
            assert len(exported["metrics"]) == 6, "Wrong metric count in export"
        results.add_pass("Metrics export")
        
        # Clean up
        import os
        os.remove(export_path)
        
    except Exception as e:
        results.add_fail("Performance monitor tests", str(e))

# ========== VLLM SERVER TESTS ==========

async def test_vllm_server():
    """Test vLLM server functionality"""
    logger.info("\n🚄 Testing vLLM Server...")
    
    try:
        # Test status check
        status = vllm_server.get_status()
        assert "is_running" in status, "Missing is_running field"
        assert "model" in status, "Missing model field"
        assert "optimizations" in status, "Missing optimizations field"
        results.add_pass("vLLM status check")
        
        # Check if vLLM package is available
        try:
            import vllm
            results.add_pass("vLLM package available")
        except ImportError:
            results.add_warning("vLLM package not installed - skipping server tests")
        
    except Exception as e:
        results.add_fail("vLLM server tests", str(e))

# ========== INTEGRATION TESTS ==========

async def test_integration():
    """Test integration between modules"""
    logger.info("\n🔗 Testing Module Integration...")
    
    try:
        # Test GPU optimizer integration with batch processor
        optimal_batch = gpu_optimizer.get_optimal_batch_size()
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(optimal_batch)
        ]
        
        batches = batch_processor.create_optimal_batches(test_images)
        assert len(batches) > 0, "Failed to create batches"
        results.add_pass("GPU optimizer + Batch processor integration")
        
        # Test performance monitor with GPU stats
        performance_monitor.record_inference(
            duration_ms=750,
            batch_size=optimal_batch,
            agent="integration_test"
        )
        
        stats = performance_monitor.get_current_stats()
        assert "gpu_memory" in stats, "Missing GPU memory in performance stats"
        results.add_pass("Performance monitor + GPU optimizer integration")
        
        # Test Ollama optimizer with GPU config
        ollama_status = ollama_optimizer.get_optimization_status()
        assert "gpu_stats" in ollama_status, "Missing GPU stats in Ollama status"
        gpu_stats = ollama_status["gpu_stats"]
        assert "total_mb" in gpu_stats, "Missing total memory in GPU stats"
        results.add_pass("Ollama optimizer + GPU optimizer integration")
        
    except Exception as e:
        results.add_fail("Integration tests", str(e))

# ========== MAIN TEST RUNNER ==========

async def main():
    """Run all tests"""
    logger.info("🧪 RUNNING COMPREHENSIVE INTEGRATION TESTS")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Run synchronous tests
    test_gpu_optimizer()
    test_performance_monitor()
    
    # Run async tests
    await test_vlm_initializer()
    await test_ollama_optimizer()
    await test_batch_processor()
    await test_vllm_server()
    await test_integration()
    
    # Print summary
    elapsed = time.time() - start_time
    logger.info(f"\n⏱️ Tests completed in {elapsed:.2f} seconds")
    
    success = results.print_summary()
    
    if success:
        logger.info("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        logger.info(f"\n❌ {results.failed} TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)