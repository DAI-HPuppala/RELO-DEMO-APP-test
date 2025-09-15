#!/usr/bin/env python3
"""
GPU Optimization Warmup Script - GPU Only Version
Streamlined GPU-accelerated VLM initialization without fallbacks
"""

import asyncio
import logging
import sys
import time
import numpy as np
from pathlib import Path

# Add src directory to Python path (same as run_backend.py)
backend_dir = Path(__file__).parent
src_dir = backend_dir / "src"
sys.path.insert(0, str(src_dir))

# Now import our modules
from config.gpu_optimizer import GPUOptimizer
from services.vlm_gpu_loader import VLMGPULoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUWarmupService:
    """GPU-accelerated warmup service - GPU Required"""
    
    def __init__(self):
        self.gpu_optimizer = GPUOptimizer()
        self.vlm_loader = None
        
    async def run_warmup(self) -> bool:
        """Run complete GPU warmup process"""
        
        logger.info("=" * 60)
        logger.info("🚀 RELO-CLASSIFIER GPU OPTIMIZATION WARMUP")
        logger.info("=" * 60)
        
        try:
            # Step 1: GPU Detection and Configuration
            await self._detect_and_configure_gpu()
            
            # Step 2: Initialize VLM with GPU optimization
            await self._initialize_vlm_optimization()
            
            # Step 3: Run performance benchmark
            await self._run_performance_benchmark()
            
            # Step 4: Keep model loaded for production
            await self._finalize_optimization()
            
            logger.info("=" * 60)
            logger.info("✅ GPU OPTIMIZATION WARMUP COMPLETED")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU warmup failed: {e}")
            return False
    
    async def _detect_and_configure_gpu(self):
        """Detect GPU and configure optimization"""
        logger.info("🔍 Detecting GPU capabilities...")
        
        try:
            # Get GPU memory info (will raise if no GPU)
            total, used, free = self.gpu_optimizer.get_gpu_memory_info()
            
            logger.info(f"GPU Detected:")
            logger.info(f"  - Total Memory: {total}MB")
            logger.info(f"  - Used Memory: {used}MB")
            logger.info(f"  - Free Memory: {free}MB")
            
            # Calculate optimal configuration
            config = self.gpu_optimizer.calculate_optimal_config(target_fps=10)
            summary = self.gpu_optimizer.get_optimization_summary(config)
            
            logger.info(f"Optimal Configuration:")
            logger.info(f"  - Optimization Level: {summary['optimization_level']}")
            logger.info(f"  - Performance Mode: {summary['performance_mode']}")
            logger.info(f"  - CUDA Optimizations: {summary['cuda_optimizations_count']} parameters")
            
        except RuntimeError as e:
            raise RuntimeError(f"GPU required but not available: {e}")
    
    async def _initialize_vlm_optimization(self):
        """Initialize VLM with GPU optimization"""
        logger.info("🧠 Initializing VLM with GPU optimization...")
        
        self.vlm_loader = VLMGPULoader()
        
        # Progress callback for detailed logging
        async def progress_callback(message: str, percentage: int):
            logger.info(f"  [{percentage:3d}%] {message}")
        
        # Initialize with GPU optimization
        result = await self.vlm_loader.initialize_and_optimize(
            target_fps=10,
            progress_callback=progress_callback
        )
        
        if result["status"] == "success":
            logger.info("✅ VLM GPU optimization initialized successfully")
            
            # Log performance details
            stats = self.vlm_loader.get_performance_stats()
            logger.info(f"Performance Stats:")
            logger.info(f"  - Load Time: {stats.get('load_time', 0):.2f}s")
            logger.info(f"  - Warmup Time: {stats.get('warmup_time', 0):.2f}s")
            logger.info(f"  - Optimization Level: {stats.get('optimization_level', 'unknown')}")
            logger.info(f"  - Average Inference Time: {stats.get('avg_inference_time', 0):.0f}ms")
            
        else:
            raise RuntimeError(f"VLM initialization failed: {result}")
    
    async def _run_performance_benchmark(self):
        """Run performance benchmark"""
        logger.info("📊 Running performance benchmark...")
        
        if not self.vlm_loader or not self.vlm_loader.is_optimized:
            raise RuntimeError("VLM not optimized - cannot run benchmark")
        
        # Create diverse test patterns
        test_patterns = [
            self._create_test_pattern("solid", (224, 224, 3)),
            self._create_test_pattern("gradient", (448, 448, 3)),
            self._create_test_pattern("noise", (320, 240, 3)),
        ]
        
        total_time = 0
        successful_runs = 0
        
        for i, pattern in enumerate(test_patterns):
            try:
                start_time = time.time()
                
                result = await self.vlm_loader.infer_optimized(
                    [pattern], 
                    "Analyze this test pattern briefly"
                )
                
                inference_time = (time.time() - start_time) * 1000
                total_time += inference_time
                successful_runs += 1
                
                logger.info(f"  Test {i+1}: {inference_time:.0f}ms - {'Success' if not result.get('error') else 'Failed'}")
                
            except Exception as e:
                logger.error(f"  Test {i+1}: Failed - {e}")
        
        if successful_runs > 0:
            avg_time = total_time / successful_runs
            logger.info(f"Benchmark Results:")
            logger.info(f"  - Average Inference Time: {avg_time:.0f}ms")
            logger.info(f"  - Successful Runs: {successful_runs}/{len(test_patterns)}")
            
            # Performance assessment
            if avg_time < 1000:
                logger.info("🚀 Excellent performance! (<1s per inference)")
            elif avg_time < 2000:
                logger.info("✅ Good performance! (1-2s per inference)")
            else:
                logger.warning("⚠️ Performance may need optimization (>2s per inference)")
        else:
            raise RuntimeError("All benchmark tests failed")
    
    def _create_test_pattern(self, pattern_type: str, shape: tuple) -> np.ndarray:
        """Create test patterns for benchmarking"""
        
        if pattern_type == "solid":
            # Solid blue pattern
            pattern = np.full(shape, [100, 150, 200], dtype=np.uint8)
        elif pattern_type == "gradient":
            # Gradient pattern
            pattern = np.zeros(shape, dtype=np.uint8)
            for i in range(shape[0]):
                pattern[i, :, 0] = int((i / shape[0]) * 255)  # Red gradient
                pattern[i, :, 1] = 128  # Constant green
                pattern[i, :, 2] = int(((shape[0] - i) / shape[0]) * 255)  # Blue gradient
        elif pattern_type == "noise":
            # Random noise pattern
            pattern = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            # Default: solid white
            pattern = np.full(shape, 255, dtype=np.uint8)
        
        return pattern
    
    async def _finalize_optimization(self):
        """Finalize optimization and save profile"""
        logger.info("🔧 Finalizing GPU optimization for production use")
        
        if self.vlm_loader:
            # Get final stats
            stats = self.vlm_loader.get_performance_stats()
            health = await self.vlm_loader.health_check()
            
            logger.info("Final Status:")
            logger.info(f"  - Model Loaded: {stats.get('is_loaded', False)}")
            logger.info(f"  - GPU Optimized: {stats.get('is_optimized', False)}")
            logger.info(f"  - Health Status: {health.get('status', 'unknown')}")
            logger.info(f"  - Total Inferences: {stats.get('inference_count', 0)}")
            
            # Save optimization profile
            await self._save_optimization_profile(stats)
    
    async def _save_optimization_profile(self, stats: dict):
        """Save optimization profile for future use"""
        import json
        from datetime import datetime
        
        profile = {
            "timestamp": datetime.now().isoformat(),
            "gpu_optimization": {
                "optimization_level": stats.get("optimization_level", "unknown"),
                "available_memory_mb": stats.get("available_memory_mb", 0),
                "avg_inference_time": stats.get("avg_inference_time", 0)
            },
            "performance_metrics": {
                "load_time": stats.get("load_time", 0),
                "warmup_time": stats.get("warmup_time", 0),
                "inference_count": stats.get("inference_count", 0)
            }
        }
        
        profile_path = backend_dir / "gpu_optimization_profile.json"
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            logger.info(f"📝 Optimization profile saved to: {profile_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save optimization profile: {e}")

async def main():
    """Main warmup function"""
    warmup_service = GPUWarmupService()
    
    try:
        success = await warmup_service.run_warmup()
        
        if success:
            logger.info("🎉 GPU warmup completed successfully!")
            logger.info("💡 RELO-CLASSIFIER is now optimized for production use")
            return 0
        else:
            logger.error("💥 GPU warmup failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 GPU warmup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 GPU warmup error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)