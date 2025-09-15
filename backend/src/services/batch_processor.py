"""
Optimized Batch Processing for VLM Inference
Implements SOTA batch processing with dynamic batching and preprocessing
"""

import asyncio
import numpy as np
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor

from config.gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Optimized batch processor for multi-image VLM inference
    Implements dynamic batching, preprocessing pipeline, and memory management
    """
    
    def __init__(self, target_size: int = 448):
        self.target_size = target_size  # Optimal size for Qwen2.5-VL
        self.executor = ThreadPoolExecutor(max_workers=4)  # For parallel preprocessing
        self.preprocessing_cache = {}  # Cache preprocessed images
        self.batch_queue = asyncio.Queue()
        self.processing = False
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Preprocess single image for VLM inference
        Returns preprocessed array and cache key
        """
        # Generate cache key
        cache_key = f"{image.shape}_{hash(image.tobytes())}"
        
        # Check cache
        if cache_key in self.preprocessing_cache:
            return self.preprocessing_cache[cache_key], cache_key
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Resize to target size (448x448 optimal for Qwen2.5-VL)
        if pil_image.size != (self.target_size, self.target_size):
            # Use high-quality resampling
            pil_image = pil_image.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS
            )
        
        # No RGB conversion - keep original format (BGR)
        # VLM will process same format as frontend sees
        
        # Convert back to numpy
        processed = np.array(pil_image, dtype=np.uint8)
        
        # Normalize to [0, 1] range for model input
        processed = processed.astype(np.float32) / 255.0
        
        # Cache the result (limit cache size)
        if len(self.preprocessing_cache) < 100:
            self.preprocessing_cache[cache_key] = processed
        
        return processed, cache_key
    
    async def preprocess_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess batch of images in parallel
        """
        logger.info(f"Preprocessing batch of {len(images)} images...")
        start_time = time.time()
        
        # Use thread pool for parallel preprocessing
        loop = asyncio.get_event_loop()
        tasks = []
        
        for image in images:
            task = loop.run_in_executor(
                self.executor,
                self.preprocess_image,
                image
            )
            tasks.append(task)
        
        # Wait for all preprocessing to complete
        results = await asyncio.gather(*tasks)
        processed_images = [r[0] for r in results]
        
        preprocessing_time = time.time() - start_time
        logger.info(f"✓ Preprocessed {len(images)} images in {preprocessing_time:.2f}s")
        
        return processed_images
    
    def create_optimal_batches(self, images: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Create optimal batches based on GPU memory
        """
        # Get optimal batch size from GPU optimizer
        optimal_batch_size = gpu_optimizer.get_optimal_batch_size()
        
        logger.info(f"Creating batches with optimal size: {optimal_batch_size}")
        
        batches = []
        for i in range(0, len(images), optimal_batch_size):
            batch = images[i:i + optimal_batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches from {len(images)} images")
        return batches
    
    async def process_batch_with_inference(self, 
                                          batch: List[np.ndarray],
                                          prompt: str,
                                          inference_fn) -> Dict[str, Any]:
        """
        Process a batch through inference
        
        Args:
            batch: List of preprocessed images
            prompt: Text prompt for VLM
            inference_fn: Async function to run inference
        """
        start_time = time.time()
        
        # Check GPU memory before processing
        memory_stats = gpu_optimizer.get_memory_stats()
        logger.info(f"GPU memory before batch: {memory_stats['free_mb']}MB free")
        
        # Run inference
        try:
            result = await inference_fn(batch, prompt)
            
            inference_time = time.time() - start_time
            
            return {
                "result": result,
                "batch_size": len(batch),
                "inference_time": inference_time,
                "throughput": len(batch) / inference_time,  # Images per second
                "memory_used": memory_stats['used_mb']
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            
            # Try with smaller batch on OOM
            if "out of memory" in str(e).lower() and len(batch) > 1:
                logger.info("Retrying with smaller batch...")
                half_size = len(batch) // 2
                batch1 = batch[:half_size]
                batch2 = batch[half_size:]
                
                result1 = await self.process_batch_with_inference(batch1, prompt, inference_fn)
                result2 = await self.process_batch_with_inference(batch2, prompt, inference_fn)
                
                # Combine results
                return {
                    "result": {
                        "batch1": result1.get("result"),
                        "batch2": result2.get("result")
                    },
                    "batch_size": len(batch),
                    "inference_time": result1["inference_time"] + result2["inference_time"],
                    "split_batch": True
                }
            
            raise
    
    async def process_images_optimally(self,
                                      images: List[np.ndarray],
                                      prompt: str,
                                      inference_fn) -> Dict[str, Any]:
        """
        Process images with optimal batching and preprocessing
        
        This is the main entry point for batch processing
        """
        total_start = time.time()
        
        logger.info(f"📦 Processing {len(images)} images optimally...")
        
        # Step 1: Preprocess all images in parallel
        preprocessed = await self.preprocess_batch(images)
        
        # Step 2: Create optimal batches
        batches = self.create_optimal_batches(preprocessed)
        
        # Step 3: Process each batch
        results = []
        total_inference_time = 0
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} images)...")
            
            batch_result = await self.process_batch_with_inference(
                batch, prompt, inference_fn
            )
            
            results.append(batch_result)
            total_inference_time += batch_result["inference_time"]
        
        # Calculate overall statistics
        total_time = time.time() - total_start
        total_images = len(images)
        
        return {
            "total_images": total_images,
            "num_batches": len(batches),
            "batch_results": results,
            "total_time": total_time,
            "preprocessing_time": total_time - total_inference_time,
            "inference_time": total_inference_time,
            "overall_throughput": total_images / total_time,
            "inference_throughput": total_images / total_inference_time if total_inference_time > 0 else 0
        }
    
    def encode_images_for_api(self, images: List[np.ndarray]) -> List[str]:
        """
        Encode images as base64 for API transmission
        """
        encoded = []
        
        for image in images:
            # Convert to PIL
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Encode to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            encoded.append(img_base64)
        
        return encoded
    
    async def dynamic_batching_loop(self, inference_fn, max_wait_time: float = 0.1):
        """
        Dynamic batching loop that accumulates requests
        """
        self.processing = True
        accumulated = []
        last_process_time = time.time()
        
        while self.processing:
            try:
                # Try to get item with timeout
                try:
                    item = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=max_wait_time
                    )
                    accumulated.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Process if we have items and either:
                # 1. Reached optimal batch size
                # 2. Waited long enough
                current_time = time.time()
                optimal_size = gpu_optimizer.get_optimal_batch_size()
                
                should_process = (
                    len(accumulated) >= optimal_size or
                    (len(accumulated) > 0 and current_time - last_process_time > max_wait_time)
                )
                
                if should_process and accumulated:
                    # Process accumulated batch
                    batch_images = [item["image"] for item in accumulated]
                    batch_prompts = [item["prompt"] for item in accumulated]
                    
                    # Use first prompt for batch (assuming same task)
                    result = await self.process_batch_with_inference(
                        batch_images,
                        batch_prompts[0],
                        inference_fn
                    )
                    
                    # Return results to waiting tasks
                    for item in accumulated:
                        item["future"].set_result(result)
                    
                    accumulated = []
                    last_process_time = current_time
                    
            except Exception as e:
                logger.error(f"Dynamic batching error: {e}")
                
                # Return error to waiting tasks
                for item in accumulated:
                    item["future"].set_exception(e)
                
                accumulated = []
    
    def stop_dynamic_batching(self):
        """Stop the dynamic batching loop"""
        self.processing = False
    
    def clear_cache(self):
        """Clear preprocessing cache"""
        self.preprocessing_cache.clear()
        logger.info("Preprocessing cache cleared")

# Global batch processor instance
batch_processor = BatchProcessor()

__all__ = ['batch_processor', 'BatchProcessor']