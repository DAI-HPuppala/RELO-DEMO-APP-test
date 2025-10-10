"""
Production VLM GPU Loader - GPU Only Version
Optimized GPU-accelerated VLM model loading and inference without fallbacks
"""

import asyncio
import aiohttp
import json
import hashlib
import logging
import time
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from contextlib import asynccontextmanager

from config.gpu_optimizer import GPUOptimizer, GPUConfig

logger = logging.getLogger(__name__)

class VLMGPULoader:
    """Production VLM model loader with GPU optimizations - GPU Required"""

    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = None):
        self.ollama_host = ollama_host
        # Allow env override, fallback to parameter, then default
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
        self.gpu_optimizer = GPUOptimizer()
        self.gpu_config: Optional[GPUConfig] = None
        self.is_loaded = False
        self.is_optimized = False

        # Performance tracking
        self.load_time = 0.0
        self.warmup_time = 0.0
        self.optimization_applied = False

        # Warmup state management
        self.warmup_completed = False
        self.warmup_lock = asyncio.Lock()

        # Performance metrics
        self.inference_count = 0
        self.avg_inference_time = 0.0
        self.inference_times = []

        # Image preprocessing cache
        self._image_cache = {}  # Cache for preprocessed base64 images
        self._cache_max_size = 50  # Limit cache size to prevent memory bloat
        self._cache_hits = 0  # Track cache hits for performance metrics
        self._cache_misses = 0  # Track cache misses for performance metrics

        # Debug settings
        self.debug_save_frames = os.getenv('PREPROCESSING_SAVE_DEBUG_IMAGES', 'true').lower() == 'true'

        # Load preprocessing configuration
        self._load_preprocessing_config()

    def _load_preprocessing_config(self):
        """Load preprocessing configuration from environment variables"""
        # Master toggle
        self.preprocessing_enabled = os.getenv('PREPROCESSING_ENABLED', 'true').lower() == 'true'

        # Per-agent toggles
        self.preprocessing_config = {
            'initial_classifier': {
                'enabled': os.getenv('PREPROCESSING_INITIAL_ENABLED', 'false').lower() == 'true' and self.preprocessing_enabled,
                'type': os.getenv('PREPROCESSING_INITIAL_TYPE', 'none'),
                'level': os.getenv('PREPROCESSING_INITIAL_LEVEL', 'low'),
                # Contrast parameters
                'contrast_alpha': float(os.getenv('PREPROCESSING_INITIAL_CONTRAST_ALPHA', '1.2')),
                'contrast_beta': int(os.getenv('PREPROCESSING_INITIAL_CONTRAST_BETA', '5')),
                # Sharpen parameters
                'sharpen_amount': float(os.getenv('PREPROCESSING_INITIAL_SHARPEN_AMOUNT', '1.2')),
                'sharpen_sigma': float(os.getenv('PREPROCESSING_INITIAL_SHARPEN_SIGMA', '0.8')),
                # Denoise parameters
                'denoise_h': int(os.getenv('PREPROCESSING_INITIAL_DENOISE_H', '10')),
                'denoise_template': int(os.getenv('PREPROCESSING_INITIAL_DENOISE_TEMPLATE', '7')),
                'denoise_search': int(os.getenv('PREPROCESSING_INITIAL_DENOISE_SEARCH', '21'))
            },
            'detail_extractor': {
                'enabled': os.getenv('PREPROCESSING_DETAIL_ENABLED', 'true').lower() == 'true' and self.preprocessing_enabled,
                'type': os.getenv('PREPROCESSING_DETAIL_TYPE', 'clahe'),
                'level': os.getenv('PREPROCESSING_DETAIL_LEVEL', 'medium'),
                # CLAHE parameters
                'clahe_clip': float(os.getenv('PREPROCESSING_DETAIL_CLAHE_CLIP_LIMIT', '3.0')),
                'clahe_grid': int(os.getenv('PREPROCESSING_DETAIL_CLAHE_GRID_SIZE', '8')),
                # Sharpen parameters
                'sharpen_amount': float(os.getenv('PREPROCESSING_DETAIL_SHARPEN_AMOUNT', '1.5')),
                'sharpen_radius': float(os.getenv('PREPROCESSING_DETAIL_SHARPEN_RADIUS', '1.0')),
                'sharpen_threshold': int(os.getenv('PREPROCESSING_DETAIL_SHARPEN_THRESHOLD', '0')),
                'sharpen_contrast_alpha': float(os.getenv('PREPROCESSING_DETAIL_SHARPEN_CONTRAST_ALPHA', '1.05')),
                'sharpen_contrast_beta': int(os.getenv('PREPROCESSING_DETAIL_SHARPEN_CONTRAST_BETA', '3')),
                # Contrast parameters
                'contrast_alpha': float(os.getenv('PREPROCESSING_DETAIL_CONTRAST_ALPHA', '1.5')),
                'contrast_beta': int(os.getenv('PREPROCESSING_DETAIL_CONTRAST_BETA', '0'))
            },
            'damage_detector': {
                'enabled': os.getenv('PREPROCESSING_DAMAGE_ENABLED', 'false').lower() == 'true' and self.preprocessing_enabled,
                'type': os.getenv('PREPROCESSING_DAMAGE_TYPE', 'none'),
                'level': os.getenv('PREPROCESSING_DAMAGE_LEVEL', 'none'),
                # Sharpen parameters - DISABLED to avoid masking real damage
                'sharpen_amount': float(os.getenv('PREPROCESSING_DAMAGE_SHARPEN_AMOUNT', '1.0')),
                'sharpen_sigma': float(os.getenv('PREPROCESSING_DAMAGE_SHARPEN_SIGMA', '1.0')),
                'blur_size': int(os.getenv('PREPROCESSING_DAMAGE_BLUR_SIZE', '5')),
                # Edge parameters
                'edge_kernel': int(os.getenv('PREPROCESSING_DAMAGE_EDGE_KERNEL_SIZE', '3')),
                'edge_strength': float(os.getenv('PREPROCESSING_DAMAGE_EDGE_STRENGTH', '1.0')),
                # Contrast parameters - DISABLED to show raw damage
                'contrast_alpha': float(os.getenv('PREPROCESSING_DAMAGE_CONTRAST_ALPHA', '1.0')),
                'contrast_beta': int(os.getenv('PREPROCESSING_DAMAGE_CONTRAST_BETA', '0'))
            }
        }

    async def initialize_and_optimize(self, target_fps: int = 10, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Initialize and optimize VLM with GPU optimizations"""
        
        start_time = time.time()
        logger.info("🚀 Starting VLM GPU optimization and loading")
        
        try:
            # Step 1: Calculate optimal GPU configuration
            if progress_callback:
                await progress_callback("Calculating optimal GPU configuration...", 10)
            
            self.gpu_config = self.gpu_optimizer.calculate_optimal_config(target_fps)
            logger.info(f"GPU Config: {self.gpu_optimizer.get_optimization_summary(self.gpu_config)}")
            
            # Step 2: Load model with GPU optimization
            if progress_callback:
                await progress_callback("Loading model with GPU optimization...", 30)
            
            await self._load_model_with_optimization()
            
            # Step 3: Apply GPU optimizations
            if progress_callback:
                await progress_callback("Applying GPU optimizations...", 50)
            
            await self._apply_ollama_optimizations()
            
            # Mark as optimized before warmup so warmup can run
            self.is_optimized = True
            
            # Step 4: Run performance warmup
            if progress_callback:
                await progress_callback("Running performance warmup...", 70)
            
            await self._performance_warmup()
            
            # Step 5: Finalize optimization
            if progress_callback:
                await progress_callback("Finalizing GPU optimization...", 90)
            self.load_time = time.time() - start_time
            
            if progress_callback:
                await progress_callback("GPU optimization completed!", 100)
            
            return {
                "status": "success",
                "optimization_level": self.gpu_config.optimization_level.value,
                "load_time": self.load_time,
                "gpu_memory_mb": self.gpu_config.available_memory_mb,
                "inference_streams": self.gpu_config.num_inference_streams
            }
            
        except Exception as e:
            logger.error(f"VLM GPU optimization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "load_time": time.time() - start_time
            }
    
    async def _load_model_with_optimization(self):
        """Load model with GPU optimization"""
        
        async with aiohttp.ClientSession() as session:
            # Check if model is already loaded
            async with session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if self.model_name in models:
                        logger.info("Model already loaded in Ollama")
                        self.is_loaded = True
                        return
            
            # Load model if not present
            logger.info(f"Loading model {self.model_name}...")
            async with session.post(f"{self.ollama_host}/api/pull",
                                   json={"name": self.model_name}) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to load model: {response.status}")
                
                async for line in response.content:
                    if line:
                        try:
                            status = json.loads(line)
                            if status.get("status") == "success":
                                break
                        except json.JSONDecodeError:
                            continue
            
            self.is_loaded = True
    
    async def _apply_ollama_optimizations(self):
        """Apply GPU optimizations to Ollama"""
        
        # Filter CUDA optimizations to only include Ollama-compatible parameters
        ollama_compatible_options = {}
        if self.gpu_config and self.gpu_config.cuda_optimizations:
            # Only include options that Ollama actually supports
            supported_keys = {
                "temperature", "top_p", "top_k", "repeat_penalty", 
                "num_predict", "num_ctx", "num_batch", "num_thread",
                "num_gpu", "main_gpu", "f16_kv", "use_mmap", "use_mlock"
            }
            for key, value in self.gpu_config.cuda_optimizations.items():
                if key in supported_keys:
                    ollama_compatible_options[key] = value
            
            logger.info(f"Filtered Ollama options: {ollama_compatible_options}")
        
        # Generate a test inference with optimal GPU settings
        # Use a very simple prompt that will execute quickly
        optimized_payload = {
            "model": self.model_name,
            "prompt": "Hi",
            "stream": False,
            "keep_alive": -1,  # Keep model in GPU memory permanently
            "options": {
                **ollama_compatible_options,
                "num_predict": 5  # Only generate 5 tokens for warmup
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Increase timeout to 90 seconds for first inference (can take up to 30s)
                async with session.post(f"{self.ollama_host}/api/generate", 
                                       json=optimized_payload,
                                       timeout=aiohttp.ClientTimeout(total=90)) as response:
                    if response.status == 200:
                        # Properly consume the response
                        result = await response.json()
                        logger.info(f"✅ GPU optimizations applied to Ollama. Response: {result.get('response', '')[:100]}...")
                        self.optimization_applied = True
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Failed to apply GPU optimizations: {response.status} - {error_text}")
        except asyncio.TimeoutError:
            logger.error("Timeout while applying GPU optimizations - model may need more time for first inference")
            raise RuntimeError("GPU optimization timeout - try increasing timeout or check Ollama status")
        except Exception as e:
            logger.error(f"Error applying GPU optimizations: {e}")
            raise
    
    async def _performance_warmup(self):
        """Run performance warmup with GPU optimization"""
        
        async with self.warmup_lock:
            if self.warmup_completed:
                return
            
            warmup_start = time.time()
            logger.info("Starting GPU warmup sequence...")
            
            # Create simple test prompts for warmup (short to execute quickly)
            test_prompts = [
                "Hi",
                "Test", 
                "OK"
            ]
            
            warmup_times = []
            
            for i, prompt in enumerate(test_prompts):
                start_time = time.time()
                
                try:
                    # Create simple test image
                    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                    
                    result = await self.infer_optimized([test_image], prompt)
                    
                    inference_time = (time.time() - start_time) * 1000
                    warmup_times.append(inference_time)
                    
                    logger.info(f"Warmup run {i+1}: {inference_time:.0f}ms")
                    
                except Exception as e:
                    logger.warning(f"Warmup run {i+1} failed: {e}")
            
            if warmup_times:
                self.avg_inference_time = sum(warmup_times) / len(warmup_times)
                logger.info(f"GPU warmup completed. Average time: {self.avg_inference_time:.0f}ms")
            
            self.warmup_time = time.time() - warmup_start
            self.warmup_completed = True
    
    async def infer_optimized(self, frames: List[np.ndarray], prompt: str, agent_name: str = None) -> Dict[str, Any]:
        """Run optimized inference with GPU acceleration"""
        
        if not self.is_optimized:
            raise RuntimeError("VLM not optimized - call initialize_and_optimize() first")
        
        start_time = time.time()
        
        try:
            # Use optimized inference context
            async with self.optimized_inference():
                # Convert frames to base64
                import base64
                from PIL import Image
                import io
                
                base64_images = []
                processed_frames = []  # Store processed frames for saving
                for frame_idx, frame in enumerate(frames):
                    # Ensure frame is uint8 before hashing
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # Generate frame hash for caching
                    frame_hash = self._get_image_hash(frame)

                    # Determine target resolution based on agent and frame index
                    if agent_name in ["initial_classifier"]:
                        target_resolution = (448, 448)
                    elif agent_name in ["damage_detector"]:
                        # Higher resolution for damage detection to catch small defects like holes
                        target_resolution = (768, 768)
                    elif agent_name == "detail_extractor":
                        target_resolution = (768, 768)
                    else:
                        target_resolution = (512, 512)

                    # Check cache
                    cache_key = self._get_cache_key(frame_hash, agent_name or "unknown", target_resolution)

                    if cache_key in self._image_cache:
                        # Cache hit - use cached base64 image
                        base64_img = self._image_cache[cache_key]
                        self._cache_hits += 1
                        logger.info(f"🎯 Cache HIT for {agent_name} frame[{frame_idx}] - "
                                  f"Resolution: {target_resolution[0]}x{target_resolution[1]} "
                                  f"(hits: {self._cache_hits}, misses: {self._cache_misses}, "
                                  f"hit rate: {self._cache_hits/(self._cache_hits+self._cache_misses)*100:.1f}%)")
                    else:
                        # Cache miss - process image normally
                        self._cache_misses += 1

                        # Convert BGR to RGB for correct VLM color interpretation
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Apply preprocessing if enabled for this agent
                        if agent_name and self.preprocessing_config.get(agent_name, {}).get('enabled', False):
                            frame_rgb = self._apply_preprocessing(frame_rgb, agent_name)

                        image = Image.fromarray(frame_rgb)

                        # Log original frame resolution
                        original_width, original_height = image.size
                        logger.info(f"📏 Original frame resolution: {original_width}x{original_height} (aspect ratio: {original_width/original_height:.2f}:1)")

                        # Apply target resolution
                        image.thumbnail(target_resolution, Image.Resampling.LANCZOS)

                        # Log final resized resolution
                        final_width, final_height = image.size
                        logger.info(f"🔄 Resized resolution for {agent_name} frame[{frame_idx}]: {final_width}x{final_height} (aspect ratio: {final_width/final_height:.2f}:1)")

                        # Save the final processed frame (exactly what VLM sees)
                        if self.debug_save_frames:
                            processed_frames.append((image, frame_idx, agent_name))

                        # Convert to base64
                        buffer = io.BytesIO()
                        image.save(buffer, format='JPEG', quality=95)
                        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

                        # Store in cache
                        self._image_cache[cache_key] = base64_img
                        self._manage_cache_size()

                        logger.debug(f"📦 Cache MISS for {agent_name} frame[{frame_idx}] - "
                                   f"Added to cache (cache size: {len(self._image_cache)})")

                    base64_images.append(base64_img)

                # Save processed frames that VLM actually sees
                if self.debug_save_frames and processed_frames:
                    await self._save_vlm_frames(processed_frames, agent_name)

                # Build GPU-optimized options with agent-specific parameters
                # Agent-specific temperature and sampling settings
                if agent_name == "detail_extractor":
                    # OCR/Text reading needs deterministic output
                    temperature = 0.2  # Fully deterministic for text reading
                    top_p = 0.1       # Very conservative sampling
                    top_k = 1         # Only best choice for OCR accuracy
                    repeat_penalty = 1.0  # No penalty needed for OCR
                elif agent_name == "damage_detector":
                    # Damage detection - precise but sensitive to catch small defects
                    temperature = 0.3  # Lower temperature for more focused, consistent damage detection
                    top_p = 0.6       # Moderate-high sampling to consider various damage indicators
                    top_k = 20        # More choices to recognize diverse damage types
                    repeat_penalty = 1.0  # No penalty for damage descriptions
                elif agent_name == "initial_classifier":
                    # Classification needs slight variety for attributes
                    temperature = 1  # Slightly more than damage for variety
                    top_p = 0.6        # Moderate probability coverage
                    top_k = 15         # More choices for diverse attributes
                    repeat_penalty = 1.0  # No penalty for JSON outputs
                else:
                    # Default fallback settings
                    temperature = 0.2
                    top_p = 0.7
                    top_k = 20
                    repeat_penalty = 1.0

                options = {
                    "num_predict": 300,
                    "temperature": temperature,
                    "num_ctx": 8192,
                    "num_thread": 6,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty,
                    # CRITICAL GPU settings
                    "num_gpu": 40,  # Force ALL layers to GPU
                    "gpu_layers": 40,  # Ensure all 40 layers use GPU
                    "main_gpu": 0,  # Use GPU 0
                    "low_vram": False,  # Disable low VRAM mode
                    "f16_kv": True,  # Use FP16 for GPU
                    "seed": 42,  # Fixed seed for reproducibility
                }
                
                # Build payload for Ollama API
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": base64_images,
                    "stream": False,
                    "format": "json",
                    "options": options,
                    "keep_alive": -1  # Keep in GPU memory
                }
                
                # Call Ollama API
                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(f"{self.ollama_host}/api/generate", json=payload) as response:
                        if response.status == 200:
                            ollama_result = await response.json()
                            
                            # Parse the response
                            raw_response = ollama_result.get('response', '')
                            
                            # Try to parse JSON response
                            try:
                                if raw_response:
                                    import json as json_module
                                    parsed = json_module.loads(raw_response)
                                    attributes = parsed if isinstance(parsed, dict) else {}
                                else:
                                    attributes = {}
                            except:
                                # If not JSON, return raw response
                                attributes = {"raw_text": raw_response}
                            
                            # Calculate cache statistics
                            total_cache_requests = self._cache_hits + self._cache_misses
                            cache_hit_rate = (self._cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0

                            result = {
                                "attributes": attributes,
                                "confidence": 0.85,
                                "reasoning": "GPU-accelerated inference completed",
                                "raw_response": raw_response,
                                "inference_metadata": {
                                    "optimization_level": self.gpu_config.optimization_level.value if self.gpu_config else "high",
                                    "gpu_memory_used": self.gpu_config.model_memory_requirement_mb if self.gpu_config else 3000,
                                    "model_name": ollama_result.get('model', self.model_name),
                                    "total_duration": ollama_result.get('total_duration', 0) / 1e6,  # Convert to ms
                                    "load_duration": ollama_result.get('load_duration', 0) / 1e6,
                                    "eval_duration": ollama_result.get('eval_duration', 0) / 1e6,
                                    "eval_count": ollama_result.get('eval_count', 0),
                                    "cache_hits": self._cache_hits,
                                    "cache_misses": self._cache_misses,
                                    "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                                    "cache_size": len(self._image_cache)
                                }
                            }
                        else:
                            error_text = await response.text()
                            raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                # Track performance
                inference_time = (time.time() - start_time) * 1000
                self.inference_count += 1
                self.inference_times.append(inference_time)
                
                # Update running average
                if len(self.inference_times) > 10:
                    self.inference_times = self.inference_times[-10:]  # Keep last 10
                self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
                
                # Log performance with cache metrics
                if total_cache_requests > 0:
                    logger.info(f"GPU inference completed in {inference_time:.0f}ms "
                              f"(eval: {result['inference_metadata'].get('eval_duration', 0):.0f}ms) | "
                              f"Cache: {cache_hit_rate:.1f}% hit rate ({self._cache_hits} hits, {self._cache_misses} misses)")
                else:
                    logger.info(f"GPU inference completed in {inference_time:.0f}ms "
                              f"(eval: {result['inference_metadata'].get('eval_duration', 0):.0f}ms)")
                
                return result
                
        except Exception as e:
            logger.error(f"Optimized inference failed: {e}")
            return {
                "error": str(e),
                "inference_metadata": {
                    "optimization_level": self.gpu_config.optimization_level.value if self.gpu_config else "unknown"
                }
            }

    async def _save_vlm_frames(self, processed_frames: List[tuple], agent_name: str) -> None:
        """Save the final processed and resized frames exactly as VLM sees them"""
        try:
            # Create directory for VLM frames
            base_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/captured_frames")
            vlm_dir = base_dir / "vlm_preprocessed" / f"{agent_name}_vlm"
            vlm_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for pil_image, frame_idx, agent in processed_frames:
                # Save the final processed PIL image
                filename = f"{agent_name}_final_frame{frame_idx+1}_{timestamp}.jpg"
                filepath = vlm_dir / filename

                # Save PIL image directly
                await asyncio.to_thread(pil_image.save, str(filepath), 'JPEG', quality=95)

                logger.info(f"💾 VLM-ready frame saved: {filepath}")
                logger.info(f"   Resolution: {pil_image.size[0]}x{pil_image.size[1]} (exactly what VLM sees)")

        except Exception as e:
            logger.error(f"Error saving VLM frames: {e}")

    def _apply_preprocessing(self, frame_rgb: np.ndarray, agent_name: str) -> np.ndarray:
        """Apply preprocessing based on agent configuration"""
        config = self.preprocessing_config.get(agent_name, {})
        if not config.get('enabled', False):
            return frame_rgb

        preprocess_type = config.get('type', 'none')

        # Apply preprocessing based on type
        if agent_name == 'detail_extractor':
            if preprocess_type == 'clahe_sharpen':
                # Combined CLAHE + Sharpening for optimal OCR
                # Step 1: Apply minimal CLAHE
                lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=config.get('clahe_clip', 1.0),  # Minimal CLAHE
                    tileGridSize=(config.get('clahe_grid', 8), config.get('clahe_grid', 8))
                )
                l_channel = clahe.apply(l_channel)
                enhanced_lab = cv2.merge([l_channel, a, b])
                frame_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

                # Step 2: Apply sharpening
                amount = config.get('sharpen_amount', 1.15)
                gaussian = cv2.GaussianBlur(frame_rgb, (5, 5), config.get('sharpen_radius', 1.0))
                frame_rgb = cv2.addWeighted(frame_rgb, amount, gaussian, -(amount - 1), 0)

                logger.info(f"✨ Applied CLAHE + Sharpening for {agent_name} (clahe_clip={config.get('clahe_clip', 1.0)}, sharpen={amount})")
            elif preprocess_type == 'clahe':
                # CLAHE only
                lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=config.get('clahe_clip', 1.0),
                    tileGridSize=(config.get('clahe_grid', 8), config.get('clahe_grid', 8))
                )
                l_channel = clahe.apply(l_channel)
                enhanced_lab = cv2.merge([l_channel, a, b])
                frame_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                logger.info(f"✨ Applied CLAHE enhancement for {agent_name} (clip={config.get('clahe_clip')}, grid={config.get('clahe_grid')})")
            elif preprocess_type == 'sharpen':
                amount = config.get('sharpen_amount', 1.15)
                gaussian = cv2.GaussianBlur(frame_rgb, (5, 5), config.get('sharpen_radius', 1.0))
                frame_rgb = cv2.addWeighted(frame_rgb, amount, gaussian, -(amount - 1), 0)
                logger.info(f"✨ Applied sharpening for {agent_name} (sharpen={amount})")
            elif preprocess_type == 'contrast':
                frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=config.get('contrast_alpha', 1.5), beta=config.get('contrast_beta', 0))
                logger.info(f"✨ Applied contrast enhancement for {agent_name}")

        elif agent_name == 'damage_detector':
            if preprocess_type == 'sharpen':
                blur_size = config.get('blur_size', 5)
                if blur_size % 2 == 0:  # Ensure odd number
                    blur_size += 1
                amount = config.get('sharpen_amount', 1.5)
                gaussian = cv2.GaussianBlur(frame_rgb, (blur_size, blur_size), config.get('sharpen_sigma', 1.0))
                frame_rgb = cv2.addWeighted(frame_rgb, amount, gaussian, -(amount - 1), 0)
                logger.info(f"🔍 Applied sharpening for {agent_name} (amount={amount})")
            elif preprocess_type == 'edge_enhance':
                kernel_size = config.get('edge_kernel', 3)
                kernel = np.array([[-1]*kernel_size, [-1, kernel_size**2, -1], [-1]*kernel_size])
                frame_rgb = cv2.filter2D(frame_rgb, -1, kernel * config.get('edge_strength', 1.5))
                logger.info(f"🔍 Applied edge enhancement for {agent_name}")
            elif preprocess_type == 'contrast':
                frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=config.get('contrast_alpha', 1.3), beta=config.get('contrast_beta', 10))
                logger.info(f"🔍 Applied contrast for {agent_name}")

        elif agent_name == 'initial_classifier':
            if preprocess_type == 'contrast':
                frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=config.get('contrast_alpha', 1.2), beta=config.get('contrast_beta', 5))
                logger.info(f"🎨 Applied contrast for {agent_name}")
            elif preprocess_type == 'sharpen':
                amount = config.get('sharpen_amount', 1.2)
                gaussian = cv2.GaussianBlur(frame_rgb, (5, 5), config.get('sharpen_sigma', 0.8))
                frame_rgb = cv2.addWeighted(frame_rgb, amount, gaussian, -(amount - 1), 0)
                logger.info(f"🎨 Applied sharpening for {agent_name} (amount={amount})")
            elif preprocess_type == 'denoise':
                frame_rgb = cv2.fastNlMeansDenoisingColored(
                    frame_rgb,
                    None,
                    config.get('denoise_h', 10),
                    config.get('denoise_h', 10),
                    config.get('denoise_template', 7),
                    config.get('denoise_search', 21)
                )
                logger.info(f"🎨 Applied denoising for {agent_name}")

        return frame_rgb

    def _get_image_hash(self, frame: np.ndarray) -> str:
        """Generate unique hash for frame content"""
        return hashlib.md5(frame.tobytes()).hexdigest()

    def _get_cache_key(self, frame_hash: str, agent_name: str, resolution: tuple) -> str:
        """Generate cache key from frame hash, agent, and resolution"""
        return f"{frame_hash}_{agent_name}_{resolution[0]}x{resolution[1]}"

    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds max size (LRU)"""
        if len(self._image_cache) > self._cache_max_size:
            # Remove oldest 10 entries to make room
            for key in list(self._image_cache.keys())[:10]:
                del self._image_cache[key]
            logger.debug(f"Cache pruned: removed 10 oldest entries, current size: {len(self._image_cache)}")

    @asynccontextmanager
    async def optimized_inference(self):
        """Context manager for optimized inference sessions"""
        # In a real implementation, this would set up GPU inference context
        logger.debug("Starting optimized inference session")
        try:
            yield
        finally:
            logger.debug("Ending optimized inference session")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        base_stats = {
            "is_loaded": self.is_loaded,
            "is_optimized": self.is_optimized,
            "optimization_applied": self.optimization_applied,
            "warmup_completed": self.warmup_completed,
            "load_time": self.load_time,
            "warmup_time": self.warmup_time,
            "inference_count": self.inference_count,
            "avg_inference_time": self.avg_inference_time,
        }
        
        if self.gpu_config:
            base_stats.update({
                "optimization_level": self.gpu_config.optimization_level.value,
                "available_memory_mb": self.gpu_config.available_memory_mb,
                "model_memory_requirement_mb": self.gpu_config.model_memory_requirement_mb,
                "inference_streams": self.gpu_config.num_inference_streams,
                "mixed_precision": False,  # Not using PyTorch mixed precision
            })
        
        return base_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of GPU-optimized VLM"""
        
        try:
            # Check GPU availability
            total, used, free = self.gpu_optimizer.get_gpu_memory_info()
            gpu_healthy = free > 1000  # At least 1GB free
            
            # Check Ollama connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags", 
                                     timeout=aiohttp.ClientTimeout(total=5)) as response:
                    ollama_healthy = response.status == 200
            
            # Check model status
            model_healthy = self.is_loaded and self.is_optimized
            
            overall_status = "healthy" if (gpu_healthy and ollama_healthy and model_healthy) else "degraded"
            
            return {
                "status": overall_status,
                "gpu_healthy": gpu_healthy,
                "ollama_healthy": ollama_healthy, 
                "model_healthy": model_healthy,
                "gpu_free_memory_mb": free,
                "optimization_level": self.gpu_config.optimization_level.value if self.gpu_config else "not_set"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup GPU resources"""
        logger.info("Cleaning up VLM GPU resources...")
        
        try:
            # Unload model from GPU memory
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ollama_host}/api/generate",
                                       json={
                                           "model": self.model_name,
                                           "prompt": "cleanup",
                                           "stream": False,
                                           "keep_alive": 0  # Unload immediately
                                       },
                                       timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        logger.info("✅ VLM model unloaded from GPU")
            
            self.is_optimized = False
            self.is_loaded = False
            self.optimization_applied = False
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")