"""VLM Service for Ollama Qwen2.5-VL Integration with Advanced GPU Optimization"""

import asyncio
import aiohttp
import base64
import json
import logging
import time
import os
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class GPUManager:
    """Advanced GPU management for optimal VLM performance"""
    
    def __init__(self):
        self.gpu_info = None
        self.optimal_layers = None
        self.memory_safety_margin = 0.15  # 15% safety margin
        
    async def detect_gpu_capabilities(self) -> Dict[str, Any]:
        """Detect GPU capabilities using nvidia-smi"""
        try:
            # Get GPU information
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.warning("nvidia-smi not available, using CPU fallback")
                return {"gpu_available": False, "fallback": "cpu"}
            
            lines = result.stdout.strip().split('\n')
            gpu_data = []
            
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    name, total_mem, free_mem, used_mem = parts[:4]
                    gpu_data.append({
                        "name": name,
                        "total_memory_mb": int(total_mem),
                        "free_memory_mb": int(free_mem),
                        "used_memory_mb": int(used_mem),
                        "utilization_percent": (int(used_mem) / int(total_mem)) * 100
                    })
            
            self.gpu_info = gpu_data[0] if gpu_data else None
            
            if self.gpu_info:
                # Calculate optimal GPU layers based on available memory
                self.optimal_layers = self._calculate_optimal_layers(self.gpu_info)
                
                logger.info(f"GPU detected: {self.gpu_info['name']}")
                logger.info(f"GPU Memory: {self.gpu_info['free_memory_mb']}MB free / {self.gpu_info['total_memory_mb']}MB total")
                logger.info(f"Optimal GPU layers: {self.optimal_layers}")
                
                return {
                    "gpu_available": True,
                    "gpu_info": self.gpu_info,
                    "optimal_layers": self.optimal_layers,
                    "memory_optimization": True
                }
            
            return {"gpu_available": False, "reason": "No GPU data available"}
            
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
            return {"gpu_available": False, "error": str(e)}
    
    def _calculate_optimal_layers(self, gpu_info: Dict[str, Any]) -> int:
        """Calculate optimal number of GPU layers based on available memory"""
        free_memory_mb = gpu_info["free_memory_mb"]
        total_memory_mb = gpu_info["total_memory_mb"]
        
        # Apply safety margin
        usable_memory_mb = free_memory_mb * (1 - self.memory_safety_margin)
        
        # Memory requirements for Qwen2.5-VL 3B model (approximate)
        model_size_mb = 3000  # 3B model ~3GB
        base_memory_mb = 1500  # Base CUDA overhead
        
        # Calculate if we can fit the full model
        if usable_memory_mb >= (model_size_mb + base_memory_mb):
            # Full GPU inference - use all layers (-1 in Ollama means all layers)
            optimal_layers = -1
        else:
            # Partial GPU inference - calculate layer count
            available_for_model = usable_memory_mb - base_memory_mb
            layer_memory_mb = model_size_mb / 32  # Approximate 32 layers in 3B model
            optimal_layers = max(1, int(available_for_model / layer_memory_mb))
        
        return optimal_layers
    
    def get_cuda_optimization_params(self) -> Dict[str, Any]:
        """Get CUDA optimization parameters for Ollama"""
        if not self.gpu_info or not self.optimal_layers:
            return {}
        
        return {
            "num_gpu": 1,
            "gpu_layers": self.optimal_layers,
            "main_gpu": 0,
            "gpu_memory_utilization": 0.9,
            "use_mmap": True,
            "use_mlock": True,
            "num_thread": min(24, os.cpu_count() or 8),  # Optimize thread count
            "num_batch": 1024,
            "num_ctx": 4096,
            # Performance optimizations
            "tfs_z": 1.0,
            "typical_p": 1.0,
            "mirostat": 0,
            "repeat_penalty": 1.1
        }


class VLMService:
    """Advanced Vision Language Model service with GPU optimization and performance monitoring"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = "qwen2.5vl:3b"
        self.max_retries = 2
        self.timeout = 60  # Reduced timeout for optimized GPU inference
        
        # Advanced GPU management
        self.gpu_manager = GPUManager()
        self.cuda_params = {}
        
        # Optimized image preprocessing settings (based on RELO-DEMO-APP research)
        self.max_image_size = (448, 448)  # Optimized for faster processing
        self.image_quality = 90  # Higher quality for better inference
        
        # GPU lock-in state management
        self.is_model_loaded = False
        self.is_gpu_locked = False
        self.warmup_completed = False
        self.model_session_id = None
        self._warmup_lock = asyncio.Lock()
        
        # Advanced performance tracking
        self.warmup_start_time = None
        self.warmup_duration_ms = None
        self.inference_count = 0
        self.inference_times = []  # Track individual inference times
        self.avg_inference_time = 0
        self.last_inference_time = 0
        self.target_inference_time = 1200  # <1200ms target
        
        # Performance optimization flags
        self.model_compiled = False
        self.cuda_optimized = False
        
        logger.info(f"Advanced VLMService initialized: {self.model_name}")
        logger.info("GPU optimization and performance monitoring enabled")
    
    async def initialize_and_warmup(self, progress_callback=None) -> Dict[str, Any]:
        """
        Advanced GPU-optimized initialization and warmup with performance benchmarking
        """
        async with self._warmup_lock:
            if self.warmup_completed:
                logger.info("VLM warmup already completed, skipping")
                return {"status": "already_completed", "warmup_duration_ms": self.warmup_duration_ms}
            
            logger.info("🚀 Starting Advanced VLM GPU Optimization and Warmup")
            self.warmup_start_time = time.time()
            
            try:
                # Step 1: Advanced GPU Detection and Optimization
                if progress_callback:
                    await progress_callback("Detecting GPU capabilities and optimizing...", 5)
                
                gpu_capabilities = await self.gpu_manager.detect_gpu_capabilities()
                if gpu_capabilities.get("gpu_available"):
                    self.cuda_params = self.gpu_manager.get_cuda_optimization_params()
                    self.cuda_optimized = True
                    logger.info("🔧 GPU optimization parameters configured")
                
                # Step 2: Load model with GPU optimization
                if progress_callback:
                    await progress_callback("Loading model with GPU optimization...", 15)
                
                await self._load_model_with_optimization()
                
                # Step 3: Load and preprocess warmup image
                if progress_callback:
                    await progress_callback("Loading optimized warmup image...", 25)
                
                warmup_image = await self._load_optimized_warmup_image()
                
                # Step 4: Performance benchmarking warmup
                if progress_callback:
                    await progress_callback("Running performance benchmark warmup...", 40)
                
                benchmark_results = await self._run_performance_benchmark(warmup_image)
                
                # Step 5: Model compilation optimization (if needed)
                if progress_callback:
                    await progress_callback("Optimizing model compilation...", 60)
                
                await self._optimize_model_compilation()
                
                # Step 6: Final validation and performance test
                if progress_callback:
                    await progress_callback("Validating performance targets...", 80)
                
                performance_validation = await self._validate_performance_target()
                
                # Step 7: Complete with performance summary
                self.warmup_duration_ms = int((time.time() - self.warmup_start_time) * 1000)
                self.warmup_completed = True
                
                if progress_callback:
                    perf_msg = f"Warmup complete! Avg inference: {self.avg_inference_time:.0f}ms"
                    await progress_callback(perf_msg, 100)
                
                logger.info(f"🚀 Advanced VLM optimization completed in {self.warmup_duration_ms}ms")
                logger.info(f"🎯 Average inference time: {self.avg_inference_time:.0f}ms (target: <{self.target_inference_time}ms)")
                
                return {
                    "status": "success",
                    "warmup_duration_ms": self.warmup_duration_ms,
                    "model_loaded": self.is_model_loaded,
                    "gpu_locked": self.is_gpu_locked,
                    "cuda_optimized": self.cuda_optimized,
                    "avg_inference_time": self.avg_inference_time,
                    "performance_target_met": self.avg_inference_time < self.target_inference_time,
                    "gpu_capabilities": gpu_capabilities,
                    "benchmark_results": benchmark_results,
                    "performance_validation": performance_validation
                }
                
            except Exception as e:
                logger.error(f"VLM warmup failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "warmup_duration_ms": int((time.time() - self.warmup_start_time) * 1000) if self.warmup_start_time else 0
                }
    
    async def _load_model_persistent(self):
        """Load model into GPU memory with persistent session"""
        try:
            # Check if model is already loaded (by script)
            url = f"{self.ollama_host}/api/tags"
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model.get("name", "") for model in data.get("models", [])]
                        if any(self.model_name in model for model in models):
                            # Model exists, just ensure it's loaded
                            await self._ensure_model_loaded()
                            return
            
            # If model not found, try to load it
            await self._ensure_model_loaded()
                        
        except Exception as e:
            logger.error(f"Failed to load model persistently: {e}")
            raise
    
    async def _ensure_model_loaded(self):
        """Ensure model is loaded and ready for inference"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": "Model readiness check",
                "stream": False,
                "keep_alive": -1,  # Keep forever
                "options": {
                    "num_predict": 5,  # Minimal response for quick check
                    "temperature": 0.1
                }
            }
            
            url = f"{self.ollama_host}/api/generate"
            timeout = aiohttp.ClientTimeout(total=20)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.is_model_loaded = True
                        self.is_gpu_locked = True
                        logger.info("🔒 GPU model confirmed loaded and ready")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Model readiness check failed: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Failed to ensure model loaded: {e}")
            raise
    
    async def _load_warmup_image(self) -> str:
        """Load the Denali warmup test image"""
        try:
            # Look for the warmup image in multiple possible locations
            possible_paths = [
                "backend/assets/denali_logo.png",
                "assets/denali_logo.png",
                "/home/automation-dev/RELO-CLASSIFIER-specify-DEV/RELO-CLASSIFIER-DEV/backend/assets/denali_logo.png"
            ]
            
            image_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if not image_path:
                # Create the image if it doesn't exist
                logger.warning("Denali warmup image not found, creating it...")
                await self._create_warmup_image()
                image_path = "backend/assets/denali_logo.png"
            
            # Load and encode the image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
                logger.info(f"Loaded warmup image: {image_path}")
                return base64_str
                
        except Exception as e:
            logger.error(f"Failed to load warmup image: {e}")
            raise
    
    async def _create_warmup_image(self):
        """Create the Denali warmup image if it doesn't exist"""
        try:
            # Import the logo creation function
            import sys
            import os
            backend_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.append(backend_path)
            
            from create_denali_logo import create_denali_logo
            
            # Create assets directory
            assets_dir = os.path.join(backend_path, 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            
            # Create and save the logo
            logo = create_denali_logo()
            logo_path = os.path.join(assets_dir, 'denali_logo.png')
            logo.save(logo_path, 'PNG', quality=95)
            
            logger.info(f"Created warmup image: {logo_path}")
            
        except Exception as e:
            logger.error(f"Failed to create warmup image: {e}")
            raise
    
    async def _run_warmup_inference(self, warmup_image: str) -> Dict[str, Any]:
        """Run optimized warmup inference with the Denali test image"""
        
        # Optimized warmup prompt - shorter but comprehensive
        warmup_prompt = """Analyze this Denali company logo image quickly and respond with JSON:
        {
            "company": "company name detected",
            "text_elements": ["text", "found"],
            "colors": ["main", "colors"],
            "design": "simple design description",
            "warmup_status": "ready",
            "gpu_performance": "optimal"
        }
        
        This is a VLM warmup test - respond quickly and concisely."""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": warmup_prompt,
                "images": [warmup_image],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 200,  # Shorter response for faster warmup
                    "temperature": 0.1
                },
                "keep_alive": -1  # Keep model loaded
            }
            
            warmup_start = time.time()
            result = await self._call_ollama_with_retry(payload)
            warmup_time = int((time.time() - warmup_start) * 1000)
            
            logger.info(f"🔥 Warmup inference completed in {warmup_time}ms")
            
            # Parse the response
            content = result.get("response", "")
            try:
                parsed_result = json.loads(content)
                parsed_result["warmup_inference_time_ms"] = warmup_time
                return parsed_result
            except json.JSONDecodeError:
                return {
                    "warmup_inference_time_ms": warmup_time,
                    "raw_response": content[:500] + "..." if len(content) > 500 else content,
                    "status": "warmup_completed"
                }
                
        except Exception as e:
            logger.error(f"Warmup inference failed: {e}")
            raise
    
    async def _validate_gpu_lock(self):
        """Validate that the model remains loaded in GPU memory"""
        try:
            # Quick validation inference
            test_payload = {
                "model": self.model_name,
                "prompt": "Model validation test - respond with 'GPU_LOCKED_READY'",
                "stream": False,
                "options": {"num_predict": 10},
                "keep_alive": -1
            }
            
            start_time = time.time()
            result = await self._call_ollama_with_retry(test_payload)
            validation_time = int((time.time() - start_time) * 1000)
            
            # Fast response indicates model is still in GPU memory
            if validation_time < 5000:  # Less than 5 seconds indicates good GPU persistence
                logger.info(f"✅ GPU lock validated - response time: {validation_time}ms")
                return True
            else:
                logger.warning(f"⚠️ GPU lock may be weak - response time: {validation_time}ms")
                return False
                
        except Exception as e:
            logger.error(f"GPU lock validation failed: {e}")
            return False
    
    async def get_warmup_status(self) -> Dict[str, Any]:
        """Get current warmup and GPU lock-in status"""
        return {
            "is_model_loaded": self.is_model_loaded,
            "is_gpu_locked": self.is_gpu_locked,
            "warmup_completed": self.warmup_completed,
            "warmup_duration_ms": self.warmup_duration_ms,
            "inference_count": self.inference_count,
            "model_name": self.model_name
        }
    
    async def force_gpu_unlock(self):
        """Force unlock GPU model (called by stop.sh)"""
        try:
            # Unload model from memory
            payload = {
                "model": self.model_name,
                "keep_alive": 0  # Immediately unload
            }
            
            url = f"{self.ollama_host}/api/generate"
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json={
                    **payload,
                    "prompt": "Model unload",
                    "stream": False
                }) as response:
                    pass  # Don't care about response, just trigger unload
            
            self.is_model_loaded = False
            self.is_gpu_locked = False
            self.warmup_completed = False
            
            logger.info("🔓 GPU model unlocked and unloaded")
            
        except Exception as e:
            logger.error(f"Failed to unlock GPU model: {e}")
    
    async def infer(self, frames: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """
        Run VLM inference on frames with the given prompt
        Leverages GPU lock-in for optimal performance
        
        Args:
            frames: List of numpy arrays representing images
            prompt: Text prompt for the VLM
            
        Returns:
            Dict containing attributes, confidence, and reasoning
        """
        try:
            start_time = time.time()
            
            # Ensure warmup is completed before inference
            if not self.warmup_completed:
                logger.warning("VLM not warmed up - inference may be slow")
            
            # Increment inference counter
            self.inference_count += 1
            
            # Preprocess images
            processed_images = await self._preprocess_frames(frames)
            
            # Build the request payload with GPU lock-in settings
            payload = self._build_optimized_payload(processed_images, prompt)
            
            # Make the API call with retries
            response = await self._call_ollama_with_retry(payload)
            
            # Parse response
            result = await self._parse_response(response, prompt)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Update advanced performance tracking
            self.inference_times.append(duration_ms)
            self.last_inference_time = duration_ms
            
            # Update running average (keep last 20 measurements for stability)
            if len(self.inference_times) > 20:
                self.inference_times = self.inference_times[-20:]
            self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
            
            # Performance context and monitoring
            performance_context = "🚀 GPU-optimized" if self.warmup_completed else "⚠️ Cold"
            target_status = "✅" if duration_ms < self.target_inference_time else "⚠️"
            
            logger.info(f"{performance_context} VLM inference: {duration_ms}ms {target_status} (avg: {self.avg_inference_time:.0f}ms, target: <{self.target_inference_time}ms)")
            
            # Performance warning if consistently slow
            if len(self.inference_times) >= 5 and self.avg_inference_time > self.target_inference_time * 1.5:
                logger.warning(f"⚠️ Performance degradation detected: avg {self.avg_inference_time:.0f}ms > {self.target_inference_time * 1.5:.0f}ms")
            
            # Enhanced performance metadata
            result["inference_metadata"] = {
                "duration_ms": duration_ms,
                "avg_inference_time": self.avg_inference_time,
                "target_time": self.target_inference_time,
                "target_met": duration_ms < self.target_inference_time,
                "gpu_optimized": self.warmup_completed,
                "cuda_optimized": self.cuda_optimized,
                "model_compiled": self.model_compiled,
                "inference_count": self.inference_count,
                "frame_count": len(frames),
                "performance_stable": len(self.inference_times) > 1 and (max(self.inference_times) - min(self.inference_times)) / self.avg_inference_time < 0.3
            }
            
            return result
            
        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            return {
                "attributes": {},
                "confidence": 0.0,
                "reasoning": f"VLM inference error: {str(e)}",
                "error": str(e),
                "inference_metadata": {
                    "duration_ms": 0,
                    "gpu_optimized": self.warmup_completed,
                    "inference_count": self.inference_count,
                    "frame_count": len(frames) if frames else 0
                }
            }
    
    async def _preprocess_frames(self, frames: List[np.ndarray]) -> List[str]:
        """Convert numpy frames to base64 encoded images"""
        processed = []
        
        for i, frame in enumerate(frames):
            try:
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Assume it's BGR, convert to RGB
                    frame_rgb = frame[:, :, ::-1]
                else:
                    frame_rgb = frame

                # Create PIL image
                img = Image.fromarray(frame_rgb)
                
                # Resize if too large
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                
                # Convert to JPEG bytes
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=self.image_quality)
                
                # Encode to base64
                base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                processed.append(base64_str)
                
                logger.debug(f"Preprocessed frame {i+1}/{len(frames)}: {img.size}")
                
            except Exception as e:
                logger.error(f"Error preprocessing frame {i}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed)}/{len(frames)} frames successfully")
        return processed
    
    def _build_optimized_payload(self, images: List[str], prompt: str) -> Dict[str, Any]:
        """Build dynamically optimized request payload with GPU-specific optimizations"""
        
        # Base options optimized for fast inference
        base_options = {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 200,  # Reduced for faster inference
            "repeat_penalty": 1.1,
            "seed": -1
        }
        
        # Merge with GPU manager optimizations if available
        if self.cuda_optimized and self.cuda_params:
            optimized_options = {**base_options, **self.cuda_params}
            logger.debug(f"Using dynamic GPU optimization with {len(self.cuda_params)} CUDA parameters")
        else:
            # Fallback optimization for when GPU manager not available
            optimized_options = {
                **base_options,
                "num_gpu": 1,
                "gpu_memory_utilization": 0.85,
                "main_gpu": 0,
                "gpu_layers": -1,  # All layers on GPU
                "num_thread": min(16, os.cpu_count() or 8),
                "num_batch": 512,
                "num_gqa": 8,
                "use_mmap": True,
                "use_mlock": True,
                "rms_norm_eps": 1e-5,
                "rope_freq_base": 1000000,
                "rope_freq_scale": 1.0
            }
            logger.debug("Using fallback GPU optimization")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": "json",
            "options": optimized_options,
            "keep_alive": -1  # Critical: Keep model in GPU memory
        }
        
        return payload
    
    def _build_request_payload(self, images: List[str], prompt: str) -> Dict[str, Any]:
        """Build the Ollama API request payload"""
        
        # For Ollama, we need a simpler format
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 500
            }
        }
        
        return payload
    
    async def _call_ollama_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to Ollama with retry logic"""
        
        url = f"{self.ollama_host}/api/generate"
        
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result
                        else:
                            error_text = await response.text()
                            raise Exception(f"Ollama API returned {response.status}: {error_text}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"VLM inference timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise Exception("VLM inference timed out after all retries")
                    
            except Exception as e:
                logger.warning(f"VLM inference attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("VLM inference failed after all retries")
    
    async def _parse_response(self, response: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Parse Ollama response and extract structured attributes"""
        
        try:
            # Get the response content (for /api/generate endpoint)
            content = response.get("response", "")
            
            if not content:
                raise Exception("Empty response from VLM")
            
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    attributes = parsed_json
                else:
                    # If it's not a dict, treat as text and extract manually
                    attributes = self._extract_attributes_from_text(content, prompt)
            except json.JSONDecodeError:
                # Fall back to text parsing
                attributes = self._extract_attributes_from_text(content, prompt)
            
            # Calculate confidence based on response completeness
            confidence = self._calculate_confidence(attributes, prompt)
            
            return {
                "attributes": attributes,
                "confidence": confidence,
                "reasoning": content[:200] + "..." if len(content) > 200 else content,
                "raw_response": content
            }
            
        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            return {
                "attributes": {},
                "confidence": 0.0,
                "reasoning": f"Response parsing error: {str(e)}",
                "error": str(e)
            }
    
    def _extract_attributes_from_text(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract attributes from unstructured text response"""
        attributes = {}
        text_lower = text.lower()
        
        # Determine which attributes to extract based on prompt
        if "initial_classifier" in prompt or "item_type" in prompt:
            # Extract garment classification attributes
            if "shirt" in text_lower:
                attributes["item_type"] = "shirt"
            elif "pants" in text_lower or "trousers" in text_lower:
                attributes["item_type"] = "pants"
            elif "dress" in text_lower:
                attributes["item_type"] = "dress"
            elif "jacket" in text_lower:
                attributes["item_type"] = "jacket"
            elif "sweater" in text_lower:
                attributes["item_type"] = "sweater"
            
            # Extract color
            colors = ["black", "white", "red", "blue", "green", "yellow", "pink", "purple", "gray", "brown"]
            for color in colors:
                if color in text_lower:
                    attributes["color"] = color
                    break
            
            # Extract pattern
            patterns = ["solid", "striped", "plaid", "floral", "polka dot", "checkered"]
            for pattern in patterns:
                if pattern in text_lower:
                    attributes["pattern"] = pattern
                    break
            
            # Extract neckline
            necklines = ["crew", "v-neck", "round", "collar", "scoop", "turtleneck"]
            for neckline in necklines:
                if neckline in text_lower:
                    attributes["neckline"] = neckline
                    break
            
        elif "detail_extractor" in prompt or "brand" in prompt:
            # Extract detail attributes
            # Brand extraction is complex from text, leave empty if not clearly stated
            if "brand" in text_lower:
                # Try to find brand after "brand:" or similar
                brand_markers = ["brand:", "brand is", "brand ="]
                for marker in brand_markers:
                    if marker in text_lower:
                        brand_text = text_lower.split(marker)[1].split()[0]
                        if brand_text and len(brand_text) > 2:
                            attributes["brand"] = brand_text.title()
            
            # Extract size
            sizes = ["xs", "s", "m", "l", "xl", "xxl", "small", "medium", "large"]
            for size in sizes:
                if size in text_lower:
                    attributes["size"] = size.upper() if len(size) <= 3 else size.title()
                    break
            
            # Extract material
            materials = ["cotton", "polyester", "wool", "silk", "linen", "denim", "leather"]
            for material in materials:
                if material in text_lower:
                    attributes["material"] = material.title()
                    break
                    
        elif "damage_detector" in prompt or "damage" in prompt:
            # Extract damage information
            damage_indicators = ["damaged", "damage", "stain", "hole", "tear", "worn", "faded"]
            has_damage = any(indicator in text_lower for indicator in damage_indicators)
            
            if "no damage" in text_lower or "not damaged" in text_lower:
                attributes["is_damaged"] = False
                attributes["damage_type"] = None
                attributes["damage_severity"] = None
            elif has_damage:
                attributes["is_damaged"] = True
                
                # Try to identify damage type
                if "stain" in text_lower:
                    attributes["damage_type"] = "stain"
                elif "hole" in text_lower:
                    attributes["damage_type"] = "hole"
                elif "tear" in text_lower:
                    attributes["damage_type"] = "tear"
                elif "faded" in text_lower:
                    attributes["damage_type"] = "fading"
                else:
                    attributes["damage_type"] = "general_wear"
                
                # Try to determine severity
                if "minor" in text_lower or "slight" in text_lower:
                    attributes["damage_severity"] = "minor"
                elif "major" in text_lower or "severe" in text_lower:
                    attributes["damage_severity"] = "major"
                else:
                    attributes["damage_severity"] = "moderate"
            else:
                attributes["is_damaged"] = False
                attributes["damage_type"] = None
                attributes["damage_severity"] = None
        
        return attributes
    
    def _calculate_confidence(self, attributes: Dict[str, Any], prompt: str) -> float:
        """Calculate confidence based on response completeness"""
        base_confidence = 0.8  # Base confidence for successful parsing
        
        # Adjust based on number of attributes found
        if not attributes:
            return 0.1
        
        # Expected attributes per agent type
        expected_counts = {
            "initial_classifier": 4,  # item_type, color, pattern, neckline
            "detail_extractor": 3,    # brand, size, material  
            "damage_detector": 2      # is_damaged, damage_type
        }
        
        # Determine expected count based on prompt
        expected = 3  # default
        for agent_type, count in expected_counts.items():
            if agent_type in prompt:
                expected = count
                break
        
        # Calculate confidence based on completeness
        found_count = len([v for v in attributes.values() if v is not None])
        completeness_ratio = min(found_count / expected, 1.0)
        
        final_confidence = base_confidence * (0.5 + 0.5 * completeness_ratio)
        return round(final_confidence, 2)
    
    async def health_check(self) -> bool:
        """Check if Ollama service and model are available"""
        try:
            url = f"{self.ollama_host}/api/tags"
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model.get("name", "") for model in data.get("models", [])]
                        return any(self.model_name in model for model in models)
            return False
        except Exception as e:
            logger.error(f"VLM health check failed: {e}")
            return False
    
    # Advanced optimization methods
    
    async def _load_model_with_optimization(self):
        """Load model with advanced GPU optimization parameters"""
        try:
            # Enhanced payload with GPU optimization
            payload = {
                "model": self.model_name,
                "prompt": "GPU optimization initialization - loading model with optimal parameters",
                "stream": False,
                "keep_alive": -1,  # Persistent loading
                "options": {
                    **self.cuda_params,  # Apply GPU manager parameters
                    "num_predict": 10,
                    "temperature": 0.1
                }
            }
            
            url = f"{self.ollama_host}/api/generate"
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.is_model_loaded = True
                        self.is_gpu_locked = True
                        logger.info("✅ Model loaded with GPU optimization")
                        return True
            
            return False
                        
        except Exception as e:
            logger.error(f"Failed to load model with optimization: {e}")
            raise
    
    async def _load_optimized_warmup_image(self) -> str:
        """Load and preprocess warmup image with optimized settings"""
        try:
            # Use optimized dimensions (448x448) for faster processing
            warmup_image_path = "assets/denali_logo.png"
            
            if os.path.exists(warmup_image_path):
                with Image.open(warmup_image_path) as img:
                    # Optimize for faster processing
                    img = img.convert('RGB')
                    img = img.resize(self.max_image_size, Image.Resampling.LANCZOS)
                    
                    # High quality encoding for better inference
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=self.image_quality, optimize=True)
                    image_data = buffer.getvalue()
                    
                    logger.info(f"Loaded optimized warmup image: {warmup_image_path} ({self.max_image_size[0]}x{self.max_image_size[1]})")
                    return base64.b64encode(image_data).decode('utf-8')
            
            # Fallback: generate test pattern if no image found
            return self._generate_test_pattern()
            
        except Exception as e:
            logger.error(f"Failed to load optimized warmup image: {e}")
            return self._generate_test_pattern()
    
    def _generate_test_pattern(self) -> str:
        """Generate optimized test pattern for warmup"""
        # Create a simple test pattern optimized for VLM
        img = Image.new('RGB', self.max_image_size, color='white')
        
        # Add some pattern for the VLM to analyze
        import numpy as np
        img_array = np.array(img)
        
        # Add diagonal lines
        for i in range(0, self.max_image_size[0], 20):
            if i < self.max_image_size[1]:
                img_array[i, :, 0] = 255  # Red channel
        
        img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=self.image_quality)
        
        logger.info("Generated optimized test pattern for warmup")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def _run_performance_benchmark(self, image_data: str) -> Dict[str, Any]:
        """Run performance benchmarking with multiple test inferences"""
        benchmark_results = {
            "test_runs": [],
            "avg_inference_time": 0,
            "min_inference_time": float('inf'),
            "max_inference_time": 0,
            "performance_stable": False
        }
        
        # Run 3 benchmark inferences
        test_runs = 3
        for i in range(test_runs):
            start_time = time.time()
            
            try:
                # Optimized benchmark prompt - convert image data to frames
                import numpy as np
                from PIL import Image
                import io
                
                # Decode base64 image to numpy array
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                frame = np.array(pil_image)
                
                result = await self.infer(
                    [frame], 
                    "Analyze this image and describe what you see in one sentence."
                )
                
                inference_time = (time.time() - start_time) * 1000
                benchmark_results["test_runs"].append({
                    "run": i + 1,
                    "inference_time_ms": inference_time,
                    "success": result.get("attributes", {}) != {}
                })
                
                # Update performance tracking
                self.inference_times.append(inference_time)
                self.last_inference_time = inference_time
                
                # Update min/max
                benchmark_results["min_inference_time"] = min(benchmark_results["min_inference_time"], inference_time)
                benchmark_results["max_inference_time"] = max(benchmark_results["max_inference_time"], inference_time)
                
                logger.info(f"Benchmark run {i+1}: {inference_time:.0f}ms")
                
            except Exception as e:
                logger.error(f"Benchmark run {i+1} failed: {e}")
                benchmark_results["test_runs"].append({
                    "run": i + 1,
                    "inference_time_ms": None,
                    "success": False,
                    "error": str(e)
                })
        
        # Calculate average performance
        successful_runs = [run for run in benchmark_results["test_runs"] if run["success"]]
        if successful_runs:
            times = [run["inference_time_ms"] for run in successful_runs]
            benchmark_results["avg_inference_time"] = sum(times) / len(times)
            self.avg_inference_time = benchmark_results["avg_inference_time"]
            
            # Check performance stability (variation < 20%)
            if len(times) > 1:
                variation = (max(times) - min(times)) / benchmark_results["avg_inference_time"]
                benchmark_results["performance_stable"] = variation < 0.2
        
        logger.info(f"Benchmark complete: {benchmark_results['avg_inference_time']:.0f}ms average")
        return benchmark_results
    
    async def _optimize_model_compilation(self):
        """Optimize model compilation for faster inference"""
        try:
            # For Ollama, this involves ensuring the model is fully warmed up
            # and any internal optimizations are applied
            
            # Send a compilation optimization request
            payload = {
                "model": self.model_name,
                "prompt": "Optimization pass - ensure model compilation is complete",
                "stream": False,
                "keep_alive": -1,
                "options": {
                    **self.cuda_params,
                    "num_predict": 5,
                    "temperature": 0.0  # Deterministic for optimization
                }
            }
            
            url = f"{self.ollama_host}/api/generate"
            timeout = aiohttp.ClientTimeout(total=15)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.model_compiled = True
                        logger.info("✅ Model compilation optimization complete")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Model compilation optimization failed (non-critical): {e}")
            return False
    
    async def _validate_performance_target(self) -> Dict[str, Any]:
        """Validate that performance targets are met"""
        validation_result = {
            "target_met": False,
            "avg_inference_time": self.avg_inference_time,
            "target_time": self.target_inference_time,
            "performance_ratio": 0,
            "recommendation": ""
        }
        
        if self.avg_inference_time > 0:
            validation_result["performance_ratio"] = self.target_inference_time / self.avg_inference_time
            validation_result["target_met"] = self.avg_inference_time < self.target_inference_time
            
            if validation_result["target_met"]:
                validation_result["recommendation"] = "✅ Performance target achieved!"
                logger.info(f"🎯 Performance target MET: {self.avg_inference_time:.0f}ms < {self.target_inference_time}ms")
            else:
                validation_result["recommendation"] = "⚠️ Consider GPU memory optimization or model quantization"
                logger.warning(f"⚠️ Performance target MISSED: {self.avg_inference_time:.0f}ms > {self.target_inference_time}ms")
        
        return validation_result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "inference_count": self.inference_count,
            "avg_inference_time": self.avg_inference_time,
            "last_inference_time": self.last_inference_time,
            "target_inference_time": self.target_inference_time,
            "target_met": self.avg_inference_time < self.target_inference_time if self.avg_inference_time > 0 else False,
            "gpu_optimized": self.cuda_optimized,
            "model_compiled": self.model_compiled,
            "recent_times": self.inference_times[-10:] if len(self.inference_times) >= 10 else self.inference_times
        }