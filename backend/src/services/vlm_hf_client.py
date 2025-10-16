"""
HuggingFace VLM Client - Wrapper for HuggingFace Model Server
Provides the same interface as VLMGPULoader but calls HuggingFace server instead of Ollama
"""

import asyncio
import aiohttp
import base64
import hashlib
import logging
import os
import time
from collections import deque, OrderedDict
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from PIL import Image

from .provider_factory import get_provider_config, ModelProvider

logger = logging.getLogger(__name__)


class VLMHuggingFaceClient:
    """
    HuggingFace VLM client with same interface as VLMGPULoader
    Calls HuggingFace model server instead of Ollama
    """

    def __init__(self):
        # Get provider configuration
        provider_config = get_provider_config()

        if provider_config.provider != ModelProvider.HUGGINGFACE:
            raise ValueError("VLMHuggingFaceClient requires MODEL_PROVIDER=huggingface")

        self.hf_host = f"http://{provider_config.host}:{provider_config.port}"
        self.model_name = provider_config.model_name

        # State management
        self.is_loaded = False
        self.is_optimized = True  # HuggingFace models are "optimized" via quantization
        self.warmup_completed = False
        self.warmup_lock = asyncio.Lock()

        # Performance tracking - use deque for O(1) automatic eviction
        self.inference_count = 0
        self.avg_inference_time = 0.0
        self.inference_times = deque(maxlen=10)  # Auto-evicts oldest when full

        # Image preprocessing cache - use OrderedDict for better eviction
        self._image_cache = OrderedDict()  # Maintains insertion order for FIFO eviction
        self._cache_max_size = 50
        self._cache_hits = 0
        self._cache_misses = 0

        # Debug settings
        self.debug_save_frames = os.getenv('PREPROCESSING_SAVE_DEBUG_IMAGES', 'true').lower() == 'true'

        # Load preprocessing configuration
        self._load_preprocessing_config()

        logger.info(f"🚀 HuggingFace VLM Client initialized")
        logger.info(f"   Server: {self.hf_host}")
        logger.info(f"   Model: {self.model_name}")

    def _parse_damage_duplicate_keys(self, raw_json: str) -> str:
        """
        Parse JSON with duplicate damage keys (same type, different locations).
        Converts: {"Stain": [..], "Stain": [..], "Stain": [..]}
        To:       {"Stain_1": [..], "Stain_2": [..], "Stain_3": [..]}

        O(1) complexity per damage instance.
        """
        import re

        # Find damage object: "damage": { ... }
        match = re.search(r'"damage"\s*:\s*\{([^}]+)\}', raw_json, re.DOTALL)
        if not match:
            return raw_json

        damage_content = match.group(1)

        # Extract all "type": [bbox] pairs
        pattern = r'"([^"]+)"\s*:\s*\[([^\]]+)\]'
        matches = re.findall(pattern, damage_content)

        if not matches:
            return raw_json

        # Count occurrences with O(1) dict operations
        type_counts = {}
        new_pairs = []

        for damage_type, bbox in matches:
            # Increment count (O(1) dict get/set)
            type_counts[damage_type] = type_counts.get(damage_type, 0) + 1
            count = type_counts[damage_type]

            # Add index only for 2nd+ occurrence (keep first as-is for backward compat)
            if count > 1:
                indexed_type = f"{damage_type}_{count}"
            else:
                indexed_type = damage_type

            # Reconstruct key-value pair
            new_pairs.append(f'"{indexed_type}": [{bbox}]')

        # Rebuild damage object
        new_damage_content = ', '.join(new_pairs)
        new_json = raw_json[:match.start(1)] + new_damage_content + raw_json[match.end(1):]

        logger.info(f"📋 Parsed {len(matches)} damage instances ({len(type_counts)} unique types)")
        return new_json

    def _load_preprocessing_config(self):
        """Load preprocessing configuration from environment variables"""
        # Master toggle
        self.preprocessing_enabled = os.getenv('PREPROCESSING_ENABLED', 'true').lower() == 'true'

        # Per-agent toggles (same as VLMGPULoader)
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
                # Sharpen parameters
                'sharpen_amount': float(os.getenv('PREPROCESSING_DAMAGE_SHARPEN_AMOUNT', '1.0')),
                'sharpen_sigma': float(os.getenv('PREPROCESSING_DAMAGE_SHARPEN_SIGMA', '1.0')),
                'blur_size': int(os.getenv('PREPROCESSING_DAMAGE_BLUR_SIZE', '5')),
                # Edge parameters
                'edge_kernel': int(os.getenv('PREPROCESSING_DAMAGE_EDGE_KERNEL_SIZE', '3')),
                'edge_strength': float(os.getenv('PREPROCESSING_DAMAGE_EDGE_STRENGTH', '1.0')),
                # Contrast parameters
                'contrast_alpha': float(os.getenv('PREPROCESSING_DAMAGE_CONTRAST_ALPHA', '1.0')),
                'contrast_beta': int(os.getenv('PREPROCESSING_DAMAGE_CONTRAST_BETA', '0'))
            }
        }

    async def initialize_and_optimize(self, target_fps: int = 10, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Initialize HuggingFace VLM (no-op since server handles everything)
        Provides same interface as VLMGPULoader for compatibility
        """
        start_time = time.time()

        try:
            # Check if HuggingFace server is running
            if progress_callback:
                await progress_callback("Checking HuggingFace server...", 20)

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"{self.hf_host}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            health_data = await response.json()
                            logger.info(f"✅ HuggingFace server is healthy: {health_data}")
                            self.is_loaded = True
                        else:
                            raise RuntimeError(f"HuggingFace server health check failed: {response.status}")
                except Exception as e:
                    raise RuntimeError(f"Cannot connect to HuggingFace server at {self.hf_host}: {e}")

            if progress_callback:
                await progress_callback("HuggingFace model ready", 100)

            return {
                "status": "success",
                "optimization_level": "quantized",
                "load_time": time.time() - start_time,
                "server": self.hf_host,
                "model": self.model_name
            }

        except Exception as e:
            logger.error(f"HuggingFace client initialization failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "load_time": time.time() - start_time
            }

    async def _load_model_with_optimization(self):
        """No-op for compatibility - server handles model loading"""
        # Check server health
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.hf_host}/health") as response:
                if response.status == 200:
                    self.is_loaded = True
                    logger.info("HuggingFace model already loaded on server")
                else:
                    raise RuntimeError("HuggingFace server not ready")

    async def _apply_ollama_optimizations(self):
        """No-op for compatibility - HuggingFace doesn't use Ollama optimizations"""
        logger.info("Skipping Ollama optimizations (using HuggingFace)")
        pass

    async def _performance_warmup(self):
        """Run performance warmup"""
        async with self.warmup_lock:
            if self.warmup_completed:
                return

            logger.info("Starting HuggingFace warmup sequence...")

            # Simple warmup
            test_prompts = ["Hi", "Test", "OK"]
            warmup_times = []

            for i, prompt in enumerate(test_prompts):
                start_time = time.time()

                try:
                    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                    result = await self.infer_optimized([test_image], prompt)

                    inference_time = (time.time() - start_time) * 1000
                    warmup_times.append(inference_time)

                    logger.info(f"Warmup run {i+1}: {inference_time:.0f}ms")

                except Exception as e:
                    logger.warning(f"Warmup run {i+1} failed: {e}")

            if warmup_times:
                self.avg_inference_time = sum(warmup_times) / len(warmup_times)
                logger.info(f"Warmup completed. Average time: {self.avg_inference_time:.0f}ms")

            self.warmup_completed = True

    async def infer_optimized(self, frames: List[np.ndarray], prompt: str, agent_name: str = None) -> Dict[str, Any]:
        """
        Run inference using HuggingFace model server
        Same interface as VLMGPULoader.infer_optimized()
        """
        start_time = time.time()

        try:
            async with self.optimized_inference():
                # Convert frames to base64 (same logic as VLMGPULoader)
                base64_images = []
                processed_frames = []

                for frame_idx, frame in enumerate(frames):
                    # Ensure frame is uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # Generate frame hash for caching
                    frame_hash = self._get_image_hash(frame)

                    # Determine target resolution based on agent
                    if agent_name in ["initial_classifier"]:
                        target_resolution = (768, 768)
                    elif agent_name in ["damage_detector"]:
                        target_resolution = (1024, 1024)
                    elif agent_name == "detail_extractor":
                        target_resolution = (1280, 1280)
                    else:
                        target_resolution = (512, 512)

                    # Check cache
                    cache_key = self._get_cache_key(frame_hash, agent_name or "unknown", target_resolution)

                    if cache_key in self._image_cache:
                        # Cache hit
                        base64_img = self._image_cache[cache_key]
                        self._cache_hits += 1
                        logger.info(f"🎯 Cache HIT for {agent_name} frame[{frame_idx}] - "
                                  f"Resolution: {target_resolution[0]}x{target_resolution[1]}")
                    else:
                        # Cache miss - process image
                        self._cache_misses += 1

                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Apply preprocessing if enabled
                        if agent_name and self.preprocessing_config.get(agent_name, {}).get('enabled', False):
                            frame_rgb = self._apply_preprocessing(frame_rgb, agent_name)

                        image = Image.fromarray(frame_rgb)

                        # Apply target resolution
                        image.thumbnail(target_resolution, Image.Resampling.LANCZOS)

                        # Log resolution
                        final_width, final_height = image.size
                        logger.info(f"🔄 Resized resolution for {agent_name} frame[{frame_idx}]: {final_width}x{final_height}")

                        # Save debug frames if enabled
                        if self.debug_save_frames:
                            processed_frames.append((image, frame_idx, agent_name))

                        # Convert to base64
                        buffer = BytesIO()
                        image.save(buffer, format='JPEG', quality=95)
                        base64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

                        # Store in cache
                        self._image_cache[cache_key] = base64_img
                        self._manage_cache_size()

                        logger.debug(f"📦 Cache MISS for {agent_name} frame[{frame_idx}] - Added to cache")

                    base64_images.append(base64_img)

                # Save processed frames
                if self.debug_save_frames and processed_frames:
                    await self._save_vlm_frames(processed_frames, agent_name)

                # Agent-specific token limits - Increased to prevent JSON truncation
                if agent_name == "initial_classifier":
                    num_predict = 200  # Increased from 110 to prevent attribute truncation
                    temperature = 0.4
                    top_p = 0.6
                    top_k = 15
                elif agent_name == "detail_extractor":
                    num_predict = 180  # Increased from 100 to ensure full brand/size extraction
                    temperature = 0.2
                    top_p = 0.4
                    top_k = 5
                elif agent_name == "damage_detector":
                    num_predict = 300  # Increased from 150 for complete bbox + damage descriptions
                    temperature = 0.4
                    top_p = 0.6
                    top_k = 10
                else:
                    num_predict = 350  # Increased from 300 for safer default
                    temperature = 0.2
                    top_p = 0.85
                    top_k = 35

                # Call HuggingFace model server
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": base64_images,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "num_predict": num_predict,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    }
                }

                timeout = aiohttp.ClientTimeout(total=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(f"{self.hf_host}/api/generate", json=payload) as response:
                        if response.status == 200:
                            result_data = await response.json()

                            # Parse response
                            raw_response = result_data.get('response', '')

                            # Parse duplicate damage keys for damage_detector (before json.loads)
                            if agent_name == "damage_detector" and raw_response and '"damage"' in raw_response:
                                raw_response = self._parse_damage_duplicate_keys(raw_response)

                            # Try to parse JSON with fallback for truncated responses
                            try:
                                if raw_response:
                                    import json
                                    parsed = json.loads(raw_response)
                                    attributes = parsed if isinstance(parsed, dict) else {}
                                else:
                                    attributes = {}
                            except json.JSONDecodeError:
                                # Attempt recovery for truncated/malformed JSON
                                import re
                                text = raw_response.strip()
                                recovered = False

                                # Try to recover truncated JSON by closing braces
                                if text.startswith('{') and not text.endswith('}'):
                                    open_braces = text.count('{')
                                    close_braces = text.count('}')
                                    if open_braces > close_braces:
                                        # Remove incomplete key-value pair
                                        last_comma = text.rfind(',')
                                        if last_comma != -1 and text.rfind(':') > last_comma:
                                            text = text[:last_comma]
                                        # Add missing closing braces
                                        text += '}' * (open_braces - close_braces)
                                        try:
                                            parsed = json.loads(text)
                                            if isinstance(parsed, dict):
                                                attributes = parsed
                                                recovered = True
                                                logger.info(f"✅ Recovered truncated JSON ({len(parsed)} fields)")
                                        except:
                                            pass

                                # Fallback: extract key-value pairs using regex
                                if not recovered:
                                    pattern = r'"([^"]+)"\s*:\s*("([^"]*)"|null|true|false|[-+]?\d+(?:\.\d+)?)'
                                    matches = re.findall(pattern, raw_response)
                                    if matches:
                                        attributes = {}
                                        for match in matches:
                                            key = match[0]
                                            raw_val = match[1]
                                            if raw_val == 'null':
                                                attributes[key] = None
                                            elif raw_val in ('true', 'false'):
                                                attributes[key] = raw_val == 'true'
                                            elif raw_val.startswith('"'):
                                                attributes[key] = match[2]
                                            else:
                                                try:
                                                    attributes[key] = int(raw_val) if '.' not in raw_val else float(raw_val)
                                                except:
                                                    attributes[key] = raw_val
                                        logger.info(f"✅ Extracted {len(attributes)} fields via regex fallback")
                                    else:
                                        attributes = {"raw_text": raw_response}

                            # Calculate cache stats
                            total_cache_requests = self._cache_hits + self._cache_misses
                            cache_hit_rate = (self._cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0

                            result = {
                                "attributes": attributes,
                                "confidence": 0.85,
                                "reasoning": "HuggingFace inference completed",
                                "raw_response": raw_response,
                                "inference_metadata": {
                                    "optimization_level": "quantized",
                                    "model_name": self.model_name,
                                    "provider": "huggingface",
                                    "total_duration": result_data.get('total_duration', 0) / 1e6,
                                    "load_duration": result_data.get('load_duration', 0) / 1e6,
                                    "eval_duration": result_data.get('eval_duration', 0) / 1e6,
                                    "eval_count": result_data.get('eval_count', 0),
                                    "cache_hits": self._cache_hits,
                                    "cache_misses": self._cache_misses,
                                    "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                                    "cache_size": len(self._image_cache)
                                }
                            }
                        else:
                            error_text = await response.text()
                            raise Exception(f"HuggingFace API error {response.status}: {error_text}")

                # Track performance - deque automatically evicts oldest when full
                inference_time = (time.time() - start_time) * 1000
                self.inference_count += 1

                # Update running average with O(1) calculation (exact same result as sum/len)
                if len(self.inference_times) == 10:  # Deque is full, will evict oldest
                    # Remove oldest value's contribution, add new value
                    oldest = self.inference_times[0]
                    self.avg_inference_time += (inference_time - oldest) / 10
                else:
                    # Deque not full yet, use simple incremental average
                    n = len(self.inference_times)
                    self.avg_inference_time = (self.avg_inference_time * n + inference_time) / (n + 1)

                self.inference_times.append(inference_time)  # O(1) with auto-eviction

                logger.info(f"HuggingFace inference completed in {inference_time:.0f}ms "
                          f"(eval: {result['inference_metadata'].get('eval_duration', 0):.0f}ms)")

                return result

        except Exception as e:
            logger.error(f"HuggingFace inference failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "inference_metadata": {
                    "optimization_level": "quantized",
                    "provider": "huggingface"
                }
            }

    def _apply_preprocessing(self, frame_rgb: np.ndarray, agent_name: str) -> np.ndarray:
        """Apply preprocessing based on agent configuration (same as VLMGPULoader)"""
        config = self.preprocessing_config.get(agent_name, {})
        if not config.get('enabled', False):
            return frame_rgb

        preprocess_type = config.get('type', 'none')

        # Apply preprocessing based on agent and type
        if agent_name == 'detail_extractor':
            if preprocess_type == 'clahe_sharpen':
                # Combined CLAHE + Sharpening for optimal OCR
                lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=config.get('clahe_clip', 1.0),
                    tileGridSize=(config.get('clahe_grid', 8), config.get('clahe_grid', 8))
                )
                l_channel = clahe.apply(l_channel)
                enhanced_lab = cv2.merge([l_channel, a, b])
                frame_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

                # Apply sharpening
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

    async def _save_vlm_frames(self, processed_frames: List[tuple], agent_name: str) -> None:
        """Save processed frames for debugging"""
        try:
            base_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/captured_frames")
            vlm_dir = base_dir / "vlm_preprocessed" / f"{agent_name}_vlm"
            vlm_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for pil_image, frame_idx, agent in processed_frames:
                filename = f"{agent_name}_final_frame{frame_idx+1}_{timestamp}.jpg"
                filepath = vlm_dir / filename
                await asyncio.to_thread(pil_image.save, str(filepath), 'JPEG', quality=95)
                logger.info(f"💾 VLM-ready frame saved: {filepath}")
        except Exception as e:
            logger.error(f"Error saving VLM frames: {e}")

    def _get_image_hash(self, frame: np.ndarray) -> str:
        """Generate unique hash for frame content"""
        return hashlib.md5(frame.tobytes()).hexdigest()

    def _get_cache_key(self, frame_hash: str, agent_name: str, resolution: tuple) -> str:
        """Generate cache key from frame hash, agent, and resolution"""
        return f"{frame_hash}_{agent_name}_{resolution[0]}x{resolution[1]}"

    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds max size - FIFO eviction with batch pruning"""
        if len(self._image_cache) > self._cache_max_size:
            # Remove 20% of cache to reduce thrashing (batch eviction)
            num_to_remove = max(10, int(self._cache_max_size * 0.2))
            for _ in range(num_to_remove):
                if self._image_cache:  # Check if not empty
                    self._image_cache.popitem(last=False)  # O(1) FIFO removal (remove oldest)
            logger.debug(f"Cache pruned: removed {num_to_remove} entries, current size: {len(self._image_cache)}")

    @asynccontextmanager
    async def optimized_inference(self):
        """Context manager for optimized inference sessions"""
        logger.debug("Starting HuggingFace inference session")
        try:
            yield
        finally:
            logger.debug("Ending HuggingFace inference session")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "is_loaded": self.is_loaded,
            "is_optimized": self.is_optimized,
            "warmup_completed": self.warmup_completed,
            "inference_count": self.inference_count,
            "avg_inference_time": self.avg_inference_time,
            "provider": "huggingface",
            "optimization_level": "quantized",
            "model": self.model_name,
            "server": self.hf_host
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.hf_host}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    server_healthy = response.status == 200
                    health_data = await response.json() if server_healthy else {}

            return {
                "status": "healthy" if server_healthy else "unhealthy",
                "server_healthy": server_healthy,
                "model_loaded": self.is_loaded,
                "server": self.hf_host,
                "model": self.model_name,
                **health_data
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up HuggingFace VLM client...")
        self.is_loaded = False
        self.is_optimized = False
