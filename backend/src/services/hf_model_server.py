#!/usr/bin/env python3
"""
HuggingFace Model Server - Ollama-compatible API for HuggingFace VLMs
Serves vision-language models from HuggingFace Hub with automatic download
Supports multiple formats: safetensors, GGUF, pytorch, etc.

This is a standalone server that doesn't depend on other services.
"""

import asyncio
import base64
import json
import logging
import os
import sys
from io import BytesIO
from typing import Dict, Any, Optional, List
from pathlib import Path

# NOTE: This is a standalone server - no imports from other services
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from PIL import Image
import torch
from dotenv import load_dotenv

# Load environment variables from backend/.env
backend_dir = Path(__file__).parent.parent.parent
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Use the main app's logging configuration
# No need to configure logging here since this is imported by the main app
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Request model for /api/generate endpoint"""
    model: str
    prompt: str
    images: Optional[List[str]] = None
    stream: bool = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[int] = -1


class GenerateResponse(BaseModel):
    """Response model for /api/generate endpoint"""
    model: str
    response: str
    done: bool = True
    total_duration: int = 0
    load_duration: int = 0
    eval_duration: int = 0
    eval_count: int = 0


class HuggingFaceModelServer:
    """
    HuggingFace VLM server with Ollama-compatible API
    Auto-downloads models from HuggingFace Hub
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", host: str = "0.0.0.0", port: int = 11435):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use standard HuggingFace cache directory for persistence
        # This ensures models are downloaded once and reused across runs
        self.cache_dir = Path.home() / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing HuggingFace Model Server")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Port: {self.port}")

    def _detect_model_format(self) -> str:
        """
        Detect the model format from model name or files
        Returns: 'gguf', 'safetensors', or 'pytorch'
        """
        model_name_lower = self.model_name.lower()

        # Check if GGUF model
        if 'gguf' in model_name_lower or model_name_lower.endswith('.gguf'):
            return 'gguf'

        # Check cache directory for existing model files
        model_cache_path = self.cache_dir / self.model_name.replace('/', '--')
        if model_cache_path.exists():
            # Check for safetensors files
            if list(model_cache_path.glob('*.safetensors')):
                return 'safetensors'
            # Check for GGUF files
            if list(model_cache_path.glob('*.gguf')):
                return 'gguf'

        # Default to safetensors (preferred format)
        return 'safetensors'

    async def load_model(self):
        """
        Load model from HuggingFace Hub with auto-download
        Supports multiple formats: GGUF, safetensors, pytorch
        """
        try:
            logger.info(f"Loading model: {self.model_name}")

            # Detect model format
            model_format = self._detect_model_format()
            logger.info(f"Detected model format: {model_format}")

            if model_format == 'gguf':
                await self._load_gguf_model()
            else:
                await self._load_transformers_model(prefer_safetensors=(model_format == 'safetensors'))

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Installation requirements:")
            logger.error("   Standard models: pip install transformers torch accelerate pillow")
            logger.error("   GGUF models: pip install llama-cpp-python pillow")
            raise

    async def _load_gguf_model(self):
        """Load GGUF format model using llama-cpp-python"""
        try:
            from llama_cpp import Llama

            logger.info("Loading GGUF model with llama-cpp-python...")

            # For GGUF, model_name should be a path or HF repo with .gguf file
            # Auto-download from HuggingFace if needed
            if '/' in self.model_name and not self.model_name.endswith('.gguf'):
                # It's a HuggingFace repo, try to download
                from huggingface_hub import hf_hub_download, list_repo_files

                logger.info(f"Downloading GGUF model from HuggingFace: {self.model_name}")

                # Get quantization level from environment (e.g., "Q4_0", "Q6_K")
                quantization = os.getenv("GGUF_QUANTIZATION", "Q4_0")
                logger.info(f"Using quantization: {quantization}")

                # List all files in the repo to find the right GGUF file
                try:
                    repo_files = list_repo_files(repo_id=self.model_name)
                    gguf_files = [f for f in repo_files if f.endswith('.gguf')]

                    logger.info(f"Found {len(gguf_files)} GGUF files in repo")

                    # Find the file matching the quantization level
                    matching_file = None
                    for gguf_file in gguf_files:
                        if quantization.upper() in gguf_file.upper():
                            matching_file = gguf_file
                            break

                    if not matching_file:
                        # Fallback to first GGUF file if no match
                        matching_file = gguf_files[0] if gguf_files else None
                        logger.warning(f"No file matching {quantization}, using: {matching_file}")
                    else:
                        logger.info(f"Found matching file: {matching_file}")

                    if not matching_file:
                        raise ValueError(f"No GGUF files found in {self.model_name}")

                    # Download the specific GGUF file
                    logger.info(f"Downloading: {matching_file}")
                    model_file = hf_hub_download(
                        repo_id=self.model_name,
                        filename=matching_file,
                        cache_dir=str(self.cache_dir)
                    )
                    model_path = model_file
                    logger.info(f"Model cached at: {model_path}")

                    # Download mmproj file for vision models (if exists)
                    mmproj_file = None
                    mmproj_files = [f for f in repo_files if f.startswith('mmproj') and f.endswith('.gguf')]
                    if mmproj_files:
                        # Prefer Q8_0 for quality, fallback to F16
                        mmproj_preferred = next((f for f in mmproj_files if 'Q8_0' in f), mmproj_files[0])
                        logger.info(f"Downloading vision projector: {mmproj_preferred}")
                        mmproj_file = hf_hub_download(
                            repo_id=self.model_name,
                            filename=mmproj_preferred,
                            cache_dir=str(self.cache_dir)
                        )
                        logger.info(f"Vision projector cached at: {mmproj_file}")

                except Exception as e:
                    logger.error(f"Failed to download GGUF model: {e}")
                    raise
            else:
                model_path = self.model_name

            # Load GGUF model with optimized settings for VLM
            logger.info(f"Loading GGUF model into memory...")

            # Prepare kwargs
            model_kwargs = {
                "model_path": model_path,
                "n_ctx": 4096,  # Increased context for better vision understanding
                "n_gpu_layers": -1 if self.device == "cuda" else 0,  # Use all GPU layers if available
                "n_threads": 8,  # Optimize CPU threads
                "verbose": True,  # Enable verbose for debugging
            }

            # Add mmproj for vision models if available
            if mmproj_file:
                logger.info(f"Loading with vision projector: {mmproj_file}")
                model_kwargs["mmproj"] = mmproj_file
                model_kwargs["chat_format"] = "llava-1-6"  # Use llava chat format for vision

            self.model = Llama(**model_kwargs)

            # For GGUF, we don't need a separate processor
            self.processor = None

            logger.info(f"GGUF model loaded successfully on {self.device}")

            # Log model info
            if self.device == "cuda":
                logger.info(f"   GPU Layers: All layers offloaded to GPU")
            logger.info(f"   Context Size: 4096 tokens")
            logger.info(f"   Quantization: {os.getenv('GGUF_QUANTIZATION', 'Q4_0')}")

        except ImportError:
            logger.error("llama-cpp-python not installed")
            logger.error("   Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise

    async def _load_transformers_model(self, prefer_safetensors: bool = True):
        """Load model using transformers (safetensors or pytorch format)"""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

            logger.info(f"Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Check quantization setting from environment
            quantization = os.getenv("TRANSFORMERS_QUANTIZATION", "4bit").lower()

            # Prepare load kwargs
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }

            # Configure quantization
            if quantization == "4bit" and self.device == "cuda":
                logger.info(f"Quantization: 4-bit NF4 (~3-4GB VRAM)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                load_kwargs["quantization_config"] = quantization_config

            elif quantization == "8bit" and self.device == "cuda":
                logger.info(f"Quantization: 8-bit (~5-6GB VRAM)")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                load_kwargs["quantization_config"] = quantization_config

            else:
                # Full precision
                load_kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32

            if prefer_safetensors:
                load_kwargs["use_safetensors"] = True

            logger.info(f"Loading model to GPU...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **load_kwargs
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            raise

    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_bytes))
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise

    def _extract_or_validate_json(self, text: str) -> str:
        """
        Extract or validate JSON from response text
        Handles cases where model wraps JSON in markdown or includes extra text
        """
        import re

        # Try to parse as-is first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                extracted = json_match.group(1)
                json.loads(extracted)  # Validate
                return extracted
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                extracted = json_match.group(0)
                json.loads(extracted)  # Validate
                return extracted
            except json.JSONDecodeError:
                pass

        # If all else fails, try to return as-is with warning
        logger.warning(f"Could not extract valid JSON from response, returning as-is: {text[:100]}...")
        return text

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate response using HuggingFace VLM
        Compatible with Ollama API format
        """
        import time
        start_time = time.time()

        try:
            if self.model is None or self.processor is None:
                await self.load_model()

            # Prepare conversation format
            messages = []

            # Handle images if provided
            images = []
            if request.images:
                for img_b64 in request.images:
                    image = self.decode_base64_image(img_b64)
                    images.append(image)

                # Create message with images
                content = []
                for img in images:
                    content.append({"type": "image"})
                content.append({"type": "text", "text": request.prompt})

                messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                # Text-only message
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": request.prompt}]
                })

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            if images:
                inputs = self.processor(
                    text=[text],
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True
                )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get generation parameters from options
            options = request.options or {}
            # Use MAX_RESPONSE_TOKENS from env as default (default: 512)
            default_max_tokens = int(os.getenv("MAX_RESPONSE_TOKENS", "512"))
            max_new_tokens = options.get("num_predict", default_max_tokens)
            temperature = options.get("temperature", 0.2)
            top_p = options.get("top_p", 0.85)
            top_k = options.get("top_k", 35)

            # Generate
            load_start = time.time()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            eval_duration = int((time.time() - load_start) * 1e9)

            # Decode output
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Extract only the assistant's response (after the last user message)
            if "assistant\n" in generated_text:
                response_text = generated_text.split("assistant\n")[-1].strip()
            else:
                response_text = generated_text.strip()

            # Ensure JSON format if requested
            if request.format == "json":
                # Try to extract JSON from response
                response_text = self._extract_or_validate_json(response_text)

            # Calculate durations (in nanoseconds for Ollama compatibility)
            total_duration = int((time.time() - start_time) * 1e9)
            load_duration = int((load_start - start_time) * 1e9)

            # Count tokens
            eval_count = len(outputs[0]) - len(inputs["input_ids"][0])

            logger.info(f"Generated response in {total_duration/1e6:.0f}ms "
                       f"(eval: {eval_duration/1e6:.0f}ms, tokens: {eval_count})")

            return GenerateResponse(
                model=self.model_name,
                response=response_text,
                done=True,
                total_duration=total_duration,
                load_duration=load_duration,
                eval_duration=eval_duration,
                eval_count=eval_count
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


# Create FastAPI app
app = FastAPI(
    title="HuggingFace Model Server",
    description="Ollama-compatible API for HuggingFace VLMs",
    version="1.0.0"
)

# Global server instance
server: Optional[HuggingFaceModelServer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model server on startup"""
    global server

    # Get model name from environment
    model_name = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    port = int(os.getenv("HF_SERVER_PORT", "11435"))

    server = HuggingFaceModelServer(model_name=model_name, port=port)

    # Pre-load model
    logger.info("Pre-loading model for faster inference...")
    await server.load_model()
    logger.info("Server ready!")


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate endpoint - compatible with Ollama API
    """
    if server is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    return await server.generate(request)


@app.get("/api/tags")
async def list_models():
    """
    List available models - compatible with Ollama API
    """
    if server is None:
        return {"models": []}

    return {
        "models": [
            {
                "name": server.model_name,
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 0,
                "digest": "hf_model"
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if server is None or server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model": server.model_name,
        "device": server.device
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "HuggingFace Model Server",
        "version": "1.0.0",
        "compatible_with": "Ollama API",
        "model": server.model_name if server else "Not loaded"
    }


def main():
    """Run the server"""
    # Get configuration from environment
    model_name = os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    host = os.getenv("HF_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("HF_SERVER_PORT", "11435"))

    logger.info("="*60)
    logger.info("Starting HuggingFace Model Server")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Host: {host}:{port}")
    logger.info(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*60)

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
