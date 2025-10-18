"""
Provider Factory - Abstraction layer for model serving backends
Supports: Ollama, HuggingFace
"""

import os
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers"""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class ProviderConfig:
    """Configuration for a model provider"""

    def __init__(
        self,
        provider: ModelProvider,
        host: str,
        port: int,
        model_name: str,
        base_url: Optional[str] = None
    ):
        self.provider = provider
        self.host = host
        self.port = port
        self.model_name = model_name
        self.base_url = base_url or f"http://{host}:{port}"

    @property
    def api_url(self) -> str:
        """Get the API base URL"""
        return self.base_url

    def __repr__(self) -> str:
        return f"ProviderConfig(provider={self.provider.value}, url={self.base_url}, model={self.model_name})"


class ProviderFactory:
    """
    Factory for creating provider configurations
    Reads from environment variables to determine provider
    """

    @staticmethod
    def get_provider_config() -> ProviderConfig:
        """
        Get provider configuration based on environment variables

        Environment variables:
        - MODEL_PROVIDER: "ollama" or "huggingface" (default: "ollama")
        - OLLAMA_MODEL: Model name for Ollama (default: "qwen2.5vl:3b")
        - HUGGINGFACE_MODEL: Model name for HuggingFace (default: "Qwen/Qwen2-VL-2B-Instruct")
        - OLLAMA_HOST: Ollama server host (default: "localhost")
        - OLLAMA_PORT: Ollama server port (default: 11434)
        - HF_SERVER_HOST: HuggingFace server host (default: "localhost")
        - HF_SERVER_PORT: HuggingFace server port (default: 11435)
        """

        # Get provider from environment (default: ollama)
        provider_str = os.getenv("MODEL_PROVIDER", "ollama").lower()

        try:
            provider = ModelProvider(provider_str)
        except ValueError:
            logger.warning(f"Invalid MODEL_PROVIDER '{provider_str}', defaulting to 'ollama'")
            provider = ModelProvider.OLLAMA

        # Create configuration based on provider
        if provider == ModelProvider.OLLAMA:
            config = ProviderConfig(
                provider=ModelProvider.OLLAMA,
                host=os.getenv("OLLAMA_HOST", "localhost"),
                port=int(os.getenv("OLLAMA_PORT", "11434")),
                model_name=os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
            )
        elif provider == ModelProvider.HUGGINGFACE:
            config = ProviderConfig(
                provider=ModelProvider.HUGGINGFACE,
                host=os.getenv("HF_SERVER_HOST", "localhost"),
                port=int(os.getenv("HF_SERVER_PORT", "11435")),
                model_name=os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f" Provider configuration: {config}")

        return config

    @staticmethod
    def get_ollama_host() -> str:
        """
        Get Ollama host URL (for backward compatibility)
        Uses provider factory to determine correct host
        """
        config = ProviderFactory.get_provider_config()
        return config.api_url

    @staticmethod
    def get_model_name() -> str:
        """
        Get model name (for backward compatibility)
        Uses provider factory to determine correct model
        """
        config = ProviderFactory.get_provider_config()
        return config.model_name

    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get provider information for logging/debugging"""
        config = ProviderFactory.get_provider_config()

        return {
            "provider": config.provider.value,
            "host": config.host,
            "port": config.port,
            "api_url": config.api_url,
            "model_name": config.model_name
        }


# Convenience functions for easy access
def get_provider_config() -> ProviderConfig:
    """Get the current provider configuration"""
    return ProviderFactory.get_provider_config()


def get_ollama_host() -> str:
    """Get the API host URL (compatible with existing code)"""
    return ProviderFactory.get_ollama_host()


def get_model_name() -> str:
    """Get the model name (compatible with existing code)"""
    return ProviderFactory.get_model_name()


def is_huggingface_provider() -> bool:
    """Check if using HuggingFace provider"""
    config = ProviderFactory.get_provider_config()
    return config.provider == ModelProvider.HUGGINGFACE


def is_ollama_provider() -> bool:
    """Check if using Ollama provider"""
    config = ProviderFactory.get_provider_config()
    return config.provider == ModelProvider.OLLAMA


# Log provider configuration on module import
logger.info("="*60)
logger.info(" Model Provider Configuration")
provider_info = ProviderFactory.get_provider_info()
logger.info(f"   Provider: {provider_info['provider'].upper()}")
logger.info(f"   API URL: {provider_info['api_url']}")
logger.info(f"   Model: {provider_info['model_name']}")
logger.info("="*60)
