"""V2 Configuration Management System"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    timer_seconds: float
    max_inferences: int = 10
    min_confidence: float = 0.7
    batch_size_progression: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    enable_multi_inference: bool = True
    debug_mode: bool = False


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""
    enable_checkpointing: bool = True
    checkpoint_interval_seconds: float = 60.0
    max_checkpoint_size_mb: float = 100.0
    enable_frame_sharing: bool = True
    enable_pause_resume: bool = True
    session_timeout_seconds: float = 3600.0  # 1 hour
    max_concurrent_sessions: int = 10


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    gpu_serialization: bool = True
    max_batch_size: int = 5
    inference_timeout_seconds: float = 30.0
    enable_mock_inference: bool = True  # For testing without GPU
    mock_inference_delay: float = 0.5
    confidence_threshold: float = 0.6


@dataclass
class FrameConfig:
    """Configuration for frame management"""
    save_debug_frames: bool = False
    frame_format: str = "jpeg"
    jpeg_quality: int = 85
    max_frames_per_session: int = 1000
    cleanup_after_hours: int = 24
    frame_width: int = 640
    frame_height: int = 480
    enable_frame_validation: bool = True


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket server"""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(default_factory=lambda: ["*"])
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    max_connections: int = 100


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "backend.log"
    max_log_size_mb: float = 100.0
    backup_count: int = 5
    enable_console_logging: bool = True
    enable_file_logging: bool = True


class V2Config:
    """Main configuration class for V2 system"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Default configurations
        self.agents = {
            "initial_classifier": AgentConfig(timer_seconds=4.0),
            "detail_extractor": AgentConfig(timer_seconds=3.0),
            "damage_detector": AgentConfig(timer_seconds=4.0),
            "final_compiler": AgentConfig(timer_seconds=1.0, enable_multi_inference=False)
        }
        
        self.orchestrator = OrchestratorConfig()
        self.inference = InferenceConfig()
        self.frames = FrameConfig()
        self.websocket = WebSocketConfig()
        self.logging = LoggingConfig()
        
        # Environment variable overrides
        self._load_from_env()
        
        # Load from config file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        
        # Apply logging configuration
        self._configure_logging()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Agent timers
        if os.getenv("INITIAL_CLASSIFIER_TIMER"):
            self.agents["initial_classifier"].timer_seconds = float(os.getenv("INITIAL_CLASSIFIER_TIMER"))
        if os.getenv("DETAIL_EXTRACTOR_TIMER"):
            self.agents["detail_extractor"].timer_seconds = float(os.getenv("DETAIL_EXTRACTOR_TIMER"))
        if os.getenv("DAMAGE_DETECTOR_TIMER"):
            self.agents["damage_detector"].timer_seconds = float(os.getenv("DAMAGE_DETECTOR_TIMER"))
        
        # Frame saving
        if os.getenv("SAVE_DEBUG_FRAMES"):
            self.frames.save_debug_frames = os.getenv("SAVE_DEBUG_FRAMES", "false").lower() == "true"
        
        # Mock inference
        if os.getenv("ENABLE_MOCK_INFERENCE"):
            self.inference.enable_mock_inference = os.getenv("ENABLE_MOCK_INFERENCE", "true").lower() == "true"
        
        # WebSocket
        if os.getenv("WEBSOCKET_PORT"):
            self.websocket.port = int(os.getenv("WEBSOCKET_PORT"))
        if os.getenv("WEBSOCKET_HOST"):
            self.websocket.host = os.getenv("WEBSOCKET_HOST")
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            self.logging.log_level = os.getenv("LOG_LEVEL")
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update agent configs
            if "agents" in config_data:
                for agent_name, agent_config in config_data["agents"].items():
                    if agent_name in self.agents:
                        for key, value in agent_config.items():
                            setattr(self.agents[agent_name], key, value)
            
            # Update other configs
            for section in ["orchestrator", "inference", "frames", "websocket", "logging"]:
                if section in config_data:
                    config_obj = getattr(self, section)
                    for key, value in config_data[section].items():
                        setattr(config_obj, key, value)
            
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def _configure_logging(self):
        """Configure logging based on settings"""
        import logging.handlers
        
        # Set log level
        log_level = getattr(logging, self.logging.log_level.upper(), logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if self.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(self.logging.log_format))
            root_logger.addHandler(console_handler)
        
        # Add file handler with rotation
        if self.logging.enable_file_logging and self.logging.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.log_file,
                maxBytes=int(self.logging.max_log_size_mb * 1024 * 1024),
                backupCount=self.logging.backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(self.logging.log_format))
            root_logger.addHandler(file_handler)
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file"""
        config_data = {
            "agents": {
                name: {
                    "timer_seconds": agent.timer_seconds,
                    "max_inferences": agent.max_inferences,
                    "min_confidence": agent.min_confidence,
                    "batch_size_progression": agent.batch_size_progression,
                    "enable_multi_inference": agent.enable_multi_inference,
                    "debug_mode": agent.debug_mode
                }
                for name, agent in self.agents.items()
            },
            "orchestrator": {
                "enable_checkpointing": self.orchestrator.enable_checkpointing,
                "checkpoint_interval_seconds": self.orchestrator.checkpoint_interval_seconds,
                "max_checkpoint_size_mb": self.orchestrator.max_checkpoint_size_mb,
                "enable_frame_sharing": self.orchestrator.enable_frame_sharing,
                "enable_pause_resume": self.orchestrator.enable_pause_resume,
                "session_timeout_seconds": self.orchestrator.session_timeout_seconds,
                "max_concurrent_sessions": self.orchestrator.max_concurrent_sessions
            },
            "inference": {
                "gpu_serialization": self.inference.gpu_serialization,
                "max_batch_size": self.inference.max_batch_size,
                "inference_timeout_seconds": self.inference.inference_timeout_seconds,
                "enable_mock_inference": self.inference.enable_mock_inference,
                "mock_inference_delay": self.inference.mock_inference_delay,
                "confidence_threshold": self.inference.confidence_threshold
            },
            "frames": {
                "save_debug_frames": self.frames.save_debug_frames,
                "frame_format": self.frames.frame_format,
                "jpeg_quality": self.frames.jpeg_quality,
                "max_frames_per_session": self.frames.max_frames_per_session,
                "cleanup_after_hours": self.frames.cleanup_after_hours,
                "frame_width": self.frames.frame_width,
                "frame_height": self.frames.frame_height,
                "enable_frame_validation": self.frames.enable_frame_validation
            },
            "websocket": {
                "host": self.websocket.host,
                "port": self.websocket.port,
                "cors_origins": self.websocket.cors_origins,
                "max_message_size": self.websocket.max_message_size,
                "ping_interval": self.websocket.ping_interval,
                "ping_timeout": self.websocket.ping_timeout,
                "max_connections": self.websocket.max_connections
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_format": self.logging.log_format,
                "log_file": self.logging.log_file,
                "max_log_size_mb": self.logging.max_log_size_mb,
                "backup_count": self.logging.backup_count,
                "enable_console_logging": self.logging.enable_console_logging,
                "enable_file_logging": self.logging.enable_file_logging
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")
    
    def get_agent_timer(self, agent_name: str) -> float:
        """Get timer for specific agent"""
        if agent_name in self.agents:
            return self.agents[agent_name].timer_seconds
        return 4.0  # Default
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate agent timers
        for name, agent in self.agents.items():
            if agent.timer_seconds <= 0:
                errors.append(f"Invalid timer for {name}: {agent.timer_seconds}")
            if agent.max_inferences <= 0:
                errors.append(f"Invalid max_inferences for {name}: {agent.max_inferences}")
        
        # Validate ports
        if not 1 <= self.websocket.port <= 65535:
            errors.append(f"Invalid WebSocket port: {self.websocket.port}")
        
        # Validate file paths
        if self.logging.log_file:
            log_dir = Path(self.logging.log_file).parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create log directory: {e}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "agents": {name: vars(agent) for name, agent in self.agents.items()},
            "orchestrator": vars(self.orchestrator),
            "inference": vars(self.inference),
            "frames": vars(self.frames),
            "websocket": vars(self.websocket),
            "logging": vars(self.logging)
        }


# Global configuration instance
v2_config = V2Config()

# Load from default config file if it exists
default_config_path = Path(__file__).parent.parent.parent / "config" / "v2_config.json"
if default_config_path.exists():
    v2_config = V2Config(str(default_config_path))