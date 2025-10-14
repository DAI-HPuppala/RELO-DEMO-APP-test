"""Configuration Manager for Barcode Detector."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class DetectionConfig:
    """Configuration for detection parameters."""
    mode: str = "hybrid"
    min_confidence: float = 0.7
    max_detection_time: int = 1000
    enable_multiple_detection: bool = True


@dataclass
class CameraConfig:
    """Configuration for camera parameters."""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class CustomDetectorConfig:
    """Configuration for custom detector."""
    algorithm: str = "code128"
    preprocessing: bool = True
    edge_threshold: int = 50
    min_bar_width: int = 2
    max_bar_width: int = 20


@dataclass
class LibraryConfig:
    """Configuration for library fallback."""
    fallback_enabled: bool = True
    primary: str = "pyzbar"
    secondary: str = "zxing"


@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    enable_gpu_acceleration: bool = False
    enable_multithreading: bool = True
    max_threads: int = 4


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    file: str = "logs/barcode_detector.log"
    enable_performance_logging: bool = True


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    show_bounding_boxes: bool = True
    show_detection_time: bool = True
    save_annotated_images: bool = False
    output_directory: str = "output/"


class ConfigManager:
    """Manages configuration for the barcode detector."""

    def __init__(self, config_file: Optional[str] = None) -> None:
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file (.env)
        """
        self._config_file = config_file or ".env"
        self._load_config()
        
    def _load_config(self) -> None:
        """Load configuration from environment variables."""
        if os.path.exists(self._config_file):
            load_dotenv(self._config_file)
            
        # Detection configuration
        self.detection = DetectionConfig(
            mode=os.getenv("DETECTION_MODE", "hybrid"),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7")),
            max_detection_time=int(os.getenv("MAX_DETECTION_TIME", "1000")),
            enable_multiple_detection=os.getenv("ENABLE_MULTIPLE_DETECTION", "true").lower() == "true"
        )
        
        # Input source configuration
        self.input_source = os.getenv("INPUT_SOURCE", "camera")
        self.image_path = os.getenv("IMAGE_PATH", "examples/sample_barcodes/")
        
        # Camera configuration
        self.camera = CameraConfig(
            index=int(os.getenv("CAMERA_INDEX", "0")),
            width=int(os.getenv("CAMERA_WIDTH", "640")),
            height=int(os.getenv("CAMERA_HEIGHT", "480")),
            fps=int(os.getenv("CAMERA_FPS", "30"))
        )
        
        # Custom detector configuration
        self.custom_detector = CustomDetectorConfig(
            algorithm=os.getenv("CUSTOM_DETECTOR_ALGORITHM", "code128"),
            preprocessing=os.getenv("CUSTOM_DETECTOR_PREPROCESSING", "true").lower() == "true",
            edge_threshold=int(os.getenv("CUSTOM_DETECTOR_EDGE_THRESHOLD", "50")),
            min_bar_width=int(os.getenv("CUSTOM_DETECTOR_MIN_BAR_WIDTH", "2")),
            max_bar_width=int(os.getenv("CUSTOM_DETECTOR_MAX_BAR_WIDTH", "20"))
        )
        
        # Library configuration
        self.library = LibraryConfig(
            fallback_enabled=os.getenv("LIBRARY_FALLBACK_ENABLED", "true").lower() == "true",
            primary=os.getenv("LIBRARY_PRIMARY", "pyzbar"),
            secondary=os.getenv("LIBRARY_SECONDARY", "zxing")
        )
        
        # Performance configuration
        self.performance = PerformanceConfig(
            enable_gpu_acceleration=os.getenv("ENABLE_GPU_ACCELERATION", "false").lower() == "true",
            enable_multithreading=os.getenv("ENABLE_MULTITHREADING", "true").lower() == "true",
            max_threads=int(os.getenv("MAX_THREADS", "4"))
        )
        
        # Logging configuration
        self.logging = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file=os.getenv("LOG_FILE", "logs/barcode_detector.log"),
            enable_performance_logging=os.getenv("ENABLE_PERFORMANCE_LOGGING", "true").lower() == "true"
        )
        
        # Output configuration
        self.output = OutputConfig(
            show_bounding_boxes=os.getenv("SHOW_BOUNDING_BOXES", "true").lower() == "true",
            show_detection_time=os.getenv("SHOW_DETECTION_TIME", "true").lower() == "true",
            save_annotated_images=os.getenv("SAVE_ANNOTATED_IMAGES", "false").lower() == "true",
            output_directory=os.getenv("OUTPUT_DIRECTORY", "output/")
        )
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return os.getenv(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        os.environ[key] = str(value)
        
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        
    def validate(self) -> Dict[str, Any]:
        """Validate current configuration.
        
        Returns:
            Dictionary of validation results
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate detection mode
        if self.detection.mode not in ["custom", "library", "hybrid"]:
            validation_results["errors"].append(
                f"Invalid detection mode: {self.detection.mode}"
            )
            validation_results["valid"] = False
            
        # Validate input source
        if self.input_source not in ["image", "camera"]:
            validation_results["errors"].append(
                f"Invalid input source: {self.input_source}"
            )
            validation_results["valid"] = False
            
        # Validate image path if using image input
        if self.input_source == "image":
            if not os.path.exists(self.image_path):
                validation_results["errors"].append(
                    f"Image path does not exist: {self.image_path}"
                )
                validation_results["valid"] = False
                
        # Validate confidence threshold
        if not 0.0 <= self.detection.min_confidence <= 1.0:
            validation_results["errors"].append(
                f"Invalid confidence threshold: {self.detection.min_confidence}"
            )
            validation_results["valid"] = False
            
        # Create output directory if it doesn't exist
        output_dir = Path(self.output.output_directory)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                validation_results["warnings"].append(
                    f"Created output directory: {self.output.output_directory}"
                )
            except Exception as e:
                validation_results["errors"].append(
                    f"Could not create output directory: {e}"
                )
                validation_results["valid"] = False
                
        # Create logs directory if it doesn't exist
        log_dir = Path(self.logging.file).parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                validation_results["warnings"].append(
                    f"Created logs directory: {log_dir}"
                )
            except Exception as e:
                validation_results["errors"].append(
                    f"Could not create logs directory: {e}"
                )
                validation_results["valid"] = False
                
        return validation_results
        
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""ConfigManager:
  Detection Mode: {self.detection.mode}
  Input Source: {self.input_source}
  Camera: {self.camera.width}x{self.camera.height}@{self.camera.fps}fps
  Confidence: {self.detection.min_confidence}
  Multithreading: {self.performance.enable_multithreading}
  Log Level: {self.logging.level}"""