"""Barcode Detection System - A modular barcode and QR code detection library.

This package provides a comprehensive solution for barcode and QR code detection
using multiple camera types (OAK-D PoE, CV60) and detection strategies.

Quick Start:
    ```python
    # Using OAK-D camera
    from barcode_detector import OAKDCamera, PreprocessingPipeline

    camera = OAKDCamera(resolution_4k=True, ip_address="169.254.1.222")
    if camera.connect():
        ret, frame = camera.read()
        # Process frame
        camera.disconnect()

    # Using preprocessing
    from barcode_detector.preprocessing import PreprocessingPipeline

    pipeline = PreprocessingPipeline.create_default_pipeline()
    processed_images = pipeline.process(frame)
    ```

Main Components:
    - cameras: Camera implementations (OAKDCamera, CV60Camera)
    - detectors: Detection algorithms (CustomDetector, LibraryDetector)
    - preprocessing: Image preprocessing utilities
    - utils: Logging and utility functions
"""

__version__ = "1.0.0"
__author__ = "DenaliAI Automation"

# Camera implementations
from .cameras import (
    BaseCamera,
    OAKDCamera,
    CV60Camera,
    create_oakd_camera,
    create_cv60_camera,
)

# Detector implementations
from .detectors import (
    CustomBarcodeDetector,
    LibraryBarcodeDetector,
    HybridBarcodeDetector,
    Code128Decoder,
)

# Preprocessing utilities
from .preprocessing import (
    PreprocessingPipeline,
    apply_clahe,
    apply_sharpening,
    apply_bilateral_filter,
    apply_binary_threshold,
    apply_adaptive_threshold,
)

# Core utilities
from .core.config_manager import ConfigManager
from .core.detector_factory import DetectorFactory

# Robust detector (supports both OAK-D and CV60)
from .robust_detector import RobustBarcodeDetector

# Logger (optional - may not be available)
try:
    from .utils.logger import setup_logger
except ImportError:
    setup_logger = None

# Export public API
__all__ = [
    # Version
    '__version__',
    '__author__',

    # Cameras
    'BaseCamera',
    'OAKDCamera',
    'CV60Camera',
    'create_oakd_camera',
    'create_cv60_camera',

    # Detectors
    'CustomBarcodeDetector',
    'LibraryBarcodeDetector',
    'HybridBarcodeDetector',
    'Code128Decoder',

    # Preprocessing
    'PreprocessingPipeline',
    'apply_clahe',
    'apply_sharpening',
    'apply_bilateral_filter',
    'apply_binary_threshold',
    'apply_adaptive_threshold',

    # Core
    'ConfigManager',
    'DetectorFactory',
    'RobustBarcodeDetector',

    # Utils
    'setup_logger',
]
