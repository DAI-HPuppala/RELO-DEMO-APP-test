"""Abstract interface for barcode detectors."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class DetectionResult:
    """Result of a barcode detection operation."""
    
    data: str
    type: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, width, height)
    processing_time_ms: float
    algorithm: str
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def corners(self) -> List[Tuple[int, int]]:
        """Get corners of bounding box as list of (x, y) tuples."""
        x, y, w, h = self.bounding_box
        return [
            (x, y),           # top-left
            (x + w, y),       # top-right
            (x + w, y + h),   # bottom-right
            (x, y + h)        # bottom-left
        ]
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary."""
        return {
            "data": self.data,
            "type": self.type,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "processing_time_ms": self.processing_time_ms,
            "algorithm": self.algorithm,
            "metadata": self.metadata
        }


class BarcodeDetector(ABC):
    """Abstract base class for barcode detectors."""
    
    def __init__(self, name: str) -> None:
        """Initialize detector with name.
        
        Args:
            name: Name of the detector
        """
        self.name = name
        self._is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
        
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect barcodes in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection results
        """
        pass
        
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup detector resources."""
        pass
        
    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self._is_initialized
        
    @property
    def supported_types(self) -> List[str]:
        """Get list of supported barcode types."""
        return []
        
    def can_detect_type(self, barcode_type: str) -> bool:
        """Check if detector can detect specific barcode type.
        
        Args:
            barcode_type: Type of barcode to check
            
        Returns:
            True if supported, False otherwise
        """
        return barcode_type.lower() in [t.lower() for t in self.supported_types]
        
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self.is_initialized})"