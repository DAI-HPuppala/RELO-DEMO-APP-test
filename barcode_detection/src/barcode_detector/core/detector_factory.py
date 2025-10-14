"""Factory for creating barcode detectors based on configuration."""

from typing import Optional, Dict, Any
from ..interfaces.barcode_detector import BarcodeDetector
from ..core.config_manager import ConfigManager
from ..utils.logger import BarcodeLogger


class DetectorFactory:
    """Factory class for creating barcode detectors."""
    
    def __init__(self, config: ConfigManager, logger: BarcodeLogger) -> None:
        """Initialize detector factory.
        
        Args:
            config: Configuration manager
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        
    def create_detector(self) -> Optional[BarcodeDetector]:
        """Create detector based on configuration.
        
        Returns:
            Detector instance or None if creation failed
        """
        detection_mode = self.config.detection.mode.lower()
        
        try:
            if detection_mode == "custom":
                return self._create_custom_detector()
            elif detection_mode == "library":
                return self._create_library_detector()
            elif detection_mode == "hybrid":
                return self._create_hybrid_detector()
            else:
                self.logger.get_logger("factory").error(
                    f"Unknown detection mode: {detection_mode}"
                )
                return None
                
        except Exception as e:
            self.logger.log_error(e, "detector_factory_create")
            return None
            
    def _create_custom_detector(self) -> Optional[BarcodeDetector]:
        """Create custom detector.
        
        Returns:
            Custom detector instance
        """
        try:
            from ..detectors.custom_detector import CustomBarcodeDetector
            
            custom_config = {
                "algorithm": self.config.custom_detector.algorithm,
                "preprocessing": self.config.custom_detector.preprocessing,
                "edge_threshold": self.config.custom_detector.edge_threshold,
                "min_bar_width": self.config.custom_detector.min_bar_width,
                "max_bar_width": self.config.custom_detector.max_bar_width,
            }
            
            detector = CustomBarcodeDetector(custom_config)
            
            if detector.initialize():
                self.logger.get_logger("factory").info("Custom detector created successfully")
                return detector
            else:
                self.logger.get_logger("factory").error("Failed to initialize custom detector")
                return None
                
        except ImportError as e:
            self.logger.get_logger("factory").error(f"Failed to import custom detector: {e}")
            return None
            
    def _create_library_detector(self) -> Optional[BarcodeDetector]:
        """Create library detector.
        
        Returns:
            Library detector instance
        """
        try:
            from ..detectors.library_detector import LibraryBarcodeDetector
            
            library_config = {
                "primary": self.config.library.primary,
                "secondary": self.config.library.secondary,
                "fallback_enabled": self.config.library.fallback_enabled,
            }
            
            detector = LibraryBarcodeDetector(library_config)
            
            if detector.initialize():
                self.logger.get_logger("factory").info("Library detector created successfully")
                return detector
            else:
                self.logger.get_logger("factory").error("Failed to initialize library detector")
                return None
                
        except ImportError as e:
            self.logger.get_logger("factory").error(f"Failed to import library detector: {e}")
            return None
            
    def _create_hybrid_detector(self) -> Optional[BarcodeDetector]:
        """Create hybrid detector.
        
        Returns:
            Hybrid detector instance
        """
        try:
            from ..detectors.library_detector import HybridBarcodeDetector
            
            custom_config = {
                "algorithm": self.config.custom_detector.algorithm,
                "preprocessing": self.config.custom_detector.preprocessing,
                "edge_threshold": self.config.custom_detector.edge_threshold,
                "min_bar_width": self.config.custom_detector.min_bar_width,
                "max_bar_width": self.config.custom_detector.max_bar_width,
            }
            
            library_config = {
                "primary": self.config.library.primary,
                "secondary": self.config.library.secondary,
                "fallback_enabled": self.config.library.fallback_enabled,
            }
            
            detector = HybridBarcodeDetector(custom_config, library_config)
            
            if detector.initialize():
                self.logger.get_logger("factory").info("Hybrid detector created successfully")
                return detector
            else:
                self.logger.get_logger("factory").error("Failed to initialize hybrid detector")
                return None
                
        except ImportError as e:
            self.logger.get_logger("factory").error(f"Failed to import hybrid detector: {e}")
            return None
            
    def get_available_detectors(self) -> Dict[str, bool]:
        """Get availability status of different detector types.
        
        Returns:
            Dictionary mapping detector type to availability
        """
        availability = {
            "custom": False,
            "library": False,
            "hybrid": False
        }
        
        # Check custom detector
        try:
            from ..detectors.custom_detector import CustomBarcodeDetector
            availability["custom"] = True
        except ImportError:
            pass
            
        # Check library detector
        try:
            from ..detectors.library_detector import LibraryBarcodeDetector
            
            # Create temporary instance to check library availability
            temp_detector = LibraryBarcodeDetector({})
            if temp_detector.initialize():
                availability["library"] = True
                temp_detector.cleanup()
                
        except ImportError:
            pass
            
        # Hybrid is available if either custom or library is available
        availability["hybrid"] = availability["custom"] or availability["library"]
        
        return availability