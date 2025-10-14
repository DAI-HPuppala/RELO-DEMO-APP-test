"""Library-based barcode detector using pyzbar and zxing as fallbacks."""

import time
from typing import List, Optional, Dict, Any
import numpy as np

from ..interfaces.barcode_detector import BarcodeDetector, DetectionResult
from ..utils.logger import performance_monitor, log_function_call


class LibraryBarcodeDetector(BarcodeDetector):
    """Library-based barcode detector with multiple fallback options."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize library detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("Library")
        self.config = config
        self.primary_library = config.get("primary", "pyzbar")
        self.secondary_library = config.get("secondary", "zxing")
        self.fallback_enabled = config.get("fallback_enabled", True)
        
        # Import libraries
        self.pyzbar = None
        self.zxing = None
        self._import_libraries()
        
    def _import_libraries(self) -> None:
        """Import available libraries."""
        try:
            import pyzbar.pyzbar as pyzbar
            self.pyzbar = pyzbar
        except ImportError:
            pass
            
        try:
            import zxing
            self.zxing = zxing
        except ImportError:
            pass
            
    @performance_monitor("library_detector_init")
    def initialize(self) -> bool:
        """Initialize the library detector."""
        # Check if at least one library is available
        if self.pyzbar is None and self.zxing is None:
            return False
            
        self._is_initialized = True
        return True
        
    @property
    def supported_types(self) -> List[str]:
        """Get supported barcode types."""
        types = []
        
        if self.pyzbar:
            types.extend([
                "CODE128", "CODE39", "CODE93", "CODABAR",
                "EAN8", "EAN13", "EAN14", "UPCA", "UPCE",
                "ITF", "QRCODE", "PDF417", "DATAMATRIX", "AZTEC"
            ])
            
        if self.zxing:
            types.extend([
                "QRCODE", "DATAMATRIX", "AZTEC", "PDF417",
                "CODE128", "CODE39", "CODE93", "CODABAR",
                "EAN8", "EAN13", "UPCA", "UPCE", "ITF"
            ])
            
        return list(set(types))  # Remove duplicates
        
    @performance_monitor("library_detect")
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect barcodes using library implementations.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection results
        """
        if not self.is_initialized:
            return []
            
        start_time = time.perf_counter()
        results = []
        
        # Try primary library first
        if self.primary_library == "pyzbar":
            results = self._detect_with_pyzbar(image)
        elif self.primary_library == "zxing":
            results = self._detect_with_zxing(image)
            
        # Try fallback if no results and fallback is enabled
        if not results and self.fallback_enabled:
            if self.secondary_library == "pyzbar" and self.pyzbar:
                results = self._detect_with_pyzbar(image)
            elif self.secondary_library == "zxing" and self.zxing:
                results = self._detect_with_zxing(image)
                
        # Update processing time for all results
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        
        for result in results:
            result.processing_time_ms = processing_time
            
        return results
        
    @log_function_call
    def _detect_with_pyzbar(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect barcodes using pyzbar library.
        
        Args:
            image: Input image
            
        Returns:
            List of detection results
        """
        if not self.pyzbar:
            return []
            
        try:
            # Convert BGR to RGB for pyzbar
            rgb_image = image[:, :, ::-1]
            
            # Detect barcodes
            barcodes = self.pyzbar.decode(rgb_image)
            
            results = []
            for barcode in barcodes:
                # Extract bounding box
                if hasattr(barcode, 'rect'):
                    x, y, w, h = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height
                else:
                    # Fallback: calculate from polygon
                    points = barcode.polygon
                    if points:
                        xs = [p.x for p in points]
                        ys = [p.y for p in points]
                        x, y = min(xs), min(ys)
                        w, h = max(xs) - x, max(ys) - y
                    else:
                        x, y, w, h = 0, 0, 0, 0
                
                # Decode data
                try:
                    data = barcode.data.decode('utf-8')
                except UnicodeDecodeError:
                    data = str(barcode.data)
                
                # Create detection result
                result = DetectionResult(
                    data=data,
                    type=barcode.type,
                    confidence=1.0,  # pyzbar doesn't provide confidence
                    bounding_box=(x, y, w, h),
                    processing_time_ms=0.0,  # Will be updated by caller
                    algorithm="pyzbar",
                    metadata={
                        "orientation": getattr(barcode, 'orientation', None),
                        "quality": getattr(barcode, 'quality', None)
                    }
                )
                
                results.append(result)
                
            return results
            
        except Exception as e:
            # Log error but don't raise
            return []
            
    @log_function_call
    def _detect_with_zxing(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect barcodes using zxing library.
        
        Args:
            image: Input image
            
        Returns:
            List of detection results
        """
        if not self.zxing:
            return []
            
        try:
            # zxing-cpp expects grayscale or RGB
            if len(image.shape) == 3:
                # Convert BGR to RGB
                rgb_image = image[:, :, ::-1]
            else:
                rgb_image = image
                
            # Create zxing reader
            from zxing import BarcodeReader
            reader = BarcodeReader()
            
            # Detect barcodes
            results = []
            
            # Try to read barcode
            barcode = reader.decode(rgb_image)
            
            if barcode:
                # Extract bounding box (zxing-cpp provides position)
                if hasattr(barcode, 'position') and barcode.position:
                    # Calculate bounding box from position points
                    points = barcode.position
                    if len(points) >= 4:
                        xs = [p.x for p in points]
                        ys = [p.y for p in points]
                        x, y = int(min(xs)), int(min(ys))
                        w, h = int(max(xs) - x), int(max(ys) - y)
                    else:
                        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
                else:
                    # No position info, use full image
                    x, y, w, h = 0, 0, image.shape[1], image.shape[0]
                
                result = DetectionResult(
                    data=barcode.text,
                    type=barcode.format.name,
                    confidence=getattr(barcode, 'confidence', 1.0),
                    bounding_box=(x, y, w, h),
                    processing_time_ms=0.0,  # Will be updated by caller
                    algorithm="zxing",
                    metadata={
                        "orientation": getattr(barcode, 'orientation', None),
                        "symbology_identifier": getattr(barcode, 'symbology_identifier', None)
                    }
                )
                
                results.append(result)
                
            return results
            
        except Exception as e:
            # Log error but don't raise
            return []
            
    def cleanup(self) -> None:
        """Cleanup detector resources."""
        self._is_initialized = False


class HybridBarcodeDetector(BarcodeDetector):
    """Hybrid detector that combines custom and library implementations."""
    
    def __init__(self, custom_config: Dict[str, Any], library_config: Dict[str, Any]) -> None:
        """Initialize hybrid detector.
        
        Args:
            custom_config: Configuration for custom detector
            library_config: Configuration for library detector
        """
        super().__init__("Hybrid")
        
        # Import custom detector
        from .custom_detector import CustomBarcodeDetector
        
        self.custom_detector = CustomBarcodeDetector(custom_config)
        self.library_detector = LibraryBarcodeDetector(library_config)
        
    @performance_monitor("hybrid_detector_init")
    def initialize(self) -> bool:
        """Initialize both detectors."""
        custom_init = self.custom_detector.initialize()
        library_init = self.library_detector.initialize()
        
        # Hybrid is initialized if at least one detector works
        self._is_initialized = custom_init or library_init
        return self._is_initialized
        
    @property
    def supported_types(self) -> List[str]:
        """Get combined supported types."""
        types = []
        types.extend(self.custom_detector.supported_types)
        types.extend(self.library_detector.supported_types)
        return list(set(types))  # Remove duplicates
        
    @performance_monitor("hybrid_detect")
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect using both custom and library detectors.
        
        Args:
            image: Input image
            
        Returns:
            Combined detection results
        """
        if not self.is_initialized:
            return []
            
        all_results = []
        
        # Try custom detector first (faster for simple cases)
        if self.custom_detector.is_initialized:
            custom_results = self.custom_detector.detect(image)
            all_results.extend(custom_results)
            
        # Try library detector
        if self.library_detector.is_initialized:
            library_results = self.library_detector.detect(image)
            all_results.extend(library_results)
            
        # Remove duplicates based on data and proximity
        unique_results = self._remove_duplicates(all_results)
        
        return unique_results
        
    def _remove_duplicates(self, results: List[DetectionResult]) -> List[DetectionResult]:
        """Remove duplicate detections.
        
        Args:
            results: List of detection results
            
        Returns:
            List with duplicates removed
        """
        if not results:
            return results
            
        unique_results = []
        
        for result in results:
            is_duplicate = False
            
            for unique_result in unique_results:
                # Check if same data
                if result.data == unique_result.data:
                    # Check if bounding boxes overlap significantly
                    overlap = self._calculate_overlap(
                        result.bounding_box,
                        unique_result.bounding_box
                    )
                    
                    if overlap > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        # Keep the one with higher confidence
                        if result.confidence > unique_result.confidence:
                            unique_results.remove(unique_result)
                            unique_results.append(result)
                        break
                        
            if not is_duplicate:
                unique_results.append(result)
                
        return unique_results
        
    def _calculate_overlap(
        self, 
        box1: tuple[int, int, int, int], 
        box2: tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two bounding boxes.
        
        Args:
            box1: First bounding box (x, y, w, h)
            box2: Second bounding box (x, y, w, h)
            
        Returns:
            Overlap ratio (0-1)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return 0.0
            
        intersection_area = (right - left) * (bottom - top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
        
    def cleanup(self) -> None:
        """Cleanup both detectors."""
        self.custom_detector.cleanup()
        self.library_detector.cleanup()
        self._is_initialized = False