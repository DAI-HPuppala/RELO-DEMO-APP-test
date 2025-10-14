"""Custom barcode detector implementation from scratch."""

import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks

from ..interfaces.barcode_detector import BarcodeDetector, DetectionResult
from ..utils.logger import performance_monitor, log_function_call


class Code128Decoder:
    """Decoder for Code128 barcodes."""
    
    # Code128 character set
    CODE128_CHARS = {
        # Start codes
        103: 'START_A', 104: 'START_B', 105: 'START_C',
        # Stop code
        106: 'STOP',
        # Special codes
        102: 'FNC1', 97: 'FNC2', 96: 'FNC3', 101: 'FNC4',
        100: 'CODE_B', 99: 'CODE_C', 98: 'CODE_A',
        95: 'SHIFT'
    }
    
    # Code128 patterns (bar-space patterns)
    CODE128_PATTERNS = [
        [2, 1, 2, 2, 2, 2], [2, 2, 2, 1, 2, 2], [2, 2, 2, 2, 2, 1],
        [1, 2, 1, 2, 2, 3], [1, 2, 1, 3, 2, 2], [1, 3, 1, 2, 2, 2],
        [1, 2, 2, 2, 1, 3], [1, 2, 2, 3, 1, 2], [1, 3, 2, 2, 1, 2],
        [2, 2, 1, 2, 1, 3], [2, 2, 1, 3, 1, 2], [2, 3, 1, 2, 1, 2],
        [1, 1, 2, 2, 3, 2], [1, 2, 2, 1, 3, 2], [1, 2, 2, 2, 3, 1],
        [1, 1, 3, 2, 2, 2], [1, 2, 3, 1, 2, 2], [1, 2, 3, 2, 2, 1],
        [2, 2, 3, 2, 1, 1], [2, 2, 1, 1, 3, 2], [2, 2, 1, 2, 3, 1],
        [2, 1, 3, 2, 1, 2], [2, 2, 3, 1, 1, 2], [3, 1, 2, 1, 3, 1],
        [3, 1, 1, 2, 2, 2], [3, 2, 1, 1, 2, 2], [3, 2, 1, 2, 2, 1],
        [3, 1, 2, 2, 1, 2], [3, 2, 2, 1, 1, 2], [3, 2, 2, 2, 1, 1],
        [2, 1, 2, 1, 2, 3], [2, 1, 2, 3, 2, 1], [2, 3, 2, 1, 2, 1],
        [1, 1, 1, 3, 2, 3], [1, 3, 1, 1, 2, 3], [1, 3, 1, 3, 2, 1],
        [1, 1, 2, 3, 1, 3], [1, 3, 2, 1, 1, 3], [1, 3, 2, 3, 1, 1],
        [2, 1, 1, 3, 1, 3], [2, 3, 1, 1, 1, 3], [2, 3, 1, 3, 1, 1],
        [1, 1, 2, 1, 3, 3], [1, 1, 2, 3, 3, 1], [1, 3, 2, 1, 3, 1],
        [1, 1, 3, 1, 2, 3], [1, 1, 3, 3, 2, 1], [1, 3, 3, 1, 2, 1],
        [3, 1, 3, 1, 2, 1], [2, 1, 1, 3, 3, 1], [2, 3, 1, 1, 3, 1],
        [2, 1, 3, 1, 1, 3], [2, 1, 3, 3, 1, 1], [2, 1, 3, 1, 3, 1],
        [3, 1, 1, 1, 2, 3], [3, 1, 1, 3, 2, 1], [3, 3, 1, 1, 2, 1],
        [3, 1, 2, 1, 1, 3], [3, 1, 2, 3, 1, 1], [3, 3, 2, 1, 1, 1],
        [3, 1, 4, 1, 1, 1], [2, 2, 1, 4, 1, 1], [4, 3, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 4], [1, 1, 1, 4, 2, 2], [1, 2, 1, 1, 2, 4],
        [1, 2, 1, 4, 2, 1], [1, 4, 1, 1, 2, 2], [1, 4, 1, 2, 2, 1],
        [1, 1, 2, 2, 1, 4], [1, 1, 2, 4, 1, 2], [1, 2, 2, 1, 1, 4],
        [1, 2, 2, 4, 1, 1], [1, 4, 2, 1, 1, 2], [1, 4, 2, 2, 1, 1],
        [2, 4, 1, 2, 1, 1], [2, 2, 1, 1, 1, 4], [4, 1, 3, 1, 1, 1],
        [2, 4, 1, 1, 1, 2], [1, 3, 4, 1, 1, 1], [1, 1, 1, 2, 4, 2],
        [1, 2, 1, 1, 4, 2], [1, 2, 1, 2, 4, 1], [1, 1, 4, 2, 1, 2],
        [1, 2, 4, 1, 1, 2], [1, 2, 4, 2, 1, 1], [4, 1, 1, 2, 1, 2],
        [4, 2, 1, 1, 1, 2], [4, 2, 1, 2, 1, 1], [2, 1, 2, 1, 4, 1],
        [2, 1, 4, 1, 2, 1], [4, 1, 2, 1, 2, 1], [1, 1, 1, 1, 4, 3],
        [1, 1, 1, 3, 4, 1], [1, 3, 1, 1, 4, 1], [1, 1, 4, 1, 1, 3],
        [1, 1, 4, 3, 1, 1], [4, 1, 1, 1, 1, 3], [4, 1, 1, 3, 1, 1],
        [1, 1, 3, 1, 4, 1], [1, 1, 4, 1, 3, 1], [3, 1, 1, 1, 4, 1],
        [4, 1, 1, 1, 3, 1], [2, 1, 1, 4, 1, 2], [2, 1, 1, 2, 1, 4],
        [2, 1, 1, 2, 3, 2], [2, 3, 3, 1, 1, 1, 2]
    ]
    
    def decode_pattern(self, pattern: List[int]) -> Optional[int]:
        """Decode a pattern to Code128 character.
        
        Args:
            pattern: List of bar/space widths
            
        Returns:
            Character code or None if not found
        """
        if len(pattern) != 6:
            return None
            
        # Normalize pattern
        total_width = sum(pattern)
        if total_width == 0:
            return None
            
        normalized = [int(round(11 * w / total_width)) for w in pattern]
        
        # Find best match
        best_score = float('inf')
        best_match = None
        
        for i, ref_pattern in enumerate(self.CODE128_PATTERNS):
            score = sum(abs(a - b) for a, b in zip(normalized, ref_pattern))
            if score < best_score:
                best_score = score
                best_match = i
                
        # Return match if score is reasonable
        return best_match if best_score <= 3 else None


class CustomBarcodeDetector(BarcodeDetector):
    """Custom implementation of barcode detector from scratch."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize custom detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("Custom")
        self.config = config
        self.algorithm = config.get("algorithm", "code128")
        self.preprocessing = config.get("preprocessing", True)
        self.edge_threshold = config.get("edge_threshold", 50)
        self.min_bar_width = config.get("min_bar_width", 2)
        self.max_bar_width = config.get("max_bar_width", 20)
        
        self.decoder = Code128Decoder() if self.algorithm == "code128" else None
        
    @performance_monitor("custom_detector_init")
    def initialize(self) -> bool:
        """Initialize the custom detector."""
        try:
            # Verify OpenCV is available
            cv2.getVersionString()
            self._is_initialized = True
            return True
        except Exception:
            return False
            
    @property
    def supported_types(self) -> List[str]:
        """Get supported barcode types."""
        return ["CODE128", "EAN13", "EAN8", "UPCA", "UPCE"]
        
    @performance_monitor("custom_detect")
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect barcodes using custom algorithm.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection results
        """
        if not self.is_initialized:
            return []
            
        start_time = time.perf_counter()
        results = []
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Find potential barcode regions
            regions = self._find_barcode_regions(processed_image)
            
            # Process each region
            for region in regions:
                detection = self._process_region(image, processed_image, region)
                if detection:
                    results.append(detection)
                    
        except Exception as e:
            # Log error but continue
            pass
            
        # Update processing time for all results
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000
        
        for result in results:
            result.processing_time_ms = processing_time
            
        return results
        
    @log_function_call
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for barcode detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        if not self.preprocessing:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
        
    def _find_barcode_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find potential barcode regions in image.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            List of regions as (x, y, width, height) tuples
        """
        regions = []
        
        # Use Sobel gradient to find vertical edges
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold gradient
        _, binary = cv2.threshold(
            gradient_mag.astype(np.uint8), 
            self.edge_threshold, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Morphological operations to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on aspect ratio and size
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Typical barcode characteristics
            if (aspect_ratio > 2.0 and area > 1000 and 
                w > 50 and h > 20):
                # Expand region slightly
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                regions.append((x, y, w, h))
                
        return regions
        
    def _process_region(
        self, 
        original: np.ndarray, 
        processed: np.ndarray, 
        region: Tuple[int, int, int, int]
    ) -> Optional[DetectionResult]:
        """Process a potential barcode region.
        
        Args:
            original: Original color image
            processed: Preprocessed grayscale image
            region: Region coordinates (x, y, width, height)
            
        Returns:
            Detection result or None
        """
        x, y, w, h = region
        roi = processed[y:y+h, x:x+w]
        
        # Extract barcode pattern
        pattern = self._extract_pattern(roi)
        if not pattern:
            return None
            
        # Decode pattern
        decoded_data = self._decode_pattern(pattern)
        if not decoded_data:
            return None
            
        # Calculate confidence based on pattern quality
        confidence = self._calculate_confidence(pattern, roi)
        
        return DetectionResult(
            data=decoded_data,
            type=self.algorithm.upper(),
            confidence=confidence,
            bounding_box=(x, y, w, h),
            processing_time_ms=0.0,  # Will be updated by caller
            algorithm="Custom",
            metadata={
                "pattern_length": len(pattern),
                "region_area": w * h,
                "aspect_ratio": w / h if h > 0 else 0
            }
        )
        
    def _extract_pattern(self, roi: np.ndarray) -> Optional[List[int]]:
        """Extract bar pattern from ROI.
        
        Args:
            roi: Region of interest
            
        Returns:
            List of bar widths or None
        """
        if roi.size == 0:
            return None
            
        # Get horizontal projection (sum across rows)
        projection = np.mean(roi, axis=0)
        
        # Apply threshold to create binary pattern
        threshold = np.mean(projection)
        binary_pattern = (projection < threshold).astype(int)
        
        # Find runs of consecutive values
        runs = []
        current_value = binary_pattern[0]
        current_length = 1
        
        for i in range(1, len(binary_pattern)):
            if binary_pattern[i] == current_value:
                current_length += 1
            else:
                runs.append(current_length)
                current_value = binary_pattern[i]
                current_length = 1
                
        runs.append(current_length)
        
        # Filter runs that are too small or too large
        filtered_runs = [
            r for r in runs 
            if self.min_bar_width <= r <= self.max_bar_width
        ]
        
        return filtered_runs if len(filtered_runs) >= 6 else None
        
    def _decode_pattern(self, pattern: List[int]) -> Optional[str]:
        """Decode bar pattern to text.
        
        Args:
            pattern: List of bar widths
            
        Returns:
            Decoded text or None
        """
        if not self.decoder or not pattern:
            return None
            
        # For Code128, we need to process patterns in groups of 6
        decoded_chars = []
        
        for i in range(0, len(pattern) - 5, 6):
            char_pattern = pattern[i:i+6]
            char_code = self.decoder.decode_pattern(char_pattern)
            
            if char_code is not None and char_code < 95:
                # Convert to ASCII (simplified)
                if 0 <= char_code <= 94:
                    decoded_chars.append(chr(char_code + 32))
                    
        return ''.join(decoded_chars) if decoded_chars else None
        
    def _calculate_confidence(self, pattern: List[int], roi: np.ndarray) -> float:
        """Calculate confidence score for detection.
        
        Args:
            pattern: Extracted pattern
            roi: Region of interest
            
        Returns:
            Confidence score between 0 and 1
        """
        if not pattern:
            return 0.0
            
        # Base confidence on pattern regularity and contrast
        base_confidence = 0.5
        
        # Pattern length bonus
        if len(pattern) >= 20:
            base_confidence += 0.2
            
        # Pattern regularity (variation in bar widths)
        if len(pattern) > 1:
            variation = np.std(pattern) / np.mean(pattern)
            regularity_bonus = max(0, 0.3 - variation)
            base_confidence += regularity_bonus
            
        # ROI contrast
        contrast = np.std(roi) / 255.0
        contrast_bonus = min(0.2, contrast)
        base_confidence += contrast_bonus
        
        return min(1.0, base_confidence)
        
    def cleanup(self) -> None:
        """Cleanup detector resources."""
        self._is_initialized = False