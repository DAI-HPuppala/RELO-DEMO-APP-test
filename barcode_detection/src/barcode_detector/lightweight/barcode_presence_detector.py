"""Lightweight barcode presence detector using gradient analysis.

This detector uses fast computer vision techniques to identify
barcode-like patterns without actually decoding them. Ideal for
filtering frames before running heavy detection algorithms.

Technique:
- Sobel gradient in X-direction to find vertical edges (barcode bars)
- Morphological operations to find rectangular regions
- Variance analysis to confirm regular bar patterns
- ~2-5ms per frame on modern CPU
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BarcodeRegion:
    """Detected barcode-like region."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float  # 0.0 to 1.0
    bar_pattern_score: float  # Quality of bar pattern
    orientation: str  # 'horizontal' or 'vertical'


class BarcodePresenceDetector:
    """Fast barcode presence detector using gradient analysis.

    This detector is designed to be extremely fast (~2-5ms per frame)
    and can be run on every frame to detect barcode presence before
    running slower decoding algorithms.

    Args:
        min_bar_width: Minimum expected bar width in pixels (default: 2)
        max_bar_width: Maximum expected bar width in pixels (default: 10)
        min_bars: Minimum number of bars to consider a barcode (default: 10)
        confidence_threshold: Minimum confidence to report detection (default: 0.5)

    Example:
        >>> detector = BarcodePresenceDetector()
        >>> frame = cv2.imread('image.jpg')
        >>> regions = detector.detect(frame)
        >>> if regions:
        >>>     print(f"Found {len(regions)} potential barcodes")
        >>>     # Now run heavy decoder only on these frames
    """

    def __init__(self,
                 min_bar_width: int = 2,
                 max_bar_width: int = 10,
                 min_bars: int = 10,
                 confidence_threshold: float = 0.5):
        self.min_bar_width = min_bar_width
        self.max_bar_width = max_bar_width
        self.min_bars = min_bars
        self.confidence_threshold = confidence_threshold

        # Performance tuning
        self.resize_width = 640  # Resize for faster processing
        self.use_clahe = False  # Disable by default for speed

    def detect(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[BarcodeRegion]:
        """Detect barcode-like regions in frame.

        Args:
            frame: Input image (BGR or grayscale)
            roi: Optional region of interest (x, y, w, h) to search within

        Returns:
            List of BarcodeRegion objects with confidence scores
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Extract ROI if specified
        original_shape = gray.shape
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
            roi_offset = (x, y)
        else:
            roi_offset = (0, 0)

        # Resize for faster processing
        h, w = gray.shape
        if w > self.resize_width:
            scale = self.resize_width / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        # Detect horizontal barcodes
        horizontal_regions = self._detect_orientation(gray, 'horizontal')

        # Detect vertical barcodes (rotated)
        vertical_regions = self._detect_orientation(gray, 'vertical')

        # Combine and scale back
        all_regions = horizontal_regions + vertical_regions

        # Scale bounding boxes back to original size and add ROI offset
        scaled_regions = []
        for region in all_regions:
            x, y, rw, rh = region.bbox
            x = int(x / scale) + roi_offset[0]
            y = int(y / scale) + roi_offset[1]
            rw = int(rw / scale)
            rh = int(rh / scale)

            scaled_regions.append(BarcodeRegion(
                bbox=(x, y, rw, rh),
                confidence=region.confidence,
                bar_pattern_score=region.bar_pattern_score,
                orientation=region.orientation
            ))

        # Filter by confidence
        filtered = [r for r in scaled_regions if r.confidence >= self.confidence_threshold]

        return filtered

    def _detect_orientation(self, gray: np.ndarray, orientation: str) -> List[BarcodeRegion]:
        """Detect barcodes in specific orientation."""
        regions = []

        # Apply CLAHE if enabled (slower but better for low contrast)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Compute gradient
        if orientation == 'horizontal':
            # For horizontal barcodes, detect vertical bars (X gradient)
            gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        else:
            # For vertical barcodes, detect horizontal bars (Y gradient)
            gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Absolute gradient
        gradient = np.abs(gradient)
        gradient = np.uint8(np.clip(gradient, 0, 255))

        # Threshold gradient
        _, gradient_thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to connect bars
        if orientation == 'horizontal':
            # Close gaps horizontally, keep vertical structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        else:
            # Close gaps vertically, keep horizontal structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))

        closed = cv2.morphologyEx(gradient_thresh, cv2.MORPH_CLOSE, kernel)

        # Dilate to merge nearby regions
        if orientation == 'horizontal':
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        else:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))

        dilated = cv2.dilate(closed, dilate_kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze each contour
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filters
            if orientation == 'horizontal':
                # Horizontal barcodes are wide
                if w < 50 or h < 20 or w / h < 1.5:
                    continue
            else:
                # Vertical barcodes are tall
                if h < 50 or w < 20 or h / w < 1.5:
                    continue

            # Extract region
            roi = gray[y:y+h, x:x+w]

            # Analyze bar pattern
            bar_score, confidence = self._analyze_bar_pattern(roi, orientation)

            if confidence > 0:
                regions.append(BarcodeRegion(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    bar_pattern_score=bar_score,
                    orientation=orientation
                ))

        return regions

    def _analyze_bar_pattern(self, roi: np.ndarray, orientation: str) -> Tuple[float, float]:
        """Analyze if region contains barcode-like bar pattern.

        Returns:
            (bar_pattern_score, confidence)
        """
        if roi.size == 0:
            return 0.0, 0.0

        h, w = roi.shape

        # Get profile (average intensity across bars)
        if orientation == 'horizontal':
            # For horizontal barcode, average vertically
            profile = np.mean(roi, axis=0)
        else:
            # For vertical barcode, average horizontally
            profile = np.mean(roi, axis=1)

        if len(profile) < self.min_bars * 2:
            return 0.0, 0.0

        # Find transitions (bar edges)
        # Normalize profile
        profile = profile - profile.min()
        if profile.max() > 0:
            profile = profile / profile.max()

        # Detect edges in profile
        profile_grad = np.abs(np.diff(profile))

        # Count significant transitions
        threshold = np.mean(profile_grad) + np.std(profile_grad)
        transitions = np.where(profile_grad > threshold)[0]

        num_transitions = len(transitions)

        # Barcodes have many transitions
        if num_transitions < self.min_bars:
            return 0.0, 0.0

        # Check for regularity (barcodes have somewhat regular spacing)
        if num_transitions >= 2:
            # Compute spacing between transitions
            spacings = np.diff(transitions)

            # Good barcodes have varied but reasonable spacing
            if len(spacings) > 0:
                mean_spacing = np.mean(spacings)
                std_spacing = np.std(spacings)

                # Filter invalid spacings
                if mean_spacing < self.min_bar_width or mean_spacing > self.max_bar_width * 2:
                    return 0.0, 0.0

                # Coefficient of variation (lower is more regular)
                if mean_spacing > 0:
                    regularity = 1.0 - min(std_spacing / mean_spacing, 1.0)
                else:
                    regularity = 0.0
            else:
                regularity = 0.0
        else:
            regularity = 0.0

        # Check contrast (barcodes have high local contrast)
        contrast = np.std(roi) / 128.0  # Normalize to 0-1
        contrast = min(contrast, 1.0)

        # Compute bar pattern score
        bar_score = (
            0.4 * min(num_transitions / 30.0, 1.0) +  # More transitions = better
            0.3 * regularity +  # More regular = better
            0.3 * contrast  # Higher contrast = better
        )

        # Compute confidence
        confidence = bar_score

        # Boost confidence if many transitions found
        if num_transitions > 20:
            confidence = min(confidence * 1.2, 1.0)

        return bar_score, confidence

    def visualize_detection(self, frame: np.ndarray, regions: List[BarcodeRegion]) -> np.ndarray:
        """Draw detected regions on frame for debugging.

        Args:
            frame: Original frame
            regions: List of detected regions

        Returns:
            Annotated frame
        """
        display = frame.copy()

        for region in regions:
            x, y, w, h = region.bbox

            # Color based on confidence
            if region.confidence > 0.7:
                color = (0, 255, 0)  # Green - high confidence
            elif region.confidence > 0.5:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 128, 255)  # Orange - low confidence

            # Draw bounding box
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)

            # Draw label
            label = f"{region.orientation[0].upper()}: {region.confidence:.2f}"
            cv2.putText(display, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return display
