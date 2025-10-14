"""Image preprocessing pipeline for barcode detection."""

import cv2
import numpy as np
from typing import Dict, Callable, List, Tuple


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        frame: Input grayscale image
        clip_limit: Clipping limit for contrast
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE-processed image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(frame)


def apply_sharpening(frame: np.ndarray) -> np.ndarray:
    """Apply sharpening kernel to enhance edges.

    Args:
        frame: Input grayscale image

    Returns:
        Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)


def apply_bilateral_filter(frame: np.ndarray, d: int = 11, sigma_color: int = 17, sigma_space: int = 17) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing.

    Args:
        frame: Input grayscale image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Filtered image
    """
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)


def apply_binary_threshold(frame: np.ndarray, thresh_value: int = 127) -> np.ndarray:
    """Apply binary threshold.

    Args:
        frame: Input grayscale image
        thresh_value: Threshold value

    Returns:
        Binary image
    """
    _, binary = cv2.threshold(frame, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def apply_adaptive_threshold(frame: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Apply adaptive threshold.

    Args:
        frame: Input grayscale image
        block_size: Size of pixel neighborhood
        C: Constant subtracted from weighted mean

    Returns:
        Binary image
    """
    return cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)


class PreprocessingPipeline:
    """Preprocessing pipeline for barcode detection.

    This class provides a flexible pipeline for applying multiple
    preprocessing filters to images for optimal barcode detection.

    Example:
        ```python
        from barcode_detector.preprocessing import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        pipeline.add_step('grayscale', pipeline.to_grayscale)
        pipeline.add_step('clahe', lambda f: apply_clahe(f))
        pipeline.add_step('sharpened', lambda f: apply_sharpening(f))

        processed_images = pipeline.process(frame)
        # Returns dict: {'grayscale': ..., 'clahe': ..., 'sharpened': ...}
        ```
    """

    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.steps: List[Tuple[str, Callable]] = []

    def add_step(self, name: str, func: Callable[[np.ndarray], np.ndarray]) -> 'PreprocessingPipeline':
        """Add a preprocessing step.

        Args:
            name: Name of the step
            func: Function that takes and returns a numpy array

        Returns:
            Self for method chaining
        """
        self.steps.append((name, func))
        return self

    def clear(self) -> 'PreprocessingPipeline':
        """Clear all steps.

        Returns:
            Self for method chaining
        """
        self.steps.clear()
        return self

    def process(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Process frame through all steps.

        Args:
            frame: Input frame

        Returns:
            Dictionary mapping step names to processed images
        """
        results = {}
        current = frame.copy()

        for name, func in self.steps:
            try:
                current = func(current)
                results[name] = current
            except Exception as e:
                # Skip failed steps
                continue

        return results

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if needed.

        Args:
            frame: Input frame

        Returns:
            Grayscale image
        """
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame.copy()

    @staticmethod
    def create_default_pipeline() -> 'PreprocessingPipeline':
        """Create a default preprocessing pipeline for barcode detection.

        Returns:
            PreprocessingPipeline configured with common preprocessing steps
        """
        pipeline = PreprocessingPipeline()

        # Add standard preprocessing steps
        pipeline.add_step('original', lambda f: f.copy())
        pipeline.add_step('grayscale', PreprocessingPipeline.to_grayscale)
        pipeline.add_step('clahe', lambda f: apply_clahe(PreprocessingPipeline.to_grayscale(f)))
        pipeline.add_step('bilateral', lambda f: apply_bilateral_filter(PreprocessingPipeline.to_grayscale(f)))
        pipeline.add_step('binary', lambda f: apply_binary_threshold(PreprocessingPipeline.to_grayscale(f)))
        pipeline.add_step('adaptive', lambda f: apply_adaptive_threshold(PreprocessingPipeline.to_grayscale(f)))
        pipeline.add_step('sharpened', lambda f: apply_sharpening(PreprocessingPipeline.to_grayscale(f)))

        return pipeline

    @staticmethod
    def create_barcode_pipeline() -> 'PreprocessingPipeline':
        """Create a pipeline optimized for linear barcodes.

        Returns:
            PreprocessingPipeline configured for barcode detection
        """
        pipeline = PreprocessingPipeline()

        def to_gray(f):
            return PreprocessingPipeline.to_grayscale(f)

        # Optimized for barcodes
        pipeline.add_step('original', lambda f: to_gray(f))
        pipeline.add_step('sharpened', lambda f: apply_sharpening(to_gray(f)))
        pipeline.add_step('bilateral', lambda f: apply_bilateral_filter(to_gray(f), 5, 50, 50))
        pipeline.add_step('clahe', lambda f: apply_clahe(to_gray(f), 3.0, (8, 8)))
        pipeline.add_step('binary_otsu', lambda f: cv2.threshold(to_gray(f), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        pipeline.add_step('adaptive', lambda f: apply_adaptive_threshold(to_gray(f), 31, 5))

        return pipeline


__all__ = [
    'PreprocessingPipeline',
    'apply_clahe',
    'apply_sharpening',
    'apply_bilateral_filter',
    'apply_binary_threshold',
    'apply_adaptive_threshold',
]
