"""Image processing pipeline for barcode detection."""

import time
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

from ..interfaces.barcode_detector import BarcodeDetector, DetectionResult
from ..utils.logger import BarcodeLogger, performance_monitor, log_function_call


class ImageProcessor:
    """Handles image loading, processing, and result annotation."""
    
    def __init__(self, logger: BarcodeLogger) -> None:
        """Initialize image processor.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
        
    @performance_monitor("load_image")
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (BGR format) or None if failed
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                self.logger.get_logger("processor").error(f"Image file not found: {image_path}")
                return None
                
            # Try OpenCV first
            image = cv2.imread(str(image_path))
            
            if image is not None:
                return image
                
            # Fallback to PIL
            try:
                pil_image = Image.open(image_path)
                # Convert to RGB then to BGR for OpenCV compatibility
                rgb_image = np.array(pil_image.convert('RGB'))
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                return bgr_image
                
            except Exception as e:
                self.logger.get_logger("processor").error(
                    f"Failed to load image {image_path}: {e}"
                )
                return None
                
        except Exception as e:
            self.logger.log_error(e, f"load_image({image_path})")
            return None
            
    @performance_monitor("process_image")
    def process_image(
        self, 
        image: np.ndarray, 
        detector: BarcodeDetector,
        source_info: Optional[str] = None
    ) -> List[DetectionResult]:
        """Process image for barcode detection.
        
        Args:
            image: Input image
            detector: Detector to use
            source_info: Optional source information for logging
            
        Returns:
            List of detection results
        """
        if image is None or image.size == 0:
            return []
            
        source = source_info or "unknown"
        
        # Log detection start
        self.logger.log_detection_start(source, detector.name)
        
        start_time = time.perf_counter()
        
        try:
            # Detect barcodes
            results = detector.detect(image)
            
            end_time = time.perf_counter()
            processing_time = (end_time - start_time) * 1000
            
            # Log results
            self.logger.log_detection_result(
                success=len(results) > 0,
                barcodes_found=len(results),
                processing_time=processing_time,
                algorithm=detector.name
            )
            
            return results
            
        except Exception as e:
            end_time = time.perf_counter()
            processing_time = (end_time - start_time) * 1000
            
            self.logger.log_error(e, f"process_image({source})")
            self.logger.log_detection_result(
                success=False,
                barcodes_found=0,
                processing_time=processing_time,
                algorithm=detector.name
            )
            
            return []
            
    @log_function_call
    def annotate_image(
        self, 
        image: np.ndarray, 
        results: List[DetectionResult],
        show_bounding_boxes: bool = True,
        show_detection_time: bool = True,
        show_data: bool = True
    ) -> np.ndarray:
        """Annotate image with detection results.
        
        Args:
            image: Input image
            results: Detection results
            show_bounding_boxes: Whether to draw bounding boxes
            show_detection_time: Whether to show processing time
            show_data: Whether to show detected data
            
        Returns:
            Annotated image
        """
        if not results:
            return image.copy()
            
        annotated = image.copy()
        
        for i, result in enumerate(results):
            x, y, w, h = result.bounding_box
            
            # Choose color based on confidence
            if result.confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif result.confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
                
            if show_bounding_boxes:
                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                
                # Draw corner markers
                corner_length = 20
                thickness = 3
                
                # Top-left corner
                cv2.line(annotated, (x, y), (x + corner_length, y), color, thickness)
                cv2.line(annotated, (x, y), (x, y + corner_length), color, thickness)
                
                # Top-right corner
                cv2.line(annotated, (x + w, y), (x + w - corner_length, y), color, thickness)
                cv2.line(annotated, (x + w, y), (x + w, y + corner_length), color, thickness)
                
                # Bottom-left corner
                cv2.line(annotated, (x, y + h), (x + corner_length, y + h), color, thickness)
                cv2.line(annotated, (x, y + h), (x, y + h - corner_length), color, thickness)
                
                # Bottom-right corner
                cv2.line(annotated, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
                cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
                
            # Text annotations
            text_y = max(y - 10, 30)  # Position text above bbox or at top
            
            if show_data:
                # Show barcode data
                data_text = f"Data: {result.data[:30]}..." if len(result.data) > 30 else f"Data: {result.data}"
                cv2.putText(
                    annotated, 
                    data_text, 
                    (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    color, 
                    2
                )
                text_y += 25
                
                # Show type and confidence
                info_text = f"Type: {result.type}, Conf: {result.confidence:.2f}"
                cv2.putText(
                    annotated, 
                    info_text, 
                    (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    1
                )
                text_y += 20
                
            if show_detection_time:
                # Show processing time
                time_text = f"Time: {result.processing_time_ms:.1f}ms"
                cv2.putText(
                    annotated, 
                    time_text, 
                    (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    color, 
                    1
                )
                
        # Add summary information
        if results:
            summary_text = f"Found {len(results)} barcode(s)"
            cv2.putText(
                annotated, 
                summary_text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
            
        return annotated
        
    @performance_monitor("save_image")
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> bool:
        """Save image to file.
        
        Args:
            image: Image to save
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = cv2.imwrite(str(output_path), image)
            
            if success:
                self.logger.get_logger("processor").info(f"Image saved: {output_path}")
                return True
            else:
                self.logger.get_logger("processor").error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            self.logger.log_error(e, f"save_image({output_path})")
            return False
            
    def resize_image(
        self, 
        image: np.ndarray, 
        max_width: int = 1920, 
        max_height: int = 1080
    ) -> np.ndarray:
        """Resize image if it's too large.
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        if image is None:
            return image
            
        height, width = image.shape[:2]
        
        if width <= max_width and height <= max_height:
            return image
            
        # Calculate scale factor
        width_scale = max_width / width
        height_scale = max_height / height
        scale = min(width_scale, height_scale)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        self.logger.get_logger("processor").info(
            f"Image resized from {width}x{height} to {new_width}x{new_height}"
        )
        
        return resized
        
    def enhance_image_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Enhance image to improve barcode detection.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        if image is None:
            return image
            
        enhanced = image.copy()
        
        # Convert to grayscale for processing
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced
            
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(equalized, -1, kernel)
        
        # Convert back to BGR if original was color
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = sharpened
            
        return enhanced
        
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Get information about an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image information
        """
        if image is None:
            return {}
            
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_mb": (image.nbytes / 1024 / 1024),
            "aspect_ratio": width / height if height > 0 else 0
        }