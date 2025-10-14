"""Camera processing for live barcode detection."""

import time
import threading
from typing import Optional, List, Callable, Dict, Any
from queue import Queue, Empty
import numpy as np
import cv2

from ..interfaces.barcode_detector import BarcodeDetector, DetectionResult
from ..core.image_processor import ImageProcessor
from ..core.config_manager import ConfigManager
from ..utils.logger import BarcodeLogger, performance_monitor
from .cv60_camera import create_cv60_camera


class CameraProcessor:
    """Handles live camera feed processing for barcode detection."""
    
    def __init__(
        self, 
        config: ConfigManager, 
        logger: BarcodeLogger,
        image_processor: ImageProcessor
    ) -> None:
        """Initialize camera processor.
        
        Args:
            config: Configuration manager
            logger: Logger instance
            image_processor: Image processor instance
        """
        self.config = config
        self.logger = logger
        self.image_processor = image_processor
        
        # Camera settings
        self.camera_index = config.camera.index
        self.camera_width = config.camera.width
        self.camera_height = config.camera.height
        self.camera_fps = config.camera.fps
        
        # Processing settings
        self.show_bounding_boxes = config.output.show_bounding_boxes
        self.show_detection_time = config.output.show_detection_time
        self.save_annotated = config.output.save_annotated_images
        self.output_dir = config.output.output_directory
        
        # State
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        self.detection_queue = Queue(maxsize=10)
        self.result_callback = None
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture, preferring CV60 camera.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to use CV60 camera first
            self.logger.get_logger("camera").info("Attempting to initialize CV60 camera...")
            self.camera = create_cv60_camera(self.camera_index)

            if not self.camera.isOpened():
                self.logger.get_logger("camera").error(
                    f"Failed to open camera {self.camera_index}"
                )
                return False

            # Set camera properties (works for both CV60 and OpenCV)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)

            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

            # Check if this is CV60 camera
            camera_type = "CV60" if hasattr(self.camera, 'camera_ip') else "OpenCV"

            self.logger.get_logger("camera").info(
                f"{camera_type} camera initialized: {actual_width}x{actual_height}@{actual_fps:.1f}fps"
            )

            return True

        except Exception as e:
            self.logger.log_error(e, "initialize_camera")
            return False
            
    def start_detection(
        self, 
        detector: BarcodeDetector,
        result_callback: Optional[Callable[[List[DetectionResult]], None]] = None
    ) -> bool:
        """Start live detection.
        
        Args:
            detector: Detector to use
            result_callback: Optional callback for detection results
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            self.logger.get_logger("camera").warning("Detection already running")
            return False
            
        if not self.camera or not self.camera.isOpened():
            self.logger.get_logger("camera").error("Camera not initialized")
            return False
            
        self.detector = detector
        self.result_callback = result_callback
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.get_logger("camera").info("Live detection started")
        return True
        
    def stop_detection(self) -> None:
        """Stop live detection."""
        self.is_running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        self.logger.get_logger("camera").info("Live detection stopped")
        
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from camera.
        
        Returns:
            Frame as numpy array or None if failed
        """
        if not self.camera or not self.camera.isOpened():
            return None
            
        ret, frame = self.camera.read()
        return frame if ret else None
        
    @performance_monitor("process_frame")
    def process_frame(
        self, 
        frame: np.ndarray, 
        detector: BarcodeDetector
    ) -> tuple[np.ndarray, List[DetectionResult]]:
        """Process a single frame for barcode detection.
        
        Args:
            frame: Input frame
            detector: Detector to use
            
        Returns:
            Tuple of (annotated_frame, detection_results)
        """
        # Detect barcodes
        results = self.image_processor.process_image(
            frame, 
            detector, 
            source_info="camera"
        )
        
        # Annotate frame
        annotated_frame = self.image_processor.annotate_image(
            frame,
            results,
            show_bounding_boxes=self.show_bounding_boxes,
            show_detection_time=self.show_detection_time,
            show_data=True
        )
        
        # Add FPS and statistics overlay
        annotated_frame = self._add_overlay_info(annotated_frame, results)
        
        # Update statistics
        self.frame_count += 1
        if results:
            self.detection_count += 1
            
        return annotated_frame, results
        
    def _processing_loop(self) -> None:
        """Main processing loop for live detection."""
        self.logger.get_logger("camera").info("Processing loop started")
        
        frame_skip = 0
        skip_frames = max(1, self.camera_fps // 10)  # Process ~10 FPS max
        
        while self.is_running:
            try:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                    
                # Skip frames for performance
                frame_skip += 1
                if frame_skip % skip_frames != 0:
                    continue
                    
                # Process frame
                annotated_frame, results = self.process_frame(frame, self.detector)
                
                # Add to display queue (non-blocking)
                try:
                    self.detection_queue.put_nowait({
                        'frame': annotated_frame,
                        'results': results,
                        'timestamp': time.time()
                    })
                except:
                    # Queue full, skip this frame
                    pass
                    
                # Call result callback if provided
                if self.result_callback and results:
                    try:
                        self.result_callback(results)
                    except Exception as e:
                        self.logger.log_error(e, "result_callback")
                        
                # Save annotated frame if enabled
                if self.save_annotated and results:
                    self._save_annotated_frame(annotated_frame)
                    
            except Exception as e:
                self.logger.log_error(e, "processing_loop")
                time.sleep(0.1)  # Prevent busy loop on errors
                
        self.logger.get_logger("camera").info("Processing loop ended")
        
    def get_latest_frame(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get latest processed frame from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with frame data or None
        """
        try:
            return self.detection_queue.get(timeout=timeout)
        except Empty:
            return None
            
    def display_live_feed(self, window_name: str = "Barcode Detection") -> None:
        """Display live feed with detection results.
        
        Args:
            window_name: OpenCV window name
        """
        self.logger.get_logger("camera").info(f"Starting live display: {window_name}")
        
        # Create window
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.is_running:
                # Get latest frame
                frame_data = self.get_latest_frame(timeout=0.033)  # ~30 FPS display
                
                if frame_data:
                    frame = frame_data['frame']
                    results = frame_data['results']
                    
                    # Display frame
                    cv2.imshow(window_name, frame)
                    
                    # Log detections
                    if results:
                        for result in results:
                            self.logger.get_logger("camera").info(
                                f"Detected: {result.type} = '{result.data}' "
                                f"(confidence: {result.confidence:.2f})"
                            )
                            
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.logger.get_logger("camera").info("Exit key pressed")
                    break
                elif key == ord('s'):  # 's' to save screenshot
                    if frame_data:
                        self._save_screenshot(frame_data['frame'])
                        
        except KeyboardInterrupt:
            self.logger.get_logger("camera").info("Interrupted by user")
        finally:
            cv2.destroyWindow(window_name)
            self.logger.get_logger("camera").info("Live display ended")
            
    def _add_overlay_info(
        self, 
        frame: np.ndarray, 
        results: List[DetectionResult]
    ) -> np.ndarray:
        """Add overlay information to frame.
        
        Args:
            frame: Input frame
            results: Detection results
            
        Returns:
            Frame with overlay
        """
        # Calculate FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
        else:
            self.fps_counter += 1
            
        # Add overlay text
        overlay_y = frame.shape[0] - 60
        
        # FPS
        fps_text = f"FPS: {getattr(self, 'fps', 0)}"
        cv2.putText(frame, fps_text, (10, overlay_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        # Frame count
        frame_text = f"Frames: {self.frame_count}"
        cv2.putText(frame, frame_text, (10, overlay_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        # Detection count
        detection_text = f"Detections: {self.detection_count}"
        cv2.putText(frame, detection_text, (10, overlay_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        # Controls info
        controls_text = "Press 'q' to quit, 's' to save screenshot"
        cv2.putText(frame, controls_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                   
        return frame
        
    def _save_annotated_frame(self, frame: np.ndarray) -> None:
        """Save annotated frame to disk.
        
        Args:
            frame: Annotated frame to save
        """
        try:
            from pathlib import Path
            
            output_path = Path(self.output_dir) / "camera_captures"
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}_{self.detection_count:04d}.jpg"
            file_path = output_path / filename
            
            success = cv2.imwrite(str(file_path), frame)
            if success:
                self.logger.get_logger("camera").debug(f"Saved frame: {file_path}")
                
        except Exception as e:
            self.logger.log_error(e, "save_annotated_frame")
            
    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Save screenshot manually triggered.
        
        Args:
            frame: Frame to save
        """
        try:
            from pathlib import Path
            
            output_path = Path(self.output_dir) / "screenshots"
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            file_path = output_path / filename
            
            success = cv2.imwrite(str(file_path), frame)
            if success:
                self.logger.get_logger("camera").info(f"Screenshot saved: {file_path}")
                
        except Exception as e:
            self.logger.log_error(e, "save_screenshot")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "fps": getattr(self, 'fps', 0),
            "is_running": self.is_running,
            "camera_initialized": self.camera is not None and self.camera.isOpened()
        }
        
    def cleanup(self) -> None:
        """Cleanup camera resources."""
        self.stop_detection()
        
        if self.camera:
            self.camera.release()
            self.camera = None
            
        cv2.destroyAllWindows()
        
        self.logger.get_logger("camera").info("Camera processor cleaned up")
        
    def __enter__(self):
        """Context manager entry."""
        self.initialize_camera()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()