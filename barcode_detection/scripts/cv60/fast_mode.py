#!/usr/bin/env python3
"""Fast CV60 QR/Barcode Detector optimized for 30fps display."""

import sys
import os
import cv2
import numpy as np
import time
import threading
import logging
from queue import Queue, Empty
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Setup paths for eBUS SDK
sys.path.insert(0, '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib')
# Add project src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, '/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/backend')

# Import CV60 module
try:
    from cv60_ebus_module.cv60_ebus import CV60Camera
    CV60_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import cv60_ebus_module: {e}")
    CV60_MODULE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/qr_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import detection libraries
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    logger.info("pyzbar library loaded")
except ImportError:
    logger.warning("pyzbar not available")


@dataclass
class DetectionResult:
    """Container for detection results."""
    data: str
    type: str
    bbox: Any
    timestamp: float
    frame_number: int


class FastQRDetector:
    """Fast QR/Barcode detector optimized for real-time display."""

    def __init__(self, camera_ip: str = "192.168.1.21"):
        self.camera_ip = camera_ip
        self.camera = None

        # Detection settings
        self.detect_interval = 5  # Detect every N frames
        self.use_simple_detection = True  # Use only fastest method

        # Threading components
        self.frame_queue = Queue(maxsize=2)  # Small queue for latest frames
        self.detection_queue = Queue(maxsize=10)
        self.detection_thread = None
        self.display_thread = None
        self.stop_threads = False

        # Detection components
        self.qr_detector = cv2.QRCodeDetector()

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.detected_codes = set()
        self.fps_counter = []
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Current detection result for display
        self.current_detection = None
        self.detection_display_duration = 2.0  # Show detection for 2 seconds

    def initialize_camera(self) -> bool:
        """Initialize CV60 camera."""
        if not CV60_MODULE_AVAILABLE:
            logger.error("CV60 module not available")
            return False

        try:
            logger.info(f"Connecting to CV60 camera at {self.camera_ip}...")
            self.camera = CV60Camera(ip_address=self.camera_ip)

            if self.camera.connect():
                settings = self.camera.get_camera_settings()
                logger.info(f"Connected to CV60 camera")
                logger.info(f"Resolution: {settings.get('Width')}x{settings.get('Height')}")
                logger.info(f"Pixel Format: {settings.get('PixelFormat')}")

                # Start acquisition
                if self.camera.start_acquisition():
                    logger.info("Camera acquisition started")
                    return True
                else:
                    logger.error("Failed to start acquisition")
                    self.camera.disconnect()
                    return False
            else:
                logger.error(f"Failed to connect to CV60 at {self.camera_ip}")
                return False

        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    def detect_fast(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Fast detection using only the most efficient method."""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Try pyzbar first (fastest and most reliable)
            if PYZBAR_AVAILABLE:
                decoded = pyzbar.decode(gray)
                if decoded:
                    obj = decoded[0]  # Take first detection
                    data = obj.data.decode('utf-8', errors='ignore')
                    return DetectionResult(
                        data=data,
                        type=str(obj.type),
                        bbox=obj.rect,
                        timestamp=time.time(),
                        frame_number=self.frame_count
                    )

            # Fallback to OpenCV QR detector
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)
            if data:
                return DetectionResult(
                    data=data,
                    type='QRCODE',
                    bbox=bbox,
                    timestamp=time.time(),
                    frame_number=self.frame_count
                )

        except Exception as e:
            logger.debug(f"Detection error: {e}")

        return None

    def detection_worker(self):
        """Worker thread for detection processing."""
        logger.info("Detection worker started")
        frame_skip_counter = 0

        while not self.stop_threads:
            try:
                # Get frame from detection queue (with timeout to check stop flag)
                frame_data = self.detection_queue.get(timeout=0.1)

                # Perform detection
                result = self.detect_fast(frame_data['frame'])

                if result and result.data not in self.detected_codes:
                    self.detected_codes.add(result.data)
                    self.detection_count += 1
                    self.current_detection = result

                    # Log detection
                    print(f"\n[DETECTED] {result.data}")
                    print(f"  Type: {result.type}")
                    print(f"  Frame: {result.frame_number}")
                    sys.stdout.flush()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Detection worker error: {e}")

        logger.info("Detection worker stopped")

    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time)

        # Keep only last second of timestamps
        self.fps_counter = [t for t in self.fps_counter if current_time - t < 1.0]

        if len(self.fps_counter) > 1:
            self.current_fps = len(self.fps_counter)

        return self.current_fps

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay information on frame."""
        # Calculate FPS
        fps = self.calculate_fps()

        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw detection count
        cv2.putText(frame, f"Detections: {self.detection_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw current detection if recent
        if self.current_detection:
            age = time.time() - self.current_detection.timestamp
            if age < self.detection_display_duration:
                # Draw detection info
                alpha = max(0, 1 - (age / self.detection_display_duration))
                color = (0, int(255 * alpha), int(255 * alpha))

                cv2.putText(frame, f"DETECTED: {self.current_detection.data[:50]}",
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, color, 2)

                # Draw bounding box if available
                if self.current_detection.bbox is not None:
                    bbox = self.current_detection.bbox
                    if isinstance(bbox, tuple) and len(bbox) == 4:
                        x, y, w, h = [int(v) for v in bbox]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw status
        status_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"Detection interval: 1/{self.detect_interval} frames",
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        return frame

    def run(self):
        """Main execution loop with optimized frame handling."""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return

        print("\n" + "="*60)
        print("CV60 FAST QR/BARCODE DETECTOR (30 FPS)")
        print("="*60)
        print(f"Camera IP: {self.camera_ip}")
        print(f"Detection interval: Every {self.detect_interval} frames")
        print(f"Target FPS: 30")
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  '+' - Increase detection frequency")
        print("  '-' - Decrease detection frequency")
        print("  's' - Save current frame")
        print("="*60 + "\n")

        # Create display window
        window_name = "CV60 Fast QR Detection - 30 FPS"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        # Start detection worker thread
        self.detection_thread = threading.Thread(target=self.detection_worker, daemon=True)
        self.detection_thread.start()

        try:
            frame_time = 1.0 / 30.0  # Target 30 FPS
            last_frame_time = time.time()
            detection_counter = 0

            # Main display loop
            for frame in self.camera.stream():
                current_time = time.time()

                # Increment counters
                self.frame_count += 1
                detection_counter += 1

                # Send frame for detection (non-blocking) at intervals
                if detection_counter >= self.detect_interval:
                    # Only add to detection queue if it's not full
                    try:
                        self.detection_queue.put_nowait({
                            'frame': frame.copy(),
                            'frame_number': self.frame_count
                        })
                        detection_counter = 0
                    except:
                        pass  # Queue full, skip this detection

                # Prepare display frame
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                # Draw overlay
                display_frame = self.draw_overlay(display_frame)

                # Show frame immediately (no waiting for detection)
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Quit
                    break
                elif key == ord('+'):  # Increase detection frequency
                    self.detect_interval = max(1, self.detect_interval - 1)
                    print(f"Detection interval: 1/{self.detect_interval} frames")
                elif key == ord('-'):  # Decrease detection frequency
                    self.detect_interval = min(30, self.detect_interval + 1)
                    print(f"Detection interval: 1/{self.detect_interval} frames")
                elif key == ord('s'):  # Save frame
                    filename = f"captured_frame_{self.frame_count}.png"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")

                # Maintain target frame rate
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_frame_time = time.time()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Stop threads
            self.stop_threads = True
            if self.detection_thread:
                self.detection_thread.join(timeout=1)

            # Print summary
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Frames processed: {self.frame_count}")
            print(f"Average FPS: {self.current_fps:.1f}")
            print(f"Total detections: {self.detection_count}")
            print(f"Unique codes: {len(self.detected_codes)}")

            if self.detected_codes:
                print("\nDetected codes:")
                for code in self.detected_codes:
                    print(f"  - {code}")

            print("="*60)

            # Cleanup
            if self.camera:
                self.camera.stop_acquisition()
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Fast CV60 QR/Barcode Detector (30 FPS)')
    parser.add_argument('--ip', default='192.168.1.21',
                       help='CV60 camera IP address')
    parser.add_argument('--interval', type=int, default=5,
                       help='Detection interval (detect every N frames)')
    args = parser.parse_args()

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    detector = FastQRDetector(camera_ip=args.ip)
    detector.detect_interval = args.interval
    detector.run()


if __name__ == "__main__":
    main()