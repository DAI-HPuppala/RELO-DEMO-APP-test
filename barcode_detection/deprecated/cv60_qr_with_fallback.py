#!/usr/bin/env python3
"""CV60 QR/Barcode Detector with fallback mode for testing."""

import sys
import os
import cv2
import numpy as np
import time
import logging
from typing import Optional, Generator

# Setup paths for eBUS SDK
sys.path.insert(0, '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib')
sys.path.insert(0, '/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/backend')

# Try to import CV60 module
try:
    from cv60_ebus_module.cv60_ebus import CV60Camera
    CV60_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import cv60_ebus_module: {e}")
    CV60_MODULE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try to import pyzbar
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("[WARNING] pyzbar not available - using OpenCV QR detector only")


class FallbackCamera:
    """Fallback camera that generates test patterns with QR codes."""

    def __init__(self):
        self.frame_count = 0
        self.width = 640
        self.height = 480

    def connect(self) -> bool:
        """Simulate connection."""
        return True

    def disconnect(self):
        """Simulate disconnection."""
        pass

    def start_acquisition(self) -> bool:
        """Simulate starting acquisition."""
        return True

    def stop_acquisition(self):
        """Simulate stopping acquisition."""
        pass

    def get_camera_settings(self) -> dict:
        """Return simulated settings."""
        return {
            'Width': self.width,
            'Height': self.height,
            'PixelFormat': 'TestPattern'
        }

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generate test frames with QR codes."""
        qr = cv2.QRCodeWriter() if hasattr(cv2, 'QRCodeWriter') else None

        while True:
            # Create base frame
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200

            # Add gradient
            for i in range(self.height):
                frame[i, :] = [200 - i//4, 200 - i//4, 200 - i//4]

            # Add text
            cv2.putText(frame, "TEST MODE - NO CV60", (150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Add a QR code every 5 frames
            if self.frame_count % 5 == 0 and qr:
                try:
                    # Generate QR code data
                    test_data = f"TEST_QR_{self.frame_count}"
                    qr_img = np.array(qr.create(test_data), dtype=np.uint8) * 255

                    # Resize QR code
                    qr_size = 200
                    qr_resized = cv2.resize(qr_img, (qr_size, qr_size))

                    # Place QR code in frame
                    y_pos = 150
                    x_pos = (self.width - qr_size) // 2
                    if len(qr_resized.shape) == 2:
                        frame[y_pos:y_pos+qr_size, x_pos:x_pos+qr_size] = cv2.cvtColor(
                            qr_resized, cv2.COLOR_GRAY2BGR)
                except:
                    # If QR generation fails, add a simple pattern
                    cv2.rectangle(frame, (200, 150), (440, 390), (0, 0, 0), 2)
                    cv2.putText(frame, f"QR_{self.frame_count}", (250, 270),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Add frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            self.frame_count += 1
            yield frame
            time.sleep(0.1)  # Simulate camera frame rate


class CV60QRDetector:
    """QR/Barcode detector using CV60 camera with fallback mode."""

    def __init__(self, camera_ip: str = "192.168.1.21", use_fallback: bool = False):
        self.camera_ip = camera_ip
        self.use_fallback = use_fallback
        self.camera = None
        self.detected_codes = set()

        # Initialize OpenCV QR detector
        self.qr_detector = cv2.QRCodeDetector()

        # Stats
        self.frame_count = 0
        self.detection_count = 0

    def initialize_camera(self) -> bool:
        """Initialize CV60 camera or fallback."""
        if self.use_fallback:
            print("[INFO] Using fallback test camera")
            self.camera = FallbackCamera()
            return self.camera.connect()

        if not CV60_MODULE_AVAILABLE:
            print("[WARNING] CV60 module not available, using fallback")
            self.camera = FallbackCamera()
            return self.camera.connect()

        try:
            print(f"[INFO] Connecting to CV60 camera at {self.camera_ip}...")
            self.camera = CV60Camera(ip_address=self.camera_ip)

            if self.camera.connect():
                settings = self.camera.get_camera_settings()
                print(f"[SUCCESS] Connected to CV60 camera")
                print(f"[INFO] Resolution: {settings.get('Width')}x{settings.get('Height')}")
                print(f"[INFO] Pixel Format: {settings.get('PixelFormat')}")

                self.camera.start_acquisition()
                return True
            else:
                print(f"[WARNING] Failed to connect to CV60, using fallback")
                self.camera = FallbackCamera()
                return self.camera.connect()

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            print("[INFO] Using fallback test camera")
            self.camera = FallbackCamera()
            return self.camera.connect()

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better QR detection."""
        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        return enhanced

    def detect_barcodes(self, frame: np.ndarray) -> list:
        """Detect barcodes in frame using multiple methods."""
        results = []

        # Preprocess frame
        processed = self.preprocess_frame(frame)

        # Try pyzbar first (if available)
        if PYZBAR_AVAILABLE:
            try:
                decoded = pyzbar.decode(processed)
                for obj in decoded:
                    try:
                        data = obj.data.decode('utf-8')
                    except:
                        data = str(obj.data)

                    results.append({
                        'data': data,
                        'type': str(obj.type),
                        'method': 'pyzbar'
                    })
            except Exception as e:
                logger.debug(f"pyzbar error: {e}")

        # Try OpenCV QR detector
        if not results:  # Only if pyzbar didn't find anything
            try:
                data, bbox, _ = self.qr_detector.detectAndDecode(processed)
                if data:
                    results.append({
                        'data': data,
                        'type': 'QRCODE',
                        'method': 'opencv'
                    })
            except Exception as e:
                logger.debug(f"OpenCV QR error: {e}")

        return results

    def run(self):
        """Main detection loop."""
        if not self.initialize_camera():
            print("[ERROR] Failed to initialize camera")
            return

        print("\n" + "="*60)
        print("CV60 QR/BARCODE DETECTOR")
        print("="*60)
        print(f"Camera: {self.camera_ip if not isinstance(self.camera, FallbackCamera) else 'TEST MODE'}")
        print("Press 'q' or ESC to quit")
        print("Detected barcodes will appear below:")
        print("="*60 + "\n")

        # Create display window
        window_name = "CV60 QR Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            # Stream frames from camera
            for frame in self.camera.stream():
                self.frame_count += 1

                # Detect barcodes
                detections = self.detect_barcodes(frame)

                # Print new detections
                for detection in detections:
                    barcode_value = detection['data']
                    if barcode_value and barcode_value not in self.detected_codes:
                        print(f"[DETECTED] {barcode_value} (method: {detection['method']})")
                        sys.stdout.flush()
                        self.detected_codes.add(barcode_value)
                        self.detection_count += 1

                # Prepare display frame
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                # Add overlay
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Detections: {self.detection_count}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if detections:
                    cv2.putText(display_frame, f"Current: {detections[0]['data'][:30]}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Show frame
                cv2.imshow(window_name, display_frame)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"\n[INFO] Frames processed: {self.frame_count}")
            print(f"[INFO] Barcodes detected: {self.detection_count}")

            # Cleanup
            if self.camera:
                self.camera.stop_acquisition()
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CV60 QR/Barcode Detector')
    parser.add_argument('--ip', default='192.168.1.21',
                       help='CV60 camera IP address')
    parser.add_argument('--fallback', action='store_true',
                       help='Use fallback test mode')
    args = parser.parse_args()

    detector = CV60QRDetector(camera_ip=args.ip, use_fallback=args.fallback)
    detector.run()


if __name__ == "__main__":
    main()