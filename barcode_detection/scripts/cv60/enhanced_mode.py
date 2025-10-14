#!/usr/bin/env python3
"""Enhanced CV60 QR/Barcode Detector with multiple detection methods."""

import sys
import os
import cv2
import numpy as np
import time
import logging
from typing import Optional, List, Dict, Any

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
        logging.FileHandler('logs/qr_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import detection libraries
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    logger.info("pyzbar library loaded successfully")
except ImportError as e:
    logger.warning(f"pyzbar not available: {e}")

# Check for OpenCV contrib modules
HAS_WECHAT_QRCODE = hasattr(cv2, 'wechat_qrcode_WeChatQRCode')
if HAS_WECHAT_QRCODE:
    logger.info("WeChat QRCode detector available")

# Check for additional OpenCV barcode detectors
HAS_BARCODE_DETECTOR = hasattr(cv2, 'barcode_BarcodeDetector')
if HAS_BARCODE_DETECTOR:
    logger.info("OpenCV Barcode detector available")


class MultiMethodDetector:
    """Multi-method barcode/QR code detector."""

    def __init__(self):
        # Initialize OpenCV QR detector
        self.qr_detector = cv2.QRCodeDetector()

        # Initialize QRCodeDetectorAruco if available (better performance)
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            try:
                self.qr_aruco = cv2.QRCodeDetectorAruco()
                logger.info("QRCodeDetectorAruco initialized")
            except:
                self.qr_aruco = None
        else:
            self.qr_aruco = None

        # Initialize WeChat QRCode detector if available
        self.wechat_detector = None
        if HAS_WECHAT_QRCODE:
            try:
                # WeChat QRCode requires model files, try without them first
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                logger.info("WeChat QRCode detector initialized")
            except:
                logger.warning("WeChat QRCode detector initialization failed")

        # Initialize OpenCV Barcode detector if available
        self.barcode_detector = None
        if HAS_BARCODE_DETECTOR:
            try:
                self.barcode_detector = cv2.barcode_BarcodeDetector()
                logger.info("OpenCV Barcode detector initialized")
            except:
                logger.warning("OpenCV Barcode detector initialization failed")

    def preprocess_for_detection(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocess image with multiple techniques for better detection."""
        processed_images = []

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Original grayscale
        processed_images.append(gray)

        # Enhanced contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed_images.append(enhanced)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, 10
        )
        processed_images.append(thresh)

        # Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(binary)

        # Sharp filter
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        processed_images.append(sharpened)

        return processed_images

    def detect_with_pyzbar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using pyzbar library."""
        results = []
        if not PYZBAR_AVAILABLE:
            return results

        try:
            decoded = pyzbar.decode(image)
            for obj in decoded:
                data = obj.data.decode('utf-8', errors='ignore')
                barcode_type = str(obj.type)

                # Get bounding box
                points = obj.polygon
                if points:
                    x_coords = [p.x for p in points]
                    y_coords = [p.y for p in points]
                    bbox = (min(x_coords), min(y_coords),
                           max(x_coords) - min(x_coords),
                           max(y_coords) - min(y_coords))
                else:
                    bbox = obj.rect

                results.append({
                    'data': data,
                    'type': barcode_type,
                    'bbox': bbox,
                    'method': 'pyzbar',
                    'confidence': 1.0
                })
        except Exception as e:
            logger.debug(f"pyzbar detection error: {e}")

        return results

    def detect_with_opencv_qr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using OpenCV QRCodeDetector."""
        results = []

        try:
            # Standard QRCodeDetector
            data, bbox, rectified = self.qr_detector.detectAndDecode(image)
            if data:
                results.append({
                    'data': data,
                    'type': 'QRCODE',
                    'bbox': bbox,
                    'method': 'opencv_qr',
                    'confidence': 0.9
                })

            # QRCodeDetectorAruco (if available)
            if self.qr_aruco:
                data, bbox, rectified = self.qr_aruco.detectAndDecode(image)
                if data and data not in [r['data'] for r in results]:
                    results.append({
                        'data': data,
                        'type': 'QRCODE',
                        'bbox': bbox,
                        'method': 'opencv_qr_aruco',
                        'confidence': 0.95
                    })

        except Exception as e:
            logger.debug(f"OpenCV QR detection error: {e}")

        return results

    def detect_with_wechat(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using WeChat QRCode detector."""
        results = []
        if not self.wechat_detector:
            return results

        try:
            res, points = self.wechat_detector.detectAndDecode(image)
            if res:
                for i, data in enumerate(res):
                    if data:
                        bbox = points[i] if i < len(points) else None
                        results.append({
                            'data': data,
                            'type': 'QRCODE',
                            'bbox': bbox,
                            'method': 'wechat_qr',
                            'confidence': 0.98
                        })
        except Exception as e:
            logger.debug(f"WeChat QR detection error: {e}")

        return results

    def detect_with_opencv_barcode(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using OpenCV BarcodeDetector."""
        results = []
        if not self.barcode_detector:
            return results

        try:
            retval, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data:
                        results.append({
                            'data': data,
                            'type': decoded_type[i] if i < len(decoded_type) else 'BARCODE',
                            'bbox': points[i] if i < len(points) else None,
                            'method': 'opencv_barcode',
                            'confidence': 0.85
                        })
        except Exception as e:
            logger.debug(f"OpenCV Barcode detection error: {e}")

        return results

    def detect_all(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect barcodes/QR codes using all available methods."""
        all_results = []
        seen_data = set()

        # Get preprocessed versions of the image
        processed_images = self.preprocess_for_detection(image)

        # Try each detection method on each processed image
        for idx, processed in enumerate(processed_images):
            # pyzbar (most reliable for various barcode types)
            if PYZBAR_AVAILABLE:
                results = self.detect_with_pyzbar(processed)
                for r in results:
                    if r['data'] not in seen_data:
                        r['preprocess'] = idx
                        all_results.append(r)
                        seen_data.add(r['data'])

            # OpenCV QR detectors
            results = self.detect_with_opencv_qr(processed)
            for r in results:
                if r['data'] not in seen_data:
                    r['preprocess'] = idx
                    all_results.append(r)
                    seen_data.add(r['data'])

            # WeChat QR detector
            results = self.detect_with_wechat(processed)
            for r in results:
                if r['data'] not in seen_data:
                    r['preprocess'] = idx
                    all_results.append(r)
                    seen_data.add(r['data'])

            # OpenCV Barcode detector
            results = self.detect_with_opencv_barcode(processed)
            for r in results:
                if r['data'] not in seen_data:
                    r['preprocess'] = idx
                    all_results.append(r)
                    seen_data.add(r['data'])

        return all_results


class CV60QRDetector:
    """Main QR/Barcode detector for CV60 camera."""

    def __init__(self, camera_ip: str = "192.168.1.21"):
        self.camera_ip = camera_ip
        self.camera = None
        self.detector = MultiMethodDetector()
        self.detected_codes = set()

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.method_stats = {}

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

    def draw_detection(self, frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
        """Draw detection overlay on frame."""
        # Draw bounding box if available
        if detection.get('bbox') is not None:
            bbox = detection['bbox']
            if isinstance(bbox, np.ndarray) and len(bbox.shape) > 1:
                # Draw polygon
                pts = bbox.astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            elif isinstance(bbox, tuple) and len(bbox) == 4:
                # Draw rectangle (x, y, width, height)
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add text overlay
        text = f"{detection['type']}: {detection['data'][:50]}"
        cv2.putText(frame, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 255), 2)

        # Add method info
        method_text = f"Method: {detection['method']}"
        cv2.putText(frame, method_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 0), 1)

        return frame

    def run(self):
        """Main detection loop."""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return

        print("\n" + "="*60)
        print("CV60 ENHANCED QR/BARCODE DETECTOR")
        print("="*60)
        print(f"Camera IP: {self.camera_ip}")
        print(f"Detection methods available:")
        print(f"  - pyzbar: {PYZBAR_AVAILABLE}")
        print(f"  - OpenCV QR: True")
        print(f"  - WeChat QR: {HAS_WECHAT_QRCODE}")
        print(f"  - OpenCV Barcode: {HAS_BARCODE_DETECTOR}")
        print("Press 'q' or ESC to quit")
        print("="*60 + "\n")

        # Create display window
        window_name = "CV60 Enhanced QR/Barcode Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        try:
            # Stream frames from camera
            for frame in self.camera.stream():
                self.frame_count += 1

                # Detect barcodes/QR codes
                detections = self.detector.detect_all(frame)

                # Process new detections
                for detection in detections:
                    barcode_value = detection['data']
                    if barcode_value and barcode_value not in self.detected_codes:
                        print(f"\n[DETECTED] {barcode_value}")
                        print(f"  Type: {detection['type']}")
                        print(f"  Method: {detection['method']}")
                        print(f"  Confidence: {detection.get('confidence', 'N/A')}")

                        self.detected_codes.add(barcode_value)
                        self.detection_count += 1

                        # Update method statistics
                        method = detection['method']
                        self.method_stats[method] = self.method_stats.get(method, 0) + 1

                # Prepare display frame
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                # Add overlay
                cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Total Detections: {self.detection_count}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Unique Codes: {len(self.detected_codes)}", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Draw current detections
                if detections:
                    display_frame = self.draw_detection(display_frame, detections[0])

                # Show frame
                cv2.imshow(window_name, display_frame)

                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    # Save current frame
                    cv2.imwrite(f"captured_frame_{self.frame_count}.png", frame)
                    print(f"Saved frame to captured_frame_{self.frame_count}.png")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "="*60)
            print("DETECTION SUMMARY")
            print("="*60)
            print(f"Frames processed: {self.frame_count}")
            print(f"Total detections: {self.detection_count}")
            print(f"Unique codes: {len(self.detected_codes)}")

            if self.method_stats:
                print("\nDetection method statistics:")
                for method, count in self.method_stats.items():
                    print(f"  {method}: {count} detections")

            print("="*60)

            # Cleanup
            if self.camera:
                self.camera.stop_acquisition()
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Enhanced CV60 QR/Barcode Detector')
    parser.add_argument('--ip', default='192.168.1.21',
                       help='CV60 camera IP address')
    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    detector = CV60QRDetector(camera_ip=args.ip)
    detector.run()


if __name__ == "__main__":
    main()