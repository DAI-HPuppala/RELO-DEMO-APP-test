#!/usr/bin/env python3
"""Auto-detecting Barcode Scanner - Continuously scans at 1-5 FPS."""

import sys
import os
import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import json

# Setup paths for eBUS SDK - CRITICAL: Must be before imports
EBUS_SDK_PATH = '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'
if EBUS_SDK_PATH not in sys.path:
    sys.path.insert(0, EBUS_SDK_PATH)

# Setup LD_LIBRARY_PATH for native libraries
lib_paths = [
    '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib',
    '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/genicam/bin/Linux64_x64'
]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths + [current_ld_path])

# Try to import our modular CV60 camera wrapper
CV60_MODULE_AVAILABLE = False
try:
    # Try the new modular version first
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from barcode_detector.core.cv60_camera import CV60Camera
    from barcode_detector.core.oak_d_camera import create_oak_d_camera, DEPTHAI_AVAILABLE
    CV60_MODULE_AVAILABLE = True
    print("✓ Using modular camera modules")
except ImportError as e1:
    print(f"[WARNING] Could not import camera modules: {e1}")
    CV60_MODULE_AVAILABLE = False

# Import pyzbar - best for barcode detection
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
    print("✓ pyzbar library loaded - PRIMARY BARCODE DETECTOR")
except ImportError:
    print("✗ pyzbar not available - CRITICAL for barcode detection!")
    print("  Install with: pip install pyzbar")
    print("  System lib: sudo apt-get install libzbar0")


@dataclass
class BarcodeDetection:
    """Container for barcode detection results."""
    data: str
    barcode_type: str
    method: str
    preprocessing: str
    position: Any = None
    quality_score: float = 0.0
    timestamp: float = 0.0


class AutoBarcodeDetector:
    """Auto-detecting barcode scanner using lightweight detection + full decoding."""

    def __init__(self, camera_type: str = "cv60", camera_ip: str = "192.168.1.21",
                 detection_interval: int = 5, oak_d_ip: str = "169.254.1.222"):
        self.camera_type = camera_type
        self.camera_ip = camera_ip
        self.oak_d_ip = oak_d_ip
        self.camera = None

        # Detection settings
        self.detection_interval = detection_interval  # Detect every N frames
        self.frame_skip_counter = 0

        # Tracking
        self.frame_count = 0
        self.detection_count = 0
        self.current_frame = None
        self.unique_barcodes = {}  # data -> detection info

        # FPS
        self.fps_counter = []
        self.current_fps = 0

        # Display
        self.show_feedback = False
        self.feedback_time = 0
        self.last_detection = None

        # Initialize detectors
        self.qr_detector = cv2.QRCodeDetector()

        # Create output directory
        self.output_dir = "auto_detected_barcodes"
        os.makedirs(self.output_dir, exist_ok=True)

    def _apply_clahe(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to grayscale frame."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_frame)

    def _apply_sharpening(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply sharpening kernel to grayscale frame."""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv2.filter2D(gray_frame, -1, kernel)

    def initialize_camera(self) -> bool:
        """Initialize camera based on type."""
        if not CV60_MODULE_AVAILABLE:
            print("[ERROR] Camera modules not available")
            return False

        try:
            if self.camera_type == "oakd":
                if not DEPTHAI_AVAILABLE:
                    print("[ERROR] DepthAI not available for OAK-D")
                    return False

                print(f"\nConnecting to OAK-D PoE camera at {self.oak_d_ip}...")
                self.camera = create_oak_d_camera(resolution_4k=True, fps=30, ip_address=self.oak_d_ip)

                if self.camera and self.camera.isOpened():
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ OAK-D PoE camera initialized: {width}x{height}")
                    print(f"✓ Connected to: {self.oak_d_ip}")
                    return True
                else:
                    print("[ERROR] Failed to initialize OAK-D camera")
                    return False
            else:
                print(f"\nConnecting to CV60 camera at {self.camera_ip}...")
                self.camera = CV60Camera(camera_ip=self.camera_ip)

                if self.camera.initialize():
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ CV60 camera initialized: {width}x{height}")
                    return True
                else:
                    print("[ERROR] Failed to initialize CV60 camera")
                    return False

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def detect_with_pyzbar(self, image: np.ndarray, prep_name: str) -> List[BarcodeDetection]:
        """Detect barcodes using pyzbar."""
        detections = []
        if not PYZBAR_AVAILABLE:
            return detections

        try:
            decoded = pyzbar.decode(image)

            for obj in decoded:
                # Skip QR codes if looking for barcodes only
                # if obj.type == 'QRCODE':
                #     continue

                try:
                    data = obj.data.decode('utf-8', errors='ignore')
                    if data:
                        detections.append(BarcodeDetection(
                            data=data,
                            barcode_type=str(obj.type),
                            method='pyzbar',
                            preprocessing=prep_name,
                            position=obj.rect,
                            quality_score=1.0,
                            timestamp=time.time()
                        ))
                except:
                    pass
        except Exception as e:
            pass

        return detections

    def detect_with_opencv(self, image: np.ndarray, prep_name: str) -> List[BarcodeDetection]:
        """Detect QR codes using OpenCV."""
        detections = []

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            data, bbox, _ = self.qr_detector.detectAndDecode(gray)
            if data:
                detections.append(BarcodeDetection(
                    data=data,
                    barcode_type='QRCODE',
                    method='opencv_qr',
                    preprocessing=prep_name,
                    position=bbox,
                    quality_score=0.7,
                    timestamp=time.time()
                ))
        except Exception as e:
            pass

        return detections

    def detect_barcodes_sequential(self, frame: np.ndarray) -> List[BarcodeDetection]:
        """Sequential detection: try pyzbar with preprocessing steps until found."""
        all_results = []

        # Preprocessing steps (stop when found)
        preprocessing_steps = [
            ('original', lambda f: f),
            ('grayscale', lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f),
            ('clahe', lambda f: self._apply_clahe(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
            ('bilateral_filter', lambda f: cv2.bilateralFilter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 11, 17, 17)),
            ('binary_threshold', lambda f: cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 127, 255, cv2.THRESH_BINARY)[1]),
            ('adaptive_threshold', lambda f: cv2.adaptiveThreshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('sharpening', lambda f: self._apply_sharpening(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
        ]

        # PHASE 1: Try pyzbar sequentially (stop when found)
        if PYZBAR_AVAILABLE:
            for preproc_name, preproc_func in preprocessing_steps:
                try:
                    processed = preproc_func(frame)
                    detections = self.detect_with_pyzbar(processed, preproc_name)

                    if detections:
                        all_results.extend(detections)
                        break  # STOP - found something!
                except:
                    continue

        # PHASE 2: If pyzbar found nothing, try OpenCV
        if not all_results:
            for preproc_name, preproc_func in preprocessing_steps:
                try:
                    processed = preproc_func(frame)
                    detections = self.detect_with_opencv(processed, preproc_name)

                    if detections:
                        all_results.extend(detections)
                        break  # STOP - found something!
                except:
                    continue

        # Deduplicate
        unique_results = []
        seen_data = set()
        for detection in all_results:
            if detection.data not in seen_data:
                unique_results.append(detection)
                seen_data.add(detection.data)

        return unique_results

    def process_detection(self, frame: np.ndarray, detections: List[BarcodeDetection]) -> None:
        """Process and save detection results."""
        if not detections:
            return

        self.detection_count += 1
        timestamp = datetime.now()

        # Save frame with detection
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"auto_barcode_{timestamp_str}_frame{self.frame_count}"
        image_path = os.path.join(self.output_dir, f"{base_filename}.png")

        # Annotate frame
        annotated = frame.copy()
        for detection in detections:
            # Draw detection info at top
            text = f"{detection.barcode_type}: {detection.data}"
            cv2.putText(annotated, text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imwrite(image_path, annotated)

        # Print detection info
        print("\n" + "="*70)
        print(f"AUTO-DETECTED at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print("="*70)

        for detection in detections:
            print(f"  • Data: {detection.data}")
            print(f"    Type: {detection.barcode_type}")
            print(f"    Method: {detection.method}")
            print(f"    Preprocessing: {detection.preprocessing}")

            # Track unique barcodes
            if detection.data not in self.unique_barcodes:
                self.unique_barcodes[detection.data] = {
                    'type': detection.barcode_type,
                    'first_seen': timestamp.isoformat(),
                    'frame': self.frame_count,
                    'method': detection.method,
                    'preprocessing': detection.preprocessing
                }

        print(f"\n💾 Saved: {image_path}")
        print(f"📊 Total unique barcodes: {len(self.unique_barcodes)}")
        print("="*70 + "\n")

        # Save JSON results
        results_file = os.path.join(self.output_dir, f"{base_filename}_results.json")
        results_data = {
            'timestamp': timestamp.isoformat(),
            'frame': self.frame_count,
            'barcodes': [
                {
                    'data': d.data,
                    'type': d.barcode_type,
                    'method': d.method,
                    'preprocessing': d.preprocessing
                }
                for d in detections
            ]
        }
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Update display feedback
        self.last_detection = detections[0]
        self.show_feedback = True
        self.feedback_time = time.time()

    def calculate_fps(self):
        """Calculate FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time)
        self.fps_counter = [t for t in self.fps_counter if current_time - t < 1.0]
        if len(self.fps_counter) > 1:
            self.current_fps = len(self.fps_counter)
        return self.current_fps

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay on frame."""
        fps = self.calculate_fps()

        # FPS
        fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Stats
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Auto-detections: {self.detection_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.unique_barcodes:
            cv2.putText(frame, f"Unique codes: {len(self.unique_barcodes)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mode indicator
        camera_name = "OAK-D" if self.camera_type == "oakd" else "CV60"
        cv2.putText(frame, f"AUTO SCANNER ({camera_name})", (frame.shape[1]//2 - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Detection interval info
        cv2.putText(frame, f"Scanning every {self.detection_interval} frames", (10, frame.shape[0] - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, f"Next scan in: {self.detection_interval - self.frame_skip_counter} frames",
                   (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)
        cv2.putText(frame, "'q'=quit | 'c'=clear | '+/-'=adjust", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Detection feedback
        if self.show_feedback:
            elapsed = time.time() - self.feedback_time
            if elapsed < 2.0:
                if self.last_detection:
                    color = (0, 255, 0)
                    text = f"FOUND: {self.last_detection.data}"
                else:
                    color = (0, 0, 255)
                    text = "SCANNING..."

                alpha = 1 - (elapsed / 2.0)
                color = tuple(int(c * alpha) for c in color)
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 3)
                cv2.putText(frame, text, (frame.shape[1]//2 - 200, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                self.show_feedback = False

        # ROI guide
        h, w = frame.shape[:2]
        roi_width = int(w * 0.6)
        roi_height = int(h * 0.4)
        roi_x1 = (w - roi_width) // 2
        roi_y1 = (h - roi_height) // 2
        roi_x2 = roi_x1 + roi_width
        roi_y2 = roi_y1 + roi_height

        roi_color = (0, 255, 255)
        cv2.line(frame, (roi_x1, roi_y1), (roi_x1 + 50, roi_y1), roi_color, 2)
        cv2.line(frame, (roi_x1, roi_y1), (roi_x1, roi_y1 + 50), roi_color, 2)
        cv2.line(frame, (roi_x2, roi_y1), (roi_x2 - 50, roi_y1), roi_color, 2)
        cv2.line(frame, (roi_x2, roi_y1), (roi_x2, roi_y1 + 50), roi_color, 2)
        cv2.line(frame, (roi_x1, roi_y2), (roi_x1 + 50, roi_y2), roi_color, 2)
        cv2.line(frame, (roi_x1, roi_y2), (roi_x1, roi_y2 - 50), roi_color, 2)
        cv2.line(frame, (roi_x2, roi_y2), (roi_x2 - 50, roi_y2), roi_color, 2)
        cv2.line(frame, (roi_x2, roi_y2), (roi_x2, roi_y2 - 50), roi_color, 2)

        return frame

    def run(self):
        """Main loop with auto-detection."""
        if not self.initialize_camera():
            print("[ERROR] Failed to initialize camera")
            return

        print("\n" + "="*70)
        camera_name = "OAK-D PoE" if self.camera_type == "oakd" else "CV60"
        print(f"AUTO BARCODE SCANNER - {camera_name}")
        print("="*70)
        print(f"Auto-detecting every {self.detection_interval} frames")
        print("\nCONTROLS:")
        print("  'q' or ESC = Quit")
        print("  'c' = Clear history")
        print("  '+' = Increase detection interval (slower)")
        print("  '-' = Decrease detection interval (faster)")
        print("\nDETECTORS:")
        print(f"  • pyzbar: {PYZBAR_AVAILABLE} (PRIMARY)")
        print(f"  • OpenCV QR: Available (backup)")
        print("="*70 + "\n")

        # Create window
        window_name = f"Auto Barcode Scanner - {camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        try:
            frame_time = 1.0 / 30.0
            last_frame_time = time.time()

            while True:
                current_time = time.time()

                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_count += 1
                self.current_frame = frame

                # Auto-detection logic
                if self.frame_skip_counter >= self.detection_interval:
                    # Time to detect!
                    detections = self.detect_barcodes_sequential(frame)
                    if detections:
                        self.process_detection(frame, detections)
                    self.frame_skip_counter = 0
                else:
                    self.frame_skip_counter += 1

                # Display
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                display_frame = self.draw_overlay(display_frame)
                cv2.imshow(window_name, display_frame)

                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('c') or key == ord('C'):  # clear history
                    self.unique_barcodes.clear()
                    print("[INFO] Cleared barcode history")
                elif key == ord('+') or key == ord('='):  # increase interval
                    self.detection_interval = min(30, self.detection_interval + 1)
                    print(f"[INFO] Detection interval: {self.detection_interval} frames")
                elif key == ord('-') or key == ord('_'):  # decrease interval
                    self.detection_interval = max(1, self.detection_interval - 1)
                    print(f"[INFO] Detection interval: {self.detection_interval} frames")

                # Maintain FPS
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_frame_time = time.time()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        finally:
            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Frames: {self.frame_count}")
            print(f"Auto-detections: {self.detection_count}")
            print(f"Unique barcodes: {len(self.unique_barcodes)}")
            if self.unique_barcodes:
                print("\nDetected barcodes:")
                for code, info in self.unique_barcodes.items():
                    print(f"  • {code} ({info['type']})")
            print("="*70)

            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Auto Barcode Scanner')
    parser.add_argument('--camera', choices=['cv60', 'oakd'], default='cv60',
                       help='Camera type (cv60 or oakd)')
    parser.add_argument('--ip', default='192.168.1.21',
                       help='CV60 camera IP address')
    parser.add_argument('--oak-ip', default='169.254.1.222',
                       help='OAK-D PoE camera IP address')
    parser.add_argument('--interval', type=int, default=5,
                       help='Detection interval in frames (default: 5)')
    args = parser.parse_args()

    scanner = AutoBarcodeDetector(
        camera_type=args.camera,
        camera_ip=args.ip,
        detection_interval=args.interval,
        oak_d_ip=args.oak_ip
    )
    scanner.run()


if __name__ == "__main__":
    main()
