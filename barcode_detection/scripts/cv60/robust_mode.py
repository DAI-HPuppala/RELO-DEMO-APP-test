#!/usr/bin/env python3
"""Robust CV60 QR/Barcode Detector with comprehensive detection and debugging."""

import sys
import os
import cv2
import numpy as np
import time
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import json

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

# Import detection libraries
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
    print("✓ pyzbar library loaded")
except ImportError:
    print("✗ pyzbar not available")

# Try importing qrcode for generation/testing
try:
    import qrcode
    QRCODE_GEN_AVAILABLE = True
except:
    QRCODE_GEN_AVAILABLE = False


@dataclass
class DetectionResult:
    """Container for detection results."""
    method: str
    data: str
    type: str
    preprocessing: str
    bbox: Any = None
    confidence: float = 0.0


class RobustBarcodeDetector:
    """Robust barcode/QR detector with comprehensive preprocessing."""

    def __init__(self):
        # Initialize all detectors
        self.qr_detector = cv2.QRCodeDetector()

        # Try to initialize additional detectors
        self.qr_aruco = None
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            try:
                self.qr_aruco = cv2.QRCodeDetectorAruco()
                print("✓ QRCodeDetectorAruco initialized")
            except:
                print("✗ QRCodeDetectorAruco not available")

        # Initialize barcode detector if available
        self.barcode_detector = None
        if hasattr(cv2, 'barcode'):
            try:
                self.barcode_detector = cv2.barcode.BarcodeDetector()
                print("✓ OpenCV BarcodeDetector initialized")
            except:
                try:
                    # Try alternative import
                    self.barcode_detector = cv2.barcode_BarcodeDetector()
                    print("✓ OpenCV BarcodeDetector initialized (alt)")
                except:
                    print("✗ OpenCV BarcodeDetector not available")

    def preprocess_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply comprehensive preprocessing for barcode detection."""
        preprocessed = {}

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        preprocessed['original_gray'] = gray

        # 1. Denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        preprocessed['denoised'] = denoised

        # 2. Sharpening (important for barcode edges)
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        preprocessed['sharpened'] = sharpened

        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessed['clahe'] = enhanced

        # 4. Bilateral filter (edge-preserving smoothing)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed['bilateral'] = bilateral

        # 5. Adaptive threshold
        thresh_adaptive = cv2.adaptiveThreshold(gray, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 51, 10)
        preprocessed['adaptive_thresh'] = thresh_adaptive

        # 6. OTSU threshold
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed['otsu_thresh'] = thresh_otsu

        # 7. Morphological operations (helps with broken barcodes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        preprocessed['morphological'] = morph

        # 8. Histogram equalization
        equalized = cv2.equalizeHist(gray)
        preprocessed['equalized'] = equalized

        # 9. Multiple scales
        for scale in [0.5, 0.75, 1.5, 2.0]:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
            preprocessed[f'scale_{scale}'] = scaled

        return preprocessed

    def detect_with_pyzbar(self, image: np.ndarray, preprocess_name: str) -> List[DetectionResult]:
        """Detect using pyzbar with various symbol types."""
        results = []
        if not PYZBAR_AVAILABLE:
            return results

        try:
            # Try detection with all symbols
            decoded_all = pyzbar.decode(image)
            for obj in decoded_all:
                try:
                    data = obj.data.decode('utf-8', errors='ignore')
                    if data:  # Only add if data is not empty
                        results.append(DetectionResult(
                            method='pyzbar',
                            data=data,
                            type=str(obj.type),
                            preprocessing=preprocess_name,
                            bbox=obj.rect
                        ))
                except:
                    pass

            # Also try with specific symbol types if no results
            if not decoded_all:
                # Try QR only
                decoded_qr = pyzbar.decode(image, symbols=[ZBarSymbol.QRCODE])
                for obj in decoded_qr:
                    try:
                        data = obj.data.decode('utf-8', errors='ignore')
                        if data:
                            results.append(DetectionResult(
                                method='pyzbar_qr',
                                data=data,
                                type='QRCODE',
                                preprocessing=preprocess_name,
                                bbox=obj.rect
                            ))
                    except:
                        pass

        except Exception as e:
            pass  # Silent fail for this preprocessing

        return results

    def detect_with_opencv_qr(self, image: np.ndarray, preprocess_name: str) -> List[DetectionResult]:
        """Detect using OpenCV QR detectors."""
        results = []

        # Standard QR detector
        try:
            data, bbox, straight = self.qr_detector.detectAndDecode(image)
            if data and len(data) > 0:
                results.append(DetectionResult(
                    method='opencv_qr',
                    data=data,
                    type='QRCODE',
                    preprocessing=preprocess_name,
                    bbox=bbox
                ))
        except:
            pass

        # QR detector with Aruco
        if self.qr_aruco:
            try:
                data, bbox, straight = self.qr_aruco.detectAndDecode(image)
                if data and len(data) > 0:
                    results.append(DetectionResult(
                        method='opencv_qr_aruco',
                        data=data,
                        type='QRCODE',
                        preprocessing=preprocess_name,
                        bbox=bbox
                    ))
            except:
                pass

        # Try multi-detection
        try:
            retval, decoded_info, points, straight = self.qr_detector.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data and len(data) > 0:
                        results.append(DetectionResult(
                            method='opencv_qr_multi',
                            data=data,
                            type='QRCODE',
                            preprocessing=preprocess_name,
                            bbox=points[i] if i < len(points) else None
                        ))
        except:
            pass

        return results

    def detect_with_opencv_barcode(self, image: np.ndarray, preprocess_name: str) -> List[DetectionResult]:
        """Detect using OpenCV barcode detector."""
        results = []
        if not self.barcode_detector:
            return results

        try:
            retval, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data and len(data) > 0:
                        barcode_type = decoded_type[i] if i < len(decoded_type) else 'BARCODE'
                        results.append(DetectionResult(
                            method='opencv_barcode',
                            data=data,
                            type=barcode_type,
                            preprocessing=preprocess_name,
                            bbox=points[i] if i < len(points) else None
                        ))
        except:
            pass

        return results

    def detect_comprehensive(self, frame: np.ndarray) -> Tuple[List[DetectionResult], Dict[str, np.ndarray]]:
        """Perform comprehensive detection with all methods and preprocessing."""
        all_results = []

        # Get all preprocessed versions
        preprocessed = self.preprocess_frame(frame)

        # Try each detection method on each preprocessed image
        for prep_name, prep_image in preprocessed.items():
            # Skip if image is None or empty
            if prep_image is None or prep_image.size == 0:
                continue

            # pyzbar detection
            results = self.detect_with_pyzbar(prep_image, prep_name)
            all_results.extend(results)

            # OpenCV QR detection
            results = self.detect_with_opencv_qr(prep_image, prep_name)
            all_results.extend(results)

            # OpenCV barcode detection
            results = self.detect_with_opencv_barcode(prep_image, prep_name)
            all_results.extend(results)

        return all_results, preprocessed


class CV60RobustDetector:
    """Main application for robust CV60 barcode detection."""

    def __init__(self, camera_ip: str = "192.168.1.21"):
        self.camera_ip = camera_ip
        self.camera = None
        self.detector = RobustBarcodeDetector()

        # Frame tracking
        self.frame_count = 0
        self.capture_count = 0
        self.current_frame = None

        # Detection history
        self.all_detections = []
        self.unique_codes = set()

        # FPS tracking
        self.fps_counter = []
        self.current_fps = 0

        # Display state
        self.show_feedback = False
        self.feedback_time = 0
        self.last_detection_results = []

    def initialize_camera(self) -> bool:
        """Initialize CV60 camera."""
        if not CV60_MODULE_AVAILABLE:
            print("[ERROR] CV60 module not available")
            return False

        try:
            print(f"Connecting to CV60 camera at {self.camera_ip}...")
            self.camera = CV60Camera(ip_address=self.camera_ip)

            if self.camera.connect():
                settings = self.camera.get_camera_settings()
                print(f"✓ Connected to CV60 camera")
                print(f"  Resolution: {settings.get('Width')}x{settings.get('Height')}")
                print(f"  Pixel Format: {settings.get('PixelFormat')}")

                if self.camera.start_acquisition():
                    print("✓ Camera acquisition started")
                    return True
                else:
                    print("[ERROR] Failed to start acquisition")
                    self.camera.disconnect()
                    return False
            else:
                print(f"[ERROR] Failed to connect to CV60 at {self.camera_ip}")
                return False

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def capture_and_detect(self):
        """Capture current frame and perform comprehensive detection."""
        if self.current_frame is None:
            print("\n[ERROR] No frame available")
            return

        # Capture current frame
        frame = self.current_frame.copy()
        self.capture_count += 1
        timestamp = datetime.now()

        # Create test directory
        test_dir = "test_captured_frames"
        os.makedirs(test_dir, exist_ok=True)

        # Save original frame
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"frame_{timestamp_str}_{self.frame_count}"

        # Save original
        original_path = os.path.join(test_dir, f"{base_filename}_original.png")
        cv2.imwrite(original_path, frame)

        print("\n" + "="*80)
        print(f"CAPTURE #{self.capture_count} - Frame {self.frame_count}")
        print("="*80)
        print(f"Time: {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"Original saved: {original_path}")
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

        # Get frame info
        if len(frame.shape) == 3:
            print(f"Color channels: {frame.shape[2]}")
            print(f"Mean values (B,G,R): {frame[:,:,0].mean():.1f}, {frame[:,:,1].mean():.1f}, {frame[:,:,2].mean():.1f}")
        else:
            print(f"Grayscale, mean: {frame.mean():.1f}")

        print("\nRUNNING COMPREHENSIVE DETECTION...")
        print("-"*80)

        # Perform detection
        results, preprocessed_images = self.detector.detect_comprehensive(frame)

        # Save preprocessed images for analysis
        print("\nSaving preprocessed images for analysis:")
        for prep_name, prep_image in list(preprocessed_images.items())[:5]:  # Save first 5
            if prep_image is not None and prep_image.size > 0:
                prep_path = os.path.join(test_dir, f"{base_filename}_{prep_name}.png")
                cv2.imwrite(prep_path, prep_image)
                print(f"  - {prep_name}: {prep_path}")

        # Analyze results
        print("\n" + "-"*80)
        print("DETECTION RESULTS:")

        if results:
            # Group by unique data
            unique_detections = {}
            for r in results:
                if r.data not in unique_detections:
                    unique_detections[r.data] = []
                unique_detections[r.data].append(r)

            print(f"\n✓ Found {len(unique_detections)} unique code(s):")
            for data, detections in unique_detections.items():
                print(f"\n  Code: '{data}'")
                print(f"  Detected by {len(detections)} method(s):")
                for d in detections[:5]:  # Show first 5
                    print(f"    - {d.method} with {d.preprocessing} preprocessing")

                # Add to history
                self.unique_codes.add(data)

            # Save detection info
            detection_file = os.path.join(test_dir, f"{base_filename}_detections.json")
            detection_data = {
                'timestamp': timestamp.isoformat(),
                'frame_number': self.frame_count,
                'detections': [
                    {
                        'data': r.data,
                        'type': r.type,
                        'method': r.method,
                        'preprocessing': r.preprocessing
                    }
                    for r in results
                ]
            }
            with open(detection_file, 'w') as f:
                json.dump(detection_data, f, indent=2)
            print(f"\nDetection data saved: {detection_file}")

        else:
            print("\n✗ No barcodes/QR codes detected")
            print("\nTrying additional analysis:")

            # Try to understand why detection failed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Check image quality metrics
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            print(f"  - Blur metric (Laplacian variance): {laplacian_var:.2f}")
            if laplacian_var < 100:
                print("    → Image might be too blurry")

            # Check contrast
            contrast = gray.std()
            print(f"  - Contrast (std dev): {contrast:.2f}")
            if contrast < 30:
                print("    → Image might have low contrast")

            # Check if image is too dark or bright
            mean_val = gray.mean()
            print(f"  - Mean brightness: {mean_val:.2f}")
            if mean_val < 50:
                print("    → Image might be too dark")
            elif mean_val > 200:
                print("    → Image might be too bright/overexposed")

        self.last_detection_results = results
        self.all_detections.extend(results)

        print("\n" + "-"*80)
        print(f"Total unique codes found in session: {len(self.unique_codes)}")
        print("="*80 + "\n")

        # Visual feedback
        self.show_feedback = True
        self.feedback_time = time.time()

    def calculate_fps(self):
        """Calculate current FPS."""
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

        # Frame info
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Capture count
        cv2.putText(frame, f"Captures: {self.capture_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Unique codes
        if self.unique_codes:
            cv2.putText(frame, f"Codes found: {len(self.unique_codes)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "Press '+' to capture & detect", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, "'t' = test pattern | 'q' = quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Capture feedback
        if self.show_feedback:
            elapsed = time.time() - self.feedback_time
            if elapsed < 1.0:
                alpha = 1 - elapsed
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 3)

                if self.last_detection_results:
                    text = f"DETECTED: {len(set(r.data for r in self.last_detection_results))} code(s)"
                    cv2.putText(frame, text, (frame.shape[1]//2 - 150, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                self.show_feedback = False

        return frame

    def create_test_pattern(self):
        """Create a test QR code pattern."""
        if not QRCODE_GEN_AVAILABLE:
            print("qrcode library not available for test pattern")
            return None

        test_data = f"TEST_QR_{self.frame_count}"
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(test_data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        test_frame = np.array(img, dtype=np.uint8) * 255

        # Convert to BGR
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_GRAY2BGR)

        # Add text
        cv2.putText(test_frame, f"Test Pattern: {test_data}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return test_frame

    def run(self):
        """Main execution loop."""
        if not self.initialize_camera():
            print("[ERROR] Failed to initialize camera")
            return

        print("\n" + "="*80)
        print("CV60 ROBUST DETECTOR - 30 FPS with Comprehensive Detection")
        print("="*80)
        print(f"Camera: {self.camera_ip}")
        print("\nCONTROLS:")
        print("  '+' = Capture and detect (comprehensive)")
        print("  't' = Generate test pattern")
        print("  'q' = Quit")
        print("\nDETECTION METHODS:")
        print(f"  • pyzbar: {PYZBAR_AVAILABLE}")
        print(f"  • OpenCV QR: True")
        print(f"  • OpenCV QR Aruco: {self.detector.qr_aruco is not None}")
        print(f"  • OpenCV Barcode: {self.detector.barcode_detector is not None}")
        print("="*80 + "\n")

        # Create directories
        os.makedirs("test_captured_frames", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Window
        window_name = "CV60 Robust Detector - 30 FPS"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        try:
            frame_time = 1.0 / 30.0
            last_frame_time = time.time()

            for frame in self.camera.stream():
                current_time = time.time()
                self.frame_count += 1
                self.current_frame = frame

                # Prepare display
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                # Draw overlay
                display_frame = self.draw_overlay(display_frame)

                # Show
                cv2.imshow(window_name, display_frame)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('+') or key == ord('='):
                    self.capture_and_detect()
                elif key == ord('t'):
                    test_frame = self.create_test_pattern()
                    if test_frame is not None:
                        self.current_frame = test_frame
                        self.capture_and_detect()

                # Maintain FPS
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_frame_time = time.time()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        finally:
            print("\n" + "="*80)
            print("SESSION SUMMARY")
            print("="*80)
            print(f"Frames: {self.frame_count}")
            print(f"Captures: {self.capture_count}")
            print(f"Unique codes: {len(self.unique_codes)}")
            if self.unique_codes:
                print("\nDetected codes:")
                for code in list(self.unique_codes)[:10]:
                    print(f"  • {code}")
            print("="*80)

            if self.camera:
                self.camera.stop_acquisition()
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Robust CV60 QR/Barcode Detector')
    parser.add_argument('--ip', default='192.168.1.21', help='Camera IP')
    args = parser.parse_args()

    detector = CV60RobustDetector(camera_ip=args.ip)
    detector.run()


if __name__ == "__main__":
    main()