#!/usr/bin/env python3
"""CV60 Barcode Detector - Optimized specifically for linear barcodes."""

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
    CV60_MODULE_AVAILABLE = True
    print("✓ Using modular CV60Camera from barcode_detector")
except ImportError as e1:
    # Fallback to old cv60_ebus_module if it exists
    try:
        sys.path.insert(0, '/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/backend')
        from cv60_ebus_module.cv60_ebus import CV60Camera
        CV60_MODULE_AVAILABLE = True
        print("✓ Using legacy cv60_ebus_module")
    except ImportError as e2:
        print(f"[WARNING] Could not import CV60 camera module")
        print(f"  Tried modular: {e1}")
        print(f"  Tried legacy: {e2}")
        CV60_MODULE_AVAILABLE = False

# Import pyzbar - best for barcode detection
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
    print("✓ pyzbar library loaded - PRIMARY BARCODE DETECTOR")

    # List supported barcode types
    barcode_types = [
        'CODE128', 'CODE39', 'CODE93', 'CODABAR',
        'EAN13', 'EAN8', 'UPCA', 'UPCE',
        'I25', 'PDF417', 'DATABAR', 'DATAMATRIX'
    ]
    print(f"  Supported barcode types: {', '.join(barcode_types)}")
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
    orientation: str = "horizontal"


class BarcodeFocusedDetector:
    """Detector optimized for linear barcodes using sequential detection."""

    def __init__(self):
        print("\nInitializing barcode detectors...")

        # Initialize OpenCV QR detector as backup
        self.qr_detector = cv2.QRCodeDetector()

        # OpenCV barcode detector (backup method)
        self.cv_barcode = None
        if hasattr(cv2, 'barcode'):
            try:
                self.cv_barcode = cv2.barcode.BarcodeDetector()
                print("✓ OpenCV BarcodeDetector available (backup)")
            except:
                try:
                    self.cv_barcode = cv2.barcode_BarcodeDetector()
                    print("✓ OpenCV BarcodeDetector available (alt, backup)")
                except:
                    print("✗ OpenCV BarcodeDetector not available")

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

    def preprocess_for_barcodes(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocessing specifically optimized for linear barcodes."""
        preprocessed = {}

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        height, width = gray.shape
        preprocessed['original'] = gray

        # 1. CRITICAL: Sharpening for barcode edges
        # Strong sharpening kernel for barcode bars
        sharp_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, cv2.CV_8U, sharp_kernel)
        preprocessed['sharpened'] = sharpened

        # 2. Denoising while preserving edges (bilateral filter)
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        preprocessed['denoised'] = denoised

        # 3. Contrast enhancement (CLAHE) - moderate for barcodes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        preprocessed['contrast_enhanced'] = enhanced

        # 4. Binary thresholding - critical for barcodes
        # OTSU threshold
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed['binary_otsu'] = binary_otsu

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 5)
        preprocessed['adaptive_thresh'] = adaptive

        # 5. Morphological operations to connect broken bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        morph_close = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        preprocessed['morphological'] = morph_close

        # 6. Gradient magnitude (helps detect barcode regions)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = np.uint8(np.clip(gradient, 0, 255))
        preprocessed['gradient'] = gradient

        # 7. Rotation correction attempts (barcodes must be horizontal/vertical)
        for angle in [-5, -3, -1, 0, 1, 3, 5]:
            if angle != 0:
                center = (width // 2, height // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(sharpened, M, (width, height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                preprocessed[f'rotated_{angle}'] = rotated

        # 8. Multiple resolutions (barcodes need sufficient resolution)
        # Upscaling for small/distant barcodes
        for scale in [1.5, 2.0, 3.0]:
            scaled = cv2.resize(sharpened, None, fx=scale, fy=1.0,  # Scale horizontally more
                              interpolation=cv2.INTER_CUBIC)
            preprocessed[f'upscaled_x{scale}'] = scaled

        # Downscaling for large/close barcodes
        if width > 1000:
            downscaled = cv2.resize(sharpened, None, fx=0.75, fy=0.75,
                                   interpolation=cv2.INTER_AREA)
            preprocessed['downscaled_0.75'] = downscaled

        # 9. Histogram equalization for low contrast
        equalized = cv2.equalizeHist(gray)
        preprocessed['equalized'] = equalized

        # 10. Invert (for negative barcodes - white bars on black)
        inverted = cv2.bitwise_not(binary_otsu)
        preprocessed['inverted'] = inverted

        return preprocessed

    def detect_with_pyzbar(self, image: np.ndarray, prep_name: str) -> List[BarcodeDetection]:
        """Detect barcodes using pyzbar - the most reliable method."""
        detections = []
        if not PYZBAR_AVAILABLE:
            return detections

        try:
            # Decode with all barcode types (excluding QR)
            # First try with all symbols
            decoded = pyzbar.decode(image)

            for obj in decoded:
                # Skip QR codes
                if obj.type == 'QRCODE':
                    continue

                try:
                    data = obj.data.decode('utf-8', errors='ignore')
                    if data:
                        detections.append(BarcodeDetection(
                            data=data,
                            barcode_type=str(obj.type),
                            method='pyzbar',
                            preprocessing=prep_name,
                            position=obj.rect,
                            quality_score=1.0
                        ))
                except:
                    # Try with different encoding
                    try:
                        data = obj.data.decode('latin-1', errors='ignore')
                        if data:
                            detections.append(BarcodeDetection(
                                data=data,
                                barcode_type=str(obj.type),
                                method='pyzbar_latin1',
                                preprocessing=prep_name,
                                position=obj.rect,
                                quality_score=0.9
                            ))
                    except:
                        pass

            # If no detections, try with specific barcode symbols
            if not decoded:
                # Try common barcode types explicitly
                barcode_symbols = [
                    ZBarSymbol.CODE128,
                    ZBarSymbol.CODE39,
                    ZBarSymbol.EAN13,
                    ZBarSymbol.EAN8,
                    ZBarSymbol.UPCA,
                    ZBarSymbol.UPCE,
                    ZBarSymbol.I25,
                    ZBarSymbol.CODABAR,
                    ZBarSymbol.CODE93,
                    ZBarSymbol.DATABAR,
                    ZBarSymbol.PDF417
                ]

                for symbol in barcode_symbols:
                    try:
                        decoded_specific = pyzbar.decode(image, symbols=[symbol])
                        for obj in decoded_specific:
                            data = obj.data.decode('utf-8', errors='ignore')
                            if data and not any(d.data == data for d in detections):
                                detections.append(BarcodeDetection(
                                    data=data,
                                    barcode_type=str(obj.type),
                                    method=f'pyzbar_{symbol.name}',
                                    preprocessing=prep_name,
                                    position=obj.rect,
                                    quality_score=0.8
                                ))
                    except:
                        pass

        except Exception as e:
            pass  # Silent fail for this preprocessing

        return detections

    def detect_with_opencv(self, image: np.ndarray, prep_name: str) -> List[BarcodeDetection]:
        """Detect barcodes using OpenCV barcode detector."""
        detections = []
        if not self.cv_barcode:
            return detections

        try:
            retval, decoded_info, decoded_type, points = self.cv_barcode.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for i, data in enumerate(decoded_info):
                    if data and len(data) > 0:
                        barcode_type = decoded_type[i] if i < len(decoded_type) else 'UNKNOWN'
                        # Skip if it's a QR code
                        if 'QR' not in barcode_type.upper():
                            detections.append(BarcodeDetection(
                                data=data,
                                barcode_type=barcode_type,
                                method='opencv_barcode',
                                preprocessing=prep_name,
                                position=points[i] if i < len(points) else None,
                                quality_score=0.7
                            ))
        except:
            pass

        return detections

    def detect_barcodes_robust(self, frame: np.ndarray) -> List[BarcodeDetection]:
        """Detect barcodes using sequential preprocessing (OAK-D style).

        Tries pyzbar with each preprocessing step until found,
        then tries OpenCV if pyzbar didn't find anything.
        """
        all_results = []

        # Preprocessing steps to try in order (stop when found)
        preprocessing_steps = [
            ('original', lambda f: f),
            ('grayscale', lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f),
            ('clahe', lambda f: self._apply_clahe(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
            ('bilateral_filter', lambda f: cv2.bilateralFilter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 11, 17, 17)),
            ('binary_threshold', lambda f: cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 127, 255, cv2.THRESH_BINARY)[1]),
            ('adaptive_threshold', lambda f: cv2.adaptiveThreshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('sharpening', lambda f: self._apply_sharpening(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
        ]

        print(f"\nTrying sequential preprocessing (stops when found)...")

        # PHASE 1: Try pyzbar with each preprocessing step (STOP when found)
        if PYZBAR_AVAILABLE:
            for i, (preproc_name, preproc_func) in enumerate(preprocessing_steps):
                try:
                    processed = preproc_func(frame)
                except Exception as e:
                    print(f"  [{i+1}/7] {preproc_name}: preprocessing failed")
                    continue

                # Try pyzbar detection
                detections = self.detect_with_pyzbar(processed, preproc_name)

                if detections:
                    print(f"  [{i+1}/7] {preproc_name}: ✓ FOUND {len(detections)} barcode(s) with pyzbar")
                    all_results.extend(detections)
                    break  # STOP - we found something!
                else:
                    print(f"  [{i+1}/7] {preproc_name}: ✗ not found")

        # PHASE 2: If pyzbar didn't find anything, try OpenCV sequentially
        if not all_results:
            print("\npyzbar found nothing, trying OpenCV QR detector...")
            for i, (preproc_name, preproc_func) in enumerate(preprocessing_steps):
                try:
                    processed = preproc_func(frame)
                except Exception as e:
                    print(f"  [{i+1}/7] {preproc_name}: preprocessing failed")
                    continue

                # Try OpenCV detection
                detections = self.detect_with_opencv(processed, preproc_name)

                if detections:
                    print(f"  [{i+1}/7] {preproc_name}: ✓ FOUND {len(detections)} code(s) with OpenCV")
                    all_results.extend(detections)
                    break  # STOP - we found something!
                else:
                    print(f"  [{i+1}/7] {preproc_name}: ✗ not found")

        # Deduplicate by data content
        unique_results = []
        seen_data = set()
        for detection in all_results:
            if detection.data not in seen_data:
                unique_results.append(detection)
                seen_data.add(detection.data)

        return unique_results


class CV60BarcodeScanner:
    """Main application for CV60 barcode scanning."""

    def __init__(self, camera_ip: str = "192.168.1.21"):
        self.camera_ip = camera_ip
        self.camera = None
        self.detector = BarcodeFocusedDetector()

        # Tracking
        self.frame_count = 0
        self.capture_count = 0
        self.current_frame = None
        self.unique_barcodes = {}  # data -> first detection info

        # FPS
        self.fps_counter = []
        self.current_fps = 0

        # Display
        self.show_feedback = False
        self.feedback_time = 0
        self.last_detections = []

    def initialize_camera(self) -> bool:
        """Initialize CV60 camera."""
        if not CV60_MODULE_AVAILABLE:
            print("[ERROR] CV60 module not available")
            return False

        try:
            print(f"\nConnecting to CV60 camera at {self.camera_ip}...")
            self.camera = CV60Camera(camera_ip=self.camera_ip)

            if self.camera.initialize():
                # Get camera info
                print(f"✓ Connected to CV60 camera")
                print(f"  Resolution: {int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                print("✓ Camera acquisition started")
                return True
            else:
                print(f"[ERROR] Failed to connect to CV60 at {self.camera_ip}")
                return False

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def capture_and_detect(self):
        """Capture current frame and detect barcodes."""
        if self.current_frame is None:
            print("\n[ERROR] No frame available")
            return

        # Capture
        frame = self.current_frame.copy()
        self.capture_count += 1
        timestamp = datetime.now()

        # Create directory
        test_dir = "test_captured_frames"
        os.makedirs(test_dir, exist_ok=True)

        # Save original
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"barcode_{timestamp_str}_frame{self.frame_count}"
        original_path = os.path.join(test_dir, f"{base_filename}_original.png")
        cv2.imwrite(original_path, frame)

        print("\n" + "="*80)
        print(f"BARCODE SCAN #{self.capture_count}")
        print("="*80)
        print(f"Time: {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"Frame: {self.frame_count}")
        print(f"Saved: {original_path}")

        # Image info
        if len(frame.shape) == 3:
            h, w, c = frame.shape
            print(f"Image: {w}x{h}, {c} channels")
        else:
            h, w = frame.shape
            print(f"Image: {w}x{h}, grayscale")

        # Quality check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Focus score: {blur_score:.1f} {'(sharp)' if blur_score > 100 else '(blurry)'}")

        print("\nSCANNING FOR BARCODES...")
        print("-"*80)

        # Detect using sequential approach (like OAK-D)
        detection_start = time.time()
        detections = self.detector.detect_barcodes_robust(frame)
        detection_time = (time.time() - detection_start) * 1000  # ms

        # Results
        print("\n" + "-"*80)
        print(f"⏱️  TOTAL INFERENCE TIME: {detection_time:.2f}ms")
        print("-"*80)

        if detections:
            print(f"✓ FOUND {len(detections)} BARCODE(S):\n")

            for i, detection in enumerate(detections, 1):
                print(f"  [{i}] Data: '{detection.data}'")
                print(f"      Type: {detection.barcode_type}")
                print(f"      Method: {detection.method}")
                print(f"      Preprocessing: {detection.preprocessing}")

                # Add to unique barcodes
                if detection.data not in self.unique_barcodes:
                    self.unique_barcodes[detection.data] = {
                        'type': detection.barcode_type,
                        'first_seen': timestamp.isoformat(),
                        'frame': self.frame_count,
                        'method': detection.method,
                        'preprocessing': detection.preprocessing
                    }
                print()

            # Save results
            results_file = os.path.join(test_dir, f"{base_filename}_results.json")
            results_data = {
                'timestamp': timestamp.isoformat(),
                'frame': self.frame_count,
                'detection_time_ms': round(detection_time, 2),
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
            print(f"Results saved: {results_file}")

        else:
            print("✗ NO BARCODES DETECTED")
            print("\nPossible issues:")

            if not PYZBAR_AVAILABLE:
                print("  • pyzbar not installed (CRITICAL for barcode detection)")

            if blur_score < 100:
                print("  • Image too blurry - barcode edges not clear")

            contrast = gray.std()
            if contrast < 30:
                print(f"  • Low contrast ({contrast:.1f}) - increase lighting")

            mean_brightness = gray.mean()
            if mean_brightness < 50:
                print(f"  • Too dark ({mean_brightness:.1f}) - increase exposure")
            elif mean_brightness > 200:
                print(f"  • Too bright ({mean_brightness:.1f}) - reduce exposure")

            print("\nTips for better detection:")
            print("  • Ensure barcode is in focus")
            print("  • Barcode should be horizontal or vertical")
            print("  • Good, even lighting without glare")
            print("  • Barcode should fill reasonable portion of frame")

        print("\n" + "-"*80)
        print(f"Total unique barcodes in session: {len(self.unique_barcodes)}")
        if self.unique_barcodes:
            print("Scanned barcodes:")
            for code, info in list(self.unique_barcodes.items())[:5]:
                print(f"  • {code} ({info['type']})")
        print("="*80 + "\n")

        self.last_detections = detections
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
        cv2.putText(frame, f"Scans: {self.capture_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.unique_barcodes:
            cv2.putText(frame, f"Barcodes found: {len(self.unique_barcodes)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "BARCODE SCANNER", (frame.shape[1]//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'S' or SPACE to scan", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, "'q'=quit | 'c'=clear", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Scan feedback
        if self.show_feedback:
            elapsed = time.time() - self.feedback_time
            if elapsed < 1.0:
                if self.last_detections:
                    color = (0, 255, 0)
                    text = f"FOUND: {self.last_detections[0].data}"
                else:
                    color = (0, 0, 255)
                    text = "NO BARCODE"

                alpha = 1 - elapsed
                color = tuple(int(c * alpha) for c in color)
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 3)
                cv2.putText(frame, text, (frame.shape[1]//2 - 150, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                self.show_feedback = False

        # Crosshair for alignment
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2 - 50, h//2), (w//2 + 50, h//2), (0, 255, 0), 1)
        cv2.line(frame, (w//2, h//2 - 50), (w//2, h//2 + 50), (0, 255, 0), 1)

        return frame

    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            print("[ERROR] Failed to initialize camera")
            return

        print("\n" + "="*80)
        print("CV60 BARCODE SCANNER - 30 FPS")
        print("="*80)
        print("Optimized for linear barcodes (Code128, Code39, EAN, etc.)")
        print("\nCONTROLS:")
        print("  'S' or SPACEBAR = Scan for barcodes")
        print("  '+' = Scan for barcodes (also works)")
        print("  'q' or ESC = Quit")
        print("  'c' = Clear history")
        print("\nDETECTORS:")
        print(f"  • pyzbar: {PYZBAR_AVAILABLE} {'(PRIMARY)' if PYZBAR_AVAILABLE else '(INSTALL REQUIRED!)'}")
        print(f"  • OpenCV: {self.detector.cv_barcode is not None} (backup)")
        print("="*80 + "\n")

        # Create directories
        os.makedirs("test_captured_frames", exist_ok=True)

        # Window
        window_name = "CV60 Barcode Scanner"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        try:
            frame_time = 1.0 / 30.0
            last_frame_time = time.time()

            while True:
                current_time = time.time()

                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_count += 1
                self.current_frame = frame

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
                elif key == ord('s') or key == ord('S') or key == 32 or key == ord('+') or key == ord('='):  # s, S, SPACE, +, =
                    self.capture_and_detect()
                elif key == ord('c') or key == ord('C'):  # clear history
                    self.unique_barcodes.clear()
                    print("[INFO] Cleared barcode history")

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
            print(f"Scans: {self.capture_count}")
            print(f"Unique barcodes: {len(self.unique_barcodes)}")
            if self.unique_barcodes:
                print("\nScanned barcodes:")
                for code, info in self.unique_barcodes.items():
                    print(f"  • {code} ({info['type']})")
            print("="*80)

            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CV60 Barcode Scanner')
    parser.add_argument('--ip', default='192.168.1.21', help='Camera IP')
    args = parser.parse_args()

    scanner = CV60BarcodeScanner(camera_ip=args.ip)
    scanner.run()


if __name__ == "__main__":
    main()