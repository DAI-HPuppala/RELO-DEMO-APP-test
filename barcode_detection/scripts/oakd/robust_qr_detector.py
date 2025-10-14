#!/usr/bin/env python3
"""Robust QR/Barcode detector for CV60 camera with multiple detection methods."""

import sys
import os
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
import time
from datetime import datetime
import subprocess

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Suppress scipy warning
import warnings
warnings.filterwarnings("ignore", message="A NumPy version")

from barcode_detector.core.cv60_camera import create_cv60_camera
from barcode_detector.core.oak_d_camera import create_oak_d_camera, DEPTHAI_AVAILABLE

# Try to import detection libraries
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    print("[INFO] pyzbar library available")
except ImportError:
    print("[WARNING] pyzbar not available - install with: pip install pyzbar")
    print("[WARNING] Also need libzbar0: sudo apt-get install libzbar0")

# Try to import psutil for CPU monitoring
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    print("[WARNING] psutil not available - install with: pip install psutil")


def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.cpu_percent(interval=0.1)
    return 0.0


def get_gpu_usage() -> tuple:
    """Get GPU usage (utilization%, memory_used_MB, memory_total_MB).

    Returns:
        tuple: (gpu_util%, memory_used_MB, memory_total_MB) or (0, 0, 0) if unavailable
    """
    try:
        # Try nvidia-smi first
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(',')
            gpu_util = float(values[0].strip())
            mem_used = float(values[1].strip())
            mem_total = float(values[2].strip())
            return (gpu_util, mem_used, mem_total)
    except:
        pass

    return (0.0, 0.0, 0.0)


class RobustBarcodeDetector:
    """Robust barcode detector with multiple detection strategies."""

    def __init__(self, debug_mode: bool = True, use_oak_d: bool = True, oak_d_ip: str = "169.254.1.222"):
        self.camera = None
        self.detected_codes = {}  # Track detected codes with timestamps
        self.debug_mode = debug_mode
        self.use_oak_d = use_oak_d
        self.oak_d_ip = oak_d_ip  # IP address for OAK-D PoE camera
        self.camera_type = None  # Will be set during initialization
        self.frame_count = 0
        self.detection_count = 0

        # Create cache directory for saved frames
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cached_detections")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Focus control
        self.current_focus = 128  # Default focus value (0-255)
        self.saved_focus = None  # Saved/locked focus value
        self.focus_mode = "auto"  # "auto" or "manual"

        # Button positions (will be set in run method)
        self.save_focus_button = None
        self.autofocus_button = None

        # Initialize OpenCV detectors
        self.qr_detector = cv2.QRCodeDetector()

        # Try to create QR detector with better parameters
        try:
            self.qr_detector_advanced = cv2.QRCodeDetectorAruco()
        except:
            self.qr_detector_advanced = None

        print("[INFO] Robust Barcode Detector initialized")
        print(f"[INFO] Debug mode: {self.debug_mode}")
        print(f"[INFO] Prefer OAK-D camera: {self.use_oak_d}")
        if self.use_oak_d:
            print(f"[INFO] OAK-D PoE IP: {self.oak_d_ip}")
        print(f"[INFO] Cache directory: {self.cache_dir}")

    def initialize_camera(self) -> bool:
        """Initialize camera with priority: OAK-D > CV60 > OpenCV fallback."""
        try:
            # Try OAK-D camera first if enabled
            if self.use_oak_d and DEPTHAI_AVAILABLE:
                print(f"[INFO] Attempting to initialize OAK-D PoE camera at {self.oak_d_ip} with 4K resolution...")
                self.camera = create_oak_d_camera(resolution_4k=True, fps=30, ip_address=self.oak_d_ip)
                if self.camera and self.camera.isOpened():
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.camera_type = f"OAK-D PoE"
                    print(f"[SUCCESS] OAK-D PoE camera initialized: {width}x{height}")
                    print(f"[INFO] Connected to: {self.oak_d_ip}")
                    print(f"[INFO] Camera features: 4K resolution, Autofocus (CONTINUOUS_VIDEO), Auto-exposure")
                    return True
                else:
                    print("[WARNING] OAK-D initialization failed, falling back to CV60...")

            # Fall back to CV60 camera
            print("[INFO] Initializing CV60 camera...")
            self.camera = create_cv60_camera()
            if self.camera and self.camera.isOpened():
                width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.camera_type = "CV60"
                print(f"[INFO] CV60 camera initialized: {width}x{height}")
                return True
            else:
                print("[ERROR] Failed to initialize any camera")
                return False
        except Exception as e:
            print(f"[ERROR] Camera initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect_with_pyzbar(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect barcodes using pyzbar library."""
        results = []
        if not PYZBAR_AVAILABLE:
            return results

        try:
            # Try different frame formats
            frames_to_try = []

            # Original frame
            frames_to_try.append(frame)

            # If color, also try grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_to_try.append(gray)

            for test_frame in frames_to_try:
                decoded_objects = pyzbar.decode(test_frame)
                for obj in decoded_objects:
                    try:
                        barcode_data = obj.data.decode('utf-8')
                    except:
                        barcode_data = str(obj.data)

                    barcode_type = obj.type

                    # Get bounding box
                    points = obj.polygon
                    if points:
                        results.append({
                            'data': barcode_data,
                            'type': barcode_type,
                            'method': 'pyzbar',
                            'points': points
                        })

                if results:  # If we found something, stop trying
                    break

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] pyzbar error: {e}")

        return results

    def detect_with_opencv(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect QR codes using OpenCV."""
        results = []

        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Try basic QR detector
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)
            if data:
                results.append({
                    'data': data,
                    'type': 'QRCODE',
                    'method': 'opencv_qr',
                    'bbox': bbox
                })

            # Try advanced QR detector if available
            if self.qr_detector_advanced and not results:
                try:
                    data, bbox, _ = self.qr_detector_advanced.detectAndDecode(gray)
                    if data:
                        results.append({
                            'data': data,
                            'type': 'QRCODE',
                            'method': 'opencv_qr_aruco',
                            'bbox': bbox
                        })
                except:
                    pass

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] OpenCV QR error: {e}")

        return results

    def detect_barcodes_robust(self, frame: np.ndarray, timing_breakdown: dict = None) -> List[Dict[str, Any]]:
        """Detect barcodes using multiple methods and preprocessing.

        First tries pyzbar with all preprocessing filters sequentially,
        then tries OpenCV with all preprocessing filters sequentially.
        """
        all_results = []

        # Initialize detection timing
        if timing_breakdown is not None:
            timing_breakdown['pyzbar_attempts'] = []
            timing_breakdown['opencv_attempts'] = []
            timing_breakdown['total_preprocessing'] = 0.0

        # Preprocessing steps to try in order
        preprocessing_steps = [
            ('original', lambda f: f),
            ('grayscale', lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)),
            ('clahe', lambda f: self._apply_clahe(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
            ('bilateral_filter', lambda f: cv2.bilateralFilter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 11, 17, 17)),
            ('binary_threshold', lambda f: cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 127, 255, cv2.THRESH_BINARY)[1]),
            ('adaptive_threshold', lambda f: cv2.adaptiveThreshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('sharpening', lambda f: self._apply_sharpening(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
        ]

        # PHASE 1: Try pyzbar with ALL preprocessing steps sequentially
        if PYZBAR_AVAILABLE:
            for i, (preproc_name, preproc_func) in enumerate(preprocessing_steps):
                # Apply preprocessing
                t_preproc = time.time()
                try:
                    processed = preproc_func(frame)
                except Exception as e:
                    if self.debug_mode:
                        print(f"[DEBUG] Preprocessing {preproc_name} failed: {e}")
                    continue

                preproc_time = (time.time() - t_preproc) * 1000

                if timing_breakdown is not None:
                    timing_breakdown[f'pyzbar_{preproc_name}'] = preproc_time
                    timing_breakdown['total_preprocessing'] += preproc_time

                # Try pyzbar detection
                t0 = time.time()
                results = self.detect_with_pyzbar(processed)
                pyzbar_time = (time.time() - t0) * 1000

                if timing_breakdown is not None:
                    timing_breakdown['pyzbar_attempts'].append({
                        'preprocessing': preproc_name,
                        'time_ms': pyzbar_time,
                        'found': len(results) > 0
                    })

                if results:
                    # pyzbar decoded it! Add results and stop completely
                    for r in results:
                        r['preprocessing'] = i
                        r['preprocessing_name'] = preproc_name
                    all_results.extend(results)
                    break  # Stop trying more preprocessing steps

        # PHASE 2: If pyzbar didn't find anything, try OpenCV with ALL preprocessing steps sequentially
        if not all_results:
            for i, (preproc_name, preproc_func) in enumerate(preprocessing_steps):
                # Apply preprocessing
                t_preproc = time.time()
                try:
                    processed = preproc_func(frame)
                except Exception as e:
                    if self.debug_mode:
                        print(f"[DEBUG] Preprocessing {preproc_name} failed: {e}")
                    continue

                preproc_time = (time.time() - t_preproc) * 1000

                if timing_breakdown is not None:
                    timing_breakdown[f'opencv_{preproc_name}'] = preproc_time
                    timing_breakdown['total_preprocessing'] += preproc_time

                # Try OpenCV detection
                t0 = time.time()
                results = self.detect_with_opencv(processed)
                opencv_time = (time.time() - t0) * 1000

                if timing_breakdown is not None:
                    timing_breakdown['opencv_attempts'].append({
                        'preprocessing': preproc_name,
                        'time_ms': opencv_time,
                        'found': len(results) > 0
                    })

                if results:
                    # OpenCV decoded it! Add results and stop completely
                    for r in results:
                        r['preprocessing'] = i
                        r['preprocessing_name'] = preproc_name
                    all_results.extend(results)
                    break  # Stop trying more preprocessing steps

        # Remove duplicates
        unique_results = {}
        for result in all_results:
            data = result['data']
            if data not in unique_results:
                unique_results[data] = result

        return list(unique_results.values())

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

    def on_focus_change(self, focus_value: int) -> None:
        """Callback for focus slider change."""
        self.current_focus = focus_value

        # Only apply if using OAK-D camera with manual focus capability
        if hasattr(self.camera, 'set_manual_focus'):
            self.camera.set_manual_focus(focus_value)
            if self.debug_mode:
                print(f"[DEBUG] Focus set to: {focus_value}")

    def save_focus_setting(self) -> None:
        """Save the current focus value and lock it."""
        self.saved_focus = self.current_focus
        self.focus_mode = "manual"

        # Apply manual focus to camera
        if hasattr(self.camera, 'set_manual_focus'):
            self.camera.set_manual_focus(self.saved_focus)
            print(f"[INFO] Focus saved and locked at: {self.saved_focus}")
        else:
            print("[WARNING] Camera does not support manual focus")

    def reset_autofocus(self) -> None:
        """Reset to autofocus mode."""
        self.focus_mode = "auto"
        self.saved_focus = None

        if hasattr(self.camera, 'set_auto_focus'):
            self.camera.set_auto_focus("CONTINUOUS_VIDEO")
            print("[INFO] Autofocus re-enabled (CONTINUOUS_VIDEO mode)")
        else:
            print("[WARNING] Camera does not support autofocus control")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on buttons."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if Save Focus button was clicked
            if self.save_focus_button:
                bx, by, bw, bh = self.save_focus_button
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.save_focus_setting()

            # Check if Autofocus button was clicked
            if self.autofocus_button:
                bx, by, bw, bh = self.autofocus_button
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self.reset_autofocus()

    def draw_focus_buttons(self, frame: np.ndarray) -> np.ndarray:
        """Draw focus control buttons on the frame."""
        frame_h, frame_w = frame.shape[:2]

        # Button properties
        button_height = 40
        button_width = 140
        margin = 10
        y_start = frame_h - 150  # Position near bottom

        # Save Focus button
        save_x = margin
        save_y = y_start
        self.save_focus_button = (save_x, save_y, button_width, button_height)

        # Determine button color based on focus mode
        save_color = (0, 180, 0) if self.focus_mode == "manual" else (100, 100, 100)
        cv2.rectangle(frame, (save_x, save_y),
                     (save_x + button_width, save_y + button_height),
                     save_color, -1)
        cv2.rectangle(frame, (save_x, save_y),
                     (save_x + button_width, save_y + button_height),
                     (255, 255, 255), 2)
        cv2.putText(frame, "SAVE FOCUS", (save_x + 10, save_y + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Autofocus button
        auto_x = save_x + button_width + margin
        auto_y = y_start
        self.autofocus_button = (auto_x, auto_y, button_width, button_height)

        auto_color = (0, 180, 0) if self.focus_mode == "auto" else (100, 100, 100)
        cv2.rectangle(frame, (auto_x, auto_y),
                     (auto_x + button_width, auto_y + button_height),
                     auto_color, -1)
        cv2.rectangle(frame, (auto_x, auto_y),
                     (auto_x + button_width, auto_y + button_height),
                     (255, 255, 255), 2)
        cv2.putText(frame, "AUTOFOCUS", (auto_x + 10, auto_y + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display current focus mode and value
        mode_text = f"Focus Mode: {self.focus_mode.upper()}"
        if self.focus_mode == "manual" and self.saved_focus is not None:
            mode_text += f" ({self.saved_focus})"
        else:
            mode_text += f" (slider: {self.current_focus})"

        cv2.putText(frame, mode_text, (margin, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def process_and_save_frame(self, frame: np.ndarray) -> None:
        """Process frame, detect barcodes, and save with annotations."""
        # Capture current focus settings
        if self.focus_mode == "manual" and self.saved_focus is not None:
            focus_value = self.saved_focus
            focus_info = f"Manual: {focus_value}"
        else:
            # Try to get actual lens position from camera
            actual_lens_pos = None
            if hasattr(self.camera, 'get_lens_position'):
                actual_lens_pos = self.camera.get_lens_position()

            if self.focus_mode == "auto" and actual_lens_pos is not None:
                focus_value = int(actual_lens_pos)
                focus_info = f"Auto: {focus_value}"
            elif self.focus_mode == "auto":
                focus_value = None
                focus_info = "Auto"
            else:
                focus_value = self.current_focus
                focus_info = f"Slider: {focus_value}"

        # Measure CPU/GPU usage before detection
        cpu_before = get_cpu_usage()
        gpu_before, gpu_mem_before, gpu_mem_total = get_gpu_usage()

        # Measure detection time with detailed breakdown
        timing_breakdown = {}
        detection_start = time.time()
        detections = self.detect_barcodes_robust(frame, timing_breakdown)
        detection_time = (time.time() - detection_start) * 1000  # Convert to ms

        # Measure CPU/GPU usage after detection
        cpu_after = get_cpu_usage()
        gpu_after, gpu_mem_after, _ = get_gpu_usage()

        # Calculate average utilization
        cpu_util = (cpu_before + cpu_after) / 2
        gpu_util = (gpu_before + gpu_after) / 2
        gpu_mem_used = gpu_mem_after

        # Create annotated frame
        annotated_frame = frame.copy()

        # Prepare result text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results_text = []

        if detections:
            # Annotate frame with detection results at top-left (no bounding boxes)
            y_offset = 40
            for i, detection in enumerate(detections):
                barcode_data = detection['data']
                barcode_type = detection.get('type', 'UNKNOWN')
                method = detection.get('method', 'unknown')

                # Draw text on image
                text = f"{barcode_type}: {barcode_data}"
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                y_offset += 50

                # Collect for terminal output
                results_text.append(f"  - Type: {barcode_type}")
                results_text.append(f"    Data: {barcode_data}")
                results_text.append(f"    Method: {method}")
        else:
            # No detection
            cv2.putText(annotated_frame, "NO BARCODE DETECTED", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Add inference time to image
        time_text = f"Inference: {detection_time:.1f}ms"
        cv2.putText(annotated_frame, time_text, (10, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Add CPU/GPU utilization to image
        util_y = annotated_frame.shape[0] - 60
        cpu_text = f"CPU: {cpu_util:.1f}%"
        cv2.putText(annotated_frame, cpu_text, (10, util_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if gpu_util > 0:
            gpu_text = f"GPU: {gpu_util:.1f}% | VRAM: {gpu_mem_used:.0f}MB"
            cv2.putText(annotated_frame, gpu_text, (10, util_y - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Add timestamp to image
        cv2.putText(annotated_frame, timestamp, (10, annotated_frame.shape[0] - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add focus information to image
        focus_text = f"Focus: {focus_info}"
        cv2.putText(annotated_frame, focus_text, (10, annotated_frame.shape[0] - 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save image
        filename = f"barcode_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = os.path.join(self.cache_dir, filename)
        cv2.imwrite(filepath, annotated_frame)

        # Print results to terminal
        print("\n" + "="*70)
        print(f"FRAME CAPTURED AND PROCESSED")
        print("="*70)
        print(f"Timestamp: {timestamp}")
        print(f"Camera Focus: {focus_info}")
        print(f"\n📊 TIMING BREAKDOWN:")
        print("-" * 70)

        # Preprocessing times
        print(f"\n🔧 PREPROCESSING (Total: {timing_breakdown.get('total_preprocessing', 0):.2f}ms):")
        preprocessing_steps = ['original', 'grayscale', 'clahe', 'bilateral_filter',
                              'binary_threshold', 'adaptive_threshold', 'sharpening']
        for step in preprocessing_steps:
            if step in timing_breakdown:
                print(f"  • {step:20s}: {timing_breakdown[step]:6.2f}ms")

        # Detection times
        print(f"\n🔍 DETECTION ATTEMPTS:")

        if 'pyzbar_attempts' in timing_breakdown:
            print(f"\n  pyzbar:")
            for attempt in timing_breakdown['pyzbar_attempts']:
                status = "✓ FOUND" if attempt['found'] else "✗ not found"
                print(f"    • {attempt['preprocessing']:20s}: {attempt['time_ms']:6.2f}ms  {status}")

        if 'opencv_attempts' in timing_breakdown:
            print(f"\n  OpenCV QR:")
            for attempt in timing_breakdown['opencv_attempts']:
                status = "✓ FOUND" if attempt['found'] else "✗ not found"
                print(f"    • {attempt['preprocessing']:20s}: {attempt['time_ms']:6.2f}ms  {status}")

        print(f"\n⏱️  TOTAL INFERENCE TIME: {detection_time:.2f}ms")
        print(f"💻 CPU utilization: {cpu_util:.1f}%")
        if gpu_util > 0:
            print(f"🎮 GPU utilization: {gpu_util:.1f}%")
            print(f"💾 GPU memory: {gpu_mem_used:.0f}MB / {gpu_mem_total:.0f}MB")

        print(f"\n💾 Saved to: {filepath}")

        if detections:
            print(f"\n✅ DETECTED {len(detections)} BARCODE(S):")
            for det in detections:
                print(f"  • Type: {det.get('type', 'UNKNOWN')}")
                print(f"    Data: {det['data']}")
                print(f"    Method: {det.get('method', 'unknown')}")
                print(f"    Preprocessing: {det.get('preprocessing_name', 'unknown')}")
        else:
            print("\n❌ NO BARCODES DETECTED")

        print("="*70 + "\n")
        sys.stdout.flush()

    def draw_detection(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection results on frame."""
        display_frame = frame.copy()

        for detection in detections:
            # Draw bounding box if available
            if 'points' in detection:  # pyzbar format
                points = detection['points']
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            elif 'bbox' in detection and detection['bbox'] is not None:  # OpenCV format
                bbox = detection['bbox']
                if bbox is not None and len(bbox) > 0:
                    pts = np.array(bbox, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

            # Add text label (only data value)
            y_offset = 30
            cv2.putText(display_frame, detection['data'],
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)

        return display_frame

    def run(self):
        """Run the robust QR detector."""
        if not self.initialize_camera():
            print("[ERROR] Could not initialize camera")
            return

        # Create window for live feed
        window_name = "Robust QR/Barcode Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        # Create focus control trackbar (0-255)
        # Note: OpenCV trackbar callbacks don't support instance methods directly,
        # so we'll handle the value in the main loop
        cv2.createTrackbar("Focus (0=far, 255=near)", window_name,
                          self.current_focus, 255, lambda x: None)

        # Set mouse callback for button clicks
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "="*60)
        print("ROBUST QR/BARCODE DETECTOR RUNNING")
        print("="*60)
        print("Press 'S' or SPACEBAR to SAVE frame and detect barcodes")
        print("Press 'F' to SAVE FOCUS (lock current focus value)")
        print("Press 'A' to enable AUTOFOCUS")
        print("Press 'q' or ESC to quit")
        print("Press 'd' to toggle debug mode")
        print("Press 'c' to clear detected codes")
        print("\nFocus Control:")
        print("  - Use slider to adjust focus manually")
        print("  - Press 'F' to lock focus at current slider position")
        print("  - Press 'A' to return to autofocus mode")
        print("="*60 + "\n")

        last_detection_time = time.time()
        fps_time = time.time()
        fps_frame_count = 0
        current_fps = 0

        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    continue

                self.frame_count += 1
                fps_frame_count += 1

                # Calculate FPS
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    current_fps = fps_frame_count
                    fps_frame_count = 0
                    fps_time = current_time

                # Just display the frame (no continuous detection for better FPS)
                display_frame = frame.copy()

                # Handle focus slider changes
                trackbar_focus = cv2.getTrackbarPos("Focus (0=far, 255=near)", window_name)
                if trackbar_focus != self.current_focus and self.focus_mode != "manual":
                    self.on_focus_change(trackbar_focus)

                # Draw ROI (Region of Interest) box for barcode placement guide
                frame_h, frame_w = display_frame.shape[:2]
                roi_width = int(frame_w * 0.6)  # 60% of frame width
                roi_height = int(frame_h * 0.4)  # 40% of frame height
                roi_x1 = (frame_w - roi_width) // 2
                roi_y1 = (frame_h - roi_height) // 2
                roi_x2 = roi_x1 + roi_width
                roi_y2 = roi_y1 + roi_height

                # Draw ROI rectangle with corners
                roi_color = (0, 255, 255)  # Yellow
                roi_thickness = 3
                corner_length = 50

                # Top-left corner
                cv2.line(display_frame, (roi_x1, roi_y1), (roi_x1 + corner_length, roi_y1), roi_color, roi_thickness)
                cv2.line(display_frame, (roi_x1, roi_y1), (roi_x1, roi_y1 + corner_length), roi_color, roi_thickness)

                # Top-right corner
                cv2.line(display_frame, (roi_x2, roi_y1), (roi_x2 - corner_length, roi_y1), roi_color, roi_thickness)
                cv2.line(display_frame, (roi_x2, roi_y1), (roi_x2, roi_y1 + corner_length), roi_color, roi_thickness)

                # Bottom-left corner
                cv2.line(display_frame, (roi_x1, roi_y2), (roi_x1 + corner_length, roi_y2), roi_color, roi_thickness)
                cv2.line(display_frame, (roi_x1, roi_y2), (roi_x1, roi_y2 - corner_length), roi_color, roi_thickness)

                # Bottom-right corner
                cv2.line(display_frame, (roi_x2, roi_y2), (roi_x2 - corner_length, roi_y2), roi_color, roi_thickness)
                cv2.line(display_frame, (roi_x2, roi_y2), (roi_x2, roi_y2 - corner_length), roi_color, roi_thickness)

                # Add instruction text
                instruction_text = "Place barcode in frame and press 'S' to scan"
                text_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (frame_w - text_size[0]) // 2
                text_y = roi_y1 - 20
                cv2.putText(display_frame, instruction_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

                # Draw focus control buttons
                display_frame = self.draw_focus_buttons(display_frame)

                # Add status info in debug mode
                if self.debug_mode:
                    # Add camera type
                    if self.camera_type:
                        cv2.putText(display_frame, f"Camera: {self.camera_type}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Add FPS counter
                    cv2.putText(display_frame, f"FPS: {current_fps}",
                               (display_frame.shape[1] - 100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Add frame counter
                    cv2.putText(display_frame, f"Frame: {self.frame_count}",
                               (display_frame.shape[1] - 150, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Add detection counter
                    cv2.putText(display_frame, f"Detections: {self.detection_count}",
                               (display_frame.shape[1] - 180, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    # Add detection methods available
                    methods = []
                    if PYZBAR_AVAILABLE:
                        methods.append("pyzbar")
                    methods.append("opencv")
                    cv2.putText(display_frame, f"Methods: {', '.join(methods)}",
                               (10, display_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Show live feed
                cv2.imshow(window_name, display_frame)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('s') or key == ord('S') or key == 32:  # 's', 'S', or spacebar
                    # Save and process frame
                    print("[INFO] Capturing frame for barcode detection...")
                    self.process_and_save_frame(frame)
                elif key == ord('f') or key == ord('F'):
                    # Save/lock focus
                    self.save_focus_setting()
                elif key == ord('a') or key == ord('A'):
                    # Enable autofocus
                    self.reset_autofocus()
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"[INFO] Debug mode: {self.debug_mode}")
                elif key == ord('c'):
                    self.detected_codes.clear()
                    print("[INFO] Cleared detected codes history")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            print(f"\n[INFO] Total frames processed: {self.frame_count}")
            print(f"[INFO] Total barcodes detected: {self.detection_count}")

            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Robust QR/Barcode Detector')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug mode')
    parser.add_argument('--no-oak-d', action='store_true',
                       help='Disable OAK-D camera (use CV60 instead)')
    parser.add_argument('--oak-ip', default='169.254.1.222',
                       help='OAK-D PoE camera IP address (default: 169.254.1.222)')
    args = parser.parse_args()

    detector = RobustBarcodeDetector(
        debug_mode=not args.no_debug,
        use_oak_d=not args.no_oak_d,
        oak_d_ip=args.oak_ip
    )
    detector.run()


if __name__ == "__main__":
    main()