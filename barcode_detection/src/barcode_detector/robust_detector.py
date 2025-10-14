"""Importable robust barcode detector supporting OAK-D and CV60 cameras.

This module provides the same robust detection functionality as the standalone script,
but in a clean, importable API that can be used in other projects.

Example Usage:
    ```python
    from barcode_detector import RobustBarcodeDetector

    # Auto mode (tries OAK-D first, falls back to CV60)
    detector = RobustBarcodeDetector(camera_type='auto')
    detector.initialize_camera()

    # Get a frame and detect
    ret, frame = detector.camera.read()
    detections = detector.detect_barcodes_robust(frame)

    for det in detections:
        print(f"Found: {det['data']} - Type: {det['type']}")

    detector.cleanup()
    ```

    Or use specific camera:
    ```python
    # Use CV60 only
    detector = RobustBarcodeDetector(camera_type='cv60', camera_ip='192.168.1.21')

    # Use OAK-D only
    detector = RobustBarcodeDetector(camera_type='oakd', camera_ip='169.254.1.222')
    ```
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Literal
import time
from datetime import datetime
import subprocess
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="A NumPy version")

# Import camera modules
try:
    from .core.cv60_camera import create_cv60_camera
    from .core.oak_d_camera import create_oak_d_camera, DEPTHAI_AVAILABLE
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from barcode_detector.core.cv60_camera import create_cv60_camera
    from barcode_detector.core.oak_d_camera import create_oak_d_camera, DEPTHAI_AVAILABLE

# Try to import detection libraries
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    pass

# Try to import psutil for monitoring
PSUTIL_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass


def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.cpu_percent(interval=0.1)
    return 0.0


def get_gpu_usage() -> tuple:
    """Get GPU usage (utilization%, memory_used_MB, memory_total_MB)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(',')
            return (float(values[0].strip()), float(values[1].strip()), float(values[2].strip()))
    except:
        pass
    return (0.0, 0.0, 0.0)


class RobustBarcodeDetector:
    """Robust barcode detector with dual camera support (OAK-D and CV60).

    This class provides comprehensive barcode detection using multiple methods:
    - pyzbar (primary detector for all barcode types)
    - OpenCV QR detector (fallback for QR codes)
    - Multiple preprocessing strategies (CLAHE, filters, thresholding, etc.)

    Attributes:
        camera_type_requested (str): Type of camera requested ('oakd', 'cv60', 'auto')
        camera_type_actual (str): Actual camera type initialized
        camera: Camera instance (OAK-D or CV60)
        debug_mode (bool): Enable debug output

    Example:
        >>> detector = RobustBarcodeDetector(camera_type='auto')
        >>> if detector.initialize_camera():
        ...     ret, frame = detector.camera.read()
        ...     results = detector.detect_barcodes_robust(frame)
        ...     print(f"Found {len(results)} barcodes")
        ...     detector.cleanup()
    """

    def __init__(
        self,
        camera_type: Literal['oakd', 'cv60', 'auto'] = 'auto',
        camera_ip: Optional[str] = None,
        debug_mode: bool = False,
        cache_dir: Optional[str] = None
    ):
        """Initialize the robust barcode detector.

        Args:
            camera_type: Camera to use ('oakd', 'cv60', or 'auto' for automatic selection)
            camera_ip: IP address for camera (auto-detected if None)
            debug_mode: Enable debug output and timing information
            cache_dir: Directory to save detected frames (default: project_root/cached_detections)
        """
        self.camera = None
        self.detected_codes = {}
        self.debug_mode = debug_mode
        self.camera_type_requested = camera_type
        self.camera_type_actual = None
        self.frame_count = 0
        self.detection_count = 0

        # Set camera IP based on type
        if camera_ip:
            self.camera_ip = camera_ip
        elif camera_type == 'oakd':
            self.camera_ip = "169.254.1.222"  # Default OAK-D PoE IP
        elif camera_type == 'cv60':
            self.camera_ip = "192.168.1.21"  # Default CV60 IP
        else:  # auto
            self.camera_ip = "169.254.1.222"  # Try OAK-D first

        # Setup cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(script_dir, '..', '..')
            self.cache_dir = os.path.abspath(os.path.join(project_root, "cached_detections"))
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize OpenCV detectors
        self.qr_detector = cv2.QRCodeDetector()
        try:
            self.qr_detector_advanced = cv2.QRCodeDetectorAruco()
        except:
            self.qr_detector_advanced = None

        if self.debug_mode:
            print(f"[INFO] RobustBarcodeDetector initialized")
            print(f"[INFO] Camera type: {camera_type}")
            print(f"[INFO] Camera IP: {self.camera_ip}")
            print(f"[INFO] Cache dir: {self.cache_dir}")
            print(f"[INFO] pyzbar available: {PYZBAR_AVAILABLE}")
            print(f"[INFO] DepthAI available: {DEPTHAI_AVAILABLE}")

    def initialize_camera(self) -> bool:
        """Initialize camera with automatic fallback.

        Priority order:
        - auto: OAK-D -> CV60 -> OpenCV fallback
        - oakd: OAK-D only
        - cv60: CV60 only

        Returns:
            bool: True if camera initialized successfully
        """
        try:
            # Auto mode: try OAK-D first, then CV60
            if self.camera_type_requested == 'auto':
                if DEPTHAI_AVAILABLE:
                    if self.debug_mode:
                        print(f"[INFO] Trying OAK-D at {self.camera_ip}...")
                    self.camera = create_oak_d_camera(
                        resolution_4k=True,
                        fps=30,
                        ip_address=self.camera_ip
                    )
                    if self.camera and self.camera.isOpened():
                        self.camera_type_actual = "OAK-D PoE"
                        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[SUCCESS] OAK-D initialized: {width}x{height}")
                        return True

                # Fallback to CV60
                if self.debug_mode:
                    print("[INFO] Trying CV60...")
                self.camera = create_cv60_camera()
                if self.camera and self.camera.isOpened():
                    self.camera_type_actual = "CV60"
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[SUCCESS] CV60 initialized: {width}x{height}")
                    return True

            # OAK-D only mode
            elif self.camera_type_requested == 'oakd':
                if not DEPTHAI_AVAILABLE:
                    print("[ERROR] DepthAI not available. Install: pip install depthai==2.24.0.0")
                    return False

                self.camera = create_oak_d_camera(
                    resolution_4k=True,
                    fps=30,
                    ip_address=self.camera_ip
                )
                if self.camera and self.camera.isOpened():
                    self.camera_type_actual = "OAK-D PoE"
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[SUCCESS] OAK-D initialized: {width}x{height}")
                    return True

            # CV60 only mode
            elif self.camera_type_requested == 'cv60':
                self.camera = create_cv60_camera()
                if self.camera and self.camera.isOpened():
                    self.camera_type_actual = "CV60"
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[SUCCESS] CV60 initialized: {width}x{height}")
                    return True

            print("[ERROR] Failed to initialize any camera")
            return False

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return False

    def detect_with_pyzbar(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect barcodes using pyzbar library."""
        results = []
        if not PYZBAR_AVAILABLE:
            return results

        try:
            frames_to_try = [frame]
            if len(frame.shape) == 3:
                frames_to_try.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            for test_frame in frames_to_try:
                decoded_objects = pyzbar.decode(test_frame)
                for obj in decoded_objects:
                    try:
                        barcode_data = obj.data.decode('utf-8')
                    except:
                        barcode_data = str(obj.data)

                    results.append({
                        'data': barcode_data,
                        'type': obj.type,
                        'method': 'pyzbar',
                        'points': obj.polygon
                    })

                if results:
                    break

        except Exception as e:
            if self.debug_mode:
                print(f"[DEBUG] pyzbar error: {e}")

        return results

    def detect_with_opencv(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect QR codes using OpenCV."""
        results = []

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            # Try basic QR detector
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)
            if data:
                results.append({
                    'data': data,
                    'type': 'QRCODE',
                    'method': 'opencv_qr',
                    'bbox': bbox
                })

            # Try advanced QR detector
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
                print(f"[DEBUG] OpenCV error: {e}")

        return results

    def _apply_clahe(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to grayscale frame."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_frame)

    def _apply_sharpening(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply sharpening kernel."""
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        return cv2.filter2D(gray_frame, -1, kernel)

    def detect_barcodes_robust(
        self,
        frame: np.ndarray,
        timing_breakdown: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """Detect barcodes using multiple methods and preprocessing strategies.

        This is the main detection method. It tries pyzbar first with various
        preprocessing filters, then OpenCV QR if nothing is found.

        Args:
            frame: Input frame (BGR or grayscale)
            timing_breakdown: Optional dict to store timing information

        Returns:
            List of detected barcodes with data, type, method, and preprocessing info
        """
        all_results = []

        if timing_breakdown is not None:
            timing_breakdown['pyzbar_attempts'] = []
            timing_breakdown['opencv_attempts'] = []
            timing_breakdown['total_preprocessing'] = 0.0

        # Preprocessing steps to try
        preprocessing_steps = [
            ('original', lambda f: f),
            ('grayscale', lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f),
            ('clahe', lambda f: self._apply_clahe(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
            ('bilateral_filter', lambda f: cv2.bilateralFilter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 11, 17, 17)),
            ('binary_threshold', lambda f: cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 127, 255, cv2.THRESH_BINARY)[1]),
            ('adaptive_threshold', lambda f: cv2.adaptiveThreshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('sharpening', lambda f: self._apply_sharpening(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
        ]

        # PHASE 1: Try pyzbar with all preprocessing
        if PYZBAR_AVAILABLE:
            for i, (preproc_name, preproc_func) in enumerate(preprocessing_steps):
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
                    for r in results:
                        r['preprocessing'] = i
                        r['preprocessing_name'] = preproc_name
                    all_results.extend(results)
                    break

        # PHASE 2: Try OpenCV if pyzbar found nothing
        if not all_results:
            for i, (preproc_name, preproc_func) in enumerate(preprocessing_steps):
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
                    for r in results:
                        r['preprocessing'] = i
                        r['preprocessing_name'] = preproc_name
                    all_results.extend(results)
                    break

        # Remove duplicates
        unique_results = {}
        for result in all_results:
            data = result['data']
            if data not in unique_results:
                unique_results[data] = result

        return list(unique_results.values())

    def detect_single_frame(self, frame: np.ndarray, verbose: bool = False) -> List[Dict[str, Any]]:
        """Convenience method to detect barcodes in a single frame.

        Args:
            frame: Input frame
            verbose: Print detection results

        Returns:
            List of detected barcodes
        """
        timing = {} if verbose else None
        results = self.detect_barcodes_robust(frame, timing)

        if verbose:
            if results:
                print(f"✓ Found {len(results)} barcode(s):")
                for r in results:
                    print(f"  - {r['type']}: {r['data']}")
                    print(f"    Method: {r['method']}, Preprocessing: {r.get('preprocessing_name', 'N/A')}")
            else:
                print("✗ No barcodes detected")

            if timing:
                total_time = sum(t for k, t in timing.items() if k.startswith('pyzbar_') or k.startswith('opencv_'))
                print(f"⏱ Detection time: {total_time:.2f}ms")

        return results

    def cleanup(self):
        """Release camera resources."""
        if self.camera:
            self.camera.release()
            self.camera = None
            if self.debug_mode:
                print("[INFO] Camera released")

    def __enter__(self):
        """Context manager entry."""
        self.initialize_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
