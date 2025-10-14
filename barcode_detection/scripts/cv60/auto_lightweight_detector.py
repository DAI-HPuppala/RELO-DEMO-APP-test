#!/usr/bin/env python3
"""Lightweight Auto-Detecting Barcode Scanner.

Uses two-stage detection:
1. Fast lightweight gradient-based barcode presence detection (~2-5ms per frame)
2. Heavy pyzbar decoding only when barcode detected (async, non-blocking)

This approach minimizes CPU/GPU overhead while maintaining high detection rates.
"""

import sys
import os
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json

# Setup paths
EBUS_SDK_PATH = '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'
if EBUS_SDK_PATH not in sys.path:
    sys.path.insert(0, EBUS_SDK_PATH)

lib_paths = [
    '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib',
    '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/genicam/bin/Linux64_x64'
]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths + [current_ld_path])

# Import camera modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

CV60_MODULE_AVAILABLE = False
try:
    from barcode_detector.core.cv60_camera import CV60Camera
    from barcode_detector.core.oak_d_camera import create_oak_d_camera, DEPTHAI_AVAILABLE
    from barcode_detector.lightweight import BarcodePresenceDetector
    CV60_MODULE_AVAILABLE = True
    print("✓ Camera and lightweight detection modules loaded")
except ImportError as e:
    print(f"[ERROR] Could not import modules: {e}")
    sys.exit(1)

# Import pyzbar
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    print("✓ pyzbar loaded - DECODER")
except ImportError:
    print("✗ pyzbar not available!")


@dataclass
class BarcodeDetection:
    """Barcode detection result."""
    data: str
    barcode_type: str
    method: str
    preprocessing: str
    timestamp: float = 0.0
    detection_time_ms: float = 0.0


class LightweightAutoBarcodeScanner:
    """Auto-scanner with lightweight presence detection + async heavy decoding."""

    def __init__(self, camera_type: str = "cv60", camera_ip: str = "192.168.1.21",
                 oak_d_ip: str = "169.254.1.222"):
        self.camera_type = camera_type
        self.camera_ip = camera_ip
        self.oak_d_ip = oak_d_ip
        self.camera = None

        # Lightweight presence detector (fast!)
        self.presence_detector = BarcodePresenceDetector(
            min_bar_width=2,
            max_bar_width=10,
            min_bars=10,
            confidence_threshold=0.5
        )

        # Async decoding
        self.decode_queue = Queue(maxsize=2)  # Limit queue size
        self.decode_thread = None
        self.stop_decode_thread = False
        self.decoding_active = False

        # Tracking
        self.frame_count = 0
        self.presence_check_count = 0
        self.decode_attempt_count = 0
        self.detection_count = 0
        self.unique_barcodes = {}

        # Performance stats
        self.fps_counter = []
        self.current_fps = 0
        self.avg_presence_time_ms = 0
        self.avg_decode_time_ms = 0

        # Display
        self.show_feedback = False
        self.feedback_time = 0
        self.last_detection = None
        self.last_presence_regions = []

        # Output
        self.output_dir = "lightweight_detected_barcodes"
        os.makedirs(self.output_dir, exist_ok=True)

    def initialize_camera(self) -> bool:
        """Initialize camera."""
        try:
            if self.camera_type == "oakd":
                if not DEPTHAI_AVAILABLE:
                    print("[ERROR] DepthAI not available")
                    return False

                print(f"\nConnecting to OAK-D PoE camera at {self.oak_d_ip}...")
                self.camera = create_oak_d_camera(resolution_4k=True, fps=30, ip_address=self.oak_d_ip)

                if self.camera and self.camera.isOpened():
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ OAK-D camera: {width}x{height}")
                    return True
            else:
                print(f"\nConnecting to CV60 camera at {self.camera_ip}...")
                self.camera = CV60Camera(camera_ip=self.camera_ip)

                if self.camera.initialize():
                    width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"✓ CV60 camera: {width}x{height}")
                    return True

            print("[ERROR] Failed to initialize camera")
            return False

        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False

    def _apply_clahe(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_frame)

    def _apply_sharpening(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply sharpening."""
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        return cv2.filter2D(gray_frame, -1, kernel)

    def decode_barcode_sequential(self, frame: np.ndarray) -> List[BarcodeDetection]:
        """Heavy sequential decoding (runs in background thread)."""
        results = []

        preprocessing_steps = [
            ('original', lambda f: f),
            ('grayscale', lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f),
            ('clahe', lambda f: self._apply_clahe(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
            ('bilateral', lambda f: cv2.bilateralFilter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 11, 17, 17)),
            ('binary', lambda f: cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 127, 255, cv2.THRESH_BINARY)[1]),
            ('adaptive', lambda f: cv2.adaptiveThreshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('sharpen', lambda f: self._apply_sharpening(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
        ]

        if PYZBAR_AVAILABLE:
            for preproc_name, preproc_func in preprocessing_steps:
                try:
                    processed = preproc_func(frame)
                    decoded = pyzbar.decode(processed)

                    for obj in decoded:
                        try:
                            data = obj.data.decode('utf-8', errors='ignore')
                            if data:
                                results.append(BarcodeDetection(
                                    data=data,
                                    barcode_type=str(obj.type),
                                    method='pyzbar',
                                    preprocessing=preproc_name,
                                    timestamp=time.time()
                                ))
                        except:
                            pass

                    if results:
                        break  # Found something!
                except:
                    continue

        # Deduplicate
        unique = []
        seen = set()
        for r in results:
            if r.data not in seen:
                unique.append(r)
                seen.add(r.data)

        return unique

    def decode_worker(self):
        """Background thread for heavy decoding."""
        print("[INFO] Decode worker thread started")

        while not self.stop_decode_thread:
            try:
                # Get frame from queue
                item = self.decode_queue.get(timeout=0.1)
                if item is None:
                    continue

                frame, frame_num = item
                self.decoding_active = True

                # Measure decode time
                decode_start = time.time()
                detections = self.decode_barcode_sequential(frame)
                decode_time = (time.time() - decode_start) * 1000

                # Update stats
                self.avg_decode_time_ms = 0.9 * self.avg_decode_time_ms + 0.1 * decode_time

                if detections:
                    for det in detections:
                        det.detection_time_ms = decode_time
                    self.process_detection(frame, detections, frame_num)

                self.decoding_active = False

            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Decode worker error: {e}")
                self.decoding_active = False

        print("[INFO] Decode worker thread stopped")

    def process_detection(self, frame: np.ndarray, detections: List[BarcodeDetection], frame_num: int):
        """Process successful detection."""
        if not detections:
            return

        self.detection_count += 1
        timestamp = datetime.now()

        # Save
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_filename = f"lightweight_barcode_{timestamp_str}_frame{frame_num}"
        image_path = os.path.join(self.output_dir, f"{base_filename}.png")

        annotated = frame.copy()
        for detection in detections:
            text = f"{detection.barcode_type}: {detection.data}"
            cv2.putText(annotated, text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imwrite(image_path, annotated)

        # Terminal output
        print("\n" + "="*70)
        print(f"LIGHTWEIGHT AUTO-DETECTED at {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
        print("="*70)

        for detection in detections:
            print(f"  • Data: {detection.data}")
            print(f"    Type: {detection.barcode_type}")
            print(f"    Preprocessing: {detection.preprocessing}")
            print(f"    Decode time: {detection.detection_time_ms:.1f}ms")

            if detection.data not in self.unique_barcodes:
                self.unique_barcodes[detection.data] = {
                    'type': detection.barcode_type,
                    'first_seen': timestamp.isoformat(),
                    'frame': frame_num
                }

        print(f"\n💾 Saved: {image_path}")
        print(f"📊 Unique codes: {len(self.unique_barcodes)}")
        print("="*70 + "\n")

        # Update feedback
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
        """Draw overlay."""
        fps = self.calculate_fps()

        # FPS
        fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Stats
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Presence checks: {self.presence_check_count}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Decode attempts: {self.decode_attempt_count}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Detections: {self.detection_count}", (10, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Performance
        cv2.putText(frame, f"Presence: {self.avg_presence_time_ms:.1f}ms", (10, frame.shape[0] - 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)
        cv2.putText(frame, f"Decode: {self.avg_decode_time_ms:.1f}ms", (10, frame.shape[0] - 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

        # Mode
        camera_name = "OAK-D" if self.camera_type == "oakd" else "CV60"
        cv2.putText(frame, f"LIGHTWEIGHT AUTO ({camera_name})", (frame.shape[1]//2 - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Decoding indicator
        if self.decoding_active:
            cv2.putText(frame, "DECODING...", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Controls
        cv2.putText(frame, "'q'=quit | 'c'=clear | 'd'=debug", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Detection feedback
        if self.show_feedback:
            elapsed = time.time() - self.feedback_time
            if elapsed < 2.0:
                if self.last_detection:
                    color = (0, 255, 0)
                    text = f"FOUND: {self.last_detection.data}"
                else:
                    color = (0, 255, 255)
                    text = "CHECKING..."

                alpha = 1 - (elapsed / 2.0)
                color = tuple(int(c * alpha) for c in color)
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 3)
                cv2.putText(frame, text, (frame.shape[1]//2 - 200, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else:
                self.show_feedback = False

        # Draw detected regions from lightweight detector
        for region in self.last_presence_regions:
            x, y, w, h = region.bbox
            color = (0, 255, 255)  # Yellow for detected regions
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"Conf: {region.confidence:.2f}"
            cv2.putText(frame, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            return

        # Start decode worker thread
        self.decode_thread = threading.Thread(target=self.decode_worker, daemon=True)
        self.decode_thread.start()

        print("\n" + "="*70)
        camera_name = "OAK-D PoE" if self.camera_type == "oakd" else "CV60"
        print(f"LIGHTWEIGHT AUTO BARCODE SCANNER - {camera_name}")
        print("="*70)
        print("Two-stage detection:")
        print("  1. Fast presence check every frame (~2-5ms)")
        print("  2. Heavy decode only when barcode detected (async)")
        print("\nCONTROLS:")
        print("  'q' or ESC = Quit")
        print("  'c' = Clear history")
        print("  'd' = Toggle debug visualization")
        print("="*70 + "\n")

        window_name = f"Lightweight Auto Scanner - {camera_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        debug_mode = False

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

                # STAGE 1: Fast presence check (~2-5ms)
                presence_start = time.time()
                regions = self.presence_detector.detect(frame)
                presence_time = (time.time() - presence_start) * 1000

                self.presence_check_count += 1
                self.avg_presence_time_ms = 0.9 * self.avg_presence_time_ms + 0.1 * presence_time
                self.last_presence_regions = regions

                # STAGE 2: If barcode detected, queue for async decoding
                if regions and not self.decoding_active:
                    # Try to add to decode queue (non-blocking)
                    try:
                        self.decode_queue.put_nowait((frame.copy(), self.frame_count))
                        self.decode_attempt_count += 1
                    except:
                        pass  # Queue full, skip this frame

                # Display
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                # Debug mode: show presence detection
                if debug_mode and regions:
                    display_frame = self.presence_detector.visualize_detection(display_frame, regions)

                display_frame = self.draw_overlay(display_frame)
                cv2.imshow(window_name, display_frame)

                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('c') or key == ord('C'):
                    self.unique_barcodes.clear()
                    print("[INFO] Cleared barcode history")
                elif key == ord('d') or key == ord('D'):
                    debug_mode = not debug_mode
                    print(f"[INFO] Debug visualization: {debug_mode}")

                # Maintain FPS
                elapsed = current_time - last_frame_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_frame_time = time.time()

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted")
        finally:
            # Stop decode thread
            self.stop_decode_thread = True
            if self.decode_thread:
                self.decode_thread.join(timeout=2.0)

            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Frames: {self.frame_count}")
            print(f"Presence checks: {self.presence_check_count}")
            print(f"Decode attempts: {self.decode_attempt_count}")
            print(f"Successful detections: {self.detection_count}")
            print(f"Unique barcodes: {len(self.unique_barcodes)}")
            print(f"Avg presence time: {self.avg_presence_time_ms:.1f}ms")
            print(f"Avg decode time: {self.avg_decode_time_ms:.1f}ms")
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
    parser = argparse.ArgumentParser(description='Lightweight Auto Barcode Scanner')
    parser.add_argument('--camera', choices=['cv60', 'oakd'], default='cv60')
    parser.add_argument('--ip', default='192.168.1.21')
    parser.add_argument('--oak-ip', default='169.254.1.222')
    args = parser.parse_args()

    scanner = LightweightAutoBarcodeScanner(
        camera_type=args.camera,
        camera_ip=args.ip,
        oak_d_ip=args.oak_ip
    )
    scanner.run()


if __name__ == "__main__":
    main()
