#!/usr/bin/env python3
"""CV60 QR/Barcode Detector with Manual Trigger - 30fps display with on-demand detection."""

import sys
import os
import cv2
import numpy as np
import time
import threading
import logging
from datetime import datetime
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
        logging.FileHandler('logs/qr_manual.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import all detection libraries
PYZBAR_AVAILABLE = False
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    logger.info("pyzbar library loaded")
except ImportError:
    logger.warning("pyzbar not available")

# Check for OpenCV detectors
HAS_WECHAT_QRCODE = hasattr(cv2, 'wechat_qrcode_WeChatQRCode')
HAS_BARCODE_DETECTOR = hasattr(cv2, 'barcode_BarcodeDetector')


@dataclass
class CapturedFrame:
    """Registry entry for captured frames."""
    frame: np.ndarray
    timestamp: float
    frame_number: int
    filename: str
    detections: List[Dict[str, Any]]


class ManualQRDetector:
    """Manual trigger QR/Barcode detector with 30fps display."""

    def __init__(self, camera_ip: str = "192.168.1.21"):
        self.camera_ip = camera_ip
        self.camera = None

        # Frame registry
        self.frame_registry = []
        self.max_registry_size = 100  # Keep last 100 captured frames

        # Current frame info
        self.current_frame = None
        self.frame_count = 0
        self.capture_count = 0

        # Detection components
        self.qr_detector = cv2.QRCodeDetector()
        self.qr_aruco = None
        self.wechat_detector = None
        self.barcode_detector = None

        # Initialize additional detectors
        self._init_detectors()

        # FPS tracking
        self.fps_counter = []
        self.current_fps = 0

        # Display state
        self.show_capture_feedback = False
        self.capture_feedback_time = 0

    def _init_detectors(self):
        """Initialize all available detection methods."""
        # QRCodeDetectorAruco
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            try:
                self.qr_aruco = cv2.QRCodeDetectorAruco()
                logger.info("QRCodeDetectorAruco initialized")
            except:
                pass

        # WeChat QRCode
        if HAS_WECHAT_QRCODE:
            try:
                self.wechat_detector = cv2.wechat_qrcode_WeChatQRCode()
                logger.info("WeChat QRCode detector initialized")
            except:
                pass

        # OpenCV Barcode
        if HAS_BARCODE_DETECTOR:
            try:
                self.barcode_detector = cv2.barcode_BarcodeDetector()
                logger.info("OpenCV Barcode detector initialized")
            except:
                pass

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

    def capture_and_process(self):
        """Capture current frame and process with all detection methods."""
        if self.current_frame is None:
            print("\n[ERROR] No frame available to capture")
            return

        # Get the latest frame (this is the current displayed frame)
        captured_frame = self.current_frame.copy()
        capture_time = time.time()
        self.capture_count += 1

        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{timestamp_str}_frame{self.frame_count}.png"
        filepath = os.path.join("captured_frames", filename)

        # Create directory if needed
        os.makedirs("captured_frames", exist_ok=True)

        # Save frame
        cv2.imwrite(filepath, captured_frame)

        print("\n" + "="*70)
        print(f"FRAME CAPTURED #{self.capture_count}")
        print("="*70)
        print(f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        print(f"Frame number: {self.frame_count}")
        print(f"Saved to: {filepath}")
        print(f"Frame shape: {captured_frame.shape}")
        print("\nAPPLYING ALL DETECTION METHODS:")
        print("-"*70)

        all_detections = []

        # Prepare grayscale version
        if len(captured_frame.shape) == 3:
            gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = captured_frame

        # 1. PYZBAR DETECTION
        print("\n[1] PYZBAR DETECTION:")
        if PYZBAR_AVAILABLE:
            try:
                decoded = pyzbar.decode(gray)
                if decoded:
                    for obj in decoded:
                        data = obj.data.decode('utf-8', errors='ignore')
                        barcode_type = str(obj.type)
                        print(f"   ✓ Found: '{data}' (Type: {barcode_type})")
                        all_detections.append({
                            'method': 'pyzbar',
                            'data': data,
                            'type': barcode_type
                        })
                else:
                    print("   ✗ No barcodes detected")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print("   ✗ Not available")

        # 2. OPENCV QR DETECTOR
        print("\n[2] OPENCV QR DETECTOR:")
        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)
            if data:
                print(f"   ✓ Found: '{data}'")
                all_detections.append({
                    'method': 'opencv_qr',
                    'data': data,
                    'type': 'QRCODE'
                })
            else:
                print("   ✗ No QR code detected")
        except Exception as e:
            print(f"   ✗ Error: {e}")

        # 3. OPENCV QR ARUCO
        print("\n[3] OPENCV QR ARUCO DETECTOR:")
        if self.qr_aruco:
            try:
                data, bbox, _ = self.qr_aruco.detectAndDecode(gray)
                if data:
                    print(f"   ✓ Found: '{data}'")
                    all_detections.append({
                        'method': 'opencv_qr_aruco',
                        'data': data,
                        'type': 'QRCODE'
                    })
                else:
                    print("   ✗ No QR code detected")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print("   ✗ Not available")

        # 4. WECHAT QR DETECTOR
        print("\n[4] WECHAT QR DETECTOR:")
        if self.wechat_detector:
            try:
                res, points = self.wechat_detector.detectAndDecode(gray)
                if res and res[0]:
                    for data in res:
                        if data:
                            print(f"   ✓ Found: '{data}'")
                            all_detections.append({
                                'method': 'wechat_qr',
                                'data': data,
                                'type': 'QRCODE'
                            })
                else:
                    print("   ✗ No QR code detected")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print("   ✗ Not available")

        # 5. OPENCV BARCODE DETECTOR
        print("\n[5] OPENCV BARCODE DETECTOR:")
        if self.barcode_detector:
            try:
                retval, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecodeMulti(gray)
                if retval and decoded_info:
                    for i, data in enumerate(decoded_info):
                        if data:
                            barcode_type = decoded_type[i] if i < len(decoded_type) else 'BARCODE'
                            print(f"   ✓ Found: '{data}' (Type: {barcode_type})")
                            all_detections.append({
                                'method': 'opencv_barcode',
                                'data': data,
                                'type': barcode_type
                            })
                else:
                    print("   ✗ No barcodes detected")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        else:
            print("   ✗ Not available")

        # Apply preprocessing and retry detection
        print("\n[6] DETECTION WITH PREPROCESSING:")

        # CLAHE enhancement
        print("   A. With CLAHE enhancement:")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        self._try_all_on_processed(enhanced, "CLAHE", all_detections)

        # Adaptive threshold
        print("   B. With adaptive threshold:")
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 10)
        self._try_all_on_processed(thresh, "Threshold", all_detections)

        # Binary threshold
        print("   C. With binary threshold:")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._try_all_on_processed(binary, "Binary", all_detections)

        # Summary
        print("\n" + "-"*70)
        print("DETECTION SUMMARY:")
        if all_detections:
            unique_codes = set(d['data'] for d in all_detections)
            print(f"   Total detections: {len(all_detections)}")
            print(f"   Unique codes: {len(unique_codes)}")
            for code in unique_codes:
                methods = [d['method'] for d in all_detections if d['data'] == code]
                print(f"   • '{code}' - detected by: {', '.join(methods)}")
        else:
            print("   No codes detected by any method")

        # Add to registry
        registry_entry = CapturedFrame(
            frame=captured_frame,
            timestamp=capture_time,
            frame_number=self.frame_count,
            filename=filename,
            detections=all_detections
        )

        self.frame_registry.append(registry_entry)

        # Keep registry size limited
        if len(self.frame_registry) > self.max_registry_size:
            self.frame_registry.pop(0)

        print(f"\nFrame added to registry (total: {len(self.frame_registry)} frames)")
        print("="*70 + "\n")

        # Show visual feedback
        self.show_capture_feedback = True
        self.capture_feedback_time = time.time()

    def _try_all_on_processed(self, processed_img, preprocess_name, all_detections):
        """Try all detection methods on a preprocessed image."""
        found_any = False

        # Try pyzbar
        if PYZBAR_AVAILABLE:
            try:
                decoded = pyzbar.decode(processed_img)
                if decoded:
                    for obj in decoded:
                        data = obj.data.decode('utf-8', errors='ignore')
                        print(f"      • pyzbar found: '{data}'")
                        found_any = True
            except:
                pass

        # Try OpenCV QR
        try:
            data, _, _ = self.qr_detector.detectAndDecode(processed_img)
            if data:
                print(f"      • OpenCV QR found: '{data}'")
                found_any = True
        except:
            pass

        if not found_any:
            print(f"      • No detections with {preprocess_name}")

    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time)
        self.fps_counter = [t for t in self.fps_counter if current_time - t < 1.0]
        if len(self.fps_counter) > 1:
            self.current_fps = len(self.fps_counter)
        return self.current_fps

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay information on frame."""
        # Calculate FPS
        fps = self.calculate_fps()

        # FPS indicator
        fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Capture count
        cv2.putText(frame, f"Captures: {self.capture_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Instructions
        cv2.putText(frame, "Press '+' to capture & detect", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, "Press 'q' or ESC to quit", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Show capture feedback
        if self.show_capture_feedback:
            elapsed = time.time() - self.capture_feedback_time
            if elapsed < 0.5:  # Show for 0.5 seconds
                alpha = 1 - (elapsed / 0.5)
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.putText(frame, "CAPTURED!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                # Draw border
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 5)
            else:
                self.show_capture_feedback = False

        return frame

    def run(self):
        """Main execution loop - 30fps display with manual capture trigger."""
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return

        print("\n" + "="*70)
        print("CV60 MANUAL TRIGGER DETECTOR - 30 FPS Display")
        print("="*70)
        print(f"Camera IP: {self.camera_ip}")
        print(f"Mode: Manual capture with all detection methods")
        print("\nCONTROLS:")
        print("  '+' - Capture frame and run all detection methods")
        print("  'q' or ESC - Quit")
        print("\nDETECTION METHODS AVAILABLE:")
        print(f"  • pyzbar: {PYZBAR_AVAILABLE}")
        print(f"  • OpenCV QR: True")
        print(f"  • OpenCV QR Aruco: {self.qr_aruco is not None}")
        print(f"  • WeChat QR: {self.wechat_detector is not None}")
        print(f"  • OpenCV Barcode: {self.barcode_detector is not None}")
        print("="*70 + "\n")

        # Create display window
        window_name = "CV60 Manual Detection - 30 FPS"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)

        # Create output directories
        os.makedirs("captured_frames", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        try:
            frame_time = 1.0 / 30.0  # Target 30 FPS
            last_frame_time = time.time()

            # Main display loop
            for frame in self.camera.stream():
                current_time = time.time()
                self.frame_count += 1

                # Store current frame (always the latest)
                self.current_frame = frame

                # Prepare display frame
                if len(frame.shape) == 2:
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display_frame = frame.copy()

                # Draw overlay
                display_frame = self.draw_overlay(display_frame)

                # Show frame
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Quit
                    break
                elif key == ord('+') or key == ord('='):  # Capture and process
                    self.capture_and_process()

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
            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Total frames displayed: {self.frame_count}")
            print(f"Average FPS: {self.current_fps:.1f}")
            print(f"Frames captured: {self.capture_count}")
            print(f"Frames in registry: {len(self.frame_registry)}")

            if self.frame_registry:
                print("\nCAPTURED FRAMES:")
                for i, entry in enumerate(self.frame_registry[-10:], 1):  # Show last 10
                    print(f"  {i}. {entry.filename} - {len(entry.detections)} detections")

            print("="*70)

            # Cleanup
            if self.camera:
                self.camera.stop_acquisition()
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CV60 Manual Trigger QR/Barcode Detector')
    parser.add_argument('--ip', default='192.168.1.21',
                       help='CV60 camera IP address')
    args = parser.parse_args()

    detector = ManualQRDetector(camera_ip=args.ip)
    detector.run()


if __name__ == "__main__":
    main()