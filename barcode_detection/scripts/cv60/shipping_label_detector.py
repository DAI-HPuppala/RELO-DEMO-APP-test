#!/usr/bin/env python3
"""CV60 Shipping Label Detector - Optimized for blurry shipping labels with multiple codes."""

import sys
import os
import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

# Setup paths
sys.path.insert(0, '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib')
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
    print("✓ pyzbar loaded")
except ImportError:
    print("✗ pyzbar not available")

# Import OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✓ Tesseract OCR available")
except:
    TESSERACT_AVAILABLE = False
    print("✗ Tesseract OCR not available")


class ShippingLabelDetector:
    """Detector optimized for shipping labels with multiple codes."""

    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()

        # Try QR with Aruco
        try:
            self.qr_aruco = cv2.QRCodeDetectorAruco()
            print("✓ QRCodeDetectorAruco available")
        except:
            self.qr_aruco = None

    def aggressive_preprocessing(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Aggressive preprocessing for very blurry images."""
        preprocessed = {}

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        h, w = gray.shape
        print(f"\nPreprocessing {w}x{h} image...")

        # 1. CRITICAL: Deblurring with Wiener filter approximation
        print("  1. Applying deblurring...")
        # First denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

        # Unsharp mask for deblurring
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        deblurred = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
        preprocessed['deblurred'] = deblurred

        # 2. Super aggressive sharpening
        print("  2. Aggressive sharpening...")
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
        super_sharp = cv2.filter2D(deblurred, -1, kernel)
        preprocessed['super_sharp'] = super_sharp

        # 3. Adaptive thresholding with different parameters
        print("  3. Multiple threshold attempts...")
        # For barcodes
        thresh1 = cv2.adaptiveThreshold(super_sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        preprocessed['thresh_11'] = thresh1

        thresh2 = cv2.adaptiveThreshold(super_sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 21, 5)
        preprocessed['thresh_21'] = thresh2

        # 4. CLAHE with high clip limit for very low contrast
        print("  4. Extreme contrast enhancement...")
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        extreme_contrast = clahe.apply(deblurred)
        preprocessed['extreme_contrast'] = extreme_contrast

        # 5. Morphological operations to reconstruct broken codes
        print("  5. Morphological reconstruction...")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        closed = cv2.morphologyEx(super_sharp, cv2.MORPH_CLOSE, kernel, iterations=2)
        preprocessed['morph_closed'] = closed

        # 6. Crop to label area (detect white rectangle)
        print("  6. Finding label region...")
        edges = cv2.Canny(deblurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest rectangular contour (the label)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4 and cv2.contourArea(contour) > (w * h * 0.1):
                # Found label, crop and process
                x, y, cw, ch = cv2.boundingRect(contour)
                if cw > w * 0.3 and ch > h * 0.3:  # Reasonable size
                    cropped = deblurred[y:y+ch, x:x+cw]
                    preprocessed['label_crop'] = cropped

                    # Process cropped region
                    crop_sharp = cv2.filter2D(cropped, -1, kernel)
                    preprocessed['label_sharp'] = crop_sharp
                    break

        # 7. Upscale for better detection
        print("  7. Upscaling for detail...")
        upscaled = cv2.resize(super_sharp, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        preprocessed['upscaled_2x'] = upscaled

        # 8. Histogram stretching
        print("  8. Histogram stretching...")
        p2, p98 = np.percentile(deblurred, (2, 98))
        stretched = np.clip((deblurred - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        preprocessed['stretched'] = stretched

        return preprocessed

    def detect_all_codes(self, image: np.ndarray, prep_name: str) -> List[Dict]:
        """Try all detection methods on an image."""
        detections = []

        # 1. Try pyzbar
        if PYZBAR_AVAILABLE:
            try:
                # Try all symbols
                decoded = pyzbar.decode(image)
                for obj in decoded:
                    data = obj.data.decode('utf-8', errors='ignore')
                    if data:
                        detections.append({
                            'data': data,
                            'type': str(obj.type),
                            'method': f'pyzbar_{prep_name}'
                        })

                # Specifically try QR
                decoded_qr = pyzbar.decode(image, symbols=[ZBarSymbol.QRCODE])
                for obj in decoded_qr:
                    data = obj.data.decode('utf-8', errors='ignore')
                    if data and not any(d['data'] == data for d in detections):
                        detections.append({
                            'data': data,
                            'type': 'QRCODE',
                            'method': f'pyzbar_qr_{prep_name}'
                        })

                # Specifically try Code128
                decoded_128 = pyzbar.decode(image, symbols=[ZBarSymbol.CODE128])
                for obj in decoded_128:
                    data = obj.data.decode('utf-8', errors='ignore')
                    if data and not any(d['data'] == data for d in detections):
                        detections.append({
                            'data': data,
                            'type': 'CODE128',
                            'method': f'pyzbar_128_{prep_name}'
                        })
            except:
                pass

        # 2. OpenCV QR
        try:
            # Single QR
            data, bbox, _ = self.qr_detector.detectAndDecode(image)
            if data and not any(d['data'] == data for d in detections):
                detections.append({
                    'data': data,
                    'type': 'QRCODE',
                    'method': f'opencv_qr_{prep_name}'
                })

            # Multi QR
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(image)
            if retval and decoded_info:
                for data in decoded_info:
                    if data and not any(d['data'] == data for d in detections):
                        detections.append({
                            'data': data,
                            'type': 'QRCODE',
                            'method': f'opencv_multi_{prep_name}'
                        })
        except:
            pass

        # 3. QR Aruco
        if self.qr_aruco:
            try:
                data, bbox, _ = self.qr_aruco.detectAndDecode(image)
                if data and not any(d['data'] == data for d in detections):
                    detections.append({
                        'data': data,
                        'type': 'QRCODE',
                        'method': f'aruco_{prep_name}'
                    })
            except:
                pass

        return detections

    def perform_ocr(self, image: np.ndarray) -> str:
        """Extract text using OCR."""
        if not TESSERACT_AVAILABLE:
            return ""

        try:
            # Configure for shipping labels
            config = '--oem 3 --psm 6'  # Assume uniform block of text
            text = pytesseract.image_to_string(image, config=config)
            return text
        except:
            return ""


class CV60ShippingScanner:
    """Main shipping label scanner application."""

    def __init__(self, camera_ip: str = "192.168.1.21"):
        self.camera_ip = camera_ip
        self.camera = None
        self.detector = ShippingLabelDetector()

        self.frame_count = 0
        self.capture_count = 0
        self.current_frame = None
        self.all_detections = {}

        self.fps_counter = []
        self.current_fps = 0

    def initialize_camera(self) -> bool:
        """Initialize CV60 camera."""
        if not CV60_MODULE_AVAILABLE:
            print("[ERROR] CV60 module not available")
            return False

        try:
            print(f"\nConnecting to CV60 at {self.camera_ip}...")
            self.camera = CV60Camera(ip_address=self.camera_ip)

            if self.camera.connect():
                settings = self.camera.get_camera_settings()
                print(f"✓ Connected: {settings.get('Width')}x{settings.get('Height')}")

                # IMPORTANT: Try to improve focus/exposure
                try:
                    # Increase exposure for better clarity
                    self.camera.set_exposure_time(10000)  # 10ms
                    print("✓ Adjusted exposure")
                except:
                    pass

                if self.camera.start_acquisition():
                    print("✓ Acquisition started")
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] {e}")
            return False

    def capture_and_process(self):
        """Capture and process for shipping labels."""
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        self.capture_count += 1
        timestamp = datetime.now()

        # Save
        test_dir = "test_captured_frames"
        os.makedirs(test_dir, exist_ok=True)

        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        base_name = f"shipping_{timestamp_str}_frame{self.frame_count}"

        original_path = os.path.join(test_dir, f"{base_name}_original.png")
        cv2.imwrite(original_path, frame)

        print("\n" + "="*80)
        print(f"SHIPPING LABEL SCAN #{self.capture_count}")
        print("="*80)
        print(f"Saved: {original_path}")

        # Check focus
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Focus score: {blur_score:.1f}")

        if blur_score < 50:
            print("⚠️ IMAGE VERY BLURRY - Applying aggressive deblurring...")

        # Process
        print("\nAPPLYING AGGRESSIVE PREPROCESSING...")
        preprocessed = self.detector.aggressive_preprocessing(frame)

        # Save key preprocessed versions
        for name in ['deblurred', 'super_sharp', 'label_crop']:
            if name in preprocessed:
                path = os.path.join(test_dir, f"{base_name}_{name}.png")
                cv2.imwrite(path, preprocessed[name])

        # Detect on all versions
        print("\nDETECTING CODES...")
        all_results = []

        for prep_name, prep_img in preprocessed.items():
            if prep_img is None or prep_img.size == 0:
                continue

            results = self.detector.detect_all_codes(prep_img, prep_name)
            all_results.extend(results)

        # OCR attempt
        print("\nOCR TEXT EXTRACTION...")
        if 'label_crop' in preprocessed:
            text = self.detector.perform_ocr(preprocessed['label_crop'])
            if text:
                print("OCR found:")
                for line in text.split('\n'):
                    if line.strip():
                        print(f"  • {line.strip()}")

        # Results
        print("\n" + "-"*80)
        if all_results:
            unique = {}
            for r in all_results:
                if r['data'] not in unique:
                    unique[r['data']] = r

            print(f"✓ FOUND {len(unique)} UNIQUE CODE(S):\n")
            for data, info in unique.items():
                print(f"  • {data}")
                print(f"    Type: {info['type']}, Method: {info['method']}")
                self.all_detections[data] = info
        else:
            print("✗ NO CODES DETECTED")
            print("\nRECOMMENDATIONS:")
            print("  1. CRITICAL: Hold camera steadier for better focus")
            print("  2. Move closer/further to get optimal focus")
            print("  3. Ensure even lighting on label")
            print("  4. Try different angles to reduce glare")

        print("\n" + "-"*80)
        print(f"Session total: {len(self.all_detections)} unique codes")
        print("="*80)

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

        # Title
        cv2.putText(frame, "SHIPPING LABEL SCANNER", (frame.shape[1]//2 - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        fps_color = (0, 255, 0) if fps > 25 else (0, 255, 255) if fps > 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Stats
        cv2.putText(frame, f"Scans: {self.capture_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.all_detections:
            cv2.putText(frame, f"Codes found: {len(self.all_detections)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Instructions
        cv2.putText(frame, "Press '+' to scan label", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, "HOLD STEADY for better focus!", (10, frame.shape[0] - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Focus guide
        h, w = frame.shape[:2]
        # Draw rectangle where label should be
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(frame, "Position label here", (w//2 - 70, h//4 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            return

        print("\n" + "="*80)
        print("SHIPPING LABEL SCANNER - Optimized for blurry multi-code labels")
        print("="*80)
        print("\nTIPS FOR SUCCESS:")
        print("  • HOLD CAMERA STEADY - Focus is critical!")
        print("  • Position label in center rectangle")
        print("  • Ensure even lighting, no shadows")
        print("  • Press '+' when label is visible")
        print("="*80 + "\n")

        os.makedirs("test_captured_frames", exist_ok=True)

        window = "CV60 Shipping Label Scanner"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)

        try:
            frame_time = 1.0 / 30.0
            last_time = time.time()

            for frame in self.camera.stream():
                current_time = time.time()
                self.frame_count += 1
                self.current_frame = frame

                # Display
                if len(frame.shape) == 2:
                    display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    display = frame.copy()

                display = self.draw_overlay(display)
                cv2.imshow(window, display)

                # Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('+') or key == ord('='):
                    self.capture_and_process()

                # FPS
                elapsed = current_time - last_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last_time = time.time()

        except KeyboardInterrupt:
            pass
        finally:
            print("\n" + "="*80)
            print("SESSION COMPLETE")
            print("="*80)
            print(f"Total scans: {self.capture_count}")
            print(f"Unique codes found: {len(self.all_detections)}")

            if self.all_detections:
                print("\nDetected codes:")
                for code in self.all_detections:
                    print(f"  • {code}")

            print("="*80)

            if self.camera:
                self.camera.stop_acquisition()
                self.camera.disconnect()
            cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', default='192.168.1.21')
    args = parser.parse_args()

    scanner = CV60ShippingScanner(camera_ip=args.ip)
    scanner.run()


if __name__ == "__main__":
    main()