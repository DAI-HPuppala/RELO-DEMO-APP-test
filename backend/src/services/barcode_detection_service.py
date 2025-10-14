"""Barcode Detection Service for RELO Classification System.

This service integrates the barcode_detection module into the RELO backend,
providing barcode detection capabilities using WebRTC frames.

Architecture:
- Uses 2-stage detection (fast presence check + async decode)
- Accepts frames from WebRTC (no separate camera connection)
- Filters detections by user-configured barcode types
- Returns standardized detection results

Performance:
- Stage 1 (Presence): ~2-5ms per frame
- Stage 2 (Decode): ~50-200ms when barcode detected
- Non-blocking: doesn't affect WebRTC stream
"""

import sys
import os
import asyncio
import logging
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add barcode_detection module to path
BARCODE_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', 'barcode_detection', 'src'
)
if os.path.exists(BARCODE_MODULE_PATH) and BARCODE_MODULE_PATH not in sys.path:
    sys.path.insert(0, BARCODE_MODULE_PATH)

logger = logging.getLogger(__name__)

# Try to import barcode detection components
BARCODE_AVAILABLE = False
try:
    from barcode_detector.lightweight import BarcodePresenceDetector
    from pyzbar import pyzbar
    BARCODE_AVAILABLE = True
    logger.info("✓ Barcode detection module loaded successfully")
except ImportError as e:
    logger.error(f"✗ Failed to load barcode detection module: {e}")
    logger.error("  Make sure barcode_detection module is installed")
    logger.error("  pip install pyzbar scipy")
    logger.error("  sudo apt-get install libzbar0")
    BarcodePresenceDetector = None
    pyzbar = None


class BarcodeDetectionService:
    """Service for detecting barcodes in video frames using 2-stage detection."""

    # Supported barcode types
    SUPPORTED_TYPES = [
        'CODE128', 'CODE39', 'CODE93', 'CODABAR',
        'EAN13', 'EAN8', 'UPCA', 'UPCE', 'ITF',
        'PDF417', 'QRCODE', 'DATAMATRIX', 'AZTEC'
    ]

    def __init__(self):
        """Initialize barcode detection service."""
        self.is_initialized = False
        self.presence_detector = None
        self.allowed_types: List[str] = []
        self.save_frames = os.getenv('BARCODE_SAVE_FRAMES', 'true').lower() == 'true'
        self.output_dir = Path('captured_frames') / 'barcode_detection'

        # Performance tracking
        self.detection_count = 0
        self.avg_presence_time_ms = 0.0
        self.avg_decode_time_ms = 0.0

        # Create output directory if saving frames
        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Barcode frames will be saved to: {self.output_dir}")

    async def initialize(self) -> bool:
        """Initialize barcode detection components.

        Returns:
            True if initialization successful, False otherwise
        """
        if not BARCODE_AVAILABLE:
            logger.error("Barcode detection module not available")
            return False

        try:
            logger.info("Initializing barcode detection service...")

            # Initialize lightweight presence detector (Stage 1)
            self.presence_detector = BarcodePresenceDetector(
                min_bar_width=2,
                max_bar_width=10,
                min_bars=10,
                confidence_threshold=0.5
            )

            # Load allowed types from environment
            default_types = os.getenv('BARCODE_ALLOWED_TYPES', 'CODE128,EAN13,QRCODE')
            self.allowed_types = [t.strip().upper() for t in default_types.split(',') if t.strip()]

            # Validate types
            invalid_types = [t for t in self.allowed_types if t not in self.SUPPORTED_TYPES]
            if invalid_types:
                logger.warning(f"Invalid barcode types configured: {invalid_types}")
                self.allowed_types = [t for t in self.allowed_types if t in self.SUPPORTED_TYPES]

            if not self.allowed_types:
                # Default to all types if none specified
                self.allowed_types = self.SUPPORTED_TYPES
                logger.info("No valid types configured, detecting all types")

            self.is_initialized = True
            logger.info(f"✓ Barcode detection service initialized")
            logger.info(f"  Allowed types: {', '.join(self.allowed_types)}")
            logger.info(f"  2-stage detection ready")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize barcode detection service: {e}", exc_info=True)
            self.is_initialized = False
            return False

    async def reinitialize(self) -> bool:
        """Retry initialization after failure.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Retrying barcode detection service initialization...")
        return await self.initialize()

    def set_allowed_types(self, types: List[str]) -> None:
        """Set which barcode types to detect.

        Args:
            types: List of barcode type strings (e.g., ['CODE128', 'QRCODE'])
        """
        if not types:
            self.allowed_types = self.SUPPORTED_TYPES
            logger.info("Allowed types set to ALL")
            return

        # Normalize and validate
        normalized = [t.strip().upper() for t in types if t.strip()]
        valid_types = [t for t in normalized if t in self.SUPPORTED_TYPES]
        invalid_types = [t for t in normalized if t not in self.SUPPORTED_TYPES]

        if invalid_types:
            logger.warning(f"Invalid barcode types ignored: {invalid_types}")

        if valid_types:
            self.allowed_types = valid_types
            logger.info(f"Allowed types updated: {', '.join(self.allowed_types)}")
        else:
            logger.warning("No valid types provided, keeping existing configuration")

    def _apply_clahe(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE preprocessing."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_frame)

    def _apply_sharpening(self, gray_frame: np.ndarray) -> np.ndarray:
        """Apply sharpening filter."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(gray_frame, -1, kernel)

    async def detect_barcode(self, frame: np.ndarray, allowed_types: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Detect barcode in frame using 2-stage process.

        Stage 1: Fast presence check (~2-5ms)
        Stage 2: Decode if barcode present (~50-200ms)

        Args:
            frame: Input frame from WebRTC (full resolution)
            allowed_types: Optional override for allowed types

        Returns:
            Detection dict with keys:
                - data: Barcode string
                - type: Barcode type (CODE128, QRCODE, etc.)
                - confidence: Detection confidence
                - timestamp: Detection timestamp
                - presence_time_ms: Stage 1 time
                - decode_time_ms: Stage 2 time
            or None if no barcode detected
        """
        if not self.is_initialized:
            logger.error("Barcode detection service not initialized")
            return None

        if frame is None or frame.size == 0:
            logger.error("Invalid frame provided")
            return None

        # Use provided types or default allowed types
        types_to_detect = allowed_types if allowed_types else self.allowed_types

        try:
            # STAGE 1: Fast presence check (~2-5ms)
            presence_start = time.time()
            regions = self.presence_detector.detect(frame)
            presence_time_ms = (time.time() - presence_start) * 1000

            # Update running average
            self.avg_presence_time_ms = 0.9 * self.avg_presence_time_ms + 0.1 * presence_time_ms

            if not regions:
                # No barcode detected
                return None

            logger.debug(f"Stage 1: Barcode presence detected ({presence_time_ms:.1f}ms)")

            # STAGE 2: Heavy decode (async, ~50-200ms)
            decode_start = time.time()
            detection = await self._decode_barcode_async(frame, types_to_detect)
            decode_time_ms = (time.time() - decode_start) * 1000

            # Update running average
            self.avg_decode_time_ms = 0.9 * self.avg_decode_time_ms + 0.1 * decode_time_ms

            if detection:
                self.detection_count += 1
                detection['presence_time_ms'] = presence_time_ms
                detection['decode_time_ms'] = decode_time_ms

                logger.info(f"✓ Barcode detected: {detection['type']} = {detection['data']}")
                logger.debug(f"  Stage 1: {presence_time_ms:.1f}ms, Stage 2: {decode_time_ms:.1f}ms")

                # Save frame if configured
                if self.save_frames:
                    await self._save_detection_frame(frame, detection)

                return detection
            else:
                logger.debug(f"Stage 2: Failed to decode barcode ({decode_time_ms:.1f}ms)")
                return None

        except Exception as e:
            logger.error(f"Error during barcode detection: {e}", exc_info=True)
            return None

    async def _decode_barcode_async(self, frame: np.ndarray, allowed_types: List[str]) -> Optional[Dict[str, Any]]:
        """Decode barcode with preprocessing strategies.

        Tries multiple preprocessing methods:
        1. Original frame
        2. Grayscale
        3. CLAHE
        4. Bilateral filter
        5. Binary threshold
        6. Adaptive threshold
        7. Sharpening

        Args:
            frame: Input frame
            allowed_types: List of allowed barcode types

        Returns:
            Detection dict or None
        """
        # Run decode in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._decode_barcode_sequential,
            frame,
            allowed_types
        )
        return result

    def _decode_barcode_sequential(self, frame: np.ndarray, allowed_types: List[str]) -> Optional[Dict[str, Any]]:
        """Sequential barcode decoding with multiple preprocessing strategies."""
        preprocessing_steps = [
            ('original', lambda f: f),
            ('grayscale', lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f),
            ('clahe', lambda f: self._apply_clahe(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
            ('bilateral', lambda f: cv2.bilateralFilter(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 11, 17, 17)),
            ('binary', lambda f: cv2.threshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 127, 255, cv2.THRESH_BINARY)[1]),
            ('adaptive', lambda f: cv2.adaptiveThreshold(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ('sharpen', lambda f: self._apply_sharpening(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f)),
        ]

        for preproc_name, preproc_func in preprocessing_steps:
            try:
                processed = preproc_func(frame)
                decoded = pyzbar.decode(processed)

                for obj in decoded:
                    try:
                        data = obj.data.decode('utf-8', errors='ignore')
                        barcode_type = str(obj.type)

                        # Filter by allowed types
                        if barcode_type not in allowed_types:
                            continue

                        if data:
                            return {
                                'data': data,
                                'type': barcode_type,
                                'confidence': 1.0,  # pyzbar doesn't provide confidence
                                'preprocessing': preproc_name,
                                'timestamp': time.time(),
                                'timestamp_str': datetime.now().isoformat()
                            }
                    except Exception as e:
                        logger.debug(f"Error decoding barcode object: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error in preprocessing '{preproc_name}': {e}")
                continue

        return None

    async def _save_detection_frame(self, frame: np.ndarray, detection: Dict[str, Any]) -> None:
        """Save detected frame to disk with text overlay (no bounding boxes).

        Args:
            frame: Original frame
            detection: Detection result dictionary
        """
        try:
            # Create clean frame copy (no bounding boxes)
            annotated = frame.copy()

            # Add text overlay in top-right corner
            text = f"{detection['type']}: {detection['data']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (0, 255, 0)  # Green text

            # Get text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Position in top-right corner (10px padding)
            x = frame.shape[1] - text_width - 10
            y = text_height + 10

            # Add background rectangle for better readability
            cv2.rectangle(annotated,
                         (x - 5, y - text_height - 5),
                         (x + text_width + 5, y + baseline + 5),
                         (0, 0, 0), -1)  # Black background

            # Add text
            cv2.putText(annotated, text, (x, y), font, font_scale, color, thickness)

            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            # Sanitize barcode data for filename
            safe_data = "".join(c for c in detection['data'] if c.isalnum() or c in ('-', '_'))[:20]
            filename = f"barcode_{timestamp_str}_{detection['type']}_{safe_data}.png"
            filepath = self.output_dir / filename

            # Save frame
            cv2.imwrite(str(filepath), annotated)
            logger.info(f"  💾 Saved detection frame: {filepath.name}")

        except Exception as e:
            logger.error(f"Failed to save detection frame: {e}", exc_info=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'initialized': self.is_initialized,
            'detection_count': self.detection_count,
            'avg_presence_time_ms': round(self.avg_presence_time_ms, 2),
            'avg_decode_time_ms': round(self.avg_decode_time_ms, 2),
            'allowed_types': self.allowed_types,
            'supported_types': self.SUPPORTED_TYPES,
            'save_frames_enabled': self.save_frames
        }

    def reset_statistics(self) -> None:
        """Reset detection statistics."""
        self.detection_count = 0
        self.avg_presence_time_ms = 0.0
        self.avg_decode_time_ms = 0.0
        logger.info("Barcode detection statistics reset")


# Singleton instance
_barcode_service_instance: Optional[BarcodeDetectionService] = None


def get_barcode_service() -> BarcodeDetectionService:
    """Get singleton instance of barcode detection service.

    Returns:
        BarcodeDetectionService instance
    """
    global _barcode_service_instance
    if _barcode_service_instance is None:
        _barcode_service_instance = BarcodeDetectionService()
        logger.info(f"Created new barcode service instance")
    return _barcode_service_instance
