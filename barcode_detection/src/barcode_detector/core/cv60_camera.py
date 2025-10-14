"""CV60 camera integration for barcode detection using eBUS SDK."""

import os
import sys
import site
import numpy as np
import cv2
import logging
import time
import threading
from queue import Queue, Empty
from typing import Optional, Tuple, Any

# Setup logging
logger = logging.getLogger(__name__)

# Setup eBUS SDK paths - CRITICAL: Must be done before import
EBUS_SDK_PATH = "/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib"

# Add eBUS SDK to Python path FIRST
if os.path.isdir(EBUS_SDK_PATH) and EBUS_SDK_PATH not in sys.path:
    sys.path.insert(0, EBUS_SDK_PATH)
    logger.info(f"Added eBUS SDK path: {EBUS_SDK_PATH}")

# Setup eBUS SDK paths
for p in site.getsitepackages():
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# Add additional common paths for Debian packages
for path in [
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.10/site-packages",
]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Add LD_LIBRARY_PATH for native libraries - CRITICAL for .so files
lib_paths = [
    '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib',
    '/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib/genicam/bin/Linux64_x64'
]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_paths + [current_ld_path])

try:
    import eBUS as eb
    EBUS_AVAILABLE = True
    logger.info("Successfully imported eBUS module for QR code detection")
except ImportError as e:
    logger.warning(f"eBUS SDK not available: {e}. Will fallback to OpenCV camera.")
    eb = None
    EBUS_AVAILABLE = False

# Constants
BUFFER_COUNT = 16
CV60_IP = "192.168.1.21"  # Default CV60 camera IP


class CV60Camera:
    """CV60 camera wrapper for barcode detection."""

    def __init__(self, camera_ip: str = CV60_IP):
        """Initialize CV60 camera.

        Args:
            camera_ip: IP address of the CV60 camera
        """
        self.camera_ip = camera_ip
        self.is_initialized = False
        self.device = None
        self.stream = None
        self.buffers = []
        self._stop_capture = False
        self._frame_queue = Queue(maxsize=10)
        self._capture_thread = None
        self._last_frame = None
        self._opencv_fallback = None

    def initialize(self) -> bool:
        """Initialize the CV60 camera using eBUS SDK.

        Returns:
            True if successful, False otherwise
        """
        if not EBUS_AVAILABLE:
            logger.warning("eBUS SDK not available, using OpenCV fallback")
            return self._initialize_opencv_fallback()

        try:
            logger.info(f"Connecting to CV60 camera at {self.camera_ip}")

            # Connect to device directly
            result, self.device = eb.PvDevice.CreateAndConnect(self.camera_ip)
            if not self.device:
                raise Exception(f"Failed to connect to device {self.camera_ip}")

            # Open stream
            logger.info("Opening stream...")
            result, self.stream = eb.PvStream.CreateAndOpen(self.camera_ip)
            if not self.stream:
                raise Exception("Failed to open stream")

            # Configure packet size for GigE
            if isinstance(self.device, eb.PvDeviceGEV):
                system = eb.PvSystem()
                system.Find()
                for k in range(system.GetInterfaceCount()):
                    iface = system.GetInterface(k)
                    try:
                        ip = str(iface.GetIPAddress(0))
                    except:
                        continue
                    if ip.startswith(('169.254', '192.168')):
                        self.device.NegotiatePacketSize()
                        self.device.SetStreamDestination(ip, self.stream.GetLocalPort())
                        break

            # Allocate buffers
            size = self.device.GetPayloadSize()
            buf_count = min(self.stream.GetQueuedBufferMaximum(), BUFFER_COUNT)
            self.buffers = []
            for _ in range(buf_count):
                buf = eb.PvBuffer()
                buf.Alloc(size)
                self.buffers.append(buf)
                self.stream.QueueBuffer(buf)

            # Start acquisition
            self.device.StreamEnable()
            self.device.GetParameters().Get("AcquisitionStart").Execute()

            logger.info(f"CV60 camera initialized successfully at {self.camera_ip}")
            self.is_initialized = True

            # Start capture thread
            self._start_capture_thread()

            return True

        except Exception as e:
            logger.error(f"Failed to initialize CV60 camera: {e}")
            return self._initialize_opencv_fallback()

    def _initialize_opencv_fallback(self) -> bool:
        """Initialize OpenCV camera as fallback.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing OpenCV camera fallback")

            # Try to find a camera
            for index in range(5):
                self._opencv_fallback = cv2.VideoCapture(index)
                if self._opencv_fallback.isOpened():
                    # Set resolution for better QR detection
                    self._opencv_fallback.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    self._opencv_fallback.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

                    # Check actual resolution
                    actual_w = int(self._opencv_fallback.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self._opencv_fallback.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    logger.info(f"OpenCV camera initialized at index {index}: {actual_w}x{actual_h}")
                    self.is_initialized = True

                    # Start capture thread for OpenCV
                    self._start_capture_thread()
                    return True

            logger.error("No OpenCV camera found")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize OpenCV fallback: {e}")
            return False

    def _start_capture_thread(self):
        """Start the frame capture thread."""
        if self._capture_thread is not None:
            return

        self._stop_capture = False
        self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._capture_thread.start()
        logger.info("Frame capture thread started")

    def _capture_frames(self):
        """Continuously capture frames in a separate thread."""
        while not self._stop_capture:
            try:
                if self._opencv_fallback is not None:
                    # OpenCV fallback mode
                    ret, frame = self._opencv_fallback.read()
                    if ret:
                        # Add to queue
                        try:
                            self._frame_queue.put_nowait(frame)
                            self._last_frame = frame
                        except:
                            # Queue full, drop oldest
                            try:
                                self._frame_queue.get_nowait()
                                self._frame_queue.put_nowait(frame)
                            except:
                                pass
                    time.sleep(0.033)  # ~30 FPS

                elif EBUS_AVAILABLE and self.stream:
                    # eBUS mode
                    result, buf, op = self.stream.RetrieveBuffer(500)

                    if result.IsOK() and op and op.IsOK() and buf.GetPayloadType() == eb.PvPayloadTypeImage:
                        img = buf.GetImage()
                        w, h = img.GetWidth(), img.GetHeight()
                        raw = np.frombuffer(img.GetDataPointer(), dtype=np.uint8).copy()

                        # Convert based on pixel format
                        pixel_type = img.GetPixelType()

                        if len(raw) == h * w * 3:
                            # Already RGB
                            frame = raw.reshape((h, w, 3))
                        elif len(raw) == h * w:
                            # Single channel - needs conversion
                            temp = raw.reshape((h, w))

                            if pixel_type == eb.PvPixelMono8:
                                # Grayscale
                                frame = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
                            else:
                                # Try Bayer patterns
                                try:
                                    frame = cv2.cvtColor(temp, cv2.COLOR_BayerBG2BGR)
                                except:
                                    try:
                                        frame = cv2.cvtColor(temp, cv2.COLOR_BayerRG2BGR)
                                    except:
                                        # Fallback to grayscale
                                        frame = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
                        else:
                            # Unknown format
                            frame = np.zeros((h, w, 3), dtype=np.uint8)

                        # Return buffer
                        self.stream.QueueBuffer(buf)

                        # Add to queue
                        try:
                            self._frame_queue.put_nowait(frame)
                            self._last_frame = frame
                        except:
                            # Queue full, drop oldest
                            try:
                                self._frame_queue.get_nowait()
                                self._frame_queue.put_nowait(frame)
                            except:
                                pass

                    else:
                        if buf:
                            self.stream.QueueBuffer(buf)
                        time.sleep(0.05)

                else:
                    # No camera available
                    time.sleep(0.1)

            except Exception as e:
                logger.debug(f"Frame capture error: {e}")
                time.sleep(0.05)

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the camera.

        Returns:
            Frame as numpy array or None if no frame available
        """
        try:
            # Try to get from queue first
            frame = self._frame_queue.get_nowait()
            self._last_frame = frame
            return frame
        except Empty:
            # Return last frame if available
            return self._last_frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame (OpenCV-compatible interface).

        Returns:
            Tuple of (success, frame)
        """
        frame = self.get_frame()
        return (frame is not None, frame)

    def isOpened(self) -> bool:
        """Check if camera is opened.

        Returns:
            True if camera is initialized
        """
        return self.is_initialized

    def set(self, prop_id: int, value: float) -> bool:
        """Set camera property (OpenCV-compatible interface).

        Args:
            prop_id: OpenCV property ID
            value: Property value

        Returns:
            True if successful
        """
        if self._opencv_fallback:
            return self._opencv_fallback.set(prop_id, value)
        # TODO: Map OpenCV properties to eBUS parameters
        return True

    def get(self, prop_id: int) -> float:
        """Get camera property (OpenCV-compatible interface).

        Args:
            prop_id: OpenCV property ID

        Returns:
            Property value
        """
        if self._opencv_fallback:
            return self._opencv_fallback.get(prop_id)

        # Return default values for eBUS camera
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 1760  # CV60 default width
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 1696  # CV60 default height
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30
        return 0

    def release(self):
        """Release camera resources."""
        # Stop capture thread
        self._stop_capture = True
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

        # Release OpenCV camera if used
        if self._opencv_fallback:
            self._opencv_fallback.release()
            self._opencv_fallback = None

        # Release eBUS resources
        if EBUS_AVAILABLE and self.device:
            try:
                # Stop acquisition
                self.device.GetParameters().Get("AcquisitionStop").Execute()
            except:
                pass

            try:
                if self.stream:
                    self.stream.Close()
            except:
                pass

            try:
                self.device.Disconnect()
            except:
                pass

        self.is_initialized = False
        logger.info("CV60 camera released")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def create_cv60_camera(camera_index: int = 0) -> Any:
    """Factory function to create a CV60 camera or OpenCV fallback.

    Args:
        camera_index: Camera index (ignored for CV60, used for OpenCV fallback)

    Returns:
        CV60Camera instance or cv2.VideoCapture
    """
    # Try CV60 camera first
    cv60 = CV60Camera()
    if cv60.initialize():
        logger.info("Using CV60 camera for QR code detection")
        return cv60

    # Fallback to OpenCV
    logger.warning("CV60 initialization failed, using standard OpenCV camera")
    return cv2.VideoCapture(camera_index)