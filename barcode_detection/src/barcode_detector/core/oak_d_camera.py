"""OAK-D camera integration for barcode/QR code detection using DepthAI."""

import logging
import numpy as np
import cv2
from typing import Optional, Tuple
import time

# Setup logging
logger = logging.getLogger(__name__)

# Try to import depthai
try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
    logger.info("Successfully imported depthai module for OAK-D camera")
except ImportError as e:
    logger.warning(f"DepthAI module not available: {e}. Install with: pip install depthai")
    dai = None
    DEPTHAI_AVAILABLE = False


class OAKDCamera:
    """OAK-D camera wrapper for barcode/QR code detection with 4K resolution and autofocus."""

    def __init__(self, resolution_4k: bool = True, fps: int = 30, ip_address: Optional[str] = None):
        """Initialize OAK-D camera.

        Args:
            resolution_4k: Use 4K resolution (3840x2160). If False, uses 1080p
            fps: Target FPS (default: 30)
            ip_address: IP address for PoE devices (e.g., "169.254.1.222"). If None, uses USB connection
        """
        self.resolution_4k = resolution_4k
        self.fps = fps
        self.ip_address = ip_address
        self.is_initialized = False
        self.device = None
        self.pipeline = None
        self.video_queue = None
        self.control_queue = None
        self._last_frame = None
        self._last_lens_position = None  # Track lens position from metadata

    def initialize(self) -> bool:
        """Initialize the OAK-D camera with optimal settings for barcode detection.

        Returns:
            True if successful, False otherwise
        """
        if not DEPTHAI_AVAILABLE:
            logger.error("DepthAI module not available")
            return False

        max_retries = 3
        retry_delay = 1.5

        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing OAK-D camera (attempt {attempt + 1}/{max_retries})...")

                # Create pipeline
                self.pipeline = dai.Pipeline()

                # Define ColorCamera node
                cam_rgb = self.pipeline.create(dai.node.ColorCamera)

                # Set resolution - 4K for maximum detail in barcode detection
                if self.resolution_4k:
                    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
                    cam_rgb.setIspScale(1, 1)  # No downscaling, use full 4K
                    logger.info("Configured for 4K resolution (3840x2160)")
                else:
                    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                    logger.info("Configured for 1080p resolution (1920x1080)")

                # Set FPS
                cam_rgb.setFps(self.fps)

                # Set color order to BGR for OpenCV compatibility
                cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                cam_rgb.setInterleaved(False)
                cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

                # Enable initial camera controls optimized for barcode detection
                # Use CONTINUOUS_VIDEO autofocus mode - best for detecting moving barcodes
                cam_rgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

                # Enable auto exposure
                cam_rgb.initialControl.setAutoExposureEnable()

                # Enable auto white balance for consistent color reproduction
                cam_rgb.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)

                # Create XLinkOut nodes for video stream
                xout_video = self.pipeline.create(dai.node.XLinkOut)
                xout_video.setStreamName("video")

                # Link camera video output to XLinkOut
                cam_rgb.video.link(xout_video.input)

                # Create control input for dynamic camera control
                xin_control = self.pipeline.create(dai.node.XLinkIn)
                xin_control.setStreamName("control")
                xin_control.out.link(cam_rgb.inputControl)

                # Connect to device and start pipeline
                # Try dynamic discovery first, then fall back to static IP
                device_info = None

                if self.ip_address:
                    # For PoE: try auto-discovery first
                    devices = dai.Device.getAllAvailableDevices()
                    logger.info(f"Found {len(devices)} device(s) via discovery")

                    # Look for PoE device via auto-discovery
                    for dev in devices:
                        if dev.protocol == dai.XLinkProtocol.X_LINK_TCP_IP:
                            logger.info(f"✓ Dynamically discovered PoE device: {dev.name}")
                            device_info = dev
                            break

                    # Fall back to static IP if no device found via discovery
                    if device_info is None:
                        logger.info(f"No PoE device found via auto-discovery, using static IP: {self.ip_address}")
                        device_info = dai.DeviceInfo(self.ip_address)

                    # Connect to the device
                    self.device = dai.Device(self.pipeline, device_info)
                    logger.info(f"✓ Connected to OAK-D PoE device")
                else:
                    # For USB: direct connection
                    logger.info("Connecting to OAK-D USB device...")
                    self.device = dai.Device(self.pipeline)

                # Get output queue for video frames
                self.video_queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)

                # Get input queue for camera controls
                self.control_queue = self.device.getInputQueue(name="control")

                # Get actual camera properties
                self.is_initialized = True
                connection_type = f"PoE @ {self.ip_address}" if self.ip_address else "USB"
                logger.info(f"OAK-D camera initialized successfully ({connection_type})")
                logger.info(f"Settings: {self.fps} FPS, Autofocus: CONTINUOUS_VIDEO, Auto-exposure: Enabled")

                return True

            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1}/{max_retries} failed: {e}")

                # Cleanup partial initialization
                if self.device:
                    try:
                        self.device.close()
                    except:
                        pass
                    self.device = None
                self.video_queue = None
                self.control_queue = None
                self.pipeline = None

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    import traceback
                    traceback.print_exc()
                    return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the camera.

        Returns:
            Frame as numpy array (BGR format) or None if no frame available
        """
        if not self.is_initialized or not self.video_queue:
            return None

        try:
            # Get frame from queue (non-blocking)
            in_video = self.video_queue.tryGet()

            if in_video is not None:
                # Get frame data as numpy array
                frame = in_video.getCvFrame()
                self._last_frame = frame

                # Try to get lens position from metadata
                try:
                    metadata = in_video.getMetadata()
                    if hasattr(metadata, 'getLensPosition'):
                        self._last_lens_position = metadata.getLensPosition()
                except:
                    pass  # Metadata not available

                return frame
            else:
                # Return last frame if no new frame available
                return self._last_frame

        except Exception as e:
            logger.debug(f"Error getting frame: {e}")
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
        # OAK-D uses DepthAI control system, not OpenCV properties
        # This is a compatibility method
        logger.debug(f"set() called with prop_id={prop_id}, value={value} (not implemented for OAK-D)")
        return True

    def get(self, prop_id: int) -> float:
        """Get camera property (OpenCV-compatible interface).

        Args:
            prop_id: OpenCV property ID

        Returns:
            Property value
        """
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 3840 if self.resolution_4k else 1920
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 2160 if self.resolution_4k else 1080
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0

    def set_manual_focus(self, focus_value: int):
        """Set manual focus position.

        Args:
            focus_value: Focus value 0-255 (0=far, 255=near)
        """
        if not self.is_initialized or not self.control_queue:
            logger.warning("Camera not initialized")
            return

        try:
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(focus_value)
            self.control_queue.send(ctrl)
            logger.debug(f"Set manual focus to {focus_value}")
        except Exception as e:
            logger.error(f"Failed to set manual focus: {e}")

    def set_auto_focus(self, mode: str = "CONTINUOUS_VIDEO"):
        """Set autofocus mode.

        Args:
            mode: Autofocus mode - "CONTINUOUS_VIDEO", "CONTINUOUS_PICTURE", "AUTO", or "OFF"
        """
        if not self.is_initialized or not self.control_queue:
            logger.warning("Camera not initialized")
            return

        try:
            ctrl = dai.CameraControl()

            mode_map = {
                "CONTINUOUS_VIDEO": dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO,
                "CONTINUOUS_PICTURE": dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE,
                "AUTO": dai.CameraControl.AutoFocusMode.AUTO,
                "OFF": dai.CameraControl.AutoFocusMode.OFF,
            }

            if mode in mode_map:
                ctrl.setAutoFocusMode(mode_map[mode])
                self.control_queue.send(ctrl)
                logger.info(f"Set autofocus mode to {mode}")
            else:
                logger.warning(f"Unknown autofocus mode: {mode}")

        except Exception as e:
            logger.error(f"Failed to set autofocus mode: {e}")

    def set_exposure(self, exposure_time_us: int, iso_sensitivity: int):
        """Set manual exposure.

        Args:
            exposure_time_us: Exposure time in microseconds
            iso_sensitivity: ISO sensitivity (100-1600)
        """
        if not self.is_initialized or not self.control_queue:
            logger.warning("Camera not initialized")
            return

        try:
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(exposure_time_us, iso_sensitivity)
            self.control_queue.send(ctrl)
            logger.debug(f"Set manual exposure: {exposure_time_us}us, ISO {iso_sensitivity}")
        except Exception as e:
            logger.error(f"Failed to set manual exposure: {e}")

    def set_auto_exposure(self, enable: bool = True):
        """Enable or disable auto exposure.

        Args:
            enable: True to enable auto exposure
        """
        if not self.is_initialized or not self.control_queue:
            logger.warning("Camera not initialized")
            return

        try:
            ctrl = dai.CameraControl()
            if enable:
                ctrl.setAutoExposureEnable()
            else:
                # To disable, set manual exposure to current values
                ctrl.setManualExposure(10000, 800)  # Default values
            self.control_queue.send(ctrl)
            logger.info(f"Auto exposure {'enabled' if enable else 'disabled'}")
        except Exception as e:
            logger.error(f"Failed to set auto exposure: {e}")

    def get_lens_position(self) -> Optional[int]:
        """Get the current lens position (focus value).

        Returns:
            Lens position (0-255) or None if not available
        """
        return self._last_lens_position

    def release(self):
        """Release camera resources."""
        if self.device:
            try:
                self.device.close()
                logger.info("OAK-D camera closed")
            except Exception as e:
                logger.error(f"Error closing OAK-D device: {e}")

        self.is_initialized = False
        self.device = None
        self.pipeline = None
        self.video_queue = None
        self.control_queue = None
        self._last_frame = None

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Destructor."""
        self.release()


def create_oak_d_camera(resolution_4k: bool = True, fps: int = 30, ip_address: Optional[str] = None) -> Optional[OAKDCamera]:
    """Factory function to create an OAK-D camera.

    Args:
        resolution_4k: Use 4K resolution
        fps: Target FPS
        ip_address: IP address for PoE devices (e.g., "169.254.1.222"). If None, uses USB connection

    Returns:
        OAKDCamera instance if successful, None otherwise
    """
    if not DEPTHAI_AVAILABLE:
        logger.error("DepthAI not available. Install with: pip install depthai")
        return None

    camera = OAKDCamera(resolution_4k=resolution_4k, fps=fps, ip_address=ip_address)
    if camera.initialize():
        return camera

    return None
