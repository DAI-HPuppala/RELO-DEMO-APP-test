"""OAK-D Pro camera service for WebRTC streaming using DepthAI."""
import logging
import asyncio
import time
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any
import numpy as np
import cv2

from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)

# OAK-D PoE fallback IP address (used if auto-discovery fails)
OAKD_POE_FALLBACK_IP = "169.254.1.222"

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
    logger.info("Successfully imported DepthAI module")
except ImportError as e:
    logger.error(f"ERROR importing DepthAI: {e}")
    DEPTHAI_AVAILABLE = False
    dai = None


class OakDVideoTrack(VideoStreamTrack):
    """Video track that reads from OAK-D Pro camera via DepthAI."""

    kind = "video"

    # Camera resolution configuration (change here to update everywhere)
    # High resolution for garment detail capture
    CAMERA_WIDTH = 2400
    CAMERA_HEIGHT = 2160

    def __init__(self, session_id: str = None):
        super().__init__()
        self.session_id = session_id
        self.is_initialized = False
        self._stop_capture = False
        self._frame_queue = Queue(maxsize=1)  # Single frame queue for minimum latency
        self._capture_thread = None
        self._last_frame_time = 0
        self._target_fps = 30
        self._frame_interval = 1.0 / self._target_fps
        self._last_valid_frame = None
        self._device = None
        self._pipeline = None
        self._q_rgb = None
        self._q_control = None

        logger.info(f"OakDVideoTrack initialized for session: {session_id}")
        self._initialize_oakd()

    def _initialize_oakd(self):
        """Initialize OAK-D Pro camera using DepthAI with retry logic."""
        if not DEPTHAI_AVAILABLE:
            logger.error("DepthAI SDK not available, using test pattern")
            self.is_initialized = False
            self._start_capture_thread()
            return False

        max_retries = 3
        retry_delay = 1.5  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing OAK-D Pro camera (attempt {attempt + 1}/{max_retries})...")

                # Create pipeline (exactly like reference)
                self._pipeline = self._create_pipeline()

                # Connect to device and start pipeline
                # Use static IP directly for PoE (more reliable after reloads)
                logger.info(f"Connecting directly to OAK-D PoE at static IP: {OAKD_POE_FALLBACK_IP}")
                device_info = dai.DeviceInfo(OAKD_POE_FALLBACK_IP)

                # Connect to the device
                self._device = dai.Device(self._pipeline, device_info)
                if device_info:
                    logger.info(f"✓ Connected to OAK-D Pro camera successfully")
                else:
                    logger.info(f"✓ Connected to OAK-D Pro camera at fallback IP: {OAKD_POE_FALLBACK_IP}")

                # Get output queue with minimal buffering for low latency
                # Reduced from 4 to 1 to minimize hardware queue delay
                self._q_rgb = self._device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

                # Get control input queue for tap-to-focus
                self._q_control = self._device.getInputQueue(name="control", maxSize=8, blocking=False)

                self.is_initialized = True
                logger.info(f"OAK-D Pro camera initialized successfully (attempt {attempt + 1}/{max_retries})")
                self._start_capture_thread()
                return True

            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1}/{max_retries} failed: {e}")

                # Cleanup partial initialization
                if self._device:
                    try:
                        self._device.close()
                    except:
                        pass
                    self._device = None
                self._q_rgb = None
                self._pipeline = None

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed - falling back to test pattern")
                    self.is_initialized = False
                    self._start_capture_thread()
                    return False

    def _create_pipeline(self):
        """Create DepthAI pipeline for RGB camera with full ISP optimization."""
        pipeline = dai.Pipeline()

        # Define source - RGB camera with ISP enhancements
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setVideoSize(self.CAMERA_WIDTH, self.CAMERA_HEIGHT)  # Configurable resolution
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(30)

        # ============================================================
        # ISP ENHANCEMENTS (Hardware-accelerated, 0% CPU/FPS impact)
        # ============================================================

        # Image quality controls (all via initialControl)
        cam_rgb.initialControl.setSharpness(1)         # Slight sharpening for fabric texture (0-4)
        cam_rgb.initialControl.setLumaDenoise(1)       # Reduce brightness noise (0-4)
        cam_rgb.initialControl.setChromaDenoise(1)     # Reduce color noise - reduced from 4 to prevent desaturation (0-4)
        cam_rgb.initialControl.setSaturation(0)        # Neutral saturation - no boost needed with lower chroma denoise (0-4)
        # 3A (Auto-Exposure, Auto-White Balance) controls
        cam_rgb.initialControl.setAutoExposureEnable()  # Enable auto-exposure
        cam_rgb.initialControl.setAutoWhiteBalanceMode(
            dai.CameraControl.AutoWhiteBalanceMode.AUTO
        )  # Auto white balance for natural colors
        cam_rgb.initialControl.setAutoExposureLimit(22000)  # 33ms max (prevents motion blur at 30fps)
        cam_rgb.initialControl.setAntiBandingMode(
            dai.CameraControl.AntiBandingMode.MAINS_60_HZ
        )  # Reduce 50Hz flicker (change to MAINS_60_HZ for US/Canada)
        cam_rgb.initialControl.setAutoExposureCompensation(-2)  # Neutral exposure


        # Create output (exactly like reference)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam_rgb.video.link(xout.input)

        # Create control input for tap-to-focus
        xin_control = pipeline.create(dai.node.XLinkIn)
        xin_control.setStreamName("control")
        xin_control.out.link(cam_rgb.inputControl)

        return pipeline

    def _start_capture_thread(self):
        """Start the frame capture thread."""
        if self._capture_thread and self._capture_thread.is_alive():
            return

        self._stop_capture = False
        self._capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._capture_thread.start()
        logger.info("Frame capture thread started")

    def _capture_frames(self):
        """Capture frames from OAK-D Pro camera or generate test pattern."""
        while not self._stop_capture:
            try:
                frame = None

                if self.is_initialized and self._q_rgb:
                    # Get frame from OAK-D Pro
                    in_rgb = self._q_rgb.get()
                    if in_rgb is not None:
                        # Already in BGR format from camera
                        frame = in_rgb.getCvFrame()
                        self._last_valid_frame = frame.copy()
                        # Log raw frame resolution from OAK-D Pro only once
                        if not hasattr(self, '_resolution_logged'):
                            logger.info(f"📐 OAK-D Pro raw frame resolution: {frame.shape[1]}x{frame.shape[0]} (WxH)")
                            self._resolution_logged = True

                # Generate test pattern if no real frame
                if frame is None:
                    frame = self._generate_test_pattern()

                # Add frame to queue
                try:
                    self._frame_queue.put_nowait(frame)
                except:
                    # Queue full, remove oldest frame and add newest
                    try:
                        self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
                    except:
                        pass

                # No artificial delay - OAK-D controls frame rate at hardware level

            except Exception as e:
                logger.error(f"Error in capture thread: {e}")
                time.sleep(0.1)

    def _generate_test_pattern(self):
        """Generate test pattern when camera is not available."""
        frame = np.zeros((self.CAMERA_HEIGHT, self.CAMERA_WIDTH, 3), dtype=np.uint8)

        # Add gradient background
        for i in range(self.CAMERA_HEIGHT):
            frame[i, :] = [(i//8) % 256, 100, (255 - i//8) % 256]

        # Add text to indicate test mode (scaled for 4K)
        if self.is_initialized:
            text = "OAK-D Pro - No Frame"
            subtext = "Camera connected but no data"
        else:
            text = "OAK-D Pro - Test Pattern"
            subtext = "Camera not available"

        cv2.putText(frame, text, (1200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, subtext, (1300, 1200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4, cv2.LINE_AA)

        # Add timestamp (scaled for 4K)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 255, 0), 6, cv2.LINE_AA)

        return frame

    async def recv(self):
        """Receive next video frame for WebRTC."""
        pts, time_base = await self.next_timestamp()

        # Get ONLY the latest frame - don't waste time extracting all frames
        frame = None
        try:
            # Try to get the newest frame without extracting all
            frame = self._frame_queue.get_nowait()

            # If there are more frames queued, clear them and keep only the newest
            if not self._frame_queue.empty():
                # Clear old frames quickly
                while not self._frame_queue.empty():
                    try:
                        frame = self._frame_queue.get_nowait()  # Keep updating to newest
                    except Empty:
                        break
        except Empty:
            # Queue is empty - use last valid frame
            if self._last_valid_frame is not None:
                frame = self._last_valid_frame  # No copy needed, read-only
            else:
                # Only generate test pattern if camera truly isn't working
                # Wait a tiny bit to see if a frame arrives
                await asyncio.sleep(0.01)  # 10ms wait
                try:
                    frame = self._frame_queue.get_nowait()
                except Empty:
                    # Still no frame - camera might be initializing or disconnected
                    if self.is_initialized:
                        # Camera is initialized but no frames - keep using last valid if available
                        if self._last_valid_frame is not None:
                            frame = self._last_valid_frame
                        else:
                            logger.warning("No frames available from initialized camera")
                            frame = self._generate_test_pattern()
                    else:
                        frame = self._generate_test_pattern()

        # Convert frame to VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame

    def set_focus_point(self, x: float, y: float):
        """
        Set manual focus point via tap-to-focus.

        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
        """
        if not self.is_initialized or not self._device:
            logger.warning("Cannot set focus point - camera not initialized")
            return

        try:
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(x * self.CAMERA_WIDTH)
            pixel_y = int(y * self.CAMERA_HEIGHT)

            # Calculate focus region around tap point (200x200 pixel region)
            region_width = 200
            region_height = 200
            start_x = max(0, pixel_x - region_width // 2)
            start_y = max(0, pixel_y - region_height // 2)

            # Ensure region doesn't exceed camera bounds
            if start_x + region_width > self.CAMERA_WIDTH:
                start_x = self.CAMERA_WIDTH - region_width
            if start_y + region_height > self.CAMERA_HEIGHT:
                start_y = self.CAMERA_HEIGHT - region_height

            # Create and send camera control command
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusRegion(startX=start_x, startY=start_y, width=region_width, height=region_height)
            ctrl.setAutoFocusTrigger()

            # Send control to camera via control queue
            if self._q_control:
                self._q_control.send(ctrl)
                logger.info(f"✓ Tap-to-focus: ({pixel_x}, {pixel_y}), region ({start_x}, {start_y}, {region_width}x{region_height})")
            else:
                logger.warning("Camera control queue not available")

        except Exception as e:
            logger.error(f"Error setting focus point: {e}")

    def stop(self):
        """Stop the video track and cleanup resources."""
        logger.info("Stopping OAK-D Pro video track")
        self._stop_capture = True

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # Clear queues first
        self._q_rgb = None
        self._q_control = None

        # Close device with proper cleanup
        if self._device:
            try:
                logger.info("Closing OAK-D PoE device connection...")
                self._device.close()
                # Add small delay for PoE device to fully disconnect
                import time
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Error closing device: {e}")
            finally:
                self._device = None

        self._pipeline = None
        logger.info("OAK-D Pro video track stopped and resources cleaned")


class OakDCameraService:
    """OAK-D Pro camera service for WebRTC (replaces CV60CameraService)."""

    def __init__(self):
        """Initialize the camera service."""
        self.initialized = False
        logger.info("OakDCameraService initialized")

    @staticmethod
    def get_camera_resolution() -> Dict[str, int]:
        """Get the configured camera resolution."""
        return {
            "width": OakDVideoTrack.CAMERA_WIDTH,
            "height": OakDVideoTrack.CAMERA_HEIGHT,
            "aspect_ratio": round(OakDVideoTrack.CAMERA_WIDTH / OakDVideoTrack.CAMERA_HEIGHT, 4)
        }

    def initialize_camera(self, camera_source: str = "oakd"):
        """Initialize camera (compatibility method)."""
        self.initialized = True
        logger.info(f"Camera initialized: {camera_source}")
        return {"camera_id": "oakd_pro", "camera_type": camera_source}

    def get_video_track(self, session_id: str = None):
        """Get a new video track."""
        return OakDVideoTrack(session_id=session_id)

    def capture_depth_frame(self):
        """
        Capture a single depth frame for height measurement.
        Creates a temporary pipeline with stereo depth.

        Returns:
            numpy.ndarray: Depth map in millimeters, or None on failure
        """
        if not DEPTHAI_AVAILABLE:
            logger.error("DepthAI not available for depth capture")
            return None

        try:
            logger.info("Creating stereo depth pipeline for height measurement...")

            # Create pipeline with stereo depth
            pipeline = dai.Pipeline()

            # Mono cameras for stereo depth
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)
            stereo = pipeline.create(dai.node.StereoDepth)

            # Configure mono cameras
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_left.setFps(30)
            mono_right.setFps(30)

            # Configure stereo depth
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(False)
            stereo.setSubpixel(True)

            # Link mono cameras to stereo
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            # Create output for depth
            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

            # Connect to device
            with dai.Device(pipeline) as device:
                logger.info("✅ Connected to OAK-D for depth capture")

                # Get depth queue
                q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

                # Wait a moment for auto-exposure to stabilize
                import time
                time.sleep(0.5)

                # Capture depth frame
                depth_frame = q_depth.get()
                depth_array = depth_frame.getFrame()

                logger.info(f"📏 Captured depth frame: {depth_array.shape}")
                return depth_array

        except Exception as e:
            logger.error(f"Error capturing depth frame: {e}")
            return None

    @staticmethod
    def get_camera_status() -> Dict[str, Any]:
        """Get status of OAK-D Pro camera."""
        if not DEPTHAI_AVAILABLE:
            return {
                "available": False,
                "error": "DepthAI SDK not available",
                "devices": []
            }

        try:
            # Check for available OAK devices
            devices = dai.Device.getAllAvailableDevices()
            device_list = []

            for device_info in devices:
                device_list.append({
                    "name": device_info.name,
                    "state": device_info.state.name,
                    "protocol": device_info.protocol.name
                })

            return {
                "available": len(devices) > 0,
                "device_count": len(devices),
                "devices": device_list,
                "sdk_version": dai.__version__ if hasattr(dai, '__version__') else "unknown"
            }

        except Exception as e:
            logger.error(f"Error getting OAK-D camera status: {e}")
            return {
                "available": False,
                "error": str(e),
                "devices": []
            }