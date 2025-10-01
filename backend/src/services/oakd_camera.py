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

    def __init__(self, session_id: str = None):
        super().__init__()
        self.session_id = session_id
        self.is_initialized = False
        self._stop_capture = False
        self._frame_queue = Queue(maxsize=2)  # Minimal queue for low latency
        self._capture_thread = None
        self._last_frame_time = 0
        self._target_fps = 30
        self._frame_interval = 1.0 / self._target_fps
        self._last_valid_frame = None
        self._device = None
        self._pipeline = None
        self._q_rgb = None

        logger.info(f"OakDVideoTrack initialized for session: {session_id}")
        self._initialize_oakd()

    def _initialize_oakd(self):
        """Initialize OAK-D Pro camera using DepthAI."""
        if not DEPTHAI_AVAILABLE:
            logger.error("DepthAI SDK not available, using test pattern")
            self.is_initialized = False
            self._start_capture_thread()
            return False

        try:
            logger.info("Initializing OAK-D Pro camera...")

            # Create pipeline (exactly like reference)
            self._pipeline = self._create_pipeline()

            # Connect to device and start pipeline
            # Try dynamic discovery first, then fall back to static IP
            device_info = None
            devices = dai.Device.getAllAvailableDevices()

            # Look for PoE device via auto-discovery
            for dev in devices:
                if dev.protocol == dai.XLinkProtocol.X_LINK_TCP_IP:
                    logger.info(f"✓ Dynamically discovered PoE device: {dev.name}")
                    device_info = dev
                    break

            # Fall back to static IP if no device found via discovery
            if device_info is None:
                logger.info(f"No PoE device found via auto-discovery, falling back to static IP: {OAKD_POE_FALLBACK_IP}")
                device_info = dai.DeviceInfo(OAKD_POE_FALLBACK_IP)

            # Connect to the device
            self._device = dai.Device(self._pipeline, device_info)
            if device_info:
                logger.info(f"✓ Connected to OAK-D Pro camera successfully")
            else:
                logger.info(f"✓ Connected to OAK-D Pro camera at fallback IP: {OAKD_POE_FALLBACK_IP}")

            # Get output queue (exactly like reference)
            self._q_rgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            self.is_initialized = True
            logger.info("OAK-D Pro camera initialized successfully")
            self._start_capture_thread()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize OAK-D Pro camera: {e}")
            self.is_initialized = False
            self._start_capture_thread()
            return False

    def _create_pipeline(self):
        """Create DepthAI pipeline for RGB camera with full ISP optimization."""
        pipeline = dai.Pipeline()

        # Define source - RGB camera with ISP enhancements
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setVideoSize(1796, 2160)  # True 4K resolution
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
        cam_rgb.initialControl.setChromaDenoise(4)     # Reduce color noise - max recommended (0-4)

        # 3A (Auto-Exposure, Auto-White Balance) controls
        cam_rgb.initialControl.setAutoExposureEnable()  # Enable auto-exposure
        cam_rgb.initialControl.setAutoWhiteBalanceMode(
            dai.CameraControl.AutoWhiteBalanceMode.AUTO
        )  # Auto white balance for natural colors
        cam_rgb.initialControl.setAutoExposureLimit(33000)  # 33ms max (prevents motion blur at 30fps)
        cam_rgb.initialControl.setAntiBandingMode(
            dai.CameraControl.AntiBandingMode.MAINS_50_HZ
        )  # Reduce 50Hz flicker (change to MAINS_60_HZ for US/Canada)
        cam_rgb.initialControl.setAutoExposureCompensation(0)  # Neutral exposure


        # Create output (exactly like reference)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")
        cam_rgb.video.link(xout.input)

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
        frame = np.zeros((2160, 1796, 3), dtype=np.uint8)

        # Add gradient background
        for i in range(2160):
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

        # Get LATEST frame from queue (skip old frames for low latency)
        frame = None
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
            except Empty:
                break

        # Use last valid frame if queue is empty
        if frame is None:
            if self._last_valid_frame is not None:
                frame = self._last_valid_frame.copy()
            else:
                frame = self._generate_test_pattern()

        # Convert frame to VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame

    def stop(self):
        """Stop the video track and cleanup resources."""
        logger.info("Stopping OAK-D Pro video track")
        self._stop_capture = True

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        if self._device:
            try:
                self._device.close()
            except:
                pass
            self._device = None

        self._q_rgb = None
        self._pipeline = None
        logger.info("OAK-D Pro video track stopped")


class OakDCameraService:
    """OAK-D Pro camera service for WebRTC (replaces CV60CameraService)."""

    def __init__(self):
        """Initialize the camera service."""
        self.initialized = False
        logger.info("OakDCameraService initialized")

    def initialize_camera(self, camera_source: str = "oakd"):
        """Initialize camera (compatibility method)."""
        self.initialized = True
        logger.info(f"Camera initialized: {camera_source}")
        return {"camera_id": "oakd_pro", "camera_type": camera_source}

    def get_video_track(self, session_id: str = None):
        """Get a new video track."""
        return OakDVideoTrack(session_id=session_id)

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