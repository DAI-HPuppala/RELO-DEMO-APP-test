"""CV60 camera service for WebRTC streaming using eBUS SDK."""
import os
import sys
import site
import numpy as np
import cv2
import logging
import asyncio
import time
import threading
from queue import Queue, Empty
from typing import Optional, Dict, Any
from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)

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
    "/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib"
]:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Add LD_LIBRARY_PATH for native libraries
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/jai/ebus_sdk/Ubuntu-22.04-x86_64/lib'

try:
    import eBUS as eb
    logger.info("Successfully imported eBUS module")
except ImportError as e:
    logger.error(f"ERROR importing eBUS: {e}")
    eb = None

# Constants
BUFFER_COUNT = 16  # Match working test script buffer count
CV60_IP = "192.168.1.21"  # CV60 camera IP


class CV60VideoTrack(VideoStreamTrack):
    """Video track that reads from Zebra CV60 camera via eBUS."""

    kind = "video"

    # Persistent camera contexts (shared across instances)
    _contexts = {}
    _contexts_lock = threading.Lock()

    def __init__(self, session_id: str = None):
        super().__init__()
        self.connection_id = CV60_IP
        self.is_initialized = False
        self._stop_capture = False
        self._frame_queue = Queue(maxsize=30)  # Larger queue to prevent empty situations
        self._capture_thread = None
        self._last_frame_time = 0
        self._target_fps = 30
        self._frame_interval = 1.0 / self._target_fps
        self.current_session_id = session_id  # Store session ID for frame buffer
        self._last_valid_frame = None  # Cache for brief gaps only
        self._last_frame_use_count = 0  # Track how many times we reused last frame
        self._test_mode_logged = False  # Track if test mode fallback has been logged
        self._creation_time = time.time()  # Track when this track was created
        self._initialize_cv60()

    def _initialize_cv60(self):
        """Initialize CV60 camera using eBUS SDK with retry logic."""
        if not eb:
            logger.error("eBUS SDK not available, using test pattern")
            self.is_initialized = False
            self._start_capture_thread()
            return False

        max_retries = 3
        retry_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                with self._contexts_lock:
                    # Check if context already exists for this camera
                    if self.connection_id in self._contexts:
                        ctx = self._contexts[self.connection_id]
                        if ctx and ctx.get('device'):
                            logger.info(f"Reusing existing camera context for {self.connection_id}")
                            self.is_initialized = True
                            self._start_capture_thread()
                            return True

                    logger.info(f"Connecting to CV60 camera at {self.connection_id} (attempt {attempt + 1}/{max_retries})")

                    # Attempt device connection
                    try:
                        result, device = eb.PvDevice.CreateAndConnect(self.connection_id)
                        if not device:
                            raise Exception(f"Failed to connect to device {self.connection_id}")
                    except Exception as e:
                        logger.error(f"Connection error on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            logger.error(f"All {max_retries} connection attempts failed")
                            self.is_initialized = False
                            self._start_capture_thread()
                            return False

                    # Open stream
                    logger.info("Opening stream...")
                    try:
                        result, stream = eb.PvStream.CreateAndOpen(self.connection_id)
                        if not stream:
                            raise Exception("Failed to open stream")
                    except Exception as e:
                        logger.error(f"Stream open error on attempt {attempt + 1}: {e}")
                        device.Disconnect()
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            self.is_initialized = False
                            self._start_capture_thread()
                            return False

                    # Configure packet size for GigE
                    if isinstance(device, eb.PvDeviceGEV):
                        # Use SDK's deterministic method to get the actual stream IP (CRITICAL FIX)
                        stream_ip = stream.GetLocalIPAddress()
                        stream_port = stream.GetLocalPort()

                        logger.info(f"Configuring stream destination: {stream_ip}:{stream_port}")
                        logger.info(f"Camera IP: {self.connection_id}")

                        # Validate that stream IP is on same subnet as camera (basic check)
                        camera_subnet = '.'.join(self.connection_id.split('.')[:3])
                        stream_subnet = '.'.join(str(stream_ip).split('.')[:3])

                        if camera_subnet != stream_subnet:
                            logger.warning(f"  Stream IP {stream_ip} is not on same subnet as camera {self.connection_id}")
                            logger.warning(f"   Camera subnet: {camera_subnet}.x, Stream subnet: {stream_subnet}.x")
                            logger.warning(f"   This may cause frame retrieval issues!")
                        else:
                            logger.info(f" Subnet validation passed: {camera_subnet}.x")

                        device.NegotiatePacketSize()
                        device.SetStreamDestination(stream_ip, stream_port)

                        logger.info(f" Stream destination configured: {stream_ip}:{stream_port}")

                    # Allocate buffers (match working test script exactly)
                    size = device.GetPayloadSize()
                    logger.info(f"Payload size: {size} bytes ({size/1024/1024:.2f} MB)")
                    buffers = []
                    for i in range(BUFFER_COUNT):
                        buf = eb.PvBuffer()
                        buf.Alloc(size)
                        buffers.append(buf)
                        stream.QueueBuffer(buf)
                    logger.info(f"Allocated and queued {BUFFER_COUNT} buffers")

                    # Start acquisition
                    device.StreamEnable()

                    # Execute AcquisitionStart with error checking
                    acq_start = device.GetParameters().Get("AcquisitionStart")
                    if acq_start:
                        result = acq_start.Execute()
                        logger.info(f"AcquisitionStart executed: {result}")
                    else:
                        logger.warning("AcquisitionStart parameter not available")

                    # Give camera time to start streaming (CV60 needs 1s minimum)
                    time.sleep(1.0)

                    # Store context
                    ctx = {
                        "device": device,
                        "stream": stream,
                        "buffers": buffers,
                        "last_capture_time": None,
                        "last_error": None
                    }
                    self._contexts[self.connection_id] = ctx

                    logger.info(f"CV60 camera initialized successfully at {self.connection_id} (attempt {attempt + 1}/{max_retries})")
                    self.is_initialized = True

                    # Start frame capture thread
                    self._start_capture_thread()
                    return True

            except Exception as e:
                logger.error(f"Failed to initialize CV60 on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying camera initialization in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Clean up any partial initialization
                    try:
                        if 'device' in locals() and device:
                            device.Disconnect()
                        if 'stream' in locals() and stream:
                            stream.Close()
                    except:
                        pass
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed to initialize CV60 camera")
                    self.is_initialized = False
                    self._start_capture_thread()
                    return False

        # Should not reach here, but just in case
        self.is_initialized = False
        self._start_capture_thread()
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
        consecutive_errors = 0
        max_consecutive_errors = 10
        last_reconnect_attempt = 0
        reconnect_interval = 30  # Try to reconnect every 30 seconds if in test mode

        while not self._stop_capture:
            try:
                current_time = time.time()

                # Rate limiting - maintain target FPS
                if current_time - self._last_frame_time < self._frame_interval:
                    time.sleep(0.001)
                    continue

                # If not initialized, attempt reconnection periodically
                if not self.is_initialized and eb:
                    if current_time - last_reconnect_attempt > reconnect_interval:
                        logger.info("Attempting to reconnect to CV60 camera...")
                        last_reconnect_attempt = current_time
                        # Try to reinitialize (will use retry logic)
                        if self._initialize_cv60():
                            logger.info("Successfully reconnected to CV60 camera!")
                            consecutive_errors = 0
                            continue
                        else:
                            logger.warning("Failed to reconnect, continuing with test pattern")

                # If not initialized or eBUS not available, generate test frames
                if not self.is_initialized or not eb:
                    reason = "eBUS SDK not available" if not eb else "CV60 camera failed to initialize"
                    frame_data = self._generate_test_pattern(reason)
                    frame_timestamp = current_time

                    # Add to queue
                    try:
                        self._frame_queue.put_nowait((frame_data, frame_timestamp))
                    except:
                        # Queue full, drop oldest frame
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait((frame_data, frame_timestamp))
                        except Empty:
                            pass

                    self._last_frame_time = current_time
                    time.sleep(self._frame_interval)
                    continue

                # Get context
                with self._contexts_lock:
                    ctx = self._contexts.get(self.connection_id)
                    if not ctx:
                        logger.error(f"No context for {self.connection_id}")
                        time.sleep(0.1)
                        continue

                device = ctx["device"]
                stream = ctx["stream"]

                # Retrieve buffer from stream (CV60 needs 5s timeout like eBUSPlayer)
                result, buf, op = stream.RetrieveBuffer(5000)  # 5000ms timeout matching working test script

                if result.IsOK() and op and op.IsOK() and buf.GetPayloadType() == eb.PvPayloadTypeImage:
                    img = buf.GetImage()
                    w, h = img.GetWidth(), img.GetHeight()
                    raw = np.frombuffer(img.GetDataPointer(), dtype=np.uint8).copy()

                    # Convert based on pixel format (no preprocessing)
                    # COMMENTED OUT COLOR CONVERSIONS - Using raw format
                    # if img.GetPixelType() == eb.PvPixelMono8:
                    #     frame_data = cv2.cvtColor(raw.reshape((h, w)), cv2.COLOR_GRAY2BGR)
                    # else:
                    #     # Assuming BayerRG8 format
                    #     temp = raw.reshape((h, w))
                    #     frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerRG2BGR)

                    # CV60 is sending Bayer format data that needs conversion
                    # The eBUS Player does demosaicing for display, but raw stream is still Bayer

                    # Log the pixel type for debugging
                    pixel_type = img.GetPixelType()

                    # Check pixel format and convert accordingly
                    if len(raw) == h * w * 3:
                        # Already 3 channels RGB - use directly
                        frame_data = raw.reshape((h, w, 3))
                        logger.debug(f"Using RGB frame directly: shape {frame_data.shape}")
                    elif len(raw) == h * w:
                        # Single channel Bayer or Mono - needs conversion
                        temp = raw.reshape((h, w))

                        # Check if it's Mono8 format
                        if pixel_type == eb.PvPixelMono8:
                            # Grayscale - convert to BGR
                            frame_data = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
                            logger.debug(f"Converted Mono8 to BGR")
                        else:
                            # Pixel type 17301513 seems to be BayerBG pattern based on color issues
                            # Try BayerBG first (since RG gave us inverted colors)
                            try:
                                # Try BayerBG pattern first
                                frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerBG2BGR)
                            except:
                                try:
                                    # Fallback to BayerRG
                                    frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerRG2BGR)
                                except:
                                    try:
                                        # Try other Bayer patterns
                                        frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerGB2BGR)
                                    except:
                                        try:
                                            frame_data = cv2.cvtColor(temp, cv2.COLOR_BayerGR2BGR)
                                        except:
                                            # Final fallback - duplicate channels
                                            logger.warning(f"All Bayer conversions failed, using grayscale")
                                            frame_data = np.stack([temp, temp, temp], axis=2)
                    else:
                        # Unknown format
                        logger.error(f"Unknown frame format! Size: {len(raw)}, Expected: {h*w} or {h*w*3}")
                        frame_data = np.zeros((h, w, 3), dtype=np.uint8)

                    # Resize to standard resolution if needed
                    # Commenting out resize to allow native CV60 resolution (1760x1696) for frontend
                    # if frame_data.shape[:2] != (480, 640):
                    #     frame_data = cv2.resize(frame_data, (640, 480))

                    frame_timestamp = current_time

                    # Return buffer to stream
                    stream.QueueBuffer(buf)

                    # Add to queue
                    try:
                        self._frame_queue.put_nowait((frame_data, frame_timestamp))
                    except:
                        # Queue full, drop oldest frame
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait((frame_data, frame_timestamp))
                        except Empty:
                            pass

                    self._last_frame_time = current_time
                    consecutive_errors = 0

                    # Update cached frame
                    self._last_valid_frame = frame_data
                    self._last_frame_use_count = 0


                    # Update context
                    ctx['last_capture_time'] = time.time()
                    ctx['last_error'] = None

                else:
                    # Failed to retrieve buffer
                    if buf:
                        stream.QueueBuffer(buf)
                    consecutive_errors += 1

                    # Log the actual error
                    error_msg = f"RetrieveBuffer failed - result: {result.GetCodeString() if result else 'None'}"
                    if consecutive_errors == 1 or consecutive_errors % 5 == 0:
                        logger.warning(f"Frame grab error #{consecutive_errors}: {error_msg}")

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors}), resetting camera. Last error: {error_msg}")
                        self._reset_camera_context()
                        consecutive_errors = 0
                    time.sleep(0.05)

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive frame capture errors ({consecutive_errors})")
                    self._reset_camera_context()
                    consecutive_errors = 0

                if consecutive_errors == 1:
                    logger.debug(f"Frame capture error: {e}")
                time.sleep(0.05)

    def _reset_camera_context(self):
        """Reset the camera context."""
        with self._contexts_lock:
            ctx = self._contexts.pop(self.connection_id, None)
            if ctx:
                try:
                    device = ctx["device"]
                    stream = ctx["stream"]

                    # Stop acquisition
                    try:
                        device.GetParameters().Get("AcquisitionStop").Execute()
                    except:
                        pass

                    # Close stream and disconnect
                    try:
                        stream.Close()
                    except:
                        pass
                    try:
                        device.Disconnect()
                    except:
                        pass

                    logger.info(f"Reset camera context for {self.connection_id}")
                except Exception as e:
                    logger.error(f"Error during context reset: {e}")

        # Mark as not initialized
        self.is_initialized = False

    async def recv(self):
        """Receive the next video frame."""
        current_time = time.time()

        frame = None
        frame_timestamp = current_time
        session_id = getattr(self, 'current_session_id', None)

        # Get latest frame from queue
        if not self._frame_queue.empty():
            try:
                frame_data, frame_timestamp = self._frame_queue.get_nowait()
                frame = frame_data
                # Cache this frame for potential reuse during brief gaps
                self._last_valid_frame = frame_data
                self._last_frame_use_count = 0  # Reset reuse counter

                # Add frame to WebRTC manager's buffer if session exists
                if session_id:
                    try:
                        from services.webrtc_manager import WebRTCManager
                        from api.main import get_webrtc_manager
                        webrtc_manager = get_webrtc_manager()
                        if webrtc_manager:
                            webrtc_manager.add_frame_to_buffer(session_id, frame_data)
                    except Exception as e:
                        logger.debug(f"Could not add frame to buffer: {e}")

                # Track frame count silently
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                else:
                    self._frame_count = 0
            except Empty:
                # Wait longer for a new frame if camera is initialized
                if self.is_initialized:
                    # Try waiting up to 200ms for a new frame
                    for retry in range(4):
                        await asyncio.sleep(0.05)  # Wait 50ms each time
                        try:
                            frame_data, frame_timestamp = self._frame_queue.get_nowait()
                            frame = frame_data
                            self._last_valid_frame = frame_data  # Update cache
                            self._last_frame_use_count = 0  # Reset reuse counter
                            break
                        except Empty:
                            continue
                    else:
                        # Still no frame after waiting
                        # Use last valid frame for brief gaps (up to 3 times)
                        if self._last_valid_frame is not None and self._last_frame_use_count < 3:
                            frame = self._last_valid_frame
                            self._last_frame_use_count += 1
                            logger.debug(f"Using cached frame (reuse #{self._last_frame_use_count})")
                        else:
                            # Only use test pattern if truly no frames available
                            frame = self._generate_test_pattern("No camera frames available after waiting - camera may be disconnected")
                            logger.debug("No frames available, using test pattern")
                else:
                    frame = self._generate_test_pattern("Camera not initialized during frame retrieval")
                    logger.debug("Camera not initialized, using test pattern")
        else:
            # No frames available - wait for new frame if camera is initialized
            if self.is_initialized:
                # Wait up to 200ms for frames to arrive
                for retry in range(4):
                    await asyncio.sleep(0.05)
                    if not self._frame_queue.empty():
                        try:
                            frame_data, frame_timestamp = self._frame_queue.get_nowait()
                            frame = frame_data
                            self._last_valid_frame = frame_data  # Update cache
                            self._last_frame_use_count = 0  # Reset reuse counter
                            break
                        except Empty:
                            continue
                else:
                    # Use cached frame if available
                    if self._last_valid_frame is not None and self._last_frame_use_count < 3:
                        frame = self._last_valid_frame
                        self._last_frame_use_count += 1
                    else:
                        frame = self._generate_test_pattern("Frame queue empty and no cached frames available")
            else:
                frame = self._generate_test_pattern("Camera not initialized and frame queue is empty")

        # Convert to VideoFrame with proper timing
        # Using BGR format for WebRTC compatibility
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")

        # Use aiortc's timestamp system
        pts, time_base = await self.next_timestamp()
        av_frame.pts = pts
        av_frame.time_base = time_base

        return av_frame

    def _generate_test_pattern(self, fallback_reason="Unknown reason"):
        """Generate a test pattern when camera is not available."""
        # Log test mode fallback only once per instance
        if not self._test_mode_logged:
            logger.error(" CV60 CAMERA FALLBACK TO TEST MODE")
            logger.error(f"    Camera IP: {self.connection_id}")
            logger.error(f"    eBUS SDK Available: {eb is not None}")
            logger.error(f"    Camera Initialized: {self.is_initialized}")
            logger.error(f"   ⏰ Track Age: {time.time() - self._creation_time:.1f}s")
            logger.error(f"   🆔 Session ID: {self.current_session_id}")
            logger.error(f"   ❗ Fallback Reason: {fallback_reason}")
            logger.error("    This indicates camera hardware/network issues or page refresh recovery")
            logger.error("    Will attempt reconnection every 30 seconds")
            self._test_mode_logged = True

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add gradient background
        for i in range(480):
            frame[i, :] = [i//4, 50 + i//4, 128]

        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")

        # Add text overlay
        cv2.putText(
            frame,
            "CV60 TEST MODE - NO CAMERA",
            (150, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Camera IP: {self.connection_id}",
            (220, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Time: {timestamp}",
            (250, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        # Add moving element
        moving_x = int((time.time() * 100) % 640)
        cv2.circle(frame, (moving_x, 400), 10, (0, 255, 0), -1)

        return frame

    def stop(self):
        """Stop the video track."""
        # Stop capture thread
        self._stop_capture = True
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)

        # Note: We don't disconnect the camera here as it's shared
        # The context will be cleaned up when needed or on app shutdown

        self.is_initialized = False
        logger.info("CV60 video track stopped")


class CV60CameraService:
    """CV60 camera service for WebRTC."""

    def __init__(self):
        """Initialize the camera service."""
        self.initialized = False
        logger.info("CV60CameraService initialized")

    def initialize_camera(self, camera_source: str = "cv60"):
        """Initialize camera (compatibility method)."""
        self.initialized = True
        logger.info(f"Camera initialized: {camera_source}")
        return {"camera_id": "cv60_zebra", "camera_type": camera_source}

    def get_video_track(self, session_id: str = None):
        """Get a new video track."""
        return CV60VideoTrack(session_id=session_id)

    @staticmethod
    def get_camera_status() -> Dict[str, Any]:
        """Get status of all camera contexts."""
        statuses = []
        with CV60VideoTrack._contexts_lock:
            for cam_id, ctx in CV60VideoTrack._contexts.items():
                statuses.append({
                    "device": cam_id,
                    "stream_open": ctx.get("stream") is not None,
                    "buffers_queued": len(ctx.get("buffers", [])),
                    "last_capture_time": ctx.get("last_capture_time"),
                    "last_error": ctx.get("last_error")
                })
        return {"cameras": statuses}

    @staticmethod
    def reset_camera(connection_id: str = CV60_IP) -> bool:
        """Reset a camera context."""
        with CV60VideoTrack._contexts_lock:
            ctx = CV60VideoTrack._contexts.pop(connection_id, None)
            if not ctx:
                return False

            try:
                device = ctx["device"]
                stream = ctx["stream"]

                # Stop acquisition
                try:
                    device.GetParameters().Get("AcquisitionStop").Execute()
                except:
                    pass

                # Close stream and disconnect
                try:
                    stream.Close()
                except:
                    pass
                try:
                    device.Disconnect()
                except:
                    pass

                logger.info(f"Reset camera {connection_id}")
                return True
            except Exception as e:
                logger.error(f"Error resetting camera: {e}")
                return False