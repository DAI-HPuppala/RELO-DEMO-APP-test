"""RealSense camera service for WebRTC streaming."""
import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import asyncio
import time
import threading
from queue import Queue, Empty
from typing import Optional
from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)

class RealSenseVideoTrack(VideoStreamTrack):
    """Video track that reads from Intel RealSense camera."""
    
    kind = "video"
    
    def __init__(self, session_id: str = None):
        super().__init__()
        self.pipeline = None
        self.config = None
        self.is_initialized = False
        self._stop_capture = False
        self._frame_queue = Queue(maxsize=2)  # Small queue for real-time
        self._capture_thread = None
        self._last_frame_time = 0
        self._target_fps = 30
        self._frame_interval = 1.0 / self._target_fps
        self.current_session_id = session_id  # Store session ID for frame buffer
        self._initialize_realsense()
    
    def _initialize_realsense(self):
        """Initialize RealSense camera."""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Get connected devices
            ctx = rs.context()
            devices = ctx.devices
            
            if len(devices) == 0:
                logger.warning("No RealSense devices found")
                return False
            
            # Get first device
            device = devices[0]
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            
            logger.info(f"Found RealSense device: {name} (Serial: {serial})")
            
            # Enable device by serial number
            self.config.enable_device(serial)
            
            # Configure streams - use same settings as RELO-DEMO-APP
            self.config.enable_stream(
                rs.stream.color, 
                640, 
                480, 
                rs.format.bgr8, 
                30
            )
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Configure camera settings like RELO-DEMO-APP
            device = profile.get_device()
            
            # Find color sensor
            color_sensor = None
            for sensor in device.query_sensors():
                if sensor.get_info(rs.camera_info.name).lower().find('rgb') != -1:
                    color_sensor = sensor
                    break
            
            if not color_sensor and len(device.query_sensors()) > 1:
                color_sensor = device.query_sensors()[1]
            
            if color_sensor:
                # Disable auto-exposure for manual control
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    logger.info("Disabled auto-exposure")
                
                # Set manual exposure (low value like RELO-DEMO-APP)
                if color_sensor.supports(rs.option.exposure):
                    color_sensor.set_option(rs.option.exposure, 200)  # 200 microseconds
                    logger.info("Set manual exposure to 200")
                
                # Set gain
                if color_sensor.supports(rs.option.gain):
                    color_sensor.set_option(rs.option.gain, 16)
                    logger.info("Set gain to 16")
                
                # Set white balance
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                    logger.info("Enabled auto white balance")
            
            # Quick warmup - 5 frames
            logger.info("Camera warmup...")
            for _ in range(5):
                self.pipeline.wait_for_frames()
            
            self.is_initialized = True
            logger.info("RealSense camera initialized successfully")
            
            # Start frame capture thread for real-time streaming
            self._start_capture_thread()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RealSense: {str(e)}")
            self.pipeline = None
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
        while not self._stop_capture and self.pipeline and self.is_initialized:
            try:
                current_time = time.time()
                
                # Rate limiting - maintain target FPS
                if current_time - self._last_frame_time < self._frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy wait
                    continue
                
                # Get frame from RealSense
                frames = self.pipeline.wait_for_frames(timeout_ms=33)  # ~30 FPS timeout
                color_frame = frames.get_color_frame()
                
                if color_frame:
                    # Convert to numpy array
                    frame_data = np.asanyarray(color_frame.get_data())
                    frame_timestamp = current_time
                    
                    # Add to queue (drop old frames if queue is full)
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
                    
            except Exception as e:
                logger.debug(f"Frame capture error: {e}")
                time.sleep(0.01)  # Brief pause on error
    
    async def recv(self):
        """Receive the next video frame."""
        # Use real timestamp for proper timing
        current_time = time.time()
        
        frame = None
        frame_timestamp = current_time
        session_id = getattr(self, 'current_session_id', None)
        
        # Get latest frame from queue
        if self.is_initialized and not self._frame_queue.empty():
            try:
                frame_data, frame_timestamp = self._frame_queue.get_nowait()
                frame = frame_data
                
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
                frame = self._generate_test_pattern()
                logger.debug("Queue empty, using test pattern")
        else:
            # No camera or no frames, generate test pattern
            frame = self._generate_test_pattern()
            if not self.is_initialized:
                logger.debug("Camera not initialized, using test pattern")
        
        # Convert to VideoFrame with proper timing
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        
        # Use aiortc's timestamp system but ensure proper timing
        pts, time_base = await self.next_timestamp()
        av_frame.pts = pts
        av_frame.time_base = time_base
        
        return av_frame
    
    def _generate_test_pattern(self):
        """Generate a test pattern when camera is not available."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add gradient background
        for i in range(480):
            frame[i, :] = [i//2, 100, 255 - i//2]
        
        # Add text
        cv2.putText(
            frame,
            "RealSense Not Available",
            (150, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Check RealSense connection",
            (120, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        return frame
    
    def stop(self):
        """Stop the video track."""
        # Stop capture thread
        self._stop_capture = True
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        
        # Stop pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        
        self.is_initialized = False
        logger.info("RealSense video track stopped")

class RealSenseCameraService:
    """RealSense camera service for WebRTC."""
    
    def __init__(self):
        """Initialize the camera service."""
        self.initialized = False
        logger.info("RealSenseCameraService initialized")
    
    def initialize_camera(self, camera_source: str = "realsense"):
        """Initialize camera (compatibility method)."""
        self.initialized = True
        logger.info(f"Camera initialized: {camera_source}")
        return {"camera_id": "realsense_d415", "camera_type": camera_source}
    
    def get_video_track(self, session_id: str = None):
        """Get a new video track."""
        return RealSenseVideoTrack(session_id=session_id)