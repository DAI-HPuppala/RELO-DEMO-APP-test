"""Camera service for handling RealSense and other camera sources."""
import asyncio
import logging
import numpy as np
from typing import Optional, Any
from datetime import datetime
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logging.warning("pyrealsense2 not available. RealSense support disabled.")

from aiortc import VideoStreamTrack
from av import VideoFrame

from models import CameraFeed, CameraType, ConnectionStatus

logger = logging.getLogger(__name__)


class CameraService:
    """Service for managing camera connections and streams."""
    
    def __init__(self):
        """Initialize camera service."""
        self.camera_feed = None
        self.pipeline = None
        self.config = None
        self.current_source = None
        self.is_streaming = False
    
    def initialize_camera(self, camera_type: str = "realsense") -> CameraFeed:
        """Initialize camera based on type."""
        camera_feed = CameraFeed(
            camera_id=f"{camera_type}_001",
            camera_type=CameraType(camera_type)
        )
        
        try:
            if camera_type == "realsense" and REALSENSE_AVAILABLE:
                self._initialize_realsense(camera_feed)
            else:
                self._initialize_webcam(camera_feed)
            
            camera_feed.connect()
            self.camera_feed = camera_feed
            logger.info(f"Camera initialized: {camera_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            camera_feed.connection_status = ConnectionStatus.ERROR
            camera_feed.record_error(str(e))
        
        return camera_feed
    
    def _initialize_realsense(self, camera_feed: CameraFeed):
        """Initialize Intel RealSense camera."""
        if not REALSENSE_AVAILABLE:
            raise ValueError("RealSense support not available")
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(
            rs.stream.color, 
            camera_feed.resolution_width,
            camera_feed.resolution_height,
            rs.format.bgr8,
            camera_feed.fps
        )
        
        # Start pipeline
        profile = self.pipeline.start(self.config)
        
        # Get device information
        device = profile.get_device()
        camera_feed.camera_id = device.get_info(rs.camera_info.serial_number)
        
        logger.info(f"RealSense initialized: {device.get_info(rs.camera_info.name)}")
    
    def _initialize_webcam(self, camera_feed: CameraFeed):
        """Initialize standard webcam."""
        # Will be handled by OpenCV VideoCapture in the track
        camera_feed.camera_type = CameraType.WEBCAM
        logger.info("Webcam initialized")
    
    def get_video_track(self) -> 'CameraVideoTrack':
        """Get video track for WebRTC streaming."""
        if not self.camera_feed:
            self.initialize_camera()
        
        return CameraVideoTrack(self)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the camera."""
        try:
            if self.pipeline and REALSENSE_AVAILABLE:
                # Get frame from RealSense
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    return None
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Update metrics
                if self.camera_feed:
                    self.camera_feed.frames_captured += 1
                
                return frame
            
            else:
                # Get frame from webcam (handled in VideoTrack)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get frame: {e}")
            if self.camera_feed:
                self.camera_feed.record_error(str(e))
            return None
    
    def stop_camera(self):
        """Stop camera streaming."""
        if self.pipeline and REALSENSE_AVAILABLE:
            self.pipeline.stop()
            self.pipeline = None
        
        if self.camera_feed:
            self.camera_feed.disconnect()
        
        self.is_streaming = False
        logger.info("Camera stopped")
    
    def get_camera_status(self) -> dict:
        """Get current camera status."""
        if not self.camera_feed:
            return {
                "connected": False,
                "type": "none",
                "error": "Camera not initialized"
            }
        
        return self.camera_feed.to_status_dict()


class CameraVideoTrack(VideoStreamTrack):
    """Video track for streaming camera feed via WebRTC."""
    
    kind = "video"
    
    def __init__(self, camera_service: CameraService):
        """Initialize video track."""
        super().__init__()
        self.camera_service = camera_service
        self.cap = None
        
        # Initialize webcam if not using RealSense
        if not camera_service.pipeline:
            # Try to find an available camera using simple index approach
            self.cap = None
            
            # Try direct index 0 first (as hemanth.py uses index 0)
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    logger.info(f"Using camera at index 0")
                else:
                    self.cap.release()
                    self.cap = None
            except Exception as e:
                logger.debug(f"Could not open camera at index 0: {e}")
                self.cap = None
            
            # If index 0 fails, try other indices
            if not self.cap:
                for i in [4, 2, 1, 3, 5]:
                    try:
                        self.cap = cv2.VideoCapture(i)
                        if self.cap.isOpened():
                            ret, test_frame = self.cap.read()
                            if ret and test_frame is not None:
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.cap.set(cv2.CAP_PROP_FPS, 30)
                                logger.info(f"Using camera at index {i}")
                                break
                            else:
                                self.cap.release()
                                self.cap = None
                        else:
                            self.cap.release()
                            self.cap = None
                    except Exception as e:
                        logger.debug(f"Could not open camera at index {i}: {e}")
                        self.cap = None
            
            if not self.cap:
                logger.warning("No camera found, will generate test pattern")
    
    async def recv(self):
        """Receive next video frame."""
        pts, time_base = await self.next_timestamp()
        
        frame = None
        
        # Get frame from RealSense
        if self.camera_service.pipeline:
            frame = self.camera_service.get_frame()
        
        # Get frame from webcam
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                frame = None
        
        # Generate a test pattern if no camera
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add gradient background
            for i in range(480):
                frame[i, :] = [i//2, 100, 255 - i//2]
            
            # Add text to indicate no camera
            cv2.putText(
                frame,
                "Test Pattern - No Camera",
                (120, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                "Connect a camera to start",
                (140, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Convert frame to VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        
        return av_frame
    
    def stop(self):
        """Stop the video track."""
        if self.cap:
            self.cap.release()
            self.cap = None