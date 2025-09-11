"""Simple camera service that works with webcam at index 0."""
import cv2
import numpy as np
import logging
from typing import Optional
import asyncio
from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)

class SimpleVideoTrack(VideoStreamTrack):
    """Simple video track that reads from camera index 0."""
    
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera at index 0."""
        # Try index 0 directly (as your hemanth.py uses)
        self.cap = cv2.VideoCapture(0)
        
        if self.cap.isOpened():
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test read
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                logger.info("Camera initialized successfully at index 0")
                return
            else:
                logger.warning("Camera opened but cannot read frames")
                self.cap.release()
                self.cap = None
        else:
            logger.warning("Cannot open camera at index 0")
            self.cap = None
    
    async def recv(self):
        """Receive the next video frame."""
        pts, time_base = await self.next_timestamp()
        
        frame = None
        
        # Read frame from camera
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # If read fails, generate test pattern
                frame = self._generate_test_pattern()
        else:
            # No camera, generate test pattern
            frame = self._generate_test_pattern()
        
        # Convert to VideoFrame
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
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
            "Camera Not Available",
            (150, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "Please check camera at index 0",
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
        if self.cap:
            self.cap.release()
            self.cap = None

class SimpleCameraService:
    """Simple camera service for WebRTC."""
    
    def __init__(self):
        """Initialize the camera service."""
        self.initialized = False
        logger.info("SimpleCameraService initialized")
    
    def initialize_camera(self, camera_source: str = "webcam"):
        """Initialize camera (compatibility method)."""
        self.initialized = True
        logger.info(f"Camera initialized: {camera_source}")
        return {"camera_id": "webcam_0", "camera_type": camera_source}
    
    def get_video_track(self):
        """Get a new video track."""
        return SimpleVideoTrack()