"""CameraFeed data model."""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum


class CameraType(str, Enum):
    """Camera type enumeration."""
    WEBCAM = "webcam"
    REALSENSE = "realsense"
    ZEBRA_CV60 = "zebra"
    BUILTIN = "builtin"


class ConnectionStatus(str, Enum):
    """Connection status enumeration."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ConnectionQuality(str, Enum):
    """Connection quality enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    POOR = "poor"
    UNSTABLE = "unstable"


class CameraFeed(BaseModel):
    """Live video stream configuration and status."""
    
    # Identification
    camera_id: str
    camera_type: CameraType = CameraType.WEBCAM
    
    # Configuration
    resolution_width: int = 640
    resolution_height: int = 480
    fps: int = 30
    backend: str = "opencv"  # OpenCV backend being used
    
    # Connection
    connection_status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    webrtc_peer_id: Optional[str] = None
    
    # Stream Metrics
    frames_captured: int = 0
    frames_dropped: int = 0
    current_fps: float = 0.0
    bandwidth_mbps: float = 0.0
    
    # Health
    last_frame_timestamp: Optional[datetime] = None
    connection_quality: ConnectionQuality = ConnectionQuality.GOOD
    error_count: int = 0
    last_error: Optional[str] = None
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        use_enum_values=True
    )
    
    def connect(self, peer_id: Optional[str] = None):
        """Mark camera as connected."""
        self.connection_status = ConnectionStatus.CONNECTED
        self.webrtc_peer_id = peer_id
        self.error_count = 0
        self.last_error = None
    
    def disconnect(self):
        """Mark camera as disconnected."""
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.webrtc_peer_id = None
    
    def update_metrics(self, fps: float, bandwidth: float):
        """Update stream metrics."""
        self.current_fps = fps
        self.bandwidth_mbps = bandwidth
        self.last_frame_timestamp = datetime.now()
        
        # Update connection quality based on metrics
        if fps >= 25 and bandwidth >= 2.0:
            self.connection_quality = ConnectionQuality.EXCELLENT
        elif fps >= 20 and bandwidth >= 1.5:
            self.connection_quality = ConnectionQuality.GOOD
        elif fps >= 15 and bandwidth >= 1.0:
            self.connection_quality = ConnectionQuality.POOR
        else:
            self.connection_quality = ConnectionQuality.UNSTABLE
    
    def record_error(self, error_message: str):
        """Record an error occurrence."""
        self.error_count += 1
        self.last_error = error_message
        
        if self.error_count > 3:
            self.connection_status = ConnectionStatus.ERROR
    
    def to_status_dict(self) -> dict:
        """Convert to status dictionary."""
        return {
            "camera_id": self.camera_id,
            "type": self.camera_type,
            "status": self.connection_status,
            "resolution": f"{self.resolution_width}x{self.resolution_height}",
            "fps": self.current_fps,
            "quality": self.connection_quality,
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped
        }