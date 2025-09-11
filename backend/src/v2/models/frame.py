"""Frame model for captured image data"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from uuid import uuid4


@dataclass
class Frame:
    """Captured frame with metadata"""
    
    # Identification
    frame_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    agent_name: str = ""
    
    # Frame data
    data: bytes = b""  # Image bytes (JPEG encoded)
    width: int = 640
    height: int = 480
    format: str = "bgr24"  # "bgr24", "rgb24", etc.
    
    # Metadata
    captured_at: datetime = field(default_factory=datetime.now)
    frame_number: int = 0  # Sequential number from stream
    is_shared: bool = False  # True if shared between agents
    shared_with: List[str] = field(default_factory=list)  # Agent names
    
    # Storage
    file_path: Optional[str] = None  # Debug storage location
    
    def share_with(self, agent_name: str) -> None:
        """Mark frame as shared with an agent"""
        if agent_name not in self.shared_with:
            self.shared_with.append(agent_name)
            self.is_shared = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without binary data)"""
        return {
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "captured_at": self.captured_at.isoformat(),
            "frame_number": self.frame_number,
            "is_shared": self.is_shared,
            "shared_with": self.shared_with,
            "file_path": self.file_path,
            "data_size": len(self.data)
        }
