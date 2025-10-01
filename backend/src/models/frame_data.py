"""FrameData data model."""
from typing import Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field
import uuid


class FrameData(BaseModel):
    """Individual frame extracted from video stream."""
    
    # Identification
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    
    # Frame Info
    timestamp: datetime = Field(default_factory=datetime.now)
    sequence_number: int
    
    # Image Data
    width: int
    height: int
    format: str = "RGB"  # RGB, BGR, etc.
    image_data: Optional[bytes] = Field(default=None, exclude=True)  # Raw frame data (deleted after processing)
    
    # Processing
    processed: bool = False
    processing_agent: Optional[str] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    
    # Auto-deletion
    marked_for_deletion: bool = False
    deletion_scheduled: Optional[datetime] = None
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    def start_processing(self, agent_name: str):
        """Mark frame as being processed by an agent."""
        self.processing_agent = agent_name
        self.processing_started = datetime.now()
    
    def complete_processing(self):
        """Mark frame processing as complete."""
        self.processed = True
        self.processing_completed = datetime.now()
        self.schedule_deletion()
    
    def schedule_deletion(self, delay_seconds: int = 5):
        """Schedule frame data for deletion."""
        self.marked_for_deletion = True
        self.deletion_scheduled = datetime.now() + timedelta(seconds=delay_seconds)
    
    def delete_image_data(self):
        """Delete the raw image data to free memory."""
        self.image_data = None
        self.marked_for_deletion = False
        self.deletion_scheduled = None
    
    def to_metadata_dict(self) -> dict:
        """Convert to metadata dictionary (without image data)."""
        return {
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp.isoformat(),
            "dimensions": f"{self.width}x{self.height}",
            "format": self.format,
            "processed": self.processed,
            "processing_agent": self.processing_agent
        }