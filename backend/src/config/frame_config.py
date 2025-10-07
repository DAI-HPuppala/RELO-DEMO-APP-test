"""Frame Storage Configuration"""

import os
from pathlib import Path
from typing import Optional

class FrameConfig:
    """Configuration for frame capture and storage"""
    
    def __init__(self):
        # Base directories
        self.base_dir = Path(__file__).parent.parent.parent  # backend/
        self.captured_frames_dir = self.base_dir / "captured_frames"
        
        # Frame storage settings
        self.save_debug_frames = os.environ.get("SAVE_DEBUG_FRAMES", "false").lower() == "true"
        self.frame_format = "jpeg"  # Format for saved frames
        self.jpeg_quality = int(os.environ.get("JPEG_QUALITY", "85"))  # JPEG compression quality (1-100)
        
        # Frame dimensions (WebRTC standard)
        self.default_width = 640
        self.default_height = 480
        
        # Cleanup settings
        self.cleanup_after_hours = 24  # Delete old frames after 24 hours
        self.max_frames_per_session = 100  # Maximum frames to keep per session
        
    def get_agent_frame_dir(self, agent_name: str) -> Path:
        """Get directory for an agent's captured frames"""
        agent_dir = self.captured_frames_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        return agent_dir

    def get_frame_path(self, agent_name: str, frame_id: str) -> Path:
        """Get full path for a frame file"""
        agent_dir = self.get_agent_frame_dir(agent_name)
        return agent_dir / f"{frame_id}.{self.frame_format}"

    def should_save_frame(self, inference_num: int = 1) -> bool:
        """Determine if frame should be saved for debugging"""
        if not self.save_debug_frames:
            return False
        # Always save first frame and every 5th frame for debugging
        return inference_num == 1 or inference_num % 5 == 0

# Global frame configuration instance
frame_config = FrameConfig()