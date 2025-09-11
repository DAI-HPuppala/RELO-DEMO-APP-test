"""Frame Registry service for managing frame lifecycle and sharing"""

import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from uuid import uuid4
import time

from ..models.frame import Frame

logger = logging.getLogger(__name__)


class FrameRegistry:
    """Manages frame lifecycle and sharing between agents"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frames: Dict[str, Frame] = {}  # frame_id -> Frame
        
        # Sharing management
        self.shared_frames: Dict[str, str] = {}  # share_key -> frame_id
        self.reference_counts: Dict[str, int] = {}  # frame_id -> ref count
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
        logger.info(f"FrameRegistry initialized for session {session_id}")
    
    async def register_frame(self, frame_data: np.ndarray, agent_name: str, 
                            frame_number: int = 0) -> str:
        """Register a new frame and return its ID"""
        async with self._lock:
            frame_id = f"{agent_name}_{int(time.time()*1000)}_{uuid4().hex[:8]}"
            
            # Convert numpy array to bytes (JPEG encoding would go here)
            frame_bytes = frame_data.tobytes()
            
            frame = Frame(
                frame_id=frame_id,
                session_id=self.session_id,
                agent_name=agent_name,
                data=frame_bytes,
                width=frame_data.shape[1] if len(frame_data.shape) > 1 else 640,
                height=frame_data.shape[0] if len(frame_data.shape) > 0 else 480,
                frame_number=frame_number
            )
            
            self.frames[frame_id] = frame
            self.reference_counts[frame_id] = 1
            
            logger.debug(f"Registered frame {frame_id} for agent {agent_name}")
            return frame_id
    
    async def get_frame(self, frame_id: str) -> Optional[Frame]:
        """Get a frame by ID"""
        async with self._lock:
            return self.frames.get(frame_id)
    
    async def share_frame(self, frame_id: str, share_key: str) -> bool:
        """Share a frame with a specific key (e.g., 'initial_first')"""
        async with self._lock:
            if frame_id not in self.frames:
                logger.error(f"Cannot share non-existent frame {frame_id}")
                return False
            
            self.shared_frames[share_key] = frame_id
            self.reference_counts[frame_id] = self.reference_counts.get(frame_id, 0) + 1
            
            frame = self.frames[frame_id]
            frame.is_shared = True
            
            logger.info(f"Shared frame {frame_id} with key '{share_key}'")
            return True
    
    async def get_shared_frame(self, share_key: str) -> Optional[Frame]:
        """Get a shared frame by its share key"""
        async with self._lock:
            frame_id = self.shared_frames.get(share_key)
            if frame_id:
                return self.frames.get(frame_id)
            return None
    
    async def register_initial_first_frame(self, frame_data: np.ndarray) -> str:
        """Special method to register and share initial classifier's first frame"""
        frame_id = await self.register_frame(frame_data, "initial_classifier", 1)
        await self.share_frame(frame_id, "initial_first")
        
        # Mark it as shared with damage_detector
        async with self._lock:
            if frame_id in self.frames:
                self.frames[frame_id].share_with("damage_detector")
        
        logger.info(f"Registered initial first frame: {frame_id} for damage detector sharing")
        return frame_id
    
    async def release_frame(self, frame_id: str) -> None:
        """Release a frame reference and delete if no longer needed"""
        async with self._lock:
            if frame_id not in self.reference_counts:
                return
            
            self.reference_counts[frame_id] -= 1
            
            if self.reference_counts[frame_id] <= 0:
                # No more references, safe to delete
                if frame_id in self.frames:
                    del self.frames[frame_id]
                del self.reference_counts[frame_id]
                
                # Remove from shared frames if present
                keys_to_remove = [k for k, v in self.shared_frames.items() if v == frame_id]
                for key in keys_to_remove:
                    del self.shared_frames[key]
                
                logger.debug(f"Released and deleted frame {frame_id}")
    
    async def get_frame_for_damage_detector(self) -> Optional[np.ndarray]:
        """Get the shared initial frame for damage detector's first inference"""
        frame = await self.get_shared_frame("initial_first")
        if frame and frame.data:
            # Convert bytes back to numpy array
            # In real implementation, would decode from JPEG
            try:
                array = np.frombuffer(frame.data, dtype=np.uint8)
                array = array.reshape((frame.height, frame.width, 3))
                return array
            except Exception as e:
                logger.error(f"Failed to decode shared frame: {e}")
        return None
    
    async def cleanup(self) -> None:
        """Clean up all frames for this session"""
        async with self._lock:
            frame_count = len(self.frames)
            self.frames.clear()
            self.shared_frames.clear()
            self.reference_counts.clear()
            logger.info(f"Cleaned up {frame_count} frames for session {self.session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "session_id": self.session_id,
            "total_frames": len(self.frames),
            "shared_frames": len(self.shared_frames),
            "shared_keys": list(self.shared_frames.keys()),
            "memory_usage_bytes": sum(len(f.data) for f in self.frames.values())
        }