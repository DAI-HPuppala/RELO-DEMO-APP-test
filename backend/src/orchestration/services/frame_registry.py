"""Frame Registry service for managing frame lifecycle and sharing"""

import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
import numpy as np
from uuid import uuid4
import time
from collections import OrderedDict

from ..models.frame import Frame

logger = logging.getLogger(__name__)


class FrameRegistry:
    """Manages frame lifecycle and sharing between agents"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frames: OrderedDict[str, Frame] = OrderedDict()  # frame_id -> Frame (LRU order)

        # Frame capacity limit (50 frames = ~45MB)
        self.MAX_FRAMES = 50
        self.eviction_count = 0  # Track evictions for monitoring

        # Sharing management
        self.shared_frames: Dict[str, str] = {}  # share_key -> frame_id
        self.reference_counts: Dict[str, int] = {}  # frame_id -> ref count

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info(f"FrameRegistry initialized for session {session_id} with {self.MAX_FRAMES} frame limit")
    
    async def register_frame(self, frame_data: np.ndarray, agent_name: str,
                            frame_number: int = 0) -> str:
        """Register a new frame and return its ID (with LRU eviction if at capacity)"""
        async with self._lock:
            # Check if at capacity and need to evict
            if len(self.frames) >= self.MAX_FRAMES:
                evicted = await self._evict_oldest_unused_frame()
                if not evicted:
                    logger.warning(f"At frame capacity ({self.MAX_FRAMES}) but no frames available for eviction!")

            frame_id = f"{agent_name}_{int(time.time()*1000)}_{uuid4().hex[:8]}"

            # Store frame metadata for proper reconstruction
            dtype_str = str(frame_data.dtype)
            shape = frame_data.shape

            # Convert numpy array to bytes with metadata
            # Format: dtype_str|shape|data_bytes
            frame_bytes = f"{dtype_str}|{shape}|".encode() + frame_data.tobytes()

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
            self.frames.move_to_end(frame_id)  # Mark as most recently used
            self.reference_counts[frame_id] = 0  # Start with 0 references (can be evicted)

            logger.debug(f"Registered frame {frame_id} for agent {agent_name} (total: {len(self.frames)})")
            return frame_id

    async def _evict_oldest_unused_frame(self) -> bool:
        """Evict the oldest frame that is not shared or referenced (LRU)"""
        # Iterate from oldest to newest
        for frame_id in list(self.frames.keys()):
            frame = self.frames[frame_id]
            ref_count = self.reference_counts.get(frame_id, 0)

            # Check if frame can be evicted (not shared, not referenced)
            if not frame.is_shared and ref_count <= 0:
                # Evict this frame
                del self.frames[frame_id]
                if frame_id in self.reference_counts:
                    del self.reference_counts[frame_id]

                # Remove from shared frames if present
                keys_to_remove = [k for k, v in self.shared_frames.items() if v == frame_id]
                for key in keys_to_remove:
                    del self.shared_frames[key]

                self.eviction_count += 1
                logger.info(f"Evicted frame {frame_id} (LRU) - Total evictions: {self.eviction_count}")
                return True

        # No frame could be evicted (all are shared or referenced)
        return False
    
    async def get_frame(self, frame_id: str) -> Optional[Frame]:
        """Get a frame by ID (updates LRU order)"""
        async with self._lock:
            frame = self.frames.get(frame_id)
            if frame:
                # Mark as recently used by moving to end
                self.frames.move_to_end(frame_id)
            return frame

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
            "max_frames": self.MAX_FRAMES,
            "capacity_used": f"{len(self.frames)}/{self.MAX_FRAMES}",
            "shared_frames": len(self.shared_frames),
            "shared_keys": list(self.shared_frames.keys()),
            "memory_usage_bytes": sum(len(f.data) for f in self.frames.values()),
            "memory_usage_mb": sum(len(f.data) for f in self.frames.values()) / (1024 * 1024),
            "eviction_count": self.eviction_count
        }