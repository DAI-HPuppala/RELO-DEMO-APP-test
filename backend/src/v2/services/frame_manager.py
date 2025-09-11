"""Frame Manager - Enhanced frame management with debug saving and cleanup"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import cv2
import json

from ..services.frame_registry import FrameRegistry
from config.frame_config import frame_config

logger = logging.getLogger(__name__)


class FrameManager:
    """Manages frame lifecycle, debug saving, and cleanup"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frame_registry = FrameRegistry(session_id)
        self.frame_config = frame_config
        self.frame_count = 0
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info(f"FrameManager initialized for session {session_id}")
    
    async def capture_and_register_frame(self, webrtc_track, agent_name: str) -> Optional[Dict[str, Any]]:
        """Capture frame from WebRTC and register it"""
        try:
            # Receive frame from WebRTC track
            frame = await webrtc_track.recv()
            
            # Convert to numpy array
            frame_array = frame.to_ndarray(format="bgr24")
            
            # Register in frame registry
            self.frame_count += 1
            frame_id = await self.frame_registry.register_frame(
                frame_array, agent_name, self.frame_count
            )
            
            # Save debug frame if enabled
            if self.frame_config.save_debug_frames:
                await self._save_debug_frame(frame_array, frame_id, agent_name)
            
            return {
                "frame_id": frame_id,
                "frame_array": frame_array,
                "frame_number": self.frame_count,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    async def _save_debug_frame(self, frame_array: np.ndarray, frame_id: str, agent_name: str):
        """Save frame to disk for debugging"""
        try:
            # Get save directory
            agent_dir = self.frame_config.get_agent_frame_dir(agent_name)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{frame_id}.jpg"
            filepath = agent_dir / filename
            
            # Save frame as JPEG
            success = cv2.imwrite(
                str(filepath),
                frame_array,
                [cv2.IMWRITE_JPEG_QUALITY, self.frame_config.jpeg_quality]
            )
            
            if success:
                logger.debug(f"Saved debug frame to {filepath}")
                
                # Save metadata
                metadata = {
                    "frame_id": frame_id,
                    "agent_name": agent_name,
                    "session_id": self.session_id,
                    "frame_number": self.frame_count,
                    "timestamp": datetime.now().isoformat(),
                    "shape": frame_array.shape,
                    "file": filename
                }
                
                metadata_path = agent_dir / f"{timestamp}_{frame_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                logger.error(f"Failed to save frame to {filepath}")
                
        except Exception as e:
            logger.error(f"Error saving debug frame: {e}")
    
    async def setup_frame_sharing(self, initial_frame: np.ndarray) -> str:
        """Setup frame sharing from initial classifier to damage detector"""
        frame_id = await self.frame_registry.register_initial_first_frame(initial_frame)
        logger.info(f"Frame sharing setup complete: {frame_id}")
        return frame_id
    
    async def get_shared_frame_for_damage(self) -> Optional[np.ndarray]:
        """Get the shared frame for damage detector"""
        return await self.frame_registry.get_frame_for_damage_detector()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_frames()
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_old_frames(self):
        """Clean up old frames from disk"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.frame_config.cleanup_after_hours)
            
            # Clean up each agent directory
            for agent_name in ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]:
                agent_dir = self.frame_config.get_agent_frame_dir(agent_name)
                
                if not agent_dir.exists():
                    continue
                
                # Remove old files
                for file_path in agent_dir.iterdir():
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
                            logger.debug(f"Deleted old frame: {file_path}")
            
            logger.info(f"Cleanup completed for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def cleanup(self):
        """Clean up all resources"""
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Clean up frame registry
        await self.frame_registry.cleanup()
        
        # Optionally clean up debug frames immediately
        if not self.frame_config.save_debug_frames:
            await self._cleanup_all_frames()
        
        logger.info(f"FrameManager cleaned up for session {self.session_id}")
    
    async def _cleanup_all_frames(self):
        """Clean up all frames for this session immediately"""
        try:
            # Remove all files from agent directories for this session
            for agent_name in ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]:
                agent_dir = self.frame_config.get_agent_frame_dir(agent_name)
                
                if not agent_dir.exists():
                    continue
                
                # Remove files matching this session
                for file_path in agent_dir.glob(f"*{self.session_id}*"):
                    if file_path.is_file():
                        file_path.unlink()
                        logger.debug(f"Deleted session frame: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up session frames: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get frame management statistics"""
        registry_stats = self.frame_registry.get_stats()
        
        # Count debug frames on disk
        debug_frame_count = 0
        if self.frame_config.save_debug_frames:
            for agent_name in ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]:
                agent_dir = self.frame_config.get_agent_frame_dir(agent_name)
                if agent_dir.exists():
                    debug_frame_count += len(list(agent_dir.glob("*.jpg")))
        
        return {
            **registry_stats,
            "total_captured": self.frame_count,
            "debug_frames_saved": debug_frame_count,
            "debug_mode": self.frame_config.save_debug_frames
        }