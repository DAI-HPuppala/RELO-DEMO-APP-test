"""State Persistence Service for JSON checkpointing"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import aiofiles

from ..models.session_state import SessionState, SessionStatus, SessionMode
from ..models.agent_state import AgentState, AgentStatus
from config.frame_config import frame_config

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles async JSON checkpointing for session and agent states"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.checkpoint_dir = frame_config.get_session_dir(session_id)
        self.checkpoint_path = self.checkpoint_dir / "checkpoint.json"
        self.backup_path = self.checkpoint_dir / "checkpoint.backup.json"
        
        # Async checkpoint queue
        self.checkpoint_queue = asyncio.Queue()
        self.checkpoint_task: Optional[asyncio.Task] = None
        
        # Start checkpoint writer
        self._start_checkpoint_writer()
        
        logger.info(f"StatePersistence initialized for session {session_id}")
    
    def _start_checkpoint_writer(self):
        """Start background task for async checkpointing"""
        async def writer_loop():
            while True:
                try:
                    state_data = await self.checkpoint_queue.get()
                    if state_data is None:  # Shutdown signal
                        break
                    await self._write_checkpoint(state_data)
                except Exception as e:
                    logger.error(f"Checkpoint writer error: {e}")
        
        self.checkpoint_task = asyncio.create_task(writer_loop())
    
    async def save_checkpoint(self, session_state: SessionState) -> None:
        """Queue a checkpoint save (non-blocking)"""
        try:
            # Serialize state to dict
            state_dict = {
                "session_id": session_state.session_id,
                "status": session_state.status.value,
                "mode": session_state.mode.value,
                "current_agent": session_state.current_agent,
                "agents_completed": session_state.agents_completed,
                "paused_agent": session_state.paused_agent,
                "timestamp": datetime.now().isoformat(),
                "agent_states": {}
            }
            
            # Add agent states
            for agent_name, agent_state in session_state.agent_states.items():
                state_dict["agent_states"][agent_name] = {
                    "status": agent_state.status.value,
                    "timer_seconds": agent_state.timer_seconds,
                    "timer_remaining": agent_state.timer_remaining,
                    "inference_count": agent_state.inference_count,
                    "current_inference_num": agent_state.current_inference_num,
                    "finalized_attributes": agent_state.finalized_attributes,
                    "aggregation_method": agent_state.aggregation_method,
                    "error_message": agent_state.error_message,
                    # Don't persist inference results (too large)
                    "has_results": len(agent_state.inference_results) > 0
                }
            
            # Queue for async write
            await self.checkpoint_queue.put(state_dict)
            
        except Exception as e:
            logger.error(f"Failed to queue checkpoint: {e}")
    
    async def _write_checkpoint(self, state_dict: Dict[str, Any]) -> None:
        """Write checkpoint to disk (async)"""
        try:
            # Backup existing checkpoint
            if self.checkpoint_path.exists():
                self.checkpoint_path.rename(self.backup_path)

            # Write new checkpoint asynchronously
            async with aiofiles.open(self.checkpoint_path, 'w') as f:
                await f.write(json.dumps(state_dict, indent=2))

            logger.debug(f"Checkpoint saved for session {self.session_id}")

        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")
            # Restore backup if write failed
            if self.backup_path.exists():
                self.backup_path.rename(self.checkpoint_path)
    
    async def load_checkpoint(self) -> Optional[SessionState]:
        """Load session state from checkpoint (async)"""
        try:
            if not self.checkpoint_path.exists():
                logger.info(f"No checkpoint found for session {self.session_id}")
                return None

            async with aiofiles.open(self.checkpoint_path, 'r') as f:
                content = await f.read()
                state_dict = json.loads(content)
            
            # Reconstruct session state
            session_state = SessionState(session_id=state_dict["session_id"])
            session_state.status = SessionStatus[state_dict["status"]]
            session_state.mode = SessionMode[state_dict["mode"]]
            session_state.current_agent = state_dict.get("current_agent")
            session_state.agents_completed = state_dict.get("agents_completed", [])
            session_state.paused_agent = state_dict.get("paused_agent")
            
            # Reconstruct agent states
            for agent_name, agent_dict in state_dict.get("agent_states", {}).items():
                agent_state = AgentState(
                    agent_name=agent_name,
                    timer_seconds=agent_dict["timer_seconds"]
                )
                agent_state.status = AgentStatus[agent_dict["status"]]
                agent_state.timer_remaining = agent_dict["timer_remaining"]
                agent_state.inference_count = agent_dict["inference_count"]
                agent_state.current_inference_num = agent_dict["current_inference_num"]
                agent_state.finalized_attributes = agent_dict.get("finalized_attributes")
                agent_state.aggregation_method = agent_dict.get("aggregation_method")
                agent_state.error_message = agent_dict.get("error_message")
                
                session_state.agent_states[agent_name] = agent_state
            
            logger.info(f"Checkpoint loaded for session {self.session_id}")
            return session_state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    async def delete_checkpoint(self) -> None:
        """Delete checkpoint files"""
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
            if self.backup_path.exists():
                self.backup_path.unlink()
            logger.info(f"Checkpoints deleted for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to delete checkpoints: {e}")
    
    async def cleanup(self) -> None:
        """Clean up persistence resources"""
        # Signal checkpoint writer to stop
        if self.checkpoint_queue:
            await self.checkpoint_queue.put(None)
        
        # Wait for writer to finish
        if self.checkpoint_task:
            await self.checkpoint_task
        
        logger.info(f"StatePersistence cleaned up for session {self.session_id}")
    
    async def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about existing checkpoint (async)"""
        if not self.checkpoint_path.exists():
            return {"exists": False}

        try:
            stat = self.checkpoint_path.stat()
            async with aiofiles.open(self.checkpoint_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            return {
                "exists": True,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "session_id": data.get("session_id"),
                "status": data.get("status"),
                "current_agent": data.get("current_agent"),
                "checkpoint_time": data.get("timestamp")
            }
        except Exception as e:
            logger.error(f"Failed to get checkpoint info: {e}")
            return {"exists": True, "error": str(e)}