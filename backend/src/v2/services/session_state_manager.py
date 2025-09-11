"""Session State Manager for persistence and recovery"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

from ..models.session_state import SessionState
from ..models.agent_state import AgentState
from config.frame_config import frame_config

logger = logging.getLogger(__name__)


class SessionStateManager:
    """Manages session state persistence and recovery"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self.frame_config = frame_config
        self._checkpoint_interval = 5.0  # Checkpoint every 5 seconds
        self._checkpoint_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("SessionStateManager initialized")
    
    async def create_session(self, session_id: str) -> SessionState:
        """Create a new session"""
        session = SessionState(session_id=session_id)
        self.sessions[session_id] = session
        
        # Start checkpointing
        self._start_checkpointing(session_id)
        
        logger.info(f"Created session {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get an existing session"""
        return self.sessions.get(session_id)
    
    async def save_session(self, session_id: str) -> bool:
        """Save session state to disk"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        try:
            # Get session directory
            session_dir = self.frame_config.get_session_dir(session_id)
            state_path = session_dir / "state.json"
            
            # Prepare state for serialization
            state_dict = {
                "session_id": session.session_id,
                "mode": session.mode.value,
                "status": session.status.value,
                "agents_sequence": session.agents_sequence,
                "current_agent_index": session.current_agent_index,
                "agents_completed": session.agents_completed,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                "paused_at": session.paused_at.isoformat() if session.paused_at else None,
                "paused_agent": session.paused_agent,
                "resume_from_agent": session.resume_from_agent,
                "agent_states": {}
            }
            
            # Serialize agent states
            for agent_name, agent_state in session.agent_states.items():
                if isinstance(agent_state, AgentState):
                    state_dict["agent_states"][agent_name] = agent_state.to_dict()
            
            # Write to file
            async with aiofiles.open(state_path, 'w') as f:
                await f.write(json.dumps(state_dict, indent=2))
            
            logger.debug(f"Saved session {session_id} to {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[SessionState]:
        """Load session state from disk"""
        try:
            session_dir = self.frame_config.get_session_dir(session_id)
            state_path = session_dir / "state.json"
            
            if not state_path.exists():
                logger.warning(f"No saved state found for session {session_id}")
                return None
            
            async with aiofiles.open(state_path, 'r') as f:
                state_dict = json.loads(await f.read())
            
            # Reconstruct session
            session = SessionState(session_id=session_id)
            
            # Restore basic fields
            from ..models.session_state import SessionMode, SessionStatus
            session.mode = SessionMode(state_dict["mode"])
            session.status = SessionStatus(state_dict["status"])
            session.agents_sequence = state_dict["agents_sequence"]
            session.current_agent_index = state_dict["current_agent_index"]
            session.agents_completed = state_dict["agents_completed"]
            
            # Restore timestamps
            if state_dict.get("created_at"):
                session.created_at = datetime.fromisoformat(state_dict["created_at"])
            if state_dict.get("updated_at"):
                session.updated_at = datetime.fromisoformat(state_dict["updated_at"])
            if state_dict.get("completed_at"):
                session.completed_at = datetime.fromisoformat(state_dict["completed_at"])
            if state_dict.get("paused_at"):
                session.paused_at = datetime.fromisoformat(state_dict["paused_at"])
            
            session.paused_agent = state_dict.get("paused_agent")
            session.resume_from_agent = state_dict.get("resume_from_agent")
            
            # Restore agent states
            for agent_name, agent_dict in state_dict.get("agent_states", {}).items():
                session.agent_states[agent_name] = AgentState.from_dict(agent_dict)
            
            self.sessions[session_id] = session
            
            # Resume checkpointing
            self._start_checkpointing(session_id)
            
            logger.info(f"Loaded session {session_id} from disk")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def _start_checkpointing(self, session_id: str) -> None:
        """Start automatic checkpointing for a session"""
        if session_id in self._checkpoint_tasks:
            return
        
        async def checkpoint_loop():
            while session_id in self.sessions:
                await asyncio.sleep(self._checkpoint_interval)
                await self.save_session(session_id)
        
        task = asyncio.create_task(checkpoint_loop())
        self._checkpoint_tasks[session_id] = task
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its data"""
        # Stop checkpointing
        if session_id in self._checkpoint_tasks:
            self._checkpoint_tasks[session_id].cancel()
            del self._checkpoint_tasks[session_id]
        
        # Remove from memory
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Delete files
        try:
            session_dir = self.frame_config.get_session_dir(session_id)
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
            
            logger.info(f"Deleted session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def cleanup_old_sessions(self, hours: int = 24) -> int:
        """Clean up sessions older than specified hours"""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=hours)
        deleted_count = 0
        
        for session_id, session in list(self.sessions.items()):
            if session.created_at < cutoff:
                if await self.delete_session(session_id):
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": session.session_id,
                "status": session.status.value,
                "current_agent": session.current_agent,
                "created_at": session.created_at.isoformat(),
                "agent_progress": f"{len(session.agents_completed)}/{len(session.agents_sequence)}"
            }
            for session in self.sessions.values()
        ]
