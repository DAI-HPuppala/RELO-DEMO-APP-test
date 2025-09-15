"""Session management service."""
import json
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import logging

from models import (
    ClassificationSession, 
    SessionStatus, 
    SessionMode,
    ClothingItem,
    ProcessingStatus
)

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages classification sessions."""
    
    def __init__(self, logs_dir: str = "backend/logs/sessions"):
        """Initialize session manager."""
        self.active_sessions: Dict[str, ClassificationSession] = {}
        self.processing_status: Dict[str, ProcessingStatus] = {}
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_task = None
    
    async def create_session(
        self, 
        mode: SessionMode,
        camera_source: str = "webcam",
        agent_timers: Optional[Dict[str, float]] = None
    ) -> ClassificationSession:
        """Create a new classification session."""
        session = ClassificationSession(
            mode=mode,
            camera_source=camera_source
        )
        
        if agent_timers:
            session.agent_timers.update(agent_timers)
        
        # Initialize processing status
        status = ProcessingStatus(session_id=session.session_id)
        
        # Store in memory
        self.active_sessions[session.session_id] = session
        self.processing_status[session.session_id] = status
        
        logger.info(f"Created session {session.session_id} in {mode} mode")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ClassificationSession]:
        """Get a session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_processing_status(self, session_id: str) -> Optional[ProcessingStatus]:
        """Get processing status for a session."""
        return self.processing_status.get(session_id)
    
    async def update_session_status(
        self, 
        session_id: str, 
        status: SessionStatus,
        current_agent: Optional[str] = None
    ):
        """Update session status."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.status = status
        session.current_agent = current_agent
        session.update_progress()
        
        # Update processing status
        proc_status = self.get_processing_status(session_id)
        if proc_status and current_agent:
            proc_status.current_agent = current_agent
    
    async def stop_session(self, session_id: str) -> ClassificationSession:
        """Stop an active session."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Mark session as stopped
        session.status = SessionStatus.STOPPED
        
        # Save to JSON for potential recovery (optional for manual mode)
        if session.mode == SessionMode.MANUAL:
            session.interrupt_session()
            await self.save_session_to_json(session)
        
        # Always remove from active sessions - each start creates a new session
        del self.active_sessions[session_id]
        if session_id in self.processing_status:
            del self.processing_status[session_id]
        
        logger.info(f"Stopped session {session_id} ({session.mode}) - removed from memory")
        return session
    
    async def reactivate_session(self, session_id: str, mode: Optional[SessionMode] = None) -> ClassificationSession:
        """Reactivate a stopped session (typically for manual mode)."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status != SessionStatus.STOPPED:
            logger.warning(f"Session {session_id} is not stopped, current status: {session.status}")
        
        # Update session
        session.status = SessionStatus.ACTIVE
        if mode:
            session.mode = mode
        session.stopped_at = None  # Clear stopped timestamp
        
        # Ensure processing status exists
        if session_id not in self.processing_status:
            self.processing_status[session_id] = ProcessingStatus(session_id=session_id)
        
        logger.info(f"Reactivated session {session_id} in {session.mode} mode")
        return session
    
    async def resume_session(self, session_id: str) -> Dict:
        """Resume an interrupted manual mode session."""
        session = self.get_session(session_id)
        
        # If not in memory, try to load from JSON
        if not session:
            session = await self.load_session_from_json(session_id)
            if session:
                self.active_sessions[session_id] = session
                self.processing_status[session_id] = ProcessingStatus(session_id=session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if not session.is_resumable:
            raise ValueError(f"Session {session_id} is not resumable")
        
        checkpoint = session.resume_from_checkpoint()
        logger.info(f"Resumed session {session_id} from checkpoint")
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "last_checkpoint": checkpoint
        }
    
    async def trigger_agent(
        self, 
        session_id: str,
        agent_name: Optional[str] = None,
        skip_current: bool = False
    ) -> Dict:
        """Manually trigger an agent in manual mode."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.mode != SessionMode.MANUAL:
            raise ValueError("Cannot trigger agent in automatic mode")
        
        if session.status == SessionStatus.COMPLETED:
            raise ValueError("Session is already completed")
        
        proc_status = self.get_processing_status(session_id)
        if not proc_status:
            raise ValueError(f"Processing status not found for session {session_id}")
        
        # Determine which agent to trigger
        if skip_current and proc_status.current_agent:
            proc_status.complete_agent(proc_status.current_agent)
        
        if agent_name:
            triggered_agent = agent_name
        elif proc_status.agents_pending:
            triggered_agent = proc_status.agents_pending[0]
        else:
            raise ValueError("No agents left to trigger")
        
        # Start the agent
        proc_status.start_agent(triggered_agent)
        session.current_agent = triggered_agent
        session.status = SessionStatus.PROCESSING
        
        logger.info(f"Triggered agent {triggered_agent} for session {session_id}")
        
        return {
            "triggered_agent": triggered_agent,
            "status": "processing"
        }
    
    async def complete_agent(
        self,
        session_id: str,
        agent_name: str,
        results: Dict
    ):
        """Mark an agent as completed and store results."""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Store agent results
        agent_result = {
            "agent_name": agent_name,
            "attributes": results.get("attributes", {}),
            "confidence": results.get("confidence", 0.0),
            "reasoning": results.get("reasoning", ""),
            "frames_processed": results.get("frames_processed", 0)
        }
        session.agent_results.append(agent_result)
        
        # Update processing status
        proc_status = self.get_processing_status(session_id)
        if proc_status:
            proc_status.complete_agent(agent_name)
        
        # Check if all agents completed
        if not proc_status.agents_pending:
            await self.finalize_session(session_id)
        
        logger.info(f"Completed agent {agent_name} for session {session_id}")
    
    async def finalize_session(self, session_id: str):
        """Finalize a session with final classification."""
        session = self.get_session(session_id)
        if not session:
            return
        
        # Compile final classification from agent results
        final_classification = self.compile_final_classification(session.agent_results)
        session.final_classification = final_classification
        session.complete_session()
        
        # Save to JSON
        await self.save_session_to_json(session)
        
        logger.info(f"Finalized session {session_id}")
    
    def compile_final_classification(self, agent_results: List[Dict]) -> Dict:
        """Compile final classification from all agent results."""
        final = {
            "type": None,
            "color_primary": None,
            "color_secondary": None,
            "pattern": None,
            "brand": None,
            "size": None,
            "has_damage": False,
            "damage_type": None,
            "damage_locations": []
        }
        
        # Merge results from all agents (simplified logic)
        for result in agent_results:
            attrs = result.get("attributes", {})
            
            # Update with non-null values
            for key in final:
                if key in attrs and attrs[key] is not None:
                    if key == "has_damage" and attrs[key]:
                        final[key] = True
                    elif key == "damage_locations" and attrs[key]:
                        final[key].extend(attrs[key])
                    else:
                        final[key] = attrs[key]
        
        return final
    
    async def save_session_to_json(self, session: ClassificationSession):
        """Save session to JSON file."""
        file_path = self.logs_dir / f"{session.session_id}.json"
        
        session_data = {
            "session_id": session.session_id,
            "mode": session.mode,
            "started_at": session.started_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "status": session.status,
            "final_classification": session.final_classification,
            "agent_results": session.agent_results,
            "total_frames_analyzed": session.total_frames_analyzed,
            "is_resumable": session.is_resumable,
            "last_checkpoint": session.last_checkpoint
        }
        
        with open(file_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Saved session {session.session_id} to {file_path}")
    
    async def load_session_from_json(self, session_id: str) -> Optional[ClassificationSession]:
        """Load session from JSON file."""
        file_path = self.logs_dir / f"{session_id}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct session
        session = ClassificationSession(
            session_id=data["session_id"],
            mode=data["mode"],
            status=data["status"]
        )
        
        session.started_at = datetime.fromisoformat(data["started_at"])
        if data["completed_at"]:
            session.completed_at = datetime.fromisoformat(data["completed_at"])
        
        session.final_classification = data.get("final_classification")
        session.agent_results = data.get("agent_results", [])
        session.total_frames_analyzed = data.get("total_frames_analyzed", 0)
        session.is_resumable = data.get("is_resumable", False)
        session.last_checkpoint = data.get("last_checkpoint")
        
        logger.info(f"Loaded session {session_id} from {file_path}")
        
        return session
    
    async def export_session(self, session_id: str, format: str = "json") -> Dict:
        """Export session results."""
        session = self.get_session(session_id)
        if not session:
            # Try loading from JSON
            session = await self.load_session_from_json(session_id)
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status != SessionStatus.COMPLETED:
            raise ValueError(f"Session {session_id} is not complete")
        
        export_data = session.to_results_dict()
        
        # Save export file
        export_path = self.logs_dir / f"{session_id}_export.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return {
            "download_url": f"/api/session/{session_id}/export/download",
            "format": format,
            "size_bytes": export_path.stat().st_size,
            "expires_at": datetime.now().isoformat()
        }
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions from memory."""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            age_hours = (current_time - session.started_at).total_seconds() / 3600
            
            if age_hours > max_age_hours and session.status in [
                SessionStatus.COMPLETED, 
                SessionStatus.ERROR,
                SessionStatus.STOPPED
            ]:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            if session_id in self.processing_status:
                del self.processing_status[session_id]
            logger.info(f"Cleaned up old session {session_id}")
        
        return len(sessions_to_remove)