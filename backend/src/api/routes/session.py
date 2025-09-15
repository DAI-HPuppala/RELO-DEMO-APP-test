"""Session management API routes."""
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import uuid

from models import SessionMode

router = APIRouter()


class SessionStartRequest(BaseModel):
    """Request model for starting a session."""
    mode: SessionMode
    camera_source: Optional[str] = "webcam"
    agent_timers: Optional[Dict[str, float]] = None


class SessionStartResponse(BaseModel):
    """Response model for session start."""
    session_id: str
    status: str
    mode: str
    websocket_url: str


class TriggerRequest(BaseModel):
    """Request model for triggering an agent."""
    agent: Optional[str] = None
    skip_current: bool = False


class ExportRequest(BaseModel):
    """Request model for exporting results."""
    format: str = "json"


def get_session_manager():
    """Get session manager dependency."""
    from api.main import get_session_manager as _get_session_manager
    return _get_session_manager()


@router.post("/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    session_manager = Depends(get_session_manager)
):
    """Initialize a new classification session."""
    try:
        session = await session_manager.create_session(
            mode=request.mode,
            camera_source=request.camera_source,
            agent_timers=request.agent_timers
        )
        
        return SessionStartResponse(
            session_id=session.session_id,
            status=session.status,
            mode=session.mode,
            websocket_url="ws://localhost:8000/ws/stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/status")
async def get_session_status(
    session_id: str,
    session_manager = Depends(get_session_manager)
):
    """Get current session status and progress."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_status_dict()


@router.post("/{session_id}/trigger")
async def trigger_agent(
    session_id: str,
    request: TriggerRequest,
    session_manager = Depends(get_session_manager)
):
    """Manually trigger next agent (manual mode only)."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    try:
        result = await session_manager.trigger_agent(
            session_id,
            agent_name=request.agent,
            skip_current=request.skip_current
        )
        return result
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        elif "automatic mode" in str(e).lower():
            raise HTTPException(status_code=400, detail="Cannot trigger in automatic mode")
        elif "completed" in str(e).lower():
            raise HTTPException(status_code=400, detail="Session is already completed")
        else:
            raise HTTPException(status_code=400, detail=str(e))


@router.get("/{session_id}/results")
async def get_session_results(
    session_id: str,
    session_manager = Depends(get_session_manager)
):
    """Get final classification results."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    session = session_manager.get_session(session_id)
    if not session:
        # Try loading from JSON
        session = await session_manager.load_session_from_json(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "completed":
        raise HTTPException(status_code=425, detail="Results not ready yet")
    
    return session.to_results_dict()


@router.post("/{session_id}/export")
async def export_session(
    session_id: str,
    request: ExportRequest,
    session_manager = Depends(get_session_manager)
):
    """Export session results."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    # Validate format
    if request.format not in ["json"]:
        raise HTTPException(status_code=422, detail="Invalid export format")
    
    try:
        export_info = await session_manager.export_session(session_id, request.format)
        return export_info
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        elif "not complete" in str(e).lower():
            raise HTTPException(status_code=425, detail="Session not complete")
        else:
            raise HTTPException(status_code=400, detail=str(e))


@router.post("/{session_id}/stop")
async def stop_session(
    session_id: str,
    session_manager = Depends(get_session_manager)
):
    """Stop an active session."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    try:
        session = await session_manager.stop_session(session_id)
        return {
            "session_id": session_id,
            "final_status": session.status,
            "results_available": session.final_classification is not None
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{session_id}/resume")
async def resume_session(
    session_id: str,
    session_manager = Depends(get_session_manager)
):
    """Resume an interrupted session (manual mode only)."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    try:
        result = await session_manager.resume_session(session_id)
        return result
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        elif "not resumable" in str(e).lower():
            raise HTTPException(status_code=400, detail="Session not resumable")
        else:
            raise HTTPException(status_code=400, detail=str(e))


@router.get("/{session_id}/export/download")
async def download_export(
    session_id: str,
    session_manager = Depends(get_session_manager)
):
    """Download exported session file."""
    # Validate UUID format
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    
    # Load export file
    from pathlib import Path
    export_path = Path(f"backend/logs/sessions/{session_id}_export.json")
    
    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    import json
    with open(export_path, 'r') as f:
        data = json.load(f)
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=data,
        headers={
            "Content-Disposition": f"attachment; filename={session_id}_export.json",
            "Content-Type": "application/json"
        }
    )