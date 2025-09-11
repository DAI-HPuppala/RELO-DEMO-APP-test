"""WebSocket endpoint for real-time communication."""
import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for WebRTC signaling and real-time updates."""
    await websocket.accept()
    
    session_id = None
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                response = await handle_websocket_message(message, websocket)
                
                # Track session association
                if message.get("type") == "start_session" and response:
                    session_id = response.get("session_id")
                    active_connections[session_id] = websocket
                elif message.get("type") == "associate_session":
                    session_id = message.get("session_id")
                    active_connections[session_id] = websocket
                elif message.get("session_id") and not session_id:
                    # If message has session_id but we don't have it tracked yet
                    session_id = message.get("session_id")
                    active_connections[session_id] = websocket
                
                # Send response
                if response:
                    await websocket.send_text(json.dumps(response))
                    
                # For WebRTC signaling messages, keep the connection alive
                if message.get("type") in ["offer", "answer", "ice_candidate"]:
                    logger.info(f"WebRTC signaling complete for {session_id}, keeping connection alive")
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error_code": "INVALID_JSON",
                    "message": "Invalid JSON format",
                    "recoverable": True
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error_code": "PROCESSING_ERROR",
                    "message": str(e),
                    "recoverable": True
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id and session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if session_id and session_id in active_connections:
            del active_connections[session_id]


async def handle_websocket_message(message: Dict[str, Any], websocket: WebSocket) -> Dict:
    """Handle incoming WebSocket message and return response."""
    import sys
    import os
    # Add src to path if not already there
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from api.main import get_session_manager, get_webrtc_manager
    
    session_manager = get_session_manager()
    webrtc_manager = get_webrtc_manager()
    
    # Check if managers are initialized
    if not session_manager or not webrtc_manager:
        logger.error("Session or WebRTC manager not initialized")
        return {
            "type": "error",
            "error_code": "SYSTEM_NOT_READY",
            "message": "System is still initializing, please try again",
            "recoverable": True
        }
    
    msg_type = message.get("type")
    logger.info(f"Handling WebSocket message type: {msg_type}")
    
    if msg_type == "start_session":
        # Create new session
        mode = message.get("mode", "automatic")
        camera_config = message.get("camera_config", {})
        
        from models import SessionMode
        session = await session_manager.create_session(
            mode=SessionMode(mode),
            camera_source=camera_config.get("source", "webcam"),
            agent_timers=message.get("agent_timers")
        )
        
        return {
            "type": "session_created",
            "session_id": session.session_id,
            "mode": session.mode,
            "status": session.status
        }
    
    elif msg_type == "stop_session":
        # Stop session
        session_id = message.get("session_id")
        session = await session_manager.stop_session(session_id)
        
        return {
            "type": "session_stopped",
            "session_id": session_id,
            "status": session.status
        }
    
    elif msg_type == "offer":
        # WebRTC offer
        session_id = message.get("session_id")
        offer = {
            "sdp": message.get("sdp"),
            "type": "offer"
        }
        
        # Get camera source from session if available
        session = session_manager.get_session(session_id)
        camera_source = session.camera_source if session else "realsense"
        
        answer = await webrtc_manager.handle_offer(session_id, offer, camera_source)
        return answer
    
    elif msg_type == "ice_candidate":
        # ICE candidate
        session_id = message.get("session_id")
        candidate = message.get("candidate")
        
        await webrtc_manager.handle_ice_candidate(session_id, candidate)
        
        return {
            "type": "ice_candidate_received",
            "session_id": session_id
        }
    
    elif msg_type == "switch_mode":
        # Switch operating mode
        session_id = message.get("session_id")
        new_mode = message.get("mode")
        
        session = session_manager.get_session(session_id)
        if session:
            from models import SessionMode
            session.mode = SessionMode(new_mode)
            
            return {
                "type": "mode_switched",
                "session_id": session_id,
                "mode": new_mode
            }
        else:
            raise ValueError(f"Session {session_id} not found")
    
    elif msg_type == "manual_trigger":
        # Manual agent trigger
        session_id = message.get("session_id")
        agent = message.get("agent")
        
        # Import V2 stateful orchestrator
        from src.v2.services.stateful_orchestrator import StatefulOrchestrator
        orchestrator = StatefulOrchestrator(session_id)
        
        # Get frame provider from WebRTC manager and set up orchestrator
        frame_provider = webrtc_manager.get_frame_provider(session_id)
        orchestrator.frame_provider = frame_provider
        
        # Set up V2 message callback for manual trigger
        async def v2_message_callback(v2_msg):
            v1_msg = await translate_v2_to_v1_message(v2_msg)
            if v1_msg:
                await webrtc_manager.send_data_channel_message(session_id, v1_msg)
        
        orchestrator.send_message = v2_message_callback
        
        # Process the specific agent using V2 timer-based approach
        timer_seconds = orchestrator.get_agent_timer(agent)
        agent_state = await orchestrator.run_agent_with_timer(agent, timer_seconds)
        
        # Extract result from agent state
        result = agent_state.final_result if agent_state else {}
        
        return {
            "type": "agent_result",
            "session_id": session_id,
            "agent": agent,
            "result": result
        }
    
    elif msg_type == "start_automatic":
        # Start automatic processing
        session_id = message.get("session_id")
        
        # Import V2 stateful orchestrator  
        from v2.services.stateful_orchestrator import StatefulOrchestrator
        orchestrator = StatefulOrchestrator(session_id)
        
        # Set custom timers if provided
        if message.get("agent_timers"):
            orchestrator.set_custom_timers(message["agent_timers"])
        
        # Get frame provider from WebRTC manager
        frame_provider = webrtc_manager.get_frame_provider(session_id)
        orchestrator.frame_provider = frame_provider
        
        # Set up V2 message callback to send to V1 frontend via data channel
        async def v2_message_callback(v2_msg):
            # Translate V2 messages to V1 format
            v1_msg = await translate_v2_to_v1_message(v2_msg)
            if v1_msg:
                await webrtc_manager.send_data_channel_message(session_id, v1_msg)
        
        orchestrator.send_message = v2_message_callback
        
        # Start automatic processing (runs async) using V2 monitoring flow
        import asyncio
        asyncio.create_task(run_v1_to_v2_monitoring_flow(session_id, orchestrator))
        
        return {
            "type": "automatic_started",
            "session_id": session_id
        }
    
    elif msg_type == "export_results":
        # Export results request
        session_id = message.get("session_id")
        format = message.get("format", "json")
        
        try:
            export_info = await session_manager.export_session(session_id, format)
            
            return {
                "type": "export_ready",
                "session_id": session_id,
                "download_url": export_info["download_url"],
                "format": export_info["format"],
                "size_bytes": export_info["size_bytes"]
            }
        except ValueError as e:
            return {
                "type": "error",
                "error_code": "EXPORT_FAILED",
                "message": str(e),
                "recoverable": False
            }
    
    elif msg_type == "associate_session":
        # Associate WebSocket with existing session
        session_id = message.get("session_id")
        logger.info(f"Associated WebSocket with session {session_id}")
        return None  # No response needed
    
    elif msg_type == "reconnect":
        # Reconnection with existing session
        session_id = message.get("session_id")
        session = session_manager.get_session(session_id)
        
        if session:
            return {
                "type": "reconnected",
                "session_id": session_id,
                "status": session.status,
                "agents_completed": [r["agent_name"] for r in session.agent_results]
            }
        else:
            return {
                "type": "error",
                "error_code": "SESSION_NOT_FOUND",
                "message": f"Session {session_id} not found",
                "recoverable": False
            }
    
    else:
        # Unknown message type
        return {
            "type": "error",
            "error_code": "UNKNOWN_MESSAGE_TYPE",
            "message": f"Unknown message type: {msg_type}",
            "recoverable": True
        }


async def broadcast_to_session(session_id: str, message: Dict):
    """Broadcast message to a specific session's WebSocket."""
    websocket = active_connections.get(session_id)
    
    if websocket:
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error broadcasting to session {session_id}: {e}")
            # Remove failed connection
            if session_id in active_connections:
                del active_connections[session_id]


async def translate_v2_to_v1_message(v2_msg: Dict) -> Dict:
    """Translate V2 stateful messages to V1 format for frontend compatibility"""
    msg_type = v2_msg.get("type")
    
    if msg_type == "agent_started":
        # V2: {"type": "agent_started", "agent": "initial_classifier", "timer_seconds": 4.0, ...}
        # V1: {"type": "agent_started", "agent": "initial_classifier", ...}
        return {
            "type": "agent_started", 
            "agent": v2_msg.get("agent"),
            "session_id": v2_msg.get("session_id"),
            "mode": v2_msg.get("mode", "automatic")
        }
        
    elif msg_type == "inference_update":
        # V2: {"type": "inference_update", "agent": "initial", "inference_num": 2, ...}  
        # V1: {"type": "progress_update", "agent": "initial", "progress": {...}, ...}
        return {
            "type": "progress_update",
            "agent": v2_msg.get("agent"),
            "session_id": v2_msg.get("session_id"),
            "progress": v2_msg.get("inference_num", 1),
            "timer_remaining": v2_msg.get("timer_remaining", 0),
            "frames_used": v2_msg.get("frames_used", 1)
        }
        
    elif msg_type == "agent_completed":
        # V2: {"type": "agent_completed", "agent": "initial", "results": {...}}
        # V1: {"type": "agent_completed", "agent": "initial", "results": {...}}
        return v2_msg  # Direct pass-through
        
    elif msg_type == "classification_complete":
        # V2: {"type": "classification_complete", "final_result": {...}}
        # V1: {"type": "final_results", "classification": {...}}
        return {
            "type": "final_results",
            "session_id": v2_msg.get("session_id"),
            "classification": v2_msg.get("final_result", {}),
            "processing_time": v2_msg.get("processing_time", 0),
            "agents_completed": v2_msg.get("agents_completed", 4)
        }
        
    elif msg_type == "error":
        # Pass through errors
        return v2_msg
        
    # Unknown message types - pass through
    return v2_msg


async def run_v1_to_v2_monitoring_flow(session_id: str, orchestrator):
    """Run V2 monitoring flow adapted for V1 frontend"""
    try:
        logger.info(f"Starting V1-to-V2 monitoring flow for session {session_id}")
        
        # Initialize V2 agents
        from src.v2.agents.initial_classifier_v2 import InitialClassifierV2
        from src.v2.agents.detail_extractor_v2 import DetailExtractorV2  
        from src.v2.agents.damage_detector_v2 import DamageDetectorV2
        from src.v2.agents.final_compiler_v2 import FinalCompilerV2
        
        agents = {
            "initial_classifier": InitialClassifierV2(),
            "detail_extractor": DetailExtractorV2(),
            "damage_detector": DamageDetectorV2(), 
            "final_compiler": FinalCompilerV2()
        }
        
        # Wire frame management for V2 agents
        for agent_name, agent in agents.items():
            if hasattr(agent, 'set_frame_registry'):
                agent.set_frame_registry(orchestrator.frame_registry)
            if hasattr(agent, 'set_frame_provider'):
                agent.set_frame_provider(orchestrator.frame_provider)
            if hasattr(agent, 'set_inference_engine'):
                agent.set_inference_engine(orchestrator.inference_engine)
        
        results = {}
        
        # Run each agent with V2 timer-based processing
        for agent_name in ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]:
            agent = agents[agent_name]
            timer_seconds = orchestrator.get_agent_timer(agent_name)
            
            logger.info(f"Running {agent_name} with {timer_seconds}s timer")
            
            # Send agent_started via V2 message system
            if orchestrator.send_message:
                await orchestrator.send_message({
                    "type": "agent_started",
                    "session_id": session_id,
                    "agent": agent_name,
                    "timer_seconds": timer_seconds,
                    "mode": "automatic"
                })
            
            # Run agent with timer through orchestrator
            agent_state = await orchestrator.run_agent_with_timer(agent_name, timer_seconds)
            results[agent_name] = agent_state
            
            # Send agent_completed
            if orchestrator.send_message:
                await orchestrator.send_message({
                    "type": "agent_completed", 
                    "session_id": session_id,
                    "agent": agent_name,
                    "results": agent_state.final_result if agent_state else {}
                })
        
        # Send final results
        if orchestrator.send_message:
            await orchestrator.send_message({
                "type": "classification_complete",
                "session_id": session_id,
                "final_result": results.get("final_compiler", {}).get("final_result", {}),
                "processing_time": sum(r.timer_seconds for r in results.values() if r),
                "agents_completed": len([r for r in results.values() if r])
            })
            
        logger.info(f"V1-to-V2 monitoring flow completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error in V1-to-V2 monitoring flow: {e}", exc_info=True)
        if orchestrator.send_message:
            await orchestrator.send_message({
                "type": "error",
                "session_id": session_id,
                "message": f"Monitoring flow error: {str(e)}"
            })