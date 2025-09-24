"""WebSocket endpoint for real-time communication."""
import json
import logging
import time
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

from services.vlm_service_ultra import UltraVLMService as VLMService
from v2.utils.key_normalizer import KeyNormalizer
from services.manual_mode_handler import ManualModeHandler
from models.manual_session import ManualSessionState, NavigationDirection

logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store monitoring control for each session
monitoring_controls: Dict[str, Dict[str, Any]] = {}

# Store manual mode handlers for each session
manual_mode_handlers: Dict[str, ManualModeHandler] = {}

# Store manual mode session states
manual_mode_sessions: Dict[str, ManualSessionState] = {}

# Store orchestrators for each session
orchestrators: Dict[str, Any] = {}

# Store reference to webrtc_manager for control message handling
_webrtc_manager = None
_control_handler_setup = False

# Global VLM service instance
_vlm_service = None

async def initialize_vlm_service():
    """Initialize VLM service if not already done"""
    global _vlm_service
    if _vlm_service is None:
        _vlm_service = VLMService()
        # Check if VLM is available
        vlm_healthy = await _vlm_service.health_check()
        if vlm_healthy:
            logger.info("✅ VLM service initialized and healthy")
        else:
            logger.error("❌ VLM service health check failed - check Ollama setup")
    return _vlm_service

async def setup_orchestrator_vlm(orchestrator):
    """Setup VLM integration for orchestrator"""
    # The MultiInferenceEngine already has its own VLM engine
    # We just need to ensure it's initialized
    try:
        if hasattr(orchestrator, 'inference_engine'):
            # The inference engine already has VLM integrated
            logger.info("✅ VLM already integrated in MultiInferenceEngine")
            return True
        else:
            logger.error("❌ Orchestrator missing inference_engine")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to verify VLM integration: {e}")
        return False


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
                    # Clean up any stale manual mode handlers for this session
                    if session_id in manual_mode_handlers:
                        logger.info(f"Cleaning stale manual mode handler for new session {session_id}")
                        del manual_mode_handlers[session_id]
                    if session_id in manual_mode_sessions:
                        del manual_mode_sessions[session_id]
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
    
    # Import at module level to avoid import issues
    from api import main
    
    session_manager = main.session_manager
    webrtc_manager = main.webrtc_manager
    
    # Debug logging
    logger.info(f"Session manager: {session_manager}")
    logger.info(f"WebRTC manager: {webrtc_manager}")
    
    # Check if managers are initialized
    if not session_manager or not webrtc_manager:
        logger.error(f"Session or WebRTC manager not initialized - session_manager: {session_manager}, webrtc_manager: {webrtc_manager}")
        return {
            "type": "error",
            "error_code": "SYSTEM_NOT_READY",
            "message": "System is still initializing, please try again",
            "recoverable": True
        }
    
    # Set up control message handler on first call
    global _webrtc_manager, _control_handler_setup
    if not _control_handler_setup:
        _webrtc_manager = webrtc_manager
        
        async def handle_control_message_from_data_channel(session_id: str, message: Dict):
            """Handle control messages forwarded from data channel."""
            logger.info(f"Handling control message from data channel: {message.get('type')} for session {session_id}")
            
            # Get the websocket for this session
            ws = active_connections.get(session_id)
            if not ws:
                logger.warning(f"No websocket found for session {session_id}, using a placeholder")
                # Use a placeholder since we're handling via data channel
                ws = None
            
            # Ensure the message has the session_id
            if 'session_id' not in message:
                message['session_id'] = session_id
            
            # Process the message through the normal websocket handler
            return await handle_websocket_message(message, ws)
        
        webrtc_manager.set_control_message_handler(handle_control_message_from_data_channel)
        _control_handler_setup = True
        logger.info("Control message handler set up for WebRTC manager")
    
    msg_type = message.get("type")
    logger.info(f"Handling WebSocket message type: {msg_type}")
    
    if msg_type == "start_session":
        # Always create a new session
        mode = message.get("mode", "automatic")
        camera_config = message.get("camera_config", {})
        
        from models import SessionMode
        
        # Create new session
        session = await session_manager.create_session(
            mode=SessionMode(mode),
            camera_source=camera_config.get("source", "webcam"),
            agent_timers=message.get("agent_timers")
        )
        
        logger.info(f"Created new session {session.session_id} in {mode} mode")
        
        return {
            "type": "session_created",
            "session_id": session.session_id,
            "mode": session.mode,
            "status": session.status
        }
    
    elif msg_type == "stop_session":
        # Stop session and monitoring immediately
        session_id = message.get("session_id")
        logger.info(f"Stop session requested for {session_id}")
        
        # Immediately stop all monitoring for this session and any cycle sessions
        keys_to_stop = []
        for key in list(monitoring_controls.keys()):
            # Stop base session and any cycle sessions
            if key == session_id or key.startswith(f"{session_id}_"):
                keys_to_stop.append(key)
        
        for key in keys_to_stop:
            logger.info(f"Stopping monitoring for {key}")
            monitoring_controls[key]['running'] = False
            monitoring_controls[key]['paused'] = False
        
        # Clean up monitoring controls
        for key in keys_to_stop:
            if key in monitoring_controls:
                del monitoring_controls[key]

        # Clean up orchestrators
        if session_id in orchestrators:
            del orchestrators[session_id]

        # Clean up manual mode handlers and sessions
        if session_id in manual_mode_handlers:
            logger.info(f"Cleaning up manual mode handler for {session_id}")
            manual_mode_handlers[session_id].deactivate_manual_mode()
            del manual_mode_handlers[session_id]
        
        if session_id in manual_mode_sessions:
            logger.info(f"Cleaning up manual mode session state for {session_id}")
            del manual_mode_sessions[session_id]
        
        # Try to stop the session, but don't fail if it doesn't exist
        try:
            session = await session_manager.stop_session(session_id)
            status = session.status
        except ValueError as e:
            logger.warning(f"Session stop error (may already be stopped): {e}")
            status = "stopped"
        
        logger.info(f"Session {session_id} stop completed")
        
        return {
            "type": "session_stopped",
            "session_id": session_id,
            "status": status
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
        
        # Setup VLM integration
        await setup_orchestrator_vlm(orchestrator)
        
        # Get frame provider from WebRTC manager and wrap it for V2 async compatibility
        v1_frame_provider = webrtc_manager.get_frame_provider(session_id)
        
        async def async_frame_provider():
            """Async wrapper for V1 frame provider"""
            try:
                # Check if v1_frame_provider is async (coroutine function)
                if v1_frame_provider:
                    if asyncio.iscoroutinefunction(v1_frame_provider):
                        frame = await v1_frame_provider()
                    else:
                        # Fallback for sync functions
                        frame = v1_frame_provider()
                else:
                    frame = None
                return frame
            except Exception as e:
                logger.error(f"Frame provider error: {e}")
                return None
        
        orchestrator.frame_provider = async_frame_provider
        
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
        is_manual_mode = message.get("manual_mode", False)
        logger.info(f"Received start_automatic: session={session_id}, manual_mode={is_manual_mode}")
        
        # Check if monitoring is already running for this session
        if session_id in monitoring_controls and monitoring_controls[session_id].get('running', False):
            logger.warning(f"Monitoring already running for session {session_id}")
            return {
                "type": "error",
                "message": "Monitoring already running for this session"
            }
        
        # Ensure VLM service is initialized before starting
        vlm_service = await initialize_vlm_service()
        if not vlm_service:
            logger.error("VLM service not available, cannot start automatic mode")
            return {
                "type": "error",
                "error_code": "VLM_NOT_READY",
                "message": "Vision model service is not ready. Please wait a moment and try again.",
                "recoverable": True
            }
        
        # Verify data channel is ready
        if not webrtc_manager.is_data_channel_ready(session_id):
            logger.warning(f"Data channel not ready for session {session_id}, waiting...")
            # Try to wait for data channel
            ready = await webrtc_manager.wait_for_data_channel(session_id, timeout=3000)
            if not ready:
                logger.error(f"Data channel failed to open for session {session_id}")
                return {
                    "type": "error",
                    "error_code": "DATA_CHANNEL_NOT_READY",
                    "message": "Communication channel not established. Please refresh and try again.",
                    "recoverable": True
                }
        
        # Create monitoring control for this session
        monitoring_controls[session_id] = {'running': True, 'paused': False}
        
        # Import V2 stateful orchestrator
        from v2.services.stateful_orchestrator import StatefulOrchestrator
        orchestrator = StatefulOrchestrator(session_id, manual_mode=is_manual_mode)
        
        # Setup VLM integration
        vlm_setup_success = await setup_orchestrator_vlm(orchestrator)
        if not vlm_setup_success:
            logger.error("Failed to setup VLM for orchestrator")
            monitoring_controls[session_id]['running'] = False
            return {
                "type": "error", 
                "error_code": "VLM_SETUP_FAILED",
                "message": "Failed to initialize vision model. Please try again.",
                "recoverable": True
            }
        
        # Set custom timers if provided
        if message.get("agent_timers"):
            orchestrator.set_custom_timers(message["agent_timers"])
        
        # Get frame provider from WebRTC manager and wrap it for V2 async compatibility
        v1_frame_provider = webrtc_manager.get_frame_provider(session_id)
        
        async def async_frame_provider():
            """Async wrapper for V1 frame provider"""
            try:
                # Check if v1_frame_provider is async (coroutine function)
                if v1_frame_provider:
                    if asyncio.iscoroutinefunction(v1_frame_provider):
                        frame = await v1_frame_provider()
                    else:
                        # Fallback for sync functions
                        frame = v1_frame_provider()
                else:
                    frame = None
                return frame
            except Exception as e:
                logger.error(f"Frame provider error: {e}")
                return None
        
        orchestrator.frame_provider = async_frame_provider
        
        # Set up V2 message callback to send to V1 frontend via data channel
        async def v2_message_callback(v2_msg):
            # Translate V2 messages to V1 format
            v1_msg = await translate_v2_to_v1_message(v2_msg)
            if v1_msg:
                msg_type = v1_msg.get('type')
                agent = v1_msg.get('agent', 'N/A')
                
                # Log what we're sending to frontend with clear naming
                if msg_type == "progress_update" and v1_msg.get('partial_results'):
                    logger.info(f"[TO FRONTEND - PROGRESSIVE] Sending progressive results for {agent}")
                    logger.info(f"[TO FRONTEND - PROGRESSIVE] Attributes: {v1_msg.get('partial_results')}")
                elif msg_type == "agent_completed":
                    logger.info(f"[TO FRONTEND - AGENT FINAL] Sending finalized results for {agent}")
                    logger.info(f"[TO FRONTEND - AGENT FINAL] Attributes: {v1_msg.get('results')}")
                elif msg_type == "final_results":
                    logger.info(f"[TO FRONTEND - FINAL CLASSIFICATION] Sending complete classification")
                    logger.info(f"[TO FRONTEND - FINAL CLASSIFICATION] Results: {v1_msg.get('results')}")
                else:
                    logger.info(f"[TO FRONTEND] Sending {msg_type} for {agent}")
                
                result = await webrtc_manager.send_data_channel_message(session_id, v1_msg)
                if not result:
                    logger.error(f"Failed to send message via data channel for session {session_id}")
        
        orchestrator.send_message = v2_message_callback

        # Store orchestrator for later use
        orchestrators[session_id] = orchestrator

        # Check if manual mode requested
        if is_manual_mode:
            logger.info(f"Entering manual mode for session {session_id}")
            # Create manual mode handler
            if session_id not in manual_mode_handlers:
                manual_mode_handlers[session_id] = ManualModeHandler(session_id)
                manual_mode_sessions[session_id] = ManualSessionState(session_id=session_id)
            
            handler = manual_mode_handlers[session_id]
            handler.activate_manual_mode()

            # Pass orchestrator reference to handler
            handler.set_orchestrator(orchestrator)

            # Set up callbacks
            async def send_manual_status(status):
                await webrtc_manager.send_manual_mode_status(session_id, status)

            handler.set_callbacks(send_status=send_manual_status)
            
            # Mark as manual mode in monitoring controls
            monitoring_controls[session_id]['manual_mode'] = True
            
            # Start manual mode flow
            logger.info(f"Creating manual mode flow task for session {session_id}")
            asyncio.create_task(run_manual_mode_flow(session_id, orchestrator, monitoring_controls[session_id], handler))
            
            logger.info(f"Manual mode started successfully for session {session_id}")
            return {
                "type": "manual_mode_started",
                "session_id": session_id,
                "mode": "manual"
            }
        else:
            # Start automatic processing (runs async) using V2 monitoring flow
            asyncio.create_task(run_v1_to_v2_monitoring_flow(session_id, orchestrator, monitoring_controls[session_id]))
            
            return {
                "type": "automatic_started",
                "session_id": session_id
            }
    
    elif msg_type == "stop_monitoring" or msg_type == "stop_automatic":
        # Stop automatic monitoring
        session_id = message.get("session_id")
        
        if session_id in monitoring_controls:
            logger.info(f"Stopping monitoring for session {session_id}")
            monitoring_controls[session_id]['running'] = False
            # Clean up the control entry
            del monitoring_controls[session_id]
            
            return {
                "type": "monitoring_stopped",
                "session_id": session_id
            }
        else:
            logger.warning(f"No active monitoring found for session {session_id}")
            return {
                "type": "error",
                "message": "No active monitoring for this session"
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
    
    elif msg_type == "pause_flow":
        # Pause the monitoring flow
        session_id = message.get("session_id")
        
        if session_id in monitoring_controls:
            monitoring_controls[session_id]['paused'] = True
            monitoring_controls[session_id]['should_interrupt'] = True  # Signal to interrupt current agent
            current_agent = monitoring_controls[session_id].get('current_agent', 'unknown')
            logger.info(f"Paused monitoring flow for session {session_id} at agent {current_agent}")
            
            # Store which agent to restart when resumed
            monitoring_controls[session_id]['paused_at_agent'] = current_agent
            
            return {
                "type": "flow_paused",
                "session_id": session_id,
                "paused_agent": current_agent
            }
        else:
            return {
                "type": "error",
                "message": "No active monitoring to pause"
            }
    
    elif msg_type == "resume_flow":
        # Resume the monitoring flow
        session_id = message.get("session_id")
        restart_agent = message.get("restart_agent", False)
        
        if session_id in monitoring_controls:
            monitoring_controls[session_id]['paused'] = False
            monitoring_controls[session_id]['should_interrupt'] = False
            monitoring_controls[session_id]['restart_agent'] = True  # Signal to restart the paused agent
            current_agent = monitoring_controls[session_id].get('paused_at_agent', monitoring_controls[session_id].get('current_agent', 'unknown'))
            logger.info(f"Resumed monitoring flow for session {session_id}, will restart agent {current_agent}")
            
            return {
                "type": "flow_resumed",
                "session_id": session_id,
                "resuming_agent": current_agent,
                "restart_agent": True
            }
        else:
            return {
                "type": "error",
                "message": "No active monitoring to resume"
            }
    
    elif msg_type == "start_manual_mode":
        # Start manual mode for session
        session_id = message.get("session_id")
        logger.info(f"Starting manual mode for session {session_id}")
        
        # Create manual mode handler if not exists
        if session_id not in manual_mode_handlers:
            manual_mode_handlers[session_id] = ManualModeHandler(session_id)
            manual_mode_sessions[session_id] = ManualSessionState(session_id=session_id)
        
        handler = manual_mode_handlers[session_id]
        handler.activate_manual_mode()
        
        # Set up callbacks
        async def send_manual_status(status):
            await webrtc_manager.send_manual_mode_status(session_id, status)
        
        handler.set_callbacks(send_status=send_manual_status)

        # Pass orchestrator reference if available
        if session_id in orchestrators:
            handler.set_orchestrator(orchestrators[session_id])

        # Stop any automatic monitoring
        if session_id in monitoring_controls:
            monitoring_controls[session_id]['running'] = False
            monitoring_controls[session_id]['manual_mode'] = True
        
        return {
            "type": "manual_mode_started",
            "session_id": session_id,
            "current_agent": handler.get_current_agent()
        }
    
    elif msg_type in ["manual_previous", "manual_next", "manual_redo", 
                     "manual_select_agent"]:
        # Handle manual mode commands
        session_id = message.get("session_id")
        logger.info(f"Manual command {msg_type} received with session_id: {session_id}")
        
        # Check if session exists at all
        from models import SessionMode
        if session_id and session_manager:
            session = session_manager.get_session(session_id)
            if not session:
                logger.warning(f"Manual command {msg_type} received for non-existent session {session_id}")
                return {
                    "type": "error",
                    "error_code": "SESSION_NOT_FOUND",
                    "message": "Session has been terminated. Please start a new session.",
                    "recoverable": False
                }
        
        # Check if manual mode handler exists
        if session_id not in manual_mode_handlers:
            logger.error(f"No manual mode handler for session {session_id} - manual mode not active")
            return {
                "type": "error",
                "error_code": "MANUAL_MODE_NOT_ACTIVE",
                "message": "Manual mode is not active. Please start monitoring in manual mode.",
                "recoverable": False
            }
        
        handler = manual_mode_handlers[session_id]
        
        # Handle the command
        result = await handler.handle_command(message)
        
        # Check if we need to run an agent or final compiler
        if result.get("success"):
            action = result.get("action")
            
            if action in ["navigate_next", "navigate_previous", "select_agent", "redo_agent"]:
                # We're ready to run the agent
                if action == "redo_agent":
                    # For redo, use the agent specified in the result
                    agent_name = result.get("agent", handler.get_current_agent())
                else:
                    # For navigation, use the current agent
                    agent_name = handler.get_current_agent()
                
                # Mark in monitoring controls for manual mode flow
                if session_id not in monitoring_controls:
                    monitoring_controls[session_id] = {'running': True, 'manual_mode': True}
                
                monitoring_controls[session_id]['manual_agent'] = agent_name
                monitoring_controls[session_id]['waiting_for_manual_agent'] = True
                
                # The monitoring flow will pick this up and run the agent
                logger.info(f"Manual mode: Ready to run agent {agent_name}")
                
            elif action == "run_final_compiler":
                # Run final compiler
                monitoring_controls[session_id]['run_final_compiler'] = True
                logger.info("Manual mode: Ready to run final compiler")
        
        return result
    
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
    
    # Map backend agent names to frontend short names
    agent_name_map = {
        "initial_classifier": "initial",
        "detail_extractor": "detail", 
        "damage_detector": "damage",
        "final_compiler": "final"
    }
    
    # Get short agent name for frontend
    agent = v2_msg.get("agent", "")
    short_agent = agent_name_map.get(agent, agent)
    
    if msg_type == "agent_started":
        # V2: {"type": "agent_started", "agent": "initial_classifier", "timer_seconds": 4.0, ...}
        # V1: {"type": "agent_started", "agent": "initial", ...}
        return {
            "type": "agent_started", 
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "mode": v2_msg.get("mode", "automatic")
        }
        
    elif msg_type == "inference_update":
        # V2: {"type": "inference_update", "agent": "initial_classifier", "inference_num": 2, ...}  
        # V1: {"type": "progress_update", "agent": "initial", "progress": {...}, ...}
        return {
            "type": "progress_update",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "progress": v2_msg.get("inference_num", 1),
            "timer_remaining": v2_msg.get("timer_remaining", 0),
            "frames_used": v2_msg.get("frames_used", 1)
        }
    
    elif msg_type == "inference_result":
        # V2: {"type": "inference_result", "agent": "initial_classifier", "attributes": {...}, ...}
        # V1: {"type": "progress_update", "agent": "initial", "partial_results": {...}, ...}
        
        # Normalize attributes before sending to frontend
        raw_attributes = v2_msg.get("attributes", {})
        normalized_attributes = KeyNormalizer.normalize_agent_attributes(agent, raw_attributes)
        
        return {
            "type": "progress_update",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "inference_num": v2_msg.get("inference_num", 1),
            "partial_results": normalized_attributes,
            "confidence": v2_msg.get("confidence", 0),
            "reasoning": v2_msg.get("reasoning", "")
        }
        
    elif msg_type == "agent_completed":
        # V2: {"type": "agent_completed", "agent": "initial_classifier", "results": {...}}
        # V1: {"type": "agent_completed", "agent": "initial", "results": {...}}
        
        # Normalize results before sending to frontend
        raw_results = v2_msg.get("results", {})
        normalized_results = KeyNormalizer.normalize_agent_attributes(agent, raw_results)
        
        return {
            "type": "agent_completed",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "results": normalized_results
        }
        
    elif msg_type == "classification_complete":
        # V2: {"type": "classification_complete", "final_result": {...}}
        # V1: {"type": "final_results", "results": {...}}
        final_result = v2_msg.get("final_result", {})
        # Ensure we have the correct structure for frontend
        return {
            "type": "final_results",
            "session_id": v2_msg.get("session_id"),
            "results": final_result,  # Frontend expects "results", not "classification"
            "processing_time": v2_msg.get("processing_time", 0),
            "agents_completed": v2_msg.get("agents_completed", 4),
            "cycle_number": v2_msg.get("cycle_number", 1)
        }
        
    elif msg_type == "error":
        # Pass through errors
        return v2_msg
        
    # Unknown message types - pass through
    return v2_msg


async def run_v1_to_v2_monitoring_flow(session_id: str, orchestrator, monitoring_control):
    """Run V2 monitoring flow adapted for V1 frontend - runs continuously until stopped"""
    try:
        logger.info(f"Starting continuous V1-to-V2 monitoring flow for session {session_id}")
        cycle_count = 0
        
        # Ensure VLM is initialized before starting
        if not orchestrator.is_vlm_configured():
            logger.info("Initializing GPU optimization for orchestrator...")
            try:
                init_success = await orchestrator.initialize_gpu_optimization()
                if not init_success:
                    logger.error("Failed to initialize VLM for monitoring flow")
                    # Send error to frontend
                    if orchestrator.send_message:
                        await orchestrator.send_message({
                            "type": "error",
                            "error_code": "VLM_INIT_FAILED",
                            "message": "Vision model initialization failed. Please check Ollama is running.",
                            "recoverable": True
                        })
                    monitoring_control['running'] = False
                    return
            except Exception as e:
                logger.error(f"Exception during VLM initialization: {e}")
                # Send error to frontend
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "error",
                        "error_code": "VLM_INIT_ERROR",
                        "message": f"Vision model error: {str(e)}",
                        "recoverable": True
                    })
                monitoring_control['running'] = False
                return
        
        # Double-check VLM is ready
        logger.info(f"VLM configured status: {orchestrator.is_vlm_configured()}")
        if not orchestrator.is_vlm_configured():
            logger.error("VLM still not configured after initialization attempt")
            monitoring_control['running'] = False
            return
        
        # Run continuously until stop signal
        while monitoring_control.get('running', True):
            cycle_count += 1
            
            # Create a new session ID for each cycle (keeps base session ID with cycle suffix)
            cycle_session_id = f"{session_id}_cycle_{cycle_count}"
            logger.info(f"Starting monitoring cycle {cycle_count} with session {cycle_session_id}")
            
            # Track cycle timing
            cycle_start_time = time.time()
            cycle_paused_time = 0  # Track total paused time
            
            # Send cycle started message
            if orchestrator.send_message:
                await orchestrator.send_message({
                    "type": "monitoring_cycle_started",
                    "session_id": session_id,
                    "cycle_session_id": cycle_session_id,
                    "cycle_number": cycle_count
                })
            
            # Initialize V2 agents fresh for each cycle
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
            
            # Run classification agents with V2 timer-based processing
            agent_list = ["initial_classifier", "detail_extractor", "damage_detector"]
            current_agent_index = 0
            
            while current_agent_index < len(agent_list):
                agent_name = agent_list[current_agent_index]
                
                # Check for pause with restart handling
                pause_start = None
                while monitoring_control.get('paused', False):
                    if pause_start is None:
                        pause_start = time.time()
                        logger.debug(f"Monitoring paused for session {cycle_session_id}")
                    await asyncio.sleep(0.5)
                    if not monitoring_control.get('running', True):
                        logger.info(f"Monitoring stopped during pause for session {cycle_session_id}")
                        return
                    
                    # Check if we should restart the agent when resumed
                    if not monitoring_control.get('paused', False) and monitoring_control.get('restart_agent', False):
                        if pause_start:
                            cycle_paused_time += time.time() - pause_start
                            pause_start = None
                        logger.info(f"Restarting agent {agent_name} after resume")
                        monitoring_control['restart_agent'] = False
                        # Clear any partial results for this agent
                        if agent_name in results:
                            del results[agent_name]
                        break
                
                # Add pause time if we just resumed
                if pause_start:
                    cycle_paused_time += time.time() - pause_start
                    pause_start = None
                
                # Update current agent in control
                short_name = agent_name.replace('_classifier', '').replace('_extractor', '').replace('_detector', '')
                monitoring_control['current_agent'] = short_name
                
                # Check if monitoring should stop
                if not monitoring_control.get('running', True):
                    logger.info(f"Monitoring stopped during {agent_name}")
                    break
                    
                agent = agents[agent_name]
                timer_seconds = orchestrator.get_agent_timer(agent_name)
                
                logger.info(f"Running {agent_name} with {timer_seconds}s timer")
                
                # Send agent_started via V2 message system
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "agent_started",
                        "session_id": cycle_session_id,
                        "agent": agent_name,
                        "timer_seconds": timer_seconds,
                        "mode": "automatic"
                    })
                
                # Run agent with timer through orchestrator - with interruption support
                try:
                    # Verify VLM is still ready before running agent
                    if not orchestrator.is_vlm_configured():
                        logger.error(f"VLM not configured before running {agent_name}, attempting re-init...")
                        init_success = await orchestrator.initialize_gpu_optimization()
                        if not init_success:
                            raise RuntimeError("VLM engine not available")
                    
                    # Create a task for the agent
                    agent_task = asyncio.create_task(orchestrator.run_agent_with_timer(agent_name, timer_seconds))
                    
                    # Wait for agent to complete or interruption signal
                    while not agent_task.done():
                        await asyncio.sleep(0.1)
                        
                        # Check for interruption
                        if monitoring_control.get('should_interrupt', False):
                            logger.info(f"Interrupting agent {agent_name} due to pause")
                            agent_task.cancel()
                            try:
                                await agent_task
                            except asyncio.CancelledError:
                                pass
                            monitoring_control['should_interrupt'] = False
                            # Don't increment index, we'll restart this agent
                            break
                    
                    if not monitoring_control.get('paused', False):
                        # Agent completed normally
                        agent_state = agent_task.result() if agent_task.done() and not agent_task.cancelled() else None
                        if agent_state:
                            results[agent_name] = agent_state
                            
                            # Send agent_completed with attributes
                            if orchestrator.send_message:
                                await orchestrator.send_message({
                                    "type": "agent_completed", 
                                    "session_id": cycle_session_id,
                                    "agent": agent_name,
                                    "results": agent_state.finalized_attributes
                                })
                            
                            # Move to next agent
                            current_agent_index += 1
                        else:
                            logger.warning(f"Agent {agent_name} returned no state")
                            current_agent_index += 1
                    
                except Exception as e:
                    logger.error(f"Error running agent {agent_name}: {e}")
                    # Send error to frontend
                    if orchestrator.send_message:
                        await orchestrator.send_message({
                            "type": "agent_error",
                            "agent": agent_name,
                            "error": str(e),
                            "session_id": cycle_session_id
                        })
                    # Skip to next agent
                    current_agent_index += 1
            
            # Now run final_compiler to aggregate all results
            if not monitoring_control.get('running', True):
                logger.info("Monitoring stopped before final_compiler")
            else:
                logger.info("Running final_compiler to aggregate results")
                
                # Send agent_started for final_compiler
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "agent_started",
                        "session_id": cycle_session_id,
                        "agent": "final_compiler",
                        "timer_seconds": 1.0,
                        "mode": "automatic"
                    })
                
                # Prepare agent results for final compiler
                agent_results = {}
                for agent_name, agent_state in results.items():
                    if agent_state and agent_state.finalized_attributes:
                        agent_results[agent_name] = {
                            "attributes": agent_state.finalized_attributes,
                            "confidence": agent_state.finalized_attributes.get("confidence", 0.9),
                            "total_inferences": agent_state.inference_count
                        }
                
                # Run final compiler to aggregate
                final_compiler = agents["final_compiler"]
                final_result = await final_compiler.compile_results(agent_results)
                
                # Calculate actual cycle duration
                cycle_end_time = time.time()
                actual_cycle_duration = (cycle_end_time - cycle_start_time) - cycle_paused_time
                
                # Send agent_completed for final_compiler
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "agent_completed", 
                        "session_id": cycle_session_id,
                        "agent": "final_compiler",
                        "results": final_result.to_dict() if hasattr(final_result, 'to_dict') else final_result
                    })
            
                # Send final classification complete message
                if orchestrator.send_message:
                    final_dict = final_result.to_dict() if hasattr(final_result, 'to_dict') else final_result
                    await orchestrator.send_message({
                        "type": "classification_complete",
                        "session_id": cycle_session_id,
                        "cycle_number": cycle_count,
                        "final_result": final_dict,
                        "processing_time": actual_cycle_duration,  # Use actual measured time
                        "agents_completed": len(results) + 1  # Include final_compiler
                    })
                
                # Log cycle timing details
                logger.info(f"Cycle {cycle_count} completed: {actual_cycle_duration:.1f}s (paused: {cycle_paused_time:.1f}s)")
            
            logger.info(f"Monitoring cycle {cycle_count} completed for session {cycle_session_id}")
            
            # Clear orchestrator state for next cycle
            orchestrator.session_state.reset_for_new_cycle()
            
            # Small delay between cycles
            await asyncio.sleep(4)
        
        logger.info(f"V1-to-V2 monitoring flow stopped after {cycle_count} cycles")
        
    except Exception as e:
        logger.error(f"Error in V1-to-V2 monitoring flow: {e}", exc_info=True)
        if orchestrator.send_message:
            await orchestrator.send_message({
                "type": "error",
                "session_id": session_id,
                "message": f"Monitoring flow error: {str(e)}"
            })
    finally:
        # Clean up monitoring control
        logger.info(f"Cleaning up monitoring flow for session {session_id}")
        monitoring_control['running'] = False
        monitoring_control['paused'] = False
        
        # Clean up any cycle-specific session data
        if session_id in monitoring_controls:
            del monitoring_controls[session_id]
        
        # Clean up any associated data channels that might be stuck
        from api import main
        if main.webrtc_manager:
            # Clear any pending messages for this session
            logger.info(f"Cleaning up WebRTC resources for session {session_id}")

async def run_manual_mode_flow(session_id: str, orchestrator, monitoring_control, manual_handler: ManualModeHandler):
    """Run manual mode monitoring flow with timer-based processing and command waiting"""
    try:
        logger.info(f"Starting manual mode flow for session {session_id}")
        
        # Ensure VLM is initialized
        if not orchestrator.is_vlm_configured():
            logger.info("Initializing GPU optimization for manual mode...")
            init_success = await orchestrator.initialize_gpu_optimization()
            if not init_success:
                logger.error("Failed to initialize VLM for manual mode")
                monitoring_control['running'] = False
                return
        
        cycle_count = 0
        
        # Run until stopped
        while monitoring_control.get('running', True) and manual_handler.is_active():
            cycle_count += 1
            manual_handler.current_cycle = cycle_count
            
            # Create cycle session ID
            cycle_session_id = f"{session_id}_manual_cycle_{cycle_count}"
            logger.info(f"Starting manual mode cycle {cycle_count}")

            # Send cycle started message to frontend
            if orchestrator.send_message:
                await orchestrator.send_message({
                    "type": "monitoring_cycle_started",
                    "session_id": session_id,
                    "cycle_session_id": cycle_session_id,
                    "cycle_number": cycle_count
                })

            # Initialize agents for this cycle
            from v2.agents.initial_classifier_v2 import InitialClassifierV2
            from v2.agents.detail_extractor_v2 import DetailExtractorV2
            from v2.agents.damage_detector_v2 import DamageDetectorV2
            from v2.agents.final_compiler_v2 import FinalCompilerV2
            
            agents = {
                "initial_classifier": InitialClassifierV2(),
                "detail_extractor": DetailExtractorV2(),
                "damage_detector": DamageDetectorV2(),
                "final_compiler": FinalCompilerV2()
            }
            
            # Wire frame management
            for agent_name, agent in agents.items():
                if hasattr(agent, 'set_frame_registry'):
                    agent.set_frame_registry(orchestrator.frame_registry)
                if hasattr(agent, 'set_frame_provider'):
                    agent.set_frame_provider(orchestrator.frame_provider)
                if hasattr(agent, 'set_inference_engine'):
                    agent.set_inference_engine(orchestrator.inference_engine)
            
            # Skip reset here - already done by reset_after_final_compiler()
            # Only reset if this is the very first cycle
            if cycle_count == 1:
                manual_handler.reset()
            
            # Start with the first agent automatically
            first_agent_run = True
            
            # Ensure clean state for new cycle
            monitoring_control['waiting_for_manual_agent'] = False
            monitoring_control['manual_agent'] = None
            # Starting cycle with clean monitoring control state
            
            # Run agents with manual control
            while not manual_handler.can_compile():
                # For the first agent, run it immediately
                if first_agent_run:
                    agent_name = manual_handler.get_current_agent()
                    logger.info(f"Manual mode: Starting with first agent {agent_name}")
                    first_agent_run = False
                else:
                    # Wait for user command
                    manual_handler.waiting_for_command = True
                    await manual_handler._send_status_update()
                    
                    logger.info(f"Manual mode: Waiting for command, current agent: {manual_handler.get_current_agent()}")
                    logger.debug(f"Monitoring control state: waiting_for_manual_agent={monitoring_control.get('waiting_for_manual_agent', False)}, manual_agent={monitoring_control.get('manual_agent')}")
                    
                    # Wait for command signal from monitoring_control
                    while not monitoring_control.get('waiting_for_manual_agent', False):
                        await asyncio.sleep(0.1)
                        if not monitoring_control.get('running', True):
                            logger.info("Manual mode stopped")
                            return
                    
                    # Get the agent to run
                    agent_name = monitoring_control.get('manual_agent')
                    monitoring_control['waiting_for_manual_agent'] = False
                
                if not agent_name:
                    continue
                
                # Run the agent with timer
                logger.info(f"Manual mode: Running agent {agent_name}")
                
                # Get timer for this agent
                timer_seconds = orchestrator.get_agent_timer(agent_name)
                
                # Send agent_started
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "agent_started",
                        "session_id": cycle_session_id,
                        "agent": agent_name,
                        "timer_seconds": timer_seconds,
                        "mode": "manual"
                    })
                
                # Mark agent as started in manual session
                if session_id in manual_mode_sessions:
                    manual_mode_sessions[session_id].record_agent_start(agent_name, timer_seconds)
                
                # Run agent with timer
                try:
                    agent_state = await orchestrator.run_agent_with_timer(agent_name, timer_seconds)
                    
                    if agent_state:
                        # Mark agent as completed
                        manual_handler.mark_agent_completed(agent_name, agent_state.finalized_attributes)
                        
                        # Record in session state
                        if session_id in manual_mode_sessions:
                            manual_mode_sessions[session_id].record_agent_completion(
                                agent_name, 
                                agent_state.finalized_attributes
                            )
                        
                        # Send agent_completed
                        if orchestrator.send_message:
                            await orchestrator.send_message({
                                "type": "agent_completed",
                                "session_id": cycle_session_id,
                                "agent": agent_name,
                                "results": agent_state.finalized_attributes,
                                "waiting_for_command": True
                            })
                        
                        # Send status update
                        await manual_handler._send_status_update()
                        
                except Exception as e:
                    logger.error(f"Error running agent {agent_name} in manual mode: {e}")
                    if orchestrator.send_message:
                        await orchestrator.send_message({
                            "type": "agent_error",
                            "agent": agent_name,
                            "error": str(e),
                            "session_id": cycle_session_id
                        })
            
            # Check if we should run final compiler
            if monitoring_control.get('run_final_compiler', False):
                monitoring_control['run_final_compiler'] = False
                
                logger.info("Manual mode: Running final compiler")
                
                # Send agent_started for final compiler
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "agent_started",
                        "session_id": cycle_session_id,
                        "agent": "final_compiler",
                        "timer_seconds": 1.0,
                        "mode": "manual"
                    })
                
                # Prepare results for final compiler
                agent_results = {}
                for agent_name in manual_handler.agent_sequence:
                    if agent_name in manual_handler.agent_results:
                        agent_results[agent_name] = {
                            "attributes": manual_handler.agent_results[agent_name],
                            "confidence": manual_handler.agent_results[agent_name].get("confidence", 0.9)
                        }
                
                # Run final compiler
                final_compiler = agents["final_compiler"]
                final_result = await final_compiler.compile_results(agent_results)
                
                # Send final results
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "classification_complete",
                        "session_id": cycle_session_id,
                        "cycle_number": cycle_count,
                        "final_result": final_result.to_dict() if hasattr(final_result, 'to_dict') else final_result,
                        "mode": "manual"
                    })
                
                # Record in session state
                if session_id in manual_mode_sessions:
                    manual_mode_sessions[session_id].record_final_compilation(
                        final_result.to_dict() if hasattr(final_result, 'to_dict') else final_result
                    )
                
                # Start new cycle - use the new reset method
                manual_handler.reset_after_final_compiler()
                # Set flag to run first agent of new cycle
                first_agent_run = True
                
                # Clear monitoring control flags for clean state
                monitoring_control['waiting_for_manual_agent'] = False
                monitoring_control['manual_agent'] = None
                logger.info("Cleared monitoring control flags for new cycle")
                
            # Brief delay before next cycle
            await asyncio.sleep(1.0)
            
    except Exception as e:
        logger.error(f"Error in manual mode flow: {e}", exc_info=True)
    finally:
        monitoring_control['running'] = False
        manual_handler.deactivate_manual_mode()
        logger.info(f"Manual mode flow ended for session {session_id}")
