"""WebSocket endpoint for real-time communication."""
import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store monitoring control for each session
monitoring_controls: Dict[str, Dict[str, Any]] = {}

# Store reference to webrtc_manager for control message handling
_webrtc_manager = None
_control_handler_setup = False


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
                logger.warning(f"No websocket found for session {session_id}")
                return None
            
            # Process the message through the normal websocket handler
            return await handle_websocket_message(message, ws)
        
        webrtc_manager.set_control_message_handler(handle_control_message_from_data_channel)
        _control_handler_setup = True
        logger.info("Control message handler set up for WebRTC manager")
    
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
        
        # Get frame provider from WebRTC manager and wrap it for V2 async compatibility
        v1_frame_provider = webrtc_manager.get_frame_provider(session_id)
        
        async def async_frame_provider():
            """Async wrapper for V1 frame provider"""
            try:
                frame = v1_frame_provider() if v1_frame_provider else None
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
        
        # Check if monitoring is already running for this session
        if session_id in monitoring_controls and monitoring_controls[session_id].get('running', False):
            logger.warning(f"Monitoring already running for session {session_id}")
            return {
                "type": "error",
                "message": "Monitoring already running for this session"
            }
        
        # Create monitoring control for this session
        monitoring_controls[session_id] = {'running': True}
        
        # Import V2 stateful orchestrator  
        from v2.services.stateful_orchestrator import StatefulOrchestrator
        orchestrator = StatefulOrchestrator(session_id)
        
        # Set custom timers if provided
        if message.get("agent_timers"):
            orchestrator.set_custom_timers(message["agent_timers"])
        
        # Get frame provider from WebRTC manager and wrap it for V2 async compatibility
        v1_frame_provider = webrtc_manager.get_frame_provider(session_id)
        
        async def async_frame_provider():
            """Async wrapper for V1 frame provider"""
            try:
                frame = v1_frame_provider() if v1_frame_provider else None
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
        
        # Start automatic processing (runs async) using V2 monitoring flow
        import asyncio
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
        return {
            "type": "progress_update",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "inference_num": v2_msg.get("inference_num", 1),
            "partial_results": v2_msg.get("attributes", {}),
            "confidence": v2_msg.get("confidence", 0),
            "reasoning": v2_msg.get("reasoning", "")
        }
        
    elif msg_type == "agent_completed":
        # V2: {"type": "agent_completed", "agent": "initial_classifier", "results": {...}}
        # V1: {"type": "agent_completed", "agent": "initial", "results": {...}}
        return {
            "type": "agent_completed",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "results": v2_msg.get("results", {})
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
        
        # Run continuously until stop signal
        while monitoring_control.get('running', True):
            cycle_count += 1
            
            # Create a new session ID for each cycle (keeps base session ID with cycle suffix)
            cycle_session_id = f"{session_id}_cycle_{cycle_count}"
            logger.info(f"Starting monitoring cycle {cycle_count} with session {cycle_session_id}")
            
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
                while monitoring_control.get('paused', False):
                    logger.debug(f"Monitoring paused for session {cycle_session_id}")
                    await asyncio.sleep(0.5)
                    if not monitoring_control.get('running', True):
                        logger.info(f"Monitoring stopped during pause for session {cycle_session_id}")
                        return
                    
                    # Check if we should restart the agent when resumed
                    if not monitoring_control.get('paused', False) and monitoring_control.get('restart_agent', False):
                        logger.info(f"Restarting agent {agent_name} after resume")
                        monitoring_control['restart_agent'] = False
                        # Clear any partial results for this agent
                        if agent_name in results:
                            del results[agent_name]
                        break
                
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
                        "processing_time": sum(r.timer_seconds for r in results.values() if r) + 1.0,  # Add compiler time
                        "agents_completed": len(results) + 1  # Include final_compiler
                    })
            
            logger.info(f"Monitoring cycle {cycle_count} completed for session {cycle_session_id}")
            
            # Clear orchestrator state for next cycle
            orchestrator.session_state.reset_for_new_cycle()
            
            # Small delay between cycles
            import asyncio
            await asyncio.sleep(1)
        
        logger.info(f"V1-to-V2 monitoring flow stopped after {cycle_count} cycles")
        
    except Exception as e:
        logger.error(f"Error in V1-to-V2 monitoring flow: {e}", exc_info=True)
        if orchestrator.send_message:
            await orchestrator.send_message({
                "type": "error",
                "session_id": session_id,
                "message": f"Monitoring flow error: {str(e)}"
            })