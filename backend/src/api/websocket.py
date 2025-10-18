"""WebSocket endpoint for real-time communication."""
import json
import logging
import os
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

from orchestration.utils.key_normalizer import KeyNormalizer
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

# Global cycle counter - persists across sessions, resets only on backend restart
_global_cycle_counter = 0

# Cleanup task reference
_cleanup_task = None


async def cleanup_stale_monitoring_controls():
    """Periodic cleanup of stale monitoring_controls entries.

    Runs every 5 minutes and removes entries that are:
    - NOT running (stopped sessions)
    - Have no active connection (user disconnected)
    - Have been inactive for 30+ minutes

    This prevents unbounded growth from abandoned sessions while preserving
    active sessions and recently stopped sessions that might be resumed.
    """
    import time

    # Track last activity time for each session
    last_activity: Dict[str, float] = {}

    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes

            current_time = time.time()
            stale_threshold = 1800  # 30 minutes in seconds

            entries_to_remove = []

            for session_id, control in list(monitoring_controls.items()):
                # Update last activity if session is running
                if control.get('running', False):
                    last_activity[session_id] = current_time
                    continue

                # Check if session has active connection
                has_connection = session_id in active_connections
                if has_connection:
                    last_activity[session_id] = current_time
                    continue

                # Check if session is stale (no activity for 30+ min)
                last_seen = last_activity.get(session_id, current_time)
                inactive_duration = current_time - last_seen

                if inactive_duration > stale_threshold:
                    entries_to_remove.append(session_id)

            # Remove stale entries
            for session_id in entries_to_remove:
                del monitoring_controls[session_id]
                if session_id in last_activity:
                    del last_activity[session_id]
                logger.info(f"Cleaned up stale monitoring_controls entry for {session_id}")

            if entries_to_remove:
                logger.info(f"Cleanup: Removed {len(entries_to_remove)} stale entries from monitoring_controls")

        except Exception as e:
            logger.error(f"Error in monitoring_controls cleanup task: {e}")


def start_cleanup_task():
    """Start the periodic cleanup task for monitoring_controls."""
    global _cleanup_task

    if _cleanup_task is None:
        _cleanup_task = asyncio.create_task(cleanup_stale_monitoring_controls())
        logger.info("Started periodic cleanup task for monitoring_controls")


def escape_csv_value(value):
    """Escape CSV values to handle commas, quotes, and newlines."""
    if value is None:
        return ''

    string_value = str(value)

    # Check if value needs escaping
    if ',' in string_value or '"' in string_value or '\n' in string_value:
        # Escape quotes by doubling them
        escaped = string_value.replace('"', '""')
        return f'"{escaped}"'

    return string_value


async def generate_clean_csv(table_data: List[Dict]) -> str:
    """
    Generate clean CSV (data only, no metadata).
    Reverses order to oldest→newest (opposite of frontend table display).

    Args:
        table_data: List of row data from frontend (newest first)

    Returns:
        CSV string content
    """
    if not table_data:
        return ''

    # Headers matching frontend clean CSV export
    headers = ['#', 'Type', 'Color', 'Pattern', 'Brand', 'Size', 'Neckline',
               'Sleeve', 'Closure', 'Damaged', 'Defect', 'Time',
               'Initial Images', 'Detail Images', 'Damage Images']

    rows = [','.join(headers)]

    # Reverse to get oldest→newest order
    reversed_data = list(reversed(table_data))

    for row_data in reversed_data:
        cells = row_data.get('cells', [])
        metadata = row_data.get('metadata', {})

        # Ensure we have at least 12 cells (visible columns)
        while len(cells) < 12:
            cells.append('')

        # Escape cell values
        escaped_cells = [escape_csv_value(cell) for cell in cells[:12]]

        # Add final images from metadata
        final_images = metadata.get('final_images', {})

        def format_image_list(images):
            if not images or not isinstance(images, list) or len(images) == 0:
                return '-'
            return f'"{"; ".join(images)}"'

        escaped_cells.append(format_image_list(final_images.get('initial_classifier', [])))
        escaped_cells.append(format_image_list(final_images.get('detail_extractor', [])))
        escaped_cells.append(format_image_list(final_images.get('damage_detector', [])))

        rows.append(','.join(escaped_cells))

    return '\n'.join(rows)


async def generate_metadata_csv(table_data: List[Dict]) -> str:
    """
    Generate metadata CSV (full details with images, cycle info, inference counts).
    Reverses order to oldest→newest (opposite of frontend table display).

    Args:
        table_data: List of row data from frontend (newest first)

    Returns:
        CSV string content
    """
    if not table_data:
        return ''

    # Headers matching frontend metadata CSV export
    headers = [
        '#', 'Cycle ID', 'Mode', 'Type', 'Color', 'Pattern', 'Brand', 'Size',
        'Neckline', 'Sleeve', 'Closure', 'Damage', 'Damage Type', 'Timestamp',
        'Initial Inferences', 'Detail Inferences', 'Damage Inferences', 'Edited Cells',
        'Final Initial Image', 'Final Detail Image', 'Final Damage Image',
        'All Initial Images', 'All Detail Images', 'All Damage Images'
    ]

    rows = [','.join(headers)]

    # Reverse to get oldest→newest order
    reversed_data = list(reversed(table_data))

    for row_data in reversed_data:
        cells = row_data.get('cells', [])
        metadata = row_data.get('metadata', {})
        cycle_id = row_data.get('cycle_id', 'unknown')

        # Ensure we have at least 12 cells
        while len(cells) < 12:
            cells.append('')

        # Build row with metadata
        row = []

        # Basic cells (visible columns)
        row.extend([escape_csv_value(cell) for cell in cells[:12]])

        # Add cycle ID after row number
        row.insert(1, escape_csv_value(cycle_id))

        # Add mode
        mode = metadata.get('mode', 'unknown')
        row.insert(2, escape_csv_value(mode))

        # Timestamp is already in cells[11], so we have it

        # Add inference counts
        inference_counts = metadata.get('inference_counts', {})
        row.append(escape_csv_value(inference_counts.get('initial_classifier', 0)))
        row.append(escape_csv_value(inference_counts.get('detail_extractor', 0)))
        row.append(escape_csv_value(inference_counts.get('damage_detector', 0)))

        # Add edited cells info
        edits = metadata.get('edits', {})
        edited_cells_list = list(edits.keys()) if edits else []
        row.append(escape_csv_value('; '.join(edited_cells_list) if edited_cells_list else '-'))

        # Add final images (single image per agent)
        final_images = metadata.get('final_images', {})

        def format_single_image(images):
            if not images or not isinstance(images, list) or len(images) == 0:
                return '-'
            # Return first image (final successful image)
            return escape_csv_value(images[0])

        row.append(format_single_image(final_images.get('initial_classifier', [])))
        row.append(format_single_image(final_images.get('detail_extractor', [])))
        row.append(format_single_image(final_images.get('damage_detector', [])))

        # Add all images (all attempts including retries)
        saved_images = metadata.get('saved_images', {})

        def format_all_images(images):
            if not images or not isinstance(images, list) or len(images) == 0:
                return '-'
            return f'"{"; ".join(images)}"'

        row.append(format_all_images(saved_images.get('initial_classifier', [])))
        row.append(format_all_images(saved_images.get('detail_extractor', [])))
        row.append(format_all_images(saved_images.get('damage_detector', [])))

        rows.append(','.join(row))

    return '\n'.join(rows)


async def setup_orchestrator_vlm(orchestrator):
    """Setup VLM integration for orchestrator"""
    # The MultiInferenceEngine already has its own VLM engine
    # We just need to ensure it's initialized
    try:
        if hasattr(orchestrator, 'inference_engine'):
            # The inference engine already has VLM integrated
            logger.info("VLM already integrated in MultiInferenceEngine")
            return True
        else:
            logger.error("Orchestrator missing inference_engine")
            return False
    except Exception as e:
        logger.error(f"Failed to verify VLM integration: {e}")
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
        if session_id:
            # Clean up active connection
            if session_id in active_connections:
                del active_connections[session_id]

            # CRITICAL: Release WebRTC connection and camera
            # This triggers the camera release when browser refreshes/disconnects
            from api import main
            if main.webrtc_manager:
                logger.info(f"Releasing WebRTC connection for session {session_id}")
                await main.webrtc_manager.close_connection(session_id)
                logger.info(f"WebRTC connection released for {session_id}")

            # Stop monitoring and clean up orchestrator if running
            if session_id in monitoring_controls:
                logger.info(f"Stopping monitoring for disconnected session {session_id}")
                monitoring_controls[session_id]['running'] = False
                monitoring_controls[session_id]['paused'] = False

            # Clean up orchestrator
            if session_id in orchestrators:
                logger.info(f"Cleaning up orchestrator for disconnected session {session_id}")
                try:
                    await orchestrators[session_id].cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up orchestrator: {e}")
                del orchestrators[session_id]

            # Clean up manual mode if active
            if session_id in manual_mode_handlers:
                logger.info(f"Cleaning up manual mode for disconnected session {session_id}")
                manual_mode_handlers[session_id].deactivate_manual_mode()
                del manual_mode_handlers[session_id]

            if session_id in manual_mode_sessions:
                del manual_mode_sessions[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if session_id:
            # Clean up active connection
            if session_id in active_connections:
                del active_connections[session_id]

            # Release WebRTC connection on any error
            from api import main
            if main.webrtc_manager:
                logger.info(f"Releasing WebRTC connection for session {session_id} due to error")
                await main.webrtc_manager.close_connection(session_id)
                logger.info(f"WebRTC connection released for {session_id}")

            # Stop monitoring and clean up orchestrator if running
            if session_id in monitoring_controls:
                logger.info(f"Stopping monitoring for errored session {session_id}")
                monitoring_controls[session_id]['running'] = False
                monitoring_controls[session_id]['paused'] = False

            # Clean up orchestrator
            if session_id in orchestrators:
                logger.info(f"Cleaning up orchestrator for errored session {session_id}")
                try:
                    await orchestrators[session_id].cleanup()
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up orchestrator: {cleanup_error}")
                del orchestrators[session_id]

            # Clean up manual mode if active
            if session_id in manual_mode_handlers:
                logger.info(f"Cleaning up manual mode for errored session {session_id}")
                manual_mode_handlers[session_id].deactivate_manual_mode()
                del manual_mode_handlers[session_id]

            if session_id in manual_mode_sessions:
                del manual_mode_sessions[session_id]


async def handle_websocket_message(message: Dict[str, Any], websocket: WebSocket) -> Dict:
    """Handle incoming WebSocket message and return response."""
    
    # Import at module level to avoid import issues
    from api import main
    
    session_manager = main.session_manager
    webrtc_manager = main.webrtc_manager
    
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

        # Get table row limit from environment for frontend tracking
        table_row_limit = int(os.getenv('TABLE_ROW_LIMIT_WARNING', '200'))

        return {
            "type": "session_created",
            "session_id": session.session_id,
            "mode": session.mode,
            "status": session.status,
            "table_row_limit": table_row_limit
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

        # Keep monitoring_controls for potential session resumption
        logger.info(f"Session {session_id} stopped but monitoring controls preserved")

        # Clean up orchestrators
        if session_id in orchestrators:
            orchestrator = orchestrators[session_id]
            try:
                await orchestrator.cleanup()
                logger.info(f"Orchestrator cleanup completed for {session_id}")
            except Exception as e:
                logger.error(f"Error during orchestrator cleanup: {e}")
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
        
        # Import stateful orchestrator
        from src.orchestration.services.stateful_orchestrator import StatefulOrchestrator
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

        # Note: VLM initialization is handled by the orchestrator in run_v1_to_v2_monitoring_flow
        # The orchestrator uses vlm_singleton which is provider-aware (Ollama or HuggingFace)

        # Verify data channel is ready
        if not webrtc_manager.is_data_channel_ready(session_id):
            logger.warning(f"Data channel not ready for session {session_id}, waiting...")
            # Try to wait for data channel (10s timeout - must be longer than ICE gathering timeout)
            ready = await webrtc_manager.wait_for_data_channel(session_id, timeout=10000)
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
        
        # Import stateful orchestrator
        from orchestration.services.stateful_orchestrator import StatefulOrchestrator
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
                
                # Debug logging for development (converted to DEBUG level for production)
                logger.debug(f"Sending {msg_type} to frontend for {agent}")
                
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
            # Keep monitoring_controls for potential session resumption
            # Will be cleaned up on stop_session or backend restart
            logger.info(f"Session {session_id} preserved for potential resumption")

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
        # Pause the monitoring flow using orchestrator
        session_id = message.get("session_id")

        if session_id in monitoring_controls and session_id in orchestrators:
            # Set monitoring control flags
            monitoring_controls[session_id]['paused'] = True
            monitoring_controls[session_id]['should_interrupt'] = True  # Signal to interrupt current agent
            current_agent = monitoring_controls[session_id].get('current_agent', 'unknown')
            current_agent_full = monitoring_controls[session_id].get('current_agent_full', None)

            # Detect if pausing between cycles (no active agent)
            orchestrator = orchestrators[session_id]

            if current_agent_full is None:
                # Pausing between cycles - no specific agent to restart
                monitoring_controls[session_id]['paused_at_agent'] = None
                monitoring_controls[session_id]['paused_between_cycles'] = True
                monitoring_controls[session_id]['agent_had_inferences'] = False
                logger.info(f"Paused between cycles for session {session_id} (no active agent)")
            else:
                # Pausing during an agent - check if agent had inferences BEFORE state is cleared
                agent_state = orchestrator.session_state.agent_states.get(current_agent_full)
                had_inferences = agent_state and agent_state.inference_count > 0

                # Store pause information
                monitoring_controls[session_id]['paused_at_agent'] = current_agent_full
                monitoring_controls[session_id]['paused_between_cycles'] = False
                monitoring_controls[session_id]['agent_had_inferences'] = had_inferences

                logger.info(f"Paused during agent {current_agent_full} for session {session_id} (had {agent_state.inference_count if agent_state else 0} inferences)")

            # Call orchestrator's pause_flow with the FULL agent name to handle state cleanup
            orchestrator_response = await orchestrator.pause_flow(current_agent=current_agent_full)

            logger.info(f"Paused monitoring flow for session {session_id} - state cleared")

            return {
                "type": "flow_paused",
                "session_id": session_id,
                "paused_agent": current_agent,
                "state_cleared": orchestrator_response.get("state_cleared", False),
                "had_partial_results": orchestrator_response.get("had_partial_results", False)
            }
        else:
            return {
                "type": "error",
                "message": "No active monitoring to pause"
            }
    
    elif msg_type == "resume_flow":
        # Resume the monitoring flow using orchestrator
        session_id = message.get("session_id")
        restart_agent = message.get("restart_agent", False)

        if session_id in monitoring_controls and session_id in orchestrators:
            # Update monitoring control flags
            monitoring_controls[session_id]['paused'] = False
            monitoring_controls[session_id]['should_interrupt'] = False

            # Check if paused between cycles or during an agent
            paused_between_cycles = monitoring_controls[session_id].get('paused_between_cycles', False)
            current_agent = monitoring_controls[session_id].get('paused_at_agent', monitoring_controls[session_id].get('current_agent', 'unknown'))

            orchestrator = orchestrators[session_id]

            if paused_between_cycles:
                # Paused between cycles - just continue to next cycle (no agent to restart)
                monitoring_controls[session_id]['restart_agent'] = False
                monitoring_controls[session_id]['paused_between_cycles'] = False  # Clear flag
                logger.info(f"Resume from inter-cycle pause for session {session_id} - continuing to next cycle")

                return {
                    "type": "flow_resumed",
                    "session_id": session_id,
                    "resuming_agent": None,
                    "fresh_start": True,
                    "between_cycles": True
                }
            else:
                # Paused during an agent - restart that specific agent
                monitoring_controls[session_id]['restart_agent'] = True  # Signal to restart the paused agent

                # Use the stored value from pause (agent state is already cleared)
                agent_had_inferences = monitoring_controls[session_id].get('agent_had_inferences', False)

                # Increment redo counter ONLY if agent had completed at least one inference before pause
                if agent_had_inferences:
                    # This is a redo - agent was paused AFTER starting inferences
                    if current_agent not in orchestrator.redo_counts:
                        orchestrator.redo_counts[current_agent] = 0
                    orchestrator.redo_counts[current_agent] += 1
                    logger.info(f"Resume (auto mode) - Rerun attempt #{orchestrator.redo_counts[current_agent]} for {current_agent} (was paused after inferences)")
                else:
                    # This is the FIRST run - agent was paused BEFORE completing any inference
                    logger.info(f"Resume (auto mode) - First run for {current_agent} (was paused before first inference)")

                # Call orchestrator's resume_flow (state already cleared during pause)
                orchestrator_response = await orchestrator.resume_flow(restart_agent)

                logger.info(f"Resumed monitoring flow for session {session_id} - agent {current_agent} will start fresh")

                return {
                    "type": "flow_resumed",
                    "session_id": session_id,
                    "resuming_agent": current_agent,
                    "fresh_start": orchestrator_response.get("fresh_start", True),
                    "timer_seconds": orchestrator_response.get("timer_seconds", 4.0)
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
    
    elif msg_type == "tap_to_focus":
        # Handle tap-to-focus command
        session_id = message.get("session_id")
        x = message.get("x", 0.5)  # Normalized coordinates (0-1)
        y = message.get("y", 0.5)

        logger.info(f"Tap-to-focus command received for session {session_id} at ({x:.3f}, {y:.3f})")

        # Get the camera service and trigger focus
        if _webrtc_manager:
            video_track = _webrtc_manager.get_video_track(session_id)
            if video_track and hasattr(video_track, 'set_focus_point'):
                video_track.set_focus_point(x, y)
                logger.info(f"Focus set to ({x:.3f}, {y:.3f}) for session {session_id}")
                return {
                    "type": "tap_to_focus_ack",
                    "status": "success",
                    "x": x,
                    "y": y
                }
            else:
                logger.warning(f"Video track does not support tap-to-focus for session {session_id}")
                return {
                    "type": "tap_to_focus_ack",
                    "status": "not_supported"
                }

        return {
            "type": "tap_to_focus_ack",
            "status": "error",
            "message": "WebRTC manager not available"
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
                if action == "redo_agent" or action == "select_agent":
                    # For redo or select_agent (jumping back), use the agent specified
                    agent_name = result.get("agent", handler.get_current_agent())

                    # Increment redo counter for this agent in orchestrator
                    if session_id in orchestrators:
                        orchestrator = orchestrators[session_id]
                        orchestrator.is_redo = True

                        # Increment redo count for this agent
                        if agent_name not in orchestrator.redo_counts:
                            orchestrator.redo_counts[agent_name] = 0
                        orchestrator.redo_counts[agent_name] += 1

                        action_label = "Redo" if action == "redo_agent" else "Rerun (node click)"
                        logger.info(f"{action_label} attempt #{orchestrator.redo_counts[agent_name]} for {agent_name} in {session_id}")
                else:
                    # For normal forward navigation (next/previous), use the current agent
                    agent_name = handler.get_current_agent()

                    # Clear redo flag for normal navigation (but keep redo counts)
                    if session_id in orchestrators:
                        orchestrators[session_id].is_redo = False

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
    
    elif msg_type == "auto_export_csv":
        # Auto-export CSV files when table limit is reached
        session_id = message.get("session_id")
        table_data = message.get("table_data", [])

        # Validate input
        if not table_data:
            logger.warning(f"Auto-export received empty table_data for session {session_id}")
            return {
                "type": "error",
                "error_code": "EMPTY_TABLE_DATA",
                "message": "No table data to export",
                "recoverable": False
            }

        if not isinstance(table_data, list):
            logger.error(f"Auto-export received invalid table_data type: {type(table_data)}")
            return {
                "type": "error",
                "error_code": "INVALID_TABLE_DATA",
                "message": "Table data must be a list",
                "recoverable": False
            }

        try:
            # Ensure exports directory exists
            exports_dir = Path(__file__).parent.parent.parent / "exports"

            if not exports_dir.exists():
                exports_dir.mkdir(parents=True, exist_ok=True)

            # Check directory is writable
            if not os.access(exports_dir, os.W_OK):
                logger.error(f"Exports directory is not writable: {exports_dir}")
                return {
                    "type": "error",
                    "error_code": "DIRECTORY_NOT_WRITABLE",
                    "message": f"Cannot write to exports directory: {exports_dir}",
                    "recoverable": False
                }

            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Generate CSVs
            clean_csv = await generate_clean_csv(table_data)
            if not clean_csv:
                logger.warning("Clean CSV generation returned empty string")

            metadata_csv = await generate_metadata_csv(table_data)
            if not metadata_csv:
                logger.warning("Metadata CSV generation returned empty string")

            # Define filenames
            clean_filename = f"relo_classification_{timestamp}.csv"
            metadata_filename = f"relo_classification_metadata_{timestamp}.csv"

            # Write files
            clean_filepath = exports_dir / clean_filename
            metadata_filepath = exports_dir / metadata_filename

            try:
                with open(clean_filepath, 'w', encoding='utf-8') as f:
                    f.write(clean_csv)
            except Exception as write_error:
                logger.error(f"Failed to write clean CSV: {write_error}")
                raise

            try:
                with open(metadata_filepath, 'w', encoding='utf-8') as f:
                    f.write(metadata_csv)
            except Exception as write_error:
                logger.error(f"Failed to write metadata CSV: {write_error}")
                raise

            logger.info(f"Auto-export completed: {clean_filename}, {metadata_filename} ({len(table_data)} rows)")

            return {
                "type": "auto_export_success",
                "session_id": session_id,
                "files": {
                    "clean_csv": clean_filename,
                    "metadata_csv": metadata_filename
                },
                "row_count": len(table_data),
                "export_directory": str(exports_dir.absolute())
            }

        except Exception as e:
            logger.error(f"Error during auto-export: {e}", exc_info=True)
            return {
                "type": "error",
                "error_code": "AUTO_EXPORT_FAILED",
                "message": f"Failed to export CSVs: {str(e)}",
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

        # Results are already normalized by frame_aggregator, no need to normalize again
        results = v2_msg.get("results", {})

        return {
            "type": "agent_completed",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "results": results
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
            "cycle_number": v2_msg.get("cycle_number", 1),
            "saved_images": v2_msg.get("saved_images", {}),
            "final_images": v2_msg.get("final_images", {}),
            "inference_counts": v2_msg.get("inference_counts", {})
        }
        
    elif msg_type == "agent_paused_with_results":
        # V2: {"type": "agent_paused_with_results", "agent": "initial_classifier", "attributes": {...}, "is_partial": true}
        # V1: {"type": "progress_update", "agent": "initial", "partial_results": {...}, "is_partial": true}

        # Normalize attributes before sending to frontend
        raw_attributes = v2_msg.get("attributes", {})
        normalized_attributes = KeyNormalizer.normalize_agent_attributes(agent, raw_attributes)

        return {
            "type": "progress_update",
            "agent": short_agent,  # Use short name for frontend
            "session_id": v2_msg.get("session_id"),
            "partial_results": normalized_attributes,
            "confidence": v2_msg.get("confidence", 0),
            "is_partial": True,  # Mark as partial/deprecated
            "total_inferences": v2_msg.get("total_inferences", 0),
            "reason": "paused"  # Tell frontend why these are partial
        }

    elif msg_type == "error":
        # Pass through errors
        return v2_msg

    # Unknown message types - pass through
    return v2_msg


async def run_v1_to_v2_monitoring_flow(session_id: str, orchestrator, monitoring_control):
    """Run V2 monitoring flow adapted for V1 frontend - runs continuously until stopped"""
    global _global_cycle_counter

    try:
        logger.info(f"Starting continuous V1-to-V2 monitoring flow for session {session_id}")
        # Use global cycle counter instead of local counter (persists across sessions)
        
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

        if not orchestrator.is_vlm_configured():
            logger.error("VLM still not configured after initialization attempt")
            monitoring_control['running'] = False
            return
        
        # Run continuously until stop signal
        while monitoring_control.get('running', True):
            _global_cycle_counter += 1

            # Update orchestrator cycle number for image tracking
            orchestrator.current_cycle = _global_cycle_counter

            # Create a new session ID for each cycle (keeps base session ID with cycle suffix)
            cycle_session_id = f"{session_id}_cycle_{_global_cycle_counter}"

            # Log cycle start with box format
            logger.info("=" * 80)
            logger.info(f"CYCLE {_global_cycle_counter} STARTED")
            logger.info(f"Session: {cycle_session_id}")
            logger.info("=" * 80)
            
            # Track cycle timing
            cycle_start_time = time.time()
            cycle_paused_time = 0  # Track total paused time
            
            # Send cycle started message
            if orchestrator.send_message:
                await orchestrator.send_message({
                    "type": "monitoring_cycle_started",
                    "session_id": session_id,
                    "cycle_session_id": cycle_session_id,
                    "cycle_number": _global_cycle_counter
                })
            
            # Initialize agents fresh for each cycle
            from src.orchestration.agents.initial_classifier_v2 import InitialClassifierV2
            from src.orchestration.agents.detail_extractor_v2 import DetailExtractorV2
            from src.orchestration.agents.damage_detector_v2 import DamageDetectorV2
            from src.orchestration.agents.final_compiler_v2 import FinalCompilerV2
            
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
                
                # Update current agent in control (store both short and full names)
                short_name = agent_name.replace('_classifier', '').replace('_extractor', '').replace('_detector', '')
                monitoring_control['current_agent'] = short_name
                monitoring_control['current_agent_full'] = agent_name  # Store full name for pause_flow
                
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

                    # Extract initial classifier context for damage detector (auto mode)
                    initial_classifier_context = None
                    if agent_name == "damage_detector" and "initial_classifier" in results:
                        initial_classifier_state = results["initial_classifier"]
                        if initial_classifier_state and initial_classifier_state.finalized_attributes:
                            initial_classifier_context = initial_classifier_state.finalized_attributes
                            logger.info(f"Auto mode: Extracted initial classifier context for damage_detector: {initial_classifier_context}")

                    # Create a task for the agent
                    agent_task = asyncio.create_task(
                        orchestrator.run_agent_with_timer(
                            agent_name,
                            timer_seconds,
                            initial_classifier_context=initial_classifier_context
                        )
                    )
                    
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

            # Wait if paused before final_compiler
            pause_start = None
            while monitoring_control.get('paused', False):
                if pause_start is None:
                    pause_start = time.time()
                await asyncio.sleep(0.5)
                if not monitoring_control.get('running', True):
                    logger.info(f"Monitoring stopped during pause before final_compiler for session {cycle_session_id}")
                    return

            # Add pause time if we just resumed
            if pause_start:
                cycle_paused_time += time.time() - pause_start
                pause_start = None

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

                    # Get saved images for this cycle from inference engine
                    saved_images = orchestrator.inference_engine.get_saved_images_for_cycle(_global_cycle_counter)
                    final_images = orchestrator.inference_engine.get_final_images_for_cycle(_global_cycle_counter)

                    # Collect per-agent inference counts for frontend metadata
                    inference_counts = {}
                    for agent_name, agent_state in results.items():
                        if agent_state:
                            inference_counts[agent_name] = agent_state.inference_count

                    await orchestrator.send_message({
                        "type": "classification_complete",
                        "session_id": cycle_session_id,
                        "cycle_number": _global_cycle_counter,
                        "final_result": final_dict,
                        "processing_time": actual_cycle_duration,  # Use actual measured time
                        "agents_completed": len(results) + 1,  # Include final_compiler
                        "saved_images": saved_images,  # All images (including retries/redos)
                        "final_images": final_images,   # Final successful image per agent
                        "inference_counts": inference_counts  # Per-agent inference counts for CSV metadata
                    })

                # Log cycle timing details
                logger.info(f"Cycle {_global_cycle_counter} completed: {actual_cycle_duration:.1f}s (paused: {cycle_paused_time:.1f}s)")

            logger.info(f"Monitoring cycle {_global_cycle_counter} completed for session {cycle_session_id}")

            # Clear orchestrator state for next cycle
            orchestrator.session_state.reset_for_new_cycle()
            orchestrator.reset_for_new_cycle()  # Clear dynamic buffer timings

            # Reset redo counters for new cycle (auto mode doesn't use redo, but reset for consistency)
            orchestrator.redo_counts = {}

            # Clear current agent tracking (we're between cycles now, no active agent)
            if session_id in monitoring_controls:
                monitoring_controls[session_id]['current_agent'] = None
                monitoring_controls[session_id]['current_agent_full'] = None

            # ============================================================
            # INTER-CYCLE DELAY (Dynamic based on damage_detector inference time)
            # ============================================================
            TARGET_DELAY = 3.0  # Default 3-second cycle delay

            # Adjust delay based on damage_detector inference time (O(1) complexity)
            if "damage_detector" in results:
                damage_state = results["damage_detector"]
                if damage_state and hasattr(damage_state, 'inference_results') and damage_state.inference_results:
                    # Get last inference result (O(1) list access)
                    last_result = damage_state.inference_results[-1]

                    # InferenceResult stores metadata in attributes['inference_metadata']
                    # (VLM puts it at top level, but it gets stored inside attributes)
                    if hasattr(last_result, 'attributes') and isinstance(last_result.attributes, dict):
                        inference_meta = last_result.attributes.get('inference_metadata', {})

                        if inference_meta and 'total_duration' in inference_meta:
                            # total_duration is in milliseconds (nanoseconds / 1e6)
                            total_duration_ms = inference_meta.get('total_duration', 0)

                            # Convert to seconds (O(1) arithmetic)
                            inference_time_sec = total_duration_ms / 1000.0

                            logger.info(f"Damage detector last inference: {inference_time_sec:.2f}s")

                            # Apply conditional logic (O(1) comparisons)
                            if inference_time_sec > 7:
                                TARGET_DELAY = 0.3
                                logger.info(f"Damage inference {inference_time_sec:.1f}s > 7s → delay = 0.3s")
                            elif inference_time_sec > 6:
                                TARGET_DELAY = 1.0  # 3 - 2
                                logger.info(f"Damage inference {inference_time_sec:.1f}s > 6s → delay = 1.0s")
                            elif inference_time_sec > 5:
                                TARGET_DELAY = 2.0  # 3 - 1
                                logger.info(f"Damage inference {inference_time_sec:.1f}s > 5s → delay = 2.0s")
                            else:
                                logger.info(f"Damage inference {inference_time_sec:.1f}s ≤ 5s → default delay = 3.0s")
                        else:
                            logger.warning(f"No total_duration in inference_metadata: {inference_meta}")
                    else:
                        logger.warning("last_result.attributes is not a dict or doesn't exist")

            await asyncio.sleep(TARGET_DELAY)

        logger.info(f"V1-to-V2 monitoring flow stopped after {_global_cycle_counter} cycles")
        
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

        # Keep monitoring_controls for potential session resumption
        # Note: Export is now global/session-independent
        logger.info(f"Monitoring flow stopped, session {session_id} controls preserved")
        
        # Clean up any associated data channels that might be stuck
        from api import main
        if main.webrtc_manager:
            # Clear any pending messages for this session
            logger.info(f"Cleaning up WebRTC resources for session {session_id}")

async def run_manual_mode_flow(session_id: str, orchestrator, monitoring_control, manual_handler: ManualModeHandler):
    """Run manual mode monitoring flow with timer-based processing and command waiting"""
    global _global_cycle_counter

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

        # Track if this is first cycle for this manual session
        session_first_cycle = True
        cycle_session_id = None  # Will be set when cycle actually starts

        # Run until stopped
        while monitoring_control.get('running', True) and manual_handler.is_active():
            # Cycle counter and messages will be sent after final_compiler or on first run
            # This prevents spam every second

            # Initialize agents for this cycle
            from orchestration.agents.initial_classifier_v2 import InitialClassifierV2
            from orchestration.agents.detail_extractor_v2 import DetailExtractorV2
            from orchestration.agents.damage_detector_v2 import DamageDetectorV2
            from orchestration.agents.final_compiler_v2 import FinalCompilerV2
            
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
            # Only reset if this is the very first cycle for this session
            if session_first_cycle:
                # First cycle - increment global counter and send message
                _global_cycle_counter += 1
                manual_handler.current_cycle = _global_cycle_counter

                # Update orchestrator cycle number for image tracking
                orchestrator.current_cycle = _global_cycle_counter

                # Create cycle session ID
                cycle_session_id = f"{session_id}_manual_cycle_{_global_cycle_counter}"

                # Log cycle start with box format
                logger.info("=" * 80)
                logger.info(f"CYCLE {_global_cycle_counter} STARTED (MANUAL MODE)")
                logger.info(f"Session: {cycle_session_id}")
                logger.info("=" * 80)

                session_first_cycle = False

                # Send cycle started message to frontend
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "monitoring_cycle_started",
                        "session_id": session_id,
                        "cycle_session_id": cycle_session_id,
                        "cycle_number": _global_cycle_counter
                    })

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

                # Extract initial classifier context for damage detector (manual mode)
                initial_classifier_context = None
                if agent_name == "damage_detector" and "initial_classifier" in manual_handler.agent_results:
                    initial_classifier_context = manual_handler.agent_results["initial_classifier"]
                    logger.info(f"Manual mode: Extracted initial classifier context for damage_detector: {initial_classifier_context}")

                # Run agent with timer
                try:
                    agent_state = await orchestrator.run_agent_with_timer(
                        agent_name,
                        timer_seconds,
                        initial_classifier_context=initial_classifier_context
                    )
                    
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
                                "session_id": session_id,
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
                    # Get saved images for this cycle from inference engine
                    saved_images = orchestrator.inference_engine.get_saved_images_for_cycle(_global_cycle_counter)
                    final_images = orchestrator.inference_engine.get_final_images_for_cycle(_global_cycle_counter)

                    # Collect per-agent inference counts for frontend metadata
                    inference_counts = {}
                    for agent_name in manual_handler.agent_sequence:
                        agent_state = orchestrator.session_state.agent_states.get(agent_name)
                        if agent_state:
                            inference_counts[agent_name] = agent_state.inference_count

                    await orchestrator.send_message({
                        "type": "classification_complete",
                        "session_id": cycle_session_id,
                        "cycle_number": _global_cycle_counter,
                        "final_result": final_result.to_dict() if hasattr(final_result, 'to_dict') else final_result,
                        "mode": "manual",
                        "saved_images": saved_images,  # All images (including retries/redos)
                        "final_images": final_images,   # Final successful image per agent
                        "inference_counts": inference_counts  # Per-agent inference counts for CSV metadata
                    })

                # Record in session state
                if session_id in manual_mode_sessions:
                    manual_mode_sessions[session_id].record_final_compilation(
                        final_result.to_dict() if hasattr(final_result, 'to_dict') else final_result
                    )
                
                # Start new cycle - use the new reset method
                manual_handler.reset_after_final_compiler()

                # Reset redo counters and dynamic buffer timings for new cycle
                if session_id in orchestrators:
                    orchestrators[session_id].redo_counts = {}
                    orchestrators[session_id].reset_for_new_cycle()  # Clear dynamic buffer timings
                    logger.info(f"Reset redo counters for new cycle {_global_cycle_counter + 1}")

                # Increment global cycle counter for next cycle
                _global_cycle_counter += 1
                manual_handler.current_cycle = _global_cycle_counter

                # Update orchestrator cycle number for image tracking
                orchestrator.current_cycle = _global_cycle_counter

                # Create new cycle session ID
                cycle_session_id = f"{session_id}_manual_cycle_{_global_cycle_counter}"
                logger.info(f"Starting manual mode cycle {_global_cycle_counter} after final_compiler")

                # Send cycle started message to frontend
                if orchestrator.send_message:
                    await orchestrator.send_message({
                        "type": "monitoring_cycle_started",
                        "session_id": session_id,
                        "cycle_session_id": cycle_session_id,
                        "cycle_number": _global_cycle_counter
                    })

                # Set flag to run first agent of new cycle
                first_agent_run = True

                # Clear monitoring control flags for clean state
                monitoring_control['waiting_for_manual_agent'] = False
                monitoring_control['manual_agent'] = None

            # Brief delay before next cycle
            await asyncio.sleep(1.0)
            
    except Exception as e:
        logger.error(f"Error in manual mode flow: {e}", exc_info=True)
    finally:
        monitoring_control['running'] = False
        manual_handler.deactivate_manual_mode()
        logger.info(f"Manual mode flow ended for session {session_id}")
