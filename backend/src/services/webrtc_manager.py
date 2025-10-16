"""WebRTC connection and stream management service."""
import asyncio
import json
import logging
import socket
import os
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import time

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
    RTCConfiguration,
    RTCIceServer
)
from aiortc.rtcconfiguration import RTCBundlePolicy
from aiortc.contrib.media import MediaRelay

from models import CameraFeed, ConnectionStatus

logger = logging.getLogger(__name__)


class WebRTCManager:
    """Manages WebRTC connections and data channels."""
    
    # Class-level camera service and video track (singleton)
    _camera_service = None
    _video_track = None
    _camera_service_lock = asyncio.Lock()
    
    def __init__(self):
        """Initialize WebRTC manager."""
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.data_channels: Dict[str, RTCDataChannel] = {}
        self.camera_feeds: Dict[str, CameraFeed] = {}
        self.media_relay = MediaRelay()
        # OFFLINE MODE: Empty ICE servers for local network operation without internet
        self.ice_servers = []
        # Frame buffer for each session - stores latest frames with automatic eviction
        self.frame_buffers: Dict[str, deque] = {}
        self.max_buffer_size = 2  # Reduced from 10 to minimize latency (67ms at 30fps vs 333ms)
        # Callback for handling control messages from data channel
        self.control_message_handler = None

        # Session management for TTL-based cleanup
        self.session_timestamps: Dict[str, float] = {}  # Track when each session was created
        self.session_ttl_seconds = 3600  # 1 hour TTL by default
        self.cleanup_task = None  # Background cleanup task
        self.cleanup_interval_seconds = 60  # Run cleanup every minute
    
    async def create_peer_connection(self, session_id: str) -> RTCPeerConnection:
        """Create a new peer connection for a session."""
        # OFFLINE MODE: Create configuration for local network without STUN/TURN servers
        from aiortc import RTCConfiguration
        config = RTCConfiguration(iceServers=self.ice_servers)
        pc = RTCPeerConnection(configuration=config)
        
        # Store the connection with timestamp
        self.peer_connections[session_id] = pc
        self.session_timestamps[session_id] = time.time()  # Track creation time
        
        # Store WebSocket for sending ICE candidates
        self.websocket_for_session = getattr(self, 'websocket_for_session', {})
        
        # Set up event handlers
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"WebRTC Connection for {session_id}: state={pc.connectionState}, ICE={pc.iceConnectionState}, gathering={pc.iceGatheringState}")
            
            if pc.connectionState == "connected":
                await self.on_connection_established(session_id)
            elif pc.connectionState == "failed":
                await self.on_connection_failed(session_id)

        @pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            logger.info(f"ICE gathering state for {session_id}: {pc.iceGatheringState}")
            if pc.iceGatheringState == "complete":
                logger.info(f"ICE gathering complete for {session_id}")
        
        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state changed for {session_id}: {pc.iceConnectionState}")
            if pc.iceConnectionState == "connected":
                # Create data channel once ICE is connected
                if session_id not in self.data_channels:
                    logger.info(f"ICE connected, creating data channel for {session_id}")
                    channel = pc.createDataChannel("results", ordered=True)
                    self.data_channels[session_id] = channel

                    @channel.on("open")
                    def on_open():
                        logger.info(f"Data channel opened for {session_id}")
                        self.data_channels[session_id] = channel

                    @channel.on("close")
                    def on_close():
                        logger.info(f"Data channel closed for {session_id}")
                        if session_id in self.data_channels:
                            del self.data_channels[session_id]

                    @channel.on("message")
                    async def on_message(message):
                        await self.handle_data_channel_message(session_id, message)

                    logger.info(f"Data channel created for {session_id}, state: {channel.readyState}")
        
        # Listen for data channels from client (backward compatibility)
        @pc.on("datachannel")
        def on_datachannel(client_channel):
            logger.info(f"Data channel received from client for {session_id}: {client_channel.label}")
            # Store client-created channel if we don't have one
            if session_id not in self.data_channels:
                self.data_channels[session_id] = client_channel
                logger.info(f"Using client data channel for {session_id}")
                
                @client_channel.on("open")
                def on_open():
                    logger.info(f"Client data channel opened for {session_id}")
                
                @client_channel.on("message")
                async def on_message(message):
                    await self.handle_data_channel_message(session_id, message)
        
        logger.info(f"Created peer connection for session {session_id}")
        
        return pc
    
    async def handle_offer(self, session_id: str, offer: Dict, camera_source: str = "realsense") -> Dict:
        """Handle WebRTC offer and create answer."""
        logger.info(f"WebRTC handle_offer for {session_id}, camera: {camera_source}")
        try:
            pc = await self.create_peer_connection(session_id)

            # Initialize camera service and video track (singleton pattern with reset logic)
            async with WebRTCManager._camera_service_lock:
                if not WebRTCManager._camera_service:
                    # Get camera type from environment variable (default to CV60)
                    camera_type = os.getenv('CAMERA_TYPE', 'CV60').upper()

                    if camera_type == 'OAKD':
                        # Use OAK-D Pro camera service
                        try:
                            from services.oakd_camera import OakDCameraService
                        except ImportError:
                            from .oakd_camera import OakDCameraService
                        WebRTCManager._camera_service = OakDCameraService()
                        WebRTCManager._camera_service.initialize_camera(camera_source)
                        logger.info("OakDCameraService initialized")
                    else:
                        # Use CV60 camera service (default)
                        try:
                            from services.cv60_camera import CV60CameraService
                        except ImportError:
                            from .cv60_camera import CV60CameraService
                        WebRTCManager._camera_service = CV60CameraService()
                        WebRTCManager._camera_service.initialize_camera(camera_source)
                        logger.info("CV60CameraService initialized")

                # Check if existing video track is in test mode and needs reset
                if WebRTCManager._video_track:
                    # Check if the video track is in test mode (not properly initialized)
                    if hasattr(WebRTCManager._video_track, 'is_initialized') and not WebRTCManager._video_track.is_initialized:
                        logger.warning(f"Camera reset: video track in test mode, reinitializing for {session_id}")
                        # Stop the old track
                        if hasattr(WebRTCManager._video_track, 'stop'):
                            WebRTCManager._video_track.stop()
                        WebRTCManager._video_track = None
                    # Also reset if the track has been running for too long (stale)
                    elif hasattr(WebRTCManager._video_track, '_creation_time'):
                        import time
                        track_age = time.time() - WebRTCManager._video_track._creation_time
                        # Reset if track is older than 5 minutes (300s) - handles refresh/reconnect better
                        if track_age > 300:  # Reduced from 3600 to 300 for better refresh handling
                            logger.warning(f"Camera reset: video track is stale ({track_age:.0f}s old)")
                            if hasattr(WebRTCManager._video_track, 'stop'):
                                WebRTCManager._video_track.stop()
                            WebRTCManager._video_track = None

                # Create video track with session ID for frame buffer
                if not WebRTCManager._video_track:
                    # Add delay for OAK-D PoE device to reset after release
                    await asyncio.sleep(5.0)  # 5 second delay for PoE camera reset
                    logger.info(f"Creating new video track for session {session_id} after camera reset delay...")
                    WebRTCManager._video_track = WebRTCManager._camera_service.get_video_track(session_id)
                    logger.info(f"New video track created for session {session_id}")
            
            # Set remote description and create answer
            remote_sdp = RTCSessionDescription(
                sdp=offer["sdp"],
                type=offer.get("type", "offer")
            )
            await pc.setRemoteDescription(remote_sdp)

            # Add video track
            video_track = WebRTCManager._video_track
            pc.addTrack(video_track)

            # Create answer
            answer = await pc.createAnswer()

            # Wait for ICE gathering to complete or timeout
            # Register handler BEFORE setLocalDescription to avoid race condition
            gathering_complete = asyncio.Event()

            @pc.on("icegatheringstatechange")
            async def on_gathering_change():
                if pc.iceGatheringState == "complete":
                    gathering_complete.set()

            # Set local description (triggers ICE gathering)
            await pc.setLocalDescription(answer)
            logger.info(f"ICE gathering state: {pc.iceGatheringState}")

            # Check if ICE gathering already completed
            if pc.iceGatheringState == "complete":
                gathering_complete.set()

            # Wait up to 10 seconds for ICE gathering
            try:
                await asyncio.wait_for(gathering_complete.wait(), timeout=10.0)
                logger.info("ICE gathering completed")
            except asyncio.TimeoutError:
                logger.warning("ICE gathering timeout after 10s, proceeding with current candidates")
            
            # Get the final SDP with candidates
            final_sdp = pc.localDescription.sdp

            # OFFLINE MODE FIX: Force localhost ONLY for invalid/link-local IPs
            if "c=IN IP4 0.0.0.0" in final_sdp or "c=IN IP4 169.254." in final_sdp:
                logger.info("Replacing invalid/link-local IPs with 127.0.0.1 in SDP")

                # Replace ONLY invalid/link-local IPs with localhost in connection lines
                import re
                final_sdp = re.sub(r'c=IN IP4 0\.0\.0\.0', 'c=IN IP4 127.0.0.1', final_sdp)
                final_sdp = re.sub(r'c=IN IP4 169\.254\.\d+\.\d+', 'c=IN IP4 127.0.0.1', final_sdp)

                # Also fix ICE candidates for invalid/link-local IPs
                final_sdp = re.sub(
                    r'(a=candidate:[^\s]+\s+\d+\s+\w+\s+\d+\s+)0\.0\.0\.0',
                    r'\g<1>127.0.0.1',
                    final_sdp
                )
                final_sdp = re.sub(
                    r'(a=candidate:[^\s]+\s+\d+\s+\w+\s+\d+\s+)169\.254\.\d+\.\d+',
                    r'\g<1>127.0.0.1',
                    final_sdp
                )

            logger.info(f"WebRTC answer created for {session_id}")

            return {
                "type": "answer",
                "sdp": final_sdp,
                "session_id": session_id
            }
        except Exception as e:
            logger.error(f"Error in handle_offer: {str(e)}", exc_info=True)
            raise
    
    async def handle_ice_candidate(self, session_id: str, candidate: Dict):
        """Handle ICE candidate."""
        pc = self.peer_connections.get(session_id)
        if not pc:
            raise ValueError(f"No peer connection for session {session_id}")
        
        # In aiortc, ICE candidates are handled automatically via the SDP exchange
        # We don't need to manually add them like in browser WebRTC
        # Just log that we received it
        logger.info(f"Received ICE candidate for session {session_id} (handled automatically by aiortc)")
    
    def get_video_track(self, session_id: str = None):
        """Get the video track for a session (or the singleton track)."""
        return WebRTCManager._video_track

    def is_data_channel_ready(self, session_id: str) -> bool:
        """Check if data channel is ready for communication."""
        channel = self.data_channels.get(session_id)
        if not channel:
            return False
        return channel.readyState == "open"
    
    async def wait_for_data_channel(self, session_id: str, timeout: int = 5000) -> bool:
        """Wait for data channel to be ready."""
        start_time = asyncio.get_event_loop().time()
        timeout_seconds = timeout / 1000.0
        
        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            if self.is_data_channel_ready(session_id):
                logger.info(f"Data channel ready for session {session_id}")
                return True
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Data channel timeout for session {session_id} after {timeout}ms")
        return False
    
    async def send_data_channel_message(self, session_id: str, message: Dict):
        """Send message via data channel."""
        # Use session_id directly, not with _results suffix
        channel = self.data_channels.get(session_id)

        if not channel:
            logger.warning(f"No data channel found for session {session_id}")
            return False
        
        if channel.readyState != "open":
            logger.warning(f"Data channel not open for session {session_id}, state: {channel.readyState}")
            # Try to wait briefly for it to open
            if channel.readyState == "connecting":
                await self.wait_for_data_channel(session_id, timeout=1000)
                # Re-check after wait
                if channel.readyState != "open":
                    return False
            else:
                return False
        
        try:
            msg_str = json.dumps(message)
            channel.send(msg_str)
            return True
        except Exception as e:
            logger.error(f"Error sending data channel message: {e}")
            return False
    
    def set_control_message_handler(self, handler):
        """Set the handler for control messages from data channel."""
        self.control_message_handler = handler
    
    async def handle_data_channel_message(self, session_id: str, message: str):
        """Handle incoming data channel message."""
        try:
            data = json.loads(message)
            logger.info(f"Received data channel message for {session_id}: {data.get('type')}")
            
            msg_type = data.get("type")
            
            # Handle different message types
            if msg_type == "frame_request":
                # Client requesting frame analysis
                pass
            elif msg_type == "client_log":
                # Frontend log message
                logger.info(f"Client log [{session_id}]: {data.get('message')}")
            elif msg_type in ["stop_session", "pause_flow", "resume_flow"]:
                # Forward control messages to websocket handler and send response back
                logger.info(f"Forwarding control message {msg_type} to websocket handler")
                if self.control_message_handler:
                    response = await self.control_message_handler(session_id, data)
                    # Send response back through data channel
                    if response:
                        await self.send_data_channel_message(session_id, response)
                else:
                    logger.warning(f"No control message handler set for {msg_type}")
            elif msg_type in ["manual_previous", "manual_next", "manual_redo",
                            "manual_select_agent", "manual_confirm_final", "tap_to_focus"]:
                # Forward manual mode commands and tap-to-focus to websocket handler and send response back
                logger.info(f"Forwarding command {msg_type} to websocket handler")
                if self.control_message_handler:
                    # Add session_id to the message if not present
                    data["session_id"] = session_id
                    response = await self.control_message_handler(session_id, data)
                    # Send response back through data channel
                    if response:
                        await self.send_data_channel_message(session_id, response)
                else:
                    logger.warning(f"No control message handler set for manual command {msg_type}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in data channel message: {message}")
    
    async def send_progressive_update(
        self, 
        session_id: str,
        agent: str,
        attributes: Dict,
        confidence: float = 0.0,
        frames_processed: int = 0
    ):
        """Send progressive update via data channel."""
        message = {
            "type": "progressive_update",
            "session_id": session_id,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes,
            "confidence": confidence,
            "frames_processed": frames_processed
        }
        
        return await self.send_data_channel_message(session_id, message)
    
    async def send_status_update(
        self,
        session_id: str,
        current_agent: Optional[str],
        progress: int,
        fps: float = 0.0,
        timer_remaining: float = 0.0
    ):
        """Send status update via data channel."""
        message = {
            "type": "status_update",
            "session_id": session_id,
            "current_agent": current_agent,
            "progress": progress,
            "timer_remaining": timer_remaining,
            "fps": fps,
            "gpu_usage": 0,  # Will be updated from actual monitoring
            "connection_quality": "excellent"
        }
        
        return await self.send_data_channel_message(session_id, message)
    
    async def send_agent_started(
        self,
        session_id: str,
        agent: str,
        mode: str,
        timer_seconds: Optional[float] = None
    ):
        """Send agent started notification."""
        message = {
            "type": "agent_started",
            "session_id": session_id,
            "agent": agent,
            "mode": mode
        }
        
        if timer_seconds:
            message["timer_seconds"] = timer_seconds
        
        return await self.send_data_channel_message(session_id, message)
    
    async def send_agent_completed(
        self,
        session_id: str,
        agent: str,
        results: Dict
    ):
        """Send agent completed notification."""
        message = {
            "type": "agent_completed",
            "session_id": session_id,
            "agent": agent,
            "results": results.get("attributes", {}),
            "reasoning": results.get("reasoning", ""),
            "frames_analyzed": results.get("frames_processed", 0)
        }
        
        return await self.send_data_channel_message(session_id, message)
    
    async def send_final_results(
        self,
        session_id: str,
        classification: Dict,
        processing_time: float,
        total_frames: int
    ):
        """Send final classification results."""
        message = {
            "type": "final_results",
            "session_id": session_id,
            "classification": classification,
            "processing_time": processing_time,
            "total_frames": total_frames,
            "confidence_overall": 0.95  # Calculate from actual results
        }
        
        return await self.send_data_channel_message(session_id, message)
    
    async def send_error(
        self,
        session_id: str,
        error_code: str,
        message: str,
        recoverable: bool = True
    ):
        """Send error message."""
        error_message = {
            "type": "error",
            "session_id": session_id,
            "error_code": error_code,
            "message": message,
            "recoverable": recoverable,
            "suggested_action": "reconnect" if recoverable else None
        }
        
        return await self.send_data_channel_message(session_id, error_message)
    
    async def send_manual_mode_status(
        self,
        session_id: str,
        status: Dict
    ):
        """Send manual mode status update."""
        message = {
            "type": "manual_mode_status",
            "session_id": session_id,
            "current_agent": status.get("current_agent"),
            "agent_status": status.get("agent_status"),
            "waiting_for_command": status.get("waiting_for_command", True),
            "completed_agents": status.get("completed_agents", []),
            "completed_agents_set": status.get("completed_agents_set", []),  # Add this field
            "pending_agents": status.get("pending_agents", []),
            "can_compile": status.get("can_compile", False),
            "node_clicking_enabled": status.get("node_clicking_enabled", False),  # Add this field
            "cycle_number": status.get("cycle_number", 1),
            "manual_mode_active": status.get("manual_mode_active", False)
        }
        
        return await self.send_data_channel_message(session_id, message)
    
    async def on_connection_established(self, session_id: str):
        """Handle successful connection establishment."""
        # Update camera feed status
        if session_id in self.camera_feeds:
            self.camera_feeds[session_id].connect(session_id)
        
        logger.info(f"WebRTC connection established for {session_id}")
    
    async def on_connection_failed(self, session_id: str):
        """Handle connection failure."""
        # Update camera feed status
        if session_id in self.camera_feeds:
            self.camera_feeds[session_id].connection_status = ConnectionStatus.ERROR
            self.camera_feeds[session_id].record_error("Connection failed")
        
        # Send error notification
        await self.send_error(
            session_id,
            "WEBRTC_FAILED",
            "WebRTC connection failed",
            recoverable=True
        )
        
        logger.error(f"WebRTC connection failed for {session_id}")
    
    async def close_connection(self, session_id: str):
        """Close a peer connection and clean up all associated resources."""
        pc = self.peer_connections.get(session_id)
        if pc:
            await pc.close()
            del self.peer_connections[session_id]

        # Clean up data channels
        for key in list(self.data_channels.keys()):
            if session_id in key:
                del self.data_channels[key]

        # Update camera feed
        if session_id in self.camera_feeds:
            self.camera_feeds[session_id].disconnect()
            del self.camera_feeds[session_id]

        # Clean up frame buffers to free memory
        if session_id in self.frame_buffers:
            del self.frame_buffers[session_id]

        # Clean up session timestamp
        if session_id in self.session_timestamps:
            del self.session_timestamps[session_id]

        # CRITICAL: Stop and reset video track if no more active connections
        # This releases the camera resource when the browser disconnects
        if len(self.peer_connections) == 0:
            async with WebRTCManager._camera_service_lock:
                if WebRTCManager._video_track:
                    logger.info("Camera release: No active connections, stopping video track")
                    try:
                        if hasattr(WebRTCManager._video_track, 'stop'):
                            WebRTCManager._video_track.stop()
                        WebRTCManager._video_track = None
                        logger.info("Camera resource released - ready for reconnection")
                    except Exception as e:
                        logger.error(f"Error stopping video track: {e}")

        logger.info(f"Closed connection and cleaned resources for session {session_id}")
    
    async def cleanup_stale_connections(self):
        """Clean up only failed or closed connections - keeps active sessions alive."""
        stale_sessions = []

        # Only check for failed/closed connections - NOT based on age
        for session_id, pc in self.peer_connections.items():
            if pc.connectionState in ["failed", "closed"]:
                stale_sessions.append(session_id)
                logger.info(f"Session {session_id} marked for cleanup: connection {pc.connectionState}")

        # Clean up all stale sessions
        for session_id in stale_sessions:
            await self.close_connection(session_id)

        if stale_sessions:
            logger.info(f"Cleaned up {len(stale_sessions)} dead sessions. "
                       f"Active sessions: {len(self.peer_connections)}")

        return len(stale_sessions)
    
    def add_frame_to_buffer(self, session_id: str, frame):
        """Add a frame to the session's buffer with automatic FIFO eviction.

        Args:
            session_id: Session identifier
            frame: Video frame (numpy array)
        """
        import numpy as np

        if session_id not in self.frame_buffers:
            # Initialize deque with maxlen for automatic eviction
            self.frame_buffers[session_id] = deque(maxlen=self.max_buffer_size)

        # Frame is already independent copy from camera hardware (cv60_camera.py:304)
        # No need to copy again - just store reference
        self.frame_buffers[session_id].append(frame)
    
    def get_latest_frame(self, session_id: str):
        """Get the latest frame for a session.

        Args:
            session_id: Session identifier

        Returns:
            Latest frame from buffer (already independent copy) or None if no frames available
        """
        import numpy as np

        if session_id in self.frame_buffers and self.frame_buffers[session_id]:
            # Buffer stores independent frames from camera hardware
            # No additional copy needed (agents read-only, no concurrent modification)
            return self.frame_buffers[session_id][-1]
        return None
    
    def get_frame_buffer(self, session_id: str, count: int = 5) -> list:
        """Get the latest frames from the buffer.

        Args:
            session_id: Session identifier
            count: Number of frames to return

        Returns:
            List of frames from buffer (already independent copies, up to count)
        """
        import numpy as np

        if session_id in self.frame_buffers:
            buffer = self.frame_buffers[session_id]
            # Return frames from buffer (already independent from hardware)
            # No additional copies needed (agents read-only, no concurrent modification)
            return list(buffer[-count:] if len(buffer) > count else buffer)
        return []
    
    def clear_frame_buffer(self, session_id: str):
        """Clear the frame buffer for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.frame_buffers:
            del self.frame_buffers[session_id]
    
    def get_frame_provider(self, session_id: str):
        """Get a frame provider function for agents.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Callable that returns latest frame(s) - always copies
        """
        import numpy as np
        
        async def provider():
            # Optimized: Don't maintain old frames, just get fresh ones
            # Clear buffer to prevent stale frames
            if session_id in self.frame_buffers:
                self.frame_buffers[session_id].clear()

            # Try to get fresh frame from video track
            if hasattr(WebRTCManager, '_video_track'):
                video_track = WebRTCManager._video_track
                if video_track:
                    try:
                        # Call recv() to get a fresh frame from camera
                        av_frame = await video_track.recv()
                        if av_frame:
                            # Convert to numpy array (creates fresh memory automatically)
                            # Use BGR format to match OpenCV/aiortc convention
                            frame_data = av_frame.to_ndarray(format="bgr24")
                            # Add to buffer for potential reuse
                            self.add_frame_to_buffer(session_id, frame_data)
                            return frame_data
                    except Exception:
                        pass

            # Fallback to buffer if video track not available
            frame = self.get_latest_frame(session_id)
            return frame  # Frame from buffer (already independent from hardware)

        return provider

    async def start_periodic_cleanup(self):
        """Start background task for periodic cleanup of stale connections."""
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._periodic_cleanup_loop())
            logger.info(f"Started periodic cleanup task (interval: {self.cleanup_interval_seconds}s)")

    async def stop_periodic_cleanup(self):
        """Stop the periodic cleanup task."""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped periodic cleanup task")

    async def _periodic_cleanup_loop(self):
        """Background loop that runs cleanup periodically."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_seconds)
                cleaned = await self.cleanup_stale_connections()
                if cleaned > 0:
                    logger.info(f"Periodic cleanup removed {cleaned} stale sessions")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")