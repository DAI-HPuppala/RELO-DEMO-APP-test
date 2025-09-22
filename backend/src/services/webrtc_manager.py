"""WebRTC connection and stream management service."""
import asyncio
import json
import logging
import socket
from typing import Dict, Optional, Any
from datetime import datetime

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
        self.ice_servers = [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
        # Frame buffer for each session - stores latest frames
        self.frame_buffers: Dict[str, list] = {}
        self.max_buffer_size = 10  # Keep moderate buffer to avoid starvation
        # Callback for handling control messages from data channel
        self.control_message_handler = None
    
    async def create_peer_connection(self, session_id: str) -> RTCPeerConnection:
        """Create a new peer connection for a session."""
        # Create simple configuration for local network
        # Don't specify configuration to use defaults which work better locally
        pc = RTCPeerConnection()
        
        # Store the connection
        self.peer_connections[session_id] = pc
        
        # Store WebSocket for sending ICE candidates
        self.websocket_for_session = getattr(self, 'websocket_for_session', {})
        
        # Set up event handlers
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state for {session_id}: {pc.connectionState}")
            
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
        try:
            pc = await self.create_peer_connection(session_id)

            # Initialize camera service and video track (singleton pattern with reset logic)
            async with WebRTCManager._camera_service_lock:
                if not WebRTCManager._camera_service:
                    try:
                        # Use the CV60 camera service
                        from services.cv60_camera import CV60CameraService
                    except ImportError:
                        # Try alternative import path
                        from .cv60_camera import CV60CameraService
                    WebRTCManager._camera_service = CV60CameraService()
                    WebRTCManager._camera_service.initialize_camera(camera_source)
                    logger.info("CV60CameraService initialized")

                # Check if existing video track is in test mode and needs reset
                if WebRTCManager._video_track:
                    # Check if the video track is in test mode (not properly initialized)
                    if hasattr(WebRTCManager._video_track, 'is_initialized') and not WebRTCManager._video_track.is_initialized:
                        logger.warning("🔄 CAMERA RESET: Existing video track is in test mode")
                        logger.info(f"   Previous session: {getattr(WebRTCManager._video_track, 'current_session_id', 'unknown')}")
                        logger.info(f"   New session: {session_id}")
                        # Stop the old track
                        if hasattr(WebRTCManager._video_track, 'stop'):
                            WebRTCManager._video_track.stop()
                        WebRTCManager._video_track = None
                        logger.info("✅ Video track reset - will attempt to reconnect to camera")
                    # Also reset if the track has been running for too long (stale)
                    elif hasattr(WebRTCManager._video_track, '_creation_time'):
                        import time
                        track_age = time.time() - WebRTCManager._video_track._creation_time
                        if track_age > 3600:  # Reset if track is older than 1 hour
                            logger.warning(f"🔄 CAMERA RESET: Video track is stale ({track_age:.0f}s old)")
                            if hasattr(WebRTCManager._video_track, 'stop'):
                                WebRTCManager._video_track.stop()
                            WebRTCManager._video_track = None
                            logger.info("✅ Stale video track cleared")

                # Create video track with session ID for frame buffer
                if not WebRTCManager._video_track:
                    WebRTCManager._video_track = WebRTCManager._camera_service.get_video_track(session_id)
                    logger.info(f"New video track created for session {session_id}")
            
            # First set the remote description to understand what the client wants
            remote_sdp = RTCSessionDescription(
                sdp=offer["sdp"],
                type=offer.get("type", "offer")
            )
            await pc.setRemoteDescription(remote_sdp)
            
            # Add the video track directly without relay for simplicity
            # The relay can cause issues if not properly configured
            video_track = WebRTCManager._video_track
            
            # Add video track to peer connection
            # Use addTrack instead of manipulating transceivers
            pc.addTrack(video_track)
            logger.info(f"Added video track to peer connection")
            
            # Create answer after adding tracks
            answer = await pc.createAnswer()
            
            # Set local description
            await pc.setLocalDescription(answer)
            
            # Wait for ICE gathering to complete or timeout
            gathering_complete = asyncio.Event()
            
            @pc.on("icegatheringstatechange")
            async def on_gathering_change():
                if pc.iceGatheringState == "complete":
                    gathering_complete.set()
            
            # Wait up to 2 seconds for ICE gathering
            try:
                await asyncio.wait_for(gathering_complete.wait(), timeout=2.0)
                logger.info("ICE gathering completed")
            except asyncio.TimeoutError:
                logger.warning("ICE gathering timeout, proceeding with current candidates")
            
            # Get the final SDP with candidates
            final_sdp = pc.localDescription.sdp
            
            logger.info(f"Created answer for session {session_id}")
            logger.info(f"Answer SDP (first 500 chars): {final_sdp[:500]}")
            
            # Check if we have valid IP in SDP
            if "c=IN IP4 0.0.0.0" in final_sdp:
                logger.warning("SDP contains 0.0.0.0, this may cause connection issues")
                # Try to get local IP and update SDP if needed
                import socket
                try:
                    # Get local IP address
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    s.close()
                    logger.info(f"Local IP address: {local_ip}")
                    # Update SDP with actual IP
                    final_sdp = final_sdp.replace("c=IN IP4 0.0.0.0", f"c=IN IP4 {local_ip}")
                except Exception as e:
                    logger.error(f"Could not determine local IP: {e}")
            
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
            
            # Check if channel exists but not open yet
            channel = self.data_channels.get(session_id)
            if channel:
                logger.debug(f"Data channel state for {session_id}: {channel.readyState}")
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"Data channel timeout for session {session_id} after {timeout}ms")
        return False
    
    async def send_data_channel_message(self, session_id: str, message: Dict):
        """Send message via data channel."""
        # Use session_id directly, not with _results suffix
        channel = self.data_channels.get(session_id)
        
        if not channel:
            logger.warning(f"No data channel found for session {session_id}")
            logger.debug(f"Available channels: {list(self.data_channels.keys())}")
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
            logger.debug(f"Sent data channel message: {message.get('type')} for {message.get('agent', 'N/A')}")
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
                # Forward control messages to websocket handler
                logger.info(f"Forwarding control message {msg_type} to websocket handler")
                if self.control_message_handler:
                    await self.control_message_handler(session_id, data)
                else:
                    logger.warning(f"No control message handler set for {msg_type}")
            elif msg_type in ["manual_previous", "manual_next", "manual_redo", 
                            "manual_select_agent", "manual_confirm_final"]:
                # Forward manual mode commands to websocket handler
                logger.info(f"Forwarding manual command {msg_type} to websocket handler")
                if self.control_message_handler:
                    # Add session_id to the message if not present
                    data["session_id"] = session_id
                    await self.control_message_handler(session_id, data)
                else:
                    logger.warning(f"No control message handler set for manual command {msg_type}")
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
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
        """Close a peer connection."""
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
        
        logger.info(f"Closed connection for session {session_id}")
    
    async def cleanup_stale_connections(self):
        """Clean up stale connections."""
        stale_sessions = []
        
        for session_id, pc in self.peer_connections.items():
            if pc.connectionState in ["failed", "closed"]:
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            await self.close_connection(session_id)
        
        return len(stale_sessions)
    
    def add_frame_to_buffer(self, session_id: str, frame):
        """Add a frame to the session's buffer.
        
        Args:
            session_id: Session identifier
            frame: Video frame (numpy array)
        """
        import numpy as np
        
        if session_id not in self.frame_buffers:
            self.frame_buffers[session_id] = []
        
        buffer = self.frame_buffers[session_id]
        # Create a copy of the frame to prevent interference with live stream
        frame_copy = np.copy(frame) if isinstance(frame, np.ndarray) else frame
        buffer.append(frame_copy)
        
        # Keep only the latest frames
        if len(buffer) > self.max_buffer_size:
            self.frame_buffers[session_id] = buffer[-self.max_buffer_size:]
    
    def get_latest_frame(self, session_id: str):
        """Get the latest frame for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Copy of latest frame or None if no frames available
        """
        import numpy as np
        
        if session_id in self.frame_buffers and self.frame_buffers[session_id]:
            frame = self.frame_buffers[session_id][-1]
            # Return a copy to prevent modification
            return np.copy(frame) if isinstance(frame, np.ndarray) else frame
        return None
    
    def get_frame_buffer(self, session_id: str, count: int = 5) -> list:
        """Get the latest frames from the buffer.
        
        Args:
            session_id: Session identifier
            count: Number of frames to return
            
        Returns:
            List of frame copies (up to count)
        """
        import numpy as np
        
        if session_id in self.frame_buffers:
            buffer = self.frame_buffers[session_id]
            frames = buffer[-count:] if len(buffer) > count else buffer
            # Return copies of frames to prevent modification
            return [np.copy(f) if isinstance(f, np.ndarray) else f for f in frames]
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
            # First, clear old frames from buffer to ensure freshness
            if session_id in self.frame_buffers:
                buffer = self.frame_buffers[session_id]
                # Keep only the last 2 frames to ensure freshness
                if len(buffer) > 2:
                    self.frame_buffers[session_id] = buffer[-2:]

            # Try to get fresh frame from video track
            if hasattr(WebRTCManager, '_video_track'):
                video_track = WebRTCManager._video_track
                if video_track:
                    try:
                        # Call recv() to get a fresh frame from camera
                        av_frame = await video_track.recv()
                        if av_frame:
                            # Convert to numpy array
                            # Use BGR format to match OpenCV/aiortc convention
                            frame_data = av_frame.to_ndarray(format="bgr24")
                            # Add to buffer
                            self.add_frame_to_buffer(session_id, frame_data)
                            # Return a copy
                            return np.copy(frame_data)
                    except Exception as e:
                        logger.debug(f"Could not get fresh frame from video track: {e}")

            # Fallback to buffer if video track not available
            frame = self.get_latest_frame(session_id)
            return frame  # Already a copy from get_latest_frame
        
        return provider