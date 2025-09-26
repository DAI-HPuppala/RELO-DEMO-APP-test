"""WebSocket Handler V2 with stateful orchestration support"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from uuid import uuid4
from fastapi import WebSocket, WebSocketDisconnect
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay

from v2.services.stateful_orchestrator import StatefulOrchestrator
from v2.services.session_state_manager import SessionStateManager
from v2.agents.initial_classifier_v2 import InitialClassifierV2
from v2.agents.detail_extractor_v2 import DetailExtractorV2
from v2.agents.damage_detector_v2 import DamageDetectorV2
from v2.agents.final_compiler_v2 import FinalCompilerV2
from services.vlm_service_ultra import UltraVLMService as VLMService

logger = logging.getLogger(__name__)


class WebSocketHandlerV2:
    """Enhanced WebSocket handler for v2 protocol with pause/resume"""
    
    def __init__(self):
        self.sessions = {}  # session_id -> orchestrator
        self.connections = {}  # session_id -> websocket
        self.peer_connections = {}  # session_id -> RTCPeerConnection
        self.session_manager = SessionStateManager()
        self.media_relay = MediaRelay()
        
        # Initialize VLM service
        self.vlm_service = VLMService()
        
        logger.info("WebSocketHandlerV2 initialized with VLM service")
    
    async def handle_message(self, websocket, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route incoming WebSocket messages"""
        msg_type = message.get("type")
        session_id = message.get("session_id")
        
        logger.debug(f"Handling message type: {msg_type} for session: {session_id}")
        
        # Route to appropriate handler
        handlers = {
            "offer": self.handle_offer,
            "start_monitoring": self.handle_start_monitoring,
            "pause_flow": self.handle_pause_flow,
            "resume_flow": self.handle_resume_flow,
            "get_session_state": self.handle_get_session_state,
            "set_custom_timers": self.handle_set_custom_timers,
            "recover_session": self.handle_recover_session,
            "get_checkpoint_info": self.handle_get_checkpoint_info,
            "get_performance_metrics": self.handle_get_performance_metrics
        }
        
        handler = handlers.get(msg_type)
        if handler:
            return await handler(message, websocket)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return {"type": "error", "message": f"Unknown message type: {msg_type}"}
    
    async def handle_offer(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Handle WebRTC offer and setup peer connection"""
        session_id = message.get("session_id", str(uuid4()))
        sdp = message.get("sdp")
        
        # Create peer connection
        pc = RTCPeerConnection()
        self.peer_connections[session_id] = pc
        self.connections[session_id] = websocket
        
        # Set remote description
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        
        # Setup video track handler
        @pc.on("track")
        def on_track(track):
            logger.info(f"Track received: {track.kind}")
            if track.kind == "video":
                # Create orchestrator for this session
                asyncio.create_task(self._setup_orchestrator(session_id, track))
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "type": "answer",
            "sdp": pc.localDescription.sdp,
            "session_id": session_id
        }
    
    async def _setup_orchestrator(self, session_id: str, video_track):
        """Setup stateful orchestrator for session"""
        # Create session
        session = await self.session_manager.create_session(session_id)
        
        # Create orchestrator
        orchestrator = StatefulOrchestrator(session_id)
        orchestrator.session_state = session
        
        # Set frame provider from video track
        async def frame_provider():
            try:
                frame = await video_track.recv()
                return frame.to_ndarray(format="bgr24")
            except:
                return None
        
        orchestrator.frame_provider = frame_provider
        
        # Set message callback
        orchestrator.send_message = lambda msg: self._send_to_client(session_id, msg)
        
        # Setup VLM service integration
        await self._setup_vlm_integration(orchestrator)
        
        # Validate setup and get comprehensive status
        validation = await orchestrator.validate_setup()
        performance_metrics = await orchestrator.get_performance_metrics()
        
        self.sessions[session_id] = orchestrator
        
        # Send enhanced validation results to frontend
        await self._send_to_client(session_id, {
            "type": "orchestrator_ready",
            "session_id": session_id,
            "validation": validation,
            "performance_metrics": performance_metrics
        })
        
        # Log setup completion with optimization details
        gpu_status = validation.get('gpu_optimization', {}).get('status', 'unknown')
        logger.info(f"Orchestrator setup complete for session {session_id}")
        logger.info(f"  - Status: {validation['status']}")
        logger.info(f"  - GPU optimization: {gpu_status}")
        logger.info(f"  - VLM configured: {validation['vlm_configured']}")
    
    async def handle_start_monitoring(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Start automatic monitoring flow"""
        session_id = message.get("session_id")
        mode = message.get("mode", "automatic")
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        
        # Start monitoring in background
        asyncio.create_task(self._run_monitoring_flow(session_id, orchestrator))
        
        return {
            "type": "monitoring_started",
            "session_id": session_id,
            "mode": mode
        }
    
    async def _run_monitoring_flow(self, session_id: str, orchestrator: StatefulOrchestrator):
        """Run the complete monitoring flow"""
        try:
            # Check if resuming from checkpoint
            if await orchestrator.load_from_checkpoint():
                logger.info(f"Resuming session {session_id} from checkpoint")
                await self._send_to_client(session_id, {
                    "type": "session_resumed",
                    "session_id": session_id,
                    "checkpoint_info": await orchestrator.get_checkpoint_info()
                })
            
            # Initialize agents
            agents = {
                "initial_classifier": InitialClassifierV2(),
                "detail_extractor": DetailExtractorV2(),
                "damage_detector": DamageDetectorV2()
            }
            
            # Wire frame management and callbacks for all agents
            for agent_name, agent in agents.items():
                # Set frame management
                agent.set_frame_registry(orchestrator.frame_registry)
                agent.set_frame_provider(orchestrator.frame_provider)
                agent.set_inference_engine(orchestrator.inference_engine)
                
                # Set callbacks with proper closure
                def make_update_callback(name):
                    async def callback(update):
                        await self._send_inference_update(session_id, name, update)
                    return callback
                
                agent.on_inference_update = make_update_callback(agent_name)
            
            results = {}
            
            # Run each agent in sequence
            for agent_name in ["initial_classifier", "detail_extractor", "damage_detector"]:
                if orchestrator.session_state.status == "PAUSED":
                    logger.info(f"Flow paused at {agent_name}")
                    break
                
                agent = agents[agent_name]
                
                # Send agent_started with timer
                await self._send_to_client(session_id, {
                    "type": "agent_started",
                    "session_id": session_id,
                    "agent": agent_name,
                    "timer_seconds": agent.timer_seconds,
                    "mode": "automatic",
                    "sequence_position": orchestrator.session_state.agents_sequence.index(agent_name) + 1,
                    "total_agents": len(orchestrator.session_state.agents_sequence)
                })
                
                # Update orchestrator state
                orchestrator.session_state.current_agent = agent_name
                
                # Run agent with timer through orchestrator for proper state tracking
                try:
                    result = await agent.run_with_timer()
                    results[agent_name] = result
                except Exception as agent_error:
                    logger.error(f"Agent {agent_name} failed: {agent_error}")
                    await self._send_to_client(session_id, {
                        "type": "agent_error",
                        "session_id": session_id,
                        "agent": agent_name,
                        "error": str(agent_error),
                        "action": "continuing_to_next"
                    })
                    # Continue to next agent
                    results[agent_name] = {"error": str(agent_error), "attributes": {}}
                
                # Send agent_completed
                await self._send_to_client(session_id, {
                    "type": "agent_completed",
                    "session_id": session_id,
                    "agent": agent_name,
                    "result": result
                })
                
                # Update session state
                orchestrator.session_state.advance_to_next_agent()
            
            # Run final compiler
            if orchestrator.session_state.status != "PAUSED":
                final_compiler = FinalCompilerV2()
                final_result = await final_compiler.process(results)

                await self._send_to_client(session_id, {
                    "type": "classification_complete",
                    "session_id": session_id,
                    "final_result": final_result
                })
                
                orchestrator.session_state.complete()
            
        except Exception as e:
            logger.error(f"Monitoring flow error: {e}")
            await self._send_to_client(session_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def handle_pause_flow(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Handle pause flow request"""
        session_id = message.get("session_id")
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        response = await orchestrator.pause_flow()
        
        # Send to client
        await self._send_to_client(session_id, response)
        
        return response
    
    async def handle_resume_flow(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Handle resume flow request"""
        session_id = message.get("session_id")
        restart_agent = message.get("restart_agent", False)
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        response = await orchestrator.resume_flow(restart_agent)
        
        # Send to client
        await self._send_to_client(session_id, response)
        
        # Resume monitoring flow
        asyncio.create_task(self._run_monitoring_flow(session_id, orchestrator))
        
        return response
    
    async def handle_get_session_state(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Get current session state"""
        session_id = message.get("session_id")
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        session = orchestrator.session_state
        
        return {
            "type": "session_state",
            "session_id": session_id,
            "status": session.status.value,
            "current_agent": session.current_agent,
            "agents_completed": session.agents_completed,
            "paused_agent": session.paused_agent
        }
    
    async def handle_set_custom_timers(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Set custom timers for testing"""
        session_id = message.get("session_id")
        timers = message.get("timers", {})
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        orchestrator.set_custom_timers(timers)
        
        logger.info(f"Custom timers set for session {session_id}: {timers}")
        
        return {
            "type": "custom_timers_set",
            "session_id": session_id,
            "timers": timers
        }
    
    async def _send_inference_update(self, session_id: str, agent_name: str, update: Dict[str, Any]):
        """Send inference update to client with validation"""
        # Apply damage override for progressive updates
        if agent_name == "damage_detector" and "attributes" in update:
            update = update.copy()  # Don't modify original
            update["attributes"] = self._validate_damage_attributes(update["attributes"])

        message = {
            "type": "inference_update",
            "session_id": session_id,
            "agent": agent_name,
            **update
        }
        await self._send_to_client(session_id, message)
    
    async def _send_to_client(self, session_id: str, message: Dict[str, Any]):
        """Send message to client via WebSocket"""
        if session_id in self.connections:
            websocket = self.connections[session_id]
            try:
                await websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
    
    def get_agent_timer(self, agent_name: str) -> float:
        """Get default timer for agent"""
        timers = {
            "initial_classifier": 4.0,
            "detail_extractor": 3.0,
            "damage_detector": 4.0,
            "final_compiler": 1.0
        }
        return timers.get(agent_name, 4.0)

    def _validate_damage_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Validate damage attributes for progressive updates"""
        if not attributes:
            return attributes

        is_damaged = attributes.get("is_damaged")
        damage_type = attributes.get("damage_type")

        # Override logic: if is_damaged="yes" but damage_type indicates no damage
        if self._is_positive_damage(is_damaged) and self._is_no_damage_type(damage_type):
            logger.info(f"WebSocketV2: Progressive update override: is_damaged from '{is_damaged}' to 'No' "
                       f"due to damage_type: '{damage_type}'")
            attributes = attributes.copy()  # Don't modify original
            attributes["is_damaged"] = "No"
            # Also clear damage-related fields for consistency
            attributes["damage_severity"] = None
            attributes["damage_location"] = None
            attributes["repair_feasibility"] = None

        return attributes

    def _is_positive_damage(self, value) -> bool:
        """Check if value indicates damage"""
        if not value:
            return False
        return str(value).lower() in ['yes', 'true', '1', 'damaged']

    def _is_no_damage_type(self, value) -> bool:
        """Check if damage_type indicates no damage"""
        if not value:
            return True
        value_lower = str(value).lower()
        return (value_lower in ['null', 'undefined', 'none', 'no', 'no damage', 'clean'] or
                'no' in value_lower or 'none' in value_lower)
    
    async def handle_recover_session(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Handle session recovery from checkpoint"""
        session_id = message.get("session_id")
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        
        # Try to load from checkpoint
        recovered = await orchestrator.load_from_checkpoint()
        
        if recovered:
            logger.info(f"Session {session_id} recovered from checkpoint")
            return {
                "type": "session_recovered",
                "session_id": session_id,
                "checkpoint_info": await orchestrator.get_checkpoint_info(),
                "session_state": orchestrator.session_state.to_dict() if hasattr(orchestrator.session_state, 'to_dict') else {}
            }
        else:
            return {
                "type": "no_checkpoint",
                "session_id": session_id,
                "message": "No checkpoint found for this session"
            }
    
    async def handle_get_checkpoint_info(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Get checkpoint information for a session"""
        session_id = message.get("session_id")
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        info = await orchestrator.get_checkpoint_info()
        
        return {
            "type": "checkpoint_info",
            "session_id": session_id,
            **info
        }
    
    async def _setup_vlm_integration(self, orchestrator: StatefulOrchestrator):
        """Setup VLM service integration with GPU optimization for the orchestrator"""
        try:
            logger.info("🚀 Setting up VLM integration with GPU optimization...")
            
            # Initialize GPU optimization first
            gpu_init_success = await orchestrator.initialize_gpu_optimization(
                progress_callback=lambda msg, pct: asyncio.create_task(
                    self._send_gpu_optimization_progress(orchestrator.session_id, msg, pct)
                )
            )
            
            if gpu_init_success:
                logger.info("✅ GPU optimization initialized successfully")
            else:
                logger.warning("⚠️ GPU optimization failed, falling back to legacy VLM")
                
                # Fallback to legacy VLM service
                vlm_healthy = await self.vlm_service.health_check()
                
                if vlm_healthy:
                    # Legacy VLM fallback not supported with MultiInferenceEngine
                    # The MultiInferenceEngine has its own internal vlm_engine
                    logger.info("⚠️ Legacy VLM service available but not integrated (GPU-only mode)")
                else:
                    logger.error("❌ Both GPU optimization and legacy VLM failed")
                    
        except Exception as e:
            logger.error(f"Failed to setup VLM integration: {e}")
            # Try legacy fallback in case of error
            try:
                vlm_healthy = await self.vlm_service.health_check()
                if vlm_healthy:
                    # Legacy VLM fallback not supported with MultiInferenceEngine
                    # The MultiInferenceEngine has its own internal vlm_engine
                    logger.info("⚠️ Legacy VLM service available but not integrated (GPU-only mode)")
            except Exception as fallback_error:
                logger.error(f"Legacy VLM fallback check failed: {fallback_error}")
    
    async def _send_gpu_optimization_progress(self, session_id: str, message: str, percentage: int):
        """Send GPU optimization progress to client"""
        await self._send_to_client(session_id, {
            "type": "gpu_optimization_progress",
            "session_id": session_id,
            "message": message,
            "percentage": percentage
        })
    
    async def handle_get_performance_metrics(self, message: Dict[str, Any], websocket) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        session_id = message.get("session_id")
        
        if session_id not in self.sessions:
            return {"type": "error", "message": "Session not found"}
        
        orchestrator = self.sessions[session_id]
        
        try:
            metrics = await orchestrator.get_performance_metrics()
            return {
                "type": "performance_metrics",
                "session_id": session_id,
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"type": "error", "message": str(e)}
    
    async def cleanup_session(self, session_id: str):
        """Clean up session resources with GPU optimization cleanup"""
        if session_id in self.sessions:
            orchestrator = self.sessions[session_id]
            await orchestrator.cleanup()
            del self.sessions[session_id]
        
        if session_id in self.peer_connections:
            await self.peer_connections[session_id].close()
            del self.peer_connections[session_id]
        
        if session_id in self.connections:
            del self.connections[session_id]


# Create global handler instance
handler_v2 = WebSocketHandlerV2()


async def websocket_v2_endpoint(websocket: WebSocket):
    """WebSocket endpoint for V2 stateful orchestration"""
    await websocket.accept()
    session_id = None
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                session_id = message.get("session_id")
                
                response = await handler_v2.handle_message(websocket, message)
                
                if response:
                    await websocket.send_text(json.dumps(response))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id:
            await handler_v2.cleanup_session(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if session_id:
            await handler_v2.cleanup_session(session_id)
        logger.info(f"Session {session_id} cleaned up")