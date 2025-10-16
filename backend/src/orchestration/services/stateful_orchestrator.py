"""Stateful Orchestrator service for managing agent pipeline with timers"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Any, Callable
from datetime import datetime

from ..models.agent_state import AgentState, AgentStatus
from ..models.session_state import SessionState, SessionStatus
from ..models.inference_result import InferenceResult
from .frame_registry import FrameRegistry
from .frame_manager import FrameManager
from .multi_inference_engine import MultiInferenceEngine
from .frame_aggregator import FrameAggregator

logger = logging.getLogger(__name__)


class StatefulOrchestrator:
    """Orchestrates stateful agent execution with multi-inference and timers"""
    
    def __init__(self, session_id: str, manual_mode: bool = False):
        self.session_id = session_id
        self.manual_mode = manual_mode
        self.session_state = SessionState(session_id=session_id)
        self.frame_registry = FrameRegistry(session_id)
        self.frame_manager = FrameManager(session_id)
        self.inference_engine = MultiInferenceEngine()
        self.frame_aggregator = FrameAggregator()

        # Cycle tracking for image export
        self.current_cycle = 1
        self.is_redo = False  # Track if current run is a redo
        self.redo_counts = {}  # Track redo attempts per agent: {agent_name: count}

        # Dynamic buffer tracking - use first inference time as threshold
        self.first_inference_times = {}  # {agent_name: first_inference_duration_seconds}

        # GPU optimization state
        self.gpu_optimization_initialized = False
        self.gpu_optimization_status = "not_initialized"
        
        # Agent configurations
        self.agent_timers = {
            "initial_classifier": 4.0,
            "detail_extractor": 3.0,
            "damage_detector": 4.0,
            "final_compiler": 1.0  # Aggregation only
        }
        
        # Custom timers for testing
        self.custom_timers: Optional[Dict[str, float]] = None
        
        # Frame provider callback
        self.frame_provider: Optional[Callable] = None
        
        # WebSocket message callback
        self.send_message: Optional[Callable] = None
        
        logger.info(f"StatefulOrchestrator initialized for session {session_id} with GPU optimization enabled")
    
    def is_vlm_configured(self) -> bool:
        """Check if VLM inference is properly configured"""
        # Check GPU-optimized VLM configuration
        gpu_ready = self.inference_engine.is_gpu_optimized()
        return gpu_ready
    
    async def initialize_gpu_optimization(self, progress_callback=None) -> bool:
        """Initialize GPU optimization for VLM inference"""
        if self.gpu_optimization_initialized:
            logger.info("GPU optimization already initialized")
            return True
        
        logger.info("🚀 Initializing GPU optimization for VLM inference...")
        
        try:
            success = await self.inference_engine.initialize_gpu_optimization(progress_callback)
            
            if success:
                self.gpu_optimization_initialized = True
                self.gpu_optimization_status = "optimized"
                
                # Get optimization details
                opt_status = self.inference_engine.get_optimization_status()
                logger.info(f"✅ GPU optimization initialized: {opt_status.get('optimization_level', 'unknown')} level")
                logger.info(f"GPU memory available: {opt_status.get('gpu_memory_mb', 0)}MB")
            else:
                self.gpu_optimization_status = "failed"
                logger.error("❌ GPU optimization failed - GPU required for operation")
                raise RuntimeError("GPU optimization required but failed")
            
            return success
            
        except Exception as e:
            self.gpu_optimization_status = "failed"
            logger.error(f"GPU optimization initialization error: {e}")
            return False
    
    async def validate_setup(self) -> Dict[str, Any]:
        """Validate that all required services are properly configured"""
        
        # Get optimization status
        opt_status = self.inference_engine.get_optimization_status()
        
        validation = {
            "session_id": self.session_id,
            "vlm_configured": self.is_vlm_configured(),
            "frame_provider_set": self.frame_provider is not None,
            "websocket_callback_set": self.send_message is not None,
            "gpu_optimization": {
                "initialized": self.gpu_optimization_initialized,
                "status": self.gpu_optimization_status,
                "details": opt_status
            },
            "status": "ready" if (self.is_vlm_configured() and self.frame_provider) else "not_ready",
            "warnings": []
        }
        
        # VLM validation
        if not validation["vlm_configured"]:
            validation["warnings"].append("VLM inference service not configured - check Ollama setup")
        
        if not validation["frame_provider_set"]:
            validation["warnings"].append("Frame provider not set - video stream may not be available")
        
        # GPU optimization warnings
        if not self.gpu_optimization_initialized:
            validation["errors"].append("GPU optimization not initialized - GPU required for operation")
        elif self.gpu_optimization_status == "failed":
            validation["errors"].append("GPU optimization failed - GPU required for operation")
        
        mode = "GPU-optimized" if opt_status.get("gpu_optimized", False) else "GPU-required"
        logger.info(f"Orchestrator validation: {validation['status']} - VLM: {validation['vlm_configured']} ({mode}), Frames: {validation['frame_provider_set']}")
        
        return validation
    
    def set_custom_timers(self, timers: Dict[str, float]) -> None:
        """Set custom timers for edge case testing"""
        self.custom_timers = timers
        logger.info(f"Custom timers set: {timers}")
    
    def get_agent_timer(self, agent_name: str) -> float:
        """Get timer duration for an agent"""
        if self.custom_timers and agent_name in self.custom_timers:
            return self.custom_timers[agent_name]
        return self.agent_timers.get(agent_name, 4.0)
    
    async def run_agent_with_timer(self, agent_name: str,
                                  timer_seconds: Optional[float] = None,
                                  initial_classifier_context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Run an agent with timer-based multi-inference"""
        
        # Ensure VLM is ready before processing
        if not self.is_vlm_configured():
            logger.error(f"VLM not configured for agent {agent_name}, attempting initialization...")
            # Try to initialize if not already done
            if not self.gpu_optimization_initialized:
                init_success = await self.initialize_gpu_optimization()
                if not init_success:
                    logger.error(f"Failed to initialize VLM for agent {agent_name}")
                    # Return empty state to prevent crash
                    return AgentState(agent_name=agent_name, timer_seconds=0)
        
        timer_seconds = timer_seconds or self.get_agent_timer(agent_name)
        
        # Initialize or get agent state
        if agent_name not in self.session_state.agent_states:
            logger.info(f"Creating NEW agent state for {agent_name} with timer={timer_seconds}s")
            self.session_state.agent_states[agent_name] = AgentState(
                agent_name=agent_name,
                timer_seconds=timer_seconds
            )
        else:
            # DEFENSIVE: In manual mode, agent state should have been reset!
            # If it exists with completed inferences, previous results could contaminate new run
            old_state = self.session_state.agent_states[agent_name]
            if self.manual_mode and old_state.inference_count > 0:
                logger.error(f"⚠️ CONTAMINATION RISK: Agent {agent_name} has existing state with {old_state.inference_count} inferences in manual mode!")
                logger.error(f"   Old inference contexts could contaminate new run. Forcing reset...")
                # Force reset to prevent contamination
                del self.session_state.agent_states[agent_name]
                self.session_state.agent_states[agent_name] = AgentState(
                    agent_name=agent_name,
                    timer_seconds=timer_seconds
                )
                logger.info(f"✅ Forced fresh state for {agent_name} - preventing context contamination")
            else:
                logger.warning(f"Agent state for {agent_name} already exists with inference_count={old_state.inference_count} (auto mode continuation)")

        agent_state = self.session_state.agent_states[agent_name]
        logger.info(f"Starting timer for {agent_name} - inference_count={agent_state.inference_count}")
        agent_state.start_timer()
        
        # Send agent_started message with optimization info
        opt_info = self.inference_engine.get_optimization_status()
        await self._send_agent_started(agent_name, timer_seconds, opt_info)
        
        # Run inference loop until timer expires (with dynamic buffer)
        end_time = time.time() + timer_seconds
        # Start with default 1.0s buffer, will become dynamic after first inference
        buffer_time = 1.0

        while time.time() < (end_time - buffer_time):
            agent_state.update_timer()
            
            if agent_state.status != AgentStatus.RUNNING:
                break  # Paused or stopped
            
            # Determine batch size
            inference_num = agent_state.inference_count + 1
            batch_size = self._get_batch_size(agent_name, inference_num, agent_state.timer_remaining)
            
            # Collect frames
            frames = await self._collect_frames(agent_name, batch_size, inference_num)
            if not frames:
                logger.warning(f"No frames available for {agent_name} inference #{inference_num}, waiting...")
                await asyncio.sleep(0.1)
                continue
            
            # Validate we have actual frame data
            import numpy as np
            valid_frames = 0
            for f in frames:
                if isinstance(f, np.ndarray) and f.size > 0:
                    valid_frames += 1
                elif f is not None:
                    valid_frames += 1
            
            if valid_frames == 0:
                logger.error(f"All frames invalid for {agent_name} inference #{inference_num}")
                await asyncio.sleep(0.1)
                continue
            
            logger.debug(f"Processing {valid_frames} valid frames for {agent_name}")

            # Get previous context for inference #2 and beyond
            previous_context = None
            if inference_num > 1:
                previous_context = agent_state.get_previous_context()
                if previous_context:
                    logger.info(f"  📚 Using context from inference #{previous_context['inference_num']} for {agent_name}")
                    # Log the context details
                    logger.info("*" * 30 + f" CONTEXT FOR {agent_name.upper()} " + "*" * 30)
                    logger.info(f"  Previous Attributes: {previous_context.get('attributes', {})}")
                    logger.info(f"  Previous Confidence: {previous_context.get('confidence', 0.0):.1%}")
                    logger.info(f"  Frames Used: {previous_context.get('frames_used', 'unknown')}")
                    logger.info("*" * (60 + len(agent_name)))
                else:
                    logger.debug(f"No previous context available for {agent_name} inference #{inference_num}")

            # Run inference with context and cycle metadata
            # Get redo attempt number for this agent (0 = not a redo)
            redo_attempt = self.redo_counts.get(agent_name, 0)

            # Determine mode for filename
            mode = "manual" if self.manual_mode else "auto"

            # Start timing BEFORE creating the task (task starts immediately)
            inference_start = time.time()

            inference_task = asyncio.create_task(
                self.inference_engine.run_inference(
                    agent_name, frames, inference_num, previous_context,
                    cycle_num=self.current_cycle,
                    redo_attempt=redo_attempt,
                    mode=mode,
                    initial_classifier_context=initial_classifier_context
                )
            )

            # Send inference update
            await self._send_inference_update(
                agent_name, inference_num, len(frames), agent_state.timer_remaining
            )

            # Wait for inference to complete (even if timer expires)
            result = await inference_task
            inference_duration = time.time() - inference_start

            # Track first inference time for dynamic buffer calculation (AUTO MODE ONLY)
            if not self.manual_mode and inference_num == 1 and agent_name not in self.first_inference_times:
                self.first_inference_times[agent_name] = inference_duration
                # Update buffer to be first_inference_time - 0.3s (aggressive timing)
                buffer_time = inference_duration - 0.3
                logger.info(f"📊 {agent_name} inference #1 took {inference_duration:.2f}s")
                logger.info(f"⏱️  Dynamic buffer updated to {buffer_time:.2f}s (inference time - 0.3s for aggressive second run)")
                logger.info(f"   Second inference will run if time remaining > {buffer_time:.2f}s")
            elif self.manual_mode:
                logger.debug(f"📊 {agent_name} inference #{inference_num} took {inference_duration:.2f}s (manual mode: using fixed 1.0s buffer)")

            # Add result
            agent_state.add_inference_result(result)

            # Store context for future inferences (only for initial, detail, damage agents)
            if agent_name in ["initial_classifier", "detail_extractor", "damage_detector"]:
                # Pass the first frame from current inference for batch processing in next inference
                # We use the first frame as it's the most representative (new frame, not shared)
                frame_to_store = frames[0] if frames else None
                agent_state.add_inference_context(len(frames), result, frame_to_store)
                logger.info(f"  💾 Context saved for {agent_name} future inferences (including frame for batch processing)")
            
            # Log the raw inference result
            logger.info(f"{agent_name} inference #{inference_num} raw result: {result.raw_response}")
            
            # Send inference result to frontend
            await self._send_inference_result(agent_name, inference_num, result)
            
            # Update timer
            agent_state.update_timer()
        
        # Aggregate results
        if agent_state.inference_results:
            aggregated = await self.frame_aggregator.aggregate(
                agent_name, agent_state.inference_results
            )
            agent_state.finalized_attributes = aggregated.attributes
            agent_state.aggregation_method = aggregated.aggregation_method
        
        agent_state.complete()
        logger.info(f"Agent {agent_name} completed with {agent_state.inference_count} inferences")

        return agent_state
    
    async def _collect_frames(self, agent_name: str, batch_size: int, 
                            inference_num: int) -> List[Any]:
        """Collect frames for inference"""
        frames = []

        # Collect new frames
        if self.frame_provider:
            for i in range(batch_size):
                try:
                    frame = await self.frame_provider()
                    if frame is not None:
                        # Validate frame has content
                        import numpy as np
                        if isinstance(frame, np.ndarray):
                            if frame.size > 0:
                                frames.append(frame)
                                if batch_size == 1:
                                    logger.debug(f"Collected single frame with shape {frame.shape}")
                                else:
                                    logger.debug(f"Collected frame {i+1}/{batch_size} with shape {frame.shape}")
                            else:
                                logger.warning(f"Frame {i+1} is empty numpy array")
                        else:
                            frames.append(frame)
                            logger.debug(f"Collected non-numpy frame {i+1}/{batch_size}")
                        
                    else:
                        logger.warning(f"Frame provider returned None for frame {i+1}/{batch_size}")
                except Exception as e:
                    logger.error(f"Error collecting frame {i+1}/{batch_size}: {e}")
        else:
            logger.error("No frame provider set for orchestrator")
        
        if len(frames) == 0:
            logger.warning(f"No frames collected for {agent_name} inference #{inference_num}")
        else:
            if inference_num > 1:
                logger.info(f"Collected {len(frames)} frame{'s' if len(frames) > 1 else ''} for {agent_name} inference #{inference_num} (with context)")
            else:
                logger.info(f"Collected {len(frames)} frame{'s' if len(frames) > 1 else ''} for {agent_name} inference #{inference_num}")
        
        return frames
    
    def _get_batch_size(self, agent_name: str, inference_num: int,
                       timer_remaining: float) -> int:
        """Determine batch size for inference - always 1 frame for consistency"""
        # All agents: always capture exactly 1 frame per inference
        # This ensures consistent single frame capture regardless of time or inference number
        return 1
    
    async def pause_flow(self, current_agent: Optional[str] = None) -> Dict[str, Any]:
        """Pause the current flow and send partial results

        Args:
            current_agent: The agent to pause (if None, uses session_state.current_agent)
        """
        if current_agent is None:
            current_agent = self.session_state.current_agent
            logger.debug(f"pause_flow: No agent specified, using session_state.current_agent = {current_agent}")
        else:
            logger.debug(f"pause_flow: Using specified agent = {current_agent}")

        if not current_agent:
            logger.warning("pause_flow: No agent currently running")
            return {"type": "error", "message": "No agent currently running"}

        if current_agent == "final_compiler":
            return {"type": "error", "message": "Cannot pause during final_compiler"}

        agent_state = self.session_state.agent_states.get(current_agent)

        if not agent_state:
            logger.warning(f"pause_flow: Agent {current_agent} has no state (may have already completed)")
            # Still proceed to set pause status
            self.session_state.pause(current_agent)
            return {
                "type": "flow_paused",
                "session_id": self.session_id,
                "paused_agent": current_agent,
                "had_partial_results": False,
                "state_cleared": False,
                "status": "PAUSED"
            }

        # If agent has partial results, aggregate and send to frontend
        partial_results = None
        if agent_state and agent_state.inference_results:
            logger.info(f"Agent {current_agent} paused with {agent_state.inference_count} partial inferences - aggregating...")

            # Aggregate partial results
            aggregated = await self.frame_aggregator.aggregate(
                current_agent, agent_state.inference_results
            )

            partial_results = {
                "agent": current_agent,
                "attributes": aggregated.attributes,
                "confidence": aggregated.overall_confidence,
                "total_inferences": agent_state.inference_count,
                "aggregation_method": aggregated.aggregation_method,
                "is_partial": True  # Mark as partial/deprecated
            }

            # Send partial results to frontend for progressive updates
            if self.send_message:
                await self.send_message({
                    "type": "agent_paused_with_results",
                    "session_id": self.session_id,
                    **partial_results
                })

            logger.info(f"Sent partial results from {current_agent} to frontend")

        # Clear agent state NOW (save 10ms on resume)
        self.reset_agent_state(current_agent)
        logger.info(f"Cleared {current_agent} state on pause - will restart fresh on resume")

        self.session_state.pause(current_agent)

        response = {
            "type": "flow_paused",
            "session_id": self.session_id,
            "paused_agent": current_agent,
            "had_partial_results": partial_results is not None,
            "state_cleared": True,
            "status": "PAUSED"
        }

        logger.info(f"Flow paused - {current_agent} state cleared, ready for fresh resume")
        return response
    
    async def resume_flow(self, restart_agent: bool = False) -> Dict[str, Any]:
        """Resume the paused flow (state already cleared on pause)"""
        if self.session_state.status != SessionStatus.PAUSED:
            return {"type": "error", "message": "Session not paused"}

        paused_agent = self.session_state.paused_agent
        if not paused_agent:
            return {"type": "error", "message": "No paused agent found"}

        # State already cleared in pause_flow() - resume immediately!
        self.session_state.resume(restart_agent)

        # Get full timer for fresh start
        timer_seconds = self.get_agent_timer(paused_agent)

        response = {
            "type": "flow_resumed",
            "session_id": self.session_id,
            "resuming_agent": paused_agent,
            "timer_seconds": timer_seconds,
            "fresh_start": True
        }

        # Send agent_started with full timer for fresh start
        opt_info = self.inference_engine.get_optimization_status()
        await self._send_agent_started(paused_agent, timer_seconds, opt_info)

        logger.info(f"Flow resumed - {paused_agent} starting fresh with {timer_seconds}s timer (state pre-cleared)")
        return response
    
    async def _send_agent_started(self, agent: str, timer: float, optimization_info: Dict[str, Any] = None) -> None:
        """Send agent_started message with timer and optimization info"""
        if self.send_message:
            message = {
                "type": "agent_started",
                "session_id": self.session_id,
                "agent": agent,
                "timer_seconds": timer,
                "mode": self.session_state.mode.value,
                "sequence_position": self.session_state.agents_sequence.index(agent) + 1,
                "total_agents": len(self.session_state.agents_sequence)
            }
            
            # Add GPU optimization info if available
            if optimization_info:
                message["gpu_optimization"] = {
                    "enabled": optimization_info.get("gpu_optimization", False),
                    "mode": optimization_info.get("mode", "unknown"),
                    "optimization_level": optimization_info.get("optimization_level", "unknown")
                }
            
            await self.send_message(message)
    
    async def _send_inference_update(self, agent: str, inference_num: int, 
                                    frames_used: int, timer_remaining: float) -> None:
        """Send inference update message"""
        if self.send_message:
            await self.send_message({
                "type": "inference_update",
                "session_id": self.session_id,
                "agent": agent,
                "inference_num": inference_num,
                "frames_used": frames_used,
                "timer_remaining": round(timer_remaining, 1),
                "inference_type": "single_image" if frames_used == 1 else "multi_image",
                "status": "processing"
            })
    
    async def _send_inference_result(self, agent: str, inference_num: int, result: Any) -> None:
        """Send inference result message with raw response"""
        if self.send_message:
            await self.send_message({
                "type": "inference_result",
                "session_id": self.session_id,
                "agent": agent,
                "inference_num": inference_num,
                "attributes": result.attributes,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "raw_response": result.raw_response,
                "inference_type": result.inference_type,
                "duration_ms": result.duration_ms
            })
    
    def get_agent_state(self, agent_name: str) -> Optional[AgentState]:
        """Get state for a specific agent"""
        return self.session_state.agent_states.get(agent_name)

    def reset_agent_state(self, agent_name: str) -> None:
        """Reset a specific agent's state (for manual mode redo/jump)"""
        if agent_name in self.session_state.agent_states:
            # Log the current state before deletion
            old_state = self.session_state.agent_states[agent_name]
            logger.info(f"Deleting agent state for {agent_name} - had inference_count={old_state.inference_count}")
            del self.session_state.agent_states[agent_name]
            logger.info(f"Reset agent state for {agent_name} - will restart from inference #1")
            # Verify it's deleted
            if agent_name in self.session_state.agent_states:
                logger.error(f"ERROR: Agent {agent_name} still exists after deletion!")
            else:
                logger.info(f"Confirmed: Agent {agent_name} deleted from session_state.agent_states")

        # Clear first inference time for fresh dynamic buffer calculation
        if agent_name in self.first_inference_times:
            del self.first_inference_times[agent_name]
            logger.info(f"Cleared first inference time for {agent_name} - will recalculate on next run")

    def reset_all_agent_states(self) -> None:
        """Reset all agent states for new cycle (manual mode only)"""
        if self.manual_mode:
            self.session_state.agent_states.clear()
            self.first_inference_times.clear()  # Clear all dynamic buffer timings
            logger.info("Reset all agent states for new manual mode cycle")

            # Clean up frame registry
            if self.frame_registry:
                asyncio.create_task(self.frame_registry.cleanup())

    def reset_for_new_cycle(self) -> None:
        """Reset orchestrator state for a new cycle (both auto and manual modes)"""
        self.first_inference_times.clear()
        logger.info(f"🔄 Cleared dynamic buffer timings for new cycle (mode: {'manual' if self.manual_mode else 'auto'})")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            # Get inference engine metrics
            engine_metrics = await self.inference_engine.health_check()
            
            # Get session state metrics
            session_metrics = {
                "session_id": self.session_id,
                "session_status": self.session_state.status.value,
                "current_agent": self.session_state.current_agent,
                "agents_completed": len([name for name, state in self.session_state.agent_states.items() if state.status.value == "COMPLETED"]),
                "total_agents": len(self.session_state.agents_sequence)
            }
            
            # Combine metrics
            return {
                "session": session_metrics,
                "inference_engine": engine_metrics,
                "gpu_optimization": {
                    "initialized": self.gpu_optimization_initialized,
                    "status": self.gpu_optimization_status
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    async def cleanup(self) -> None:
        """Clean up orchestrator resources"""
        logger.info(f"Cleaning up orchestrator for session {self.session_id}...")
        
        # Cleanup inference engine (includes GPU cleanup)
        try:
            await self.inference_engine.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down inference engine: {e}")
        
        # Cleanup other resources
        await self.frame_registry.cleanup()
        await self.frame_manager.cleanup()

        logger.info(f"Orchestrator cleaned up for session {self.session_id}")
