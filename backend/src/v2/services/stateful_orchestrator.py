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
from .state_persistence import StatePersistence

logger = logging.getLogger(__name__)


class StatefulOrchestrator:
    """Orchestrates stateful agent execution with multi-inference and timers"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_state = SessionState(session_id=session_id)
        self.frame_registry = FrameRegistry(session_id)
        self.frame_manager = FrameManager(session_id)
        self.state_persistence = StatePersistence(session_id)
        self.inference_engine = MultiInferenceEngine()
        self.frame_aggregator = FrameAggregator()
        
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
        
        logger.info(f"StatefulOrchestrator initialized for session {session_id}")
    
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
                                  timer_seconds: Optional[float] = None) -> AgentState:
        """Run an agent with timer-based multi-inference"""
        
        timer_seconds = timer_seconds or self.get_agent_timer(agent_name)
        
        # Initialize or get agent state
        if agent_name not in self.session_state.agent_states:
            self.session_state.agent_states[agent_name] = AgentState(
                agent_name=agent_name,
                timer_seconds=timer_seconds
            )
        
        agent_state = self.session_state.agent_states[agent_name]
        agent_state.start_timer()
        
        # Send agent_started message
        await self._send_agent_started(agent_name, timer_seconds)
        
        # Run inference loop until timer expires (with 1-second buffer)
        end_time = time.time() + timer_seconds
        buffer_time = 1.0  # Reserve for aggregation
        
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
                await asyncio.sleep(0.1)
                continue
            
            # Run inference
            inference_task = asyncio.create_task(
                self.inference_engine.run_inference(agent_name, frames, inference_num)
            )
            
            # Send inference update
            await self._send_inference_update(
                agent_name, inference_num, len(frames), agent_state.timer_remaining
            )
            
            # Wait for inference to complete (even if timer expires)
            result = await inference_task
            
            # Add result
            agent_state.add_inference_result(result)
            
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
        
        # Save checkpoint after agent completion
        await self.state_persistence.save_checkpoint(self.session_state)
        
        return agent_state
    
    async def _collect_frames(self, agent_name: str, batch_size: int, 
                            inference_num: int) -> List[Any]:
        """Collect frames for inference"""
        frames = []
        
        # Special case: damage_detector first inference includes shared frame
        if agent_name == "damage_detector" and inference_num == 1:
            shared_frame = await self.frame_registry.get_frame_for_damage_detector()
            if shared_frame is not None:
                frames.append(shared_frame)
                batch_size -= 1  # Reduce new frames needed
        
        # Collect new frames
        if self.frame_provider:
            for _ in range(batch_size):
                frame = await self.frame_provider()
                if frame is not None:
                    frames.append(frame)
                    
                    # Register first frame from initial_classifier
                    if agent_name == "initial_classifier" and inference_num == 1 and len(frames) == 1:
                        await self.frame_registry.register_initial_first_frame(frame)
        
        return frames
    
    def _get_batch_size(self, agent_name: str, inference_num: int, 
                       timer_remaining: float) -> int:
        """Determine batch size for inference"""
        if inference_num == 1:
            # First inference is always single (except damage_detector)
            return 2 if agent_name == "damage_detector" else 1
        
        # Progressive accumulation based on timer
        if timer_remaining < 2.0:
            return min(2, 5)
        elif timer_remaining < 3.0:
            return min(3, 5)
        else:
            return min(inference_num + 1, 5)  # Cap at 5
    
    async def pause_flow(self) -> Dict[str, Any]:
        """Pause the current flow"""
        current_agent = self.session_state.current_agent
        
        if not current_agent:
            return {"type": "error", "message": "No agent currently running"}
        
        if current_agent == "final_compiler":
            return {"type": "error", "message": "Cannot pause during final_compiler"}
        
        agent_state = self.session_state.agent_states.get(current_agent)
        if agent_state:
            agent_state.pause()
        
        self.session_state.pause(current_agent)
        
        # Save checkpoint when paused
        await self.state_persistence.save_checkpoint(self.session_state)
        
        response = {
            "type": "flow_paused",
            "session_id": self.session_id,
            "paused_agent": current_agent,
            "timer_remaining": agent_state.timer_remaining if agent_state else 0,
            "inference_count": agent_state.inference_count if agent_state else 0,
            "status": "PAUSED"
        }
        
        logger.info(f"Flow paused at agent {current_agent}")
        return response
    
    async def resume_flow(self, restart_agent: bool = False) -> Dict[str, Any]:
        """Resume the paused flow"""
        if self.session_state.status != SessionStatus.PAUSED:
            return {"type": "error", "message": "Session not paused"}
        
        paused_agent = self.session_state.paused_agent
        if not paused_agent:
            return {"type": "error", "message": "No paused agent found"}
        
        agent_state = self.session_state.agent_states.get(paused_agent)
        if agent_state:
            agent_state.resume(restart=restart_agent)
        
        self.session_state.resume(restart_agent)
        
        # Save checkpoint when resumed
        await self.state_persistence.save_checkpoint(self.session_state)
        
        response = {
            "type": "flow_resumed",
            "session_id": self.session_id,
            "resuming_agent": paused_agent,
            "timer_seconds": agent_state.timer_seconds if agent_state else 0,
            "restarted": restart_agent,
            "inference_count_before_pause": agent_state.inference_count if agent_state else 0
        }
        
        # Send agent_started for resumed agent
        if agent_state:
            await self._send_agent_started(paused_agent, agent_state.timer_remaining)
        
        logger.info(f"Flow resumed at agent {paused_agent} (restart={restart_agent})")
        return response
    
    async def _send_agent_started(self, agent: str, timer: float) -> None:
        """Send agent_started message with timer info"""
        if self.send_message:
            await self.send_message({
                "type": "agent_started",
                "session_id": self.session_id,
                "agent": agent,
                "timer_seconds": timer,
                "mode": self.session_state.mode.value,
                "sequence_position": self.session_state.agents_sequence.index(agent) + 1,
                "total_agents": len(self.session_state.agents_sequence)
            })
    
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
    
    async def load_from_checkpoint(self) -> bool:
        """Load session state from checkpoint if it exists"""
        try:
            saved_state = await self.state_persistence.load_checkpoint()
            if saved_state:
                self.session_state = saved_state
                logger.info(f"Session {self.session_id} restored from checkpoint")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about existing checkpoint"""
        return self.state_persistence.get_checkpoint_info()
    
    async def cleanup(self) -> None:
        """Clean up orchestrator resources"""
        await self.frame_registry.cleanup()
        await self.frame_manager.cleanup()
        await self.state_persistence.cleanup()
        logger.info(f"Orchestrator cleaned up for session {self.session_id}")
