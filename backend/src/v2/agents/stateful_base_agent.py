"""Stateful Base Agent with timer-based multi-inference support"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ..models.agent_state import AgentState, AgentStatus
from ..models.inference_result import InferenceResult
from ..services.multi_inference_engine import MultiInferenceEngine
from ..services.frame_aggregator import FrameAggregator
from ..services.frame_registry import FrameRegistry

logger = logging.getLogger(__name__)


class StatefulBaseAgent(ABC):
    """Base class for all stateful agents with multi-inference capability"""
    
    def __init__(self, agent_name: str, timer_seconds: float = 4.0):
        self.agent_name = agent_name
        self.timer_seconds = timer_seconds
        
        # State tracking
        self.state = AgentState(
            agent_name=agent_name,
            timer_seconds=timer_seconds
        )
        
        # Services
        self.inference_engine = MultiInferenceEngine()
        self.frame_aggregator = FrameAggregator()
        self.frame_registry: Optional[FrameRegistry] = None
        
        # Frame provider (WebRTC or mock)
        self.frame_provider: Optional[Callable] = None
        
        # Callbacks
        self.on_inference_update: Optional[Callable] = None
        self.on_agent_complete: Optional[Callable] = None
        
        logger.info(f"StatefulBaseAgent {agent_name} initialized with {timer_seconds}s timer")
    
    @abstractmethod
    def get_expected_attributes(self) -> List[str]:
        """Get list of attributes this agent should detect"""
        pass
    
    @abstractmethod
    def validate_result(self, attributes: Dict[str, Any]) -> bool:
        """Validate that result contains expected attributes"""
        pass
    
    async def run_with_timer(self) -> Dict[str, Any]:
        """Execute agent with timer-based multi-inference loop"""
        logger.info(f"Starting {self.agent_name} with {self.timer_seconds}s timer")
        
        # Start timer
        self.state.start_timer()
        end_time = time.time() + self.timer_seconds
        buffer_time = 1.0  # Reserve for aggregation
        
        # Inference loop
        while time.time() < (end_time - buffer_time):
            # Check if paused
            if self.state.status != AgentStatus.RUNNING:
                logger.info(f"{self.agent_name} paused or stopped")
                break
            
            # Update timer
            self.state.update_timer()
            
            # Determine inference parameters
            inference_num = self.state.inference_count + 1
            batch_size = self._calculate_batch_size(inference_num, self.state.timer_remaining)
            
            # Capture frames with retry
            frames = await self._capture_frames_with_retry(
                batch_size, 
                inference_num,
                max_time=self.state.timer_remaining - buffer_time
            )
            
            if not frames:
                logger.warning(f"{self.agent_name}: No frames captured, continuing")
                await asyncio.sleep(0.1)
                continue
            
            # Update current inference number
            self.state.current_inference_num = inference_num
            
            # Run inference (complete even if timer expires)
            inference_task = asyncio.create_task(
                self.inference_engine.run_inference(
                    self.agent_name,
                    frames,
                    inference_num
                )
            )
            
            # Send update if callback registered
            if self.on_inference_update:
                await self.on_inference_update({
                    "agent": self.agent_name,
                    "inference_num": inference_num,
                    "frames_used": len(frames),
                    "timer_remaining": self.state.timer_remaining
                })
            
            # Wait for inference completion
            try:
                result = await inference_task
                self.state.add_inference_result(result)
                logger.info(f"{self.agent_name} completed inference #{inference_num}")
            except Exception as e:
                logger.error(f"{self.agent_name} inference #{inference_num} failed: {e}")
            
            # Update timer after inference
            self.state.update_timer()
        
        # Aggregate results using the rules
        final_result = await self._aggregate_results()
        
        # Mark complete
        self.state.complete()
        
        # Callback
        if self.on_agent_complete:
            await self.on_agent_complete(self.agent_name, final_result)
        
        logger.info(f"{self.agent_name} completed with {self.state.inference_count} inferences")
        return final_result
    
    async def _capture_frames_with_retry(self, batch_size: int, 
                                        inference_num: int,
                                        max_time: float) -> List[np.ndarray]:
        """Capture frames with retry logic until timer expires"""
        frames = []
        end_time = time.time() + max_time
        
        # Special handling for damage_detector's first inference
        if self.agent_name == "damage_detector" and inference_num == 1:
            # Get shared frame from initial classifier
            if self.frame_registry:
                shared_frame = await self.frame_registry.get_frame_for_damage_detector()
                if shared_frame is not None:
                    frames.append(shared_frame)
                    batch_size -= 1
                    logger.debug(f"damage_detector using shared frame from initial_classifier")
        
        # Capture new frames
        while len(frames) < batch_size and time.time() < end_time:
            if self.frame_provider:
                try:
                    frame = await self.frame_provider()
                    if frame is not None:
                        frames.append(frame)
                        
                        # Register first frame from initial_classifier for sharing
                        if (self.agent_name == "initial_classifier" and 
                            inference_num == 1 and len(frames) == 1 and 
                            self.frame_registry):
                            frame_id = await self.frame_registry.register_initial_first_frame(frame)
                            logger.info(f"Registered initial first frame for sharing: {frame_id}")
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Frame capture failed: {e}")
                    await asyncio.sleep(0.1)
            else:
                # Mock frames for testing
                frames.append(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            
            if len(frames) >= batch_size:
                break
        
        if not frames and time.time() >= end_time:
            logger.error(f"{self.agent_name}: Frame capture timeout after {max_time}s")
            # Notify frontend about error but continue to next agent
            if self.on_inference_update:
                await self.on_inference_update({
                    "agent": self.agent_name,
                    "error": "Frame capture timeout - continuing to next agent"
                })
        
        return frames
    
    def _calculate_batch_size(self, inference_num: int, timer_remaining: float) -> int:
        """Calculate optimal batch size based on inference number and remaining time"""
        # First inference is always single (except damage_detector)
        if inference_num == 1:
            if self.agent_name == "damage_detector":
                return 2  # Will get 1 shared + 1 new
            return 1
        
        # Progressive accumulation
        if timer_remaining < 2.0:
            return min(2, 5)
        elif timer_remaining < 3.0:
            return min(3, 5)
        else:
            # Progressively increase: 2, 3, 4, 5
            return min(inference_num + 1, 5)
    
    async def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate multiple inference results based on rules"""
        if not self.state.inference_results:
            logger.warning(f"{self.agent_name}: No results to aggregate")
            return {"error": "No inference results"}
        
        # Use frame aggregator service
        aggregated = await self.frame_aggregator.aggregate(
            self.agent_name,
            self.state.inference_results
        )
        
        # Store in state
        self.state.finalized_attributes = aggregated.attributes
        self.state.aggregation_method = aggregated.aggregation_method
        
        # Build final result
        result = {
            "agent": self.agent_name,
            "total_inferences": self.state.inference_count,
            "aggregation_method": aggregated.aggregation_method,
            "attributes": aggregated.attributes,
            "confidence": aggregated.overall_confidence,
            "conflicts_resolved": [c.to_dict() for c in aggregated.conflicts_resolved]
        }
        
        # Validate result
        if not self.validate_result(aggregated.attributes):
            logger.warning(f"{self.agent_name}: Result validation failed")
            result["validation"] = "failed"
        else:
            result["validation"] = "passed"
        
        return result
    
    def pause(self) -> None:
        """Pause the agent processing"""
        self.state.pause()
        logger.info(f"{self.agent_name} paused with {self.state.timer_remaining:.1f}s remaining")
    
    def resume(self, restart: bool = False) -> None:
        """Resume agent processing"""
        self.state.resume(restart)
        if restart:
            logger.info(f"{self.agent_name} restarted with full {self.timer_seconds}s timer")
        else:
            logger.info(f"{self.agent_name} resumed with {self.state.timer_remaining:.1f}s remaining")
    
    def reset(self) -> None:
        """Reset agent to initial state"""
        self.state.reset()
        logger.info(f"{self.agent_name} reset to initial state")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return self.state.to_dict()
    
    def set_frame_provider(self, provider: Callable) -> None:
        """Set the frame provider function"""
        self.frame_provider = provider
    
    def set_frame_registry(self, registry: FrameRegistry) -> None:
        """Set the frame registry for sharing"""
        self.frame_registry = registry
    
    def set_inference_engine(self, engine: MultiInferenceEngine) -> None:
        """Set custom inference engine"""
        self.inference_engine = engine