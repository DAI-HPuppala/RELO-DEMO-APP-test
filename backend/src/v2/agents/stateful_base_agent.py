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
        
        # Removed verbose initialization logging
    
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

            # Log frame capture clearly
            logger.info("\n" + "*"*80)
            logger.info(f"🎥 FRAME CAPTURE COMPLETE for {self.agent_name.upper()}")
            logger.info(f"  🔢 Inference Number: #{inference_num}")
            logger.info(f"  📷 Frames Captured: {len(frames)} frame{'s' if len(frames) > 1 else ''} from camera")
            if inference_num > 1:
                logger.info(f"  🔄 Single frame capture for inference #{inference_num} (context-aware)")
            logger.info(f"  ⏱️ Timer Remaining: {self.state.timer_remaining:.1f}s")
            logger.info(f"  ➡️ Next Step: Sending frame to inference engine with context..." if inference_num > 1 else f"  ➡️ Next Step: Sending frame to inference engine...")
            logger.info("*"*80 + "\n")

            # Update current inference number
            self.state.current_inference_num = inference_num

            # Get previous context for inference #2 and beyond
            previous_context = None
            if inference_num > 1:
                previous_context = self.state.get_previous_context()
                if previous_context:
                    logger.info(f"  📚 Using context from inference #{previous_context['inference_num']}")

            # Run inference (complete even if timer expires)
            inference_task = asyncio.create_task(
                self.inference_engine.run_inference(
                    self.agent_name,
                    frames,
                    inference_num,
                    previous_context
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

                # Store context for future inferences (only for initial, detail, damage agents)
                if self.agent_name in ["initial_classifier", "detail_extractor", "damage_detector"]:
                    self.state.add_inference_context(len(frames), result)
                    logger.info(f"  💾 Context saved for future inferences")

                logger.info("\n" + "✅"*40)
                logger.info(f"✅ INFERENCE COMPLETE: {self.agent_name} - Inference #{inference_num}")
                logger.info(f"  📊 Attributes Found: {len(result.attributes) if hasattr(result, 'attributes') else 'N/A'}")
                logger.info(f"  🎯 Confidence: {result.confidence if hasattr(result, 'confidence') else 'N/A'}")
                if inference_num > 1 and previous_context:
                    logger.info(f"  🔄 Context-aware inference used")
                logger.info("✅"*40 + "\n")
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
                    logger.info(f"✅ damage_detector successfully retrieved shared frame from initial_classifier")
                    logger.debug(f"Shared frame shape: {shared_frame.shape if hasattr(shared_frame, 'shape') else 'unknown'}")
                else:
                    logger.warning(f"⚠️ damage_detector could not retrieve shared frame - will capture {batch_size} new frames")
        
        # Capture new frames
        while len(frames) < batch_size and time.time() < end_time:
            if self.frame_provider:
                try:
                    # Frame provider is now async
                    if asyncio.iscoroutinefunction(self.frame_provider):
                        frame = await self.frame_provider()
                    else:
                        frame = await asyncio.get_event_loop().run_in_executor(None, self.frame_provider)
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
                # No frame provider available - this is now an error condition
                logger.error(f"{self.agent_name}: No frame provider available")
                break
            
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
        """Calculate batch size - always 1 frame per inference for consistency"""
        # Special case: damage_detector's first inference gets shared frame + 1 new
        if self.agent_name == "damage_detector" and inference_num == 1:
            return 2  # Will get 1 shared + 1 new

        # All other cases: always capture exactly 1 frame per inference
        # This ensures consistent frame capture regardless of time remaining
        return 1
    
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