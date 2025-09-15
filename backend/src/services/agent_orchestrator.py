"""Agent Orchestrator Service - Manages the sequential execution of classification agents."""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import numpy as np

from agents.initial_classifier import InitialClassifier
from agents.detail_extractor import DetailExtractor
from agents.damage_detector import DamageDetector
from agents.final_compiler import FinalCompiler

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the execution of multiple agents in sequence."""
    
    def __init__(self):
        """Initialize the orchestrator with all agents."""
        self.agents = {
            "initial_classifier": InitialClassifier(),
            "detail_extractor": DetailExtractor(),
            "damage_detector": DamageDetector(),
            "final_compiler": FinalCompiler()
        }
        
        # Default agent sequence for automatic mode
        self.default_sequence = [
            "initial_classifier",
            "detail_extractor",
            "damage_detector",
            "final_compiler"
        ]
        
        # Agent timers (in seconds) for automatic mode
        self.agent_timers = {
            "initial_classifier": 4.0,
            "detail_extractor": 3.0,
            "damage_detector": 4.0,
            "final_compiler": 2.0  # Quick final compilation
        }
        
        self.current_session = None
        self.processing = False
        self.cancel_event = asyncio.Event()
        
    async def process_automatic(
        self,
        session_id: str,
        frame_provider: Callable,
        progress_callback: Optional[Callable] = None,
        result_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process agents in automatic sequence with timers.
        
        Args:
            session_id: Session identifier
            frame_provider: Callable that returns current frame(s)
            progress_callback: Optional callback for progress updates
            result_callback: Optional callback for agent results
            
        Returns:
            Final compiled results
        """
        self.current_session = session_id
        self.processing = True
        self.cancel_event.clear()
        
        results = []
        start_time = time.time()
        
        try:
            logger.info(f"[Orchestrator] Starting automatic processing for session {session_id}")
            logger.info(f"[Orchestrator] Agent sequence: {self.default_sequence}")
            logger.info(f"[Orchestrator] Agent timers: {self.agent_timers}")
            
            for i, agent_name in enumerate(self.default_sequence):
                if self.cancel_event.is_set():
                    logger.info(f"[Orchestrator] Processing cancelled for session {session_id}")
                    break
                
                agent = self.agents[agent_name]
                timer = self.agent_timers.get(agent_name, 5.0)
                logger.info(f"[Orchestrator] Starting agent {agent_name} ({i+1}/{len(self.default_sequence)}) with {timer}s timer")
                
                # Notify agent started
                if progress_callback:
                    await progress_callback({
                        "type": "agent_started",
                        "session_id": session_id,
                        "agent": agent_name,
                        "mode": "automatic",
                        "timer_seconds": timer,
                        "sequence_position": i + 1,
                        "total_agents": len(self.default_sequence)
                    })
                    logger.debug(f"[Orchestrator] Sent agent_started notification for {agent_name}")
                
                # Process with timer
                agent_result = await self._process_agent_with_timer(
                    agent,
                    agent_name,
                    frame_provider,
                    timer,
                    progress_callback
                )
                
                if agent_result:
                    results.append(agent_result)
                    logger.info(f"[Orchestrator] Agent {agent_name} completed with confidence {agent_result.get('confidence', 0):.2f}")
                    
                    # Send result via callback
                    if result_callback:
                        await result_callback({
                            "type": "agent_completed",
                            "session_id": session_id,
                            "agent": agent_name,
                            "results": agent_result
                        })
                        logger.debug(f"[Orchestrator] Sent agent_completed notification for {agent_name}")
                
                # Small delay between agents
                if i < len(self.default_sequence) - 1:
                    await asyncio.sleep(0.5)
            
            # Compile final results
            logger.info(f"[Orchestrator] Compiling final results from {len(results)} agents")
            final_results = await self._compile_final_results(results, frame_provider)
            
            processing_time = time.time() - start_time
            logger.info(f"[Orchestrator] Automatic processing completed in {processing_time:.2f}s")
            
            # Send final results
            if result_callback:
                await result_callback({
                    "type": "final_results",
                    "session_id": session_id,
                    "classification": final_results,
                    "processing_time": processing_time,
                    "agents_completed": len(results)
                })
                logger.debug("[Orchestrator] Sent final_results notification")
            
            return final_results
            
        except Exception as e:
            logger.error(f"[Orchestrator] Error in automatic processing: {str(e)}", exc_info=True)
            return self._error_result(str(e))
        finally:
            self.processing = False
            self.current_session = None
    
    async def process_manual(
        self,
        session_id: str,
        agent_name: str,
        frame_provider: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process a single agent manually.
        
        Args:
            session_id: Session identifier
            agent_name: Name of agent to trigger
            frame_provider: Callable that returns current frame(s)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Agent results
        """
        if agent_name not in self.agents:
            logger.error(f"[Orchestrator] Unknown agent: {agent_name}")
            return self._error_result(f"Unknown agent: {agent_name}")
        
        logger.info(f"[Orchestrator] Starting manual processing for agent {agent_name}")
        
        self.current_session = session_id
        self.processing = True
        
        try:
            agent = self.agents[agent_name]
            
            # Notify agent started
            if progress_callback:
                await progress_callback({
                    "type": "agent_started",
                    "agent": agent_name,
                    "mode": "manual"
                })
            
            # Get frames
            frames = frame_provider()
            # Fix numpy array comparison issue
            if frames is None or (isinstance(frames, list) and len(frames) == 0):
                return self._error_result("No frames available")
            
            # Process frames
            if isinstance(frames, list):
                result = await agent.process_frames(frames, max_frames=3)
            else:
                result = await agent.process_frame(frames)
            
            # Log the result
            logger.info(f"[Orchestrator] Manual {agent_name} completed with result: {result}")
            
            # Notify completion
            if progress_callback:
                await progress_callback({
                    "type": "agent_completed",
                    "agent": agent_name,
                    "results": result
                })
            
            return result
            
        except Exception as e:
            logger.error(f"[Orchestrator] Error in manual processing: {str(e)}", exc_info=True)
            return self._error_result(str(e))
        finally:
            self.processing = False
            self.current_session = None
    
    async def _process_agent_with_timer(
        self,
        agent: Any,
        agent_name: str,
        frame_provider: Callable,
        timer_seconds: float,
        progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """Process agent with a timer, collecting frames over time.
        
        Args:
            agent: Agent instance
            agent_name: Agent name
            frame_provider: Callable for frames
            timer_seconds: Time to run agent
            progress_callback: Progress updates
            
        Returns:
            Agent results
        """
        frames_collected = []
        start_time = time.time()
        update_interval = 0.5  # Update progress every 0.5 seconds
        last_update = start_time
        
        logger.info(f"[Orchestrator] Collecting frames for {agent_name} over {timer_seconds}s")
        
        # Collect frames during timer period
        while time.time() - start_time < timer_seconds:
            if self.cancel_event.is_set():
                break
            
            # Get current frame
            frame = frame_provider()
            if frame is not None:
                if isinstance(frame, list) and frame:
                    frames_collected.append(frame[0])
                elif isinstance(frame, np.ndarray):
                    frames_collected.append(frame)
            
            # Send progress update
            current_time = time.time()
            if current_time - last_update >= update_interval and progress_callback:
                elapsed = current_time - start_time
                progress = min(int((elapsed / timer_seconds) * 100), 100)
                timer_remaining = max(0, timer_seconds - elapsed)
                
                await progress_callback({
                    "type": "progress_update",
                    "agent": agent_name,
                    "progress": progress,
                    "timer_remaining": timer_remaining,
                    "frames_collected": len(frames_collected)
                })
                last_update = current_time
                logger.debug(f"[Orchestrator] {agent_name} progress: {progress}%, {timer_remaining:.1f}s remaining, {len(frames_collected)} frames")
            
            # Small delay to avoid overwhelming
            await asyncio.sleep(0.1)
        
        # Process collected frames
        if not frames_collected:
            logger.warning(f"[Orchestrator] No frames collected for {agent_name}")
            return self._error_result("No frames collected")
        
        # Use last few frames for analysis (most recent)
        frames_to_analyze = frames_collected[-5:] if len(frames_collected) > 5 else frames_collected
        
        logger.info(f"[Orchestrator] Processing {len(frames_to_analyze)} frames for {agent_name} (collected {len(frames_collected)} total)")
        result = await agent.process_frames(frames_to_analyze, max_frames=3)
        
        return result
    
    async def _compile_final_results(
        self,
        agent_results: List[Dict[str, Any]],
        frame_provider: Callable
    ) -> Dict[str, Any]:
        """Compile final results using the FinalCompiler agent.
        
        Args:
            agent_results: Results from all agents
            frame_provider: Callable for current frame
            
        Returns:
            Final compiled classification
        """
        compiler = self.agents["final_compiler"]
        
        # Set previous results for context
        compiler.set_previous_results(agent_results)
        
        # Get current frame for final visual verification
        frame = frame_provider()
        if frame is not None:
            if isinstance(frame, list) and frame:
                final_result = await compiler.process_frame(frame[0])
            elif isinstance(frame, np.ndarray):
                final_result = await compiler.process_frame(frame)
            else:
                # Use synthesis without visual verification
                final_result = {
                    "agent_name": "FinalCompiler",
                    "timestamp": datetime.now().isoformat(),
                    "attributes": compiler._synthesize_from_previous(),
                    "confidence": 0.7
                }
        else:
            # No frame available, synthesize from previous results
            final_result = {
                "agent_name": "FinalCompiler",
                "timestamp": datetime.now().isoformat(),
                "attributes": compiler._synthesize_from_previous(),
                "confidence": 0.6
            }
        
        # Build comprehensive result
        classification = final_result.get("attributes", {})
        
        # Add detailed breakdown from all agents
        classification["detailed_analysis"] = {
            agent_result["agent_name"]: agent_result.get("attributes", {})
            for agent_result in agent_results
        }
        
        # Calculate overall confidence
        confidences = [r.get("confidence", 0) for r in agent_results]
        confidences.append(final_result.get("confidence", 0))
        classification["overall_confidence"] = sum(confidences) / len(confidences) if confidences else 0
        
        return classification
    
    def cancel_processing(self):
        """Cancel current processing."""
        self.cancel_event.set()
        logger.info(f"Cancelling processing for session {self.current_session}")
    
    async def validate_agents(self) -> Dict[str, bool]:
        """Validate that all agents can connect to Ollama.
        
        Returns:
            Dictionary of agent validation status
        """
        validation_results = {}
        
        for name, agent in self.agents.items():
            try:
                is_valid = await agent.validate_ollama_connection()
                validation_results[name] = is_valid
                if not is_valid:
                    logger.warning(f"Agent {name} failed validation")
            except Exception as e:
                logger.error(f"Error validating agent {name}: {str(e)}")
                validation_results[name] = False
        
        return validation_results
    
    def set_agent_timers(self, timers: Dict[str, float]):
        """Update agent timers for automatic mode.
        
        Args:
            timers: Dictionary of agent names to timer values in seconds
        """
        self.agent_timers.update(timers)
        logger.info(f"Updated agent timers: {self.agent_timers}")
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure.
        
        Args:
            error_message: Error description
            
        Returns:
            Error result dictionary
        """
        return {
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session,
            "classification": {
                "final_type": "unknown",
                "final_category": "unknown",
                "final_condition": "unknown",
                "recommended_action": "review",
                "confidence_level": "low"
            }
        }