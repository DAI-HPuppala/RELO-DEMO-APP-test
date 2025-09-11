"""Base Agent class for clothing classification agents."""
import asyncio
import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import aiohttp
import cv2

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all classification agents."""
    
    def __init__(self, name: str, model: str = "qwen2.5vl:3b"):
        """Initialize base agent.
        
        Args:
            name: Agent name for identification
            model: Ollama model to use for vision-language processing
        """
        self.name = name
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.processing = False
        self.last_result = None
        
    @abstractmethod
    def get_prompt(self) -> str:
        """Get the agent-specific prompt for the VLM.
        
        Returns:
            String prompt tailored to this agent's task
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the VLM response into structured data.
        
        Args:
            response: Raw text response from the VLM
            
        Returns:
            Structured dictionary with agent-specific attributes
        """
        pass
    
    async def process_frame(self, frame: np.ndarray, frame_id: str = None) -> Dict[str, Any]:
        """Process a single frame through the VLM.
        
        Args:
            frame: Video frame as numpy array (BGR format)
            frame_id: Optional frame identifier for debugging
            
        Returns:
            Agent analysis results
        """
        try:
            self.processing = True
            timestamp = datetime.now()
            
            # Generate frame ID if not provided
            if not frame_id:
                frame_id = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Save frame for debugging
            self._save_frame_for_debug(frame, frame_id, timestamp)
            
            logger.info(f"[{self.name}] Processing frame {frame_id} - Shape: {frame.shape}")
            
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare request for Ollama
            prompt = self.get_prompt()
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
            
            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ollama_url,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        vlm_response = result.get("response", "")
                        
                        # Log raw response for debugging
                        logger.info(f"[{self.name}] RAW RESPONSE for frame {frame_id}: {vlm_response[:500]}...")
                        
                        # Parse response into structured data
                        parsed = self.parse_response(vlm_response)
                        
                        # Add metadata
                        confidence = self._calculate_confidence(parsed)
                        self.last_result = {
                            "agent_name": self.name,
                            "timestamp": datetime.now().isoformat(),
                            "frame_id": frame_id,
                            "attributes": parsed,
                            "raw_response": vlm_response,
                            "confidence": confidence
                        }
                        
                        logger.info(f"[{self.name}] Completed analysis of frame {frame_id} - Confidence: {confidence:.2f}")
                        logger.info(f"[{self.name}] Parsed attributes: {parsed}")
                        return self.last_result
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"[{self.name}] Ollama API error for frame {frame_id}: {response.status} - {error_text}")
                        return self._error_result(f"VLM API error: {response.status}")
                        
        except asyncio.TimeoutError:
            logger.error(f"[{self.name}] Timeout processing frame {frame_id}")
            return self._error_result("Processing timeout")
        except Exception as e:
            logger.error(f"[{self.name}] Error processing frame {frame_id}: {str(e)}")
            return self._error_result(str(e))
        finally:
            self.processing = False
    
    async def process_frames(
        self, 
        frames: list, 
        max_frames: int = 5,
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Process multiple frames until confidence threshold is met.
        
        Args:
            frames: List of video frames
            max_frames: Maximum frames to process
            confidence_threshold: Stop when this confidence is reached
            
        Returns:
            Best result from frame analysis
        """
        best_result = None
        best_confidence = 0.0
        
        logger.info(f"[{self.name}] Processing {min(len(frames), max_frames)} frames")
        
        for i, frame in enumerate(frames[:max_frames]):
            frame_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
            logger.info(f"[{self.name}] INFERENCE #{i+1}/{min(len(frames), max_frames)} - Processing frame {frame_id}")
            result = await self.process_frame(frame, frame_id)
            
            if result and not result.get("error"):
                confidence = result.get("confidence", 0.0)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = result
                    best_result["frames_processed"] = i + 1
                
                # Stop if we reach confidence threshold
                if confidence >= confidence_threshold:
                    logger.info(f"[{self.name}] Reached confidence threshold {confidence:.2f} at frame {i+1}")
                    break
            
            # Small delay between frames to avoid overloading
            if i < len(frames) - 1:
                await asyncio.sleep(0.1)
        
        if best_result:
            return best_result
        else:
            return self._error_result("No valid results from frame processing")
    
    def _calculate_confidence(self, attributes: Dict[str, Any]) -> float:
        """Calculate confidence score based on attributes detected.
        
        Args:
            attributes: Parsed attributes from VLM
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base implementation: more attributes = higher confidence
        # Agents can override for specific logic
        if not attributes:
            return 0.0
        
        # Count non-null/non-empty values
        valid_count = sum(1 for v in attributes.values() if v)
        total_count = len(attributes)
        
        if total_count == 0:
            return 0.0
        
        return min(valid_count / total_count, 1.0)
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result structure.
        
        Args:
            error_message: Error description
            
        Returns:
            Error result dictionary
        """
        return {
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "attributes": {},
            "confidence": 0.0
        }
    
    async def validate_ollama_connection(self) -> bool:
        """Check if Ollama is accessible and model is available.
        
        Returns:
            True if Ollama is ready, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get(
                    "http://localhost:11434/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        
                        # Check if our model is available
                        model_names = [m.get("name", "") for m in models]
                        if any(self.model in name for name in model_names):
                            logger.info(f"[{self.name}] Ollama model {self.model} is available")
                            return True
                        else:
                            logger.warning(f"[{self.name}] Model {self.model} not found in Ollama")
                            return False
                    else:
                        logger.error(f"Ollama API returned status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            return False
    
    def _save_frame_for_debug(self, frame: np.ndarray, frame_id: str, timestamp: datetime):
        """Save frame to captured_frames directory for debugging.
        
        Args:
            frame: Video frame to save
            frame_id: Unique frame identifier
            timestamp: When frame was captured
        """
        try:
            # Create captured_frames directory if it doesn't exist
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "captured_frames")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create subdirectory for this agent
            agent_dir = os.path.join(debug_dir, self.name.lower().replace(" ", "_"))
            os.makedirs(agent_dir, exist_ok=True)
            
            # Save frame with descriptive filename
            filename = f"{self.name}_{frame_id}.jpg"
            filepath = os.path.join(agent_dir, filename)
            
            cv2.imwrite(filepath, frame)
            logger.debug(f"[{self.name}] Saved debug frame to {filepath}")
            
        except Exception as e:
            logger.warning(f"[{self.name}] Could not save debug frame: {e}")