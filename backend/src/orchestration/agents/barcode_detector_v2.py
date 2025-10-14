"""Barcode Detector Agent v2 - First agent in RELO classification pipeline.

This agent runs BEFORE all other agents (initial, detail, damage, final) and
detects barcodes using WebRTC frames with 2-stage detection.

Features:
- First agent in pipeline
- No timeout by default (continuous until detected or manual skip)
- Uses WebRTC frames (no separate camera connection)
- 2-stage detection (fast presence check + decode)
- One-time detection per cycle (no re-running)
- Manual override with text input
- Saves detected frames to disk

Workflow:
1. Agent starts, begins capturing frames from WebRTC
2. Each frame: Stage 1 presence check (~2-5ms)
3. If barcode present: Stage 2 decode (~50-200ms)
4. If detected: Save frame, store in state, complete
5. If manual skip: Prompt user for manual entry or skip
"""

import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional
import numpy as np

from .stateful_base_agent import StatefulBaseAgent
from ...services.barcode_detection_service import get_barcode_service

logger = logging.getLogger(__name__)


class BarcodeDetectorV2(StatefulBaseAgent):
    """Barcode detection agent - first in classification pipeline."""

    def __init__(self, timer_seconds: float = 0):
        """Initialize barcode detector agent.

        Args:
            timer_seconds: Timer duration (0 = no timeout, continuous until detected)
        """
        # Load timeout from environment
        timeout = float(os.getenv('BARCODE_DETECTION_TIMEOUT', '0'))
        super().__init__("barcode_detector", timer_seconds=timeout)

        # Barcode service
        self.barcode_service = get_barcode_service()

        # Detection state
        self.detected_barcode: Optional[Dict[str, Any]] = None
        self.allowed_types: List[str] = []
        self.detection_complete = False

        # Manual entry support
        self.manual_entry_data: Optional[str] = None
        self.was_manually_entered = False

        logger.info(f"BarcodeDetectorV2 initialized (timeout: {timeout}s or continuous)")

    async def initialize(self) -> bool:
        """Initialize barcode detection service.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.barcode_service.is_initialized:
                logger.info("Initializing barcode detection service...")
                success = await self.barcode_service.initialize()
                if not success:
                    logger.error("Failed to initialize barcode detection service")
                    return False

            logger.info("✓ Barcode detector ready")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize barcode detector: {e}", exc_info=True)
            return False

    def set_allowed_types(self, types: List[str]) -> None:
        """Set which barcode types to detect.

        Args:
            types: List of barcode type strings (e.g., ['CODE128', 'QRCODE'])
        """
        self.allowed_types = types
        self.barcode_service.set_allowed_types(types)
        logger.info(f"Barcode detector configured for types: {', '.join(types)}")

    def get_expected_attributes(self) -> List[str]:
        """Get list of attributes this agent should detect.

        Returns:
            List of expected attribute names
        """
        return ["barcode_data", "barcode_type"]

    def validate_result(self, attributes: Dict[str, Any]) -> bool:
        """Validate that result contains expected attributes.

        Args:
            attributes: Attributes dictionary

        Returns:
            True if valid, False otherwise
        """
        return (
            "barcode_data" in attributes and
            attributes["barcode_data"] is not None and
            len(str(attributes["barcode_data"])) > 0
        )

    async def run_with_timer(self) -> Dict[str, Any]:
        """Execute barcode detection with optional timer.

        Overrides base class to implement barcode-specific logic:
        - No timer by default (continuous until detected)
        - Single frame per detection attempt
        - Manual skip support

        Returns:
            Detection result dictionary
        """
        logger.info(f"🔍 Starting barcode detection agent")
        logger.info(f"  Allowed types: {', '.join(self.allowed_types) if self.allowed_types else 'ALL'}")
        logger.info(f"  Timeout: {self.timer_seconds}s" if self.timer_seconds > 0 else "  Timeout: Continuous (no timeout)")

        # Start timer if configured
        if self.timer_seconds > 0:
            self.state.start_timer()
            end_time = time.time() + self.timer_seconds
        else:
            # No timeout - run until detected or manual skip
            self.state.start_timer()  # Still track elapsed time
            end_time = float('inf')

        # Detection loop
        while time.time() < end_time and not self.detection_complete:
            # Check if paused or stopped
            if self.state.status.value != "running":
                logger.info(f"Barcode detector paused or stopped")
                break

            # Update timer (for elapsed time tracking)
            self.state.update_timer()

            # Capture single frame from WebRTC
            if not self.frame_provider:
                logger.error("No frame provider available")
                await asyncio.sleep(0.1)
                continue

            try:
                # Get frame from WebRTC
                if asyncio.iscoroutinefunction(self.frame_provider):
                    frame = await self.frame_provider()
                else:
                    frame = await asyncio.get_event_loop().run_in_executor(None, self.frame_provider)

                if frame is None:
                    await asyncio.sleep(0.1)
                    continue

                # Run barcode detection (2-stage: presence + decode)
                detection = await self.barcode_service.detect_barcode(
                    frame,
                    allowed_types=self.allowed_types if self.allowed_types else None
                )

                if detection:
                    # Barcode detected!
                    self.detected_barcode = detection
                    self.detection_complete = True

                    logger.info(f"\n{'='*70}")
                    logger.info(f"✅ BARCODE DETECTED")
                    logger.info(f"{'='*70}")
                    logger.info(f"  Type: {detection['type']}")
                    logger.info(f"  Data: {detection['data']}")
                    logger.info(f"  Preprocessing: {detection.get('preprocessing', 'N/A')}")
                    logger.info(f"  Stage 1 (Presence): {detection.get('presence_time_ms', 0):.1f}ms")
                    logger.info(f"  Stage 2 (Decode): {detection.get('decode_time_ms', 0):.1f}ms")
                    logger.info(f"  Elapsed Time: {self.state.timer_elapsed:.1f}s")
                    logger.info(f"{'='*70}\n")

                    # Notify via callback if registered
                    if self.on_inference_update:
                        await self.on_inference_update({
                            "agent": self.agent_name,
                            "status": "detected",
                            "barcode_data": detection['data'],
                            "barcode_type": detection['type'],
                            "elapsed_time": self.state.timer_elapsed
                        })

                    break

                # Still searching...
                if self.state.inference_count == 0:
                    # First update
                    if self.on_inference_update:
                        await self.on_inference_update({
                            "agent": self.agent_name,
                            "status": "searching",
                            "elapsed_time": self.state.timer_elapsed
                        })
                    self.state.inference_count = 1

                # Small delay to avoid CPU spinning
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error during barcode detection: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        # Finalize result
        final_result = await self._finalize_result()

        # Mark complete
        self.state.complete()

        # Callback
        if self.on_agent_complete:
            await self.on_agent_complete(self.agent_name, final_result)

        return final_result

    async def _finalize_result(self) -> Dict[str, Any]:
        """Finalize detection result.

        Returns:
            Final result dictionary
        """
        if self.detected_barcode:
            # Successful detection
            return {
                "agent": self.agent_name,
                "status": "detected",
                "barcode_data": self.detected_barcode['data'],
                "barcode_type": self.detected_barcode['type'],
                "confidence": self.detected_barcode.get('confidence', 1.0),
                "preprocessing": self.detected_barcode.get('preprocessing', 'N/A'),
                "elapsed_time": self.state.timer_elapsed,
                "presence_time_ms": self.detected_barcode.get('presence_time_ms', 0),
                "decode_time_ms": self.detected_barcode.get('decode_time_ms', 0),
                "manually_entered": False,
                "validation": "passed"
            }
        elif self.manual_entry_data:
            # Manual entry
            return {
                "agent": self.agent_name,
                "status": "manual_entry",
                "barcode_data": self.manual_entry_data,
                "barcode_type": "MANUAL",
                "confidence": 1.0,
                "elapsed_time": self.state.timer_elapsed,
                "manually_entered": True,
                "validation": "passed"
            }
        else:
            # Skipped or timeout
            return {
                "agent": self.agent_name,
                "status": "skipped",
                "barcode_data": None,
                "barcode_type": None,
                "confidence": 0.0,
                "elapsed_time": self.state.timer_elapsed,
                "manually_entered": False,
                "validation": "failed",
                "message": "Barcode detection skipped or timed out"
            }

    def set_manual_entry(self, barcode_data: str) -> None:
        """Set barcode data from manual entry.

        Args:
            barcode_data: Manually entered barcode string
        """
        self.manual_entry_data = barcode_data
        self.was_manually_entered = True
        self.detection_complete = True
        logger.info(f"Barcode manually entered: {barcode_data}")

    def skip(self) -> None:
        """Skip barcode detection (go to next agent without barcode)."""
        self.detection_complete = True
        logger.info("Barcode detection skipped by user")

    def reset(self) -> None:
        """Reset agent to initial state."""
        super().reset()
        self.detected_barcode = None
        self.manual_entry_data = None
        self.was_manually_entered = False
        self.detection_complete = False
        logger.info("Barcode detector reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state.

        Returns:
            State dictionary
        """
        base_state = super().get_state()
        base_state.update({
            "detected_barcode": self.detected_barcode,
            "manual_entry_data": self.manual_entry_data,
            "was_manually_entered": self.was_manually_entered,
            "detection_complete": self.detection_complete,
            "allowed_types": self.allowed_types
        })
        return base_state
