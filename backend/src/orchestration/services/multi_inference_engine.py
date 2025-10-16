"""
Multi-Inference Engine - GPU Only Version
Streamlined GPU-accelerated multi-image inference without fallbacks
"""

import asyncio
import logging
import time
from typing import List, Any, Dict, Optional
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path

from ..models.inference_result import InferenceResult
from config.gpu_config import gpu_config
from config.frame_config import frame_config
from services.vlm_singleton import vlm_singleton

logger = logging.getLogger(__name__)

class MultiInferenceEngine:
    """GPU-accelerated multi-image inference engine using VLM Singleton"""

    # O(1) lookup set for non-damaging issues (cosmetic only, don't affect resale value)
    NON_DAMAGING_ISSUES = frozenset({
        'wrinkle', 'wrinkles', 'wrinkled',
        'fold', 'folds', 'folded', 'folding',
        'pilling', 'pills',
        'dust', 'dusty',
        'minor dirt', 'light dirt',
        'wrinkled fabric'
    })

    def __init__(self):
        self.gpu_config = gpu_config
        self.frame_config = frame_config

        # Use VLM singleton instead of creating new instance
        self.vlm_singleton = vlm_singleton

        # Error handling settings
        self.max_retries = 2
        self.retry_delay = 1.0  # seconds

        # Performance tracking
        self.inference_history = []
        self.total_inferences = 0
        self.successful_inferences = 0

        # Frame counter for sequential numbering (legacy, kept for backwards compatibility)
        self.frame_counter = 0

        # Track image paths per cycle and agent for export
        self.saved_images = {}  # {cycle_num: {agent_name: [paths]}}
    
    async def initialize_gpu_optimization(self, progress_callback=None) -> bool:
        """Initialize GPU optimization for VLM inference"""
        logger.info("Checking VLM singleton initialization status")

        try:
            # Check if VLM singleton is already initialized
            if self.vlm_singleton.is_ready():
                logger.info("VLM singleton already initialized")
                return True

            # Initialize VLM singleton (will only initialize once)
            result = await self.vlm_singleton.initialize_once(progress_callback)

            if result.get("status") in ["success", "already_initialized"]:
                logger.info("VLM singleton ready for multi-inference")
                return True
            else:
                raise RuntimeError(f"VLM initialization failed: {result}")
                
        except Exception as e:
            logger.error(f"VLM singleton initialization failed: {e}")
            raise RuntimeError(f"VLM singleton initialization failed: {e}")
    
    def is_gpu_optimized(self) -> bool:
        """Check if GPU optimization is active and ready"""
        return self.vlm_singleton.is_ready()
    
    async def run_inference(self, agent_name: str, frames: List[np.ndarray], inference_num: int, previous_context: Dict[str, Any] = None, cycle_num: int = 1, redo_attempt: int = 0, mode: str = "auto", initial_classifier_context: Optional[Dict[str, Any]] = None) -> InferenceResult:
        """Run GPU-accelerated inference for multiple frames"""

        if not self.vlm_singleton.is_ready():
            logger.warning(f"VLM singleton not ready for {agent_name}, attempting initialization...")
            # Initialize VLM singleton if not ready (will only init once)
            try:
                result = await self.vlm_singleton.initialize_once()
                if result.get("status") not in ["success", "already_initialized"]:
                    raise RuntimeError(f"VLM initialization failed: {result}")
                logger.info("VLM singleton initialized successfully on demand")
            except Exception as e:
                logger.error(f"VLM initialization error: {e}")
                raise RuntimeError(f"VLM singleton not initialized: {str(e)}")

        # Prepare frames for batch processing (save to disk AFTER inference to not waste timer)
        frames_to_process = frames.copy()

        # For inference #2 and beyond, include previous frame for batch processing
        if inference_num > 1 and previous_context and previous_context.get('frame_data') is not None:
            # Insert previous frame at the beginning for batch processing
            prev_frame = previous_context.get('frame_data')
            frames_to_process = [prev_frame] + frames_to_process

        # Log inference start
        logger.info(f"VLM inference start: {agent_name} inference #{inference_num}, frames={len(frames)}")

        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                # Get VLM GPU loader from singleton
                vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()

                # Create prompt based on agent name and context
                prompt = self._get_agent_prompt(agent_name, inference_num, previous_context, initial_classifier_context)

                # Use GPU-optimized inference with agent name for better detection
                # Use frames_to_process which includes previous frame for batch processing
                result_data = await vlm_loader.infer_optimized(frames_to_process, prompt, agent_name)
                logger.info(f"[VLM RAW RESPONSE] {agent_name}: {result_data}")
                # Create InferenceResult object
                result = InferenceResult(
                    agent_name=agent_name,
                    inference_num=inference_num,
                    frame_count=len(frames)
                )
                
                # Extract attributes and confidence from result
                attributes = result_data.get("attributes", {})
                confidence = result_data.get("confidence", 0.8)
                reasoning = result_data.get("reasoning", "")

                # Extract inference_metadata if available and add to attributes
                inference_metadata = result_data.get("inference_metadata", {})
                if inference_metadata:
                    attributes['inference_metadata'] = inference_metadata

                # Filter out non-damaging issues for damage_detector agent
                if agent_name == "damage_detector":
                    # Step 1: Filter cosmetic issues (wrinkles, folds, etc.)
                    attributes = self._filter_non_damaging_issues(attributes)

                    # Step 2: Only check bounding box if damage still exists after Step 1
                    # Handle both boolean True and string 'true' from VLM
                    damaged_value = attributes.get('damaged')
                    is_damaged = damaged_value == True or damaged_value == 'true' or damaged_value == 'True'

                    if is_damaged and attributes.get('damage_location'):
                        # Also filter vague detections with oversized bounding boxes
                        # Pass frame shape and agent name to get actual VLM resolution dynamically
                        frame_shape = frames[0].shape if frames else None
                        attributes = self._filter_vague_damage_detections(attributes, frame_shape, agent_name)

                result.complete(attributes, confidence, reasoning)
                
                # Track performance
                inference_time = time.time() - start_time
                self.total_inferences += 1
                self.successful_inferences += 1
                self._record_performance(agent_name, inference_time, "success")

                logger.info(f"VLM inference completed: {agent_name} in {inference_time:.2f}s, attributes={len(attributes)}")

                # Save frames AFTER inference (doesn't impact timer)
                saved_frame_paths = await self._save_debug_frames(
                    agent_name, frames, inference_num, cycle_num=cycle_num, redo_attempt=redo_attempt, mode=mode
                )

                # Save annotated version for damage_detector ONLY if damage passed filters (bbox < 60%)
                # Check for damage in RAW format (before normalization)
                # RAW format: {'damage': {'Hole': [0.45, 0.25, 0.5, 0.3]}}
                # Also check normalized format for backward compatibility
                raw_damage = attributes.get('damage')
                damaged_value = attributes.get('damaged')
                damage_locations = attributes.get('damage_locations')
                damage_location = attributes.get('damage_location')

                # Determine if damaged based on available fields
                is_damaged = False
                if isinstance(raw_damage, dict) and raw_damage:
                    # New raw format from VLM
                    is_damaged = True
                elif damaged_value == True or damaged_value == 'true' or damaged_value == 'True':
                    # Already normalized format
                    is_damaged = True

                # Check if we have location data
                has_damage_location = raw_damage or damage_locations or damage_location

                if agent_name == "damage_detector" and is_damaged and has_damage_location:
                    await self._save_annotated_damage_frames(
                        frames, saved_frame_paths, attributes, cycle_num, inference_num, mode, redo_attempt
                    )

                return result
                
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"GPU inference attempt {attempt + 1} failed for {agent_name}: {e}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    # Final attempt failed
                    inference_time = time.time() - start_time
                    self.total_inferences += 1
                    self._record_performance(agent_name, inference_time, "failed")

                    logger.error(f"VLM inference failed for {agent_name} after {self.max_retries} retries: {e}")
                    
                    # Return error result
                    error_result = InferenceResult(
                        agent_name=agent_name,
                        inference_num=inference_num,
                        frame_count=len(frames)
                    )
                    error_result.complete({}, 0.0, f"GPU inference failed: {str(e)}")
                    error_result.duration_ms = int(inference_time * 1000)
                    return error_result

    async def _save_debug_frames(self, agent_name: str, frames: List[np.ndarray],
                                inference_num: int, cycle_num: int = 1, redo_attempt: int = 0,
                                is_previous: bool = False, is_shared: bool = False, mode: str = "auto") -> List[str]:
        """
        Save frames to disk with enhanced metadata-rich filenames.

        Filename format: {agent}_c{cycle:03d}_i{inf:02d}_{mode}_{suffix}.jpg
        Examples:
            - initial_classifier_c001_i01_auto.jpg (auto mode, main frame)
            - initial_classifier_c001_i01_manual.jpg (manual mode, main frame)
            - initial_classifier_c001_i01_auto_redo_1.jpg (auto mode, first redo)
            - initial_classifier_c001_i01_manual_redo_2.jpg (manual mode, second redo)
            - damage_detector_c001_i01_auto_shared.jpg (auto mode, shared from initial)
        """
        saved_paths = []

        try:
            # Use frame_config to get correct agent directory
            agent_dir = self.frame_config.get_agent_frame_dir(agent_name)

            # Initialize cycle tracking
            if cycle_num not in self.saved_images:
                self.saved_images[cycle_num] = {}
            if agent_name not in self.saved_images[cycle_num]:
                self.saved_images[cycle_num][agent_name] = []

            for idx, frame in enumerate(frames):
                # Build suffix based on context
                suffix_parts = []

                # Add mode (auto/manual) first
                suffix_parts.append(mode)

                if is_previous:
                    suffix_parts.append("prev")
                if redo_attempt > 0:
                    # Add redo with attempt number: redo_1, redo_2, etc.
                    suffix_parts.append(f"redo_{redo_attempt}")
                if is_shared:
                    suffix_parts.append("shared")

                suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

                # Generate filename: {agent}_c{cycle:03d}_i{inf:02d}_{mode}_{suffix}.jpg
                filename = f"{agent_name}_c{cycle_num:03d}_i{inference_num:02d}{suffix}.jpg"
                filepath = agent_dir / filename

                # Save frame using JPEG quality from ENV
                success = cv2.imwrite(
                    str(filepath),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.frame_config.jpeg_quality]
                )

                if success:
                    # Store relative path for CSV export
                    relative_path = f"{agent_name}/{filename}"
                    saved_paths.append(relative_path)

                    # Track in saved_images for this cycle
                    self.saved_images[cycle_num][agent_name].append(relative_path)
                else:
                    logger.error(f"Failed to save frame: {filepath}")

        except Exception as e:
            logger.error(f"Error saving debug frames for {agent_name}: {e}")

        return saved_paths

    def get_saved_images_for_cycle(self, cycle_num: int) -> Dict[str, List[str]]:
        """
        Get all saved image paths for a specific cycle.

        Returns:
            Dict mapping agent_name to list of image paths
            Example: {
                "initial_classifier": ["initial_classifier/initial_classifier_c001_i01.jpg", ...],
                "detail_extractor": ["detail_extractor/detail_extractor_c001_i01.jpg"],
                "damage_detector": ["damage_detector/damage_detector_c001_i01.jpg"]
            }
        """
        return self.saved_images.get(cycle_num, {})

    def get_final_images_for_cycle(self, cycle_num: int) -> Dict[str, list]:
        """
        Get ALL images from the FINAL attempt for each agent in a cycle.

        If redos exist, returns all images from the last redo attempt.
        If no redos, returns all images from the original run.

        Returns:
            Dict mapping agent_name to list of final image paths
            Example: {
                "initial_classifier": ["initial_classifier_c001_i01_redo_2.jpg", "...i02_redo_2.jpg"],
                "detail_extractor": ["detail_extractor_c001_i01.jpg", "...i02.jpg"]
            }
        """
        cycle_images = self.saved_images.get(cycle_num, {})
        final_images = {}

        for agent_name, image_list in cycle_images.items():
            if not image_list:
                continue

            # Filter out "_prev" images (batch processing context, not actual outputs)
            main_images = [img for img in image_list if "_prev" not in img]

            if not main_images:
                continue

            # Find the highest redo number in the images
            import re
            highest_redo = 0
            for img in main_images:
                redo_match = re.search(r'_redo_(\d+)', img)
                if redo_match:
                    redo_num = int(redo_match.group(1))
                    highest_redo = max(highest_redo, redo_num)

            # Filter images from the final attempt
            if highest_redo > 0:
                # Get all images from the last redo attempt
                final_attempt_images = [
                    img for img in main_images
                    if f"_redo_{highest_redo}" in img
                ]
            else:
                # No redos - get all non-redo images (original run)
                final_attempt_images = [
                    img for img in main_images
                    if "_redo_" not in img
                ]

            if final_attempt_images:
                final_images[agent_name] = final_attempt_images

        return final_images

    def clear_cycle_images(self, cycle_num: int):
        """Clear tracked images for a cycle (called after export)"""
        if cycle_num in self.saved_images:
            del self.saved_images[cycle_num]

    async def _save_annotated_damage_frames(self, frames: List[np.ndarray], saved_paths: List[str],
                                            attributes: Dict[str, Any], cycle_num: int, inference_num: int,
                                            mode: str, redo_attempt: int):
        """
        Save annotated versions of damage detector frames with bounding boxes drawn.

        Handles coordinate scaling from VLM-processed resolution (768x691) to original frame resolution.
        Supports both old format (damage_location) and new format (damage_locations dict).

        Args:
            frames: Original frame data (numpy arrays at original resolution, e.g., 2400x2160)
            saved_paths: Paths where original frames were saved
            attributes: Damage detection attributes including damage_location or damage_locations
            cycle_num: Cycle number
            inference_num: Inference number
            mode: Mode (auto/manual)
            redo_attempt: Redo attempt number
        """
        try:
            # Check for RAW format first (from VLM before normalization)
            raw_damage = attributes.get('damage')
            if raw_damage and isinstance(raw_damage, dict):
                # RAW VLM format: {'damage': {'Hole': [0.45, 0.25, 0.5, 0.3]}}
                logger.info(f"Using RAW damage format with {len(raw_damage)} damage type(s)")
                await self._save_annotated_frames_new_format(
                    frames, saved_paths, raw_damage, cycle_num, inference_num, mode, redo_attempt
                )
                return

            # Check for normalized damage_locations format
            damage_locations = attributes.get('damage_locations')
            if damage_locations and isinstance(damage_locations, dict):
                # Normalized format: {damage_type: bbox, ...}
                logger.info(f"Using normalized damage_locations format with {len(damage_locations)} damage type(s)")
                await self._save_annotated_frames_new_format(
                    frames, saved_paths, damage_locations, cycle_num, inference_num, mode, redo_attempt
                )
                return

            # Fall back to old format
            damage_location = attributes.get('damage_location')
            damage_type = attributes.get('damage_type', 'damage')

            if not damage_location:
                logger.debug("No damage_location found, skipping annotation")
                return

            logger.info("Using old damage_location format")
            # Parse damage location - handle string or list format, supporting multiple bboxes
            bboxes = []  # List of bounding boxes to draw

            if isinstance(damage_location, str):
                import re
                # Handle multiple bboxes: "[x1,y1,x2,y2], [x1,y1,x2,y2], ..."
                bbox_pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
                matches = re.findall(bbox_pattern, damage_location)

                if matches:
                    # Found formatted bboxes like "[313, 100, 342, 124], [342, 100, 370, 124]"
                    bboxes = [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
                    logger.debug(f"Parsed {len(bboxes)} bounding boxes from string format")
                else:
                    # Fallback: try to parse as single bbox with comma-separated numbers
                    numbers = re.findall(r'[\d.]+', damage_location)
                    if len(numbers) >= 4:
                        # Group numbers into sets of 4 for multiple bboxes
                        for i in range(0, len(numbers) - 3, 4):
                            bboxes.append([float(n) for n in numbers[i:i+4]])
                        logger.debug(f"Parsed {len(bboxes)} bounding boxes from comma-separated format")
                    else:
                        logger.warning(f"Invalid damage_location string format: {damage_location}")
                        return

            elif isinstance(damage_location, list):
                if len(damage_location) >= 4 and all(isinstance(x, (int, float)) for x in damage_location[:4]):
                    # Single bbox as flat list: [x1, y1, x2, y2]
                    bboxes = [[float(n) for n in damage_location[:4]]]
                elif all(isinstance(item, list) and len(item) >= 4 for item in damage_location):
                    # Multiple bboxes as list of lists: [[x1,y1,x2,y2], [x1,y1,x2,y2]]
                    bboxes = [[float(n) for n in bbox[:4]] for bbox in damage_location]
                    logger.debug(f"Parsed {len(bboxes)} bounding boxes from list format")
                else:
                    logger.warning(f"Invalid damage_location list format: {damage_location}")
                    return
            else:
                logger.warning(f"Invalid damage_location format: {type(damage_location)}")
                return

            if not bboxes:
                logger.warning("No valid bounding boxes found")
                return

            # Get agent directory
            agent_dir = self.frame_config.get_agent_frame_dir("damage_detector")

            # VLM processes images at this resolution (from vlm_gpu_loader.py)
            VLM_RESIZE_WIDTH = 768
            VLM_RESIZE_HEIGHT = 691

            for idx, (frame, saved_path) in enumerate(zip(frames, saved_paths)):
                try:
                    # Read the saved original frame
                    import cv2
                    from PIL import Image, ImageDraw, ImageFont

                    # Build filename for annotated version
                    # Original: damage_detector_c001_i01_auto.jpg
                    # Annotated: damage_detector_c001_i01_auto_annotated.jpg
                    original_filename = saved_path.split('/')[-1]  # Get filename from path
                    base_name = original_filename.rsplit('.', 1)[0]  # Remove extension
                    annotated_filename = f"{base_name}_annotated.jpg"
                    annotated_filepath = agent_dir / annotated_filename

                    # Convert frame to PIL Image for drawing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_img)

                    # Get actual frame dimensions
                    img_width, img_height = pil_img.size

                    # Calculate scale factors once (O(1) operation)
                    scale_x = img_width / VLM_RESIZE_WIDTH
                    scale_y = img_height / VLM_RESIZE_HEIGHT

                    # Load font once (O(1))
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    except:
                        font = None  # Use default font

                    # Draw all bounding boxes (O(N) total where N = number of boxes, O(1) per box)
                    bbox_count = len(bboxes)
                    for bbox_idx, bbox in enumerate(bboxes):
                        # Parse bounding box coordinates (O(1))
                        x1, y1, x2, y2 = bbox[:4]

                        # Determine coordinate type and scale appropriately (O(1))
                        if all(0 <= v <= 1 for v in bbox):
                            # Case 1: Normalized coordinates (0-1 range)
                            x1_px = int(x1 * img_width)
                            y1_px = int(y1 * img_height)
                            x2_px = int(x2 * img_width)
                            y2_px = int(y2 * img_height)
                            logger.debug(f"Box {bbox_idx+1}/{bbox_count}: Normalized coords {bbox} → Pixels [{x1_px}, {y1_px}, {x2_px}, {y2_px}]")

                        elif all(v <= max(VLM_RESIZE_WIDTH, VLM_RESIZE_HEIGHT) for v in bbox):
                            # Case 2: Pixel coordinates from VLM-resized image (768x691)
                            # Scale up to original resolution using pre-calculated factors (O(1))
                            x1_px = int(x1 * scale_x)
                            y1_px = int(y1 * scale_y)
                            x2_px = int(x2 * scale_x)
                            y2_px = int(y2 * scale_y)
                            logger.debug(f"Box {bbox_idx+1}/{bbox_count}: VLM coords {bbox} → Scaled [{x1_px}, {y1_px}, {x2_px}, {y2_px}]")

                        else:
                            # Case 3: Pixel coordinates already at original resolution
                            x1_px = int(x1)
                            y1_px = int(y1)
                            x2_px = int(x2)
                            y2_px = int(y2)
                            logger.debug(f"Box {bbox_idx+1}/{bbox_count}: Original pixel coords {bbox}")

                        # Clamp coordinates to image bounds (O(1) operations)
                        x1_px = max(0, min(x1_px, img_width - 1))
                        y1_px = max(0, min(y1_px, img_height - 1))
                        x2_px = max(0, min(x2_px, img_width))
                        y2_px = max(0, min(y2_px, img_height))

                        # Validate bounding box (O(1))
                        if x2_px <= x1_px or y2_px <= y1_px:
                            logger.warning(f"Box {bbox_idx+1}/{bbox_count}: Invalid after scaling [{x1_px}, {y1_px}, {x2_px}, {y2_px}] - skipping")
                            continue

                        # Draw bounding box (O(1))
                        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline='red', width=4)

                    # Add label with damage type and count (drawn once on first bbox)
                    if bboxes:
                        first_bbox = bboxes[0]
                        x1, y1, x2, y2 = first_bbox[:4]

                        # Scale first bbox coordinates for label placement
                        if all(0 <= v <= 1 for v in first_bbox):
                            x1_px = int(x1 * img_width)
                            y1_px = int(y1 * img_height)
                        elif all(v <= max(VLM_RESIZE_WIDTH, VLM_RESIZE_HEIGHT) for v in first_bbox):
                            x1_px = int(x1 * scale_x)
                            y1_px = int(y1 * scale_y)
                        else:
                            x1_px = int(x1)
                            y1_px = int(y1)

                        # Create label with count if multiple boxes
                        if bbox_count > 1:
                            label_text = f"{damage_type} ({bbox_count})"
                        else:
                            label_text = f"{damage_type}"

                        # Draw text with background (O(1))
                        text_bbox = draw.textbbox((x1_px + 5, y1_px - 30), label_text, font=font)
                        draw.rectangle(text_bbox, fill='red')
                        draw.text((x1_px + 5, y1_px - 30), label_text, fill='white', font=font)

                    # Convert back to BGR for OpenCV saving
                    annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # Save annotated frame
                    success = cv2.imwrite(
                        str(annotated_filepath),
                        annotated_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.frame_config.jpeg_quality]
                    )

                    if success:
                        logger.info(f"Annotated frame saved: damage_detector/{annotated_filename}, resolution={img_width}x{img_height}, boxes={bbox_count}")
                    else:
                        logger.error(f"Failed to save annotated frame: {annotated_filepath}")

                except Exception as frame_error:
                    logger.error(f"Error annotating frame {idx}: {frame_error}")
                    continue

        except Exception as e:
            logger.error(f"Error saving annotated damage frames: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _save_annotated_frames_new_format(self, frames: List[np.ndarray], saved_paths: List[str],
                                                 damage_locations: Dict[str, Any], cycle_num: int,
                                                 inference_num: int, mode: str, redo_attempt: int):
        """
        Save annotated frames using new format with individual damage type labels.

        New format: {"Hole": [0.45, 0.25, 0.5, 0.3], "Stain": [0.1, 0.2, 0.3, 0.4]}

        Args:
            frames: Original frame data
            saved_paths: Paths where original frames were saved
            damage_locations: Dict mapping damage types to bounding boxes
            cycle_num: Cycle number
            inference_num: Inference number
            mode: Mode (auto/manual)
            redo_attempt: Redo attempt number
        """
        try:
            # Get agent directory
            agent_dir = self.frame_config.get_agent_frame_dir("damage_detector")

            # VLM processes images at this resolution (from vlm_gpu_loader.py)
            VLM_RESIZE_WIDTH = 768
            VLM_RESIZE_HEIGHT = 691

            for idx, (frame, saved_path) in enumerate(zip(frames, saved_paths)):
                try:
                    import cv2
                    from PIL import Image, ImageDraw, ImageFont

                    # Build filename for annotated version
                    original_filename = saved_path.split('/')[-1]
                    base_name = original_filename.rsplit('.', 1)[0]
                    annotated_filename = f"{base_name}_annotated.jpg"
                    annotated_filepath = agent_dir / annotated_filename

                    # Convert frame to PIL Image for drawing
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_img)

                    # Get actual frame dimensions
                    img_width, img_height = pil_img.size

                    # Calculate scale factors once
                    scale_x = img_width / VLM_RESIZE_WIDTH
                    scale_y = img_height / VLM_RESIZE_HEIGHT

                    # Load font once
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                    except:
                        font = None  # Use default font

                    # Draw each damage type with its own bounding box and label
                    damage_count = len(damage_locations)
                    logger.info(f"Drawing {damage_count} damage annotations")

                    for damage_type, bbox in damage_locations.items():
                        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                            logger.warning(f"Invalid bbox for {damage_type}: {bbox}")
                            continue

                        # Parse bounding box coordinates
                        x1, y1, x2, y2 = bbox[:4]

                        # Determine coordinate type and scale appropriately
                        if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
                            # Case 1: Normalized coordinates (0-1 range)
                            x1_px = int(x1 * img_width)
                            y1_px = int(y1 * img_height)
                            x2_px = int(x2 * img_width)
                            y2_px = int(y2 * img_height)
                            logger.debug(f"{damage_type}: Normalized coords {bbox} → Pixels [{x1_px}, {y1_px}, {x2_px}, {y2_px}]")

                        elif all(v <= max(VLM_RESIZE_WIDTH, VLM_RESIZE_HEIGHT) for v in [x1, y1, x2, y2]):
                            # Case 2: Pixel coordinates from VLM-resized image
                            x1_px = int(x1 * scale_x)
                            y1_px = int(y1 * scale_y)
                            x2_px = int(x2 * scale_x)
                            y2_px = int(y2 * scale_y)
                            logger.debug(f"{damage_type}: VLM coords {bbox} → Scaled [{x1_px}, {y1_px}, {x2_px}, {y2_px}]")

                        else:
                            # Case 3: Pixel coordinates already at original resolution
                            x1_px = int(x1)
                            y1_px = int(y1)
                            x2_px = int(x2)
                            y2_px = int(y2)
                            logger.debug(f"{damage_type}: Original pixel coords {bbox}")

                        # Clamp coordinates to image bounds
                        x1_px = max(0, min(x1_px, img_width - 1))
                        y1_px = max(0, min(y1_px, img_height - 1))
                        x2_px = max(0, min(x2_px, img_width))
                        y2_px = max(0, min(y2_px, img_height))

                        # Validate bounding box
                        if x2_px <= x1_px or y2_px <= y1_px:
                            logger.warning(f"{damage_type}: Invalid bbox after scaling [{x1_px}, {y1_px}, {x2_px}, {y2_px}] - skipping")
                            continue

                        # Draw bounding box
                        draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline='red', width=4)

                        # Draw individual label for this damage type
                        label_text = damage_type
                        text_bbox = draw.textbbox((x1_px + 5, y1_px - 30), label_text, font=font)
                        draw.rectangle(text_bbox, fill='red')
                        draw.text((x1_px + 5, y1_px - 30), label_text, fill='white', font=font)

                    # Convert back to BGR for OpenCV saving
                    annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                    # Save annotated frame
                    success = cv2.imwrite(
                        str(annotated_filepath),
                        annotated_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.frame_config.jpeg_quality]
                    )

                    if success:
                        logger.info(f"Annotated frame saved: damage_detector/{annotated_filename}, resolution={img_width}x{img_height}, types={damage_count}")
                    else:
                        logger.error(f"Failed to save annotated frame: {annotated_filepath}")

                except Exception as frame_error:
                    logger.error(f"Error annotating frame {idx}: {frame_error}")
                    continue

        except Exception as e:
            logger.error(f"Error saving annotated damage frames (new format): {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _filter_non_damaging_issues(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out non-damaging cosmetic issues from damage detection results.
        Uses O(1) set lookup for efficiency.

        Args:
            attributes: Damage detector attributes with 'damaged' and 'damage_type' keys

        Returns:
            Filtered attributes with non-damaging issues removed
        """
        if not attributes or 'damage_type' not in attributes:
            return attributes

        damage_type = attributes.get('damage_type')

        # If no damage detected, return as-is
        if not damage_type or damage_type == 'null' or damage_type == 'NULL':
            return attributes

        # Handle both list and string formats
        if isinstance(damage_type, list):
            # Already a list - convert each item to lowercase and strip
            damage_list = [str(d).strip().lower() for d in damage_type if d]
        else:
            # String format - convert to lowercase and parse comma-separated
            damage_type_lower = str(damage_type).lower()
            damage_list = [d.strip() for d in damage_type_lower.split(',') if d.strip()]

        # Filter out non-damaging issues using O(1) set lookup
        real_damages = [d for d in damage_list if d not in self.NON_DAMAGING_ISSUES]

        # Update attributes based on filtered results
        filtered_attrs = attributes.copy()

        if not real_damages:
            # All damages were cosmetic - mark as undamaged
            filtered_attrs['damaged'] = False
            filtered_attrs['damage_type'] = None
            filtered_attrs['damage_location'] = None  # Clear bbox to prevent annotation
            logger.info(f"Filtered out cosmetic issues: {damage_type} → No damage")
        else:
            # Real damages remain - preserve original format (list or string)
            if isinstance(damage_type, list):
                filtered_attrs['damage_type'] = real_damages
            else:
                filtered_attrs['damage_type'] = ', '.join(real_damages)
            logger.info(f"Filtered damage types: {damage_type} → {filtered_attrs['damage_type']}")

        return filtered_attrs

    def _calculate_vlm_dimensions(self, agent_name: str, frame_shape: tuple) -> tuple:
        """
        Calculate actual VLM dimensions after thumbnail operation.
        Replicates the same logic used in vlm_gpu_loader.py

        Args:
            agent_name: Name of the agent (e.g., 'damage_detector')
            frame_shape: Original frame shape (height, width, channels)

        Returns:
            Tuple of (vlm_width, vlm_height)
        """
        # Determine target resolution based on agent (from vlm_gpu_loader.py)
        if agent_name in ["initial_classifier"]:
            target_resolution = (448, 448)
        elif agent_name in ["damage_detector"]:
            target_resolution = (768, 768)
        elif agent_name == "detail_extractor":
            target_resolution = (768, 768)
        else:
            target_resolution = (512, 512)

        # Get original dimensions from frame_shape
        orig_height, orig_width = frame_shape[0], frame_shape[1]
        target_w, target_h = target_resolution

        # PIL's thumbnail logic: preserves aspect ratio, fits within target
        # Calculate aspect ratio
        aspect = orig_width / orig_height

        # Determine which dimension is the limiting factor
        if aspect > 1:
            # Width is larger - limit by width
            vlm_width = target_w
            vlm_height = int(target_w / aspect)
        else:
            # Height is larger or equal - limit by height
            vlm_height = target_h
            vlm_width = int(target_h * aspect)

        return vlm_width, vlm_height

    def _filter_vague_damage_detections(self, attributes: Dict[str, Any], frame_shape: tuple = None, agent_name: str = "damage_detector") -> Dict[str, Any]:
        """
        Filter out vague damage detections with overly large bounding boxes.
        If a bounding box covers >60% of the image, it's likely a false positive.

        Args:
            attributes: Damage detector attributes with 'damage_location' key
            frame_shape: Optional numpy array shape (height, width, channels) for dynamic resolution
            agent_name: Agent name for determining VLM target resolution

        Returns:
            Filtered attributes with vague detections removed
        """
        if not attributes or 'damage_location' not in attributes:
            return attributes

        damage_location = attributes.get('damage_location')

        # If no damage location, return as-is
        if not damage_location:
            return attributes

        # Parse damage_location - handle string or list format
        if isinstance(damage_location, str):
            # Parse string format like "[0.0, 0.0, 0.822, 0.984]"
            import re
            numbers = re.findall(r'[\d.]+', damage_location)
            if len(numbers) >= 4:
                bbox = [float(n) for n in numbers[:4]]
            else:
                return attributes
        elif isinstance(damage_location, list) and len(damage_location) >= 4:
            bbox = damage_location
        else:
            return attributes

        # Calculate bounding box coverage
        # bbox format: [x1, y1, x2, y2] - can be normalized (0-1) or pixel coords
        x1, y1, x2, y2 = bbox[:4]

        # Determine if normalized (0-1) or pixel coordinates
        if all(0 <= v <= 1 for v in bbox):
            # Normalized coordinates
            width = x2 - x1
            height = y2 - y1
        else:
            # Pixel coordinates - use VLM dimensions, NOT original frame dimensions
            if frame_shape is not None:
                # Calculate actual VLM dimensions using same method as vlm_gpu_loader.py
                img_width, img_height = self._calculate_vlm_dimensions(agent_name, frame_shape)
                logger.debug(f"Using dynamically calculated VLM dimensions: {img_width}x{img_height} (agent: {agent_name})")
            else:
                # Fallback to common damage_detector resolution if frame shape not provided
                img_width, img_height = 768, 691
                logger.warning(f"Frame shape not provided, using fallback resolution: {img_width}x{img_height}")

            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

        # Calculate area coverage as percentage
        area_coverage = width * height

        # Threshold: Reject if bbox covers >60% of image
        VAGUE_DETECTION_THRESHOLD = 0.60

        filtered_attrs = attributes.copy()

        if area_coverage > VAGUE_DETECTION_THRESHOLD:
            # Bounding box too large - likely a false positive
            filtered_attrs['damaged'] = False
            filtered_attrs['damage_type'] = None
            filtered_attrs['damage_location'] = None
            resolution_info = f"{img_width}x{img_height}" if 'img_width' in locals() else "normalized"
            logger.info(f"Rejected vague damage detection: bbox covers {area_coverage*100:.1f}% of {resolution_info} image (threshold: {VAGUE_DETECTION_THRESHOLD*100:.0f}%)")

        return filtered_attrs

    def _filter_garment_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant garment attributes for damage detection context.
        Only includes: type, color, pattern, closure.
        Skips null/empty/n/a values.

        Complexity: O(1) - fixed set of 4 attributes

        Args:
            attributes: Initial classifier attributes

        Returns:
            Filtered dict with only relevant, non-null attributes
        """
        valid_keys = {'type', 'color', 'pattern', 'closure'}
        skip_values = {'none', 'no', 'n/a', 'not applicable', 'null', '', 'unknown'}

        filtered = {}
        for key in valid_keys:
            value = attributes.get(key)
            # Skip if value is None or in skip list (case-insensitive)
            if value is not None:
                value_str = str(value).lower().strip()
                if value_str and value_str not in skip_values:
                    filtered[key] = value

        return filtered

    def _get_agent_prompt(self, agent_name: str, inference_num: int, previous_context: Dict[str, Any] = None, initial_classifier_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate appropriate prompt based on agent name and context"""
        prompts = {
            "initial_classifier": "You are a state-of-the-art fashion vision analyst. Inspect the entire garment carefully before answering. Determine garment type, color, pattern, neckline, sleeve_length, and closure. Reason step-by-step about silhouette, fabric cues, lighting, and camera angle to validate each attribute. Type, color, pattern are required. If any attribute cannot be confidently inferred, return null for that attribute.",
            "detail_extractor": "You are a state-of-the-art garment label specialist. Analyze all tags, labels, and printed regions at multiple scales to detect brand name and size. Use advanced OCR reasoning to disambiguate stylized fonts or partial views. Validate the brand name with your knowledge of fashion brands. If brand or size cannot be read, return null.",
            "damage_detector": """Analyze the given garment image and ignore any other objects/items other than the garment. Find PHYSICAL DAMAGE affecting resale.

STEP 1 - FOLLOW HANDS: If hands present → they show damage location. Examine that area FIRST at maximum zoom.

STEP 2 - SCAN HARDWARE (inspect each individually):
• Belt loops/strings: Frayed edges? Cut/torn fabric? Loose threads?
• Buttons: Cracks? Chips? Missing? Loose attachment?
• Zippers: Broken teeth? Stuck? Bent pull tab?
• Fasteners/snaps: Bent? Broken? Missing?
Scan from pixel level to find each damage with its location.

STEP 3 - SCAN FABRIC (top→bottom):
• Stains, holes, tears, cuts, abrasions, discoloration
• Hair, fuzz, lint, dirty areas
• Is the apparel color faded?

STEP 4 - CLASSIFY & LOCATE:
Damage types: Stain, Hole, Tear, Cut, Color fade, Scratch, Discoloration, Hair, Lint, Dirty, Frayed Belt Loop/String, Damaged/Missing Button, Broken Zipper, Damaged Fastener... etc.
Not limited to the list above.
NOT damage: Pockets, seams, patterns, wrinkles, folds, intact hardware
Mark bbox [x1,y1,x2,y2] NORMALIZED (0.0-1.0 ONLY). Example: 400px ÷ 800px width = 0.5
Multiple damage_types can have same name but UNIQUE LOCATIONS. Each damage location should be unique with only one entry per damage type and should only have 4 normalized coordinates [x1, y1, x2, y2]. (normalized coordinates 0.0-1.0)
JSON:
{
  "damage": {
    "type1": [x_min, y_min, x_max, y_max], (normalized coordinates 0.0-1.0)
    "type2": [x_min, y_min, x_max, y_max], (normalized coordinates 0.0-1.0)
    ...
  }
}""",
            "final_compiler": "You are a state-of-the-art fashion assistant consolidating prior analyses. Review all available attributes and reconcile conflicts using best-evidence reasoning. Produce the final structured classification, ensuring every field is justified by visual cues or prior agent outputs, and mark any unknown attribute as null."
        }

        base_prompt = prompts.get(agent_name, "Analyze this image of a garment")

        # Add context for inference #2 and beyond (not for final_compiler)
        if inference_num > 1 and previous_context and agent_name != "final_compiler":
            prev_attrs = previous_context.get('attributes', {})
            prev_confidence = previous_context.get('confidence', 0.0)

            # Build context string based on agent type
            context_str = "\n\nContext from your previous analysis:"

            if agent_name == "initial_classifier":
                # For initial classifier, mention previous garment attributes
                if prev_attrs:
                    context_str += f"\n- Previous: {', '.join([f'{k}={v}' for k, v in prev_attrs.items() if v])}"
                context_str += "\nAnalyzing 2 images. Confirm or update attributes."
                base_prompt = "You are a state-of-the-art fashion vision analyst. Inspect the entire garment carefully before answering. Determine garment type, color, pattern, neckline, sleeve_length, and closure. Reason step-by-step about silhouette, fabric cues, lighting, and camera angle to validate each attribute. Type, color, pattern are required. If any attribute cannot be confidently inferred, return null for that attribute."
                base_prompt += context_str
            elif agent_name == "detail_extractor":
                # For detail extractor, mention if brand/size was found before
                brand = prev_attrs.get('brand', 'not found')
                size = prev_attrs.get('size', 'not found')
                context_str += f"\n- Previous: brand={brand}, size={size}"
                context_str += "\n2 images. Check both for labels."
                base_prompt = "You are a state-of-the-art garment label specialist. Analyze all tags, labels, and printed regions at multiple scales to detect brand name and size. Use advanced OCR reasoning to disambiguate stylized fonts or partial views. Validate the brand name with your knowledge of fashion brands. If brand or size cannot be read, return null."
                base_prompt += context_str
            elif agent_name == "damage_detector":
                # For damage detector, mention previous damage findings
                damaged = prev_attrs.get('damaged', 'unknown')
                damage_type = prev_attrs.get('damage_type', 'none')
                context_str += f"\n- Previous: damaged={damaged}, type={damage_type}"
                context_str += "\n2 images. Confirm damage."
                base_prompt += context_str
            
            

        # Add garment context for damage_detector if available
        if agent_name == "damage_detector" and initial_classifier_context:
            filtered_context = self._filter_garment_attributes(initial_classifier_context)
            if filtered_context:
                context_str = f"\nGarment: {', '.join([f'{k}={v}' for k, v in filtered_context.items()])}"
                context_str += "\nUse context to distinguish design features from damage."
                base_prompt += context_str
                logger.debug(f"Added garment context to damage_detector prompt")
        base_prompt += " JSON format."
        return base_prompt
    
    def _record_performance(self, agent_name: str, inference_time: float, status: str):
        """Record performance metrics"""
        performance_record = {
            "timestamp": time.time(),
            "agent_name": agent_name,
            "inference_time": inference_time,
            "status": status
        }
        
        self.inference_history.append(performance_record)
        
        # Keep only last 100 records
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-100:]
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get GPU optimization status"""
        
        vlm_status = self.vlm_singleton.get_status()
        
        base_status = {
            "gpu_optimized": True,
            "engine_ready": self.vlm_singleton.is_ready(),
            "vlm_state": vlm_status.get("state"),
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "success_rate": (self.successful_inferences / max(1, self.total_inferences)) * 100
        }
        
        # Add VLM singleton metrics
        if self.vlm_singleton.is_ready():
            vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()
            vlm_metrics = vlm_loader.get_performance_stats()
            base_status.update({
                "vlm_metrics": vlm_metrics,
                "optimization_level": vlm_metrics.get("optimization_level", "unknown"),
                "avg_inference_time_ms": vlm_metrics.get("avg_inference_time", 0)
            })
        
        return base_status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of GPU-optimized inference engine"""
        
        try:
            # Check VLM singleton health
            if self.vlm_singleton.is_ready():
                vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()
                vlm_health = await vlm_loader.health_check()
                engine_healthy = vlm_health.get("status") == "healthy"
            else:
                vlm_health = {"status": "error", "message": "VLM singleton not initialized"}
                engine_healthy = False
            
            # Calculate performance health
            success_rate = (self.successful_inferences / max(1, self.total_inferences)) * 100
            performance_healthy = success_rate > 80  # 80% success rate threshold
            
            overall_status = "healthy" if (engine_healthy and performance_healthy) else "degraded"
            
            return {
                "status": overall_status,
                "engine_healthy": engine_healthy,
                "performance_healthy": performance_healthy,
                "success_rate": success_rate,
                "total_inferences": self.total_inferences,
                "vlm_health": vlm_health
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Shutdown GPU-optimized inference engine"""
        logger.info("Shutting down GPU-optimized multi-inference engine")

        try:
            # VLM singleton manages its own lifecycle
            # No need to shutdown as it's shared across the application
            logger.info("GPU-optimized multi-inference engine shutdown completed")

        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last period"""
        
        if not self.inference_history:
            return {"message": "No inference history available"}
        
        # Calculate metrics from recent history
        recent_inferences = self.inference_history[-20:]  # Last 20 inferences
        
        successful = [r for r in recent_inferences if r["status"] == "success"]
        total_time = sum(r["inference_time"] for r in successful)
        avg_time = total_time / len(successful) if successful else 0
        
        # Group by agent
        agent_stats = {}
        for record in recent_inferences:
            agent = record["agent_name"]
            if agent not in agent_stats:
                agent_stats[agent] = {"count": 0, "avg_time": 0, "success_count": 0}
            
            agent_stats[agent]["count"] += 1
            if record["status"] == "success":
                agent_stats[agent]["success_count"] += 1
                agent_stats[agent]["avg_time"] = (
                    (agent_stats[agent]["avg_time"] * (agent_stats[agent]["success_count"] - 1) + 
                     record["inference_time"]) / agent_stats[agent]["success_count"]
                )
        
        return {
            "total_inferences": len(recent_inferences),
            "successful_inferences": len(successful),
            "success_rate": (len(successful) / len(recent_inferences)) * 100,
            "avg_inference_time": avg_time,
            "agent_statistics": agent_stats,
            "gpu_optimized": True
        }
