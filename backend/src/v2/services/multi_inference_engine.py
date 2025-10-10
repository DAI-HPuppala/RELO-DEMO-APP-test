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
        'wrinkle', 'wrinkles', 'wrinkled', 'crease', 'creases', 'creased', 'creasing',
        'fold', 'folds', 'folded', 'folding',
        'pilling', 'pills',
        'dust', 'dusty',
        'minor dirt', 'light dirt',
        'minor fading',
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
        logger.info("🚀 Checking VLM singleton initialization status...")
        
        try:
            # Check if VLM singleton is already initialized
            if self.vlm_singleton.is_ready():
                logger.info("✅ VLM singleton already initialized - no warmup needed")
                return True
            
            # Initialize VLM singleton (will only initialize once)
            result = await self.vlm_singleton.initialize_once(progress_callback)
            
            if result.get("status") in ["success", "already_initialized"]:
                logger.info("✅ VLM singleton ready for multi-inference")
                return True
            else:
                raise RuntimeError(f"VLM initialization failed: {result}")
                
        except Exception as e:
            logger.error(f"VLM singleton initialization failed: {e}")
            raise RuntimeError(f"VLM singleton initialization failed: {e}")
    
    def is_gpu_optimized(self) -> bool:
        """Check if GPU optimization is active and ready"""
        is_ready = self.vlm_singleton.is_ready()
        logger.debug(f"VLM singleton ready status: {is_ready}")
        return is_ready
    
    async def run_inference(self, agent_name: str, frames: List[np.ndarray], inference_num: int, previous_context: Dict[str, Any] = None, cycle_num: int = 1, redo_attempt: int = 0, mode: str = "auto") -> InferenceResult:
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

        # Clear logging for debugging
        logger.info("="*80)
        logger.info(f"🎯 INFERENCE START: {agent_name.upper()} - Inference #{inference_num}")
        logger.info("="*80)

        if inference_num == 1:
            logger.info(f"📸 FRAME CAPTURED: {len(frames)} {'frame' if len(frames) == 1 else 'frames'} captured from camera (first inference)")
        else:
            logger.info(f"📸 FRAME CAPTURED: 1 new frame captured for inference #{inference_num}")
            if previous_context and previous_context.get('frame_data') is not None:
                logger.info(f"🖼️ BATCH PROCESSING: Using 2 images (1 previous + 1 new) for context-aware inference")
                logger.info(f"📚 CONTEXT: Using results and image from inference #{previous_context.get('inference_num', inference_num-1)}")
                # Log the actual context being used
                logger.info("*" * 30 + " CONTEXT " + "*" * 30)
                prev_attrs = previous_context.get('attributes', {})
                if prev_attrs:
                    logger.info(f"Previous Attributes: {prev_attrs}")
                logger.info(f"Previous Confidence: {previous_context.get('confidence', 0.0):.1%}")
                if previous_context.get('reasoning'):
                    logger.info(f"Previous Reasoning: {previous_context.get('reasoning')}")
                logger.info("*" * 69)

        if inference_num > 1 and previous_context and previous_context.get('frame_data') is not None:
            logger.info(f"  📁 PREVIOUS Frame: Used from inference #{previous_context.get('inference_num')} (in memory)")
            logger.info(f"  🔍 Batch processing with 2 images (previous + new) for {agent_name} inference #{inference_num}")
            logger.info("*" * 30 + f" BATCH DETAILS FOR {agent_name.upper()} " + "*" * 30)
            logger.info(f"  Image 1: Previous frame from inference #{previous_context.get('inference_num')}")
            logger.info(f"  Image 2: New frame just captured")
            logger.info(f"  Previous Results: {previous_context.get('attributes')}")
            logger.info("*" * (60 + len(agent_name) + 14))
        else:
            logger.info(f"  🔍 This frame will be used for {agent_name} inference #{inference_num}")
        logger.info("-"*80)

        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                # Get VLM GPU loader from singleton
                vlm_loader = self.vlm_singleton.get_vlm_gpu_loader()

                # Create prompt based on agent name and context
                prompt = self._get_agent_prompt(agent_name, inference_num, previous_context)

                # Use GPU-optimized inference with agent name for better detection
                # Use frames_to_process which includes previous frame for batch processing
                result_data = await vlm_loader.infer_optimized(frames_to_process, prompt, agent_name)
                logger.info("=/*"*40)
                logger.info(f"GPU inference completed for {agent_name} in {time.time() - start_time:.2f}s and agent response: {result_data}")
                logger.info("=/*"*40)
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

                # Filter out non-damaging issues for damage_detector agent
                if agent_name == "damage_detector":
                    # Step 1: Filter cosmetic issues (wrinkles, folds, etc.)
                    attributes = self._filter_non_damaging_issues(attributes)

                    # Step 2: Only check bounding box if damage still exists after Step 1
                    if attributes.get('damaged') == True and attributes.get('damage_location'):
                        # Also filter vague detections with oversized bounding boxes
                        # Pass frame shape to get actual resolution dynamically
                        frame_shape = frames[0].shape if frames else None
                        attributes = self._filter_vague_damage_detections(attributes, frame_shape)

                result.complete(attributes, confidence, reasoning)
                
                # Track performance
                inference_time = time.time() - start_time
                self.total_inferences += 1
                self.successful_inferences += 1
                self._record_performance(agent_name, inference_time, "success")

                logger.info(f"✅ GPU INFERENCE COMPLETED for {agent_name} in {inference_time:.2f}s")
                logger.info(f"📊 Results: {len(attributes)} attributes detected")
                logger.info("="*80)

                # Save frames AFTER inference (doesn't impact timer)
                saved_frame_paths = await self._save_debug_frames(
                    agent_name, frames, inference_num, cycle_num=cycle_num, redo_attempt=redo_attempt, mode=mode
                )

                # Log saved frames
                for i, path in enumerate(saved_frame_paths, 1):
                    logger.info(f"  📁 Frame SAVED TO: {path}")

                # Save annotated version for damage_detector ONLY if damage passed filters (bbox < 60%)
                if agent_name == "damage_detector" and attributes.get('damaged') == True and attributes.get('damage_location'):
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
                    
                    logger.error(f"❌ GPU inference failed for {agent_name} after {self.max_retries} retries: {e}")
                    
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

                    logger.debug(f"Saved frame: {filepath}")
                else:
                    logger.error(f"❌ Failed to save frame: {filepath}")

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

        Args:
            frames: Original frame data
            saved_paths: Paths where original frames were saved
            attributes: Damage detection attributes including damage_location
            cycle_num: Cycle number
            inference_num: Inference number
            mode: Mode (auto/manual)
            redo_attempt: Redo attempt number
        """
        try:
            damage_location = attributes.get('damage_location')
            damage_type = attributes.get('damage_type', 'damage')

            if not damage_location:
                return

            # Parse damage location
            if isinstance(damage_location, str):
                import re
                numbers = re.findall(r'[\d.]+', damage_location)
                if len(numbers) >= 4:
                    bbox = [float(n) for n in numbers[:4]]
                else:
                    return
            elif isinstance(damage_location, list) and len(damage_location) >= 4:
                bbox = damage_location
            else:
                return

            # Get agent directory
            agent_dir = self.frame_config.get_agent_frame_dir("damage_detector")

            for idx, (frame, saved_path) in enumerate(zip(frames, saved_paths)):
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

                # Get image dimensions
                img_width, img_height = pil_img.size

                # Parse bounding box coordinates
                x1, y1, x2, y2 = bbox[:4]

                # Convert normalized coords to pixels if needed
                if all(0 <= v <= 1 for v in bbox):
                    # Normalized coordinates
                    x1 = int(x1 * img_width)
                    y1 = int(y1 * img_height)
                    x2 = int(x2 * img_width)
                    y2 = int(y2 * img_height)
                else:
                    # Already pixel coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

                # Add label with damage type
                label_text = f"{damage_type}"
                draw.text((x1 + 5, y1 + 5), label_text, fill='red')

                # Convert back to BGR for OpenCV saving
                annotated_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # Save annotated frame
                success = cv2.imwrite(
                    str(annotated_filepath),
                    annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.frame_config.jpeg_quality]
                )

                if success:
                    logger.info(f"  🎨 Annotated frame SAVED TO: damage_detector/{annotated_filename}")
                else:
                    logger.error(f"❌ Failed to save annotated frame: {annotated_filepath}")

        except Exception as e:
            logger.error(f"Error saving annotated damage frames: {e}")

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
            logger.info(f"🧹 Filtered out cosmetic issues: {damage_type} → No damage")
        else:
            # Real damages remain - preserve original format (list or string)
            if isinstance(damage_type, list):
                filtered_attrs['damage_type'] = real_damages
            else:
                filtered_attrs['damage_type'] = ', '.join(real_damages)
            logger.info(f"🔍 Filtered damage types: {damage_type} → {filtered_attrs['damage_type']}")

        return filtered_attrs

    def _filter_vague_damage_detections(self, attributes: Dict[str, Any], frame_shape: tuple = None) -> Dict[str, Any]:
        """
        Filter out vague damage detections with overly large bounding boxes.
        If a bounding box covers >60% of the image, it's likely a false positive.

        Args:
            attributes: Damage detector attributes with 'damage_location' key
            frame_shape: Optional numpy array shape (height, width, channels) for dynamic resolution

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
            # Pixel coordinates - get actual image dimensions dynamically
            if frame_shape is not None:
                # frame_shape is (height, width, channels) from numpy array
                img_height, img_width = frame_shape[0], frame_shape[1]
            else:
                # Fallback to common resolution if frame shape not provided
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
            logger.info(f"🚫 Rejected vague damage detection: bbox covers {area_coverage*100:.1f}% of {resolution_info} image (threshold: {VAGUE_DETECTION_THRESHOLD*100:.0f}%)")
        else:
            resolution_info = f"{img_width}x{img_height}" if 'img_width' in locals() else "normalized"
            logger.debug(f"✅ Accepted damage detection: bbox covers {area_coverage*100:.1f}% of {resolution_info} image")

        return filtered_attrs

    def _get_agent_prompt(self, agent_name: str, inference_num: int, previous_context: Dict[str, Any] = None) -> str:
        """Generate appropriate prompt based on agent name and context"""
        prompts = {
            "initial_classifier": "Analyze this garment and identify: type (e.g., T-shirt, Dress, Pants, Shoes, Shirt, Shorts, Jacket, Sweatshirt, Sweater, Hoodie, Bag), color, pattern (max 3 words - only if necessary), neckline style, sleeve length, and closure type. Neckline, closure type, and sleeve length are optional or could be null for Shoes. Return null for unrecognizable attributes.",
            "detail_extractor": "Analyze this garment and Search for brand name/logo and size on this garment. Return brand and size, or null if not visible.",
            "damage_detector": """Analyze this garment and identify: damaged (true/false), damage_type (comma-separated words/small-description or NULL if no damages found), and damage_location (bounding box coordinates or null if no damage found). Only consider damages on the garment and not on the background/other objects.""",
            "final_compiler": "Compile final classification based on all attributes."
        }

        base_prompt = prompts.get(agent_name, "Analyze this clothing item")

        # Add context for inference #2 and beyond (not for final_compiler)
        if inference_num > 1 and previous_context and agent_name != "final_compiler":
            prev_attrs = previous_context.get('attributes', {})
            prev_confidence = previous_context.get('confidence', 0.0)

            # Build context string based on agent type
            context_str = "\n\nContext from your previous analysis:"

            if agent_name == "initial_classifier":
                # For initial classifier, mention previous garment attributes
                if prev_attrs:
                    context_str += f"\n- Previously identified: {', '.join([f'{k}={v}' for k, v in prev_attrs.items() if v])}"
                context_str += "\n\nYou are now analyzing TWO images in batch: your previous inference image AND a new image of the same garment. Compare both views to confirm or refine your findings."

            elif agent_name == "detail_extractor":
                # For detail extractor, mention if brand/size was found before
                brand = prev_attrs.get('brand', 'not found')
                size = prev_attrs.get('size', 'not found')
                context_str += f"\n- Previous brand detection: {brand}"
                context_str += f"\n- Previous size detection: {size}"
                context_str += "\n\nYou are now analyzing TWO images in batch: your previous inference image AND a new image. Check both images for brand and size labels - they may be clearer in one view."

            elif agent_name == "damage_detector":
                # For damage detector, mention previous damage findings
                damaged = prev_attrs.get('damaged', 'unknown')
                damage_type = prev_attrs.get('damage_type', 'none')
                context_str += f"\n- Previous damage detection: {damaged}"
                if damaged == 'yes':
                    context_str += f"\n- Damage type found: {damage_type}"
                context_str += "\n\nYou are now analyzing TWO images in batch: your previous inference image AND a new image. Examine both views for damage to confirm findings or detect additional issues."

            base_prompt += context_str

        return f"{base_prompt} (Inference #{inference_num})"
    
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
        logger.info("Shutting down GPU-optimized multi-inference engine...")
        
        try:
            # VLM singleton manages its own lifecycle
            # No need to shutdown as it's shared across the application
            logger.info("✅ GPU-optimized multi-inference engine shutdown completed")
            
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