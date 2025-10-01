#!/usr/bin/env python3
"""
Robust GPU-Only EasyOCR Agent with Time-Buffered Multi-Inference
Intelligent OCR agent with 1-second time budget and context awareness
"""

import cv2
import numpy as np
import easyocr
import torch
import json
from pathlib import Path
import time
import os
import warnings
from collections import deque
from typing import List, Tuple, Dict, Optional
import re
from datetime import datetime

warnings.filterwarnings('ignore')

# GPU optimization settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True


class RobustGPUOCRAgent:
    """
    Intelligent OCR Agent with:
    - GPU-only processing (no CPU fallback)
    - 1-second time budget per image
    - Multiple inference strategies within time budget
    - Context awareness from previous detections
    - Adaptive preprocessing based on results
    """

    def __init__(self, time_budget: float = None):
        self.vlm_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/vlm_optimization_experiments")
        self.output_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/text_zoom_experiments/gpu_robust_agent_results")
        self.output_dir.mkdir(exist_ok=True)

        # Time budget disabled for now (set to None)
        self.time_budget = time_budget if time_budget else 9999  # Effectively no limit

        # Context memory for previous detections
        self.context_memory = deque(maxlen=20)  # Remember last 20 detections
        self.vocabulary_cache = set()  # Common words found
        self.brand_patterns = set()  # Detected brand patterns

        print("="*70)
        print("ROBUST GPU-ONLY OCR AGENT")
        print("="*70)

        # Verify GPU availability (REQUIRED)
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is REQUIRED. No GPU detected!")

        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"⏱️  Time Budget: {'Disabled' if self.time_budget == 9999 else f'{self.time_budget}s per image'}")
        print(f"📁 Output: {self.output_dir}\n")

        # Clear GPU cache
        torch.cuda.empty_cache()

        # Initialize GPU reader
        print("Initializing GPU-Only Reader...")
        start = time.time()
        self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        print(f"  Reader loaded: {time.time() - start:.2f}s")

        # Warm up GPU
        dummy = np.ones((100, 100), dtype=np.uint8)
        self.reader.readtext(dummy, detail=0)
        torch.cuda.synchronize()
        print("  GPU warmed up\n")

        # Define the 4 optimal parameter combinations
        self.param_combinations = [
            {  # COMBO 1: Sensitive + Unsharp Mask
                'name': 'sensitive_unsharp',
                'preprocessing': 'unsharp_mask',
                'clahe_clip': 2.0,
                'tile_size': (8, 8),
                'text_threshold': 0.5,
                'link_threshold': 0.3,
                'low_text': 0.3,
                'mag_ratio': 1.5,
                'width_ths': 0.7,
                'height_ths': 0.7
            },
            {  # COMBO 2: Default + Bilateral Filter
                'name': 'default_bilateral',
                'preprocessing': 'bilateral',
                'clahe_clip': 2.0,
                'tile_size': (8, 8),
                'text_threshold': 0.7,
                'link_threshold': 0.4,
                'low_text': 0.4,
                'mag_ratio': 1.0,
                'width_ths': 0.7,
                'height_ths': 0.7
            },
            {  # COMBO 3: Very Sensitive + Standard
                'name': 'very_sensitive_standard',
                'preprocessing': 'standard',
                'clahe_clip': 2.0,
                'tile_size': (4, 4),
                'text_threshold': 0.3,
                'link_threshold': 0.2,
                'low_text': 0.2,
                'mag_ratio': 1.5,
                'width_ths': 0.5,
                'height_ths': 0.5
            },
            {  # COMBO 4: Sensitive + Bilateral (NEW)
                'name': 'sensitive_bilateral',
                'preprocessing': 'bilateral',
                'clahe_clip': 2.0,
                'tile_size': (8, 8),
                'text_threshold': 0.5,
                'link_threshold': 0.3,
                'low_text': 0.3,
                'mag_ratio': 1.0,
                'width_ths': 0.7,
                'height_ths': 0.7
            }
        ]

        # Initialize results storage
        self.all_results = {}
        self.gpu_metrics = []

    def update_context(self, text: str, confidence: float):
        """Update context memory with new detection"""
        if text and confidence > 0.3:
            self.context_memory.append({
                'text': text,
                'confidence': confidence,
                'timestamp': time.time()
            })

            # Extract vocabulary
            words = re.findall(r'\b[A-Z]+\b', text.upper())
            self.vocabulary_cache.update(words)

            # Detect brand patterns
            if any(brand in text.upper() for brand in ['AMAZON', 'NIKE', 'ADIDAS', 'PUMA']):
                self.brand_patterns.add(text.upper())

    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization"""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj

    def get_context_hints(self) -> Dict:
        """Get hints from context memory"""
        hints = {
            'common_words': list(self.vocabulary_cache)[:10],
            'recent_detections': [item['text'] for item in list(self.context_memory)[-5:]],
            'brand_patterns': list(self.brand_patterns)
        }
        return hints

    def calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness of image"""
        if len(image.shape) == 3:
            # Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate mean brightness (0-255)
        brightness = np.mean(gray)
        return brightness

    def get_gpu_metrics(self) -> Dict:
        """Get current GPU memory and utilization metrics"""
        try:
            if torch.cuda.is_available():
                # Get memory info
                free_memory, total_memory = torch.cuda.mem_get_info()
                used_memory = total_memory - free_memory

                # Convert to GB
                free_gb = free_memory / (1024**3)
                used_gb = used_memory / (1024**3)
                total_gb = total_memory / (1024**3)

                # Get utilization if possible
                utilization = torch.cuda.utilization()

                return {
                    'timestamp': time.time(),
                    'free_gb': round(free_gb, 3),
                    'used_gb': round(used_gb, 3),
                    'total_gb': round(total_gb, 3),
                    'utilization_percent': utilization,
                    'memory_percent': round((used_gb / total_gb) * 100, 1)
                }
        except Exception as e:
            return {
                'timestamp': time.time(),
                'error': str(e),
                'free_gb': 0,
                'used_gb': 0,
                'total_gb': 0,
                'utilization_percent': 0,
                'memory_percent': 0
            }
        return {}

    def preprocess_image(self, image: np.ndarray, strategy: str, combo: Dict = None) -> np.ndarray:
        """Apply preprocessing based on strategy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if strategy == 'standard':
            # Standard CLAHE preprocessing
            clip_limit = combo.get('clahe_clip', 2.0) if combo else 2.0
            tile_size = combo.get('tile_size', (8,8)) if combo else (8,8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            return clahe.apply(gray)

        elif strategy == 'unsharp_mask':
            # Unsharp masking for edge enhancement
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            return sharpened

        elif strategy == 'bilateral':
            # Bilateral filter for edge-preserving smoothing
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            return filtered

        elif strategy == 'enhanced':
            # Stronger enhancement for difficult images
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            # Add sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            return sharpened

        elif strategy == 'minimal':
            # Minimal processing for clear images
            return gray

        elif strategy == 'inverted':
            # For white text on dark background
            inverted = cv2.bitwise_not(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(inverted)

        elif strategy == 'adaptive':
            # Context-aware preprocessing
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # If we've seen inverted patterns before, try both
            if self.brand_patterns:
                return enhanced
            else:
                # Adaptive thresholding
                thresh = cv2.adaptiveThreshold(enhanced, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
                return thresh

        return gray

    def inference_pass(self, image: np.ndarray, combo: Dict) -> Dict:
        """
        Single inference pass with given combination
        Returns: Dict with results and metrics
        """
        start = time.time()

        # Get GPU metrics before inference
        gpu_before = self.get_gpu_metrics()

        try:
            # Preprocess
            processed = self.preprocess_image(image, combo['preprocessing'], combo)

            # Build OCR parameters
            ocr_params = {
                'detail': 1,
                'paragraph': False,
                'text_threshold': combo.get('text_threshold', 0.7),
                'link_threshold': combo.get('link_threshold', 0.4),
                'low_text': combo.get('low_text', 0.4),
                'mag_ratio': combo.get('mag_ratio', 1.0),
                'width_ths': combo.get('width_ths', 0.7),
                'height_ths': combo.get('height_ths', 0.7),
                'canvas_size': 2560
            }

            # Run OCR
            results = self.reader.readtext(processed, **ocr_params)

            # Get GPU metrics after inference
            gpu_after = self.get_gpu_metrics()

            # Find best result and all detected texts
            best_text = ""
            best_conf = 0
            all_detections = []
            bboxes = []

            for bbox, text, conf in results:
                clean_text = text.upper()
                clean_text = re.sub(r'[^A-Z0-9\s\-]', '', clean_text).strip()

                # Convert bbox to regular Python list to avoid JSON serialization issues
                bbox_list = [[float(x), float(y)] for x, y in bbox] if bbox else []

                all_detections.append({
                    'text': clean_text,
                    'confidence': round(float(conf), 3),
                    'bbox': bbox_list
                })
                bboxes.append(bbox)

                if conf > best_conf and clean_text:
                    best_text = clean_text
                    best_conf = conf

            elapsed = time.time() - start

            return {
                'text': best_text,
                'confidence': round(best_conf, 3),
                'time': round(elapsed, 3),
                'strategy': combo['name'],
                'preprocessing': combo['preprocessing'],
                'params': ocr_params,
                'all_detections': all_detections,
                'bboxes': bboxes,
                'gpu_before': gpu_before,
                'gpu_after': gpu_after,
                'gpu_delta': {
                    'memory_change_gb': round(gpu_after.get('used_gb', 0) - gpu_before.get('used_gb', 0), 3),
                    'utilization_change': gpu_after.get('utilization_percent', 0) - gpu_before.get('utilization_percent', 0)
                }
            }

        except Exception as e:
            elapsed = time.time() - start
            gpu_after = self.get_gpu_metrics()

            return {
                'text': '',
                'confidence': 0,
                'time': round(elapsed, 3),
                'strategy': combo['name'],
                'preprocessing': combo['preprocessing'],
                'error': str(e),
                'gpu_before': gpu_before,
                'gpu_after': gpu_after
            }

    def multi_inference(self, image_path: Path) -> Dict:
        """
        Perform exactly 4 inference attempts using optimal parameter combinations
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                'file': image_path.name,
                'text': '',
                'confidence': 0,
                'time': 0,
                'attempts': 0,
                'error': 'Image not found'
            }

        # Get initial GPU metrics
        gpu_idle_before = self.get_gpu_metrics()

        # Calculate image brightness
        brightness = self.calculate_brightness(img)

        # Track all results
        all_attempts = []
        total_time = 0
        time_start = time.time()

        # Force exactly 4 attempts with our optimal combinations
        for attempt_num, combo in enumerate(self.param_combinations, 1):
            # Clear GPU cache before each attempt
            torch.cuda.empty_cache()

            # Run inference with this combination
            result = self.inference_pass(img, combo)
            result['attempt_num'] = attempt_num
            result['brightness'] = round(brightness, 1)

            # Track timing
            total_time += result['time']

            # Store all attempts (not just successful ones)
            all_attempts.append(result)

            # Update context if text was found
            if result['text']:
                self.update_context(result['text'], result['confidence'])

        # Get GPU metrics after all attempts
        gpu_idle_after = self.get_gpu_metrics()

        # Find best result
        best_result = None
        best_confidence = 0
        for attempt in all_attempts:
            if attempt['confidence'] > best_confidence:
                best_confidence = attempt['confidence']
                best_result = attempt

        # Prepare final result
        total_elapsed = time.time() - time_start

        return {
            'file': image_path.name,
            'file_path': str(image_path),
            'brightness': round(brightness, 1),
            'total_time': round(total_elapsed, 3),
            'attempts': len(all_attempts),
            'all_attempts': all_attempts,
            'best_result': {
                'text': best_result['text'] if best_result else '',
                'confidence': best_result['confidence'] if best_result else 0,
                'strategy': best_result['strategy'] if best_result else 'none',
                'attempt_num': best_result['attempt_num'] if best_result else 0
            },
            'gpu_metrics': {
                'idle_before': gpu_idle_before,
                'idle_after': gpu_idle_after,
                'total_memory_change_gb': round(
                    gpu_idle_after.get('used_gb', 0) - gpu_idle_before.get('used_gb', 0), 3
                )
            },
            'context': {
                'vocabulary_size': len(self.vocabulary_cache),
                'brands_detected': list(self.brand_patterns)
            }
        }

    def run_agent(self):
        """Run the OCR agent on all images and save comprehensive JSON results"""

        # Get all images
        image_files = sorted(list(self.vlm_dir.glob("*.jpg")))

        print("="*70)
        print(f"Processing {len(image_files)} images with 4 optimal parameter combinations...")
        print(f"Results will be saved to JSON (console output disabled)")
        print("="*70 + "\n")

        # Initialize results storage
        all_image_results = []
        successful = 0
        session_start = time.time()

        # Progress tracking (minimal console output)
        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx:2d}/{len(image_files)}] Processing {img_path.name[:40]}...")

            # Run multi-inference with 4 attempts
            result = self.multi_inference(img_path)

            # Check if successful
            if result['best_result']['text']:
                successful += 1

            # Create enhanced visualization with all 4 attempts
            self._create_enhanced_visualization(img_path, result, idx)

            # Store result
            all_image_results.append(result)

            # Minimal progress indicator
            print(f"        Completed. Best: '{result['best_result']['text']}' ({result['best_result']['confidence']:.2f})")

        # Prepare comprehensive results JSON
        session_end = time.time()

        # Calculate statistics
        total_images = len(all_image_results)
        avg_time = sum(r['total_time'] for r in all_image_results) / total_images if total_images > 0 else 0

        # Strategy effectiveness
        strategy_stats = {}
        for img_result in all_image_results:
            for attempt in img_result['all_attempts']:
                strategy = attempt['strategy']
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'successful': 0, 'failed': 0, 'total': 0, 'avg_confidence': 0}

                strategy_stats[strategy]['total'] += 1
                if attempt['text']:
                    strategy_stats[strategy]['successful'] += 1
                    strategy_stats[strategy]['avg_confidence'] += attempt['confidence']
                else:
                    strategy_stats[strategy]['failed'] += 1

        # Calculate average confidence for each strategy
        for strategy in strategy_stats:
            if strategy_stats[strategy]['successful'] > 0:
                strategy_stats[strategy]['avg_confidence'] /= strategy_stats[strategy]['successful']
                strategy_stats[strategy]['avg_confidence'] = round(strategy_stats[strategy]['avg_confidence'], 3)

        # Get final GPU metrics
        final_gpu = self.get_gpu_metrics()

        # Create comprehensive summary
        comprehensive_results = {
            'session_info': {
                'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                'duration_seconds': round(session_end - session_start, 2),
                'gpu_device': torch.cuda.get_device_name(0),
                'gpu_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
                'time_budget': 'Disabled' if self.time_budget == 9999 else self.time_budget
            },
            'summary': {
                'total_images': total_images,
                'successful_detections': successful,
                'detection_rate_percent': round((successful/total_images*100) if total_images > 0 else 0, 1),
                'avg_time_per_image': round(avg_time, 3),
                'total_attempts': total_images * 4  # Always 4 attempts per image
            },
            'parameter_combinations': self.param_combinations,
            'strategy_effectiveness': strategy_stats,
            'context_learning': {
                'vocabulary_learned': list(self.vocabulary_cache),
                'vocabulary_count': len(self.vocabulary_cache),
                'brands_detected': list(self.brand_patterns),
                'brands_count': len(self.brand_patterns)
            },
            'gpu_metrics_final': final_gpu,
            'detailed_results': all_image_results
        }

        # Convert any numpy types to Python native types before JSON serialization
        comprehensive_results = self._convert_numpy_types(comprehensive_results)

        # Save to JSON file
        json_path = self.output_dir / f"ocr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)

        # Print minimal summary
        print("="*70)
        print(f"✅ Processing Complete!")
        print(f"📊 Detection Rate: {successful}/{total_images} ({successful/total_images*100:.1f}%)")
        print(f"⏱️  Average Time: {avg_time:.2f}s per image")
        print(f"📄 Results saved to: {json_path}")
        print(f"🖼️  Visualizations saved to: {self.output_dir}/")
        print("="*70)

        return comprehensive_results

    def _create_enhanced_visualization(self, img_path: Path, result: Dict, idx: int):
        """Create visualization showing all 4 inference attempts and their results"""
        img = cv2.imread(str(img_path))
        if img is None:
            return

        h, w = img.shape[:2]

        # Create a larger canvas to show all attempts
        canvas_height = h + 250  # Extra space for results
        canvas = np.zeros((canvas_height, w, 3), dtype=np.uint8)

        # Copy original image to top
        canvas[:h, :] = img

        # Add black background for results
        canvas[h:, :] = (30, 30, 30)  # Dark gray background

        # Draw separator line
        cv2.line(canvas, (0, h), (w, h), (100, 100, 100), 2)

        # Title section
        cv2.putText(canvas, f"4-Attempt OCR Results | Image: {img_path.name[:30]}",
                   (10, h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display all 4 attempts in a grid
        y_offset = h + 60
        for i, attempt in enumerate(result['all_attempts'], 1):
            # Determine color based on confidence
            if attempt['confidence'] > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif attempt['confidence'] > 0.5:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 100, 255)  # Orange for low

            # Format text
            text = attempt['text'] if attempt['text'] else "NO TEXT"
            conf = attempt['confidence']
            strategy = attempt['strategy']
            time_ms = attempt['time'] * 1000

            # Draw attempt info
            y_pos = y_offset + (i-1) * 35
            cv2.putText(canvas, f"#{i} [{strategy}]:",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(canvas, f"{text[:40]}",
                       (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(canvas, f"Conf: {conf:.3f} | Time: {time_ms:.0f}ms",
                       (w-300, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Best result highlight
        best = result['best_result']
        if best['text']:
            y_best = h + 200
            cv2.rectangle(canvas, (5, y_best-5), (w-5, y_best+30), (0, 100, 0), 2)
            cv2.putText(canvas, f"BEST: {best['text'][:50]}",
                       (10, y_best+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, f"Strategy: {best['strategy']} | Confidence: {best['confidence']:.3f}",
                       (10, y_best+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Try to highlight text location if bboxes available
        if result['all_attempts'] and result['best_result']['text']:
            # Find the best attempt's bboxes
            for attempt in result['all_attempts']:
                if attempt['text'] == best['text'] and 'bboxes' in attempt and attempt['bboxes']:
                    # Draw bounding boxes on original image area
                    for bbox in attempt['bboxes'][:3]:  # Limit to 3 boxes
                        if len(bbox) >= 4:
                            pts = np.array(bbox[:4], dtype=np.int32)
                            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
                    break

        # Save the enhanced visualization
        output_path = self.output_dir / f"result_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), canvas)

        # Also save a simple version with just the best result
        simple = img.copy()
        h_simple, w_simple = simple.shape[:2]

        # Add overlay at bottom
        overlay_h = 60
        cv2.rectangle(simple, (0, h_simple-overlay_h), (w_simple, h_simple), (0, 0, 0), -1)

        if best['text']:
            cv2.putText(simple, f"{best['text'][:50]}",
                       (10, h_simple-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(simple, f"Conf: {best['confidence']:.2f} | {best['strategy']}",
                       (10, h_simple-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(simple, "NO TEXT DETECTED",
                       (10, h_simple-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        simple_path = self.output_dir / f"simple_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(simple_path), simple)


if __name__ == "__main__":
    # Initialize agent with disabled time budget (4 fixed attempts)
    agent = RobustGPUOCRAgent(time_budget=None)

    # Run the agent with 4 optimal parameter combinations
    results = agent.run_agent()