#!/usr/bin/env python3
"""
EasyOCR Timed Test - Measure inference speed and accuracy
Process VLM optimization experiments images with timing data
"""

import cv2
import numpy as np
import easyocr
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


class EasyOCRTimedTest:
    def __init__(self):
        self.vlm_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/vlm_optimization_experiments")
        # New directory for EasyOCR timed results
        self.output_dir = self.vlm_dir / "easyocr_timed_results"
        self.output_dir.mkdir(exist_ok=True)

        print("Initializing EasyOCR with Timing Analysis...")
        print("Output Directory:", self.output_dir)

        # Initialize EasyOCR (this takes time on first run)
        start_init = time.time()
        self.reader = easyocr.Reader(['en'], gpu=False)
        init_time = time.time() - start_init
        print(f"EasyOCR initialization time: {init_time:.2f} seconds")

    def preprocess_for_easyocr(self, image):
        """Preprocessing optimized for EasyOCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape
        if h < 100:
            scale = 100 / h
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, 100), interpolation=cv2.INTER_CUBIC)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        return enhanced

    def extract_text_timed(self, image_path):
        """Extract text with timing measurements"""
        # Load image
        load_start = time.time()
        image = cv2.imread(str(image_path))
        load_time = time.time() - load_start

        if image is None:
            return "", 0, 0, 0

        # Preprocessing
        preprocess_start = time.time()
        processed = self.preprocess_for_easyocr(image)
        preprocess_time = time.time() - preprocess_start

        # OCR inference
        ocr_start = time.time()
        all_texts = []

        try:
            # Run EasyOCR with different settings
            results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                width_ths=0.5,
                height_ths=0.5,
                y_ths=0.5,
                x_ths=1.0,
                threshold=0.01,
                mag_ratio=1.5
            )

            for bbox, text, conf in results:
                if text and conf > 0.01:
                    all_texts.append((text.upper(), conf))

        except Exception as e:
            print(f"    OCR Error: {e}")

        ocr_time = time.time() - ocr_start

        # Also try on original color image
        try:
            ocr2_start = time.time()
            results2 = self.reader.readtext(
                image,
                detail=1,
                threshold=0.01
            )

            for bbox, text, conf in results2:
                if text and conf > 0.01:
                    all_texts.append((text.upper(), conf))

            ocr_time += (time.time() - ocr2_start)
        except:
            pass

        # Select best result
        if all_texts:
            # Sort by confidence
            all_texts.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            best_text = self.clean_text(all_texts[0][0])
            return best_text, load_time, preprocess_time, ocr_time

        return "", load_time, preprocess_time, ocr_time

    def clean_text(self, text):
        """Clean extracted text"""
        import re
        cleaned = re.sub(r'[^\w\s\-]', '', text).strip()
        return cleaned.upper() if cleaned else ""

    def create_overlay_with_timing(self, image_path, text, total_time):
        """Create overlay with OCR result and timing info"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        # Main text
        display_text = text if text else "NO TEXT DETECTED"

        # Text settings
        font = cv2.FONT_HERSHEY_TRIPLEX
        scale = 1.2
        thickness = 2

        if len(display_text) > 20:
            scale = 0.8
        elif len(display_text) > 15:
            scale = 1.0

        text_size = cv2.getTextSize(display_text, font, scale, thickness)[0]

        # Position at bottom
        x = max(10, (w - text_size[0]) // 2)
        y = h - 30

        # Black background for main text
        cv2.rectangle(img,
                     (x-10, y-text_size[1]-10),
                     (min(w-5, x+text_size[0]+10), y+10),
                     (0,0,0), -1)

        # Cyan text for result
        cv2.putText(img, display_text, (x, y), font, scale, (255, 255, 0), thickness)

        # Add timing info at top
        timing_text = f"Time: {total_time:.3f}s"
        cv2.rectangle(img, (5, 5), (150, 35), (0,0,0), -1)
        cv2.putText(img, timing_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        return img

    def process_all_with_timing(self):
        """Process all VLM images with detailed timing"""
        image_files = sorted(list(self.vlm_dir.glob("*.jpg")))

        print("\n" + "="*70)
        print("EASYOCR TIMED TEST - VLM OPTIMIZATION EXPERIMENTS")
        print("="*70)
        print(f"\nProcessing {len(image_files)} images with timing analysis...\n")
        print(f"Results will be saved to: {self.output_dir}\n")

        results = []
        texts_found = 0
        total_processing_time = 0

        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] {img_path.name}")

            # Process with timing
            total_start = time.time()
            text, load_time, preprocess_time, ocr_time = self.extract_text_timed(img_path)
            total_time = time.time() - total_start

            if text:
                texts_found += 1
                print(f"    ✅ EasyOCR: '{text}'")
            else:
                print(f"    ⚠️  No text detected")

            print(f"    ⏱️  Times: Load={load_time:.3f}s, Preprocess={preprocess_time:.3f}s, OCR={ocr_time:.3f}s, Total={total_time:.3f}s")

            total_processing_time += total_time

            # Create overlay with timing info
            overlay = self.create_overlay_with_timing(img_path, text, total_time)

            # Save overlay
            output_name = f"easyocr_timed_{img_path.stem}.jpg"
            output_path = self.output_dir / output_name

            if overlay is not None:
                cv2.imwrite(str(output_path), overlay)

            results.append({
                "index": idx,
                "original": img_path.name,
                "easyocr_text": text if text else "",
                "detected": bool(text),
                "timing": {
                    "load_time_ms": round(load_time * 1000, 2),
                    "preprocess_time_ms": round(preprocess_time * 1000, 2),
                    "ocr_time_ms": round(ocr_time * 1000, 2),
                    "total_time_ms": round(total_time * 1000, 2)
                },
                "output": output_name
            })

        print("\n" + "="*70)
        print("EASYOCR TIMING ANALYSIS COMPLETE")
        print("="*70)

        accuracy = (texts_found / len(image_files)) * 100 if image_files else 0
        avg_time = total_processing_time / len(image_files) if image_files else 0

        print(f"\n📊 EasyOCR Results:")
        print(f"  Total Images: {len(image_files)}")
        print(f"  Text Detected: {texts_found}")
        print(f"  No Text: {len(image_files) - texts_found}")
        print(f"  Detection Rate: {accuracy:.1f}%")

        print(f"\n⏱️  Timing Statistics:")
        print(f"  Total Processing Time: {total_processing_time:.2f} seconds")
        print(f"  Average Time per Image: {avg_time:.3f} seconds")

        # Calculate timing statistics
        if results:
            ocr_times = [r['timing']['ocr_time_ms'] for r in results]
            total_times = [r['timing']['total_time_ms'] for r in results]

            print(f"  Min OCR Time: {min(ocr_times):.2f}ms")
            print(f"  Max OCR Time: {max(ocr_times):.2f}ms")
            print(f"  Avg OCR Time: {np.mean(ocr_times):.2f}ms")
            print(f"  Min Total Time: {min(total_times):.2f}ms")
            print(f"  Max Total Time: {max(total_times):.2f}ms")
            print(f"  Avg Total Time: {np.mean(total_times):.2f}ms")

        # Show detected texts with timing
        if texts_found > 0:
            print(f"\n📝 EasyOCR Detections with Timing:")
            for r in results:
                if r['detected']:
                    print(f"  - {r['original'][:30]}: '{r['easyocr_text']}' ({r['timing']['total_time_ms']:.0f}ms)")

        # Text distribution
        text_counts = {}
        for r in results:
            if r['easyocr_text']:
                text = r['easyocr_text']
                text_counts[text] = text_counts.get(text, 0) + 1

        if text_counts:
            print(f"\n📈 Detected Text Distribution:")
            for text, count in sorted(text_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - '{text}': {count} image{'s' if count > 1 else ''}")

        # Save results with timing data
        final = {
            "ocr_engine": "EasyOCR",
            "gpu_enabled": False,
            "directory": str(self.vlm_dir),
            "output_directory": str(self.output_dir),
            "total_images": len(image_files),
            "texts_detected": texts_found,
            "no_text": len(image_files) - texts_found,
            "detection_rate_percent": round(accuracy, 2),
            "timing_summary": {
                "total_processing_time_seconds": round(total_processing_time, 2),
                "average_time_per_image_seconds": round(avg_time, 3),
                "average_ocr_time_ms": round(np.mean(ocr_times), 2) if results else 0,
                "average_total_time_ms": round(np.mean(total_times), 2) if results else 0
            },
            "results": results
        }

        json_path = self.output_dir / "easyocr_timed_results.json"
        with open(json_path, 'w') as f:
            json.dump(final, f, indent=2)

        print(f"\n📁 OUTPUT DIRECTORY: {self.output_dir}/")
        print(f"📄 Results JSON: {json_path}")
        print(f"\n✅ All processed images with timing data saved to:")
        print(f"   {self.output_dir}/")
        print(f"\n   This directory contains {len(image_files)} images with OCR overlay and timing info")

        return final


if __name__ == "__main__":
    ocr = EasyOCRTimedTest()
    results = ocr.process_all_with_timing()