#!/usr/bin/env python3
"""
Quick EasyOCR Timing Test - Process a subset with timing data
"""

import cv2
import numpy as np
import easyocr
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


def quick_easyocr_test():
    vlm_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/vlm_optimization_experiments")

    # Create output directory for EasyOCR results
    output_dir = vlm_dir / "easyocr_timed_output"
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("EASYOCR QUICK TIMING TEST")
    print("="*70)
    print(f"\nOutput Directory: {output_dir}")
    print("This directory will contain all OCR processed images\n")

    # Initialize EasyOCR
    print("Initializing EasyOCR...")
    init_start = time.time()
    reader = easyocr.Reader(['en'], gpu=False)
    init_time = time.time() - init_start
    print(f"EasyOCR initialization: {init_time:.2f}s\n")

    # Get first 5 images for quick test
    image_files = sorted(list(vlm_dir.glob("*.jpg")))[:5]

    print(f"Processing {len(image_files)} sample images...\n")

    results = []
    total_time_all = 0

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}] {img_path.name}")

        # Load and preprocess
        start_time = time.time()
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"    Error loading image")
            continue

        # Simple preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # OCR
        ocr_start = time.time()
        try:
            # Quick OCR with basic settings
            ocr_results = reader.readtext(enhanced, detail=1, threshold=0.01)

            # Get text
            text = ""
            if ocr_results:
                # Get highest confidence text
                texts_with_conf = [(r[1].upper(), r[2]) for r in ocr_results if r[1]]
                if texts_with_conf:
                    texts_with_conf.sort(key=lambda x: x[1], reverse=True)
                    text = texts_with_conf[0][0]
        except:
            text = ""

        ocr_time = time.time() - ocr_start
        total_time = time.time() - start_time

        if text:
            print(f"    ✅ Text: '{text}'")
        else:
            print(f"    ⚠️  No text detected")

        print(f"    ⏱️  OCR: {ocr_time:.3f}s, Total: {total_time:.3f}s")

        total_time_all += total_time

        # Create overlay image
        h, w = img.shape[:2]
        display_text = text if text else "NO TEXT"

        # Add text overlay
        font = cv2.FONT_HERSHEY_TRIPLEX
        scale = 1.0
        thickness = 2

        # Black bar at bottom
        cv2.rectangle(img, (0, h-60), (w, h), (0,0,0), -1)

        # OCR text in yellow
        cv2.putText(img, display_text, (10, h-35), font, scale, (0,255,255), thickness)

        # Timing info in green
        timing_text = f"Time: {total_time:.3f}s"
        cv2.putText(img, timing_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        # Save image
        output_path = output_dir / f"easyocr_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), img)

        results.append({
            "file": img_path.name,
            "text": text,
            "ocr_time_seconds": round(ocr_time, 3),
            "total_time_seconds": round(total_time, 3),
            "output": output_path.name
        })

    # Summary
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)

    avg_time = total_time_all / len(results) if results else 0
    print(f"\n📊 Performance:")
    print(f"  Total processing time: {total_time_all:.2f}s")
    print(f"  Average per image: {avg_time:.3f}s")
    print(f"  Images processed: {len(results)}")

    detected = sum(1 for r in results if r['text'])
    print(f"\n📝 Detection:")
    print(f"  Text found: {detected}/{len(results)}")

    print(f"\n📁 RESULTS DIRECTORY: {output_dir}/")
    print(f"   Contains {len(results)} processed images with OCR overlay")

    # Save JSON results
    summary = {
        "engine": "EasyOCR",
        "output_directory": str(output_dir),
        "total_images": len(results),
        "text_detected": detected,
        "average_time_seconds": round(avg_time, 3),
        "results": results
    }

    json_path = output_dir / "timing_results.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📄 JSON results: {json_path}")

    # Individual timings
    print(f"\n⏱️  Individual Image Timings:")
    for r in results:
        print(f"  - {r['file'][:30]}: OCR={r['ocr_time_seconds']}s, Total={r['total_time_seconds']}s")

    return output_dir


if __name__ == "__main__":
    output_dir = quick_easyocr_test()
    print(f"\n✅ Complete! All results in: {output_dir}/")