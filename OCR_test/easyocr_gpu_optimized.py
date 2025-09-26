#!/usr/bin/env python3
"""
Optimized EasyOCR with GPU - Balance of speed and accuracy
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
warnings.filterwarnings('ignore')

# GPU optimization settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True


class OptimizedGPUOCR:
    def __init__(self):
        self.vlm_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/vlm_optimization_experiments")
        self.output_dir = self.vlm_dir / "1easyocr_gpu_optimized1"
        self.output_dir.mkdir(exist_ok=True)

        print("="*70)
        print("OPTIMIZED EASYOCR GPU BENCHMARK")
        print("="*70)

        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            torch.cuda.empty_cache()

        print(f"📁 Output: {self.output_dir}\n")

    def optimized_preprocess(self, image):
        """Optimized preprocessing for speed and accuracy"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Moderate resize for speed
        h, w = gray.shape
        
        # if w > 640:
        #     scale = 640 / w
        #     new_h = int(h * scale)
        #     gray = cv2.resize(gray, (640, new_h), cv2.INTER_LINEAR)
        # elif h < 50:
        #     scale = 50 / h
        #     new_w = int(w * scale)
        #     gray = cv2.resize(gray, (new_w, 50), cv2.INTER_LINEAR)

        # Simple but effective enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        return enhanced

    def run_ocr(self, reader, image_path, device_name="Device"):
        """Run OCR with optimized settings"""
        img = cv2.imread(str(image_path))
        if img is None:
            return "", 0

        # Preprocess
        processed = self.optimized_preprocess(img)

        start = time.time()

        try:
            # Optimized parameters for speed
            results = reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                threshold=0.1,  # Higher threshold for speed
                mag_ratio=1.0,   # Lower magnification for speed
                batch_size=1,
                workers=0
            )

            # Also try on original if no results
            if not results and len(img.shape) == 3:
                results = reader.readtext(
                    img,
                    detail=1,
                    paragraph=False,
                    threshold=0.1
                )

            # Get best text
            if results:
                best_text = ""
                best_conf = 0
                for bbox, text, conf in results:
                    if conf > best_conf and text:
                        best_text = text.upper()
                        best_conf = conf

                elapsed = time.time() - start
                return self.clean_text(best_text), elapsed

        except Exception as e:
            print(f"  {device_name} error: {e}")

        elapsed = time.time() - start
        return "", elapsed

    def clean_text(self, text):
        """Clean text"""
        import re
        cleaned = re.sub(r'[^\w\s\-]', '', text).strip()
        return cleaned.upper()

    def run_quick_benchmark(self):
        """Quick benchmark with first 5 images"""

        print("Initializing readers...\n")

        # GPU Reader
        if self.gpu_available:
            start = time.time()
            reader_gpu = easyocr.Reader(['en'], gpu=True, verbose=False)
            gpu_init = time.time() - start
            print(f"GPU reader: {gpu_init:.2f}s")

            # Warm up
            dummy = np.zeros((100, 100), dtype=np.uint8)
            reader_gpu.readtext(dummy, detail=0)
        else:
            reader_gpu = None
            gpu_init = 0

        # CPU Reader
        start = time.time()
        reader_cpu = easyocr.Reader(['en'], gpu=False, verbose=False)
        cpu_init = time.time() - start
        print(f"CPU reader: {cpu_init:.2f}s\n")

        # Test on first 5 images
        image_files = sorted(list(self.vlm_dir.glob("*.jpg")))[:5]

        print("="*70)
        print(f"Testing {len(image_files)} images...")
        print("="*70 + "\n")

        results = []
        total_gpu = 0
        total_cpu = 0

        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}] {img_path.name[:40]}")

            # GPU
            if reader_gpu:
                gpu_text, gpu_time = self.run_ocr(reader_gpu, img_path, "GPU")
                total_gpu += gpu_time
                print(f"  GPU: '{gpu_text[:30]}' - {gpu_time:.3f}s")
            else:
                gpu_text, gpu_time = "", 0

            # CPU
            cpu_text, cpu_time = self.run_ocr(reader_cpu, img_path, "CPU")
            total_cpu += cpu_time
            print(f"  CPU: '{cpu_text[:30]}' - {cpu_time:.3f}s")

            if gpu_time > 0 and cpu_time > 0:
                print(f"  Speedup: {cpu_time/gpu_time:.1f}x\n")

            # Save comparison image
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            cv2.rectangle(img, (0, h-80), (w, h), (0,0,0), -1)

            # Show results
            if gpu_text:
                cv2.putText(img, f"GPU: {gpu_text[:20]} ({gpu_time:.2f}s)",
                           (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(img, f"GPU: NO TEXT ({gpu_time:.2f}s)",
                           (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if cpu_text:
                cv2.putText(img, f"CPU: {cpu_text[:20]} ({cpu_time:.2f}s)",
                           (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                cv2.putText(img, f"CPU: NO TEXT ({cpu_time:.2f}s)",
                           (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            output_path = self.output_dir / f"result_{idx:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), img)

            results.append({
                "file": img_path.name,
                "gpu_text": gpu_text,
                "gpu_time": round(gpu_time, 3),
                "cpu_text": cpu_text,
                "cpu_time": round(cpu_time, 3)
            })

        # Summary
        print("="*70)
        print("SUMMARY")
        print("="*70)

        if self.gpu_available and total_gpu > 0:
            avg_gpu = total_gpu / len(results)
            avg_cpu = total_cpu / len(results)
            print(f"\nAverage times:")
            print(f"  GPU: {avg_gpu:.3f}s")
            print(f"  CPU: {avg_cpu:.3f}s")
            print(f"  Speedup: {avg_cpu/avg_gpu:.1f}x")

        print(f"\nDetected texts:")
        for r in results:
            if r['gpu_text'] or r['cpu_text']:
                print(f"  {r['file'][:30]}:")
                if r['gpu_text']:
                    print(f"    GPU: {r['gpu_text']}")
                if r['cpu_text']:
                    print(f"    CPU: {r['cpu_text']}")

        # Save
        summary = {
            "gpu_available": self.gpu_available,
            "results": results,
            "average_gpu": round(avg_gpu, 3) if self.gpu_available else 0,
            "average_cpu": round(avg_cpu, 3),
            "speedup": round(avg_cpu/avg_gpu, 1) if self.gpu_available and avg_gpu > 0 else 0
        }

        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Results saved to: {self.output_dir}/")

        return summary


if __name__ == "__main__":
    ocr = OptimizedGPUOCR()
    ocr.run_quick_benchmark()