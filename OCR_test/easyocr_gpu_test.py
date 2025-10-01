#!/usr/bin/env python3
"""
EasyOCR with GPU Acceleration - Performance Test
Testing on NVIDIA RTX A1000
"""

import cv2
import numpy as np
import easyocr
import torch
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')


class EasyOCRGPUTest:
    def __init__(self):
        self.vlm_dir = Path("/home/denaliai/RELO-CLASSIFIER-DEV/RELO-DEMO-APP-test-RELO-DEV-APP-test/vlm_optimization_experiments")
        self.output_dir = self.vlm_dir / "easyocr_gpu_results"
        self.output_dir.mkdir(exist_ok=True)

        print("="*70)
        print("EASYOCR GPU ACCELERATION TEST")
        print("="*70)

        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()

        if self.gpu_available:
            print(f"\n✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   GPU Available Memory: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")
        else:
            print("\n❌ No GPU detected, will use CPU")

        print("\n📁 Output Directory:", self.output_dir)
        print("-"*70)

    def initialize_readers(self):
        """Initialize both GPU and CPU readers for comparison"""
        print("\nInitializing OCR readers...")

        # GPU Reader
        if self.gpu_available:
            print("Loading GPU-accelerated EasyOCR...")
            start_gpu = time.time()
            self.reader_gpu = easyocr.Reader(['en'], gpu=True, verbose=False)
            gpu_load_time = time.time() - start_gpu
            print(f"  GPU Reader loaded: {gpu_load_time:.2f}s")
        else:
            self.reader_gpu = None
            gpu_load_time = 0

        # CPU Reader for comparison
        print("Loading CPU EasyOCR...")
        start_cpu = time.time()
        self.reader_cpu = easyocr.Reader(['en'], gpu=False, verbose=False)
        cpu_load_time = time.time() - start_cpu
        print(f"  CPU Reader loaded: {cpu_load_time:.2f}s")

        return gpu_load_time, cpu_load_time

    def preprocess_image(self, image):
        """Standard preprocessing"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Moderate resize for balance between speed and accuracy
        h, w = gray.shape
        if w > 800:
            scale = 800 / w
            new_h = int(h * scale)
            gray = cv2.resize(gray, (800, new_h), interpolation=cv2.INTER_LINEAR)

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        return enhanced

    def run_ocr_gpu(self, image):
        """Run OCR with GPU"""
        if self.reader_gpu is None:
            return "", 0

        start = time.time()

        try:
            results = self.reader_gpu.readtext(
                image,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                threshold=0.01,
                mag_ratio=1.5
            )

            text = ""
            if results:
                # Get highest confidence text
                best_conf = 0
                for bbox, detected_text, conf in results:
                    if conf > best_conf and detected_text:
                        text = detected_text.upper()
                        best_conf = conf

        except Exception as e:
            print(f"    GPU OCR Error: {e}")
            text = ""

        elapsed = time.time() - start
        return text, elapsed

    def run_ocr_cpu(self, image):
        """Run OCR with CPU for comparison"""
        start = time.time()

        try:
            results = self.reader_cpu.readtext(
                image,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                threshold=0.01,
                mag_ratio=1.5
            )

            text = ""
            if results:
                # Get highest confidence text
                best_conf = 0
                for bbox, detected_text, conf in results:
                    if conf > best_conf and detected_text:
                        text = detected_text.upper()
                        best_conf = conf

        except Exception as e:
            print(f"    CPU OCR Error: {e}")
            text = ""

        elapsed = time.time() - start
        return text, elapsed

    def run_comparison_test(self):
        """Compare GPU vs CPU performance"""
        # Initialize readers
        gpu_init_time, cpu_init_time = self.initialize_readers()

        # Get test images
        image_files = sorted(list(self.vlm_dir.glob("*.jpg")))

        print(f"\n{'='*70}")
        print(f"Processing {len(image_files)} images...")
        print(f"{'='*70}\n")

        results = []
        total_gpu_time = 0
        total_cpu_time = 0
        gpu_detected = 0
        cpu_detected = 0

        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] {img_path.name[:40]}")

            # Load and preprocess
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ❌ Error loading image")
                continue

            processed = self.preprocess_image(img)

            # GPU OCR
            if self.gpu_available and self.reader_gpu:
                gpu_text, gpu_time = self.run_ocr_gpu(processed)
                total_gpu_time += gpu_time
                if gpu_text:
                    gpu_detected += 1
                    print(f"    🚀 GPU: '{gpu_text[:20]}' - {gpu_time:.3f}s")
                else:
                    print(f"    🚀 GPU: No text - {gpu_time:.3f}s")
            else:
                gpu_text, gpu_time = "", 0

            # CPU OCR
            cpu_text, cpu_time = self.run_ocr_cpu(processed)
            total_cpu_time += cpu_time
            if cpu_text:
                cpu_detected += 1
                print(f"    🖥️  CPU: '{cpu_text[:20]}' - {cpu_time:.3f}s")
            else:
                print(f"    🖥️  CPU: No text - {cpu_time:.3f}s")

            # Speedup calculation
            if gpu_time > 0 and cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"    ⚡ Speedup: {speedup:.2f}x faster on GPU")

            # Create comparison image
            h, w = img.shape[:2]

            # Add overlay
            cv2.rectangle(img, (0, h-100), (w, h), (0,0,0), -1)

            # GPU result
            gpu_display = f"GPU: {gpu_text[:20] if gpu_text else 'NO TEXT'} ({gpu_time:.3f}s)"
            cv2.putText(img, gpu_display, (10, h-65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # CPU result
            cpu_display = f"CPU: {cpu_text[:20] if cpu_text else 'NO TEXT'} ({cpu_time:.3f}s)"
            cv2.putText(img, cpu_display, (10, h-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # Speedup
            if gpu_time > 0 and cpu_time > 0:
                speedup_display = f"Speedup: {speedup:.1f}x"
                cv2.putText(img, speedup_display, (10, h-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Save comparison image
            output_path = self.output_dir / f"gpu_comparison_{idx:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), img)

            results.append({
                "index": idx,
                "file": img_path.name,
                "gpu_text": gpu_text,
                "gpu_time": round(gpu_time, 3),
                "cpu_text": cpu_text,
                "cpu_time": round(cpu_time, 3),
                "speedup": round(speedup, 2) if gpu_time > 0 and cpu_time > 0 else 0,
                "output": output_path.name
            })

        # Calculate statistics
        avg_gpu_time = total_gpu_time / len(results) if results and self.gpu_available else 0
        avg_cpu_time = total_cpu_time / len(results) if results else 0
        avg_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0

        # Summary
        print(f"\n{'='*70}")
        print("PERFORMANCE COMPARISON SUMMARY")
        print(f"{'='*70}")

        print(f"\n📊 Detection Results:")
        if self.gpu_available:
            print(f"  GPU: {gpu_detected}/{len(results)} texts detected")
        print(f"  CPU: {cpu_detected}/{len(results)} texts detected")

        print(f"\n⏱️  Timing Statistics:")
        print(f"  Initialization:")
        if self.gpu_available:
            print(f"    GPU: {gpu_init_time:.2f}s")
        print(f"    CPU: {cpu_init_time:.2f}s")

        print(f"\n  Processing Time:")
        if self.gpu_available:
            print(f"    GPU Average: {avg_gpu_time:.3f}s per image")
            print(f"    GPU Total:   {total_gpu_time:.2f}s for {len(results)} images")
        print(f"    CPU Average: {avg_cpu_time:.3f}s per image")
        print(f"    CPU Total:   {total_cpu_time:.2f}s for {len(results)} images")

        if self.gpu_available and avg_gpu_time > 0:
            print(f"\n⚡ Performance Improvement:")
            print(f"    Average Speedup: {avg_speedup:.2f}x faster on GPU")
            print(f"    Time Saved: {total_cpu_time - total_gpu_time:.2f}s")
            print(f"    Efficiency Gain: {((1 - avg_gpu_time/avg_cpu_time) * 100):.1f}%")

        # Best and worst speedups
        if self.gpu_available and results:
            speedups = [r['speedup'] for r in results if r['speedup'] > 0]
            if speedups:
                print(f"\n  Speedup Range:")
                print(f"    Best:  {max(speedups):.2f}x faster")
                print(f"    Worst: {min(speedups):.2f}x faster")

        # Save results
        summary = {
            "gpu_available": self.gpu_available,
            "gpu_device": torch.cuda.get_device_name(0) if self.gpu_available else None,
            "total_images": len(results),
            "gpu_detected": gpu_detected,
            "cpu_detected": cpu_detected,
            "timing": {
                "gpu_init_seconds": round(gpu_init_time, 2),
                "cpu_init_seconds": round(cpu_init_time, 2),
                "gpu_avg_seconds": round(avg_gpu_time, 3),
                "cpu_avg_seconds": round(avg_cpu_time, 3),
                "gpu_total_seconds": round(total_gpu_time, 2),
                "cpu_total_seconds": round(total_cpu_time, 2),
                "average_speedup": round(avg_speedup, 2)
            },
            "results": results
        }

        json_path = self.output_dir / "gpu_comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n📁 Results saved to:")
        print(f"   Directory: {self.output_dir}/")
        print(f"   JSON: gpu_comparison_results.json")
        print(f"\n✅ GPU acceleration test complete!")

        return summary


if __name__ == "__main__":
    tester = EasyOCRGPUTest()
    results = tester.run_comparison_test()