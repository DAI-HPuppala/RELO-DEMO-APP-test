"""Performance benchmarking utilities for barcode detection."""

import time
import statistics
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

from ..core.config_manager import ConfigManager
from ..core.detector_factory import DetectorFactory
from ..core.image_processor import ImageProcessor
from ..utils.logger import BarcodeLogger, performance_monitor
from ..interfaces.barcode_detector import DetectionResult


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(
        self,
        detector_name: str,
        total_images: int,
        total_detections: int,
        processing_times: List[float],
        detection_accuracy: float = 0.0,
        memory_usage_mb: float = 0.0
    ):
        """Initialize benchmark result.
        
        Args:
            detector_name: Name of the detector
            total_images: Total number of images processed
            total_detections: Total number of barcodes detected
            processing_times: List of processing times in milliseconds
            detection_accuracy: Detection accuracy score (0-1)
            memory_usage_mb: Memory usage in MB
        """
        self.detector_name = detector_name
        self.total_images = total_images
        self.total_detections = total_detections
        self.processing_times = processing_times
        self.detection_accuracy = detection_accuracy
        self.memory_usage_mb = memory_usage_mb
        
        # Calculate statistics
        if processing_times:
            self.avg_processing_time = statistics.mean(processing_times)
            self.median_processing_time = statistics.median(processing_times)
            self.std_processing_time = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0
            self.min_processing_time = min(processing_times)
            self.max_processing_time = max(processing_times)
            self.fps = 1000.0 / self.avg_processing_time if self.avg_processing_time > 0 else 0.0
        else:
            self.avg_processing_time = 0.0
            self.median_processing_time = 0.0
            self.std_processing_time = 0.0
            self.min_processing_time = 0.0
            self.max_processing_time = 0.0
            self.fps = 0.0
            
        self.detection_rate = total_detections / total_images if total_images > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detector_name": self.detector_name,
            "total_images": self.total_images,
            "total_detections": self.total_detections,
            "detection_rate": self.detection_rate,
            "detection_accuracy": self.detection_accuracy,
            "avg_processing_time_ms": self.avg_processing_time,
            "median_processing_time_ms": self.median_processing_time,
            "std_processing_time_ms": self.std_processing_time,
            "min_processing_time_ms": self.min_processing_time,
            "max_processing_time_ms": self.max_processing_time,
            "fps": self.fps,
            "memory_usage_mb": self.memory_usage_mb
        }
        
    def __str__(self) -> str:
        """String representation."""
        return f"""BenchmarkResult for {self.detector_name}:
  Images processed: {self.total_images}
  Detections: {self.total_detections} ({self.detection_rate:.2%} rate)
  Avg processing time: {self.avg_processing_time:.1f}ms
  FPS: {self.fps:.1f}
  Accuracy: {self.detection_accuracy:.2%}
  Memory usage: {self.memory_usage_mb:.1f}MB"""


class PerformanceBenchmark:
    """Performance benchmarking system for barcode detectors."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize benchmark system.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = BarcodeLogger(config)
        self.image_processor = ImageProcessor(self.logger)
        self.detector_factory = DetectorFactory(self.config, self.logger)
        
    @performance_monitor("benchmark_detector")
    def benchmark_detector(
        self,
        detector_name: str,
        image_paths: List[Path],
        ground_truth: Optional[Dict[str, List[str]]] = None,
        warmup_runs: int = 5,
        measure_memory: bool = True
    ) -> BenchmarkResult:
        """Benchmark a specific detector.
        
        Args:
            detector_name: Name of detector to benchmark
            image_paths: List of image paths to test
            ground_truth: Optional ground truth data for accuracy calculation
            warmup_runs: Number of warmup runs before measuring
            measure_memory: Whether to measure memory usage
            
        Returns:
            Benchmark results
        """
        self.logger.get_logger("benchmark").info(f"Benchmarking detector: {detector_name}")
        
        # Create detector
        original_mode = self.config.detection.mode
        self.config.detection.mode = detector_name
        detector = self.detector_factory.create_detector()
        
        if not detector:
            self.logger.get_logger("benchmark").error(f"Failed to create detector: {detector_name}")
            return BenchmarkResult(detector_name, 0, 0, [])
            
        try:
            # Memory measurement setup
            memory_usage = 0.0
            if measure_memory:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                
            processing_times = []
            total_detections = 0
            correct_detections = 0
            processed_images = 0
            
            # Warmup runs
            if warmup_runs > 0 and image_paths:
                self.logger.get_logger("benchmark").info(f"Performing {warmup_runs} warmup runs")
                warmup_image = self.image_processor.load_image(image_paths[0])
                if warmup_image is not None:
                    for _ in range(warmup_runs):
                        detector.detect(warmup_image)
                        
            # Measure memory after warmup
            if measure_memory:
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
            # Process each image
            for image_path in image_paths:
                self.logger.get_logger("benchmark").debug(f"Processing: {image_path}")
                
                # Load image
                image = self.image_processor.load_image(image_path)
                if image is None:
                    continue
                    
                # Measure detection time
                start_time = time.perf_counter()
                results = detector.detect(image)
                end_time = time.perf_counter()
                
                processing_time = (end_time - start_time) * 1000  # ms
                processing_times.append(processing_time)
                
                total_detections += len(results)
                processed_images += 1
                
                # Calculate accuracy if ground truth available
                if ground_truth and str(image_path) in ground_truth:
                    expected_codes = set(ground_truth[str(image_path)])
                    detected_codes = set(result.data for result in results)
                    
                    # Count correct detections
                    correct_detections += len(expected_codes.intersection(detected_codes))
                    
            # Calculate accuracy
            detection_accuracy = 0.0
            if ground_truth:
                total_expected = sum(len(codes) for codes in ground_truth.values())
                detection_accuracy = correct_detections / total_expected if total_expected > 0 else 0.0
                
            return BenchmarkResult(
                detector_name=detector_name,
                total_images=processed_images,
                total_detections=total_detections,
                processing_times=processing_times,
                detection_accuracy=detection_accuracy,
                memory_usage_mb=memory_usage
            )
            
        finally:
            detector.cleanup()
            self.config.detection.mode = original_mode
            
    def benchmark_all_detectors(
        self,
        image_paths: List[Path],
        ground_truth: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark all available detectors.
        
        Args:
            image_paths: List of image paths to test
            ground_truth: Optional ground truth data
            
        Returns:
            Dictionary of benchmark results by detector name
        """
        available_detectors = self.detector_factory.get_available_detectors()
        results = {}
        
        for detector_name, is_available in available_detectors.items():
            if is_available:
                try:
                    result = self.benchmark_detector(detector_name, image_paths, ground_truth)
                    results[detector_name] = result
                except Exception as e:
                    self.logger.log_error(e, f"benchmark_{detector_name}")
                    
        return results
        
    def benchmark_concurrent_processing(
        self,
        image_paths: List[Path],
        max_workers: int = 4,
        use_processes: bool = False
    ) -> Dict[str, Any]:
        """Benchmark concurrent processing performance.
        
        Args:
            image_paths: List of image paths to test
            max_workers: Maximum number of concurrent workers
            use_processes: Whether to use processes instead of threads
            
        Returns:
            Dictionary with concurrent processing results
        """
        self.logger.get_logger("benchmark").info(
            f"Benchmarking concurrent processing with {max_workers} workers"
        )
        
        # Create detector
        detector = self.detector_factory.create_detector()
        if not detector:
            return {"error": "Failed to create detector"}
            
        try:
            # Sequential processing baseline
            sequential_start = time.perf_counter()
            sequential_results = []
            
            for image_path in image_paths:
                image = self.image_processor.load_image(image_path)
                if image is not None:
                    results = detector.detect(image)
                    sequential_results.extend(results)
                    
            sequential_time = time.perf_counter() - sequential_start
            
            # Concurrent processing
            executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
            
            concurrent_start = time.perf_counter()
            concurrent_results = []
            
            def process_image(image_path: Path) -> List[DetectionResult]:
                """Process single image."""
                image = self.image_processor.load_image(image_path)
                if image is not None:
                    return detector.detect(image)
                return []
                
            with executor_class(max_workers=max_workers) as executor:
                futures = [executor.submit(process_image, path) for path in image_paths]
                
                for future in futures:
                    try:
                        results = future.result()
                        concurrent_results.extend(results)
                    except Exception as e:
                        self.logger.log_error(e, "concurrent_processing")
                        
            concurrent_time = time.perf_counter() - concurrent_start
            
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0
            efficiency = speedup / max_workers
            
            return {
                "sequential_time": sequential_time,
                "concurrent_time": concurrent_time,
                "speedup": speedup,
                "efficiency": efficiency,
                "sequential_detections": len(sequential_results),
                "concurrent_detections": len(concurrent_results),
                "max_workers": max_workers,
                "use_processes": use_processes
            }
            
        finally:
            detector.cleanup()
            
    def memory_leak_test(
        self,
        image_path: Path,
        iterations: int = 100,
        detector_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test for memory leaks.
        
        Args:
            image_path: Path to test image
            iterations: Number of iterations to run
            detector_name: Specific detector to test
            
        Returns:
            Dictionary with memory usage statistics
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Use specified detector or create default
        if detector_name:
            original_mode = self.config.detection.mode
            self.config.detection.mode = detector_name
            
        detector = self.detector_factory.create_detector()
        if not detector:
            return {"error": "Failed to create detector"}
            
        try:
            # Load test image
            test_image = self.image_processor.load_image(image_path)
            if test_image is None:
                return {"error": "Failed to load test image"}
                
            memory_samples = []
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(baseline_memory)
            
            # Run iterations
            for i in range(iterations):
                results = detector.detect(test_image)
                
                # Sample memory every 10 iterations
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    
            final_memory = process.memory_info().rss / 1024 / 1024
            
            # Calculate statistics
            memory_growth = final_memory - baseline_memory
            max_memory = max(memory_samples)
            avg_memory = statistics.mean(memory_samples)
            
            return {
                "detector_name": detector_name or "default",
                "iterations": iterations,
                "baseline_memory_mb": baseline_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": memory_growth,
                "max_memory_mb": max_memory,
                "avg_memory_mb": avg_memory,
                "memory_samples": memory_samples,
                "potential_leak": memory_growth > (baseline_memory * 0.1)  # >10% growth
            }
            
        finally:
            detector.cleanup()
            if detector_name:
                self.config.detection.mode = original_mode
                
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate benchmark report.
        
        Args:
            results: Benchmark results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "BARCODE DETECTION PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {self.config.detection.mode}",
            "",
        ]
        
        if not results:
            report_lines.append("No benchmark results available.")
            report = "\n".join(report_lines)
            
            if output_path:
                output_path.write_text(report)
                
            return report
            
        # Summary table
        report_lines.extend([
            "SUMMARY",
            "-" * 80,
            f"{'Detector':<15} {'Images':<8} {'Detections':<11} {'Avg Time':<10} {'FPS':<8} {'Accuracy':<10} {'Memory':<10}",
            "-" * 80,
        ])
        
        for name, result in results.items():
            report_lines.append(
                f"{name:<15} {result.total_images:<8} {result.total_detections:<11} "
                f"{result.avg_processing_time:<10.1f} {result.fps:<8.1f} "
                f"{result.detection_accuracy:<10.2%} {result.memory_usage_mb:<10.1f}"
            )
            
        report_lines.append("")
        
        # Detailed results
        for name, result in results.items():
            report_lines.extend([
                f"DETAILED RESULTS: {name}",
                "-" * 80,
                str(result),
                "",
            ])
            
        # Performance ranking
        if len(results) > 1:
            report_lines.extend([
                "PERFORMANCE RANKING",
                "-" * 80,
            ])
            
            # Rank by FPS
            fps_ranking = sorted(results.items(), key=lambda x: x[1].fps, reverse=True)
            report_lines.append("By Speed (FPS):")
            for i, (name, result) in enumerate(fps_ranking, 1):
                report_lines.append(f"  {i}. {name}: {result.fps:.1f} FPS")
                
            report_lines.append("")
            
            # Rank by accuracy
            if any(result.detection_accuracy > 0 for result in results.values()):
                accuracy_ranking = sorted(results.items(), key=lambda x: x[1].detection_accuracy, reverse=True)
                report_lines.append("By Accuracy:")
                for i, (name, result) in enumerate(accuracy_ranking, 1):
                    report_lines.append(f"  {i}. {name}: {result.detection_accuracy:.2%}")
                    
            report_lines.append("")
            
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            self.logger.get_logger("benchmark").info(f"Report saved to: {output_path}")
            
        return report
        
    def save_results_json(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: Path
    ) -> None:
        """Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results
            output_path: Output file path
        """
        json_data = {
            "timestamp": time.time(),
            "config": {
                "detection_mode": self.config.detection.mode,
                "input_source": self.config.input_source,
                "min_confidence": self.config.detection.min_confidence
            },
            "results": {name: result.to_dict() for name, result in results.items()}
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
        self.logger.get_logger("benchmark").info(f"Results saved to JSON: {output_path}")


def create_synthetic_test_images(output_dir: Path, count: int = 10) -> List[Path]:
    """Create synthetic barcode images for testing.
    
    Args:
        output_dir: Directory to save images
        count: Number of images to create
        
    Returns:
        List of created image paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    
    for i in range(count):
        # Create synthetic barcode image
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        # Add barcode pattern
        bar_positions = np.arange(50, 350, 8)
        bar_widths = np.random.choice([2, 3, 4, 5], size=len(bar_positions))
        
        for pos, width in zip(bar_positions, bar_widths):
            if pos + width < image.shape[1]:
                image[50:150, pos:pos+width] = 0
                
        # Add noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        # Save image
        image_path = output_dir / f"synthetic_barcode_{i:03d}.jpg"
        cv2.imwrite(str(image_path), image)
        image_paths.append(image_path)
        
    return image_paths


def run_benchmark(config: ConfigManager) -> int:
    """Run benchmark from command line.
    
    Args:
        config: Configuration manager
        
    Returns:
        Exit code
    """
    benchmark = PerformanceBenchmark(config)
    
    try:
        # Create or use existing test images
        image_dir = Path("benchmark_images")
        if not image_dir.exists() or not list(image_dir.glob("*.jpg")):
            print("Creating synthetic test images...")
            image_paths = create_synthetic_test_images(image_dir, 20)
        else:
            image_paths = list(image_dir.glob("*.jpg"))
            
        print(f"Benchmarking with {len(image_paths)} images...")
        
        # Run benchmarks
        results = benchmark.benchmark_all_detectors(image_paths)
        
        # Generate and display report
        report = benchmark.generate_report(results, Path("benchmark_report.txt"))
        print(report)
        
        # Save JSON results
        benchmark.save_results_json(results, Path("benchmark_results.json"))
        
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1