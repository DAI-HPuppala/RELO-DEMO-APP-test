"""Main application entry point for barcode detector."""

import sys
import time
from pathlib import Path
from typing import List, Optional
import argparse

from .core.config_manager import ConfigManager
from .core.detector_factory import DetectorFactory
from .core.image_processor import ImageProcessor
from .core.camera_processor import CameraProcessor
from .utils.logger import BarcodeLogger
from .interfaces.barcode_detector import DetectionResult


class BarcodeDetectorApp:
    """Main barcode detector application."""
    
    def __init__(self, config_file: Optional[str] = None) -> None:
        """Initialize application.
        
        Args:
            config_file: Optional configuration file path
        """
        # Initialize configuration
        self.config = ConfigManager(config_file)
        
        # Validate configuration
        validation = self.config.validate()
        if not validation["valid"]:
            print(f"Configuration validation failed: {validation['errors']}")
            sys.exit(1)
            
        if validation["warnings"]:
            print(f"Configuration warnings: {validation['warnings']}")
            
        # Initialize logger
        self.logger = BarcodeLogger(self.config)
        
        # Initialize components
        self.detector_factory = DetectorFactory(self.config, self.logger)
        self.image_processor = ImageProcessor(self.logger)
        self.camera_processor = CameraProcessor(
            self.config, self.logger, self.image_processor
        )
        
        # Application state
        self.detector = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize the application.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.get_logger("app").info("Initializing Barcode Detector Application")
        self.logger.get_logger("app").info(str(self.config))
        
        # Create detector
        self.detector = self.detector_factory.create_detector()
        if not self.detector:
            self.logger.get_logger("app").error("Failed to create detector")
            return False
            
        self.logger.get_logger("app").info(f"Created detector: {self.detector.name}")
        
        return True
        
    def run_image_detection(self) -> bool:
        """Run detection on images from configured directory.
        
        Returns:
            True if successful, False otherwise
        """
        image_path = Path(self.config.image_path)
        
        if not image_path.exists():
            self.logger.get_logger("app").error(f"Image path does not exist: {image_path}")
            return False
            
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Find image files
        if image_path.is_file():
            image_files = [image_path] if image_path.suffix.lower() in image_extensions else []
        else:
            image_files = [
                f for f in image_path.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
        if not image_files:
            self.logger.get_logger("app").error(f"No image files found in: {image_path}")
            return False
            
        self.logger.get_logger("app").info(f"Processing {len(image_files)} image(s)")
        
        total_detections = 0
        
        for image_file in image_files:
            self.logger.get_logger("app").info(f"Processing: {image_file.name}")
            
            # Load image
            image = self.image_processor.load_image(image_file)
            if image is None:
                continue
                
            # Process image
            results = self.image_processor.process_image(
                image, 
                self.detector, 
                source_info=str(image_file)
            )
            
            if results:
                total_detections += len(results)
                
                # Log results
                for result in results:
                    self.logger.get_logger("app").info(
                        f"  Detected: {result.type} = '{result.data}' "
                        f"(confidence: {result.confidence:.2f}, "
                        f"time: {result.processing_time_ms:.1f}ms)"
                    )
                    
                # Save annotated image if enabled
                if self.config.output.save_annotated_images:
                    annotated = self.image_processor.annotate_image(
                        image,
                        results,
                        show_bounding_boxes=self.config.output.show_bounding_boxes,
                        show_detection_time=self.config.output.show_detection_time
                    )
                    
                    output_path = Path(self.config.output.output_directory) / f"annotated_{image_file.name}"
                    self.image_processor.save_image(annotated, output_path)
                    
            else:
                self.logger.get_logger("app").info("  No barcodes detected")
                
        self.logger.get_logger("app").info(
            f"Image processing complete. Total detections: {total_detections}"
        )
        
        return True
        
    def run_camera_detection(self) -> bool:
        """Run live camera detection.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.get_logger("app").info("Starting live camera detection")
        
        # Initialize camera
        if not self.camera_processor.initialize_camera():
            self.logger.get_logger("app").error("Failed to initialize camera")
            return False
            
        # Setup detection result callback
        def on_detection_result(results: List[DetectionResult]) -> None:
            for result in results:
                self.logger.get_logger("app").info(
                    f"Live detected: {result.type} = '{result.data}' "
                    f"(confidence: {result.confidence:.2f}, "
                    f"time: {result.processing_time_ms:.1f}ms)"
                )
                
        # Start detection
        if not self.camera_processor.start_detection(self.detector, on_detection_result):
            self.logger.get_logger("app").error("Failed to start camera detection")
            return False
            
        try:
            # Display live feed
            self.camera_processor.display_live_feed("Barcode Detection - Live Feed")
            
        except KeyboardInterrupt:
            self.logger.get_logger("app").info("Interrupted by user")
        finally:
            # Cleanup
            self.camera_processor.cleanup()
            
        # Show final statistics
        stats = self.camera_processor.get_statistics()
        self.logger.get_logger("app").info(f"Camera detection statistics: {stats}")
        
        return True
        
    def run(self) -> bool:
        """Run the application based on configuration.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.initialize():
            return False
            
        self.running = True
        
        try:
            if self.config.input_source == "image":
                return self.run_image_detection()
            elif self.config.input_source == "camera":
                return self.run_camera_detection()
            else:
                self.logger.get_logger("app").error(
                    f"Unknown input source: {self.config.input_source}"
                )
                return False
                
        except Exception as e:
            self.logger.log_error(e, "main_run")
            return False
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        """Cleanup application resources."""
        if self.detector:
            self.detector.cleanup()
            
        self.camera_processor.cleanup()
        self.running = False
        
        self.logger.get_logger("app").info("Application cleanup complete")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.
    
    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description="Production-level barcode detector with custom and library implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration (camera mode)
  python -m barcode_detector.main
  
  # Run in image mode with custom config
  python -m barcode_detector.main --config custom.env --source image --path /path/to/images
  
  # Run in custom detection mode
  python -m barcode_detector.main --mode custom --source camera
  
  # Run performance benchmark
  python -m barcode_detector.main --benchmark --path /path/to/test/images
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path (default: .env)"
    )
    
    parser.add_argument(
        "--source", "-s",
        choices=["image", "camera"],
        help="Input source (overrides config)"
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        help="Image path (for image source, overrides config)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["custom", "library", "hybrid"],
        help="Detection mode (overrides config)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run performance benchmark"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    return parser


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Create application
        app = BarcodeDetectorApp(args.config)
        
        # Override configuration with command line arguments
        if args.source:
            app.config.set("INPUT_SOURCE", args.source)
            app.config.input_source = args.source
            
        if args.path:
            app.config.set("IMAGE_PATH", args.path)
            app.config.image_path = args.path
            
        if args.mode:
            app.config.set("DETECTION_MODE", args.mode)
            app.config.detection.mode = args.mode
            
        if args.output:
            app.config.set("OUTPUT_DIRECTORY", args.output)
            app.config.output.output_directory = args.output
            
        if args.verbose:
            app.config.set("LOG_LEVEL", "DEBUG")
            app.config.logging.level = "DEBUG"
            
        # Validate configuration if requested
        if args.validate_config:
            validation = app.config.validate()
            if validation["valid"]:
                print("✓ Configuration is valid")
                if validation["warnings"]:
                    print(f"Warnings: {validation['warnings']}")
                return 0
            else:
                print("✗ Configuration validation failed")
                print(f"Errors: {validation['errors']}")
                return 1
                
        # Run benchmark if requested
        if args.benchmark:
            from .utils.benchmark import run_benchmark
            return run_benchmark(app.config)
            
        # Run application
        success = app.run()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())