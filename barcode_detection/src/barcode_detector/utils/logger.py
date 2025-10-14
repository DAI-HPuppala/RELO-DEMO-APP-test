"""Logging utilities for barcode detector."""

import sys
import time
from pathlib import Path
from typing import Optional, Any, Dict
from functools import wraps
from loguru import logger

from ..core.config_manager import ConfigManager


class BarcodeLogger:
    """Enhanced logger for barcode detection system."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize logger with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self._setup_logger()
        
    def _setup_logger(self) -> None:
        """Setup loguru logger with custom configuration."""
        # Remove default logger
        logger.remove()
        
        # Create log directory if it doesn't exist
        log_file = Path(self.config.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Console handler with colors
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=self.config.logging.level,
            colorize=True,
            enqueue=True
        )
        
        # File handler
        logger.add(
            self.config.logging.file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                   "{name}:{function}:{line} | {message}",
            level=self.config.logging.level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True
        )
        
        # Performance log handler (if enabled)
        if self.config.logging.enable_performance_logging:
            performance_log = log_file.parent / "performance.log"
            logger.add(
                performance_log,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message}",
                filter=lambda record: record["extra"].get("performance", False),
                rotation="5 MB",
                retention="3 days",
                enqueue=True
            )
            
    def get_logger(self, name: str) -> Any:
        """Get a logger instance with given name.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logger.bind(name=name)
        
    def log_detection_start(self, source: str, algorithm: str) -> None:
        """Log detection start event.
        
        Args:
            source: Detection source (image/camera)
            algorithm: Detection algorithm used
        """
        logger.info(f"Starting detection - Source: {source}, Algorithm: {algorithm}")
        
    def log_detection_result(
        self, 
        success: bool, 
        barcodes_found: int, 
        processing_time: float,
        algorithm: str
    ) -> None:
        """Log detection result.
        
        Args:
            success: Whether detection was successful
            barcodes_found: Number of barcodes found
            processing_time: Processing time in milliseconds
            algorithm: Algorithm used
        """
        if success:
            logger.success(
                f"Detection completed - Found: {barcodes_found} barcodes, "
                f"Time: {processing_time:.2f}ms, Algorithm: {algorithm}"
            )
        else:
            logger.warning(
                f"Detection failed - Time: {processing_time:.2f}ms, "
                f"Algorithm: {algorithm}"
            )
            
    def log_performance(
        self, 
        operation: str, 
        duration: float, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in milliseconds
            details: Additional performance details
        """
        if self.config.logging.enable_performance_logging:
            details_str = f", Details: {details}" if details else ""
            logger.bind(performance=True).info(
                f"Operation: {operation}, Duration: {duration:.2f}ms{details_str}"
            )
            
    def log_error(self, error: Exception, context: str) -> None:
        """Log error with context.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
        """
        logger.error(f"Error in {context}: {type(error).__name__}: {error}")
        logger.exception("Exception details:")
        
    def log_fallback_usage(self, primary: str, fallback: str, reason: str) -> None:
        """Log fallback detector usage.
        
        Args:
            primary: Primary detector that failed
            fallback: Fallback detector used
            reason: Reason for fallback
        """
        logger.warning(
            f"Fallback triggered - Primary: {primary}, Fallback: {fallback}, "
            f"Reason: {reason}"
        )


def performance_monitor(operation_name: str):
    """Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = e
                
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Log performance
            logger.bind(performance=True).info(
                f"Operation: {operation_name}, Duration: {duration_ms:.2f}ms, "
                f"Success: {success}"
            )
            
            if not success:
                logger.error(f"Operation {operation_name} failed: {error}")
                raise error
                
            return result
        return wrapper
    return decorator


def log_function_call(func):
    """Decorator to log function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    return wrapper