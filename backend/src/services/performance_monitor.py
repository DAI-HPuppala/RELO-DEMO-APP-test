"""
Performance Monitoring and Benchmarking Service
Tracks VLM inference metrics and provides optimization insights
"""

import time
import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Deque
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
import json

from config.gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)

@dataclass
class InferenceMetric:
    """Single inference metric data"""
    timestamp: float
    duration_ms: float
    batch_size: int
    model: str
    agent: str
    success: bool
    gpu_memory_used: Optional[int] = None
    ttft_ms: Optional[float] = None  # Time to first token
    tokens_per_second: Optional[float] = None
    error: Optional[str] = None

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for VLM inference
    Tracks metrics, identifies bottlenecks, and provides optimization suggestions
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: Deque[InferenceMetric] = deque(maxlen=history_size)
        self.agent_metrics: Dict[str, Deque[InferenceMetric]] = defaultdict(lambda: deque(maxlen=100))
        self.start_time = time.time()
        self.total_inferences = 0
        self.successful_inferences = 0
        self.total_tokens_generated = 0
        
        # Performance thresholds (milliseconds)
        self.thresholds = {
            "excellent": 500,   # < 500ms
            "good": 1000,       # < 1s
            "acceptable": 2000, # < 2s
            "slow": 3000       # < 3s
        }
        
        # Track GPU memory over time
        self.memory_history: Deque[Tuple[float, int, int, int]] = deque(maxlen=100)
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def record_inference(self, 
                         duration_ms: float,
                         batch_size: int = 1,
                         model: str = "qwen2.5vl:3b",
                         agent: str = "unknown",
                         success: bool = True,
                         ttft_ms: Optional[float] = None,
                         tokens_per_second: Optional[float] = None,
                         error: Optional[str] = None) -> None:
        """Record a single inference metric"""
        
        # Get current GPU memory
        try:
            _, used, _ = gpu_optimizer.get_gpu_memory_info()
            gpu_memory = used
        except:
            gpu_memory = None
        
        metric = InferenceMetric(
            timestamp=time.time(),
            duration_ms=duration_ms,
            batch_size=batch_size,
            model=model,
            agent=agent,
            success=success,
            gpu_memory_used=gpu_memory,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            error=error
        )
        
        self.metrics.append(metric)
        self.agent_metrics[agent].append(metric)
        
        self.total_inferences += 1
        if success:
            self.successful_inferences += 1
        
        if tokens_per_second:
            self.total_tokens_generated += int(tokens_per_second * (duration_ms / 1000))
        
        # Log if slow
        if duration_ms > self.thresholds["slow"]:
            logger.warning(f"Slow inference detected: {duration_ms:.0f}ms for {agent}")
    
    def get_performance_category(self, duration_ms: float) -> str:
        """Categorize performance based on duration"""
        if duration_ms < self.thresholds["excellent"]:
            return "excellent"
        elif duration_ms < self.thresholds["good"]:
            return "good"
        elif duration_ms < self.thresholds["acceptable"]:
            return "acceptable"
        elif duration_ms < self.thresholds["slow"]:
            return "slow"
        else:
            return "very_slow"
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.metrics:
            return {"message": "No metrics recorded yet"}
        
        recent_metrics = list(self.metrics)[-100:]  # Last 100 inferences
        recent_durations = [m.duration_ms for m in recent_metrics if m.success]
        
        if not recent_durations:
            return {"message": "No successful inferences yet"}
        
        # Calculate statistics
        stats = {
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "success_rate": (self.successful_inferences / self.total_inferences * 100) if self.total_inferences > 0 else 0,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "recent_performance": {
                "count": len(recent_durations),
                "avg_ms": statistics.mean(recent_durations),
                "median_ms": statistics.median(recent_durations),
                "min_ms": min(recent_durations),
                "max_ms": max(recent_durations),
                "p95_ms": self._calculate_percentile(recent_durations, 95),
                "p99_ms": self._calculate_percentile(recent_durations, 99)
            }
        }
        
        # Add throughput metrics
        if self.total_tokens_generated > 0:
            total_time = time.time() - self.start_time
            stats["throughput"] = {
                "tokens_per_second": self.total_tokens_generated / total_time,
                "inferences_per_minute": (self.total_inferences / total_time) * 60
            }
        
        # Add performance distribution
        distribution = defaultdict(int)
        for duration in recent_durations:
            category = self.get_performance_category(duration)
            distribution[category] += 1
        
        stats["performance_distribution"] = dict(distribution)
        
        # Add GPU memory stats
        memory_stats = gpu_optimizer.get_memory_stats()
        stats["gpu_memory"] = memory_stats
        
        return stats
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics for a specific agent"""
        if agent_name not in self.agent_metrics:
            return {"error": f"No metrics for agent: {agent_name}"}
        
        agent_metrics = list(self.agent_metrics[agent_name])
        if not agent_metrics:
            return {"error": "No metrics recorded for this agent"}
        
        durations = [m.duration_ms for m in agent_metrics if m.success]
        
        if not durations:
            return {"error": "No successful inferences for this agent"}
        
        return {
            "agent": agent_name,
            "total_inferences": len(agent_metrics),
            "successful": len(durations),
            "success_rate": (len(durations) / len(agent_metrics) * 100),
            "performance": {
                "avg_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p95_ms": self._calculate_percentile(durations, 95)
            }
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Provide optimization suggestions based on metrics"""
        suggestions = []
        
        if not self.metrics:
            return ["No metrics available for analysis"]
        
        recent_metrics = list(self.metrics)[-100:]
        recent_durations = [m.duration_ms for m in recent_metrics if m.success]
        
        if not recent_durations:
            return ["No successful inferences to analyze"]
        
        avg_duration = statistics.mean(recent_durations)
        
        # Check performance
        if avg_duration > self.thresholds["slow"]:
            suggestions.append("⚠️ Average inference time is slow. Consider:")
            suggestions.append("  - Reducing batch size")
            suggestions.append("  - Enabling Flash Attention")
            suggestions.append("  - Using 4-bit quantization")
        
        # Check memory pressure
        memory_stats = gpu_optimizer.get_memory_stats()
        if memory_stats["pressure"] > 0.9:
            suggestions.append("⚠️ High GPU memory pressure detected. Consider:")
            suggestions.append("  - Reducing model precision to bfloat16")
            suggestions.append("  - Lowering batch size")
            suggestions.append("  - Unloading unused models")
        
        # Check batch efficiency
        batch_sizes = [m.batch_size for m in recent_metrics]
        avg_batch = statistics.mean(batch_sizes)
        if avg_batch < 2:
            suggestions.append("💡 Low batch utilization. Consider:")
            suggestions.append("  - Implementing dynamic batching")
            suggestions.append("  - Accumulating requests before processing")
        
        # Check success rate
        if self.successful_inferences < self.total_inferences * 0.95:
            suggestions.append("⚠️ Success rate below 95%. Check:")
            suggestions.append("  - GPU memory availability")
            suggestions.append("  - Model loading issues")
            suggestions.append("  - Timeout settings")
        
        if not suggestions:
            suggestions.append("✅ Performance is optimal!")
        
        return suggestions
    
    async def start_monitoring(self, interval: int = 10):
        """Start continuous monitoring loop"""
        if self._monitoring_task:
            logger.info("Monitoring already running")
            return
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Started performance monitoring (interval: {interval}s)")
    
    async def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while True:
            try:
                # Record GPU memory
                total, used, free = gpu_optimizer.get_gpu_memory_info()
                self.memory_history.append((time.time(), total, used, free))
                
                # Log current stats periodically
                stats = self.get_current_stats()
                if stats.get("recent_performance"):
                    avg_ms = stats["recent_performance"]["avg_ms"]
                    p95_ms = stats["recent_performance"]["p95_ms"]
                    logger.info(f"📊 Performance: avg={avg_ms:.0f}ms, p95={p95_ms:.0f}ms")
                
                # Check for issues
                if len(self.memory_history) > 5:
                    recent_memory = [m[2] for m in list(self.memory_history)[-5:]]
                    if all(m > total * 0.95 for m in recent_memory):
                        logger.warning("⚠️ Sustained high GPU memory usage detected!")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop monitoring loop"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
            logger.info("Stopped performance monitoring")
    
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        metrics_data = {
            "export_time": datetime.now().isoformat(),
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "metrics": [
                {
                    "timestamp": m.timestamp,
                    "duration_ms": m.duration_ms,
                    "batch_size": m.batch_size,
                    "model": m.model,
                    "agent": m.agent,
                    "success": m.success,
                    "gpu_memory_used": m.gpu_memory_used
                }
                for m in self.metrics
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.agent_metrics.clear()
        self.memory_history.clear()
        self.total_inferences = 0
        self.successful_inferences = 0
        self.total_tokens_generated = 0
        self.start_time = time.time()
        logger.info("Performance metrics reset")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# Convenience function for recording
def record_inference(duration_ms: float, **kwargs):
    """Convenience function to record inference metric"""
    performance_monitor.record_inference(duration_ms, **kwargs)

__all__ = ['performance_monitor', 'PerformanceMonitor', 'record_inference', 'InferenceMetric']