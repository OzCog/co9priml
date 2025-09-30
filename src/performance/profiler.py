"""
Performance profiling and monitoring utilities.

Provides comprehensive profiling of cognitive operations including:
- Execution time tracking
- Memory usage monitoring
- Operation frequency analysis
- Performance bottleneck identification
"""

import time
import threading
import functools
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance measurement."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    thread_id: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics for an operation."""
    operation_name: str
    call_count: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    std_duration_ms: float
    total_memory_delta_mb: float
    avg_memory_delta_mb: float
    recent_calls: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """Add a new metric measurement."""
        self.call_count += 1
        self.total_duration_ms += metric.duration_ms
        self.total_memory_delta_mb += metric.memory_delta_mb
        
        # Update aggregations
        self.avg_duration_ms = self.total_duration_ms / self.call_count
        self.avg_memory_delta_mb = self.total_memory_delta_mb / self.call_count
        
        self.min_duration_ms = min(self.min_duration_ms, metric.duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, metric.duration_ms)
        
        self.recent_calls.append(metric)
        
        # Calculate standard deviation for recent calls
        if len(self.recent_calls) > 1:
            recent_durations = [m.duration_ms for m in self.recent_calls]
            mean = sum(recent_durations) / len(recent_durations)
            variance = sum((d - mean) ** 2 for d in recent_durations) / len(recent_durations)
            self.std_duration_ms = variance ** 0.5


class PerformanceProfiler:
    """Comprehensive performance profiler for cognitive operations."""
    
    def __init__(self, enabled: bool = True, max_history: int = 1000):
        self.enabled = enabled
        self.max_history = max_history
        
        # Metrics storage
        self._metrics: deque = deque(maxlen=max_history)
        self._aggregated_metrics: Dict[str, AggregatedMetrics] = {}
        self._active_operations: Dict[int, Dict[str, Any]] = defaultdict(dict)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance thresholds for warnings
        self.slow_operation_threshold_ms = 100
        self.memory_leak_threshold_mb = 10
        
        # Statistics
        self._start_time = time.time()
        
        logger.info(f"PerformanceProfiler initialized (enabled={enabled})")
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback - return 0 if psutil not available
            return 0.0
    
    @contextmanager
    def profile_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling an operation."""
        if not self.enabled:
            yield
            return
        
        thread_id = threading.get_ident()
        start_time = time.time()
        start_memory = self._get_memory_usage_mb()
        
        # Store active operation info
        with self._lock:
            self._active_operations[thread_id][operation_name] = {
                'start_time': start_time,
                'start_memory': start_memory
            }
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage_mb()
            
            # Create metric
            duration_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            metric = PerformanceMetric(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_before_mb=start_memory,
                memory_after_mb=end_memory,
                memory_delta_mb=memory_delta,
                thread_id=thread_id,
                metadata=metadata or {}
            )
            
            # Store metric
            with self._lock:
                self._metrics.append(metric)
                
                # Update aggregated metrics
                if operation_name not in self._aggregated_metrics:
                    self._aggregated_metrics[operation_name] = AggregatedMetrics(
                        operation_name=operation_name,
                        call_count=0,
                        total_duration_ms=0,
                        avg_duration_ms=0,
                        min_duration_ms=float('inf'),
                        max_duration_ms=0,
                        std_duration_ms=0,
                        total_memory_delta_mb=0,
                        avg_memory_delta_mb=0
                    )
                
                self._aggregated_metrics[operation_name].add_metric(metric)
                
                # Clean up active operation
                if thread_id in self._active_operations:
                    self._active_operations[thread_id].pop(operation_name, None)
            
            # Check for performance issues
            self._check_performance_warnings(metric)
    
    def profile_function(self, operation_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            func_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_operation(func_name, metadata):
                    return func(*args, **kwargs)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.profile_operation(func_name, metadata):
                    return await func(*args, **kwargs)
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        
        return decorator
    
    def _check_performance_warnings(self, metric: PerformanceMetric) -> None:
        """Check for performance issues and log warnings."""
        if metric.duration_ms > self.slow_operation_threshold_ms:
            logger.warning(f"Slow operation detected: {metric.operation_name} took {metric.duration_ms:.2f}ms")
        
        if metric.memory_delta_mb > self.memory_leak_threshold_mb:
            logger.warning(f"High memory usage: {metric.operation_name} used {metric.memory_delta_mb:.2f}MB")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            if not self._metrics:
                return {'message': 'No performance data available'}
            
            # Overall statistics
            total_operations = len(self._metrics)
            uptime_seconds = time.time() - self._start_time
            operations_per_second = total_operations / max(1, uptime_seconds)
            
            # Top slowest operations
            sorted_by_avg_duration = sorted(
                self._aggregated_metrics.values(),
                key=lambda m: m.avg_duration_ms,
                reverse=True
            )[:10]
            
            # Most memory intensive operations
            sorted_by_memory = sorted(
                self._aggregated_metrics.values(),
                key=lambda m: m.avg_memory_delta_mb,
                reverse=True
            )[:10]
            
            # Most frequent operations
            sorted_by_frequency = sorted(
                self._aggregated_metrics.values(),
                key=lambda m: m.call_count,
                reverse=True
            )[:10]
            
            return {
                'uptime_seconds': uptime_seconds,
                'total_operations': total_operations,
                'operations_per_second': operations_per_second,
                'unique_operations': len(self._aggregated_metrics),
                'slowest_operations': [
                    {
                        'name': m.operation_name,
                        'avg_duration_ms': m.avg_duration_ms,
                        'call_count': m.call_count
                    } for m in sorted_by_avg_duration
                ],
                'memory_intensive_operations': [
                    {
                        'name': m.operation_name,
                        'avg_memory_delta_mb': m.avg_memory_delta_mb,
                        'call_count': m.call_count
                    } for m in sorted_by_memory
                ],
                'most_frequent_operations': [
                    {
                        'name': m.operation_name,
                        'call_count': m.call_count,
                        'avg_duration_ms': m.avg_duration_ms
                    } for m in sorted_by_frequency
                ]
            }
    
    def get_operation_details(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific operation."""
        with self._lock:
            if operation_name not in self._aggregated_metrics:
                return None
            
            agg = self._aggregated_metrics[operation_name]
            
            # Recent performance trend
            recent_durations = [m.duration_ms for m in agg.recent_calls]
            trend = "stable"
            if len(recent_durations) >= 10:
                first_half = recent_durations[:len(recent_durations)//2]
                second_half = recent_durations[len(recent_durations)//2:]
                
                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)
                
                if second_avg > first_avg * 1.2:
                    trend = "deteriorating"
                elif second_avg < first_avg * 0.8:
                    trend = "improving"
            
            return {
                'operation_name': agg.operation_name,
                'call_count': agg.call_count,
                'avg_duration_ms': agg.avg_duration_ms,
                'min_duration_ms': agg.min_duration_ms,
                'max_duration_ms': agg.max_duration_ms,
                'std_duration_ms': agg.std_duration_ms,
                'avg_memory_delta_mb': agg.avg_memory_delta_mb,
                'performance_trend': trend,
                'recent_calls': len(agg.recent_calls)
            }
    
    def identify_bottlenecks(self, min_call_count: int = 10) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        with self._lock:
            bottlenecks = []
            
            for agg in self._aggregated_metrics.values():
                if agg.call_count < min_call_count:
                    continue
                
                # Calculate bottleneck score based on:
                # - Average duration
                # - Frequency of calls
                # - Memory usage
                duration_score = agg.avg_duration_ms / 100  # Normalize to ~1 for 100ms operations
                frequency_score = agg.call_count / 100  # Normalize to ~1 for 100 calls
                memory_score = max(0, agg.avg_memory_delta_mb)  # Memory growth is bad
                
                bottleneck_score = duration_score * frequency_score + memory_score
                
                if bottleneck_score > 1.0:  # Threshold for considering it a bottleneck
                    bottlenecks.append({
                        'operation_name': agg.operation_name,
                        'bottleneck_score': bottleneck_score,
                        'avg_duration_ms': agg.avg_duration_ms,
                        'call_count': agg.call_count,
                        'avg_memory_delta_mb': agg.avg_memory_delta_mb,
                        'total_time_ms': agg.total_duration_ms
                    })
            
            # Sort by bottleneck score
            bottlenecks.sort(key=lambda x: x['bottleneck_score'], reverse=True)
            
            return bottlenecks
    
    def export_metrics_to_csv(self, filename: str) -> None:
        """Export detailed metrics to CSV file."""
        import csv
        
        with self._lock:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'operation_name', 'start_time', 'duration_ms', 
                    'memory_delta_mb', 'thread_id'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for metric in self._metrics:
                    writer.writerow({
                        'operation_name': metric.operation_name,
                        'start_time': metric.start_time,
                        'duration_ms': metric.duration_ms,
                        'memory_delta_mb': metric.memory_delta_mb,
                        'thread_id': metric.thread_id
                    })
        
        logger.info(f"Metrics exported to {filename}")
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._aggregated_metrics.clear()
            self._active_operations.clear()
        
        logger.info("Performance metrics cleared")


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None
_profiler_lock = threading.Lock()


def get_global_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance (thread-safe singleton)."""
    global _global_profiler
    
    if _global_profiler is None:
        with _profiler_lock:
            if _global_profiler is None:
                _global_profiler = PerformanceProfiler()
    
    return _global_profiler


def profile(operation_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """Convenient decorator for profiling functions using global profiler."""
    return get_global_profiler().profile_function(operation_name, metadata)


def profile_context(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenient context manager for profiling using global profiler."""
    return get_global_profiler().profile_operation(operation_name, metadata)


# Fix missing import
import asyncio