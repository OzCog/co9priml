"""
Comprehensive cognitive performance optimizer.

Integrates all performance optimization techniques:
- Memory pool management
- Pattern caching
- Parallel processing
- JIT compilation
- Performance profiling
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass

from .memory_pool import get_global_memory_pool, MemoryPool
from .pattern_cache import get_global_pattern_cache, PatternCache
from .parallel_processor import get_global_parallel_processor, ParallelCognitiveProcessor
from .jit_optimizer import get_global_jit_compiler, AdaptiveJITCompiler
from .profiler import get_global_profiler, PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for cognitive performance optimization."""
    # Memory optimization
    enable_memory_pooling: bool = True
    max_memory_pool_mb: int = 1024
    memory_optimization_interval: int = 300  # seconds
    
    # Caching optimization
    enable_pattern_caching: bool = True
    cache_hit_target: float = 0.6
    cache_cleanup_interval: int = 600  # seconds
    
    # Parallel execution
    enable_parallel_processing: bool = True
    max_parallel_workers: int = 8
    task_timeout_seconds: float = 30.0
    
    # JIT compilation
    enable_jit_compilation: bool = True
    jit_hot_threshold: float = 0.5
    min_calls_before_jit: int = 10
    
    # Performance monitoring
    enable_profiling: bool = True
    performance_report_interval: int = 900  # seconds
    
    # Optimization triggers
    memory_pressure_threshold: float = 0.8  # 80% of max memory
    slow_operation_threshold_ms: float = 100
    low_cache_hit_threshold: float = 0.3


class CognitiveOptimizer:
    """Comprehensive cognitive performance optimizer."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Initialize performance components
        self.memory_pool: MemoryPool = get_global_memory_pool()
        self.pattern_cache: PatternCache = get_global_pattern_cache()
        self.parallel_processor: ParallelCognitiveProcessor = get_global_parallel_processor()
        self.jit_compiler: AdaptiveJITCompiler = get_global_jit_compiler()
        self.profiler: PerformanceProfiler = get_global_profiler()
        
        # Optimization state
        self.optimization_enabled = True
        self.last_optimization_time = time.time()
        self.optimization_stats = {
            'memory_optimizations': 0,
            'cache_optimizations': 0,
            'jit_compilations': 0,
            'parallel_optimizations': 0,
            'performance_improvements': []
        }
        
        # Background optimization task
        self._optimization_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("CognitiveOptimizer initialized with comprehensive performance optimization")
    
    async def start_optimization(self) -> None:
        """Start background optimization processes."""
        if self._optimization_task is None or self._optimization_task.done():
            self._optimization_task = asyncio.create_task(self._optimization_loop())
            logger.info("Started background optimization processes")
    
    async def stop_optimization(self) -> None:
        """Stop background optimization processes."""
        self._shutdown_event.set()
        if self._optimization_task:
            await self._optimization_task
        logger.info("Stopped background optimization processes")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop running in background."""
        while not self._shutdown_event.is_set():
            try:
                await self._run_optimization_cycle()
                
                # Wait for next cycle or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=min(
                        self.config.memory_optimization_interval,
                        self.config.cache_cleanup_interval,
                        self.config.performance_report_interval
                    )
                )
                
            except asyncio.TimeoutError:
                # Continue optimization cycle
                continue
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _run_optimization_cycle(self) -> None:
        """Run a complete optimization cycle."""
        current_time = time.time()
        
        # Memory optimization
        if (current_time - self.last_optimization_time) >= self.config.memory_optimization_interval:
            await self._optimize_memory()
        
        # Cache optimization
        if (current_time - self.last_optimization_time) >= self.config.cache_cleanup_interval:
            await self._optimize_caches()
        
        # Performance analysis and JIT optimization
        if (current_time - self.last_optimization_time) >= self.config.performance_report_interval:
            await self._analyze_and_optimize_performance()
            self.last_optimization_time = current_time
    
    async def _optimize_memory(self) -> None:
        """Optimize memory usage."""
        if not self.config.enable_memory_pooling:
            return
        
        try:
            # Get current memory statistics
            memory_stats = self.memory_pool.get_memory_statistics()
            
            # Check if memory optimization is needed
            memory_usage_ratio = memory_stats['estimated_usage_mb'] / memory_stats['max_usage_mb']
            
            if memory_usage_ratio > self.config.memory_pressure_threshold:
                logger.info(f"Memory pressure detected ({memory_usage_ratio:.2%}), optimizing...")
                
                # Optimize memory usage
                optimization_result = self.memory_pool.optimize_memory_usage()
                
                self.optimization_stats['memory_optimizations'] += 1
                self.optimization_stats['performance_improvements'].append({
                    'type': 'memory',
                    'timestamp': time.time(),
                    'memory_freed_mb': optimization_result['memory_freed_mb'],
                    'usage_before': optimization_result['before']['estimated_usage_mb'],
                    'usage_after': optimization_result['after']['estimated_usage_mb']
                })
                
                logger.info(f"Memory optimization freed {optimization_result['memory_freed_mb']:.2f}MB")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    async def _optimize_caches(self) -> None:
        """Optimize cache performance."""
        if not self.config.enable_pattern_caching:
            return
        
        try:
            # Get cache efficiency report
            cache_report = self.pattern_cache.get_cache_efficiency_report()
            
            # Check if cache optimization is needed
            if cache_report['overall_hit_rate'] < self.config.low_cache_hit_threshold:
                logger.info(f"Low cache hit rate detected ({cache_report['overall_hit_rate']:.2%}), optimizing...")
                
                # Optimize cache sizes
                optimization_result = self.pattern_cache.optimize_cache_sizes()
                
                self.optimization_stats['cache_optimizations'] += 1
                self.optimization_stats['performance_improvements'].append({
                    'type': 'cache',
                    'timestamp': time.time(),
                    'hit_rate_before': cache_report['overall_hit_rate'],
                    'optimization_suggestions': optimization_result['optimization_suggestions']
                })
                
                logger.info(f"Cache optimization completed with {len(optimization_result['optimization_suggestions'])} suggestions")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    async def _analyze_and_optimize_performance(self) -> None:
        """Analyze performance and apply optimizations."""
        try:
            # Get performance summary
            perf_summary = self.profiler.get_performance_summary()
            
            # Identify bottlenecks
            bottlenecks = self.profiler.identify_bottlenecks()
            
            if bottlenecks:
                logger.info(f"Identified {len(bottlenecks)} performance bottlenecks")
                
                # JIT compile hot functions if enabled
                if self.config.enable_jit_compilation:
                    await self._optimize_with_jit(bottlenecks)
                
                # Optimize parallel processing
                if self.config.enable_parallel_processing:
                    await self._optimize_parallel_processing()
            
            # Log performance report
            self._log_performance_report(perf_summary, bottlenecks)
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    async def _optimize_with_jit(self, bottlenecks: List[Dict[str, Any]]) -> None:
        """Apply JIT compilation optimizations."""
        jit_report = self.jit_compiler.get_compilation_report()
        
        # Check for hot function candidates
        hot_candidates = jit_report.get('hot_functions_candidates', [])
        
        if hot_candidates:
            logger.info(f"Found {len(hot_candidates)} JIT compilation candidates")
            self.optimization_stats['jit_compilations'] += len(hot_candidates)
    
    async def _optimize_parallel_processing(self) -> None:
        """Optimize parallel processing configuration."""
        parallel_stats = self.parallel_processor.get_comprehensive_statistics()
        optimization_result = self.parallel_processor.optimize_worker_allocation()
        
        if optimization_result['optimization_suggestions']:
            logger.info(f"Parallel processing optimization suggestions: {optimization_result['optimization_suggestions']}")
            self.optimization_stats['parallel_optimizations'] += 1
    
    def _log_performance_report(self, perf_summary: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> None:
        """Log comprehensive performance report."""
        logger.info("=== COGNITIVE PERFORMANCE REPORT ===")
        logger.info(f"Uptime: {perf_summary.get('uptime_seconds', 0):.1f}s")
        logger.info(f"Total operations: {perf_summary.get('total_operations', 0)}")
        logger.info(f"Operations/second: {perf_summary.get('operations_per_second', 0):.2f}")
        
        if bottlenecks:
            logger.info(f"Top bottleneck: {bottlenecks[0]['operation_name']} ({bottlenecks[0]['avg_duration_ms']:.2f}ms avg)")
        
        # Memory stats
        memory_stats = self.memory_pool.get_memory_statistics()
        logger.info(f"Memory usage: {memory_stats['estimated_usage_mb']:.1f}MB "
                   f"(reuse ratio: {memory_stats['total_reuse_ratio']:.2%})")
        
        # Cache stats
        cache_stats = self.pattern_cache.get_cache_efficiency_report()
        logger.info(f"Cache hit rate: {cache_stats['overall_hit_rate']:.2%} "
                   f"(memory: {cache_stats['total_memory_usage_mb']:.1f}MB)")
        
        # JIT stats
        jit_report = self.jit_compiler.get_compilation_report()
        logger.info(f"JIT compiled functions: {jit_report['statistics']['functions_jit_compiled']}")
        
        logger.info("=====================================")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        return {
            'config': {
                'memory_pooling': self.config.enable_memory_pooling,
                'pattern_caching': self.config.enable_pattern_caching,
                'parallel_processing': self.config.enable_parallel_processing,
                'jit_compilation': self.config.enable_jit_compilation,
                'profiling': self.config.enable_profiling
            },
            'optimization_stats': self.optimization_stats.copy(),
            'memory_stats': self.memory_pool.get_memory_statistics(),
            'cache_stats': self.pattern_cache.get_cache_efficiency_report(),
            'parallel_stats': self.parallel_processor.get_comprehensive_statistics(),
            'jit_stats': self.jit_compiler.get_compilation_report(),
            'performance_summary': self.profiler.get_performance_summary()
        }
    
    def force_full_optimization(self) -> Dict[str, Any]:
        """Force immediate comprehensive optimization."""
        logger.info("Starting forced full optimization...")
        
        results = {}
        
        # Memory optimization
        if self.config.enable_memory_pooling:
            results['memory'] = self.memory_pool.optimize_memory_usage()
        
        # Cache optimization  
        if self.config.enable_pattern_caching:
            results['cache'] = self.pattern_cache.optimize_cache_sizes()
        
        # Performance analysis
        bottlenecks = self.profiler.identify_bottlenecks()
        results['bottlenecks'] = bottlenecks
        
        # JIT optimization
        if self.config.enable_jit_compilation:
            results['jit'] = self.jit_compiler.get_compilation_report()
        
        # Parallel optimization
        if self.config.enable_parallel_processing:
            results['parallel'] = self.parallel_processor.optimize_worker_allocation()
        
        logger.info("Forced full optimization completed")
        return results
    
    async def benchmark_optimizations(self, test_operations: List[Callable]) -> Dict[str, Any]:
        """Benchmark the effectiveness of optimizations."""
        logger.info("Starting optimization benchmark...")
        
        benchmark_results = {}
        
        for i, operation in enumerate(test_operations):
            operation_name = f"benchmark_operation_{i}"
            
            # Benchmark without optimizations
            self.optimization_enabled = False
            
            start_time = time.time()
            for _ in range(100):  # Run 100 iterations
                operation()
            baseline_time = time.time() - start_time
            
            # Benchmark with optimizations
            self.optimization_enabled = True
            
            start_time = time.time()
            for _ in range(100):
                operation()
            optimized_time = time.time() - start_time
            
            # Calculate improvement
            improvement = ((baseline_time - optimized_time) / baseline_time) * 100
            
            benchmark_results[operation_name] = {
                'baseline_time_ms': baseline_time * 1000,
                'optimized_time_ms': optimized_time * 1000,
                'improvement_percent': improvement,
                'speedup_factor': baseline_time / optimized_time if optimized_time > 0 else 0
            }
        
        logger.info("Optimization benchmark completed")
        return benchmark_results


# Global cognitive optimizer instance
_global_cognitive_optimizer: Optional[CognitiveOptimizer] = None
_optimizer_lock = threading.Lock()


def get_global_cognitive_optimizer() -> CognitiveOptimizer:
    """Get the global cognitive optimizer instance (thread-safe singleton)."""
    global _global_cognitive_optimizer
    
    if _global_cognitive_optimizer is None:
        with _optimizer_lock:
            if _global_cognitive_optimizer is None:
                _global_cognitive_optimizer = CognitiveOptimizer()
    
    return _global_cognitive_optimizer


def configure_global_cognitive_optimizer(config: OptimizationConfig) -> None:
    """Configure the global cognitive optimizer."""
    global _global_cognitive_optimizer
    
    with _optimizer_lock:
        if _global_cognitive_optimizer is not None:
            # Stop existing optimizer
            asyncio.create_task(_global_cognitive_optimizer.stop_optimization())
        
        _global_cognitive_optimizer = CognitiveOptimizer(config)
        logger.info("Global cognitive optimizer reconfigured")


async def start_global_optimization() -> None:
    """Start global optimization processes."""
    optimizer = get_global_cognitive_optimizer()
    await optimizer.start_optimization()


async def stop_global_optimization() -> None:
    """Stop global optimization processes."""
    optimizer = get_global_cognitive_optimizer()
    await optimizer.stop_optimization()