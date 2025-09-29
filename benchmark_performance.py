#!/usr/bin/env python3
"""
Performance Optimization Benchmark

Demonstrates the performance improvements achieved through the comprehensive
optimization system implemented for the CogPrime cognitive architecture.
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import performance optimizations
    from src.performance import (
        get_global_memory_pool,
        get_global_pattern_cache,
        get_global_profiler,
        get_global_jit_compiler,
        get_global_parallel_processor,
        get_global_cognitive_optimizer,
        OptimizationConfig,
        profile,
        jit_profile
    )
    
    # Import cognitive modules (if available)
    try:
        from src.modules.reasoning import ReasoningModule
        from src.unified_cognitive_kernel.cognitive_kernel import UnifiedCognitiveKernel, CognitiveKernelConfig
        COGNITIVE_MODULES_AVAILABLE = True
    except ImportError:
        COGNITIVE_MODULES_AVAILABLE = False
        logger.warning("Cognitive modules not available, using simulated operations")
    
    PERFORMANCE_AVAILABLE = True
    
except ImportError as e:
    logger.error(f"Performance optimizations not available: {e}")
    PERFORMANCE_AVAILABLE = False


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.results = {}
        self.baseline_times = {}
        self.optimized_times = {}
        
        if PERFORMANCE_AVAILABLE:
            # Initialize performance components
            self.memory_pool = get_global_memory_pool()
            self.pattern_cache = get_global_pattern_cache()
            self.profiler = get_global_profiler()
            self.jit_compiler = get_global_jit_compiler()
            self.parallel_processor = get_global_parallel_processor()
            self.cognitive_optimizer = get_global_cognitive_optimizer()
            
            logger.info("Performance benchmark initialized with all optimizations")
        else:
            logger.warning("Performance optimizations not available - benchmark limited")
    
    def simulate_cognitive_operation(self, complexity: int = 100, use_optimizations: bool = True) -> Dict[str, Any]:
        """Simulate a cognitive operation with variable complexity."""
        
        @profile('simulated_cognitive_op') if use_optimizations and PERFORMANCE_AVAILABLE else lambda x: x
        @jit_profile('simulated_cognitive_op') if use_optimizations and PERFORMANCE_AVAILABLE else lambda x: x
        def cognitive_work():
            # Simulate pattern matching
            patterns = []
            for i in range(complexity):
                pattern_strength = (i * 0.1) % 1.0
                pattern_type = ['temporal', 'spatial', 'hierarchical'][i % 3]
                patterns.append({
                    'strength': pattern_strength,
                    'type': pattern_type,
                    'frequency': i % 10
                })
            
            # Simulate memory retrieval and consolidation
            memories = []
            for i in range(min(complexity // 10, 20)):
                memory_relevance = 1.0 / (i + 1)  # Decay function
                memories.append({
                    'content': f"memory_{i}",
                    'relevance': memory_relevance,
                    'age': i * 10
                })
            
            # Simulate reasoning computation
            reasoning_result = sum(p['strength'] * p['frequency'] for p in patterns) / len(patterns)
            
            # Simulate attention allocation
            attention_weights = [1.0 / (i + 1) for i in range(len(memories))]
            total_attention = sum(attention_weights)
            normalized_attention = [w / total_attention for w in attention_weights] if total_attention > 0 else []
            
            return {
                'patterns_detected': len(patterns),
                'memories_retrieved': len(memories),
                'reasoning_score': reasoning_result,
                'attention_distribution': normalized_attention,
                'cognitive_load': complexity
            }
        
        return cognitive_work()
    
    def benchmark_memory_operations(self, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark memory pool operations."""
        if not PERFORMANCE_AVAILABLE:
            return {'error': 'Performance optimizations not available'}
        
        logger.info(f"Benchmarking memory operations ({iterations} iterations)")
        
        # Baseline: Standard tensor allocation
        baseline_times = []
        for _ in range(iterations):
            start = time.time()
            
            # Simulate tensor allocation without pool
            try:
                import torch
                tensors = [torch.randn(100, 50) for _ in range(5)]
                del tensors  # Force cleanup
            except ImportError:
                import numpy as np
                arrays = [np.random.randn(100, 50) for _ in range(5)]
                del arrays
            
            baseline_times.append(time.time() - start)
        
        # Optimized: Memory pool allocation
        optimized_times = []
        for _ in range(iterations):
            start = time.time()
            
            # Use memory pool
            tensors = []
            for _ in range(5):
                tensor = self.memory_pool.get_working_tensor((100, 50))
                tensors.append(tensor)
            
            # Return to pool
            for tensor in tensors:
                self.memory_pool.return_tensor(tensor, 'working')
            
            optimized_times.append(time.time() - start)
        
        baseline_avg = statistics.mean(baseline_times) * 1000  # Convert to ms
        optimized_avg = statistics.mean(optimized_times) * 1000
        improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
        
        return {
            'baseline_avg_ms': baseline_avg,
            'optimized_avg_ms': optimized_avg,
            'improvement_percent': improvement,
            'speedup_factor': baseline_avg / optimized_avg if optimized_avg > 0 else 0
        }
    
    def benchmark_pattern_caching(self, iterations: int = 500) -> Dict[str, float]:
        """Benchmark pattern caching effectiveness."""
        if not PERFORMANCE_AVAILABLE:
            return {'error': 'Performance optimizations not available'}
        
        logger.info(f"Benchmarking pattern caching ({iterations} iterations)")
        
        # Create test patterns
        test_patterns = [
            {'pattern_id': f'pattern_{i}', 'content_hash': f'hash_{i % 50}'}  # 50 unique patterns, repeated
            for i in range(iterations)
        ]
        
        # Baseline: No caching
        baseline_times = []
        for pattern in test_patterns:
            start = time.time()
            
            # Simulate expensive pattern matching computation
            result = self.simulate_cognitive_operation(complexity=20, use_optimizations=False)
            
            baseline_times.append(time.time() - start)
        
        # Optimized: With caching
        optimized_times = []
        for pattern in test_patterns:
            start = time.time()
            
            # Check cache first
            cached_result = self.pattern_cache.get_pattern_match(
                pattern['pattern_id'], pattern['content_hash']
            )
            
            if cached_result is None:
                # Compute and cache
                result = self.simulate_cognitive_operation(complexity=20, use_optimizations=True)
                self.pattern_cache.cache_pattern_match(
                    pattern['pattern_id'], pattern['content_hash'], result
                )
            else:
                result = cached_result
            
            optimized_times.append(time.time() - start)
        
        baseline_avg = statistics.mean(baseline_times) * 1000
        optimized_avg = statistics.mean(optimized_times) * 1000
        improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
        
        # Get cache statistics
        cache_stats = self.pattern_cache.get_cache_efficiency_report()
        
        return {
            'baseline_avg_ms': baseline_avg,
            'optimized_avg_ms': optimized_avg,
            'improvement_percent': improvement,
            'cache_hit_rate': cache_stats['overall_hit_rate'],
            'speedup_factor': baseline_avg / optimized_avg if optimized_avg > 0 else 0
        }
    
    def benchmark_parallel_processing(self, num_tasks: int = 100) -> Dict[str, float]:
        """Benchmark parallel processing improvements."""
        if not PERFORMANCE_AVAILABLE:
            return {'error': 'Performance optimizations not available'}
        
        logger.info(f"Benchmarking parallel processing ({num_tasks} tasks)")
        
        def task_function(task_id: int) -> Dict[str, Any]:
            return self.simulate_cognitive_operation(complexity=50, use_optimizations=False)
        
        # Baseline: Sequential execution
        start = time.time()
        sequential_results = []
        for i in range(num_tasks):
            result = task_function(i)
            sequential_results.append(result)
        baseline_time = time.time() - start
        
        # Optimized: Parallel execution
        start = time.time()
        operations = [
            {'function': task_function, 'args': (i,)}
            for i in range(num_tasks)
        ]
        parallel_results = self.parallel_processor.batch_execute_independent_operations(operations)
        optimized_time = time.time() - start
        
        improvement = ((baseline_time - optimized_time) / baseline_time) * 100
        
        return {
            'baseline_time_s': baseline_time,
            'optimized_time_s': optimized_time,
            'improvement_percent': improvement,
            'speedup_factor': baseline_time / optimized_time if optimized_time > 0 else 0,
            'successful_tasks': sum(1 for r in parallel_results if r.success)
        }
    
    def benchmark_jit_compilation(self, iterations: int = 1000) -> Dict[str, float]:
        """Benchmark JIT compilation improvements."""
        if not PERFORMANCE_AVAILABLE:
            return {'error': 'Performance optimizations not available'}
        
        logger.info(f"Benchmarking JIT compilation ({iterations} iterations)")
        
        def math_intensive_function(x: float) -> float:
            result = x
            for _ in range(100):
                result = result * 1.01 + 0.001
                result = result ** 0.99
            return result
        
        # Create JIT profiled version
        @jit_profile('math_intensive_jit')
        def jit_math_function(x: float) -> float:
            return math_intensive_function(x)
        
        # Baseline: Regular function
        baseline_times = []
        for i in range(iterations):
            start = time.time()
            result = math_intensive_function(float(i))
            baseline_times.append(time.time() - start)
        
        # Optimized: JIT compiled function (warm up first)
        # Warm-up JIT compilation
        for i in range(50):
            jit_math_function(float(i))
        
        optimized_times = []
        for i in range(iterations):
            start = time.time()
            result = jit_math_function(float(i))
            optimized_times.append(time.time() - start)
        
        baseline_avg = statistics.mean(baseline_times) * 1000
        optimized_avg = statistics.mean(optimized_times) * 1000
        improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
        
        # Get JIT compilation report
        jit_report = self.jit_compiler.get_compilation_report()
        
        return {
            'baseline_avg_ms': baseline_avg,
            'optimized_avg_ms': optimized_avg,
            'improvement_percent': improvement,
            'speedup_factor': baseline_avg / optimized_avg if optimized_avg > 0 else 0,
            'functions_compiled': jit_report['statistics']['functions_jit_compiled']
        }
    
    async def benchmark_cognitive_cycle(self, iterations: int = 50) -> Dict[str, float]:
        """Benchmark complete cognitive cycle performance."""
        if not PERFORMANCE_AVAILABLE:
            return {'error': 'Performance optimizations not available'}
        
        logger.info(f"Benchmarking cognitive cycle ({iterations} iterations)")
        
        # Start optimization system
        await self.cognitive_optimizer.start_optimization()
        
        try:
            # Simulate cognitive cycles
            cycle_times = []
            
            for i in range(iterations):
                start = time.time()
                
                # Simulate a complete cognitive cycle
                input_data = {
                    'sensory_input': f'input_{i}',
                    'cognitive_content': {
                        'complexity': i % 10 + 1,
                        'novelty': (i * 0.1) % 1.0,
                        'urgency': (i * 0.05) % 1.0
                    }
                }
                
                # Simulate tensor processing
                result = self.simulate_cognitive_operation(
                    complexity=input_data['cognitive_content']['complexity'] * 10,
                    use_optimizations=True
                )
                
                cycle_time = time.time() - start
                cycle_times.append(cycle_time)
                
                # Brief pause between cycles
                await asyncio.sleep(0.001)
            
            # Force optimization to see improvements
            optimization_results = self.cognitive_optimizer.force_full_optimization()
            
            avg_cycle_time_ms = statistics.mean(cycle_times) * 1000
            
            # Get comprehensive optimization summary
            optimization_summary = self.cognitive_optimizer.get_optimization_summary()
            
            return {
                'avg_cycle_time_ms': avg_cycle_time_ms,
                'total_cycles': iterations,
                'optimization_summary': optimization_summary,
                'meets_realtime_target': avg_cycle_time_ms < 10.0,  # 10ms target
                'memory_reuse_ratio': optimization_summary['memory_stats']['total_reuse_ratio'],
                'cache_hit_rate': optimization_summary['cache_stats']['overall_hit_rate']
            }
            
        finally:
            await self.cognitive_optimizer.stop_optimization()
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite."""
        if not PERFORMANCE_AVAILABLE:
            logger.error("Performance optimizations not available")
            return {'error': 'Performance optimizations not available'}
        
        logger.info("Starting comprehensive performance benchmark...")
        
        benchmark_results = {}
        
        # Individual component benchmarks
        benchmark_results['memory_operations'] = self.benchmark_memory_operations()
        benchmark_results['pattern_caching'] = self.benchmark_pattern_caching()
        benchmark_results['parallel_processing'] = self.benchmark_parallel_processing()
        benchmark_results['jit_compilation'] = self.benchmark_jit_compilation()
        
        # Integrated cognitive cycle benchmark
        benchmark_results['cognitive_cycle'] = await self.benchmark_cognitive_cycle()
        
        # Calculate overall performance improvement
        improvements = []
        for component, results in benchmark_results.items():
            if isinstance(results, dict) and 'improvement_percent' in results:
                improvements.append(results['improvement_percent'])
        
        if improvements:
            overall_improvement = statistics.mean(improvements)
            benchmark_results['overall_improvement_percent'] = overall_improvement
        
        # Final system statistics
        benchmark_results['final_system_stats'] = {
            'profiler_summary': self.profiler.get_performance_summary(),
            'memory_stats': self.memory_pool.get_memory_statistics(),
            'cache_stats': self.pattern_cache.get_cache_efficiency_report(),
            'parallel_stats': self.parallel_processor.get_comprehensive_statistics(),
            'jit_stats': self.jit_compiler.get_compilation_report()
        }
        
        logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    def print_benchmark_report(self, results: Dict[str, Any]) -> None:
        """Print formatted benchmark report."""
        print("\n" + "="*80)
        print("CogPrime Performance Optimization Benchmark Report")
        print("="*80)
        
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            return
        
        # Overall improvement
        if 'overall_improvement_percent' in results:
            print(f"\nOVERALL PERFORMANCE IMPROVEMENT: {results['overall_improvement_percent']:.1f}%")
        
        print("\nCOMPONENT PERFORMANCE RESULTS:")
        print("-" * 40)
        
        # Memory operations
        if 'memory_operations' in results:
            mem_results = results['memory_operations']
            print(f"Memory Pool Operations:")
            print(f"  Baseline: {mem_results['baseline_avg_ms']:.3f}ms")
            print(f"  Optimized: {mem_results['optimized_avg_ms']:.3f}ms")
            print(f"  Improvement: {mem_results['improvement_percent']:.1f}%")
            print(f"  Speedup: {mem_results['speedup_factor']:.2f}x")
        
        # Pattern caching
        if 'pattern_caching' in results:
            cache_results = results['pattern_caching']
            print(f"\nPattern Caching:")
            print(f"  Baseline: {cache_results['baseline_avg_ms']:.3f}ms")
            print(f"  Optimized: {cache_results['optimized_avg_ms']:.3f}ms")
            print(f"  Improvement: {cache_results['improvement_percent']:.1f}%")
            print(f"  Cache Hit Rate: {cache_results['cache_hit_rate']:.1%}")
            print(f"  Speedup: {cache_results['speedup_factor']:.2f}x")
        
        # Parallel processing
        if 'parallel_processing' in results:
            parallel_results = results['parallel_processing']
            print(f"\nParallel Processing:")
            print(f"  Sequential: {parallel_results['baseline_time_s']:.3f}s")
            print(f"  Parallel: {parallel_results['optimized_time_s']:.3f}s")
            print(f"  Improvement: {parallel_results['improvement_percent']:.1f}%")
            print(f"  Speedup: {parallel_results['speedup_factor']:.2f}x")
            print(f"  Successful Tasks: {parallel_results['successful_tasks']}")
        
        # JIT compilation
        if 'jit_compilation' in results:
            jit_results = results['jit_compilation']
            print(f"\nJIT Compilation:")
            print(f"  Baseline: {jit_results['baseline_avg_ms']:.3f}ms")
            print(f"  JIT Optimized: {jit_results['optimized_avg_ms']:.3f}ms")
            print(f"  Improvement: {jit_results['improvement_percent']:.1f}%")
            print(f"  Speedup: {jit_results['speedup_factor']:.2f}x")
        
        # Cognitive cycle
        if 'cognitive_cycle' in results:
            cycle_results = results['cognitive_cycle']
            print(f"\nCognitive Cycle Performance:")
            print(f"  Average Cycle Time: {cycle_results['avg_cycle_time_ms']:.2f}ms")
            print(f"  Real-time Target (<10ms): {'✓' if cycle_results['meets_realtime_target'] else '✗'}")
            print(f"  Memory Reuse Ratio: {cycle_results['memory_reuse_ratio']:.1%}")
            print(f"  Cache Hit Rate: {cycle_results['cache_hit_rate']:.1%}")
        
        print("\n" + "="*80)
        print("Benchmark completed successfully!")
        print("="*80)


async def main():
    """Run the performance benchmark."""
    benchmark = PerformanceBenchmark()
    
    if PERFORMANCE_AVAILABLE:
        results = await benchmark.run_comprehensive_benchmark()
        benchmark.print_benchmark_report(results)
        
        # Save results to file
        import json
        with open('benchmark_results.json', 'w') as f:
            # Convert numpy arrays and other non-serializable objects
            def serialize_results(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: serialize_results(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_results(item) for item in obj]
                else:
                    return obj
            
            json.dump(serialize_results(results), f, indent=2)
        
        print("\nDetailed results saved to 'benchmark_results.json'")
    else:
        print("Performance optimizations not available - cannot run benchmark")


if __name__ == "__main__":
    asyncio.run(main())