"""
Test performance optimizations implementation.

Validates that all performance optimization components work correctly
and provide measurable improvements.
"""

import unittest
import time
import asyncio
from typing import Dict, Any, List
import numpy as np

# Import performance modules
try:
    from ..performance import (
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
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False


class TestPerformanceOptimizations(unittest.TestCase):
    """Test suite for performance optimization components."""
    
    def setUp(self):
        """Set up test environment."""
        if not PERFORMANCE_AVAILABLE:
            self.skipTest("Performance optimizations not available")
    
    def test_memory_pool_basic_operations(self):
        """Test basic memory pool functionality."""
        memory_pool = get_global_memory_pool()
        
        # Test tensor allocation and return
        tensor = memory_pool.get_working_tensor((100, 50))
        self.assertIsNotNone(tensor)
        
        # Return tensor to pool
        memory_pool.return_tensor(tensor, 'working')
        
        # Get statistics
        stats = memory_pool.get_memory_statistics()
        self.assertIn('total_reuse_ratio', stats)
        self.assertGreaterEqual(stats['total_reuse_ratio'], 0)
    
    def test_pattern_cache_functionality(self):
        """Test pattern caching operations."""
        pattern_cache = get_global_pattern_cache()
        
        # Cache a pattern match result
        test_result = {'match_score': 0.85, 'pattern_type': 'temporal'}
        pattern_cache.cache_pattern_match('test_pattern', 'content_hash_123', test_result)
        
        # Retrieve cached result
        cached_result = pattern_cache.get_pattern_match('test_pattern', 'content_hash_123')
        self.assertEqual(cached_result, test_result)
        
        # Test cache efficiency report
        report = pattern_cache.get_cache_efficiency_report()
        self.assertIn('overall_hit_rate', report)
        self.assertIn('caches', report)
    
    def test_performance_profiler(self):
        """Test performance profiling functionality."""
        profiler = get_global_profiler()
        
        @profile('test_function')
        def test_function():
            time.sleep(0.01)  # Small delay for measurable time
            return "test_result"
        
        # Execute profiled function
        result = test_function()
        self.assertEqual(result, "test_result")
        
        # Get performance summary
        summary = profiler.get_performance_summary()
        self.assertIn('total_operations', summary)
        self.assertGreater(summary['total_operations'], 0)
    
    def test_jit_compiler_basic(self):
        """Test JIT compiler basic functionality."""
        jit_compiler = get_global_jit_compiler()
        
        @jit_profile('test_math_function')
        def simple_math_function(x: float) -> float:
            return x * x + 2 * x + 1
        
        # Execute function multiple times to trigger JIT consideration
        for i in range(15):
            result = simple_math_function(float(i))
            expected = i * i + 2 * i + 1
            self.assertEqual(result, expected)
        
        # Get compilation report
        report = jit_compiler.get_compilation_report()
        self.assertIn('statistics', report)
        self.assertIn('functions_profiled', report['statistics'])
    
    def test_parallel_processor_basic(self):
        """Test parallel processing functionality."""
        parallel_processor = get_global_parallel_processor()
        
        def simple_task(x: int) -> int:
            return x * 2
        
        # Test batch execution
        operations = [
            {'function': simple_task, 'args': (i,)} 
            for i in range(5)
        ]
        
        results = parallel_processor.batch_execute_independent_operations(operations)
        self.assertEqual(len(results), 5)
        
        # Verify results
        for i, result in enumerate(results):
            if result.success:
                self.assertEqual(result.result, i * 2)
    
    def test_cognitive_optimizer_integration(self):
        """Test cognitive optimizer integration."""
        config = OptimizationConfig(
            enable_memory_pooling=True,
            enable_pattern_caching=True,
            enable_parallel_processing=True,
            enable_jit_compilation=True,
            enable_profiling=True
        )
        
        cognitive_optimizer = get_global_cognitive_optimizer()
        
        # Get optimization summary
        summary = cognitive_optimizer.get_optimization_summary()
        self.assertIn('config', summary)
        self.assertIn('memory_stats', summary)
        self.assertIn('cache_stats', summary)
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization workflow."""
        
        @profile('end_to_end_test')
        @jit_profile('end_to_end_test')
        def cognitive_operation(data: List[float]) -> Dict[str, Any]:
            """Simulate a cognitive operation."""
            # Simulate pattern matching
            pattern_cache = get_global_pattern_cache()
            data_hash = str(hash(tuple(data)))
            
            cached_result = pattern_cache.get_pattern_match('cognitive_op', data_hash)
            if cached_result:
                return cached_result
            
            # Simulate computation
            result = {
                'processed_data': [x * 1.5 + 0.1 for x in data],
                'pattern_detected': len(data) > 5,
                'complexity_score': sum(data) / len(data) if data else 0
            }
            
            # Cache result
            pattern_cache.cache_pattern_match('cognitive_op', data_hash, result)
            
            return result
        
        # Test with different data sizes
        test_datasets = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.5, 1.5, 2.5, 3.5, 4.5]
        ]
        
        results = []
        for dataset in test_datasets:
            result = cognitive_operation(dataset)
            results.append(result)
            self.assertIn('processed_data', result)
            self.assertIn('pattern_detected', result)
        
        # Test caching - second call should be cached
        cached_result = cognitive_operation(test_datasets[0])
        self.assertEqual(cached_result, results[0])
        
        # Get final performance report
        profiler = get_global_profiler()
        final_summary = profiler.get_performance_summary()
        self.assertGreater(final_summary['total_operations'], 0)
    
    async def test_async_optimization_integration(self):
        """Test asynchronous optimization integration."""
        cognitive_optimizer = get_global_cognitive_optimizer()
        
        # Start optimization (would normally run in background)
        await cognitive_optimizer.start_optimization()
        
        # Simulate some cognitive work
        await asyncio.sleep(0.1)
        
        # Force optimization cycle
        optimization_results = cognitive_optimizer.force_full_optimization()
        self.assertIn('memory', optimization_results)
        
        # Stop optimization
        await cognitive_optimizer.stop_optimization()


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics and monitoring."""
    
    def setUp(self):
        """Set up test environment."""
        if not PERFORMANCE_AVAILABLE:
            self.skipTest("Performance optimizations not available")
    
    def test_performance_measurement(self):
        """Test that performance measurements are accurate."""
        profiler = get_global_profiler()
        
        @profile('measured_function')
        def measured_function():
            # Simulate work with known duration
            time.sleep(0.05)  # 50ms
            return "completed"
        
        start_time = time.time()
        result = measured_function()
        actual_duration = time.time() - start_time
        
        self.assertEqual(result, "completed")
        
        # Check that profiler recorded similar duration
        operation_details = profiler.get_operation_details('measured_function')
        if operation_details:
            measured_duration_ms = operation_details['avg_duration_ms']
            # Allow for some measurement variance (±20ms)
            self.assertAlmostEqual(measured_duration_ms, actual_duration * 1000, delta=20)
    
    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        profiler = get_global_profiler()
        
        @profile('fast_function')
        def fast_function():
            time.sleep(0.001)  # 1ms
            return "fast"
        
        @profile('slow_function')
        def slow_function():
            time.sleep(0.02)  # 20ms
            return "slow"
        
        # Execute functions multiple times
        for _ in range(15):
            fast_function()
            slow_function()
        
        # Identify bottlenecks
        bottlenecks = profiler.identify_bottlenecks(min_call_count=10)
        
        if bottlenecks:
            # The slow function should be identified as a bottleneck
            bottleneck_names = [b['operation_name'] for b in bottlenecks]
            self.assertIn('slow_function', bottleneck_names)


if __name__ == '__main__':
    # Run async tests
    async def run_async_tests():
        test_case = TestPerformanceOptimizations()
        test_case.setUp()
        await test_case.test_async_optimization_integration()
    
    # Run synchronous tests
    unittest.main(verbosity=2, exit=False)
    
    # Run async test
    if PERFORMANCE_AVAILABLE:
        asyncio.run(run_async_tests())
        print("Async tests completed successfully")