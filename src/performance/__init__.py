"""
Performance optimization utilities for the CogPrime cognitive architecture.

This module provides:
- Memory pool management for efficient tensor allocation
- Caching mechanisms for frequently accessed patterns
- Performance profiling and monitoring tools
- Just-in-time compilation utilities
- Parallel processing for cognitive operations
- Optimized tensor operations and mathematical functions
"""

from .memory_pool import MemoryPool, get_global_memory_pool, configure_global_memory_pool
from .pattern_cache import PatternCache, get_global_pattern_cache, configure_global_pattern_cache
from .profiler import PerformanceProfiler, get_global_profiler, profile, profile_context
from .jit_optimizer import AdaptiveJITCompiler, get_global_jit_compiler, jit_profile, force_jit_compile
from .parallel_processor import (
    ParallelCognitiveProcessor, 
    get_global_parallel_processor, 
    configure_global_parallel_processor
)
from .optimized_operations import (
    OptimizedTensorOperations,
    get_global_tensor_operations,
    optimized_attention,
    optimized_similarity,
    optimized_consolidation
)
from .cognitive_optimizer import (
    CognitiveOptimizer,
    OptimizationConfig,
    get_global_cognitive_optimizer,
    configure_global_cognitive_optimizer,
    start_global_optimization,
    stop_global_optimization
)

__all__ = [
    # Memory management
    'MemoryPool',
    'get_global_memory_pool',
    'configure_global_memory_pool',
    
    # Pattern caching
    'PatternCache',
    'get_global_pattern_cache',
    'configure_global_pattern_cache',
    
    # Performance profiling
    'PerformanceProfiler',
    'get_global_profiler',
    'profile',
    'profile_context',
    
    # JIT compilation
    'AdaptiveJITCompiler',
    'get_global_jit_compiler',
    'jit_profile',
    'force_jit_compile',
    
    # Parallel processing
    'ParallelCognitiveProcessor',
    'get_global_parallel_processor',
    'configure_global_parallel_processor',
    
    # Optimized operations
    'OptimizedTensorOperations',
    'get_global_tensor_operations',
    'optimized_attention',
    'optimized_similarity',
    'optimized_consolidation',
    
    # Comprehensive optimization
    'CognitiveOptimizer',
    'OptimizationConfig',
    'get_global_cognitive_optimizer',
    'configure_global_cognitive_optimizer',
    'start_global_optimization',
    'stop_global_optimization'
]