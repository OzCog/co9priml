"""
Just-In-Time (JIT) compilation optimizer for performance-critical cognitive operations.

Provides JIT compilation capabilities for hot code paths and frequently executed
cognitive functions to improve runtime performance.
"""

import functools
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
import logging
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)

# Try to import numba for JIT compilation
try:
    import numba
    from numba import jit, njit, types
    NUMBA_AVAILABLE = True
    logger.info("Numba JIT compiler available")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - JIT optimization disabled")
    
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class JITProfile:
    """Profile for tracking JIT compilation candidates."""
    
    def __init__(self, function_name: str):
        self.function_name = function_name
        self.call_count = 0
        self.total_execution_time = 0.0
        self.avg_execution_time = 0.0
        self.recent_times = deque(maxlen=100)
        self.is_jit_compiled = False
        self.jit_compilation_time = 0.0
        self.performance_improvement = 0.0
        self.last_called = time.time()
        
    def record_call(self, execution_time: float):
        """Record a function call for profiling."""
        self.call_count += 1
        self.total_execution_time += execution_time
        self.avg_execution_time = self.total_execution_time / self.call_count
        self.recent_times.append(execution_time)
        self.last_called = time.time()
    
    def get_hotness_score(self) -> float:
        """Calculate how 'hot' this function is (good JIT candidate)."""
        if self.call_count < 5:
            return 0.0
        
        # Score based on frequency and execution time
        frequency_score = min(self.call_count / 100, 1.0)  # Normalize to 0-1
        time_score = min(self.avg_execution_time * 1000, 1.0)  # ms to 0-1 range
        recency_score = max(0, 1.0 - (time.time() - self.last_called) / 3600)  # Decay over 1 hour
        
        return frequency_score * 0.4 + time_score * 0.4 + recency_score * 0.2


class AdaptiveJITCompiler:
    """Adaptive JIT compiler that automatically identifies and optimizes hot functions."""
    
    def __init__(self, 
                 hot_threshold: float = 0.5,
                 min_calls_before_jit: int = 10,
                 compilation_cache_size: int = 100):
        
        self.hot_threshold = hot_threshold
        self.min_calls_before_jit = min_calls_before_jit
        self.compilation_cache_size = compilation_cache_size
        
        # Function profiling
        self.function_profiles: Dict[str, JITProfile] = {}
        self.compiled_functions: Dict[str, Callable] = {}
        
        # JIT compilation cache
        self._compilation_cache = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'functions_profiled': 0,
            'functions_jit_compiled': 0,
            'total_compilation_time': 0.0,
            'average_performance_improvement': 0.0
        }
        
        logger.info(f"AdaptiveJITCompiler initialized (numba_available={NUMBA_AVAILABLE})")
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator to profile functions for JIT compilation."""
        def decorator(func: Callable) -> Callable:
            function_name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Check if we have a JIT compiled version
                with self._lock:
                    if function_name in self.compiled_functions:
                        try:
                            result = self.compiled_functions[function_name](*args, **kwargs)
                            execution_time = time.time() - start_time
                            
                            # Record performance
                            profile = self.function_profiles[function_name]
                            profile.record_call(execution_time)
                            
                            return result
                        except Exception as e:
                            logger.warning(f"JIT compiled function {function_name} failed, falling back: {e}")
                            # Fall through to original function
                
                # Execute original function
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Profile the execution
                with self._lock:
                    if function_name not in self.function_profiles:
                        self.function_profiles[function_name] = JITProfile(function_name)
                        self.stats['functions_profiled'] += 1
                    
                    profile = self.function_profiles[function_name]
                    profile.record_call(execution_time)
                    
                    # Check if this function should be JIT compiled
                    if (not profile.is_jit_compiled and 
                        profile.call_count >= self.min_calls_before_jit and
                        profile.get_hotness_score() >= self.hot_threshold):
                        
                        self._compile_function(function_name, func)
                
                return result
            
            return wrapper
        return decorator
    
    def _compile_function(self, function_name: str, func: Callable) -> None:
        """Compile a function with JIT if possible."""
        if not NUMBA_AVAILABLE:
            return
        
        try:
            compilation_start = time.time()
            
            # Try to compile with numba
            # Use nopython mode for better performance
            compiled_func = njit(cache=True, fastmath=True)(func)
            
            compilation_time = time.time() - compilation_start
            
            # Store compiled function
            self.compiled_functions[function_name] = compiled_func
            
            # Update profile
            profile = self.function_profiles[function_name]
            profile.is_jit_compiled = True
            profile.jit_compilation_time = compilation_time
            
            # Update statistics
            self.stats['functions_jit_compiled'] += 1
            self.stats['total_compilation_time'] += compilation_time
            
            logger.info(f"Successfully JIT compiled {function_name} in {compilation_time:.3f}s")
            
        except Exception as e:
            logger.warning(f"Failed to JIT compile {function_name}: {e}")
    
    def force_compile_function(self, func: Callable, function_name: Optional[str] = None) -> Optional[Callable]:
        """Force compile a specific function."""
        if not NUMBA_AVAILABLE:
            return func
        
        function_name = function_name or f"{func.__module__}.{func.__name__}"
        
        try:
            compiled_func = njit(cache=True, fastmath=True)(func)
            
            with self._lock:
                self.compiled_functions[function_name] = compiled_func
                
                if function_name not in self.function_profiles:
                    self.function_profiles[function_name] = JITProfile(function_name)
                
                self.function_profiles[function_name].is_jit_compiled = True
            
            logger.info(f"Force compiled {function_name}")
            return compiled_func
            
        except Exception as e:
            logger.error(f"Failed to force compile {function_name}: {e}")
            return func
    
    def get_compilation_report(self) -> Dict[str, Any]:
        """Get comprehensive JIT compilation report."""
        with self._lock:
            hot_functions = []
            compiled_functions = []
            
            for name, profile in self.function_profiles.items():
                func_data = {
                    'name': name,
                    'call_count': profile.call_count,
                    'avg_execution_time_ms': profile.avg_execution_time * 1000,
                    'hotness_score': profile.get_hotness_score(),
                    'is_compiled': profile.is_jit_compiled
                }
                
                if profile.is_jit_compiled:
                    func_data['compilation_time_ms'] = profile.jit_compilation_time * 1000
                    func_data['performance_improvement'] = profile.performance_improvement
                    compiled_functions.append(func_data)
                elif profile.get_hotness_score() >= self.hot_threshold:
                    hot_functions.append(func_data)
            
            # Sort by hotness/performance
            hot_functions.sort(key=lambda x: x['hotness_score'], reverse=True)
            compiled_functions.sort(key=lambda x: x['performance_improvement'], reverse=True)
            
            return {
                'statistics': self.stats.copy(),
                'numba_available': NUMBA_AVAILABLE,
                'hot_functions_candidates': hot_functions[:10],
                'compiled_functions': compiled_functions,
                'total_functions_tracked': len(self.function_profiles)
            }
    
    def optimize_mathematical_operations(self) -> Dict[str, Callable]:
        """Pre-compile common mathematical operations for cognitive processing."""
        if not NUMBA_AVAILABLE:
            return {}
        
        optimized_functions = {}
        
        # Vector operations
        @njit(cache=True, fastmath=True)
        def fast_cosine_similarity(a, b):
            """Optimized cosine similarity calculation."""
            dot_product = 0.0
            norm_a = 0.0
            norm_b = 0.0
            
            for i in range(len(a)):
                dot_product += a[i] * b[i]
                norm_a += a[i] * a[i]
                norm_b += b[i] * b[i]
            
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            
            return dot_product / (norm_a ** 0.5 * norm_b ** 0.5)
        
        @njit(cache=True, fastmath=True)
        def fast_euclidean_distance(a, b):
            """Optimized Euclidean distance calculation."""
            distance = 0.0
            for i in range(len(a)):
                diff = a[i] - b[i]
                distance += diff * diff
            return distance ** 0.5
        
        @njit(cache=True, fastmath=True)
        def fast_softmax(x):
            """Optimized softmax calculation."""
            # Find max for numerical stability
            max_val = x[0]
            for i in range(1, len(x)):
                if x[i] > max_val:
                    max_val = x[i]
            
            # Compute exp and sum
            exp_sum = 0.0
            result = [0.0] * len(x)
            
            for i in range(len(x)):
                exp_val = 2.718281828459045 ** (x[i] - max_val)  # e^(x-max)
                result[i] = exp_val
                exp_sum += exp_val
            
            # Normalize
            for i in range(len(result)):
                result[i] /= exp_sum
            
            return result
        
        @njit(cache=True, fastmath=True)
        def fast_attention_weights(queries, keys):
            """Optimized attention weight calculation."""
            weights = []
            for i in range(len(queries)):
                scores = []
                for j in range(len(keys)):
                    # Dot product attention
                    score = 0.0
                    for k in range(len(queries[i])):
                        score += queries[i][k] * keys[j][k]
                    scores.append(score)
                
                # Apply softmax
                weights.append(fast_softmax(scores))
            
            return weights
        
        optimized_functions.update({
            'cosine_similarity': fast_cosine_similarity,
            'euclidean_distance': fast_euclidean_distance,
            'softmax': fast_softmax,
            'attention_weights': fast_attention_weights
        })
        
        logger.info(f"Pre-compiled {len(optimized_functions)} mathematical operations")
        return optimized_functions
    
    def benchmark_jit_performance(self, func: Callable, test_args: List[Any], iterations: int = 1000) -> Dict[str, float]:
        """Benchmark JIT vs non-JIT performance."""
        if not NUMBA_AVAILABLE:
            return {'message': 'Numba not available for benchmarking'}
        
        # Benchmark original function
        original_times = []
        for _ in range(iterations):
            start = time.time()
            func(*test_args)
            original_times.append(time.time() - start)
        
        # Compile and benchmark JIT version
        try:
            jit_func = njit(cache=True, fastmath=True)(func)
            
            # Warm up JIT (first call includes compilation)
            jit_func(*test_args)
            
            jit_times = []
            for _ in range(iterations):
                start = time.time()
                jit_func(*test_args)
                jit_times.append(time.time() - start)
            
            original_avg = sum(original_times) / len(original_times)
            jit_avg = sum(jit_times) / len(jit_times)
            
            speedup = original_avg / jit_avg if jit_avg > 0 else 0
            
            return {
                'original_avg_time_ms': original_avg * 1000,
                'jit_avg_time_ms': jit_avg * 1000,
                'speedup_factor': speedup,
                'performance_improvement_percent': (speedup - 1) * 100
            }
            
        except Exception as e:
            return {'error': f'JIT compilation failed: {e}'}


# Global JIT compiler instance
_global_jit_compiler: Optional[AdaptiveJITCompiler] = None
_jit_lock = threading.Lock()


def get_global_jit_compiler() -> AdaptiveJITCompiler:
    """Get the global JIT compiler instance (thread-safe singleton)."""
    global _global_jit_compiler
    
    if _global_jit_compiler is None:
        with _jit_lock:
            if _global_jit_compiler is None:
                _global_jit_compiler = AdaptiveJITCompiler()
    
    return _global_jit_compiler


def jit_profile(func_name: Optional[str] = None):
    """Convenient decorator for JIT profiling using global compiler."""
    return get_global_jit_compiler().profile_function(func_name)


def force_jit_compile(func: Callable, function_name: Optional[str] = None) -> Callable:
    """Force compile a function using global JIT compiler."""
    return get_global_jit_compiler().force_compile_function(func, function_name) or func