"""
Memory pool management for efficient tensor allocation and reuse.

Provides pre-allocated memory pools to reduce garbage collection overhead
and improve performance for frequently allocated tensor operations.
"""

import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import time
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class TensorPool:
    """Thread-safe tensor memory pool for efficient allocation and reuse."""
    
    def __init__(self, max_tensors_per_size: int = 100):
        self.max_tensors_per_size = max_tensors_per_size
        self._pools: Dict[Tuple[int, ...], deque] = defaultdict(lambda: deque(maxlen=max_tensors_per_size))
        self._lock = threading.RLock()
        self._allocated_count = 0
        self._reused_count = 0
        
    def get_tensor(self, shape: Tuple[int, ...], dtype=None, device=None) -> Any:
        """Get a tensor from the pool or create a new one."""
        if not TORCH_AVAILABLE:
            # Fallback to basic allocation without torch
            import numpy as np
            return np.zeros(shape, dtype=dtype or np.float32)
            
        with self._lock:
            pool_key = shape
            pool = self._pools[pool_key]
            
            if pool:
                tensor = pool.popleft()
                tensor.zero_()  # Clear existing data
                self._reused_count += 1
                return tensor
            else:
                # Create new tensor
                tensor = torch.zeros(shape, dtype=dtype or torch.float32, device=device)
                self._allocated_count += 1
                return tensor
    
    def return_tensor(self, tensor: Any) -> None:
        """Return a tensor to the pool for reuse."""
        if not TORCH_AVAILABLE or tensor is None:
            return
            
        with self._lock:
            shape = tuple(tensor.shape)
            pool = self._pools[shape]
            
            if len(pool) < self.max_tensors_per_size:
                # Detach from computation graph and move to CPU for storage
                if hasattr(tensor, 'detach'):
                    tensor = tensor.detach().cpu()
                pool.append(tensor)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool usage statistics."""
        with self._lock:
            total_pooled = sum(len(pool) for pool in self._pools.values())
            pool_sizes = {str(shape): len(pool) for shape, pool in self._pools.items()}
            
            return {
                'total_pooled_tensors': total_pooled,
                'allocated_count': self._allocated_count,
                'reused_count': self._reused_count,
                'reuse_ratio': self._reused_count / max(1, self._allocated_count + self._reused_count),
                'pool_sizes': pool_sizes
            }
    
    def clear(self) -> None:
        """Clear all pools."""
        with self._lock:
            self._pools.clear()
            self._allocated_count = 0
            self._reused_count = 0


class MemoryPool:
    """Comprehensive memory management for cognitive operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.max_tensors_per_size = config.get('max_tensors_per_size', 100)
        self.max_memory_mb = config.get('max_memory_mb', 1024)  # 1GB default
        
        # Specialized pools for different use cases
        self.tensor_pool = TensorPool(self.max_tensors_per_size)
        self.pattern_cache_pool = TensorPool(50)  # Smaller pool for pattern data
        self.attention_pool = TensorPool(30)      # Pool for attention weights
        
        # Memory usage tracking
        self._start_time = time.time()
        self._memory_stats = {
            'peak_usage_mb': 0,
            'current_usage_mb': 0,
            'allocation_events': 0
        }
        
        logger.info(f"MemoryPool initialized with max {self.max_memory_mb}MB")
    
    def get_working_tensor(self, shape: Tuple[int, ...], dtype=None, device=None) -> Any:
        """Get a tensor for general cognitive computations."""
        return self.tensor_pool.get_tensor(shape, dtype, device)
    
    def get_pattern_tensor(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Get a tensor optimized for pattern storage."""
        return self.pattern_cache_pool.get_tensor(shape, dtype)
    
    def get_attention_tensor(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Get a tensor optimized for attention computations."""
        return self.attention_pool.get_attention_tensor(shape, dtype)
    
    def return_tensor(self, tensor: Any, pool_type: str = 'working') -> None:
        """Return a tensor to appropriate pool."""
        if pool_type == 'pattern':
            self.pattern_cache_pool.return_tensor(tensor)
        elif pool_type == 'attention':
            self.attention_pool.return_tensor(tensor)
        else:
            self.tensor_pool.return_tensor(tensor)
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up unused tensors."""
        stats_before = self.get_memory_statistics()
        
        # Clear least used pools if memory usage is high
        if stats_before['estimated_usage_mb'] > self.max_memory_mb * 0.8:
            # Clear pattern cache first (least critical for immediate operations)
            self.pattern_cache_pool.clear()
            
            # If still high, clear attention pool
            if stats_before['estimated_usage_mb'] > self.max_memory_mb * 0.9:
                self.attention_pool.clear()
        
        # Force garbage collection if available
        try:
            import gc
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")
        except ImportError:
            pass
        
        stats_after = self.get_memory_statistics()
        
        return {
            'before': stats_before,
            'after': stats_after,
            'memory_freed_mb': stats_before['estimated_usage_mb'] - stats_after['estimated_usage_mb']
        }
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        tensor_stats = self.tensor_pool.get_statistics()
        pattern_stats = self.pattern_cache_pool.get_statistics()
        attention_stats = self.attention_pool.get_statistics()
        
        # Estimate memory usage (rough calculation)
        total_tensors = (tensor_stats['total_pooled_tensors'] + 
                        pattern_stats['total_pooled_tensors'] +
                        attention_stats['total_pooled_tensors'])
        
        # Assume average tensor size of 4KB for estimation
        estimated_usage_mb = (total_tensors * 4) / 1024
        
        return {
            'uptime_seconds': time.time() - self._start_time,
            'estimated_usage_mb': estimated_usage_mb,
            'max_usage_mb': self.max_memory_mb,
            'tensor_pool': tensor_stats,
            'pattern_pool': pattern_stats,
            'attention_pool': attention_stats,
            'total_reuse_ratio': (
                tensor_stats['reused_count'] + 
                pattern_stats['reused_count'] + 
                attention_stats['reused_count']
            ) / max(1, 
                tensor_stats['allocated_count'] + tensor_stats['reused_count'] +
                pattern_stats['allocated_count'] + pattern_stats['reused_count'] +
                attention_stats['allocated_count'] + attention_stats['reused_count']
            )
        }


# Global memory pool instance
_global_memory_pool: Optional[MemoryPool] = None
_pool_lock = threading.Lock()


def get_global_memory_pool() -> MemoryPool:
    """Get the global memory pool instance (thread-safe singleton)."""
    global _global_memory_pool
    
    if _global_memory_pool is None:
        with _pool_lock:
            if _global_memory_pool is None:
                _global_memory_pool = MemoryPool()
    
    return _global_memory_pool


def configure_global_memory_pool(config: Dict[str, Any]) -> None:
    """Configure the global memory pool."""
    global _global_memory_pool
    
    with _pool_lock:
        _global_memory_pool = MemoryPool(config)
        logger.info("Global memory pool reconfigured")