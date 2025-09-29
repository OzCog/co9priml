"""
Pattern caching system for cognitive operations.

Provides efficient caching of frequently accessed patterns, attention weights,
and cognitive atom relationships to reduce repeated computations.
"""

import threading
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached pattern entry."""
    key: str
    data: Any
    access_count: int
    creation_time: float
    last_access_time: float
    size_bytes: int
    metadata: Dict[str, Any]
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access_time = time.time()
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.creation_time
    
    @property
    def idle_time_seconds(self) -> float:
        """Get idle time since last access."""
        return time.time() - self.last_access_time


class LRUCache:
    """Thread-safe LRU cache with size limits and TTL support."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 100,
                 default_ttl_seconds: int = 3600):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size_bytes': 0
        }
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of cached data."""
        if TORCH_AVAILABLE and hasattr(data, 'numel') and hasattr(data, 'element_size'):
            # PyTorch tensor
            return data.numel() * data.element_size()
        elif hasattr(data, '__sizeof__'):
            return data.__sizeof__()
        else:
            # Rough estimate for other types
            return len(str(data)) * 2  # Assume 2 bytes per character
    
    def _make_key(self, key: Union[str, Tuple, List]) -> str:
        """Convert various key types to string hash."""
        if isinstance(key, str):
            return key
        else:
            # Hash composite keys
            key_str = str(key)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        to_remove = []
        
        for key, entry in self._cache.items():
            if entry.age_seconds > self.default_ttl_seconds:
                to_remove.append(key)
        
        for key in to_remove:
            self._remove_entry(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries to stay within limits."""
        while (len(self._cache) > self.max_size or 
               self._stats['current_size_bytes'] > self.max_memory_bytes):
            if not self._cache:
                break
            # Remove oldest (least recently used)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._stats['evictions'] += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry and update stats."""
        if key in self._cache:
            entry = self._cache[key]
            self._stats['current_size_bytes'] -= entry.size_bytes
            del self._cache[key]
    
    def get(self, key: Union[str, Tuple, List], default: Any = None) -> Any:
        """Get value from cache."""
        cache_key = self._make_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check if expired
                if entry.age_seconds > self.default_ttl_seconds:
                    self._remove_entry(cache_key)
                    self._stats['misses'] += 1
                    return default
                
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                entry.update_access()
                self._stats['hits'] += 1
                
                return entry.data
            else:
                self._stats['misses'] += 1
                return default
    
    def put(self, key: Union[str, Tuple, List], data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Put value into cache."""
        cache_key = self._make_key(key)
        metadata = metadata or {}
        
        with self._lock:
            # Remove existing entry if present
            if cache_key in self._cache:
                self._remove_entry(cache_key)
            
            # Create new entry
            size_bytes = self._estimate_size(data)
            entry = CacheEntry(
                key=cache_key,
                data=data,
                access_count=1,
                creation_time=time.time(),
                last_access_time=time.time(),
                size_bytes=size_bytes,
                metadata=metadata
            )
            
            self._cache[cache_key] = entry
            self._stats['current_size_bytes'] += size_bytes
            
            # Evict if necessary
            self._evict_expired()
            self._evict_lru()
    
    def invalidate(self, key: Union[str, Tuple, List]) -> bool:
        """Remove specific key from cache."""
        cache_key = self._make_key(key)
        
        with self._lock:
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
            self._stats['current_size_bytes'] = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / max(1, total_accesses)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._stats['current_size_bytes'] / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions']
            }


class PatternCache:
    """Specialized cache for cognitive patterns and frequent computations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Specialized caches for different types of data
        self.pattern_cache = LRUCache(
            max_size=config.get('pattern_max_size', 500),
            max_memory_mb=config.get('pattern_max_memory_mb', 50),
            default_ttl_seconds=config.get('pattern_ttl_seconds', 1800)  # 30 minutes
        )
        
        self.attention_cache = LRUCache(
            max_size=config.get('attention_max_size', 200),
            max_memory_mb=config.get('attention_max_memory_mb', 20),
            default_ttl_seconds=config.get('attention_ttl_seconds', 600)  # 10 minutes
        )
        
        self.inference_cache = LRUCache(
            max_size=config.get('inference_max_size', 300),
            max_memory_mb=config.get('inference_max_memory_mb', 30),
            default_ttl_seconds=config.get('inference_ttl_seconds', 900)  # 15 minutes
        )
        
        self.memory_embedding_cache = LRUCache(
            max_size=config.get('memory_max_size', 1000),
            max_memory_mb=config.get('memory_max_memory_mb', 100),
            default_ttl_seconds=config.get('memory_ttl_seconds', 3600)  # 1 hour
        )
        
        # Access pattern tracking
        self._access_patterns = defaultdict(list)
        self._lock = threading.RLock()
        
        logger.info("PatternCache initialized with specialized caches")
    
    def cache_pattern_match(self, pattern_id: str, content_hash: str, result: Any) -> None:
        """Cache pattern matching result."""
        key = f"pattern_match_{pattern_id}_{content_hash}"
        metadata = {'type': 'pattern_match', 'pattern_id': pattern_id}
        self.pattern_cache.put(key, result, metadata)
    
    def get_pattern_match(self, pattern_id: str, content_hash: str) -> Any:
        """Get cached pattern matching result."""
        key = f"pattern_match_{pattern_id}_{content_hash}"
        return self.pattern_cache.get(key)
    
    def cache_attention_weights(self, context_hash: str, weights: Any) -> None:
        """Cache computed attention weights."""
        key = f"attention_{context_hash}"
        metadata = {'type': 'attention_weights'}
        self.attention_cache.put(key, weights, metadata)
    
    def get_attention_weights(self, context_hash: str) -> Any:
        """Get cached attention weights."""
        key = f"attention_{context_hash}"
        result = self.attention_cache.get(key)
        
        # Track access pattern for adaptive caching
        if result is not None:
            with self._lock:
                self._access_patterns['attention'].append(time.time())
        
        return result
    
    def cache_inference_result(self, premises_hash: str, conclusion: Any) -> None:
        """Cache PLN inference result."""
        key = f"inference_{premises_hash}"
        metadata = {'type': 'inference_result'}
        self.inference_cache.put(key, conclusion, metadata)
    
    def get_inference_result(self, premises_hash: str) -> Any:
        """Get cached inference result."""
        key = f"inference_{premises_hash}"
        return self.inference_cache.get(key)
    
    def cache_memory_embedding(self, content_hash: str, embedding: Any) -> None:
        """Cache memory content embedding."""
        key = f"embedding_{content_hash}"
        metadata = {'type': 'memory_embedding'}
        self.memory_embedding_cache.put(key, embedding, metadata)
    
    def get_memory_embedding(self, content_hash: str) -> Any:
        """Get cached memory embedding."""
        key = f"embedding_{content_hash}"
        return self.memory_embedding_cache.get(key)
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Get comprehensive cache efficiency report."""
        pattern_stats = self.pattern_cache.get_statistics()
        attention_stats = self.attention_cache.get_statistics()
        inference_stats = self.inference_cache.get_statistics()
        memory_stats = self.memory_embedding_cache.get_statistics()
        
        # Calculate recent access patterns
        recent_accesses = {}
        current_time = time.time()
        
        with self._lock:
            for cache_type, timestamps in self._access_patterns.items():
                # Count accesses in last 5 minutes
                recent = [t for t in timestamps if current_time - t < 300]
                recent_accesses[cache_type] = len(recent)
        
        overall_hit_rate = (
            pattern_stats['hits'] + attention_stats['hits'] + 
            inference_stats['hits'] + memory_stats['hits']
        ) / max(1, 
            pattern_stats['hits'] + pattern_stats['misses'] +
            attention_stats['hits'] + attention_stats['misses'] +
            inference_stats['hits'] + inference_stats['misses'] +
            memory_stats['hits'] + memory_stats['misses']
        )
        
        return {
            'overall_hit_rate': overall_hit_rate,
            'recent_access_patterns': recent_accesses,
            'caches': {
                'patterns': pattern_stats,
                'attention': attention_stats,
                'inference': inference_stats,
                'memory': memory_stats
            },
            'total_memory_usage_mb': (
                pattern_stats['memory_usage_mb'] + 
                attention_stats['memory_usage_mb'] +
                inference_stats['memory_usage_mb'] + 
                memory_stats['memory_usage_mb']
            )
        }
    
    def optimize_cache_sizes(self) -> Dict[str, Any]:
        """Dynamically optimize cache sizes based on usage patterns."""
        stats = self.get_cache_efficiency_report()
        
        optimizations = []
        
        # Increase cache size for high hit rate caches with frequent evictions
        for cache_name, cache_stats in stats['caches'].items():
            if cache_stats['hit_rate'] > 0.8 and cache_stats['evictions'] > 100:
                optimizations.append(f"Consider increasing {cache_name} cache size")
        
        # Suggest clearing underused caches
        for cache_name, cache_stats in stats['caches'].items():
            if cache_stats['hit_rate'] < 0.3:
                optimizations.append(f"Consider reducing {cache_name} cache size or TTL")
        
        return {
            'current_stats': stats,
            'optimization_suggestions': optimizations
        }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.pattern_cache.clear()
        self.attention_cache.clear()
        self.inference_cache.clear()
        self.memory_embedding_cache.clear()
        
        with self._lock:
            self._access_patterns.clear()
        
        logger.info("All pattern caches cleared")


# Global pattern cache instance
_global_pattern_cache: Optional[PatternCache] = None
_cache_lock = threading.Lock()


def get_global_pattern_cache() -> PatternCache:
    """Get the global pattern cache instance (thread-safe singleton)."""
    global _global_pattern_cache
    
    if _global_pattern_cache is None:
        with _cache_lock:
            if _global_pattern_cache is None:
                _global_pattern_cache = PatternCache()
    
    return _global_pattern_cache


def configure_global_pattern_cache(config: Dict[str, Any]) -> None:
    """Configure the global pattern cache."""
    global _global_pattern_cache
    
    with _cache_lock:
        _global_pattern_cache = PatternCache(config)
        logger.info("Global pattern cache reconfigured")