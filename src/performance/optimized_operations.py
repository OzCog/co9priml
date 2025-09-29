"""
Optimized tensor operations for cognitive processing.

Provides high-performance implementations of common cognitive operations
using vectorization, parallelization, and JIT compilation.
"""

import functools
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import numpy as np

from .jit_optimizer import get_global_jit_compiler, jit_profile
from .memory_pool import get_global_memory_pool
from .profiler import profile

logger = logging.getLogger(__name__)

# Try to import torch for tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import numba for JIT compilation
try:
    import numba
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class OptimizedTensorOperations:
    """Collection of optimized tensor operations for cognitive processing."""
    
    def __init__(self):
        self.jit_compiler = get_global_jit_compiler()
        self.memory_pool = get_global_memory_pool()
        
        # Pre-compile common operations
        self._initialize_optimized_functions()
        
        logger.info("OptimizedTensorOperations initialized")
    
    def _initialize_optimized_functions(self):
        """Pre-compile frequently used operations."""
        if NUMBA_AVAILABLE:
            self.optimized_math = self.jit_compiler.optimize_mathematical_operations()
        else:
            self.optimized_math = {}
    
    @profile('tensor_ops.attention_computation')
    def compute_attention_weights(self, queries: Any, keys: Any, values: Any, 
                                mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Optimized attention weight computation."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for attention computation")
        
        # Use optimized attention if available
        if 'attention_weights' in self.optimized_math:
            try:
                # Convert to numpy for numba processing
                q_np = queries.detach().cpu().numpy()
                k_np = keys.detach().cpu().numpy()
                
                weights = self.optimized_math['attention_weights'](q_np, k_np)
                attention_weights = torch.from_numpy(weights).to(queries.device)
                
                # Apply attention to values
                attended_values = torch.matmul(attention_weights, values)
                
                return attended_values, attention_weights
                
            except Exception as e:
                logger.warning(f"Optimized attention failed, falling back: {e}")
        
        # Fallback to standard attention
        return self._standard_attention(queries, keys, values, mask)
    
    def _standard_attention(self, queries: Any, keys: Any, values: Any, 
                          mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Standard attention computation."""
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        
        # Scale by sqrt(d_k)
        d_k = queries.size(-1)
        scores = scores / (d_k ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)
        
        return attended_values, attention_weights
    
    @profile('tensor_ops.similarity_computation')
    @jit_profile('tensor_ops.similarity_computation')
    def compute_similarity_matrix(self, tensor_a: Any, tensor_b: Any, 
                                similarity_type: str = 'cosine') -> Any:
        """Optimized similarity matrix computation."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for similarity computation")
        
        if similarity_type == 'cosine' and 'cosine_similarity' in self.optimized_math:
            try:
                # Use optimized cosine similarity
                a_np = tensor_a.detach().cpu().numpy()
                b_np = tensor_b.detach().cpu().numpy()
                
                similarity_matrix = np.zeros((len(a_np), len(b_np)))
                
                for i in range(len(a_np)):
                    for j in range(len(b_np)):
                        similarity_matrix[i, j] = self.optimized_math['cosine_similarity'](
                            a_np[i], b_np[j]
                        )
                
                return torch.from_numpy(similarity_matrix).to(tensor_a.device)
                
            except Exception as e:
                logger.warning(f"Optimized cosine similarity failed, falling back: {e}")
        
        # Fallback implementations
        if similarity_type == 'cosine':
            return self._cosine_similarity_matrix(tensor_a, tensor_b)
        elif similarity_type == 'euclidean':
            return self._euclidean_distance_matrix(tensor_a, tensor_b)
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}")
    
    def _cosine_similarity_matrix(self, tensor_a: Any, tensor_b: Any) -> Any:
        """Standard cosine similarity matrix computation."""
        # Normalize tensors
        norm_a = torch.nn.functional.normalize(tensor_a, p=2, dim=1)
        norm_b = torch.nn.functional.normalize(tensor_b, p=2, dim=1)
        
        # Compute similarity matrix
        return torch.matmul(norm_a, norm_b.transpose(0, 1))
    
    def _euclidean_distance_matrix(self, tensor_a: Any, tensor_b: Any) -> Any:
        """Standard Euclidean distance matrix computation."""
        # Expand dimensions for broadcasting
        a_expanded = tensor_a.unsqueeze(1)  # [N, 1, D]
        b_expanded = tensor_b.unsqueeze(0)  # [1, M, D]
        
        # Compute squared differences
        diff = a_expanded - b_expanded  # [N, M, D]
        squared_diff = diff ** 2  # [N, M, D]
        
        # Sum over last dimension and take square root
        distances = torch.sqrt(torch.sum(squared_diff, dim=2))  # [N, M]
        
        return distances
    
    @profile('tensor_ops.memory_consolidation')
    def consolidate_memory_tensors(self, memory_tensors: List[Any], 
                                 consolidation_method: str = 'weighted_average',
                                 weights: Optional[List[float]] = None) -> Any:
        """Optimized memory tensor consolidation."""
        if not memory_tensors:
            return None
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for memory consolidation")
        
        # Use memory pool for intermediate tensors
        if self.memory_pool:
            # Get working tensors from pool
            target_shape = memory_tensors[0].shape
            consolidated = self.memory_pool.get_working_tensor(target_shape)
            
            try:
                if consolidation_method == 'weighted_average':
                    if weights is None:
                        weights = [1.0] * len(memory_tensors)
                    
                    # Normalize weights
                    total_weight = sum(weights)
                    normalized_weights = [w / total_weight for w in weights]
                    
                    # Weighted sum
                    consolidated.zero_()
                    for tensor, weight in zip(memory_tensors, normalized_weights):
                        consolidated.add_(tensor, alpha=weight)
                
                elif consolidation_method == 'max_pooling':
                    stacked = torch.stack(memory_tensors, dim=0)
                    consolidated.copy_(torch.max(stacked, dim=0)[0])
                
                elif consolidation_method == 'attention_weighted':
                    # Use attention mechanism to weight memories
                    stacked = torch.stack(memory_tensors, dim=0)  # [N, D]
                    
                    # Compute attention weights
                    attention_scores = torch.matmul(stacked, stacked.transpose(0, 1))
                    attention_weights = torch.softmax(attention_scores.mean(dim=1), dim=0)
                    
                    # Apply attention weights
                    consolidated.copy_(torch.sum(stacked * attention_weights.unsqueeze(1), dim=0))
                
                else:
                    raise ValueError(f"Unsupported consolidation method: {consolidation_method}")
                
                result = consolidated.clone()
                self.memory_pool.return_tensor(consolidated, 'working')
                return result
                
            except Exception as e:
                self.memory_pool.return_tensor(consolidated, 'working')
                raise e
        
        else:
            # Fallback without memory pool
            if consolidation_method == 'weighted_average':
                if weights is None:
                    return torch.mean(torch.stack(memory_tensors, dim=0), dim=0)
                else:
                    weighted_sum = sum(t * w for t, w in zip(memory_tensors, weights))
                    return weighted_sum / sum(weights)
            
            elif consolidation_method == 'max_pooling':
                return torch.max(torch.stack(memory_tensors, dim=0), dim=0)[0]
            
            else:
                raise ValueError(f"Unsupported consolidation method: {consolidation_method}")
    
    @profile('tensor_ops.pattern_embedding')
    def compute_pattern_embeddings(self, patterns: List[Dict[str, Any]], 
                                 embedding_dim: int = 512) -> Any:
        """Compute embeddings for detected patterns."""
        if not patterns:
            return None
        
        if not TORCH_AVAILABLE:
            embeddings = np.random.randn(len(patterns), embedding_dim).astype(np.float32)
            return embeddings
        
        embeddings = []
        
        for pattern in patterns:
            # Create embedding based on pattern characteristics
            embedding = torch.zeros(embedding_dim)
            
            # Pattern type encoding (first 64 dimensions)
            pattern_type = pattern.get('pattern_type', 'unknown')
            type_hash = hash(pattern_type) % 64
            embedding[type_hash] = 1.0
            
            # Confidence encoding (next 32 dimensions)
            confidence = pattern.get('confidence', 0.5)
            conf_idx = int(confidence * 31) + 64
            embedding[conf_idx] = confidence
            
            # Frequency encoding (next 32 dimensions)
            frequency = pattern.get('frequency', 0)
            freq_normalized = min(frequency / 100.0, 1.0)  # Normalize frequency
            freq_idx = int(freq_normalized * 31) + 96
            embedding[freq_idx] = freq_normalized
            
            # Random features for remaining dimensions
            remaining_dims = embedding_dim - 128
            if remaining_dims > 0:
                embedding[128:] = torch.randn(remaining_dims) * 0.1
            
            embeddings.append(embedding)
        
        return torch.stack(embeddings)
    
    @profile('tensor_ops.batch_operations')
    def batch_process_tensors(self, tensors: List[Any], operation: str, 
                            **operation_kwargs) -> List[Any]:
        """Batch process multiple tensors for efficiency."""
        if not tensors:
            return []
        
        if operation == 'normalize':
            return [torch.nn.functional.normalize(t, p=2, dim=-1) for t in tensors]
        
        elif operation == 'relu':
            return [torch.relu(t) for t in tensors]
        
        elif operation == 'softmax':
            dim = operation_kwargs.get('dim', -1)
            return [torch.softmax(t, dim=dim) for t in tensors]
        
        elif operation == 'layer_norm':
            # Apply layer normalization
            results = []
            for tensor in tensors:
                mean = tensor.mean(dim=-1, keepdim=True)
                std = tensor.std(dim=-1, keepdim=True)
                normalized = (tensor - mean) / (std + 1e-8)
                results.append(normalized)
            return results
        
        else:
            raise ValueError(f"Unsupported batch operation: {operation}")


# Global optimized operations instance
_global_tensor_operations: Optional[OptimizedTensorOperations] = None
_tensor_ops_lock = threading.Lock()


def get_global_tensor_operations() -> OptimizedTensorOperations:
    """Get the global optimized tensor operations instance (thread-safe singleton)."""
    global _global_tensor_operations
    
    if _global_tensor_operations is None:
        with _tensor_ops_lock:
            if _global_tensor_operations is None:
                _global_tensor_operations = OptimizedTensorOperations()
    
    return _global_tensor_operations


# Convenience functions for common operations
def optimized_attention(queries: Any, keys: Any, values: Any, mask: Optional[Any] = None) -> Tuple[Any, Any]:
    """Compute optimized attention weights."""
    return get_global_tensor_operations().compute_attention_weights(queries, keys, values, mask)


def optimized_similarity(tensor_a: Any, tensor_b: Any, similarity_type: str = 'cosine') -> Any:
    """Compute optimized similarity matrix."""
    return get_global_tensor_operations().compute_similarity_matrix(tensor_a, tensor_b, similarity_type)


def optimized_consolidation(memory_tensors: List[Any], method: str = 'weighted_average', 
                          weights: Optional[List[float]] = None) -> Any:
    """Consolidate memory tensors with optimization."""
    return get_global_tensor_operations().consolidate_memory_tensors(memory_tensors, method, weights)