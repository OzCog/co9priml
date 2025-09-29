"""
Parallel processing optimization for cognitive operations.

Provides thread-safe parallel execution of independent cognitive operations
to maximize utilization of available CPU cores and improve overall throughput.
"""

import asyncio
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import time
import logging
from dataclasses import dataclass
from collections import deque
import queue

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a parallel task execution."""
    task_id: str
    result: Any
    execution_time_ms: float
    worker_id: int
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None
    
    @property
    def success(self) -> bool:
        """Check if task completed successfully."""
        return self.error is None


class CognitiveTaskPool:
    """Specialized thread pool for cognitive operations."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 task_timeout: float = 30.0,
                 queue_size: int = 1000):
        
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.task_timeout = task_timeout
        
        # Create thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="cognitive_worker"
        )
        
        # Task management
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: deque = deque(maxlen=queue_size)
        
        # Performance tracking
        self.task_stats = {
            'submitted': 0,
            'completed': 0,
            'failed': 0,
            'timeouts': 0,
            'avg_execution_time_ms': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"CognitiveTaskPool initialized with {self.max_workers} workers")
    
    def submit_task(self, 
                   task_id: str,
                   func: Callable,
                   args: Tuple = (),
                   kwargs: Dict[str, Any] = None,
                   metadata: Dict[str, Any] = None) -> asyncio.Future:
        """Submit a cognitive task for parallel execution."""
        kwargs = kwargs or {}
        metadata = metadata or {}
        
        start_time = time.time()
        
        def wrapped_task():
            """Wrapper for task execution with timing and error handling."""
            worker_id = threading.get_ident()
            task_start = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.time() - task_start) * 1000
                
                task_result = TaskResult(
                    task_id=task_id,
                    result=result,
                    execution_time_ms=execution_time_ms,
                    worker_id=worker_id,
                    metadata=metadata
                )
                
                with self._lock:
                    self.task_stats['completed'] += 1
                    self.completed_tasks.append(task_result)
                    
                    # Update average execution time
                    total_time = (self.task_stats['avg_execution_time_ms'] * 
                                (self.task_stats['completed'] - 1) + execution_time_ms)
                    self.task_stats['avg_execution_time_ms'] = total_time / self.task_stats['completed']
                
                return task_result
                
            except Exception as e:
                execution_time_ms = (time.time() - task_start) * 1000
                
                task_result = TaskResult(
                    task_id=task_id,
                    result=None,
                    execution_time_ms=execution_time_ms,
                    worker_id=worker_id,
                    error=e,
                    metadata=metadata
                )
                
                with self._lock:
                    self.task_stats['failed'] += 1
                    self.completed_tasks.append(task_result)
                
                logger.error(f"Task {task_id} failed: {e}")
                return task_result
        
        # Submit to thread pool
        future = self.executor.submit(wrapped_task)
        
        with self._lock:
            self.active_tasks[task_id] = {
                'future': future,
                'start_time': start_time,
                'metadata': metadata
            }
            self.task_stats['submitted'] += 1
        
        return future
    
    def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, TaskResult]:
        """Wait for specific tasks to complete."""
        results = {}
        timeout = timeout or self.task_timeout
        
        for task_id in task_ids:
            with self._lock:
                if task_id in self.active_tasks:
                    future = self.active_tasks[task_id]['future']
                    try:
                        result = future.result(timeout=timeout)
                        results[task_id] = result
                        
                        # Clean up
                        del self.active_tasks[task_id]
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Task {task_id} timed out after {timeout}s")
                        with self._lock:
                            self.task_stats['timeouts'] += 1
                        
                        # Create timeout result
                        results[task_id] = TaskResult(
                            task_id=task_id,
                            result=None,
                            execution_time_ms=timeout * 1000,
                            worker_id=-1,
                            error=asyncio.TimeoutError(f"Task timed out after {timeout}s")
                        )
        
        return results
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get thread pool performance statistics."""
        with self._lock:
            active_count = len(self.active_tasks)
            
            return {
                'max_workers': self.max_workers,
                'active_tasks': active_count,
                'queue_size': active_count,  # Approximation
                'stats': self.task_stats.copy(),
                'success_rate': (
                    self.task_stats['completed'] / 
                    max(1, self.task_stats['submitted'])
                ),
                'recent_completed': len(self.completed_tasks)
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        logger.info("Shutting down CognitiveTaskPool")
        self.executor.shutdown(wait=wait)


class ParallelCognitiveProcessor:
    """High-level parallel processor for cognitive operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        self.max_workers = config.get('max_workers', min(8, multiprocessing.cpu_count()))
        self.task_timeout = config.get('task_timeout', 30.0)
        
        # Specialized task pools for different types of operations
        self.reasoning_pool = CognitiveTaskPool(
            max_workers=max(2, self.max_workers // 2),
            task_timeout=self.task_timeout
        )
        
        self.memory_pool = CognitiveTaskPool(
            max_workers=max(2, self.max_workers // 4),
            task_timeout=self.task_timeout * 2  # Memory operations might take longer
        )
        
        self.pattern_pool = CognitiveTaskPool(
            max_workers=max(2, self.max_workers // 4),
            task_timeout=self.task_timeout
        )
        
        # Task coordination
        self._task_counter = 0
        self._lock = threading.RLock()
        
        logger.info(f"ParallelCognitiveProcessor initialized with {self.max_workers} total workers")
    
    def _get_next_task_id(self) -> str:
        """Generate unique task ID."""
        with self._lock:
            self._task_counter += 1
            return f"task_{self._task_counter}_{int(time.time() * 1000)}"
    
    async def parallel_reasoning(self, reasoning_tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Execute multiple reasoning operations in parallel."""
        task_futures = []
        task_ids = []
        
        for task_data in reasoning_tasks:
            task_id = self._get_next_task_id()
            task_ids.append(task_id)
            
            future = self.reasoning_pool.submit_task(
                task_id=task_id,
                func=task_data['function'],
                args=task_data.get('args', ()),
                kwargs=task_data.get('kwargs', {}),
                metadata={'type': 'reasoning', 'priority': task_data.get('priority', 'normal')}
            )
            task_futures.append(future)
        
        # Wait for all tasks to complete
        results = []
        for future in as_completed(task_futures, timeout=self.task_timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel reasoning task failed: {e}")
                # Create error result
                results.append(TaskResult(
                    task_id="unknown",
                    result=None,
                    execution_time_ms=0,
                    worker_id=-1,
                    error=e
                ))
        
        return results
    
    async def parallel_pattern_matching(self, 
                                      patterns: List[Any], 
                                      content: Any,
                                      matcher_func: Callable) -> List[TaskResult]:
        """Execute pattern matching for multiple patterns in parallel."""
        task_futures = []
        task_ids = []
        
        for i, pattern in enumerate(patterns):
            task_id = self._get_next_task_id()
            task_ids.append(task_id)
            
            future = self.pattern_pool.submit_task(
                task_id=task_id,
                func=matcher_func,
                args=(pattern, content),
                metadata={'type': 'pattern_matching', 'pattern_index': i}
            )
            task_futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(task_futures, timeout=self.task_timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Pattern matching task failed: {e}")
                results.append(TaskResult(
                    task_id="pattern_unknown",
                    result=None,
                    execution_time_ms=0,
                    worker_id=-1,
                    error=e
                ))
        
        return results
    
    async def parallel_memory_retrieval(self, queries: List[Dict[str, Any]]) -> List[TaskResult]:
        """Execute multiple memory retrieval operations in parallel."""
        task_futures = []
        task_ids = []
        
        for query in queries:
            task_id = self._get_next_task_id()
            task_ids.append(task_id)
            
            future = self.memory_pool.submit_task(
                task_id=task_id,
                func=query['retrieval_function'],
                args=query.get('args', ()),
                kwargs=query.get('kwargs', {}),
                metadata={'type': 'memory_retrieval', 'query_type': query.get('type', 'general')}
            )
            task_futures.append(future)
        
        # Wait and collect results
        results = []
        for future in as_completed(task_futures, timeout=self.task_timeout * 2):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Memory retrieval task failed: {e}")
                results.append(TaskResult(
                    task_id="memory_unknown",
                    result=None,
                    execution_time_ms=0,
                    worker_id=-1,
                    error=e
                ))
        
        return results
    
    def batch_execute_independent_operations(self, 
                                           operations: List[Dict[str, Any]],
                                           pool_type: str = 'reasoning') -> List[TaskResult]:
        """Execute a batch of independent operations in parallel."""
        # Select appropriate pool
        if pool_type == 'memory':
            pool = self.memory_pool
        elif pool_type == 'pattern':
            pool = self.pattern_pool
        else:
            pool = self.reasoning_pool
        
        # Submit all tasks
        task_futures = {}
        for operation in operations:
            task_id = self._get_next_task_id()
            
            future = pool.submit_task(
                task_id=task_id,
                func=operation['function'],
                args=operation.get('args', ()),
                kwargs=operation.get('kwargs', {}),
                metadata=operation.get('metadata', {})
            )
            task_futures[task_id] = future
        
        # Wait for completion
        task_ids = list(task_futures.keys())
        return list(pool.wait_for_completion(task_ids).values())
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all pools."""
        reasoning_stats = self.reasoning_pool.get_pool_statistics()
        memory_stats = self.memory_pool.get_pool_statistics()
        pattern_stats = self.pattern_pool.get_pool_statistics()
        
        total_completed = (reasoning_stats['stats']['completed'] + 
                          memory_stats['stats']['completed'] + 
                          pattern_stats['stats']['completed'])
        
        total_submitted = (reasoning_stats['stats']['submitted'] + 
                          memory_stats['stats']['submitted'] + 
                          pattern_stats['stats']['submitted'])
        
        overall_success_rate = total_completed / max(1, total_submitted)
        
        return {
            'total_workers': self.max_workers,
            'overall_success_rate': overall_success_rate,
            'total_completed_tasks': total_completed,
            'total_submitted_tasks': total_submitted,
            'pools': {
                'reasoning': reasoning_stats,
                'memory': memory_stats,
                'pattern': pattern_stats
            }
        }
    
    def optimize_worker_allocation(self) -> Dict[str, Any]:
        """Dynamically optimize worker allocation based on usage patterns."""
        stats = self.get_comprehensive_statistics()
        
        optimizations = []
        
        # Analyze pool utilization
        reasoning_load = stats['pools']['reasoning']['active_tasks']
        memory_load = stats['pools']['memory']['active_tasks'] 
        pattern_load = stats['pools']['pattern']['active_tasks']
        
        total_load = reasoning_load + memory_load + pattern_load
        
        if total_load > 0:
            reasoning_ratio = reasoning_load / total_load
            memory_ratio = memory_load / total_load
            pattern_ratio = pattern_load / total_load
            
            # Suggest rebalancing if one pool is heavily used
            if reasoning_ratio > 0.6:
                optimizations.append("Consider increasing reasoning pool workers")
            elif memory_ratio > 0.4:
                optimizations.append("Consider increasing memory pool workers")
            elif pattern_ratio > 0.4:
                optimizations.append("Consider increasing pattern pool workers")
        
        return {
            'current_stats': stats,
            'load_distribution': {
                'reasoning': reasoning_load,
                'memory': memory_load,
                'pattern': pattern_load
            },
            'optimization_suggestions': optimizations
        }
    
    def shutdown_all_pools(self) -> None:
        """Shutdown all task pools."""
        logger.info("Shutting down all cognitive task pools")
        self.reasoning_pool.shutdown()
        self.memory_pool.shutdown()
        self.pattern_pool.shutdown()


# Global parallel processor instance
_global_parallel_processor: Optional[ParallelCognitiveProcessor] = None
_processor_lock = threading.Lock()


def get_global_parallel_processor() -> ParallelCognitiveProcessor:
    """Get the global parallel processor instance (thread-safe singleton)."""
    global _global_parallel_processor
    
    if _global_parallel_processor is None:
        with _processor_lock:
            if _global_parallel_processor is None:
                _global_parallel_processor = ParallelCognitiveProcessor()
    
    return _global_parallel_processor


def configure_global_parallel_processor(config: Dict[str, Any]) -> None:
    """Configure the global parallel processor."""
    global _global_parallel_processor
    
    with _processor_lock:
        if _global_parallel_processor is not None:
            _global_parallel_processor.shutdown_all_pools()
        
        _global_parallel_processor = ParallelCognitiveProcessor(config)
        logger.info("Global parallel processor reconfigured")