"""
Dynamic Resource Management System

Sophisticated resource management capabilities that dynamically allocate 
computational resources, memory, and attention based on current cognitive 
demands and system constraints.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import gc
import weakref
from pathlib import Path

# Performance optimization imports
try:
    from ..performance.cognitive_optimizer import get_global_cognitive_optimizer
    from ..performance.profiler import profile
    from ..performance.parallel_processor import get_global_parallel_processor
    from ..performance.memory_pool import MemoryPool
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATIONS_AVAILABLE = False
    
    # Create dummy decorators if performance modules not available
    def profile(name=None):
        def decorator(func):
            return func
        return decorator


class ResourceType(Enum):
    """Types of resources managed by the system"""
    CPU = "cpu"
    MEMORY = "memory"
    ATTENTION = "attention"
    IO = "io"
    GPU = "gpu"
    NETWORK = "network"


class Priority(Enum):
    """Priority levels for resource allocation"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ResourceState(Enum):
    """States of resource allocation"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    EXHAUSTED = "exhausted"
    DEGRADED = "degraded"


@dataclass
class ResourceConstraint:
    """Resource constraint specification"""
    resource_type: ResourceType
    min_required: float
    max_allowed: float
    preferred: float
    hard_limit: bool = False


@dataclass
class ResourceRequest:
    """Request for resource allocation"""
    request_id: str
    resource_type: ResourceType
    amount: float
    priority: Priority
    duration_estimate: Optional[float] = None
    callback: Optional[Callable] = None
    constraints: List[ResourceConstraint] = field(default_factory=list)
    submitted_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None


@dataclass
class ResourceAllocation:
    """Allocated resource tracking"""
    allocation_id: str
    request_id: str
    resource_type: ResourceType
    allocated_amount: float
    start_time: float
    estimated_duration: Optional[float]
    actual_usage: float = 0.0
    peak_usage: float = 0.0
    efficiency: float = 1.0


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    resource_type: ResourceType
    total_capacity: float
    available: float
    allocated: float
    reserved: float
    utilization_rate: float
    efficiency: float
    contention_level: float
    avg_wait_time: float
    last_updated: float = field(default_factory=time.time)


@dataclass
class PredictionModel:
    """Resource usage prediction model"""
    resource_type: ResourceType
    historical_data: deque = field(default_factory=lambda: deque(maxlen=1000))
    prediction_accuracy: float = 0.0
    last_prediction: Optional[float] = None
    trend_coefficient: float = 0.0
    seasonal_pattern: List[float] = field(default_factory=list)


class PriorityScheduler:
    """Priority-based scheduler for computational tasks"""
    
    def __init__(self, max_concurrent_tasks: int = 4):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queues = {priority: deque() for priority in Priority}
        self.active_tasks: Dict[str, ResourceAllocation] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.logger = logging.getLogger(__name__)
    
    def submit_task(self, request: ResourceRequest) -> str:
        """Submit a task for scheduling"""
        with self.lock:
            self.task_queues[request.priority].append(request)
            self.logger.debug(f"Task {request.request_id} queued with priority {request.priority}")
            return request.request_id
    
    def get_next_task(self) -> Optional[ResourceRequest]:
        """Get the next highest priority task"""
        with self.lock:
            for priority in Priority:
                if self.task_queues[priority]:
                    return self.task_queues[priority].popleft()
            return None
    
    def can_schedule_task(self, request: ResourceRequest) -> bool:
        """Check if task can be scheduled given current load"""
        with self.lock:
            return len(self.active_tasks) < self.max_concurrent_tasks
    
    def schedule_tasks(self) -> List[str]:
        """Schedule available tasks"""
        scheduled = []
        while self.can_schedule_task(None):
            task = self.get_next_task()
            if not task:
                break
            
            # Create allocation record
            allocation = ResourceAllocation(
                allocation_id=f"alloc_{task.request_id}_{int(time.time())}",
                request_id=task.request_id,
                resource_type=task.resource_type,
                allocated_amount=task.amount,
                start_time=time.time(),
                estimated_duration=task.duration_estimate
            )
            
            with self.lock:
                self.active_tasks[task.request_id] = allocation
            scheduled.append(task.request_id)
            
        return scheduled


class MemoryManager:
    """Advanced memory management with garbage collection optimization"""
    
    def __init__(self, initial_pool_size: int = 1024 * 1024 * 100):  # 100MB
        self.pool_size = initial_pool_size
        self.allocated_blocks: Dict[str, Tuple[int, float]] = {}
        self.free_blocks: List[Tuple[int, int]] = [(0, initial_pool_size)]
        self.garbage_collection_threshold = 0.8
        self.weak_references: weakref.WeakSet = weakref.WeakSet()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize memory pool if available
        if PERFORMANCE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.memory_pool = MemoryPool(initial_pool_size)
            except:
                self.memory_pool = None
        else:
            self.memory_pool = None
    
    def allocate(self, size: int, alignment: int = 8) -> Optional[str]:
        """Allocate memory block"""
        aligned_size = ((size + alignment - 1) // alignment) * alignment
        
        with self.lock:
            # Find suitable free block
            for i, (start, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Allocate from this block
                    block_id = f"mem_{start}_{aligned_size}_{int(time.time())}"
                    self.allocated_blocks[block_id] = (aligned_size, time.time())
                    
                    # Update free blocks
                    if block_size == aligned_size:
                        self.free_blocks.pop(i)
                    else:
                        self.free_blocks[i] = (start + aligned_size, block_size - aligned_size)
                    
                    self.logger.debug(f"Allocated {aligned_size} bytes as {block_id}")
                    return block_id
            
            # No suitable block found, try garbage collection
            if self.get_utilization() > self.garbage_collection_threshold:
                self.garbage_collect()
                return self.allocate(size, alignment)  # Retry after GC
            
            return None
    
    def deallocate(self, block_id: str) -> bool:
        """Deallocate memory block"""
        with self.lock:
            if block_id not in self.allocated_blocks:
                return False
            
            size, _ = self.allocated_blocks.pop(block_id)
            # In a real implementation, would merge adjacent free blocks
            self.logger.debug(f"Deallocated block {block_id} ({size} bytes)")
            return True
    
    def garbage_collect(self) -> int:
        """Perform garbage collection"""
        collected = 0
        with self.lock:
            # Remove expired weak references
            expired_refs = [ref for ref in self.weak_references if ref() is None]
            for ref in expired_refs:
                self.weak_references.discard(ref)
                collected += 1
            
            # Force Python garbage collection
            collected += gc.collect()
            
            self.logger.info(f"Garbage collection completed, collected {collected} objects")
            return collected
    
    def get_utilization(self) -> float:
        """Get memory utilization ratio"""
        with self.lock:
            allocated = sum(size for size, _ in self.allocated_blocks.values())
            return allocated / self.pool_size
    
    def get_fragmentation(self) -> float:
        """Get memory fragmentation ratio"""
        with self.lock:
            if not self.free_blocks:
                return 0.0
            
            total_free = sum(size for _, size in self.free_blocks)
            largest_free = max(size for _, size in self.free_blocks) if self.free_blocks else 0
            
            if total_free == 0:
                return 0.0
                
            return 1.0 - (largest_free / total_free)


class AttentionResourceAllocator:
    """Manages attention resource allocation with cognitive load balancing"""
    
    def __init__(self, total_attention_capacity: float = 100.0):
        self.total_capacity = total_attention_capacity
        self.allocations: Dict[str, float] = {}
        self.priorities: Dict[str, Priority] = {}
        self.focus_history: deque = deque(maxlen=100)
        self.attention_decay_rate = 0.95
        self.minimum_attention_quantum = 0.1
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def allocate_attention(self, target: str, amount: float, priority: Priority) -> bool:
        """Allocate attention to a target"""
        with self.lock:
            available = self.get_available_attention()
            
            if amount > available and priority != Priority.CRITICAL:
                # Try to reallocate from lower priority targets
                self._rebalance_attention(amount, priority)
                available = self.get_available_attention()
            
            if amount <= available or priority == Priority.CRITICAL:
                self.allocations[target] = self.allocations.get(target, 0) + amount
                self.priorities[target] = priority
                self.focus_history.append((target, amount, time.time()))
                self.logger.debug(f"Allocated {amount} attention to {target}")
                return True
            
            return False
    
    def release_attention(self, target: str, amount: Optional[float] = None) -> float:
        """Release attention from a target"""
        with self.lock:
            if target not in self.allocations:
                return 0.0
            
            if amount is None:
                released = self.allocations.pop(target, 0.0)
                self.priorities.pop(target, None)
            else:
                current = self.allocations.get(target, 0.0)
                released = min(amount, current)
                self.allocations[target] = max(0, current - released)
                if self.allocations[target] < self.minimum_attention_quantum:
                    self.allocations.pop(target)
                    self.priorities.pop(target, None)
            
            self.logger.debug(f"Released {released} attention from {target}")
            return released
    
    def get_available_attention(self) -> float:
        """Get currently available attention capacity"""
        with self.lock:
            allocated = sum(self.allocations.values())
            return max(0, self.total_capacity - allocated)
    
    def _rebalance_attention(self, needed: float, priority: Priority) -> float:
        """Rebalance attention by reducing lower priority allocations"""
        reallocated = 0.0
        with self.lock:
            # Sort targets by priority (higher priority value = lower importance)
            targets_by_priority = sorted(
                [(target, alloc, self.priorities.get(target, Priority.NORMAL)) 
                 for target, alloc in self.allocations.items()],
                key=lambda x: x[2].value,
                reverse=True
            )
            
            for target, allocation, target_priority in targets_by_priority:
                if target_priority.value > priority.value:  # Lower priority
                    reduction = min(allocation * 0.2, needed - reallocated)
                    self.allocations[target] -= reduction
                    reallocated += reduction
                    
                    if reallocated >= needed:
                        break
            
            return reallocated
    
    def decay_attention(self) -> None:
        """Apply attention decay over time"""
        with self.lock:
            for target in list(self.allocations.keys()):
                self.allocations[target] *= self.attention_decay_rate
                if self.allocations[target] < self.minimum_attention_quantum:
                    self.allocations.pop(target)
                    self.priorities.pop(target, None)


class LoadBalancer:
    """Adaptive load balancing across cognitive modules"""
    
    def __init__(self, modules: List[str]):
        self.modules = modules
        self.module_loads: Dict[str, float] = {module: 0.0 for module in modules}
        self.module_capacities: Dict[str, float] = {module: 100.0 for module in modules}
        self.load_history: Dict[str, deque] = {module: deque(maxlen=100) for module in modules}
        self.rebalance_threshold = 0.8
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def assign_task(self, task_load: float) -> Optional[str]:
        """Assign task to the least loaded suitable module"""
        with self.lock:
            # Calculate current utilization for each module
            module_utilizations = [
                (module, self.module_loads[module] / self.module_capacities[module])
                for module in self.modules
                if self.module_loads[module] + task_load <= self.module_capacities[module]
            ]
            
            if not module_utilizations:
                # Try to rebalance if no modules can handle the task
                self._rebalance_modules()
                module_utilizations = [
                    (module, self.module_loads[module] / self.module_capacities[module])
                    for module in self.modules
                    if self.module_loads[module] + task_load <= self.module_capacities[module]
                ]
            
            if module_utilizations:
                # Select module with lowest utilization
                selected_module = min(module_utilizations, key=lambda x: x[1])[0]
                self.module_loads[selected_module] += task_load
                self.load_history[selected_module].append((task_load, time.time()))
                self.logger.debug(f"Assigned task (load {task_load}) to {selected_module}")
                return selected_module
            
            return None
    
    def complete_task(self, module: str, task_load: float) -> None:
        """Mark task as completed on a module"""
        with self.lock:
            if module in self.module_loads:
                self.module_loads[module] = max(0, self.module_loads[module] - task_load)
                self.logger.debug(f"Completed task (load {task_load}) on {module}")
    
    def _rebalance_modules(self) -> None:
        """Rebalance load across modules"""
        with self.lock:
            total_load = sum(self.module_loads.values())
            avg_load = total_load / len(self.modules)
            
            # Identify overloaded and underloaded modules
            overloaded = [(module, load) for module, load in self.module_loads.items() 
                         if load > avg_load * 1.2]
            underloaded = [(module, load) for module, load in self.module_loads.items() 
                          if load < avg_load * 0.8]
            
            # Simple rebalancing strategy
            for over_module, over_load in overloaded:
                for under_module, under_load in underloaded:
                    transfer_amount = min(
                        over_load - avg_load,
                        self.module_capacities[under_module] - under_load
                    ) * 0.1  # Transfer 10% at a time
                    
                    if transfer_amount > 0:
                        self.module_loads[over_module] -= transfer_amount
                        self.module_loads[under_module] += transfer_amount
                        break


class ResourceMonitor:
    """Real-time resource monitoring and alerting system"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics: Dict[ResourceType, ResourceMetrics] = {}
        self.alert_thresholds: Dict[ResourceType, float] = {
            ResourceType.CPU: 0.9,
            ResourceType.MEMORY: 0.85,
            ResourceType.ATTENTION: 0.95
        }
        self.alert_callbacks: List[Callable] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                self._check_alerts()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _update_system_metrics(self) -> None:
        """Update system resource metrics"""
        with self.lock:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics[ResourceType.CPU] = ResourceMetrics(
                resource_type=ResourceType.CPU,
                total_capacity=100.0,
                available=100.0 - cpu_percent,
                allocated=cpu_percent,
                reserved=0.0,
                utilization_rate=cpu_percent / 100.0,
                efficiency=1.0,
                contention_level=max(0, cpu_percent - 80) / 20.0,
                avg_wait_time=0.0
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics[ResourceType.MEMORY] = ResourceMetrics(
                resource_type=ResourceType.MEMORY,
                total_capacity=memory.total,
                available=memory.available,
                allocated=memory.used,
                reserved=0.0,
                utilization_rate=memory.percent / 100.0,
                efficiency=1.0,
                contention_level=max(0, memory.percent - 80) / 20.0,
                avg_wait_time=0.0
            )
    
    def _check_alerts(self) -> None:
        """Check for alert conditions"""
        with self.lock:
            for resource_type, metrics in self.metrics.items():
                threshold = self.alert_thresholds.get(resource_type, 0.9)
                if metrics.utilization_rate > threshold:
                    alert_data = {
                        'resource_type': resource_type,
                        'utilization': metrics.utilization_rate,
                        'threshold': threshold,
                        'timestamp': time.time()
                    }
                    
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert_data)
                        except Exception as e:
                            self.logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback for resource alerts"""
        self.alert_callbacks.append(callback)
    
    def get_metrics(self, resource_type: Optional[ResourceType] = None) -> Dict[ResourceType, ResourceMetrics]:
        """Get current resource metrics"""
        with self.lock:
            if resource_type:
                return {resource_type: self.metrics.get(resource_type)}
            return self.metrics.copy()


class ResourcePredictor:
    """Resource usage prediction and preallocation system"""
    
    def __init__(self):
        self.prediction_models: Dict[ResourceType, PredictionModel] = {
            resource_type: PredictionModel(resource_type=resource_type)
            for resource_type in ResourceType
        }
        self.prediction_horizon = 60.0  # 1 minute ahead
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def record_usage(self, resource_type: ResourceType, usage: float) -> None:
        """Record resource usage data point"""
        with self.lock:
            model = self.prediction_models[resource_type]
            model.historical_data.append((time.time(), usage))
    
    def predict_usage(self, resource_type: ResourceType, 
                     time_ahead: float = None) -> Optional[float]:
        """Predict future resource usage"""
        if time_ahead is None:
            time_ahead = self.prediction_horizon
            
        with self.lock:
            model = self.prediction_models[resource_type]
            
            if len(model.historical_data) < 10:
                return None
            
            # Simple linear prediction based on recent trend
            recent_data = list(model.historical_data)[-10:]
            times = np.array([t for t, _ in recent_data])
            values = np.array([v for _, v in recent_data])
            
            if len(times) < 2:
                return None
            
            # Linear regression
            coeffs = np.polyfit(times, values, 1)
            future_time = time.time() + time_ahead
            prediction = np.polyval(coeffs, future_time)
            
            model.last_prediction = prediction
            model.trend_coefficient = coeffs[0]
            
            return max(0, prediction)  # Ensure non-negative prediction
    
    def get_prediction_accuracy(self, resource_type: ResourceType) -> float:
        """Get prediction accuracy for a resource type"""
        with self.lock:
            return self.prediction_models[resource_type].prediction_accuracy
    
    def update_prediction_accuracy(self, resource_type: ResourceType, 
                                 predicted: float, actual: float) -> None:
        """Update prediction accuracy based on actual vs predicted values"""
        with self.lock:
            model = self.prediction_models[resource_type]
            error = abs(predicted - actual) / max(actual, 1.0)
            accuracy = 1.0 - min(error, 1.0)
            
            # Exponential moving average
            if model.prediction_accuracy == 0.0:
                model.prediction_accuracy = accuracy
            else:
                model.prediction_accuracy = 0.9 * model.prediction_accuracy + 0.1 * accuracy


class DynamicResourceManager:
    """Main dynamic resource management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.scheduler = PriorityScheduler(
            max_concurrent_tasks=self.config.get('max_concurrent_tasks', 4)
        )
        self.memory_manager = MemoryManager(
            initial_pool_size=self.config.get('memory_pool_size', 1024 * 1024 * 100)
        )
        self.attention_allocator = AttentionResourceAllocator(
            total_attention_capacity=self.config.get('attention_capacity', 100.0)
        )
        
        cognitive_modules = self.config.get('cognitive_modules', 
                                          ['reasoning', 'memory', 'attention', 'planning'])
        self.load_balancer = LoadBalancer(cognitive_modules)
        
        self.monitor = ResourceMonitor(
            update_interval=self.config.get('monitor_interval', 1.0)
        )
        self.predictor = ResourcePredictor()
        
        # Resource state tracking
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.resource_constraints: Dict[ResourceType, List[ResourceConstraint]] = defaultdict(list)
        self.degradation_strategies: Dict[ResourceType, Callable] = {}
        
        # Performance tracking
        self.allocation_history: deque = deque(maxlen=1000)
        self.performance_metrics = {
            'response_time': deque(maxlen=100),
            'throughput': deque(maxlen=100),
            'efficiency': deque(maxlen=100)
        }
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Setup alert callback
        self.monitor.add_alert_callback(self._handle_resource_alert)
    
    async def initialize(self) -> None:
        """Initialize the resource management system"""
        self.logger.info("Initializing Dynamic Resource Management System")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Initialize prediction models with some default data
        for resource_type in ResourceType:
            for i in range(10):
                self.predictor.record_usage(resource_type, 50.0 + np.random.normal(0, 10))
        
        self.logger.info("Dynamic Resource Management System initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the resource management system"""
        self.logger.info("Shutting down Dynamic Resource Management System")
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Cleanup scheduler
        self.scheduler.executor.shutdown(wait=True)
        
        self.logger.info("Dynamic Resource Management System shutdown complete")
    
    @profile(name="resource_allocation")
    async def allocate_resources(self, request: ResourceRequest) -> Optional[str]:
        """Allocate resources based on request"""
        start_time = time.time()
        
        try:
            # Check constraints
            if not self._check_constraints(request):
                self.logger.warning(f"Resource request {request.request_id} failed constraint check")
                return None
            
            # Predict future resource needs
            predicted_usage = self.predictor.predict_usage(request.resource_type)
            if predicted_usage and predicted_usage > 80.0:  # High predicted usage
                self.logger.info(f"High predicted usage for {request.resource_type}, adjusting allocation")
                request.amount *= 0.8  # Reduce allocation
            
            # Handle different resource types
            allocation_id = None
            
            if request.resource_type == ResourceType.CPU:
                allocation_id = await self._allocate_cpu(request)
            elif request.resource_type == ResourceType.MEMORY:
                allocation_id = await self._allocate_memory(request)
            elif request.resource_type == ResourceType.ATTENTION:
                allocation_id = await self._allocate_attention(request)
            else:
                allocation_id = await self._allocate_generic(request)
            
            if allocation_id:
                # Record allocation
                allocation = ResourceAllocation(
                    allocation_id=allocation_id,
                    request_id=request.request_id,
                    resource_type=request.resource_type,
                    allocated_amount=request.amount,
                    start_time=start_time,
                    estimated_duration=request.duration_estimate
                )
                
                with self.lock:
                    self.active_allocations[allocation_id] = allocation
                
                # Record performance metrics
                response_time = time.time() - start_time
                self.performance_metrics['response_time'].append(response_time)
                
                self.logger.info(f"Allocated {request.resource_type} resources: {allocation_id}")
                
                # Execute callback if provided
                if request.callback:
                    try:
                        await request.callback(allocation_id)
                    except Exception as e:
                        self.logger.error(f"Error in allocation callback: {e}")
            
            return allocation_id
            
        except Exception as e:
            self.logger.error(f"Error allocating resources for {request.request_id}: {e}")
            return None
    
    async def _allocate_cpu(self, request: ResourceRequest) -> Optional[str]:
        """Allocate CPU resources"""
        # Use load balancer to find suitable module
        module = self.load_balancer.assign_task(request.amount)
        if module:
            task_id = self.scheduler.submit_task(request)
            return f"cpu_{module}_{task_id}"
        return None
    
    async def _allocate_memory(self, request: ResourceRequest) -> Optional[str]:
        """Allocate memory resources"""
        block_id = self.memory_manager.allocate(int(request.amount))
        if block_id:
            return f"mem_{block_id}"
        return None
    
    async def _allocate_attention(self, request: ResourceRequest) -> Optional[str]:
        """Allocate attention resources"""
        success = self.attention_allocator.allocate_attention(
            request.request_id, request.amount, request.priority
        )
        if success:
            return f"att_{request.request_id}"
        return None
    
    async def _allocate_generic(self, request: ResourceRequest) -> Optional[str]:
        """Allocate generic resources"""
        # Simple allocation strategy for other resource types
        return f"gen_{request.resource_type.value}_{request.request_id}"
    
    def _check_constraints(self, request: ResourceRequest) -> bool:
        """Check if request meets resource constraints"""
        constraints = self.resource_constraints.get(request.resource_type, [])
        
        for constraint in constraints:
            if constraint.resource_type == request.resource_type:
                if request.amount < constraint.min_required:
                    return False
                if constraint.hard_limit and request.amount > constraint.max_allowed:
                    return False
        
        return True
    
    def _handle_resource_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle resource alert"""
        resource_type = alert_data['resource_type']
        utilization = alert_data['utilization']
        
        self.logger.warning(f"Resource alert: {resource_type} at {utilization:.1%} utilization")
        
        # Apply degradation strategy if available
        if resource_type in self.degradation_strategies:
            try:
                self.degradation_strategies[resource_type](alert_data)
            except Exception as e:
                self.logger.error(f"Error applying degradation strategy: {e}")
    
    async def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources"""
        with self.lock:
            if allocation_id not in self.active_allocations:
                return False
            
            allocation = self.active_allocations.pop(allocation_id)
        
        try:
            # Handle different resource types
            if allocation_id.startswith('cpu_'):
                parts = allocation_id.split('_')
                if len(parts) >= 3:
                    module = parts[1]
                    self.load_balancer.complete_task(module, allocation.allocated_amount)
            
            elif allocation_id.startswith('mem_'):
                block_id = allocation_id[4:]  # Remove 'mem_' prefix
                self.memory_manager.deallocate(block_id)
            
            elif allocation_id.startswith('att_'):
                target = allocation.request_id
                self.attention_allocator.release_attention(target)
            
            # Record completion
            allocation.actual_usage = allocation.allocated_amount  # Simplified
            allocation.efficiency = allocation.actual_usage / allocation.allocated_amount
            
            self.allocation_history.append(allocation)
            
            self.logger.info(f"Released resources: {allocation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error releasing resources {allocation_id}: {e}")
            return False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        with self.lock:
            return {
                'active_allocations': len(self.active_allocations),
                'cpu_utilization': self.load_balancer.module_loads.copy(),
                'memory_utilization': self.memory_manager.get_utilization(),
                'memory_fragmentation': self.memory_manager.get_fragmentation(),
                'attention_available': self.attention_allocator.get_available_attention(),
                'system_metrics': self.monitor.get_metrics(),
                'performance_metrics': {
                    metric: list(values)[-10:] if values else []
                    for metric, values in self.performance_metrics.items()
                }
            }
    
    def add_resource_constraint(self, constraint: ResourceConstraint) -> None:
        """Add resource constraint"""
        self.resource_constraints[constraint.resource_type].append(constraint)
        self.logger.info(f"Added constraint for {constraint.resource_type}")
    
    def set_degradation_strategy(self, resource_type: ResourceType, 
                               strategy: Callable) -> None:
        """Set degradation strategy for resource type"""
        self.degradation_strategies[resource_type] = strategy
        self.logger.info(f"Set degradation strategy for {resource_type}")
    
    async def preallocation_cycle(self) -> None:
        """Run preallocation based on predictions"""
        try:
            for resource_type in ResourceType:
                predicted_usage = self.predictor.predict_usage(resource_type, 30.0)  # 30 seconds ahead
                
                if predicted_usage and predicted_usage > 70.0:  # High predicted demand
                    # Pre-allocate some resources
                    prealloc_request = ResourceRequest(
                        request_id=f"prealloc_{resource_type.value}_{int(time.time())}",
                        resource_type=resource_type,
                        amount=predicted_usage * 0.1,  # Pre-allocate 10% of predicted usage
                        priority=Priority.BACKGROUND
                    )
                    
                    await self.allocate_resources(prealloc_request)
                    self.logger.debug(f"Pre-allocated {resource_type} resources based on prediction")
        
        except Exception as e:
            self.logger.error(f"Error in preallocation cycle: {e}")