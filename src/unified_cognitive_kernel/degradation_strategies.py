"""
Graceful Degradation Strategies

Implements sophisticated degradation strategies for maintaining core functionality
under resource pressure. These strategies ensure the system remains operational
while gracefully reducing non-essential capabilities.
"""

import logging
import time
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

from .resource_manager import ResourceType, Priority


class DegradationLevel(Enum):
    """Levels of system degradation"""
    NONE = 0
    MINIMAL = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class DegradationStrategy(Enum):
    """Types of degradation strategies"""
    RESOURCE_THROTTLING = "resource_throttling"
    FEATURE_REDUCTION = "feature_reduction"
    QUALITY_REDUCTION = "quality_reduction"
    CACHE_EVICTION = "cache_eviction"
    BACKGROUND_SUSPENSION = "background_suspension"
    EMERGENCY_CLEANUP = "emergency_cleanup"


@dataclass
class DegradationAction:
    """Represents a degradation action that can be taken"""
    strategy: DegradationStrategy
    description: str
    resource_type: ResourceType
    severity: DegradationLevel
    action: Callable[[Dict[str, Any]], None]
    recovery_action: Optional[Callable[[Dict[str, Any]], None]] = None
    min_trigger_threshold: float = 0.8
    max_trigger_threshold: float = 1.0


class GracefulDegradationManager:
    """Manages graceful degradation under resource pressure"""
    
    def __init__(self):
        self.degradation_actions: Dict[ResourceType, List[DegradationAction]] = {}
        self.active_degradations: Dict[str, DegradationAction] = {}
        self.degradation_history: List[Dict[str, Any]] = []
        self.recovery_callbacks: List[Callable] = []
        self.current_degradation_level = DegradationLevel.NONE
        self.logger = logging.getLogger(__name__)
        
        # Initialize default degradation strategies
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self):
        """Initialize default degradation strategies for each resource type"""
        
        # CPU degradation strategies
        cpu_strategies = [
            DegradationAction(
                strategy=DegradationStrategy.BACKGROUND_SUSPENSION,
                description="Suspend non-critical background tasks",
                resource_type=ResourceType.CPU,
                severity=DegradationLevel.MINIMAL,
                action=self._suspend_background_tasks,
                recovery_action=self._resume_background_tasks,
                min_trigger_threshold=0.75,
                max_trigger_threshold=0.85
            ),
            DegradationAction(
                strategy=DegradationStrategy.QUALITY_REDUCTION,
                description="Reduce computational quality for non-critical operations",
                resource_type=ResourceType.CPU,
                severity=DegradationLevel.MODERATE,
                action=self._reduce_computation_quality,
                recovery_action=self._restore_computation_quality,
                min_trigger_threshold=0.85,
                max_trigger_threshold=0.95
            ),
            DegradationAction(
                strategy=DegradationStrategy.RESOURCE_THROTTLING,
                description="Throttle CPU-intensive operations",
                resource_type=ResourceType.CPU,
                severity=DegradationLevel.SEVERE,
                action=self._throttle_cpu_operations,
                recovery_action=self._unthrottle_cpu_operations,
                min_trigger_threshold=0.95,
                max_trigger_threshold=1.0
            )
        ]
        
        # Memory degradation strategies
        memory_strategies = [
            DegradationAction(
                strategy=DegradationStrategy.CACHE_EVICTION,
                description="Evict least recently used cache entries",
                resource_type=ResourceType.MEMORY,
                severity=DegradationLevel.MINIMAL,
                action=self._evict_lru_cache,
                recovery_action=self._restore_cache_size,
                min_trigger_threshold=0.70,
                max_trigger_threshold=0.80
            ),
            DegradationAction(
                strategy=DegradationStrategy.EMERGENCY_CLEANUP,
                description="Force garbage collection and cleanup temporary objects",
                resource_type=ResourceType.MEMORY,
                severity=DegradationLevel.MODERATE,
                action=self._emergency_memory_cleanup,
                recovery_action=None,
                min_trigger_threshold=0.80,
                max_trigger_threshold=0.90
            ),
            DegradationAction(
                strategy=DegradationStrategy.FEATURE_REDUCTION,
                description="Disable memory-intensive features",
                resource_type=ResourceType.MEMORY,
                severity=DegradationLevel.SEVERE,
                action=self._disable_memory_intensive_features,
                recovery_action=self._enable_memory_intensive_features,
                min_trigger_threshold=0.90,
                max_trigger_threshold=1.0
            )
        ]
        
        # Attention degradation strategies
        attention_strategies = [
            DegradationAction(
                strategy=DegradationStrategy.RESOURCE_THROTTLING,
                description="Reduce attention allocation to non-critical processes",
                resource_type=ResourceType.ATTENTION,
                severity=DegradationLevel.MINIMAL,
                action=self._throttle_attention_allocation,
                recovery_action=self._restore_attention_allocation,
                min_trigger_threshold=0.80,
                max_trigger_threshold=0.90
            ),
            DegradationAction(
                strategy=DegradationStrategy.FEATURE_REDUCTION,
                description="Focus attention on critical cognitive processes only",
                resource_type=ResourceType.ATTENTION,
                severity=DegradationLevel.MODERATE,
                action=self._focus_critical_attention,
                recovery_action=self._restore_attention_distribution,
                min_trigger_threshold=0.90,
                max_trigger_threshold=1.0
            )
        ]
        
        self.degradation_actions[ResourceType.CPU] = cpu_strategies
        self.degradation_actions[ResourceType.MEMORY] = memory_strategies
        self.degradation_actions[ResourceType.ATTENTION] = attention_strategies
    
    def add_degradation_strategy(self, action: DegradationAction):
        """Add a custom degradation strategy"""
        if action.resource_type not in self.degradation_actions:
            self.degradation_actions[action.resource_type] = []
        
        self.degradation_actions[action.resource_type].append(action)
        self.logger.info(f"Added degradation strategy: {action.description}")
    
    def evaluate_degradation_need(self, resource_metrics: Dict[ResourceType, Any]) -> List[DegradationAction]:
        """Evaluate which degradation actions should be taken based on current metrics"""
        needed_actions = []
        
        for resource_type, metrics in resource_metrics.items():
            if resource_type not in self.degradation_actions:
                continue
            
            utilization = getattr(metrics, 'utilization_rate', 0.0)
            
            # Find applicable degradation actions
            for action in self.degradation_actions[resource_type]:
                action_id = f"{action.resource_type.name}_{action.strategy.value}"
                
                # Check if action should be triggered
                if (utilization >= action.min_trigger_threshold and 
                    utilization <= action.max_trigger_threshold and
                    action_id not in self.active_degradations):
                    
                    needed_actions.append(action)
                    self.logger.warning(
                        f"Degradation needed: {action.description} "
                        f"(utilization: {utilization:.1%})"
                    )
                
                # Check if action should be recovered
                elif (utilization < action.min_trigger_threshold * 0.9 and  # Hysteresis
                      action_id in self.active_degradations):
                    
                    # Schedule recovery
                    if action.recovery_action:
                        needed_actions.append(action)  # Will be handled in apply_degradation
        
        return needed_actions
    
    async def apply_degradation(self, action: DegradationAction, context: Dict[str, Any] = None):
        """Apply a degradation action"""
        if context is None:
            context = {}
        
        action_id = f"{action.resource_type.name}_{action.strategy.value}"
        
        try:
            # Check if this is a recovery action
            if action_id in self.active_degradations:
                # This is a recovery
                if action.recovery_action:
                    self.logger.info(f"Recovering from degradation: {action.description}")
                    await self._safe_execute_action(action.recovery_action, context)
                    
                    # Remove from active degradations
                    del self.active_degradations[action_id]
                    
                    # Record recovery
                    self.degradation_history.append({
                        'action': 'recovery',
                        'strategy': action.strategy.value,
                        'resource_type': action.resource_type.name,
                        'timestamp': time.time(),
                        'description': action.description
                    })
                    
                    self.logger.info(f"✅ Recovered from: {action.description}")
                
            else:
                # This is a new degradation
                self.logger.warning(f"Applying degradation: {action.description}")
                await self._safe_execute_action(action.action, context)
                
                # Add to active degradations
                self.active_degradations[action_id] = action
                
                # Update degradation level
                self._update_degradation_level()
                
                # Record degradation
                self.degradation_history.append({
                    'action': 'degradation',
                    'strategy': action.strategy.value,
                    'resource_type': action.resource_type.name,
                    'severity': action.severity.name,
                    'timestamp': time.time(),
                    'description': action.description
                })
                
                self.logger.warning(f"🚨 Applied degradation: {action.description}")
        
        except Exception as e:
            self.logger.error(f"Error applying degradation action {action.description}: {e}")
    
    async def _safe_execute_action(self, action_func: Callable, context: Dict[str, Any]):
        """Safely execute a degradation action"""
        try:
            if asyncio.iscoroutinefunction(action_func):
                await action_func(context)
            else:
                action_func(context)
        except Exception as e:
            self.logger.error(f"Error executing degradation action: {e}")
            # Don't re-raise - degradation failures shouldn't crash the system
    
    def _update_degradation_level(self):
        """Update the current degradation level based on active degradations"""
        if not self.active_degradations:
            self.current_degradation_level = DegradationLevel.NONE
            return
        
        max_severity = max(action.severity for action in self.active_degradations.values())
        self.current_degradation_level = max_severity
        
        self.logger.info(f"System degradation level: {self.current_degradation_level.name}")
    
    # Default degradation action implementations
    
    def _suspend_background_tasks(self, context: Dict[str, Any]):
        """Suspend non-critical background tasks"""
        self.logger.info("🔄 Suspending background tasks to free CPU resources")
        # In a real implementation, this would communicate with task scheduler
        # to suspend or delay non-critical tasks
        context['background_tasks_suspended'] = True
    
    def _resume_background_tasks(self, context: Dict[str, Any]):
        """Resume background tasks"""
        self.logger.info("🔄 Resuming background tasks")
        context['background_tasks_suspended'] = False
    
    def _reduce_computation_quality(self, context: Dict[str, Any]):
        """Reduce computational quality for non-critical operations"""
        self.logger.info("📉 Reducing computation quality to save CPU resources")
        # This might involve using faster but less accurate algorithms,
        # reducing precision, or simplifying calculations
        context['computation_quality_reduced'] = True
        context['quality_reduction_factor'] = 0.7  # 70% of normal quality
    
    def _restore_computation_quality(self, context: Dict[str, Any]):
        """Restore normal computation quality"""
        self.logger.info("📈 Restoring normal computation quality")
        context['computation_quality_reduced'] = False
        context['quality_reduction_factor'] = 1.0
    
    def _throttle_cpu_operations(self, context: Dict[str, Any]):
        """Throttle CPU-intensive operations"""
        self.logger.info("🐌 Throttling CPU-intensive operations")
        # This might involve adding delays, reducing parallelism,
        # or limiting operation frequency
        context['cpu_throttled'] = True
        context['cpu_throttle_factor'] = 0.5  # 50% throttling
    
    def _unthrottle_cpu_operations(self, context: Dict[str, Any]):
        """Remove CPU throttling"""
        self.logger.info("🚀 Removing CPU throttling")
        context['cpu_throttled'] = False
        context['cpu_throttle_factor'] = 1.0
    
    def _evict_lru_cache(self, context: Dict[str, Any]):
        """Evict least recently used cache entries"""
        self.logger.info("🗑️ Evicting LRU cache entries to free memory")
        # In a real implementation, this would interface with cache managers
        # to evict less important cached data
        context['cache_eviction_active'] = True
        context['cache_eviction_percentage'] = 0.3  # Evict 30% of cache
    
    def _restore_cache_size(self, context: Dict[str, Any]):
        """Restore normal cache size"""
        self.logger.info("💾 Restoring normal cache size")
        context['cache_eviction_active'] = False
        context['cache_eviction_percentage'] = 0.0
    
    def _emergency_memory_cleanup(self, context: Dict[str, Any]):
        """Perform emergency memory cleanup"""
        self.logger.info("🚨 Performing emergency memory cleanup")
        # Force garbage collection, clear temporary objects, etc.
        import gc
        collected = gc.collect()
        self.logger.info(f"🗑️ Emergency cleanup collected {collected} objects")
        context['emergency_cleanup_performed'] = True
    
    def _disable_memory_intensive_features(self, context: Dict[str, Any]):
        """Disable memory-intensive features"""
        self.logger.info("❌ Disabling memory-intensive features")
        # This might disable caching, reduce buffer sizes, 
        # or turn off memory-hungry algorithms
        context['memory_intensive_features_disabled'] = True
        context['disabled_features'] = ['large_buffers', 'extensive_caching', 'history_logging']
    
    def _enable_memory_intensive_features(self, context: Dict[str, Any]):
        """Re-enable memory-intensive features"""
        self.logger.info("✅ Re-enabling memory-intensive features")
        context['memory_intensive_features_disabled'] = False
        context['disabled_features'] = []
    
    def _throttle_attention_allocation(self, context: Dict[str, Any]):
        """Throttle attention allocation to non-critical processes"""
        self.logger.info("🎯 Throttling attention allocation to non-critical processes")
        # Reduce attention available to low-priority processes
        context['attention_throttled'] = True
        context['attention_throttle_factor'] = 0.6  # 60% of normal attention
    
    def _restore_attention_allocation(self, context: Dict[str, Any]):
        """Restore normal attention allocation"""
        self.logger.info("🎯 Restoring normal attention allocation")
        context['attention_throttled'] = False
        context['attention_throttle_factor'] = 1.0
    
    def _focus_critical_attention(self, context: Dict[str, Any]):
        """Focus attention on critical cognitive processes only"""
        self.logger.info("🔍 Focusing attention on critical processes only")
        # Reallocate attention from non-critical to critical processes
        context['attention_focused_critical'] = True
        context['critical_processes_only'] = True
    
    def _restore_attention_distribution(self, context: Dict[str, Any]):
        """Restore normal attention distribution"""
        self.logger.info("🔍 Restoring normal attention distribution")
        context['attention_focused_critical'] = False
        context['critical_processes_only'] = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation status"""
        return {
            'current_level': self.current_degradation_level.name,
            'active_degradations': len(self.active_degradations),
            'degradation_details': [
                {
                    'resource_type': action.resource_type.name,
                    'strategy': action.strategy.value,
                    'description': action.description,
                    'severity': action.severity.name
                }
                for action in self.active_degradations.values()
            ],
            'history_count': len(self.degradation_history)
        }
    
    def get_degradation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent degradation history"""
        return self.degradation_history[-limit:] if self.degradation_history else []
    
    def add_recovery_callback(self, callback: Callable):
        """Add a callback to be called when system recovers from degradation"""
        self.recovery_callbacks.append(callback)
    
    async def check_full_recovery(self):
        """Check if system has fully recovered and call callbacks if so"""
        if (self.current_degradation_level == DegradationLevel.NONE and 
            len(self.active_degradations) == 0):
            
            for callback in self.recovery_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.error(f"Error in recovery callback: {e}")
            
            self.logger.info("🎉 System fully recovered from all degradations")


class ResourcePressureDetector:
    """Detects resource pressure patterns and predicts degradation needs"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.pressure_history: Dict[ResourceType, List[float]] = {
            resource_type: [] for resource_type in ResourceType
        }
        self.pressure_trends: Dict[ResourceType, float] = {}
        self.logger = logging.getLogger(__name__)
    
    def record_pressure(self, resource_type: ResourceType, utilization: float):
        """Record resource pressure measurement"""
        if resource_type not in self.pressure_history:
            self.pressure_history[resource_type] = []
        
        history = self.pressure_history[resource_type]
        history.append(utilization)
        
        # Maintain history size
        if len(history) > self.history_size:
            history.pop(0)
        
        # Update trend
        self._update_pressure_trend(resource_type)
    
    def _update_pressure_trend(self, resource_type: ResourceType):
        """Update pressure trend for a resource type"""
        history = self.pressure_history[resource_type]
        
        if len(history) < 10:
            self.pressure_trends[resource_type] = 0.0
            return
        
        # Simple trend calculation: compare recent vs older measurements
        recent_avg = sum(history[-5:]) / 5
        older_avg = sum(history[-10:-5]) / 5
        
        self.pressure_trends[resource_type] = recent_avg - older_avg
    
    def predict_pressure_spike(self, resource_type: ResourceType, 
                              threshold: float = 0.8) -> bool:
        """Predict if a resource pressure spike is likely"""
        if resource_type not in self.pressure_history:
            return False
        
        history = self.pressure_history[resource_type]
        trend = self.pressure_trends.get(resource_type, 0.0)
        
        if len(history) < 5:
            return False
        
        current_pressure = history[-1]
        
        # Predict spike if current pressure is high and trending upward
        predicted_pressure = current_pressure + trend * 3  # 3 time steps ahead
        
        return predicted_pressure > threshold
    
    def get_pressure_status(self) -> Dict[str, Any]:
        """Get current pressure status for all resource types"""
        status = {}
        
        for resource_type in ResourceType:
            history = self.pressure_history.get(resource_type, [])
            trend = self.pressure_trends.get(resource_type, 0.0)
            current_pressure = history[-1] if history else 0.0
            
            status[resource_type.name] = {
                'current_pressure': current_pressure,
                'trend': trend,
                'spike_predicted': self.predict_pressure_spike(resource_type),
                'history_length': len(history)
            }
        
        return status