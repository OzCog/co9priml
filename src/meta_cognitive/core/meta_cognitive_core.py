"""
Meta-Cognitive Core Framework
============================

This module implements the core meta-cognitive framework that serves as the
central orchestrator for all meta-cognitive processes within the CogPrime
architecture. It provides the foundation for higher-order thinking about
thinking, self-awareness, and reasoning about cognitive processes.
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging


class MetaCognitiveLevel(Enum):
    """Levels of meta-cognitive processing depth."""
    OBJECT_LEVEL = 0       # Direct cognitive processing
    META_LEVEL_1 = 1       # Thinking about thinking
    META_LEVEL_2 = 2       # Thinking about thinking about thinking
    META_LEVEL_3 = 3       # Higher-order recursive meta-cognition


class MetaCognitiveMode(Enum):
    """Different modes of meta-cognitive operation."""
    MONITORING = "monitoring"           # Observing cognitive processes
    EVALUATING = "evaluating"          # Assessing cognitive performance
    CONTROLLING = "controlling"        # Directing cognitive processes
    PLANNING = "planning"              # Strategic cognitive planning
    REFLECTING = "reflecting"          # Post-hoc cognitive analysis


@dataclass
class MetaCognitiveContext:
    """Context information for meta-cognitive processes."""
    level: MetaCognitiveLevel
    mode: MetaCognitiveMode
    task_type: str
    cognitive_load: float
    resources_available: Dict[str, Any]
    timestamp: float
    parent_context: Optional['MetaCognitiveContext'] = None


@dataclass
class CognitiveProcess:
    """Representation of a cognitive process being monitored."""
    process_id: str
    process_type: str
    state: str
    performance_metrics: Dict[str, float]
    resources_used: Dict[str, float]
    start_time: float
    duration: Optional[float] = None
    meta_data: Dict[str, Any] = None


class MetaCognitiveCore:
    """
    The core meta-cognitive framework that orchestrates all meta-cognitive
    processes within the CogPrime architecture.
    
    This system provides:
    - Meta-cognitive process monitoring and control
    - Higher-order thinking capabilities
    - Self-awareness and introspection
    - Cognitive process analysis and optimization
    - Strategy selection and adaptation
    - Recursive meta-cognitive processing
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the meta-cognitive core framework."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core state
        self.active_processes: Dict[str, CognitiveProcess] = {}
        self.meta_cognitive_stack: List[MetaCognitiveContext] = []
        self.current_context: Optional[MetaCognitiveContext] = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.strategy_effectiveness: Dict[str, float] = {}
        
        # Configuration
        self.max_recursion_depth = self.config.get('max_recursion_depth', 3)
        self.monitoring_frequency = self.config.get('monitoring_frequency', 0.1)
        self.enable_introspection = self.config.get('enable_introspection', True)
        
        # Initialize subsystems (will be injected by framework)
        self.higher_order_thinking = None
        self.self_awareness = None
        self.process_analyzer = None
        self.strategy_selector = None
        self.recursive_processor = None
        self.meta_knowledge = None
        self.meta_learner = None
        
        self.logger.info("Meta-cognitive core framework initialized")
    
    def register_subsystem(self, name: str, subsystem: Any) -> None:
        """Register a meta-cognitive subsystem."""
        setattr(self, name, subsystem)
        self.logger.debug(f"Registered meta-cognitive subsystem: {name}")
    
    def enter_meta_cognitive_mode(self, 
                                  mode: MetaCognitiveMode,
                                  task_type: str = "general",
                                  level: MetaCognitiveLevel = MetaCognitiveLevel.META_LEVEL_1) -> MetaCognitiveContext:
        """Enter a meta-cognitive processing mode."""
        # Check recursion depth
        if len(self.meta_cognitive_stack) >= self.max_recursion_depth:
            self.logger.warning(f"Max recursion depth {self.max_recursion_depth} reached")
            return self.current_context
        
        # Create new context
        context = MetaCognitiveContext(
            level=level,
            mode=mode,
            task_type=task_type,
            cognitive_load=self._calculate_cognitive_load(),
            resources_available=self._assess_available_resources(),
            timestamp=time.time(),
            parent_context=self.current_context
        )
        
        # Update state
        self.meta_cognitive_stack.append(context)
        self.current_context = context
        
        self.logger.debug(f"Entered meta-cognitive mode: {mode.value} at level {level.value}")
        return context
    
    def exit_meta_cognitive_mode(self) -> Optional[MetaCognitiveContext]:
        """Exit current meta-cognitive mode and return to parent."""
        if not self.meta_cognitive_stack:
            return None
        
        # Pop current context
        exited_context = self.meta_cognitive_stack.pop()
        
        # Restore parent context
        self.current_context = exited_context.parent_context
        
        # Record performance
        self._record_context_performance(exited_context)
        
        self.logger.debug(f"Exited meta-cognitive mode: {exited_context.mode.value}")
        return exited_context
    
    def monitor_cognitive_process(self, process: CognitiveProcess) -> Dict[str, Any]:
        """Monitor and analyze a cognitive process."""
        self.active_processes[process.process_id] = process
        
        # Perform monitoring
        monitoring_result = {
            'process_id': process.process_id,
            'performance_assessment': self._assess_process_performance(process),
            'resource_utilization': self._assess_resource_utilization(process),
            'optimization_suggestions': self._generate_optimization_suggestions(process),
            'timestamp': time.time()
        }
        
        # Update knowledge base
        if self.meta_knowledge:
            self.meta_knowledge.record_process_observation(process, monitoring_result)
        
        return monitoring_result
    
    def reflect_on_cognition(self, reflection_depth: int = 1) -> Dict[str, Any]:
        """Perform meta-cognitive reflection on recent cognitive processes."""
        if not self.enable_introspection:
            return {'reflection': 'introspection_disabled'}
        
        # Enter reflective mode
        context = self.enter_meta_cognitive_mode(
            MetaCognitiveMode.REFLECTING,
            level=MetaCognitiveLevel(min(reflection_depth, self.max_recursion_depth))
        )
        
        reflection_result = {
            'reflection_depth': reflection_depth,
            'processes_analyzed': len(self.active_processes),
            'insights': [],
            'patterns_detected': [],
            'improvement_recommendations': []
        }
        
        try:
            # Analyze recent processes
            if self.process_analyzer:
                analysis = self.process_analyzer.analyze_recent_processes(
                    list(self.active_processes.values())
                )
                reflection_result['insights'].extend(analysis.get('insights', []))
                reflection_result['patterns_detected'].extend(analysis.get('patterns', []))
            
            # Generate higher-order insights
            if self.higher_order_thinking:
                higher_order_insights = self.higher_order_thinking.generate_insights(
                    self.performance_history,
                    self.active_processes
                )
                reflection_result['insights'].extend(higher_order_insights)
            
            # Self-awareness assessment
            if self.self_awareness:
                awareness_state = self.self_awareness.assess_current_state()
                reflection_result['self_awareness_state'] = awareness_state
            
            # Strategy effectiveness evaluation
            if self.strategy_selector:
                strategy_assessment = self.strategy_selector.evaluate_current_strategies()
                reflection_result['strategy_effectiveness'] = strategy_assessment
            
        except Exception as e:
            self.logger.error(f"Error during meta-cognitive reflection: {e}")
            reflection_result['error'] = str(e)
        
        finally:
            # Exit reflective mode
            self.exit_meta_cognitive_mode()
        
        return reflection_result
    
    def optimize_cognitive_strategy(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cognitive strategy based on current context and past performance."""
        # Enter planning mode
        context = self.enter_meta_cognitive_mode(
            MetaCognitiveMode.PLANNING,
            task_type=task_context.get('task_type', 'general')
        )
        
        optimization_result = {
            'current_strategy': None,
            'recommended_strategy': None,
            'expected_improvement': 0.0,
            'confidence': 0.0
        }
        
        try:
            # Analyze current context
            if self.process_analyzer:
                context_analysis = self.process_analyzer.analyze_context(task_context)
                optimization_result['context_analysis'] = context_analysis
            
            # Select optimal strategy
            if self.strategy_selector:
                strategy_recommendation = self.strategy_selector.select_optimal_strategy(
                    task_context, self.performance_history
                )
                optimization_result.update(strategy_recommendation)
            
            # Learn from optimization
            if self.meta_learner:
                self.meta_learner.update_from_optimization(
                    task_context, optimization_result
                )
        
        except Exception as e:
            self.logger.error(f"Error during strategy optimization: {e}")
            optimization_result['error'] = str(e)
        
        finally:
            # Exit planning mode
            self.exit_meta_cognitive_mode()
        
        return optimization_result
    
    def recursive_meta_analyze(self, depth: int = 2) -> Dict[str, Any]:
        """Perform recursive meta-cognitive analysis."""
        if self.recursive_processor:
            return self.recursive_processor.recursive_analyze(
                self.active_processes,
                self.performance_history,
                depth=depth
            )
        return {'error': 'recursive_processor_not_available'}
    
    def get_meta_cognitive_state(self) -> Dict[str, Any]:
        """Get comprehensive meta-cognitive state information."""
        return {
            'current_context': {
                'level': self.current_context.level.value if self.current_context else None,
                'mode': self.current_context.mode.value if self.current_context else None,
                'task_type': self.current_context.task_type if self.current_context else None,
                'cognitive_load': self.current_context.cognitive_load if self.current_context else 0.0
            },
            'recursion_depth': len(self.meta_cognitive_stack),
            'active_processes': len(self.active_processes),
            'performance_history_size': len(self.performance_history),
            'strategy_effectiveness': dict(self.strategy_effectiveness),
            'subsystems_available': {
                'higher_order_thinking': self.higher_order_thinking is not None,
                'self_awareness': self.self_awareness is not None,
                'process_analyzer': self.process_analyzer is not None,
                'strategy_selector': self.strategy_selector is not None,
                'recursive_processor': self.recursive_processor is not None,
                'meta_knowledge': self.meta_knowledge is not None,
                'meta_learner': self.meta_learner is not None
            }
        }
    
    # Private helper methods
    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load."""
        base_load = len(self.active_processes) * 0.1
        recursion_load = len(self.meta_cognitive_stack) * 0.2
        return min(base_load + recursion_load, 1.0)
    
    def _assess_available_resources(self) -> Dict[str, Any]:
        """Assess currently available cognitive resources."""
        return {
            'memory_available': True,  # Simplified
            'processing_capacity': 1.0 - self._calculate_cognitive_load(),
            'attention_available': len(self.active_processes) < 10
        }
    
    def _assess_process_performance(self, process: CognitiveProcess) -> Dict[str, float]:
        """Assess the performance of a cognitive process."""
        # Simplified performance assessment
        return {
            'efficiency': process.performance_metrics.get('efficiency', 0.5),
            'accuracy': process.performance_metrics.get('accuracy', 0.5),
            'speed': process.performance_metrics.get('speed', 0.5),
            'resource_efficiency': 1.0 - sum(process.resources_used.values()) / len(process.resources_used) if process.resources_used else 0.5
        }
    
    def _assess_resource_utilization(self, process: CognitiveProcess) -> Dict[str, float]:
        """Assess resource utilization of a process."""
        return process.resources_used or {'cpu': 0.1, 'memory': 0.1, 'attention': 0.1}
    
    def _generate_optimization_suggestions(self, process: CognitiveProcess) -> List[str]:
        """Generate optimization suggestions for a process."""
        suggestions = []
        
        performance = self._assess_process_performance(process)
        if performance['efficiency'] < 0.5:
            suggestions.append("Consider alternative processing strategies")
        if performance['speed'] < 0.3:
            suggestions.append("Optimize for faster processing")
        if sum(process.resources_used.values()) > 0.8:
            suggestions.append("Reduce resource consumption")
        
        return suggestions
    
    def _record_context_performance(self, context: MetaCognitiveContext) -> None:
        """Record performance metrics for a completed context."""
        duration = time.time() - context.timestamp
        
        performance_record = {
            'context_level': context.level.value,
            'context_mode': context.mode.value,
            'task_type': context.task_type,
            'duration': duration,
            'cognitive_load': context.cognitive_load,
            'timestamp': context.timestamp
        }
        
        self.performance_history.append(performance_record)
        
        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-800:]