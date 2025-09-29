"""
Meta-Cognitive Interface
=======================

This module defines the standard interfaces and protocols for meta-cognitive
systems within the CogPrime architecture. It provides abstract base classes
and protocols that all meta-cognitive components must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol, Union
from dataclasses import dataclass


@dataclass
class MetaCognitiveCapability:
    """Represents a meta-cognitive capability."""
    name: str
    description: str
    complexity_level: int  # 1-5, where 5 is most complex
    requires_recursion: bool
    resource_intensive: bool


class MetaCognitiveProcessor(Protocol):
    """Protocol for meta-cognitive processing components."""
    
    def process(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Process meta-cognitive input and return results."""
        ...
    
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of capabilities this processor provides."""
        ...


class MetaCognitiveMonitor(Protocol):
    """Protocol for meta-cognitive monitoring components."""
    
    def monitor(self, process_id: str, process_data: Any) -> Dict[str, Any]:
        """Monitor a cognitive process and return observations."""
        ...
    
    def get_monitoring_frequency(self) -> float:
        """Return recommended monitoring frequency in Hz."""
        ...


class MetaCognitiveController(Protocol):
    """Protocol for meta-cognitive control components."""
    
    def control(self, process_id: str, control_signal: Any) -> bool:
        """Send control signal to a cognitive process."""
        ...
    
    def can_control(self, process_type: str) -> bool:
        """Check if this controller can control the given process type."""
        ...


class MetaCognitiveInterface(ABC):
    """
    Abstract base class for meta-cognitive system interfaces.
    
    This interface defines the standard methods that all meta-cognitive
    components should implement to integrate with the CogPrime architecture.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the meta-cognitive interface."""
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.priority = self.config.get('priority', 1.0)
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the meta-cognitive component."""
        pass
    
    @abstractmethod 
    def shutdown(self) -> bool:
        """Shutdown the meta-cognitive component cleanly."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of meta-cognitive capabilities provided."""
        pass
    
    @abstractmethod
    def process_meta_cognitive_request(self, 
                                     request_type: str,
                                     request_data: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meta-cognitive request."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if this component is enabled."""
        return self.enabled
    
    def get_priority(self) -> float:
        """Get the priority of this component."""
        return self.priority
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable this component."""
        self.enabled = enabled
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the component."""
        return {
            'enabled': self.enabled,
            'priority': self.priority,
            'capabilities': [cap.name for cap in self.get_capabilities()]
        }


class HigherOrderThinkingInterface(MetaCognitiveInterface):
    """Interface for higher-order thinking components."""
    
    @abstractmethod
    def think_about_thinking(self, 
                           thought_process: Any,
                           analysis_depth: int = 1) -> Dict[str, Any]:
        """Analyze and reason about a thought process."""
        pass
    
    @abstractmethod
    def generate_meta_insights(self, 
                             cognitive_data: List[Any]) -> List[Dict[str, Any]]:
        """Generate higher-order insights from cognitive data."""
        pass
    
    @abstractmethod
    def abstract_reasoning(self, 
                         concrete_examples: List[Any],
                         abstraction_level: int = 1) -> Dict[str, Any]:
        """Perform abstract reasoning from concrete examples."""
        pass


class SelfAwarenessInterface(MetaCognitiveInterface):
    """Interface for self-awareness components."""
    
    @abstractmethod
    def assess_self_state(self) -> Dict[str, Any]:
        """Assess current self-state and capabilities."""
        pass
    
    @abstractmethod
    def introspect(self, 
                  focus_area: Optional[str] = None) -> Dict[str, Any]:
        """Perform introspective analysis."""
        pass
    
    @abstractmethod
    def monitor_self_change(self) -> Dict[str, Any]:
        """Monitor changes in self-state over time."""
        pass


class ProcessAnalysisInterface(MetaCognitiveInterface):
    """Interface for cognitive process analysis components."""
    
    @abstractmethod
    def analyze_process_efficiency(self, 
                                 process_data: Any) -> Dict[str, float]:
        """Analyze the efficiency of a cognitive process."""
        pass
    
    @abstractmethod
    def identify_bottlenecks(self, 
                           process_chain: List[Any]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in a process chain."""
        pass
    
    @abstractmethod
    def suggest_optimizations(self, 
                            process_data: Any) -> List[str]:
        """Suggest optimizations for a process."""
        pass


class StrategySelectionInterface(MetaCognitiveInterface):
    """Interface for meta-cognitive strategy selection components."""
    
    @abstractmethod
    def select_strategy(self, 
                       task_context: Dict[str, Any],
                       available_strategies: List[str]) -> str:
        """Select optimal strategy for given context."""
        pass
    
    @abstractmethod
    def evaluate_strategy_effectiveness(self, 
                                      strategy: str,
                                      performance_data: Dict[str, Any]) -> float:
        """Evaluate the effectiveness of a strategy."""
        pass
    
    @abstractmethod
    def adapt_strategy(self, 
                      current_strategy: str,
                      feedback: Dict[str, Any]) -> str:
        """Adapt strategy based on feedback."""
        pass


class RecursiveProcessingInterface(MetaCognitiveInterface):
    """Interface for recursive meta-cognitive processing components."""
    
    @abstractmethod
    def recursive_analyze(self, 
                        data: Any,
                        depth: int = 2) -> Dict[str, Any]:
        """Perform recursive analysis at specified depth."""
        pass
    
    @abstractmethod
    def check_recursion_termination(self, 
                                  current_depth: int,
                                  analysis_quality: float) -> bool:
        """Check if recursion should terminate."""
        pass
    
    @abstractmethod
    def manage_recursive_resources(self, 
                                 depth: int) -> Dict[str, Any]:
        """Manage resources for recursive processing."""
        pass


class MetaKnowledgeInterface(MetaCognitiveInterface):
    """Interface for meta-cognitive knowledge representation components."""
    
    @abstractmethod
    def store_meta_knowledge(self, 
                           knowledge_type: str,
                           knowledge_data: Any) -> bool:
        """Store meta-cognitive knowledge."""
        pass
    
    @abstractmethod
    def retrieve_meta_knowledge(self, 
                              knowledge_type: str,
                              query: Dict[str, Any]) -> List[Any]:
        """Retrieve meta-cognitive knowledge."""
        pass
    
    @abstractmethod
    def update_meta_knowledge(self, 
                            knowledge_id: str,
                            updates: Dict[str, Any]) -> bool:
        """Update existing meta-cognitive knowledge."""
        pass


class MetaLearningInterface(MetaCognitiveInterface):
    """Interface for meta-cognitive learning components."""
    
    @abstractmethod
    def learn_from_experience(self, 
                            experience_data: Dict[str, Any]) -> bool:
        """Learn from meta-cognitive experience."""
        pass
    
    @abstractmethod
    def adapt_meta_strategies(self, 
                            performance_feedback: Dict[str, Any]) -> bool:
        """Adapt meta-cognitive strategies based on feedback."""
        pass
    
    @abstractmethod
    def transfer_meta_knowledge(self, 
                              source_domain: str,
                              target_domain: str) -> Dict[str, Any]:
        """Transfer meta-knowledge between domains."""
        pass


# Factory function for creating meta-cognitive components
def create_meta_cognitive_component(component_type: str, 
                                  config: Dict[str, Any] = None) -> Optional[MetaCognitiveInterface]:
    """Factory function to create meta-cognitive components."""
    # This would be implemented with actual component classes
    component_registry = {
        'higher_order_thinking': None,  # Would map to actual implementations
        'self_awareness': None,
        'process_analysis': None,
        'strategy_selection': None,
        'recursive_processing': None,
        'meta_knowledge': None,
        'meta_learning': None
    }
    
    component_class = component_registry.get(component_type)
    if component_class:
        return component_class(config)
    return None