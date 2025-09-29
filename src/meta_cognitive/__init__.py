"""
Meta-Cognitive Synthesis Framework
==================================

This package implements a comprehensive meta-cognitive synthesis framework
that integrates insights from cognitive science, philosophy of mind, and 
artificial intelligence to create a unified approach to meta-cognition
within the CogPrime architecture.

Core Components:
- Meta-cognitive architecture and interfaces
- Higher-order thinking capabilities  
- Self-awareness and introspection mechanisms
- Cognitive process reasoning and analysis
- Meta-cognitive strategy selection and optimization
- Recursive meta-cognitive processing
- Meta-cognitive knowledge representation
- Meta-cognitive learning and adaptation

This framework enables higher-order thinking about thinking, self-awareness,
and the ability to reason about cognitive processes themselves, creating a
foundation for advanced artificial general intelligence capabilities.
"""

from .core.meta_cognitive_core import MetaCognitiveCore
from .interfaces.meta_cognitive_interface import MetaCognitiveInterface
from .processing.higher_order_thinking import HigherOrderThinking
from .awareness.self_awareness import SelfAwarenessSystem
from .analysis.process_analyzer import CognitiveProcessAnalyzer
from .strategy.strategy_selector import MetaCognitiveStrategySelector
from .recursive.recursive_processor import RecursiveMetaCognitiveProcessor
from .knowledge.meta_knowledge_system import MetaKnowledgeSystem
from .learning.meta_learner import MetaCognitiveLearner

# Meta-Cognitive Synthesis Framework Factory
def create_meta_cognitive_framework(config: dict = None) -> MetaCognitiveCore:
    """
    Factory function to create a complete meta-cognitive synthesis framework.
    
    Args:
        config: Configuration dictionary for the framework
        
    Returns:
        Configured MetaCognitiveCore with all subsystems initialized
    """
    # Create core framework
    meta_core = MetaCognitiveCore(config)
    
    # Create and register all subsystems
    higher_order_thinking = HigherOrderThinking(config)
    self_awareness = SelfAwarenessSystem(config)
    process_analyzer = CognitiveProcessAnalyzer(config)
    strategy_selector = MetaCognitiveStrategySelector(config)
    recursive_processor = RecursiveMetaCognitiveProcessor(config)
    meta_knowledge = MetaKnowledgeSystem(config)
    meta_learner = MetaCognitiveLearner(config)
    
    # Register subsystems with core
    meta_core.register_subsystem('higher_order_thinking', higher_order_thinking)
    meta_core.register_subsystem('self_awareness', self_awareness)
    meta_core.register_subsystem('process_analyzer', process_analyzer)
    meta_core.register_subsystem('strategy_selector', strategy_selector)
    meta_core.register_subsystem('recursive_processor', recursive_processor)
    meta_core.register_subsystem('meta_knowledge', meta_knowledge)
    meta_core.register_subsystem('meta_learner', meta_learner)
    
    # Initialize all subsystems
    subsystems = [
        higher_order_thinking, self_awareness, process_analyzer,
        strategy_selector, recursive_processor, meta_knowledge, meta_learner
    ]
    
    for subsystem in subsystems:
        if hasattr(subsystem, 'initialize'):
            subsystem.initialize()
    
    return meta_core

__all__ = [
    'MetaCognitiveCore',
    'MetaCognitiveInterface', 
    'HigherOrderThinking',
    'SelfAwarenessSystem',
    'CognitiveProcessAnalyzer',
    'MetaCognitiveStrategySelector',
    'RecursiveMetaCognitiveProcessor',
    'MetaKnowledgeSystem',
    'MetaCognitiveLearner'
]