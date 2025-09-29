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