"""
CogPrime: An Integrative Architecture for Embodied Artificial General Intelligence

This package integrates multiple cognitive frameworks including:
- Core cognitive architectures with AtomSpace
- Vervaeke-inspired relevance realization systems  
- Advanced cognitive science modules
- Learning and adaptation systems
- Action generation and selection
- Integration frameworks for multi-system coordination
"""

# Core cognitive systems
from .core.cognitive_core import CogPrimeCore
from .core.state import CognitiveState
from .core.vervaeke_cognitive_core import CognitiveCore, CognitiveFrame, KnowingMode
from .core.relevance_core import RelevanceCore, RelevanceMode

# Vervaeke-inspired systems (now integrated)
from .learning.relevance_learning import RelevanceLearner, RelevanceExperience
from .action.relevance_action import ActionGenerator, Action, ActionType

__all__ = [
    # Original CogPrime core
    'CogPrimeCore',
    'CognitiveState',
    
    # Vervaeke cognitive components
    'CognitiveCore',
    'CognitiveFrame', 
    'KnowingMode',
    'RelevanceCore',
    'RelevanceMode',
    
    # Learning systems
    'RelevanceLearner',
    'RelevanceExperience',
    
    # Action systems  
    'ActionGenerator',
    'Action',
    'ActionType'
] 