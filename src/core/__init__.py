"""
CogPrime Core Systems

This package contains the core cognitive architecture including:
- Main CogPrime cognitive core with cognitive cycles
- Vervaeke-inspired cognitive core with relevance realization
- Relevance realization core systems
"""

# Original CogPrime core
from .cognitive_core import CogPrimeCore
from .state import CognitiveState

# Vervaeke-inspired cognitive core
from .vervaeke_cognitive_core import CognitiveCore, CognitiveFrame, KnowingMode

# Relevance realization core
from .relevance_core import RelevanceCore, RelevanceMode

__all__ = [
    # Original CogPrime
    'CogPrimeCore',
    'CognitiveState',
    
    # Vervaeke systems
    'CognitiveCore',
    'CognitiveFrame', 
    'KnowingMode',
    'RelevanceCore',
    'RelevanceMode'
] 