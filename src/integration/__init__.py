"""
Integration modules for coordinating multiple cognitive systems.

This package handles the integration and coordination between different
cognitive frameworks and systems within CogPrime, including cross-domain
integration capabilities.
"""

# Import key integration modules
__all__ = []

try:
    from .integration_core import IntegrationCore
    __all__.append('IntegrationCore')
except ImportError:
    pass

try:
    from .aletheia import AletheiaCore
    __all__.append('AletheiaCore')
except ImportError:
    pass

try:
    from .imaginal import ImaginalCore  
    __all__.append('ImaginalCore')
except ImportError:
    pass

try:
    from .psyche import PsycheCore
    __all__.append('PsycheCore')
except ImportError:
    pass

# Cross-Domain Integration Framework
try:
    from .cross_domain_framework import (
        CrossDomainIntegrationFramework,
        DomainType,
        ModalityType,
        ConceptMapping,
        AbstractConcept,
        UnifiedRepresentationSystem,
        CrossModalAttentionMechanism,
        DomainAdaptationSystem
    )
    __all__.extend([
        'CrossDomainIntegrationFramework',
        'DomainType',
        'ModalityType', 
        'ConceptMapping',
        'AbstractConcept',
        'UnifiedRepresentationSystem',
        'CrossModalAttentionMechanism',
        'DomainAdaptationSystem'
    ])
except ImportError:
    pass

try:
    from .cross_domain_reasoning import (
        CrossDomainReasoningEngine,
        ReasoningType,
        CrossDomainInference,
        CrossDomainKnowledgeGraph,
        AnalogicalReasoningEngine,
        CausalReasoningEngine
    )
    __all__.extend([
        'CrossDomainReasoningEngine',
        'ReasoningType',
        'CrossDomainInference', 
        'CrossDomainKnowledgeGraph',
        'AnalogicalReasoningEngine',
        'CausalReasoningEngine'
    ])
except ImportError:
    pass

try:
    from .multimodal_knowledge_graph import (
        MultiModalKnowledgeGraph,
        ModalityFeature,
        ModalityEmbeddingType,
        CrossModalCorrespondence,
        MultiModalEntity,
        ModalityProcessor,
        VisualProcessor,
        AudioProcessor,
        LanguageProcessor
    )
    __all__.extend([
        'MultiModalKnowledgeGraph',
        'ModalityFeature',
        'ModalityEmbeddingType',
        'CrossModalCorrespondence',
        'MultiModalEntity',
        'ModalityProcessor', 
        'VisualProcessor',
        'AudioProcessor',
        'LanguageProcessor'
    ])
except ImportError:
    pass