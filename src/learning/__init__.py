"""
Learning systems for adaptive cognitive architectures.

This package contains learning mechanisms including relevance learning,
adaptive cognitive systems, and experience-based adaptation.
"""

from .relevance_learning import RelevanceLearner, RelevanceExperience
from .meta_learning import MetaLearner, TransferLearning, FewShotLearner, AdaptiveLearningManager, KnowledgeDistiller, CurriculumLearner, LearningStrategy

__all__ = [
    'RelevanceLearner',
    'RelevanceExperience',
    'MetaLearner',
    'TransferLearning', 
    'FewShotLearner',
    'AdaptiveLearningManager',
    'KnowledgeDistiller',
    'CurriculumLearner',
    'LearningStrategy'
]