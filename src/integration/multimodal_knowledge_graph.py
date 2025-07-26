"""
Cross-Modal Knowledge Graph Integration

This module implements knowledge graph integration across different modalities,
enabling unified representation and reasoning over multi-modal information
including visual, auditory, linguistic, and other sensory data.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import copy
import json
import logging

from .cross_domain_framework import (
    DomainType, ModalityType, ConceptMapping, AbstractConcept,
    UnifiedRepresentationSystem, CrossDomainIntegrationFramework
)
from .cross_domain_reasoning import (
    CrossDomainKnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
)


class ModalityEmbeddingType(Enum):
    """Types of embeddings for different modalities"""
    VISUAL_CNN = "visual_cnn"
    VISUAL_TRANSFORMER = "visual_transformer"
    AUDIO_MFCC = "audio_mfcc"
    AUDIO_SPECTROGRAM = "audio_spectrogram"
    TEXT_BERT = "text_bert"
    TEXT_WORD2VEC = "text_word2vec"
    SPATIAL_COORDINATES = "spatial_coordinates"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    MULTIMODAL_CLIP = "multimodal_clip"
    UNIFIED_REPRESENTATION = "unified_representation"


@dataclass
class ModalityFeature:
    """Feature extracted from a specific modality"""
    feature_id: str
    modality: ModalityType
    embedding_type: ModalityEmbeddingType
    feature_vector: np.ndarray
    confidence: float
    temporal_window: Optional[Tuple[float, float]] = None  # Start and end time
    spatial_region: Optional[Dict[str, float]] = None  # Spatial bounding information
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossModalCorrespondence:
    """Correspondence between features across modalities"""
    correspondence_id: str
    modality_features: Dict[ModalityType, ModalityFeature]
    correspondence_strength: float
    temporal_alignment: float  # How well temporally aligned
    spatial_alignment: float   # How well spatially aligned
    semantic_similarity: float # Semantic correspondence strength
    evidence_sources: List[str] = field(default_factory=list)
    uncertainty: float = 0.0


@dataclass
class MultiModalEntity:
    """Entity that spans multiple modalities"""
    entity_id: str
    entity_type: str
    modality_manifestations: Dict[ModalityType, List[ModalityFeature]] = field(default_factory=dict)
    cross_modal_correspondences: List[CrossModalCorrespondence] = field(default_factory=list)
    temporal_span: Optional[Tuple[float, float]] = None
    spatial_extent: Optional[Dict[str, Any]] = None
    confidence_scores: Dict[ModalityType, float] = field(default_factory=dict)
    entity_embeddings: Dict[ModalityEmbeddingType, np.ndarray] = field(default_factory=dict)


class ModalityProcessor(ABC):
    """Abstract base class for processing different modalities"""
    
    @abstractmethod
    def extract_features(self, input_data: Any, context: Dict[str, Any] = None) -> List[ModalityFeature]:
        """Extract features from input data"""
        pass
    
    @abstractmethod
    def compute_similarity(self, feature1: ModalityFeature, feature2: ModalityFeature) -> float:
        """Compute similarity between two features"""
        pass
    
    @abstractmethod
    def get_modality_type(self) -> ModalityType:
        """Get the modality type this processor handles"""
        pass


class VisualProcessor(ModalityProcessor):
    """Processor for visual modality"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
    def extract_features(self, input_data: Any, context: Dict[str, Any] = None) -> List[ModalityFeature]:
        """Extract visual features from input data"""
        features = []
        
        # Simulate feature extraction (in practice, would use actual CNN/Vision Transformer)
        if isinstance(input_data, np.ndarray):
            # Simulate CNN features
            cnn_features = np.random.randn(self.embedding_dim) * 0.1
            cnn_features /= np.linalg.norm(cnn_features)
            
            visual_feature = ModalityFeature(
                feature_id=f"visual_{hash(str(input_data.tolist()))}",
                modality=ModalityType.VISION,
                embedding_type=ModalityEmbeddingType.VISUAL_CNN,
                feature_vector=cnn_features,
                confidence=0.8,
                spatial_region=context.get('spatial_region') if context else None,
                metadata={'input_shape': input_data.shape}
            )
            features.append(visual_feature)
        
        elif isinstance(input_data, str):
            # Visual description - convert to embedding
            desc_embedding = self._text_to_visual_embedding(input_data)
            
            visual_feature = ModalityFeature(
                feature_id=f"visual_desc_{hash(input_data)}",
                modality=ModalityType.VISION,
                embedding_type=ModalityEmbeddingType.VISUAL_TRANSFORMER,
                feature_vector=desc_embedding,
                confidence=0.6,
                metadata={'description': input_data}
            )
            features.append(visual_feature)
        
        return features
    
    def _text_to_visual_embedding(self, description: str) -> np.ndarray:
        """Convert visual description to embedding"""
        # Simple hash-based embedding (in practice, would use actual vision-language model)
        embedding = np.random.randn(self.embedding_dim) * 0.1
        
        # Add some structure based on common visual words
        visual_words = ['red', 'blue', 'large', 'small', 'round', 'square', 'bright', 'dark']
        for i, word in enumerate(visual_words):
            if word in description.lower():
                embedding[i * 64:(i + 1) * 64] += 0.5
        
        return embedding / np.linalg.norm(embedding)
    
    def compute_similarity(self, feature1: ModalityFeature, feature2: ModalityFeature) -> float:
        """Compute similarity between visual features"""
        if feature1.modality != ModalityType.VISION or feature2.modality != ModalityType.VISION:
            return 0.0
        
        # Cosine similarity
        cosine_sim = np.dot(feature1.feature_vector, feature2.feature_vector) / (
            np.linalg.norm(feature1.feature_vector) * np.linalg.norm(feature2.feature_vector)
        )
        
        # Weight by confidence
        confidence_weight = (feature1.confidence + feature2.confidence) / 2
        
        return float(cosine_sim * confidence_weight)
    
    def get_modality_type(self) -> ModalityType:
        """Get the modality type"""
        return ModalityType.VISION


class AudioProcessor(ModalityProcessor):
    """Processor for audio modality"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
    def extract_features(self, input_data: Any, context: Dict[str, Any] = None) -> List[ModalityFeature]:
        """Extract audio features from input data"""
        features = []
        
        if isinstance(input_data, np.ndarray):
            # Simulate MFCC features
            mfcc_features = np.random.randn(self.embedding_dim) * 0.1
            mfcc_features /= np.linalg.norm(mfcc_features)
            
            audio_feature = ModalityFeature(
                feature_id=f"audio_{hash(str(input_data.tolist()))}",
                modality=ModalityType.HEARING,
                embedding_type=ModalityEmbeddingType.AUDIO_MFCC,
                feature_vector=mfcc_features,
                confidence=0.8,
                temporal_window=context.get('temporal_window') if context else None,
                metadata={'sample_rate': context.get('sample_rate', 44100) if context else 44100}
            )
            features.append(audio_feature)
        
        elif isinstance(input_data, str):
            # Audio description - convert to embedding
            desc_embedding = self._text_to_audio_embedding(input_data)
            
            audio_feature = ModalityFeature(
                feature_id=f"audio_desc_{hash(input_data)}",
                modality=ModalityType.HEARING,
                embedding_type=ModalityEmbeddingType.AUDIO_SPECTROGRAM,
                feature_vector=desc_embedding,
                confidence=0.6,
                metadata={'description': input_data}
            )
            features.append(audio_feature)
        
        return features
    
    def _text_to_audio_embedding(self, description: str) -> np.ndarray:
        """Convert audio description to embedding"""
        embedding = np.random.randn(self.embedding_dim) * 0.1
        
        # Add structure based on audio characteristics
        audio_words = ['loud', 'quiet', 'high', 'low', 'music', 'voice', 'noise', 'melody']
        for i, word in enumerate(audio_words):
            if word in description.lower():
                embedding[i * 64:(i + 1) * 64] += 0.5
        
        return embedding / np.linalg.norm(embedding)
    
    def compute_similarity(self, feature1: ModalityFeature, feature2: ModalityFeature) -> float:
        """Compute similarity between audio features"""
        if feature1.modality != ModalityType.HEARING or feature2.modality != ModalityType.HEARING:
            return 0.0
        
        cosine_sim = np.dot(feature1.feature_vector, feature2.feature_vector) / (
            np.linalg.norm(feature1.feature_vector) * np.linalg.norm(feature2.feature_vector)
        )
        
        confidence_weight = (feature1.confidence + feature2.confidence) / 2
        
        return float(cosine_sim * confidence_weight)
    
    def get_modality_type(self) -> ModalityType:
        """Get the modality type"""
        return ModalityType.HEARING


class LanguageProcessor(ModalityProcessor):
    """Processor for language modality"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        
    def extract_features(self, input_data: Any, context: Dict[str, Any] = None) -> List[ModalityFeature]:
        """Extract language features from input data"""
        features = []
        
        if isinstance(input_data, str):
            # Simulate BERT-like embedding
            text_embedding = self._text_to_embedding(input_data)
            
            lang_feature = ModalityFeature(
                feature_id=f"lang_{hash(input_data)}",
                modality=ModalityType.LANGUAGE,
                embedding_type=ModalityEmbeddingType.TEXT_BERT,
                feature_vector=text_embedding,
                confidence=0.9,
                metadata={'text': input_data, 'length': len(input_data)}
            )
            features.append(lang_feature)
        
        elif isinstance(input_data, list):
            # List of words/tokens
            for i, token in enumerate(input_data):
                token_embedding = self._text_to_embedding(str(token))
                
                token_feature = ModalityFeature(
                    feature_id=f"token_{i}_{hash(str(token))}",
                    modality=ModalityType.LANGUAGE,
                    embedding_type=ModalityEmbeddingType.TEXT_WORD2VEC,
                    feature_vector=token_embedding,
                    confidence=0.8,
                    metadata={'token': str(token), 'position': i}
                )
                features.append(token_feature)
        
        return features
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding"""
        # Simple word-based embedding
        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)
        
        # Create embedding based on word characteristics
        for i, word in enumerate(words):
            word_hash = hash(word) % self.embedding_dim
            embedding[word_hash] += 1.0 / (i + 1)  # Position-weighted
        
        # Add some semantic structure
        semantic_words = {
            'positive': np.array([1.0] * 64 + [0.0] * (self.embedding_dim - 64)),
            'negative': np.array([0.0] * 64 + [-1.0] * 64 + [0.0] * (self.embedding_dim - 128)),
            'action': np.array([0.0] * 128 + [1.0] * 64 + [0.0] * (self.embedding_dim - 192)),
            'object': np.array([0.0] * 192 + [1.0] * 64 + [0.0] * (self.embedding_dim - 256))
        }
        
        for category, vector in semantic_words.items():
            if any(word in text.lower() for word in [category]):
                embedding += vector * 0.5
        
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    
    def compute_similarity(self, feature1: ModalityFeature, feature2: ModalityFeature) -> float:
        """Compute similarity between language features"""
        if feature1.modality != ModalityType.LANGUAGE or feature2.modality != ModalityType.LANGUAGE:
            return 0.0
        
        cosine_sim = np.dot(feature1.feature_vector, feature2.feature_vector) / (
            np.linalg.norm(feature1.feature_vector) * np.linalg.norm(feature2.feature_vector)
        )
        
        confidence_weight = (feature1.confidence + feature2.confidence) / 2
        
        return float(cosine_sim * confidence_weight)
    
    def get_modality_type(self) -> ModalityType:
        """Get the modality type"""
        return ModalityType.LANGUAGE


class CrossModalCorrespondenceFinder:
    """Finds correspondences between features across modalities"""
    
    def __init__(self, temporal_threshold: float = 1.0, spatial_threshold: float = 0.8):
        self.temporal_threshold = temporal_threshold  # Maximum time difference for correspondence
        self.spatial_threshold = spatial_threshold    # Minimum spatial overlap for correspondence
        self.learned_correspondences: List[CrossModalCorrespondence] = []
        
    def find_correspondences(self, features_by_modality: Dict[ModalityType, List[ModalityFeature]],
                           context: Dict[str, Any] = None) -> List[CrossModalCorrespondence]:
        """Find correspondences between features across modalities"""
        correspondences = []
        
        modalities = list(features_by_modality.keys())
        
        # Pairwise correspondence detection
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                features1 = features_by_modality[mod1]
                features2 = features_by_modality[mod2]
                
                pair_correspondences = self._find_pairwise_correspondences(
                    mod1, features1, mod2, features2, context
                )
                correspondences.extend(pair_correspondences)
        
        # Multi-modal correspondence detection (3+ modalities)
        if len(modalities) >= 3:
            multi_correspondences = self._find_multimodal_correspondences(
                features_by_modality, context
            )
            correspondences.extend(multi_correspondences)
        
        return correspondences
    
    def _find_pairwise_correspondences(self, mod1: ModalityType, features1: List[ModalityFeature],
                                     mod2: ModalityType, features2: List[ModalityFeature],
                                     context: Dict[str, Any] = None) -> List[CrossModalCorrespondence]:
        """Find correspondences between two modalities"""
        correspondences = []
        
        for feat1 in features1:
            for feat2 in features2:
                # Check temporal alignment
                temporal_alignment = self._compute_temporal_alignment(feat1, feat2)
                
                # Check spatial alignment
                spatial_alignment = self._compute_spatial_alignment(feat1, feat2)
                
                # Check semantic similarity
                semantic_similarity = self._compute_cross_modal_similarity(feat1, feat2)
                
                # Overall correspondence strength
                correspondence_strength = (
                    0.3 * temporal_alignment +
                    0.3 * spatial_alignment +
                    0.4 * semantic_similarity
                )
                
                if correspondence_strength > 0.5:  # Threshold for valid correspondence
                    correspondence = CrossModalCorrespondence(
                        correspondence_id=f"corr_{feat1.feature_id}_{feat2.feature_id}",
                        modality_features={mod1: feat1, mod2: feat2},
                        correspondence_strength=correspondence_strength,
                        temporal_alignment=temporal_alignment,
                        spatial_alignment=spatial_alignment,
                        semantic_similarity=semantic_similarity,
                        evidence_sources=[f"pairwise_{mod1.value}_{mod2.value}"]
                    )
                    correspondences.append(correspondence)
        
        return correspondences
    
    def _find_multimodal_correspondences(self, features_by_modality: Dict[ModalityType, List[ModalityFeature]],
                                       context: Dict[str, Any] = None) -> List[CrossModalCorrespondence]:
        """Find correspondences involving 3+ modalities"""
        correspondences = []
        
        # For now, simple approach: extend pairwise correspondences to include third modality
        # In practice, would use more sophisticated multi-modal alignment algorithms
        
        modalities = list(features_by_modality.keys())
        
        if len(modalities) >= 3:
            # Find features that are temporally and spatially co-located across all modalities
            for mod1 in modalities:
                for feat1 in features_by_modality[mod1]:
                    aligned_features = {mod1: feat1}
                    
                    for mod2 in modalities:
                        if mod2 == mod1:
                            continue
                        
                        best_alignment = 0.0
                        best_feature = None
                        
                        for feat2 in features_by_modality[mod2]:
                            temporal_align = self._compute_temporal_alignment(feat1, feat2)
                            spatial_align = self._compute_spatial_alignment(feat1, feat2)
                            
                            alignment = (temporal_align + spatial_align) / 2
                            if alignment > best_alignment and alignment > 0.6:
                                best_alignment = alignment
                                best_feature = feat2
                        
                        if best_feature:
                            aligned_features[mod2] = best_feature
                    
                    # If we have features from 3+ modalities that are aligned
                    if len(aligned_features) >= 3:
                        correspondence_strength = np.mean([
                            self._compute_cross_modal_similarity(feat, aligned_features[mod1])
                            for mod, feat in aligned_features.items() if mod != mod1
                        ])
                        
                        if correspondence_strength > 0.4:
                            correspondence = CrossModalCorrespondence(
                                correspondence_id=f"multi_corr_{hash(str(aligned_features.keys()))}",
                                modality_features=aligned_features,
                                correspondence_strength=correspondence_strength,
                                temporal_alignment=np.mean([
                                    self._compute_temporal_alignment(feat, aligned_features[mod1])
                                    for mod, feat in aligned_features.items() if mod != mod1
                                ]),
                                spatial_alignment=np.mean([
                                    self._compute_spatial_alignment(feat, aligned_features[mod1])
                                    for mod, feat in aligned_features.items() if mod != mod1
                                ]),
                                semantic_similarity=correspondence_strength,
                                evidence_sources=[f"multimodal_{len(aligned_features)}_way"]
                            )
                            correspondences.append(correspondence)
        
        return correspondences
    
    def _compute_temporal_alignment(self, feat1: ModalityFeature, feat2: ModalityFeature) -> float:
        """Compute temporal alignment between two features"""
        if feat1.temporal_window is None or feat2.temporal_window is None:
            return 1.0  # Assume aligned if no temporal info
        
        # Calculate overlap between temporal windows
        start1, end1 = feat1.temporal_window
        start2, end2 = feat2.temporal_window
        
        # Find intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start >= intersection_end:
            return 0.0  # No overlap
        
        intersection_duration = intersection_end - intersection_start
        union_duration = max(end1, end2) - min(start1, start2)
        
        return intersection_duration / union_duration if union_duration > 0 else 0.0
    
    def _compute_spatial_alignment(self, feat1: ModalityFeature, feat2: ModalityFeature) -> float:
        """Compute spatial alignment between two features"""
        if feat1.spatial_region is None or feat2.spatial_region is None:
            return 1.0  # Assume aligned if no spatial info
        
        # Simple bounding box overlap calculation
        region1 = feat1.spatial_region
        region2 = feat2.spatial_region
        
        # Assume regions have 'x', 'y', 'width', 'height' fields
        if all(key in region1 for key in ['x', 'y', 'width', 'height']) and \
           all(key in region2 for key in ['x', 'y', 'width', 'height']):
            
            # Calculate intersection
            x1_left, x1_right = region1['x'], region1['x'] + region1['width']
            y1_top, y1_bottom = region1['y'], region1['y'] + region1['height']
            
            x2_left, x2_right = region2['x'], region2['x'] + region2['width']
            y2_top, y2_bottom = region2['y'], region2['y'] + region2['height']
            
            intersection_x = max(0, min(x1_right, x2_right) - max(x1_left, x2_left))
            intersection_y = max(0, min(y1_bottom, y2_bottom) - max(y1_top, y2_top))
            
            intersection_area = intersection_x * intersection_y
            
            area1 = region1['width'] * region1['height']
            area2 = region2['width'] * region2['height']
            union_area = area1 + area2 - intersection_area
            
            return intersection_area / union_area if union_area > 0 else 0.0
        
        return 0.5  # Default moderate alignment if format unknown
    
    def _compute_cross_modal_similarity(self, feat1: ModalityFeature, feat2: ModalityFeature) -> float:
        """Compute semantic similarity between features from different modalities"""
        # Project features to a common semantic space
        common_feat1 = self._project_to_common_space(feat1)
        common_feat2 = self._project_to_common_space(feat2)
        
        # Compute cosine similarity in common space
        cosine_sim = np.dot(common_feat1, common_feat2) / (
            np.linalg.norm(common_feat1) * np.linalg.norm(common_feat2)
        )
        
        # Weight by confidence
        confidence_weight = (feat1.confidence + feat2.confidence) / 2
        
        return float(cosine_sim * confidence_weight)
    
    def _project_to_common_space(self, feature: ModalityFeature) -> np.ndarray:
        """Project feature to common semantic space"""
        # Simple projection - in practice would use learned cross-modal embeddings
        
        if feature.modality == ModalityType.VISION:
            # Visual features get spatial emphasis
            projection = feature.feature_vector.copy()
            projection[:128] *= 1.2  # Emphasize spatial components
            
        elif feature.modality == ModalityType.HEARING:
            # Audio features get temporal emphasis
            projection = feature.feature_vector.copy()
            projection[128:256] *= 1.2  # Emphasize temporal components
            
        elif feature.modality == ModalityType.LANGUAGE:
            # Language features get semantic emphasis
            projection = feature.feature_vector.copy()
            projection[256:384] *= 1.2  # Emphasize semantic components
            
        else:
            projection = feature.feature_vector.copy()
        
        return projection / (np.linalg.norm(projection) + 1e-8)


class MultiModalKnowledgeGraph:
    """Knowledge graph that integrates information across modalities"""
    
    def __init__(self, base_kg: CrossDomainKnowledgeGraph):
        self.base_kg = base_kg
        self.modality_processors: Dict[ModalityType, ModalityProcessor] = {}
        self.correspondence_finder = CrossModalCorrespondenceFinder()
        self.multimodal_entities: Dict[str, MultiModalEntity] = {}
        self.modality_features: Dict[ModalityType, List[ModalityFeature]] = defaultdict(list)
        
        # Initialize default processors
        self._initialize_processors()
    
    def _initialize_processors(self):
        """Initialize default modality processors"""
        self.modality_processors[ModalityType.VISION] = VisualProcessor()
        self.modality_processors[ModalityType.HEARING] = AudioProcessor()
        self.modality_processors[ModalityType.LANGUAGE] = LanguageProcessor()
    
    def add_modality_processor(self, processor: ModalityProcessor):
        """Add a custom modality processor"""
        self.modality_processors[processor.get_modality_type()] = processor
    
    def process_multimodal_input(self, inputs: Dict[ModalityType, Any],
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process multi-modal input and integrate into knowledge graph"""
        results = {
            'extracted_features': {},
            'correspondences': [],
            'created_entities': [],
            'updated_entities': [],
            'integration_success': True
        }
        
        # Extract features from each modality
        for modality, input_data in inputs.items():
            if modality in self.modality_processors:
                processor = self.modality_processors[modality]
                features = processor.extract_features(input_data, context)
                
                self.modality_features[modality].extend(features)
                results['extracted_features'][modality] = features
        
        # Find cross-modal correspondences
        correspondences = self.correspondence_finder.find_correspondences(
            results['extracted_features'], context
        )
        results['correspondences'] = correspondences
        
        # Create or update multi-modal entities
        for correspondence in correspondences:
            entity = self._create_or_update_entity(correspondence)
            if entity:
                if entity.entity_id in self.multimodal_entities:
                    results['updated_entities'].append(entity.entity_id)
                else:
                    results['created_entities'].append(entity.entity_id)
                
                self.multimodal_entities[entity.entity_id] = entity
        
        # Integrate entities into base knowledge graph
        self._integrate_entities_to_kg()
        
        return results
    
    def _create_or_update_entity(self, correspondence: CrossModalCorrespondence) -> Optional[MultiModalEntity]:
        """Create or update a multi-modal entity from correspondence"""
        # Generate entity ID based on correspondence
        entity_id = f"entity_{correspondence.correspondence_id}"
        
        # Check if entity already exists
        if entity_id in self.multimodal_entities:
            entity = self.multimodal_entities[entity_id]
            
            # Update existing entity
            for modality, feature in correspondence.modality_features.items():
                if modality not in entity.modality_manifestations:
                    entity.modality_manifestations[modality] = []
                
                entity.modality_manifestations[modality].append(feature)
                entity.confidence_scores[modality] = max(
                    entity.confidence_scores.get(modality, 0), feature.confidence
                )
            
            entity.cross_modal_correspondences.append(correspondence)
            
        else:
            # Create new entity
            entity = MultiModalEntity(
                entity_id=entity_id,
                entity_type="multimodal_object",  # Could be inferred from features
                modality_manifestations={
                    modality: [feature] for modality, feature in correspondence.modality_features.items()
                },
                cross_modal_correspondences=[correspondence],
                confidence_scores={
                    modality: feature.confidence for modality, feature in correspondence.modality_features.items()
                }
            )
            
            # Compute temporal span
            temporal_windows = [
                feature.temporal_window for feature in correspondence.modality_features.values()
                if feature.temporal_window is not None
            ]
            
            if temporal_windows:
                min_start = min(window[0] for window in temporal_windows)
                max_end = max(window[1] for window in temporal_windows)
                entity.temporal_span = (min_start, max_end)
            
            # Compute spatial extent
            spatial_regions = [
                feature.spatial_region for feature in correspondence.modality_features.values()
                if feature.spatial_region is not None
            ]
            
            if spatial_regions:
                # Simple bounding box union
                entity.spatial_extent = self._compute_spatial_union(spatial_regions)
        
        # Update entity embeddings
        self._update_entity_embeddings(entity)
        
        return entity
    
    def _compute_spatial_union(self, spatial_regions: List[Dict[str, float]]) -> Dict[str, Any]:
        """Compute union of spatial regions"""
        if not spatial_regions:
            return {}
        
        # Assume regions have 'x', 'y', 'width', 'height' format
        min_x = min(region.get('x', 0) for region in spatial_regions)
        min_y = min(region.get('y', 0) for region in spatial_regions)
        max_x = max(region.get('x', 0) + region.get('width', 0) for region in spatial_regions)
        max_y = max(region.get('y', 0) + region.get('height', 0) for region in spatial_regions)
        
        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def _update_entity_embeddings(self, entity: MultiModalEntity):
        """Update embeddings for a multi-modal entity"""
        # Create unified embedding from all modalities
        embeddings = []
        weights = []
        
        for modality, features in entity.modality_manifestations.items():
            if features:
                # Average features from this modality
                modality_embedding = np.mean([feat.feature_vector for feat in features], axis=0)
                embeddings.append(modality_embedding)
                
                # Weight by confidence
                avg_confidence = np.mean([feat.confidence for feat in features])
                weights.append(avg_confidence)
        
        if embeddings:
            # Weighted average of modality embeddings
            weights = np.array(weights)
            weights /= weights.sum()
            
            unified_embedding = np.average(embeddings, axis=0, weights=weights)
            entity.entity_embeddings[ModalityEmbeddingType.UNIFIED_REPRESENTATION] = unified_embedding
            
            # Store individual modality embeddings
            for modality, features in entity.modality_manifestations.items():
                if features:
                    modality_embedding = np.mean([feat.feature_vector for feat in features], axis=0)
                    
                    # Map modality to embedding type
                    if modality == ModalityType.VISION:
                        embedding_type = ModalityEmbeddingType.VISUAL_CNN
                    elif modality == ModalityType.HEARING:
                        embedding_type = ModalityEmbeddingType.AUDIO_MFCC
                    elif modality == ModalityType.LANGUAGE:
                        embedding_type = ModalityEmbeddingType.TEXT_BERT
                    else:
                        continue
                    
                    entity.entity_embeddings[embedding_type] = modality_embedding
    
    def _integrate_entities_to_kg(self):
        """Integrate multi-modal entities into the base knowledge graph"""
        for entity in self.multimodal_entities.values():
            # Create knowledge graph node for entity
            kg_node = KnowledgeGraphNode(
                node_id=entity.entity_id,
                node_type=entity.entity_type,
                domain=DomainType.ABSTRACT,  # Multi-modal entities are abstract
                attributes={
                    'modalities': [mod.value for mod in entity.modality_manifestations.keys()],
                    'confidence_scores': entity.confidence_scores,
                    'temporal_span': entity.temporal_span,
                    'spatial_extent': entity.spatial_extent
                },
                embeddings={
                    emb_type.value: embedding for emb_type, embedding in entity.entity_embeddings.items()
                }
            )
            
            self.base_kg.add_node(kg_node)
            
            # Create edges between modality manifestations
            modalities = list(entity.modality_manifestations.keys())
            for i, mod1 in enumerate(modalities):
                for mod2 in modalities[i+1:]:
                    # Find correspondence strength between these modalities
                    correspondence_strength = 0.0
                    for correspondence in entity.cross_modal_correspondences:
                        if mod1 in correspondence.modality_features and mod2 in correspondence.modality_features:
                            correspondence_strength = max(correspondence_strength, 
                                                        correspondence.correspondence_strength)
                    
                    if correspondence_strength > 0:
                        edge = KnowledgeGraphEdge(
                            edge_id=f"crossmodal_{entity.entity_id}_{mod1.value}_{mod2.value}",
                            source_node=entity.entity_id,
                            target_node=entity.entity_id,  # Self-edge representing internal cross-modal binding
                            relation_type="cross_modal_correspondence",
                            strength=correspondence_strength,
                            properties={
                                'source_modality': mod1.value,
                                'target_modality': mod2.value,
                                'entity_id': entity.entity_id
                            }
                        )
                        
                        self.base_kg.add_edge(edge)
    
    def query_by_modality(self, modality: ModalityType, query_features: np.ndarray,
                         similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Query entities by similarity in a specific modality"""
        results = []
        
        for entity in self.multimodal_entities.values():
            if modality in entity.modality_manifestations:
                # Compute similarity
                entity_features = [feat.feature_vector for feat in entity.modality_manifestations[modality]]
                
                if entity_features:
                    avg_entity_features = np.mean(entity_features, axis=0)
                    
                    # Cosine similarity
                    similarity = np.dot(query_features, avg_entity_features) / (
                        np.linalg.norm(query_features) * np.linalg.norm(avg_entity_features)
                    )
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            'entity_id': entity.entity_id,
                            'similarity': float(similarity),
                            'modalities': list(entity.modality_manifestations.keys()),
                            'confidence': entity.confidence_scores.get(modality, 0.0)
                        })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def query_cross_modal(self, query_modality: ModalityType, query_features: np.ndarray,
                         target_modality: ModalityType, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Query using one modality and retrieve information from another modality"""
        results = []
        
        # First find entities similar in query modality
        query_results = self.query_by_modality(query_modality, query_features, similarity_threshold)
        
        # Then extract information from target modality
        for result in query_results:
            entity = self.multimodal_entities[result['entity_id']]
            
            if target_modality in entity.modality_manifestations:
                target_features = entity.modality_manifestations[target_modality]
                
                cross_modal_result = {
                    'entity_id': entity.entity_id,
                    'query_similarity': result['similarity'],
                    'target_modality': target_modality.value,
                    'target_features': [
                        {
                            'feature_id': feat.feature_id,
                            'confidence': feat.confidence,
                            'metadata': feat.metadata
                        } for feat in target_features
                    ],
                    'cross_modal_strength': 0.0
                }
                
                # Calculate cross-modal strength
                for correspondence in entity.cross_modal_correspondences:
                    if (query_modality in correspondence.modality_features and 
                        target_modality in correspondence.modality_features):
                        cross_modal_result['cross_modal_strength'] = max(
                            cross_modal_result['cross_modal_strength'],
                            correspondence.correspondence_strength
                        )
                
                results.append(cross_modal_result)
        
        return results
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about multi-modal integration"""
        stats = {
            'total_entities': len(self.multimodal_entities),
            'modality_coverage': defaultdict(int),
            'cross_modal_correspondences': 0,
            'average_confidence': defaultdict(list),
            'temporal_entities': 0,
            'spatial_entities': 0
        }
        
        for entity in self.multimodal_entities.values():
            # Count modalities per entity
            for modality in entity.modality_manifestations.keys():
                stats['modality_coverage'][modality.value] += 1
                
                if modality in entity.confidence_scores:
                    stats['average_confidence'][modality.value].append(entity.confidence_scores[modality])
            
            # Count correspondences
            stats['cross_modal_correspondences'] += len(entity.cross_modal_correspondences)
            
            # Count temporal and spatial entities
            if entity.temporal_span is not None:
                stats['temporal_entities'] += 1
            
            if entity.spatial_extent is not None:
                stats['spatial_entities'] += 1
        
        # Calculate average confidences
        for modality, confidences in stats['average_confidence'].items():
            stats['average_confidence'][modality] = np.mean(confidences) if confidences else 0.0
        
        stats['modality_coverage'] = dict(stats['modality_coverage'])
        stats['average_confidence'] = dict(stats['average_confidence'])
        
        return stats