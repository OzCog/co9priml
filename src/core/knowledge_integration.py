"""
Knowledge Integration Module for Relevance-Based Prioritization

This module implements knowledge integration mechanisms that prioritize relevant
information based on contextual importance and cognitive relevance assessment.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from .relevance_optimization import TimeScale, MultiScaleRelevanceAssessment


class KnowledgeType(Enum):
    """Types of knowledge for integration"""
    DECLARATIVE = "declarative"      # Facts and information
    PROCEDURAL = "procedural"        # Skills and procedures
    CONDITIONAL = "conditional"      # When and where to apply knowledge
    METACOGNITIVE = "metacognitive"  # Knowledge about knowledge


@dataclass
class KnowledgeItem:
    """Represents an item of knowledge with relevance properties"""
    content: Any
    knowledge_type: KnowledgeType
    relevance_score: float = 0.0
    confidence: float = 0.0
    connections: Set[str] = field(default_factory=set)
    temporal_relevance: Dict[TimeScale, float] = field(default_factory=dict)
    creation_time: float = field(default_factory=lambda: np.datetime64('now').astype(float))
    last_accessed: float = field(default_factory=lambda: np.datetime64('now').astype(float))
    access_count: int = 0


@dataclass
class KnowledgeIntegrationMetrics:
    """Metrics for tracking knowledge integration performance"""
    integration_efficiency: float = 0.0
    relevance_accuracy: float = 0.0
    knowledge_coherence: float = 0.0
    temporal_consistency: float = 0.0
    connection_quality: float = 0.0


class KnowledgeIntegrator:
    """
    Advanced knowledge integration system that prioritizes relevant information
    and maintains coherent knowledge organization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Knowledge storage
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.knowledge_connections: Dict[str, Set[str]] = {}
        
        # Relevance-based prioritization
        self.priority_weights = {
            KnowledgeType.DECLARATIVE: 0.25,
            KnowledgeType.PROCEDURAL: 0.30,
            KnowledgeType.CONDITIONAL: 0.25,
            KnowledgeType.METACOGNITIVE: 0.20
        }
        
        # Integration parameters
        self.relevance_threshold = self.config.get('relevance_threshold', 0.4)
        self.max_integration_items = self.config.get('max_integration_items', 20)
        self.connection_strength_threshold = self.config.get('connection_threshold', 0.3)
        
        # Performance tracking
        self.integration_history: List[KnowledgeIntegrationMetrics] = []
        
    def integrate_knowledge(self, new_knowledge: Dict[str, Any], 
                          context: Dict[str, Any],
                          relevance_assessments: Dict[str, MultiScaleRelevanceAssessment] = None) -> Dict[str, Any]:
        """
        Integrate new knowledge based on relevance prioritization.
        
        Args:
            new_knowledge: Dictionary of new knowledge items
            context: Current context for integration
            relevance_assessments: Optional pre-computed relevance assessments
            
        Returns:
            Integration results with prioritized knowledge
        """
        integration_results = {
            'integrated_items': [],
            'prioritized_connections': [],
            'relevance_updates': {},
            'integration_metrics': None
        }
        
        # Process each new knowledge item
        for item_id, item_data in new_knowledge.items():
            # Determine knowledge type
            knowledge_type = self._classify_knowledge_type(item_data)
            
            # Get relevance assessment
            if relevance_assessments and item_id in relevance_assessments:
                relevance_assessment = relevance_assessments[item_id]
                relevance_score = relevance_assessment.combined_score
                temporal_relevance = {
                    TimeScale.IMMEDIATE: relevance_assessment.immediate_relevance,
                    TimeScale.SHORT_TERM: relevance_assessment.short_term_relevance,
                    TimeScale.MEDIUM_TERM: relevance_assessment.medium_term_relevance,
                    TimeScale.LONG_TERM: relevance_assessment.long_term_relevance
                }
            else:
                # Compute basic relevance if not provided
                relevance_score = self._compute_basic_relevance(item_data, context)
                temporal_relevance = self._compute_temporal_relevance(item_data, context)
            
            # Only integrate if above relevance threshold
            if relevance_score >= self.relevance_threshold:
                # Create knowledge item
                knowledge_item = KnowledgeItem(
                    content=item_data,
                    knowledge_type=knowledge_type,
                    relevance_score=relevance_score,
                    confidence=context.get('confidence', 0.8),
                    temporal_relevance=temporal_relevance
                )
                
                # Find connections to existing knowledge
                connections = self._find_knowledge_connections(item_id, item_data)
                knowledge_item.connections = connections
                
                # Add to knowledge base
                self.knowledge_base[item_id] = knowledge_item
                self.knowledge_connections[item_id] = connections
                
                integration_results['integrated_items'].append({
                    'item_id': item_id,
                    'relevance_score': relevance_score,
                    'knowledge_type': knowledge_type.value,
                    'connections': list(connections)
                })
        
        # Prioritize connections based on relevance
        prioritized_connections = self._prioritize_connections(context)
        integration_results['prioritized_connections'] = prioritized_connections
        
        # Update relevance scores for existing knowledge
        relevance_updates = self._update_existing_relevance(context)
        integration_results['relevance_updates'] = relevance_updates
        
        # Compute integration metrics
        metrics = self._compute_integration_metrics()
        integration_results['integration_metrics'] = metrics
        self.integration_history.append(metrics)
        
        return integration_results
    
    def retrieve_relevant_knowledge(self, query: str, context: Dict[str, Any],
                                   max_items: int = 10) -> List[Tuple[str, KnowledgeItem, float]]:
        """
        Retrieve knowledge items most relevant to query and context.
        
        Args:
            query: Query for knowledge retrieval
            context: Current context
            max_items: Maximum number of items to return
            
        Returns:
            List of (item_id, knowledge_item, relevance_score) tuples
        """
        relevant_items = []
        
        for item_id, knowledge_item in self.knowledge_base.items():
            # Compute relevance to query
            query_relevance = self._compute_query_relevance(query, knowledge_item.content)
            
            # Compute contextual relevance
            context_relevance = self._compute_context_relevance(knowledge_item, context)
            
            # Combine with stored relevance
            combined_relevance = (
                0.4 * query_relevance +
                0.3 * context_relevance +
                0.3 * knowledge_item.relevance_score
            )
            
            # Update access statistics
            knowledge_item.last_accessed = np.datetime64('now').astype(float)
            knowledge_item.access_count += 1
            
            relevant_items.append((item_id, knowledge_item, combined_relevance))
        
        # Sort by relevance and return top items
        relevant_items.sort(key=lambda x: x[2], reverse=True)
        return relevant_items[:max_items]
    
    def update_knowledge_relevance(self, feedback: Dict[str, float]) -> Dict[str, float]:
        """
        Update knowledge relevance based on feedback.
        
        Args:
            feedback: Dictionary of item_id -> feedback_score
            
        Returns:
            Dictionary of updated relevance scores
        """
        updated_scores = {}
        learning_rate = 0.1
        
        for item_id, feedback_score in feedback.items():
            if item_id in self.knowledge_base:
                knowledge_item = self.knowledge_base[item_id]
                
                # Update relevance score with feedback
                old_score = knowledge_item.relevance_score
                error = feedback_score - old_score
                new_score = old_score + learning_rate * error
                
                knowledge_item.relevance_score = np.clip(new_score, 0.0, 1.0)
                updated_scores[item_id] = knowledge_item.relevance_score
        
        return updated_scores
    
    def _classify_knowledge_type(self, item_data: Any) -> KnowledgeType:
        """Classify the type of knowledge item"""
        item_str = str(item_data).lower()
        
        # Simple heuristic classification
        if any(word in item_str for word in ['how', 'procedure', 'method', 'algorithm']):
            return KnowledgeType.PROCEDURAL
        elif any(word in item_str for word in ['when', 'where', 'if', 'condition']):
            return KnowledgeType.CONDITIONAL
        elif any(word in item_str for word in ['meta', 'learning', 'strategy', 'thinking']):
            return KnowledgeType.METACOGNITIVE
        else:
            return KnowledgeType.DECLARATIVE
    
    def _compute_basic_relevance(self, item_data: Any, context: Dict[str, Any]) -> float:
        """Compute basic relevance score for knowledge item"""
        relevance = 0.0
        item_str = str(item_data).lower()
        
        # Context keyword matching
        for key, value in context.items():
            if isinstance(value, str) and value.lower() in item_str:
                relevance += 0.2
            elif key.lower() in item_str:
                relevance += 0.1
        
        # Task type relevance
        task_type = context.get('task_type', '')
        if task_type and task_type.lower() in item_str:
            relevance += 0.3
        
        return min(1.0, relevance)
    
    def _compute_temporal_relevance(self, item_data: Any, context: Dict[str, Any]) -> Dict[TimeScale, float]:
        """Compute temporal relevance across different time scales"""
        temporal_relevance = {}
        
        # Simple heuristic based on content
        item_str = str(item_data).lower()
        
        # Immediate relevance
        immediate_keywords = ['urgent', 'now', 'immediate', 'current']
        immediate_score = sum(0.2 for word in immediate_keywords if word in item_str)
        temporal_relevance[TimeScale.IMMEDIATE] = min(1.0, immediate_score + 0.3)
        
        # Short-term relevance
        short_term_keywords = ['soon', 'next', 'upcoming', 'near']
        short_term_score = sum(0.15 for word in short_term_keywords if word in item_str)
        temporal_relevance[TimeScale.SHORT_TERM] = min(1.0, short_term_score + 0.4)
        
        # Medium-term relevance
        medium_term_keywords = ['plan', 'goal', 'objective', 'target']
        medium_term_score = sum(0.1 for word in medium_term_keywords if word in item_str)
        temporal_relevance[TimeScale.MEDIUM_TERM] = min(1.0, medium_term_score + 0.5)
        
        # Long-term relevance
        long_term_keywords = ['strategy', 'vision', 'future', 'learning']
        long_term_score = sum(0.1 for word in long_term_keywords if word in item_str)
        temporal_relevance[TimeScale.LONG_TERM] = min(1.0, long_term_score + 0.4)
        
        return temporal_relevance
    
    def _find_knowledge_connections(self, item_id: str, item_data: Any) -> Set[str]:
        """Find connections between new knowledge and existing knowledge base"""
        connections = set()
        item_str = str(item_data).lower()
        item_words = set(item_str.split())
        
        for existing_id, existing_item in self.knowledge_base.items():
            existing_str = str(existing_item.content).lower()
            existing_words = set(existing_str.split())
            
            # Word overlap
            overlap = len(item_words & existing_words)
            if overlap > 2:  # Threshold for connection
                connections.add(existing_id)
            
            # Semantic similarity (simplified)
            if self._compute_semantic_similarity(item_str, existing_str) > self.connection_strength_threshold:
                connections.add(existing_id)
        
        return connections
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts (simplified)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _prioritize_connections(self, context: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """Prioritize knowledge connections based on relevance"""
        prioritized = []
        
        for item_id, connections in self.knowledge_connections.items():
            if item_id in self.knowledge_base:
                item_relevance = self.knowledge_base[item_id].relevance_score
                
                for connected_id in connections:
                    if connected_id in self.knowledge_base:
                        connected_relevance = self.knowledge_base[connected_id].relevance_score
                        
                        # Connection strength based on mutual relevance
                        connection_strength = (item_relevance + connected_relevance) / 2
                        
                        prioritized.append((item_id, connected_id, connection_strength))
        
        # Sort by connection strength
        prioritized.sort(key=lambda x: x[2], reverse=True)
        return prioritized[:self.max_integration_items]
    
    def _update_existing_relevance(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Update relevance scores for existing knowledge based on new context"""
        updates = {}
        
        for item_id, knowledge_item in self.knowledge_base.items():
            # Recompute relevance in new context
            new_relevance = self._compute_basic_relevance(knowledge_item.content, context)
            
            # Update with decay for older items
            current_time = np.datetime64('now').astype(float)
            age = current_time - knowledge_item.creation_time
            decay_factor = np.exp(-age / (24 * 3600))  # Daily decay
            
            updated_relevance = 0.7 * knowledge_item.relevance_score + 0.3 * new_relevance
            updated_relevance *= decay_factor
            
            if abs(updated_relevance - knowledge_item.relevance_score) > 0.1:
                knowledge_item.relevance_score = updated_relevance
                updates[item_id] = updated_relevance
        
        return updates
    
    def _compute_query_relevance(self, query: str, content: Any) -> float:
        """Compute relevance of knowledge item to query"""
        query_words = set(query.lower().split())
        content_words = set(str(content).lower().split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = len(query_words & content_words)
        return intersection / len(query_words)
    
    def _compute_context_relevance(self, knowledge_item: KnowledgeItem, context: Dict[str, Any]) -> float:
        """Compute contextual relevance of knowledge item"""
        # Use temporal relevance based on context urgency
        urgency = context.get('urgency', 0.5)
        
        if urgency > 0.7:
            return knowledge_item.temporal_relevance.get(TimeScale.IMMEDIATE, 0.5)
        elif urgency > 0.4:
            return knowledge_item.temporal_relevance.get(TimeScale.SHORT_TERM, 0.5)
        elif urgency > 0.2:
            return knowledge_item.temporal_relevance.get(TimeScale.MEDIUM_TERM, 0.5)
        else:
            return knowledge_item.temporal_relevance.get(TimeScale.LONG_TERM, 0.5)
    
    def _compute_integration_metrics(self) -> KnowledgeIntegrationMetrics:
        """Compute metrics for knowledge integration performance"""
        if not self.knowledge_base:
            return KnowledgeIntegrationMetrics()
        
        # Integration efficiency (how well items are connected)
        total_connections = sum(len(connections) for connections in self.knowledge_connections.values())
        total_items = len(self.knowledge_base)
        integration_efficiency = min(1.0, total_connections / (total_items * 2))
        
        # Relevance accuracy (average relevance of integrated items)
        relevance_scores = [item.relevance_score for item in self.knowledge_base.values()]
        relevance_accuracy = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Knowledge coherence (consistency of connections)
        coherence = self._compute_knowledge_coherence()
        
        # Temporal consistency (relevance consistency across time scales)
        temporal_consistency = self._compute_temporal_consistency()
        
        # Connection quality (strength of connections)
        connection_quality = self._compute_connection_quality()
        
        return KnowledgeIntegrationMetrics(
            integration_efficiency=integration_efficiency,
            relevance_accuracy=relevance_accuracy,
            knowledge_coherence=coherence,
            temporal_consistency=temporal_consistency,
            connection_quality=connection_quality
        )
    
    def _compute_knowledge_coherence(self) -> float:
        """Compute coherence of knowledge connections"""
        if not self.knowledge_connections:
            return 0.5
        
        coherence_scores = []
        for item_id, connections in self.knowledge_connections.items():
            if item_id in self.knowledge_base and connections:
                item_type = self.knowledge_base[item_id].knowledge_type
                
                # Check type consistency among connections
                type_matches = 0
                for connected_id in connections:
                    if connected_id in self.knowledge_base:
                        connected_type = self.knowledge_base[connected_id].knowledge_type
                        if item_type == connected_type:
                            type_matches += 1
                
                if connections:
                    coherence_score = type_matches / len(connections)
                    coherence_scores.append(coherence_score)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _compute_temporal_consistency(self) -> float:
        """Compute temporal consistency across time scales"""
        consistency_scores = []
        
        for knowledge_item in self.knowledge_base.values():
            temporal_values = list(knowledge_item.temporal_relevance.values())
            if len(temporal_values) > 1:
                # Measure variance - lower variance = higher consistency
                variance = np.var(temporal_values)
                consistency = 1.0 / (1.0 + variance)  # Convert to 0-1 scale
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _compute_connection_quality(self) -> float:
        """Compute quality of knowledge connections"""
        quality_scores = []
        
        for item_id, connections in self.knowledge_connections.items():
            if item_id in self.knowledge_base and connections:
                item_relevance = self.knowledge_base[item_id].relevance_score
                
                connection_relevances = []
                for connected_id in connections:
                    if connected_id in self.knowledge_base:
                        connected_relevance = self.knowledge_base[connected_id].relevance_score
                        connection_relevances.append(connected_relevance)
                
                if connection_relevances:
                    # Quality is how well connected items maintain high relevance
                    avg_connection_relevance = np.mean(connection_relevances)
                    quality = (item_relevance + avg_connection_relevance) / 2
                    quality_scores.append(quality)
        
        return np.mean(quality_scores) if quality_scores else 0.5