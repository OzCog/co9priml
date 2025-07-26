"""
Cross-Domain Reasoning and Inference Engine

This module implements reasoning and inference capabilities that operate
across multiple domains, enabling the system to leverage knowledge from
one domain to make inferences in another and create logical connections
across different cognitive domains.
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import copy
import logging

from .cross_domain_framework import (
    DomainType, ModalityType, ConceptMapping, AbstractConcept,
    UnifiedRepresentationSystem, CrossDomainIntegrationFramework
)


class ReasoningType(Enum):
    """Types of reasoning patterns"""
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    METAPHORICAL = "metaphorical"
    COMPOSITIONAL = "compositional"
    RELATIONAL = "relational"


class InferenceStrength(Enum):
    """Strength levels for inferences"""
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.8
    CERTAIN = 0.95


@dataclass
class CrossDomainInference:
    """Represents an inference made across domains"""
    inference_id: str
    reasoning_type: ReasoningType
    source_domain: DomainType
    target_domain: DomainType
    source_facts: List[str]
    inferred_fact: str
    inference_strength: float
    evidence_chain: List[Dict[str, Any]] = field(default_factory=list)
    context_dependencies: Dict[str, Any] = field(default_factory=dict)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class ReasoningPattern:
    """Represents a reusable reasoning pattern"""
    pattern_id: str
    pattern_type: ReasoningType
    source_template: Dict[str, str]  # Template for source domain facts
    target_template: Dict[str, str]  # Template for target domain inferences
    applicability_conditions: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    domain_pairs: List[Tuple[DomainType, DomainType]] = field(default_factory=list)


@dataclass
class KnowledgeGraphNode:
    """Node in the cross-domain knowledge graph"""
    node_id: str
    node_type: str  # concept, relation, entity, etc.
    domain: DomainType
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)  # Different embedding types
    uncertainty: float = 0.0
    temporal_validity: Optional[Tuple[float, float]] = None  # Valid time range


@dataclass
class KnowledgeGraphEdge:
    """Edge in the cross-domain knowledge graph"""
    edge_id: str
    source_node: str
    target_node: str
    relation_type: str
    strength: float
    domain_context: Optional[DomainType] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


class CrossDomainKnowledgeGraph:
    """Knowledge graph that spans multiple domains"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeGraphNode] = {}
        self.edges: Dict[str, KnowledgeGraphEdge] = {}
        self.domain_subgraphs: Dict[DomainType, Set[str]] = defaultdict(set)  # domain -> node_ids
        self.relation_types: Set[str] = set()
        self.cross_domain_bridges: List[KnowledgeGraphEdge] = []
        
    def add_node(self, node: KnowledgeGraphNode) -> bool:
        """Add a node to the knowledge graph"""
        if node.node_id in self.nodes:
            # Merge with existing node
            existing = self.nodes[node.node_id]
            existing.attributes.update(node.attributes)
            existing.embeddings.update(node.embeddings)
            return True
        
        self.nodes[node.node_id] = node
        self.domain_subgraphs[node.domain].add(node.node_id)
        return True
    
    def add_edge(self, edge: KnowledgeGraphEdge) -> bool:
        """Add an edge to the knowledge graph"""
        if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
            return False
        
        self.edges[edge.edge_id] = edge
        self.relation_types.add(edge.relation_type)
        
        # Check if this is a cross-domain bridge
        source_domain = self.nodes[edge.source_node].domain
        target_domain = self.nodes[edge.target_node].domain
        
        if source_domain != target_domain:
            self.cross_domain_bridges.append(edge)
        
        return True
    
    def find_path(self, source_node_id: str, target_node_id: str, 
                  max_depth: int = 5) -> List[List[str]]:
        """Find paths between nodes in the graph"""
        if source_node_id not in self.nodes or target_node_id not in self.nodes:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current_node: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current_node == target:
                paths.append(path + [current_node])
                return
            
            if current_node in visited:
                return
            
            visited.add(current_node)
            
            # Find outgoing edges
            for edge in self.edges.values():
                if edge.source_node == current_node:
                    dfs(edge.target_node, target, path + [current_node], depth + 1)
            
            visited.remove(current_node)
        
        dfs(source_node_id, target_node_id, [], 0)
        return paths
    
    def get_neighbors(self, node_id: str, relation_type: str = None) -> List[str]:
        """Get neighboring nodes"""
        neighbors = []
        
        for edge in self.edges.values():
            if edge.source_node == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append(edge.target_node)
            elif edge.target_node == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append(edge.source_node)
        
        return neighbors
    
    def get_cross_domain_connections(self, source_domain: DomainType, 
                                   target_domain: DomainType) -> List[KnowledgeGraphEdge]:
        """Get edges connecting two domains"""
        connections = []
        
        for edge in self.cross_domain_bridges:
            source_node_domain = self.nodes[edge.source_node].domain
            target_node_domain = self.nodes[edge.target_node].domain
            
            if ((source_node_domain == source_domain and target_node_domain == target_domain) or
                (source_node_domain == target_domain and target_node_domain == source_domain)):
                connections.append(edge)
        
        return connections
    
    def query_by_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """Query the graph using a pattern"""
        results = []
        
        # Simple pattern matching for now
        if 'node_type' in pattern:
            matching_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.node_type == pattern['node_type']
            ]
            
            for node_id in matching_nodes:
                result = {'node_id': node_id}
                if 'relation_type' in pattern:
                    neighbors = self.get_neighbors(node_id, pattern['relation_type'])
                    result['neighbors'] = neighbors
                results.append(result)
        
        return results


class AnalogicalReasoningEngine:
    """Engine for analogical reasoning across domains"""
    
    def __init__(self, knowledge_graph: CrossDomainKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.analogy_patterns: Dict[str, ReasoningPattern] = {}
        self.structural_mappings: Dict[Tuple[DomainType, DomainType], Dict[str, str]] = {}
        
    def find_analogies(self, source_domain: DomainType, source_concept: str,
                      target_domains: List[DomainType]) -> List[Dict[str, Any]]:
        """Find analogical mappings between concepts across domains"""
        analogies = []
        
        # Find source concept in knowledge graph
        source_nodes = [
            node_id for node_id, node in self.knowledge_graph.nodes.items()
            if node.domain == source_domain and source_concept.lower() in node.attributes.get('name', '').lower()
        ]
        
        if not source_nodes:
            return analogies
        
        source_node_id = source_nodes[0]  # Use first match
        source_node = self.knowledge_graph.nodes[source_node_id]
        
        # Get structural information about source concept
        source_structure = self._analyze_concept_structure(source_node_id)
        
        # Search for analogous structures in target domains
        for target_domain in target_domains:
            target_nodes = [
                node_id for node_id, node in self.knowledge_graph.nodes.items()
                if node.domain == target_domain
            ]
            
            for target_node_id in target_nodes:
                target_structure = self._analyze_concept_structure(target_node_id)
                
                # Calculate structural similarity
                similarity = self._calculate_structural_similarity(source_structure, target_structure)
                
                if similarity > 0.5:  # Threshold for analogical match
                    target_node = self.knowledge_graph.nodes[target_node_id]
                    analogy = {
                        'source_concept': source_concept,
                        'source_domain': source_domain.value,
                        'target_concept': target_node.attributes.get('name', target_node_id),
                        'target_domain': target_domain.value,
                        'similarity_score': similarity,
                        'structural_mapping': self._create_structural_mapping(
                            source_structure, target_structure
                        ),
                        'reasoning_type': ReasoningType.ANALOGICAL.value
                    }
                    analogies.append(analogy)
        
        # Sort by similarity score
        analogies.sort(key=lambda x: x['similarity_score'], reverse=True)
        return analogies
    
    def _analyze_concept_structure(self, node_id: str) -> Dict[str, Any]:
        """Analyze the structural properties of a concept"""
        node = self.knowledge_graph.nodes[node_id]
        
        structure = {
            'node_type': node.node_type,
            'attributes': node.attributes,
            'relations': defaultdict(list),
            'relation_counts': defaultdict(int),
            'neighbor_types': defaultdict(int),
            'depth_2_neighbors': set()
        }
        
        # Analyze immediate relations
        for edge in self.knowledge_graph.edges.values():
            if edge.source_node == node_id:
                structure['relations'][edge.relation_type].append(edge.target_node)
                structure['relation_counts'][edge.relation_type] += 1
                
                target_node = self.knowledge_graph.nodes[edge.target_node]
                structure['neighbor_types'][target_node.node_type] += 1
                
                # Get depth-2 neighbors
                second_level_neighbors = self.knowledge_graph.get_neighbors(edge.target_node)
                structure['depth_2_neighbors'].update(second_level_neighbors)
        
        structure['depth_2_neighbors'] = len(structure['depth_2_neighbors'])
        return structure
    
    def _calculate_structural_similarity(self, struct1: Dict[str, Any], 
                                       struct2: Dict[str, Any]) -> float:
        """Calculate similarity between two concept structures"""
        similarity_factors = []
        
        # Node type similarity
        if struct1['node_type'] == struct2['node_type']:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Relation type similarity
        relations1 = set(struct1['relation_counts'].keys())
        relations2 = set(struct2['relation_counts'].keys())
        
        if relations1 or relations2:
            relation_similarity = len(relations1.intersection(relations2)) / len(relations1.union(relations2))
            similarity_factors.append(relation_similarity)
        
        # Neighbor type similarity
        neighbors1 = set(struct1['neighbor_types'].keys())
        neighbors2 = set(struct2['neighbor_types'].keys())
        
        if neighbors1 or neighbors2:
            neighbor_similarity = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
            similarity_factors.append(neighbor_similarity)
        
        # Structural complexity similarity
        depth1 = struct1['depth_2_neighbors']
        depth2 = struct2['depth_2_neighbors']
        max_depth = max(depth1, depth2, 1)
        complexity_similarity = 1.0 - abs(depth1 - depth2) / max_depth
        similarity_factors.append(complexity_similarity)
        
        return np.mean(similarity_factors) if similarity_factors else 0.0
    
    def _create_structural_mapping(self, source_struct: Dict[str, Any], 
                                 target_struct: Dict[str, Any]) -> Dict[str, str]:
        """Create a mapping between structural elements"""
        mapping = {}
        
        # Map relation types
        source_relations = set(source_struct['relation_counts'].keys())
        target_relations = set(target_struct['relation_counts'].keys())
        
        common_relations = source_relations.intersection(target_relations)
        for relation in common_relations:
            mapping[f"relation_{relation}"] = relation
        
        # Map neighbor types
        source_neighbors = set(source_struct['neighbor_types'].keys())
        target_neighbors = set(target_struct['neighbor_types'].keys())
        
        common_neighbors = source_neighbors.intersection(target_neighbors)
        for neighbor_type in common_neighbors:
            mapping[f"neighbor_{neighbor_type}"] = neighbor_type
        
        return mapping


class CausalReasoningEngine:
    """Engine for causal reasoning across domains"""
    
    def __init__(self, knowledge_graph: CrossDomainKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.causal_patterns: Dict[str, ReasoningPattern] = {}
        self.causal_strength_cache: Dict[Tuple[str, str], float] = {}
        
    def infer_causal_relations(self, source_domain: DomainType, target_domain: DomainType,
                              source_event: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Infer causal relations across domains"""
        causal_inferences = []
        
        # Find causal patterns from source domain
        source_patterns = self._extract_causal_patterns(source_domain, source_event)
        
        # Apply patterns to target domain
        for pattern in source_patterns:
            target_inferences = self._apply_causal_pattern(pattern, target_domain, context)
            causal_inferences.extend(target_inferences)
        
        return causal_inferences
    
    def _extract_causal_patterns(self, domain: DomainType, event: str) -> List[Dict[str, Any]]:
        """Extract causal patterns from a domain"""
        patterns = []
        
        # Find event nodes in the domain
        event_nodes = [
            node_id for node_id, node in self.knowledge_graph.nodes.items()
            if node.domain == domain and event.lower() in node.attributes.get('name', '').lower()
        ]
        
        for event_node_id in event_nodes:
            # Find causal relations
            causal_edges = [
                edge for edge in self.knowledge_graph.edges.values()
                if (edge.source_node == event_node_id and 'cause' in edge.relation_type.lower()) or
                   (edge.target_node == event_node_id and 'cause' in edge.relation_type.lower())
            ]
            
            for edge in causal_edges:
                pattern = {
                    'event_node': event_node_id,
                    'causal_relation': edge.relation_type,
                    'related_node': edge.target_node if edge.source_node == event_node_id else edge.source_node,
                    'causal_strength': edge.strength,
                    'direction': 'cause' if edge.source_node == event_node_id else 'effect'
                }
                patterns.append(pattern)
        
        return patterns
    
    def _apply_causal_pattern(self, pattern: Dict[str, Any], target_domain: DomainType,
                            context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Apply a causal pattern to a target domain"""
        inferences = []
        
        # Find analogous events in target domain
        source_node = self.knowledge_graph.nodes[pattern['event_node']]
        
        # Simple matching based on node attributes
        target_candidates = [
            node_id for node_id, node in self.knowledge_graph.nodes.items()
            if (node.domain == target_domain and 
                node.node_type == source_node.node_type)
        ]
        
        for candidate_id in target_candidates:
            candidate_node = self.knowledge_graph.nodes[candidate_id]
            
            # Calculate similarity
            similarity = self._calculate_event_similarity(source_node, candidate_node)
            
            if similarity > 0.4:  # Threshold for causal transfer
                inference_strength = similarity * pattern['causal_strength']
                
                inference = {
                    'inference_type': 'causal',
                    'source_domain': source_node.domain.value,
                    'target_domain': target_domain.value,
                    'source_event': source_node.attributes.get('name', pattern['event_node']),
                    'target_event': candidate_node.attributes.get('name', candidate_id),
                    'causal_relation': pattern['causal_relation'],
                    'causal_direction': pattern['direction'],
                    'inference_strength': inference_strength,
                    'similarity_score': similarity
                }
                
                inferences.append(inference)
        
        return inferences
    
    def _calculate_event_similarity(self, event1: KnowledgeGraphNode, 
                                  event2: KnowledgeGraphNode) -> float:
        """Calculate similarity between two events"""
        similarity_factors = []
        
        # Node type similarity
        if event1.node_type == event2.node_type:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.0)
        
        # Attribute similarity
        attrs1 = set(event1.attributes.keys())
        attrs2 = set(event2.attributes.keys())
        
        if attrs1 or attrs2:
            attr_similarity = len(attrs1.intersection(attrs2)) / len(attrs1.union(attrs2))
            similarity_factors.append(attr_similarity)
        
        # Embedding similarity if available
        if 'default' in event1.embeddings and 'default' in event2.embeddings:
            emb1 = event1.embeddings['default']
            emb2 = event2.embeddings['default']
            
            cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity_factors.append(float(cosine_sim))
        
        return np.mean(similarity_factors) if similarity_factors else 0.0


class CrossDomainReasoningEngine:
    """Main reasoning engine that coordinates different reasoning types"""
    
    def __init__(self, integration_framework: CrossDomainIntegrationFramework):
        self.integration_framework = integration_framework
        self.knowledge_graph = CrossDomainKnowledgeGraph()
        self.analogical_engine = AnalogicalReasoningEngine(self.knowledge_graph)
        self.causal_engine = CausalReasoningEngine(self.knowledge_graph)
        
        # Reasoning history and patterns
        self.inference_history: List[CrossDomainInference] = []
        self.reasoning_patterns: Dict[str, ReasoningPattern] = {}
        self.pattern_success_rates: Dict[str, float] = {}
        
        # Configuration
        self.confidence_threshold = 0.5
        self.max_inference_depth = 3
        
    def populate_knowledge_graph(self, domain_knowledge: Dict[DomainType, Dict[str, Any]]):
        """Populate the knowledge graph with domain-specific knowledge"""
        for domain, knowledge in domain_knowledge.items():
            # Add concepts as nodes
            if 'concepts' in knowledge:
                for concept_name, concept_data in knowledge['concepts'].items():
                    node = KnowledgeGraphNode(
                        node_id=f"{domain.value}_{concept_name}",
                        node_type="concept",
                        domain=domain,
                        attributes={'name': concept_name, **concept_data.get('attributes', {})},
                        embeddings=concept_data.get('embeddings', {}),
                        uncertainty=concept_data.get('uncertainty', 0.0)
                    )
                    self.knowledge_graph.add_node(node)
            
            # Add relations as edges
            if 'relations' in knowledge:
                for relation_data in knowledge['relations']:
                    edge = KnowledgeGraphEdge(
                        edge_id=f"{domain.value}_{relation_data['id']}",
                        source_node=f"{domain.value}_{relation_data['source']}",
                        target_node=f"{domain.value}_{relation_data['target']}",
                        relation_type=relation_data['type'],
                        strength=relation_data.get('strength', 1.0),
                        domain_context=domain,
                        properties=relation_data.get('properties', {}),
                        evidence=relation_data.get('evidence', [])
                    )
                    self.knowledge_graph.add_edge(edge)
    
    def make_cross_domain_inference(self, reasoning_type: ReasoningType,
                                   source_domain: DomainType, target_domain: DomainType,
                                   source_facts: List[str], context: Dict[str, Any] = None) -> List[CrossDomainInference]:
        """Make inferences across domains using specified reasoning type"""
        inferences = []
        
        if reasoning_type == ReasoningType.ANALOGICAL:
            inferences = self._make_analogical_inferences(
                source_domain, target_domain, source_facts, context
            )
        elif reasoning_type == ReasoningType.CAUSAL:
            inferences = self._make_causal_inferences(
                source_domain, target_domain, source_facts, context
            )
        elif reasoning_type == ReasoningType.DEDUCTIVE:
            inferences = self._make_deductive_inferences(
                source_domain, target_domain, source_facts, context
            )
        elif reasoning_type == ReasoningType.INDUCTIVE:
            inferences = self._make_inductive_inferences(
                source_domain, target_domain, source_facts, context
            )
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            inferences = self._make_abductive_inferences(
                source_domain, target_domain, source_facts, context
            )
        
        # Filter by confidence threshold
        valid_inferences = [
            inf for inf in inferences 
            if inf.inference_strength >= self.confidence_threshold
        ]
        
        # Add to history
        self.inference_history.extend(valid_inferences)
        
        return valid_inferences
    
    def _make_analogical_inferences(self, source_domain: DomainType, target_domain: DomainType,
                                   source_facts: List[str], context: Dict[str, Any] = None) -> List[CrossDomainInference]:
        """Make analogical inferences"""
        inferences = []
        
        for fact in source_facts:
            # Extract concept from fact (simple extraction)
            concept = fact.split()[-1] if fact.split() else fact
            
            # Find analogies
            analogies = self.analogical_engine.find_analogies(
                source_domain, concept, [target_domain]
            )
            
            for analogy in analogies:
                inference = CrossDomainInference(
                    inference_id=f"analogical_{len(self.inference_history)}",
                    reasoning_type=ReasoningType.ANALOGICAL,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_facts=[fact],
                    inferred_fact=f"Similar property applies to {analogy['target_concept']}",
                    inference_strength=analogy['similarity_score'],
                    evidence_chain=[{
                        'type': 'analogical_mapping',
                        'mapping': analogy['structural_mapping'],
                        'similarity': analogy['similarity_score']
                    }],
                    context_dependencies=context or {},
                    confidence_factors={'analogical_similarity': analogy['similarity_score']},
                    timestamp=len(self.inference_history)
                )
                inferences.append(inference)
        
        return inferences
    
    def _make_causal_inferences(self, source_domain: DomainType, target_domain: DomainType,
                               source_facts: List[str], context: Dict[str, Any] = None) -> List[CrossDomainInference]:
        """Make causal inferences"""
        inferences = []
        
        for fact in source_facts:
            # Extract event from fact
            event = fact.split()[-1] if fact.split() else fact
            
            # Find causal relations
            causal_relations = self.causal_engine.infer_causal_relations(
                source_domain, target_domain, event, context
            )
            
            for relation in causal_relations:
                inference = CrossDomainInference(
                    inference_id=f"causal_{len(self.inference_history)}",
                    reasoning_type=ReasoningType.CAUSAL,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_facts=[fact],
                    inferred_fact=f"Causal relation: {relation['target_event']} {relation['causal_relation']}",
                    inference_strength=relation['inference_strength'],
                    evidence_chain=[{
                        'type': 'causal_pattern',
                        'relation': relation['causal_relation'],
                        'direction': relation['causal_direction'],
                        'strength': relation['inference_strength']
                    }],
                    context_dependencies=context or {},
                    confidence_factors={'causal_strength': relation['inference_strength']},
                    timestamp=len(self.inference_history)
                )
                inferences.append(inference)
        
        return inferences
    
    def _make_deductive_inferences(self, source_domain: DomainType, target_domain: DomainType,
                                  source_facts: List[str], context: Dict[str, Any] = None) -> List[CrossDomainInference]:
        """Make deductive inferences using logical rules"""
        inferences = []
        
        # Simple deductive reasoning using domain mappings
        concept_mappings = self.integration_framework.unified_representation.concept_mappings
        key = (source_domain, target_domain)
        
        if key in concept_mappings:
            for mapping in concept_mappings[key]:
                for fact in source_facts:
                    if mapping.source_concept.lower() in fact.lower():
                        inferred_fact = fact.replace(
                            mapping.source_concept, mapping.target_concept
                        )
                        
                        inference = CrossDomainInference(
                            inference_id=f"deductive_{len(self.inference_history)}",
                            reasoning_type=ReasoningType.DEDUCTIVE,
                            source_domain=source_domain,
                            target_domain=target_domain,
                            source_facts=[fact],
                            inferred_fact=inferred_fact,
                            inference_strength=mapping.mapping_strength,
                            evidence_chain=[{
                                'type': 'concept_mapping',
                                'mapping': f"{mapping.source_concept} -> {mapping.target_concept}",
                                'strength': mapping.mapping_strength
                            }],
                            context_dependencies=context or {},
                            confidence_factors={'mapping_strength': mapping.mapping_strength},
                            timestamp=len(self.inference_history)
                        )
                        inferences.append(inference)
        
        return inferences
    
    def _make_inductive_inferences(self, source_domain: DomainType, target_domain: DomainType,
                                  source_facts: List[str], context: Dict[str, Any] = None) -> List[CrossDomainInference]:
        """Make inductive inferences by generalizing patterns"""
        inferences = []
        
        # Find patterns in source facts
        if len(source_facts) >= 2:  # Need multiple facts for induction
            # Simple pattern detection: look for common terms
            common_terms = set(source_facts[0].split())
            for fact in source_facts[1:]:
                common_terms = common_terms.intersection(set(fact.split()))
            
            if common_terms:
                # Create generalization
                pattern = " ".join(common_terms)
                inferred_fact = f"Pattern '{pattern}' likely applies in {target_domain.value}"
                
                # Strength based on number of supporting facts
                strength = min(0.9, len(source_facts) / 10.0)
                
                inference = CrossDomainInference(
                    inference_id=f"inductive_{len(self.inference_history)}",
                    reasoning_type=ReasoningType.INDUCTIVE,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_facts=source_facts,
                    inferred_fact=inferred_fact,
                    inference_strength=strength,
                    evidence_chain=[{
                        'type': 'pattern_generalization',
                        'pattern': pattern,
                        'supporting_facts': len(source_facts)
                    }],
                    context_dependencies=context or {},
                    confidence_factors={'pattern_support': len(source_facts)},
                    timestamp=len(self.inference_history)
                )
                inferences.append(inference)
        
        return inferences
    
    def _make_abductive_inferences(self, source_domain: DomainType, target_domain: DomainType,
                                  source_facts: List[str], context: Dict[str, Any] = None) -> List[CrossDomainInference]:
        """Make abductive inferences to find best explanations"""
        inferences = []
        
        # For each fact, try to find the best explanation in the target domain
        for fact in source_facts:
            # Find potential explanations by looking at causal patterns
            explanations = self._find_potential_explanations(fact, target_domain)
            
            # Rank explanations by plausibility
            for explanation in explanations:
                inference = CrossDomainInference(
                    inference_id=f"abductive_{len(self.inference_history)}",
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    source_facts=[fact],
                    inferred_fact=explanation['explanation'],
                    inference_strength=explanation['plausibility'],
                    evidence_chain=[{
                        'type': 'best_explanation',
                        'explanation': explanation['explanation'],
                        'plausibility': explanation['plausibility']
                    }],
                    context_dependencies=context or {},
                    confidence_factors={'explanation_plausibility': explanation['plausibility']},
                    timestamp=len(self.inference_history)
                )
                inferences.append(inference)
        
        return inferences
    
    def _find_potential_explanations(self, fact: str, target_domain: DomainType) -> List[Dict[str, Any]]:
        """Find potential explanations for a fact in the target domain"""
        explanations = []
        
        # Extract key terms from the fact
        fact_terms = fact.split()
        
        # Look for nodes in target domain that might explain the fact
        target_nodes = [
            node for node in self.knowledge_graph.nodes.values()
            if node.domain == target_domain
        ]
        
        for node in target_nodes:
            # Calculate relevance of this node as an explanation
            node_terms = node.attributes.get('name', '').split()
            
            # Simple relevance based on term overlap
            overlap = len(set(fact_terms).intersection(set(node_terms)))
            total_terms = len(set(fact_terms).union(set(node_terms)))
            
            if total_terms > 0:
                relevance = overlap / total_terms
                
                if relevance > 0.1:  # Minimum relevance threshold
                    explanation = {
                        'explanation': f"{node.attributes.get('name', node.node_id)} explains the observed fact",
                        'plausibility': relevance,
                        'explaining_node': node.node_id
                    }
                    explanations.append(explanation)
        
        # Sort by plausibility
        explanations.sort(key=lambda x: x['plausibility'], reverse=True)
        return explanations[:5]  # Return top 5 explanations
    
    def validate_inference_consistency(self) -> Dict[str, float]:
        """Validate consistency of cross-domain inferences"""
        consistency_metrics = {
            'logical_consistency': 0.0,
            'temporal_consistency': 0.0,
            'strength_consistency': 0.0,
            'overall_consistency': 0.0
        }
        
        if not self.inference_history:
            return consistency_metrics
        
        # Check logical consistency
        contradictions = 0
        total_pairs = 0
        
        for i, inf1 in enumerate(self.inference_history):
            for inf2 in self.inference_history[i+1:]:
                total_pairs += 1
                
                # Check for logical contradictions
                if self._check_logical_contradiction(inf1, inf2):
                    contradictions += 1
        
        if total_pairs > 0:
            consistency_metrics['logical_consistency'] = 1.0 - (contradictions / total_pairs)
        
        # Check temporal consistency
        temporal_violations = 0
        temporal_checks = 0
        
        for inference in self.inference_history:
            if inference.context_dependencies.get('temporal_constraints'):
                temporal_checks += 1
                # Simple check: later inferences should not contradict earlier ones
                # without sufficient evidence
                if not self._check_temporal_consistency(inference):
                    temporal_violations += 1
        
        if temporal_checks > 0:
            consistency_metrics['temporal_consistency'] = 1.0 - (temporal_violations / temporal_checks)
        else:
            consistency_metrics['temporal_consistency'] = 1.0
        
        # Check strength consistency
        strength_inconsistencies = 0
        strength_checks = 0
        
        for inference in self.inference_history:
            strength_checks += 1
            expected_strength = self._calculate_expected_strength(inference)
            actual_strength = inference.inference_strength
            
            if abs(expected_strength - actual_strength) > 0.3:  # Threshold
                strength_inconsistencies += 1
        
        if strength_checks > 0:
            consistency_metrics['strength_consistency'] = 1.0 - (strength_inconsistencies / strength_checks)
        
        # Overall consistency
        consistency_metrics['overall_consistency'] = np.mean([
            consistency_metrics['logical_consistency'],
            consistency_metrics['temporal_consistency'],
            consistency_metrics['strength_consistency']
        ])
        
        return consistency_metrics
    
    def _check_logical_contradiction(self, inf1: CrossDomainInference, 
                                   inf2: CrossDomainInference) -> bool:
        """Check if two inferences contradict each other"""
        # Simple contradiction detection
        if (inf1.source_domain == inf2.source_domain and 
            inf1.target_domain == inf2.target_domain):
            
            # Check if facts are contradictory
            fact1_words = set(inf1.inferred_fact.lower().split())
            fact2_words = set(inf2.inferred_fact.lower().split())
            
            # Look for negation words
            negation_words = {'not', 'no', 'never', 'none', 'cannot', 'impossible'}
            
            has_negation1 = bool(fact1_words.intersection(negation_words))
            has_negation2 = bool(fact2_words.intersection(negation_words))
            
            # If one has negation and they share common terms, might be contradiction
            if has_negation1 != has_negation2:
                common_terms = fact1_words.intersection(fact2_words) - negation_words
                if len(common_terms) > 0:
                    return True
        
        return False
    
    def _check_temporal_consistency(self, inference: CrossDomainInference) -> bool:
        """Check if an inference is temporally consistent"""
        # Simple temporal consistency check
        # Later inferences should generally have higher or similar confidence
        # when dealing with the same concepts
        
        related_inferences = [
            inf for inf in self.inference_history
            if (inf.source_domain == inference.source_domain and
                inf.target_domain == inference.target_domain and
                inf.timestamp < inference.timestamp)
        ]
        
        if related_inferences:
            avg_previous_strength = np.mean([inf.inference_strength for inf in related_inferences])
            
            # Allow for some variance, but significant drops in confidence might indicate issues
            if inference.inference_strength < avg_previous_strength - 0.4:
                return False
        
        return True
    
    def _calculate_expected_strength(self, inference: CrossDomainInference) -> float:
        """Calculate expected strength for an inference based on its evidence"""
        evidence_strength = 0.0
        evidence_count = len(inference.evidence_chain)
        
        if evidence_count == 0:
            return 0.1  # Very low confidence without evidence
        
        for evidence in inference.evidence_chain:
            if 'strength' in evidence:
                evidence_strength += evidence['strength']
            elif 'similarity' in evidence:
                evidence_strength += evidence['similarity']
            else:
                evidence_strength += 0.5  # Default moderate strength
        
        # Average evidence strength, with bonus for multiple pieces of evidence
        base_strength = evidence_strength / evidence_count
        evidence_bonus = min(0.2, evidence_count * 0.05)  # Up to 20% bonus
        
        return min(0.95, base_strength + evidence_bonus)
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning performance"""
        if not self.inference_history:
            return {
                'total_inferences': 0,
                'reasoning_types': {},
                'domain_pairs': {},
                'average_strength': 0.0,
                'consistency_scores': {}
            }
        
        stats = {
            'total_inferences': len(self.inference_history),
            'reasoning_types': defaultdict(int),
            'domain_pairs': defaultdict(int),
            'average_strength': 0.0,
            'consistency_scores': self.validate_inference_consistency()
        }
        
        total_strength = 0.0
        
        for inference in self.inference_history:
            stats['reasoning_types'][inference.reasoning_type.value] += 1
            domain_pair = f"{inference.source_domain.value} -> {inference.target_domain.value}"
            stats['domain_pairs'][domain_pair] += 1
            total_strength += inference.inference_strength
        
        stats['average_strength'] = total_strength / len(self.inference_history)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        stats['reasoning_types'] = dict(stats['reasoning_types'])
        stats['domain_pairs'] = dict(stats['domain_pairs'])
        
        return stats