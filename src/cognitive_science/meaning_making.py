from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from ..core.relevance_core import RelevanceCore, RelevanceMode

class CommunicationMaxim(Enum):
    """Grice's maxims as described by Vervaeke"""
    QUALITY = "quality"  # Be truthful/sincere
    QUANTITY = "quantity"  # Right amount of info
    MANNER = "manner"  # Clear presentation
    RELEVANCE = "relevance"  # Be relevant

class MeaningLevel(Enum):
    """Levels of meaning representation and processing"""
    SUBSYMBOLIC = "subsymbolic"  # Neural patterns, embeddings
    SYMBOLIC = "symbolic"  # Concepts, relations, rules
    NARRATIVE = "narrative"  # Stories, temporal sequences
    CULTURAL = "cultural"  # Social norms, collective meanings
    METACOGNITIVE = "metacognitive"  # Reflection on meaning

class EmotionalValence(Enum):
    """Emotional dimensions of meaning"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class SemanticNode:
    """Represents a semantic concept with multi-level properties"""
    concept: str
    embedding: Optional[np.ndarray] = None
    relations: Dict[str, List[str]] = field(default_factory=dict)
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    activation_strength: float = 0.0
    meaning_level: MeaningLevel = MeaningLevel.SYMBOLIC

@dataclass
class MeaningStructure:
    """Hierarchical meaning structure with validation"""
    core_nodes: List[SemanticNode] = field(default_factory=list)
    relations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    coherence_score: float = 0.0
    cultural_embedding: Dict[str, Any] = field(default_factory=dict)
    temporal_context: Dict[str, Any] = field(default_factory=dict)
    
    def validate_consistency(self) -> bool:
        """Validate internal consistency of meaning structure"""
        if not self.core_nodes:
            return False
        
        # Check relation consistency
        for source, targets in self.relations.items():
            source_exists = any(node.concept == source for node in self.core_nodes)
            if not source_exists:
                return False
                
            for target in targets.keys():
                target_exists = any(node.concept == target for node in self.core_nodes)
                if not target_exists:
                    return False
        
        return True

@dataclass
class MeaningContext:
    """Enhanced context for meaning-making with multi-level integration"""
    nomological: Dict  # Causal/scientific relations
    normative: Dict  # Value/ethical relations  
    narrative: Dict  # Story/temporal relations
    participatory: Dict  # Embodied/interactive relations
    emotional: Dict = field(default_factory=dict)  # Emotional associations
    cultural: Dict = field(default_factory=dict)  # Cultural context
    semantic_network: Dict[str, SemanticNode] = field(default_factory=dict)
    meaning_structures: List[MeaningStructure] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)

class SymbolicSubsymbolicBridge(ABC):
    """Abstract base for bridging symbolic and subsymbolic representations"""
    
    @abstractmethod
    def symbolic_to_subsymbolic(self, symbolic_content: Dict) -> np.ndarray:
        """Convert symbolic representation to subsymbolic"""
        pass
    
    @abstractmethod
    def subsymbolic_to_symbolic(self, subsymbolic_content: np.ndarray) -> Dict:
        """Convert subsymbolic representation to symbolic"""
        pass

class NeuralSemanticBridge(SymbolicSubsymbolicBridge):
    """Neural network-based bridge between symbolic and subsymbolic"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        
    def symbolic_to_subsymbolic(self, symbolic_content: Dict) -> np.ndarray:
        """Convert symbolic concepts to neural embeddings"""
        embeddings = []
        
        for concept, weight in symbolic_content.items():
            if concept not in self.concept_embeddings:
                # Generate random embedding for new concepts
                self.concept_embeddings[concept] = np.random.randn(self.embedding_dim)
            
            embedding = self.concept_embeddings[concept] * weight
            embeddings.append(embedding)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def subsymbolic_to_symbolic(self, subsymbolic_content: np.ndarray) -> Dict:
        """Convert neural embedding to symbolic concepts"""
        if not self.concept_embeddings:
            return {}
        
        # Find most similar concepts
        similarities = {}
        for concept, embedding in self.concept_embeddings.items():
            similarity = np.dot(subsymbolic_content, embedding) / (
                np.linalg.norm(subsymbolic_content) * np.linalg.norm(embedding) + 1e-8
            )
            similarities[concept] = max(0, similarity)  # Only positive similarities
        
        # Normalize to get probability distribution
        total = sum(similarities.values())
        if total > 0:
            return {k: v/total for k, v in similarities.items() if v > 0.1}
        else:
            return {}

class EmotionalCognitiveSynthesizer:
    """Synthesizes emotional and cognitive aspects of meaning"""
    
    def __init__(self):
        self.emotion_concept_map: Dict[str, EmotionalValence] = {}
        self.cognitive_emotional_weights: Dict[str, float] = {}
    
    def synthesize_meaning(self, cognitive_content: Dict, 
                          emotional_context: Dict) -> Dict:
        """Synthesize cognitive and emotional aspects into unified meaning"""
        synthesized = {}
        
        # Weight cognitive content by emotional associations
        for concept, cognitive_weight in cognitive_content.items():
            emotional_weight = emotional_context.get(concept, 0.5)
            
            # Emotional modulation of cognitive content
            if concept in self.emotion_concept_map:
                valence = self.emotion_concept_map[concept]
                if valence == EmotionalValence.POSITIVE:
                    emotional_weight *= 1.2
                elif valence == EmotionalValence.NEGATIVE:
                    emotional_weight *= 0.8
                    
            synthesized[concept] = cognitive_weight * emotional_weight
        
        # Add purely emotional concepts
        for emotion, weight in emotional_context.items():
            if emotion not in synthesized and weight > 0.3:
                synthesized[emotion] = weight * 0.7  # Lower weight for pure emotions
        
        return synthesized
    
    def update_emotion_associations(self, concept: str, valence: EmotionalValence):
        """Update emotional associations for concepts"""
        self.emotion_concept_map[concept] = valence

class CulturalContextProcessor:
    """Processes cultural and social context in meaning-making"""
    
    def __init__(self):
        self.cultural_norms: Dict[str, Dict[str, float]] = {}
        self.social_contexts: Dict[str, Dict[str, Any]] = {}
        
    def process_cultural_meaning(self, content: Dict, 
                                cultural_context: str) -> Dict:
        """Process meaning through cultural lens"""
        if cultural_context not in self.cultural_norms:
            return content  # No cultural modification if context unknown
        
        cultural_weights = self.cultural_norms[cultural_context]
        processed = {}
        
        for concept, weight in content.items():
            cultural_modifier = cultural_weights.get(concept, 1.0)
            processed[concept] = weight * cultural_modifier
            
        return processed
    
    def add_cultural_norm(self, context: str, concept: str, modifier: float):
        """Add cultural norm that modifies concept weights"""
        if context not in self.cultural_norms:
            self.cultural_norms[context] = {}
        self.cultural_norms[context][concept] = modifier

class MeaningMaker:
    """Enhanced meaning-making system with comprehensive semantic processing.
    
    This implements Vervaeke's framework for meaning cultivation through
    the integration of multiple ways of knowing and relating, enhanced with
    multi-level semantic representation, emotional-cognitive synthesis,
    and cultural context understanding.
    """
    
    def __init__(self, relevance_core: RelevanceCore, embedding_dim: int = 512):
        self.relevance_core = relevance_core
        self.current_context = MeaningContext(
            nomological={},
            normative={},
            narrative={},
            participatory={}
        )
        
        # Enhanced components for comprehensive meaning-making
        self.symbolic_bridge = NeuralSemanticBridge(embedding_dim)
        self.emotional_synthesizer = EmotionalCognitiveSynthesizer()
        self.cultural_processor = CulturalContextProcessor()
        
        # Meaning validation and consistency
        self.validation_threshold = 0.6
        self.feedback_history: List[Dict] = []
        self.adaptation_rate = 0.1
        
        # Multi-level meaning integration
        self.meaning_levels: Dict[MeaningLevel, Dict] = {
            level: {} for level in MeaningLevel
        }
    def construct_contextual_meaning(self, experience: Dict,
                                    context: Optional[Dict] = None,
                                    cultural_context: Optional[str] = None) -> MeaningStructure:
        """Construct meaning from experience with full contextual processing.
        
        Args:
            experience: Features of the experience
            context: Optional additional context
            cultural_context: Cultural framework for interpretation
            
        Returns:
            Constructed meaning structure
        """
        # Extract semantic nodes from experience
        semantic_nodes = self._extract_semantic_nodes(experience)
        
        # Build relations between nodes
        relations = self._build_semantic_relations(semantic_nodes, context)
        
        # Apply cultural processing if context provided
        if cultural_context:
            relations = self.cultural_processor.process_cultural_meaning(
                relations, cultural_context
            )
        
        # Create meaning structure
        meaning_structure = MeaningStructure(
            core_nodes=semantic_nodes,
            relations=relations,
            cultural_embedding=context.get('cultural', {}) if context else {}
        )
        
        # Validate and compute coherence
        if meaning_structure.validate_consistency():
            meaning_structure.coherence_score = self._compute_coherence(meaning_structure)
        
        return meaning_structure
    
    def integrate_multi_level_meaning(self, structures: List[MeaningStructure]) -> MeaningStructure:
        """Integrate meaning structures across multiple levels.
        
        Args:
            structures: List of meaning structures to integrate
            
        Returns:
            Integrated meaning structure
        """
        if not structures:
            return MeaningStructure()
        
        # Collect all nodes and relations
        all_nodes = []
        all_relations = {}
        total_coherence = 0.0
        
        for structure in structures:
            all_nodes.extend(structure.core_nodes)
            
            # Merge relations with weighted averaging
            for source, targets in structure.relations.items():
                if source not in all_relations:
                    all_relations[source] = {}
                
                for target, weight in targets.items():
                    if target in all_relations[source]:
                        # Average existing weights
                        all_relations[source][target] = (
                            all_relations[source][target] + weight
                        ) / 2
                    else:
                        all_relations[source][target] = weight
            
            total_coherence += structure.coherence_score
        
        # Remove duplicate nodes (keep highest activation)
        unique_nodes = {}
        for node in all_nodes:
            if node.concept not in unique_nodes:
                unique_nodes[node.concept] = node
            elif node.activation_strength > unique_nodes[node.concept].activation_strength:
                unique_nodes[node.concept] = node
        
        integrated = MeaningStructure(
            core_nodes=list(unique_nodes.values()),
            relations=all_relations,
            coherence_score=total_coherence / len(structures) if structures else 0.0
        )
        
        return integrated
    
    def bridge_symbolic_subsymbolic(self, symbolic_content: Dict) -> Tuple[np.ndarray, Dict]:
        """Bridge between symbolic and subsymbolic representations.
        
        Args:
            symbolic_content: Symbolic meaning representation
            
        Returns:
            Tuple of (subsymbolic_embedding, reconstructed_symbolic)
        """
        # Convert to subsymbolic
        subsymbolic = self.symbolic_bridge.symbolic_to_subsymbolic(symbolic_content)
        
        # Reconstruct symbolic to test fidelity
        reconstructed = self.symbolic_bridge.subsymbolic_to_symbolic(subsymbolic)
        
        return subsymbolic, reconstructed
    
    def synthesize_emotional_cognitive_meaning(self, cognitive_content: Dict,
                                             emotional_context: Dict) -> Dict:
        """Synthesize emotional and cognitive aspects of meaning.
        
        Args:
            cognitive_content: Cognitive aspects of meaning
            emotional_context: Emotional context and associations
            
        Returns:
            Synthesized meaning with emotional-cognitive integration
        """
        return self.emotional_synthesizer.synthesize_meaning(
            cognitive_content, emotional_context
        )
    
    def validate_meaning_consistency(self, meaning_structure: MeaningStructure) -> Tuple[bool, List[str]]:
        """Validate meaning consistency and coherence.
        
        Args:
            meaning_structure: Structure to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Basic structure validation
        if not meaning_structure.validate_consistency():
            issues.append("Basic consistency check failed")
        
        # Coherence threshold check
        if meaning_structure.coherence_score < self.validation_threshold:
            issues.append(f"Coherence score {meaning_structure.coherence_score:.2f} below threshold {self.validation_threshold}")
        
        # Check for conflicting emotional valences
        valences = [node.emotional_valence for node in meaning_structure.core_nodes]
        if EmotionalValence.POSITIVE in valences and EmotionalValence.NEGATIVE in valences:
            # This might be valid for complex meanings, so just note it
            issues.append("Mixed emotional valences detected - verify intentional")
        
        # Check relation strength distribution
        if meaning_structure.relations:
            all_weights = []
            for targets in meaning_structure.relations.values():
                all_weights.extend(targets.values())
            
            if all_weights:
                weight_variance = np.var(all_weights)
                if weight_variance < 0.01:  # Very low variance
                    issues.append("Relation weights lack diversity - may indicate over-simplification")
        
        return len(issues) == 0, issues
    
    def adapt_meaning_from_feedback(self, feedback: Dict) -> None:
        """Adapt meaning-making based on feedback.
        
        Args:
            feedback: Feedback containing evaluation and suggestions
        """
        self.feedback_history.append(feedback)
        
        # Adapt validation threshold based on feedback
        if 'coherence_target' in feedback:
            target = feedback['coherence_target']
            current_threshold = self.validation_threshold
            self.validation_threshold += self.adaptation_rate * (target - current_threshold)
            self.validation_threshold = max(0.1, min(0.9, self.validation_threshold))
        
        # Update emotional associations based on feedback
        if 'emotion_corrections' in feedback:
            for concept, valence in feedback['emotion_corrections'].items():
                self.emotional_synthesizer.update_emotion_associations(concept, valence)
        
        # Update cultural norms based on feedback
        if 'cultural_corrections' in feedback:
            for context, norms in feedback['cultural_corrections'].items():
                for concept, modifier in norms.items():
                    self.cultural_processor.add_cultural_norm(context, concept, modifier)
        
        # Limit feedback history size
        if len(self.feedback_history) > 100:
            self.feedback_history = self.feedback_history[-50:]
    
    def communicate(self, message: str, context: Dict) -> Tuple[str, float]:
        """Generate communication following Grice's maxims.
        
        Args:
            message: Core message to communicate
            context: Communication context
            
        Returns:
            Tuple of (refined message, confidence)
        """
        # Check maxims using relevance core
        maxim_scores = {}
        for maxim in CommunicationMaxim:
            relevant_aspects = self.relevance_core.evaluate_relevance(
                {message},
                context={
                    "maxim": maxim,
                    "context": context
                }
            )
            maxim_scores[maxim] = relevant_aspects[1]  # Get confidence
            
        # Refine message based on maxim scores
        refined_message = self._refine_message(
            message, maxim_scores, context
        )
        
        # Calculate overall confidence
        confidence = np.mean(list(maxim_scores.values()))
        
        return refined_message, confidence
        
    def cultivate_meaning(self, experience: Dict,
                         context: Optional[Dict] = None) -> MeaningContext:
        """Cultivate meaning from experience across multiple domains.
        
        Args:
            experience: Features of the experience
            context: Optional additional context
            
        Returns:
            Updated meaning context
        """
        # Update each aspect of meaning
        self._update_nomological(experience, context)
        self._update_normative(experience, context)
        self._update_narrative(experience, context)
        self._update_participatory(experience, context)
        
        return self.current_context
        
    def _refine_message(self, message: str,
                       maxim_scores: Dict[CommunicationMaxim, float],
                       context: Dict) -> str:
        """Refine message based on maxim scores."""
        # Placeholder for more sophisticated message refinement
        if maxim_scores[CommunicationMaxim.QUANTITY] < 0.5:
            message += " [More detail needed]"
        if maxim_scores[CommunicationMaxim.MANNER] < 0.5:
            message += " [Clarity needed]"
        return message
        
    def _update_nomological(self, experience: Dict,
                           context: Optional[Dict]) -> None:
        """Update causal/scientific relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "nomological"}
        )[0]
        
        self.current_context.nomological.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
        
    def _update_normative(self, experience: Dict,
                         context: Optional[Dict]) -> None:
        """Update value/ethical relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "normative"}
        )[0]
        
        self.current_context.normative.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
        
    def _update_narrative(self, experience: Dict,
                         context: Optional[Dict]) -> None:
        """Update story/temporal relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "narrative"}
        )[0]
        
        self.current_context.narrative.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
        
    def _update_participatory(self, experience: Dict,
                            context: Optional[Dict]) -> None:
        """Update embodied/interactive relations."""
        relevant_features = self.relevance_core.evaluate_relevance(
            set(experience.keys()),
            context={"domain": "participatory"}
        )[0]
        
        self.current_context.participatory.update({
            k: experience[k] for k in relevant_features
            if k in experience
        })
    
    def _extract_semantic_nodes(self, experience: Dict) -> List[SemanticNode]:
        """Extract semantic nodes from experience data"""
        nodes = []
        
        for concept, value in experience.items():
            # Create embedding for concept
            embedding = None
            if isinstance(value, (int, float)):
                # Simple numerical embedding
                embedding = np.array([value] * self.symbolic_bridge.embedding_dim)
            elif isinstance(value, str):
                # Hash-based embedding for strings
                hash_val = hash(value) % 1000000
                np.random.seed(hash_val)
                embedding = np.random.randn(self.symbolic_bridge.embedding_dim)
            
            # Determine emotional valence (placeholder logic)
            valence = EmotionalValence.NEUTRAL
            if isinstance(value, (int, float)):
                if value > 0.7:
                    valence = EmotionalValence.POSITIVE
                elif value < 0.3:
                    valence = EmotionalValence.NEGATIVE
            
            node = SemanticNode(
                concept=concept,
                embedding=embedding,
                emotional_valence=valence,
                activation_strength=abs(value) if isinstance(value, (int, float)) else 0.5
            )
            nodes.append(node)
        
        return nodes
    
    def _build_semantic_relations(self, nodes: List[SemanticNode], 
                                 context: Optional[Dict]) -> Dict[str, Dict[str, float]]:
        """Build semantic relations between nodes"""
        relations = {}
        
        for i, node1 in enumerate(nodes):
            relations[node1.concept] = {}
            
            for j, node2 in enumerate(nodes):
                if i != j and node1.embedding is not None and node2.embedding is not None:
                    # Compute similarity as relation strength
                    similarity = np.dot(node1.embedding, node2.embedding) / (
                        np.linalg.norm(node1.embedding) * np.linalg.norm(node2.embedding) + 1e-8
                    )
                    
                    # Only keep positive similarities above threshold
                    if similarity > 0.3:
                        relations[node1.concept][node2.concept] = float(similarity)
        
        return relations
    
    def _compute_coherence(self, meaning_structure: MeaningStructure) -> float:
        """Compute coherence score for meaning structure"""
        if not meaning_structure.core_nodes:
            return 0.0
        
        coherence_factors = []
        
        # Factor 1: Node activation consistency
        activations = [node.activation_strength for node in meaning_structure.core_nodes]
        if activations:
            activation_variance = np.var(activations)
            # Lower variance means more consistent activation
            coherence_factors.append(1.0 / (1.0 + activation_variance))
        
        # Factor 2: Relation density and strength
        if meaning_structure.relations:
            all_weights = []
            for targets in meaning_structure.relations.values():
                all_weights.extend(targets.values())
            
            if all_weights:
                avg_weight = np.mean(all_weights)
                relation_density = len(all_weights) / (len(meaning_structure.core_nodes) ** 2)
                coherence_factors.append(avg_weight * relation_density)
        
        # Factor 3: Emotional valence consistency
        valences = [node.emotional_valence for node in meaning_structure.core_nodes]
        unique_valences = set(valences)
        if len(unique_valences) == 1:
            coherence_factors.append(1.0)  # Perfect consistency
        elif len(unique_valences) == 2 and EmotionalValence.NEUTRAL in unique_valences:
            coherence_factors.append(0.8)  # Good consistency
        else:
            coherence_factors.append(0.4)  # Mixed valences
        
        return np.mean(coherence_factors) if coherence_factors else 0.0