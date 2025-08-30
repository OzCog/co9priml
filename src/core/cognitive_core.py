import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from .state import CognitiveState
from ..modules.perception import PerceptionModule, SensoryInput
from ..modules.reasoning import ReasoningModule, Thought
from ..modules.action import ActionSelectionModule, Action
from ..modules.learning import ReinforcementLearner, Experience
from ..learning.meta_learning import MetaLearner, MetaExperience, LearningStrategy

# Import Vervaeke 4E Cognition Framework
from ..modules.vervaeke_4e import Vervaeke4ECognitionFramework, CognitionMode, KnowingMode

# Import the new AtomSpace and Memory modules
from ..atomspace import AtomSpace, Node, Link, BackendType
from ..memory import Memory

# Import Cross-Domain Integration Framework
from ..integration.cross_domain_framework import (
    CrossDomainIntegrationFramework, DomainType, ModalityType,
    ConceptMapping, AbstractConcept
)
from ..integration.cross_domain_reasoning import (
    CrossDomainReasoningEngine, ReasoningType
)
from ..integration.multimodal_knowledge_graph import (
    MultiModalKnowledgeGraph, ModalityFeature, ModalityEmbeddingType
)

class CogPrimeCore:
    """
    The core cognitive architecture of CogPrime system.
    Implements the basic cognitive cycle and main AGI components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.perception = PerceptionModule(config)
        self.reasoning = ReasoningModule(config)
        self.action_selector = ActionSelectionModule(config)
        self.learner = ReinforcementLearner(config)
        
        # Initialize meta-learning system
        self.meta_learner = MetaLearner(self.config.get('meta_learning_config', {}))
        
        # Initialize Vervaeke 4E Cognition Framework
        self.vervaeke_framework = Vervaeke4ECognitionFramework(self.config.get('vervaeke_config', {}))
        
        # Track current domain and task for meta-learning
        self.current_domain = self.config.get('default_domain', 'general')
        self.current_task = 'cognitive_processing'
        
        # Initialize AtomSpace with the specified backend
        atomspace_backend = self.config.get('atomspace_backend', 'local')
        atomspace_config = self.config.get('atomspace_config', {})
        self.atomspace = AtomSpace(backend_type=atomspace_backend, config=atomspace_config)
        
        # Initialize Memory with the specified backend
        memory_backend = self.config.get('memory_backend', 'mem0')
        memory_config = self.config.get('memory_config', {})
        self.memory = Memory(backend_type=memory_backend, config=memory_config)
        
        # Initialize Cross-Domain Integration Framework
        cross_domain_config = self.config.get('cross_domain_config', {})
        self.cross_domain_framework = CrossDomainIntegrationFramework(cross_domain_config)
        
        # Initialize Cross-Domain Reasoning Engine
        self.reasoning_engine = CrossDomainReasoningEngine(self.cross_domain_framework)
        
        # Initialize Multi-Modal Knowledge Graph
        self.multimodal_kg = MultiModalKnowledgeGraph(self.reasoning_engine.knowledge_graph)
        
        # Register default domains and modalities
        self._initialize_cross_domain_components()
        
        # Initialize cognitive state
        self.state = CognitiveState(
            attention_focus=torch.zeros(512),  # Initial attention vector
            working_memory={},
            emotional_valence=0.0,
            goal_stack=[],
            sensory_buffer={},
            current_thought=None,
            last_action=None,
            last_reward=0.0,
            total_reward=0.0
        )
        
        # Register callbacks
        self._cycle_callbacks = []
        
        # Create AtomSpace nodes for core concepts
        self._create_core_atoms()
    
    def _initialize_cross_domain_components(self):
        """Initialize cross-domain integration components"""
        # Register core domains
        core_domains = [
            DomainType.VISUAL, DomainType.AUDITORY, DomainType.LINGUISTIC,
            DomainType.SPATIAL, DomainType.TEMPORAL, DomainType.ABSTRACT
        ]
        
        for domain in core_domains:
            self.cross_domain_framework.register_domain(domain)
        
        # Register core modalities  
        core_modalities = [
            ModalityType.VISION, ModalityType.HEARING, ModalityType.LANGUAGE,
            ModalityType.REASONING, ModalityType.MEMORY, ModalityType.ATTENTION
        ]
        
        for modality in core_modalities:
            self.cross_domain_framework.register_modality(modality)
        
        # Create basic concept mappings
        self._create_basic_concept_mappings()
        
        # Populate reasoning engine with initial knowledge
        self._populate_initial_knowledge()
    
    def _create_basic_concept_mappings(self):
        """Create basic cross-domain concept mappings"""
        # Visual-Linguistic mappings
        visual_linguistic_mappings = [
            ("red", "red color"),
            ("large", "big size"),
            ("round", "circular shape"),
            ("bright", "high luminosity")
        ]
        
        for visual_concept, linguistic_concept in visual_linguistic_mappings:
            mapping = ConceptMapping(
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.LINGUISTIC,
                source_concept=visual_concept,
                target_concept=linguistic_concept,
                mapping_strength=0.8,
                semantic_similarity=0.9,
                bidirectional=True
            )
            self.cross_domain_framework.unified_representation.add_concept_mapping(mapping)
        
        # Audio-Linguistic mappings
        audio_linguistic_mappings = [
            ("loud", "high volume"),
            ("high_pitch", "high frequency"),
            ("melody", "musical sequence"),
            ("rhythm", "temporal pattern")
        ]
        
        for audio_concept, linguistic_concept in audio_linguistic_mappings:
            mapping = ConceptMapping(
                source_domain=DomainType.AUDITORY,
                target_domain=DomainType.LINGUISTIC,
                source_concept=audio_concept,
                target_concept=linguistic_concept,
                mapping_strength=0.7,
                semantic_similarity=0.8,
                bidirectional=True
            )
            self.cross_domain_framework.unified_representation.add_concept_mapping(mapping)
        
        # Spatial-Temporal mappings
        spatial_temporal_mappings = [
            ("near", "soon"),
            ("far", "distant_future"),
            ("above", "before"),
            ("below", "after")
        ]
        
        for spatial_concept, temporal_concept in spatial_temporal_mappings:
            mapping = ConceptMapping(
                source_domain=DomainType.SPATIAL,
                target_domain=DomainType.TEMPORAL,
                source_concept=spatial_concept,
                target_concept=temporal_concept,
                mapping_strength=0.6,
                semantic_similarity=0.7,
                bidirectional=True
            )
            self.cross_domain_framework.unified_representation.add_concept_mapping(mapping)
    
    def _populate_initial_knowledge(self):
        """Populate reasoning engine with initial domain knowledge"""
        # Visual domain knowledge
        visual_knowledge = {
            'concepts': {
                'red': {'attributes': {'color': True, 'wavelength': 700}, 'uncertainty': 0.1},
                'blue': {'attributes': {'color': True, 'wavelength': 450}, 'uncertainty': 0.1},
                'large': {'attributes': {'size': True, 'relative': True}, 'uncertainty': 0.2},
                'small': {'attributes': {'size': True, 'relative': True}, 'uncertainty': 0.2}
            },
            'relations': [
                {'id': 'red_blue_contrast', 'source': 'red', 'target': 'blue', 'type': 'contrasts_with', 'strength': 0.8},
                {'id': 'large_small_opposite', 'source': 'large', 'target': 'small', 'type': 'opposite_of', 'strength': 0.9}
            ]
        }
        
        # Auditory domain knowledge
        auditory_knowledge = {
            'concepts': {
                'loud': {'attributes': {'volume': True, 'intensity': 'high'}, 'uncertainty': 0.1},
                'quiet': {'attributes': {'volume': True, 'intensity': 'low'}, 'uncertainty': 0.1},
                'high_pitch': {'attributes': {'frequency': True, 'value': 'high'}, 'uncertainty': 0.2},
                'low_pitch': {'attributes': {'frequency': True, 'value': 'low'}, 'uncertainty': 0.2}
            },
            'relations': [
                {'id': 'loud_quiet_opposite', 'source': 'loud', 'target': 'quiet', 'type': 'opposite_of', 'strength': 0.9},
                {'id': 'high_low_pitch_opposite', 'source': 'high_pitch', 'target': 'low_pitch', 'type': 'opposite_of', 'strength': 0.9}
            ]
        }
        
        # Linguistic domain knowledge
        linguistic_knowledge = {
            'concepts': {
                'red_color': {'attributes': {'semantic_field': 'color', 'type': 'adjective'}, 'uncertainty': 0.05},
                'big_size': {'attributes': {'semantic_field': 'size', 'type': 'adjective'}, 'uncertainty': 0.1},
                'high_volume': {'attributes': {'semantic_field': 'sound', 'type': 'adjective'}, 'uncertainty': 0.1}
            },
            'relations': [
                {'id': 'color_size_different_fields', 'source': 'red_color', 'target': 'big_size', 
                 'type': 'different_semantic_field', 'strength': 0.7}
            ]
        }
        
        domain_knowledge = {
            DomainType.VISUAL: visual_knowledge,
            DomainType.AUDITORY: auditory_knowledge,
            DomainType.LINGUISTIC: linguistic_knowledge
        }
        
        self.reasoning_engine.populate_knowledge_graph(domain_knowledge)
    
    def _create_core_atoms(self):
        """Create foundational atoms in the AtomSpace"""
        # Create concept nodes for core components
        self.concept_perception = Node("ConceptNode", "Perception")
        self.concept_reasoning = Node("ConceptNode", "Reasoning")
        self.concept_action = Node("ConceptNode", "Action")
        self.concept_learning = Node("ConceptNode", "Learning")
        self.concept_emotion = Node("ConceptNode", "Emotion")
        self.concept_attention = Node("ConceptNode", "Attention")
        self.concept_goal = Node("ConceptNode", "Goal")
        
        # Add to AtomSpace
        for concept in [
            self.concept_perception, self.concept_reasoning, 
            self.concept_action, self.concept_learning,
            self.concept_emotion, self.concept_attention,
            self.concept_goal
        ]:
            self.atomspace.add(concept)
    
    def cognitive_cycle(self, sensory_input: SensoryInput, reward: float = 0.0) -> Optional[Action]:
        """Execute one cognitive cycle with 4E cognition and learning"""
        # Store current state for learning
        current_state = self.state.attention_focus
        
        # Process through Vervaeke 4E Cognition Framework
        environmental_context = self._extract_environmental_context(sensory_input)
        vervaeke_results = self.vervaeke_framework.process_4e_cognition(
            sensory_input, 
            self.state.last_action, 
            environmental_context
        )
        
        # Store 4E cognition results in state for cross-domain reasoning
        self.state.vervaeke_4e_state = vervaeke_results
        
        # Process cycle with enhanced 4E awareness
        self._perceive(sensory_input, vervaeke_results)
        self._reason()
        action = self._act()
        
        # Update rewards
        self.state.last_reward = reward
        self.state.total_reward += reward
        
        # Learn from experience if we have a previous action
        if action and self.state.last_action:
            experience = Experience(
                state=current_state,
                action=self.state.last_action.name,
                reward=reward,
                next_state=self.state.attention_focus,
                done=False  # Could be based on goal achievement
            )
            
            # Update learning system
            learning_stats = self.learner.learn(experience)
            self.state.working_memory['learning_stats'] = learning_stats
            
            # Meta-learning integration: Create training data for meta-learning
            meta_training_data = [(current_state, reward)]
            meta_results = self.meta_learner.learn_meta_task(
                domain=self.current_domain,
                task=self.current_task,
                training_data=meta_training_data
            )
            
            # Update working memory with meta-learning results
            self.state.working_memory['meta_learning_stats'] = meta_results
            
            # Store experience in memory for later batch learning
            self.memory.save_experience(
                state=self.state,
                action=self.state.last_action,
                reward=reward,
                next_state=self.state
            )
            
            # Update exploration rate
            self.learner.update_exploration()
        
        # Store cognitive state in memory
        self.memory.store_cognitive_state(f"state_{np.random.randint(10000)}", self.state)
        
        # Trigger callbacks
        for callback in self._cycle_callbacks:
            callback(sensory_input, reward, action)
        
        return action
    
    def _perceive(self, sensory_input: SensoryInput, vervaeke_results: Optional[Dict] = None) -> None:
        """Enhanced perception phase with 4E cognition and cross-modal integration"""
        # Process sensory input through enhanced perception module
        attended_features, attention_weights, processing_info = self.perception.process_input(sensory_input)
        
        # Integrate Vervaeke 4E cognition results into perception
        if vervaeke_results:
            # Use embodied cognition to enhance sensorimotor integration
            if 'embodied' in vervaeke_results:
                embodied_state = vervaeke_results['embodied']['embodied_state']
                # Modulate attention based on embodied state - handle dimension mismatch
                if embodied_state.numel() >= attention_weights.shape[0]:
                    embodied_modulation = embodied_state[:attention_weights.shape[0]]
                else:
                    padding = torch.zeros(attention_weights.shape[0] - embodied_state.numel())
                    embodied_modulation = torch.cat([embodied_state, padding])
                
                attention_weights = attention_weights + 0.1 * embodied_modulation
                processing_info['embodied_integration'] = True
            
            # Use embedded cognition for context-aware perception
            if 'embedded' in vervaeke_results:
                environmental_context = vervaeke_results['embedded']['environmental_context']
                affordances = vervaeke_results['embedded']['affordances']
                processing_info['environmental_affordances'] = affordances
                processing_info['context_stability'] = vervaeke_results['embedded']['context_stability']
            
            # Use salience landscape for attention guidance
            if 'attention_focus' in vervaeke_results and vervaeke_results['attention_focus']:
                processing_info['vervaeke_attention_focus'] = vervaeke_results['attention_focus']
                processing_info['salience_landscape'] = vervaeke_results['salience_landscape']
        
        # Prepare multi-modal inputs for cross-domain processing
        multimodal_inputs = {}
        
        # Convert sensory input to modality-specific features
        if hasattr(sensory_input, 'visual') and sensory_input.visual is not None:
            multimodal_inputs[ModalityType.VISION] = sensory_input.visual.numpy() if hasattr(sensory_input.visual, 'numpy') else sensory_input.visual
        
        if hasattr(sensory_input, 'auditory') and sensory_input.auditory is not None:
            multimodal_inputs[ModalityType.HEARING] = sensory_input.auditory.numpy() if hasattr(sensory_input.auditory, 'numpy') else sensory_input.auditory
        
        # Process through multi-modal knowledge graph
        if multimodal_inputs:
            cross_modal_results = self.multimodal_kg.process_multimodal_input(
                multimodal_inputs,
                context={'attention_weights': attention_weights, 'processing_info': processing_info}
            )
            
            # Store cross-modal results in sensory buffer
            processing_info['cross_modal_results'] = cross_modal_results
        
        # Update cognitive state with enhanced information
        self.state.attention_focus = attention_weights
        self.state.sensory_buffer = {
            'attended_features': attended_features,
            'raw_input': sensory_input,
            'processing_info': processing_info
        }
        
        # Create atoms for perception in AtomSpace
        perception_node = Node("PerceptionNode", f"perception_{np.random.randint(10000)}")
        self.atomspace.add(perception_node)
        
        # Create link between perception and attended features
        features_list = attended_features.tolist() if hasattr(attended_features, 'tolist') else attended_features
        features_str = str(features_list)[:100]  # Truncate for readability
        features_node = Node("ConceptNode", f"features_{features_str}")
        self.atomspace.add(features_node)
        
        # Link perception to features
        perception_link = Link("EvaluationLink", [
            self.concept_perception,
            Link("ListLink", [perception_node, features_node])
        ])
        self.atomspace.add(perception_link)
        
        # Store enhanced processing information
        if processing_info.get('cross_modal_integration', False):
            cross_modal_node = Node("ConceptNode", "cross_modal_integration")
            self.atomspace.add(cross_modal_node)
            integration_link = Link("EvaluationLink", [
                cross_modal_node, perception_node
            ])
            self.atomspace.add(integration_link)
        
        # Add cross-modal correspondences to AtomSpace
        if 'cross_modal_results' in processing_info:
            correspondences = processing_info['cross_modal_results'].get('correspondences', [])
            for correspondence in correspondences:
                corr_node = Node("CorrespondenceNode", correspondence.correspondence_id)
                self.atomspace.add(corr_node)
                
                # Link correspondence to perception
                corr_link = Link("EvaluationLink", [
                    Node("ConceptNode", "cross_modal_correspondence"),
                    Link("ListLink", [perception_node, corr_node])
                ])
                self.atomspace.add(corr_link)
    
    def _reason(self) -> None:
        """Enhanced reasoning phase with cross-domain inference"""
        # Get attended features from sensory buffer
        attended_features = self.state.sensory_buffer['attended_features']
        
        # Process through reasoning module
        thought, updated_memory = self.reasoning(
            attended_features,
            self.state.working_memory
        )
        
        # Perform cross-domain reasoning if we have active domains
        cross_domain_inferences = []
        if len(self.cross_domain_framework.active_domains) > 1:
            # Extract concepts from thought content
            thought_concepts = self._extract_concepts_from_thought(thought.content)
            
            # Attempt cross-domain reasoning
            active_domains = list(self.cross_domain_framework.active_domains)
            for i, source_domain in enumerate(active_domains):
                for target_domain in active_domains[i+1:]:
                    # Try different reasoning types
                    for reasoning_type in [ReasoningType.ANALOGICAL, ReasoningType.CAUSAL, ReasoningType.DEDUCTIVE]:
                        inferences = self.reasoning_engine.make_cross_domain_inference(
                            reasoning_type=reasoning_type,
                            source_domain=source_domain,
                            target_domain=target_domain,
                            source_facts=thought_concepts,
                            context={'current_thought': thought.content, 'salience': thought.salience}
                        )
                        cross_domain_inferences.extend(inferences)
        
        # Integrate cross-domain inferences into thought
        if cross_domain_inferences:
            # Enhance thought content with cross-domain insights
            inference_summary = self._summarize_inferences(cross_domain_inferences)
            enhanced_content = f"{thought.content} | Cross-domain insights: {inference_summary}"
            
            # Update thought with enhanced content and increased salience
            thought.content = enhanced_content
            thought.salience = min(1.0, thought.salience + 0.1 * len(cross_domain_inferences))
            
            # Store inferences in working memory
            updated_memory['cross_domain_inferences'] = cross_domain_inferences
        
        # Update cognitive state
        self.state.current_thought = thought
        self.state.working_memory = updated_memory
        
        # Update emotional valence based on thought salience and rewards
        self.state.emotional_valence = (
            self.state.emotional_valence * 0.7 +  # Decay factor
            thought.salience * 0.2 +  # Thought contribution
            np.tanh(self.state.last_reward) * 0.1  # Reward contribution
        )
        
        # Create atoms for thought in AtomSpace
        thought_node = Node("ThoughtNode", f"thought_{thought.content[:50]}")
        self.atomspace.add(thought_node)
        
        # Create link between thought and its content
        content_node = Node("ConceptNode", f"content_{thought.content[:50]}")
        self.atomspace.add(content_node)
        
        # Link thought to content
        thought_link = Link("EvaluationLink", [
            self.concept_reasoning,
            Link("ListLink", [thought_node, content_node])
        ])
        self.atomspace.add(thought_link)
        
        # Add cross-domain inference atoms
        if cross_domain_inferences:
            for inference in cross_domain_inferences:
                inference_node = Node("InferenceNode", inference.inference_id)
                self.atomspace.add(inference_node)
                
                # Link inference to thought
                inference_link = Link("EvaluationLink", [
                    Node("ConceptNode", "cross_domain_inference"),
                    Link("ListLink", [thought_node, inference_node])
                ])
                self.atomspace.add(inference_link)
                
                # Add inference details
                source_domain_node = Node("DomainNode", inference.source_domain.value)
                target_domain_node = Node("DomainNode", inference.target_domain.value)
                reasoning_type_node = Node("ReasoningTypeNode", inference.reasoning_type.value)
                
                self.atomspace.add(source_domain_node)
                self.atomspace.add(target_domain_node)
                self.atomspace.add(reasoning_type_node)
                
                # Create detailed inference structure
                inference_structure = Link("EvaluationLink", [
                    Node("ConceptNode", "inference_structure"),
                    Link("ListLink", [
                        inference_node,
                        source_domain_node,
                        target_domain_node,
                        reasoning_type_node
                    ])
                ])
                self.atomspace.add(inference_structure)
        
        # Store thought in memory
        self.memory.store(f"thought_{np.random.randint(10000)}", {
            "content": thought.content,
            "salience": thought.salience,
            "timestamp": np.datetime64('now'),
            "cross_domain_inferences": len(cross_domain_inferences)
        })
        
        # Extract facts from thought content using memory system
        if hasattr(self.memory, 'extract_facts'):
            facts = self.memory.extract_facts(thought.content)
            if facts:
                # Store extracted facts in working memory
                self.state.working_memory['extracted_facts'] = facts
    
    def _extract_concepts_from_thought(self, thought_content: torch.Tensor) -> List[str]:
        """Extract key concepts from thought content for cross-domain reasoning"""
        # Convert tensor to string representation for concept extraction
        # In practice, this would use proper NLP techniques
        if isinstance(thought_content, torch.Tensor):
            # Use tensor statistics as proxy for concepts
            tensor_stats = {
                'mean': float(thought_content.mean()),
                'std': float(thought_content.std()),
                'max': float(thought_content.max()),
                'min': float(thought_content.min())
            }
            # Generate concept words based on tensor characteristics
            concepts = []
            if tensor_stats['mean'] > 0.02:
                concepts.append('activation')
            if tensor_stats['std'] > 0.05:
                concepts.append('variability')
            if tensor_stats['max'] > 0.1:
                concepts.append('salience')
            if tensor_stats['min'] < -0.1:
                concepts.append('inhibition')
            return concepts if concepts else ['neutral']
        else:
            # Fallback for string input
            words = str(thought_content).lower().split()
        
        # Filter for meaningful concepts (exclude common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return unique concepts
        return list(set(concepts))
    
    def _summarize_inferences(self, inferences) -> str:
        """Create a summary of cross-domain inferences"""
        if not inferences:
            return "None"
        
        summary_parts = []
        reasoning_counts = {}
        
        for inference in inferences:
            reasoning_type = inference.reasoning_type.value
            reasoning_counts[reasoning_type] = reasoning_counts.get(reasoning_type, 0) + 1
        
        for reasoning_type, count in reasoning_counts.items():
            summary_parts.append(f"{count} {reasoning_type}")
        
        return ", ".join(summary_parts)
    
    def _act(self) -> Optional[Action]:
        """Action phase of the cognitive cycle with learning influence"""
        if self.state.current_thought is None:
            return None
            
        # Get action suggestion from learner
        learner_action, confidence = self.learner.select_action(
            self.state.current_thought.content
        )
        
        # Combine with action selector
        selected_action = self.action_selector(
            self.state.current_thought.content,
            self.state.goal_stack,
            self.state.emotional_valence
        )
        
        # Use learner's suggestion if confidence is high enough
        if selected_action and confidence > 0.8:
            selected_action.name = learner_action
            selected_action.confidence = confidence
        
        # Update cognitive state
        self.state.last_action = selected_action
        
        # Create atoms for action in AtomSpace
        if selected_action:
            action_node = Node("ActionNode", f"action_{selected_action.name}")
            self.atomspace.add(action_node)
            
            # Create link between action and its parameters
            params_str = str(selected_action.parameters)[:50]
            params_node = Node("ConceptNode", f"params_{params_str}")
            self.atomspace.add(params_node)
            
            # Link action to parameters
            action_link = Link("EvaluationLink", [
                self.concept_action,
                Link("ListLink", [action_node, params_node])
            ])
            self.atomspace.add(action_link)
            
            # Create link between thought and action (causality)
            thought_node = Node("ThoughtNode", f"thought_{self.state.current_thought.content[:50]}")
            causality_link = Link("CausalLink", [thought_node, action_node])
            self.atomspace.add(causality_link)
        
        return selected_action
    
    def update_goals(self, new_goal: str) -> None:
        """Update the system's goal stack"""
        self.state.goal_stack.append(new_goal)
        
        # Create atoms for goal in AtomSpace
        goal_node = Node("GoalNode", f"goal_{new_goal}")
        self.atomspace.add(goal_node)
        
        # Create link between goal concept and this goal
        goal_link = Link("EvaluationLink", [
            self.concept_goal,
            goal_node
        ])
        self.atomspace.add(goal_link)
        
        # Store goal in memory
        self.memory.store(f"goal_{np.random.randint(10000)}", {
            "content": new_goal,
            "timestamp": np.datetime64('now'),
            "active": True
        })
    
    def get_cognitive_state(self) -> CognitiveState:
        """Return current cognitive state"""
        return self.state
    
    def register_cycle_callback(self, callback: Callable) -> None:
        """Register a callback to be called after each cognitive cycle
        
        Args:
            callback: Function to call with signature (sensory_input, reward, action)
        """
        self._cycle_callbacks.append(callback)
    
    def unregister_cycle_callback(self, callback: Callable) -> bool:
        """Unregister a previously registered callback
        
        Args:
            callback: The callback to unregister
            
        Returns:
            True if the callback was found and removed, False otherwise
        """
        if callback in self._cycle_callbacks:
            self._cycle_callbacks.remove(callback)
            return True
        return False
    
    def query_knowledge(self, pattern: Any) -> List[Any]:
        """Query the AtomSpace for knowledge matching a pattern
        
        Args:
            pattern: Query pattern (can be an Atom or a dict for advanced queries)
            
        Returns:
            List of matching results
        """
        if isinstance(pattern, dict):
            return self.atomspace.pattern_match(pattern)
        else:
            return self.atomspace.query(pattern)
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search memory for semantically similar content
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of (key, value, similarity_score) tuples
        """
        return self.memory.semantic_search(query, limit)
    
    def save_state(self, path: str) -> bool:
        """Save the current cognitive state to persistent storage
        
        Args:
            path: Path to save the state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save cognitive state to memory
            state_id = f"saved_state_{np.random.randint(10000)}"
            self.memory.store_cognitive_state(state_id, self.state)
            
            # Store the state ID for later retrieval
            self.memory.store("last_saved_state", state_id)
            
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, path: str = None) -> bool:
        """Load a previously saved cognitive state
        
        Args:
            path: Path to load the state from, or None to load the last saved state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if path is None:
                # Retrieve the last saved state ID
                state_id = self.memory.retrieve("last_saved_state")
                if not state_id:
                    return False
            else:
                state_id = path
            
            # Load cognitive state from memory
            loaded_state = self.memory.retrieve_cognitive_state(state_id)
            if loaded_state:
                self.state = loaded_state
                return True
            
            return False
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def set_domain_and_task(self, domain: str, task: str) -> None:
        """Set the current domain and task for meta-learning optimization
        
        Args:
            domain: The domain identifier (e.g., 'vision', 'language', 'robotics')
            task: The specific task within the domain
        """
        self.current_domain = domain
        self.current_task = task
        
        # Register domain with transfer learning if not already registered
        if domain not in self.meta_learner.transfer_learning.domain_knowledge:
            # Use attention focus size as feature dimension
            feature_dim = self.state.attention_focus.size(-1)
            self.meta_learner.transfer_learning.register_domain(domain, feature_dim)
    
    def trigger_meta_learning_batch(self, batch_data: List[Tuple[torch.Tensor, Any]],
                                   validation_data: List[Tuple[torch.Tensor, Any]] = None) -> Dict[str, Any]:
        """Trigger a meta-learning batch update with accumulated data
        
        Args:
            batch_data: List of (input, target) pairs for training
            validation_data: Optional validation data for evaluation
            
        Returns:
            Meta-learning results and statistics
        """
        return self.meta_learner.learn_meta_task(
            domain=self.current_domain,
            task=self.current_task,
            training_data=batch_data,
            validation_data=validation_data
        )
    
    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics and performance metrics
        
        Returns:
            Dictionary containing meta-learning performance statistics
        """
        return self.meta_learner.get_meta_learning_stats()
    
    def optimize_learning_strategy(self, context: Dict[str, Any] = None) -> LearningStrategy:
        """Get the optimal learning strategy for the current context
        
        Args:
            context: Optional context information to guide strategy selection
            
        Returns:
            The recommended learning strategy
        """
        if context is None:
            context = {
                'domain': self.current_domain,
                'task': self.current_task,
                'current_performance': self.state.total_reward / max(1, abs(self.state.total_reward)) if self.state.total_reward != 0 else 0.5
            }
        
        return self.meta_learner.adaptive_manager.select_strategy(context)
    
    def transfer_knowledge_from_domain(self, source_domain: str, adaptation_rate: float = 0.1) -> bool:
        """Transfer knowledge from a source domain to the current domain
        
        Args:
            source_domain: The domain to transfer knowledge from
            adaptation_rate: Rate of knowledge adaptation (0.0 to 1.0)
            
        Returns:
            True if transfer was successful, False otherwise
        """
        if self.current_domain == source_domain:
            return False
            
        # Ensure both domains are registered
        current_feature_dim = self.state.attention_focus.size(-1)
        
        if source_domain not in self.meta_learner.transfer_learning.domain_knowledge:
            self.meta_learner.transfer_learning.register_domain(source_domain, current_feature_dim)
            
        if self.current_domain not in self.meta_learner.transfer_learning.domain_knowledge:
            self.meta_learner.transfer_learning.register_domain(self.current_domain, current_feature_dim)
        
        # Perform knowledge transfer
        return self.meta_learner.transfer_learning.transfer_knowledge(
            source_domain, self.current_domain, adaptation_rate
        )
    
    def create_few_shot_prototype(self, task_name: str, examples: List[torch.Tensor],
                                labels: List[int] = None) -> bool:
        """Create a few-shot learning prototype for quick task adaptation
        
        Args:
            task_name: Name of the task to create prototype for
            examples: List of example inputs
            labels: Optional labels for the examples
            
        Returns:
            True if prototype was created successfully
        """
        if not examples:
            return False
            
        # Default labels if none provided
        if labels is None:
            labels = [0] * len(examples)
            
        try:
            prototype = self.meta_learner.few_shot_learner.create_prototype(
                task_name, examples, labels
            )
            return True
        except Exception as e:
            print(f"Error creating few-shot prototype: {e}")
            return False
    
    def adapt_learning_rate(self, recent_performance: List[float]) -> float:
        """Adapt the learning rate based on recent performance trends
        
        Args:
            recent_performance: List of recent performance scores
            
        Returns:
            Adapted learning rate
        """
        current_lr = self.learner.optimizer.param_groups[0]['lr']
        new_lr = self.meta_learner.adaptive_manager.adapt_learning_rate(
            current_lr, recent_performance
        )
        
        # Update the learner's learning rate
        for param_group in self.learner.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get current curriculum learning progress
        
        Returns:
            Dictionary containing curriculum progress information
        """
        return self.meta_learner.curriculum_learner.get_curriculum_progress()
    
    def add_curriculum_level(self, level_id: str, difficulty: float,
                           tasks: List[Any], prerequisites: List[str] = None):
        """Add a level to the curriculum learning system
        
        Args:
            level_id: Unique identifier for the curriculum level
            difficulty: Difficulty score (0.0 to 1.0)
            tasks: List of tasks for this level
            prerequisites: List of prerequisite level IDs
        """
        self.meta_learner.curriculum_learner.add_curriculum_level(
            level_id, difficulty, tasks, prerequisites
        )
    
    def get_next_curriculum_task(self, current_performance: float) -> Optional[Any]:
        """Get the next task from the curriculum based on current performance
        
        Args:
            current_performance: Current performance score (0.0 to 1.0)
            
        Returns:
            Next curriculum task or None if curriculum is complete
        """
        return self.meta_learner.curriculum_learner.get_next_task(current_performance)
    
    def save_meta_knowledge(self, filepath: str) -> bool:
        """Save accumulated meta-learning knowledge to persistent storage
        
        Args:
            filepath: Path to save the meta-knowledge
            
        Returns:
            True if successful, False otherwise
        """
        return self.meta_learner.save_meta_knowledge(filepath)
    
    def load_meta_knowledge(self, filepath: str) -> bool:
        """Load previously saved meta-learning knowledge
        
        Args:
            filepath: Path to load the meta-knowledge from
            
        Returns:
            True if successful, False otherwise  
        """
        return self.meta_learner.load_meta_knowledge(filepath)
    
    def _extract_environmental_context(self, sensory_input: SensoryInput) -> Dict[str, Any]:
        """Extract environmental context for 4E cognition processing"""
        context = {}
        
        # Extract context from sensory input
        if sensory_input.visual is not None:
            visual_stats = {
                'visual_intensity': float(torch.mean(torch.abs(sensory_input.visual))),
                'visual_complexity': float(torch.std(sensory_input.visual)),
                'visual_pattern': 'complex' if torch.std(sensory_input.visual) > 0.1 else 'simple'
            }
            context['visual_environment'] = visual_stats
            
        if sensory_input.auditory is not None:
            auditory_stats = {
                'audio_volume': float(torch.mean(torch.abs(sensory_input.auditory))),
                'audio_variability': float(torch.std(sensory_input.auditory)),
                'audio_pattern': 'dynamic' if torch.std(sensory_input.auditory) > 0.1 else 'static'
            }
            context['auditory_environment'] = auditory_stats
        
        # Add goal context for centrality computation
        if self.state.goal_stack:
            context['goals'] = [goal for goal in self.state.goal_stack[:3]]  # Top 3 goals
            context['urgency'] = 0.8  # High urgency if goals present
        else:
            context['urgency'] = 0.3  # Low urgency if no goals
            
        # Add attention history for temporal relevance
        if hasattr(self.state, 'attention_history'):
            context['attention_history'] = self.state.attention_history[-5:]  # Last 5 focus items
        
        return context
    
    # Cross-Domain Integration Methods
    
    def add_cross_domain_concept_mapping(self, source_domain: DomainType, target_domain: DomainType,
                                        source_concept: str, target_concept: str,
                                        mapping_strength: float = 0.8) -> bool:
        """Add a concept mapping between domains"""
        mapping = ConceptMapping(
            source_domain=source_domain,
            target_domain=target_domain,
            source_concept=source_concept,
            target_concept=target_concept,
            mapping_strength=mapping_strength,
            semantic_similarity=mapping_strength,  # Simple default
            bidirectional=True
        )
        
        return self.cross_domain_framework.unified_representation.add_concept_mapping(mapping)
    
    def register_abstract_concept(self, concept_name: str, 
                                 domain_instantiations: Dict[DomainType, List[str]],
                                 abstraction_level: int = 1) -> bool:
        """Register an abstract concept that spans multiple domains"""
        abstract_concept = AbstractConcept(
            concept_id=f"abstract_{concept_name}",
            concept_name=concept_name,
            abstraction_level=abstraction_level,
            domain_instantiations=domain_instantiations,
            semantic_features={f"feature_{i}": 0.5 for i in range(10)},  # Default features
            hierarchical_relations={},
            cross_domain_analogies=[]
        )
        
        return self.cross_domain_framework.unified_representation.register_abstract_concept(abstract_concept)
    
    def perform_cross_domain_reasoning(self, reasoning_type: ReasoningType,
                                     source_domain: DomainType, target_domain: DomainType,
                                     source_facts: List[str], context: Dict[str, Any] = None):
        """Perform cross-domain reasoning with specified parameters"""
        return self.reasoning_engine.make_cross_domain_inference(
            reasoning_type=reasoning_type,
            source_domain=source_domain,
            target_domain=target_domain,
            source_facts=source_facts,
            context=context
        )
    
    def transfer_concept_across_domains(self, source_domain: DomainType, target_domain: DomainType,
                                      concept: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Transfer knowledge about a concept between domains"""
        return self.cross_domain_framework.transfer_knowledge(
            source_domain, target_domain, concept, context
        )
    
    def process_multimodal_input(self, inputs: Dict[ModalityType, Any],
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process multi-modal input through the cross-domain framework"""
        return self.multimodal_kg.process_multimodal_input(inputs, context)
    
    def query_cross_modal_knowledge(self, query_modality: ModalityType, 
                                   query_data: Any, target_modality: ModalityType,
                                   similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Query knowledge using one modality and retrieve from another"""
        # Extract features from query data
        if query_modality in self.multimodal_kg.modality_processors:
            processor = self.multimodal_kg.modality_processors[query_modality]
            features = processor.extract_features(query_data)
            
            if features:
                query_features = features[0].feature_vector
                return self.multimodal_kg.query_cross_modal(
                    query_modality, query_features, target_modality, similarity_threshold
                )
        
        return []
    
    def get_cross_domain_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive status of cross-domain integration"""
        framework_status = self.cross_domain_framework.get_integration_status()
        reasoning_stats = self.reasoning_engine.get_reasoning_statistics()
        multimodal_stats = self.multimodal_kg.get_integration_statistics()
        consistency_scores = self.reasoning_engine.validate_inference_consistency()
        
        return {
            'framework_status': framework_status,
            'reasoning_statistics': reasoning_stats,
            'multimodal_statistics': multimodal_stats,
            'consistency_validation': consistency_scores,
            'overall_health': self._assess_cross_domain_health(
                framework_status, reasoning_stats, consistency_scores
            )
        }
    
    def _assess_cross_domain_health(self, framework_status: Dict[str, Any],
                                  reasoning_stats: Dict[str, Any],
                                  consistency_scores: Dict[str, float]) -> Dict[str, str]:
        """Assess overall health of cross-domain integration"""
        health = {}
        
        # Framework health
        framework_health = framework_status.get('framework_health', {}).get('overall', 'poor')
        health['framework'] = framework_health
        
        # Reasoning health
        if reasoning_stats['total_inferences'] > 10:
            avg_strength = reasoning_stats['average_strength']
            if avg_strength > 0.7:
                health['reasoning'] = 'excellent'
            elif avg_strength > 0.5:
                health['reasoning'] = 'good'
            elif avg_strength > 0.3:
                health['reasoning'] = 'fair'
            else:
                health['reasoning'] = 'poor'
        else:
            health['reasoning'] = 'insufficient_data'
        
        # Consistency health
        overall_consistency = consistency_scores.get('overall_consistency', 0.0)
        if overall_consistency > 0.8:
            health['consistency'] = 'excellent'
        elif overall_consistency > 0.6:
            health['consistency'] = 'good'
        elif overall_consistency > 0.4:
            health['consistency'] = 'fair'
        else:
            health['consistency'] = 'poor'
        
        # Overall assessment
        health_values = [health['framework'], health['reasoning'], health['consistency']]
        excellent_count = health_values.count('excellent')
        good_count = health_values.count('good')
        
        if excellent_count >= 2:
            health['overall'] = 'excellent'
        elif excellent_count + good_count >= 2:
            health['overall'] = 'good'
        elif 'poor' not in health_values:
            health['overall'] = 'fair'
        else:
            health['overall'] = 'poor'
        
        return health
    
    def validate_cross_domain_knowledge_consistency(self) -> Dict[str, Any]:
        """Validate consistency of cross-domain knowledge"""
        framework_consistency = self.cross_domain_framework.validate_cross_domain_consistency()
        reasoning_consistency = self.reasoning_engine.validate_inference_consistency()
        
        return {
            'framework_consistency': framework_consistency,
            'reasoning_consistency': reasoning_consistency,
            'combined_score': (
                framework_consistency.get('overall_consistency', 0.0) +
                reasoning_consistency.get('overall_consistency', 0.0)
            ) / 2,
            'recommendations': self._get_consistency_recommendations(
                framework_consistency, reasoning_consistency
            )
        }
    
    def _get_consistency_recommendations(self, framework_consistency: Dict[str, float],
                                       reasoning_consistency: Dict[str, float]) -> List[str]:
        """Get recommendations for improving consistency"""
        recommendations = []
        
        # Framework consistency recommendations
        if framework_consistency.get('concept_mapping_consistency', 0.0) < 0.6:
            recommendations.append("Review and validate concept mappings across domains")
        
        if framework_consistency.get('adaptation_accuracy', 0.0) < 0.7:
            recommendations.append("Improve domain adaptation algorithms with more training data")
        
        if framework_consistency.get('cross_modal_coherence', 0.0) < 0.6:
            recommendations.append("Strengthen cross-modal bindings and correspondences")
        
        # Reasoning consistency recommendations
        if reasoning_consistency.get('logical_consistency', 0.0) < 0.7:
            recommendations.append("Review inference rules to reduce logical contradictions")
        
        if reasoning_consistency.get('temporal_consistency', 0.0) < 0.8:
            recommendations.append("Implement better temporal reasoning constraints")
        
        if reasoning_consistency.get('strength_consistency', 0.0) < 0.6:
            recommendations.append("Calibrate inference strength calculations")
        
        if not recommendations:
            recommendations.append("Cross-domain integration is performing well")
        
        return recommendations
