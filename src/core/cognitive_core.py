import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from .state import CognitiveState
from ..modules.perception import PerceptionModule, SensoryInput
from ..modules.reasoning import ReasoningModule, Thought
from ..modules.action import ActionSelectionModule, Action
from ..modules.learning import ReinforcementLearner, Experience

# Import the new AtomSpace and Memory modules
from ..atomspace import AtomSpace, Node, Link, BackendType
from ..memory import Memory

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
        
        # Initialize AtomSpace with the specified backend
        atomspace_backend = self.config.get('atomspace_backend', 'local')
        atomspace_config = self.config.get('atomspace_config', {})
        self.atomspace = AtomSpace(backend_type=atomspace_backend, config=atomspace_config)
        
        # Initialize Memory with the specified backend
        memory_backend = self.config.get('memory_backend', 'mem0')
        memory_config = self.config.get('memory_config', {})
        self.memory = Memory(backend_type=memory_backend, config=memory_config)
        
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
        """Execute one cognitive cycle with learning"""
        # Store current state for learning
        current_state = self.state.attention_focus
        
        # Process cycle
        self._perceive(sensory_input)
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
    
    def _perceive(self, sensory_input: SensoryInput) -> None:
        """Enhanced perception phase of the cognitive cycle"""
        # Process sensory input through enhanced perception module
        attended_features, attention_weights, processing_info = self.perception.process_input(sensory_input)
        
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
    
    def _reason(self) -> None:
        """Reasoning phase of the cognitive cycle"""
        # Get attended features from sensory buffer
        attended_features = self.state.sensory_buffer['attended_features']
        
        # Process through reasoning module
        thought, updated_memory = self.reasoning(
            attended_features,
            self.state.working_memory
        )
        
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
        
        # Store thought in memory
        self.memory.store(f"thought_{np.random.randint(10000)}", {
            "content": thought.content,
            "salience": thought.salience,
            "timestamp": np.datetime64('now')
        })
        
        # Extract facts from thought content using memory system
        if hasattr(self.memory, 'extract_facts'):
            facts = self.memory.extract_facts(thought.content)
            if facts:
                # Store extracted facts in working memory
                self.state.working_memory['extracted_facts'] = facts
    
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
