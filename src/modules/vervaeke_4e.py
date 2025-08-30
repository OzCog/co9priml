"""
Vervaeke 4E Cognition Framework Integration

This module implements John Vervaeke's 4E cognition framework:
- Embodied: Cognition grounded in sensorimotor experience
- Embedded: Cognition shaped by environmental context
- Enacted: Cognition through active engagement with world
- Extended: Cognition distributed across tools and environment

The implementation preserves theoretical rigor while making the concepts
computationally tractable within the CogPrime architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..modules.perception import SensoryInput
from ..modules.action import Action


class CognitionMode(Enum):
    """4E Cognition modes following Vervaeke's framework"""
    EMBODIED = "embodied"      # Sensorimotor grounding
    EMBEDDED = "embedded"      # Environmental context
    ENACTED = "enacted"        # Active engagement
    EXTENDED = "extended"      # Tool/environment coupling


class KnowingMode(Enum):
    """Vervaeke's four kinds of knowing"""
    PROPOSITIONAL = "propositional"  # Knowing that
    PROCEDURAL = "procedural"        # Knowing how
    PERSPECTIVAL = "perspectival"    # Knowing what it's like
    PARTICIPATORY = "participatory"   # Knowing by participating


@dataclass
class SalienceVector:
    """Three-dimensional salience vector using ACT framework"""
    aspectuality: float  # How something is aspectualized/configured
    centrality: float   # How central/important to the agent
    temporality: float  # Temporal relevance/urgency


@dataclass
class CognitiveFrame:
    """A cognitive frame represents a particular way of seeing/understanding"""
    salience_weights: Dict[str, float]  # What aspects are considered relevant
    active_knowing_modes: List[KnowingMode]
    context: Dict[str, Any]
    affordances: List[str]  # Available action possibilities


class EmbodiedCognitionModule(nn.Module):
    """
    Implements embodied cognition principles - cognition grounded in 
    sensorimotor experience and bodily interaction with the world.
    """
    
    def __init__(self, sensory_dim: int = 512, motor_dim: int = 256):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        
        # Sensorimotor integration network
        self.sensorimotor_integration = nn.Sequential(
            nn.Linear(sensory_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, motor_dim)
        )
        
        # Body schema - internal model of body capabilities
        self.body_schema = nn.Parameter(torch.randn(motor_dim))
        
        # Proprioceptive state tracker
        self.proprioception = nn.GRU(motor_dim, 64, batch_first=True)
        self.proprioceptive_state = None
        
    def forward(self, sensory_input: torch.Tensor, 
                current_action: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process embodied cognition through sensorimotor integration
        
        Args:
            sensory_input: Current sensory information
            current_action: Current motor action state
            
        Returns:
            Dict containing embodied processing results
        """
        # Integrate sensory input with body schema
        motor_prediction = self.sensorimotor_integration(sensory_input)
        embodied_state = motor_prediction + self.body_schema
        
        # Update proprioceptive awareness
        if current_action is not None:
            # Ensure action has correct dimension for proprioception
            if current_action.numel() != self.motor_dim:
                if current_action.numel() > self.motor_dim:
                    action_input = current_action[:self.motor_dim]
                else:
                    padding = torch.zeros(self.motor_dim - current_action.numel())
                    action_input = torch.cat([current_action, padding])
            else:
                action_input = current_action
                
            prop_input = action_input.unsqueeze(0) if action_input.dim() == 1 else action_input
            prop_output, self.proprioceptive_state = self.proprioception(
                prop_input.unsqueeze(0), self.proprioceptive_state
            )
        
        return {
            'embodied_state': embodied_state,
            'motor_prediction': motor_prediction,
            'proprioceptive_awareness': self.proprioceptive_state.squeeze(0) if self.proprioceptive_state is not None else torch.zeros(64),
            'body_schema': self.body_schema
        }


class EmbeddedCognitionModule(nn.Module):
    """
    Implements embedded cognition - cognition shaped by environmental 
    context and situated within specific ecological niches.
    """
    
    def __init__(self, context_dim: int = 256, feature_dim: int = 512):
        super().__init__()
        self.context_dim = context_dim
        self.feature_dim = feature_dim
        
        # Context encoding network
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, context_dim)
        )
        
        # Environmental affordance detector
        self.affordance_detector = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Predict available affordances
            nn.Sigmoid()
        )
        
        # Context history for temporal embedding
        self.context_history = []
        self.max_history = 10
        
    def forward(self, environmental_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process embedded cognition through environmental context analysis
        
        Args:
            environmental_features: Features extracted from environment
            
        Returns:
            Dict containing embedded processing results
        """
        # Encode current environmental context
        current_context = self.context_encoder(environmental_features)
        
        # Detect available affordances
        affordances = self.affordance_detector(current_context)
        
        # Update context history
        self.context_history.append(current_context.detach().clone())
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)
        
        # Compute contextual stability
        context_stability = 1.0
        if len(self.context_history) > 1:
            recent_contexts = torch.stack(self.context_history[-3:])
            context_stability = 1.0 - torch.std(recent_contexts, dim=0).mean().item()
        
        return {
            'environmental_context': current_context,
            'affordances': affordances,
            'context_stability': torch.tensor(context_stability),
            'context_history_length': torch.tensor(len(self.context_history))
        }


class EnactedCognitionModule(nn.Module):
    """
    Implements enacted cognition - cognition through active engagement 
    and exploration of the world, where knowing emerges through doing.
    """
    
    def __init__(self, action_dim: int = 64, perception_dim: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.perception_dim = perception_dim
        
        # Action-perception coupling network
        self.action_perception_coupling = nn.Sequential(
            nn.Linear(action_dim + perception_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Exploration motivation calculator
        self.exploration_motivation = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Enactive knowledge accumulator
        self.enactive_knowledge = nn.Parameter(torch.zeros(64))
        self.knowledge_decay = 0.95
        
    def forward(self, action_intention: torch.Tensor, 
                perceptual_consequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process enacted cognition through action-perception coupling
        
        Args:
            action_intention: Intended action vector
            perceptual_consequence: Resulting perceptual state
            
        Returns:
            Dict containing enacted processing results
        """
        # Couple action and perception
        action_perception_input = torch.cat([action_intention, perceptual_consequence], dim=-1)
        coupling_state = self.action_perception_coupling(action_perception_input)
        
        # Calculate exploration motivation
        exploration_drive = self.exploration_motivation(coupling_state)
        
        # Update enactive knowledge through coupling
        knowledge_update = coupling_state.mean(dim=0) if coupling_state.dim() > 1 else coupling_state
        
        # Update parameter data directly to avoid assignment error
        with torch.no_grad():
            self.enactive_knowledge.data = self.knowledge_decay * self.enactive_knowledge.data + (1 - self.knowledge_decay) * knowledge_update.detach()
        
        return {
            'action_perception_coupling': coupling_state,
            'exploration_motivation': exploration_drive,
            'enactive_knowledge': self.enactive_knowledge,
            'coupling_strength': torch.norm(coupling_state).unsqueeze(0)
        }


class ExtendedCognitionModule(nn.Module):
    """
    Implements extended cognition - cognition distributed across tools,
    technology, and environmental structures that become part of the 
    cognitive system.
    """
    
    def __init__(self, tool_dim: int = 128, cognitive_dim: int = 256):
        super().__init__()
        self.tool_dim = tool_dim
        self.cognitive_dim = cognitive_dim
        
        # Tool integration network - fix dimension to match tool_dim + cognitive_dim
        total_input_dim = tool_dim + cognitive_dim  # 128 + 256 = 384
        self.tool_integration = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, cognitive_dim)
        )
        
        # External memory interface
        self.external_memory_interface = nn.Sequential(
            nn.Linear(cognitive_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, tool_dim)
        )
        
        # Tool affordance tracker
        self.available_tools = nn.Parameter(torch.zeros(10, tool_dim))  # Up to 10 tools
        self.tool_usage_history = torch.zeros(10)
        
    def forward(self, internal_cognitive_state: torch.Tensor,
                available_tools: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process extended cognition through tool and environment coupling
        
        Args:
            internal_cognitive_state: Current internal cognitive state
            available_tools: Available external tools/resources
            
        Returns:
            Dict containing extended processing results
        """
        if available_tools is None:
            available_tools = self.available_tools[:3]  # Use top 3 default tools
        
        # Select most relevant tool
        # Fix dimension mismatch by projecting cognitive state to tool dimension
        if internal_cognitive_state.numel() >= self.tool_dim:
            cognitive_state_proj = internal_cognitive_state[:self.tool_dim]
        else:
            padding = torch.zeros(self.tool_dim - internal_cognitive_state.numel())
            cognitive_state_proj = torch.cat([internal_cognitive_state, padding])
        
        # Make sure cognitive_state_proj is the right size
        cognitive_state_proj = cognitive_state_proj[:self.tool_dim]
        
        tool_relevance = torch.matmul(cognitive_state_proj.unsqueeze(0), available_tools.T)
        best_tool_idx = torch.argmax(tool_relevance)
        selected_tool = available_tools[best_tool_idx]
        
        # Integrate tool with internal cognition
        # Use full cognitive state for integration, not the truncated version
        full_cognitive_state = internal_cognitive_state
        if full_cognitive_state.numel() != self.cognitive_dim:
            if full_cognitive_state.numel() > self.cognitive_dim:
                full_cognitive_state = full_cognitive_state[:self.cognitive_dim]
            else:
                padding = torch.zeros(self.cognitive_dim - full_cognitive_state.numel())
                full_cognitive_state = torch.cat([full_cognitive_state, padding])
        
        tool_cognitive_input = torch.cat([selected_tool, full_cognitive_state], dim=-1)
        
        extended_state = self.tool_integration(tool_cognitive_input)
        
        # Generate external memory query
        external_query = self.external_memory_interface(extended_state)
        
        # Update tool usage
        self.tool_usage_history[best_tool_idx] += 1
        
        return {
            'extended_cognitive_state': extended_state,
            'selected_tool': selected_tool,
            'external_memory_query': external_query,
            'tool_usage_pattern': self.tool_usage_history.clone(),
            'cognitive_offloading': torch.norm(external_query).unsqueeze(0)
        }


class SalienceLandscapeNavigator:
    """
    Implements Vervaeke's salience landscape navigation for dynamic
    attention allocation and relevance realization.
    """
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.current_landscape: Dict[str, SalienceVector] = {}
        self.salience_threshold = 0.5
        self.attention_focus = None
        
    def update_salience(self, item_id: str, features: Dict[str, Any], 
                       context: Optional[Dict] = None) -> SalienceVector:
        """Update salience vector for an item"""
        # Compute aspectuality (how it's configured)
        aspectuality = self._compute_aspectuality(features)
        
        # Compute centrality (importance to agent)  
        centrality = self._compute_centrality(features, context)
        
        # Compute temporality (temporal relevance)
        temporality = self._compute_temporality(features, context)
        
        vector = SalienceVector(
            aspectuality=aspectuality,
            centrality=centrality, 
            temporality=temporality
        )
        self.current_landscape[item_id] = vector
        return vector
    
    def navigate_to_most_salient(self) -> Optional[str]:
        """Navigate attention to the most salient item"""
        if not self.current_landscape:
            return None
            
        max_salience = 0
        most_salient = None
        
        for item_id, vector in self.current_landscape.items():
            # Combined salience using ACT framework
            total_salience = (vector.aspectuality + vector.centrality + vector.temporality) / 3
            
            if total_salience > max_salience and total_salience > self.salience_threshold:
                max_salience = total_salience
                most_salient = item_id
                
        self.attention_focus = most_salient
        return most_salient
    
    def _compute_aspectuality(self, features: Dict[str, Any]) -> float:
        """Compute how the item is aspectualized/configured"""
        # Simple implementation - in practice would be more sophisticated
        if not features:
            return 0.0
        
        feature_values = [v for v in features.values() if isinstance(v, (int, float))]
        if not feature_values:
            return 0.5
            
        # Use feature variance as proxy for aspectuality
        return min(1.0, np.std(feature_values) * 2)
    
    def _compute_centrality(self, features: Dict[str, Any], context: Optional[Dict] = None) -> float:
        """Compute centrality/importance to the agent"""
        # Simple implementation - goal relevance  
        if context and 'goals' in context:
            goal_overlap = len(set(features.keys()).intersection(set(context['goals'])))
            return min(1.0, goal_overlap / max(1, len(context['goals'])))
        return 0.5
    
    def _compute_temporality(self, features: Dict[str, Any], context: Optional[Dict] = None) -> float:
        """Compute temporal relevance"""
        if context and 'urgency' in context:
            return min(1.0, context['urgency'])
        return 0.5


class Vervaeke4ECognitionFramework:
    """
    Integrated 4E Cognition Framework combining all four modes
    with salience landscape navigation and knowing modes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize 4E cognition modules
        self.embodied = EmbodiedCognitionModule()
        self.embedded = EmbeddedCognitionModule()
        self.enacted = EnactedCognitionModule()
        self.extended = ExtendedCognitionModule()
        
        # Initialize salience landscape navigator
        self.salience_navigator = SalienceLandscapeNavigator()
        
        # Current cognitive frame
        self.current_frame: Optional[CognitiveFrame] = None
        
        # Active knowing modes
        self.active_knowing_modes: Set[KnowingMode] = {KnowingMode.PERSPECTIVAL}
        
    def process_4e_cognition(self, sensory_input: SensoryInput, 
                           current_action: Optional[Action] = None,
                           environmental_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process input through the integrated 4E cognition framework
        
        Args:
            sensory_input: Current sensory information
            current_action: Current action being performed
            environmental_context: Environmental context information
            
        Returns:
            Integrated 4E cognition processing results
        """
        results = {}
        
        # Process through embodied cognition
        if sensory_input.visual is not None or sensory_input.auditory is not None:
            # Combine available sensory modalities
            sensory_tensor = self._combine_sensory_input(sensory_input)
            action_tensor = self._action_to_tensor(current_action) if current_action else None
            
            embodied_results = self.embodied(sensory_tensor, action_tensor)
            results['embodied'] = embodied_results
            
            # Process through embedded cognition
            embedded_results = self.embedded(sensory_tensor)
            results['embedded'] = embedded_results
            
            # Process through enacted cognition
            if action_tensor is not None:
                enacted_results = self.enacted(action_tensor, sensory_tensor)
                results['enacted'] = enacted_results
            
            # Process through extended cognition
            cognitive_state = embodied_results['embodied_state']
            extended_results = self.extended(cognitive_state)
            results['extended'] = extended_results
        
        # Update salience landscape
        if environmental_context:
            for item_id, item_features in environmental_context.items():
                if isinstance(item_features, dict):
                    self.salience_navigator.update_salience(item_id, item_features, environmental_context)
        
        # Navigate salience landscape
        most_salient = self.salience_navigator.navigate_to_most_salient()
        results['attention_focus'] = most_salient
        results['salience_landscape'] = dict(self.salience_navigator.current_landscape)
        
        return results
    
    def update_knowing_modes(self, new_modes: List[KnowingMode]):
        """Update active knowing modes"""
        self.active_knowing_modes = set(new_modes)
    
    def get_perspectival_knowing(self) -> Dict[str, Any]:
        """Generate perspectival knowing - what it's like perspective"""
        if KnowingMode.PERSPECTIVAL not in self.active_knowing_modes:
            return {}
            
        return {
            'experiential_quality': 'embodied_awareness',
            'subjective_perspective': 'agent_centered',
            'phenomenological_richness': 0.7
        }
    
    def get_participatory_knowing(self) -> Dict[str, Any]:
        """Generate participatory knowing - knowing through participation"""
        if KnowingMode.PARTICIPATORY not in self.active_knowing_modes:
            return {}
            
        return {
            'engagement_level': 'active_participation',
            'co_constitution': 'agent_environment_coupling',
            'transformative_potential': 0.8
        }
    
    def _combine_sensory_input(self, sensory_input: SensoryInput) -> torch.Tensor:
        """Combine different sensory modalities into single tensor"""
        tensors = []
        
        if sensory_input.visual is not None:
            tensors.append(sensory_input.visual.flatten())
        if sensory_input.auditory is not None:
            tensors.append(sensory_input.auditory.flatten())
        if sensory_input.proprioceptive is not None:
            tensors.append(sensory_input.proprioceptive.flatten())
            
        if not tensors:
            return torch.zeros(512)
            
        # Pad tensors to same length and combine
        max_len = max(t.numel() for t in tensors)
        padded_tensors = []
        for t in tensors:
            if t.numel() < max_len:
                padding = torch.zeros(max_len - t.numel())
                padded_tensors.append(torch.cat([t, padding]))
            else:
                padded_tensors.append(t[:max_len])
        
        combined = torch.stack(padded_tensors).mean(dim=0)
        
        # Ensure output is the expected size
        if combined.numel() > 512:
            combined = combined[:512]
        elif combined.numel() < 512:
            padding = torch.zeros(512 - combined.numel())
            combined = torch.cat([combined, padding])
            
        return combined
    
    def _action_to_tensor(self, action: Action) -> torch.Tensor:
        """Convert action to tensor representation"""
        # Simple action encoding - in practice would be more sophisticated
        action_features = []
        
        # Encode action name as hash
        action_hash = hash(action.name) % 1000
        action_features.append(action_hash / 1000.0)
        
        # Encode confidence
        action_features.append(action.confidence)
        
        # Encode priority
        action_features.append(action.priority)
        
        # Pad to fixed size
        while len(action_features) < 64:
            action_features.append(0.0)
            
        return torch.tensor(action_features[:64], dtype=torch.float32)