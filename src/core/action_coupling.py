"""
Action Coupling Module for Relevance-Informed Behavior Selection

This module implements mechanisms that couple relevance assessments with action
selection, enabling relevance-informed behavior and decision making.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from .relevance_optimization import TimeScale, MultiScaleRelevanceAssessment, SalienceType


class ActionType(Enum):
    """Types of actions in the cognitive system"""
    ATTENTION_SHIFT = "attention_shift"
    MEMORY_RETRIEVAL = "memory_retrieval"
    GOAL_PURSUIT = "goal_pursuit"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    LEARNING = "learning"
    REFLECTION = "reflection"
    COMMUNICATION = "communication"


@dataclass
class Action:
    """Represents an action with relevance properties"""
    action_type: ActionType
    parameters: Dict[str, Any]
    expected_outcome: str
    relevance_score: float = 0.0
    confidence: float = 0.0
    urgency: float = 0.0
    resource_cost: float = 0.0
    estimated_duration: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)


@dataclass
class ActionCouplingMetrics:
    """Metrics for tracking action coupling performance"""
    action_relevance_alignment: float = 0.0
    behavior_coherence: float = 0.0
    goal_achievement_rate: float = 0.0
    efficiency_score: float = 0.0
    adaptation_speed: float = 0.0


class ActionCoupler:
    """
    Advanced action coupling system that selects and executes actions based
    on relevance assessments and contextual demands.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Action repertoire
        self.available_actions: Dict[str, Action] = {}
        self.action_history: List[Tuple[str, Action, float]] = []  # (action_id, action, outcome)
        
        # Action selection parameters
        self.selection_weights = {
            'relevance': 0.4,
            'confidence': 0.2,
            'urgency': 0.2,
            'efficiency': 0.1,
            'novelty': 0.1
        }
        
        # Behavioral patterns
        self.behavior_patterns: Dict[str, List[str]] = {}  # context -> action sequence
        self.pattern_success_rates: Dict[str, float] = {}
        
        # Performance tracking
        self.coupling_metrics_history: List[ActionCouplingMetrics] = []
        
        # Learning parameters
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.exploration_rate = self.config.get('exploration_rate', 0.2)
        
        # Initialize default actions
        self._initialize_default_actions()
    
    def select_action(self, context: Dict[str, Any],
                     relevance_assessments: Dict[str, MultiScaleRelevanceAssessment],
                     available_resources: Dict[str, float] = None) -> Tuple[str, Action]:
        """
        Select the most appropriate action based on relevance assessments and context.
        
        Args:
            context: Current situational context
            relevance_assessments: Multi-scale relevance assessments for various items
            available_resources: Optional resource constraints
            
        Returns:
            Tuple of (action_id, selected_action)
        """
        available_resources = available_resources or {'time': 1.0, 'attention': 1.0, 'memory': 1.0}
        
        # Generate candidate actions
        candidate_actions = self._generate_candidate_actions(context, relevance_assessments)
        
        # Score each candidate action
        action_scores = {}
        for action_id, action in candidate_actions.items():
            score = self._score_action(action, context, relevance_assessments, available_resources)
            action_scores[action_id] = score
        
        # Select best action (with exploration)
        if np.random.random() < self.exploration_rate:
            # Exploration: select randomly from top candidates
            top_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            selected_id = np.random.choice([action_id for action_id, _ in top_actions])
        else:
            # Exploitation: select best action
            selected_id = max(action_scores, key=action_scores.get)
        
        selected_action = candidate_actions[selected_id]
        
        # Update action relevance based on selection
        selected_action.relevance_score = action_scores[selected_id]
        
        return selected_id, selected_action
    
    def execute_action_sequence(self, context: Dict[str, Any],
                               relevance_assessments: Dict[str, MultiScaleRelevanceAssessment],
                               sequence_length: int = 3) -> List[Tuple[str, Action, float]]:
        """
        Execute a sequence of relevance-informed actions.
        
        Args:
            context: Current context
            relevance_assessments: Relevance assessments
            sequence_length: Number of actions in sequence
            
        Returns:
            List of (action_id, action, outcome_score) tuples
        """
        action_sequence = []
        current_context = context.copy()
        current_resources = {'time': 1.0, 'attention': 1.0, 'memory': 1.0}
        
        for step in range(sequence_length):
            # Select action for current state
            action_id, action = self.select_action(
                current_context, relevance_assessments, current_resources
            )
            
            # Simulate action execution
            outcome_score = self._simulate_action_execution(action, current_context)
            
            # Update context and resources based on action
            current_context = self._update_context_after_action(
                current_context, action, outcome_score
            )
            current_resources = self._update_resources_after_action(
                current_resources, action
            )
            
            # Record action
            action_sequence.append((action_id, action, outcome_score))
            self.action_history.append((action_id, action, outcome_score))
        
        # Learn from sequence
        self._learn_from_action_sequence(action_sequence, context)
        
        return action_sequence
    
    def adapt_behavior_pattern(self, context_pattern: str, feedback: List[float]) -> Dict[str, Any]:
        """
        Adapt behavioral patterns based on feedback.
        
        Args:
            context_pattern: Pattern identifier for context
            feedback: Feedback scores for recent actions
            
        Returns:
            Adaptation results
        """
        adaptation_results = {
            'pattern_updated': False,
            'new_success_rate': 0.0,
            'behavior_changes': []
        }
        
        if context_pattern in self.behavior_patterns:
            # Update success rate
            current_rate = self.pattern_success_rates.get(context_pattern, 0.5)
            feedback_avg = np.mean(feedback) if feedback else 0.5
            
            new_rate = (current_rate * 0.7) + (feedback_avg * 0.3)
            self.pattern_success_rates[context_pattern] = new_rate
            adaptation_results['new_success_rate'] = new_rate
            
            # Adapt pattern if performance is poor
            if new_rate < 0.6:
                self._modify_behavior_pattern(context_pattern, feedback)
                adaptation_results['pattern_updated'] = True
                adaptation_results['behavior_changes'] = self._get_pattern_changes(context_pattern)
        
        return adaptation_results
    
    def couple_relevance_to_action(self, item: str, 
                                  relevance_assessment: MultiScaleRelevanceAssessment,
                                  context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Generate action recommendations based on relevance assessment.
        
        Args:
            item: Item with relevance assessment
            relevance_assessment: Multi-scale relevance assessment
            context: Current context
            
        Returns:
            List of (action_id, recommendation_strength) tuples
        """
        recommendations = []
        
        # Map relevance to action types
        action_mappings = self._get_relevance_action_mappings(relevance_assessment, context)
        
        for action_type, strength in action_mappings.items():
            # Find available actions of this type
            matching_actions = [
                (action_id, action) for action_id, action in self.available_actions.items()
                if action.action_type == action_type
            ]
            
            for action_id, action in matching_actions:
                # Adjust strength based on action properties
                adjusted_strength = strength * self._compute_action_suitability(
                    action, relevance_assessment, context
                )
                
                if adjusted_strength > 0.3:  # Threshold for recommendation
                    recommendations.append((action_id, adjusted_strength))
        
        # Sort by recommendation strength
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def measure_coupling_performance(self, context: Dict[str, Any]) -> ActionCouplingMetrics:
        """
        Measure the performance of action-relevance coupling.
        
        Args:
            context: Current context for measurement
            
        Returns:
            Action coupling performance metrics
        """
        if not self.action_history:
            return ActionCouplingMetrics()
        
        # Action-relevance alignment
        recent_actions = self.action_history[-10:]
        relevance_alignment = np.mean([
            action.relevance_score for _, action, _ in recent_actions
        ]) if recent_actions else 0.0
        
        # Behavior coherence
        behavior_coherence = self._measure_behavior_coherence()
        
        # Goal achievement rate
        goal_achievement = self._measure_goal_achievement()
        
        # Efficiency score
        efficiency = self._measure_action_efficiency()
        
        # Adaptation speed
        adaptation_speed = self._measure_adaptation_speed()
        
        metrics = ActionCouplingMetrics(
            action_relevance_alignment=relevance_alignment,
            behavior_coherence=behavior_coherence,
            goal_achievement_rate=goal_achievement,
            efficiency_score=efficiency,
            adaptation_speed=adaptation_speed
        )
        
        self.coupling_metrics_history.append(metrics)
        return metrics
    
    def _initialize_default_actions(self):
        """Initialize default action repertoire"""
        default_actions = {
            'focus_attention': Action(
                action_type=ActionType.ATTENTION_SHIFT,
                parameters={'target': 'high_relevance_item'},
                expected_outcome='increased_attention_on_relevant_item',
                confidence=0.8,
                resource_cost=0.2,
                estimated_duration=0.1
            ),
            'retrieve_memory': Action(
                action_type=ActionType.MEMORY_RETRIEVAL,
                parameters={'query': 'contextual_information'},
                expected_outcome='relevant_memory_retrieved',
                confidence=0.7,
                resource_cost=0.3,
                estimated_duration=0.2
            ),
            'pursue_goal': Action(
                action_type=ActionType.GOAL_PURSUIT,
                parameters={'goal_id': 'active_goal'},
                expected_outcome='progress_toward_goal',
                confidence=0.6,
                resource_cost=0.5,
                estimated_duration=0.5
            ),
            'explore_environment': Action(
                action_type=ActionType.EXPLORATION,
                parameters={'scope': 'local_environment'},
                expected_outcome='new_information_discovered',
                confidence=0.5,
                resource_cost=0.4,
                estimated_duration=0.3
            ),
            'learn_pattern': Action(
                action_type=ActionType.LEARNING,
                parameters={'pattern_type': 'relevance_pattern'},
                expected_outcome='improved_relevance_detection',
                confidence=0.7,
                resource_cost=0.6,
                estimated_duration=0.8
            ),
            'reflect_on_actions': Action(
                action_type=ActionType.REFLECTION,
                parameters={'scope': 'recent_actions'},
                expected_outcome='behavioral_insights',
                confidence=0.6,
                resource_cost=0.3,
                estimated_duration=0.4
            )
        }
        
        self.available_actions.update(default_actions)
    
    def _generate_candidate_actions(self, context: Dict[str, Any],
                                   relevance_assessments: Dict[str, MultiScaleRelevanceAssessment]) -> Dict[str, Action]:
        """Generate candidate actions based on context and relevance"""
        candidates = {}
        
        # Include all available actions as base candidates
        candidates.update(self.available_actions)
        
        # Generate context-specific actions
        if context.get('urgency', 0) > 0.7:
            # High urgency - add immediate response actions
            urgent_action = Action(
                action_type=ActionType.ATTENTION_SHIFT,
                parameters={'target': 'urgent_item', 'intensity': 'high'},
                expected_outcome='immediate_response_to_urgency',
                confidence=0.8,
                urgency=0.9,
                resource_cost=0.3,
                estimated_duration=0.05
            )
            candidates['urgent_response'] = urgent_action
        
        # Generate relevance-specific actions
        for item, assessment in relevance_assessments.items():
            if assessment.combined_score > 0.7:
                # High relevance - generate focused action
                focused_action = Action(
                    action_type=ActionType.GOAL_PURSUIT,
                    parameters={'target': item, 'focus_level': 'high'},
                    expected_outcome=f'focused_processing_of_{item}',
                    confidence=assessment.combined_score,
                    resource_cost=0.4,
                    estimated_duration=0.3
                )
                candidates[f'focus_on_{item}'] = focused_action
        
        return candidates
    
    def _score_action(self, action: Action, context: Dict[str, Any],
                     relevance_assessments: Dict[str, MultiScaleRelevanceAssessment],
                     available_resources: Dict[str, float]) -> float:
        """Score an action based on multiple criteria"""
        score = 0.0
        
        # Relevance component
        relevance_score = self._compute_action_relevance(action, relevance_assessments)
        score += self.selection_weights['relevance'] * relevance_score
        
        # Confidence component
        score += self.selection_weights['confidence'] * action.confidence
        
        # Urgency component
        urgency_match = min(action.urgency, context.get('urgency', 0.5))
        score += self.selection_weights['urgency'] * urgency_match
        
        # Efficiency component (inverse of resource cost)
        efficiency = max(0.0, 1.0 - action.resource_cost)
        score += self.selection_weights['efficiency'] * efficiency
        
        # Novelty component (prefer less recently used actions)
        novelty = self._compute_action_novelty(action)
        score += self.selection_weights['novelty'] * novelty
        
        # Resource feasibility penalty
        if action.resource_cost > sum(available_resources.values()) / len(available_resources):
            score *= 0.5  # Penalty for resource-intensive actions
        
        return np.clip(score, 0.0, 1.0)
    
    def _compute_action_relevance(self, action: Action,
                                 relevance_assessments: Dict[str, MultiScaleRelevanceAssessment]) -> float:
        """Compute how relevant an action is to current assessments"""
        if not relevance_assessments:
            return 0.5
        
        action_relevance = 0.0
        
        # Check if action targets relate to high-relevance items
        target = action.parameters.get('target', '')
        if target in relevance_assessments:
            assessment = relevance_assessments[target]
            action_relevance = assessment.combined_score
        else:
            # General action relevance based on action type
            avg_relevance = np.mean([
                assessment.combined_score 
                for assessment in relevance_assessments.values()
            ])
            
            # Map action types to relevance contexts
            if action.action_type == ActionType.ATTENTION_SHIFT and avg_relevance > 0.6:
                action_relevance = avg_relevance * 0.8
            elif action.action_type == ActionType.MEMORY_RETRIEVAL and avg_relevance > 0.5:
                action_relevance = avg_relevance * 0.7
            elif action.action_type == ActionType.GOAL_PURSUIT:
                action_relevance = avg_relevance * 0.9
            else:
                action_relevance = avg_relevance * 0.6
        
        return action_relevance
    
    def _compute_action_novelty(self, action: Action) -> float:
        """Compute novelty of action (prefer less recently used)"""
        if not self.action_history:
            return 1.0
        
        # Check recent usage
        recent_actions = self.action_history[-10:]
        usage_count = sum(1 for _, hist_action, _ in recent_actions 
                         if hist_action.action_type == action.action_type)
        
        novelty = max(0.0, 1.0 - (usage_count / 10.0))
        return novelty
    
    def _simulate_action_execution(self, action: Action, context: Dict[str, Any]) -> float:
        """Simulate action execution and return outcome score"""
        # Simple simulation based on action properties and context
        base_outcome = action.confidence
        
        # Context modulation
        if action.action_type == ActionType.ATTENTION_SHIFT:
            if context.get('attention_demand', 0.5) > 0.7:
                base_outcome *= 1.2  # Good for high attention demand
        elif action.action_type == ActionType.MEMORY_RETRIEVAL:
            if context.get('memory_relevant', True):
                base_outcome *= 1.1
        elif action.action_type == ActionType.GOAL_PURSUIT:
            if context.get('goal_active', True):
                base_outcome *= 1.3
        
        # Add some randomness
        noise = np.random.normal(0, 0.1)
        outcome = np.clip(base_outcome + noise, 0.0, 1.0)
        
        return outcome
    
    def _update_context_after_action(self, context: Dict[str, Any], 
                                    action: Action, outcome: float) -> Dict[str, Any]:
        """Update context based on action execution"""
        new_context = context.copy()
        
        # Update based on action type
        if action.action_type == ActionType.ATTENTION_SHIFT:
            new_context['attention_focus'] = action.parameters.get('target', 'unknown')
            new_context['attention_level'] = outcome
        elif action.action_type == ActionType.MEMORY_RETRIEVAL:
            new_context['memory_active'] = outcome > 0.6
            new_context['retrieved_info'] = action.parameters.get('query', '')
        elif action.action_type == ActionType.LEARNING:
            new_context['learning_progress'] = new_context.get('learning_progress', 0.0) + outcome * 0.1
        
        # Update urgency (generally decreases after action)
        new_context['urgency'] = max(0.0, new_context.get('urgency', 0.5) - 0.1)
        
        return new_context
    
    def _update_resources_after_action(self, resources: Dict[str, float], action: Action) -> Dict[str, float]:
        """Update available resources after action execution"""
        new_resources = resources.copy()
        
        # Deduct resource cost
        cost_per_resource = action.resource_cost / len(new_resources)
        for resource in new_resources:
            new_resources[resource] = max(0.0, new_resources[resource] - cost_per_resource)
        
        # Some recovery over time (simplified)
        recovery_rate = 0.1
        for resource in new_resources:
            new_resources[resource] = min(1.0, new_resources[resource] + recovery_rate)
        
        return new_resources
    
    def _learn_from_action_sequence(self, action_sequence: List[Tuple[str, Action, float]], 
                                   context: Dict[str, Any]):
        """Learn from executed action sequence"""
        if len(action_sequence) < 2:
            return
        
        # Extract pattern
        pattern_key = self._extract_context_pattern(context)
        action_ids = [action_id for action_id, _, _ in action_sequence]
        
        # Update behavior pattern
        if pattern_key not in self.behavior_patterns:
            self.behavior_patterns[pattern_key] = action_ids
        else:
            # Blend with existing pattern
            existing_pattern = self.behavior_patterns[pattern_key]
            new_pattern = []
            
            for i in range(max(len(existing_pattern), len(action_ids))):
                if i < len(action_ids) and i < len(existing_pattern):
                    # Keep action with better historical performance
                    if self._get_action_performance(action_ids[i]) > self._get_action_performance(existing_pattern[i]):
                        new_pattern.append(action_ids[i])
                    else:
                        new_pattern.append(existing_pattern[i])
                elif i < len(action_ids):
                    new_pattern.append(action_ids[i])
                elif i < len(existing_pattern):
                    new_pattern.append(existing_pattern[i])
            
            self.behavior_patterns[pattern_key] = new_pattern
        
        # Update success rate
        sequence_outcome = np.mean([outcome for _, _, outcome in action_sequence])
        current_rate = self.pattern_success_rates.get(pattern_key, 0.5)
        new_rate = current_rate * 0.8 + sequence_outcome * 0.2
        self.pattern_success_rates[pattern_key] = new_rate
    
    def _extract_context_pattern(self, context: Dict[str, Any]) -> str:
        """Extract a pattern key from context"""
        key_elements = []
        
        # Include important context elements
        if 'task_type' in context:
            key_elements.append(f"task_{context['task_type']}")
        
        urgency = context.get('urgency', 0.5)
        if urgency > 0.7:
            key_elements.append("high_urgency")
        elif urgency < 0.3:
            key_elements.append("low_urgency")
        else:
            key_elements.append("medium_urgency")
        
        if context.get('goal_active', False):
            key_elements.append("goal_active")
        
        return "_".join(key_elements) if key_elements else "default"
    
    def _get_action_performance(self, action_id: str) -> float:
        """Get historical performance of an action"""
        outcomes = [outcome for aid, _, outcome in self.action_history if aid == action_id]
        return np.mean(outcomes) if outcomes else 0.5
    
    def _get_relevance_action_mappings(self, assessment: MultiScaleRelevanceAssessment,
                                      context: Dict[str, Any]) -> Dict[ActionType, float]:
        """Map relevance assessment to action types"""
        mappings = {}
        
        # Immediate relevance -> attention shift
        if assessment.immediate_relevance > 0.6:
            mappings[ActionType.ATTENTION_SHIFT] = assessment.immediate_relevance
        
        # Short-term relevance -> memory retrieval
        if assessment.short_term_relevance > 0.5:
            mappings[ActionType.MEMORY_RETRIEVAL] = assessment.short_term_relevance
        
        # Medium-term relevance -> goal pursuit
        if assessment.medium_term_relevance > 0.4:
            mappings[ActionType.GOAL_PURSUIT] = assessment.medium_term_relevance
        
        # Long-term relevance -> learning
        if assessment.long_term_relevance > 0.5:
            mappings[ActionType.LEARNING] = assessment.long_term_relevance
        
        # High combined score -> exploration
        if assessment.combined_score > 0.7:
            mappings[ActionType.EXPLORATION] = assessment.combined_score * 0.6
        
        return mappings
    
    def _compute_action_suitability(self, action: Action,
                                   assessment: MultiScaleRelevanceAssessment,
                                   context: Dict[str, Any]) -> float:
        """Compute how suitable an action is for the relevance assessment"""
        suitability = 1.0
        
        # Match action urgency to assessment dominant scale
        if assessment.dominant_scale == TimeScale.IMMEDIATE and action.urgency < 0.5:
            suitability *= 0.7
        elif assessment.dominant_scale == TimeScale.LONG_TERM and action.urgency > 0.7:
            suitability *= 0.8
        
        # Resource appropriateness
        if assessment.combined_score > 0.8 and action.resource_cost < 0.3:
            suitability *= 0.9  # High relevance deserves more resources
        
        return suitability
    
    def _modify_behavior_pattern(self, pattern_key: str, feedback: List[float]):
        """Modify behavior pattern based on poor performance"""
        if pattern_key not in self.behavior_patterns:
            return
        
        pattern = self.behavior_patterns[pattern_key]
        new_pattern = pattern.copy()
        
        # Replace worst performing actions
        if len(feedback) == len(pattern):
            worst_indices = sorted(range(len(feedback)), key=lambda i: feedback[i])[:2]
            
            for idx in worst_indices:
                # Replace with a random alternative action
                alternative_actions = [aid for aid in self.available_actions.keys() 
                                     if aid not in pattern]
                if alternative_actions:
                    new_pattern[idx] = np.random.choice(alternative_actions)
        
        self.behavior_patterns[pattern_key] = new_pattern
    
    def _get_pattern_changes(self, pattern_key: str) -> List[str]:
        """Get description of changes made to behavior pattern"""
        # Simplified - just return the current pattern
        return self.behavior_patterns.get(pattern_key, [])
    
    def _measure_behavior_coherence(self) -> float:
        """Measure coherence of behavior patterns"""
        if not self.behavior_patterns:
            return 0.5
        
        coherence_scores = []
        for pattern, actions in self.behavior_patterns.items():
            if len(actions) > 1:
                # Measure action type consistency
                action_types = [self.available_actions[aid].action_type for aid in actions 
                              if aid in self.available_actions]
                
                if action_types:
                    type_consistency = len(set(action_types)) / len(action_types)
                    coherence_scores.append(1.0 - type_consistency)  # Lower diversity = higher coherence
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _measure_goal_achievement(self) -> float:
        """Measure goal achievement rate"""
        if not self.action_history:
            return 0.5
        
        goal_actions = [outcome for _, action, outcome in self.action_history 
                       if action.action_type == ActionType.GOAL_PURSUIT]
        
        return np.mean(goal_actions) if goal_actions else 0.5
    
    def _measure_action_efficiency(self) -> float:
        """Measure efficiency of action selection"""
        if not self.action_history:
            return 0.5
        
        recent_actions = self.action_history[-20:]
        efficiency_scores = []
        
        for _, action, outcome in recent_actions:
            # Efficiency = outcome / resource_cost
            if action.resource_cost > 0:
                efficiency = outcome / action.resource_cost
                efficiency_scores.append(min(1.0, efficiency))
        
        return np.mean(efficiency_scores) if efficiency_scores else 0.5
    
    def _measure_adaptation_speed(self) -> float:
        """Measure how quickly the system adapts behavior"""
        if len(self.coupling_metrics_history) < 3:
            return 0.5
        
        # Measure improvement rate in recent metrics
        recent_metrics = self.coupling_metrics_history[-3:]
        alignment_improvements = []
        
        for i in range(1, len(recent_metrics)):
            improvement = (recent_metrics[i].action_relevance_alignment - 
                          recent_metrics[i-1].action_relevance_alignment)
            alignment_improvements.append(improvement)
        
        if alignment_improvements:
            avg_improvement = np.mean(alignment_improvements)
            return np.clip(0.5 + avg_improvement * 2, 0.0, 1.0)
        
        return 0.5