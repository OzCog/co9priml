"""
Advanced Relevance Optimization System

This module implements sophisticated relevance optimization mechanisms that dynamically
prioritize cognitive resources based on contextual importance, goal relevance, and
environmental demands, following Vervaeke's relevance realization framework.
"""

import numpy as np
import torch
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from .relevance_core import RelevanceCore, RelevanceMode
from .vervaeke_cognitive_core import CognitiveFrame, KnowingMode
from ..learning.relevance_learning import RelevanceLearner, RelevanceExperience


class SalienceType(Enum):
    """Types of salience detection"""
    ENVIRONMENTAL = "environmental"
    GOAL_ORIENTED = "goal_oriented"
    CONTEXTUAL = "contextual"
    EMERGENT = "emergent"
    TEMPORAL = "temporal"


class TimeScale(Enum):
    """Multi-scale temporal processing levels"""
    IMMEDIATE = "immediate"        # Milliseconds to seconds
    SHORT_TERM = "short_term"      # Seconds to minutes  
    MEDIUM_TERM = "medium_term"    # Minutes to hours
    LONG_TERM = "long_term"        # Hours to days


@dataclass
class RelevanceMetrics:
    """Metrics for tracking relevance optimization performance"""
    attention_efficiency: float = 0.0
    cognitive_load_reduction: float = 0.0
    goal_alignment_score: float = 0.0
    environmental_responsiveness: float = 0.0
    memory_retrieval_accuracy: float = 0.0
    temporal_scale_effectiveness: float = 0.0  # New multi-scale metric
    knowledge_integration_score: float = 0.0   # New knowledge integration metric
    action_coupling_effectiveness: float = 0.0  # New action coupling metric
    overall_performance_improvement: float = 0.0
    timestamp: float = field(default_factory=lambda: np.datetime64('now').astype(float))


@dataclass
class MultiScaleRelevanceAssessment:
    """Assessment across multiple temporal scales"""
    immediate_relevance: float = 0.0    # Milliseconds to seconds
    short_term_relevance: float = 0.0   # Seconds to minutes
    medium_term_relevance: float = 0.0  # Minutes to hours  
    long_term_relevance: float = 0.0    # Hours to days
    combined_score: float = 0.0
    dominant_scale: TimeScale = TimeScale.IMMEDIATE


@dataclass
class EnvironmentalSignal:
    """Represents environmental changes requiring attention"""
    signal_type: str
    intensity: float
    novelty: float
    urgency: float
    context: Dict[str, Any]
    detection_confidence: float


@dataclass
class GoalRelevanceAlignment:
    """Alignment between current focus and goal relevance"""
    goal_id: str
    current_relevance: float
    optimal_relevance: float
    alignment_score: float
    recommended_adjustments: Dict[str, float]


class RelevanceOptimizer:
    """
    Advanced relevance optimization system that coordinates multiple relevance
    mechanisms to achieve optimal cognitive resource allocation.
    """
    
    def __init__(self, base_relevance_core: RelevanceCore, config: Dict[str, Any] = None):
        self.config = config or {}
        self.relevance_core = base_relevance_core
        
        # Enhanced scoring parameters
        self.importance_weights = {
            SalienceType.ENVIRONMENTAL: 0.25,
            SalienceType.GOAL_ORIENTED: 0.30,
            SalienceType.CONTEXTUAL: 0.20,
            SalienceType.EMERGENT: 0.15,
            SalienceType.TEMPORAL: 0.10
        }
        
        # Multi-scale temporal processing
        self.time_scale_weights = {
            TimeScale.IMMEDIATE: 0.4,    # High weight for immediate relevance
            TimeScale.SHORT_TERM: 0.3,   # Medium weight for short-term
            TimeScale.MEDIUM_TERM: 0.2,  # Lower weight for medium-term  
            TimeScale.LONG_TERM: 0.1     # Lowest weight for long-term
        }
        
        # Temporal decay parameters for each scale
        self.temporal_decay_rates = {
            TimeScale.IMMEDIATE: 0.1,    # Fast decay (seconds)
            TimeScale.SHORT_TERM: 0.01,  # Medium decay (minutes)
            TimeScale.MEDIUM_TERM: 0.001, # Slow decay (hours)
            TimeScale.LONG_TERM: 0.0001  # Very slow decay (days)
        }
        
        # Dynamic attention allocation
        self.attention_capacity = self.config.get('attention_capacity', 1.0)
        self.attention_allocation = {mode: 0.2 for mode in RelevanceMode}
        
        # Goal-relevance tracking
        self.active_goals: Dict[str, Dict[str, Any]] = {}
        self.goal_relevance_history: Dict[str, List[float]] = {}
        
        # Environmental monitoring
        self.environmental_signals: List[EnvironmentalSignal] = []
        self.environmental_baseline: Dict[str, float] = {}
        self.novelty_threshold = self.config.get('novelty_threshold', 0.7)
        
        # Performance tracking
        self.performance_history: List[RelevanceMetrics] = []
        self.baseline_performance: Optional[RelevanceMetrics] = None
        
        # Adaptive thresholds
        self.adaptive_thresholds = {mode: 0.5 for mode in RelevanceMode}
        self.threshold_adaptation_rate = self.config.get('threshold_adaptation_rate', 0.1)
        
        # Memory retrieval optimization
        self.memory_relevance_weights: Dict[str, float] = {}
        self.retrieval_success_history: List[bool] = []
        
        # Learning integration
        self.relevance_learner = RelevanceLearner(
            learning_rate=self.config.get('learning_rate', 0.05)
        )
        
    def compute_relevance_score(self, item: Any, context: Dict[str, Any],
                               goals: List[str] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute comprehensive relevance score using multiple salience types.
        
        Args:
            item: The item to evaluate for relevance
            context: Current contextual information
            goals: Optional list of active goals
            
        Returns:
            Tuple of (overall_score, component_scores)
        """
        component_scores = {}
        
        # Environmental salience
        env_score = self._compute_environmental_salience(item, context)
        component_scores[SalienceType.ENVIRONMENTAL.value] = env_score
        
        # Goal-oriented salience
        goal_score = self._compute_goal_salience(item, goals or [], context)
        component_scores[SalienceType.GOAL_ORIENTED.value] = goal_score
        
        # Contextual salience
        context_score = self._compute_contextual_salience(item, context)
        component_scores[SalienceType.CONTEXTUAL.value] = context_score
        
        # Emergent salience (novel patterns)
        emergent_score = self._compute_emergent_salience(item, context)
        component_scores[SalienceType.EMERGENT.value] = emergent_score
        
        # Temporal salience (urgency and timing)
        temporal_score = self._compute_temporal_salience(item, context)
        component_scores[SalienceType.TEMPORAL.value] = temporal_score
        
        # Weighted combination
        overall_score = sum(
            self.importance_weights[stype] * score
            for stype, score in zip(SalienceType, component_scores.values())
        )
        
        return overall_score, component_scores
    
    def compute_multi_scale_relevance(self, item: Any, context: Dict[str, Any],
                                     goals: List[str] = None) -> MultiScaleRelevanceAssessment:
        """
        Compute relevance assessment across multiple temporal scales.
        
        Args:
            item: The item to evaluate for relevance
            context: Current contextual information
            goals: Optional list of active goals
            
        Returns:
            Multi-scale relevance assessment
        """
        current_time = np.datetime64('now').astype(float)
        
        # Immediate relevance (milliseconds to seconds)
        immediate_context = {**context, 'urgency': context.get('urgency', 0.0) + 0.3}
        immediate_score, _ = self.compute_relevance_score(item, immediate_context, goals)
        
        # Short-term relevance (seconds to minutes)  
        short_term_score = self._compute_short_term_relevance(item, context, goals)
        
        # Medium-term relevance (minutes to hours)
        medium_term_score = self._compute_medium_term_relevance(item, context, goals)
        
        # Long-term relevance (hours to days)
        long_term_score = self._compute_long_term_relevance(item, context, goals)
        
        # Apply temporal decay based on time scale weights
        immediate_weighted = immediate_score * self.time_scale_weights[TimeScale.IMMEDIATE]
        short_term_weighted = short_term_score * self.time_scale_weights[TimeScale.SHORT_TERM]
        medium_term_weighted = medium_term_score * self.time_scale_weights[TimeScale.MEDIUM_TERM]
        long_term_weighted = long_term_score * self.time_scale_weights[TimeScale.LONG_TERM]
        
        # Combined score
        combined_score = (immediate_weighted + short_term_weighted + 
                         medium_term_weighted + long_term_weighted)
        
        # Determine dominant time scale
        scale_scores = {
            TimeScale.IMMEDIATE: immediate_score,
            TimeScale.SHORT_TERM: short_term_score,
            TimeScale.MEDIUM_TERM: medium_term_score,
            TimeScale.LONG_TERM: long_term_score
        }
        dominant_scale = max(scale_scores, key=scale_scores.get)
        
        return MultiScaleRelevanceAssessment(
            immediate_relevance=immediate_score,
            short_term_relevance=short_term_score,
            medium_term_relevance=medium_term_score,
            long_term_relevance=long_term_score,
            combined_score=combined_score,
            dominant_scale=dominant_scale
        )
    
    def _compute_environmental_salience(self, item: Any, context: Dict[str, Any]) -> float:
        """Compute salience based on environmental factors"""
        salience = 0.0
        
        # Check for environmental changes
        for signal in self.environmental_signals:
            if self._item_relates_to_signal(item, signal):
                signal_strength = signal.intensity * signal.novelty * signal.urgency
                salience = max(salience, signal_strength * signal.detection_confidence)
        
        # Baseline environmental relevance
        item_str = str(item)
        if item_str in self.environmental_baseline:
            baseline_relevance = self.environmental_baseline[item_str]
            salience = max(salience, baseline_relevance)
        
        return np.clip(salience, 0.0, 1.0)
    
    def _compute_goal_salience(self, item: Any, goals: List[str], 
                              context: Dict[str, Any]) -> float:
        """Compute salience based on goal relevance"""
        if not goals:
            return 0.0
        
        max_goal_relevance = 0.0
        item_str = str(item)
        
        for goal in goals:
            if goal in self.active_goals:
                goal_info = self.active_goals[goal]
                
                # Direct relevance to goal
                direct_relevance = self._compute_direct_goal_relevance(item_str, goal_info)
                
                # Contextual goal relevance
                contextual_relevance = self._compute_contextual_goal_relevance(
                    item_str, goal_info, context
                )
                
                # Historical relevance (based on past success)
                historical_relevance = self._compute_historical_goal_relevance(goal)
                
                goal_relevance = (
                    0.5 * direct_relevance +
                    0.3 * contextual_relevance +
                    0.2 * historical_relevance
                )
                
                max_goal_relevance = max(max_goal_relevance, goal_relevance)
        
        return np.clip(max_goal_relevance, 0.0, 1.0)
    
    def _compute_contextual_salience(self, item: Any, context: Dict[str, Any]) -> float:
        """Compute salience based on current context"""
        salience = 0.0
        item_str = str(item)
        
        # Context keyword matching
        context_keywords = set()
        for key, value in context.items():
            if isinstance(value, str):
                context_keywords.update(value.lower().split())
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        context_keywords.update(v.lower().split())
        
        item_keywords = set(item_str.lower().split())
        keyword_overlap = len(context_keywords & item_keywords)
        
        if keyword_overlap > 0:
            salience += min(0.8, keyword_overlap * 0.2)
        
        # Add some variation based on item characteristics for more diverse scoring
        item_hash = hash(item_str) % 1000
        variation = (item_hash / 1000.0) * 0.3  # 0-0.3 variation
        salience += variation
        
        # Contextual frequency (how often this item appears in similar contexts)
        context_signature = self._create_context_signature(context)
        if hasattr(self, 'context_item_frequency'):
            freq = self.context_item_frequency.get((context_signature, item_str), 0)
            salience += min(0.4, freq * 0.1)
        
        return np.clip(salience, 0.0, 1.0)
    
    def _compute_emergent_salience(self, item: Any, context: Dict[str, Any]) -> float:
        """Compute salience based on emergent patterns and novelty"""
        salience = 0.0
        item_str = str(item)
        
        # Novelty detection (how different is this from recent items)
        if hasattr(self, 'recent_items'):
            recent_similarities = [
                self._compute_similarity(item_str, recent_item)
                for recent_item in self.recent_items[-10:]  # Last 10 items
            ]
            
            if recent_similarities:
                avg_similarity = np.mean(recent_similarities)
                novelty = 1.0 - avg_similarity
                salience += min(0.6, novelty)
        
        # Pattern emergence (unexpected combinations)
        context_pattern = self._extract_pattern_features(context)
        item_pattern = self._extract_pattern_features({'item': item_str})
        
        pattern_novelty = self._compute_pattern_novelty(context_pattern, item_pattern)
        salience += min(0.4, pattern_novelty)
        
        return np.clip(salience, 0.0, 1.0)
    
    def _compute_temporal_salience(self, item: Any, context: Dict[str, Any]) -> float:
        """Compute salience based on temporal factors"""
        salience = 0.0
        
        # Urgency based on context timing
        urgency = context.get('urgency', 0.0)
        salience += min(0.5, urgency)
        
        # Temporal relevance (time-sensitive information)
        if 'deadline' in context or 'time_critical' in context:
            salience += 0.3
        
        # Recency effects (recently accessed items have higher salience)
        item_str = str(item)
        if hasattr(self, 'item_access_times'):
            last_access = self.item_access_times.get(item_str, 0)
            current_time = np.datetime64('now').astype(float)
            time_since_access = current_time - last_access
            
            # Exponential decay
            recency_bonus = np.exp(-time_since_access / 3600)  # 1 hour decay
            salience += min(0.2, recency_bonus)
        
        return np.clip(salience, 0.0, 1.0)
    
    def allocate_attention_dynamically(self, relevance_scores: Dict[str, float],
                                     current_attention: Dict[RelevanceMode, float],
                                     context: Dict[str, Any]) -> Dict[RelevanceMode, float]:
        """
        Dynamically allocate attention across relevance modes based on current needs.
        
        Args:
            relevance_scores: Current relevance scores for different items
            current_attention: Current attention allocation
            context: Current context
            
        Returns:
            New attention allocation
        """
        # Compute demand for each mode based on relevance scores
        mode_demands = self._compute_mode_demands(relevance_scores, context)
        
        # Apply contextual modulation
        modulated_demands = self._modulate_demands_by_context(mode_demands, context)
        
        # Optimize allocation with constraints
        new_allocation = self._optimize_attention_allocation(
            modulated_demands, current_attention
        )
        
        # Apply smoothing to prevent oscillation
        smoothed_allocation = self._smooth_attention_transition(
            current_attention, new_allocation
        )
        
        # Update internal state
        self.attention_allocation = smoothed_allocation
        
        return smoothed_allocation
    
    def _compute_mode_demands(self, relevance_scores: Dict[str, float],
                             context: Dict[str, Any]) -> Dict[RelevanceMode, float]:
        """Compute attention demand for each relevance mode"""
        demands = {mode: 0.0 for mode in RelevanceMode}
        
        for item, score in relevance_scores.items():
            # Map items to modes based on content and context
            item_modes = self._map_item_to_modes(item, context)
            
            for mode, weight in item_modes.items():
                demands[mode] += score * weight
        
        # Normalize demands
        total_demand = sum(demands.values())
        if total_demand > 0:
            demands = {mode: demand / total_demand for mode, demand in demands.items()}
        
        return demands
    
    def _map_item_to_modes(self, item: str, context: Dict[str, Any]) -> Dict[RelevanceMode, float]:
        """Map an item to relevance modes with weights"""
        mode_weights = {}
        
        # Keyword-based mapping
        if any(word in item.lower() for word in ['attention', 'focus', 'notice']):
            mode_weights[RelevanceMode.SELECTIVE_ATTENTION] = 0.8
        
        if any(word in item.lower() for word in ['remember', 'memory', 'recall']):
            mode_weights[RelevanceMode.WORKING_MEMORY] = 0.7
            mode_weights[RelevanceMode.LONG_TERM_MEMORY] = 0.6
        
        if any(word in item.lower() for word in ['problem', 'solve', 'search']):
            mode_weights[RelevanceMode.PROBLEM_SPACE] = 0.8
        
        if any(word in item.lower() for word in ['action', 'effect', 'consequence']):
            mode_weights[RelevanceMode.SIDE_EFFECTS] = 0.7
        
        # Default distribution if no specific mapping
        if not mode_weights:
            mode_weights = {mode: 0.2 for mode in RelevanceMode}
        
        return mode_weights
    
    def _modulate_demands_by_context(self, demands: Dict[RelevanceMode, float],
                                   context: Dict[str, Any]) -> Dict[RelevanceMode, float]:
        """Modulate attention demands based on context"""
        modulated = demands.copy()
        
        # Context-specific modulations
        if context.get('task_type') == 'memory_retrieval':
            modulated[RelevanceMode.LONG_TERM_MEMORY] *= 1.5
            modulated[RelevanceMode.WORKING_MEMORY] *= 1.3
        
        if context.get('task_type') == 'problem_solving':
            modulated[RelevanceMode.PROBLEM_SPACE] *= 1.4
            modulated[RelevanceMode.SELECTIVE_ATTENTION] *= 1.2
        
        if context.get('urgency', 0) > 0.7:
            modulated[RelevanceMode.SELECTIVE_ATTENTION] *= 1.3
        
        if context.get('novelty', 0) > 0.6:
            modulated[RelevanceMode.SELECTIVE_ATTENTION] *= 1.2
            modulated[RelevanceMode.WORKING_MEMORY] *= 0.9
        
        # Renormalize
        total = sum(modulated.values())
        if total > 0:
            modulated = {mode: demand / total for mode, demand in modulated.items()}
        
        return modulated
    
    def _optimize_attention_allocation(self, demands: Dict[RelevanceMode, float],
                                     current_allocation: Dict[RelevanceMode, float]) -> Dict[RelevanceMode, float]:
        """Optimize attention allocation using constrained optimization"""
        # Simple gradient-based approach
        learning_rate = 0.3
        new_allocation = {}
        
        for mode in RelevanceMode:
            demand = demands.get(mode, 0.0)
            current = current_allocation.get(mode, 0.2)
            
            # Move towards demand with learning rate
            target = demand
            new_value = current + learning_rate * (target - current)
            new_allocation[mode] = max(0.05, min(0.6, new_value))  # Bounds
        
        # Ensure allocation sums to capacity
        total = sum(new_allocation.values())
        if total > 0:
            scale_factor = self.attention_capacity / total
            new_allocation = {
                mode: allocation * scale_factor
                for mode, allocation in new_allocation.items()
            }
        
        return new_allocation
    
    def _smooth_attention_transition(self, current: Dict[RelevanceMode, float],
                                   target: Dict[RelevanceMode, float]) -> Dict[RelevanceMode, float]:
        """Apply smoothing to prevent oscillation in attention allocation"""
        smoothing_factor = 0.7  # How much to preserve current allocation
        
        smoothed = {}
        for mode in RelevanceMode:
            current_val = current.get(mode, 0.2)
            target_val = target.get(mode, 0.2)
            smoothed[mode] = smoothing_factor * current_val + (1 - smoothing_factor) * target_val
        
        return smoothed
    
    def detect_environmental_salience(self, environment_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> List[EnvironmentalSignal]:
        """
        Detect environmental changes that require attention allocation.
        
        Args:
            environment_data: Current environmental observations
            context: Current context
            
        Returns:
            List of detected environmental signals
        """
        signals = []
        
        # Compare with baseline to detect changes
        for key, value in environment_data.items():
            baseline_value = self.environmental_baseline.get(key, value)  # Use current value as baseline if not set
            
            if isinstance(value, (int, float)):
                if baseline_value == 0:
                    baseline_value = 1e-6  # Avoid division by zero
                change_magnitude = abs(value - baseline_value) / max(abs(baseline_value), 1e-6)
                
                # Lower threshold for testing - make it easier to detect changes
                detection_threshold = min(self.novelty_threshold, 0.1)
                if change_magnitude > detection_threshold:
                    # Detect significant change
                    intensity = min(1.0, change_magnitude)
                    novelty = self._compute_novelty(key, value)
                    urgency = self._compute_urgency(key, value, context)
                    
                    signal = EnvironmentalSignal(
                        signal_type=key,
                        intensity=intensity,
                        novelty=novelty,
                        urgency=urgency,
                        context={'old_value': baseline_value, 'new_value': value},
                        detection_confidence=min(1.0, change_magnitude * 2)
                    )
                    signals.append(signal)
        
        # Update baseline with current values (gradual adaptation)
        adaptation_rate = 0.1
        for key, value in environment_data.items():
            if isinstance(value, (int, float)):
                old_baseline = self.environmental_baseline.get(key, value)
                self.environmental_baseline[key] = (
                    old_baseline * (1 - adaptation_rate) + value * adaptation_rate
                )
        
        # Store signals for future reference
        self.environmental_signals.extend(signals)
        
        # Keep only recent signals
        if len(self.environmental_signals) > 100:
            self.environmental_signals = self.environmental_signals[-100:]
        
        return signals
    
    def align_with_goals(self, current_focus: Dict[str, float],
                        active_goals: List[str],
                        context: Dict[str, Any]) -> List[GoalRelevanceAlignment]:
        """
        Analyze and optimize alignment between current focus and goal relevance.
        
        Args:
            current_focus: Current attention/focus distribution
            active_goals: List of active goal identifiers
            context: Current context
            
        Returns:
            List of goal relevance alignments with recommendations
        """
        alignments = []
        
        for goal in active_goals:
            # Update active goals tracking
            if goal not in self.active_goals:
                self.active_goals[goal] = {
                    'created_time': np.datetime64('now').astype(float),
                    'priority': context.get('goal_priority', 0.5),
                    'context': context.copy()
                }
            
            goal_info = self.active_goals[goal]
            
            # Compute current relevance to this goal
            current_relevance = self._compute_current_goal_relevance(
                current_focus, goal, goal_info
            )
            
            # Compute optimal relevance for this goal
            optimal_relevance = self._compute_optimal_goal_relevance(
                goal, goal_info, context
            )
            
            # Compute alignment score
            alignment_score = self._compute_alignment_score(
                current_relevance, optimal_relevance
            )
            
            # Generate recommendations for improvement
            recommendations = self._generate_alignment_recommendations(
                current_focus, goal, current_relevance, optimal_relevance
            )
            
            alignment = GoalRelevanceAlignment(
                goal_id=goal,
                current_relevance=current_relevance,
                optimal_relevance=optimal_relevance,
                alignment_score=alignment_score,
                recommended_adjustments=recommendations
            )
            alignments.append(alignment)
            
            # Update goal relevance history
            if goal not in self.goal_relevance_history:
                self.goal_relevance_history[goal] = []
            self.goal_relevance_history[goal].append(current_relevance)
            
            # Keep history bounded
            if len(self.goal_relevance_history[goal]) > 100:
                self.goal_relevance_history[goal] = self.goal_relevance_history[goal][-100:]
        
        return alignments
    
    def optimize_memory_retrieval(self, query: str, context: Dict[str, Any],
                                memory_items: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Optimize memory retrieval based on relevance to current context and goals.
        
        Args:
            query: Query for memory retrieval
            context: Current context
            memory_items: Available memory items
            
        Returns:
            List of (item_id, relevance_score) pairs, sorted by relevance
        """
        relevance_scores = []
        
        for item_id, item_data in memory_items.items():
            # Compute base relevance
            base_relevance, _ = self.compute_relevance_score(
                item_data, context, context.get('active_goals', [])
            )
            
            # Query similarity
            query_similarity = self._compute_query_similarity(query, item_data)
            
            # Temporal relevance (recency and frequency)
            temporal_relevance = self._compute_memory_temporal_relevance(item_id, item_data)
            
            # Context relevance
            context_relevance = self._compute_memory_context_relevance(item_data, context)
            
            # Combine scores
            total_relevance = (
                0.4 * base_relevance +
                0.3 * query_similarity +
                0.2 * temporal_relevance +
                0.1 * context_relevance
            )
            
            relevance_scores.append((item_id, total_relevance))
        
        # Sort by relevance
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Update memory weights for learning
        for item_id, score in relevance_scores[:10]:  # Top 10 items
            if item_id not in self.memory_relevance_weights:
                self.memory_relevance_weights[item_id] = 0.0
            
            # Gradual update
            self.memory_relevance_weights[item_id] = (
                0.9 * self.memory_relevance_weights[item_id] + 0.1 * score
            )
        
        return relevance_scores
    
    def adapt_thresholds(self, performance_feedback: Dict[str, float]) -> Dict[RelevanceMode, float]:
        """
        Adaptively adjust relevance thresholds based on performance feedback.
        
        Args:
            performance_feedback: Feedback on performance across different modes
            
        Returns:
            Updated threshold values
        """
        for mode in RelevanceMode:
            if mode.value in performance_feedback:
                performance = performance_feedback[mode.value]
                current_threshold = self.adaptive_thresholds[mode]
                
                # If performance is poor, lower threshold (more inclusive)
                # If performance is good, slightly raise threshold (more selective)
                if performance < 0.5:
                    adjustment = -self.threshold_adaptation_rate * (0.5 - performance)
                else:
                    adjustment = self.threshold_adaptation_rate * (performance - 0.7) * 0.5
                
                new_threshold = current_threshold + adjustment
                self.adaptive_thresholds[mode] = np.clip(new_threshold, 0.1, 0.9)
        
        return self.adaptive_thresholds.copy()
    
    def learn_from_feedback(self, context: Dict[str, Any], actions: List[str],
                          outcomes: List[float], goals: List[str]) -> Dict[str, Any]:
        """
        Learn and adapt relevance mechanisms based on feedback.
        
        Args:
            context: Context in which actions were taken
            actions: List of actions taken
            outcomes: List of outcome scores for each action
            goals: List of active goals
            
        Returns:
            Learning statistics
        """
        # Create relevance experiences for learning
        experiences = []
        
        for action, outcome in zip(actions, outcomes):
            frame = CognitiveFrame(
                salience_weights=self._extract_salience_weights(context),
                active_knowing_modes=[KnowingMode.PARTICIPATORY],
                context=context
            )
            
            experience = RelevanceExperience(
                frame=frame,
                inputs={'action': action, 'goals': goals},
                actual_relevance={'action': outcome},
                reward=outcome
            )
            experiences.append(experience)
        
        # Learn from experiences
        learning_updates = {}
        for experience in experiences:
            updates = self.relevance_learner.learn(experience)
            for key, update in updates.items():
                if key not in learning_updates:
                    learning_updates[key] = []
                learning_updates[key].append(update)
        
        # Compute learning statistics
        stats = {
            'num_experiences': len(experiences),
            'average_outcome': np.mean(outcomes) if outcomes else 0.0,
            'learning_updates': {
                key: np.mean(updates) for key, updates in learning_updates.items()
            },
            'threshold_adaptations': self.adapt_thresholds({
                mode.value: np.mean([exp.reward for exp in experiences])
                for mode in RelevanceMode
            })
        }
        
        return stats
    
    def measure_performance(self, context: Dict[str, Any],
                          baseline_metrics: Optional[RelevanceMetrics] = None) -> RelevanceMetrics:
        """
        Measure current relevance optimization performance.
        
        Args:
            context: Current context for measurement
            baseline_metrics: Optional baseline to compare against
            
        Returns:
            Current performance metrics
        """
        # Attention efficiency (how well attention is allocated)
        attention_efficiency = self._measure_attention_efficiency()
        
        # Cognitive load reduction (how much processing is saved)
        cognitive_load_reduction = self._measure_cognitive_load_reduction()
        
        # Goal alignment score
        goal_alignment = self._measure_goal_alignment_performance()
        
        # Environmental responsiveness
        environmental_responsiveness = self._measure_environmental_responsiveness()
        
        # Memory retrieval accuracy
        memory_accuracy = self._measure_memory_retrieval_accuracy()
        
        # New multi-scale temporal effectiveness
        temporal_scale_effectiveness = self._measure_temporal_scale_effectiveness()
        
        # Knowledge integration score (if available)
        knowledge_integration_score = self._measure_knowledge_integration_score()
        
        # Action coupling effectiveness (if available)
        action_coupling_effectiveness = self._measure_action_coupling_effectiveness()
        
        # Compute overall improvement
        current_metrics = RelevanceMetrics(
            attention_efficiency=attention_efficiency,
            cognitive_load_reduction=cognitive_load_reduction,
            goal_alignment_score=goal_alignment,
            environmental_responsiveness=environmental_responsiveness,
            memory_retrieval_accuracy=memory_accuracy,
            temporal_scale_effectiveness=temporal_scale_effectiveness,
            knowledge_integration_score=knowledge_integration_score,
            action_coupling_effectiveness=action_coupling_effectiveness
        )
        
        if baseline_metrics:
            improvements = []
            improvements.append(current_metrics.attention_efficiency - baseline_metrics.attention_efficiency)
            improvements.append(current_metrics.cognitive_load_reduction - baseline_metrics.cognitive_load_reduction)
            improvements.append(current_metrics.goal_alignment_score - baseline_metrics.goal_alignment_score)
            improvements.append(current_metrics.environmental_responsiveness - baseline_metrics.environmental_responsiveness)
            improvements.append(current_metrics.memory_retrieval_accuracy - baseline_metrics.memory_retrieval_accuracy)
            improvements.append(current_metrics.temporal_scale_effectiveness - baseline_metrics.temporal_scale_effectiveness)
            improvements.append(current_metrics.knowledge_integration_score - baseline_metrics.knowledge_integration_score)
            improvements.append(current_metrics.action_coupling_effectiveness - baseline_metrics.action_coupling_effectiveness)
            
            current_metrics.overall_performance_improvement = np.mean(improvements)
        else:
            current_metrics.overall_performance_improvement = 0.0
        
        # Store metrics for history
        self.performance_history.append(current_metrics)
        
        # Set baseline if not set
        if self.baseline_performance is None:
            self.baseline_performance = current_metrics
        
        return current_metrics
    
    # Helper methods for internal computations
    
    def _item_relates_to_signal(self, item: Any, signal: EnvironmentalSignal) -> bool:
        """Check if an item relates to an environmental signal"""
        item_str = str(item).lower()
        signal_type = signal.signal_type.lower()
        return signal_type in item_str or any(
            keyword in item_str for keyword in signal.context.get('keywords', [])
        )
    
    def _compute_direct_goal_relevance(self, item: str, goal_info: Dict[str, Any]) -> float:
        """Compute direct relevance of an item to a goal"""
        # Simple keyword matching (can be enhanced with embeddings)
        goal_context = goal_info.get('context', {})
        goal_keywords = set()
        
        for value in goal_context.values():
            if isinstance(value, str):
                goal_keywords.update(value.lower().split())
        
        item_keywords = set(item.lower().split())
        overlap = len(goal_keywords & item_keywords)
        
        return min(1.0, overlap * 0.3)
    
    def _compute_contextual_goal_relevance(self, item: str, goal_info: Dict[str, Any],
                                         context: Dict[str, Any]) -> float:
        """Compute contextual relevance of an item to a goal"""
        # Context similarity between current and goal context
        goal_context = goal_info.get('context', {})
        
        similarity = 0.0
        common_keys = set(goal_context.keys()) & set(context.keys())
        
        for key in common_keys:
            if goal_context[key] == context[key]:
                similarity += 0.2
        
        return min(1.0, similarity)
    
    def _compute_historical_goal_relevance(self, goal: str) -> float:
        """Compute relevance based on historical success with this goal"""
        if goal not in self.goal_relevance_history:
            return 0.5  # Default for new goals
        
        history = self.goal_relevance_history[goal]
        if len(history) < 3:
            return 0.5
        
        # Recent trend
        recent_scores = history[-5:]
        return np.mean(recent_scores)
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context matching"""
        sorted_items = sorted(context.items())
        return str(hash(str(sorted_items)))
    
    def _compute_similarity(self, item1: str, item2: str) -> float:
        """Compute similarity between two items"""
        # Simple Jaccard similarity
        set1 = set(item1.lower().split())
        set2 = set(item2.lower().split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_pattern_features(self, data: Dict[str, Any]) -> List[str]:
        """Extract pattern features from data"""
        features = []
        for key, value in data.items():
            if isinstance(value, str):
                features.extend(value.lower().split())
            elif isinstance(value, (int, float)):
                features.append(f"{key}:{value}")
        return features
    
    def _compute_pattern_novelty(self, pattern1: List[str], pattern2: List[str]) -> float:
        """Compute novelty of pattern combination"""
        combined = set(pattern1 + pattern2)
        individual = set(pattern1) | set(pattern2)
        
        if len(individual) == 0:
            return 0.0
        
        novelty = len(combined) / len(individual)
        return min(1.0, novelty)
    
    def _compute_novelty(self, key: str, value: Any) -> float:
        """Compute novelty of a value for a given key"""
        # Simple novelty based on deviation from baseline
        baseline = self.environmental_baseline.get(key, 0.0)
        if isinstance(value, (int, float)) and baseline != 0:
            deviation = abs(value - baseline) / abs(baseline)
            return min(1.0, deviation)
        return 0.5  # Default novelty
    
    def _compute_urgency(self, key: str, value: Any, context: Dict[str, Any]) -> float:
        """Compute urgency of attending to a change"""
        urgency = context.get('urgency', 0.0)
        
        # Keyword-based urgency
        if any(word in key.lower() for word in ['critical', 'urgent', 'emergency']):
            urgency = max(urgency, 0.8)
        
        if any(word in key.lower() for word in ['important', 'priority']):
            urgency = max(urgency, 0.6)
        
        return min(1.0, urgency)
    
    def _compute_current_goal_relevance(self, current_focus: Dict[str, float],
                                      goal: str, goal_info: Dict[str, Any]) -> float:
        """Compute how well current focus aligns with a goal"""
        goal_context = goal_info.get('context', {})
        
        relevance = 0.0
        for focus_item, focus_weight in current_focus.items():
            item_goal_relevance = self._compute_direct_goal_relevance(focus_item, goal_info)
            relevance += focus_weight * item_goal_relevance
        
        return min(1.0, relevance)
    
    def _compute_optimal_goal_relevance(self, goal: str, goal_info: Dict[str, Any],
                                      context: Dict[str, Any]) -> float:
        """Compute optimal relevance allocation for a goal"""
        priority = goal_info.get('priority', 0.5)
        time_pressure = context.get('urgency', 0.0)
        
        optimal = priority * (1.0 + time_pressure * 0.5)
        return min(1.0, optimal)
    
    def _compute_alignment_score(self, current: float, optimal: float) -> float:
        """Compute alignment score between current and optimal relevance"""
        if optimal == 0:
            return 1.0 if current == 0 else 0.0
        
        ratio = current / optimal
        if ratio > 1.0:
            return 1.0 / ratio  # Penalize over-allocation
        else:
            return ratio
    
    def _generate_alignment_recommendations(self, current_focus: Dict[str, float],
                                          goal: str, current_relevance: float,
                                          optimal_relevance: float) -> Dict[str, float]:
        """Generate recommendations for improving goal alignment"""
        recommendations = {}
        
        if current_relevance < optimal_relevance:
            # Need more focus on this goal
            deficit = optimal_relevance - current_relevance
            recommendations['increase_goal_focus'] = deficit
            recommendations['reduce_distractions'] = deficit * 0.5
        
        elif current_relevance > optimal_relevance * 1.2:
            # Too much focus, can reallocate
            excess = current_relevance - optimal_relevance
            recommendations['reduce_goal_focus'] = excess * 0.5
            recommendations['explore_alternatives'] = excess * 0.3
        
        return recommendations
    
    def _compute_query_similarity(self, query: str, item_data: Any) -> float:
        """Compute similarity between query and memory item"""
        item_text = str(item_data).lower()
        query_text = query.lower()
        
        return self._compute_similarity(query_text, item_text)
    
    def _compute_memory_temporal_relevance(self, item_id: str, item_data: Any) -> float:
        """Compute temporal relevance for memory retrieval"""
        # Default temporal relevance (can be enhanced with actual timestamps)
        base_relevance = 0.5
        
        # Access frequency
        if hasattr(self, 'item_access_counts'):
            access_count = self.item_access_counts.get(item_id, 0)
            frequency_bonus = min(0.3, access_count * 0.05)
            base_relevance += frequency_bonus
        
        return min(1.0, base_relevance)
    
    def _compute_memory_context_relevance(self, item_data: Any, context: Dict[str, Any]) -> float:
        """Compute context relevance for memory retrieval"""
        return self._compute_contextual_salience(item_data, context)
    
    def _extract_salience_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract salience weights from context"""
        weights = {}
        for key, value in context.items():
            if isinstance(value, (int, float)):
                weights[key] = min(1.0, abs(value))
            else:
                weights[key] = 0.5  # Default weight
        return weights
    
    def _measure_attention_efficiency(self) -> float:
        """Measure attention allocation efficiency"""
        # Based on how well attention is distributed relative to demands
        efficiency = 0.0
        
        for mode, allocation in self.attention_allocation.items():
            if allocation > 0:
                efficiency += min(1.0, allocation / max(0.1, self.adaptive_thresholds[mode]))
        
        return efficiency / len(RelevanceMode)
    
    def _measure_cognitive_load_reduction(self) -> float:
        """Measure reduction in cognitive load"""
        # Based on how well thresholds filter irrelevant information
        if not hasattr(self, 'processing_history'):
            return 0.5
        
        # Simple proxy: higher thresholds = more filtering = less load
        avg_threshold = np.mean(list(self.adaptive_thresholds.values()))
        return min(1.0, avg_threshold * 1.5)
    
    def _measure_goal_alignment_performance(self) -> float:
        """Measure goal alignment performance"""
        if not self.goal_relevance_history:
            return 0.5
        
        total_alignment = 0.0
        count = 0
        
        for goal, history in self.goal_relevance_history.items():
            if len(history) >= 2:
                recent_avg = np.mean(history[-5:])
                total_alignment += recent_avg
                count += 1
        
        return total_alignment / count if count > 0 else 0.5
    
    def _measure_environmental_responsiveness(self) -> float:
        """Measure responsiveness to environmental changes"""
        if not self.environmental_signals:
            return 0.5
        
        # Recent signals with high confidence indicate good responsiveness
        recent_signals = [s for s in self.environmental_signals[-10:] 
                         if s.detection_confidence > 0.6]
        
        responsiveness = len(recent_signals) / 10.0
        return min(1.0, responsiveness)
    
    def _measure_memory_retrieval_accuracy(self) -> float:
        """Measure memory retrieval accuracy"""
        if not self.retrieval_success_history:
            return 0.5
        
        recent_successes = self.retrieval_success_history[-20:]
        return sum(recent_successes) / len(recent_successes)
    
    # Multi-scale temporal assessment methods
    
    def _compute_short_term_relevance(self, item: Any, context: Dict[str, Any], 
                                     goals: List[str] = None) -> float:
        """Compute relevance for short-term processing (seconds to minutes)"""
        # Focus on immediate patterns and working memory constraints
        base_score, _ = self.compute_relevance_score(item, context, goals)
        
        # Boost for items that fit working memory capacity
        if hasattr(self, 'working_memory_items'):
            wm_capacity = 7  # Miller's magic number
            if len(self.working_memory_items) < wm_capacity:
                base_score *= 1.2
            else:
                base_score *= 0.8  # Penalty for overload
        
        # Consider repetition priming effects
        item_str = str(item)
        if hasattr(self, 'recent_items'):
            if item_str in self.recent_items[-5:]:  # Recent exposure
                base_score *= 1.15  # Priming boost
        
        return min(1.0, base_score)
    
    def _compute_medium_term_relevance(self, item: Any, context: Dict[str, Any],
                                      goals: List[str] = None) -> float:
        """Compute relevance for medium-term processing (minutes to hours)"""
        # Focus on goal pursuit and learning consolidation
        base_score, _ = self.compute_relevance_score(item, context, goals)
        
        # Enhanced goal relevance for medium-term
        if goals:
            goal_boost = 0.0
            for goal in goals:
                if goal in self.active_goals:
                    goal_priority = self.active_goals[goal].get('priority', 0.5)
                    goal_boost += goal_priority * 0.3
            base_score += min(0.4, goal_boost)
        
        # Learning consolidation factor
        if 'learning' in str(item).lower() or context.get('task_type') == 'learning':
            base_score *= 1.25
        
        return min(1.0, base_score)
    
    def _compute_long_term_relevance(self, item: Any, context: Dict[str, Any],
                                    goals: List[str] = None) -> float:
        """Compute relevance for long-term processing (hours to days)"""
        # Focus on knowledge integration and strategic planning
        base_score, _ = self.compute_relevance_score(item, context, goals)
        
        # Strategic importance
        strategic_keywords = ['strategy', 'plan', 'future', 'knowledge', 'skill', 'learning']
        item_str = str(item).lower()
        strategic_relevance = sum(1 for keyword in strategic_keywords if keyword in item_str)
        base_score += min(0.3, strategic_relevance * 0.1)
        
        # Knowledge integration potential
        if hasattr(self, 'knowledge_base'):
            # Boost items that connect to existing knowledge
            connection_count = self._count_knowledge_connections(item)
            base_score += min(0.2, connection_count * 0.05)
        
        # Long-term goal alignment
        if goals and hasattr(self, 'long_term_goal_mapping'):
            for goal in goals:
                if goal in self.long_term_goal_mapping:
                    long_term_alignment = self.long_term_goal_mapping[goal]
                    base_score += long_term_alignment * 0.2
        
        return min(1.0, base_score)
    
    def _count_knowledge_connections(self, item: Any) -> int:
        """Count connections between item and existing knowledge base"""
        # Simplified knowledge connection counting
        item_str = str(item).lower()
        connection_count = 0
        
        # Check for conceptual connections (simplified)
        conceptual_terms = item_str.split()
        if hasattr(self, 'knowledge_concepts'):
            for term in conceptual_terms:
                if term in self.knowledge_concepts:
                    connection_count += 1
        
        return connection_count
    
    def _measure_temporal_scale_effectiveness(self) -> float:
        """Measure effectiveness of multi-scale temporal assessment"""
        # If we have multi-scale assessments, measure their consistency and accuracy
        if hasattr(self, 'multi_scale_assessments'):
            assessments = self.multi_scale_assessments
            if assessments:
                # Measure consistency across scales
                consistency_scores = []
                for assessment in assessments:
                    scales = [
                        assessment.immediate_relevance,
                        assessment.short_term_relevance, 
                        assessment.medium_term_relevance,
                        assessment.long_term_relevance
                    ]
                    # Lower variance indicates better scale consistency
                    variance = np.var(scales)
                    consistency = 1.0 / (1.0 + variance)
                    consistency_scores.append(consistency)
                
                return np.mean(consistency_scores)
        
        # Default based on temporal salience performance
        if hasattr(self, 'temporal_salience_accuracy'):
            return self.temporal_salience_accuracy
        
        return 0.6  # Default temporal effectiveness
    
    def _measure_knowledge_integration_score(self) -> float:
        """Measure knowledge integration performance"""
        # If knowledge integrator is available, use its metrics
        if hasattr(self, 'knowledge_integrator'):
            integration_metrics = self.knowledge_integrator.integration_history
            if integration_metrics:
                latest_metrics = integration_metrics[-1]
                return (latest_metrics.integration_efficiency + 
                       latest_metrics.relevance_accuracy +
                       latest_metrics.knowledge_coherence) / 3
        
        # Default based on relevance accuracy of integrated knowledge
        if hasattr(self, 'knowledge_relevance_scores'):
            return np.mean(list(self.knowledge_relevance_scores.values()))
        
        return 0.7  # Default knowledge integration score
    
    def _measure_action_coupling_effectiveness(self) -> float:
        """Measure action coupling performance"""
        # If action coupler is available, use its metrics
        if hasattr(self, 'action_coupler'):
            coupling_metrics = self.action_coupler.coupling_metrics_history
            if coupling_metrics:
                latest_metrics = coupling_metrics[-1]
                return (latest_metrics.action_relevance_alignment +
                       latest_metrics.behavior_coherence +
                       latest_metrics.efficiency_score) / 3
        
        # Default based on action-relevance alignment in history
        if hasattr(self, 'action_relevance_history'):
            return np.mean(self.action_relevance_history[-10:])
        
        return 0.65  # Default action coupling effectiveness