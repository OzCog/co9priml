"""
Meta-Cognitive Strategy Selector
===============================

This module implements meta-cognitive strategy selection and optimization
capabilities that enable the system to choose optimal cognitive strategies
based on context, past performance, and current objectives.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging
import random
from ..interfaces.meta_cognitive_interface import (
    StrategySelectionInterface, MetaCognitiveCapability
)


class StrategyType(Enum):
    """Types of meta-cognitive strategies."""
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"
    EXPLORATORY = "exploratory"
    FOCUSED = "focused"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
    REFLECTIVE = "reflective"


class ContextType(Enum):
    """Types of contexts for strategy selection."""
    PROBLEM_SOLVING = "problem_solving"
    LEARNING = "learning"
    DECISION_MAKING = "decision_making"
    PATTERN_RECOGNITION = "pattern_recognition"
    CREATIVITY = "creativity"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"


@dataclass
class Strategy:
    """Represents a meta-cognitive strategy."""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    contexts: List[ContextType]
    resource_requirements: Dict[str, float]
    expected_performance: Dict[str, float]
    complexity_level: int
    adaptation_rate: float


@dataclass
class StrategyPerformance:
    """Records performance of a strategy in a specific context."""
    strategy_id: str
    context_type: ContextType
    task_context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    success_rate: float
    execution_time: float
    timestamp: float


@dataclass
class StrategyRecommendation:
    """Recommendation for strategy selection."""
    recommended_strategy: str
    confidence: float
    expected_performance: Dict[str, float]
    rationale: List[str]
    alternatives: List[Tuple[str, float]]  # (strategy_id, confidence)
    context_match: float


class MetaCognitiveStrategySelector(StrategySelectionInterface):
    """
    Implementation of meta-cognitive strategy selection and optimization.
    
    This system provides:
    - Context-aware strategy selection
    - Performance-based strategy evaluation
    - Adaptive strategy optimization
    - Strategy effectiveness learning
    - Multi-objective strategy balancing
    - Strategy combination and hybridization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the meta-cognitive strategy selector."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.exploration_rate = self.config.get('exploration_rate', 0.2)
        self.strategy_decay_rate = self.config.get('strategy_decay_rate', 0.95)
        self.context_similarity_threshold = self.config.get('context_similarity_threshold', 0.7)
        
        # Strategy library
        self.strategies: Dict[str, Strategy] = {}
        self.strategy_performance_history: List[StrategyPerformance] = []
        self.context_strategy_map: Dict[ContextType, List[str]] = {}
        
        # Learning and adaptation
        self.strategy_effectiveness: Dict[str, Dict[str, float]] = {}
        self.context_patterns: Dict[str, Dict[str, Any]] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        self.logger.info("Meta-cognitive strategy selector initialized")
    
    def initialize(self) -> bool:
        """Initialize the strategy selector component."""
        try:
            self._build_context_strategy_mappings()
            self._initialize_performance_baselines()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy selector: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the strategy selector component."""
        return True
    
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of strategy selection capabilities."""
        return [
            MetaCognitiveCapability(
                name="context_aware_selection",
                description="Context-aware strategy selection",
                complexity_level=3,
                requires_recursion=False,
                resource_intensive=False
            ),
            MetaCognitiveCapability(
                name="performance_optimization",
                description="Performance-based strategy optimization",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="adaptive_learning",
                description="Adaptive strategy learning from experience",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="strategy_combination",
                description="Combination and hybridization of strategies",
                complexity_level=5,
                requires_recursion=True,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="multi_objective_balancing",
                description="Balancing multiple objectives in strategy selection",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            )
        ]
    
    def process_meta_cognitive_request(self, 
                                     request_type: str,
                                     request_data: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meta-cognitive request."""
        if request_type == "select_strategy":
            task_context = request_data if isinstance(request_data, dict) else {}
            available_strategies = context.get('available_strategies', list(self.strategies.keys()))
            selected = self.select_strategy(task_context, available_strategies)
            return {'selected_strategy': selected}
        elif request_type == "evaluate_effectiveness":
            strategy = request_data.get('strategy', '')
            performance = request_data.get('performance', {})
            effectiveness = self.evaluate_strategy_effectiveness(strategy, performance)
            return {'effectiveness': effectiveness}
        elif request_type == "adapt_strategy":
            current = request_data.get('current_strategy', '')
            feedback = request_data.get('feedback', {})
            adapted = self.adapt_strategy(current, feedback)
            return {'adapted_strategy': adapted}
        else:
            return {'error': f'Unknown request type: {request_type}'}
    
    def select_strategy(self, 
                       task_context: Dict[str, Any],
                       available_strategies: List[str]) -> str:
        """Select optimal strategy for given context."""
        try:
            # Determine context type
            context_type = self._determine_context_type(task_context)
            
            # Get strategy recommendations
            recommendation = self._get_strategy_recommendation(
                context_type, task_context, available_strategies
            )
            
            # Apply exploration-exploitation balance
            selected_strategy = self._apply_exploration_exploitation(
                recommendation, available_strategies
            )
            
            # Log selection decision for learning
            self._log_strategy_selection(
                selected_strategy, context_type, task_context, recommendation
            )
            
            return selected_strategy
            
        except Exception as e:
            self.logger.error(f"Error in strategy selection: {e}")
            # Fallback to default strategy
            return available_strategies[0] if available_strategies else "analytical"
    
    def evaluate_strategy_effectiveness(self, 
                                      strategy: str,
                                      performance_data: Dict[str, Any]) -> float:
        """Evaluate the effectiveness of a strategy."""
        try:
            if strategy not in self.strategies:
                return 0.5  # Unknown strategy
            
            # Extract performance metrics
            metrics = performance_data.get('metrics', {})
            success_rate = metrics.get('success_rate', 0.5)
            efficiency = metrics.get('efficiency', 0.5)
            quality = metrics.get('quality', 0.5)
            resource_efficiency = metrics.get('resource_efficiency', 0.5)
            
            # Calculate weighted effectiveness
            effectiveness = (
                success_rate * 0.3 +
                efficiency * 0.25 +
                quality * 0.25 +
                resource_efficiency * 0.2
            )
            
            # Update strategy effectiveness history
            context_type = performance_data.get('context_type', 'unknown')
            if strategy not in self.strategy_effectiveness:
                self.strategy_effectiveness[strategy] = {}
            
            # Exponential moving average update
            current_effectiveness = self.strategy_effectiveness[strategy].get(context_type, 0.5)
            updated_effectiveness = (
                current_effectiveness * (1 - self.learning_rate) +
                effectiveness * self.learning_rate
            )
            self.strategy_effectiveness[strategy][context_type] = updated_effectiveness
            
            return effectiveness
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy effectiveness: {e}")
            return 0.5
    
    def adapt_strategy(self, 
                      current_strategy: str,
                      feedback: Dict[str, Any]) -> str:
        """Adapt strategy based on feedback."""
        try:
            if current_strategy not in self.strategies:
                return current_strategy
            
            # Analyze feedback
            performance_issues = feedback.get('performance_issues', [])
            resource_constraints = feedback.get('resource_constraints', [])
            context_changes = feedback.get('context_changes', {})
            
            # Determine adaptation needed
            adaptation_type = self._determine_adaptation_type(
                performance_issues, resource_constraints, context_changes
            )
            
            # Apply adaptation
            adapted_strategy = self._apply_strategy_adaptation(
                current_strategy, adaptation_type, feedback
            )
            
            # Record adaptation
            self._record_adaptation(current_strategy, adapted_strategy, feedback)
            
            return adapted_strategy
            
        except Exception as e:
            self.logger.error(f"Error adapting strategy: {e}")
            return current_strategy
    
    def select_optimal_strategy(self, 
                              task_context: Dict[str, Any],
                              performance_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select optimal strategy for meta-cognitive core."""
        recommendation = {
            'current_strategy': None,
            'recommended_strategy': None,
            'expected_improvement': 0.0,
            'confidence': 0.0
        }
        
        try:
            # Determine context
            context_type = self._determine_context_type(task_context)
            
            # Analyze performance history for current strategy
            current_strategy = self._infer_current_strategy(performance_history)
            recommendation['current_strategy'] = current_strategy
            
            # Get all available strategies
            available_strategies = list(self.strategies.keys())
            
            # Select optimal strategy
            optimal_strategy = self.select_strategy(task_context, available_strategies)
            recommendation['recommended_strategy'] = optimal_strategy
            
            # Calculate expected improvement
            if current_strategy and current_strategy != optimal_strategy:
                current_effectiveness = self._get_strategy_effectiveness(
                    current_strategy, context_type
                )
                optimal_effectiveness = self._get_strategy_effectiveness(
                    optimal_strategy, context_type
                )
                expected_improvement = optimal_effectiveness - current_effectiveness
                recommendation['expected_improvement'] = max(expected_improvement, 0.0)
            
            # Calculate confidence
            recommendation['confidence'] = self._calculate_selection_confidence(
                optimal_strategy, context_type, task_context
            )
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal strategy: {e}")
            recommendation['error'] = str(e)
        
        return recommendation
    
    def evaluate_current_strategies(self) -> Dict[str, Any]:
        """Evaluate current strategies for meta-cognitive core."""
        evaluation = {
            'strategy_count': len(self.strategies),
            'performance_summary': {},
            'adaptation_rate': 0.0,
            'exploration_effectiveness': 0.0
        }
        
        try:
            # Performance summary across all strategies
            performance_summary = {}
            for strategy_id, effectiveness_map in self.strategy_effectiveness.items():
                if effectiveness_map:
                    avg_effectiveness = sum(effectiveness_map.values()) / len(effectiveness_map)
                    performance_summary[strategy_id] = avg_effectiveness
            
            evaluation['performance_summary'] = performance_summary
            
            # Calculate adaptation rate
            if self.adaptation_history:
                recent_adaptations = len([
                    a for a in self.adaptation_history[-20:] 
                    if time.time() - a.get('timestamp', 0) < 3600  # Last hour
                ])
                evaluation['adaptation_rate'] = recent_adaptations / 20.0
            
            # Exploration effectiveness
            if self.strategy_performance_history:
                recent_explorations = [
                    p for p in self.strategy_performance_history[-50:]
                    if self._was_exploration(p)
                ]
                if recent_explorations:
                    avg_exploration_performance = sum(
                        p.success_rate for p in recent_explorations
                    ) / len(recent_explorations)
                    evaluation['exploration_effectiveness'] = avg_exploration_performance
            
        except Exception as e:
            self.logger.error(f"Error evaluating current strategies: {e}")
            evaluation['error'] = str(e)
        
        return evaluation
    
    def get_strategy_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get multiple strategy recommendations with rationales."""
        recommendations = []
        
        try:
            context_type = self._determine_context_type(context)
            available_strategies = list(self.strategies.keys())
            
            # Score all strategies for this context
            strategy_scores = {}
            for strategy_id in available_strategies:
                score = self._calculate_strategy_score(strategy_id, context_type, context)
                strategy_scores[strategy_id] = score
            
            # Sort by score and create recommendations
            sorted_strategies = sorted(
                strategy_scores.items(), key=lambda x: x[1], reverse=True
            )
            
            for i, (strategy_id, score) in enumerate(sorted_strategies[:5]):  # Top 5
                recommendation = {
                    'strategy_id': strategy_id,
                    'strategy_name': self.strategies[strategy_id].name,
                    'score': score,
                    'rank': i + 1,
                    'rationale': self._generate_rationale(strategy_id, context_type, score),
                    'expected_performance': self._estimate_performance(strategy_id, context)
                }
                recommendations.append(recommendation)
        
        except Exception as e:
            self.logger.error(f"Error generating strategy recommendations: {e}")
            recommendations.append({'error': str(e)})
        
        return recommendations
    
    # Private helper methods
    def _initialize_default_strategies(self) -> None:
        """Initialize the default strategy library."""
        default_strategies = [
            Strategy(
                strategy_id="analytical",
                strategy_type=StrategyType.ANALYTICAL,
                name="Analytical Strategy",
                description="Systematic analysis using logical reasoning",
                contexts=[ContextType.PROBLEM_SOLVING, ContextType.ANALYSIS],
                resource_requirements={'memory': 0.6, 'processing': 0.7, 'attention': 0.8},
                expected_performance={'accuracy': 0.8, 'speed': 0.6, 'reliability': 0.9},
                complexity_level=3,
                adaptation_rate=0.3
            ),
            Strategy(
                strategy_id="intuitive",
                strategy_type=StrategyType.INTUITIVE,
                name="Intuitive Strategy", 
                description="Pattern-based intuitive reasoning",
                contexts=[ContextType.CREATIVITY, ContextType.PATTERN_RECOGNITION],
                resource_requirements={'memory': 0.4, 'processing': 0.5, 'attention': 0.6},
                expected_performance={'accuracy': 0.6, 'speed': 0.8, 'creativity': 0.9},
                complexity_level=2,
                adaptation_rate=0.7
            ),
            Strategy(
                strategy_id="systematic",
                strategy_type=StrategyType.SYSTEMATIC,
                name="Systematic Strategy",
                description="Step-by-step systematic approach",
                contexts=[ContextType.LEARNING, ContextType.OPTIMIZATION],
                resource_requirements={'memory': 0.7, 'processing': 0.6, 'attention': 0.9},
                expected_performance={'accuracy': 0.9, 'speed': 0.5, 'completeness': 0.9},
                complexity_level=4,
                adaptation_rate=0.2
            ),
            Strategy(
                strategy_id="creative",
                strategy_type=StrategyType.CREATIVE,
                name="Creative Strategy",
                description="Divergent thinking and creative exploration",
                contexts=[ContextType.CREATIVITY, ContextType.SYNTHESIS],
                resource_requirements={'memory': 0.5, 'processing': 0.6, 'attention': 0.7},
                expected_performance={'creativity': 0.9, 'novelty': 0.8, 'flexibility': 0.8},
                complexity_level=3,
                adaptation_rate=0.8
            ),
            Strategy(
                strategy_id="adaptive",
                strategy_type=StrategyType.ADAPTIVE,
                name="Adaptive Strategy",
                description="Context-adaptive flexible approach",
                contexts=[ContextType.DECISION_MAKING, ContextType.LEARNING],
                resource_requirements={'memory': 0.6, 'processing': 0.8, 'attention': 0.7},
                expected_performance={'flexibility': 0.9, 'adaptability': 0.9, 'resilience': 0.8},
                complexity_level=5,
                adaptation_rate=0.9
            )
        ]
        
        for strategy in default_strategies:
            self.strategies[strategy.strategy_id] = strategy
    
    def _build_context_strategy_mappings(self) -> None:
        """Build mappings between contexts and suitable strategies."""
        for context_type in ContextType:
            self.context_strategy_map[context_type] = []
            
        for strategy_id, strategy in self.strategies.items():
            for context in strategy.contexts:
                if context not in self.context_strategy_map:
                    self.context_strategy_map[context] = []
                self.context_strategy_map[context].append(strategy_id)
    
    def _initialize_performance_baselines(self) -> None:
        """Initialize performance baselines for strategies."""
        for strategy_id, strategy in self.strategies.items():
            self.strategy_effectiveness[strategy_id] = {}
            for context in strategy.contexts:
                # Initialize with expected performance
                baseline_effectiveness = sum(strategy.expected_performance.values()) / len(strategy.expected_performance)
                self.strategy_effectiveness[strategy_id][context.value] = baseline_effectiveness
    
    def _determine_context_type(self, task_context: Dict[str, Any]) -> ContextType:
        """Determine the context type from task context."""
        # Simple heuristic-based context classification
        context_keywords = {
            ContextType.PROBLEM_SOLVING: ['problem', 'solve', 'solution', 'issue'],
            ContextType.LEARNING: ['learn', 'study', 'understand', 'knowledge'],
            ContextType.DECISION_MAKING: ['decide', 'choice', 'option', 'alternative'],
            ContextType.PATTERN_RECOGNITION: ['pattern', 'recognize', 'identify', 'classify'],
            ContextType.CREATIVITY: ['create', 'generate', 'innovate', 'creative'],
            ContextType.OPTIMIZATION: ['optimize', 'improve', 'enhance', 'efficient'],
            ContextType.ANALYSIS: ['analyze', 'examine', 'investigate', 'study'],
            ContextType.SYNTHESIS: ['synthesize', 'combine', 'integrate', 'merge']
        }
        
        # Count keyword matches
        context_scores = {}
        task_text = str(task_context).lower()
        
        for context_type, keywords in context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_text)
            if score > 0:
                context_scores[context_type] = score
        
        # Return context with highest score, or default to PROBLEM_SOLVING
        if context_scores:
            return max(context_scores.items(), key=lambda x: x[1])[0]
        else:
            return ContextType.PROBLEM_SOLVING
    
    def _get_strategy_recommendation(self, 
                                   context_type: ContextType,
                                   task_context: Dict[str, Any],
                                   available_strategies: List[str]) -> StrategyRecommendation:
        """Get strategy recommendation for context."""
        # Filter strategies suitable for context
        suitable_strategies = [
            s for s in available_strategies 
            if s in self.strategies and context_type in self.strategies[s].contexts
        ]
        
        if not suitable_strategies:
            suitable_strategies = available_strategies  # Fallback to all available
        
        # Score strategies
        strategy_scores = {}
        for strategy_id in suitable_strategies:
            score = self._calculate_strategy_score(strategy_id, context_type, task_context)
            strategy_scores[strategy_id] = score
        
        # Select best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
        
        # Create recommendation
        recommendation = StrategyRecommendation(
            recommended_strategy=best_strategy[0],
            confidence=best_strategy[1],
            expected_performance=self._estimate_performance(best_strategy[0], task_context),
            rationale=self._generate_rationale(best_strategy[0], context_type, best_strategy[1]),
            alternatives=sorted(
                [(s, score) for s, score in strategy_scores.items() if s != best_strategy[0]],
                key=lambda x: x[1], reverse=True
            )[:3],  # Top 3 alternatives
            context_match=self._calculate_context_match(best_strategy[0], context_type)
        )
        
        return recommendation
    
    def _calculate_strategy_score(self, 
                                strategy_id: str,
                                context_type: ContextType,
                                task_context: Dict[str, Any]) -> float:
        """Calculate score for a strategy in given context."""
        if strategy_id not in self.strategies:
            return 0.0
        
        strategy = self.strategies[strategy_id]
        score = 0.0
        
        # Context match score
        if context_type in strategy.contexts:
            score += 0.4
        
        # Historical effectiveness score
        effectiveness = self.strategy_effectiveness.get(strategy_id, {}).get(context_type.value, 0.5)
        score += effectiveness * 0.3
        
        # Resource availability score
        available_resources = task_context.get('available_resources', {'memory': 1.0, 'processing': 1.0, 'attention': 1.0})
        resource_match = 1.0
        for resource, required in strategy.resource_requirements.items():
            available = available_resources.get(resource, 1.0)
            if available < required:
                resource_match *= available / required
        score += resource_match * 0.2
        
        # Complexity appropriateness score
        task_complexity = task_context.get('complexity', 0.5)
        complexity_match = 1.0 - abs(strategy.complexity_level / 5.0 - task_complexity)
        score += complexity_match * 0.1
        
        return min(score, 1.0)
    
    def _apply_exploration_exploitation(self, 
                                      recommendation: StrategyRecommendation,
                                      available_strategies: List[str]) -> str:
        """Apply exploration-exploitation balance to strategy selection."""
        # Exploitation: use recommended strategy
        if random.random() > self.exploration_rate:
            return recommendation.recommended_strategy
        
        # Exploration: try alternative strategies
        # Bias towards less-tested strategies
        strategy_test_counts = {}
        for strategy_id in available_strategies:
            count = len([
                p for p in self.strategy_performance_history 
                if p.strategy_id == strategy_id
            ])
            strategy_test_counts[strategy_id] = count
        
        # Select strategy with lowest test count (with some randomness)
        min_count = min(strategy_test_counts.values()) if strategy_test_counts else 0
        exploration_candidates = [
            s for s, count in strategy_test_counts.items() 
            if count <= min_count + 2
        ]
        
        return random.choice(exploration_candidates) if exploration_candidates else recommendation.recommended_strategy
    
    def _log_strategy_selection(self, 
                              selected_strategy: str,
                              context_type: ContextType,
                              task_context: Dict[str, Any],
                              recommendation: StrategyRecommendation) -> None:
        """Log strategy selection for learning."""
        selection_log = {
            'selected_strategy': selected_strategy,
            'context_type': context_type.value,
            'task_context_summary': self._summarize_task_context(task_context),
            'recommendation_confidence': recommendation.confidence,
            'was_exploration': selected_strategy != recommendation.recommended_strategy,
            'timestamp': time.time()
        }
        
        # Store in adaptation history for analysis
        self.adaptation_history.append(selection_log)
        
        # Keep history bounded
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-800:]
    
    def _determine_adaptation_type(self, 
                                 performance_issues: List[str],
                                 resource_constraints: List[str],
                                 context_changes: Dict[str, Any]) -> str:
        """Determine what type of adaptation is needed."""
        if performance_issues:
            if 'accuracy' in performance_issues or 'quality' in performance_issues:
                return 'improve_accuracy'
            elif 'speed' in performance_issues or 'efficiency' in performance_issues:
                return 'improve_speed'
        
        if resource_constraints:
            return 'reduce_resources'
        
        if context_changes:
            return 'adapt_context'
        
        return 'general_adaptation'
    
    def _apply_strategy_adaptation(self, 
                                 current_strategy: str,
                                 adaptation_type: str,
                                 feedback: Dict[str, Any]) -> str:
        """Apply strategy adaptation based on type and feedback."""
        # Strategy adaptation logic
        adaptations = {
            'improve_accuracy': {
                'intuitive': 'analytical',
                'creative': 'systematic',
                'exploratory': 'focused'
            },
            'improve_speed': {
                'systematic': 'intuitive',
                'analytical': 'parallel',
                'sequential': 'parallel'
            },
            'reduce_resources': {
                'systematic': 'intuitive',
                'parallel': 'sequential',
                'analytical': 'focused'
            },
            'adapt_context': {
                # Any strategy can adapt to adaptive
                'any': 'adaptive'
            }
        }
        
        # Get adaptation mapping
        adaptation_map = adaptations.get(adaptation_type, {})
        
        # Apply adaptation
        if current_strategy in adaptation_map:
            return adaptation_map[current_strategy]
        elif adaptation_type == 'adapt_context':
            return 'adaptive'
        else:
            return current_strategy  # No adaptation available
    
    def _record_adaptation(self, 
                         original_strategy: str,
                         adapted_strategy: str,
                         feedback: Dict[str, Any]) -> None:
        """Record strategy adaptation for learning."""
        adaptation_record = {
            'original_strategy': original_strategy,
            'adapted_strategy': adapted_strategy,
            'feedback_summary': self._summarize_feedback(feedback),
            'adaptation_reason': feedback.get('reason', 'performance_feedback'),
            'timestamp': time.time()
        }
        
        self.adaptation_history.append(adaptation_record)
    
    def _infer_current_strategy(self, performance_history: List[Dict[str, Any]]) -> Optional[str]:
        """Infer current strategy from performance history."""
        if not performance_history:
            return None
        
        # Look for strategy indicators in recent history
        recent_history = performance_history[-5:]  # Last 5 entries
        
        # Simple heuristic: look for consistent patterns
        for entry in reversed(recent_history):
            if 'strategy' in entry:
                return entry['strategy']
            elif 'context_mode' in entry:
                # Map context modes to strategies
                mode_strategy_map = {
                    'monitoring': 'systematic',
                    'evaluating': 'analytical',
                    'controlling': 'adaptive',
                    'planning': 'systematic',
                    'reflecting': 'reflective'
                }
                mode = entry['context_mode']
                if mode in mode_strategy_map:
                    return mode_strategy_map[mode]
        
        return 'analytical'  # Default assumption
    
    def _get_strategy_effectiveness(self, strategy_id: str, context_type: ContextType) -> float:
        """Get effectiveness of a strategy for a context type."""
        return self.strategy_effectiveness.get(strategy_id, {}).get(context_type.value, 0.5)
    
    def _calculate_selection_confidence(self, 
                                      strategy_id: str,
                                      context_type: ContextType,
                                      task_context: Dict[str, Any]) -> float:
        """Calculate confidence in strategy selection."""
        base_confidence = 0.5
        
        # Increase confidence based on historical effectiveness
        effectiveness = self._get_strategy_effectiveness(strategy_id, context_type)
        base_confidence += (effectiveness - 0.5) * 0.4
        
        # Increase confidence based on context match
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            if context_type in strategy.contexts:
                base_confidence += 0.2
        
        # Decrease confidence if strategy is untested
        test_count = len([
            p for p in self.strategy_performance_history 
            if p.strategy_id == strategy_id
        ])
        if test_count < 3:
            base_confidence *= 0.8
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _was_exploration(self, performance: StrategyPerformance) -> bool:
        """Check if a performance record was from exploration."""
        # Simple heuristic: low-tested strategies are likely exploration
        test_count = len([
            p for p in self.strategy_performance_history 
            if p.strategy_id == performance.strategy_id and p.timestamp < performance.timestamp
        ])
        return test_count < 5
    
    def _estimate_performance(self, strategy_id: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Estimate expected performance for a strategy in context."""
        if strategy_id not in self.strategies:
            return {'accuracy': 0.5, 'speed': 0.5, 'efficiency': 0.5}
        
        strategy = self.strategies[strategy_id]
        return strategy.expected_performance.copy()
    
    def _generate_rationale(self, strategy_id: str, context_type: ContextType, score: float) -> List[str]:
        """Generate rationale for strategy selection."""
        rationale = []
        
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            
            if context_type in strategy.contexts:
                rationale.append(f"Strategy is designed for {context_type.value} contexts")
            
            effectiveness = self._get_strategy_effectiveness(strategy_id, context_type)
            if effectiveness > 0.7:
                rationale.append("Strategy has high historical effectiveness")
            elif effectiveness < 0.3:
                rationale.append("Strategy has low historical effectiveness but may benefit from exploration")
            
            if score > 0.8:
                rationale.append("High overall score based on context match and performance")
            elif score < 0.4:
                rationale.append("Lower score but may be worth exploring")
        
        return rationale
    
    def _calculate_context_match(self, strategy_id: str, context_type: ContextType) -> float:
        """Calculate how well a strategy matches the context."""
        if strategy_id not in self.strategies:
            return 0.0
        
        strategy = self.strategies[strategy_id]
        if context_type in strategy.contexts:
            return 1.0
        else:
            return 0.3  # Partial match for flexibility
    
    def _summarize_task_context(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize task context for logging."""
        return {
            'task_type': task_context.get('task_type', 'unknown'),
            'complexity': task_context.get('complexity', 0.5),
            'resource_constraints': bool(task_context.get('resource_constraints')),
            'time_constraints': bool(task_context.get('time_constraints'))
        }
    
    def _summarize_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize feedback for logging."""
        return {
            'performance_issues': len(feedback.get('performance_issues', [])),
            'resource_constraints': len(feedback.get('resource_constraints', [])),
            'context_changes': bool(feedback.get('context_changes')),
            'overall_satisfaction': feedback.get('satisfaction', 0.5)
        }