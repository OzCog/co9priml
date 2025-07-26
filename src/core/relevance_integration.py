"""
Integration module for Relevance Optimization System with Cognitive Architecture

This module provides integration utilities to connect the advanced relevance optimization
system with the existing cognitive core and other components.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..core.relevance_optimization import RelevanceOptimizer, RelevanceMetrics, SalienceType
from ..core.relevance_core import RelevanceCore, RelevanceMode
from ..core.cognitive_core import CogPrimeCore
from ..modules.perception import SensoryInput
from ..modules.action import Action


class EnhancedCognitiveCore(CogPrimeCore):
    """
    Enhanced cognitive core with integrated relevance optimization system.
    
    Extends the base CogPrimeCore to include advanced relevance optimization
    capabilities for improved cognitive resource allocation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Initialize relevance optimization system
        base_relevance_core = RelevanceCore()
        self.relevance_optimizer = RelevanceOptimizer(
            base_relevance_core,
            config.get('relevance_optimization_config', {})
        )
        
        # Performance tracking
        self.baseline_metrics: Optional[RelevanceMetrics] = None
        self.performance_history: List[RelevanceMetrics] = []
        
        # Enhanced cognitive state tracking
        self.environmental_state: Dict[str, Any] = {}
        self.goal_priorities: Dict[str, float] = {}
        
    def enhanced_cognitive_cycle(self, sensory_input: SensoryInput, 
                                reward: float = 0.0,
                                environment_data: Dict[str, Any] = None) -> Optional[Action]:
        """
        Enhanced cognitive cycle with relevance optimization.
        
        Args:
            sensory_input: Sensory input data
            reward: Reward signal from environment
            environment_data: Environmental state data for salience detection
            
        Returns:
            Selected action with optimized relevance-based processing
        """
        # Update environmental state
        if environment_data:
            self.environmental_state.update(environment_data)
        
        # Detect environmental changes requiring attention
        environmental_signals = []
        if environment_data:
            environmental_signals = self.relevance_optimizer.detect_environmental_salience(
                environment_data, self._get_current_context()
            )
        
        # Extract current goals and priorities
        active_goals = [goal for goal in self.state.goal_stack]
        
        # Compute relevance scores for current sensory input
        input_items = self._extract_items_from_sensory_input(sensory_input)
        relevance_scores = {}
        
        for item in input_items:
            score, components = self.relevance_optimizer.compute_relevance_score(
                item, self._get_current_context(), active_goals
            )
            relevance_scores[item] = score
        
        # Optimize attention allocation based on relevance
        current_attention = self._get_current_attention_allocation()
        optimized_attention = self.relevance_optimizer.allocate_attention_dynamically(
            relevance_scores, current_attention, self._get_current_context()
        )
        
        # Update attention focus based on optimization
        self._apply_attention_allocation(optimized_attention)
        
        # Align processing with goals
        goal_alignments = self.relevance_optimizer.align_with_goals(
            relevance_scores, active_goals, self._get_current_context()
        )
        
        # Apply goal alignment recommendations
        self._apply_goal_alignment_recommendations(goal_alignments)
        
        # Execute standard cognitive cycle with optimized processing
        action = self.cognitive_cycle(sensory_input, reward)
        
        # Learn from feedback
        if action:
            learning_stats = self.relevance_optimizer.learn_from_feedback(
                self._get_current_context(),
                [action.name],
                [reward],
                active_goals
            )
            
            # Store learning stats in working memory
            self.state.working_memory['relevance_learning_stats'] = learning_stats
        
        # Measure and track performance
        current_metrics = self.relevance_optimizer.measure_performance(
            self._get_current_context(), self.baseline_metrics
        )
        self.performance_history.append(current_metrics)
        
        # Set baseline if not established
        if self.baseline_metrics is None:
            self.baseline_metrics = current_metrics
        
        # Store optimization results in working memory
        self.state.working_memory.update({
            'environmental_signals': environmental_signals,
            'relevance_scores': relevance_scores,
            'attention_allocation': optimized_attention,
            'goal_alignments': goal_alignments,
            'performance_metrics': current_metrics
        })
        
        return action
    
    def optimize_memory_retrieval(self, query: str, 
                                 memory_items: Dict[str, Any] = None) -> List[Tuple[str, float]]:
        """
        Retrieve memories using relevance optimization.
        
        Args:
            query: Query for memory retrieval
            memory_items: Optional specific memory items to search
            
        Returns:
            List of (memory_id, relevance_score) pairs
        """
        if memory_items is None:
            # In a full implementation, this would interface with the memory system
            memory_items = self._get_memory_items_from_system()
        
        return self.relevance_optimizer.optimize_memory_retrieval(
            query, self._get_current_context(), memory_items
        )
    
    def get_relevance_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive relevance optimization performance report.
        
        Returns:
            Performance report with metrics and improvements
        """
        if not self.performance_history:
            return {'status': 'No performance data available'}
        
        latest_metrics = self.performance_history[-1]
        
        # Calculate trends over last 10 measurements
        recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        trends = {}
        if len(recent_metrics) > 1:
            for metric_name in ['attention_efficiency', 'cognitive_load_reduction', 
                              'goal_alignment_score', 'environmental_responsiveness', 
                              'memory_retrieval_accuracy']:
                values = [getattr(m, metric_name) for m in recent_metrics]
                trend = np.polyfit(range(len(values)), values, 1)[0]  # Linear trend
                trends[metric_name] = trend
        
        report = {
            'current_performance': {
                'attention_efficiency': latest_metrics.attention_efficiency,
                'cognitive_load_reduction': latest_metrics.cognitive_load_reduction,
                'goal_alignment_score': latest_metrics.goal_alignment_score,
                'environmental_responsiveness': latest_metrics.environmental_responsiveness,
                'memory_retrieval_accuracy': latest_metrics.memory_retrieval_accuracy,
                'overall_improvement': latest_metrics.overall_performance_improvement
            },
            'performance_trends': trends,
            'baseline_comparison': {
                'improvement_vs_baseline': latest_metrics.overall_performance_improvement,
                'meets_35_percent_target': latest_metrics.overall_performance_improvement >= 0.35
            },
            'optimization_status': {
                'total_measurements': len(self.performance_history),
                'environmental_signals_detected': len(self.relevance_optimizer.environmental_signals),
                'active_goals_tracked': len(self.relevance_optimizer.active_goals),
                'adaptive_thresholds': self.relevance_optimizer.adaptive_thresholds
            }
        }
        
        return report
    
    def set_goal_priority(self, goal: str, priority: float) -> None:
        """
        Set priority for a specific goal.
        
        Args:
            goal: Goal identifier
            priority: Priority value (0.0 to 1.0)
        """
        self.goal_priorities[goal] = np.clip(priority, 0.0, 1.0)
        
        # Update relevance optimizer's goal tracking
        if goal not in self.relevance_optimizer.active_goals:
            self.relevance_optimizer.active_goals[goal] = {
                'created_time': np.datetime64('now').astype(float),
                'priority': priority,
                'context': self._get_current_context()
            }
        else:
            self.relevance_optimizer.active_goals[goal]['priority'] = priority
    
    def configure_salience_weights(self, salience_weights: Dict[SalienceType, float]) -> None:
        """
        Configure the importance weights for different salience types.
        
        Args:
            salience_weights: Dictionary mapping salience types to weight values
        """
        for salience_type, weight in salience_weights.items():
            if salience_type in self.relevance_optimizer.importance_weights:
                self.relevance_optimizer.importance_weights[salience_type] = np.clip(weight, 0.0, 1.0)
        
        # Renormalize weights
        total_weight = sum(self.relevance_optimizer.importance_weights.values())
        if total_weight > 0:
            for salience_type in self.relevance_optimizer.importance_weights:
                self.relevance_optimizer.importance_weights[salience_type] /= total_weight
    
    # Helper methods for integration
    
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for relevance computation"""
        context = {
            'emotional_valence': self.state.emotional_valence,
            'active_goals': [goal for goal in self.state.goal_stack],
            'last_reward': self.state.last_reward,
            'total_reward': self.state.total_reward,
            'working_memory_items': len(self.state.working_memory),
            'current_domain': getattr(self, 'current_domain', 'general'),
            'current_task': getattr(self, 'current_task', 'cognitive_processing')
        }
        
        # Add environmental context
        context.update(self.environmental_state)
        
        # Add goal priorities
        if self.goal_priorities:
            context['goal_priorities'] = self.goal_priorities
        
        return context
    
    def _extract_items_from_sensory_input(self, sensory_input: SensoryInput) -> List[str]:
        """Extract discrete items from sensory input for relevance scoring"""
        items = []
        
        # Create descriptive items from sensory modalities
        if hasattr(sensory_input, 'visual') and sensory_input.visual is not None:
            items.extend([
                'visual_input', 'visual_features', 'visual_attention_targets'
            ])
        
        if hasattr(sensory_input, 'auditory') and sensory_input.auditory is not None:
            items.extend([
                'auditory_input', 'audio_features', 'sound_patterns'
            ])
        
        # Add items from working memory context
        for key in self.state.working_memory.keys():
            items.append(f"memory_{key}")
        
        # Add goal-related items
        for goal in self.state.goal_stack:
            items.append(f"goal_{goal}")
        
        return items
    
    def _get_current_attention_allocation(self) -> Dict[RelevanceMode, float]:
        """Get current attention allocation across relevance modes"""
        # In a full implementation, this would derive from the actual attention state
        # For now, use the optimizer's current allocation
        return self.relevance_optimizer.attention_allocation.copy()
    
    def _apply_attention_allocation(self, allocation: Dict[RelevanceMode, float]) -> None:
        """Apply optimized attention allocation to cognitive state"""
        # Update the relevance optimizer's allocation
        self.relevance_optimizer.attention_allocation = allocation
        
        # In a full implementation, this would modify the actual attention mechanisms
        # For now, store in working memory for tracking
        self.state.working_memory['optimized_attention_allocation'] = allocation
    
    def _apply_goal_alignment_recommendations(self, alignments: List) -> None:
        """Apply goal alignment recommendations to improve focus"""
        for alignment in alignments:
            goal = alignment.goal_id
            recommendations = alignment.recommended_adjustments
            
            # Apply recommendations (simplified implementation)
            if 'increase_goal_focus' in recommendations:
                increase = recommendations['increase_goal_focus']
                self.set_goal_priority(goal, min(1.0, self.goal_priorities.get(goal, 0.5) + increase))
            
            if 'reduce_goal_focus' in recommendations:
                decrease = recommendations['reduce_goal_focus']
                self.set_goal_priority(goal, max(0.0, self.goal_priorities.get(goal, 0.5) - decrease))
    
    def _get_memory_items_from_system(self) -> Dict[str, Any]:
        """Get memory items from the memory system"""
        # In a full implementation, this would interface with the actual memory system
        # For now, return items from working memory and recent thoughts
        memory_items = {}
        
        # Add working memory items
        for key, value in self.state.working_memory.items():
            memory_items[f"wm_{key}"] = {
                'content': str(value),
                'timestamp': np.datetime64('now').astype(float),
                'source': 'working_memory'
            }
        
        # Add recent thoughts if available
        if self.state.current_thought:
            memory_items['current_thought'] = {
                'content': self.state.current_thought.content,
                'timestamp': np.datetime64('now').astype(float),
                'source': 'current_thought',
                'salience': self.state.current_thought.salience
            }
        
        return memory_items


class RelevanceOptimizationDemo:
    """
    Demonstration class showing how to use the relevance optimization system.
    """
    
    def __init__(self):
        self.cognitive_core = EnhancedCognitiveCore({
            'relevance_optimization_config': {
                'novelty_threshold': 0.5,
                'attention_capacity': 1.0,
                'threshold_adaptation_rate': 0.1
            }
        })
    
    def demonstrate_environmental_adaptation(self):
        """Demonstrate environmental salience detection and adaptation"""
        print("=== Environmental Adaptation Demo ===")
        
        # Simulate changing environment
        environments = [
            {'light_level': 70, 'noise_level': 40, 'temperature': 20},
            {'light_level': 90, 'noise_level': 45, 'temperature': 22},  # Bright and noisy
            {'light_level': 30, 'noise_level': 20, 'temperature': 18},  # Dim and quiet
        ]
        
        for i, env_data in enumerate(environments):
            print(f"\nEnvironment {i+1}: {env_data}")
            
            # Create dummy sensory input
            sensory_input = SensoryInput(
                visual=np.random.randn(784),
                auditory=np.random.randn(256)
            )
            
            # Run enhanced cognitive cycle
            action = self.cognitive_core.enhanced_cognitive_cycle(
                sensory_input, reward=0.5, environment_data=env_data
            )
            
            # Check detected signals
            signals = self.cognitive_core.state.working_memory.get('environmental_signals', [])
            print(f"Detected {len(signals)} environmental changes")
            
            for signal in signals:
                print(f"  - {signal.signal_type}: intensity={signal.intensity:.3f}, novelty={signal.novelty:.3f}")
    
    def demonstrate_goal_alignment(self):
        """Demonstrate goal-relevance alignment"""
        print("\n=== Goal Alignment Demo ===")
        
        # Set up goals with different priorities
        goals = [
            ('learn_patterns', 0.8),
            ('solve_problems', 0.6),
            ('explore_environment', 0.4)
        ]
        
        for goal, priority in goals:
            self.cognitive_core.update_goals(goal)
            self.cognitive_core.set_goal_priority(goal, priority)
        
        # Simulate cognitive cycle with goal-relevant input
        sensory_input = SensoryInput(
            visual=np.random.randn(784),
            auditory=np.random.randn(256)
        )
        
        action = self.cognitive_core.enhanced_cognitive_cycle(sensory_input, reward=0.7)
        
        # Display goal alignments
        alignments = self.cognitive_core.state.working_memory.get('goal_alignments', [])
        print(f"Goal alignment analysis:")
        
        for alignment in alignments:
            print(f"  Goal: {alignment.goal_id}")
            print(f"    Current relevance: {alignment.current_relevance:.3f}")
            print(f"    Optimal relevance: {alignment.optimal_relevance:.3f}")
            print(f"    Alignment score: {alignment.alignment_score:.3f}")
            
            if alignment.recommended_adjustments:
                print(f"    Recommendations: {alignment.recommended_adjustments}")
    
    def demonstrate_memory_optimization(self):
        """Demonstrate memory retrieval optimization"""
        print("\n=== Memory Retrieval Optimization Demo ===")
        
        # Create sample memory items
        memory_items = {
            'memory_1': {'content': 'pattern recognition algorithm', 'timestamp': 100},
            'memory_2': {'content': 'environmental adaptation strategy', 'timestamp': 200},
            'memory_3': {'content': 'goal-directed behavior model', 'timestamp': 150},
            'memory_4': {'content': 'unrelated random facts', 'timestamp': 80},
            'memory_5': {'content': 'cognitive load optimization technique', 'timestamp': 180}
        }
        
        # Test different queries
        queries = [
            'pattern recognition',
            'environmental adaptation',
            'goal achievement',
            'cognitive optimization'
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            ranked_items = self.cognitive_core.optimize_memory_retrieval(query, memory_items)
            
            print("Ranked results:")
            for i, (item_id, score) in enumerate(ranked_items[:3]):
                content = memory_items[item_id]['content']
                print(f"  {i+1}. {item_id}: {score:.3f} - {content}")
    
    def demonstrate_performance_tracking(self):
        """Demonstrate performance measurement and tracking"""
        print("\n=== Performance Tracking Demo ===")
        
        # Run several cognitive cycles to build performance history
        for i in range(10):
            sensory_input = SensoryInput(
                visual=np.random.randn(784),
                auditory=np.random.randn(256)
            )
            
            # Simulate varying rewards and environments
            reward = 0.5 + 0.3 * np.sin(i / 3.0)  # Varying performance
            env_data = {
                'light_level': 70 + 10 * np.sin(i / 2.0),
                'noise_level': 40 + 5 * np.cos(i / 2.0)
            }
            
            self.cognitive_core.enhanced_cognitive_cycle(
                sensory_input, reward=reward, environment_data=env_data
            )
        
        # Get performance report
        report = self.cognitive_core.get_relevance_performance_report()
        
        print("Performance Report:")
        print(f"  Current Performance:")
        for metric, value in report['current_performance'].items():
            print(f"    {metric}: {value:.3f}")
        
        print(f"\n  Baseline Comparison:")
        print(f"    Overall improvement: {report['baseline_comparison']['improvement_vs_baseline']:.1%}")
        print(f"    Meets 35% target: {report['baseline_comparison']['meets_35_percent_target']}")
        
        print(f"\n  Optimization Status:")
        for metric, value in report['optimization_status'].items():
            print(f"    {metric}: {value}")
    
    def run_full_demonstration(self):
        """Run complete demonstration of relevance optimization features"""
        print("🧠 RELEVANCE OPTIMIZATION SYSTEM DEMONSTRATION 🧠")
        print("=" * 60)
        
        self.demonstrate_environmental_adaptation()
        self.demonstrate_goal_alignment()
        self.demonstrate_memory_optimization()
        self.demonstrate_performance_tracking()
        
        print("\n" + "=" * 60)
        print("✅ RELEVANCE OPTIMIZATION DEMONSTRATION COMPLETE ✅")


if __name__ == "__main__":
    # Run the demonstration
    demo = RelevanceOptimizationDemo()
    demo.run_full_demonstration()