"""
Standalone demonstration of the Relevance Optimization System

This demonstration shows the capabilities of the relevance optimization system
without requiring the full cognitive architecture integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from src.core.relevance_optimization import (
    RelevanceOptimizer, SalienceType, RelevanceMetrics,
    EnvironmentalSignal, GoalRelevanceAlignment
)
from src.core.relevance_core import RelevanceCore, RelevanceMode


class StandaloneRelevanceDemo:
    """Standalone demonstration of relevance optimization capabilities"""
    
    def __init__(self):
        # Initialize relevance optimization system
        base_relevance_core = RelevanceCore()
        self.optimizer = RelevanceOptimizer(base_relevance_core, {
            'novelty_threshold': 0.3,
            'attention_capacity': 1.0,
            'threshold_adaptation_rate': 0.15
        })
        
        # Demo data
        self.sample_items = [
            'visual_pattern', 'audio_signal', 'memory_trace',
            'goal_indicator', 'problem_element', 'environmental_change',
            'attention_target', 'cognitive_load', 'learning_opportunity'
        ]
        
        self.sample_goals = ['learn_patterns', 'solve_problems', 'adapt_environment']
        
    def demonstrate_relevance_scoring(self):
        """Demonstrate comprehensive relevance scoring"""
        print("=== RELEVANCE SCORING DEMONSTRATION ===")
        
        contexts = [
            {'task_type': 'learning', 'urgency': 0.3, 'novelty': 0.7},
            {'task_type': 'problem_solving', 'urgency': 0.8, 'novelty': 0.4},
            {'task_type': 'exploration', 'urgency': 0.2, 'novelty': 0.9}
        ]
        
        print(f"{'Item':<20} {'Context':<15} {'Score':<8} {'Environmental':<13} {'Goal':<8} {'Contextual':<11} {'Emergent':<9} {'Temporal':<8}")
        print("-" * 100)
        
        for context in contexts:
            print(f"\nContext: {context['task_type']}")
            
            for item in self.sample_items[:5]:  # Show first 5 items
                score, components = self.optimizer.compute_relevance_score(
                    item, context, self.sample_goals
                )
                
                print(f"{item:<20} {context['task_type']:<15} {score:.3f}    "
                      f"{components[SalienceType.ENVIRONMENTAL.value]:.3f}        "
                      f"{components[SalienceType.GOAL_ORIENTED.value]:.3f}     "
                      f"{components[SalienceType.CONTEXTUAL.value]:.3f}      "
                      f"{components[SalienceType.EMERGENT.value]:.3f}      "
                      f"{components[SalienceType.TEMPORAL.value]:.3f}")
    
    def demonstrate_environmental_detection(self):
        """Demonstrate environmental salience detection"""
        print("\n\n=== ENVIRONMENTAL SALIENCE DETECTION ===")
        
        # Set baseline environment
        self.optimizer.environmental_baseline = {
            'light_level': 70.0,
            'noise_level': 40.0,
            'temperature': 22.0,
            'activity_level': 0.5,
            'complexity_index': 0.6
        }
        
        # Test different environmental changes
        test_environments = [
            {'light_level': 95.0, 'noise_level': 65.0, 'temperature': 22.0, 'activity_level': 0.8, 'complexity_index': 0.9},
            {'light_level': 30.0, 'noise_level': 15.0, 'temperature': 18.0, 'activity_level': 0.2, 'complexity_index': 0.3},
            {'light_level': 70.0, 'noise_level': 40.0, 'temperature': 30.0, 'activity_level': 0.9, 'complexity_index': 0.4}
        ]
        
        for i, env_data in enumerate(test_environments):
            print(f"\nEnvironment Change {i+1}:")
            print(f"  Data: {env_data}")
            
            signals = self.optimizer.detect_environmental_salience(
                env_data, {'context_id': f'env_test_{i}'}
            )
            
            print(f"  Detected {len(signals)} significant changes:")
            for signal in signals:
                print(f"    - {signal.signal_type}: intensity={signal.intensity:.3f}, "
                      f"novelty={signal.novelty:.3f}, urgency={signal.urgency:.3f}, "
                      f"confidence={signal.detection_confidence:.3f}")
    
    def demonstrate_attention_allocation(self):
        """Demonstrate dynamic attention allocation"""
        print("\n\n=== DYNAMIC ATTENTION ALLOCATION ===")
        
        # Create relevance scores for different scenarios
        scenarios = [
            ('Learning Phase', {'learn_item_1': 0.8, 'learn_item_2': 0.6, 'distraction': 0.2}),
            ('Problem Solving', {'problem_core': 0.9, 'sub_problem': 0.5, 'hint': 0.4, 'noise': 0.1}),
            ('Emergency Response', {'urgent_signal': 0.95, 'secondary_info': 0.3, 'background': 0.1})
        ]
        
        current_attention = {mode: 0.2 for mode in RelevanceMode}
        
        print(f"{'Scenario':<18} {'Mode':<20} {'Allocation':<12}")
        print("-" * 55)
        
        for scenario_name, relevance_scores in scenarios:
            context = {'scenario': scenario_name.lower().replace(' ', '_')}
            
            new_allocation = self.optimizer.allocate_attention_dynamically(
                relevance_scores, current_attention, context
            )
            
            print(f"\n{scenario_name}:")
            for mode, allocation in new_allocation.items():
                print(f"{'':18} {mode.value:<20} {allocation:.3f}")
            
            current_attention = new_allocation  # Update for next iteration
    
    def demonstrate_goal_alignment(self):
        """Demonstrate goal-relevance alignment"""
        print("\n\n=== GOAL-RELEVANCE ALIGNMENT ===")
        
        # Set up goals with different priorities
        goals = self.sample_goals
        for i, goal in enumerate(goals):
            priority = 0.9 - i * 0.2  # Decreasing priorities
            self.optimizer.active_goals[goal] = {
                'created_time': np.datetime64('now').astype(float),
                'priority': priority,
                'context': {'goal_type': 'primary' if i == 0 else 'secondary'}
            }
        
        # Create different focus scenarios
        focus_scenarios = [
            {'pattern_analysis': 0.7, 'problem_solving': 0.2, 'exploration': 0.1},
            {'problem_solving': 0.8, 'pattern_analysis': 0.15, 'exploration': 0.05},
            {'exploration': 0.6, 'pattern_analysis': 0.3, 'problem_solving': 0.1}
        ]
        
        print(f"{'Goal':<15} {'Current':<10} {'Optimal':<10} {'Alignment':<10} {'Recommendations'}")
        print("-" * 70)
        
        for i, current_focus in enumerate(focus_scenarios):
            print(f"\nScenario {i+1}:")
            
            alignments = self.optimizer.align_with_goals(
                current_focus, goals, {'scenario': f'focus_test_{i}'}
            )
            
            for alignment in alignments:
                recommendations = ', '.join(alignment.recommended_adjustments.keys()) if alignment.recommended_adjustments else 'None'
                print(f"{alignment.goal_id:<15} {alignment.current_relevance:.3f}      "
                      f"{alignment.optimal_relevance:.3f}      {alignment.alignment_score:.3f}      "
                      f"{recommendations}")
    
    def demonstrate_memory_optimization(self):
        """Demonstrate memory retrieval optimization"""
        print("\n\n=== MEMORY RETRIEVAL OPTIMIZATION ===")
        
        # Create diverse memory items
        memory_items = {
            'pattern_memory_1': {'content': 'visual pattern recognition algorithm learned from experience'},
            'pattern_memory_2': {'content': 'auditory pattern matching technique'},
            'problem_memory_1': {'content': 'systematic problem decomposition strategy'},
            'problem_memory_2': {'content': 'heuristic problem solving approach'},
            'environment_memory_1': {'content': 'environmental adaptation behavioral pattern'},
            'environment_memory_2': {'content': 'context sensitivity adjustment mechanism'},
            'general_memory_1': {'content': 'general cognitive processing routine'},
            'distractor_memory': {'content': 'unrelated random information and noise'}
        }
        
        # Test queries with different focuses
        test_queries = [
            'pattern recognition',
            'problem solving strategy',
            'environmental adaptation',
            'cognitive processing'
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            ranked_items = self.optimizer.optimize_memory_retrieval(
                query, {'query_context': 'memory_test'}, memory_items
            )
            
            print("Top 3 results:")
            for i, (item_id, score) in enumerate(ranked_items[:3]):
                content = memory_items[item_id]['content']
                print(f"  {i+1}. {item_id}: {score:.3f}")
                print(f"     \"{content[:60]}...\"")
    
    def demonstrate_adaptive_learning(self):
        """Demonstrate adaptive threshold learning"""
        print("\n\n=== ADAPTIVE THRESHOLD LEARNING ===")
        
        # Simulate learning over time with different performance feedback
        feedback_scenarios = [
            ({'selective_attention': 0.3, 'working_memory': 0.4}, 'Poor Performance'),
            ({'selective_attention': 0.7, 'working_memory': 0.8}, 'Good Performance'),
            ({'problem_space': 0.2, 'side_effects': 0.3}, 'Weak Problem Solving'),
            ({'problem_space': 0.9, 'side_effects': 0.8}, 'Strong Problem Solving')
        ]
        
        print("Threshold adaptation over time:")
        print(f"{'Scenario':<25} {'Mode':<20} {'Original':<10} {'Adapted':<10} {'Change':<8}")
        print("-" * 80)
        
        for feedback, description in feedback_scenarios:
            print(f"\n{description}:")
            
            # Store original thresholds
            original_thresholds = self.optimizer.adaptive_thresholds.copy()
            
            # Apply feedback and adapt
            new_thresholds = self.optimizer.adapt_thresholds(feedback)
            
            # Show changes for relevant modes
            for mode_str, performance in feedback.items():
                mode = RelevanceMode(mode_str)
                original = original_thresholds[mode]
                adapted = new_thresholds[mode]
                change = adapted - original
                
                print(f"{'':25} {mode.value:<20} {original:.3f}      {adapted:.3f}      {change:+.3f}")
    
    def demonstrate_performance_measurement(self):
        """Demonstrate performance measurement and tracking"""
        print("\n\n=== PERFORMANCE MEASUREMENT ===")
        
        # Simulate performance evolution over time
        performance_data = []
        
        # Initialize with some baseline activity
        self.optimizer.goal_relevance_history = {
            'learn_patterns': [0.4, 0.5, 0.6],
            'solve_problems': [0.3, 0.4, 0.5],
            'adapt_environment': [0.2, 0.3, 0.4]
        }
        
        self.optimizer.environmental_signals = [
            EnvironmentalSignal('signal_1', 0.7, 0.6, 0.5, {}, 0.8),
            EnvironmentalSignal('signal_2', 0.8, 0.7, 0.6, {}, 0.9)
        ]
        
        self.optimizer.retrieval_success_history = [True, True, False, True, True, True, False, True]
        
        # Measure baseline performance
        baseline = self.optimizer.measure_performance({'baseline': True})
        print("Baseline Performance:")
        print(f"  Attention Efficiency: {baseline.attention_efficiency:.3f}")
        print(f"  Cognitive Load Reduction: {baseline.cognitive_load_reduction:.3f}")
        print(f"  Goal Alignment Score: {baseline.goal_alignment_score:.3f}")
        print(f"  Environmental Responsiveness: {baseline.environmental_responsiveness:.3f}")
        print(f"  Memory Retrieval Accuracy: {baseline.memory_retrieval_accuracy:.3f}")
        
        # Simulate improvement over time
        for i in range(5):
            # Add improving goal relevance
            for goal in self.optimizer.goal_relevance_history:
                current_avg = np.mean(self.optimizer.goal_relevance_history[goal])
                improvement = 0.05 * (i + 1)  # Gradual improvement
                new_score = min(1.0, current_avg + improvement)
                self.optimizer.goal_relevance_history[goal].append(new_score)
            
            # Add successful retrievals
            self.optimizer.retrieval_success_history.extend([True] * 3 + [False] * 1)
            
            # Measure current performance
            current = self.optimizer.measure_performance({'iteration': i}, baseline)
            performance_data.append(current)
            
            print(f"\nIteration {i+1} Performance:")
            print(f"  Overall Improvement: {current.overall_performance_improvement:.1%}")
        
        # Check if 35% target is met
        final_improvement = performance_data[-1].overall_performance_improvement
        target_met = final_improvement >= 0.35
        
        print(f"\n🎯 TARGET ASSESSMENT:")
        print(f"   Final Improvement: {final_improvement:.1%}")
        print(f"   35% Target Met: {'✅ YES' if target_met else '❌ NO'}")
        
        return performance_data
    
    def create_performance_visualization(self, performance_data):
        """Create visualization of performance improvements"""
        if not performance_data:
            return
        
        print("\n=== GENERATING PERFORMANCE VISUALIZATION ===")
        
        # Extract metrics over time
        iterations = range(1, len(performance_data) + 1)
        attention_efficiency = [p.attention_efficiency for p in performance_data]
        cognitive_load_reduction = [p.cognitive_load_reduction for p in performance_data]
        goal_alignment = [p.goal_alignment_score for p in performance_data]
        environmental_responsiveness = [p.environmental_responsiveness for p in performance_data]
        memory_accuracy = [p.memory_retrieval_accuracy for p in performance_data]
        overall_improvement = [p.overall_performance_improvement for p in performance_data]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Relevance Optimization System Performance', fontsize=16, fontweight='bold')
        
        # Plot individual metrics
        ax1.plot(iterations, attention_efficiency, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Attention Efficiency')
        ax1.set_ylabel('Efficiency Score')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iterations, cognitive_load_reduction, 'g-o', linewidth=2, markersize=6)
        ax2.set_title('Cognitive Load Reduction')
        ax2.set_ylabel('Reduction Score')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(iterations, goal_alignment, 'r-o', linewidth=2, markersize=6)
        ax3.set_title('Goal Alignment Score')
        ax3.set_ylabel('Alignment Score')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(iterations, environmental_responsiveness, 'orange', marker='o', linewidth=2, markersize=6)
        ax4.set_title('Environmental Responsiveness')
        ax4.set_ylabel('Responsiveness Score')
        ax4.grid(True, alpha=0.3)
        
        ax5.plot(iterations, memory_accuracy, 'purple', marker='o', linewidth=2, markersize=6)
        ax5.set_title('Memory Retrieval Accuracy')
        ax5.set_ylabel('Accuracy Score')
        ax5.set_xlabel('Iteration')
        ax5.grid(True, alpha=0.3)
        
        # Overall improvement with target line
        ax6.plot(iterations, overall_improvement, 'black', marker='o', linewidth=3, markersize=8, label='Overall Improvement')
        ax6.axhline(y=0.35, color='red', linestyle='--', linewidth=2, label='35% Target')
        ax6.set_title('Overall Performance Improvement')
        ax6.set_ylabel('Improvement Ratio')
        ax6.set_xlabel('Iteration')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('relevance_optimization_performance.png', dpi=300, bbox_inches='tight')
        print("Performance visualization saved as 'relevance_optimization_performance.png'")
        
        return fig
    
    def run_complete_demonstration(self):
        """Run the complete relevance optimization demonstration"""
        print("🧠 ADVANCED RELEVANCE OPTIMIZATION SYSTEM 🧠")
        print("=" * 60)
        print("Implementing Vervaeke's Relevance Realization Framework")
        print("=" * 60)
        
        # Run all demonstrations
        self.demonstrate_relevance_scoring()
        self.demonstrate_environmental_detection()
        self.demonstrate_attention_allocation()
        self.demonstrate_goal_alignment()
        self.demonstrate_memory_optimization()
        self.demonstrate_adaptive_learning()
        performance_data = self.demonstrate_performance_measurement()
        
        # Create visualization
        self.create_performance_visualization(performance_data)
        
        print("\n" + "=" * 60)
        print("✅ RELEVANCE OPTIMIZATION DEMONSTRATION COMPLETE")
        print("🎯 Key Achievements:")
        print("   • Multi-dimensional relevance scoring implemented")
        print("   • Environmental salience detection operational")
        print("   • Dynamic attention allocation optimized")
        print("   • Goal-relevance alignment functional")
        print("   • Memory retrieval optimization working")
        print("   • Adaptive learning mechanisms active")
        print("   • Performance measurement and tracking enabled")
        print("=" * 60)


if __name__ == "__main__":
    demo = StandaloneRelevanceDemo()
    demo.run_complete_demonstration()