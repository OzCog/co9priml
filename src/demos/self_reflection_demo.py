#!/usr/bin/env python3
"""
Demonstration of the comprehensive self-reflection and introspection system.

This demo showcases the key features:
- Meta-cognitive monitoring architecture
- Cognitive state introspection functions
- Decision quality assessment mechanisms
- Performance tracking and analytics
- Self-optimization feedback loops
- Cognitive bias detection and correction
- Reflective learning mechanisms
- Confidence estimation for cognitive outputs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random
from cognitive_science.self_reflection import (
    SelfReflectionCore, ReflectionMode, CognitiveState, BiasType, MetricType,
    CognitiveMetrics, DecisionQuality, BiasDetection, OptimizationFeedback
)
from cognitive_science.math_utils import clip, mean, entropy


def demo_meta_cognitive_monitoring():
    """Demonstrate meta-cognitive monitoring capabilities"""
    print("🧠 Meta-Cognitive Monitoring Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Simulate cognitive state changes
    contexts = [
        {'active_processing': True, 'task': 'problem_solving'},
        {'learning_active': True, 'subject': 'mathematics'},
        {'reflection_triggered': True, 'depth': 0.8},
        {'optimization_needed': True, 'component': 'decision_making'},
        {'idle': True}
    ]
    
    print("Cognitive State Transitions:")
    for i, context in enumerate(contexts):
        state = core.monitor_cognitive_state(context)
        print(f"  Step {i+1}: {state.value} (context: {context})")
        time.sleep(0.1)  # Simulate processing time
    
    print(f"\nState transition history: {len(core.state_transitions)} transitions recorded")
    
    # Perform introspection at different depths
    print("\nIntrospection Analysis:")
    for depth in [0.3, 0.6, 0.9]:
        report = core.perform_introspection(depth=depth)
        print(f"  Depth {depth}: {len(report)} insights generated")
        if depth == 0.9:
            print(f"    - Meta-cognitive patterns: {len(report.get('state_transition_patterns', {}))}")
    
    return core


def demo_decision_quality_assessment():
    """Demonstrate decision quality assessment"""
    print("\n🎯 Decision Quality Assessment Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Simulate a series of decisions
    decisions = [
        {
            'id': 'investment_decision',
            'expected_outcome': 0.8,
            'context': {
                'information_quality': 0.9,
                'complexity': 0.6,
                'time_pressure': 0.2,
                'uncertainty': 0.3,
                'stakes': 0.8
            },
            'actual_outcome': 0.85  # Simulated outcome
        },
        {
            'id': 'hiring_decision',
            'expected_outcome': 0.7,
            'context': {
                'information_quality': 0.6,
                'complexity': 0.8,
                'time_pressure': 0.5,
                'uncertainty': 0.6,
                'stakes': 0.9
            },
            'actual_outcome': 0.5  # Simulated outcome
        },
        {
            'id': 'strategic_decision',
            'expected_outcome': 0.9,
            'context': {
                'information_quality': 0.8,
                'complexity': 0.9,
                'time_pressure': 0.7,
                'uncertainty': 0.8,
                'stakes': 0.95
            },
            'actual_outcome': 0.92  # Simulated outcome
        }
    ]
    
    print("Decision Quality Assessment:")
    for decision in decisions:
        # Assess decision quality
        quality = core.assess_decision_quality(
            decision['id'],
            decision['expected_outcome'],
            decision['context']
        )
        print(f"  {decision['id']}: confidence={quality.confidence:.3f}, expected={decision['expected_outcome']}")
        
        # Update with actual outcome
        core.update_decision_outcome(decision['id'], decision['actual_outcome'])
        quality_score = quality.compute_quality_score()
        print(f"    -> actual={decision['actual_outcome']}, quality_score={quality_score:.3f}")
    
    return core


def demo_bias_detection():
    """Demonstrate cognitive bias detection"""
    print("\n🔍 Cognitive Bias Detection Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Simulate different bias scenarios
    scenarios = [
        {
            'name': 'Confirmation Bias Scenario',
            'context': {
                'confirmation_seeking': True,
                'contradictory_evidence_ignored': True,
                'selective_attention': True
            }
        },
        {
            'name': 'Anchoring Bias Scenario',
            'context': {
                'first_option_preference': True,
                'initial_value_fixation': True,
                'insufficient_adjustment': True
            }
        },
        {
            'name': 'Availability Bias Scenario',
            'context': {
                'recent_example_focus': True,
                'memorable_instance_weight': True,
                'ease_of_recall_bias': True
            }
        },
        {
            'name': 'Overconfidence Bias Scenario',
            'context': {
                'stated_confidence': 0.95,
                'objective_difficulty': 0.4,
                'calibration_history': [(0.9, 0.6), (0.8, 0.5), (0.95, 0.4)]
            }
        }
    ]
    
    print("Bias Detection Results:")
    for scenario in scenarios:
        biases = core.detect_cognitive_biases(scenario['context'])
        print(f"  {scenario['name']}: {len(biases)} biases detected")
        for bias in biases:
            print(f"    - {bias.bias_type.value}: severity={bias.severity:.2f}, confidence={bias.confidence:.2f}")
            print(f"      Mitigation: {bias.mitigation_strategy}")
    
    return core


def demo_performance_optimization():
    """Demonstrate performance tracking and optimization"""
    print("\n📈 Performance Optimization Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Simulate performance metrics for different components
    components = ['reasoning', 'memory', 'attention', 'decision_making']
    
    print("Initial Performance Metrics:")
    for component in components:
        # Generate realistic but suboptimal metrics
        metrics = CognitiveMetrics(
            accuracy=random.uniform(0.5, 0.7),
            precision=random.uniform(0.4, 0.6),
            recall=random.uniform(0.5, 0.7),
            confidence=random.uniform(0.4, 0.6),
            efficiency=random.uniform(0.3, 0.5),
            coherence=random.uniform(0.5, 0.7),
            depth=random.uniform(0.4, 0.6)
        )
        core.track_performance_metrics(component, metrics)
        print(f"  {component}: accuracy={metrics.accuracy:.3f}, confidence={metrics.confidence:.3f}, efficiency={metrics.efficiency:.3f}")
    
    # Generate optimization feedback
    print("\nOptimization Feedback Generation:")
    feedback = core.generate_optimization_feedback()
    print(f"  Generated {len(feedback)} optimization recommendations")
    
    high_priority_feedback = [fb for fb in feedback if fb.priority > 0.7]
    print(f"  High priority items: {len(high_priority_feedback)}")
    
    for fb in high_priority_feedback[:3]:  # Show top 3
        print(f"    - {fb.component}.{fb.metric.value}: {fb.current_value:.3f} -> {fb.target_value:.3f}")
        print(f"      Strategy: {fb.improvement_strategy}")
    
    # Apply optimization feedback
    print("\nApplying Optimization Feedback:")
    results = core.apply_optimization_feedback(high_priority_feedback)
    print(f"  Applied {len(results['applied_optimizations'])} optimizations")
    print(f"  Updated parameters: {list(results['updated_parameters'].keys())}")
    
    # Simulate improved performance
    print("\nImproved Performance Metrics:")
    for component in components:
        # Generate improved metrics
        improved_metrics = CognitiveMetrics(
            accuracy=random.uniform(0.7, 0.9),
            precision=random.uniform(0.7, 0.9),
            recall=random.uniform(0.7, 0.9),
            confidence=random.uniform(0.7, 0.9),
            efficiency=random.uniform(0.6, 0.8),
            coherence=random.uniform(0.7, 0.9),
            depth=random.uniform(0.7, 0.9)
        )
        core.track_performance_metrics(component, improved_metrics)
        print(f"  {component}: accuracy={improved_metrics.accuracy:.3f}, confidence={improved_metrics.confidence:.3f}, efficiency={improved_metrics.efficiency:.3f}")
    
    return core


def demo_confidence_estimation():
    """Demonstrate confidence estimation capabilities"""
    print("\n🎯 Confidence Estimation Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Add some performance history
    metrics = CognitiveMetrics(accuracy=0.8, precision=0.75, recall=0.85)
    core.track_performance_metrics('system', metrics)
    
    # Test confidence estimation in different scenarios
    scenarios = [
        {
            'name': 'High Quality Context',
            'context': {
                'data_quality': 0.9,
                'process_reliability': 0.9,
                'coherence': 0.8,
                'complexity': 0.3,
                'uncertainty': 0.2
            }
        },
        {
            'name': 'Medium Quality Context',
            'context': {
                'data_quality': 0.7,
                'process_reliability': 0.7,
                'coherence': 0.6,
                'complexity': 0.5,
                'uncertainty': 0.5
            }
        },
        {
            'name': 'Low Quality Context',
            'context': {
                'data_quality': 0.4,
                'process_reliability': 0.5,
                'coherence': 0.4,
                'complexity': 0.8,
                'uncertainty': 0.8
            }
        }
    ]
    
    print("Confidence Estimation Results:")
    for scenario in scenarios:
        confidence = core.estimate_confidence(scenario['context'])
        print(f"  {scenario['name']}: confidence={confidence:.3f}")
        
        # Show calibration factors
        complexity_penalty = scenario['context'].get('complexity', 0.3) * 0.1
        uncertainty_penalty = scenario['context'].get('uncertainty', 0.4) * 0.1
        print(f"    Complexity penalty: {complexity_penalty:.3f}, Uncertainty penalty: {uncertainty_penalty:.3f}")
    
    return core


def demo_reflective_learning():
    """Demonstrate reflective learning capabilities"""
    print("\n🌱 Reflective Learning Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Simulate learning experiences
    experiences = [
        {
            'name': 'Successful Problem Solving',
            'experience': {
                'performance_data': {'accuracy': 0.95, 'efficiency': 0.8},
                'patterns': ['systematic_approach', 'hypothesis_testing', 'iterative_refinement'],
                'performance_sequence': [0.6, 0.7, 0.8, 0.9, 0.95],
                'decision_sequence': [
                    {'confidence': 0.7, 'outcome': 0.75},
                    {'confidence': 0.8, 'outcome': 0.85},
                    {'confidence': 0.9, 'outcome': 0.92}
                ],
                'adaptation_success': 0.9
            }
        },
        {
            'name': 'Challenging Learning Task',
            'experience': {
                'performance_data': {'accuracy': 0.6, 'efficiency': 0.4},
                'patterns': ['complexity_management', 'resource_allocation'],
                'performance_sequence': [0.3, 0.4, 0.5, 0.55, 0.6],
                'decision_sequence': [
                    {'confidence': 0.8, 'outcome': 0.4},
                    {'confidence': 0.6, 'outcome': 0.5},
                    {'confidence': 0.5, 'outcome': 0.6}
                ],
                'adaptation_success': 0.7
            }
        }
    ]
    
    print("Reflective Learning Results:")
    for exp in experiences:
        results = core.perform_reflective_learning(exp['experience'])
        print(f"\n  {exp['name']}:")
        print(f"    Insights gained: {len(results['insights_gained'])}")
        for insight in results['insights_gained'][:2]:  # Show first 2
            print(f"      - {insight}")
        
        print(f"    Patterns identified: {len(results['patterns_identified'])}")
        for pattern in results['patterns_identified']:
            print(f"      - {pattern}")
        
        print(f"    Adaptations made: {len(results['adaptations_made'])}")
        for adaptation in results['adaptations_made']:
            print(f"      - {adaptation}")
        
        print(f"    Meta-learning updates: {list(results['meta_learning_updates'].keys())}")
    
    return core


def demo_integrated_system():
    """Demonstrate the integrated self-reflection system"""
    print("\n🔄 Integrated Self-Reflection System Demo")
    print("=" * 50)
    
    core = SelfReflectionCore()
    
    # Simulate a complete cognitive cycle with self-reflection
    print("Cognitive Cycle with Self-Reflection:")
    
    # Step 1: Monitor cognitive state
    context = {'active_processing': True, 'task': 'complex_reasoning'}
    state = core.monitor_cognitive_state(context)
    print(f"  1. Cognitive state: {state.value}")
    
    # Step 2: Perform introspection
    introspection = core.perform_introspection(depth=0.7)
    print(f"  2. Introspection depth: {introspection['depth_achieved']}")
    
    # Step 3: Assess decision quality
    decision_context = {
        'information_quality': 0.8,
        'complexity': 0.6,
        'time_pressure': 0.3,
        'confirmation_seeking': True,
        'stated_confidence': 0.9,
        'objective_difficulty': 0.5
    }
    
    decision = core.assess_decision_quality('integrated_decision', 0.8, decision_context)
    print(f"  3. Decision confidence: {decision.confidence:.3f}")
    
    # Step 4: Detect biases
    biases = core.detect_cognitive_biases(decision_context)
    print(f"  4. Biases detected: {len(biases)} ({[b.bias_type.value for b in biases]})")
    
    # Step 5: Estimate confidence
    confidence = core.estimate_confidence(decision_context)
    print(f"  5. Estimated confidence: {confidence:.3f}")
    
    # Step 6: Track performance
    metrics = CognitiveMetrics(accuracy=0.85, confidence=0.8, efficiency=0.7)
    core.track_performance_metrics('integrated_system', metrics)
    print(f"  6. Performance tracked: accuracy={metrics.accuracy:.3f}")
    
    # Step 7: Generate optimization feedback
    feedback = core.generate_optimization_feedback()
    print(f"  7. Optimization feedback: {len(feedback)} recommendations")
    
    # Step 8: Apply optimizations
    high_priority = [fb for fb in feedback if fb.priority > 0.7]
    if high_priority:
        results = core.apply_optimization_feedback(high_priority)
        print(f"  8. Applied optimizations: {len(results['applied_optimizations'])}")
    
    # Step 9: Reflective learning
    experience = {
        'performance_data': {'accuracy': 0.85, 'efficiency': 0.7},
        'patterns': ['bias_detection', 'confidence_calibration'],
        'performance_sequence': [0.7, 0.75, 0.8, 0.85],
        'adaptation_success': 0.8
    }
    
    learning_results = core.perform_reflective_learning(experience)
    print(f"  9. Reflective learning: {len(learning_results['insights_gained'])} insights")
    
    # Final introspection
    final_introspection = core.perform_introspection(depth=0.9)
    print(f"  10. Final introspection: {len(final_introspection)} total insights")
    
    return core


def main():
    """Main demonstration function"""
    print("🧠 Comprehensive Self-Reflection and Introspection System Demo")
    print("=" * 70)
    print("Demonstrating meta-cognitive awareness, performance monitoring,")
    print("and self-improvement mechanisms for cognitive processes.")
    print("=" * 70)
    
    # Run all demonstrations
    demos = [
        demo_meta_cognitive_monitoring,
        demo_decision_quality_assessment,
        demo_bias_detection,
        demo_performance_optimization,
        demo_confidence_estimation,
        demo_reflective_learning,
        demo_integrated_system
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Error in {demo.__name__}: {e}")
        print()
    
    print("✅ Self-Reflection System Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("  ✓ Meta-cognitive monitoring architecture")
    print("  ✓ Cognitive state introspection functions")
    print("  ✓ Decision quality assessment mechanisms")
    print("  ✓ Performance tracking and analytics")
    print("  ✓ Self-optimization feedback loops")
    print("  ✓ Cognitive bias detection and correction")
    print("  ✓ Reflective learning mechanisms")
    print("  ✓ Confidence estimation for cognitive outputs")
    print("  ✓ Integrated system maintaining efficiency")


if __name__ == '__main__':
    main()