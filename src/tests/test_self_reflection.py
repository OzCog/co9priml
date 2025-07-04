"""
Test suite for enhanced self-reflection mechanisms.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import unittest
from unittest.mock import Mock, patch

from cognitive_science.self_reflection import (
    SelfReflectionCore, ReflectionMode, CognitiveState, BiasType, MetricType,
    CognitiveMetrics, DecisionQuality, BiasDetection, OptimizationFeedback
)
from cognitive_science.math_utils import clip, mean, entropy


class TestEnhancedSelfReflection(unittest.TestCase):
    """Test suite for enhanced self-reflection capabilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.core = SelfReflectionCore()
    
    def test_cognitive_state_monitoring(self):
        """Test cognitive state monitoring"""
        # Test initial state
        self.assertEqual(self.core.cognitive_state, CognitiveState.IDLE)
        
        # Test active processing state
        context = {'active_processing': True}
        new_state = self.core.monitor_cognitive_state(context)
        self.assertEqual(new_state, CognitiveState.ACTIVE)
        
        # Test learning state
        context = {'learning_active': True}
        new_state = self.core.monitor_cognitive_state(context)
        self.assertEqual(new_state, CognitiveState.LEARNING)
        
        # Test reflection state
        context = {'reflection_triggered': True}
        new_state = self.core.monitor_cognitive_state(context)
        self.assertEqual(new_state, CognitiveState.REFLECTING)
        
        # Test optimization state
        context = {'optimization_needed': True}
        new_state = self.core.monitor_cognitive_state(context)
        self.assertEqual(new_state, CognitiveState.OPTIMIZING)
    
    def test_introspection_capabilities(self):
        """Test introspection capabilities"""
        # Test basic introspection
        report = self.core.perform_introspection(depth=0.3)
        self.assertIsInstance(report, dict)
        self.assertIn('cognitive_state', report)
        self.assertIn('self_awareness_level', report)
        self.assertIn('meta_cognitive_capacity', report)
        
        # Test deep introspection
        deep_report = self.core.perform_introspection(depth=0.8)
        self.assertIn('performance_trends', deep_report)
        self.assertIn('efficiency_patterns', deep_report)
        
        # Test very deep introspection
        very_deep_report = self.core.perform_introspection(depth=0.9)
        self.assertIn('state_transition_patterns', very_deep_report)
        self.assertIn('decision_quality_trends', very_deep_report)
    
    def test_decision_quality_assessment(self):
        """Test decision quality assessment"""
        context = {
            'information_quality': 0.8,
            'complexity': 0.3,
            'time_pressure': 0.2,
            'uncertainty': 0.4
        }
        
        decision = self.core.assess_decision_quality('test_decision', 0.7, context)
        self.assertIsInstance(decision, DecisionQuality)
        self.assertEqual(decision.decision_id, 'test_decision')
        self.assertEqual(decision.expected_outcome, 0.7)
        self.assertGreater(decision.confidence, 0.0)
        self.assertLess(decision.confidence, 1.0)
        
        # Test decision outcome update
        success = self.core.update_decision_outcome('test_decision', 0.8)
        self.assertTrue(success)
        self.assertEqual(decision.actual_outcome, 0.8)
        
        # Test quality score computation
        quality_score = decision.compute_quality_score()
        self.assertGreater(quality_score, 0.0)
        self.assertLess(quality_score, 1.0)
    
    def test_bias_detection(self):
        """Test cognitive bias detection"""
        # Test confirmation bias
        context = {
            'confirmation_seeking': True,
            'contradictory_evidence_ignored': True
        }
        biases = self.core.detect_cognitive_biases(context)
        bias_types = [b.bias_type for b in biases]
        self.assertIn(BiasType.CONFIRMATION, bias_types)
        
        # Test anchoring bias
        context = {
            'first_option_preference': True,
            'initial_value_fixation': True
        }
        biases = self.core.detect_cognitive_biases(context)
        bias_types = [b.bias_type for b in biases]
        self.assertIn(BiasType.ANCHORING, bias_types)
        
        # Test availability bias
        context = {
            'recent_example_focus': True,
            'memorable_instance_weight': True
        }
        biases = self.core.detect_cognitive_biases(context)
        bias_types = [b.bias_type for b in biases]
        self.assertIn(BiasType.AVAILABILITY, bias_types)
        
        # Test overconfidence bias
        context = {
            'stated_confidence': 0.9,
            'objective_difficulty': 0.4
        }
        biases = self.core.detect_cognitive_biases(context)
        bias_types = [b.bias_type for b in biases]
        self.assertIn(BiasType.OVERCONFIDENCE, bias_types)
    
    def test_optimization_feedback(self):
        """Test optimization feedback generation"""
        # Add some performance metrics
        metrics = CognitiveMetrics(
            accuracy=0.6,  # Low accuracy
            confidence=0.5,  # Low confidence
            efficiency=0.4   # Low efficiency
        )
        self.core.track_performance_metrics('test_component', metrics)
        
        # Generate optimization feedback
        feedback = self.core.generate_optimization_feedback()
        self.assertIsInstance(feedback, list)
        self.assertGreater(len(feedback), 0)
        
        # Check feedback structure
        for fb in feedback:
            self.assertIsInstance(fb, OptimizationFeedback)
            self.assertIn(fb.metric, [MetricType.ACCURACY, MetricType.CONFIDENCE, MetricType.EFFICIENCY])
            self.assertGreater(fb.priority, 0.0)
            self.assertLess(fb.priority, 1.0)
    
    def test_feedback_application(self):
        """Test optimization feedback application"""
        feedback = [
            OptimizationFeedback(
                component='test_component',
                metric=MetricType.CONFIDENCE,
                current_value=0.5,
                target_value=0.8,
                improvement_strategy='Calibrate confidence',
                priority=0.8
            ),
            OptimizationFeedback(
                component='test_component',
                metric=MetricType.ACCURACY,
                current_value=0.6,
                target_value=0.8,
                improvement_strategy='Improve training',
                priority=0.9
            )
        ]
        
        initial_confidence_threshold = self.core.confidence_threshold
        initial_learning_rate = self.core.learning_rate
        
        results = self.core.apply_optimization_feedback(feedback)
        
        self.assertIsInstance(results, dict)
        self.assertIn('applied_optimizations', results)
        self.assertIn('updated_parameters', results)
        
        # Check parameters were updated
        self.assertNotEqual(self.core.confidence_threshold, initial_confidence_threshold)
        self.assertNotEqual(self.core.learning_rate, initial_learning_rate)
    
    def test_confidence_estimation(self):
        """Test confidence estimation"""
        # Test basic confidence estimation
        context = {
            'data_quality': 0.8,
            'process_reliability': 0.9,
            'coherence': 0.7
        }
        confidence = self.core.estimate_confidence(context)
        self.assertGreater(confidence, 0.0)
        self.assertLess(confidence, 1.0)
        
        # Test with performance metrics
        metrics = CognitiveMetrics(accuracy=0.85, confidence=0.8)
        self.core.track_performance_metrics('test_component', metrics)
        
        confidence_with_history = self.core.estimate_confidence(context)
        self.assertGreater(confidence_with_history, 0.0)
        self.assertLess(confidence_with_history, 1.0)
    
    def test_reflective_learning(self):
        """Test reflective learning capabilities"""
        experience = {
            'performance_data': {'accuracy': 0.95, 'efficiency': 0.8},
            'patterns': ['pattern1', 'pattern2', 'pattern3'],
            'performance_sequence': [0.6, 0.7, 0.8, 0.9],
            'decision_sequence': [
                {'confidence': 0.8, 'outcome': 0.85},
                {'confidence': 0.9, 'outcome': 0.88}
            ],
            'adaptation_success': 0.8
        }
        
        results = self.core.perform_reflective_learning(experience)
        
        self.assertIsInstance(results, dict)
        self.assertIn('insights_gained', results)
        self.assertIn('patterns_identified', results)
        self.assertIn('adaptations_made', results)
        self.assertIn('meta_learning_updates', results)
        
        # Check insights were generated
        self.assertGreater(len(results['insights_gained']), 0)
        
        # Check patterns were identified
        self.assertGreater(len(results['patterns_identified']), 0)
    
    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        # Create test metrics
        metrics1 = CognitiveMetrics(accuracy=0.8, precision=0.75, recall=0.85)
        metrics2 = CognitiveMetrics(accuracy=0.82, precision=0.78, recall=0.87)
        
        # Track metrics
        self.core.track_performance_metrics('component1', metrics1)
        self.core.track_performance_metrics('component1', metrics2)
        
        # Check metrics were stored
        self.assertIn('component1', self.core.performance_metrics)
        self.assertEqual(self.core.performance_metrics['component1'], metrics2)
        
        # Check metrics history
        self.assertIn('component1_accuracy', self.core.metrics_history)
        self.assertEqual(len(self.core.metrics_history['component1_accuracy']), 2)
    
    def test_meta_cognitive_monitoring(self):
        """Test meta-cognitive monitoring integration"""
        # Create some activity
        context1 = {'active_processing': True}
        self.core.monitor_cognitive_state(context1)
        
        context2 = {'learning_active': True}
        self.core.monitor_cognitive_state(context2)
        
        # Check state transitions were recorded
        self.assertGreater(len(self.core.state_transitions), 0)
        
        # Perform deep introspection
        deep_report = self.core.perform_introspection(depth=0.9)
        self.assertIn('state_transition_patterns', deep_report)
    
    def test_self_optimization_loop(self):
        """Test complete self-optimization loop"""
        # Step 1: Generate some low performance
        poor_metrics = CognitiveMetrics(accuracy=0.5, confidence=0.4, efficiency=0.3)
        self.core.track_performance_metrics('test_system', poor_metrics)
        
        # Step 2: Generate optimization feedback
        feedback = self.core.generate_optimization_feedback()
        self.assertGreater(len(feedback), 0)
        
        # Step 3: Apply feedback
        results = self.core.apply_optimization_feedback(feedback)
        self.assertIn('applied_optimizations', results)
        
        # Step 4: Simulate improved performance
        improved_metrics = CognitiveMetrics(accuracy=0.8, confidence=0.75, efficiency=0.7)
        self.core.track_performance_metrics('test_system', improved_metrics)
        
        # Step 5: Verify improvement
        new_feedback = self.core.generate_optimization_feedback()
        # Should have less feedback for the same component
        old_feedback_count = len([f for f in feedback if f.component == 'test_system'])
        new_feedback_count = len([f for f in new_feedback if f.component == 'test_system'])
        self.assertLessEqual(new_feedback_count, old_feedback_count)
    
    def test_integration_with_existing_reflection(self):
        """Test integration with existing reflection capabilities"""
        # Test that original reflection methods still work
        content = {
            'focus': 'learning',
            'context': 'educational',
            'patterns': ['pattern1', 'pattern2']
        }
        
        reflective_state = self.core.engage_reflection(content, ReflectionMode.NARRATIVE)
        self.assertIsNotNone(reflective_state)
        self.assertEqual(reflective_state.mode, ReflectionMode.NARRATIVE)
        
        # Test meaning construction
        from cognitive_science.self_reflection import MeaningDimension
        meaning = self.core.construct_meaning(
            [MeaningDimension.PURPOSE, MeaningDimension.COHERENCE],
            {'core': 'test', 'relations': {}, 'dynamics': {}}
        )
        self.assertIsNotNone(meaning)


def run_comprehensive_test():
    """Run comprehensive test of self-reflection system"""
    print("🧠 Running Comprehensive Self-Reflection Test...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test methods
    test_class = TestEnhancedSelfReflection
    for method_name in dir(test_class):
        if method_name.startswith('test_'):
            suite.addTest(test_class(method_name))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print("\n🔥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed'}")
    
    return success


if __name__ == '__main__':
    success = run_comprehensive_test()
    exit(0 if success else 1)