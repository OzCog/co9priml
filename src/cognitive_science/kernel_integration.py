"""
Integration adapter for connecting the enhanced self-reflection system
with the unified cognitive kernel.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Any, Optional, List
import time
from dataclasses import dataclass

from cognitive_science.self_reflection import (
    SelfReflectionCore, CognitiveState, CognitiveMetrics, 
    DecisionQuality, BiasType, ReflectionMode
)


@dataclass
class CognitiveKernelMetrics:
    """Metrics for cognitive kernel components"""
    tensor_kernel_efficiency: float = 0.0
    attention_effectiveness: float = 0.0
    grammar_accuracy: float = 0.0
    interface_quality: float = 0.0
    overall_coherence: float = 0.0


class SelfReflectionAdapter:
    """Adapter for integrating self-reflection with cognitive kernel"""
    
    def __init__(self):
        self.reflection_core = SelfReflectionCore()
        self.kernel_metrics_history: List[CognitiveKernelMetrics] = []
        
    def monitor_cognitive_kernel_cycle(self, 
                                     input_data: Dict[str, Any],
                                     kernel_response: Dict[str, Any],
                                     processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor a complete cognitive kernel cycle with self-reflection"""
        
        # Step 1: Monitor cognitive state
        cognitive_context = self._extract_cognitive_context(input_data, processing_context)
        cognitive_state = self.reflection_core.monitor_cognitive_state(cognitive_context)
        
        # Step 2: Assess decision quality if applicable
        decision_quality = None
        if 'decision_made' in processing_context:
            decision_quality = self.reflection_core.assess_decision_quality(
                processing_context.get('decision_id', f'decision_{int(time.time())}'),
                processing_context.get('expected_outcome', 0.7),
                processing_context
            )
        
        # Step 3: Detect cognitive biases
        biases = self.reflection_core.detect_cognitive_biases(processing_context)
        
        # Step 4: Estimate confidence in kernel output
        confidence = self.reflection_core.estimate_confidence({
            'data_quality': processing_context.get('input_quality', 0.7),
            'process_reliability': processing_context.get('process_reliability', 0.8),
            'coherence': processing_context.get('output_coherence', 0.7),
            'complexity': processing_context.get('task_complexity', 0.5),
            'uncertainty': processing_context.get('uncertainty_level', 0.4)
        })
        
        # Step 5: Track performance metrics
        kernel_metrics = self._extract_kernel_metrics(kernel_response, processing_context)
        self._track_kernel_performance(kernel_metrics)
        
        # Step 6: Perform introspection
        introspection_depth = self._determine_introspection_depth(cognitive_state, biases)
        introspection_report = self.reflection_core.perform_introspection(introspection_depth)
        
        # Step 7: Generate optimization feedback
        optimization_feedback = self.reflection_core.generate_optimization_feedback()
        
        # Step 8: Apply high-priority optimizations
        high_priority_feedback = [fb for fb in optimization_feedback if fb.priority > 0.8]
        optimization_results = {}
        if high_priority_feedback:
            optimization_results = self.reflection_core.apply_optimization_feedback(high_priority_feedback)
        
        # Step 9: Reflective learning from the cycle
        cycle_experience = self._create_cycle_experience(
            input_data, kernel_response, processing_context, 
            cognitive_state, decision_quality, biases, confidence
        )
        learning_results = self.reflection_core.perform_reflective_learning(cycle_experience)
        
        # Compile comprehensive reflection report
        reflection_report = {
            'cognitive_state': cognitive_state.value,
            'decision_quality': {
                'confidence': decision_quality.confidence if decision_quality else None,
                'quality_score': decision_quality.compute_quality_score() if decision_quality else None
            },
            'biases_detected': [{'type': b.bias_type.value, 'severity': b.severity} for b in biases],
            'confidence_estimate': confidence,
            'introspection': {
                'depth': introspection_depth,
                'insights_count': len(introspection_report),
                'key_patterns': list(introspection_report.get('performance_trends', {}).keys())[:3]
            },
            'optimization': {
                'feedback_count': len(optimization_feedback),
                'high_priority_count': len(high_priority_feedback),
                'applied_optimizations': len(optimization_results.get('applied_optimizations', []))
            },
            'learning': {
                'insights_gained': len(learning_results['insights_gained']),
                'patterns_identified': len(learning_results['patterns_identified']),
                'adaptations_made': len(learning_results['adaptations_made'])
            },
            'kernel_metrics': {
                'tensor_efficiency': kernel_metrics.tensor_kernel_efficiency,
                'attention_effectiveness': kernel_metrics.attention_effectiveness,
                'grammar_accuracy': kernel_metrics.grammar_accuracy,
                'interface_quality': kernel_metrics.interface_quality,
                'overall_coherence': kernel_metrics.overall_coherence
            }
        }
        
        return reflection_report
    
    def _extract_cognitive_context(self, input_data: Dict[str, Any], 
                                 processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cognitive context from input and processing data"""
        return {
            'active_processing': processing_context.get('active_processing', True),
            'learning_active': processing_context.get('learning_mode', False),
            'reflection_triggered': processing_context.get('reflection_needed', False),
            'optimization_needed': processing_context.get('performance_degradation', False),
            'task_complexity': processing_context.get('task_complexity', 0.5),
            'cognitive_load': processing_context.get('cognitive_load', 0.5)
        }
    
    def _extract_kernel_metrics(self, kernel_response: Dict[str, Any],
                               processing_context: Dict[str, Any]) -> CognitiveKernelMetrics:
        """Extract performance metrics from kernel response"""
        return CognitiveKernelMetrics(
            tensor_kernel_efficiency=processing_context.get('tensor_efficiency', 0.7),
            attention_effectiveness=processing_context.get('attention_effectiveness', 0.7),
            grammar_accuracy=processing_context.get('grammar_accuracy', 0.8),
            interface_quality=processing_context.get('interface_quality', 0.7),
            overall_coherence=processing_context.get('coherence_score', 0.75)
        )
    
    def _track_kernel_performance(self, kernel_metrics: CognitiveKernelMetrics):
        """Track kernel performance metrics"""
        self.kernel_metrics_history.append(kernel_metrics)
        
        # Convert to CognitiveMetrics for reflection system
        cognitive_metrics = CognitiveMetrics(
            accuracy=kernel_metrics.grammar_accuracy,
            precision=kernel_metrics.interface_quality,
            recall=kernel_metrics.attention_effectiveness,
            confidence=kernel_metrics.overall_coherence,
            efficiency=kernel_metrics.tensor_kernel_efficiency,
            coherence=kernel_metrics.overall_coherence
        )
        
        self.reflection_core.track_performance_metrics('cognitive_kernel', cognitive_metrics)
        
        # Keep only recent history
        if len(self.kernel_metrics_history) > 100:
            self.kernel_metrics_history = self.kernel_metrics_history[-100:]
    
    def _determine_introspection_depth(self, cognitive_state: CognitiveState,
                                     biases: List[Any]) -> float:
        """Determine appropriate introspection depth based on context"""
        base_depth = 0.5
        
        # Increase depth for certain states
        if cognitive_state == CognitiveState.REFLECTING:
            base_depth += 0.3
        elif cognitive_state == CognitiveState.LEARNING:
            base_depth += 0.2
        elif cognitive_state == CognitiveState.OPTIMIZING:
            base_depth += 0.25
        
        # Increase depth if biases detected
        if biases:
            base_depth += len(biases) * 0.1
        
        return min(1.0, base_depth)
    
    def _create_cycle_experience(self, input_data: Dict[str, Any],
                               kernel_response: Dict[str, Any],
                               processing_context: Dict[str, Any],
                               cognitive_state: CognitiveState,
                               decision_quality: Optional[DecisionQuality],
                               biases: List[Any],
                               confidence: float) -> Dict[str, Any]:
        """Create experience data for reflective learning"""
        
        # Extract performance sequence from recent metrics
        recent_metrics = self.kernel_metrics_history[-5:] if self.kernel_metrics_history else []
        performance_sequence = [m.overall_coherence for m in recent_metrics]
        
        # Create decision sequence if decision quality available
        decision_sequence = []
        if decision_quality:
            decision_sequence.append({
                'confidence': decision_quality.confidence,
                'outcome': decision_quality.actual_outcome or decision_quality.expected_outcome
            })
        
        return {
            'performance_data': {
                'accuracy': processing_context.get('grammar_accuracy', 0.8),
                'efficiency': processing_context.get('tensor_efficiency', 0.7),
                'coherence': processing_context.get('coherence_score', 0.75)
            },
            'patterns': [
                f'cognitive_state_{cognitive_state.value}',
                f'confidence_level_{confidence:.1f}',
                f'bias_count_{len(biases)}'
            ],
            'performance_sequence': performance_sequence,
            'decision_sequence': decision_sequence,
            'adaptation_success': processing_context.get('adaptation_success', 0.7)
        }
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of reflection system status"""
        return {
            'cognitive_state': self.reflection_core.cognitive_state.value,
            'self_awareness_level': self.reflection_core.self_awareness_level,
            'meta_cognitive_capacity': self.reflection_core.meta_cognitive_capacity,
            'introspection_depth': self.reflection_core.introspection_depth,
            'total_decisions': len(self.reflection_core.decision_history),
            'total_biases_detected': len(self.reflection_core.bias_detections),
            'optimization_feedback_pending': len(self.reflection_core.optimization_feedback),
            'performance_components_tracked': len(self.reflection_core.performance_metrics),
            'kernel_cycles_monitored': len(self.kernel_metrics_history)
        }
    
    def apply_kernel_optimizations(self) -> Dict[str, Any]:
        """Apply optimizations specifically for kernel performance"""
        feedback = self.reflection_core.generate_optimization_feedback()
        kernel_feedback = [fb for fb in feedback if 'kernel' in fb.component]
        
        if kernel_feedback:
            return self.reflection_core.apply_optimization_feedback(kernel_feedback)
        else:
            return {'applied_optimizations': [], 'updated_parameters': {}}


def create_enhanced_cognitive_kernel_with_reflection():
    """Create an enhanced cognitive kernel with integrated self-reflection"""
    
    class EnhancedCognitiveKernel:
        """Cognitive kernel with integrated self-reflection capabilities"""
        
        def __init__(self):
            self.reflection_adapter = SelfReflectionAdapter()
            # Placeholder for actual kernel components
            self.kernel_initialized = True
        
        async def cognitive_cycle_with_reflection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Execute cognitive cycle with comprehensive self-reflection"""
            
            # Simulate kernel processing (would be actual kernel in real implementation)
            processing_context = {
                'active_processing': True,
                'task_complexity': min(1.0, len(str(input_data)) / 1000),
                'input_quality': 0.8,
                'process_reliability': 0.85,
                'tensor_efficiency': 0.75,
                'attention_effectiveness': 0.8,
                'grammar_accuracy': 0.85,
                'interface_quality': 0.8,
                'coherence_score': 0.82,
                'uncertainty_level': 0.3
            }
            
            # Simulate kernel response
            kernel_response = {
                'processed_output': f"Processed: {input_data}",
                'confidence': 0.8,
                'coherence': processing_context['coherence_score']
            }
            
            # Apply self-reflection monitoring
            reflection_report = self.reflection_adapter.monitor_cognitive_kernel_cycle(
                input_data, kernel_response, processing_context
            )
            
            # Combine kernel response with reflection insights
            enhanced_response = {
                'kernel_output': kernel_response,
                'reflection_insights': reflection_report,
                'meta_cognitive_summary': self.reflection_adapter.get_reflection_summary()
            }
            
            return enhanced_response
        
        def get_system_status(self) -> Dict[str, Any]:
            """Get comprehensive system status including reflection capabilities"""
            return {
                'kernel_status': 'active' if self.kernel_initialized else 'inactive',
                'reflection_status': self.reflection_adapter.get_reflection_summary(),
                'integration_health': 'optimal'
            }
    
    return EnhancedCognitiveKernel()


# Demo integration
async def demo_kernel_integration():
    """Demonstrate cognitive kernel integration with self-reflection"""
    print("🔗 Cognitive Kernel Integration Demo")
    print("=" * 50)
    
    # Create enhanced kernel
    enhanced_kernel = create_enhanced_cognitive_kernel_with_reflection()
    
    # Test inputs
    test_inputs = [
        {'query': 'What is the relationship between consciousness and cognition?', 'complexity': 'high'},
        {'task': 'solve_math_problem', 'problem': '2x + 5 = 15', 'complexity': 'medium'},
        {'request': 'analyze_pattern', 'data': [1, 2, 4, 8, 16], 'complexity': 'low'}
    ]
    
    print("Processing inputs with enhanced cognitive kernel:")
    
    for i, input_data in enumerate(test_inputs, 1):
        print(f"\n  Input {i}: {input_data['complexity']} complexity task")
        
        # Process with reflection
        response = await enhanced_kernel.cognitive_cycle_with_reflection(input_data)
        
        # Display key reflection insights
        reflection = response['reflection_insights']
        print(f"    Cognitive state: {reflection['cognitive_state']}")
        print(f"    Confidence estimate: {reflection['confidence_estimate']:.3f}")
        print(f"    Biases detected: {len(reflection['biases_detected'])}")
        print(f"    Introspection depth: {reflection['introspection']['depth']:.2f}")
        print(f"    Optimizations applied: {reflection['optimization']['applied_optimizations']}")
        print(f"    Learning insights: {reflection['learning']['insights_gained']}")
    
    # Show system status
    print(f"\n  Final System Status:")
    status = enhanced_kernel.get_system_status()
    reflection_status = status['reflection_status']
    print(f"    Self-awareness level: {reflection_status['self_awareness_level']:.3f}")
    print(f"    Meta-cognitive capacity: {reflection_status['meta_cognitive_capacity']:.3f}")
    print(f"    Total cycles monitored: {reflection_status['kernel_cycles_monitored']}")
    print(f"    Integration health: {status['integration_health']}")
    
    return enhanced_kernel


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_kernel_integration())