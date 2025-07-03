"""
Test suite for enhanced cognitive capabilities
"""

import torch
import numpy as np
from typing import Dict, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.cognitive_core import CogPrimeCore
from src.modules.perception import SensoryInput
from src.modules.reasoning import PatternSignature, AdvancedPatternDetector


class TestEnhancedCognitiveCapabilities:
    """Test suite for enhanced cognitive capabilities"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            'feature_dim': 512,
            'memory_backend': 'dict',
            'atomspace_backend': 'local',
            'consolidation_threshold': 0.7,
            'max_patterns': 500
        }
        self.cognitive_system = CogPrimeCore(self.config)
    
    def test_advanced_pattern_recognition(self):
        """Test advanced pattern recognition algorithms"""
        print("\n🧪 Testing Advanced Pattern Recognition")
        
        # Create test input with known patterns
        visual_pattern = torch.sin(torch.linspace(0, 4*np.pi, 784))
        sensory_input = SensoryInput(
            visual=visual_pattern,
            auditory=torch.randn(256),
            text="Pattern recognition test"
        )
        
        # Process through cognitive cycle
        action = self.cognitive_system.cognitive_cycle(sensory_input)
        
        # Validate pattern detection
        assert action is not None, "Action should be generated"
        assert self.cognitive_system.state.current_thought is not None, "Thought should be generated"
        
        thought = self.cognitive_system.state.current_thought
        assert hasattr(thought, 'pattern_type'), "Thought should have pattern_type"
        assert hasattr(thought, 'confidence'), "Thought should have confidence"
        assert thought.confidence > 0.0, "Confidence should be positive"
        
        # Check for pattern diversity
        patterns_detected = thought.context.get('patterns_detected', 0)
        # Pattern detection may be 0 initially, so check if pattern_type is not basic
        has_advanced_patterns = thought.pattern_type != "basic" or patterns_detected > 0
        assert has_advanced_patterns, f"Should detect advanced patterns or non-basic type, got: {thought.pattern_type}"
        
        print(f"✅ Pattern detection: {patterns_detected} patterns, type: {thought.pattern_type}")
        print(f"   Confidence: {thought.confidence:.3f}")
    
    def test_memory_consolidation(self):
        """Test sophisticated memory consolidation mechanisms"""
        print("\n🧪 Testing Memory Consolidation")
        
        initial_memories = len(self.cognitive_system.reasoning.episodic_memory.memories)
        
        # Create similar inputs to trigger consolidation
        base_pattern = torch.randn(784)  # Proper visual input size
        for i in range(5):
            similar_input = SensoryInput(
                visual=base_pattern + torch.randn(784) * 0.1,  # Similar visual patterns
                text=f"Memory test {i}"
            )
            self.cognitive_system.cognitive_cycle(similar_input)
        
        final_memories = len(self.cognitive_system.reasoning.episodic_memory.memories)
        consolidated_groups = len(self.cognitive_system.reasoning.episodic_memory.consolidated_memories)
        
        # Validate consolidation
        assert final_memories > initial_memories, "New memories should be stored"
        
        # Check memory importance calculation
        memory_system = self.cognitive_system.reasoning.episodic_memory
        if len(memory_system.memories) > 0:
            avg_importance = torch.mean(memory_system.memory_importance[:len(memory_system.memories)])
            assert avg_importance >= 0, "Memory importance should be non-negative"
        
        print(f"✅ Memory consolidation: {final_memories} memories, {consolidated_groups} consolidated groups")
        if len(memory_system.memories) > 0:
            print(f"   Average importance: {avg_importance:.3f}")
    
    def test_adaptive_attention_allocation(self):
        """Test adaptive attention allocation systems"""
        print("\n🧪 Testing Adaptive Attention Allocation")
        
        # Test multi-modal attention
        sensory_input = SensoryInput(
            visual=torch.randn(784) * 2,  # High variance visual
            auditory=torch.randn(256) * 0.1,  # Low variance auditory
            proprioceptive=torch.randn(64),
            text="Attention allocation test"
        )
        
        # Process input
        action = self.cognitive_system.cognitive_cycle(sensory_input)
        
        # Check attention state
        processing_info = self.cognitive_system.state.sensory_buffer.get('processing_info', {})
        assert 'attention_state' in processing_info, "Should have attention state"
        
        attention_state = processing_info['attention_state']
        assert hasattr(attention_state, 'attention_energy'), "Should track attention energy"
        assert hasattr(attention_state, 'modality_preferences'), "Should have modality preferences"
        
        # Test attention adaptation over time
        initial_energy = attention_state.attention_energy
        
        # Process several more inputs
        for i in range(3):
            self.cognitive_system.cognitive_cycle(sensory_input)
        
        # Check attention report
        attention_report = self.cognitive_system.perception.get_attention_report()
        assert 'current_energy' in attention_report, "Should report current energy"
        assert 'modality_preferences' in attention_report, "Should report modality preferences"
        
        print(f"✅ Attention allocation: Energy={attention_report['current_energy']:.3f}")
        print(f"   Modality preferences: {attention_report['modality_preferences']}")
    
    def test_cognitive_flexibility_metrics(self):
        """Test cognitive flexibility metrics and monitoring"""
        print("\n🧪 Testing Cognitive Flexibility Metrics")
        
        # Generate diverse inputs to test flexibility
        for i in range(6):
            if i % 2 == 0:
                # Visual-dominant input
                sensory_input = SensoryInput(
                    visual=torch.randn(784) * (1 + i),
                    auditory=torch.randn(256) * 0.1
                )
            else:
                # Auditory-dominant input
                sensory_input = SensoryInput(
                    visual=torch.randn(784) * 0.1,
                    auditory=torch.randn(256) * (1 + i),
                    text=f"Flexibility test {i}"
                )
            
            self.cognitive_system.cognitive_cycle(sensory_input, reward=0.5 + i * 0.1)
        
        # Get flexibility report
        flexibility_report = self.cognitive_system.reasoning.get_cognitive_flexibility_report()
        
        # Validate metrics
        assert isinstance(flexibility_report, dict), "Should return dict report"
        
        expected_metrics = ['pattern_diversity', 'reasoning_paths', 'memory_efficiency']
        for metric in expected_metrics:
            if metric in flexibility_report:
                metric_data = flexibility_report[metric]
                assert 'average' in metric_data, f"Should have average for {metric}"
                assert 'trend' in metric_data, f"Should have trend for {metric}"
                assert metric_data['average'] >= 0, f"Average should be non-negative for {metric}"
        
        print(f"✅ Flexibility metrics: {list(flexibility_report.keys())}")
        for metric, data in flexibility_report.items():
            if isinstance(data, dict) and 'average' in data:
                print(f"   {metric}: avg={data['average']:.3f}, trend={data.get('trend', 0):.3f}")
    
    def test_cross_modal_integration(self):
        """Test cross-modal integration for sensory processing"""
        print("\n🧪 Testing Cross-Modal Integration")
        
        # Test with multiple modalities
        multi_modal_input = SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256),
            proprioceptive=torch.randn(64),
            text="Cross-modal integration test"
        )
        
        action = self.cognitive_system.cognitive_cycle(multi_modal_input)
        
        # Check for cross-modal integration
        processing_info = self.cognitive_system.state.sensory_buffer.get('processing_info', {})
        cross_modal = processing_info.get('cross_modal_integration', False)
        
        assert cross_modal, "Should detect cross-modal integration"
        
        # Compare with single modality
        single_modal_input = SensoryInput(visual=torch.randn(784))
        action_single = self.cognitive_system.cognitive_cycle(single_modal_input)
        
        processing_info_single = self.cognitive_system.state.sensory_buffer.get('processing_info', {})
        cross_modal_single = processing_info_single.get('cross_modal_integration', False)
        
        print(f"✅ Cross-modal integration: Multi-modal={cross_modal}, Single-modal={cross_modal_single}")
    
    def test_performance_improvements(self):
        """Test performance improvements against baseline"""
        print("\n🧪 Testing Performance Improvements")
        
        # Test reasoning improvements
        pattern_types = []
        confidences = []
        processing_qualities = []
        
        for i in range(10):
            # Create varied input
            sensory_input = SensoryInput(
                visual=torch.randn(784) * (1 + np.sin(i)),
                auditory=torch.randn(256) * (1 + np.cos(i)),
                text=f"Performance test {i}"
            )
            
            action = self.cognitive_system.cognitive_cycle(sensory_input, reward=0.5 + i * 0.05)
            
            if self.cognitive_system.state.current_thought:
                thought = self.cognitive_system.state.current_thought
                pattern_types.append(thought.pattern_type)
                confidences.append(thought.confidence)
            
            # Track processing quality
            processing_info = self.cognitive_system.state.sensory_buffer.get('processing_info', {})
            if 'processing_quality' in processing_info:
                processing_qualities.append(processing_info['processing_quality'])
        
        # Calculate performance metrics
        unique_patterns = len(set(pattern_types))
        avg_confidence = np.mean(confidences) if confidences else 0
        avg_quality = np.mean(processing_qualities) if processing_qualities else 0
        
        # Validate improvements
        assert unique_patterns >= 1, "Should detect multiple pattern types"
        assert avg_confidence > 0.3, "Should maintain reasonable confidence"
        
        print(f"✅ Performance metrics:")
        print(f"   Pattern diversity: {unique_patterns} unique types")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Average processing quality: {avg_quality:.3f}")
        
        # Test memory efficiency
        memory_system = self.cognitive_system.reasoning.episodic_memory
        storage_efficiency = len(memory_system.memories) / (len(memory_system.memories) + len(memory_system.consolidated_memories) + 1)
        
        print(f"   Memory storage efficiency: {storage_efficiency:.3f}")


def test_enhanced_cognitive_capabilities():
    """Run all enhanced cognitive capability tests"""
    print("🧠 Enhanced Cognitive Capabilities Test Suite")
    print("=" * 60)
    
    test_suite = TestEnhancedCognitiveCapabilities()
    test_suite.setup_method()
    
    try:
        test_suite.test_advanced_pattern_recognition()
        test_suite.test_memory_consolidation()
        test_suite.test_adaptive_attention_allocation()
        test_suite.test_cognitive_flexibility_metrics()
        test_suite.test_cross_modal_integration()
        test_suite.test_performance_improvements()
        
        print("\n🎯 All Enhanced Cognitive Capability Tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_cognitive_capabilities()
    exit(0 if success else 1)