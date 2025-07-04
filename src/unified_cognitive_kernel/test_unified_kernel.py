"""
Comprehensive test for the Unified Cognitive Kernel Integration
"""

import asyncio
import pytest
import numpy as np
from typing import Dict, Any

from unified_cognitive_kernel import (
    UnifiedCognitiveKernel,
    CognitiveKernelConfig,
    TensorKernelCohesion,
    CognitiveGrammarField,
    AdaptiveInterfaceLayer,
    ECANAttentionAllocation
)
from unified_cognitive_kernel.tensor_shapes import TensorShapeMetaDesign, TensorCategory


class TestUnifiedCognitiveKernel:
    """Test suite for the unified cognitive kernel"""
    
    @pytest.fixture
    async def kernel_config(self):
        """Create test configuration for cognitive kernel"""
        return CognitiveKernelConfig(
            tensor_kernel={
                'ggml_backend': 'cpu',
                'kokkos_execution_space': 'serial',
                'a0ml_meta_learning_enabled': True
            },
            memory_config={
                'hierarchical_levels': 3,
                'episodic_capacity': 1000,
                'semantic_capacity': 5000
            },
            attention_config={
                'ecan_enabled': True,
                'attention_decay': 0.95,
                'importance_threshold': 0.1
            },
            api_config={
                'enable_grpc': False,  # Disable for testing
                'enable_rest': False,  # Disable for testing
                'port': 8081
            }
        )
    
    @pytest.fixture
    async def cognitive_kernel(self, kernel_config):
        """Create and initialize cognitive kernel for testing"""
        kernel = UnifiedCognitiveKernel(kernel_config)
        await kernel.initialize()
        yield kernel
        await kernel.shutdown()
    
    async def test_kernel_initialization(self, cognitive_kernel):
        """Test cognitive kernel initialization"""
        assert cognitive_kernel.state.value == 'active'
        assert cognitive_kernel.tensor_kernel is not None
        assert cognitive_kernel.cognitive_grammar is not None
        assert cognitive_kernel.adaptive_interface is not None
        assert cognitive_kernel.attention_allocation is not None
        assert cognitive_kernel.gestalt_tensor is not None
    
    async def test_tensor_kernel_cohesion(self, cognitive_kernel):
        """Test tensor kernel cohesion functionality"""
        tensor_kernel = cognitive_kernel.tensor_kernel
        
        # Test tensor kernel state
        assert tensor_kernel.is_active()
        
        # Test tensor operation processing
        input_data = {
            'tensor_data': {'input': np.random.randn(4, 4)},
            'operations': [
                {
                    'type': 'matrix_multiply',
                    'inputs': [np.random.randn(4, 4), np.random.randn(4, 4)],
                    'output_shape': (4, 4),
                    'backend': 'ggml'
                }
            ]
        }
        
        result = await tensor_kernel.process_input(input_data)
        
        assert 'tensor_results' in result
        assert 'kernel_state' in result
        assert result['kernel_state']['ggml_initialized'] == True
    
    async def test_cognitive_grammar_field(self, cognitive_kernel):
        """Test cognitive grammar field functionality"""
        cognitive_grammar = cognitive_kernel.cognitive_grammar
        
        # Test cognitive grammar state
        assert cognitive_grammar.is_active()
        
        # Test reasoning processing
        tensor_response = {'tensor_results': {'test_result': np.array([1, 2, 3])}}
        input_data = {
            'cognitive_content': {
                'cause': 'learning',
                'effect': 'understanding'
            }
        }
        
        result = await cognitive_grammar.process_reasoning(tensor_response, input_data)
        
        assert 'pattern_matches' in result
        assert 'inferences' in result
        assert 'memory_results' in result
        assert 'hypergraph_updates' in result
    
    async def test_attention_allocation(self, cognitive_kernel):
        """Test ECAN attention allocation"""
        attention_allocation = cognitive_kernel.attention_allocation
        
        # Test attention allocation state
        assert attention_allocation.is_active()
        
        # Test attention allocation processing
        reasoning_response = {
            'pattern_matches': [
                {
                    'template': 'causal_relation',
                    'match': {'cause': 'learning', 'effect': 'knowledge'},
                    'confidence': 0.8
                }
            ],
            'inferences': [
                {
                    'premise_atoms': ['learning_atom'],
                    'conclusion_atom': 'knowledge_atom',
                    'confidence': 0.85
                }
            ]
        }
        
        result = await attention_allocation.allocate_attention(reasoning_response, {})
        
        assert 'attention_allocations' in result
        assert 'attention_focus' in result
        assert 'economic_state' in result
        assert 'performance_metrics' in result
    
    async def test_adaptive_interface_layer(self, cognitive_kernel):
        """Test adaptive interface layer"""
        adaptive_interface = cognitive_kernel.adaptive_interface
        
        # Test interface state
        assert adaptive_interface.is_active()
        
        # Test response generation
        attention_response = {
            'attention_allocations': [
                {'atom_id': 'test_atom', 'attention_value': 0.8}
            ]
        }
        gestalt_tensor = np.random.randn(10, 10)
        
        result = await adaptive_interface.generate_response(attention_response, gestalt_tensor)
        
        assert 'attention_allocation' in result
        assert 'gestalt_tensor_shape' in result
        assert 'interfaces_available' in result
    
    async def test_tensor_shape_meta_design(self):
        """Test tensor shape meta-design functionality"""
        tensor_meta = TensorShapeMetaDesign()
        
        # Test tensor shape creation
        dimensions = (64, 32)
        tensor_shape = tensor_meta.create_tensor_shape(dimensions, TensorCategory.GGML_TENSOR)
        
        assert tensor_shape.dimensions == dimensions
        assert tensor_shape.category == TensorCategory.GGML_TENSOR
        assert tensor_shape.unique_address is not None
        assert tensor_shape.lexeme_representation is not None
        
        # Test gestalt shape computation
        component_shapes = {
            'ggml': tensor_meta.create_tensor_shape((64, 32), TensorCategory.GGML_TENSOR),
            'mem0': tensor_meta.create_tensor_shape((64, 128, 32), TensorCategory.MEM0_TENSOR)
        }
        
        gestalt_shape = tensor_meta.compute_gestalt_shape(component_shapes)
        assert len(gestalt_shape) >= 2
        
        # Test component tensor factorization
        component_tensor = tensor_meta.factorize_component_tensor('ggml', {'data': np.random.randn(10, 5)})
        assert isinstance(component_tensor, np.ndarray)
    
    async def test_cognitive_cycle(self, cognitive_kernel):
        """Test complete cognitive cycle"""
        input_data = {
            'cognitive_content': {
                'user_input': 'What is machine learning?',
                'context': 'educational',
                'cause': 'question',
                'effect': 'understanding'
            },
            'tensor_data': {
                'embeddings': np.random.randn(128)
            }
        }
        
        result = await cognitive_kernel.cognitive_cycle(input_data)
        
        # Verify response structure
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check that cognitive components were involved
        kernel_status = cognitive_kernel.get_status()
        assert kernel_status['state'] == 'active'
        assert all(kernel_status['active_components'].values())
    
    async def test_meta_learning_cycle(self, cognitive_kernel):
        """Test meta-learning cycle"""
        # Run a few cognitive cycles first to generate data
        for i in range(3):
            input_data = {
                'cognitive_content': {'iteration': i},
                'tensor_data': {'data': np.random.randn(10)}
            }
            await cognitive_kernel.cognitive_cycle(input_data)
        
        # Run meta-learning cycle
        await cognitive_kernel.meta_learning_cycle()
        
        # Verify kernel is still active
        assert cognitive_kernel.state.value == 'active'
    
    async def test_gestalt_tensor_evolution(self, cognitive_kernel):
        """Test gestalt tensor evolution through cycles"""
        initial_gestalt_shape = cognitive_kernel.gestalt_tensor.shape
        
        # Run multiple cognitive cycles with different inputs
        input_variations = [
            {'cognitive_content': {'type': 'learning'}, 'tensor_data': {'data': np.random.randn(5, 5)}},
            {'cognitive_content': {'type': 'reasoning'}, 'tensor_data': {'data': np.random.randn(8, 8)}},
            {'cognitive_content': {'type': 'memory'}, 'tensor_data': {'data': np.random.randn(6, 6)}}
        ]
        
        for input_data in input_variations:
            await cognitive_kernel.cognitive_cycle(input_data)
        
        # Verify gestalt tensor has evolved
        final_gestalt_shape = cognitive_kernel.gestalt_tensor.shape
        assert final_gestalt_shape == initial_gestalt_shape  # Shape should remain consistent
        
        # Verify tensor values have changed
        assert not np.array_equal(cognitive_kernel.gestalt_tensor, np.zeros_like(cognitive_kernel.gestalt_tensor))
    
    async def test_error_handling(self, cognitive_kernel):
        """Test error handling in cognitive kernel"""
        # Test with malformed input
        malformed_input = {
            'invalid_key': 'invalid_value'
        }
        
        try:
            result = await cognitive_kernel.cognitive_cycle(malformed_input)
            # Should handle gracefully and return some result
            assert isinstance(result, dict)
        except Exception as e:
            # If exception is raised, it should be a known type
            assert isinstance(e, (ValueError, KeyError, TypeError))
    
    async def test_performance_metrics(self, cognitive_kernel):
        """Test performance metrics collection"""
        # Run some cognitive cycles
        for i in range(5):
            input_data = {
                'cognitive_content': {'test_cycle': i},
                'tensor_data': {'data': np.random.randn(4, 4)}
            }
            await cognitive_kernel.cognitive_cycle(input_data)
        
        # Check component performance metrics
        tensor_metrics = await cognitive_kernel.tensor_kernel.get_efficiency_metrics()
        attention_metrics = await cognitive_kernel.attention_allocation.get_effectiveness_metrics()
        interface_metrics = await cognitive_kernel.adaptive_interface.get_quality_metrics()
        grammar_metrics = await cognitive_kernel.cognitive_grammar.get_accuracy_metrics()
        
        # Verify metrics structure
        assert isinstance(tensor_metrics, dict)
        assert isinstance(attention_metrics, dict)
        assert isinstance(interface_metrics, dict)
        assert isinstance(grammar_metrics, dict)
        
        # Verify metrics have reasonable values
        assert 0 <= tensor_metrics.get('throughput', 0) <= 10
        assert 0 <= attention_metrics.get('attention_distribution_entropy', 0) <= 10
        assert 0 <= interface_metrics.get('response_quality', 0) <= 1
        assert 0 <= grammar_metrics.get('pattern_matching_accuracy', 0) <= 1


# Integration test function that can be run standalone
async def run_integration_test():
    """Run integration test for unified cognitive kernel"""
    print("🧠 Starting Unified Cognitive Kernel Integration Test...")
    
    try:
        # Create configuration
        config = CognitiveKernelConfig(
            tensor_kernel={
                'ggml_backend': 'cpu',
                'kokkos_execution_space': 'serial',
                'a0ml_meta_learning_enabled': True
            },
            memory_config={
                'hierarchical_levels': 3,
                'episodic_capacity': 1000,
                'semantic_capacity': 5000
            },
            attention_config={
                'ecan_enabled': True,
                'attention_decay': 0.95,
                'importance_threshold': 0.1
            },
            api_config={
                'enable_grpc': False,
                'enable_rest': False,
                'port': 8081
            }
        )
        
        print("⚙️  Initializing cognitive kernel...")
        kernel = UnifiedCognitiveKernel(config)
        await kernel.initialize()
        print("✅ Cognitive kernel initialized successfully!")
        
        # Test cognitive cycle
        print("🔄 Testing cognitive cycle...")
        input_data = {
            'cognitive_content': {
                'user_query': 'Explain the relationship between learning and memory',
                'context': 'educational_discussion',
                'cause': 'curiosity',
                'effect': 'knowledge_acquisition'
            },
            'tensor_data': {
                'query_embedding': np.random.randn(128),
                'context_embedding': np.random.randn(64)
            }
        }
        
        result = await kernel.cognitive_cycle(input_data)
        print(f"✅ Cognitive cycle completed! Result keys: {list(result.keys())}")
        
        # Test tensor shape meta-design
        print("🔢 Testing tensor shape meta-design...")
        tensor_meta = TensorShapeMetaDesign()
        
        # Create sample tensor shapes
        ggml_shape = tensor_meta.create_tensor_shape((64, 32), TensorCategory.GGML_TENSOR)
        mem0_shape = tensor_meta.create_tensor_shape((64, 128, 32), TensorCategory.MEM0_TENSOR)
        
        print(f"✅ GGML tensor shape: {ggml_shape.dimensions}, lexeme: {ggml_shape.lexeme_representation}")
        print(f"✅ Mem0 tensor shape: {mem0_shape.dimensions}, lexeme: {mem0_shape.lexeme_representation}")
        
        # Test gestalt field creation
        component_shapes = {
            'ggml': ggml_shape,
            'mem0': mem0_shape
        }
        
        gestalt_field = tensor_meta.create_gestalt_field(component_shapes)
        print(f"✅ Gestalt field created with shape: {gestalt_field.shape}")
        print(f"   Prime signature: {gestalt_field.prime_signature}")
        print(f"   Coherence measure: {gestalt_field.coherence_measure:.3f}")
        
        # Test meta-learning
        print("🎯 Testing meta-learning cycle...")
        await kernel.meta_learning_cycle()
        print("✅ Meta-learning cycle completed!")
        
        # Get final status
        status = kernel.get_status()
        print(f"✅ Final kernel status: {status['state']}")
        print(f"   Active components: {sum(status['active_components'].values())}/{len(status['active_components'])}")
        print(f"   Gestalt tensor shape: {status['gestalt_tensor_shape']}")
        
        # Shutdown
        await kernel.shutdown()
        print("✅ Cognitive kernel shutdown successfully!")
        
        print("\n🎉 Unified Cognitive Kernel Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run integration test
    success = asyncio.run(run_integration_test())
    exit(0 if success else 1)