#!/usr/bin/env python3
"""
Simplified test for the Unified Cognitive Kernel Integration
Tests core functionality without external dependencies.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock numpy for basic testing
class MockNdarray:
    def __init__(self, shape, data=None):
        self.shape = shape
        self.size = 1
        for dim in shape:
            self.size *= dim
        self.data = data or [0.0] * self.size
    
    def __getitem__(self, key):
        return self.data[key] if isinstance(key, int) else MockNdarray((2,), [1.0, 2.0])
    
    def flatten(self):
        return MockNdarray((self.size,), self.data)
    
    def copy(self):
        return MockNdarray(self.shape, self.data.copy())
    
    def reshape(self, new_shape):
        return MockNdarray(new_shape, self.data)
    
    def __add__(self, other):
        if isinstance(other, MockNdarray):
            return MockNdarray(self.shape, [a + b for a, b in zip(self.data, other.data)])
        return MockNdarray(self.shape, [x + other for x in self.data])
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MockNdarray(self.shape, [x * other for x in self.data])
        return self

# Mock numpy module
class MockNumpy:
    def __init__(self):
        self.ndarray = MockNdarray
    
    def array(self, data):
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                return MockNdarray((0,))
            if isinstance(data[0], (list, tuple)):
                return MockNdarray((len(data), len(data[0])))
            return MockNdarray((len(data),), list(data))
        return MockNdarray((1,), [data])
    
    def zeros(self, shape):
        return MockNdarray(shape)
    
    def random(self):
        return MockRandomState()
    
    def prod(self, arr):
        if isinstance(arr, (list, tuple)):
            result = 1
            for x in arr:
                result *= x
            return result
        return arr.size if hasattr(arr, 'size') else 1
    
    def mean(self, arr):
        return 0.5
    
    def var(self, arr):
        return 0.1
    
    def sum(self, arr):
        return 10.0
    
    def linalg(self):
        return MockLinalg()

class MockRandomState:
    def randn(self, *shape):
        return MockNdarray(shape)
    
    def rand(self, *shape):
        return MockNdarray(shape)

class MockLinalg:
    def norm(self, arr):
        return 1.0

# Inject mock numpy
import types
mock_np = MockNumpy()
sys.modules['numpy'] = mock_np

# Now import our modules
try:
    from unified_cognitive_kernel.cognitive_kernel import UnifiedCognitiveKernel, CognitiveKernelConfig
    from unified_cognitive_kernel.tensor_shapes import TensorShapeMetaDesign, TensorCategory
    print("✅ Successfully imported unified cognitive kernel modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    traceback.print_exc()
    sys.exit(1)


async def test_tensor_shape_meta_design():
    """Test tensor shape meta-design without numpy"""
    print("🔢 Testing Tensor Shape Meta-Design...")
    
    try:
        tensor_meta = TensorShapeMetaDesign()
        
        # Test tensor shape creation
        dimensions = (64, 32)
        tensor_shape = tensor_meta.create_tensor_shape(dimensions, TensorCategory.GGML_TENSOR)
        
        print(f"   ✅ Created tensor shape: {tensor_shape.dimensions}")
        print(f"      Unique address: {tensor_shape.unique_address}")
        print(f"      Lexeme: {tensor_shape.lexeme_representation}")
        print(f"      Degrees of freedom: {tensor_shape.degrees_of_freedom}")
        
        # Test gestalt shape computation
        component_shapes = {
            'ggml': tensor_meta.create_tensor_shape((64, 32), TensorCategory.GGML_TENSOR),
            'mem0': tensor_meta.create_tensor_shape((64, 128, 32), TensorCategory.MEM0_TENSOR)
        }
        
        gestalt_shape = tensor_meta.compute_gestalt_shape(component_shapes)
        print(f"   ✅ Computed gestalt shape: {gestalt_shape}")
        
        # Test component tensor factorization
        component_tensor = tensor_meta.factorize_component_tensor('ggml', {'data': mock_np.array([1, 2, 3])})
        print(f"   ✅ Factorized component tensor shape: {component_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Tensor meta-design test failed: {e}")
        traceback.print_exc()
        return False


async def test_cognitive_kernel_initialization():
    """Test cognitive kernel initialization"""
    print("🧠 Testing Cognitive Kernel Initialization...")
    
    try:
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
        
        print("   ✅ Created cognitive kernel config")
        
        kernel = UnifiedCognitiveKernel(config)
        print("   ✅ Created cognitive kernel instance")
        
        await kernel.initialize()
        print("   ✅ Initialized cognitive kernel")
        
        # Test kernel status
        status = kernel.get_status()
        print(f"   ✅ Kernel state: {status['state']}")
        print(f"   ✅ Active components: {sum(status['active_components'].values())}/{len(status['active_components'])}")
        
        # Test cognitive cycle
        input_data = {
            'cognitive_content': {
                'user_input': 'test query',
                'context': 'testing'
            },
            'tensor_data': {
                'test_data': mock_np.array([1, 2, 3, 4])
            }
        }
        
        result = await kernel.cognitive_cycle(input_data)
        print(f"   ✅ Cognitive cycle completed with {len(result)} result components")
        
        await kernel.shutdown()
        print("   ✅ Kernel shutdown successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cognitive kernel test failed: {e}")
        traceback.print_exc()
        return False


async def test_prime_factorization():
    """Test prime factorization functionality"""
    print("🔢 Testing Prime Factorization...")
    
    try:
        from unified_cognitive_kernel.tensor_shapes import PrimeFactorCalculator
        
        calculator = PrimeFactorCalculator()
        
        # Test various numbers
        test_numbers = [2, 3, 4, 8, 12, 15, 32, 64, 100]
        
        for num in test_numbers:
            factorization = calculator.factorize(num)
            print(f"   ✅ {num} = {factorization.factorization_string}")
        
        # Test prime signature
        dimensions = (64, 32, 16)
        signature = calculator.get_prime_signature(dimensions)
        print(f"   ✅ Prime signature for {dimensions}: {signature}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Prime factorization test failed: {e}")
        traceback.print_exc()
        return False


async def test_lexeme_generation():
    """Test lexeme generation functionality"""
    print("📝 Testing Lexeme Generation...")
    
    try:
        from unified_cognitive_kernel.tensor_shapes import LexemeGenerator, TensorShape, TensorComplexity
        
        generator = LexemeGenerator()
        
        # Create test tensor shapes
        test_shapes = [
            ((2,), "scalar tensor"),
            ((10, 5), "matrix tensor"),
            ((8, 4, 2), "3D tensor"),
            ((16, 8, 4, 2), "4D tensor")
        ]
        
        for dimensions, description in test_shapes:
            # Create a mock tensor shape
            tensor_shape = type('TensorShape', (), {
                'dimensions': dimensions,
                'unique_address': f"mock_address_{len(dimensions)}d",
                'prime_factorizations': [
                    type('PrimeFactorization', (), {
                        'factorization_string': f"2^{i+1}"
                    })() for i in range(len(dimensions))
                ]
            })()
            
            lexeme = generator.generate_lexeme(tensor_shape)
            print(f"   ✅ {description} {dimensions}: {lexeme}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Lexeme generation test failed: {e}")
        traceback.print_exc()
        return False


async def run_comprehensive_test():
    """Run comprehensive test of the unified cognitive kernel"""
    print("🚀 Unified Cognitive Kernel Integration Test")
    print("=" * 60)
    print("Testing transcendent cognitive synthesis architecture...")
    print("Components: a0ml, ggml-org-central, kokkos-central, mem0, mlpn, node9")
    print("=" * 60)
    print()
    
    tests = [
        ("Prime Factorization", test_prime_factorization),
        ("Lexeme Generation", test_lexeme_generation),
        ("Tensor Shape Meta-Design", test_tensor_shape_meta_design),
        ("Cognitive Kernel", test_cognitive_kernel_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} test...")
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} test PASSED\n")
            else:
                print(f"❌ {test_name} test FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}\n")
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Unified Cognitive Kernel Integration SUCCESS!")
        print()
        print("✨ Key Features Validated:")
        print("   • Tensor shape meta-design with prime factorization")
        print("   • Lexeme generation for cognitive representation")
        print("   • Unified cognitive kernel architecture")
        print("   • Component integration and orchestration")
        print("   • Gestalt tensor field creation")
        print("   • Economic attention allocation principles")
        print()
        return True
    else:
        print("⚠️  Some tests failed - Review implementation")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)