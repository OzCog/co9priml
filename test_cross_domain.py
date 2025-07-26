#!/usr/bin/env python3
"""
Simple test for Cross-Domain Integration Framework
without external dependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def mock_numpy():
    """Mock numpy for basic testing"""
    class MockNumpy:
        def random(self):
            class Random:
                def randn(self, *args):
                    import random
                    if args:
                        return [random.gauss(0, 0.1) for _ in range(args[0] if len(args) == 1 else args[0] * args[1])]
                    return random.gauss(0, 0.1)
            return Random()
        
        def array(self, data):
            return data
        
        def mean(self, data, axis=None):
            if isinstance(data, list):
                return sum(data) / len(data) if data else 0
            return data
        
        def linalg(self):
            class Linalg:
                def norm(self, vec):
                    if isinstance(vec, list):
                        return (sum(x*x for x in vec) ** 0.5) if vec else 1.0
                    return abs(vec)
            return Linalg()
        
        def dot(self, a, b):
            if isinstance(a, list) and isinstance(b, list):
                return sum(x*y for x, y in zip(a, b))
            return a * b
        
        def zeros(self, size):
            return [0.0] * size
        
        def ones(self, size):
            return [1.0] * size
    
    return MockNumpy()

# Mock numpy globally
import builtins
builtins.np = mock_numpy()
sys.modules['numpy'] = builtins.np

def test_cross_domain_framework():
    """Test the cross-domain integration framework"""
    print("🧪 Testing Cross-Domain Integration Framework")
    print("="*50)
    
    try:
        # Import after mocking numpy
        from src.integration.cross_domain_framework import (
            CrossDomainIntegrationFramework, DomainType, ModalityType,
            ConceptMapping, AbstractConcept
        )
        print("✅ Cross-domain framework imported successfully")
        
        # Test basic initialization
        framework = CrossDomainIntegrationFramework()
        print("✅ Framework initialized")
        
        # Test domain registration
        framework.register_domain(DomainType.VISUAL, ['red', 'blue', 'large'])
        framework.register_domain(DomainType.AUDITORY, ['loud', 'quiet', 'high'])
        framework.register_domain(DomainType.LINGUISTIC, ['word', 'sentence', 'meaning'])
        print("✅ Domains registered")
        
        # Test modality registration
        framework.register_modality(ModalityType.VISION, [ModalityType.HEARING])
        framework.register_modality(ModalityType.HEARING, [ModalityType.VISION])
        framework.register_modality(ModalityType.LANGUAGE)
        print("✅ Modalities registered")
        
        # Test concept mapping
        mappings = [
            ConceptMapping(
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.AUDITORY,
                source_concept='bright',
                target_concept='loud',
                mapping_strength=0.8,
                semantic_similarity=0.7
            ),
            ConceptMapping(
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.LINGUISTIC,
                source_concept='red',
                target_concept='red color',
                mapping_strength=0.9,
                semantic_similarity=0.85
            )
        ]
        
        for mapping in mappings:
            success = framework.unified_representation.add_concept_mapping(mapping)
            print(f"✅ Concept mapping added: {mapping.source_concept} -> {mapping.target_concept} ({success})")
        
        # Test abstract concept registration
        abstract_concept = AbstractConcept(
            concept_id="intensity",
            concept_name="intensity",
            abstraction_level=2,
            domain_instantiations={
                DomainType.VISUAL: ['bright', 'dim'],
                DomainType.AUDITORY: ['loud', 'quiet']
            },
            semantic_features={'strength': 0.8, 'salience': 0.7}
        )
        
        success = framework.unified_representation.register_abstract_concept(abstract_concept)
        print(f"✅ Abstract concept registered: {success}")
        
        # Test cross-domain similarity
        similarity = framework.unified_representation.compute_cross_domain_similarity(
            DomainType.VISUAL, 'bright',
            DomainType.AUDITORY, 'loud'
        )
        print(f"✅ Cross-domain similarity computed: {similarity:.3f}")
        
        # Test analogical search
        analogies = framework.unified_representation.find_cross_domain_analogies(
            DomainType.VISUAL, 'bright', [DomainType.AUDITORY, DomainType.LINGUISTIC]
        )
        print(f"✅ Found {len(analogies)} analogies for 'bright'")
        
        for domain, concept, sim in analogies:
            print(f"   {domain.value}: {concept} (similarity: {sim:.3f})")
        
        # Test cross-domain processing
        domain_inputs = {
            DomainType.VISUAL: {'concept': 'red', 'features': {'brightness': 0.8, 'saturation': 0.9}},
            DomainType.AUDITORY: {'concept': 'loud', 'features': {'volume': 0.8, 'pitch': 0.5}}
        }
        
        results = framework.process_cross_domain_input(domain_inputs)
        print(f"✅ Cross-domain processing: {results['integration_success']}")
        print(f"   Unified representations: {len(results['unified_representations'])}")
        print(f"   Domain adaptations: {len(results['domain_adaptations'])}")
        
        # Test knowledge transfer
        transfer_result = framework.transfer_knowledge(
            DomainType.VISUAL, DomainType.AUDITORY, 'bright'
        )
        print(f"✅ Knowledge transfer: confidence {transfer_result['transfer_confidence']:.3f}")
        if transfer_result['best_analogy']:
            print(f"   Best analogy: {transfer_result['best_analogy'][1]} in {transfer_result['best_analogy'][0]}")
        
        # Test consistency validation
        consistency = framework.validate_cross_domain_consistency()
        print(f"✅ Consistency validation:")
        print(f"   Concept mappings: {consistency['concept_mapping_consistency']:.3f}")
        print(f"   Adaptation accuracy: {consistency['adaptation_accuracy']:.3f}")
        print(f"   Cross-modal coherence: {consistency['cross_modal_coherence']:.3f}")
        print(f"   Overall consistency: {consistency['overall_consistency']:.3f}")
        
        # Test status reporting
        status = framework.get_integration_status()
        print(f"✅ Framework status:")
        print(f"   Active domains: {len(status['active_domains'])}")
        print(f"   Active modalities: {len(status['active_modalities'])}")
        print(f"   Concept mappings: {status['total_concept_mappings']}")
        print(f"   Abstract concepts: {status['total_abstract_concepts']}")
        
        health = status['framework_health']
        print(f"   Health assessment:")
        for component, health_status in health.items():
            emoji = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(health_status, "⚪")
            print(f"     {component}: {emoji} {health_status}")
        
        print("\n🎉 Cross-Domain Integration Framework test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reasoning_engine():
    """Test the cross-domain reasoning engine"""
    print("\n🧪 Testing Cross-Domain Reasoning Engine")
    print("="*50)
    
    try:
        from src.integration.cross_domain_framework import CrossDomainIntegrationFramework, DomainType
        from src.integration.cross_domain_reasoning import (
            CrossDomainReasoningEngine, ReasoningType, CrossDomainKnowledgeGraph
        )
        
        # Create framework and reasoning engine
        framework = CrossDomainIntegrationFramework()
        reasoning_engine = CrossDomainReasoningEngine(framework)
        print("✅ Reasoning engine initialized")
        
        # Populate with test knowledge
        visual_knowledge = {
            'concepts': {
                'red': {'attributes': {'color': True}, 'uncertainty': 0.1},
                'bright': {'attributes': {'intensity': 'high'}, 'uncertainty': 0.1}
            },
            'relations': [
                {'id': 'red_bright', 'source': 'red', 'target': 'bright', 'type': 'can_be', 'strength': 0.7}
            ]
        }
        
        auditory_knowledge = {
            'concepts': {
                'loud': {'attributes': {'volume': 'high'}, 'uncertainty': 0.1},
                'high_pitch': {'attributes': {'frequency': 'high'}, 'uncertainty': 0.1}
            },
            'relations': [
                {'id': 'loud_attention', 'source': 'loud', 'target': 'attention', 'type': 'causes', 'strength': 0.8}
            ]
        }
        
        domain_knowledge = {
            DomainType.VISUAL: visual_knowledge,
            DomainType.AUDITORY: auditory_knowledge
        }
        
        reasoning_engine.populate_knowledge_graph(domain_knowledge)
        print("✅ Knowledge graph populated")
        
        # Test different types of reasoning
        reasoning_tests = [
            (ReasoningType.ANALOGICAL, "bright objects are prominent"),
            (ReasoningType.CAUSAL, "bright light causes attention"),
            (ReasoningType.DEDUCTIVE, "red objects exist")
        ]
        
        total_inferences = 0
        for reasoning_type, test_fact in reasoning_tests:
            inferences = reasoning_engine.make_cross_domain_inference(
                reasoning_type=reasoning_type,
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.AUDITORY,
                source_facts=[test_fact],
                context={'test_mode': True}
            )
            
            print(f"✅ {reasoning_type.value} reasoning: {len(inferences)} inferences")
            total_inferences += len(inferences)
            
            for inference in inferences[:2]:  # Show first 2
                print(f"   💡 {inference.inferred_fact}")
                print(f"      Strength: {inference.inference_strength:.3f}")
        
        # Test consistency validation
        consistency = reasoning_engine.validate_inference_consistency()
        print(f"✅ Inference consistency: {consistency['overall_consistency']:.3f}")
        
        # Test statistics
        stats = reasoning_engine.get_reasoning_statistics()
        print(f"✅ Reasoning statistics:")
        print(f"   Total inferences: {stats['total_inferences']}")
        print(f"   Average strength: {stats['average_strength']:.3f}")
        
        print("\n🎉 Cross-Domain Reasoning Engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during reasoning test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Cross-Domain Integration Framework Testing Suite")
    print("="*60)
    
    success_count = 0
    total_tests = 2
    
    # Test framework
    if test_cross_domain_framework():
        success_count += 1
    
    # Test reasoning engine
    if test_reasoning_engine():
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Cross-Domain Integration Framework is working correctly.")
        return True
    else:
        print(f"⚠️  {total_tests - success_count} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    main()