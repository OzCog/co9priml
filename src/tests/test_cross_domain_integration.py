"""
Test suite for Cross-Domain Integration Framework

This module contains comprehensive tests for the cross-domain integration
capabilities of the CogPrime cognitive architecture.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from ..core.cognitive_core import CogPrimeCore
from ..modules.perception import SensoryInput
from ..integration.cross_domain_framework import (
    DomainType, ModalityType, ConceptMapping, AbstractConcept,
    CrossDomainIntegrationFramework
)
from ..integration.cross_domain_reasoning import ReasoningType
from ..integration.multimodal_knowledge_graph import ModalityFeature, ModalityEmbeddingType


class TestCrossDomainIntegration:
    """Test class for cross-domain integration functionality"""
    
    def __init__(self):
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up test environment with sample configuration"""
        self.config = {
            'visual_dim': 784,
            'audio_dim': 256,
            'memory_size': 1000,
            'cross_domain_config': {
                'representation_dimension': 512
            },
            'meta_learning_config': {
                'learning_rate': 0.001
            }
        }
        
        self.cognitive_system = CogPrimeCore(self.config)
        
        # Create test data
        self.create_test_data()
    
    def create_test_data(self):
        """Create test data for cross-domain integration"""
        # Visual test data
        self.visual_data = {
            'red_circle': np.random.randn(784) * 0.1,
            'blue_square': np.random.randn(784) * 0.1,
            'large_object': np.random.randn(784) * 0.1
        }
        
        # Audio test data
        self.audio_data = {
            'loud_sound': np.random.randn(256) * 0.1,
            'high_pitch': np.random.randn(256) * 0.1,
            'musical_note': np.random.randn(256) * 0.1
        }
        
        # Text test data
        self.text_data = {
            'red color description': "The object is red in color",
            'size description': "This is a large object",
            'sound description': "A loud, high-pitched sound"
        }
    
    def test_unified_representation(self) -> Dict[str, bool]:
        """Test unified representation system"""
        results = {}
        
        try:
            # Test concept mapping creation
            mapping_success = self.cognitive_system.add_cross_domain_concept_mapping(
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.LINGUISTIC,
                source_concept="red",
                target_concept="red color",
                mapping_strength=0.9
            )
            results['concept_mapping_creation'] = mapping_success
            
            # Test abstract concept registration
            abstract_success = self.cognitive_system.register_abstract_concept(
                concept_name="color",
                domain_instantiations={
                    DomainType.VISUAL: ["red", "blue", "green"],
                    DomainType.LINGUISTIC: ["red color", "blue color", "green color"]
                },
                abstraction_level=2
            )
            results['abstract_concept_registration'] = abstract_success
            
            # Test cross-domain similarity computation
            unified_rep = self.cognitive_system.cross_domain_framework.unified_representation
            
            similarity = unified_rep.compute_cross_domain_similarity(
                DomainType.VISUAL, "red",
                DomainType.LINGUISTIC, "red color"
            )
            results['cross_domain_similarity'] = similarity > 0.5
            
            # Test analogical concept finding
            analogies = unified_rep.find_cross_domain_analogies(
                DomainType.VISUAL, "red", [DomainType.LINGUISTIC]
            )
            results['analogy_finding'] = len(analogies) > 0
            
        except Exception as e:
            print(f"Error in unified representation test: {e}")
            results['unified_representation_error'] = str(e)
        
        return results
    
    def test_cross_modal_attention(self) -> Dict[str, bool]:
        """Test cross-modal attention mechanisms"""
        results = {}
        
        try:
            # Create multimodal input
            multimodal_input = {
                ModalityType.VISION: self.visual_data['red_circle'],
                ModalityType.HEARING: self.audio_data['loud_sound'],
                ModalityType.LANGUAGE: "red circular object with loud sound"
            }
            
            # Process through multimodal system
            processing_results = self.cognitive_system.process_multimodal_input(multimodal_input)
            
            results['multimodal_processing'] = processing_results.get('integration_success', False)
            results['feature_extraction'] = len(processing_results.get('extracted_features', {})) > 0
            results['correspondence_detection'] = len(processing_results.get('correspondences', [])) > 0
            results['entity_creation'] = len(processing_results.get('created_entities', [])) > 0
            
        except Exception as e:
            print(f"Error in cross-modal attention test: {e}")
            results['cross_modal_attention_error'] = str(e)
        
        return results
    
    def test_domain_adaptation(self) -> Dict[str, bool]:
        """Test domain adaptation capabilities"""
        results = {}
        
        try:
            # Test feature adaptation between domains
            source_features = np.random.randn(512) * 0.1
            
            adapted_features = self.cognitive_system.cross_domain_framework.domain_adaptation.adapt_features(
                features=source_features,
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.AUDITORY
            )
            
            results['feature_adaptation'] = adapted_features is not None and len(adapted_features) == len(source_features)
            
            # Test adaptation accuracy tracking
            # Simulate training with paired examples
            for _ in range(10):
                source = np.random.randn(512) * 0.1
                target = source + np.random.randn(512) * 0.05  # Similar but with noise
                
                loss = self.cognitive_system.cross_domain_framework.domain_adaptation.update_adapter(
                    source_domain=DomainType.VISUAL,
                    target_domain=DomainType.AUDITORY,
                    source_features=source,
                    target_features=target
                )
            
            accuracy = self.cognitive_system.cross_domain_framework.domain_adaptation.get_adaptation_accuracy(
                DomainType.VISUAL, DomainType.AUDITORY
            )
            
            results['adaptation_learning'] = accuracy > 0.3  # Should improve with training
            results['adaptation_ready'] = self.cognitive_system.cross_domain_framework.domain_adaptation.is_adaptation_ready(
                DomainType.VISUAL, DomainType.AUDITORY
            )
            
        except Exception as e:
            print(f"Error in domain adaptation test: {e}")
            results['domain_adaptation_error'] = str(e)
        
        return results
    
    def test_cross_domain_reasoning(self) -> Dict[str, bool]:
        """Test cross-domain reasoning capabilities"""
        results = {}
        
        try:
            # Test analogical reasoning
            analogical_inferences = self.cognitive_system.perform_cross_domain_reasoning(
                reasoning_type=ReasoningType.ANALOGICAL,
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.AUDITORY,
                source_facts=["red object is bright", "large object is prominent"],
                context={'test_context': True}
            )
            
            results['analogical_reasoning'] = len(analogical_inferences) > 0
            
            # Test causal reasoning
            causal_inferences = self.cognitive_system.perform_cross_domain_reasoning(
                reasoning_type=ReasoningType.CAUSAL,
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.AUDITORY,
                source_facts=["bright light causes attention", "movement causes tracking"],
                context={'test_context': True}
            )
            
            results['causal_reasoning'] = len(causal_inferences) > 0
            
            # Test deductive reasoning
            deductive_inferences = self.cognitive_system.perform_cross_domain_reasoning(
                reasoning_type=ReasoningType.DEDUCTIVE,
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.LINGUISTIC,
                source_facts=["red object exists", "red implies color"],
                context={'test_context': True}
            )
            
            results['deductive_reasoning'] = len(deductive_inferences) > 0
            
            # Test knowledge transfer
            transfer_result = self.cognitive_system.transfer_concept_across_domains(
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.LINGUISTIC,
                concept="red"
            )
            
            results['knowledge_transfer'] = transfer_result.get('transfer_confidence', 0) > 0.3
            
        except Exception as e:
            print(f"Error in cross-domain reasoning test: {e}")
            results['cross_domain_reasoning_error'] = str(e)
        
        return results
    
    def test_cognitive_cycle_integration(self) -> Dict[str, bool]:
        """Test integration with cognitive cycle"""
        results = {}
        
        try:
            # Create sensory input with multiple modalities
            sensory_input = SensoryInput(
                visual=np.random.randn(784),
                auditory=np.random.randn(256)
            )
            
            # Execute cognitive cycle
            action = self.cognitive_system.cognitive_cycle(sensory_input, reward=1.0)
            
            results['cognitive_cycle_execution'] = action is not None
            
            # Check if cross-domain information is in working memory
            working_memory = self.cognitive_system.state.working_memory
            results['cross_domain_memory_integration'] = 'cross_domain_inferences' in working_memory
            
            # Check if sensory buffer contains cross-modal results
            sensory_buffer = self.cognitive_system.state.sensory_buffer
            processing_info = sensory_buffer.get('processing_info', {})
            results['cross_modal_perception_integration'] = 'cross_modal_results' in processing_info
            
            # Test multiple cycles for consistency
            consistency_results = []
            for _ in range(5):
                new_input = SensoryInput(
                    visual=np.random.randn(784),
                    auditory=np.random.randn(256)
                )
                action = self.cognitive_system.cognitive_cycle(new_input, reward=0.5)
                consistency_results.append(action is not None)
            
            results['cycle_consistency'] = all(consistency_results)
            
        except Exception as e:
            print(f"Error in cognitive cycle integration test: {e}")
            results['cognitive_cycle_integration_error'] = str(e)
        
        return results
    
    def test_knowledge_consistency_validation(self) -> Dict[str, bool]:
        """Test knowledge consistency validation"""
        results = {}
        
        try:
            # Add some test knowledge
            for _ in range(10):
                sensory_input = SensoryInput(
                    visual=np.random.randn(784),
                    auditory=np.random.randn(256)
                )
                self.cognitive_system.cognitive_cycle(sensory_input, reward=np.random.uniform(0, 1))
            
            # Validate consistency
            consistency_results = self.cognitive_system.validate_cross_domain_knowledge_consistency()
            
            results['consistency_validation_execution'] = consistency_results is not None
            results['framework_consistency_available'] = 'framework_consistency' in consistency_results
            results['reasoning_consistency_available'] = 'reasoning_consistency' in consistency_results
            results['combined_score_calculated'] = 'combined_score' in consistency_results
            results['recommendations_provided'] = len(consistency_results.get('recommendations', [])) > 0
            
            # Test consistency scores are reasonable
            combined_score = consistency_results.get('combined_score', 0)
            results['reasonable_consistency_score'] = 0 <= combined_score <= 1
            
        except Exception as e:
            print(f"Error in consistency validation test: {e}")
            results['consistency_validation_error'] = str(e)
        
        return results
    
    def test_integration_status_reporting(self) -> Dict[str, bool]:
        """Test integration status reporting"""
        results = {}
        
        try:
            # Get integration status
            status = self.cognitive_system.get_cross_domain_integration_status()
            
            results['status_reporting_execution'] = status is not None
            results['framework_status_available'] = 'framework_status' in status
            results['reasoning_statistics_available'] = 'reasoning_statistics' in status
            results['multimodal_statistics_available'] = 'multimodal_statistics' in status
            results['consistency_validation_available'] = 'consistency_validation' in status
            results['overall_health_available'] = 'overall_health' in status
            
            # Check framework status structure
            framework_status = status.get('framework_status', {})
            results['active_domains_listed'] = 'active_domains' in framework_status
            results['active_modalities_listed'] = 'active_modalities' in framework_status
            results['concept_mappings_counted'] = 'total_concept_mappings' in framework_status
            
            # Check reasoning statistics structure
            reasoning_stats = status.get('reasoning_statistics', {})
            results['total_inferences_counted'] = 'total_inferences' in reasoning_stats
            results['reasoning_types_broken_down'] = 'reasoning_types' in reasoning_stats
            results['average_strength_calculated'] = 'average_strength' in reasoning_stats
            
        except Exception as e:
            print(f"Error in integration status reporting test: {e}")
            results['integration_status_reporting_error'] = str(e)
        
        return results
    
    def run_all_tests(self) -> Dict[str, Dict[str, bool]]:
        """Run all cross-domain integration tests"""
        print("Running Cross-Domain Integration Tests...")
        
        all_results = {}
        
        print("  Testing unified representation...")
        all_results['unified_representation'] = self.test_unified_representation()
        
        print("  Testing cross-modal attention...")
        all_results['cross_modal_attention'] = self.test_cross_modal_attention()
        
        print("  Testing domain adaptation...")
        all_results['domain_adaptation'] = self.test_domain_adaptation()
        
        print("  Testing cross-domain reasoning...")
        all_results['cross_domain_reasoning'] = self.test_cross_domain_reasoning()
        
        print("  Testing cognitive cycle integration...")
        all_results['cognitive_cycle_integration'] = self.test_cognitive_cycle_integration()
        
        print("  Testing knowledge consistency validation...")
        all_results['consistency_validation'] = self.test_knowledge_consistency_validation()
        
        print("  Testing integration status reporting...")
        all_results['status_reporting'] = self.test_integration_status_reporting()
        
        # Generate summary
        self.print_test_summary(all_results)
        
        return all_results
    
    def print_test_summary(self, results: Dict[str, Dict[str, bool]]):
        """Print a summary of test results"""
        print("\n" + "="*60)
        print("CROSS-DOMAIN INTEGRATION TEST SUMMARY")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_category, test_results in results.items():
            print(f"\n{test_category.upper()}:")
            
            category_passed = 0
            category_total = 0
            
            for test_name, result in test_results.items():
                if not test_name.endswith('_error'):
                    category_total += 1
                    total_tests += 1
                    
                    if result:
                        category_passed += 1
                        passed_tests += 1
                        status = "PASS"
                    else:
                        status = "FAIL"
                    
                    print(f"  {test_name}: {status}")
                else:
                    print(f"  ERROR: {result}")
            
            if category_total > 0:
                category_percentage = (category_passed / category_total) * 100
                print(f"  Category Success Rate: {category_passed}/{category_total} ({category_percentage:.1f}%)")
        
        print(f"\n{'='*60}")
        if total_tests > 0:
            overall_percentage = (passed_tests / total_tests) * 100
            print(f"OVERALL SUCCESS RATE: {passed_tests}/{total_tests} ({overall_percentage:.1f}%)")
        else:
            print("NO TESTS COMPLETED")
        
        print("="*60)
        
        # Provide recommendations based on results
        self.provide_test_recommendations(results)
    
    def provide_test_recommendations(self, results: Dict[str, Dict[str, bool]]):
        """Provide recommendations based on test results"""
        print("\nRECOMMENDATIONS:")
        print("-" * 30)
        
        recommendations = []
        
        # Check unified representation
        unified_results = results.get('unified_representation', {})
        if not unified_results.get('concept_mapping_creation', True):
            recommendations.append("Review concept mapping creation logic")
        if not unified_results.get('cross_domain_similarity', True):
            recommendations.append("Improve cross-domain similarity computation")
        
        # Check cross-modal attention
        modal_results = results.get('cross_modal_attention', {})
        if not modal_results.get('correspondence_detection', True):
            recommendations.append("Enhance cross-modal correspondence detection")
        if not modal_results.get('entity_creation', True):
            recommendations.append("Improve multi-modal entity creation")
        
        # Check domain adaptation
        adaptation_results = results.get('domain_adaptation', {})
        if not adaptation_results.get('adaptation_learning', True):
            recommendations.append("Improve domain adaptation learning algorithms")
        
        # Check reasoning
        reasoning_results = results.get('cross_domain_reasoning', {})
        reasoning_types = ['analogical_reasoning', 'causal_reasoning', 'deductive_reasoning']
        failed_reasoning = [rt for rt in reasoning_types if not reasoning_results.get(rt, True)]
        if failed_reasoning:
            recommendations.append(f"Enhance reasoning types: {', '.join(failed_reasoning)}")
        
        # Check integration
        integration_results = results.get('cognitive_cycle_integration', {})
        if not integration_results.get('cycle_consistency', True):
            recommendations.append("Improve cognitive cycle consistency")
        
        if not recommendations:
            recommendations.append("All systems functioning well! Consider adding more advanced tests.")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print("-" * 30)


def run_cross_domain_integration_tests():
    """Run the complete cross-domain integration test suite"""
    test_suite = TestCrossDomainIntegration()
    return test_suite.run_all_tests()


if __name__ == "__main__":
    # Run tests when script is executed directly
    run_cross_domain_integration_tests()