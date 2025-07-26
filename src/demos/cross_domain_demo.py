"""
Cross-Domain Integration Framework Demo

This demo showcases the cross-domain integration capabilities of the CogPrime
cognitive architecture, demonstrating unified representation, cross-modal
processing, domain adaptation, and cross-domain reasoning.
"""

import numpy as np
from typing import Dict, List, Any

from ..core.cognitive_core import CogPrimeCore
from ..modules.perception import SensoryInput
from ..integration.cross_domain_framework import DomainType, ModalityType
from ..integration.cross_domain_reasoning import ReasoningType


class CrossDomainIntegrationDemo:
    """Demo class for cross-domain integration capabilities"""
    
    def __init__(self):
        self.setup_demo_environment()
    
    def setup_demo_environment(self):
        """Set up the demo environment"""
        print("🧠 Initializing CogPrime with Cross-Domain Integration...")
        
        # Configure the cognitive system
        config = {
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
        
        self.cognitive_system = CogPrimeCore(config)
        
        # Create demo scenarios
        self.create_demo_scenarios()
        
        print("✅ CogPrime initialized with cross-domain capabilities")
        print(f"📊 Active domains: {len(self.cognitive_system.cross_domain_framework.active_domains)}")
        print(f"🔀 Active modalities: {len(self.cognitive_system.cross_domain_framework.active_modalities)}")
    
    def create_demo_scenarios(self):
        """Create demo scenarios for cross-domain integration"""
        # Scenario 1: Visual-Language Integration
        self.scenario_1 = {
            'name': 'Visual-Language Integration',
            'description': 'A red circular object that makes a loud sound',
            'visual_data': np.random.randn(784) * 0.1,  # Simulated visual features
            'audio_data': np.random.randn(256) * 0.1,   # Simulated audio features
            'text_description': 'bright red circular object',
            'expected_concepts': ['red', 'circular', 'bright', 'loud']
        }
        
        # Scenario 2: Audio-Visual Synchrony
        self.scenario_2 = {
            'name': 'Audio-Visual Synchrony',
            'description': 'Musical performance with visual movement',
            'visual_data': np.random.randn(784) * 0.1,
            'audio_data': np.random.randn(256) * 0.1,
            'text_description': 'rhythmic movement with musical accompaniment',
            'expected_concepts': ['rhythm', 'movement', 'music', 'synchrony']
        }
        
        # Scenario 3: Cross-Domain Reasoning
        self.scenario_3 = {
            'name': 'Cross-Domain Reasoning',
            'description': 'Learning visual concepts and applying to audio domain',
            'source_domain': DomainType.VISUAL,
            'target_domain': DomainType.AUDITORY,
            'source_facts': ['bright objects attract attention', 'large objects are prominent'],
            'expected_analogies': ['loud sounds attract attention', 'strong sounds are prominent']
        }
    
    def demonstrate_unified_representation(self):
        """Demonstrate unified representation across domains"""
        print("\n" + "="*60)
        print("🔄 DEMONSTRATION 1: UNIFIED REPRESENTATION FRAMEWORK")
        print("="*60)
        
        print("\n📝 Creating cross-domain concept mappings...")
        
        # Add concept mappings
        mappings = [
            (DomainType.VISUAL, DomainType.LINGUISTIC, "red", "red color", 0.9),
            (DomainType.VISUAL, DomainType.LINGUISTIC, "bright", "luminous", 0.8),
            (DomainType.AUDITORY, DomainType.LINGUISTIC, "loud", "high volume", 0.85),
            (DomainType.SPATIAL, DomainType.TEMPORAL, "near", "soon", 0.7),
        ]
        
        for source_domain, target_domain, source_concept, target_concept, strength in mappings:
            success = self.cognitive_system.add_cross_domain_concept_mapping(
                source_domain, target_domain, source_concept, target_concept, strength
            )
            print(f"  ✅ Mapped '{source_concept}' ({source_domain.value}) -> '{target_concept}' ({target_domain.value})")
        
        print("\n🔗 Creating abstract concepts...")
        
        # Register abstract concepts
        abstract_concepts = [
            ("intensity", {
                DomainType.VISUAL: ["bright", "dim"],
                DomainType.AUDITORY: ["loud", "quiet"],
                DomainType.LINGUISTIC: ["strong", "weak"]
            }),
            ("size", {
                DomainType.VISUAL: ["large", "small"],
                DomainType.AUDITORY: ["booming", "tiny"],
                DomainType.SPATIAL: ["big", "little"]
            })
        ]
        
        for concept_name, domain_instantiations in abstract_concepts:
            success = self.cognitive_system.register_abstract_concept(
                concept_name, domain_instantiations, abstraction_level=2
            )
            print(f"  ✅ Registered abstract concept '{concept_name}' across {len(domain_instantiations)} domains")
        
        print("\n📊 Testing cross-domain similarity...")
        
        # Test similarity computation
        unified_rep = self.cognitive_system.cross_domain_framework.unified_representation
        
        similarities = [
            (DomainType.VISUAL, "red", DomainType.LINGUISTIC, "red color"),
            (DomainType.VISUAL, "bright", DomainType.AUDITORY, "loud"),
            (DomainType.SPATIAL, "near", DomainType.TEMPORAL, "soon")
        ]
        
        for domain1, concept1, domain2, concept2 in similarities:
            similarity = unified_rep.compute_cross_domain_similarity(domain1, concept1, domain2, concept2)
            print(f"  📈 Similarity between '{concept1}' and '{concept2}': {similarity:.3f}")
        
        print("\n🔍 Finding cross-domain analogies...")
        
        # Find analogies
        analogies = unified_rep.find_cross_domain_analogies(
            DomainType.VISUAL, "bright", [DomainType.AUDITORY, DomainType.LINGUISTIC]
        )
        
        for target_domain, target_concept, similarity in analogies:
            print(f"  🎯 '{target_concept}' in {target_domain.value} (similarity: {similarity:.3f})")
    
    def demonstrate_cross_modal_attention(self):
        """Demonstrate cross-modal attention and integration"""
        print("\n" + "="*60)
        print("👁️ DEMONSTRATION 2: CROSS-MODAL ATTENTION & INTEGRATION")
        print("="*60)
        
        scenario = self.scenario_1
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Create multimodal input
        multimodal_input = {
            ModalityType.VISION: scenario['visual_data'],
            ModalityType.HEARING: scenario['audio_data'],
            ModalityType.LANGUAGE: scenario['text_description']
        }
        
        print(f"\n🔄 Processing multimodal input...")
        print(f"   Visual features: {len(scenario['visual_data'])} dimensions")
        print(f"   Audio features: {len(scenario['audio_data'])} dimensions")
        print(f"   Language: '{scenario['text_description']}'")
        
        # Process through multimodal system
        results = self.cognitive_system.process_multimodal_input(
            multimodal_input,
            context={'scenario': scenario['name']}
        )
        
        print(f"\n📊 Processing Results:")
        print(f"   ✅ Integration successful: {results.get('integration_success', False)}")
        print(f"   🔍 Features extracted from {len(results.get('extracted_features', {}))} modalities")
        print(f"   🔗 Found {len(results.get('correspondences', []))} cross-modal correspondences")
        print(f"   🎯 Created {len(results.get('created_entities', []))} new entities")
        print(f"   🔄 Updated {len(results.get('updated_entities', []))} existing entities")
        
        # Show correspondence details
        correspondences = results.get('correspondences', [])
        if correspondences:
            print(f"\n🔗 Cross-Modal Correspondences:")
            for i, corr in enumerate(correspondences[:3]):  # Show first 3
                modalities = list(corr.modality_features.keys())
                print(f"   {i+1}. Correspondence between {[m.value for m in modalities]}")
                print(f"      Strength: {corr.correspondence_strength:.3f}")
                print(f"      Temporal alignment: {corr.temporal_alignment:.3f}")
                print(f"      Spatial alignment: {corr.spatial_alignment:.3f}")
    
    def demonstrate_domain_adaptation(self):
        """Demonstrate domain adaptation capabilities"""
        print("\n" + "="*60)
        print("🔧 DEMONSTRATION 3: DOMAIN ADAPTATION")
        print("="*60)
        
        print("\n📊 Testing feature adaptation between domains...")
        
        # Create sample features for adaptation
        visual_features = np.random.randn(512) * 0.1
        print(f"   Original visual features shape: {visual_features.shape}")
        
        # Adapt visual features to auditory domain
        adapted_features = self.cognitive_system.cross_domain_framework.domain_adaptation.adapt_features(
            features=visual_features,
            source_domain=DomainType.VISUAL,
            target_domain=DomainType.AUDITORY,
            context={'adaptation_strength': 0.8}
        )
        
        print(f"   Adapted features shape: {adapted_features.shape}")
        print(f"   Feature similarity: {np.corrcoef(visual_features, adapted_features)[0,1]:.3f}")
        
        print("\n🎯 Training domain adapter with examples...")
        
        # Simulate training with paired examples
        training_losses = []
        for i in range(10):
            # Create paired examples (visual -> audio)
            visual_sample = np.random.randn(512) * 0.1
            # Simulate corresponding audio features (with some transformation)
            audio_sample = visual_sample * 0.8 + np.random.randn(512) * 0.05
            
            loss = self.cognitive_system.cross_domain_framework.domain_adaptation.update_adapter(
                source_domain=DomainType.VISUAL,
                target_domain=DomainType.AUDITORY,
                source_features=visual_sample,
                target_features=audio_sample,
                learning_rate=0.01
            )
            training_losses.append(loss)
            
            if i % 3 == 0:
                print(f"   Training step {i+1}: loss = {loss:.4f}")
        
        # Check adaptation accuracy
        accuracy = self.cognitive_system.cross_domain_framework.domain_adaptation.get_adaptation_accuracy(
            DomainType.VISUAL, DomainType.AUDITORY
        )
        
        print(f"\n📈 Final adaptation accuracy: {accuracy:.3f}")
        print(f"   Training improved adaptation by {(accuracy - 0.5):.3f}")
        
        is_ready = self.cognitive_system.cross_domain_framework.domain_adaptation.is_adaptation_ready(
            DomainType.VISUAL, DomainType.AUDITORY
        )
        print(f"   Adaptation ready for use: {is_ready}")
    
    def demonstrate_cross_domain_reasoning(self):
        """Demonstrate cross-domain reasoning capabilities"""
        print("\n" + "="*60)
        print("🤔 DEMONSTRATION 4: CROSS-DOMAIN REASONING")
        print("="*60)
        
        scenario = self.scenario_3
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Source domain: {scenario['source_domain'].value}")
        print(f"   Target domain: {scenario['target_domain'].value}")
        
        print(f"\n📚 Source facts:")
        for fact in scenario['source_facts']:
            print(f"   • {fact}")
        
        # Test different reasoning types
        reasoning_types = [
            (ReasoningType.ANALOGICAL, "🔄 Analogical"),
            (ReasoningType.CAUSAL, "⚡ Causal"),
            (ReasoningType.DEDUCTIVE, "📐 Deductive"),
            (ReasoningType.INDUCTIVE, "📈 Inductive")
        ]
        
        all_inferences = []
        
        for reasoning_type, type_name in reasoning_types:
            print(f"\n{type_name} Reasoning:")
            
            inferences = self.cognitive_system.perform_cross_domain_reasoning(
                reasoning_type=reasoning_type,
                source_domain=scenario['source_domain'],
                target_domain=scenario['target_domain'],
                source_facts=scenario['source_facts'],
                context={'demo_scenario': True}
            )
            
            if inferences:
                for inference in inferences[:2]:  # Show first 2 inferences
                    print(f"   💡 {inference.inferred_fact}")
                    print(f"      Strength: {inference.inference_strength:.3f}")
                    print(f"      Evidence: {len(inference.evidence_chain)} pieces")
                
                all_inferences.extend(inferences)
            else:
                print("   🚫 No inferences generated")
        
        print(f"\n📊 Total inferences generated: {len(all_inferences)}")
        
        if all_inferences:
            avg_strength = np.mean([inf.inference_strength for inf in all_inferences])
            print(f"   Average inference strength: {avg_strength:.3f}")
            
            # Show reasoning type distribution
            type_counts = {}
            for inf in all_inferences:
                type_name = inf.reasoning_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            print(f"   Reasoning type distribution:")
            for type_name, count in type_counts.items():
                print(f"     {type_name}: {count}")
    
    def demonstrate_cognitive_cycle_integration(self):
        """Demonstrate integration with cognitive cycle"""
        print("\n" + "="*60)
        print("🔄 DEMONSTRATION 5: COGNITIVE CYCLE INTEGRATION")
        print("="*60)
        
        scenario = self.scenario_2
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        # Create sensory input
        sensory_input = SensoryInput(
            visual=scenario['visual_data'],
            auditory=scenario['audio_data']
        )
        
        print(f"\n🔄 Executing cognitive cycle with cross-domain integration...")
        
        # Execute multiple cognitive cycles
        actions = []
        rewards = [0.8, 0.6, 0.9, 0.7, 0.85]  # Varying rewards
        
        for i, reward in enumerate(rewards):
            print(f"\n   Cycle {i+1}: reward = {reward}")
            
            action = self.cognitive_system.cognitive_cycle(sensory_input, reward=reward)
            actions.append(action)
            
            # Check working memory for cross-domain information
            working_memory = self.cognitive_system.state.working_memory
            cross_domain_inferences = working_memory.get('cross_domain_inferences', [])
            
            print(f"      Action: {action.name if action else 'None'}")
            print(f"      Cross-domain inferences: {len(cross_domain_inferences)}")
            
            # Check sensory buffer for cross-modal results
            sensory_buffer = self.cognitive_system.state.sensory_buffer
            processing_info = sensory_buffer.get('processing_info', {})
            cross_modal_results = processing_info.get('cross_modal_results', {})
            
            correspondences = cross_modal_results.get('correspondences', [])
            print(f"      Cross-modal correspondences: {len(correspondences)}")
            
            # Update sensory input slightly for next cycle
            if i < len(rewards) - 1:
                sensory_input.visual += np.random.randn(*sensory_input.visual.shape) * 0.02
                sensory_input.auditory += np.random.randn(*sensory_input.auditory.shape) * 0.02
        
        print(f"\n📊 Cycle Integration Results:")
        print(f"   ✅ Successful cycles: {sum(1 for a in actions if a is not None)}/{len(actions)}")
        print(f"   💭 Total reward accumulated: {self.cognitive_system.state.total_reward:.3f}")
        print(f"   🧠 Emotional valence: {self.cognitive_system.state.emotional_valence:.3f}")
    
    def demonstrate_knowledge_validation(self):
        """Demonstrate knowledge consistency validation"""
        print("\n" + "="*60)
        print("✅ DEMONSTRATION 6: KNOWLEDGE CONSISTENCY VALIDATION")
        print("="*60)
        
        print("\n🔍 Validating cross-domain knowledge consistency...")
        
        # Validate consistency
        consistency_results = self.cognitive_system.validate_cross_domain_knowledge_consistency()
        
        print(f"\n📊 Consistency Validation Results:")
        
        # Framework consistency
        framework_consistency = consistency_results.get('framework_consistency', {})
        print(f"   Framework Consistency:")
        print(f"     Concept mappings: {framework_consistency.get('concept_mapping_consistency', 0):.3f}")
        print(f"     Domain adaptation: {framework_consistency.get('adaptation_accuracy', 0):.3f}")
        print(f"     Cross-modal coherence: {framework_consistency.get('cross_modal_coherence', 0):.3f}")
        print(f"     Overall: {framework_consistency.get('overall_consistency', 0):.3f}")
        
        # Reasoning consistency
        reasoning_consistency = consistency_results.get('reasoning_consistency', {})
        print(f"   Reasoning Consistency:")
        print(f"     Logical consistency: {reasoning_consistency.get('logical_consistency', 0):.3f}")
        print(f"     Temporal consistency: {reasoning_consistency.get('temporal_consistency', 0):.3f}")
        print(f"     Strength consistency: {reasoning_consistency.get('strength_consistency', 0):.3f}")
        print(f"     Overall: {reasoning_consistency.get('overall_consistency', 0):.3f}")
        
        # Combined score
        combined_score = consistency_results.get('combined_score', 0)
        print(f"   Combined Consistency Score: {combined_score:.3f}")
        
        # Recommendations
        recommendations = consistency_results.get('recommendations', [])
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    
    def show_integration_status(self):
        """Show comprehensive integration status"""
        print("\n" + "="*60)
        print("📈 INTEGRATION STATUS SUMMARY")
        print("="*60)
        
        status = self.cognitive_system.get_cross_domain_integration_status()
        
        # Framework status
        framework_status = status.get('framework_status', {})
        print(f"\n🔧 Framework Status:")
        print(f"   Active domains: {len(framework_status.get('active_domains', []))}")
        print(f"   Active modalities: {len(framework_status.get('active_modalities', []))}")
        print(f"   Concept mappings: {framework_status.get('total_concept_mappings', 0)}")
        print(f"   Abstract concepts: {framework_status.get('total_abstract_concepts', 0)}")
        print(f"   Domain adapters: {framework_status.get('total_domain_adapters', 0)}")
        print(f"   Cross-modal bindings: {framework_status.get('cross_modal_bindings', 0)}")
        
        # Reasoning statistics
        reasoning_stats = status.get('reasoning_statistics', {})
        print(f"\n🤔 Reasoning Statistics:")
        print(f"   Total inferences: {reasoning_stats.get('total_inferences', 0)}")
        print(f"   Average strength: {reasoning_stats.get('average_strength', 0):.3f}")
        
        reasoning_types = reasoning_stats.get('reasoning_types', {})
        if reasoning_types:
            print(f"   Reasoning types:")
            for reasoning_type, count in reasoning_types.items():
                print(f"     {reasoning_type}: {count}")
        
        # Overall health
        overall_health = status.get('overall_health', {})
        print(f"\n💚 Overall Health:")
        for component, health in overall_health.items():
            emoji = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(health, "⚪")
            print(f"   {component}: {emoji} {health}")
    
    def run_complete_demo(self):
        """Run the complete cross-domain integration demo"""
        print("🚀" + "="*59)
        print("  COGPRIME CROSS-DOMAIN INTEGRATION FRAMEWORK DEMO")
        print("🚀" + "="*59)
        
        print("\nThis demo showcases the cross-domain integration capabilities")
        print("of the CogPrime cognitive architecture, including:")
        print("• Unified representation across domains")
        print("• Cross-modal attention and integration")
        print("• Domain adaptation algorithms")
        print("• Cross-domain reasoning and inference")
        print("• Cognitive cycle integration")
        print("• Knowledge consistency validation")
        
        try:
            # Run demonstrations
            self.demonstrate_unified_representation()
            self.demonstrate_cross_modal_attention()
            self.demonstrate_domain_adaptation()
            self.demonstrate_cross_domain_reasoning()
            self.demonstrate_cognitive_cycle_integration()
            self.demonstrate_knowledge_validation()
            self.show_integration_status()
            
            print("\n" + "🎉"*60)
            print("  DEMO COMPLETED SUCCESSFULLY!")
            print("🎉"*60)
            print("\nThe Cross-Domain Integration Framework is functioning")
            print("and demonstrates successful integration across multiple")
            print("domains and modalities with consistent reasoning capabilities.")
            
        except Exception as e:
            print(f"\n❌ Demo encountered an error: {e}")
            print("This may indicate issues with the implementation that need attention.")
            return False
        
        return True


def run_cross_domain_demo():
    """Run the cross-domain integration demo"""
    demo = CrossDomainIntegrationDemo()
    return demo.run_complete_demo()


if __name__ == "__main__":
    # Run demo when script is executed directly
    run_cross_domain_demo()