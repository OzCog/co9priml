"""
Comprehensive tests for enhanced meaning-making systems.

Tests the full range of meaning-making capabilities including:
- Semantic representation and processing
- Contextual meaning construction
- Multi-level meaning integration
- Symbolic-subsymbolic bridges
- Emotional-cognitive synthesis
- Cultural context understanding
- Meaning validation and consistency
- Adaptive refinement
"""

import unittest
import numpy as np
from typing import Dict, List

# Import the enhanced meaning-making components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cognitive_science.meaning_making import (
    MeaningMaker, MeaningContext, MeaningStructure, SemanticNode,
    MeaningLevel, EmotionalValence, CommunicationMaxim,
    NeuralSemanticBridge, EmotionalCognitiveSynthesizer,
    CulturalContextProcessor
)
from core.relevance_core import RelevanceCore


class TestEnhancedMeaningMaking(unittest.TestCase):
    """Test suite for enhanced meaning-making capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.relevance_core = RelevanceCore()
        self.meaning_maker = MeaningMaker(self.relevance_core, embedding_dim=128)
        
        # Sample experience data for testing
        self.sample_experience = {
            "concept_a": 0.8,
            "concept_b": 0.6,
            "concept_c": 0.4,
            "relationship": "positive",
            "context": "learning"
        }
        
        self.sample_context = {
            "domain": "education",
            "cultural": {"context": "western_academic"},
            "temporal": {"time": "present"},
            "emotional": {"valence": "positive"}
        }
    
    def test_semantic_node_creation(self):
        """Test creation and properties of semantic nodes"""
        node = SemanticNode(
            concept="test_concept",
            embedding=np.random.randn(128),
            emotional_valence=EmotionalValence.POSITIVE,
            activation_strength=0.7
        )
        
        self.assertEqual(node.concept, "test_concept")
        self.assertEqual(node.emotional_valence, EmotionalValence.POSITIVE)
        self.assertEqual(node.activation_strength, 0.7)
        self.assertIsNotNone(node.embedding)
        self.assertEqual(len(node.embedding), 128)
    
    def test_meaning_structure_validation(self):
        """Test meaning structure validation"""
        # Create valid structure
        nodes = [
            SemanticNode(concept="node1", activation_strength=0.5),
            SemanticNode(concept="node2", activation_strength=0.7)
        ]
        relations = {
            "node1": {"node2": 0.8},
            "node2": {"node1": 0.6}
        }
        
        structure = MeaningStructure(core_nodes=nodes, relations=relations)
        self.assertTrue(structure.validate_consistency())
        
        # Create invalid structure (relation to non-existent node)
        invalid_relations = {
            "node1": {"nonexistent": 0.8}
        }
        invalid_structure = MeaningStructure(core_nodes=nodes, relations=invalid_relations)
        self.assertFalse(invalid_structure.validate_consistency())
    
    def test_contextual_meaning_construction(self):
        """Test contextual meaning construction from experience"""
        meaning_structure = self.meaning_maker.construct_contextual_meaning(
            self.sample_experience, 
            self.sample_context,
            cultural_context="western_academic"
        )
        
        self.assertIsInstance(meaning_structure, MeaningStructure)
        self.assertGreater(len(meaning_structure.core_nodes), 0)
        self.assertGreaterEqual(meaning_structure.coherence_score, 0.0)
        self.assertLessEqual(meaning_structure.coherence_score, 1.0)
    
    def test_multi_level_meaning_integration(self):
        """Test integration of multiple meaning structures"""
        # Create multiple meaning structures
        structure1 = self.meaning_maker.construct_contextual_meaning(
            {"concept_x": 0.9, "concept_y": 0.7},
            {"domain": "science"}
        )
        
        structure2 = self.meaning_maker.construct_contextual_meaning(
            {"concept_y": 0.8, "concept_z": 0.6},
            {"domain": "philosophy"}
        )
        
        # Integrate them
        integrated = self.meaning_maker.integrate_multi_level_meaning([structure1, structure2])
        
        self.assertIsInstance(integrated, MeaningStructure)
        # Should contain concepts from both structures
        concepts = [node.concept for node in integrated.core_nodes]
        self.assertIn("concept_y", concepts)  # Common concept
        
        # Should have relations
        self.assertGreater(len(integrated.relations), 0)
    
    def test_symbolic_subsymbolic_bridge(self):
        """Test bridging between symbolic and subsymbolic representations"""
        symbolic_content = {"happiness": 0.8, "learning": 0.6, "growth": 0.7}
        
        subsymbolic, reconstructed = self.meaning_maker.bridge_symbolic_subsymbolic(symbolic_content)
        
        # Check subsymbolic representation
        self.assertIsInstance(subsymbolic, np.ndarray)
        self.assertEqual(len(subsymbolic), 128)
        
        # Check reconstruction
        self.assertIsInstance(reconstructed, dict)
        # Should contain some of the original concepts
        original_concepts = set(symbolic_content.keys())
        reconstructed_concepts = set(reconstructed.keys())
        overlap = len(original_concepts.intersection(reconstructed_concepts))
        self.assertGreater(overlap, 0)
    
    def test_emotional_cognitive_synthesis(self):
        """Test synthesis of emotional and cognitive meaning aspects"""
        cognitive_content = {"learning": 0.8, "challenge": 0.6}
        emotional_context = {"learning": 0.9, "joy": 0.7, "challenge": 0.4}
        
        synthesized = self.meaning_maker.synthesize_emotional_cognitive_meaning(
            cognitive_content, emotional_context
        )
        
        self.assertIsInstance(synthesized, dict)
        self.assertIn("learning", synthesized)
        # Learning should be enhanced by positive emotional association
        self.assertGreater(synthesized["learning"], cognitive_content["learning"])
    
    def test_cultural_context_processing(self):
        """Test cultural context processing"""
        processor = CulturalContextProcessor()
        
        # Add cultural norm
        processor.add_cultural_norm("collectivist", "individual_achievement", 0.7)
        processor.add_cultural_norm("collectivist", "group_harmony", 1.3)
        
        content = {"individual_achievement": 0.8, "group_harmony": 0.6}
        processed = processor.process_cultural_meaning(content, "collectivist")
        
        # Individual achievement should be reduced, group harmony enhanced
        self.assertLess(processed["individual_achievement"], content["individual_achievement"])
        self.assertGreater(processed["group_harmony"], content["group_harmony"])
    
    def test_meaning_validation_and_consistency(self):
        """Test meaning validation and consistency checking"""
        # Create a coherent meaning structure
        nodes = [
            SemanticNode(concept="concept1", activation_strength=0.8, 
                        emotional_valence=EmotionalValence.POSITIVE),
            SemanticNode(concept="concept2", activation_strength=0.7,
                        emotional_valence=EmotionalValence.POSITIVE)
        ]
        relations = {"concept1": {"concept2": 0.9}}
        
        structure = MeaningStructure(core_nodes=nodes, relations=relations, coherence_score=0.8)
        
        is_valid, issues = self.meaning_maker.validate_meaning_consistency(structure)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Create a low-coherence structure
        low_coherence_structure = MeaningStructure(
            core_nodes=nodes, relations=relations, coherence_score=0.3
        )
        
        is_valid, issues = self.meaning_maker.validate_meaning_consistency(low_coherence_structure)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_adaptive_refinement_from_feedback(self):
        """Test adaptive refinement based on feedback"""
        initial_threshold = self.meaning_maker.validation_threshold
        
        # Provide feedback to increase coherence target
        feedback = {
            "coherence_target": 0.8,
            "emotion_corrections": {
                "test_concept": EmotionalValence.POSITIVE
            },
            "cultural_corrections": {
                "test_context": {"test_concept": 1.2}
            }
        }
        
        self.meaning_maker.adapt_meaning_from_feedback(feedback)
        
        # Check that threshold was adapted
        self.assertNotEqual(self.meaning_maker.validation_threshold, initial_threshold)
        
        # Check that feedback was recorded
        self.assertGreater(len(self.meaning_maker.feedback_history), 0)
        
        # Check emotional associations were updated
        self.assertIn("test_concept", self.meaning_maker.emotional_synthesizer.emotion_concept_map)
        
        # Check cultural norms were updated
        self.assertIn("test_context", self.meaning_maker.cultural_processor.cultural_norms)
    
    def test_communication_maxims(self):
        """Test communication following Grice's maxims"""
        message = "The system is learning effectively"
        context = {"audience": "technical", "urgency": "low"}
        
        refined_message, confidence = self.meaning_maker.communicate(message, context)
        
        self.assertIsInstance(refined_message, str)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIn("learning", refined_message.lower())
    
    def test_meaning_cultivation_integration(self):
        """Test the original meaning cultivation with enhanced features"""
        initial_context = self.meaning_maker.current_context
        
        # Cultivate meaning from experience
        updated_context = self.meaning_maker.cultivate_meaning(
            self.sample_experience, self.sample_context
        )
        
        self.assertIsInstance(updated_context, MeaningContext)
        # Should have updated semantic network
        self.assertGreaterEqual(len(updated_context.semantic_network), 0)
    
    def test_neural_semantic_bridge_fidelity(self):
        """Test fidelity of neural semantic bridge"""
        bridge = NeuralSemanticBridge(embedding_dim=64)
        
        # Test multiple conversions
        symbolic1 = {"concept_a": 0.8, "concept_b": 0.6}
        symbolic2 = {"concept_a": 0.7, "concept_c": 0.9}
        
        # Convert to subsymbolic
        sub1 = bridge.symbolic_to_subsymbolic(symbolic1)
        sub2 = bridge.symbolic_to_subsymbolic(symbolic2)
        
        # Convert back to symbolic
        reconstructed1 = bridge.subsymbolic_to_symbolic(sub1)
        reconstructed2 = bridge.subsymbolic_to_symbolic(sub2)
        
        # Check that similar symbolic content produces similar subsymbolic
        similarity = np.dot(sub1, sub2) / (np.linalg.norm(sub1) * np.linalg.norm(sub2))
        
        # Both contain concept_a, so should have some similarity
        self.assertGreater(similarity, 0.1)
        
        # Check reconstruction maintains key concepts
        self.assertTrue("concept_a" in reconstructed1 or "concept_b" in reconstructed1)
        self.assertTrue("concept_a" in reconstructed2 or "concept_c" in reconstructed2)


class TestMeaningMakerIntegration(unittest.TestCase):
    """Integration tests for the complete meaning-making system"""
    
    def setUp(self):
        self.relevance_core = RelevanceCore()
        self.meaning_maker = MeaningMaker(self.relevance_core, embedding_dim=256)
    
    def test_complete_meaning_making_pipeline(self):
        """Test the complete meaning-making pipeline from experience to validation"""
        # Step 1: Input complex experience
        complex_experience = {
            "learning_progress": 0.8,
            "emotional_state": "excited",
            "social_context": "collaborative",
            "challenge_level": 0.7,
            "achievement": "milestone_reached",
            "reflection": "growth_mindset"
        }
        
        context = {
            "cultural": {"type": "growth_oriented"},
            "temporal": {"phase": "learning"},
            "social": {"group_dynamics": "positive"}
        }
        
        # Step 2: Construct contextual meaning
        meaning_structure = self.meaning_maker.construct_contextual_meaning(
            complex_experience, context, cultural_context="growth_oriented"
        )
        
        # Step 3: Validate meaning
        is_valid, issues = self.meaning_maker.validate_meaning_consistency(meaning_structure)
        
        # Step 4: Bridge to subsymbolic
        symbolic_content = {node.concept: node.activation_strength 
                           for node in meaning_structure.core_nodes}
        subsymbolic, reconstructed = self.meaning_maker.bridge_symbolic_subsymbolic(symbolic_content)
        
        # Step 5: Synthesize with emotional context
        emotional_context = {"excitement": 0.9, "satisfaction": 0.8}
        synthesized = self.meaning_maker.synthesize_emotional_cognitive_meaning(
            symbolic_content, emotional_context
        )
        
        # Assertions
        self.assertIsNotNone(meaning_structure)
        self.assertGreater(meaning_structure.coherence_score, 0.0)
        self.assertIsInstance(subsymbolic, np.ndarray)
        self.assertIsInstance(synthesized, dict)
        
        # The pipeline should produce meaningful results
        self.assertGreater(len(meaning_structure.core_nodes), 3)
        self.assertGreater(len(synthesized), 3)
    
    def test_adaptive_learning_scenario(self):
        """Test meaning-making adaptation through feedback over time"""
        experiences = [
            {"success": 0.9, "effort": 0.8, "learning": 0.7},
            {"success": 0.3, "effort": 0.9, "learning": 0.8},  # Failure but learning
            {"success": 0.8, "effort": 0.6, "learning": 0.9},  # Easy success
        ]
        
        coherence_scores = []
        
        for i, experience in enumerate(experiences):
            # Construct meaning
            structure = self.meaning_maker.construct_contextual_meaning(experience)
            coherence_scores.append(structure.coherence_score)
            
            # Provide feedback based on the experience
            if i == 1:  # After failure experience
                feedback = {
                    "coherence_target": 0.9,  # Expect higher coherence from failure learning
                    "emotion_corrections": {
                        "effort": EmotionalValence.POSITIVE,  # Reframe effort positively
                        "learning": EmotionalValence.POSITIVE
                    }
                }
                self.meaning_maker.adapt_meaning_from_feedback(feedback)
        
        # The system should adapt and potentially improve meaning construction
        self.assertEqual(len(coherence_scores), 3)
        self.assertTrue(all(score >= 0.0 for score in coherence_scores))
        
        # Should have meaningful feedback history
        self.assertGreater(len(self.meaning_maker.feedback_history), 0)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)