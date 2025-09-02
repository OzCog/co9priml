"""
Basic tests for enhanced meaning-making systems without external dependencies.

Tests core functionality that doesn't require numpy or torch.
"""

import unittest
import sys
import os

# Simple mock for numpy arrays
class MockArray:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data] * 128
        
    def __len__(self):
        return len(self.data)
    
    def dot(self, other):
        if len(self.data) != len(other.data):
            return 0.0
        return sum(a * b for a, b in zip(self.data, other.data))

# Add mock numpy to sys.modules
class MockNumpy:
    def array(self, data):
        return MockArray(data)
    
    def random(self):
        class Random:
            def randn(self, size):
                return MockArray([0.1] * size)
            def random(self):
                return 0.5
            def seed(self, seed):
                pass
        return Random()
    
    def mean(self, data):
        if hasattr(data, 'data'):
            data = data.data
        return sum(data) / len(data) if data else 0.0
    
    def var(self, data):
        if hasattr(data, 'data'):
            data = data.data
        if not data:
            return 0.0
        mean_val = self.mean(data)
        return sum((x - mean_val) ** 2 for x in data) / len(data)
    
    def dot(self, a, b):
        if hasattr(a, 'dot'):
            return a.dot(b)
        return sum(x * y for x, y in zip(a, b))
    
    def linalg(self):
        class Linalg:
            def norm(self, data):
                if hasattr(data, 'data'):
                    data = data.data
                return (sum(x * x for x in data)) ** 0.5 if data else 1.0
        return Linalg()

sys.modules['numpy'] = MockNumpy()
import numpy as np

# Import the enhanced meaning-making components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from cognitive_science.meaning_making import (
    MeaningMaker, MeaningContext, MeaningStructure, SemanticNode,
    MeaningLevel, EmotionalValence, CommunicationMaxim,
    NeuralSemanticBridge, EmotionalCognitiveSynthesizer,
    CulturalContextProcessor
)
from core.relevance_core import RelevanceCore


class TestBasicMeaningMaking(unittest.TestCase):
    """Basic tests for meaning-making functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.relevance_core = RelevanceCore()
        self.meaning_maker = MeaningMaker(self.relevance_core, embedding_dim=8)
        
    def test_semantic_node_creation(self):
        """Test creation of semantic nodes"""
        node = SemanticNode(
            concept="test_concept",
            emotional_valence=EmotionalValence.POSITIVE,
            activation_strength=0.7
        )
        
        self.assertEqual(node.concept, "test_concept")
        self.assertEqual(node.emotional_valence, EmotionalValence.POSITIVE)
        self.assertEqual(node.activation_strength, 0.7)
    
    def test_meaning_structure_validation(self):
        """Test meaning structure validation"""
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
        
        # Test invalid structure
        invalid_relations = {"node1": {"nonexistent": 0.8}}
        invalid_structure = MeaningStructure(core_nodes=nodes, relations=invalid_relations)
        self.assertFalse(invalid_structure.validate_consistency())
    
    def test_cultural_context_processor(self):
        """Test cultural context processing"""
        processor = CulturalContextProcessor()
        
        # Add cultural norm
        processor.add_cultural_norm("collectivist", "individual_achievement", 0.7)
        processor.add_cultural_norm("collectivist", "group_harmony", 1.3)
        
        content = {"individual_achievement": 0.8, "group_harmony": 0.6}
        processed = processor.process_cultural_meaning(content, "collectivist")
        
        # Check cultural modifications were applied
        self.assertLess(processed["individual_achievement"], content["individual_achievement"])
        self.assertGreater(processed["group_harmony"], content["group_harmony"])
    
    def test_emotional_cognitive_synthesizer(self):
        """Test emotional-cognitive synthesis"""
        synthesizer = EmotionalCognitiveSynthesizer()
        
        # Update emotion association
        synthesizer.update_emotion_associations("happiness", EmotionalValence.POSITIVE)
        
        cognitive_content = {"learning": 0.8, "happiness": 0.6}
        emotional_context = {"learning": 0.9, "joy": 0.7}
        
        synthesized = synthesizer.synthesize_meaning(cognitive_content, emotional_context)
        
        self.assertIsInstance(synthesized, dict)
        self.assertIn("learning", synthesized)
        # Learning should be enhanced by emotional context
        self.assertGreater(synthesized["learning"], cognitive_content["learning"])
    
    def test_symbolic_subsymbolic_bridge(self):
        """Test symbolic-subsymbolic bridge"""
        bridge = NeuralSemanticBridge(embedding_dim=8)
        
        symbolic_content = {"happiness": 0.8, "learning": 0.6}
        
        # Convert to subsymbolic
        subsymbolic = bridge.symbolic_to_subsymbolic(symbolic_content)
        self.assertIsNotNone(subsymbolic)
        self.assertEqual(len(subsymbolic), 8)
        
        # Convert back to symbolic
        reconstructed = bridge.subsymbolic_to_symbolic(subsymbolic)
        self.assertIsInstance(reconstructed, dict)
        
        # Should have some overlap with original
        original_concepts = set(symbolic_content.keys())
        reconstructed_concepts = set(reconstructed.keys())
        # At least one concept should be reconstructed
        self.assertGreaterEqual(len(reconstructed_concepts), 1)
    
    def test_meaning_construction(self):
        """Test contextual meaning construction"""
        experience = {
            "concept_a": 0.8,
            "concept_b": 0.6,
            "relationship": "positive"
        }
        
        context = {
            "domain": "education",
            "cultural": {"type": "academic"}
        }
        
        meaning_structure = self.meaning_maker.construct_contextual_meaning(
            experience, context, cultural_context="academic"
        )
        
        self.assertIsInstance(meaning_structure, MeaningStructure)
        self.assertGreater(len(meaning_structure.core_nodes), 0)
        self.assertGreaterEqual(meaning_structure.coherence_score, 0.0)
        self.assertLessEqual(meaning_structure.coherence_score, 1.0)
    
    def test_meaning_validation(self):
        """Test meaning validation and consistency"""
        nodes = [
            SemanticNode(concept="concept1", activation_strength=0.8),
            SemanticNode(concept="concept2", activation_strength=0.7)
        ]
        relations = {"concept1": {"concept2": 0.9}}
        
        structure = MeaningStructure(core_nodes=nodes, relations=relations, coherence_score=0.8)
        
        is_valid, issues = self.meaning_maker.validate_meaning_consistency(structure)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_adaptive_feedback(self):
        """Test adaptive refinement from feedback"""
        initial_threshold = self.meaning_maker.validation_threshold
        
        feedback = {
            "coherence_target": 0.8,
            "emotion_corrections": {
                "test_concept": EmotionalValence.POSITIVE
            }
        }
        
        self.meaning_maker.adapt_meaning_from_feedback(feedback)
        
        # Check adaptation occurred
        self.assertGreater(len(self.meaning_maker.feedback_history), 0)
        self.assertIn("test_concept", 
                     self.meaning_maker.emotional_synthesizer.emotion_concept_map)
    
    def test_multi_level_integration(self):
        """Test multi-level meaning integration"""
        structure1 = self.meaning_maker.construct_contextual_meaning(
            {"concept_x": 0.9, "concept_y": 0.7}
        )
        
        structure2 = self.meaning_maker.construct_contextual_meaning(
            {"concept_y": 0.8, "concept_z": 0.6}
        )
        
        integrated = self.meaning_maker.integrate_multi_level_meaning([structure1, structure2])
        
        self.assertIsInstance(integrated, MeaningStructure)
        # Should contain concepts from both structures
        concepts = [node.concept for node in integrated.core_nodes]
        self.assertTrue(any("concept_" in concept for concept in concepts))
    
    def test_communication_maxims(self):
        """Test communication following maxims"""
        message = "The system is learning"
        context = {"audience": "technical"}
        
        refined_message, confidence = self.meaning_maker.communicate(message, context)
        
        self.assertIsInstance(refined_message, str)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestIntegrationBasic(unittest.TestCase):
    """Basic integration tests"""
    
    def setUp(self):
        self.relevance_core = RelevanceCore()
        self.meaning_maker = MeaningMaker(self.relevance_core, embedding_dim=16)
    
    def test_complete_pipeline(self):
        """Test a complete meaning-making pipeline"""
        # Complex experience
        experience = {
            "learning_progress": 0.8,
            "emotional_state": "positive",
            "social_context": "collaborative",
            "achievement": "milestone"
        }
        
        context = {
            "cultural": {"type": "collaborative"},
            "temporal": {"phase": "learning"}
        }
        
        # Construct meaning
        meaning_structure = self.meaning_maker.construct_contextual_meaning(
            experience, context, cultural_context="collaborative"
        )
        
        # Validate
        is_valid, issues = self.meaning_maker.validate_meaning_consistency(meaning_structure)
        
        # Bridge representations
        symbolic_content = {node.concept: node.activation_strength 
                           for node in meaning_structure.core_nodes}
        subsymbolic, reconstructed = self.meaning_maker.bridge_symbolic_subsymbolic(symbolic_content)
        
        # Verify pipeline results
        self.assertIsNotNone(meaning_structure)
        self.assertGreater(len(meaning_structure.core_nodes), 0)
        self.assertIsNotNone(subsymbolic)
        self.assertIsInstance(reconstructed, dict)


if __name__ == "__main__":
    # Run the test suite
    unittest.main(verbosity=2)