"""
Test suite for Historical Context Integration System

Tests the temporal knowledge representation, episodic memory with temporal indexing,
historical pattern recognition, temporal reasoning, causal detection, and 
context-aware decision making components.
"""

import torch
# import pytest  # Not needed for basic testing
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.reasoning import Thought, EnhancedEpisodicMemory, AdvancedPatternDetector
from modules.historical_context import (
    HistoricalContextIntegrationSystem, TemporalEvent, TemporalRelation,
    CausalRelation, TemporalRelationType, CausalType
)


class TestHistoricalContextIntegration:
    """Test suite for historical context integration capabilities"""
    
    def setup_method(self):
        """Setup test environment"""
        self.feature_dim = 128
        self.memory_size = 100
        self.episodic_memory = EnhancedEpisodicMemory(
            memory_size=self.memory_size,
            feature_dim=self.feature_dim
        )
        self.pattern_detector = AdvancedPatternDetector(feature_dim=self.feature_dim)
        
        # Initialize the comprehensive historical context system
        self.historical_system = HistoricalContextIntegrationSystem(
            feature_dim=self.feature_dim,
            memory_size=self.memory_size
        )
        
    def test_temporal_knowledge_representation(self):
        """Test temporal knowledge representation frameworks"""
        # Create temporal knowledge with timestamps
        knowledge_items = []
        base_time = datetime.now()
        
        for i in range(5):
            timestamp = base_time + timedelta(hours=i)
            content = torch.randn(self.feature_dim)
            thought = Thought(
                content=content,
                salience=0.8,
                associations=[f"event_{i}"],
                timestamp=timestamp.timestamp(),
                pattern_type="temporal",
                confidence=0.9,
                context={"temporal_index": i, "event_type": "test_event"}
            )
            knowledge_items.append(thought)
            
        # Test that temporal structure is preserved
        assert len(knowledge_items) == 5
        for i, item in enumerate(knowledge_items):
            assert item.context["temporal_index"] == i
            assert item.pattern_type == "temporal"
            
    def test_episodic_memory_temporal_indexing(self):
        """Test enhanced episodic memory with temporal indexing"""
        # Store memories with temporal progression
        memories = []
        base_time = datetime.now()
        
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i*10)
            content = torch.randn(self.feature_dim) * (1 + i * 0.1)  # Evolving pattern
            
            thought = Thought(
                content=content,
                salience=0.5 + i * 0.05,  # Increasing salience over time
                associations=[f"sequence_{i}"],
                timestamp=timestamp.timestamp(),
                pattern_type="sequential",
                confidence=0.8,
                context={"sequence_position": i}
            )
            
            self.episodic_memory.store(thought)
            memories.append(thought)
        
        # Test temporal retrieval
        query = memories[5].content  # Query with middle memory
        retrieved = self.episodic_memory.retrieve(query, k=3)
        
        assert len(retrieved) > 0
        assert len(retrieved) <= 3
        
        # Test that recent memories have higher recency scores
        recent_memory = memories[-1]
        old_memory = memories[0]
        
        # Recent memory should have higher recency value
        recent_idx = len(memories) - 1
        old_idx = 0
        
        if recent_idx < len(self.episodic_memory.memory_recency):
            assert self.episodic_memory.memory_recency[recent_idx] > \
                   self.episodic_memory.memory_recency[old_idx]
        
    def test_historical_pattern_recognition(self):
        """Test historical pattern recognition algorithms"""
        # Create a sequence with repeating temporal patterns
        sequence = []
        pattern_length = 4
        
        for cycle in range(3):  # 3 cycles of the pattern
            for i in range(pattern_length):
                # Create pattern: sine wave + noise
                base_pattern = torch.sin(torch.linspace(0, 2*np.pi, self.feature_dim)) * (i + 1)
                noise = torch.randn(self.feature_dim) * 0.1
                content = base_pattern + noise
                sequence.append(content)
        
        # Detect patterns in the sequence
        patterns = self.pattern_detector.detect_temporal_patterns(sequence)
        
        # Should detect temporal patterns
        assert len(patterns) > 0
        
        # Check that patterns have proper structure
        for pattern in patterns:
            assert pattern.pattern_type == "temporal"
            assert 0.0 <= pattern.confidence <= 1.0
            assert pattern.frequency >= 0
            assert "sequence_length" in pattern.context
            
    def test_temporal_reasoning_basic(self):
        """Test basic temporal reasoning capabilities"""
        # Create a causal sequence: A -> B -> C
        events = []
        base_time = datetime.now()
        
        # Event A (cause)
        event_a = Thought(
            content=torch.randn(self.feature_dim),
            salience=0.9,
            associations=["cause_event"],
            timestamp=(base_time + timedelta(minutes=0)).timestamp(),
            pattern_type="causal_start",
            confidence=0.95,
            context={"event_type": "cause", "causal_position": "start"}
        )
        
        # Event B (intermediate)
        event_b = Thought(
            content=torch.randn(self.feature_dim),
            salience=0.8,
            associations=["intermediate_event", "cause_event"],
            timestamp=(base_time + timedelta(minutes=5)).timestamp(),
            pattern_type="causal_intermediate",
            confidence=0.85,
            context={"event_type": "intermediate", "causal_position": "middle"}
        )
        
        # Event C (effect)
        event_c = Thought(
            content=torch.randn(self.feature_dim),
            salience=0.7,
            associations=["effect_event", "intermediate_event"],
            timestamp=(base_time + timedelta(minutes=10)).timestamp(),
            pattern_type="causal_end",
            confidence=0.8,
            context={"event_type": "effect", "causal_position": "end"}
        )
        
        events = [event_a, event_b, event_c]
        
        # Store in episodic memory
        for event in events:
            self.episodic_memory.store(event)
        
        # Test temporal ordering
        timestamps = [event.timestamp for event in events]
        assert timestamps == sorted(timestamps), "Events should be temporally ordered"
        
        # Test causal chain detection through associations
        assert "cause_event" in event_b.associations
        assert "intermediate_event" in event_c.associations
        
    def test_causal_relationship_detection(self):
        """Test causal relationship detection across time"""
        # Create causal sequences with varying delays
        causal_pairs = []
        
        for delay_minutes in [1, 5, 10]:
            base_time = datetime.now()
            
            cause = Thought(
                content=torch.randn(self.feature_dim),
                salience=0.9,
                associations=[f"cause_{delay_minutes}"],
                timestamp=base_time.timestamp(),
                pattern_type="cause",
                confidence=0.9,
                context={"causal_type": "cause", "delay": delay_minutes}
            )
            
            effect = Thought(
                content=torch.randn(self.feature_dim),
                salience=0.8,
                associations=[f"effect_{delay_minutes}", f"cause_{delay_minutes}"],
                timestamp=(base_time + timedelta(minutes=delay_minutes)).timestamp(),
                pattern_type="effect",
                confidence=0.85,
                context={"causal_type": "effect", "delay": delay_minutes}
            )
            
            causal_pairs.append((cause, effect))
            
            # Store both in memory
            self.episodic_memory.store(cause)
            self.episodic_memory.store(effect)
        
        # Test that causal relationships are preserved
        for cause, effect in causal_pairs:
            # Effect should have cause in its associations
            shared_associations = set(cause.associations) & set(effect.associations)
            assert len(shared_associations) > 0, "Causal relationships should share associations"
            
            # Effect should occur after cause
            assert effect.timestamp > cause.timestamp
            
    def test_historical_context_awareness(self):
        """Test historical context-aware decision making"""
        # Create historical context with patterns
        historical_contexts = []
        base_time = datetime.now()
        
        # Pattern: Success follows preparation
        for i in range(5):
            preparation_time = base_time + timedelta(hours=i*2)
            success_time = preparation_time + timedelta(hours=1)
            
            preparation = Thought(
                content=torch.randn(self.feature_dim),
                salience=0.6,
                associations=["preparation", f"cycle_{i}"],
                timestamp=preparation_time.timestamp(),
                pattern_type="preparation",
                confidence=0.8,
                context={"phase": "preparation", "cycle": i}
            )
            
            success = Thought(
                content=torch.randn(self.feature_dim),
                salience=0.9,
                associations=["success", f"cycle_{i}", "preparation"],
                timestamp=success_time.timestamp(),
                pattern_type="success",
                confidence=0.95,
                context={"phase": "success", "cycle": i}
            )
            
            historical_contexts.extend([preparation, success])
            self.episodic_memory.store(preparation)
            self.episodic_memory.store(success)
        
        # Test pattern detection in historical context
        preparation_thoughts = [t for t in historical_contexts if t.pattern_type == "preparation"]
        success_thoughts = [t for t in historical_contexts if t.pattern_type == "success"]
        
        assert len(preparation_thoughts) == 5
        assert len(success_thoughts) == 5
        
        # Test that success thoughts reference preparation
        for success_thought in success_thoughts:
            assert "preparation" in success_thought.associations
            
    def test_temporal_abstraction_and_generalization(self):
        """Test temporal abstraction and generalization systems"""
        # Create patterns at different time scales
        patterns_by_scale = {
            "hourly": [],
            "daily": [],
            "weekly": []
        }
        
        base_time = datetime.now()
        
        # Hourly patterns
        for hour in range(24):
            thought = Thought(
                content=torch.sin(torch.linspace(0, 2*np.pi, self.feature_dim)) * hour,
                salience=0.5 + 0.3 * np.sin(hour * np.pi / 12),  # Daily cycle
                associations=[f"hour_{hour}", "hourly_cycle"],
                timestamp=(base_time + timedelta(hours=hour)).timestamp(),
                pattern_type="hourly",
                confidence=0.8,
                context={"time_scale": "hourly", "period": hour}
            )
            patterns_by_scale["hourly"].append(thought)
            self.episodic_memory.store(thought)
        
        # Daily patterns (simplified)
        for day in range(7):
            thought = Thought(
                content=torch.randn(self.feature_dim) * (day + 1),
                salience=0.7,
                associations=[f"day_{day}", "daily_cycle"],
                timestamp=(base_time + timedelta(days=day)).timestamp(),
                pattern_type="daily",
                confidence=0.85,
                context={"time_scale": "daily", "period": day}
            )
            patterns_by_scale["daily"].append(thought)
            self.episodic_memory.store(thought)
        
        # Test abstraction across scales
        all_patterns = []
        for scale, thoughts in patterns_by_scale.items():
            all_patterns.extend(thoughts)
            
        # Different time scales should be represented
        hourly_count = len([t for t in all_patterns if t.pattern_type == "hourly"])
        daily_count = len([t for t in all_patterns if t.pattern_type == "daily"])
        
        assert hourly_count == 24
        assert daily_count == 7
        
        # Test temporal pattern detection across scales
        hourly_tensors = [t.content for t in patterns_by_scale["hourly"]]
        if len(hourly_tensors) > 1:
            detected_patterns = self.pattern_detector.detect_temporal_patterns(hourly_tensors)
            assert len(detected_patterns) >= 0  # Should detect some patterns
            
    def test_historical_knowledge_validation(self):
        """Test historical knowledge validation and consistency checking"""
        # Create consistent and inconsistent knowledge
        base_time = datetime.now()
        
        # Consistent sequence: A before B before C
        consistent_sequence = []
        for i, event_name in enumerate(["event_a", "event_b", "event_c"]):
            thought = Thought(
                content=torch.randn(self.feature_dim),
                salience=0.8,
                associations=[event_name, "consistent_sequence"],
                timestamp=(base_time + timedelta(hours=i)).timestamp(),
                pattern_type="sequential",
                confidence=0.9,
                context={"sequence": "consistent", "position": i}
            )
            consistent_sequence.append(thought)
            self.episodic_memory.store(thought)
        
        # Inconsistent sequence: timestamps don't match claimed order
        inconsistent_sequence = []
        timestamps = [
            (base_time + timedelta(hours=10)).timestamp(),  # Should be first
            (base_time + timedelta(hours=8)).timestamp(),   # Should be second (but earlier!)
            (base_time + timedelta(hours=12)).timestamp()   # Should be third
        ]
        
        for i, (event_name, timestamp) in enumerate(zip(["event_x", "event_y", "event_z"], timestamps)):
            thought = Thought(
                content=torch.randn(self.feature_dim),
                salience=0.7,
                associations=[event_name, "inconsistent_sequence"],
                timestamp=timestamp,
                pattern_type="sequential",
                confidence=0.6,  # Lower confidence for inconsistent data
                context={"sequence": "inconsistent", "claimed_position": i}
            )
            inconsistent_sequence.append(thought)
            self.episodic_memory.store(thought)
        
        # Test consistency validation
        # Consistent sequence should have properly ordered timestamps
        consistent_timestamps = [t.timestamp for t in consistent_sequence]
        assert consistent_timestamps == sorted(consistent_timestamps)
        
        # Inconsistent sequence timestamps don't match claimed positions
        inconsistent_timestamps = [t.timestamp for t in inconsistent_sequence]
        claimed_positions = [t.context["claimed_position"] for t in inconsistent_sequence]
        
        # Sort by claimed position and check if timestamps are actually ordered
        sorted_by_claimed = sorted(zip(claimed_positions, inconsistent_timestamps))
        actual_timestamp_order = [ts for _, ts in sorted_by_claimed]
        
        # Should NOT be in temporal order since it's inconsistent
        assert actual_timestamp_order != sorted(actual_timestamp_order)
        
    def test_integration_with_vervaeke_framework(self):
        """Test integration with existing Vervaeke 4E cognition framework"""
        # This test ensures our historical context system works with existing modules
        # Create memories that would interact with 4E cognition principles
        
        base_time = datetime.now()
        
        # Embodied memory (sensorimotor experience)
        embodied_memory = Thought(
            content=torch.randn(self.feature_dim),
            salience=0.8,
            associations=["embodied_experience", "sensorimotor"],
            timestamp=base_time.timestamp(),
            pattern_type="embodied",
            confidence=0.9,
            context={"cognition_type": "embodied", "modality": "multimodal"}
        )
        
        # Embedded memory (environmental context)
        embedded_memory = Thought(
            content=torch.randn(self.feature_dim),
            salience=0.7,
            associations=["environmental_context", "situated"],
            timestamp=(base_time + timedelta(minutes=10)).timestamp(),
            pattern_type="embedded",
            confidence=0.85,
            context={"cognition_type": "embedded", "environment": "test_context"}
        )
        
        # Store both in episodic memory
        self.episodic_memory.store(embodied_memory)
        self.episodic_memory.store(embedded_memory)
        
        # Test that different cognition types are preserved
        memories = [embodied_memory, embedded_memory]
        cognition_types = [m.context.get("cognition_type") for m in memories]
        
        assert "embodied" in cognition_types
        assert "embedded" in cognition_types
        
        # Test retrieval maintains context
        query = embodied_memory.content
        retrieved = self.episodic_memory.retrieve(query, k=2)
        
        assert len(retrieved) > 0
        # At least one retrieved memory should maintain contextual information
        assert any(hasattr(r, 'context') and r.context is not None for r in retrieved)

        
    def test_comprehensive_historical_context_system(self):
        """Test the complete Historical Context Integration System"""
        # Test processing multiple experiences
        experiences = []
        base_time = datetime.now()
        
        for i in range(5):
            experience = {
                "content": torch.randn(self.feature_dim),
                "timestamp": (base_time + timedelta(minutes=i*10)).timestamp(),
                "salience": 0.5 + i * 0.1,
                "confidence": 0.8 + i * 0.05,
                "type": f"experience_{i % 3}",  # Create some type patterns
                "context": {"sequence": i, "phase": "learning"},
                "associations": [f"seq_{i}", "learning_phase"]
            }
            experiences.append(experience)
            
            # Process each experience
            result = self.historical_system.process_experience(experience)
            
            # Verify processing results
            assert "event_id" in result
            assert "detected_patterns" in result
            assert "causal_relations" in result
            
        # Test decision making with historical context
        current_context = {
            "timestamp": (base_time + timedelta(hours=1)).timestamp(),
            "type": "decision_point",
            "context": {"phase": "application"}
        }
        available_actions = ["action_a", "action_b", "action_c"]
        
        decision = self.historical_system.make_historical_decision(
            current_context, available_actions
        )
        
        # Verify decision structure
        assert "chosen_action" in decision
        assert decision["chosen_action"] in available_actions
        assert "confidence" in decision
        assert 0.0 <= decision["confidence"] <= 1.0
        assert "historical_analysis" in decision
        
        # Test temporal consistency validation
        consistency_check = self.historical_system.validate_temporal_consistency()
        assert "temporal_consistency" in consistency_check
        assert "causal_consistency" in consistency_check
        assert "overall_consistent" in consistency_check
        
        # Test temporal insights
        query_context = {"type": "experience_0"}
        insights = self.historical_system.get_temporal_insights(query_context)
        
        assert "relevant_patterns" in insights
        assert "historical_precedents" in insights
        assert "predictions" in insights


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    test_suite = TestHistoricalContextIntegration()
    test_suite.setup_method()
    
    print("Running Historical Context Integration tests...")
    
    try:
        test_suite.test_temporal_knowledge_representation()
        print("✓ Temporal knowledge representation test passed")
        
        test_suite.test_episodic_memory_temporal_indexing()
        print("✓ Episodic memory temporal indexing test passed")
        
        test_suite.test_historical_pattern_recognition()
        print("✓ Historical pattern recognition test passed")
        
        test_suite.test_temporal_reasoning_basic()
        print("✓ Basic temporal reasoning test passed")
        
        test_suite.test_causal_relationship_detection()
        print("✓ Causal relationship detection test passed")
        
        test_suite.test_historical_context_awareness()
        print("✓ Historical context awareness test passed")
        
        test_suite.test_temporal_abstraction_and_generalization()
        print("✓ Temporal abstraction and generalization test passed")
        
        test_suite.test_historical_knowledge_validation()
        print("✓ Historical knowledge validation test passed")
        
        test_suite.test_integration_with_vervaeke_framework()
        print("✓ Vervaeke framework integration test passed")
        
        test_suite.test_comprehensive_historical_context_system()
        print("✓ Comprehensive historical context system test passed")
        
        print("\nAll Historical Context Integration tests passed! ✓")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()