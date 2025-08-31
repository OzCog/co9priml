#!/usr/bin/env python3
"""
Historical Context Integration System Demo

This demo showcases the comprehensive temporal knowledge representation,
reasoning, and decision-making capabilities of the Historical Context
Integration System.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modules.historical_context import HistoricalContextIntegrationSystem
from modules.reasoning import Thought


def print_separator(title):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def demo_temporal_knowledge_representation():
    """Demonstrate temporal knowledge representation"""
    print_separator("TEMPORAL KNOWLEDGE REPRESENTATION")
    
    # Initialize the system
    system = HistoricalContextIntegrationSystem(feature_dim=256, memory_size=200)
    
    # Create a sequence of learning experiences
    print("Creating a sequence of learning experiences...")
    base_time = datetime.now()
    
    learning_phases = [
        ("observation", "Observing the environment"),
        ("exploration", "Exploring possible actions"),
        ("practice", "Practicing learned behaviors"),
        ("mastery", "Achieving mastery"),
        ("teaching", "Teaching others")
    ]
    
    for i, (phase, description) in enumerate(learning_phases):
        experience = {
            "content": torch.randn(256) * (i + 1),  # Evolving complexity
            "timestamp": (base_time + timedelta(hours=i*2)).timestamp(),
            "salience": 0.6 + i * 0.08,
            "confidence": 0.7 + i * 0.06,
            "type": "learning_phase",
            "context": {
                "phase": phase,
                "description": description,
                "complexity": i + 1,
                "outcome": "success" if i > 1 else "in_progress"
            },
            "associations": ["learning", phase, f"step_{i}"]
        }
        
        result = system.process_experience(experience)
        print(f"  {i+1}. {phase.capitalize()}: {description}")
        print(f"     - Event ID: {result['event_id']}")
        print(f"     - Patterns detected: {len(result['detected_patterns'])}")
        print(f"     - Causal relations: {len(result['causal_relations'])}")
    
    return system


def demo_causal_relationship_detection(system):
    """Demonstrate causal relationship detection"""
    print_separator("CAUSAL RELATIONSHIP DETECTION")
    
    # Create causal scenarios
    base_time = datetime.now() + timedelta(hours=12)
    
    causal_scenarios = [
        {
            "cause": {"type": "preparation", "description": "Thorough preparation"},
            "effect": {"type": "success", "description": "Successful performance"},
            "delay": 30  # minutes
        },
        {
            "cause": {"type": "practice", "description": "Regular practice"},
            "effect": {"type": "improvement", "description": "Skill improvement"},
            "delay": 60
        },
        {
            "cause": {"type": "fatigue", "description": "Mental fatigue"},
            "effect": {"type": "error", "description": "Performance error"},
            "delay": 5
        }
    ]
    
    print("Processing causal scenarios...")
    
    for i, scenario in enumerate(causal_scenarios):
        # Process cause
        cause_exp = {
            "content": torch.randn(256),
            "timestamp": (base_time + timedelta(hours=i*2)).timestamp(),
            "salience": 0.8,
            "confidence": 0.9,
            "type": scenario["cause"]["type"],
            "context": {
                "description": scenario["cause"]["description"],
                "scenario": f"causal_{i}",
                "role": "cause"
            },
            "associations": ["causal_chain", f"scenario_{i}"]
        }
        
        # Process effect
        effect_exp = {
            "content": torch.randn(256),
            "timestamp": (base_time + timedelta(hours=i*2, minutes=scenario["delay"])).timestamp(),
            "salience": 0.9,
            "confidence": 0.85,
            "type": scenario["effect"]["type"],
            "context": {
                "description": scenario["effect"]["description"],
                "scenario": f"causal_{i}",
                "role": "effect",
                "outcome": "success" if "success" in scenario["effect"]["type"] else "neutral"
            },
            "associations": ["causal_chain", f"scenario_{i}"]
        }
        
        # Process both experiences
        cause_result = system.process_experience(cause_exp)
        effect_result = system.process_experience(effect_exp)
        
        print(f"\n  Scenario {i+1}: {scenario['cause']['description']} → {scenario['effect']['description']}")
        print(f"     - Delay: {scenario['delay']} minutes")
        print(f"     - Causal relations detected: {len(effect_result['causal_relations'])}")
        
        # Show detected causal relations
        for relation in effect_result['causal_relations']:
            print(f"       * Strength: {relation.strength:.3f}, Confidence: {relation.confidence:.3f}")


def demo_historical_decision_making(system):
    """Demonstrate historical context-aware decision making"""
    print_separator("HISTORICAL CONTEXT-AWARE DECISION MAKING")
    
    # Create decision scenarios with historical context
    current_context = {
        "timestamp": datetime.now().timestamp(),
        "type": "decision_point",
        "context": {
            "phase": "critical_decision",
            "complexity": 4,
            "urgency": "high",
            "available_time": "limited"
        }
    }
    
    # Different types of actions with varying historical success
    available_actions = [
        "thorough_analysis",    # Should score high based on preparation patterns
        "quick_decision",      # May score lower due to less preparation
        "seek_consultation",   # Should score well for complex scenarios
        "delay_decision",      # May score poorly for urgent contexts
        "incremental_approach" # Should score well based on learning patterns
    ]
    
    print("Making decision with historical context analysis...")
    print(f"Current context: {current_context['context']}")
    print(f"Available actions: {available_actions}")
    
    decision = system.make_historical_decision(current_context, available_actions)
    
    print(f"\n🎯 DECISION RESULTS:")
    print(f"   Chosen Action: {decision['chosen_action']}")
    print(f"   Confidence: {decision['confidence']:.3f}")
    print(f"   Rationale: {decision['rationale']}")
    
    print(f"\n📊 Action Scores:")
    for action, score in decision['action_scores'].items():
        indicator = "👑" if action == decision['chosen_action'] else "  "
        print(f"   {indicator} {action}: {score:.3f}")
    
    # Show historical analysis
    analysis = decision['historical_analysis']
    print(f"\n🔍 Historical Analysis:")
    print(f"   Similar situations found: {len(analysis['similar_situations'])}")
    print(f"   Successful patterns: {len(analysis['successful_patterns'])}")
    print(f"   Failure patterns: {len(analysis['failure_patterns'])}")


def demo_temporal_reasoning_and_validation(system):
    """Demonstrate temporal reasoning and knowledge validation"""
    print_separator("TEMPORAL REASONING & VALIDATION")
    
    # Check temporal consistency
    consistency = system.validate_temporal_consistency()
    
    print("🔍 Temporal Knowledge Validation:")
    print(f"   Temporal consistency: {'✅ PASS' if consistency['temporal_consistency']['consistent'] else '❌ FAIL'}")
    print(f"   Causal consistency: {'✅ PASS' if consistency['causal_consistency']['consistent'] else '❌ FAIL'}")
    print(f"   Overall consistency: {'✅ PASS' if consistency['overall_consistent'] else '❌ FAIL'}")
    
    if not consistency['temporal_consistency']['consistent']:
        print(f"   Temporal issues: {len(consistency['temporal_consistency']['inconsistencies'])}")
    
    if not consistency['causal_consistency']['consistent']:
        print(f"   Causal issues: {len(consistency['causal_consistency']['inconsistencies'])}")
    
    # Generate temporal insights
    query_context = {"type": "learning_phase"}
    insights = system.get_temporal_insights(query_context)
    
    print(f"\n🔮 Temporal Insights for '{query_context['type']}':")
    print(f"   Relevant patterns found: {len(insights['relevant_patterns'])}")
    print(f"   Historical precedents: {len(insights['historical_precedents'])}")
    print(f"   Predictions generated: {len(insights['predictions'])}")
    
    # Show predictions if any
    for prediction in insights['predictions']:
        if prediction['type'] == 'next_occurrence':
            predicted_time = datetime.fromtimestamp(prediction['predicted_time'])
            print(f"   📅 Next occurrence predicted: {predicted_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"      Confidence: {prediction['confidence']:.3f}")
            print(f"      Expected interval: {prediction['interval']/3600:.1f} hours")


def demo_integration_with_existing_systems(system):
    """Demonstrate integration with existing cognitive systems"""
    print_separator("INTEGRATION WITH EXISTING SYSTEMS")
    
    print("🔗 Integration Status:")
    print(f"   Episodic Memory: {len(system.episodic_memory.memories)} memories stored")
    print(f"   Knowledge Base: {len(system.knowledge_base.events)} events tracked")
    print(f"   Processing History: {len(system.processing_history)} processing records")
    
    # Show memory capacity and optimization
    memory_stats = {
        "current_index": system.episodic_memory.current_index,
        "memory_size": system.episodic_memory.memory_size,
        "consolidations": len(system.episodic_memory.consolidated_memories),
        "utilization": system.episodic_memory.current_index / system.episodic_memory.memory_size
    }
    
    print(f"\n📈 Memory System Statistics:")
    print(f"   Memory utilization: {memory_stats['utilization']:.1%}")
    print(f"   Consolidated memories: {memory_stats['consolidations']}")
    print(f"   Storage efficiency: {memory_stats['current_index']}/{memory_stats['memory_size']}")
    
    # Demonstrate pattern detection integration
    if len(system.episodic_memory.memories) > 2:
        recent_memories = system.episodic_memory.memories[-3:]
        memory_sequence = [m.content for m in recent_memories]
        patterns = system.pattern_detector.detect_temporal_patterns(memory_sequence)
        
        print(f"\n🎯 Pattern Detection Results:")
        print(f"   Recent patterns detected: {len(patterns)}")
        for pattern in patterns:
            print(f"   - Pattern {pattern.pattern_id}: confidence {pattern.confidence:.3f}")


def main():
    """Main demo function"""
    print("🧠 HISTORICAL CONTEXT INTEGRATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the comprehensive temporal knowledge representation,")
    print("reasoning, and decision-making capabilities implemented for CogPrime.")
    
    try:
        # Initialize and run demonstrations
        system = demo_temporal_knowledge_representation()
        demo_causal_relationship_detection(system)
        demo_historical_decision_making(system)
        demo_temporal_reasoning_and_validation(system)
        demo_integration_with_existing_systems(system)
        
        print_separator("DEMO COMPLETED SUCCESSFULLY")
        print("🎉 All Historical Context Integration System components")
        print("   have been demonstrated successfully!")
        
        # Final system statistics
        total_events = len(system.knowledge_base.events)
        total_relations = len(system.knowledge_base.temporal_relations) + len(system.knowledge_base.causal_relations)
        
        print(f"\n📊 Final System State:")
        print(f"   Total events processed: {total_events}")
        print(f"   Total relationships detected: {total_relations}")
        print(f"   Memory utilization: {system.episodic_memory.current_index}/{system.episodic_memory.memory_size}")
        print(f"   Processing history: {len(system.processing_history)} records")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)