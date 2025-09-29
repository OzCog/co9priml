#!/usr/bin/env python3
"""
Meta-Cognitive Synthesis Framework Demo
======================================

This demonstration showcases the comprehensive meta-cognitive synthesis 
framework that enables higher-order thinking about thinking, self-awareness,
and reasoning about cognitive processes within the CogPrime architecture.
"""

import sys
import os
import time
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import meta-cognitive framework components directly
from src.meta_cognitive.core.meta_cognitive_core import MetaCognitiveCore, CognitiveProcess, MetaCognitiveMode, MetaCognitiveLevel
from src.meta_cognitive.processing.higher_order_thinking import HigherOrderThinking
from src.meta_cognitive.awareness.self_awareness import SelfAwarenessSystem
from src.meta_cognitive.analysis.process_analyzer import CognitiveProcessAnalyzer
from src.meta_cognitive.strategy.strategy_selector import MetaCognitiveStrategySelector
from src.meta_cognitive.recursive.recursive_processor import RecursiveMetaCognitiveProcessor
from src.meta_cognitive.knowledge.meta_knowledge_system import MetaKnowledgeSystem
from src.meta_cognitive.learning.meta_learner import MetaCognitiveLearner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_meta_cognitive_framework(config: dict = None) -> MetaCognitiveCore:
    """
    Factory function to create a complete meta-cognitive synthesis framework.
    
    Args:
        config: Configuration dictionary for the framework
        
    Returns:
        Configured MetaCognitiveCore with all subsystems initialized
    """
    # Create core framework
    meta_core = MetaCognitiveCore(config)
    
    # Create and register all subsystems
    higher_order_thinking = HigherOrderThinking(config)
    self_awareness = SelfAwarenessSystem(config)
    process_analyzer = CognitiveProcessAnalyzer(config)
    strategy_selector = MetaCognitiveStrategySelector(config)
    recursive_processor = RecursiveMetaCognitiveProcessor(config)
    meta_knowledge = MetaKnowledgeSystem(config)
    meta_learner = MetaCognitiveLearner(config)
    
    # Register subsystems with core
    meta_core.register_subsystem('higher_order_thinking', higher_order_thinking)
    meta_core.register_subsystem('self_awareness', self_awareness)
    meta_core.register_subsystem('process_analyzer', process_analyzer)
    meta_core.register_subsystem('strategy_selector', strategy_selector)
    meta_core.register_subsystem('recursive_processor', recursive_processor)
    meta_core.register_subsystem('meta_knowledge', meta_knowledge)
    meta_core.register_subsystem('meta_learner', meta_learner)
    
    # Initialize all subsystems
    subsystems = [
        higher_order_thinking, self_awareness, process_analyzer,
        strategy_selector, recursive_processor, meta_knowledge, meta_learner
    ]
    
    for subsystem in subsystems:
        if hasattr(subsystem, 'initialize'):
            subsystem.initialize()
    
    return meta_core


def demo_meta_cognitive_framework():
    """Demonstrate the complete meta-cognitive synthesis framework."""
    print("🧠 Meta-Cognitive Synthesis Framework Demonstration")
    print("=" * 60)
    
    # Configuration for the framework
    config = {
        'max_recursion_depth': 3,
        'monitoring_frequency': 0.5,
        'enable_introspection': True,
        'learning_rate': 0.1,
        'exploration_rate': 0.2
    }
    
    # Create the meta-cognitive framework
    print("\n🔧 Initializing Meta-Cognitive Framework...")
    meta_framework = create_meta_cognitive_framework(config)
    
    # Get framework status
    framework_status = meta_framework.get_meta_cognitive_state()
    print(f"✅ Framework initialized with {len([k for k, v in framework_status['subsystems_available'].items() if v])} subsystems")
    
    # Demonstrate higher-order thinking
    print("\n🤔 Demonstrating Higher-Order Thinking...")
    thought_example = {
        'content': 'How can we optimize problem-solving strategies?',
        'context': 'meta_optimization',
        'complexity': 0.7
    }
    
    thinking_analysis = meta_framework.higher_order_thinking.think_about_thinking(
        thought_example, analysis_depth=2
    )
    
    print(f"   📊 Analysis Quality: {thinking_analysis.get('quality_assessment', {}).get('overall_efficiency', 'N/A')}")
    print(f"   💡 Insights Generated: {len(thinking_analysis.get('insights', []))}")
    print(f"   🔍 Meta-Observations: {len(thinking_analysis.get('meta_observations', []))}")
    
    # Demonstrate self-awareness
    print("\n🪞 Demonstrating Self-Awareness...")
    self_assessment = meta_framework.self_awareness.assess_self_state()
    
    print(f"   🎯 Confidence Level: {self_assessment['state_snapshot'].get('confidence_level', 'N/A'):.2f}")
    print(f"   ⚡ Cognitive Load: {self_assessment['state_snapshot'].get('cognitive_load', 'N/A'):.2f}")
    print(f"   🛠️  Active Capabilities: {len(self_assessment['state_snapshot'].get('capabilities_active', []))}")
    
    # Demonstrate introspection
    introspection_result = meta_framework.self_awareness.introspect('general')
    print(f"   🧐 Introspection Findings: {len(introspection_result.get('findings', []))}")
    print(f"   ✨ Introspection Insights: {len(introspection_result.get('insights', []))}")
    
    # Demonstrate process analysis
    print("\n📈 Demonstrating Process Analysis...")
    
    # Create a sample cognitive process
    sample_process = CognitiveProcess(
        process_id="demo_process_001",
        process_type="problem_solving",
        state="completed",
        performance_metrics={'accuracy': 0.8, 'efficiency': 0.7, 'speed': 0.6},
        resources_used={'memory': 0.4, 'processing': 0.5, 'attention': 0.6},
        start_time=time.time() - 5.0,
        duration=5.0
    )
    
    # Analyze process efficiency
    efficiency_analysis = meta_framework.process_analyzer.analyze_process_efficiency(sample_process)
    print(f"   ⏱️  Time Efficiency: {efficiency_analysis.get('time_efficiency', 'N/A'):.2f}")
    print(f"   🔋 Resource Efficiency: {efficiency_analysis.get('resource_efficiency', 'N/A'):.2f}")
    print(f"   🎯 Overall Efficiency: {efficiency_analysis.get('overall_efficiency', 'N/A'):.2f}")
    
    # Generate optimization suggestions
    optimizations = meta_framework.process_analyzer.suggest_optimizations(sample_process)
    print(f"   💡 Optimization Suggestions: {len(optimizations)}")
    for i, suggestion in enumerate(optimizations[:3], 1):
        print(f"      {i}. {suggestion}")
    
    # Demonstrate strategy selection
    print("\n🎯 Demonstrating Strategy Selection...")
    
    task_context = {
        'task_type': 'complex_analysis',
        'complexity': 0.8,
        'time_constraints': True,
        'available_resources': {'memory': 0.8, 'processing': 0.7, 'attention': 0.9}
    }
    
    available_strategies = ['analytical', 'intuitive', 'systematic', 'creative', 'adaptive']
    selected_strategy = meta_framework.strategy_selector.select_strategy(task_context, available_strategies)
    
    print(f"   🎲 Selected Strategy: {selected_strategy}")
    
    # Get strategy recommendations
    recommendations = meta_framework.strategy_selector.get_strategy_recommendations(task_context)
    print(f"   📋 Strategy Recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"      {i}. {rec.get('strategy_name', 'Unknown')} (Score: {rec.get('score', 0):.2f})")
    
    # Demonstrate recursive processing
    print("\n🔄 Demonstrating Recursive Meta-Cognitive Processing...")
    
    recursive_data = {
        'cognitive_processes': [sample_process],
        'analysis_focus': 'optimization',
        'complexity': 0.6
    }
    
    recursive_analysis = meta_framework.recursive_processor.recursive_analyze(
        recursive_data, depth=2
    )
    
    print(f"   📊 Total Depth Achieved: {recursive_analysis.get('total_depth', 'N/A')}")
    print(f"   ⏱️  Execution Time: {recursive_analysis.get('execution_time', 'N/A'):.2f}s")
    print(f"   🎯 Convergence Score: {recursive_analysis.get('convergence_score', 'N/A'):.2f}")
    print(f"   🛑 Termination Reason: {recursive_analysis.get('termination_reason', 'N/A')}")
    
    # Demonstrate meta-knowledge system
    print("\n📚 Demonstrating Meta-Knowledge System...")
    
    # Store some meta-knowledge
    knowledge_data = {
        'strategy': 'analytical',
        'context': 'problem_solving',
        'effectiveness': 0.8,
        'lessons_learned': ['systematic approach works well', 'requires more memory resources']
    }
    
    storage_success = meta_framework.meta_knowledge.store_meta_knowledge(
        'strategy_knowledge', knowledge_data
    )
    print(f"   💾 Knowledge Storage: {'✅ Success' if storage_success else '❌ Failed'}")
    
    # Retrieve knowledge
    query = {'tags': ['problem_solving'], 'min_confidence': 0.5}
    retrieved_knowledge = meta_framework.meta_knowledge.retrieve_meta_knowledge(
        'strategy_knowledge', query
    )
    print(f"   🔍 Retrieved Knowledge Items: {len(retrieved_knowledge)}")
    
    # Get knowledge statistics
    knowledge_stats = meta_framework.meta_knowledge.get_knowledge_statistics()
    print(f"   📊 Total Knowledge Items: {knowledge_stats.get('total_items', 0)}")
    
    # Demonstrate meta-learning
    print("\n🎓 Demonstrating Meta-Learning...")
    
    # Simulate learning experience
    learning_experience = {
        'type': 'strategy',
        'strategy': 'analytical',
        'context': {'context_type': 'problem_solving'},
        'outcome': {'success': True, 'improvement': 0.3},
        'novelty': 0.2
    }
    
    learning_success = meta_framework.meta_learner.learn_from_experience(learning_experience)
    print(f"   🧠 Learning from Experience: {'✅ Success' if learning_success else '❌ Failed'}")
    
    # Get learning statistics
    learning_stats = meta_framework.meta_learner.get_learning_statistics()
    print(f"   📈 Total Learning Experiences: {learning_stats.get('total_experiences', 0)}")
    print(f"   📊 Average Learning Value: {learning_stats.get('average_learning_value', 0):.2f}")
    print(f"   📉 Learning Trend: {learning_stats.get('recent_learning_trend', 'unknown')}")
    
    # Demonstrate integrated meta-cognitive reflection
    print("\n🔮 Demonstrating Integrated Meta-Cognitive Reflection...")
    
    reflection_result = meta_framework.reflect_on_cognition(reflection_depth=2)
    print(f"   🔍 Processes Analyzed: {reflection_result.get('processes_analyzed', 0)}")
    print(f"   💡 Insights Generated: {len(reflection_result.get('insights', []))}")
    print(f"   🔄 Patterns Detected: {len(reflection_result.get('patterns_detected', []))}")
    print(f"   🎯 Improvement Recommendations: {len(reflection_result.get('improvement_recommendations', []))}")
    
    # Show some specific insights
    insights = reflection_result.get('insights', [])[:3]
    if insights:
        print("   📝 Sample Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"      {i}. {insight}")
    
    # Demonstrate strategy optimization
    print("\n⚡ Demonstrating Strategy Optimization...")
    
    optimization_result = meta_framework.optimize_cognitive_strategy(task_context)
    print(f"   🎯 Current Strategy: {optimization_result.get('current_strategy', 'N/A')}")
    print(f"   💡 Recommended Strategy: {optimization_result.get('recommended_strategy', 'N/A')}")
    print(f"   📈 Expected Improvement: {optimization_result.get('expected_improvement', 0):.2f}")
    print(f"   🎲 Confidence: {optimization_result.get('confidence', 0):.2f}")
    
    # Final framework status
    print("\n📊 Final Framework Status:")
    final_status = meta_framework.get_meta_cognitive_state()
    print(f"   🔄 Current Recursion Depth: {final_status.get('recursion_depth', 0)}")
    print(f"   ⚡ Active Processes: {final_status.get('active_processes', 0)}")
    print(f"   📚 Performance History Size: {final_status.get('performance_history_size', 0)}")
    
    subsystems_status = final_status.get('subsystems_available', {})
    active_subsystems = [name for name, active in subsystems_status.items() if active]
    print(f"   🛠️  Active Subsystems: {len(active_subsystems)}/{len(subsystems_status)}")
    
    print("\n✨ Meta-Cognitive Synthesis Framework Demonstration Complete!")
    print("=" * 60)
    
    return meta_framework


def main():
    """Main demonstration function."""
    try:
        framework = demo_meta_cognitive_framework()
        
        print("\n🎯 Framework successfully demonstrated!")
        print("   The Meta-Cognitive Synthesis Framework is now ready for integration")
        print("   with the broader CogPrime architecture.")
        
        return framework
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    framework = main()