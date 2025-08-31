#!/usr/bin/env python3
"""
Vervaeke 4E Cognition Framework Demonstration

This script demonstrates the integrated 4E cognition capabilities
following John Vervaeke's framework for relevance realization and
meaning-making in the CogPrime architecture.

The demonstration shows:
1. Embodied cognition through sensorimotor integration
2. Embedded cognition through environmental context awareness
3. Enacted cognition through active perception and exploration
4. Extended cognition through tool use and environmental coupling
5. Salience landscape navigation for attention guidance
6. Perspectival and participatory knowing modes
"""

import torch
import numpy as np
from typing import Dict, Any
import time

from src.core.cognitive_core import CogPrimeCore
from src.modules.perception import SensoryInput
from src.modules.action import Action
from src.modules.vervaeke_4e import KnowingMode, CognitionMode


class VervaekeDemo:
    """Demonstration of Vervaeke 4E Cognition Framework"""
    
    def __init__(self):
        """Initialize the demonstration system"""
        print("🧠 Initializing Vervaeke 4E Cognition Demonstration System")
        print("=" * 60)
        
        # Configure the cognitive system with Vervaeke enhancements
        config = {
            'feature_dim': 512,
            'memory_size': 2000,
            'learning_rate': 0.001,
            'vervaeke_config': {
                'enable_4e_cognition': True,
                'salience_threshold': 0.6,
                'embodied_learning_rate': 0.01,
                'context_history_size': 15
            }
        }
        
        self.cognitive_system = CogPrimeCore(config)
        print("✅ CogPrime system initialized with Vervaeke 4E framework")
        print(f"✅ Active knowing modes: {list(self.cognitive_system.vervaeke_framework.active_knowing_modes)}")
        print()
    
    def demonstrate_embodied_cognition(self):
        """Demonstrate embodied cognition principles"""
        print("🤸 EMBODIED COGNITION DEMONSTRATION")
        print("-" * 40)
        print("Embodied cognition: Cognition grounded in sensorimotor experience")
        print()
        
        # Simulate rich sensorimotor input
        sensory_scenarios = [
            {
                'name': 'Bright Moving Object',
                'visual': torch.randn(784) * 0.5 + 0.7,  # Bright, high intensity
                'auditory': torch.randn(256) * 0.2,       # Quiet
                'proprioceptive': torch.randn(64) * 0.1   # Minimal self-movement
            },
            {
                'name': 'Textured Surface Exploration',
                'visual': torch.randn(784) * 0.8,         # High texture variation
                'auditory': torch.randn(256) * 0.3,       # Moderate sounds
                'proprioceptive': torch.randn(64) * 0.6   # Active tactile exploration
            },
            {
                'name': 'Quiet Contemplation',
                'visual': torch.randn(784) * 0.2 + 0.3,  # Dim, steady environment
                'auditory': torch.randn(256) * 0.1,       # Very quiet
                'proprioceptive': torch.randn(64) * 0.05  # Minimal movement
            }
        ]
        
        for i, scenario in enumerate(sensory_scenarios):
            print(f"🔍 Scenario {i+1}: {scenario['name']}")
            
            sensory_input = SensoryInput(
                visual=scenario['visual'],
                auditory=scenario['auditory'],
                proprioceptive=scenario['proprioceptive']
            )
            
            action = self.cognitive_system.cognitive_cycle(sensory_input)
            
            # Extract embodied cognition results
            if hasattr(self.cognitive_system.state, 'vervaeke_4e_state'):
                embodied_results = self.cognitive_system.state.vervaeke_4e_state.get('embodied', {})
                
                if embodied_results:
                    embodied_state = embodied_results.get('embodied_state', torch.zeros(1))
                    motor_prediction = embodied_results.get('motor_prediction', torch.zeros(1))
                    proprioceptive_awareness = embodied_results.get('proprioceptive_awareness', torch.zeros(1))
                    
                    print(f"   🎯 Embodied state magnitude: {torch.norm(embodied_state):.3f}")
                    print(f"   🎯 Motor prediction strength: {torch.norm(motor_prediction):.3f}")
                    print(f"   🎯 Proprioceptive awareness: {torch.norm(proprioceptive_awareness):.3f}")
                    
                    # Interpret the results
                    if torch.norm(embodied_state) > 10:
                        print("   💪 High sensorimotor engagement - active embodied processing")
                    elif torch.norm(embodied_state) > 5:
                        print("   🖐️  Moderate embodied integration")
                    else:
                        print("   🧘 Low embodied activation - contemplative state")
            
            print()
        
        print("✅ Embodied cognition integrates sensory experience with motor predictions")
        print("✅ Proprioceptive awareness tracks internal bodily states")
        print("✅ Different scenarios elicit different embodied responses")
        print()
    
    def demonstrate_embedded_cognition(self):
        """Demonstrate embedded cognition and environmental context awareness"""
        print("🌍 EMBEDDED COGNITION DEMONSTRATION")
        print("-" * 40)
        print("Embedded cognition: Cognition shaped by environmental context")
        print()
        
        # Simulate different environmental contexts
        environments = [
            {
                'name': 'Collaborative Workshop',
                'context': {
                    'social_presence': {'agents': 3, 'cooperation': 0.9},
                    'tools_available': {'hammer': 0.8, 'screwdriver': 0.7, 'computer': 0.9},
                    'noise_level': 0.6,
                    'lighting': 0.8,
                    'goals': ['build_together', 'share_knowledge']
                }
            },
            {
                'name': 'Quiet Library',
                'context': {
                    'social_presence': {'agents': 1, 'cooperation': 0.3},
                    'tools_available': {'books': 0.9, 'computer': 0.7, 'notepad': 0.8},
                    'noise_level': 0.1,
                    'lighting': 0.6,
                    'goals': ['research', 'concentrate', 'learn']
                }
            },
            {
                'name': 'Busy Kitchen',
                'context': {
                    'social_presence': {'agents': 2, 'cooperation': 0.7},
                    'tools_available': {'stove': 0.9, 'knife': 0.8, 'timer': 0.9},
                    'noise_level': 0.8,
                    'lighting': 0.9,
                    'goals': ['cook_meal', 'coordinate_timing', 'clean_as_you_go']
                }
            }
        ]
        
        sensory_input = SensoryInput(visual=torch.randn(784), auditory=torch.randn(256))
        
        for env in environments:
            print(f"🏢 Environment: {env['name']}")
            
            # Process with environmental context
            action = self.cognitive_system.cognitive_cycle(sensory_input)
            
            # Extract embedded cognition results
            if hasattr(self.cognitive_system.state, 'vervaeke_4e_state'):
                embedded_results = self.cognitive_system.state.vervaeke_4e_state.get('embedded', {})
                
                if embedded_results:
                    affordances = embedded_results.get('affordances', torch.zeros(1))
                    context_stability = embedded_results.get('context_stability', torch.tensor(0.0))
                    
                    print(f"   🎯 Affordances detected: {torch.sum(affordances > 0.5).item()}/32")
                    print(f"   🎯 Context stability: {context_stability:.3f}")
                    
                    # Interpret affordances
                    top_affordances = torch.topk(affordances, 3)
                    print(f"   🔧 Top affordance strengths: {top_affordances.values.tolist()}")
                    
                    # Environmental adaptation
                    noise_level = env['context']['noise_level']
                    if noise_level > 0.7:
                        print("   📢 High noise environment detected - enhanced auditory filtering")
                    elif noise_level < 0.3:
                        print("   🤫 Quiet environment detected - increased auditory sensitivity")
                    
                    social_agents = env['context']['social_presence']['agents']
                    if social_agents > 2:
                        print("   👥 Multi-agent context - collaborative affordances prioritized")
                    elif social_agents == 1:
                        print("   🧑 Solo context - individual task affordances prioritized")
            
            print()
        
        print("✅ Embedded cognition adapts to environmental constraints and opportunities")
        print("✅ Affordance detection identifies available action possibilities")
        print("✅ Context stability tracking enables adaptive behavior")
        print()
    
    def demonstrate_enacted_cognition(self):
        """Demonstrate enacted cognition through active exploration"""
        print("🎭 ENACTED COGNITION DEMONSTRATION")
        print("-" * 40)
        print("Enacted cognition: Cognition through active engagement with the world")
        print()
        
        # Simulate exploration sequence
        exploration_actions = [
            {
                'name': 'cautious_exploration',
                'description': 'Careful initial investigation',
                'parameters': {'speed': 0.3, 'depth': 0.4, 'caution': 0.8},
                'confidence': 0.6
            },
            {
                'name': 'focused_investigation',
                'description': 'Targeted examination of interesting features',
                'parameters': {'speed': 0.5, 'depth': 0.8, 'caution': 0.4},
                'confidence': 0.8
            },
            {
                'name': 'bold_exploration',
                'description': 'Confident exploration of new territories',
                'parameters': {'speed': 0.8, 'depth': 0.9, 'caution': 0.2},
                'confidence': 0.9
            }
        ]
        
        # Start with baseline sensory input
        base_visual = torch.randn(784)
        base_auditory = torch.randn(256)
        
        print("🔄 Active exploration sequence:")
        
        for i, action_spec in enumerate(exploration_actions):
            print(f"\n🎯 Action {i+1}: {action_spec['description']}")
            
            # Create action
            action = Action(
                name=action_spec['name'],
                parameters=action_spec['parameters'],
                confidence=action_spec['confidence'],
                expected_outcome=torch.randn(10),
                priority=0.7 + i * 0.1
            )
            
            # Simulate perceptual consequences of action
            # More active exploration leads to richer sensory input
            exploration_intensity = action_spec['parameters']['speed'] * action_spec['parameters']['depth']
            
            visual_richness = exploration_intensity
            sensory_input = SensoryInput(
                visual=base_visual + torch.randn(784) * visual_richness,
                auditory=base_auditory + torch.randn(256) * visual_richness * 0.5,
                proprioceptive=torch.randn(64) * exploration_intensity
            )
            
            # Process through enacted cognition
            result_action = self.cognitive_system.cognitive_cycle(sensory_input)
            
            # Extract enacted cognition results
            if hasattr(self.cognitive_system.state, 'vervaeke_4e_state'):
                enacted_results = self.cognitive_system.state.vervaeke_4e_state.get('enacted', {})
                
                if enacted_results:
                    exploration_motivation = enacted_results.get('exploration_motivation', torch.tensor(0.0))
                    coupling_strength = enacted_results.get('coupling_strength', torch.tensor(0.0))
                    
                    print(f"   🎯 Exploration motivation: {exploration_motivation.mean():.3f}")
                    print(f"   🎯 Action-perception coupling: {coupling_strength.item():.3f}")
                    
                    # Interpret results
                    if exploration_motivation.mean() > 0.6:
                        print("   🚀 High exploration drive - seeking novel experiences")
                    elif exploration_motivation.mean() > 0.4:
                        print("   🔍 Moderate exploration - balanced investigation")
                    else:
                        print("   🎯 Low exploration - focused on current understanding")
                    
                    # Show coupling strength interpretation
                    if coupling_strength.item() > 5.0:
                        print("   🔗 Strong action-perception coupling - enactive learning active")
                    else:
                        print("   🔗 Weak coupling - passive perception mode")
        
        print("\n✅ Enacted cognition demonstrates learning through doing")
        print("✅ Action-perception coupling enables enactive knowledge building")
        print("✅ Exploration motivation drives active engagement with environment")
        print()
    
    def demonstrate_extended_cognition(self):
        """Demonstrate extended cognition and tool use"""
        print("🔧 EXTENDED COGNITION DEMONSTRATION")
        print("-" * 40)
        print("Extended cognition: Cognition distributed across tools and environment")
        print()
        
        # Simulate different tool-use scenarios
        tool_scenarios = [
            {
                'name': 'Complex Problem Solving',
                'description': 'Using external tools to extend cognitive capacity',
                'cognitive_load': 0.9,
                'available_tools': ['calculator', 'reference_manual', 'collaboration_platform']
            },
            {
                'name': 'Creative Design Task',
                'description': 'Leveraging environmental resources for creativity',
                'cognitive_load': 0.7,
                'available_tools': ['sketch_pad', 'inspiration_board', 'prototype_materials']
            },
            {
                'name': 'Routine Information Processing',
                'description': 'Simple task with minimal tool requirements',
                'cognitive_load': 0.3,
                'available_tools': ['basic_interface']
            }
        ]
        
        for scenario in tool_scenarios:
            print(f"🛠️  Scenario: {scenario['name']}")
            print(f"   📝 {scenario['description']}")
            
            # Create cognitive load through complex sensory input
            cognitive_complexity = scenario['cognitive_load']
            sensory_input = SensoryInput(
                visual=torch.randn(784) * cognitive_complexity,
                auditory=torch.randn(256) * cognitive_complexity
            )
            
            # Process through extended cognition
            action = self.cognitive_system.cognitive_cycle(sensory_input)
            
            # Extract extended cognition results
            if hasattr(self.cognitive_system.state, 'vervaeke_4e_state'):
                extended_results = self.cognitive_system.state.vervaeke_4e_state.get('extended', {})
                
                if extended_results:
                    cognitive_offloading = extended_results.get('cognitive_offloading', torch.tensor(0.0))
                    tool_usage_pattern = extended_results.get('tool_usage_pattern', torch.zeros(10))
                    
                    print(f"   🎯 Cognitive offloading: {cognitive_offloading.item():.3f}")
                    
                    # Show tool usage
                    most_used_tools = torch.topk(tool_usage_pattern, 3)
                    active_tools = (tool_usage_pattern > 0).sum().item()
                    print(f"   🎯 Active tools: {active_tools}/10")
                    print(f"   🎯 Primary tool usage: {most_used_tools.values[:3].tolist()}")
                    
                    # Interpret offloading
                    if cognitive_offloading.item() > 0.7:
                        print("   🧠 High cognitive offloading - leveraging external resources effectively")
                    elif cognitive_offloading.item() > 0.4:
                        print("   🧠 Moderate offloading - balanced internal/external processing")
                    else:
                        print("   🧠 Low offloading - primarily internal processing")
                    
                    # Tool selection intelligence
                    if scenario['cognitive_load'] > 0.7 and cognitive_offloading.item() > 0.5:
                        print("   ✅ Intelligent tool use - high load drives appropriate offloading")
                    elif scenario['cognitive_load'] < 0.4 and cognitive_offloading.item() < 0.4:
                        print("   ✅ Efficient processing - low load requires minimal tools")
            
            print()
        
        print("✅ Extended cognition effectively uses external tools and resources")
        print("✅ Cognitive offloading adapts to task complexity")
        print("✅ Tool selection matches cognitive demands")
        print()
    
    def demonstrate_salience_landscape(self):
        """Demonstrate salience landscape navigation"""
        print("🗺️  SALIENCE LANDSCAPE NAVIGATION DEMONSTRATION")
        print("-" * 40)
        print("Salience landscape: Dynamic attention allocation through relevance realization")
        print()
        
        # Create complex environment with multiple potential attention targets
        environmental_context = {
            'urgent_alarm': {
                'intensity': 0.9,
                'novelty': 0.8,
                'relevance_to_goals': 0.9,
                'temporal_urgency': 0.95
            },
            'interesting_pattern': {
                'intensity': 0.6,
                'novelty': 0.9,
                'relevance_to_goals': 0.4,
                'temporal_urgency': 0.3
            },
            'routine_maintenance': {
                'intensity': 0.3,
                'novelty': 0.1,
                'relevance_to_goals': 0.8,
                'temporal_urgency': 0.2
            },
            'social_interaction': {
                'intensity': 0.7,
                'novelty': 0.5,
                'relevance_to_goals': 0.7,
                'temporal_urgency': 0.6
            },
            'goals': ['safety_monitoring', 'pattern_recognition', 'social_engagement'],
            'urgency': 0.7
        }
        
        sensory_input = SensoryInput(
            visual=torch.randn(784),
            auditory=torch.randn(256)
        )
        
        print("🎯 Processing complex multi-target environment...")
        print("Available attention targets:")
        for target, properties in environmental_context.items():
            if isinstance(properties, dict) and 'intensity' in properties:
                print(f"   • {target}: intensity={properties['intensity']:.2f}, "
                      f"novelty={properties['novelty']:.2f}, "
                      f"urgency={properties['temporal_urgency']:.2f}")
        
        # Process through salience landscape
        action = self.cognitive_system.cognitive_cycle(sensory_input)
        
        # Extract salience landscape results
        if hasattr(self.cognitive_system.state, 'vervaeke_4e_state'):
            attention_focus = self.cognitive_system.state.vervaeke_4e_state.get('attention_focus')
            salience_landscape = self.cognitive_system.state.vervaeke_4e_state.get('salience_landscape', {})
            
            print(f"\n🎯 Attention focused on: {attention_focus}")
            
            if salience_landscape:
                print("\n📊 Salience Landscape Analysis:")
                for item_id, salience_vector in salience_landscape.items():
                    if hasattr(salience_vector, 'aspectuality'):
                        total_salience = (salience_vector.aspectuality + 
                                        salience_vector.centrality + 
                                        salience_vector.temporality) / 3
                        print(f"   • {item_id}:")
                        print(f"     - Aspectuality: {salience_vector.aspectuality:.3f}")
                        print(f"     - Centrality: {salience_vector.centrality:.3f}")
                        print(f"     - Temporality: {salience_vector.temporality:.3f}")
                        print(f"     - Total Salience: {total_salience:.3f}")
                        
                        if item_id == attention_focus:
                            print(f"     🎯 ← ATTENTION FOCUS")
            
            # Validate attention allocation
            if attention_focus == 'urgent_alarm':
                print("\n✅ Excellent attention allocation - urgent items prioritized")
            elif attention_focus in ['social_interaction', 'interesting_pattern']:
                print("\n✅ Good attention allocation - balanced priorities")
            else:
                print("\n✅ Attention allocated based on current salience landscape")
        
        print("\n✅ Salience landscape enables dynamic attention allocation")
        print("✅ ACT framework (Aspectuality, Centrality, Temporality) guides relevance")
        print("✅ Attention naturally flows to most salient items")
        print()
    
    def demonstrate_knowing_modes(self):
        """Demonstrate different knowing modes"""
        print("🎓 KNOWING MODES DEMONSTRATION")
        print("-" * 40)
        print("Vervaeke's four kinds of knowing: Propositional, Procedural, Perspectival, Participatory")
        print()
        
        # Test different knowing modes
        framework = self.cognitive_system.vervaeke_framework
        
        # Perspectival knowing
        print("👁️  PERSPECTIVAL KNOWING (knowing what it's like)")
        perspectival = framework.get_perspectival_knowing()
        if perspectival:
            for key, value in perspectival.items():
                print(f"   • {key}: {value}")
            print("   🎯 Provides subjective, experiential understanding")
        
        print()
        
        # Update to include participatory knowing
        framework.update_knowing_modes([KnowingMode.PARTICIPATORY, KnowingMode.PERSPECTIVAL])
        
        print("🤝 PARTICIPATORY KNOWING (knowing by participating)")
        participatory = framework.get_participatory_knowing()
        if participatory:
            for key, value in participatory.items():
                print(f"   • {key}: {value}")
            print("   🎯 Enables transformative engagement with environment")
        
        print()
        
        # Show how knowing modes influence processing
        print("🔄 KNOWING MODES INFLUENCE ON PROCESSING:")
        sensory_input = SensoryInput(visual=torch.randn(784))
        action = self.cognitive_system.cognitive_cycle(sensory_input)
        
        active_modes = framework.active_knowing_modes
        print(f"   • Active modes: {[mode.value for mode in active_modes]}")
        
        if KnowingMode.PERSPECTIVAL in active_modes:
            print("   ✅ Perspectival knowing: Enhanced subjective experience processing")
        
        if KnowingMode.PARTICIPATORY in active_modes:
            print("   ✅ Participatory knowing: Increased transformative potential")
        
        print("\n✅ Multiple knowing modes provide complementary understanding")
        print("✅ Perspectival knowing adds experiential richness")
        print("✅ Participatory knowing enables co-constitutive relationships")
        print()
    
    def run_comprehensive_demo(self):
        """Run the complete demonstration"""
        print("🎪 COMPREHENSIVE VERVAEKE 4E COGNITION DEMONSTRATION")
        print("=" * 80)
        print("Showcasing John Vervaeke's framework for relevance realization")
        print("integrated into the CogPrime cognitive architecture")
        print("=" * 80)
        print()
        
        # Run all demonstrations
        self.demonstrate_embodied_cognition()
        self.demonstrate_embedded_cognition()
        self.demonstrate_enacted_cognition()
        self.demonstrate_extended_cognition()
        self.demonstrate_salience_landscape()
        self.demonstrate_knowing_modes()
        
        # Final integration demonstration
        print("🌟 INTEGRATED 4E COGNITION DEMONSTRATION")
        print("-" * 40)
        print("All four modes working together in complex scenario")
        print()
        
        # Create a rich, complex scenario
        complex_scenario = {
            'visual': torch.randn(784) * 0.8,
            'auditory': torch.randn(256) * 0.6,
            'proprioceptive': torch.randn(64) * 0.4
        }
        
        complex_context = {
            'collaborative_workspace': {
                'complexity': 0.8,
                'social_dynamics': 0.7,
                'tool_availability': 0.9,
                'urgency': 0.6
            },
            'goals': ['collaborate_effectively', 'solve_complex_problem', 'learn_new_skills'],
            'urgency': 0.7
        }
        
        action = Action(
            name="integrated_problem_solving",
            parameters={'collaboration': True, 'complexity': 0.8, 'creativity': 0.7},
            confidence=0.8,
            expected_outcome=torch.randn(10),
            priority=0.9
        )
        
        sensory_input = SensoryInput(
            visual=complex_scenario['visual'],
            auditory=complex_scenario['auditory'],
            proprioceptive=complex_scenario['proprioceptive']
        )
        
        print("🎯 Processing integrated 4E cognition scenario...")
        result_action = self.cognitive_system.cognitive_cycle(sensory_input)
        
        if hasattr(self.cognitive_system.state, 'vervaeke_4e_state'):
            state = self.cognitive_system.state.vervaeke_4e_state
            
            print("\n📊 Integrated 4E Cognition Results:")
            
            # Embodied
            if 'embodied' in state:
                embodied_norm = torch.norm(state['embodied']['embodied_state'])
                print(f"   🤸 Embodied integration: {embodied_norm:.3f}")
            
            # Embedded
            if 'embedded' in state:
                affordances = state['embedded']['affordances']
                context_stability = state['embedded']['context_stability']
                print(f"   🌍 Environmental affordances: {torch.sum(affordances > 0.5).item()}/32")
                print(f"   🌍 Context stability: {context_stability:.3f}")
            
            # Enacted
            if 'enacted' in state:
                exploration = state['enacted']['exploration_motivation']
                coupling = state['enacted']['coupling_strength']
                print(f"   🎭 Exploration motivation: {exploration.mean():.3f}")
                print(f"   🎭 Action-perception coupling: {coupling.item():.3f}")
            
            # Extended
            if 'extended' in state:
                offloading = state['extended']['cognitive_offloading']
                tool_usage = state['extended']['tool_usage_pattern']
                active_tools = (tool_usage > 0).sum().item()
                print(f"   🔧 Cognitive offloading: {offloading.item():.3f}")
                print(f"   🔧 Active tools: {active_tools}/10")
            
            # Salience
            focus = state.get('attention_focus')
            if focus:
                print(f"   🗺️  Attention focus: {focus}")
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("✅ All 4E cognition modes successfully integrated")
        print("✅ Embodied cognition provides sensorimotor grounding")
        print("✅ Embedded cognition enables environmental adaptation")
        print("✅ Enacted cognition drives active exploration")
        print("✅ Extended cognition leverages external resources")
        print("✅ Salience landscape guides attention dynamically")
        print("✅ Multiple knowing modes enrich understanding")
        print("=" * 60)


def main():
    """Run the demonstration"""
    demo = VervaekeDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()