"""
Main Unified Cognitive Kernel

Integrates all cognitive components into a unified, distributed neural-symbolic architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path

from .tensor_kernel import TensorKernelCohesion
from .cognitive_grammar import CognitiveGrammarField
from .adaptive_interface import AdaptiveInterfaceLayer
from .attention_allocation import ECANAttentionAllocation
from .tensor_shapes import TensorShapeMetaDesign


class CognitiveState(Enum):
    """Current state of the cognitive kernel"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    LEARNING = "learning"
    REFLECTING = "reflecting"
    IDLE = "idle"


@dataclass
class CognitiveKernelConfig:
    """Configuration for the unified cognitive kernel"""
    # Tensor kernel configuration
    tensor_kernel: Dict[str, Any] = field(default_factory=lambda: {
        'ggml_backend': 'cpu',
        'kokkos_execution_space': 'serial',
        'a0ml_meta_learning_enabled': True
    })
    
    # Memory configuration
    memory_config: Dict[str, Any] = field(default_factory=lambda: {
        'hierarchical_levels': 3,
        'episodic_capacity': 10000,
        'semantic_capacity': 50000
    })
    
    # Attention configuration
    attention_config: Dict[str, Any] = field(default_factory=lambda: {
        'ecan_enabled': True,
        'attention_decay': 0.95,
        'importance_threshold': 0.1
    })
    
    # API configuration
    api_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_grpc': True,
        'enable_rest': True,
        'port': 8080
    })


class UnifiedCognitiveKernel:
    """
    Main unified cognitive kernel that orchestrates all cognitive components.
    
    This class implements the transcendent cognitive synthesis described in the
    architecture, integrating tensor operations, memory systems, attention allocation,
    and adaptive interfaces into a coherent whole.
    """
    
    def __init__(self, config: CognitiveKernelConfig):
        self.config = config
        self.state = CognitiveState.INITIALIZING
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.tensor_kernel = TensorKernelCohesion(config.tensor_kernel)
        self.cognitive_grammar = CognitiveGrammarField()
        self.adaptive_interface = AdaptiveInterfaceLayer(config.api_config)
        self.attention_allocation = ECANAttentionAllocation(config.attention_config)
        self.tensor_meta_design = TensorShapeMetaDesign()
        
        # Cognitive state tracking
        self.cognitive_field = {}
        self.gestalt_tensor = None
        self.attention_weights = {}
        
        self.logger.info("Unified Cognitive Kernel initialized")
    
    async def initialize(self) -> None:
        """Initialize all cognitive components and establish connections"""
        self.logger.info("Starting cognitive kernel initialization...")
        
        try:
            # Initialize tensor kernel cohesion
            await self.tensor_kernel.initialize()
            self.logger.info("Tensor kernel cohesion initialized")
            
            # Initialize cognitive grammar field
            await self.cognitive_grammar.initialize()
            self.logger.info("Cognitive grammar field initialized")
            
            # Initialize attention allocation
            await self.attention_allocation.initialize()
            self.logger.info("ECAN attention allocation initialized")
            
            # Initialize adaptive interface
            await self.adaptive_interface.initialize()
            self.logger.info("Adaptive interface layer initialized")
            
            # Establish inter-component connections
            await self._establish_connections()
            
            # Generate initial gestalt field
            self.gestalt_tensor = await self._generate_gestalt_field()
            
            self.state = CognitiveState.ACTIVE
            self.logger.info("Unified Cognitive Kernel fully initialized and active")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive kernel: {e}")
            raise
    
    async def _establish_connections(self) -> None:
        """Establish connections between cognitive components"""
        # Connect tensor kernel to cognitive grammar
        self.tensor_kernel.connect_to_atomspace(self.cognitive_grammar.atomspace)
        
        # Connect memory to attention allocation
        self.attention_allocation.connect_to_memory(self.cognitive_grammar.memory)
        
        # Connect pattern matcher to tensor operations
        self.cognitive_grammar.pattern_matcher.connect_to_tensor_kernel(self.tensor_kernel)
        
        # Connect API gateway to all components
        self.adaptive_interface.register_component('tensor_kernel', self.tensor_kernel)
        self.adaptive_interface.register_component('cognitive_grammar', self.cognitive_grammar)
        self.adaptive_interface.register_component('attention_allocation', self.attention_allocation)
        
        self.logger.info("Inter-component connections established")
    
    async def _generate_gestalt_field(self) -> np.ndarray:
        """Generate unified gestalt tensor field from all components"""
        # Get tensor shapes from all components
        tensor_shapes = await self.tensor_kernel.get_tensor_shapes()
        
        # Apply tensor meta-design principles
        gestalt_shape = self.tensor_meta_design.compute_gestalt_shape(tensor_shapes)
        
        # Create unified gestalt tensor
        gestalt_tensor = np.zeros(gestalt_shape, dtype=np.float32)
        
        # Populate with component states
        component_states = {
            'ggml': await self.tensor_kernel.ggml_state(),
            'kokkos': await self.tensor_kernel.kokkos_state(),
            'a0ml': await self.tensor_kernel.a0ml_state(),
            'mem0': await self.cognitive_grammar.memory_state(),
            'mlpn': await self.cognitive_grammar.mlpn_state(),
            'node9': await self.cognitive_grammar.node9_state()
        }
        
        # Apply prime factorization addressing
        for component, state in component_states.items():
            component_tensor = self.tensor_meta_design.factorize_component_tensor(
                component, state
            )
            gestalt_tensor = self.tensor_meta_design.merge_component_tensor(
                gestalt_tensor, component_tensor
            )
        
        self.logger.info(f"Generated gestalt tensor with shape: {gestalt_tensor.shape}")
        return gestalt_tensor
    
    async def cognitive_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete cognitive cycle: sense-think-act
        
        Args:
            input_data: Sensory or contextual input data
            
        Returns:
            Cognitive response and actions
        """
        if self.state != CognitiveState.ACTIVE:
            raise RuntimeError("Cognitive kernel not active")
        
        self.logger.debug("Starting cognitive cycle")
        
        # Step 1: Tensor processing through kernel cohesion
        tensor_response = await self.tensor_kernel.process_input(input_data)
        
        # Step 2: Memory and reasoning through cognitive grammar
        reasoning_response = await self.cognitive_grammar.process_reasoning(
            tensor_response, input_data
        )
        
        # Step 3: Attention allocation and resource management
        attention_response = await self.attention_allocation.allocate_attention(
            reasoning_response, self.cognitive_field
        )
        
        # Step 4: Update gestalt field
        self.gestalt_tensor = await self._update_gestalt_field(
            tensor_response, reasoning_response, attention_response
        )
        
        # Step 5: Generate response through adaptive interface
        final_response = await self.adaptive_interface.generate_response(
            attention_response, self.gestalt_tensor
        )
        
        self.logger.debug("Cognitive cycle completed")
        return final_response
    
    async def _update_gestalt_field(self, *component_responses) -> np.ndarray:
        """Update the unified gestalt tensor field based on component responses"""
        # Combine responses from all components
        combined_state = {}
        for response in component_responses:
            if isinstance(response, dict):
                combined_state.update(response)
        
        # Apply tensor meta-design update
        updated_gestalt = self.tensor_meta_design.update_gestalt_tensor(
            self.gestalt_tensor, combined_state
        )
        
        return updated_gestalt
    
    async def meta_learning_cycle(self) -> None:
        """Execute meta-learning to improve cognitive capabilities"""
        self.state = CognitiveState.LEARNING
        self.logger.info("Starting meta-learning cycle")
        
        try:
            # Analyze recent performance
            performance_metrics = await self._analyze_performance()
            
            # Apply meta-learning through a0ml
            await self.tensor_kernel.meta_learning_update(performance_metrics)
            
            # Update attention allocation strategies
            await self.attention_allocation.meta_learning_update(performance_metrics)
            
            # Evolve cognitive grammar patterns
            await self.cognitive_grammar.evolve_patterns(performance_metrics)
            
            self.logger.info("Meta-learning cycle completed")
            
        except Exception as e:
            self.logger.error(f"Meta-learning cycle failed: {e}")
            
        finally:
            self.state = CognitiveState.ACTIVE
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze recent cognitive performance for meta-learning"""
        # Collect performance metrics from all components
        metrics = {
            'tensor_kernel_efficiency': await self.tensor_kernel.get_efficiency_metrics(),
            'attention_allocation_effectiveness': await self.attention_allocation.get_effectiveness_metrics(),
            'reasoning_accuracy': await self.cognitive_grammar.get_accuracy_metrics(),
            'response_quality': await self.adaptive_interface.get_quality_metrics()
        }
        
        return metrics
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the cognitive kernel"""
        self.logger.info("Shutting down cognitive kernel...")
        
        # Shutdown components in reverse order
        await self.adaptive_interface.shutdown()
        await self.attention_allocation.shutdown()
        await self.cognitive_grammar.shutdown()
        await self.tensor_kernel.shutdown()
        
        self.state = CognitiveState.IDLE
        self.logger.info("Cognitive kernel shutdown complete")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the cognitive kernel"""
        return {
            'state': self.state.value,
            'gestalt_tensor_shape': self.gestalt_tensor.shape if self.gestalt_tensor is not None else None,
            'active_components': {
                'tensor_kernel': self.tensor_kernel.is_active(),
                'cognitive_grammar': self.cognitive_grammar.is_active(),
                'attention_allocation': self.attention_allocation.is_active(),
                'adaptive_interface': self.adaptive_interface.is_active()
            },
            'attention_weights': self.attention_weights,
            'cognitive_field_size': len(self.cognitive_field)
        }