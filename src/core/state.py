"""
CogPrime State Definitions

This module contains the core state definitions for CogPrime to avoid circular imports.
"""

import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class CognitiveState:
    """Represents the current cognitive state of the system"""
    attention_focus: torch.Tensor
    working_memory: Dict[str, Any]
    emotional_valence: float
    goal_stack: List[str]
    sensory_buffer: Dict[str, torch.Tensor]
    current_thought: Optional[Any] = None  # Will be Thought object
    last_action: Optional[Any] = None  # Will be Action object
    last_reward: float = 0.0
    total_reward: float = 0.0