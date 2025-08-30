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
    
    # Vervaeke 4E Cognition state
    vervaeke_4e_state: Optional[Dict[str, Any]] = None
    attention_history: List[str] = None
    
    def __post_init__(self):
        """Initialize default values after dataclass creation"""
        if self.attention_history is None:
            self.attention_history = []
    
    def update_attention_history(self, focus_item: str, max_history: int = 10):
        """Update attention history for temporal relevance computation"""
        if self.attention_history is None:
            self.attention_history = []
        self.attention_history.append(focus_item)
        if len(self.attention_history) > max_history:
            self.attention_history.pop(0)