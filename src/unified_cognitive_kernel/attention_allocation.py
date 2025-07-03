"""
ECAN Attention Allocation

Implements Economic Attention Allocation Networks (ECAN) for adaptive
cognitive resource management and attention spreading in the unified
cognitive kernel.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import time
from collections import defaultdict, deque


class AttentionType(Enum):
    """Types of attention mechanisms"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    FOCUSED = "focused"
    DIFFUSE = "diffuse"
    SELECTIVE = "selective"
    DIVIDED = "divided"


class EconomicAction(Enum):
    """Economic actions in the attention economy"""
    RENT_PAYMENT = "rent_payment"
    WAGE_PAYMENT = "wage_payment"
    STIMULUS_GRANT = "stimulus_grant"
    IMPORTANCE_DECAY = "importance_decay"
    ATTENTION_SPREAD = "attention_spread"


@dataclass
class AttentionAtom:
    """Atom with attention values for ECAN processing"""
    atom_id: str
    attention_value: float = 0.0
    short_term_importance: float = 0.0
    long_term_importance: float = 0.0
    wage: float = 0.0
    rent: float = 0.0
    stimulus: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    creation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionFocus:
    """Current attention focus configuration"""
    focus_atoms: List[str]
    focus_strength: float
    focus_duration: float
    focus_type: AttentionType
    background_atoms: List[str]
    created_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EconomicTransaction:
    """Economic transaction in the attention economy"""
    transaction_id: str
    action: EconomicAction
    source_atom: Optional[str]
    target_atom: Optional[str]
    amount: float
    timestamp: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionSpreadingAlgorithm:
    """Algorithm for spreading attention across the cognitive network"""
    
    def __init__(self, spread_factor: float = 0.1, decay_factor: float = 0.95):
        self.spread_factor = spread_factor
        self.decay_factor = decay_factor
        self.logger = logging.getLogger(__name__)
    
    async def spread_attention(self, source_atom: AttentionAtom, 
                             connected_atoms: List[AttentionAtom],
                             connection_strengths: List[float]) -> List[AttentionAtom]:
        """Spread attention from source atom to connected atoms"""
        if len(connected_atoms) != len(connection_strengths):
            raise ValueError("Number of connected atoms must match connection strengths")
        
        spread_amount = source_atom.attention_value * self.spread_factor
        
        # Distribute attention based on connection strengths
        total_strength = sum(connection_strengths)
        if total_strength == 0:
            return connected_atoms
        
        for i, (atom, strength) in enumerate(zip(connected_atoms, connection_strengths)):
            attention_transfer = spread_amount * (strength / total_strength)
            atom.attention_value += attention_transfer
            atom.last_accessed = time.time()
            atom.access_count += 1
        
        # Decay source attention
        source_atom.attention_value *= self.decay_factor
        
        return connected_atoms


class ECANAttentionAllocation:
    """
    Economic Attention Allocation Networks (ECAN) implementation.
    
    This class implements the ECAN attention allocation mechanism that manages
    cognitive resources through an economic model with attention values,
    importance measures, and economic transactions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ECAN parameters
        self.attention_bank = config.get('attention_bank', 1000.0)
        self.importance_decay = config.get('importance_decay', 0.95)
        self.wage_scaling = config.get('wage_scaling', 0.1)
        self.rent_scaling = config.get('rent_scaling', 0.05)
        self.stimulus_threshold = config.get('stimulus_threshold', 0.1)
        
        # Attention management
        self.attention_atoms = {}
        self.attention_focus = None
        self.attention_history = deque(maxlen=1000)
        self.economic_transactions = deque(maxlen=1000)
        
        # Attention spreading
        self.spreading_algorithm = AttentionSpreadingAlgorithm()
        
        # Economic parameters
        self.total_attention_bank = self.attention_bank
        self.total_importance_bank = 1000.0
        self.rent_frequency = 10  # Every 10 cycles
        self.wage_frequency = 5   # Every 5 cycles
        self.cycle_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'attention_efficiency': 0.0,
            'importance_stability': 0.0,
            'economic_balance': 0.0,
            'focus_coherence': 0.0
        }
        
        # Memory system connection
        self.memory_system = None
        
        self.logger.info("ECAN attention allocation initialized")
    
    async def initialize(self) -> None:
        """Initialize ECAN attention allocation system"""
        self.logger.info("Initializing ECAN attention allocation...")
        
        try:
            # Initialize attention bank
            self.total_attention_bank = self.attention_bank
            
            # Create default attention atoms
            await self._create_default_attention_atoms()
            
            # Initialize attention focus
            await self._initialize_attention_focus()
            
            # Start economic processes
            await self._start_economic_processes()
            
            self.logger.info("ECAN attention allocation fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ECAN attention allocation: {e}")
            raise
    
    async def _create_default_attention_atoms(self) -> None:
        """Create default attention atoms for fundamental concepts"""
        default_atoms = [
            ("self", 0.8, 0.9, 0.7),  # (atom_id, attention, sti, lti)
            ("current_goal", 0.7, 0.8, 0.6),
            ("context", 0.6, 0.7, 0.8),
            ("memory", 0.5, 0.6, 0.9),
            ("reasoning", 0.6, 0.7, 0.8),
            ("perception", 0.5, 0.6, 0.7),
            ("action", 0.4, 0.5, 0.6),
            ("learning", 0.3, 0.4, 0.8)
        ]
        
        current_time = time.time()
        
        for atom_id, attention, sti, lti in default_atoms:
            attention_atom = AttentionAtom(
                atom_id=atom_id,
                attention_value=attention,
                short_term_importance=sti,
                long_term_importance=lti,
                wage=0.0,
                rent=0.0,
                stimulus=0.0,
                last_accessed=current_time,
                access_count=0,
                creation_time=current_time,
                metadata={"default": True}
            )
            self.attention_atoms[atom_id] = attention_atom
    
    async def _initialize_attention_focus(self) -> None:
        """Initialize attention focus configuration"""
        # Start with focus on current goal and context
        self.attention_focus = AttentionFocus(
            focus_atoms=["current_goal", "context"],
            focus_strength=0.8,
            focus_duration=10.0,
            focus_type=AttentionType.FOCUSED,
            background_atoms=["memory", "reasoning"],
            created_time=time.time(),
            metadata={"initial_focus": True}
        )
    
    async def _start_economic_processes(self) -> None:
        """Start economic processes for attention allocation"""
        # Initialize economic parameters for atoms
        for atom in self.attention_atoms.values():
            atom.wage = atom.short_term_importance * self.wage_scaling
            atom.rent = atom.attention_value * self.rent_scaling
        
        self.logger.info("Economic processes started")
    
    async def allocate_attention(self, reasoning_response: Dict[str, Any], 
                               cognitive_field: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate attention based on reasoning results and cognitive field state.
        
        This is the main ECAN attention allocation method that determines
        where cognitive resources should be focused.
        """
        self.logger.debug("Allocating attention via ECAN")
        
        self.cycle_count += 1
        
        # Extract relevant information
        pattern_matches = reasoning_response.get('pattern_matches', [])
        inferences = reasoning_response.get('inferences', [])
        memory_results = reasoning_response.get('memory_results', {})
        
        # Step 1: Update attention values based on reasoning
        await self._update_attention_from_reasoning(pattern_matches, inferences)
        
        # Step 2: Apply economic transactions
        await self._apply_economic_transactions()
        
        # Step 3: Spread attention across connected atoms
        await self._spread_attention()
        
        # Step 4: Update attention focus
        await self._update_attention_focus()
        
        # Step 5: Apply importance decay
        await self._apply_importance_decay()
        
        # Step 6: Calculate performance metrics
        await self._update_performance_metrics()
        
        # Generate attention allocation result
        allocation_result = {
            'attention_allocations': await self._get_attention_allocations(),
            'attention_focus': await self._get_attention_focus_state(),
            'economic_state': await self._get_economic_state(),
            'performance_metrics': self.performance_metrics,
            'cycle_count': self.cycle_count
        }
        
        # Record in history
        self.attention_history.append({
            'timestamp': time.time(),
            'allocation_result': allocation_result,
            'reasoning_input': reasoning_response
        })
        
        return allocation_result
    
    async def _update_attention_from_reasoning(self, pattern_matches: List[Dict[str, Any]], 
                                             inferences: List[Dict[str, Any]]) -> None:
        """Update attention values based on reasoning results"""
        # Increase attention for atoms involved in pattern matches
        for match in pattern_matches:
            template = match.get('template', '')
            confidence = match.get('confidence', 0.0)
            
            # Create or update attention atom for template
            if template not in self.attention_atoms:
                await self._create_attention_atom(template, confidence * 0.5)
            else:
                atom = self.attention_atoms[template]
                atom.attention_value = min(1.0, atom.attention_value + confidence * 0.2)
                atom.short_term_importance = min(1.0, atom.short_term_importance + confidence * 0.1)
                atom.last_accessed = time.time()
                atom.access_count += 1
        
        # Increase attention for atoms involved in inferences
        for inference in inferences:
            premise_atoms = inference.get('premise_atoms', [])
            conclusion_atom = inference.get('conclusion_atom', '')
            confidence = inference.get('confidence', 0.0)
            
            # Update premise atoms
            for premise in premise_atoms:
                if premise not in self.attention_atoms:
                    await self._create_attention_atom(premise, confidence * 0.3)
                else:
                    atom = self.attention_atoms[premise]
                    atom.attention_value = min(1.0, atom.attention_value + confidence * 0.15)
                    atom.short_term_importance = min(1.0, atom.short_term_importance + confidence * 0.1)
            
            # Update conclusion atom
            if conclusion_atom not in self.attention_atoms:
                await self._create_attention_atom(conclusion_atom, confidence * 0.4)
            else:
                atom = self.attention_atoms[conclusion_atom]
                atom.attention_value = min(1.0, atom.attention_value + confidence * 0.2)
                atom.short_term_importance = min(1.0, atom.short_term_importance + confidence * 0.15)
    
    async def _create_attention_atom(self, atom_id: str, initial_attention: float) -> None:
        """Create new attention atom"""
        current_time = time.time()
        
        attention_atom = AttentionAtom(
            atom_id=atom_id,
            attention_value=initial_attention,
            short_term_importance=initial_attention * 0.8,
            long_term_importance=initial_attention * 0.5,
            wage=initial_attention * self.wage_scaling,
            rent=initial_attention * self.rent_scaling,
            stimulus=0.0,
            last_accessed=current_time,
            access_count=1,
            creation_time=current_time,
            metadata={"created_from_reasoning": True}
        )
        
        self.attention_atoms[atom_id] = attention_atom
    
    async def _apply_economic_transactions(self) -> None:
        """Apply economic transactions (rent, wages, stimulus)"""
        current_time = time.time()
        
        # Apply rent payments
        if self.cycle_count % self.rent_frequency == 0:
            await self._apply_rent_payments()
        
        # Apply wage payments
        if self.cycle_count % self.wage_frequency == 0:
            await self._apply_wage_payments()
        
        # Apply stimulus grants
        await self._apply_stimulus_grants()
    
    async def _apply_rent_payments(self) -> None:
        """Apply rent payments to reduce attention for unused atoms"""
        current_time = time.time()
        
        for atom in self.attention_atoms.values():
            # Calculate rent based on attention value and time since last access
            time_since_access = current_time - atom.last_accessed
            rent_amount = atom.attention_value * self.rent_scaling * (1 + time_since_access / 3600)  # Increase with time
            
            # Apply rent payment
            atom.attention_value = max(0.0, atom.attention_value - rent_amount)
            atom.rent += rent_amount
            
            # Record transaction
            transaction = EconomicTransaction(
                transaction_id=f"rent_{atom.atom_id}_{self.cycle_count}",
                action=EconomicAction.RENT_PAYMENT,
                source_atom=atom.atom_id,
                target_atom=None,
                amount=rent_amount,
                timestamp=current_time,
                success=True,
                metadata={"time_since_access": time_since_access}
            )
            self.economic_transactions.append(transaction)
    
    async def _apply_wage_payments(self) -> None:
        """Apply wage payments to reward important atoms"""
        current_time = time.time()
        
        for atom in self.attention_atoms.values():
            # Calculate wage based on importance and recent access
            wage_amount = atom.short_term_importance * self.wage_scaling
            
            # Bonus for recent access
            if atom.access_count > 0:
                wage_amount *= (1 + min(atom.access_count, 10) / 10)
            
            # Apply wage payment
            atom.attention_value = min(1.0, atom.attention_value + wage_amount)
            atom.wage += wage_amount
            
            # Record transaction
            transaction = EconomicTransaction(
                transaction_id=f"wage_{atom.atom_id}_{self.cycle_count}",
                action=EconomicAction.WAGE_PAYMENT,
                source_atom=None,
                target_atom=atom.atom_id,
                amount=wage_amount,
                timestamp=current_time,
                success=True,
                metadata={"access_count": atom.access_count}
            )
            self.economic_transactions.append(transaction)
    
    async def _apply_stimulus_grants(self) -> None:
        """Apply stimulus grants to atoms that need attention boost"""
        current_time = time.time()
        
        # Find atoms that need stimulus (low attention but high importance)
        stimulus_candidates = []
        for atom in self.attention_atoms.values():
            if (atom.attention_value < self.stimulus_threshold and 
                atom.short_term_importance > 0.5):
                stimulus_candidates.append(atom)
        
        # Apply stimulus to selected atoms
        for atom in stimulus_candidates:
            stimulus_amount = min(0.2, atom.short_term_importance * 0.3)
            atom.attention_value = min(1.0, atom.attention_value + stimulus_amount)
            atom.stimulus += stimulus_amount
            
            # Record transaction
            transaction = EconomicTransaction(
                transaction_id=f"stimulus_{atom.atom_id}_{self.cycle_count}",
                action=EconomicAction.STIMULUS_GRANT,
                source_atom=None,
                target_atom=atom.atom_id,
                amount=stimulus_amount,
                timestamp=current_time,
                success=True,
                metadata={"reason": "low_attention_high_importance"}
            )
            self.economic_transactions.append(transaction)
    
    async def _spread_attention(self) -> None:
        """Spread attention across connected atoms"""
        # Get atoms with high attention values
        high_attention_atoms = [
            atom for atom in self.attention_atoms.values()
            if atom.attention_value > 0.7
        ]
        
        # Spread attention from high-attention atoms
        for source_atom in high_attention_atoms:
            # Find connected atoms (simplified - in full implementation would use graph structure)
            connected_atoms = await self._find_connected_atoms(source_atom)
            
            if connected_atoms:
                connection_strengths = [0.5] * len(connected_atoms)  # Simplified uniform strength
                await self.spreading_algorithm.spread_attention(
                    source_atom, connected_atoms, connection_strengths
                )
    
    async def _find_connected_atoms(self, source_atom: AttentionAtom) -> List[AttentionAtom]:
        """Find atoms connected to source atom"""
        # Simplified implementation - in full version would use proper graph traversal
        connected = []
        
        # Connect atoms with similar metadata or temporal proximity
        for atom in self.attention_atoms.values():
            if atom.atom_id != source_atom.atom_id:
                # Simple connection based on creation time proximity
                time_diff = abs(atom.creation_time - source_atom.creation_time)
                if time_diff < 3600:  # Within 1 hour
                    connected.append(atom)
        
        return connected[:5]  # Limit connections
    
    async def _update_attention_focus(self) -> None:
        """Update attention focus based on current attention values"""
        # Find atoms with highest attention values
        sorted_atoms = sorted(
            self.attention_atoms.items(),
            key=lambda x: x[1].attention_value,
            reverse=True
        )
        
        # Update focus atoms
        if sorted_atoms:
            focus_atoms = [atom_id for atom_id, _ in sorted_atoms[:3]]
            focus_strength = sum(atom.attention_value for _, atom in sorted_atoms[:3]) / 3
            
            self.attention_focus = AttentionFocus(
                focus_atoms=focus_atoms,
                focus_strength=focus_strength,
                focus_duration=10.0,
                focus_type=AttentionType.FOCUSED if focus_strength > 0.7 else AttentionType.DIFFUSE,
                background_atoms=[atom_id for atom_id, _ in sorted_atoms[3:8]],
                created_time=time.time(),
                metadata={"cycle": self.cycle_count}
            )
    
    async def _apply_importance_decay(self) -> None:
        """Apply importance decay to all atoms"""
        for atom in self.attention_atoms.values():
            atom.short_term_importance *= self.importance_decay
            atom.long_term_importance *= (self.importance_decay * 0.99)  # Slower decay for LTI
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics for ECAN system"""
        if not self.attention_atoms:
            return
        
        # Calculate attention efficiency
        total_attention = sum(atom.attention_value for atom in self.attention_atoms.values())
        max_possible_attention = len(self.attention_atoms)
        self.performance_metrics['attention_efficiency'] = total_attention / max_possible_attention
        
        # Calculate importance stability
        sti_values = [atom.short_term_importance for atom in self.attention_atoms.values()]
        sti_variance = np.var(sti_values) if sti_values else 0
        self.performance_metrics['importance_stability'] = 1.0 / (1.0 + sti_variance)
        
        # Calculate economic balance
        total_wages = sum(atom.wage for atom in self.attention_atoms.values())
        total_rents = sum(atom.rent for atom in self.attention_atoms.values())
        self.performance_metrics['economic_balance'] = min(total_wages, total_rents) / max(total_wages, total_rents, 1)
        
        # Calculate focus coherence
        if self.attention_focus:
            focus_attention = sum(
                self.attention_atoms[atom_id].attention_value 
                for atom_id in self.attention_focus.focus_atoms
                if atom_id in self.attention_atoms
            )
            self.performance_metrics['focus_coherence'] = focus_attention / len(self.attention_focus.focus_atoms)
    
    async def _get_attention_allocations(self) -> List[Dict[str, Any]]:
        """Get current attention allocations"""
        allocations = []
        
        for atom in self.attention_atoms.values():
            allocation = {
                'atom_id': atom.atom_id,
                'attention_value': atom.attention_value,
                'short_term_importance': atom.short_term_importance,
                'long_term_importance': atom.long_term_importance,
                'wage': atom.wage,
                'rent': atom.rent,
                'stimulus': atom.stimulus,
                'last_accessed': atom.last_accessed,
                'access_count': atom.access_count
            }
            allocations.append(allocation)
        
        return allocations
    
    async def _get_attention_focus_state(self) -> Dict[str, Any]:
        """Get current attention focus state"""
        if not self.attention_focus:
            return {}
        
        return {
            'focus_atoms': self.attention_focus.focus_atoms,
            'focus_strength': self.attention_focus.focus_strength,
            'focus_duration': self.attention_focus.focus_duration,
            'focus_type': self.attention_focus.focus_type.value,
            'background_atoms': self.attention_focus.background_atoms,
            'created_time': self.attention_focus.created_time
        }
    
    async def _get_economic_state(self) -> Dict[str, Any]:
        """Get current economic state"""
        return {
            'total_attention_bank': self.total_attention_bank,
            'total_importance_bank': self.total_importance_bank,
            'rent_frequency': self.rent_frequency,
            'wage_frequency': self.wage_frequency,
            'cycle_count': self.cycle_count,
            'recent_transactions': len(self.economic_transactions),
            'attention_atoms_count': len(self.attention_atoms)
        }
    
    async def meta_learning_update(self, performance_metrics: Dict[str, Any]) -> None:
        """Update ECAN parameters based on performance metrics"""
        # Adapt attention decay based on performance
        attention_efficiency = performance_metrics.get('attention_efficiency', 0.5)
        
        if attention_efficiency < 0.3:
            # Increase stimulus threshold to help struggling atoms
            self.stimulus_threshold = min(0.3, self.stimulus_threshold + 0.05)
        elif attention_efficiency > 0.8:
            # Decrease stimulus threshold for efficiency
            self.stimulus_threshold = max(0.05, self.stimulus_threshold - 0.02)
        
        # Adapt importance decay based on stability
        importance_stability = performance_metrics.get('importance_stability', 0.5)
        
        if importance_stability < 0.5:
            # Slow down decay for more stability
            self.importance_decay = min(0.98, self.importance_decay + 0.01)
        elif importance_stability > 0.9:
            # Speed up decay for more dynamism
            self.importance_decay = max(0.90, self.importance_decay - 0.01)
        
        self.logger.info(f"ECAN parameters updated: stimulus_threshold={self.stimulus_threshold}, importance_decay={self.importance_decay}")
    
    async def get_effectiveness_metrics(self) -> Dict[str, float]:
        """Get effectiveness metrics for ECAN system"""
        return {
            'attention_distribution_entropy': self._calculate_attention_entropy(),
            'focus_stability': self._calculate_focus_stability(),
            'economic_efficiency': self._calculate_economic_efficiency(),
            'resource_utilization': self._calculate_resource_utilization()
        }
    
    def _calculate_attention_entropy(self) -> float:
        """Calculate entropy of attention distribution"""
        if not self.attention_atoms:
            return 0.0
        
        attention_values = [atom.attention_value for atom in self.attention_atoms.values()]
        total_attention = sum(attention_values)
        
        if total_attention == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for attention in attention_values:
            if attention > 0:
                p = attention / total_attention
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_focus_stability(self) -> float:
        """Calculate stability of attention focus"""
        if len(self.attention_history) < 2:
            return 0.0
        
        # Compare focus atoms over time
        recent_focuses = [
            entry['allocation_result']['attention_focus'].get('focus_atoms', [])
            for entry in list(self.attention_history)[-5:]
        ]
        
        if not recent_focuses:
            return 0.0
        
        # Calculate stability as overlap between consecutive focuses
        overlaps = []
        for i in range(1, len(recent_focuses)):
            prev_focus = set(recent_focuses[i-1])
            curr_focus = set(recent_focuses[i])
            
            if prev_focus and curr_focus:
                overlap = len(prev_focus & curr_focus) / len(prev_focus | curr_focus)
                overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _calculate_economic_efficiency(self) -> float:
        """Calculate economic efficiency of attention allocation"""
        if not self.economic_transactions:
            return 0.0
        
        # Calculate ratio of successful transactions
        successful_transactions = sum(1 for t in self.economic_transactions if t.success)
        total_transactions = len(self.economic_transactions)
        
        return successful_transactions / total_transactions
    
    def _calculate_resource_utilization(self) -> float:
        """Calculate resource utilization efficiency"""
        if not self.attention_atoms:
            return 0.0
        
        # Calculate ratio of active atoms (with attention > 0)
        active_atoms = sum(1 for atom in self.attention_atoms.values() if atom.attention_value > 0)
        total_atoms = len(self.attention_atoms)
        
        return active_atoms / total_atoms
    
    def connect_to_memory(self, memory_system: Any) -> None:
        """Connect ECAN to memory system"""
        self.memory_system = memory_system
        self.logger.info("ECAN connected to memory system")
    
    def is_active(self) -> bool:
        """Check if ECAN attention allocation is active"""
        return len(self.attention_atoms) > 0
    
    async def shutdown(self) -> None:
        """Shutdown ECAN attention allocation"""
        self.logger.info("Shutting down ECAN attention allocation...")
        
        # Clear attention atoms
        self.attention_atoms.clear()
        
        # Clear history
        self.attention_history.clear()
        self.economic_transactions.clear()
        
        # Reset state
        self.attention_focus = None
        self.cycle_count = 0
        
        self.logger.info("ECAN attention allocation shutdown complete")