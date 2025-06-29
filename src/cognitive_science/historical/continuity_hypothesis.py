from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple

class CognitionLevel(Enum):
    FLUENCY = "fluency"           # Basic processing fluency
    INSIGHT = "insight"           # Individual insights
    FLOW = "flow"                 # Flow state/insight cascades
    MYSTICAL = "mystical"         # Mystical experiences
    TRANSFORMATIVE = "transformative"  # Quantum change experiences

class OptimalGripDomain(Enum):
    PERCEPTUAL = "perceptual"     # Physical object perception
    SOCIAL = "social"             # Social interaction
    COGNITIVE = "cognitive"       # Abstract thinking/categorization
    EXISTENTIAL = "existential"   # Life meaning/purpose

@dataclass
class ProcessingState:
    """Represents the current state of cognitive processing"""
    fluency_level: float  # 0-1 scale
    insight_potential: float
    flow_state: bool
    mystical_intensity: float
    transformation_depth: float

@dataclass
class OptimalGripState:
    """Represents the state of optimal grip across domains"""
    domain: OptimalGripDomain
    gestalt_clarity: float  # Overall pattern clarity
    feature_detail: float   # Detail level perception
    grip_balance: float    # Balance between gestalt/detail
    adaptation_potential: float  # Ability to shift grip

@dataclass
class DisruptiveStrategy:
    """Represents strategies that disrupt normal cognitive processing"""
    name: str
    intensity: float  # How strongly it disrupts
    duration: float   # How long it lasts
    target_domain: OptimalGripDomain
    deautomatization_level: float  # How much it breaks automatic processing

@dataclass
class InvariancePattern:
    """Represents patterns that remain constant through variation"""
    pattern_type: str  # "good" or "bad" invariant
    domain: OptimalGripDomain
    stability: float  # How stable across variations
    adaptiveness: float  # How helpful/harmful to processing
    description: str

class ContinuityManager:
    """Manages the continuity hypothesis and optimal grip dynamics"""
    
    def __init__(self):
        self.processing_state = ProcessingState(
            fluency_level=0.0,
            insight_potential=0.0,
            flow_state=False,
            mystical_intensity=0.0,
            transformation_depth=0.0
        )
        
        self.grip_states: Dict[OptimalGripDomain, OptimalGripState] = {
            domain: OptimalGripState(
                domain=domain,
                gestalt_clarity=0.0,
                feature_detail=0.0,
                grip_balance=0.0,
                adaptation_potential=0.0
            ) for domain in OptimalGripDomain
        }
        
        self.disruptive_strategies: List[DisruptiveStrategy] = []
        self.invariance_patterns: List[InvariancePattern] = []
        self.state_history: List[Tuple[ProcessingState, Dict[OptimalGripDomain, OptimalGripState]]] = []

    def apply_disruptive_strategy(self, strategy: DisruptiveStrategy) -> None:
        """Applies a disruptive strategy to increase variation and reveal invariants"""
        self.disruptive_strategies.append(strategy)
        
        # Disrupt current processing state
        self.processing_state.fluency_level *= (1.0 - strategy.intensity)
        self.processing_state.insight_potential += strategy.intensity * 0.3
        
        # Update grip state for target domain
        grip_state = self.grip_states[strategy.target_domain]
        grip_state.gestalt_clarity *= (1.0 - strategy.deautomatization_level)
        grip_state.adaptation_potential += strategy.intensity * 0.2
        
        # Record state
        self._update_state_history()

    def identify_invariants(self) -> List[InvariancePattern]:
        """Identifies both good and bad invariants through increased variation"""
        invariants = []
        
        for domain in OptimalGripDomain:
            grip_state = self.grip_states[domain]
            
            # Good invariants emerge from high adaptation potential
            if grip_state.adaptation_potential > 0.7:
                invariants.append(InvariancePattern(
                    pattern_type="good",
                    domain=domain,
                    stability=grip_state.grip_balance,
                    adaptiveness=grip_state.adaptation_potential,
                    description=f"Stable pattern in {domain.value} domain"
                ))
            
            # Bad invariants emerge from low adaptation and high gestalt clarity
            if grip_state.adaptation_potential < 0.3 and grip_state.gestalt_clarity > 0.8:
                invariants.append(InvariancePattern(
                    pattern_type="bad",
                    domain=domain,
                    stability=1.0 - grip_state.adaptation_potential,
                    adaptiveness=-0.5,
                    description=f"Rigid pattern in {domain.value} domain"
                ))
        
        self.invariance_patterns = invariants
        return invariants

    def update_processing(self, 
                         fluency_change: float,
                         insight_occurrence: bool,
                         flow_trigger: bool) -> None:
        """Updates the processing state based on cognitive events"""
        # Update fluency with bounded values
        self.processing_state.fluency_level = max(0.0, min(1.0,
            self.processing_state.fluency_level + fluency_change))
        
        # Insight increases potential and can trigger flow
        if insight_occurrence:
            self.processing_state.insight_potential += 0.2
            if self.processing_state.insight_potential >= 0.8:
                flow_trigger = True
        
        # Flow state can lead to mystical experiences
        if flow_trigger and self.processing_state.fluency_level > 0.7:
            self.processing_state.flow_state = True
            self.processing_state.mystical_intensity += 0.1
        
        # High mystical intensity can trigger transformation
        if self.processing_state.mystical_intensity > 0.8:
            self.processing_state.transformation_depth += 0.1
        
        self._update_state_history()

    def adjust_grip(self, 
                   domain: OptimalGripDomain,
                   gestalt_emphasis: float,
                   feature_emphasis: float) -> None:
        """Adjusts the optimal grip for a specific domain"""
        state = self.grip_states[domain]
        
        # Update gestalt and feature clarity with balance
        state.gestalt_clarity = max(0.0, min(1.0, gestalt_emphasis))
        state.feature_detail = max(0.0, min(1.0, feature_emphasis))
        
        # Calculate grip balance - optimal around 0.5
        balance = abs(state.gestalt_clarity - state.feature_detail)
        state.grip_balance = 1.0 - balance
        
        # Adaptation potential increases with balanced grip
        state.adaptation_potential = state.grip_balance * self.processing_state.fluency_level
        
        self._update_state_history()

    def evaluate_continuity(self) -> Dict[str, float]:
        """Evaluates the current state of the continuity hypothesis"""
        return {
            "processing_fluency": self.processing_state.fluency_level,
            "insight_readiness": self.processing_state.insight_potential,
            "flow_intensity": float(self.processing_state.flow_state),
            "mystical_depth": self.processing_state.mystical_intensity,
            "transformation_level": self.processing_state.transformation_depth,
            "average_grip_balance": sum(
                state.grip_balance for state in self.grip_states.values()
            ) / len(self.grip_states),
            "overall_adaptation": sum(
                state.adaptation_potential for state in self.grip_states.values()
            ) / len(self.grip_states)
        }

    def get_recommendations(self) -> List[str]:
        """Provides recommendations for improving cognitive processing"""
        recommendations = []
        
        # Check processing state
        if self.processing_state.fluency_level < 0.4:
            recommendations.append("Practice basic skills to improve fluency")
        if self.processing_state.insight_potential < 0.3:
            recommendations.append("Engage in insight-triggering activities")
        if not self.processing_state.flow_state:
            recommendations.append("Create conditions for flow state")
            
        # Check grip states
        for domain, state in self.grip_states.items():
            if state.grip_balance < 0.4:
                recommendations.append(
                    f"Improve balance between gestalt and detail in {domain.value} domain"
                )
            if state.adaptation_potential < 0.3:
                recommendations.append(
                    f"Increase flexibility in {domain.value} domain"
                )
                
        return recommendations

    def _update_state_history(self) -> None:
        """Records current state in history"""
        self.state_history.append((
            ProcessingState(**vars(self.processing_state)),
            {k: OptimalGripState(**vars(v)) for k, v in self.grip_states.items()}
        )) 