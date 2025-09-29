"""
Self-Awareness System
====================

This module implements self-awareness and introspection mechanisms that enable
the cognitive system to understand its own capabilities, limitations, current
state, and cognitive processes. It provides the foundation for metacognitive
self-monitoring and self-regulation.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from ..interfaces.meta_cognitive_interface import (
    SelfAwarenessInterface, MetaCognitiveCapability
)


class AwarenessLevel(Enum):
    """Levels of self-awareness depth."""
    MINIMAL = 1           # Basic state awareness
    FUNCTIONAL = 2        # Capability awareness
    REFLECTIVE = 3        # Process awareness
    META_REFLECTIVE = 4   # Awareness of awareness


class SelfState(Enum):
    """Different aspects of self-state."""
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"  
    PERFORMANCE = "performance"
    RESOURCES = "resources"
    CAPABILITIES = "capabilities"
    LIMITATIONS = "limitations"
    GOALS = "goals"
    CONTEXT = "context"


@dataclass
class SelfStateSnapshot:
    """Snapshot of current self-state."""
    timestamp: float
    cognitive_load: float
    active_processes: int
    available_resources: Dict[str, float]
    performance_metrics: Dict[str, float]
    capabilities_active: List[str]
    current_limitations: List[str]
    context_awareness: Dict[str, Any]
    confidence_level: float
    meta_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntrospectionResult:
    """Result of an introspective analysis."""
    focus_area: str
    depth_level: AwarenessLevel
    findings: List[str]
    insights: List[str]
    changes_detected: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: float


class SelfAwarenessSystem(SelfAwarenessInterface):
    """
    Implementation of self-awareness and introspection capabilities.
    
    This system provides:
    - Real-time self-state monitoring
    - Introspective analysis of cognitive processes
    - Change detection in self-state over time
    - Self-capability assessment
    - Limitation awareness and adaptation
    - Meta-awareness of awareness processes
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the self-awareness system."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.monitoring_frequency = self.config.get('monitoring_frequency', 1.0)
        self.introspection_depth = self.config.get('introspection_depth', 2)
        self.change_sensitivity = self.config.get('change_sensitivity', 0.1)
        self.maintain_history = self.config.get('maintain_history', True)
        
        # State tracking
        self.current_state: Optional[SelfStateSnapshot] = None
        self.previous_states: List[SelfStateSnapshot] = []
        self.introspection_history: List[IntrospectionResult] = []
        
        # Self-model components
        self.capability_model = CapabilityModel()
        self.limitation_tracker = LimitationTracker()
        self.change_detector = ChangeDetector(self.change_sensitivity)
        self.meta_awareness_monitor = MetaAwarenessMonitor()
        
        # Performance tracking
        self.awareness_quality_metrics: Dict[str, float] = {}
        self.last_full_assessment = 0.0
        
        self.logger.info("Self-awareness system initialized")
    
    def initialize(self) -> bool:
        """Initialize the self-awareness component."""
        try:
            self.capability_model.initialize()
            self.limitation_tracker.initialize()
            self.change_detector.initialize()
            self.meta_awareness_monitor.initialize()
            
            # Initial self-assessment
            self.current_state = self._capture_current_state()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize self-awareness system: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the self-awareness component."""
        return True
    
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of self-awareness capabilities."""
        return [
            MetaCognitiveCapability(
                name="self_state_monitoring",
                description="Real-time monitoring of cognitive state",
                complexity_level=2,
                requires_recursion=False,
                resource_intensive=False
            ),
            MetaCognitiveCapability(
                name="introspective_analysis",
                description="Deep introspective analysis of processes",
                complexity_level=4,
                requires_recursion=True,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="change_detection",
                description="Detection of changes in self-state",
                complexity_level=3,
                requires_recursion=False,
                resource_intensive=False
            ),
            MetaCognitiveCapability(
                name="capability_assessment",
                description="Assessment of own capabilities and limitations",
                complexity_level=3,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="meta_awareness",
                description="Awareness of awareness processes",
                complexity_level=5,
                requires_recursion=True,
                resource_intensive=True
            )
        ]
    
    def process_meta_cognitive_request(self, 
                                     request_type: str,
                                     request_data: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meta-cognitive request."""
        if request_type == "assess_self_state":
            return self.assess_self_state()
        elif request_type == "introspect":
            focus = request_data.get('focus_area') if isinstance(request_data, dict) else None
            return self.introspect(focus)
        elif request_type == "monitor_changes":
            return self.monitor_self_change()
        else:
            return {'error': f'Unknown request type: {request_type}'}
    
    def assess_self_state(self) -> Dict[str, Any]:
        """Assess current self-state and capabilities."""
        assessment = {
            'timestamp': time.time(),
            'state_snapshot': {},
            'capability_assessment': {},
            'limitation_analysis': {},
            'performance_evaluation': {},
            'context_awareness': {}
        }
        
        try:
            # Capture current state
            current_state = self._capture_current_state()
            assessment['state_snapshot'] = self._state_to_dict(current_state)
            
            # Assess capabilities
            capability_assessment = self.capability_model.assess_current_capabilities()
            assessment['capability_assessment'] = capability_assessment
            
            # Analyze limitations
            limitation_analysis = self.limitation_tracker.analyze_current_limitations()
            assessment['limitation_analysis'] = limitation_analysis
            
            # Evaluate performance
            performance_eval = self._evaluate_current_performance()
            assessment['performance_evaluation'] = performance_eval
            
            # Context awareness
            context_awareness = self._assess_context_awareness()
            assessment['context_awareness'] = context_awareness
            
            # Update current state
            self.current_state = current_state
            self.last_full_assessment = time.time()
            
        except Exception as e:
            self.logger.error(f"Error in self-state assessment: {e}")
            assessment['error'] = str(e)
        
        return assessment
    
    def introspect(self, focus_area: Optional[str] = None) -> Dict[str, Any]:
        """Perform introspective analysis."""
        introspection = {
            'focus_area': focus_area or 'general',
            'depth_level': self.introspection_depth,
            'findings': [],
            'insights': [],
            'patterns': [],
            'meta_observations': []
        }
        
        try:
            # Determine focus area
            if not focus_area:
                focus_area = self._determine_introspection_focus()
            
            # Perform focused introspection
            result = self._perform_focused_introspection(focus_area)
            introspection.update(result)
            
            # Meta-level introspection (thinking about introspection)
            if self.introspection_depth >= 3:
                meta_results = self._perform_meta_introspection(result)
                introspection['meta_observations'] = meta_results
            
            # Store result
            introspection_result = IntrospectionResult(
                focus_area=focus_area,
                depth_level=AwarenessLevel(min(self.introspection_depth, 4)),
                findings=introspection['findings'],
                insights=introspection['insights'],
                changes_detected=introspection.get('changes_detected', []),
                recommendations=introspection.get('recommendations', []),
                confidence=introspection.get('confidence', 0.5),
                timestamp=time.time()
            )
            
            if self.maintain_history:
                self.introspection_history.append(introspection_result)
                # Keep history bounded
                if len(self.introspection_history) > 100:
                    self.introspection_history = self.introspection_history[-80:]
            
        except Exception as e:
            self.logger.error(f"Error during introspection: {e}")
            introspection['error'] = str(e)
        
        return introspection
    
    def monitor_self_change(self) -> Dict[str, Any]:
        """Monitor changes in self-state over time."""
        change_analysis = {
            'changes_detected': False,
            'change_details': [],
            'change_magnitude': 0.0,
            'change_patterns': [],
            'adaptation_recommendations': []
        }
        
        try:
            if not self.current_state or not self.previous_states:
                return {'warning': 'insufficient_history_for_change_detection'}
            
            # Detect changes since last state
            changes = self.change_detector.detect_changes(
                self.previous_states[-1] if self.previous_states else None,
                self.current_state
            )
            
            if changes:
                change_analysis['changes_detected'] = True
                change_analysis['change_details'] = changes
                change_analysis['change_magnitude'] = self._calculate_change_magnitude(changes)
                
                # Analyze change patterns
                if len(self.previous_states) >= 3:
                    patterns = self._analyze_change_patterns()
                    change_analysis['change_patterns'] = patterns
                
                # Generate adaptation recommendations
                recommendations = self._generate_adaptation_recommendations(changes)
                change_analysis['adaptation_recommendations'] = recommendations
            
            # Meta-awareness of change monitoring
            meta_monitoring = self.meta_awareness_monitor.monitor_change_awareness(
                change_analysis
            )
            change_analysis['meta_awareness'] = meta_monitoring
            
        except Exception as e:
            self.logger.error(f"Error monitoring self-change: {e}")
            change_analysis['error'] = str(e)
        
        return change_analysis
    
    def update_self_model(self, new_information: Dict[str, Any]) -> bool:
        """Update the self-model with new information."""
        try:
            # Update capability model
            if 'capabilities' in new_information:
                self.capability_model.update(new_information['capabilities'])
            
            # Update limitation tracker
            if 'limitations' in new_information:
                self.limitation_tracker.update(new_information['limitations'])
            
            # Update current state if provided
            if 'state_update' in new_information:
                self._update_current_state(new_information['state_update'])
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating self-model: {e}")
            return False
    
    def get_awareness_quality_metrics(self) -> Dict[str, float]:
        """Get metrics on the quality of self-awareness."""
        return {
            'accuracy': self.awareness_quality_metrics.get('accuracy', 0.5),
            'completeness': self.awareness_quality_metrics.get('completeness', 0.5),
            'consistency': self.awareness_quality_metrics.get('consistency', 0.5),
            'depth': self.awareness_quality_metrics.get('depth', 0.5),
            'responsiveness': self.awareness_quality_metrics.get('responsiveness', 0.5)
        }
    
    def assess_current_state(self) -> Dict[str, Any]:
        """Assess current state for the meta-cognitive core."""
        return {
            'self_awareness_level': self.introspection_depth,
            'current_confidence': self.current_state.confidence_level if self.current_state else 0.5,
            'monitoring_active': True,
            'recent_changes': len(self.change_detector.recent_changes) if hasattr(self.change_detector, 'recent_changes') else 0,
            'introspection_frequency': len(self.introspection_history),
            'quality_metrics': self.get_awareness_quality_metrics()
        }
    
    # Private helper methods
    def _capture_current_state(self) -> SelfStateSnapshot:
        """Capture a snapshot of the current self-state."""
        return SelfStateSnapshot(
            timestamp=time.time(),
            cognitive_load=self._calculate_cognitive_load(),
            active_processes=self._count_active_processes(),
            available_resources=self._assess_available_resources(),
            performance_metrics=self._get_performance_metrics(),
            capabilities_active=self._get_active_capabilities(),
            current_limitations=self._get_current_limitations(),
            context_awareness=self._get_context_awareness(),
            confidence_level=self._calculate_confidence_level()
        )
    
    def _state_to_dict(self, state: SelfStateSnapshot) -> Dict[str, Any]:
        """Convert state snapshot to dictionary."""
        return {
            'timestamp': state.timestamp,
            'cognitive_load': state.cognitive_load,
            'active_processes': state.active_processes,
            'available_resources': state.available_resources,
            'performance_metrics': state.performance_metrics,
            'capabilities_active': state.capabilities_active,
            'current_limitations': state.current_limitations,
            'context_awareness': state.context_awareness,
            'confidence_level': state.confidence_level
        }
    
    def _determine_introspection_focus(self) -> str:
        """Determine what to focus introspection on."""
        # Simple heuristic - focus on area with recent changes or poor performance
        if self.current_state:
            if self.current_state.cognitive_load > 0.8:
                return "cognitive_load"
            elif self.current_state.confidence_level < 0.4:
                return "confidence"
            elif len(self.current_state.current_limitations) > 3:
                return "limitations"
        
        return "general"
    
    def _perform_focused_introspection(self, focus_area: str) -> Dict[str, Any]:
        """Perform introspection focused on a specific area."""
        result = {
            'findings': [],
            'insights': [],
            'confidence': 0.5
        }
        
        if focus_area == "cognitive_load":
            result['findings'].append(f"Current cognitive load: {self.current_state.cognitive_load:.2f}")
            if self.current_state.cognitive_load > 0.7:
                result['insights'].append("High cognitive load may be affecting performance")
                result['recommendations'] = ["Consider reducing concurrent processes"]
        
        elif focus_area == "capabilities":
            active_caps = len(self.current_state.capabilities_active)
            result['findings'].append(f"Currently utilizing {active_caps} capabilities")
            result['insights'].append("Capability utilization appears balanced")
        
        elif focus_area == "limitations":
            limitations = len(self.current_state.current_limitations)
            result['findings'].append(f"Currently aware of {limitations} limitations")
            if limitations > 2:
                result['insights'].append("High limitation awareness may indicate need for adaptation")
        
        return result
    
    def _perform_meta_introspection(self, introspection_result: Dict[str, Any]) -> List[str]:
        """Perform meta-level introspection (introspecting about introspection)."""
        meta_observations = []
        
        findings_count = len(introspection_result.get('findings', []))
        insights_count = len(introspection_result.get('insights', []))
        
        meta_observations.append(f"Generated {findings_count} findings and {insights_count} insights")
        
        if insights_count > findings_count:
            meta_observations.append("High insight-to-finding ratio suggests good analytical depth")
        elif findings_count > insights_count * 2:
            meta_observations.append("Low insight generation - may need deeper analysis")
        
        confidence = introspection_result.get('confidence', 0.5)
        if confidence < 0.4:
            meta_observations.append("Low confidence in introspective analysis")
        elif confidence > 0.8:
            meta_observations.append("High confidence in introspective insights")
        
        return meta_observations
    
    def _calculate_change_magnitude(self, changes: List[Dict[str, Any]]) -> float:
        """Calculate the overall magnitude of detected changes."""
        if not changes:
            return 0.0
        
        total_magnitude = sum(change.get('magnitude', 0.1) for change in changes)
        return min(total_magnitude / len(changes), 1.0)
    
    def _analyze_change_patterns(self) -> List[str]:
        """Analyze patterns in recent changes."""
        patterns = []
        
        if len(self.previous_states) >= 3:
            # Analyze cognitive load trend
            recent_loads = [state.cognitive_load for state in self.previous_states[-3:]]
            if all(recent_loads[i] < recent_loads[i+1] for i in range(len(recent_loads)-1)):
                patterns.append("Increasing cognitive load trend")
            elif all(recent_loads[i] > recent_loads[i+1] for i in range(len(recent_loads)-1)):
                patterns.append("Decreasing cognitive load trend")
            
            # Analyze confidence trend
            recent_confidence = [state.confidence_level for state in self.previous_states[-3:]]
            if all(recent_confidence[i] < recent_confidence[i+1] for i in range(len(recent_confidence)-1)):
                patterns.append("Increasing confidence trend")
        
        return patterns
    
    def _generate_adaptation_recommendations(self, changes: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for adapting to detected changes."""
        recommendations = []
        
        for change in changes:
            change_type = change.get('type', 'unknown')
            magnitude = change.get('magnitude', 0.1)
            
            if change_type == 'cognitive_load' and magnitude > 0.3:
                recommendations.append("Consider redistributing cognitive resources")
            elif change_type == 'performance' and magnitude > 0.2:
                recommendations.append("Analyze and address performance changes")
            elif change_type == 'capabilities' and magnitude > 0.1:
                recommendations.append("Update capability model and strategies")
        
        return recommendations
    
    def _update_current_state(self, state_update: Dict[str, Any]) -> None:
        """Update the current state with new information."""
        if self.current_state:
            # Store previous state
            if self.maintain_history:
                self.previous_states.append(self.current_state)
                if len(self.previous_states) > 50:
                    self.previous_states = self.previous_states[-40:]
        
        # Create new state with updates
        self.current_state = self._capture_current_state()
    
    # Simplified helper methods (would be more sophisticated in full implementation)
    def _calculate_cognitive_load(self) -> float:
        return 0.5  # Simplified
    
    def _count_active_processes(self) -> int:
        return 3  # Simplified
    
    def _assess_available_resources(self) -> Dict[str, float]:
        return {'memory': 0.7, 'processing': 0.6, 'attention': 0.8}
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        return {'accuracy': 0.75, 'speed': 0.65, 'efficiency': 0.7}
    
    def _get_active_capabilities(self) -> List[str]:
        return ['reasoning', 'perception', 'learning']
    
    def _get_current_limitations(self) -> List[str]:
        return ['memory_constraints', 'processing_speed']
    
    def _get_context_awareness(self) -> Dict[str, Any]:
        return {'task_context': 'meta_cognitive', 'environment': 'cognitive_system'}
    
    def _calculate_confidence_level(self) -> float:
        return 0.7  # Simplified
    
    def _evaluate_current_performance(self) -> Dict[str, Any]:
        return {'overall': 0.7, 'recent_trend': 'stable', 'areas_of_concern': []}
    
    def _assess_context_awareness(self) -> Dict[str, Any]:
        return {'situational_awareness': 0.8, 'goal_awareness': 0.7, 'constraint_awareness': 0.6}


# Helper classes for specialized self-awareness functions
class CapabilityModel:
    """Model of system capabilities."""
    
    def __init__(self):
        self.capabilities = {}
    
    def initialize(self) -> bool:
        return True
    
    def assess_current_capabilities(self) -> Dict[str, Any]:
        return {
            'reasoning': {'strength': 0.8, 'availability': True},
            'learning': {'strength': 0.7, 'availability': True},
            'perception': {'strength': 0.6, 'availability': True}
        }
    
    def update(self, capability_updates: Dict[str, Any]) -> None:
        self.capabilities.update(capability_updates)


class LimitationTracker:
    """Tracker for system limitations."""
    
    def __init__(self):
        self.limitations = {}
    
    def initialize(self) -> bool:
        return True
    
    def analyze_current_limitations(self) -> Dict[str, Any]:
        return {
            'memory_constraints': {'severity': 0.3, 'impact': 'moderate'},
            'processing_speed': {'severity': 0.2, 'impact': 'low'}
        }
    
    def update(self, limitation_updates: Dict[str, Any]) -> None:
        self.limitations.update(limitation_updates)


class ChangeDetector:
    """Detector for changes in self-state."""
    
    def __init__(self, sensitivity: float):
        self.sensitivity = sensitivity
        self.recent_changes = []
    
    def initialize(self) -> bool:
        return True
    
    def detect_changes(self, previous_state: Optional[SelfStateSnapshot], 
                      current_state: SelfStateSnapshot) -> List[Dict[str, Any]]:
        """Detect changes between states."""
        changes = []
        
        if not previous_state:
            return changes
        
        # Check cognitive load change
        load_diff = abs(current_state.cognitive_load - previous_state.cognitive_load)
        if load_diff > self.sensitivity:
            changes.append({
                'type': 'cognitive_load',
                'magnitude': load_diff,
                'direction': 'increase' if current_state.cognitive_load > previous_state.cognitive_load else 'decrease'
            })
        
        # Check confidence change
        conf_diff = abs(current_state.confidence_level - previous_state.confidence_level)
        if conf_diff > self.sensitivity:
            changes.append({
                'type': 'confidence',
                'magnitude': conf_diff,
                'direction': 'increase' if current_state.confidence_level > previous_state.confidence_level else 'decrease'
            })
        
        self.recent_changes = changes
        return changes


class MetaAwarenessMonitor:
    """Monitor for meta-awareness processes."""
    
    def __init__(self):
        self.meta_observations = []
    
    def initialize(self) -> bool:
        return True
    
    def monitor_change_awareness(self, change_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor the awareness of change monitoring itself."""
        return {
            'monitoring_quality': 0.7,
            'awareness_of_monitoring': True,
            'meta_insights': ["Change monitoring is functioning within normal parameters"]
        }