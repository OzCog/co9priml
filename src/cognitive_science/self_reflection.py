from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import uuid
from .math_utils import clip, mean, std, entropy, weighted_average, sigmoid, softmax, confidence_interval

class ReflectionMode(Enum):
    """Modes of self-reflection"""
    NARRATIVE = "narrative"   # Story-based reflection
    DIALOGIC = "dialogic"     # Conversational reflection
    EMBODIED = "embodied"     # Sensorimotor reflection
    SYMBOLIC = "symbolic"     # Symbol-mediated reflection

class MeaningDimension(Enum):
    """Dimensions of meaning in life"""
    PURPOSE = "purpose"       # Direction and goals
    VALUE = "value"          # Worth and significance
    COHERENCE = "coherence"  # Integration and sense
    AGENCY = "agency"        # Self-determination

class CognitiveState(Enum):
    """States of cognitive processing"""
    ACTIVE = "active"
    LEARNING = "learning"
    REFLECTING = "reflecting"
    OPTIMIZING = "optimizing"
    IDLE = "idle"

class BiasType(Enum):
    """Types of cognitive biases"""
    CONFIRMATION = "confirmation"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    OVERCONFIDENCE = "overconfidence"
    HINDSIGHT = "hindsight"
    FRAMING = "framing"
    REPRESENTATIVENESS = "representativeness"

class MetricType(Enum):
    """Types of performance metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    CONFIDENCE = "confidence"
    EFFICIENCY = "efficiency"
    COHERENCE = "coherence"
    DEPTH = "depth"

@dataclass
class ReflectiveState:
    """Represents a state of self-reflection"""
    mode: ReflectionMode
    content: Dict
    depth: float
    integration: float
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
@dataclass
class MeaningStructure:
    """Represents a structure of meaning"""
    dimensions: List[MeaningDimension]
    patterns: Dict
    coherence: float
    vitality: float
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class CognitiveMetrics:
    """Metrics for cognitive performance"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    confidence: float = 0.0
    efficiency: float = 0.0
    coherence: float = 0.0
    depth: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'confidence': self.confidence,
            'efficiency': self.efficiency,
            'coherence': self.coherence,
            'depth': self.depth,
            'timestamp': self.timestamp
        }

@dataclass
class DecisionQuality:
    """Assessment of decision quality"""
    decision_id: str
    confidence: float
    expected_outcome: float
    actual_outcome: Optional[float] = None
    factors: Dict[str, float] = field(default_factory=dict)
    biases_detected: List[BiasType] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def compute_quality_score(self) -> float:
        """Compute overall decision quality score"""
        if self.actual_outcome is None:
            return self.confidence * self.expected_outcome
        
        # Compare expected vs actual outcome
        outcome_accuracy = 1.0 - abs(self.expected_outcome - self.actual_outcome)
        confidence_calibration = 1.0 - abs(self.confidence - outcome_accuracy)
        bias_penalty = len(self.biases_detected) * 0.1
        
        return clip(outcome_accuracy * confidence_calibration - bias_penalty, 0.0, 1.0)

@dataclass
class BiasDetection:
    """Detected cognitive bias"""
    bias_type: BiasType
    confidence: float
    evidence: Dict[str, Any]
    severity: float
    mitigation_strategy: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationFeedback:
    """Feedback for self-optimization"""
    component: str
    metric: MetricType
    current_value: float
    target_value: float
    improvement_strategy: str
    priority: float
    timestamp: float = field(default_factory=time.time)

class SelfReflectionCore:
    """Handles self-reflection and meaning-making with meta-cognitive monitoring.
    
    Implements comprehensive self-reflection and introspection capabilities that
    allow the system to monitor, evaluate, and optimize its own cognitive processes.
    Includes meta-cognitive awareness, performance monitoring, and self-improvement
    mechanisms.
    """
    
    def __init__(self):
        self.reflective_states: Dict[str, ReflectiveState] = {}
        self.meaning_structures: Dict[str, MeaningStructure] = {}
        self.integration_threshold = 0.7
        
        # Meta-cognitive monitoring components
        self.cognitive_state: CognitiveState = CognitiveState.IDLE
        self.performance_metrics: Dict[str, CognitiveMetrics] = {}
        self.decision_history: Dict[str, DecisionQuality] = {}
        self.bias_detections: List[BiasDetection] = []
        self.optimization_feedback: List[OptimizationFeedback] = []
        
        # Performance tracking
        self.metrics_history: Dict[str, List[float]] = {}
        self.state_transitions: List[Tuple[CognitiveState, CognitiveState, float]] = []
        
        # Self-optimization parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.8
        self.bias_sensitivity = 0.3
        self.optimization_frequency = 10  # cycles between optimization
        self.cycle_count = 0
        
        # Introspection capabilities
        self.introspection_depth = 0.5
        self.self_awareness_level = 0.5
        self.meta_cognitive_capacity = 0.7
    # Meta-cognitive monitoring methods
    
    def monitor_cognitive_state(self, context: Dict[str, Any]) -> CognitiveState:
        """Monitor and update cognitive state based on context"""
        previous_state = self.cognitive_state
        
        # Determine new state based on context
        if context.get('learning_active', False):
            self.cognitive_state = CognitiveState.LEARNING
        elif context.get('reflection_triggered', False):
            self.cognitive_state = CognitiveState.REFLECTING
        elif context.get('optimization_needed', False):
            self.cognitive_state = CognitiveState.OPTIMIZING
        elif context.get('active_processing', False):
            self.cognitive_state = CognitiveState.ACTIVE
        else:
            self.cognitive_state = CognitiveState.IDLE
        
        # Track state transitions
        if previous_state != self.cognitive_state:
            self.state_transitions.append((previous_state, self.cognitive_state, time.time()))
        
        return self.cognitive_state
    
    def perform_introspection(self, depth: float = None) -> Dict[str, Any]:
        """Perform introspective analysis of cognitive processes"""
        if depth is None:
            depth = self.introspection_depth
            
        introspection_report = {
            'cognitive_state': self.cognitive_state.value,
            'self_awareness_level': self.self_awareness_level,
            'meta_cognitive_capacity': self.meta_cognitive_capacity,
            'active_reflections': len(self.reflective_states),
            'meaning_structures': len(self.meaning_structures),
            'recent_decisions': len([d for d in self.decision_history.values() 
                                   if time.time() - d.timestamp < 300]),  # 5 minutes
            'bias_detections': len(self.bias_detections),
            'optimization_feedback': len(self.optimization_feedback),
            'depth_achieved': depth
        }
        
        if depth > 0.5:
            # Deep introspection includes performance analysis
            introspection_report.update(self._analyze_performance_patterns())
            
        if depth > 0.8:
            # Very deep introspection includes meta-cognitive analysis
            introspection_report.update(self._analyze_meta_cognitive_patterns())
        
        return introspection_report
    
    def assess_decision_quality(self, decision_id: str, expected_outcome: float,
                               context: Dict[str, Any]) -> DecisionQuality:
        """Assess the quality of a decision"""
        confidence = self._compute_decision_confidence(context)
        factors = self._extract_decision_factors(context)
        biases = self._detect_decision_biases(context)
        
        decision_quality = DecisionQuality(
            decision_id=decision_id,
            confidence=confidence,
            expected_outcome=expected_outcome,
            factors=factors,
            biases_detected=biases
        )
        
        self.decision_history[decision_id] = decision_quality
        return decision_quality
    
    def update_decision_outcome(self, decision_id: str, actual_outcome: float) -> bool:
        """Update decision with actual outcome for learning"""
        if decision_id not in self.decision_history:
            return False
        
        decision = self.decision_history[decision_id]
        decision.actual_outcome = actual_outcome
        
        # Learn from decision quality
        quality_score = decision.compute_quality_score()
        self._update_decision_learning(decision, quality_score)
        
        return True
    
    def detect_cognitive_biases(self, context: Dict[str, Any]) -> List[BiasDetection]:
        """Detect cognitive biases in current processing"""
        detected_biases = []
        
        # Check for confirmation bias
        if self._check_confirmation_bias(context):
            detected_biases.append(BiasDetection(
                bias_type=BiasType.CONFIRMATION,
                confidence=0.7,
                evidence=context.get('confirmation_evidence', {}),
                severity=0.6,
                mitigation_strategy="Seek contradictory evidence"
            ))
        
        # Check for anchoring bias
        if self._check_anchoring_bias(context):
            detected_biases.append(BiasDetection(
                bias_type=BiasType.ANCHORING,
                confidence=0.8,
                evidence=context.get('anchoring_evidence', {}),
                severity=0.7,
                mitigation_strategy="Consider alternative reference points"
            ))
        
        # Check for availability bias
        if self._check_availability_bias(context):
            detected_biases.append(BiasDetection(
                bias_type=BiasType.AVAILABILITY,
                confidence=0.6,
                evidence=context.get('availability_evidence', {}),
                severity=0.5,
                mitigation_strategy="Seek broader data sources"
            ))
        
        # Check for overconfidence bias
        if self._check_overconfidence_bias(context):
            detected_biases.append(BiasDetection(
                bias_type=BiasType.OVERCONFIDENCE,
                confidence=0.9,
                evidence=context.get('overconfidence_evidence', {}),
                severity=0.8,
                mitigation_strategy="Implement confidence calibration"
            ))
        
        # Store detected biases
        self.bias_detections.extend(detected_biases)
        
        return detected_biases
    
    def generate_optimization_feedback(self) -> List[OptimizationFeedback]:
        """Generate feedback for self-optimization"""
        feedback = []
        
        # Analyze performance metrics
        for component, metrics in self.performance_metrics.items():
            if metrics.accuracy < 0.8:
                feedback.append(OptimizationFeedback(
                    component=component,
                    metric=MetricType.ACCURACY,
                    current_value=metrics.accuracy,
                    target_value=0.8,
                    improvement_strategy="Increase training data quality",
                    priority=0.9
                ))
            
            if metrics.confidence < 0.7:
                feedback.append(OptimizationFeedback(
                    component=component,
                    metric=MetricType.CONFIDENCE,
                    current_value=metrics.confidence,
                    target_value=0.7,
                    improvement_strategy="Implement confidence calibration",
                    priority=0.8
                ))
            
            if metrics.efficiency < 0.6:
                feedback.append(OptimizationFeedback(
                    component=component,
                    metric=MetricType.EFFICIENCY,
                    current_value=metrics.efficiency,
                    target_value=0.6,
                    improvement_strategy="Optimize processing algorithms",
                    priority=0.7
                ))
        
        # Analyze decision quality
        recent_decisions = [d for d in self.decision_history.values() 
                          if time.time() - d.timestamp < 3600]  # 1 hour
        if recent_decisions:
            avg_quality = mean([d.compute_quality_score() for d in recent_decisions])
            if avg_quality < 0.7:
                feedback.append(OptimizationFeedback(
                    component="decision_making",
                    metric=MetricType.ACCURACY,
                    current_value=avg_quality,
                    target_value=0.7,
                    improvement_strategy="Improve decision calibration",
                    priority=0.85
                ))
        
        # Analyze bias frequency
        recent_biases = [b for b in self.bias_detections 
                        if time.time() - b.timestamp < 3600]  # 1 hour
        if len(recent_biases) > 5:
            feedback.append(OptimizationFeedback(
                component="bias_detection",
                metric=MetricType.ACCURACY,
                current_value=1.0 - len(recent_biases) / 100,
                target_value=0.95,
                improvement_strategy="Implement stronger bias mitigation",
                priority=0.75
            ))
        
        self.optimization_feedback.extend(feedback)
        return feedback
    
    def apply_optimization_feedback(self, feedback: List[OptimizationFeedback]) -> Dict[str, Any]:
        """Apply optimization feedback to improve performance"""
        optimization_results = {
            'applied_optimizations': [],
            'performance_improvements': {},
            'updated_parameters': {}
        }
        
        for fb in feedback:
            if fb.priority > 0.7:
                if fb.metric == MetricType.CONFIDENCE:
                    # Adjust confidence calibration
                    self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
                    optimization_results['updated_parameters']['confidence_threshold'] = self.confidence_threshold
                
                elif fb.metric == MetricType.ACCURACY:
                    # Adjust learning rate
                    self.learning_rate = min(0.3, self.learning_rate + 0.02)
                    optimization_results['updated_parameters']['learning_rate'] = self.learning_rate
                
                elif fb.metric == MetricType.EFFICIENCY:
                    # Adjust processing parameters
                    self.optimization_frequency = max(5, self.optimization_frequency - 1)
                    optimization_results['updated_parameters']['optimization_frequency'] = self.optimization_frequency
                
                optimization_results['applied_optimizations'].append({
                    'component': fb.component,
                    'metric': fb.metric.value,
                    'strategy': fb.improvement_strategy
                })
        
        return optimization_results
    
    def estimate_confidence(self, context: Dict[str, Any]) -> float:
        """Estimate confidence in cognitive outputs"""
        factors = []
        
        # Factor 1: Data quality
        data_quality = context.get('data_quality', 0.5)
        factors.append(data_quality)
        
        # Factor 2: Process reliability
        process_reliability = context.get('process_reliability', 0.7)
        factors.append(process_reliability)
        
        # Factor 3: Historical accuracy
        if self.performance_metrics:
            avg_accuracy = mean([m.accuracy for m in self.performance_metrics.values()])
            factors.append(avg_accuracy)
        
        # Factor 4: Bias detection impact
        recent_biases = [b for b in self.bias_detections 
                        if time.time() - b.timestamp < 300]  # 5 minutes
        bias_impact = 1.0 - len(recent_biases) * 0.1
        factors.append(clip(bias_impact, 0.0, 1.0))
        
        # Factor 5: Coherence with existing knowledge
        coherence = context.get('coherence', 0.6)
        factors.append(coherence)
        
        # Compute weighted confidence
        if not factors:
            confidence = 0.5  # Default confidence
        else:
            confidence = weighted_average(factors, [0.2, 0.25, 0.25, 0.15, 0.15][:len(factors)])
        
        # Apply confidence calibration
        calibrated_confidence = self._calibrate_confidence(confidence, context)
        
        return clip(calibrated_confidence, 0.1, 1.0)  # Ensure non-zero confidence
    
    def perform_reflective_learning(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reflective learning from experience"""
        learning_results = {
            'insights_gained': [],
            'patterns_identified': [],
            'adaptations_made': [],
            'meta_learning_updates': {}
        }
        
        # Extract insights from experience
        insights = self._extract_insights(experience)
        learning_results['insights_gained'] = insights
        
        # Identify patterns
        patterns = self._identify_patterns(experience)
        learning_results['patterns_identified'] = patterns
        
        # Make adaptations
        adaptations = self._make_adaptations(experience, insights, patterns)
        learning_results['adaptations_made'] = adaptations
        
        # Update meta-learning
        meta_updates = self._update_meta_learning(experience, insights, patterns)
        learning_results['meta_learning_updates'] = meta_updates
        
        return learning_results
    
    def track_performance_metrics(self, component: str, metrics: CognitiveMetrics) -> None:
        """Track performance metrics for a component"""
        self.performance_metrics[component] = metrics
        
        # Update metrics history
        for metric_name, value in metrics.to_dict().items():
            if metric_name == 'timestamp':
                continue
            key = f"{component}_{metric_name}"
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
            
            # Keep only recent history
            if len(self.metrics_history[key]) > 100:
                self.metrics_history[key] = self.metrics_history[key][-100:]
    
    # Core reflection methods (enhanced)
    
    def engage_reflection(self,
                         content: Dict,
                         mode: ReflectionMode,
                         intensity: float = 0.5) -> Optional[ReflectiveState]:
        """Engage in self-reflection.
        
        Args:
            content: Content to reflect on
            mode: Mode of reflection
            intensity: Desired intensity
            
        Returns:
            Reflective state if successful
        """
        # Validate content
        if not self._validate_content(content):
            return None
            
        # Compute reflection depth
        depth = self._compute_depth(content, mode)
        
        # Compute integration
        integration = self._compute_integration(content, mode)
        
        # Create reflective state
        state = ReflectiveState(
            mode=mode,
            content=content,
            depth=depth,
            integration=integration
        )
        
        # Store state
        key = f"{mode.value}_{len(self.reflective_states)}"
        self.reflective_states[key] = state
        
        return state
        
    def construct_meaning(self,
                         dimensions: List[MeaningDimension],
                         patterns: Dict) -> Optional[MeaningStructure]:
        """Construct meaning structure.
        
        Args:
            dimensions: Meaning dimensions to include
            patterns: Meaning patterns
            
        Returns:
            Meaning structure if successful
        """
        # Validate dimensions and patterns
        if not self._validate_meaning(dimensions, patterns):
            return None
            
        # Compute coherence
        coherence = self._compute_coherence(dimensions, patterns)
        
        # Compute vitality
        vitality = self._compute_vitality(dimensions, patterns)
        
        # Create meaning structure
        structure = MeaningStructure(
            dimensions=dimensions,
            patterns=patterns,
            coherence=coherence,
            vitality=vitality
        )
        
        # Store structure
        key = f"meaning_{len(self.meaning_structures)}"
        self.meaning_structures[key] = structure
        
        return structure
        
    def integrate_reflection(self,
                           state_key: str,
                           structure_key: str) -> Tuple[bool, Optional[Dict]]:
        """Integrate reflection with meaning.
        
        Args:
            state_key: Key of reflective state
            structure_key: Key of meaning structure
            
        Returns:
            Tuple of (success, effects)
        """
        # Get state and structure
        if (state_key not in self.reflective_states or
            structure_key not in self.meaning_structures):
            return False, None
            
        state = self.reflective_states[state_key]
        structure = self.meaning_structures[structure_key]
        
        # Check integration potential
        if state.integration < self.integration_threshold:
            return False, None
            
        # Generate integration effects
        effects = self._generate_effects(state, structure)
        
        return True, effects
        
    def _validate_content(self, content: Dict) -> bool:
        """Validate reflection content."""
        # Check required elements
        has_elements = (
            "focus" in content and
            "context" in content and
            "patterns" in content
        )
        
        # Check coherence
        is_coherent = self._check_coherence(content)
        
        return has_elements and is_coherent
        
    def _compute_depth(self,
                      content: Dict,
                      mode: ReflectionMode) -> float:
        """Compute depth of reflection."""
        # Factors affecting depth:
        # - Pattern complexity
        # - Self-reference level
        # - Integration potential
        
        # Get base factors
        complexity = self._compute_complexity(content)
        self_reference = self._compute_self_reference(content)
        integration = self._compute_integration(content, mode)
        
        # Compute weighted depth
        depth = (0.4 * complexity +
                0.3 * self_reference +
                0.3 * integration)
        
        return clip(depth, 0, 1)
        
    def _compute_integration(self,
                           content: Dict,
                           mode: ReflectionMode) -> float:
        """Compute integration potential."""
        # Factors affecting integration:
        # - Pattern coherence
        # - Mode alignment
        # - Transformative potential
        
        # Get base factors
        coherence = self._check_coherence(content)
        alignment = self._check_mode_alignment(content, mode)
        potential = self._compute_potential(content)
        
        # Compute weighted integration
        integration = (0.4 * coherence +
                     0.3 * alignment +
                     0.3 * potential)
        
        return clip(integration, 0, 1)
        
    def _validate_meaning(self,
                         dimensions: List[MeaningDimension],
                         patterns: Dict) -> bool:
        """Validate meaning components."""
        # Check dimension coverage
        has_dimensions = len(dimensions) >= 2
        
        # Check pattern elements
        has_patterns = (
            "core" in patterns and
            "relations" in patterns and
            "dynamics" in patterns
        )
        
        return has_dimensions and has_patterns
        
    def _compute_coherence(self,
                          dimensions: List[MeaningDimension],
                          patterns: Dict) -> float:
        """Compute meaning coherence."""
        # Implementation would compute:
        # - Dimension integration
        # - Pattern consistency
        # - Dynamic stability
        return 0.8  # Placeholder
        
    def _compute_vitality(self,
                         dimensions: List[MeaningDimension],
                         patterns: Dict) -> float:
        """Compute meaning vitality."""
        # Implementation would compute:
        # - Growth potential
        # - Adaptability
        # - Resonance
        return 0.7  # Placeholder
        
    def _generate_effects(self,
                         state: ReflectiveState,
                         structure: MeaningStructure) -> Dict:
        """Generate integration effects."""
        effects = {
            "transformations": [],
            "coherence": 0.0,
            "vitality": 0.0
        }
        
        # Add mode-specific effects
        if state.mode == ReflectionMode.NARRATIVE:
            effects["transformations"].append({
                "type": "narrative_integration",
                "intensity": 0.8 * state.integration
            })
            
        elif state.mode == ReflectionMode.DIALOGIC:
            effects["transformations"].append({
                "type": "dialogic_opening",
                "intensity": 0.7 * state.integration
            })
            
        elif state.mode == ReflectionMode.EMBODIED:
            effects["transformations"].append({
                "type": "embodied_resonance",
                "intensity": 0.9 * state.integration
            })
            
        elif state.mode == ReflectionMode.SYMBOLIC:
            effects["transformations"].append({
                "type": "symbolic_transformation",
                "intensity": 0.8 * state.integration
            })
            
        # Compute overall effects
        effects["coherence"] = (
            0.6 * structure.coherence +
            0.4 * state.integration
        )
        
        effects["vitality"] = (
            0.7 * structure.vitality +
            0.3 * state.depth
        )
        
        return effects
        
    def _compute_complexity(self, content: Dict) -> float:
        """Compute content complexity."""
        # Implementation would compute:
        # - Pattern complexity
        # - Relation density
        # - Dynamic richness
        return 0.7  # Placeholder
        
    def _compute_self_reference(self, content: Dict) -> float:
        """Compute degree of self-reference."""
        # Implementation would compute:
        # - Self-model involvement
        # - Recursive depth
        # - Identity relevance
        return 0.6  # Placeholder
        
    def _compute_potential(self, content: Dict) -> float:
        """Compute transformative potential."""
        # Implementation would compute:
        # - Growth capacity
        # - Integration potential
        # - Development space
        return 0.8  # Placeholder
        
    def _check_mode_alignment(self,
                            content: Dict,
                            mode: ReflectionMode) -> float:
        """Check alignment between content and mode."""
        # Implementation would check:
        # - Mode appropriateness
        # - Content compatibility
        # - Process alignment
        return 0.7  # Placeholder
        
    def _check_coherence(self, content: Dict) -> float:
        """Check content coherence."""
        # Implementation would check:
        # - Pattern consistency
        # - Relation validity
        # - Dynamic stability
        return 0.8  # Placeholder
    
    # Helper methods for meta-cognitive monitoring
    
    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns for deep introspection"""
        patterns = {
            'performance_trends': {},
            'efficiency_patterns': {},
            'accuracy_correlations': {}
        }
        
        for key, history in self.metrics_history.items():
            if len(history) > 5:
                recent_trend = history[-5:]
                if len(recent_trend) > 1:
                    trend_direction = 'improving' if recent_trend[-1] > recent_trend[0] else 'declining'
                    patterns['performance_trends'][key] = {
                        'direction': trend_direction,
                        'magnitude': abs(recent_trend[-1] - recent_trend[0]),
                        'stability': 1.0 - std(recent_trend)
                    }
        
        return patterns
    
    def _analyze_meta_cognitive_patterns(self) -> Dict[str, Any]:
        """Analyze meta-cognitive patterns for very deep introspection"""
        meta_patterns = {
            'state_transition_patterns': {},
            'decision_quality_trends': {},
            'bias_frequency_patterns': {},
            'learning_effectiveness': {}
        }
        
        # Analyze state transitions
        if len(self.state_transitions) > 3:
            recent_transitions = self.state_transitions[-10:]
            transition_types = {}
            for prev_state, curr_state, _ in recent_transitions:
                transition_key = f"{prev_state.value}->{curr_state.value}"
                transition_types[transition_key] = transition_types.get(transition_key, 0) + 1
            
            meta_patterns['state_transition_patterns'] = transition_types
        
        # Analyze decision quality trends
        if self.decision_history:
            recent_decisions = list(self.decision_history.values())[-10:]
            quality_scores = [d.compute_quality_score() for d in recent_decisions if d.actual_outcome is not None]
            if quality_scores:
                meta_patterns['decision_quality_trends'] = {
                    'average_quality': mean(quality_scores),
                    'quality_stability': 1.0 - std(quality_scores),
                    'improvement_rate': (quality_scores[-1] - quality_scores[0]) / len(quality_scores) if len(quality_scores) > 1 else 0
                }
        
        return meta_patterns
    
    def _compute_decision_confidence(self, context: Dict[str, Any]) -> float:
        """Compute confidence in a decision"""
        factors = []
        
        # Information quality
        info_quality = context.get('information_quality', 0.7)
        factors.append(info_quality)
        
        # Time pressure (inverse relationship)
        time_pressure = context.get('time_pressure', 0.3)
        factors.append(1.0 - time_pressure)
        
        # Complexity (inverse relationship)
        complexity = context.get('complexity', 0.5)
        factors.append(1.0 - complexity * 0.5)
        
        # Historical accuracy
        if self.performance_metrics:
            avg_accuracy = mean([m.accuracy for m in self.performance_metrics.values()])
            factors.append(avg_accuracy)
        
        return mean(factors)
    
    def _extract_decision_factors(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract factors that influence decision quality"""
        return {
            'information_quality': context.get('information_quality', 0.7),
            'time_pressure': context.get('time_pressure', 0.3),
            'complexity': context.get('complexity', 0.5),
            'uncertainty': context.get('uncertainty', 0.4),
            'stakes': context.get('stakes', 0.5),
            'expertise_match': context.get('expertise_match', 0.6)
        }
    
    def _detect_decision_biases(self, context: Dict[str, Any]) -> List[BiasType]:
        """Detect biases in decision making"""
        biases = []
        
        # Simple heuristics for bias detection
        if context.get('confirmation_seeking', False):
            biases.append(BiasType.CONFIRMATION)
        
        if context.get('first_option_preference', False):
            biases.append(BiasType.ANCHORING)
        
        if context.get('recent_example_focus', False):
            biases.append(BiasType.AVAILABILITY)
        
        confidence = context.get('stated_confidence', 0.5)
        objective_difficulty = context.get('objective_difficulty', 0.5)
        if confidence > objective_difficulty + 0.2:
            biases.append(BiasType.OVERCONFIDENCE)
        
        return biases
    
    def _update_decision_learning(self, decision: DecisionQuality, quality_score: float) -> None:
        """Update learning from decision outcomes"""
        # Adjust confidence calibration
        confidence_error = abs(decision.confidence - quality_score)
        if confidence_error > 0.2:
            self.confidence_threshold = max(0.3, self.confidence_threshold - 0.02)
        
        # Adjust bias sensitivity
        if decision.biases_detected:
            self.bias_sensitivity = min(1.0, self.bias_sensitivity + 0.05)
    
    def _check_confirmation_bias(self, context: Dict[str, Any]) -> bool:
        """Check for confirmation bias"""
        return (
            context.get('confirmation_seeking', False) or
            context.get('contradictory_evidence_ignored', False) or
            context.get('selective_attention', False)
        )
    
    def _check_anchoring_bias(self, context: Dict[str, Any]) -> bool:
        """Check for anchoring bias"""
        return (
            context.get('first_option_preference', False) or
            context.get('initial_value_fixation', False) or
            context.get('insufficient_adjustment', False)
        )
    
    def _check_availability_bias(self, context: Dict[str, Any]) -> bool:
        """Check for availability bias"""
        return (
            context.get('recent_example_focus', False) or
            context.get('memorable_instance_weight', False) or
            context.get('ease_of_recall_bias', False)
        )
    
    def _check_overconfidence_bias(self, context: Dict[str, Any]) -> bool:
        """Check for overconfidence bias"""
        confidence = context.get('stated_confidence', 0.5)
        objective_difficulty = context.get('objective_difficulty', 0.5)
        calibration_history = context.get('calibration_history', [])
        
        # Check if confidence consistently exceeds accuracy
        if calibration_history:
            confidence_accuracy_gap = mean([abs(c - a) for c, a in calibration_history])
            return confidence_accuracy_gap > 0.15
        
        return confidence > objective_difficulty + 0.2
    
    def _calibrate_confidence(self, raw_confidence: float, context: Dict[str, Any]) -> float:
        """Calibrate confidence based on historical accuracy"""
        if not self.performance_metrics:
            return raw_confidence
        
        # Get historical accuracy
        historical_accuracy = mean([m.accuracy for m in self.performance_metrics.values()])
        
        # Adjust confidence based on historical performance
        calibration_factor = historical_accuracy / 0.8  # Assuming 0.8 as target accuracy
        calibrated_confidence = raw_confidence * calibration_factor
        
        # Apply additional calibration based on context
        complexity_penalty = context.get('complexity', 0.3) * 0.1
        uncertainty_penalty = context.get('uncertainty', 0.4) * 0.1
        
        final_confidence = calibrated_confidence - complexity_penalty - uncertainty_penalty
        
        return clip(final_confidence, 0.1, 0.9)
    
    def _extract_insights(self, experience: Dict[str, Any]) -> List[str]:
        """Extract insights from experience"""
        insights = []
        
        # Extract performance insights
        if 'performance_data' in experience:
            perf_data = experience['performance_data']
            if perf_data.get('accuracy', 0) > 0.9:
                insights.append("High accuracy achieved - current approach is effective")
            elif perf_data.get('accuracy', 0) < 0.6:
                insights.append("Low accuracy detected - approach needs revision")
        
        # Extract pattern insights
        if 'patterns' in experience:
            patterns = experience['patterns']
            if len(patterns) > 5:
                insights.append("Multiple patterns detected - complex situation requiring careful analysis")
            elif len(patterns) < 2:
                insights.append("Few patterns detected - may need broader perspective")
        
        return insights
    
    def _identify_patterns(self, experience: Dict[str, Any]) -> List[str]:
        """Identify patterns in experience"""
        patterns = []
        
        # Identify performance patterns
        if 'performance_sequence' in experience:
            sequence = experience['performance_sequence']
            if len(sequence) > 3:
                if all(sequence[i] <= sequence[i+1] for i in range(len(sequence)-1)):
                    patterns.append("Consistent improvement pattern")
                elif all(sequence[i] >= sequence[i+1] for i in range(len(sequence)-1)):
                    patterns.append("Consistent decline pattern")
                else:
                    patterns.append("Irregular performance pattern")
        
        # Identify decision patterns
        if 'decision_sequence' in experience:
            decisions = experience['decision_sequence']
            if len(decisions) > 2:
                avg_confidence = mean([d.get('confidence', 0.5) for d in decisions])
                if avg_confidence > 0.8:
                    patterns.append("High confidence decision pattern")
                elif avg_confidence < 0.4:
                    patterns.append("Low confidence decision pattern")
        
        return patterns
    
    def _make_adaptations(self, experience: Dict[str, Any], insights: List[str], patterns: List[str]) -> List[str]:
        """Make adaptations based on insights and patterns"""
        adaptations = []
        
        # Adapt based on insights
        for insight in insights:
            if "low accuracy" in insight.lower():
                adaptations.append("Increased learning rate for accuracy improvement")
                self.learning_rate = min(0.3, self.learning_rate + 0.05)
            elif "high accuracy" in insight.lower():
                adaptations.append("Maintained current approach for sustained performance")
        
        # Adapt based on patterns
        for pattern in patterns:
            if "improvement pattern" in pattern.lower():
                adaptations.append("Reinforced successful strategies")
            elif "decline pattern" in pattern.lower():
                adaptations.append("Implemented corrective measures")
                self.optimization_frequency = max(3, self.optimization_frequency - 1)
        
        return adaptations
    
    def _update_meta_learning(self, experience: Dict[str, Any], insights: List[str], patterns: List[str]) -> Dict[str, Any]:
        """Update meta-learning based on experience analysis"""
        updates = {}
        
        # Update introspection depth based on insight quality
        if len(insights) > 3:
            self.introspection_depth = min(1.0, self.introspection_depth + 0.05)
            updates['introspection_depth'] = self.introspection_depth
        
        # Update self-awareness based on pattern recognition
        if len(patterns) > 2:
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.03)
            updates['self_awareness_level'] = self.self_awareness_level
        
        # Update meta-cognitive capacity based on adaptation success
        adaptation_success = experience.get('adaptation_success', 0.5)
        if adaptation_success > 0.7:
            self.meta_cognitive_capacity = min(1.0, self.meta_cognitive_capacity + 0.02)
            updates['meta_cognitive_capacity'] = self.meta_cognitive_capacity
        
        return updates