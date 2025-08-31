"""
Historical Context Integration System

This module implements comprehensive temporal knowledge representation, reasoning,
and decision-making capabilities that integrate with the existing CogPrime 
cognitive architecture. It builds upon the existing EnhancedEpisodicMemory
and AdvancedPatternDetector while adding sophisticated temporal processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

# Import existing components
from .reasoning import Thought, EnhancedEpisodicMemory, AdvancedPatternDetector, PatternSignature


class TemporalRelationType(Enum):
    """Types of temporal relationships"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    STARTS = "starts"
    FINISHES = "finishes"
    EQUALS = "equals"
    CONTAINS = "contains"
    CAUSAL = "causal"


class CausalType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    NECESSARY_CONDITION = "necessary_condition"
    SUFFICIENT_CONDITION = "sufficient_condition"
    CORRELATION = "correlation"
    SPURIOUS = "spurious"


@dataclass
class TemporalRelation:
    """Represents a temporal relationship between events"""
    subject_id: str
    object_id: str
    relation_type: TemporalRelationType
    confidence: float
    temporal_distance: float  # Time difference in standard units
    evidence: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """Represents a causal relationship between events"""
    cause_id: str
    effect_id: str
    causal_type: CausalType
    strength: float  # 0.0 to 1.0
    confidence: float
    delay: float  # Temporal delay between cause and effect
    evidence: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalEvent:
    """Enhanced event representation with temporal properties"""
    event_id: str
    content: torch.Tensor
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    salience: float = 0.5
    confidence: float = 1.0
    event_type: str = "generic"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.duration is None and self.end_time is not None:
            self.duration = self.end_time - self.start_time
        elif self.end_time is None and self.duration is not None:
            self.end_time = self.start_time + self.duration


@dataclass
class TemporalPattern:
    """Represents a pattern that occurs across time"""
    pattern_id: str
    pattern_type: str
    temporal_scale: str  # "seconds", "minutes", "hours", "days", etc.
    frequency: float  # How often the pattern occurs
    confidence: float
    instances: List[str] = field(default_factory=list)  # Event IDs
    features: torch.Tensor = None
    context: Dict[str, Any] = field(default_factory=dict)


class TemporalKnowledgeBase:
    """Central repository for temporal knowledge and relationships"""
    
    def __init__(self):
        self.events: Dict[str, TemporalEvent] = {}
        self.temporal_relations: List[TemporalRelation] = []
        self.causal_relations: List[CausalRelation] = []
        self.temporal_patterns: Dict[str, TemporalPattern] = {}
        
        # Indexing for efficient retrieval
        self.time_index: Dict[float, List[str]] = defaultdict(list)  # timestamp -> event_ids
        self.type_index: Dict[str, List[str]] = defaultdict(list)   # event_type -> event_ids
        self.relation_index: Dict[str, List[TemporalRelation]] = defaultdict(list)
        
    def add_event(self, event: TemporalEvent) -> None:
        """Add a temporal event to the knowledge base"""
        self.events[event.event_id] = event
        
        # Update indices
        self.time_index[event.start_time].append(event.event_id)
        self.type_index[event.event_type].append(event.event_id)
        
    def add_temporal_relation(self, relation: TemporalRelation) -> None:
        """Add a temporal relation"""
        self.temporal_relations.append(relation)
        self.relation_index[relation.subject_id].append(relation)
        
    def add_causal_relation(self, relation: CausalRelation) -> None:
        """Add a causal relation"""
        self.causal_relations.append(relation)
        
    def get_events_in_range(self, start_time: float, end_time: float) -> List[TemporalEvent]:
        """Get all events within a time range"""
        events = []
        for timestamp, event_ids in self.time_index.items():
            if start_time <= timestamp <= end_time:
                events.extend([self.events[eid] for eid in event_ids if eid in self.events])
        return events
        
    def find_temporal_relations(self, event_id: str) -> List[TemporalRelation]:
        """Find all temporal relations involving an event"""
        relations = []
        for relation in self.temporal_relations:
            if relation.subject_id == event_id or relation.object_id == event_id:
                relations.append(relation)
        return relations
        
    def find_causal_relations(self, event_id: str) -> List[CausalRelation]:
        """Find all causal relations involving an event"""
        relations = []
        for relation in self.causal_relations:
            if relation.cause_id == event_id or relation.effect_id == event_id:
                relations.append(relation)
        return relations


class EnhancedTemporalPatternDetector(AdvancedPatternDetector):
    """Enhanced pattern detector with sophisticated temporal analysis"""
    
    def __init__(self, feature_dim: int = 512, max_patterns: int = 1000):
        super().__init__(feature_dim, max_patterns)
        
        # Additional temporal analysis components
        self.temporal_scales = ["minute", "hour", "day", "week"]
        self.causal_detector = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Pattern memory and learning
        self.pattern_memory = {}
        self.confidence_threshold = 0.3
        
    def detect_temporal_patterns(self, sequence: List[torch.Tensor]) -> List[PatternSignature]:
        """Enhanced temporal pattern detection with confidence normalization"""
        if len(sequence) < 2:
            return []
        
        patterns = []
        
        # Convert to LSTM input format
        seq_tensor = torch.stack(sequence).unsqueeze(0)  # (1, seq_len, feature_dim)
        
        # Detect patterns using LSTM
        with torch.no_grad():
            output, (hidden, cell) = self.temporal_detector(seq_tensor)
            
            # Analyze hidden states for patterns
            pattern_strength = torch.norm(hidden, dim=2).squeeze()
            
            # Normalize confidence to [0, 1] range
            normalized_confidence = torch.sigmoid(pattern_strength * 0.1)  # Scale and normalize
            
            if normalized_confidence.item() > self.confidence_threshold:
                pattern_id = f"temporal_{hash(tuple(hidden.flatten()[:10].tolist())) % 100000}"
                pattern = PatternSignature(
                    pattern_id=pattern_id,
                    pattern_type="temporal",
                    confidence=float(normalized_confidence.item()),
                    frequency=self.pattern_frequencies[pattern_id],
                    last_seen=float(torch.rand(1)),  # Placeholder timestamp
                    features=hidden.squeeze(),
                    context={"sequence_length": len(sequence)}
                )
                patterns.append(pattern)
                self.pattern_frequencies[pattern_id] += 1
        
        return patterns
        
    def detect_causal_patterns(self, cause_event: torch.Tensor, 
                             effect_event: torch.Tensor,
                             temporal_distance: float) -> float:
        """Detect potential causal relationships between events"""
        # Combine cause and effect representations
        combined = torch.cat([cause_event.flatten(), effect_event.flatten()])
        
        # Ensure the input size matches the expected size
        if combined.numel() != self.feature_dim * 2:
            # Resize to match expected input
            if combined.numel() > self.feature_dim * 2:
                combined = combined[:self.feature_dim * 2]
            else:
                padding = torch.zeros(self.feature_dim * 2 - combined.numel())
                combined = torch.cat([combined, padding])
        
        with torch.no_grad():
            causal_strength = self.causal_detector(combined.unsqueeze(0))
            
        # Adjust for temporal distance (closer events more likely to be causal)
        temporal_factor = np.exp(-temporal_distance / 3600)  # Decay over hours
        adjusted_strength = causal_strength.item() * temporal_factor
        
        return min(adjusted_strength, 1.0)
        
    def detect_multi_scale_patterns(self, events: List[TemporalEvent]) -> Dict[str, List[TemporalPattern]]:
        """Detect patterns across multiple temporal scales"""
        patterns_by_scale = {}
        
        for scale in self.temporal_scales:
            scale_patterns = []
            
            # Group events by temporal scale
            if scale == "minute":
                time_window = 60  # seconds
            elif scale == "hour":
                time_window = 3600
            elif scale == "day":
                time_window = 86400
            elif scale == "week":
                time_window = 604800
            else:
                time_window = 3600  # default
            
            # Find recurring patterns at this scale
            pattern_candidates = self._find_recurring_sequences(events, time_window)
            
            for candidate in pattern_candidates:
                if candidate['confidence'] > self.confidence_threshold:
                    pattern = TemporalPattern(
                        pattern_id=f"{scale}_{candidate['id']}",
                        pattern_type="recurring",
                        temporal_scale=scale,
                        frequency=candidate['frequency'],
                        confidence=candidate['confidence'],
                        instances=candidate['instances'],
                        context={"time_window": time_window}
                    )
                    scale_patterns.append(pattern)
            
            patterns_by_scale[scale] = scale_patterns
            
        return patterns_by_scale
        
    def _find_recurring_sequences(self, events: List[TemporalEvent], 
                                time_window: float) -> List[Dict[str, Any]]:
        """Find recurring sequences within a time window"""
        # Simplified implementation - group events by time windows and find patterns
        time_buckets = defaultdict(list)
        
        for event in events:
            bucket_key = int(event.start_time // time_window)
            time_buckets[bucket_key].append(event)
        
        # Look for similar patterns across buckets
        patterns = []
        bucket_keys = sorted(time_buckets.keys())
        
        for i, key in enumerate(bucket_keys[:-1]):
            bucket = time_buckets[key]
            if len(bucket) > 1:
                # Simple pattern: events of same type occurring together
                type_counts = defaultdict(int)
                for event in bucket:
                    type_counts[event.event_type] += 1
                
                for event_type, count in type_counts.items():
                    if count > 1:
                        patterns.append({
                            'id': f"recur_{event_type}_{key}",
                            'frequency': count / len(bucket),
                            'confidence': min(count / 3.0, 1.0),  # Confidence based on occurrence
                            'instances': [e.event_id for e in bucket if e.event_type == event_type]
                        })
        
        return patterns


class CausalRelationshipDetector:
    """Specialized system for detecting causal relationships across time"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.causal_evidence_threshold = 0.6
        self.max_causal_delay = 86400  # 1 day in seconds
        
        # Causal strength predictor
        self.causal_network = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 256),  # +1 for temporal distance
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def detect_causal_relationships(self, events: List[TemporalEvent]) -> List[CausalRelation]:
        """Detect causal relationships between events"""
        causal_relations = []
        
        # Sort events by time
        sorted_events = sorted(events, key=lambda e: e.start_time)
        
        for i, potential_cause in enumerate(sorted_events[:-1]):
            for potential_effect in sorted_events[i+1:]:
                delay = potential_effect.start_time - potential_cause.start_time
                
                # Skip if delay is too large
                if delay > self.max_causal_delay:
                    continue
                
                # Calculate causal strength
                causal_strength = self._calculate_causal_strength(
                    potential_cause, potential_effect, delay
                )
                
                if causal_strength > self.causal_evidence_threshold:
                    relation = CausalRelation(
                        cause_id=potential_cause.event_id,
                        effect_id=potential_effect.event_id,
                        causal_type=CausalType.DIRECT_CAUSE,
                        strength=causal_strength,
                        confidence=causal_strength,
                        delay=delay,
                        evidence=[f"temporal_proximity_{delay:.2f}s"],
                        context={
                            "cause_type": potential_cause.event_type,
                            "effect_type": potential_effect.event_type
                        }
                    )
                    causal_relations.append(relation)
        
        return causal_relations
        
    def _calculate_causal_strength(self, cause: TemporalEvent, 
                                 effect: TemporalEvent, delay: float) -> float:
        """Calculate the strength of a potential causal relationship"""
        # Prepare input features
        cause_features = cause.content.flatten()
        effect_features = effect.content.flatten()
        
        # Normalize temporal delay
        normalized_delay = delay / self.max_causal_delay
        
        # Combine features
        combined_features = torch.cat([
            cause_features[:self.feature_dim],
            effect_features[:self.feature_dim],
            torch.tensor([normalized_delay])
        ])
        
        with torch.no_grad():
            strength = self.causal_network(combined_features.unsqueeze(0))
            
        return float(strength.item())


class TemporalReasoningEngine:
    """Advanced temporal reasoning and inference system"""
    
    def __init__(self, knowledge_base: TemporalKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.inference_rules = self._initialize_inference_rules()
        
    def _initialize_inference_rules(self) -> Dict[str, Any]:
        """Initialize temporal inference rules"""
        return {
            "transitivity": {
                "before": lambda a, b, c: True,  # If A before B and B before C, then A before C
                "after": lambda a, b, c: True,
                "causal": lambda a, b, c: True   # If A causes B and B causes C, then A contributes to C
            },
            "consistency": {
                "temporal": self._check_temporal_consistency,
                "causal": self._check_causal_consistency
            }
        }
        
    def infer_temporal_relations(self, event_a: str, event_b: str) -> List[TemporalRelation]:
        """Infer temporal relations between two events"""
        # Direct relations
        direct_relations = []
        for relation in self.knowledge_base.temporal_relations:
            if ((relation.subject_id == event_a and relation.object_id == event_b) or
                (relation.subject_id == event_b and relation.object_id == event_a)):
                direct_relations.append(relation)
        
        if direct_relations:
            return direct_relations
        
        # Inferred relations through transitivity
        inferred_relations = self._infer_through_transitivity(event_a, event_b)
        
        return inferred_relations
        
    def _infer_through_transitivity(self, event_a: str, event_b: str) -> List[TemporalRelation]:
        """Infer relations through temporal transitivity"""
        # Simple transitivity: if A -> intermediate -> B, infer A -> B
        inferred = []
        
        # Find intermediate events
        a_relations = self.knowledge_base.find_temporal_relations(event_a)
        b_relations = self.knowledge_base.find_temporal_relations(event_b)
        
        for a_rel in a_relations:
            for b_rel in b_relations:
                # Find common intermediate event
                intermediate = None
                if a_rel.object_id == b_rel.subject_id:
                    intermediate = a_rel.object_id
                elif a_rel.subject_id == b_rel.object_id:
                    intermediate = a_rel.subject_id
                
                if intermediate:
                    # Infer relation based on transitivity
                    if (a_rel.relation_type == TemporalRelationType.BEFORE and 
                        b_rel.relation_type == TemporalRelationType.BEFORE):
                        
                        inferred_relation = TemporalRelation(
                            subject_id=event_a,
                            object_id=event_b,
                            relation_type=TemporalRelationType.BEFORE,
                            confidence=min(a_rel.confidence, b_rel.confidence) * 0.8,  # Reduced confidence for inference
                            temporal_distance=a_rel.temporal_distance + b_rel.temporal_distance,
                            evidence=[f"transitivity_via_{intermediate}"],
                            context={"inference_type": "transitivity", "intermediate": intermediate}
                        )
                        inferred.append(inferred_relation)
        
        return inferred
        
    def _check_temporal_consistency(self, relations: List[TemporalRelation]) -> Dict[str, Any]:
        """Check temporal consistency of relations"""
        inconsistencies = []
        
        # Check for logical contradictions
        for i, rel1 in enumerate(relations):
            for rel2 in relations[i+1:]:
                if (rel1.subject_id == rel2.subject_id and 
                    rel1.object_id == rel2.object_id):
                    
                    # Same events, different relations - potential inconsistency
                    if rel1.relation_type != rel2.relation_type:
                        inconsistencies.append({
                            "type": "contradictory_relations",
                            "relation1": rel1,
                            "relation2": rel2,
                            "severity": "high"
                        })
        
        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies
        }
        
    def _check_causal_consistency(self, relations: List[CausalRelation]) -> Dict[str, Any]:
        """Check causal consistency"""
        # Simplified causal consistency check
        inconsistencies = []
        
        # Check for circular causality
        causality_graph = defaultdict(list)
        for relation in relations:
            causality_graph[relation.cause_id].append(relation.effect_id)
        
        # Simple cycle detection
        def has_cycle(node, visited, path):
            if node in path:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            path.add(node)
            
            for neighbor in causality_graph.get(node, []):
                if has_cycle(neighbor, visited, path):
                    return True
            
            path.remove(node)
            return False
        
        visited = set()
        for start_node in causality_graph:
            if start_node not in visited:
                if has_cycle(start_node, visited, set()):
                    inconsistencies.append({
                        "type": "circular_causality",
                        "description": f"Circular causality detected involving {start_node}",
                        "severity": "high"
                    })
        
        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies
        }


class HistoricalContextAwareDecisionMaker:
    """Decision making system that incorporates historical context"""
    
    def __init__(self, knowledge_base: TemporalKnowledgeBase, 
                 reasoning_engine: TemporalReasoningEngine):
        self.knowledge_base = knowledge_base
        self.reasoning_engine = reasoning_engine
        self.decision_history = []
        self.learning_rate = 0.1
        
    def make_decision(self, current_context: Dict[str, Any], 
                     available_actions: List[str]) -> Dict[str, Any]:
        """Make a decision based on current context and historical patterns"""
        
        # Analyze historical context
        historical_analysis = self._analyze_historical_context(current_context)
        
        # Score available actions based on historical success
        action_scores = {}
        for action in available_actions:
            score = self._score_action_historically(action, current_context, historical_analysis)
            action_scores[action] = score
        
        # Select best action
        best_action = max(action_scores.items(), key=lambda x: x[1])
        
        decision = {
            "chosen_action": best_action[0],
            "confidence": best_action[1],
            "rationale": self._generate_rationale(best_action[0], historical_analysis),
            "historical_analysis": historical_analysis,
            "action_scores": action_scores
        }
        
        # Record decision for future learning
        self.decision_history.append(decision)
        
        return decision
        
    def _analyze_historical_context(self, current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical context relevant to current situation"""
        analysis = {
            "similar_situations": [],
            "successful_patterns": [],
            "failure_patterns": [],
            "contextual_trends": []
        }
        
        # Find similar historical situations
        current_time = current_context.get("timestamp", datetime.now().timestamp())
        time_window = current_context.get("time_window", 86400)  # 1 day default
        
        historical_events = self.knowledge_base.get_events_in_range(
            current_time - time_window, current_time
        )
        
        # Analyze patterns in historical events
        for event in historical_events:
            similarity = self._calculate_context_similarity(current_context, event.context)
            if similarity > 0.7:  # High similarity threshold
                analysis["similar_situations"].append({
                    "event": event,
                    "similarity": similarity,
                    "outcome": event.context.get("outcome", "unknown")
                })
        
        # Identify successful and failure patterns
        for situation in analysis["similar_situations"]:
            outcome = situation["outcome"]
            if outcome == "success":
                analysis["successful_patterns"].append(situation["event"])
            elif outcome == "failure":
                analysis["failure_patterns"].append(situation["event"])
        
        return analysis
        
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        # Simple similarity based on common keys and values
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
        
        return matches / len(common_keys)
        
    def _score_action_historically(self, action: str, current_context: Dict[str, Any],
                                 historical_analysis: Dict[str, Any]) -> float:
        """Score an action based on historical success"""
        base_score = 0.5  # Neutral baseline
        
        # Boost score based on successful patterns
        for pattern in historical_analysis["successful_patterns"]:
            if pattern.context.get("action") == action:
                base_score += 0.2 * pattern.confidence
        
        # Reduce score based on failure patterns
        for pattern in historical_analysis["failure_patterns"]:
            if pattern.context.get("action") == action:
                base_score -= 0.2 * pattern.confidence
        
        # Consider frequency of action in similar contexts
        action_frequency = sum(1 for s in historical_analysis["similar_situations"] 
                             if s["event"].context.get("action") == action)
        
        if action_frequency > 0:
            success_rate = sum(1 for s in historical_analysis["similar_situations"] 
                             if (s["event"].context.get("action") == action and 
                                 s["outcome"] == "success")) / action_frequency
            base_score = base_score * 0.7 + success_rate * 0.3
        
        return max(0.0, min(1.0, base_score))
        
    def _generate_rationale(self, action: str, analysis: Dict[str, Any]) -> str:
        """Generate rationale for the chosen action"""
        rationale_parts = [f"Chosen action: {action}"]
        
        if analysis["successful_patterns"]:
            rationale_parts.append(
                f"Action has {len(analysis['successful_patterns'])} successful precedents"
            )
        
        if analysis["failure_patterns"]:
            rationale_parts.append(
                f"Action has {len(analysis['failure_patterns'])} failure precedents"
            )
        
        if analysis["similar_situations"]:
            avg_similarity = np.mean([s["similarity"] for s in analysis["similar_situations"]])
            rationale_parts.append(
                f"Based on {len(analysis['similar_situations'])} similar situations "
                f"(avg similarity: {avg_similarity:.2f})"
            )
        
        return "; ".join(rationale_parts)


class HistoricalContextIntegrationSystem:
    """Main system integrating all historical context components"""
    
    def __init__(self, feature_dim: int = 512, memory_size: int = 1000):
        self.feature_dim = feature_dim
        
        # Core components
        self.knowledge_base = TemporalKnowledgeBase()
        self.pattern_detector = EnhancedTemporalPatternDetector(feature_dim)
        self.causal_detector = CausalRelationshipDetector(feature_dim)
        self.reasoning_engine = TemporalReasoningEngine(self.knowledge_base)
        self.decision_maker = HistoricalContextAwareDecisionMaker(
            self.knowledge_base, self.reasoning_engine
        )
        
        # Integration with existing systems
        self.episodic_memory = EnhancedEpisodicMemory(memory_size, feature_dim)
        
        # System state
        self.current_time = datetime.now().timestamp()
        self.processing_history = deque(maxlen=1000)
        
    def process_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Process a new experience and integrate it into historical context"""
        timestamp = experience.get("timestamp", self.current_time)
        
        # Create temporal event
        event = TemporalEvent(
            event_id=f"event_{len(self.knowledge_base.events)}_{timestamp}",
            content=experience["content"],
            start_time=timestamp,
            salience=experience.get("salience", 0.5),
            confidence=experience.get("confidence", 1.0),
            event_type=experience.get("type", "generic"),
            context=experience.get("context", {})
        )
        
        # Add to knowledge base
        self.knowledge_base.add_event(event)
        
        # Store in episodic memory
        thought = Thought(
            content=event.content,
            salience=event.salience,
            associations=experience.get("associations", []),
            timestamp=timestamp,
            pattern_type=event.event_type,
            confidence=event.confidence,
            context=event.context
        )
        self.episodic_memory.store(thought)
        
        # Detect patterns and relationships
        processing_result = self._process_temporal_relationships(event)
        
        # Update processing history
        self.processing_history.append({
            "timestamp": timestamp,
            "event_id": event.event_id,
            "processing_result": processing_result
        })
        
        return processing_result
        
    def _process_temporal_relationships(self, new_event: TemporalEvent) -> Dict[str, Any]:
        """Process temporal and causal relationships for a new event"""
        result = {
            "event_id": new_event.event_id,
            "detected_patterns": [],
            "causal_relations": [],
            "temporal_relations": [],
            "updated_patterns": []
        }
        
        # Get recent events for pattern detection
        recent_events = self.knowledge_base.get_events_in_range(
            new_event.start_time - 3600,  # Last hour
            new_event.start_time
        )
        
        if len(recent_events) > 1:
            # Detect temporal patterns
            event_sequence = [event.content for event in recent_events]
            patterns = self.pattern_detector.detect_temporal_patterns(event_sequence)
            result["detected_patterns"] = patterns
            
            # Detect causal relationships
            for prior_event in recent_events[:-1]:  # Exclude the new event itself
                causal_relations = self.causal_detector.detect_causal_relationships(
                    [prior_event, new_event]
                )
                result["causal_relations"].extend(causal_relations)
                
                # Add causal relations to knowledge base
                for relation in causal_relations:
                    self.knowledge_base.add_causal_relation(relation)
        
        return result
        
    def make_historical_decision(self, current_context: Dict[str, Any], 
                               available_actions: List[str]) -> Dict[str, Any]:
        """Make a decision using historical context"""
        return self.decision_maker.make_decision(current_context, available_actions)
        
    def validate_temporal_consistency(self) -> Dict[str, Any]:
        """Validate consistency of temporal knowledge"""
        temporal_consistency = self.reasoning_engine._check_temporal_consistency(
            self.knowledge_base.temporal_relations
        )
        causal_consistency = self.reasoning_engine._check_causal_consistency(
            self.knowledge_base.causal_relations
        )
        
        return {
            "temporal_consistency": temporal_consistency,
            "causal_consistency": causal_consistency,
            "overall_consistent": (temporal_consistency["consistent"] and 
                                 causal_consistency["consistent"])
        }
        
    def get_temporal_insights(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights from historical temporal patterns"""
        insights = {
            "relevant_patterns": [],
            "historical_precedents": [],
            "predictions": [],
            "recommendations": []
        }
        
        # Find relevant patterns
        query_type = query_context.get("type", "generic")
        relevant_events = self.knowledge_base.type_index.get(query_type, [])
        
        if relevant_events:
            # Analyze patterns in similar events
            event_objects = [self.knowledge_base.events[eid] for eid in relevant_events 
                           if eid in self.knowledge_base.events]
            
            if len(event_objects) > 1:
                multi_scale_patterns = self.pattern_detector.detect_multi_scale_patterns(event_objects)
                insights["relevant_patterns"] = multi_scale_patterns
                
                # Generate predictions based on patterns
                insights["predictions"] = self._generate_temporal_predictions(event_objects)
        
        return insights
        
    def _generate_temporal_predictions(self, events: List[TemporalEvent]) -> List[Dict[str, Any]]:
        """Generate predictions based on temporal patterns"""
        predictions = []
        
        # Simple prediction: if events occur regularly, predict next occurrence
        if len(events) >= 3:
            # Calculate average interval
            intervals = []
            sorted_events = sorted(events, key=lambda e: e.start_time)
            
            for i in range(1, len(sorted_events)):
                interval = sorted_events[i].start_time - sorted_events[i-1].start_time
                intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # Predict next event
                last_event_time = sorted_events[-1].start_time
                predicted_time = last_event_time + avg_interval
                
                predictions.append({
                    "type": "next_occurrence",
                    "predicted_time": predicted_time,
                    "confidence": max(0.1, 1.0 - (std_interval / avg_interval)) if avg_interval > 0 else 0.1,
                    "interval": avg_interval,
                    "uncertainty": std_interval
                })
        
        return predictions