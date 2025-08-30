# Relevance Optimization System Documentation

## Overview

The Relevance Optimization System is an advanced cognitive architecture component that implements Vervaeke's relevance realization framework to dynamically prioritize cognitive resources based on contextual importance, goal relevance, and environmental demands. This system addresses the fundamental challenge of cognitive resource allocation in artificial general intelligence systems.

## Architecture

### Core Components

#### 1. RelevanceOptimizer (`src/core/relevance_optimization.py`)
The main orchestration system that coordinates all relevance optimization mechanisms.

**Key Features:**
- Multi-dimensional relevance scoring
- Dynamic attention allocation
- Environmental salience detection
- Goal-relevance alignment
- Memory retrieval optimization
- Adaptive threshold learning
- Performance measurement and tracking

#### 2. SalienceType Enumeration
Defines five distinct types of salience based on Vervaeke's framework:

- **Environmental**: Bottom-up salience from environmental changes
- **Goal-oriented**: Top-down salience based on current goals
- **Contextual**: Salience derived from current context matching
- **Emergent**: Salience from novel patterns and unexpected combinations
- **Temporal**: Salience based on urgency, timing, and recency

#### 3. Data Classes

##### RelevanceMetrics
Tracks performance across key dimensions:
```python
@dataclass
class RelevanceMetrics:
    attention_efficiency: float
    cognitive_load_reduction: float
    goal_alignment_score: float
    environmental_responsiveness: float
    memory_retrieval_accuracy: float
    overall_performance_improvement: float
```

##### EnvironmentalSignal
Represents detected environmental changes:
```python
@dataclass
class EnvironmentalSignal:
    signal_type: str
    intensity: float
    novelty: float
    urgency: float
    context: Dict[str, Any]
    detection_confidence: float
```

##### GoalRelevanceAlignment
Analyzes goal-focus alignment:
```python
@dataclass
class GoalRelevanceAlignment:
    goal_id: str
    current_relevance: float
    optimal_relevance: float
    alignment_score: float
    recommended_adjustments: Dict[str, float]
```

## Key Algorithms

### 1. Multi-Dimensional Relevance Scoring

The system computes comprehensive relevance scores by combining five salience types:

```python
def compute_relevance_score(self, item, context, goals=None):
    # Environmental salience - changes requiring attention
    env_score = self._compute_environmental_salience(item, context)
    
    # Goal-oriented salience - relevance to active goals
    goal_score = self._compute_goal_salience(item, goals, context)
    
    # Contextual salience - current context matching
    context_score = self._compute_contextual_salience(item, context)
    
    # Emergent salience - novel patterns
    emergent_score = self._compute_emergent_salience(item, context)
    
    # Temporal salience - urgency and timing
    temporal_score = self._compute_temporal_salience(item, context)
    
    # Weighted combination
    overall_score = sum(
        self.importance_weights[stype] * score
        for stype, score in zip(SalienceType, component_scores.values())
    )
    
    return overall_score, component_scores
```

### 2. Dynamic Attention Allocation

Optimizes attention distribution across relevance modes:

```python
def allocate_attention_dynamically(self, relevance_scores, current_attention, context):
    # Compute demand for each mode
    mode_demands = self._compute_mode_demands(relevance_scores, context)
    
    # Apply contextual modulation
    modulated_demands = self._modulate_demands_by_context(mode_demands, context)
    
    # Optimize allocation with constraints
    new_allocation = self._optimize_attention_allocation(
        modulated_demands, current_attention
    )
    
    # Apply smoothing to prevent oscillation
    smoothed_allocation = self._smooth_attention_transition(
        current_attention, new_allocation
    )
    
    return smoothed_allocation
```

### 3. Environmental Salience Detection

Monitors environmental changes and detects significant variations:

```python
def detect_environmental_salience(self, environment_data, context):
    signals = []
    
    for key, value in environment_data.items():
        baseline_value = self.environmental_baseline.get(key, value)
        
        if isinstance(value, (int, float)):
            change_magnitude = abs(value - baseline_value) / max(abs(baseline_value), 1e-6)
            
            if change_magnitude > self.novelty_threshold:
                signal = EnvironmentalSignal(
                    signal_type=key,
                    intensity=min(1.0, change_magnitude),
                    novelty=self._compute_novelty(key, value),
                    urgency=self._compute_urgency(key, value, context),
                    context={'old_value': baseline_value, 'new_value': value},
                    detection_confidence=min(1.0, change_magnitude * 2)
                )
                signals.append(signal)
    
    return signals
```

### 4. Goal-Relevance Alignment

Analyzes and optimizes alignment between current focus and goal priorities:

```python
def align_with_goals(self, current_focus, active_goals, context):
    alignments = []
    
    for goal in active_goals:
        # Compute current and optimal relevance
        current_relevance = self._compute_current_goal_relevance(current_focus, goal, goal_info)
        optimal_relevance = self._compute_optimal_goal_relevance(goal, goal_info, context)
        
        # Compute alignment score
        alignment_score = self._compute_alignment_score(current_relevance, optimal_relevance)
        
        # Generate recommendations
        recommendations = self._generate_alignment_recommendations(
            current_focus, goal, current_relevance, optimal_relevance
        )
        
        alignment = GoalRelevanceAlignment(
            goal_id=goal,
            current_relevance=current_relevance,
            optimal_relevance=optimal_relevance,
            alignment_score=alignment_score,
            recommended_adjustments=recommendations
        )
        alignments.append(alignment)
    
    return alignments
```

### 5. Memory Retrieval Optimization

Optimizes memory retrieval based on relevance to current context:

```python
def optimize_memory_retrieval(self, query, context, memory_items):
    relevance_scores = []
    
    for item_id, item_data in memory_items.items():
        # Compute multi-faceted relevance
        base_relevance, _ = self.compute_relevance_score(item_data, context, context.get('active_goals', []))
        query_similarity = self._compute_query_similarity(query, item_data)
        temporal_relevance = self._compute_memory_temporal_relevance(item_id, item_data)
        context_relevance = self._compute_memory_context_relevance(item_data, context)
        
        # Combine scores with weights
        total_relevance = (
            0.4 * base_relevance +
            0.3 * query_similarity +
            0.2 * temporal_relevance +
            0.1 * context_relevance
        )
        
        relevance_scores.append((item_id, total_relevance))
    
    # Return sorted by relevance
    return sorted(relevance_scores, key=lambda x: x[1], reverse=True)
```

### 6. Adaptive Threshold Learning

Adjusts relevance thresholds based on performance feedback:

```python
def adapt_thresholds(self, performance_feedback):
    for mode in RelevanceMode:
        if mode.value in performance_feedback:
            performance = performance_feedback[mode.value]
            current_threshold = self.adaptive_thresholds[mode]
            
            # Poor performance → lower threshold (more inclusive)
            # Good performance → slightly raise threshold (more selective)
            if performance < 0.5:
                adjustment = -self.threshold_adaptation_rate * (0.5 - performance)
            else:
                adjustment = self.threshold_adaptation_rate * (performance - 0.7) * 0.5
            
            new_threshold = current_threshold + adjustment
            self.adaptive_thresholds[mode] = np.clip(new_threshold, 0.1, 0.9)
    
    return self.adaptive_thresholds.copy()
```

## Integration with Cognitive Architecture

### Enhanced Cognitive Core

The `EnhancedCognitiveCore` class in `src/core/relevance_integration.py` demonstrates integration with the existing cognitive architecture:

```python
class EnhancedCognitiveCore(CogPrimeCore):
    def enhanced_cognitive_cycle(self, sensory_input, reward=0.0, environment_data=None):
        # Detect environmental changes
        environmental_signals = self.relevance_optimizer.detect_environmental_salience(
            environment_data, self._get_current_context()
        )
        
        # Compute relevance scores
        input_items = self._extract_items_from_sensory_input(sensory_input)
        relevance_scores = {}
        for item in input_items:
            score, components = self.relevance_optimizer.compute_relevance_score(
                item, self._get_current_context(), active_goals
            )
            relevance_scores[item] = score
        
        # Optimize attention allocation
        current_attention = self._get_current_attention_allocation()
        optimized_attention = self.relevance_optimizer.allocate_attention_dynamically(
            relevance_scores, current_attention, self._get_current_context()
        )
        
        # Apply optimizations and execute cognitive cycle
        self._apply_attention_allocation(optimized_attention)
        action = self.cognitive_cycle(sensory_input, reward)
        
        # Learn and measure performance
        self.relevance_optimizer.learn_from_feedback(...)
        current_metrics = self.relevance_optimizer.measure_performance(...)
        
        return action
```

## Performance Measurement

The system tracks performance across multiple dimensions:

### Core Metrics

1. **Attention Efficiency**: How well attention is distributed relative to demands
2. **Cognitive Load Reduction**: Reduction in processing overhead through filtering
3. **Goal Alignment Score**: How well current focus aligns with active goals
4. **Environmental Responsiveness**: Ability to detect and respond to environmental changes
5. **Memory Retrieval Accuracy**: Quality of memory retrieval based on relevance

### Target Achievement

The system is designed to achieve a **35% performance improvement** as specified in the acceptance criteria. Performance is measured relative to baseline metrics and tracked over time.

### Performance Tracking

```python
def measure_performance(self, context, baseline_metrics=None):
    current_metrics = RelevanceMetrics(
        attention_efficiency=self._measure_attention_efficiency(),
        cognitive_load_reduction=self._measure_cognitive_load_reduction(),
        goal_alignment_score=self._measure_goal_alignment_performance(),
        environmental_responsiveness=self._measure_environmental_responsiveness(),
        memory_retrieval_accuracy=self._measure_memory_retrieval_accuracy()
    )
    
    if baseline_metrics:
        # Calculate overall improvement
        improvements = [
            current_metrics.attention_efficiency - baseline_metrics.attention_efficiency,
            current_metrics.cognitive_load_reduction - baseline_metrics.cognitive_load_reduction,
            # ... other metrics
        ]
        current_metrics.overall_performance_improvement = np.mean(improvements)
    
    return current_metrics
```

## Testing and Validation

### Comprehensive Test Suite

The system includes 20 comprehensive tests in `src/tests/test_relevance_optimization.py`:

- **Basic functionality tests**: Initialization, scoring, allocation
- **Component-specific tests**: Environmental detection, goal alignment, memory optimization
- **Integration tests**: Multi-goal scenarios, large-scale computation
- **Performance tests**: Improvement tracking, target validation

### Test Coverage

- ✅ Relevance scoring algorithms
- ✅ Environmental salience detection
- ✅ Dynamic attention allocation
- ✅ Goal-relevance alignment
- ✅ Memory retrieval optimization
- ✅ Adaptive threshold learning
- ✅ Performance measurement
- ✅ Integration capabilities

## Usage Examples

### Basic Usage

```python
from src.core.relevance_optimization import RelevanceOptimizer
from src.core.relevance_core import RelevanceCore

# Initialize the system
base_core = RelevanceCore()
optimizer = RelevanceOptimizer(base_core)

# Compute relevance scores
item = "visual_pattern"
context = {"task_type": "learning", "urgency": 0.7}
goals = ["learn_patterns", "solve_problems"]

score, components = optimizer.compute_relevance_score(item, context, goals)
print(f"Relevance score: {score:.3f}")
print(f"Components: {components}")
```

### Environmental Monitoring

```python
# Set up environmental monitoring
environment_data = {
    'light_level': 85.0,
    'noise_level': 60.0,
    'temperature': 24.0
}

signals = optimizer.detect_environmental_salience(environment_data, context)
for signal in signals:
    print(f"Detected: {signal.signal_type} - intensity: {signal.intensity:.3f}")
```

### Memory Retrieval

```python
# Optimize memory retrieval
query = "pattern recognition"
memory_items = {
    'mem1': {'content': 'visual pattern detection algorithm'},
    'mem2': {'content': 'unrelated information'},
    'mem3': {'content': 'pattern matching technique'}
}

ranked_memories = optimizer.optimize_memory_retrieval(query, context, memory_items)
for mem_id, relevance in ranked_memories:
    print(f"{mem_id}: {relevance:.3f}")
```

## Configuration

### Salience Weights

Configure the importance of different salience types:

```python
optimizer.importance_weights = {
    SalienceType.ENVIRONMENTAL: 0.25,
    SalienceType.GOAL_ORIENTED: 0.30,
    SalienceType.CONTEXTUAL: 0.20,
    SalienceType.EMERGENT: 0.15,
    SalienceType.TEMPORAL: 0.10
}
```

### Adaptive Parameters

```python
config = {
    'novelty_threshold': 0.5,           # Threshold for detecting environmental changes
    'attention_capacity': 1.0,          # Total attention capacity
    'threshold_adaptation_rate': 0.1,   # Rate of threshold adaptation
    'learning_rate': 0.05              # Learning rate for relevance adaptation
}

optimizer = RelevanceOptimizer(base_core, config)
```

## Future Enhancements

### Planned Improvements

1. **Neural Network Integration**: Replace heuristic algorithms with learned models
2. **Multi-Modal Processing**: Enhanced cross-modal relevance computation
3. **Temporal Dynamics**: Better modeling of temporal relevance patterns
4. **Social Relevance**: Integration of social and collaborative relevance
5. **Hierarchical Goals**: Support for goal hierarchies and sub-goals

### Research Directions

1. **Embodied Relevance**: Integration with robotic and embodied systems
2. **Collective Intelligence**: Multi-agent relevance coordination
3. **Meta-Relevance**: Learning to learn relevance patterns
4. **Consciousness Integration**: Connection with consciousness models

## References

- Vervaeke, J. "Relevance Realization and the Emerging Framework in Cognitive Science"
- Vervaeke, J., et al. "Zombies in Western Culture: A Twenty-First Century Crisis" 
- CogPrime Architecture Documentation
- OpenCog Hyperon Framework

## License

This implementation is part of the CogPrime/co9priml project and follows the project's licensing terms.