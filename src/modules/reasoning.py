import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
import math

@dataclass
class Thought:
    """Represents a cognitive thought pattern"""
    content: torch.Tensor
    salience: float
    associations: List[str]
    timestamp: float
    pattern_type: str = "basic"  # New: type of pattern detected
    confidence: float = 1.0  # New: confidence in the thought
    context: Dict[str, Any] = None  # New: contextual information

    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class PatternSignature:
    """Represents a detected pattern with its characteristics"""
    pattern_id: str
    pattern_type: str  # "temporal", "spatial", "hierarchical", "associative"
    confidence: float
    frequency: int
    last_seen: float
    features: torch.Tensor
    context: Dict[str, Any]

class AdvancedPatternDetector:
    """Detects various types of patterns in cognitive data"""
    
    def __init__(self, feature_dim: int = 512, max_patterns: int = 1000):
        self.feature_dim = feature_dim
        self.max_patterns = max_patterns
        
        # Pattern storage
        self.temporal_patterns = {}  # time-based patterns
        self.spatial_patterns = {}   # spatial/structural patterns  
        self.hierarchical_patterns = {}  # multi-scale patterns
        self.associative_patterns = defaultdict(list)  # association networks
        
        # Pattern detection networks
        self.temporal_detector = nn.LSTM(feature_dim, 128, batch_first=True)
        self.spatial_detector = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.hierarchical_detector = nn.ModuleList([
            nn.Linear(feature_dim, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64)
        ])
        
        # Pattern learning history
        self.pattern_history = deque(maxlen=100)
        self.pattern_frequencies = defaultdict(int)
        
    def detect_temporal_patterns(self, sequence: List[torch.Tensor]) -> List[PatternSignature]:
        """Detect temporal patterns in a sequence of tensors"""
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
            
            if pattern_strength.item() > 0.3:  # Lower threshold for pattern detection
                pattern_id = f"temporal_{hash(tuple(hidden.flatten()[:10].tolist())) % 100000}"
                pattern = PatternSignature(
                    pattern_id=pattern_id,
                    pattern_type="temporal",
                    confidence=float(pattern_strength.item()),
                    frequency=self.pattern_frequencies[pattern_id],
                    last_seen=float(torch.rand(1)),  # Placeholder timestamp
                    features=hidden.squeeze(),
                    context={"sequence_length": len(sequence)}
                )
                patterns.append(pattern)
                self.pattern_frequencies[pattern_id] += 1
        
        return patterns
    
    def detect_spatial_patterns(self, tensor: torch.Tensor) -> List[PatternSignature]:
        """Detect spatial patterns in a tensor"""
        patterns = []
        
        # Reshape for conv1d
        input_tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, feature_dim)
        
        with torch.no_grad():
            conv_output = self.spatial_detector(input_tensor)
            
            # Find peaks in convolution output
            conv_flat = conv_output.flatten()
            threshold = torch.mean(conv_flat) + torch.std(conv_flat)
            peaks = (conv_flat > threshold).nonzero().flatten()
            
            for peak_idx in peaks[:10]:  # Limit to top 10 peaks
                if peak_idx < conv_output.shape[2]:  # Ensure valid index
                    pattern_strength = float(conv_flat[peak_idx])
                    pattern_id = f"spatial_{peak_idx}_{int(pattern_strength*1000)}"
                    
                    # Extract features at valid index
                    peak_features = conv_output[:, :, peak_idx].flatten()
                    
                    pattern = PatternSignature(
                        pattern_id=pattern_id,
                        pattern_type="spatial",
                        confidence=pattern_strength,
                        frequency=self.pattern_frequencies[pattern_id],
                        last_seen=float(torch.rand(1)),
                        features=peak_features,
                        context={"peak_location": int(peak_idx)}
                    )
                    patterns.append(pattern)
                    self.pattern_frequencies[pattern_id] += 1
        
        return patterns
    
    def detect_hierarchical_patterns(self, tensor: torch.Tensor) -> List[PatternSignature]:
        """Detect hierarchical patterns at multiple scales"""
        patterns = []
        
        # Ensure input is the right size for the detector
        if tensor.numel() != self.feature_dim:
            if tensor.numel() > self.feature_dim:
                current = tensor.flatten()[:self.feature_dim]
            else:
                padding = torch.zeros(self.feature_dim - tensor.numel())
                current = torch.cat([tensor.flatten(), padding])
        else:
            current = tensor
        
        # Process through hierarchy
        for level, detector in enumerate(self.hierarchical_detector):
            with torch.no_grad():
                current = torch.relu(detector(current))
                
                # Check for pattern emergence at this level
                activation_strength = torch.norm(current)
                if activation_strength > 0.1:  # Lower threshold for better detection
                    # Create a safe hash from first few elements
                    safe_elements = current.flatten()[:min(10, current.numel())].tolist()
                    pattern_id = f"hierarchical_L{level}_{hash(tuple(safe_elements)) % 100000}"
                    
                    pattern = PatternSignature(
                        pattern_id=pattern_id,
                        pattern_type="hierarchical",
                        confidence=float(activation_strength),
                        frequency=self.pattern_frequencies[pattern_id],
                        last_seen=float(torch.rand(1)),
                        features=current.clone(),
                        context={"hierarchy_level": level, "scale": current.numel()}
                    )
                    patterns.append(pattern)
                    self.pattern_frequencies[pattern_id] += 1
        
        return patterns
    
    def detect_associative_patterns(self, current_tensor: torch.Tensor, 
                                   memory_tensors: List[torch.Tensor]) -> List[PatternSignature]:
        """Detect associative patterns between current input and memories"""
        patterns = []
        
        for i, memory_tensor in enumerate(memory_tensors):
            # Ensure tensors are the same size for similarity calculation
            if current_tensor.numel() != memory_tensor.numel():
                target_size = min(current_tensor.numel(), memory_tensor.numel())
                current_resized = current_tensor.flatten()[:target_size]
                memory_resized = memory_tensor.flatten()[:target_size]
            else:
                current_resized = current_tensor.flatten()
                memory_resized = memory_tensor.flatten()
            
            # Calculate similarity
            similarity = torch.cosine_similarity(current_resized, memory_resized, dim=0)
            
            if similarity > 0.7:  # High similarity threshold
                pattern_id = f"associative_{i}_{int(similarity*1000)}"
                
                pattern = PatternSignature(
                    pattern_id=pattern_id,
                    pattern_type="associative",
                    confidence=float(similarity),
                    frequency=self.pattern_frequencies[pattern_id],
                    last_seen=float(torch.rand(1)),
                    features=memory_tensor.clone(),
                    context={"memory_index": i, "similarity": float(similarity)}
                )
                patterns.append(pattern)
                self.pattern_frequencies[pattern_id] += 1
        
        return patterns

class EnhancedEpisodicMemory:
    """Enhanced episodic memory with consolidation and optimization"""
    
    def __init__(self, memory_size: int = 1000, feature_dim: int = 512, 
                 consolidation_threshold: float = 0.8):
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.consolidation_threshold = consolidation_threshold
        
        # Memory storage
        self.memories = []
        self.memory_matrix = torch.zeros(memory_size, feature_dim)
        self.memory_importance = torch.zeros(memory_size)
        self.memory_access_count = torch.zeros(memory_size)
        self.memory_recency = torch.zeros(memory_size)
        
        # Consolidation system
        self.consolidated_memories = {}
        self.consolidation_network = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh()
        )
        
        self.current_index = 0
        self.global_time = 0
    
    def store(self, memory: Thought) -> None:
        """Store a new memory with enhanced consolidation"""
        self.global_time += 1
        
        # Calculate memory importance
        importance = self._calculate_importance(memory)
        
        # Find storage location
        if self.current_index >= self.memory_size:
            # Memory is full, need to decide what to replace
            replace_idx = self._select_replacement_candidate()
        else:
            replace_idx = self.current_index
            self.current_index += 1
        
        # Store memory
        self.memory_matrix[replace_idx] = memory.content
        self.memory_importance[replace_idx] = importance
        self.memory_access_count[replace_idx] = 0
        self.memory_recency[replace_idx] = self.global_time
        
        if replace_idx < len(self.memories):
            self.memories[replace_idx] = memory
        else:
            self.memories.append(memory)
        
        # Check for consolidation opportunities
        self._attempt_consolidation(memory, replace_idx)
    
    def _calculate_importance(self, memory: Thought) -> float:
        """Calculate the importance of a memory"""
        # Base importance from salience
        importance = memory.salience
        
        # Boost based on confidence
        importance *= memory.confidence
        
        # Boost based on pattern type (some patterns are more important)
        pattern_weights = {
            "hierarchical": 1.5,
            "temporal": 1.3,
            "associative": 1.2,
            "spatial": 1.1,
            "basic": 1.0
        }
        importance *= pattern_weights.get(memory.pattern_type, 1.0)
        
        # Boost based on associations
        importance *= (1 + len(memory.associations) * 0.1)
        
        return importance
    
    def _select_replacement_candidate(self) -> int:
        """Select which memory to replace using importance and recency"""
        # Combine importance, access frequency, and recency
        scores = (
            self.memory_importance * 0.5 +
            self.memory_access_count * 0.3 +
            (self.global_time - self.memory_recency) * -0.2  # Penalty for old memories
        )
        
        # Find the least important memory
        return int(torch.argmin(scores))
    
    def _attempt_consolidation(self, new_memory: Thought, memory_idx: int) -> None:
        """Attempt to consolidate similar memories"""
        similarities = torch.cosine_similarity(
            new_memory.content.unsqueeze(0),
            self.memory_matrix[:len(self.memories)],
            dim=1
        )
        
        # Find highly similar memories
        similar_indices = (similarities > self.consolidation_threshold).nonzero().flatten()
        
        if len(similar_indices) > 1:  # At least one other similar memory
            # Consolidate memories
            consolidated_content = self._consolidate_memories(
                [self.memories[i] for i in similar_indices]
            )
            
            # Create consolidated memory
            consolidated_id = f"consolidated_{hash(tuple(similar_indices.tolist()))}"
            self.consolidated_memories[consolidated_id] = {
                'content': consolidated_content,
                'source_indices': similar_indices.tolist(),
                'consolidation_time': self.global_time
            }
    
    def _consolidate_memories(self, memories: List[Thought]) -> torch.Tensor:
        """Consolidate multiple memories into one"""
        if len(memories) == 1:
            return memories[0].content
        
        # Average the memory contents
        contents = torch.stack([m.content for m in memories])
        averaged = torch.mean(contents, dim=0)
        
        # Apply consolidation network for refinement
        if len(memories) >= 2:
            pairs = []
            for i in range(0, len(memories), 2):
                if i + 1 < len(memories):
                    pair = torch.cat([memories[i].content, memories[i+1].content])
                    pairs.append(pair)
            
            if pairs:
                with torch.no_grad():
                    consolidated = self.consolidation_network(pairs[0])
                    for pair in pairs[1:]:
                        temp = self.consolidation_network(pair)
                        consolidated = (consolidated + temp) / 2
                return consolidated
        
        return averaged
    
    def retrieve(self, query: torch.Tensor, k: int = 5) -> List[Thought]:
        """Enhanced retrieval with consolidation awareness"""
        if len(self.memories) == 0:
            return []
            
        # Ensure query size matches memory storage size
        if query.numel() != self.feature_dim:
            if query.numel() > self.feature_dim:
                query_resized = query.flatten()[:self.feature_dim]
            else:
                padding = torch.zeros(self.feature_dim - query.numel())
                query_resized = torch.cat([query.flatten(), padding])
        else:
            query_resized = query
        
        # Standard similarity-based retrieval
        similarities = torch.cosine_similarity(
            query_resized.unsqueeze(0),
            self.memory_matrix[:len(self.memories)],
            dim=1
        )
        
        # Boost scores based on importance and access patterns
        boosted_scores = similarities * (1 + self.memory_importance[:len(self.memories)] * 0.2)
        
        # Get top-k memories
        _, indices = torch.topk(boosted_scores, min(k, len(self.memories)))
        
        # Update access counts
        for idx in indices:
            self.memory_access_count[idx] += 1
        
        retrieved_memories = [self.memories[i] for i in indices]
        
        # Also check consolidated memories
        for cons_id, cons_data in self.consolidated_memories.items():
            cons_similarity = torch.cosine_similarity(query_resized, cons_data['content'], dim=0)
            if cons_similarity > 0.6:  # Threshold for consolidated memory relevance
                # Create a virtual memory representing the consolidation
                cons_memory = Thought(
                    content=cons_data['content'],
                    salience=float(cons_similarity),
                    associations=[f"consolidated_from_{idx}" for idx in cons_data['source_indices']],
                    timestamp=cons_data['consolidation_time'],
                    pattern_type="consolidated",
                    confidence=float(cons_similarity)
                )
                retrieved_memories.append(cons_memory)
        
        return retrieved_memories

class ReasoningModule(nn.Module):
    """Enhanced cognitive reasoning mechanisms with advanced pattern recognition"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        self.feature_dim = self.config.get('feature_dim', 512)
        
        # Enhanced memory systems
        self.episodic_memory = EnhancedEpisodicMemory(
            memory_size=self.config.get('memory_size', 1000),
            feature_dim=self.feature_dim,
            consolidation_threshold=self.config.get('consolidation_threshold', 0.8)
        )
        
        # Advanced pattern detection
        self.pattern_detector = AdvancedPatternDetector(
            feature_dim=self.feature_dim,
            max_patterns=self.config.get('max_patterns', 1000)
        )
        
        # Enhanced reasoning networks
        self.pattern_recognizer = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.inference_network = nn.Sequential(
            nn.Linear(64 + self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.Tanh()
        )
        
        # Multi-head attention for enhanced working memory
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Pattern integration network
        self.pattern_integrator = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),  # Current + pattern + memory
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Tanh()
        )
        
        # Cognitive flexibility tracker
        self.flexibility_metrics = {
            'pattern_diversity': deque(maxlen=50),
            'reasoning_paths': deque(maxlen=50),
            'adaptation_rate': deque(maxlen=50),
            'memory_efficiency': deque(maxlen=50)
        }
        
        # Keep track of input history for temporal pattern detection
        self.input_history = deque(maxlen=20)
    
    def process_thought(self, 
                       current_input: torch.Tensor,
                       working_memory: Dict[str, Any]) -> Tuple[Thought, Dict[str, Any]]:
        """Enhanced thought processing with advanced pattern recognition"""
        
        # Add to input history for temporal analysis (ensure consistent size)
        if current_input.numel() == self.feature_dim:
            self.input_history.append(current_input.clone())
        else:
            # Resize input to feature_dim
            if current_input.numel() > self.feature_dim:
                resized_input = current_input.flatten()[:self.feature_dim]
            else:
                padding = torch.zeros(self.feature_dim - current_input.numel())
                resized_input = torch.cat([current_input.flatten(), padding])
            self.input_history.append(resized_input)
        
        # Advanced pattern recognition
        patterns = self._detect_all_patterns(current_input)
        
        # Enhanced pattern recognition with input size handling
        if current_input.numel() != self.feature_dim:
            if current_input.numel() > self.feature_dim:
                resized_input = current_input.flatten()[:self.feature_dim]
            else:
                padding = torch.zeros(self.feature_dim - current_input.numel())
                resized_input = torch.cat([current_input.flatten(), padding])
        else:
            resized_input = current_input
        
        pattern_features = self.pattern_recognizer(resized_input)
        
        # Retrieve enhanced memories
        relevant_memories = self.episodic_memory.retrieve(resized_input, k=8)
        memory_tensors = [m.content for m in relevant_memories] if relevant_memories else [resized_input]
        
        # Ensure all memory tensors are the same size
        normalized_memories = []
        for memory_tensor in memory_tensors[:5]:  # Limit to top 5 for efficiency
            if memory_tensor.numel() != self.feature_dim:
                if memory_tensor.numel() > self.feature_dim:
                    normalized = memory_tensor.flatten()[:self.feature_dim]
                else:
                    padding = torch.zeros(self.feature_dim - memory_tensor.numel())
                    normalized = torch.cat([memory_tensor.flatten(), padding])
            else:
                normalized = memory_tensor
            normalized_memories.append(normalized)
        
        memory_tensor = torch.stack(normalized_memories) if normalized_memories else resized_input.unsqueeze(0)
        
        # Apply enhanced attention over memories and current input
        query = resized_input.unsqueeze(0)
        attended_memory, attention_weights = self.attention(
            query, memory_tensor, memory_tensor
        )
        
        # Integrate patterns with current processing
        if patterns:
            # Use the most confident pattern
            best_pattern = max(patterns, key=lambda p: p.confidence)
            pattern_tensor = best_pattern.features
            
            # Resize pattern tensor to match feature_dim if needed
            if pattern_tensor.numel() != self.feature_dim:
                if pattern_tensor.numel() < self.feature_dim:
                    # Pad with zeros
                    padding = torch.zeros(self.feature_dim - pattern_tensor.numel())
                    pattern_tensor = torch.cat([pattern_tensor.flatten(), padding])
                else:
                    # Truncate
                    pattern_tensor = pattern_tensor.flatten()[:self.feature_dim]
            
            # Integrate current input, pattern, and memory
            integration_input = torch.cat([
                current_input,
                pattern_tensor,
                attended_memory.squeeze(0)
            ])
            integrated_features = self.pattern_integrator(integration_input)
        else:
            integrated_features = attended_memory.squeeze(0)
        
        # Generate enhanced thought through inference
        inference_input = torch.cat([pattern_features, integrated_features])
        new_thought_content = self.inference_network(inference_input)
        
        # Determine thought characteristics
        thought_confidence = self._calculate_thought_confidence(
            patterns, attention_weights, relevant_memories
        )
        
        thought_pattern_type = patterns[0].pattern_type if patterns else "basic"
        
        # Create enhanced thought object
        thought = Thought(
            content=new_thought_content,
            salience=float(torch.max(torch.abs(new_thought_content))),
            associations=[f"memory_{i}" for i in range(len(relevant_memories))],
            timestamp=float(torch.rand(1)),  # Placeholder for actual timestamp
            pattern_type=thought_pattern_type,
            confidence=thought_confidence,
            context={
                'patterns_detected': len(patterns),
                'memory_relevance': float(torch.mean(attention_weights)) if attention_weights.numel() > 0 else 0.0,
                'pattern_types': [p.pattern_type for p in patterns]
            }
        )
        
        # Update working memory with enhanced information
        working_memory['last_thought'] = thought
        working_memory['active_patterns'] = patterns
        working_memory['pattern_features'] = pattern_features
        working_memory['attention_weights'] = attention_weights
        working_memory['flexibility_metrics'] = self._update_flexibility_metrics(
            patterns, thought, relevant_memories
        )
        
        # Store enhanced thought in episodic memory
        self.episodic_memory.store(thought)
        
        return thought, working_memory
    
    def _detect_all_patterns(self, current_input: torch.Tensor) -> List[PatternSignature]:
        """Detect all types of patterns in the current input"""
        all_patterns = []
        
        # Temporal patterns (if we have history)
        if len(self.input_history) >= 2:
            temporal_patterns = self.pattern_detector.detect_temporal_patterns(
                list(self.input_history)
            )
            all_patterns.extend(temporal_patterns)
        
        # Spatial patterns
        spatial_patterns = self.pattern_detector.detect_spatial_patterns(current_input)
        all_patterns.extend(spatial_patterns)
        
        # Hierarchical patterns
        hierarchical_patterns = self.pattern_detector.detect_hierarchical_patterns(current_input)
        all_patterns.extend(hierarchical_patterns)
        
        # Associative patterns (with recent memories)
        recent_memories = [m.content for m in self.episodic_memory.memories[-10:]]
        if recent_memories:
            associative_patterns = self.pattern_detector.detect_associative_patterns(
                current_input, recent_memories
            )
            all_patterns.extend(associative_patterns)
        
        return all_patterns
    
    def _calculate_thought_confidence(self, patterns: List[PatternSignature], 
                                    attention_weights: torch.Tensor,
                                    memories: List[Thought]) -> float:
        """Calculate confidence in the generated thought"""
        confidence = 0.5  # Base confidence
        
        # Boost based on pattern detection
        if patterns:
            pattern_confidence = sum(p.confidence for p in patterns) / len(patterns)
            confidence += pattern_confidence * 0.3
        
        # Boost based on memory relevance
        if attention_weights.numel() > 0:
            memory_relevance = float(torch.max(attention_weights))
            confidence += memory_relevance * 0.2
        
        # Boost based on memory quality
        if memories:
            avg_memory_confidence = sum(getattr(m, 'confidence', 1.0) for m in memories) / len(memories)
            confidence += avg_memory_confidence * 0.1
        
        return min(1.0, confidence)
    
    def _update_flexibility_metrics(self, patterns: List[PatternSignature], 
                                  thought: Thought, memories: List[Thought]) -> Dict[str, float]:
        """Update cognitive flexibility metrics"""
        metrics = {}
        
        # Pattern diversity - how many different pattern types detected
        pattern_types = set(p.pattern_type for p in patterns)
        self.flexibility_metrics['pattern_diversity'].append(len(pattern_types))
        metrics['pattern_diversity'] = np.mean(self.flexibility_metrics['pattern_diversity'])
        
        # Reasoning path complexity - based on pattern integration
        reasoning_complexity = len(patterns) + len(memories) + len(thought.associations)
        self.flexibility_metrics['reasoning_paths'].append(reasoning_complexity)
        metrics['reasoning_complexity'] = np.mean(self.flexibility_metrics['reasoning_paths'])
        
        # Adaptation rate - how quickly patterns change
        if len(self.input_history) >= 2:
            current_patterns = set(p.pattern_type for p in patterns)
            # This is a simplified metric - in practice would compare with previous patterns
            adaptation = len(current_patterns) / max(1, len(self.input_history))
            self.flexibility_metrics['adaptation_rate'].append(adaptation)
            metrics['adaptation_rate'] = np.mean(self.flexibility_metrics['adaptation_rate'])
        
        # Memory efficiency - ratio of memory hits to total memories
        memory_efficiency = len(memories) / max(1, len(self.episodic_memory.memories))
        self.flexibility_metrics['memory_efficiency'].append(memory_efficiency)
        metrics['memory_efficiency'] = np.mean(self.flexibility_metrics['memory_efficiency'])
        
        return metrics
    
    def get_cognitive_flexibility_report(self) -> Dict[str, Any]:
        """Get a comprehensive report on cognitive flexibility"""
        report = {}
        
        for metric_name, values in self.flexibility_metrics.items():
            if values:
                report[metric_name] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0,
                    'variability': np.std(values)
                }
        
        # Overall flexibility score - only if all metrics available
        required_metrics = ['pattern_diversity', 'reasoning_paths', 'adaptation_rate', 'memory_efficiency']
        if all(metric in report for metric in required_metrics):
            flexibility_score = (
                report['pattern_diversity']['average'] * 0.3 +
                report['reasoning_paths']['average'] * 0.25 +
                report['adaptation_rate']['average'] * 0.25 +
                report['memory_efficiency']['average'] * 0.2
            )
            report['overall_flexibility'] = flexibility_score
        
        return report
    
    def forward(self, 
                sensory_input: torch.Tensor,
                working_memory: Dict[str, Any]) -> Tuple[Thought, Dict[str, Any]]:
        """Forward pass of reasoning module"""
        return self.process_thought(sensory_input, working_memory) 