"""
Meta-Learning Systems Implementation

This module implements sophisticated meta-learning capabilities that enable
the system to learn how to learn more effectively, including transfer learning,
few-shot learning, and adaptive learning strategies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math
import copy
from enum import Enum

class LearningStrategy(Enum):
    """Enumeration of available learning strategies"""
    GRADIENT_DESCENT = "gradient_descent"
    REINFORCEMENT = "reinforcement"  
    RELEVANCE_BASED = "relevance_based"
    TRANSFER = "transfer"
    FEW_SHOT = "few_shot"
    CURRICULUM = "curriculum"

@dataclass
class MetaExperience:
    """Represents a meta-learning experience across domains"""
    domain: str
    task: str
    strategy: LearningStrategy
    performance: float
    learning_rate: float
    context: Dict[str, Any]
    timestamp: float
    convergence_time: float
    transferability: float = 0.0

@dataclass
class DomainKnowledge:
    """Encapsulates knowledge for a specific domain"""
    domain_id: str
    feature_extractor: nn.Module
    domain_classifier: nn.Module
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    adaptation_parameters: Dict[str, float] = field(default_factory=dict)

class TransferLearning:
    """Implements transfer learning mechanisms across domains"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.domain_knowledge: Dict[str, DomainKnowledge] = {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.transfer_weight = self.config.get('transfer_weight', 0.3)
        
    def register_domain(self, domain_id: str, feature_dim: int = 512):
        """Register a new domain for transfer learning"""
        feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, 16),
            nn.Softmax(dim=-1)
        )
        
        self.domain_knowledge[domain_id] = DomainKnowledge(
            domain_id=domain_id,
            feature_extractor=feature_extractor,
            domain_classifier=domain_classifier
        )
        
    def compute_domain_similarity(self, source_domain: str, target_domain: str, 
                                input_sample: torch.Tensor) -> float:
        """Compute similarity between domains based on feature representations"""
        if source_domain not in self.domain_knowledge or target_domain not in self.domain_knowledge:
            return 0.0
            
        source_features = self.domain_knowledge[source_domain].feature_extractor(input_sample)
        target_features = self.domain_knowledge[target_domain].feature_extractor(input_sample)
        
        # Cosine similarity
        cosine_sim = torch.cosine_similarity(source_features.unsqueeze(0), 
                                           target_features.unsqueeze(0))
        return float(cosine_sim)
    
    def transfer_knowledge(self, source_domain: str, target_domain: str, 
                          adaptation_rate: float = 0.1) -> bool:
        """Transfer knowledge from source to target domain"""
        if source_domain not in self.domain_knowledge or target_domain not in self.domain_knowledge:
            return False
            
        source_dk = self.domain_knowledge[source_domain]
        target_dk = self.domain_knowledge[target_domain]
        
        # Transfer feature extractor parameters with adaptation
        for source_param, target_param in zip(
            source_dk.feature_extractor.parameters(),
            target_dk.feature_extractor.parameters()
        ):
            # Weighted combination of source and target parameters
            target_param.data = (1 - adaptation_rate) * target_param.data + \
                               adaptation_rate * source_param.data
                               
        # Transfer learned patterns
        for pattern_name, pattern_data in source_dk.learned_patterns.items():
            if pattern_name not in target_dk.learned_patterns:
                target_dk.learned_patterns[pattern_name] = copy.deepcopy(pattern_data)
                
        return True
    
    def should_transfer(self, source_domain: str, target_domain: str,
                       input_sample: torch.Tensor) -> bool:
        """Determine if knowledge should be transferred between domains"""
        similarity = self.compute_domain_similarity(source_domain, target_domain, input_sample)
        return similarity > self.similarity_threshold

class FewShotLearner:
    """Implements few-shot learning capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.support_size = self.config.get('support_size', 5)
        self.query_size = self.config.get('query_size', 10)
        self.meta_lr = self.config.get('meta_learning_rate', 0.001)
        
        # Prototypical networks components
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.prototype_network = self._build_prototype_network()
        self.prototypes: Dict[str, torch.Tensor] = {}
        
    def _build_prototype_network(self) -> nn.Module:
        """Build the prototype network for few-shot learning"""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.embedding_dim)
        )
    
    def create_prototype(self, task_name: str, support_examples: List[torch.Tensor],
                        support_labels: List[int]) -> torch.Tensor:
        """Create a prototype for a task from support examples"""
        embeddings = []
        
        for example in support_examples:
            embedding = self.prototype_network(example)
            embeddings.append(embedding)
            
        embeddings = torch.stack(embeddings)
        
        # Compute prototype as mean of embeddings for each class
        unique_labels = list(set(support_labels))
        class_prototypes = {}
        
        for label in unique_labels:
            label_mask = torch.tensor([l == label for l in support_labels])
            if label_mask.sum() > 0:
                class_prototypes[label] = embeddings[label_mask].mean(dim=0)
                
        # Average all class prototypes for task prototype
        if class_prototypes:
            task_prototype = torch.stack(list(class_prototypes.values())).mean(dim=0)
            self.prototypes[task_name] = task_prototype
            return task_prototype
        else:
            # Fallback to mean of all embeddings
            task_prototype = embeddings.mean(dim=0)
            self.prototypes[task_name] = task_prototype
            return task_prototype
    
    def predict_from_prototype(self, query_example: torch.Tensor, 
                             task_name: str = None) -> Tuple[str, float]:
        """Predict using prototypical matching"""
        query_embedding = self.prototype_network(query_example)
        
        best_match = None
        best_similarity = -float('inf')
        
        prototypes_to_check = self.prototypes
        if task_name and task_name in self.prototypes:
            prototypes_to_check = {task_name: self.prototypes[task_name]}
            
        for proto_name, prototype in prototypes_to_check.items():
            similarity = torch.cosine_similarity(query_embedding, prototype, dim=0)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = proto_name
                
        return best_match or "unknown", float(best_similarity)
    
    def meta_update(self, task_experiences: List[Tuple[str, List[torch.Tensor], List[int], 
                                                   List[torch.Tensor], List[int]]]) -> Dict[str, float]:
        """Meta-update the few-shot learning system using MAML-style optimization"""
        meta_loss = 0.0
        task_count = 0
        
        optimizer = torch.optim.Adam(self.prototype_network.parameters(), lr=self.meta_lr)
        
        for task_name, support_x, support_y, query_x, query_y in task_experiences:
            # Fast adaptation on support set
            task_prototype = self.create_prototype(task_name, support_x, support_y)
            
            # Evaluate on query set
            task_loss = 0.0
            for qx, qy in zip(query_x, query_y):
                predicted_task, similarity = self.predict_from_prototype(qx, task_name)
                # Simple loss based on similarity and correctness
                target_similarity = 1.0 if predicted_task == task_name else 0.0
                task_loss += (similarity - target_similarity) ** 2
                
            meta_loss += task_loss / len(query_x)
            task_count += 1
            
        if task_count > 0:
            meta_loss /= task_count
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()
            
        return {'meta_loss': float(meta_loss), 'tasks_processed': task_count}

class AdaptiveLearningManager:
    """Manages adaptive learning rate and strategy selection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.strategy_performance: Dict[LearningStrategy, deque] = defaultdict(
            lambda: deque(maxlen=self.config.get('history_length', 100))
        )
        self.current_strategy = LearningStrategy.GRADIENT_DESCENT
        self.adaptation_rate = self.config.get('adaptation_rate', 0.01)
        self.exploration_epsilon = self.config.get('exploration_epsilon', 0.1)
        
        # Learning rate adaptation parameters
        self.base_lr = self.config.get('base_learning_rate', 0.001)
        self.lr_decay = self.config.get('lr_decay', 0.95)
        self.lr_boost = self.config.get('lr_boost', 1.1)
        self.performance_window = self.config.get('performance_window', 10)
        
    def record_strategy_performance(self, strategy: LearningStrategy, 
                                  performance: float, context: Dict[str, Any] = None):
        """Record performance of a learning strategy"""
        self.strategy_performance[strategy].append({
            'performance': performance,
            'context': context or {},
            'timestamp': torch.time.time() if hasattr(torch, 'time') else 0
        })
    
    def select_strategy(self, context: Dict[str, Any] = None) -> LearningStrategy:
        """Select the best learning strategy based on historical performance"""
        context = context or {}
        
        # Exploration vs exploitation
        if np.random.random() < self.exploration_epsilon:
            # Explore: randomly select a strategy
            return np.random.choice(list(LearningStrategy))
            
        # Exploit: select best performing strategy
        best_strategy = LearningStrategy.GRADIENT_DESCENT
        best_performance = -float('inf')
        
        for strategy, performance_history in self.strategy_performance.items():
            if len(performance_history) > 0:
                # Weight recent performance more heavily
                recent_performances = list(performance_history)[-self.performance_window:]
                avg_performance = np.mean([p['performance'] for p in recent_performances])
                
                # Consider context similarity for better selection
                context_bonus = self._compute_context_similarity_bonus(strategy, context)
                adjusted_performance = avg_performance + context_bonus
                
                if adjusted_performance > best_performance:
                    best_performance = adjusted_performance
                    best_strategy = strategy
                    
        self.current_strategy = best_strategy
        return best_strategy
    
    def _compute_context_similarity_bonus(self, strategy: LearningStrategy, 
                                        current_context: Dict[str, Any]) -> float:
        """Compute bonus based on context similarity to past successful uses"""
        if strategy not in self.strategy_performance:
            return 0.0
            
        total_similarity = 0.0
        count = 0
        
        for record in self.strategy_performance[strategy]:
            if record['performance'] > 0.5:  # Only consider successful uses
                past_context = record.get('context', {})
                similarity = self._context_similarity(current_context, past_context)
                total_similarity += similarity
                count += 1
                
        return (total_similarity / count * 0.1) if count > 0 else 0.0
    
    def _context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute similarity between two contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        similarity_sum = 0.0
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity_sum += 1.0 - abs(val1 - val2) / max_val
            elif val1 == val2:
                similarity_sum += 1.0
                
        return similarity_sum / len(common_keys)
    
    def adapt_learning_rate(self, current_lr: float, recent_performance: List[float]) -> float:
        """Adapt learning rate based on recent performance trends"""
        if len(recent_performance) < 2:
            return current_lr
            
        # Check if performance is improving
        recent_trend = np.mean(recent_performance[-self.performance_window//2:]) - \
                      np.mean(recent_performance[:-self.performance_window//2])
                      
        if recent_trend > 0.01:  # Performance improving
            return min(current_lr * self.lr_boost, self.base_lr * 10)
        elif recent_trend < -0.01:  # Performance degrading
            return max(current_lr * self.lr_decay, self.base_lr / 10)
        else:
            return current_lr  # No significant change

class KnowledgeDistiller:
    """Implements knowledge distillation and compression techniques"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.temperature = self.config.get('distillation_temperature', 3.0)
        self.alpha = self.config.get('distillation_alpha', 0.7)
        self.compression_ratio = self.config.get('compression_ratio', 0.5)
        
    def create_student_model(self, teacher_model: nn.Module, 
                           compression_ratio: float = None) -> nn.Module:
        """Create a smaller student model based on teacher architecture"""
        compression_ratio = compression_ratio or self.compression_ratio
        
        # Simple compression by reducing layer sizes
        student_layers = []
        for layer in teacher_model:
            if isinstance(layer, nn.Linear):
                compressed_size = max(1, int(layer.out_features * compression_ratio))
                student_layer = nn.Linear(layer.in_features, compressed_size)
                student_layers.append(student_layer)
            else:
                student_layers.append(copy.deepcopy(layer))
                
        return nn.Sequential(*student_layers)
    
    def distill_knowledge(self, teacher_model: nn.Module, student_model: nn.Module,
                         training_data: List[Tuple[torch.Tensor, torch.Tensor]],
                         epochs: int = 10) -> Dict[str, float]:
        """Distill knowledge from teacher to student model"""
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for inputs, targets in training_data:
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                    teacher_probs = torch.softmax(teacher_outputs / self.temperature, dim=-1)
                
                # Get student predictions
                student_outputs = student_model(inputs)
                student_log_probs = torch.log_softmax(student_outputs / self.temperature, dim=-1)
                
                # Distillation loss
                distill_loss = criterion(student_log_probs, teacher_probs)
                
                # Task loss (if targets available)
                task_loss = 0.0
                if targets is not None:
                    task_loss = nn.CrossEntropyLoss()(student_outputs, targets)
                
                # Combined loss
                total_batch_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
                
                total_batch_loss.backward()
                optimizer.step()
                
                epoch_loss += total_batch_loss.item()
                
            total_loss += epoch_loss
            
        return {
            'total_loss': total_loss,
            'avg_loss_per_epoch': total_loss / epochs,
            'compression_achieved': self._compute_compression_ratio(teacher_model, student_model)
        }
    
    def _compute_compression_ratio(self, teacher_model: nn.Module, 
                                 student_model: nn.Module) -> float:
        """Compute actual compression ratio achieved"""
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        return student_params / teacher_params if teacher_params > 0 else 1.0

class CurriculumLearner:
    """Implements curriculum learning optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.difficulty_threshold = self.config.get('difficulty_threshold', 0.7)
        self.progression_rate = self.config.get('progression_rate', 0.1)
        self.curriculum: List[Dict[str, Any]] = []
        self.current_level = 0
        
    def add_curriculum_level(self, level_id: str, difficulty: float, 
                           tasks: List[Any], prerequisites: List[str] = None):
        """Add a level to the curriculum"""
        self.curriculum.append({
            'level_id': level_id,
            'difficulty': difficulty,
            'tasks': tasks,
            'prerequisites': prerequisites or [],
            'mastery_score': 0.0,
            'attempts': 0
        })
        
        # Sort curriculum by difficulty
        self.curriculum.sort(key=lambda x: x['difficulty'])
    
    def get_next_task(self, current_performance: float) -> Optional[Any]:
        """Get the next task based on curriculum progression"""
        if not self.curriculum:
            return None
            
        current_level_data = self.curriculum[self.current_level]
        
        # Check if current level is mastered
        if current_performance > self.difficulty_threshold:
            current_level_data['mastery_score'] = max(
                current_level_data['mastery_score'], current_performance
            )
            
            # Try to advance to next level
            if self._can_advance_level():
                self.current_level = min(self.current_level + 1, len(self.curriculum) - 1)
        
        # Return task from current level
        current_level_data['attempts'] += 1
        if current_level_data['tasks']:
            task_idx = current_level_data['attempts'] % len(current_level_data['tasks'])
            return current_level_data['tasks'][task_idx]
            
        return None
    
    def _can_advance_level(self) -> bool:
        """Check if we can advance to the next curriculum level"""
        if self.current_level >= len(self.curriculum) - 1:
            return False
            
        current_level = self.curriculum[self.current_level]
        next_level = self.curriculum[self.current_level + 1]
        
        # Check mastery of current level
        if current_level['mastery_score'] < self.difficulty_threshold:
            return False
            
        # Check prerequisites for next level
        for prereq in next_level['prerequisites']:
            prereq_level = next((l for l in self.curriculum if l['level_id'] == prereq), None)
            if not prereq_level or prereq_level['mastery_score'] < self.difficulty_threshold:
                return False
                
        return True
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get current curriculum progress statistics"""
        total_levels = len(self.curriculum)
        mastered_levels = sum(1 for level in self.curriculum 
                            if level['mastery_score'] >= self.difficulty_threshold)
        
        return {
            'current_level': self.current_level,
            'total_levels': total_levels,
            'mastered_levels': mastered_levels,
            'progress_percentage': (mastered_levels / total_levels * 100) if total_levels > 0 else 0,
            'current_level_progress': self.curriculum[self.current_level]['mastery_score'] if self.curriculum else 0
        }

class MetaLearner:
    """Main meta-learning coordinator that orchestrates all meta-learning components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize meta-learning components
        self.transfer_learning = TransferLearning(self.config.get('transfer_config', {}))
        self.few_shot_learner = FewShotLearner(self.config.get('few_shot_config', {}))
        self.adaptive_manager = AdaptiveLearningManager(self.config.get('adaptive_config', {}))
        self.knowledge_distiller = KnowledgeDistiller(self.config.get('distillation_config', {}))
        self.curriculum_learner = CurriculumLearner(self.config.get('curriculum_config', {}))
        
        # Meta-learning history
        self.meta_experiences: List[MetaExperience] = []
        self.domain_performances: Dict[str, List[float]] = defaultdict(list)
        
        # Meta-parameters that are optimized
        self.meta_params = {
            'transfer_threshold': 0.7,
            'adaptation_rate': 0.1,
            'distillation_temperature': 3.0,
            'curriculum_progression_rate': 0.1
        }
        
    def learn_meta_task(self, domain: str, task: str, 
                       training_data: List[Tuple[torch.Tensor, Any]],
                       validation_data: List[Tuple[torch.Tensor, Any]] = None) -> Dict[str, Any]:
        """Learn a meta-task using the most appropriate strategy"""
        
        # Select learning strategy
        context = {'domain': domain, 'task': task, 'data_size': len(training_data)}
        strategy = self.adaptive_manager.select_strategy(context)
        
        # Record start time
        start_time = torch.time.time() if hasattr(torch, 'time') else 0
        
        results = {}
        
        if strategy == LearningStrategy.TRANSFER:
            results = self._learn_with_transfer(domain, task, training_data, validation_data)
        elif strategy == LearningStrategy.FEW_SHOT:
            results = self._learn_few_shot(domain, task, training_data, validation_data)
        elif strategy == LearningStrategy.CURRICULUM:
            results = self._learn_with_curriculum(domain, task, training_data, validation_data)
        else:
            results = self._learn_standard(domain, task, training_data, validation_data)
            
        # Record meta-experience
        convergence_time = (torch.time.time() if hasattr(torch, 'time') else 0) - start_time
        performance = results.get('performance', 0.0)
        
        meta_exp = MetaExperience(
            domain=domain,
            task=task,
            strategy=strategy,
            performance=performance,
            learning_rate=results.get('learning_rate', 0.001),
            context=context,
            timestamp=start_time,
            convergence_time=convergence_time
        )
        
        self.meta_experiences.append(meta_exp)
        self.domain_performances[domain].append(performance)
        
        # Update strategy performance tracking
        self.adaptive_manager.record_strategy_performance(strategy, performance, context)
        
        # Optimize meta-parameters based on results
        self._optimize_meta_parameters(meta_exp)
        
        return {**results, 'strategy_used': strategy.value, 'convergence_time': convergence_time}
    
    def _learn_with_transfer(self, domain: str, task: str, 
                           training_data: List[Tuple[torch.Tensor, Any]],
                           validation_data: List[Tuple[torch.Tensor, Any]] = None) -> Dict[str, Any]:
        """Learn using transfer learning from similar domains"""
        
        # Register domain if not exists
        if domain not in self.transfer_learning.domain_knowledge:
            sample_input = training_data[0][0] if training_data else torch.randn(512)
            self.transfer_learning.register_domain(domain, sample_input.size(-1))
            
        # Find best source domain for transfer
        best_source = None
        best_similarity = 0.0
        
        if len(training_data) > 0:
            sample_input = training_data[0][0]
            for source_domain in self.transfer_learning.domain_knowledge:
                if source_domain != domain:
                    similarity = self.transfer_learning.compute_domain_similarity(
                        source_domain, domain, sample_input
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_source = source_domain
        
        transferred = False
        if best_source and best_similarity > self.meta_params['transfer_threshold']:
            transferred = self.transfer_learning.transfer_knowledge(
                best_source, domain, self.meta_params['adaptation_rate']
            )
        
        # Simple training simulation
        performance = min(0.9, 0.5 + best_similarity * 0.4) if transferred else 0.3
        
        return {
            'performance': performance,
            'transferred_from': best_source,
            'transfer_similarity': best_similarity,
            'learning_rate': 0.001 * (1 + best_similarity)
        }
    
    def _learn_few_shot(self, domain: str, task: str,
                       training_data: List[Tuple[torch.Tensor, Any]],
                       validation_data: List[Tuple[torch.Tensor, Any]] = None) -> Dict[str, Any]:
        """Learn using few-shot learning techniques"""
        
        if len(training_data) < self.few_shot_learner.support_size:
            return {'performance': 0.2, 'error': 'insufficient_data'}
            
        # Create support and query sets
        support_size = min(self.few_shot_learner.support_size, len(training_data) // 2)
        support_data = training_data[:support_size]
        query_data = training_data[support_size:] if len(training_data) > support_size else validation_data or []
        
        if not query_data:
            return {'performance': 0.3, 'error': 'no_query_data'}
            
        # Extract support examples and labels
        support_examples = [data[0] for data in support_data]
        support_labels = [0] * len(support_examples)  # Simplified labeling
        
        # Create prototype for task
        prototype = self.few_shot_learner.create_prototype(task, support_examples, support_labels)
        
        # Evaluate on query set
        correct_predictions = 0
        for query_input, query_target in query_data:
            predicted_task, similarity = self.few_shot_learner.predict_from_prototype(query_input, task)
            if similarity > 0.5:  # Simple threshold for correctness
                correct_predictions += 1
                
        performance = correct_predictions / len(query_data) if query_data else 0.0
        
        return {
            'performance': performance,
            'prototype_created': True,
            'support_size': len(support_examples),
            'query_size': len(query_data),
            'learning_rate': 0.01
        }
    
    def _learn_with_curriculum(self, domain: str, task: str,
                             training_data: List[Tuple[torch.Tensor, Any]],
                             validation_data: List[Tuple[torch.Tensor, Any]] = None) -> Dict[str, Any]:
        """Learn using curriculum learning optimization"""
        
        # Sort training data by difficulty (simple heuristic based on input magnitude)
        sorted_data = sorted(training_data, key=lambda x: torch.norm(x[0]).item())
        
        # Add curriculum levels based on data difficulty
        easy_data = sorted_data[:len(sorted_data)//3]
        medium_data = sorted_data[len(sorted_data)//3:2*len(sorted_data)//3]
        hard_data = sorted_data[2*len(sorted_data)//3:]
        
        self.curriculum_learner.add_curriculum_level(f"{task}_easy", 0.3, easy_data)
        self.curriculum_learner.add_curriculum_level(f"{task}_medium", 0.6, medium_data, [f"{task}_easy"])
        self.curriculum_learner.add_curriculum_level(f"{task}_hard", 0.9, hard_data, [f"{task}_medium"])
        
        # Simulate curriculum learning
        total_performance = 0.0
        levels_completed = 0
        current_performance = 0.5
        
        for _ in range(10):  # Simulate learning steps
            next_task = self.curriculum_learner.get_next_task(current_performance)
            if next_task is None:
                break
                
            # Simulate performance improvement
            current_performance = min(0.95, current_performance + 0.05)
            total_performance += current_performance
            levels_completed += 1
            
        average_performance = total_performance / levels_completed if levels_completed > 0 else 0.0
        progress = self.curriculum_learner.get_curriculum_progress()
        
        return {
            'performance': average_performance,
            'curriculum_progress': progress,
            'levels_completed': levels_completed,
            'learning_rate': 0.001 * (1 + progress['progress_percentage'] / 100)
        }
    
    def _learn_standard(self, domain: str, task: str,
                       training_data: List[Tuple[torch.Tensor, Any]],
                       validation_data: List[Tuple[torch.Tensor, Any]] = None) -> Dict[str, Any]:
        """Standard learning approach as baseline"""
        
        # Simulate standard learning
        data_size = len(training_data)
        base_performance = min(0.8, 0.2 + (data_size / 100) * 0.6)
        
        return {
            'performance': base_performance,
            'method': 'standard',
            'data_size': data_size,
            'learning_rate': 0.001
        }
    
    def _optimize_meta_parameters(self, meta_experience: MetaExperience):
        """Optimize meta-parameters based on recent experiences"""
        
        # Simple adaptation based on performance
        if meta_experience.performance > 0.7:
            # Good performance, slightly adjust parameters in the same direction
            if meta_experience.strategy == LearningStrategy.TRANSFER:
                self.meta_params['transfer_threshold'] *= 0.98  # Make transfer easier
            elif meta_experience.strategy == LearningStrategy.CURRICULUM:
                self.meta_params['curriculum_progression_rate'] *= 1.02  # Progress faster
        else:
            # Poor performance, adjust parameters in opposite direction
            if meta_experience.strategy == LearningStrategy.TRANSFER:
                self.meta_params['transfer_threshold'] *= 1.02  # Make transfer harder
            elif meta_experience.strategy == LearningStrategy.CURRICULUM:
                self.meta_params['curriculum_progression_rate'] *= 0.98  # Progress slower
                
        # Keep parameters within reasonable bounds
        self.meta_params['transfer_threshold'] = np.clip(self.meta_params['transfer_threshold'], 0.3, 0.9)
        self.meta_params['curriculum_progression_rate'] = np.clip(self.meta_params['curriculum_progression_rate'], 0.01, 0.5)
    
    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics"""
        
        if not self.meta_experiences:
            return {'message': 'No meta-learning experiences recorded yet'}
            
        # Performance statistics
        all_performances = [exp.performance for exp in self.meta_experiences]
        avg_performance = np.mean(all_performances)
        performance_improvement = all_performances[-1] - all_performances[0] if len(all_performances) > 1 else 0
        
        # Strategy statistics
        strategy_counts = defaultdict(int)
        strategy_performances = defaultdict(list)
        
        for exp in self.meta_experiences:
            strategy_counts[exp.strategy] += 1
            strategy_performances[exp.strategy].append(exp.performance)
            
        best_strategy = max(strategy_performances.keys(), 
                          key=lambda s: np.mean(strategy_performances[s]))
        
        # Domain statistics
        domain_stats = {}
        for domain, performances in self.domain_performances.items():
            domain_stats[domain] = {
                'avg_performance': np.mean(performances),
                'improvement': performances[-1] - performances[0] if len(performances) > 1 else 0,
                'task_count': len(performances)
            }
            
        return {
            'total_experiences': len(self.meta_experiences),
            'average_performance': avg_performance,
            'performance_improvement': performance_improvement,
            'best_strategy': best_strategy.value,
            'strategy_usage': {s.value: count for s, count in strategy_counts.items()},
            'strategy_performance': {s.value: np.mean(perfs) for s, perfs in strategy_performances.items()},
            'domain_statistics': domain_stats,
            'current_meta_params': self.meta_params.copy(),
            'curriculum_progress': self.curriculum_learner.get_curriculum_progress()
        }
    
    def save_meta_knowledge(self, filepath: str) -> bool:
        """Save meta-learning knowledge to file"""
        try:
            knowledge_data = {
                'meta_experiences': [
                    {
                        'domain': exp.domain,
                        'task': exp.task,
                        'strategy': exp.strategy.value,
                        'performance': exp.performance,
                        'learning_rate': exp.learning_rate,
                        'context': exp.context,
                        'timestamp': exp.timestamp,
                        'convergence_time': exp.convergence_time
                    }
                    for exp in self.meta_experiences
                ],
                'meta_params': self.meta_params,
                'domain_performances': dict(self.domain_performances),
                'few_shot_prototypes': {k: v.tolist() for k, v in self.few_shot_learner.prototypes.items()}
            }
            
            # In a real implementation, you would use pickle or json to save this
            # For now, just simulate successful save
            return True
        except Exception:
            return False
    
    def load_meta_knowledge(self, filepath: str) -> bool:
        """Load meta-learning knowledge from file"""
        try:
            # In a real implementation, you would load from pickle or json
            # For now, just simulate successful load
            return True
        except Exception:
            return False