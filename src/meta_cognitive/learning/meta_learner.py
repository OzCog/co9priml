"""
Meta-Cognitive Learner
=====================

This module implements meta-cognitive learning and adaptation mechanisms
that enable the system to learn from meta-cognitive experiences and
continuously improve its meta-cognitive capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import logging
from ..interfaces.meta_cognitive_interface import (
    MetaLearningInterface, MetaCognitiveCapability
)


class LearningType(Enum):
    """Types of meta-cognitive learning."""
    STRATEGY_LEARNING = "strategy_learning"
    PERFORMANCE_LEARNING = "performance_learning"
    ADAPTATION_LEARNING = "adaptation_learning"
    CONTEXT_LEARNING = "context_learning"
    PATTERN_LEARNING = "pattern_learning"


@dataclass
class LearningExperience:
    """Represents a meta-cognitive learning experience."""
    experience_id: str
    learning_type: LearningType
    experience_data: Dict[str, Any]
    outcome: Dict[str, Any]
    learning_value: float
    timestamp: float
    context: Dict[str, Any]


class MetaCognitiveLearner(MetaLearningInterface):
    """
    Implementation of meta-cognitive learning and adaptation.
    
    This system provides:
    - Learning from meta-cognitive experiences
    - Strategy adaptation based on performance
    - Transfer learning across contexts
    - Continuous improvement of meta-cognitive capabilities
    - Experience-based optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the meta-cognitive learner."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.experience_retention_limit = self.config.get('experience_retention_limit', 1000)
        self.transfer_threshold = self.config.get('transfer_threshold', 0.7)
        
        # Learning state
        self.learning_experiences: List[LearningExperience] = []
        self.learned_patterns: Dict[str, Dict[str, Any]] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, float] = {}
        
        # Learning models
        self.strategy_effectiveness_model: Dict[str, float] = {}
        self.context_performance_model: Dict[str, Dict[str, float]] = {}
        self.adaptation_success_model: Dict[str, float] = {}
        
        self.logger.info("Meta-cognitive learner initialized")
    
    def initialize(self) -> bool:
        """Initialize the meta-learning component."""
        return True
    
    def shutdown(self) -> bool:
        """Shutdown the meta-learning component."""
        return True
    
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of meta-learning capabilities."""
        return [
            MetaCognitiveCapability(
                name="experience_learning",
                description="Learning from meta-cognitive experiences",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="strategy_adaptation",
                description="Adaptive strategy improvement",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="transfer_learning",
                description="Transfer learning across contexts",
                complexity_level=5,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="continuous_improvement",
                description="Continuous meta-cognitive improvement",
                complexity_level=4,
                requires_recursion=True,
                resource_intensive=True
            )
        ]
    
    def process_meta_cognitive_request(self, 
                                     request_type: str,
                                     request_data: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meta-cognitive request."""
        if request_type == "learn_experience":
            success = self.learn_from_experience(request_data)
            return {'success': success}
        elif request_type == "adapt_strategies":
            success = self.adapt_meta_strategies(request_data)
            return {'success': success}
        elif request_type == "transfer_knowledge":
            source = request_data.get('source_domain', 'general')
            target = request_data.get('target_domain', 'specific')
            result = self.transfer_meta_knowledge(source, target)
            return result
        else:
            return {'error': f'Unknown request type: {request_type}'}
    
    def learn_from_experience(self, experience_data: Dict[str, Any]) -> bool:
        """Learn from meta-cognitive experience."""
        try:
            # Extract learning components
            experience_type = self._determine_learning_type(experience_data)
            learning_value = self._calculate_learning_value(experience_data)
            
            # Create learning experience
            experience = LearningExperience(
                experience_id=f"exp_{len(self.learning_experiences)}_{time.time()}",
                learning_type=experience_type,
                experience_data=experience_data,
                outcome=experience_data.get('outcome', {}),
                learning_value=learning_value,
                timestamp=time.time(),
                context=experience_data.get('context', {})
            )
            
            # Store experience
            self.learning_experiences.append(experience)
            
            # Update learning models
            self._update_learning_models(experience)
            
            # Extract patterns
            self._extract_patterns_from_experience(experience)
            
            # Maintain experience history size
            if len(self.learning_experiences) > self.experience_retention_limit:
                self.learning_experiences = self.learning_experiences[-int(self.experience_retention_limit * 0.8):]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error learning from experience: {e}")
            return False
    
    def adapt_meta_strategies(self, performance_feedback: Dict[str, Any]) -> bool:
        """Adapt meta-cognitive strategies based on feedback."""
        try:
            # Analyze performance feedback
            performance_issues = performance_feedback.get('issues', [])
            performance_improvements = performance_feedback.get('improvements', [])
            context = performance_feedback.get('context', {})
            
            # Identify adaptation opportunities
            adaptations = self._identify_adaptations(
                performance_issues, 
                performance_improvements, 
                context
            )
            
            # Apply adaptations
            for adaptation in adaptations:
                success = self._apply_adaptation(adaptation)
                if success:
                    self.adaptation_history.append({
                        'adaptation': adaptation,
                        'timestamp': time.time(),
                        'context': context
                    })
            
            return len(adaptations) > 0
            
        except Exception as e:
            self.logger.error(f"Error adapting meta-strategies: {e}")
            return False
    
    def transfer_meta_knowledge(self, 
                              source_domain: str,
                              target_domain: str) -> Dict[str, Any]:
        """Transfer meta-knowledge between domains."""
        transfer_result = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'transferred_patterns': [],
            'transfer_success_rate': 0.0,
            'applicable_strategies': []
        }
        
        try:
            # Find transferable patterns
            source_patterns = self._get_domain_patterns(source_domain)
            transferable_patterns = []
            
            for pattern in source_patterns:
                transferability = self._assess_transferability(pattern, target_domain)
                if transferability > self.transfer_threshold:
                    transferable_patterns.append({
                        'pattern': pattern,
                        'transferability': transferability
                    })
            
            # Apply transfer
            successful_transfers = 0
            for transfer_item in transferable_patterns:
                success = self._apply_pattern_transfer(
                    transfer_item['pattern'], 
                    source_domain, 
                    target_domain
                )
                if success:
                    successful_transfers += 1
                    transfer_result['transferred_patterns'].append(transfer_item['pattern'])
            
            # Calculate success rate
            if transferable_patterns:
                transfer_result['transfer_success_rate'] = successful_transfers / len(transferable_patterns)
            
            # Identify applicable strategies
            transfer_result['applicable_strategies'] = self._identify_applicable_strategies(
                source_domain, target_domain
            )
            
        except Exception as e:
            self.logger.error(f"Error in knowledge transfer: {e}")
            transfer_result['error'] = str(e)
        
        return transfer_result
    
    def update_from_optimization(self, 
                               task_context: Dict[str, Any],
                               optimization_result: Dict[str, Any]) -> None:
        """Update learning models from optimization results."""
        try:
            # Extract optimization insights
            current_strategy = optimization_result.get('current_strategy')
            recommended_strategy = optimization_result.get('recommended_strategy')
            expected_improvement = optimization_result.get('expected_improvement', 0.0)
            
            # Create learning experience
            experience_data = {
                'type': 'optimization',
                'task_context': task_context,
                'current_strategy': current_strategy,
                'recommended_strategy': recommended_strategy,
                'expected_improvement': expected_improvement,
                'outcome': {'improvement_prediction': expected_improvement}
            }
            
            self.learn_from_experience(experience_data)
            
        except Exception as e:
            self.logger.error(f"Error updating from optimization: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about meta-learning progress."""
        stats = {
            'total_experiences': len(self.learning_experiences),
            'learning_types': {},
            'adaptation_count': len(self.adaptation_history),
            'pattern_count': len(self.learned_patterns),
            'average_learning_value': 0.0,
            'recent_learning_trend': 'stable'
        }
        
        try:
            # Count by learning type
            for experience in self.learning_experiences:
                learning_type = experience.learning_type.value
                if learning_type not in stats['learning_types']:
                    stats['learning_types'][learning_type] = 0
                stats['learning_types'][learning_type] += 1
            
            # Calculate average learning value
            if self.learning_experiences:
                total_value = sum(exp.learning_value for exp in self.learning_experiences)
                stats['average_learning_value'] = total_value / len(self.learning_experiences)
            
            # Assess recent learning trend
            if len(self.learning_experiences) >= 10:
                recent_values = [exp.learning_value for exp in self.learning_experiences[-10:]]
                older_values = [exp.learning_value for exp in self.learning_experiences[-20:-10]] if len(self.learning_experiences) >= 20 else []
                
                if older_values:
                    recent_avg = sum(recent_values) / len(recent_values)
                    older_avg = sum(older_values) / len(older_values)
                    
                    if recent_avg > older_avg * 1.1:
                        stats['recent_learning_trend'] = 'improving'
                    elif recent_avg < older_avg * 0.9:
                        stats['recent_learning_trend'] = 'declining'
            
        except Exception as e:
            self.logger.error(f"Error generating learning statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    # Private helper methods
    def _determine_learning_type(self, experience_data: Dict[str, Any]) -> LearningType:
        """Determine the type of learning from experience data."""
        data_type = experience_data.get('type', 'general')
        
        type_mapping = {
            'strategy': LearningType.STRATEGY_LEARNING,
            'performance': LearningType.PERFORMANCE_LEARNING,
            'adaptation': LearningType.ADAPTATION_LEARNING,
            'context': LearningType.CONTEXT_LEARNING,
            'pattern': LearningType.PATTERN_LEARNING,
            'optimization': LearningType.STRATEGY_LEARNING
        }
        
        return type_mapping.get(data_type, LearningType.PERFORMANCE_LEARNING)
    
    def _calculate_learning_value(self, experience_data: Dict[str, Any]) -> float:
        """Calculate the learning value of an experience."""
        base_value = 0.5
        
        # Adjust based on outcome quality
        outcome = experience_data.get('outcome', {})
        if 'improvement' in outcome:
            improvement = outcome['improvement']
            if isinstance(improvement, (int, float)):
                base_value += min(improvement, 0.5)
        
        # Adjust based on novelty
        if 'novelty' in experience_data:
            novelty = experience_data['novelty']
            if isinstance(novelty, (int, float)):
                base_value += novelty * 0.3
        
        # Adjust based on success
        if 'success' in outcome and outcome['success']:
            base_value += 0.2
        
        return max(0.0, min(1.0, base_value))
    
    def _update_learning_models(self, experience: LearningExperience) -> None:
        """Update learning models based on experience."""
        if experience.learning_type == LearningType.STRATEGY_LEARNING:
            strategy = experience.experience_data.get('strategy')
            if strategy:
                current_effectiveness = self.strategy_effectiveness_model.get(strategy, 0.5)
                new_effectiveness = (
                    current_effectiveness * (1 - self.learning_rate) +
                    experience.learning_value * self.learning_rate
                )
                self.strategy_effectiveness_model[strategy] = new_effectiveness
        
        elif experience.learning_type == LearningType.CONTEXT_LEARNING:
            context = experience.context.get('context_type', 'general')
            if context not in self.context_performance_model:
                self.context_performance_model[context] = {}
            
            strategy = experience.experience_data.get('strategy', 'default')
            current_performance = self.context_performance_model[context].get(strategy, 0.5)
            new_performance = (
                current_performance * (1 - self.learning_rate) +
                experience.learning_value * self.learning_rate
            )
            self.context_performance_model[context][strategy] = new_performance
    
    def _extract_patterns_from_experience(self, experience: LearningExperience) -> None:
        """Extract patterns from a learning experience."""
        pattern_key = f"{experience.learning_type.value}_{experience.context.get('context_type', 'general')}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                'count': 0,
                'average_value': 0.0,
                'common_elements': []
            }
        
        pattern = self.learned_patterns[pattern_key]
        pattern['count'] += 1
        
        # Update average value
        current_avg = pattern['average_value']
        new_avg = (current_avg * (pattern['count'] - 1) + experience.learning_value) / pattern['count']
        pattern['average_value'] = new_avg
        
        # Extract common elements (simplified)
        if 'strategy' in experience.experience_data:
            strategy = experience.experience_data['strategy']
            if strategy not in pattern['common_elements']:
                pattern['common_elements'].append(strategy)
    
    def _identify_adaptations(self, 
                            performance_issues: List[str],
                            performance_improvements: List[str],
                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential adaptations based on performance feedback."""
        adaptations = []
        
        # Address performance issues
        for issue in performance_issues:
            if issue == 'slow_processing':
                adaptations.append({
                    'type': 'strategy_change',
                    'from': 'systematic',
                    'to': 'intuitive',
                    'reason': 'improve_speed'
                })
            elif issue == 'low_accuracy':
                adaptations.append({
                    'type': 'strategy_change',
                    'from': 'intuitive',
                    'to': 'analytical',
                    'reason': 'improve_accuracy'
                })
        
        # Leverage improvements
        for improvement in performance_improvements:
            if improvement == 'high_creativity':
                adaptations.append({
                    'type': 'strategy_preference',
                    'strategy': 'creative',
                    'context': context.get('context_type', 'general'),
                    'reason': 'leverage_strength'
                })
        
        return adaptations
    
    def _apply_adaptation(self, adaptation: Dict[str, Any]) -> bool:
        """Apply a specific adaptation."""
        try:
            adaptation_type = adaptation.get('type')
            
            if adaptation_type == 'strategy_change':
                # Update strategy effectiveness model
                from_strategy = adaptation.get('from')
                to_strategy = adaptation.get('to')
                
                if from_strategy in self.strategy_effectiveness_model:
                    # Decrease effectiveness of from_strategy
                    self.strategy_effectiveness_model[from_strategy] *= 0.95
                
                if to_strategy not in self.strategy_effectiveness_model:
                    self.strategy_effectiveness_model[to_strategy] = 0.6
                else:
                    # Increase effectiveness of to_strategy
                    self.strategy_effectiveness_model[to_strategy] = min(
                        1.0, self.strategy_effectiveness_model[to_strategy] * 1.05
                    )
                
                return True
            
            elif adaptation_type == 'strategy_preference':
                strategy = adaptation.get('strategy')
                context = adaptation.get('context', 'general')
                
                if context not in self.context_performance_model:
                    self.context_performance_model[context] = {}
                
                if strategy not in self.context_performance_model[context]:
                    self.context_performance_model[context][strategy] = 0.6
                else:
                    # Increase preference for strategy in this context
                    self.context_performance_model[context][strategy] = min(
                        1.0, self.context_performance_model[context][strategy] * 1.1
                    )
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error applying adaptation: {e}")
        
        return False
    
    def _get_domain_patterns(self, domain: str) -> List[Dict[str, Any]]:
        """Get patterns associated with a specific domain."""
        domain_patterns = []
        
        for pattern_key, pattern_data in self.learned_patterns.items():
            if domain in pattern_key or 'general' in pattern_key:
                domain_patterns.append({
                    'key': pattern_key,
                    'data': pattern_data
                })
        
        return domain_patterns
    
    def _assess_transferability(self, pattern: Dict[str, Any], target_domain: str) -> float:
        """Assess how transferable a pattern is to a target domain."""
        base_transferability = 0.5
        
        # High-value patterns are more transferable
        if pattern['data']['average_value'] > 0.7:
            base_transferability += 0.2
        
        # Frequently observed patterns are more transferable
        if pattern['data']['count'] > 5:
            base_transferability += 0.2
        
        # Domain-specific adjustments would go here
        # (simplified for this implementation)
        
        return min(1.0, base_transferability)
    
    def _apply_pattern_transfer(self, 
                              pattern: Dict[str, Any],
                              source_domain: str,
                              target_domain: str) -> bool:
        """Apply pattern transfer from source to target domain."""
        try:
            # Create new pattern key for target domain
            source_key = pattern['key']
            target_key = source_key.replace(source_domain, target_domain)
            
            # Initialize pattern in target domain if not exists
            if target_key not in self.learned_patterns:
                self.learned_patterns[target_key] = {
                    'count': 1,  # Start with low count
                    'average_value': pattern['data']['average_value'] * 0.8,  # Reduce confidence
                    'common_elements': pattern['data']['common_elements'].copy()
                }
                return True
            
        except Exception as e:
            self.logger.error(f"Error applying pattern transfer: {e}")
        
        return False
    
    def _identify_applicable_strategies(self, 
                                      source_domain: str,
                                      target_domain: str) -> List[str]:
        """Identify strategies applicable to target domain based on source domain."""
        applicable_strategies = []
        
        # Get high-performing strategies from source domain
        source_context_model = self.context_performance_model.get(source_domain, {})
        
        for strategy, performance in source_context_model.items():
            if performance > 0.7:  # High-performing strategies
                applicable_strategies.append(strategy)
        
        return applicable_strategies