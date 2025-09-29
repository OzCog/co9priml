"""
Recursive Meta-Cognitive Processor
==================================

This module implements recursive meta-cognitive processing capabilities
that enable nested levels of meta-cognitive analysis, allowing the system
to think about thinking about thinking, with controlled depth and
termination conditions.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging
from ..interfaces.meta_cognitive_interface import (
    RecursiveProcessingInterface, MetaCognitiveCapability
)
from ..core.meta_cognitive_core import CognitiveProcess, MetaCognitiveLevel


class RecursionTerminationReason(Enum):
    """Reasons for terminating recursion."""
    MAX_DEPTH_REACHED = "max_depth_reached"
    QUALITY_THRESHOLD_MET = "quality_threshold_met"
    DIMINISHING_RETURNS = "diminishing_returns"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONVERGENCE_ACHIEVED = "convergence_achieved"
    TIME_LIMIT_EXCEEDED = "time_limit_exceeded"


@dataclass
class RecursionLevel:
    """Represents a level in recursive processing."""
    level: int
    analysis_quality: float
    resource_usage: Dict[str, float]
    insights_generated: int
    processing_time: float
    parent_level: Optional['RecursionLevel']
    children_levels: List['RecursionLevel']


@dataclass
class RecursiveAnalysisResult:
    """Result of recursive analysis."""
    total_depth: int
    termination_reason: RecursionTerminationReason
    analysis_hierarchy: List[RecursionLevel]
    final_insights: List[str]
    quality_progression: List[float]
    resource_usage_total: Dict[str, float]
    convergence_score: float
    execution_time: float


class RecursiveMetaCognitiveProcessor(RecursiveProcessingInterface):
    """
    Implementation of recursive meta-cognitive processing.
    
    This system provides:
    - Multi-level recursive analysis with depth control
    - Intelligent termination conditions
    - Resource management across recursion levels
    - Quality assessment and convergence detection
    - Hierarchical insight synthesis
    - Adaptive depth adjustment based on context
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the recursive meta-cognitive processor."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_recursion_depth = self.config.get('max_recursion_depth', 5)
        self.quality_threshold = self.config.get('quality_threshold', 0.9)
        self.diminishing_returns_threshold = self.config.get('diminishing_returns_threshold', 0.05)
        self.max_processing_time = self.config.get('max_processing_time', 300.0)  # 5 minutes
        self.convergence_threshold = self.config.get('convergence_threshold', 0.95)
        
        # Resource limits per recursion level
        self.resource_limits = self.config.get('resource_limits', {
            'memory': 0.8,
            'processing': 0.7,
            'attention': 0.6
        })
        
        # State tracking
        self.active_recursions: Dict[str, RecursiveAnalysisResult] = {}
        self.recursion_history: List[RecursiveAnalysisResult] = []
        
        # Analysis engines
        self.quality_assessor = QualityAssessor(config)
        self.convergence_detector = ConvergenceDetector(config)
        self.resource_manager = RecursiveResourceManager(config)
        self.insight_synthesizer = InsightSynthesizer(config)
        
        self.logger.info("Recursive meta-cognitive processor initialized")
    
    def initialize(self) -> bool:
        """Initialize the recursive processor component."""
        try:
            self.quality_assessor.initialize()
            self.convergence_detector.initialize()
            self.resource_manager.initialize()
            self.insight_synthesizer.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize recursive processor: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the recursive processor component."""
        return True
    
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of recursive processing capabilities."""
        return [
            MetaCognitiveCapability(
                name="recursive_analysis",
                description="Multi-level recursive meta-cognitive analysis",
                complexity_level=5,
                requires_recursion=True,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="depth_control",
                description="Intelligent depth control and termination",
                complexity_level=4,
                requires_recursion=True,
                resource_intensive=False
            ),
            MetaCognitiveCapability(
                name="convergence_detection",
                description="Detection of analysis convergence",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="hierarchical_synthesis",
                description="Synthesis of insights across recursion levels",
                complexity_level=5,
                requires_recursion=True,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="adaptive_termination",
                description="Adaptive termination based on multiple criteria",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=False
            )
        ]
    
    def process_meta_cognitive_request(self, 
                                     request_type: str,
                                     request_data: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meta-cognitive request."""
        if request_type == "recursive_analyze":
            depth = context.get('depth', 2)
            return self.recursive_analyze(request_data, {}, depth)
        elif request_type == "check_termination":
            current_depth = context.get('current_depth', 1)
            quality = context.get('analysis_quality', 0.5)
            should_terminate = self.check_recursion_termination(current_depth, quality)
            return {'should_terminate': should_terminate}
        elif request_type == "manage_resources":
            depth = context.get('depth', 1)
            return self.manage_recursive_resources(depth)
        else:
            return {'error': f'Unknown request type: {request_type}'}
    
    def recursive_analyze(self, 
                         data: Any,
                         performance_history: List[Dict[str, Any]] = None,
                         depth: int = 2) -> Dict[str, Any]:
        """Perform recursive analysis at specified depth."""
        analysis_id = f"recursive_analysis_{time.time()}"
        start_time = time.time()
        
        try:
            # Initialize recursive analysis
            result = RecursiveAnalysisResult(
                total_depth=0,
                termination_reason=RecursionTerminationReason.MAX_DEPTH_REACHED,
                analysis_hierarchy=[],
                final_insights=[],
                quality_progression=[],
                resource_usage_total={},
                convergence_score=0.0,
                execution_time=0.0
            )
            
            # Perform recursive analysis
            root_level = self._perform_recursive_level(
                data, performance_history or [], 0, depth, None, start_time
            )
            
            # Build result
            result.analysis_hierarchy = [root_level]
            result.total_depth = self._calculate_total_depth(root_level)
            result.final_insights = self._extract_final_insights(root_level)
            result.quality_progression = self._build_quality_progression(root_level)
            result.resource_usage_total = self._calculate_total_resource_usage(root_level)
            result.convergence_score = self.convergence_detector.calculate_convergence(root_level)
            result.execution_time = time.time() - start_time
            
            # Store result
            self.active_recursions[analysis_id] = result
            self.recursion_history.append(result)
            
            # Keep history bounded
            if len(self.recursion_history) > 100:
                self.recursion_history = self.recursion_history[-80:]
            
            return self._result_to_dict(result)
            
        except Exception as e:
            self.logger.error(f"Error in recursive analysis: {e}")
            return {'error': str(e), 'analysis_id': analysis_id}
    
    def check_recursion_termination(self, 
                                  current_depth: int,
                                  analysis_quality: float) -> bool:
        """Check if recursion should terminate."""
        try:
            # Check maximum depth
            if current_depth >= self.max_recursion_depth:
                return True
            
            # Check quality threshold
            if analysis_quality >= self.quality_threshold:
                return True
            
            # Check resource constraints
            if not self._has_sufficient_resources(current_depth + 1):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking recursion termination: {e}")
            return True  # Err on the side of caution
    
    def manage_recursive_resources(self, depth: int) -> Dict[str, Any]:
        """Manage resources for recursive processing."""
        try:
            resource_allocation = self.resource_manager.allocate_resources(depth)
            
            return {
                'depth': depth,
                'allocated_resources': resource_allocation,
                'resource_efficiency': self._calculate_resource_efficiency(resource_allocation),
                'projected_capacity': self._project_capacity(depth, resource_allocation)
            }
            
        except Exception as e:
            self.logger.error(f"Error managing recursive resources: {e}")
            return {'error': str(e), 'depth': depth}
    
    def get_recursion_statistics(self) -> Dict[str, Any]:
        """Get statistics about recursive processing."""
        stats = {
            'active_recursions': len(self.active_recursions),
            'total_recursions_completed': len(self.recursion_history),
            'average_depth': 0.0,
            'average_execution_time': 0.0,
            'common_termination_reasons': {},
            'quality_distribution': {}
        }
        
        try:
            if self.recursion_history:
                # Calculate averages
                stats['average_depth'] = sum(r.total_depth for r in self.recursion_history) / len(self.recursion_history)
                stats['average_execution_time'] = sum(r.execution_time for r in self.recursion_history) / len(self.recursion_history)
                
                # Termination reason distribution
                reason_counts = {}
                for result in self.recursion_history:
                    reason = result.termination_reason.value
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                stats['common_termination_reasons'] = reason_counts
                
                # Quality distribution
                quality_buckets = {'low': 0, 'medium': 0, 'high': 0}
                for result in self.recursion_history:
                    if result.quality_progression:
                        final_quality = result.quality_progression[-1]
                        if final_quality < 0.4:
                            quality_buckets['low'] += 1
                        elif final_quality < 0.7:
                            quality_buckets['medium'] += 1
                        else:
                            quality_buckets['high'] += 1
                stats['quality_distribution'] = quality_buckets
                
        except Exception as e:
            self.logger.error(f"Error calculating recursion statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def adaptive_depth_recommendation(self, 
                                    context: Dict[str, Any],
                                    resource_constraints: Dict[str, float]) -> int:
        """Recommend optimal recursion depth based on context and resources."""
        try:
            base_depth = 2  # Default depth
            
            # Adjust based on context complexity
            complexity = context.get('complexity', 0.5)
            if complexity > 0.8:
                base_depth += 2
            elif complexity > 0.6:
                base_depth += 1
            elif complexity < 0.3:
                base_depth = max(1, base_depth - 1)
            
            # Adjust based on resource constraints
            resource_availability = sum(resource_constraints.values()) / len(resource_constraints)
            if resource_availability < 0.3:
                base_depth = max(1, base_depth - 2)
            elif resource_availability < 0.6:
                base_depth = max(1, base_depth - 1)
            elif resource_availability > 0.9:
                base_depth += 1
            
            # Adjust based on historical performance
            if self.recursion_history:
                recent_results = self.recursion_history[-10:]
                avg_convergence = sum(r.convergence_score for r in recent_results) / len(recent_results)
                if avg_convergence > 0.8:
                    base_depth += 1  # Good convergence, can go deeper
                elif avg_convergence < 0.4:
                    base_depth = max(1, base_depth - 1)  # Poor convergence, stay shallow
            
            # Ensure within bounds
            recommended_depth = max(1, min(base_depth, self.max_recursion_depth))
            
            return recommended_depth
            
        except Exception as e:
            self.logger.error(f"Error in adaptive depth recommendation: {e}")
            return 2  # Safe default
    
    # Private helper methods
    def _perform_recursive_level(self, 
                               data: Any,
                               performance_history: List[Dict[str, Any]],
                               current_depth: int,
                               max_depth: int,
                               parent_level: Optional[RecursionLevel],
                               start_time: float) -> RecursionLevel:
        """Perform analysis at a single recursion level."""
        level_start_time = time.time()
        
        # Initialize level
        level = RecursionLevel(
            level=current_depth,
            analysis_quality=0.0,
            resource_usage={},
            insights_generated=0,
            processing_time=0.0,
            parent_level=parent_level,
            children_levels=[]
        )
        
        try:
            # Allocate resources for this level
            allocated_resources = self.resource_manager.allocate_resources(current_depth)
            level.resource_usage = allocated_resources
            
            # Perform analysis at this level
            analysis_result = self._analyze_at_level(data, performance_history, current_depth)
            level.analysis_quality = analysis_result['quality']
            level.insights_generated = len(analysis_result.get('insights', []))
            
            # Check termination conditions
            should_terminate = (
                self.check_recursion_termination(current_depth, level.analysis_quality) or
                current_depth >= max_depth or
                time.time() - start_time > self.max_processing_time
            )
            
            if not should_terminate:
                # Check for diminishing returns
                if parent_level and level.analysis_quality - parent_level.analysis_quality < self.diminishing_returns_threshold:
                    should_terminate = True
            
            # Recurse if not terminating
            if not should_terminate and current_depth < max_depth:
                # Prepare data for next level (meta-analysis)
                meta_data = {
                    'previous_analysis': analysis_result,
                    'current_level': current_depth,
                    'parent_insights': analysis_result.get('insights', [])
                }
                
                # Recurse
                child_level = self._perform_recursive_level(
                    meta_data, performance_history, current_depth + 1, max_depth, level, start_time
                )
                level.children_levels.append(child_level)
            
            level.processing_time = time.time() - level_start_time
            
        except Exception as e:
            self.logger.error(f"Error at recursion level {current_depth}: {e}")
            level.processing_time = time.time() - level_start_time
        
        return level
    
    def _analyze_at_level(self, 
                         data: Any,
                         performance_history: List[Dict[str, Any]],
                         level: int) -> Dict[str, Any]:
        """Perform analysis at a specific level."""
        analysis = {
            'level': level,
            'quality': 0.5,
            'insights': [],
            'patterns': [],
            'meta_observations': []
        }
        
        try:
            if level == 0:
                # Base level analysis
                analysis['insights'] = self._base_level_analysis(data, performance_history)
                analysis['quality'] = 0.6
                
            elif level == 1:
                # First meta-level: analyze the analysis
                if isinstance(data, dict) and 'previous_analysis' in data:
                    previous = data['previous_analysis']
                    analysis['insights'] = self._meta_level_analysis(previous)
                    analysis['quality'] = 0.7
                    analysis['meta_observations'] = [
                        f"Analyzing analysis from level {data.get('current_level', 0)}"
                    ]
                
            else:
                # Higher meta-levels: analyze patterns in meta-analysis
                if isinstance(data, dict) and 'previous_analysis' in data:
                    previous = data['previous_analysis']
                    analysis['insights'] = self._higher_meta_analysis(previous, level)
                    analysis['quality'] = min(0.8, 0.5 + level * 0.1)
                    analysis['meta_observations'] = [
                        f"Higher-order meta-analysis at level {level}",
                        f"Analyzing patterns in level {level-1} analysis"
                    ]
            
            # Apply quality assessment
            assessed_quality = self.quality_assessor.assess_analysis_quality(analysis)
            analysis['quality'] = assessed_quality
            
        except Exception as e:
            self.logger.error(f"Error in level {level} analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _base_level_analysis(self, data: Any, performance_history: List[Dict[str, Any]]) -> List[str]:
        """Perform base level analysis."""
        insights = []
        
        # Analyze the data itself
        if isinstance(data, list):
            insights.append(f"Data contains {len(data)} items for analysis")
            if len(data) > 10:
                insights.append("Large dataset detected - consider pattern analysis")
        elif isinstance(data, dict):
            insights.append(f"Data structure has {len(data)} key components")
            if 'performance_metrics' in data:
                insights.append("Performance metrics available for analysis")
        
        # Analyze performance history if available
        if performance_history:
            insights.append(f"Performance history contains {len(performance_history)} entries")
            if len(performance_history) > 5:
                insights.append("Sufficient history for trend analysis")
        
        return insights
    
    def _meta_level_analysis(self, previous_analysis: Dict[str, Any]) -> List[str]:
        """Perform first-level meta-analysis."""
        meta_insights = []
        
        # Analyze the quality of previous analysis
        quality = previous_analysis.get('quality', 0.5)
        if quality > 0.7:
            meta_insights.append("Previous analysis achieved high quality")
        elif quality < 0.4:
            meta_insights.append("Previous analysis quality was suboptimal")
        
        # Analyze insight generation
        insights_count = len(previous_analysis.get('insights', []))
        if insights_count > 5:
            meta_insights.append("Previous analysis was highly generative")
        elif insights_count < 2:
            meta_insights.append("Previous analysis generated few insights")
        
        # Pattern detection in analysis approach
        if 'patterns' in previous_analysis:
            meta_insights.append("Previous analysis included pattern recognition")
        
        return meta_insights
    
    def _higher_meta_analysis(self, previous_analysis: Dict[str, Any], level: int) -> List[str]:
        """Perform higher-order meta-analysis."""
        higher_insights = []
        
        # Analyze the meta-analytical process itself
        higher_insights.append(f"Performing level-{level} meta-cognitive analysis")
        
        # Look for emergent properties in the analysis process
        if 'meta_observations' in previous_analysis:
            meta_obs_count = len(previous_analysis['meta_observations'])
            if meta_obs_count > 0:
                higher_insights.append(f"Meta-observations emerging at level {level-1}")
        
        # Analyze complexity and depth
        complexity_indicators = sum(1 for key in previous_analysis.keys() if key.startswith('meta_'))
        if complexity_indicators > 2:
            higher_insights.append("High meta-cognitive complexity detected")
        
        # Recursive pattern analysis
        if level > 2:
            higher_insights.append("Deep recursive patterns may be emerging")
        
        return higher_insights
    
    def _has_sufficient_resources(self, projected_depth: int) -> bool:
        """Check if sufficient resources are available for projected depth."""
        try:
            projected_usage = self.resource_manager.project_resource_usage(projected_depth)
            
            for resource, limit in self.resource_limits.items():
                if projected_usage.get(resource, 0) > limit:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking resource sufficiency: {e}")
            return False
    
    def _calculate_total_depth(self, root_level: RecursionLevel) -> int:
        """Calculate total depth of recursion tree."""
        max_depth = root_level.level
        
        def traverse_depth(level: RecursionLevel) -> int:
            if not level.children_levels:
                return level.level
            return max(traverse_depth(child) for child in level.children_levels)
        
        return traverse_depth(root_level)
    
    def _extract_final_insights(self, root_level: RecursionLevel) -> List[str]:
        """Extract final insights from recursion tree."""
        insights = []
        
        def collect_insights(level: RecursionLevel):
            insights.append(f"Level {level.level}: Generated {level.insights_generated} insights")
            for child in level.children_levels:
                collect_insights(child)
        
        collect_insights(root_level)
        
        # Add synthesis insights
        if root_level.children_levels:
            total_insights = sum(level.insights_generated for level in self._flatten_hierarchy(root_level))
            insights.append(f"Total insights across all levels: {total_insights}")
        
        return insights
    
    def _build_quality_progression(self, root_level: RecursionLevel) -> List[float]:
        """Build quality progression across levels."""
        progression = []
        
        def collect_quality(level: RecursionLevel):
            progression.append(level.analysis_quality)
            for child in level.children_levels:
                collect_quality(child)
        
        collect_quality(root_level)
        return progression
    
    def _calculate_total_resource_usage(self, root_level: RecursionLevel) -> Dict[str, float]:
        """Calculate total resource usage across all levels."""
        total_usage = {}
        
        def accumulate_usage(level: RecursionLevel):
            for resource, usage in level.resource_usage.items():
                if resource not in total_usage:
                    total_usage[resource] = 0
                total_usage[resource] += usage
            
            for child in level.children_levels:
                accumulate_usage(child)
        
        accumulate_usage(root_level)
        return total_usage
    
    def _flatten_hierarchy(self, root_level: RecursionLevel) -> List[RecursionLevel]:
        """Flatten recursion hierarchy into a list."""
        levels = [root_level]
        
        def flatten_recursive(level: RecursionLevel):
            for child in level.children_levels:
                levels.append(child)
                flatten_recursive(child)
        
        flatten_recursive(root_level)
        return levels
    
    def _result_to_dict(self, result: RecursiveAnalysisResult) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'total_depth': result.total_depth,
            'termination_reason': result.termination_reason.value,
            'final_insights': result.final_insights,
            'quality_progression': result.quality_progression,
            'resource_usage_total': result.resource_usage_total,
            'convergence_score': result.convergence_score,
            'execution_time': result.execution_time,
            'levels_analyzed': len(result.analysis_hierarchy)
        }
    
    def _calculate_resource_efficiency(self, resource_allocation: Dict[str, float]) -> float:
        """Calculate resource efficiency score."""
        if not resource_allocation:
            return 0.0
        
        total_allocated = sum(resource_allocation.values())
        total_available = sum(self.resource_limits.values())
        
        return min(1.0, total_allocated / total_available) if total_available > 0 else 0.0
    
    def _project_capacity(self, depth: int, resource_allocation: Dict[str, float]) -> Dict[str, Any]:
        """Project remaining capacity for further recursion."""
        remaining_capacity = {}
        
        for resource, limit in self.resource_limits.items():
            used = resource_allocation.get(resource, 0)
            remaining = limit - used
            remaining_capacity[resource] = max(0, remaining)
        
        # Estimate how many more levels are possible
        min_remaining = min(remaining_capacity.values())
        estimated_levels_remaining = int(min_remaining / 0.1) if min_remaining > 0 else 0
        
        return {
            'remaining_capacity': remaining_capacity,
            'estimated_levels_remaining': estimated_levels_remaining,
            'can_continue': estimated_levels_remaining > 0
        }


# Helper classes for specialized recursive processing functions
class QualityAssessor:
    """Specialized assessor for analysis quality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def initialize(self) -> bool:
        return True
    
    def assess_analysis_quality(self, analysis: Dict[str, Any]) -> float:
        """Assess the quality of an analysis."""
        base_quality = analysis.get('quality', 0.5)
        
        # Adjust based on insight count
        insights_count = len(analysis.get('insights', []))
        if insights_count > 3:
            base_quality += 0.1
        elif insights_count == 0:
            base_quality -= 0.2
        
        # Adjust based on presence of meta-observations
        if analysis.get('meta_observations'):
            base_quality += 0.1
        
        # Adjust based on error presence
        if 'error' in analysis:
            base_quality -= 0.3
        
        return max(0.0, min(1.0, base_quality))


class ConvergenceDetector:
    """Specialized detector for analysis convergence."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.convergence_threshold = config.get('convergence_threshold', 0.95)
    
    def initialize(self) -> bool:
        return True
    
    def calculate_convergence(self, root_level: RecursionLevel) -> float:
        """Calculate convergence score for recursion tree."""
        # Simple convergence metric based on quality stabilization
        levels = self._flatten_levels(root_level)
        
        if len(levels) < 2:
            return 0.5  # Not enough data
        
        # Check quality stability
        qualities = [level.analysis_quality for level in levels]
        if len(qualities) >= 2:
            quality_variance = self._calculate_variance(qualities)
            # Lower variance indicates higher convergence
            convergence = max(0.0, 1.0 - quality_variance * 2)
            return convergence
        
        return 0.5
    
    def _flatten_levels(self, root_level: RecursionLevel) -> List[RecursionLevel]:
        """Flatten recursion levels."""
        levels = [root_level]
        
        def flatten_recursive(level: RecursionLevel):
            for child in level.children_levels:
                levels.append(child)
                flatten_recursive(child)
        
        flatten_recursive(root_level)
        return levels
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance


class RecursiveResourceManager:
    """Specialized manager for recursive resource allocation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_resource_cost = config.get('base_resource_cost', 0.1)
        self.depth_multiplier = config.get('depth_multiplier', 1.2)
    
    def initialize(self) -> bool:
        return True
    
    def allocate_resources(self, depth: int) -> Dict[str, float]:
        """Allocate resources for a specific depth."""
        base_cost = self.base_resource_cost
        depth_factor = self.depth_multiplier ** depth
        
        allocation = {
            'memory': base_cost * depth_factor * 1.0,
            'processing': base_cost * depth_factor * 1.1,
            'attention': base_cost * depth_factor * 0.9
        }
        
        return allocation
    
    def project_resource_usage(self, max_depth: int) -> Dict[str, float]:
        """Project total resource usage for maximum depth."""
        total_usage = {}
        
        for depth in range(max_depth + 1):
            level_allocation = self.allocate_resources(depth)
            for resource, usage in level_allocation.items():
                if resource not in total_usage:
                    total_usage[resource] = 0
                total_usage[resource] += usage
        
        return total_usage


class InsightSynthesizer:
    """Specialized synthesizer for insights across recursion levels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def initialize(self) -> bool:
        return True
    
    def synthesize_insights(self, levels: List[RecursionLevel]) -> List[str]:
        """Synthesize insights across multiple recursion levels."""
        synthesized = []
        
        # Combine insights from all levels
        all_insights = []
        for level in levels:
            all_insights.extend([f"L{level.level}: {level.insights_generated} insights"])
        
        # Generate synthesis
        if len(levels) > 1:
            synthesized.append(f"Cross-level analysis across {len(levels)} recursion levels")
            
            total_insights = sum(level.insights_generated for level in levels)
            synthesized.append(f"Total insights generated: {total_insights}")
            
            quality_trend = self._analyze_quality_trend(levels)
            synthesized.append(f"Quality trend: {quality_trend}")
        
        return synthesized
    
    def _analyze_quality_trend(self, levels: List[RecursionLevel]) -> str:
        """Analyze quality trend across levels."""
        if len(levels) < 2:
            return "insufficient_data"
        
        qualities = [level.analysis_quality for level in levels]
        if qualities[-1] > qualities[0]:
            return "improving"
        elif qualities[-1] < qualities[0]:
            return "declining"
        else:
            return "stable"