"""
Meta-Knowledge System
====================

This module implements meta-cognitive knowledge representation systems
that capture and organize meta-cognitive insights, patterns, and learning
for reuse across different contexts and situations.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import logging
from ..interfaces.meta_cognitive_interface import (
    MetaKnowledgeInterface, MetaCognitiveCapability
)
from ..core.meta_cognitive_core import CognitiveProcess


class KnowledgeType(Enum):
    """Types of meta-cognitive knowledge."""
    STRATEGY_KNOWLEDGE = "strategy_knowledge"
    PROCESS_PATTERNS = "process_patterns"
    PERFORMANCE_INSIGHTS = "performance_insights"
    CONTEXT_ASSOCIATIONS = "context_associations"
    LEARNING_EXPERIENCES = "learning_experiences"


@dataclass
class MetaKnowledgeItem:
    """Represents a piece of meta-cognitive knowledge."""
    knowledge_id: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    confidence: float
    usage_count: int
    last_accessed: float
    created_time: float
    context_tags: List[str]


class MetaKnowledgeSystem(MetaKnowledgeInterface):
    """
    Implementation of meta-cognitive knowledge representation and storage.
    
    This system provides:
    - Structured storage of meta-cognitive insights
    - Context-aware knowledge retrieval
    - Knowledge pattern recognition
    - Experience-based learning capture
    - Cross-context knowledge transfer
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the meta-knowledge system."""
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Knowledge storage
        self.knowledge_base: Dict[str, MetaKnowledgeItem] = {}
        self.knowledge_index: Dict[KnowledgeType, List[str]] = {}
        self.context_index: Dict[str, List[str]] = {}
        
        # Initialize indices
        for knowledge_type in KnowledgeType:
            self.knowledge_index[knowledge_type] = []
        
        self.logger.info("Meta-knowledge system initialized")
    
    def initialize(self) -> bool:
        """Initialize the meta-knowledge component."""
        return True
    
    def shutdown(self) -> bool:
        """Shutdown the meta-knowledge component."""
        return True
    
    def get_capabilities(self) -> List[MetaCognitiveCapability]:
        """Return list of meta-knowledge capabilities."""
        return [
            MetaCognitiveCapability(
                name="knowledge_storage",
                description="Storage of meta-cognitive knowledge",
                complexity_level=2,
                requires_recursion=False,
                resource_intensive=False
            ),
            MetaCognitiveCapability(
                name="context_retrieval",
                description="Context-aware knowledge retrieval",
                complexity_level=3,
                requires_recursion=False,
                resource_intensive=True
            ),
            MetaCognitiveCapability(
                name="pattern_recognition",
                description="Recognition of knowledge patterns",
                complexity_level=4,
                requires_recursion=False,
                resource_intensive=True
            )
        ]
    
    def process_meta_cognitive_request(self, 
                                     request_type: str,
                                     request_data: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a meta-cognitive request."""
        if request_type == "store_knowledge":
            knowledge_type = request_data.get('type', 'strategy_knowledge')
            knowledge_data = request_data.get('data', {})
            success = self.store_meta_knowledge(knowledge_type, knowledge_data)
            return {'success': success}
        elif request_type == "retrieve_knowledge":
            knowledge_type = request_data.get('type', 'strategy_knowledge')
            query = request_data.get('query', {})
            results = self.retrieve_meta_knowledge(knowledge_type, query)
            return {'results': results}
        else:
            return {'error': f'Unknown request type: {request_type}'}
    
    def store_meta_knowledge(self, 
                           knowledge_type: str,
                           knowledge_data: Any) -> bool:
        """Store meta-cognitive knowledge."""
        try:
            # Create knowledge item
            knowledge_id = f"knowledge_{len(self.knowledge_base)}_{time.time()}"
            
            if isinstance(knowledge_type, str):
                knowledge_type_enum = KnowledgeType(knowledge_type)
            else:
                knowledge_type_enum = knowledge_type
            
            knowledge_item = MetaKnowledgeItem(
                knowledge_id=knowledge_id,
                knowledge_type=knowledge_type_enum,
                content=knowledge_data if isinstance(knowledge_data, dict) else {'data': knowledge_data},
                confidence=0.8,  # Default confidence
                usage_count=0,
                last_accessed=time.time(),
                created_time=time.time(),
                context_tags=self._extract_context_tags(knowledge_data)
            )
            
            # Store in knowledge base
            self.knowledge_base[knowledge_id] = knowledge_item
            
            # Update indices
            self.knowledge_index[knowledge_type_enum].append(knowledge_id)
            
            for tag in knowledge_item.context_tags:
                if tag not in self.context_index:
                    self.context_index[tag] = []
                self.context_index[tag].append(knowledge_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing meta-knowledge: {e}")
            return False
    
    def retrieve_meta_knowledge(self, 
                              knowledge_type: str,
                              query: Dict[str, Any]) -> List[Any]:
        """Retrieve meta-cognitive knowledge."""
        try:
            if isinstance(knowledge_type, str):
                knowledge_type_enum = KnowledgeType(knowledge_type)
            else:
                knowledge_type_enum = knowledge_type
            
            # Get candidate knowledge items
            candidate_ids = self.knowledge_index.get(knowledge_type_enum, [])
            
            # Filter by query
            matching_items = []
            for knowledge_id in candidate_ids:
                item = self.knowledge_base[knowledge_id]
                if self._matches_query(item, query):
                    matching_items.append(item)
                    # Update usage stats
                    item.usage_count += 1
                    item.last_accessed = time.time()
            
            # Sort by relevance (usage count and confidence)
            matching_items.sort(
                key=lambda x: (x.usage_count * x.confidence),
                reverse=True
            )
            
            # Return content of matching items
            return [item.content for item in matching_items]
            
        except Exception as e:
            self.logger.error(f"Error retrieving meta-knowledge: {e}")
            return []
    
    def update_meta_knowledge(self, 
                            knowledge_id: str,
                            updates: Dict[str, Any]) -> bool:
        """Update existing meta-cognitive knowledge."""
        try:
            if knowledge_id not in self.knowledge_base:
                return False
            
            item = self.knowledge_base[knowledge_id]
            
            # Update content
            if 'content' in updates:
                item.content.update(updates['content'])
            
            # Update confidence
            if 'confidence' in updates:
                item.confidence = max(0.0, min(1.0, updates['confidence']))
            
            # Update context tags
            if 'context_tags' in updates:
                # Remove from old context indices
                for tag in item.context_tags:
                    if tag in self.context_index and knowledge_id in self.context_index[tag]:
                        self.context_index[tag].remove(knowledge_id)
                
                # Add to new context indices
                item.context_tags = updates['context_tags']
                for tag in item.context_tags:
                    if tag not in self.context_index:
                        self.context_index[tag] = []
                    if knowledge_id not in self.context_index[tag]:
                        self.context_index[tag].append(knowledge_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating meta-knowledge: {e}")
            return False
    
    def record_process_observation(self, 
                                 process: CognitiveProcess,
                                 monitoring_result: Dict[str, Any]) -> None:
        """Record observation about a cognitive process."""
        try:
            observation_data = {
                'process_id': process.process_id,
                'process_type': process.process_type,
                'performance_assessment': monitoring_result.get('performance_assessment', {}),
                'optimization_suggestions': monitoring_result.get('optimization_suggestions', []),
                'timestamp': monitoring_result.get('timestamp', time.time())
            }
            
            self.store_meta_knowledge(
                KnowledgeType.PROCESS_PATTERNS.value,
                observation_data
            )
            
        except Exception as e:
            self.logger.error(f"Error recording process observation: {e}")
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        stats = {
            'total_items': len(self.knowledge_base),
            'by_type': {},
            'most_used': [],
            'recent_additions': []
        }
        
        try:
            # Count by type
            for knowledge_type, item_ids in self.knowledge_index.items():
                stats['by_type'][knowledge_type.value] = len(item_ids)
            
            # Most used items
            sorted_items = sorted(
                self.knowledge_base.values(),
                key=lambda x: x.usage_count,
                reverse=True
            )
            stats['most_used'] = [
                {'id': item.knowledge_id, 'usage_count': item.usage_count}
                for item in sorted_items[:5]
            ]
            
            # Recent additions
            recent_items = sorted(
                self.knowledge_base.values(),
                key=lambda x: x.created_time,
                reverse=True
            )
            stats['recent_additions'] = [
                {'id': item.knowledge_id, 'type': item.knowledge_type.value}
                for item in recent_items[:5]
            ]
            
        except Exception as e:
            self.logger.error(f"Error generating knowledge statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    # Private helper methods
    def _extract_context_tags(self, knowledge_data: Any) -> List[str]:
        """Extract context tags from knowledge data."""
        tags = []
        
        if isinstance(knowledge_data, dict):
            # Extract tags from common fields
            if 'context' in knowledge_data:
                tags.append(str(knowledge_data['context']))
            if 'domain' in knowledge_data:
                tags.append(str(knowledge_data['domain']))
            if 'task_type' in knowledge_data:
                tags.append(str(knowledge_data['task_type']))
            
            # Extract from process information
            if 'process_type' in knowledge_data:
                tags.append(f"process_{knowledge_data['process_type']}")
        
        return tags
    
    def _matches_query(self, item: MetaKnowledgeItem, query: Dict[str, Any]) -> bool:
        """Check if an item matches a query."""
        # Simple matching logic
        if not query:
            return True
        
        # Check context tags
        if 'tags' in query:
            query_tags = query['tags'] if isinstance(query['tags'], list) else [query['tags']]
            if not any(tag in item.context_tags for tag in query_tags):
                return False
        
        # Check confidence threshold
        if 'min_confidence' in query:
            if item.confidence < query['min_confidence']:
                return False
        
        # Check content matching
        if 'content_match' in query:
            content_str = str(item.content).lower()
            query_str = str(query['content_match']).lower()
            if query_str not in content_str:
                return False
        
        return True