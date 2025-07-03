"""
CogPrime Memory Module - Integration with mem0

This module provides memory management capabilities for CogPrime by integrating
mem0's persistence, vector search, and knowledge graph features.

The Memory class extends CogPrime's working memory with:
- Persistent storage for cognitive states
- Vector-based semantic search
- Knowledge graph representation and queries
- LLM-based fact extraction for learning

Basic usage:
    from cogprime.memory import Memory
    
    # Create a memory instance
    memory = Memory()
    
    # Store a cognitive state
    memory.store_cognitive_state("state_1", cognitive_state)
    
    # Retrieve a cognitive state
    state = memory.retrieve_cognitive_state("state_1")
    
    # Store a fact with semantic embedding
    memory.add_fact("Cats are mammals", tags=["biology", "animals"])
    
    # Search for semantically similar facts
    results = memory.semantic_search("What animals are pets?", limit=5)
    
    # Extract facts from text using LLM
    facts = memory.extract_facts("The brain contains approximately 86 billion neurons.")
"""

import os
import json
import uuid
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Import mem0 components
try:
    from mem0.memory.main import Memory as Mem0Memory
    from mem0.memory.graph_memory import MemoryGraph
    from mem0.configs.base import MemoryConfig
    from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    
# Import CogPrime components
from ..core.state import CognitiveState

# Configure logging
logger = logging.getLogger(__name__)


class MemoryBackend:
    """Base class for memory backends."""
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value with the given key."""
        raise NotImplementedError
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete a value by key."""
        raise NotImplementedError
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for values matching the query."""
        raise NotImplementedError


class DictMemoryBackend(MemoryBackend):
    """Simple in-memory dictionary backend."""
    
    def __init__(self):
        """Initialize an in-memory dictionary backend."""
        self.data = {}
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value with the given key."""
        self.data[key] = value
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        return self.data.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value by key."""
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for values matching the query (simple substring match)."""
        results = []
        for key, value in self.data.items():
            if isinstance(value, str) and query.lower() in value.lower():
                results.append((key, value, 1.0))  # Score 1.0 for exact match
            elif isinstance(value, dict) and any(
                isinstance(v, str) and query.lower() in v.lower() 
                for v in value.values()
            ):
                results.append((key, value, 0.8))  # Score 0.8 for partial match
        
        return results[:limit]


class Mem0MemoryBackend(MemoryBackend):
    """Memory backend using mem0 for persistence, vector search, and graph memory."""
    
    def __init__(self, config: Dict = None):
        """Initialize a mem0-based memory backend.
        
        Args:
            config: Configuration for mem0
        """
        if not MEM0_AVAILABLE:
            raise ImportError(
                "mem0 is not available. Please install it using: "
                "pip install mem0"
            )
        
        self.config = config or {}
        
        # Create mem0 Memory instance
        memory_config = MemoryConfig(
            **self.config.get("memory", {})
        )
        self.memory = Mem0Memory(config=memory_config)
        
        # Enable graph memory if configured
        self.graph_enabled = self.config.get("enable_graph", True)
        
        logger.info("Initialized mem0 memory backend")
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value with the given key."""
        try:
            # Convert to JSON serializable format if needed
            if isinstance(value, CognitiveState):
                value = self._cognitive_state_to_dict(value)
            
            # Store in mem0
            content = json.dumps(value) if not isinstance(value, str) else value
            metadata = {
                "key": key,
                "timestamp": datetime.datetime.now().isoformat(),
                "type": type(value).__name__
            }
            
            self.memory.add(
                content=content,
                metadata=metadata,
                id=key
            )
            return True
        except Exception as e:
            logger.error(f"Error storing value with key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        try:
            # Get from mem0
            result = self.memory.get(key)
            if not result:
                return None
            
            # Parse the content
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            # Convert back to appropriate type
            value_type = metadata.get("type", "")
            if value_type == "CognitiveState":
                return self._dict_to_cognitive_state(json.loads(content))
            elif value_type == "dict":
                return json.loads(content)
            else:
                return content
        except Exception as e:
            logger.error(f"Error retrieving value with key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a value by key."""
        try:
            self.memory.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting value with key {key}: {e}")
            return False
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for values matching the query using vector search."""
        try:
            # Use mem0's vector search
            results = self.memory.search(
                text=query,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                key = result.get("id", "")
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                score = result.get("score", 0.0)
                
                # Convert content based on type
                value_type = metadata.get("type", "")
                if value_type == "CognitiveState":
                    value = self._dict_to_cognitive_state(json.loads(content))
                elif value_type == "dict":
                    value = json.loads(content)
                else:
                    value = content
                
                formatted_results.append((key, value, score))
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching with query {query}: {e}")
            return []
    
    def add_fact(self, fact: str, tags: List[str] = None, 
                metadata: Dict = None) -> str:
        """Add a fact to memory with vector embedding.
        
        Args:
            fact: The fact to add
            tags: List of tags for categorization
            metadata: Additional metadata
            
        Returns:
            ID of the added fact
        """
        try:
            # Prepare metadata
            meta = metadata or {}
            meta["type"] = "fact"
            meta["tags"] = tags or []
            meta["timestamp"] = datetime.datetime.now().isoformat()
            
            # Generate ID if not provided
            fact_id = meta.get("id", f"fact_{uuid.uuid4()}")
            meta["id"] = fact_id
            
            # Add to mem0
            self.memory.add(
                content=fact,
                metadata=meta,
                id=fact_id
            )
            
            return fact_id
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            return None
    
    def extract_facts(self, text: str) -> List[Dict]:
        """Extract facts from text using LLM.
        
        Args:
            text: The text to extract facts from
            
        Returns:
            List of extracted facts with metadata
        """
        try:
            # Use mem0's LLM-based fact extraction
            facts = self.memory.extract_facts(text)
            
            # Add extracted facts to memory
            for fact in facts:
                fact_text = fact.get("fact", "")
                confidence = fact.get("confidence", 0.0)
                
                if fact_text and confidence > 0.7:  # Only add high-confidence facts
                    self.add_fact(
                        fact=fact_text,
                        metadata={"confidence": confidence, "source": "extraction"}
                    )
            
            return facts
        except Exception as e:
            logger.error(f"Error extracting facts: {e}")
            return []
    
    def add_relation(self, subject: str, predicate: str, object: str, 
                    metadata: Dict = None) -> bool:
        """Add a relation to the knowledge graph.
        
        Args:
            subject: Subject of the relation
            predicate: Predicate (relation type)
            object: Object of the relation
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph_enabled:
            logger.warning("Graph memory not enabled")
            return False
        
        try:
            # Use mem0's graph memory
            self.memory.graph.add_relation(
                subject=subject,
                predicate=predicate,
                object=object,
                metadata=metadata or {}
            )
            return True
        except Exception as e:
            logger.error(f"Error adding relation: {e}")
            return False
    
    def query_graph(self, query: str) -> List[Dict]:
        """Query the knowledge graph.
        
        Args:
            query: The query to execute (Cypher format)
            
        Returns:
            List of query results
        """
        if not self.graph_enabled:
            logger.warning("Graph memory not enabled")
            return []
        
        try:
            # Use mem0's graph memory
            return self.memory.graph.query(query)
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return []
    
    def _cognitive_state_to_dict(self, state: CognitiveState) -> Dict:
        """Convert a CognitiveState to a dictionary for storage."""
        # Convert PyTorch tensors to lists
        attention_focus = state.attention_focus.tolist() if hasattr(state.attention_focus, 'tolist') else state.attention_focus
        
        # Convert working memory (may contain non-serializable objects)
        working_memory = {}
        for k, v in state.working_memory.items():
            try:
                # Try to serialize, skip if not possible
                json.dumps({k: v})
                working_memory[k] = v
            except (TypeError, OverflowError):
                working_memory[k] = str(v)
        
        # Convert sensory buffer tensors to lists
        sensory_buffer = {}
        for k, v in state.sensory_buffer.items():
            if hasattr(v, 'tolist'):
                sensory_buffer[k] = v.tolist()
            else:
                sensory_buffer[k] = v
        
        # Convert current thought and last action to string representations
        current_thought = str(state.current_thought) if state.current_thought else None
        last_action = str(state.last_action) if state.last_action else None
        
        return {
            "attention_focus": attention_focus,
            "working_memory": working_memory,
            "emotional_valence": state.emotional_valence,
            "goal_stack": state.goal_stack,
            "sensory_buffer": sensory_buffer,
            "current_thought": current_thought,
            "last_action": last_action,
            "last_reward": state.last_reward,
            "total_reward": state.total_reward,
            "_type": "CognitiveState"
        }
    
    def _dict_to_cognitive_state(self, data: Dict) -> CognitiveState:
        """Convert a dictionary to a CognitiveState."""
        import torch
        from ..modules.reasoning import Thought
        from ..modules.action import Action
        from ..core.state import CognitiveState
        
        # Convert lists back to tensors
        attention_focus = torch.tensor(data.get("attention_focus", [0] * 512))
        
        # Create a new CognitiveState
        state = CognitiveState(
            attention_focus=attention_focus,
            working_memory=data.get("working_memory", {}),
            emotional_valence=data.get("emotional_valence", 0.0),
            goal_stack=data.get("goal_stack", []),
            sensory_buffer=data.get("sensory_buffer", {}),
            last_reward=data.get("last_reward", 0.0),
            total_reward=data.get("total_reward", 0.0)
        )
        
        # Note: current_thought and last_action are stored as strings
        # and would need proper reconstruction from the original classes
        # This is a simplified version that just stores the string representation
        
        return state


class Memory:
    """Memory management for CogPrime, integrating mem0 capabilities."""
    
    def __init__(self, backend_type: str = "mem0", config: Dict = None):
        """Initialize memory with the specified backend.
        
        Args:
            backend_type: Type of backend to use ("dict" or "mem0")
            config: Configuration for the backend
        """
        self.config = config or {}
        
        # Create the appropriate backend
        if backend_type == "mem0" and MEM0_AVAILABLE:
            self.backend = Mem0MemoryBackend(config=self.config)
        else:
            if backend_type == "mem0" and not MEM0_AVAILABLE:
                logger.warning("mem0 not available, falling back to dictionary backend")
            self.backend = DictMemoryBackend()
        
        self.backend_type = backend_type
    
    def store_cognitive_state(self, key: str, state: CognitiveState) -> bool:
        """Store a cognitive state with the given key.
        
        Args:
            key: The key to store the state under
            state: The cognitive state to store
            
        Returns:
            True if successful, False otherwise
        """
        return self.backend.store(key, state)
    
    def retrieve_cognitive_state(self, key: str) -> Optional[CognitiveState]:
        """Retrieve a cognitive state by key.
        
        Args:
            key: The key to retrieve the state for
            
        Returns:
            The cognitive state, or None if not found
        """
        return self.backend.retrieve(key)
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value with the given key.
        
        Args:
            key: The key to store the value under
            value: The value to store
            
        Returns:
            True if successful, False otherwise
        """
        return self.backend.store(key, value)
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.
        
        Args:
            key: The key to retrieve the value for
            
        Returns:
            The value, or None if not found
        """
        return self.backend.retrieve(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value by key.
        
        Args:
            key: The key to delete the value for
            
        Returns:
            True if successful, False otherwise
        """
        return self.backend.delete(key)
    
    def add_fact(self, fact: str, tags: List[str] = None, 
                metadata: Dict = None) -> str:
        """Add a fact to memory with vector embedding.
        
        Args:
            fact: The fact to add
            tags: List of tags for categorization
            metadata: Additional metadata
            
        Returns:
            ID of the added fact
        """
        if hasattr(self.backend, "add_fact"):
            return self.backend.add_fact(fact, tags, metadata)
        else:
            # Fallback for simple backend
            fact_id = f"fact_{uuid.uuid4()}"
            self.backend.store(fact_id, {
                "content": fact,
                "tags": tags or [],
                "metadata": metadata or {},
                "timestamp": datetime.datetime.now().isoformat()
            })
            return fact_id
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for semantically similar content.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of (key, value, score) tuples
        """
        return self.backend.search(query, limit)
    
    def extract_facts(self, text: str) -> List[Dict]:
        """Extract facts from text using LLM.
        
        Args:
            text: The text to extract facts from
            
        Returns:
            List of extracted facts with metadata
        """
        if hasattr(self.backend, "extract_facts"):
            return self.backend.extract_facts(text)
        else:
            logger.warning("Fact extraction not supported by this backend")
            return []
    
    def add_relation(self, subject: str, predicate: str, object: str, 
                    metadata: Dict = None) -> bool:
        """Add a relation to the knowledge graph.
        
        Args:
            subject: Subject of the relation
            predicate: Predicate (relation type)
            object: Object of the relation
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.backend, "add_relation"):
            return self.backend.add_relation(subject, predicate, object, metadata)
        else:
            logger.warning("Graph operations not supported by this backend")
            return False
    
    def query_graph(self, query: str) -> List[Dict]:
        """Query the knowledge graph.
        
        Args:
            query: The query to execute (Cypher format)
            
        Returns:
            List of query results
        """
        if hasattr(self.backend, "query_graph"):
            return self.backend.query_graph(query)
        else:
            logger.warning("Graph operations not supported by this backend")
            return []
    
    def update_working_memory(self, cognitive_state: CognitiveState, 
                             key: str, value: Any) -> CognitiveState:
        """Update the working memory of a cognitive state.
        
        This is a convenience method for CogPrime's cognitive cycle.
        
        Args:
            cognitive_state: The cognitive state to update
            key: The key to update
            value: The new value
            
        Returns:
            The updated cognitive state
        """
        cognitive_state.working_memory[key] = value
        return cognitive_state
    
    def save_experience(self, state: CognitiveState, action: Any, 
                       reward: float, next_state: CognitiveState) -> str:
        """Save an experience for reinforcement learning.
        
        Args:
            state: The initial state
            action: The action taken
            reward: The reward received
            next_state: The resulting state
            
        Returns:
            ID of the saved experience
        """
        experience_id = f"exp_{uuid.uuid4()}"
        
        # Store the experience
        self.store(experience_id, {
            "state": state,
            "action": str(action),  # Convert action to string for storage
            "reward": reward,
            "next_state": next_state,
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "experience"
        })
        
        return experience_id
    
    def get_recent_experiences(self, limit: int = 10) -> List[Dict]:
        """Get recent experiences for batch learning.
        
        Args:
            limit: Maximum number of experiences to retrieve
            
        Returns:
            List of experiences
        """
        if hasattr(self.backend, "search"):
            results = self.backend.search("type:experience", limit=limit)
            return [value for _, value, _ in results]
        else:
            # Simple backend doesn't support search
            return []


# Factory function for creating memory instances

def create_memory(backend_type: str = "mem0", config: Dict = None) -> Memory:
    """Create a new memory instance.
    
    Args:
        backend_type: Type of backend to use ("dict" or "mem0")
        config: Configuration for the backend
        
    Returns:
        A new Memory instance
    """
    return Memory(backend_type=backend_type, config=config)


# Integration with CogPrime modules

def integrate_with_cognitive_core(memory: Memory, core) -> None:
    """Integrate memory with a CogPrime cognitive core.
    
    This sets up the necessary connections for memory to work with the cognitive cycle.
    
    Args:
        memory: The memory instance
        core: The CogPrime cognitive core
    """
    # Store the current cognitive state after each cycle
    def store_state_after_cycle(sensory_input, reward, action):
        state = core.get_cognitive_state()
        memory.store_cognitive_state(f"state_{uuid.uuid4()}", state)
    
    # Register the callback
    if hasattr(core, "register_cycle_callback"):
        core.register_cycle_callback(store_state_after_cycle)
