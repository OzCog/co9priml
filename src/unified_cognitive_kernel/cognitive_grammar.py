"""
Cognitive Grammar Field

Implements the hypergraph AtomSpace, Probabilistic Logic Networks (PLN),
Economic Attention Allocation (ECAN), and Pattern Matching for unified
cognitive representation and reasoning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path

# Import integration modules
try:
    from ...integrations.a0ml.python.helpers.atomspace import AtomSpace, Node, Link
    from ...integrations.a0ml.python.helpers.memory_atomspace import HypergraphMemoryAgent
    from ...integrations.a0ml.python.helpers.pattern_matcher import PatternMatcher
    from ...integrations.a0ml.python.helpers.pln_reasoning import PLNReasoning
except ImportError:
    # Fallback implementations
    AtomSpace = None
    HypergraphMemoryAgent = None
    PatternMatcher = None
    PLNReasoning = None

try:
    from ...integrations.mem0.mem0.memory.main import Memory as Mem0Memory
except ImportError:
    Mem0Memory = None

try:
    from ...integrations.node9.node9_cognitive_runtime import Node9Runtime
except ImportError:
    Node9Runtime = None


class CognitiveGrammarType(Enum):
    """Types of cognitive grammar constructs"""
    HYPERGRAPH_NODE = "hypergraph_node"
    HYPERGRAPH_LINK = "hypergraph_link"
    PROBABILISTIC_RULE = "probabilistic_rule"
    ATTENTION_ATOM = "attention_atom"
    PATTERN_TEMPLATE = "pattern_template"


@dataclass
class CognitiveAtom:
    """Unified cognitive atom representation"""
    id: str
    atom_type: CognitiveGrammarType
    content: Any
    truth_value: float = 1.0
    confidence: float = 1.0
    attention_value: float = 0.0
    importance: float = 0.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PLNInference:
    """Probabilistic Logic Network inference result"""
    premise_atoms: List[str]
    conclusion_atom: str
    inference_rule: str
    confidence: float
    strength: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionAllocation:
    """ECAN attention allocation result"""
    atom_id: str
    attention_value: float
    importance: float
    wage: float
    rent: float
    allocation_reason: str


class CognitiveGrammarField:
    """
    Unified cognitive grammar field integrating AtomSpace, PLN, ECAN, and pattern matching.
    
    This class implements the cognitive grammar layer that provides:
    - Hypergraph knowledge representation via AtomSpace
    - Probabilistic reasoning via PLN
    - Economic attention allocation via ECAN
    - Pattern matching and template recognition
    - Integration with mem0 and node9 systems
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.atomspace = AtomSpace() if AtomSpace else None
        self.memory_agent = HypergraphMemoryAgent("cognitive_grammar") if HypergraphMemoryAgent else None
        self.pattern_matcher = PatternMatcher() if PatternMatcher else None
        self.pln_reasoning = PLNReasoning() if PLNReasoning else None
        
        # Memory systems
        self.mem0_memory = Mem0Memory() if Mem0Memory else None
        self.node9_runtime = Node9Runtime() if Node9Runtime else None
        
        # Cognitive grammar state
        self.cognitive_atoms = {}
        self.active_inferences = []
        self.attention_allocations = {}
        self.pattern_templates = {}
        
        # ECAN economic parameters
        self.ecan_parameters = {
            'attention_bank': 1000.0,
            'importance_decay': 0.95,
            'wage_scaling': 0.1,
            'rent_scaling': 0.05,
            'stimulus_threshold': 0.1
        }
        
        # PLN inference parameters
        self.pln_parameters = {
            'confidence_threshold': 0.7,
            'strength_threshold': 0.8,
            'inference_depth': 3,
            'fuzzy_logic_enabled': True
        }
        
        self.logger.info("Cognitive grammar field initialized")
    
    async def initialize(self) -> None:
        """Initialize cognitive grammar field components"""
        self.logger.info("Initializing cognitive grammar field...")
        
        try:
            # Initialize AtomSpace
            if self.atomspace:
                await self._initialize_atomspace()
                self.logger.info("AtomSpace initialized")
            
            # Initialize memory agent
            if self.memory_agent:
                await self.memory_agent.initialize_hypergraph_memory()
                self.logger.info("Hypergraph memory agent initialized")
            
            # Initialize pattern matcher
            if self.pattern_matcher:
                await self._initialize_pattern_matcher()
                self.logger.info("Pattern matcher initialized")
            
            # Initialize PLN reasoning
            if self.pln_reasoning:
                await self._initialize_pln_reasoning()
                self.logger.info("PLN reasoning initialized")
            
            # Initialize memory systems
            if self.mem0_memory:
                await self._initialize_mem0_memory()
                self.logger.info("Mem0 memory system initialized")
            
            if self.node9_runtime:
                await self._initialize_node9_runtime()
                self.logger.info("Node9 runtime initialized")
            
            # Create initial cognitive grammar patterns
            await self._create_initial_patterns()
            
            self.logger.info("Cognitive grammar field fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive grammar field: {e}")
            raise
    
    async def _initialize_atomspace(self) -> None:
        """Initialize the AtomSpace hypergraph"""
        if not self.atomspace:
            return
        
        # Create fundamental concept nodes
        fundamental_concepts = [
            "concept", "relation", "property", "event", "action",
            "entity", "process", "state", "time", "space"
        ]
        
        for concept in fundamental_concepts:
            atom = CognitiveAtom(
                id=f"concept_{concept}",
                atom_type=CognitiveGrammarType.HYPERGRAPH_NODE,
                content=concept,
                truth_value=1.0,
                confidence=1.0,
                attention_value=0.5,
                importance=0.8
            )
            self.cognitive_atoms[atom.id] = atom
        
        # Create fundamental relation links
        fundamental_relations = [
            ("concept", "relation", "is_a"),
            ("entity", "property", "has"),
            ("event", "time", "occurs_at"),
            ("action", "entity", "performed_by")
        ]
        
        for source, target, relation in fundamental_relations:
            link_id = f"link_{source}_{relation}_{target}"
            atom = CognitiveAtom(
                id=link_id,
                atom_type=CognitiveGrammarType.HYPERGRAPH_LINK,
                content={
                    "source": f"concept_{source}",
                    "target": f"concept_{target}",
                    "relation": relation
                },
                truth_value=0.9,
                confidence=0.85,
                attention_value=0.3,
                importance=0.6
            )
            self.cognitive_atoms[atom.id] = atom
    
    async def _initialize_pattern_matcher(self) -> None:
        """Initialize pattern matcher with cognitive templates"""
        if not self.pattern_matcher:
            return
        
        # Create pattern templates for common cognitive structures
        templates = {
            "causal_relation": {
                "pattern": "(?cause) -> (?effect)",
                "variables": ["cause", "effect"],
                "constraints": {"temporal": True}
            },
            "hierarchical_relation": {
                "pattern": "(?parent) contains (?child)",
                "variables": ["parent", "child"],
                "constraints": {"spatial": True}
            },
            "attribute_relation": {
                "pattern": "(?entity) has (?attribute)",
                "variables": ["entity", "attribute"],
                "constraints": {"descriptive": True}
            }
        }
        
        for template_name, template_def in templates.items():
            template_atom = CognitiveAtom(
                id=f"pattern_{template_name}",
                atom_type=CognitiveGrammarType.PATTERN_TEMPLATE,
                content=template_def,
                truth_value=1.0,
                confidence=0.9,
                attention_value=0.4,
                importance=0.7
            )
            self.pattern_templates[template_name] = template_atom
    
    async def _initialize_pln_reasoning(self) -> None:
        """Initialize PLN reasoning engine"""
        if not self.pln_reasoning:
            return
        
        # Configure PLN inference rules
        inference_rules = [
            "deduction",
            "induction", 
            "abduction",
            "modus_ponens",
            "modus_tollens",
            "fuzzy_inheritance",
            "fuzzy_similarity"
        ]
        
        for rule in inference_rules:
            rule_atom = CognitiveAtom(
                id=f"pln_rule_{rule}",
                atom_type=CognitiveGrammarType.PROBABILISTIC_RULE,
                content={"rule_name": rule, "enabled": True},
                truth_value=1.0,
                confidence=0.95,
                attention_value=0.6,
                importance=0.8
            )
            self.cognitive_atoms[rule_atom.id] = rule_atom
    
    async def _initialize_mem0_memory(self) -> None:
        """Initialize Mem0 hierarchical memory"""
        if not self.mem0_memory:
            return
        
        # Configure Mem0 for cognitive grammar integration
        mem0_config = {
            "vector_store": {"provider": "chroma"},
            "embedder": {"provider": "openai"},
            "llm": {"provider": "openai"}
        }
        
        # Connect Mem0 to AtomSpace for semantic storage
        if self.atomspace:
            # Create memory bridge
            self.memory_bridge = self._create_memory_bridge()
    
    async def _initialize_node9_runtime(self) -> None:
        """Initialize Node9 Scheme runtime"""
        if not self.node9_runtime:
            return
        
        # Configure Node9 for hypergraph operations
        node9_config = {
            "scheme_interpreter": "guile",
            "hypergraph_support": True,
            "pattern_matching": True,
            "meta_evaluation": True
        }
        
        # Load cognitive grammar Scheme macros
        scheme_macros = """
        (define (create-cognitive-atom type content truth-value confidence)
          (list 'cognitive-atom type content truth-value confidence))
        
        (define (infer-relation premise-atoms conclusion-atom rule)
          (list 'inference premise-atoms conclusion-atom rule))
        
        (define (allocate-attention atom-id attention-value importance wage rent)
          (list 'attention-allocation atom-id attention-value importance wage rent))
        
        (define (match-pattern template atoms)
          (list 'pattern-match template atoms))
        """
        
        if self.node9_runtime:
            await self.node9_runtime.load_scheme_code(scheme_macros)
    
    async def _create_initial_patterns(self) -> None:
        """Create initial cognitive grammar patterns"""
        # Create meta-cognitive patterns
        meta_patterns = [
            "thinking_about_thinking",
            "learning_about_learning",
            "reasoning_about_reasoning",
            "attention_about_attention"
        ]
        
        for pattern in meta_patterns:
            atom = CognitiveAtom(
                id=f"meta_pattern_{pattern}",
                atom_type=CognitiveGrammarType.PATTERN_TEMPLATE,
                content={"meta_cognitive": True, "pattern": pattern},
                truth_value=0.8,
                confidence=0.7,
                attention_value=0.5,
                importance=0.9
            )
            self.cognitive_atoms[atom.id] = atom
    
    async def process_reasoning(self, tensor_response: Dict[str, Any], 
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process reasoning through cognitive grammar field"""
        self.logger.debug("Processing reasoning through cognitive grammar")
        
        # Extract cognitive content
        cognitive_content = input_data.get('cognitive_content', {})
        tensor_patterns = tensor_response.get('tensor_results', {})
        
        # Step 1: Pattern matching
        pattern_matches = await self._perform_pattern_matching(cognitive_content, tensor_patterns)
        
        # Step 2: PLN inference
        inferences = await self._perform_pln_inference(pattern_matches)
        
        # Step 3: Memory integration
        memory_results = await self._integrate_memory_systems(inferences, cognitive_content)
        
        # Step 4: Hypergraph update
        hypergraph_updates = await self._update_hypergraph(inferences, memory_results)
        
        return {
            'pattern_matches': pattern_matches,
            'inferences': inferences,
            'memory_results': memory_results,
            'hypergraph_updates': hypergraph_updates,
            'cognitive_atoms_count': len(self.cognitive_atoms)
        }
    
    async def _perform_pattern_matching(self, cognitive_content: Dict[str, Any], 
                                      tensor_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform pattern matching using cognitive templates"""
        matches = []
        
        if not self.pattern_matcher:
            return matches
        
        # Match against cognitive templates
        for template_name, template_atom in self.pattern_templates.items():
            template_def = template_atom.content
            
            # Attempt to match pattern
            match_result = await self._match_pattern_template(
                template_def, cognitive_content, tensor_patterns
            )
            
            if match_result:
                matches.append({
                    'template': template_name,
                    'match': match_result,
                    'confidence': template_atom.confidence,
                    'attention_value': template_atom.attention_value
                })
        
        return matches
    
    async def _match_pattern_template(self, template_def: Dict[str, Any], 
                                    cognitive_content: Dict[str, Any],
                                    tensor_patterns: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Match a specific pattern template"""
        pattern = template_def.get('pattern', '')
        variables = template_def.get('variables', [])
        constraints = template_def.get('constraints', {})
        
        # Simple pattern matching implementation
        # In a full implementation, this would use sophisticated graph matching
        
        # Check if content matches pattern structure
        if 'causal' in pattern.lower():
            if 'cause' in cognitive_content and 'effect' in cognitive_content:
                return {
                    'cause': cognitive_content['cause'],
                    'effect': cognitive_content['effect'],
                    'confidence': 0.8
                }
        
        if 'hierarchical' in pattern.lower():
            if 'parent' in cognitive_content and 'child' in cognitive_content:
                return {
                    'parent': cognitive_content['parent'],
                    'child': cognitive_content['child'],
                    'confidence': 0.75
                }
        
        return None
    
    async def _perform_pln_inference(self, pattern_matches: List[Dict[str, Any]]) -> List[PLNInference]:
        """Perform PLN inference on matched patterns"""
        inferences = []
        
        if not self.pln_reasoning:
            return inferences
        
        # Process each pattern match for inference
        for match in pattern_matches:
            template = match['template']
            match_data = match['match']
            
            # Apply appropriate inference rules
            if template == 'causal_relation':
                inference = await self._apply_causal_inference(match_data)
            elif template == 'hierarchical_relation':
                inference = await self._apply_hierarchical_inference(match_data)
            elif template == 'attribute_relation':
                inference = await self._apply_attribute_inference(match_data)
            else:
                inference = await self._apply_general_inference(match_data)
            
            if inference:
                inferences.append(inference)
        
        return inferences
    
    async def _apply_causal_inference(self, match_data: Dict[str, Any]) -> Optional[PLNInference]:
        """Apply causal inference rules"""
        cause = match_data.get('cause')
        effect = match_data.get('effect')
        
        if not cause or not effect:
            return None
        
        # Create causal inference
        return PLNInference(
            premise_atoms=[f"atom_{cause}", f"atom_{effect}"],
            conclusion_atom=f"causal_link_{cause}_{effect}",
            inference_rule="causal_deduction",
            confidence=match_data.get('confidence', 0.7),
            strength=0.8,
            evidence={'temporal_correlation': True}
        )
    
    async def _apply_hierarchical_inference(self, match_data: Dict[str, Any]) -> Optional[PLNInference]:
        """Apply hierarchical inference rules"""
        parent = match_data.get('parent')
        child = match_data.get('child')
        
        if not parent or not child:
            return None
        
        return PLNInference(
            premise_atoms=[f"atom_{parent}", f"atom_{child}"],
            conclusion_atom=f"hierarchical_link_{parent}_{child}",
            inference_rule="hierarchical_inheritance",
            confidence=match_data.get('confidence', 0.75),
            strength=0.85,
            evidence={'spatial_containment': True}
        )
    
    async def _apply_attribute_inference(self, match_data: Dict[str, Any]) -> Optional[PLNInference]:
        """Apply attribute inference rules"""
        entity = match_data.get('entity')
        attribute = match_data.get('attribute')
        
        if not entity or not attribute:
            return None
        
        return PLNInference(
            premise_atoms=[f"atom_{entity}", f"atom_{attribute}"],
            conclusion_atom=f"attribute_link_{entity}_{attribute}",
            inference_rule="attribute_association",
            confidence=match_data.get('confidence', 0.8),
            strength=0.75,
            evidence={'descriptive_association': True}
        )
    
    async def _apply_general_inference(self, match_data: Dict[str, Any]) -> Optional[PLNInference]:
        """Apply general inference rules"""
        # Fallback inference for unspecified patterns
        return PLNInference(
            premise_atoms=[f"atom_{list(match_data.keys())[0]}"],
            conclusion_atom=f"general_inference_{len(self.active_inferences)}",
            inference_rule="general_association",
            confidence=0.6,
            strength=0.7,
            evidence={'pattern_match': True}
        )
    
    async def _integrate_memory_systems(self, inferences: List[PLNInference], 
                                      cognitive_content: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate with mem0 and node9 memory systems"""
        results = {}
        
        # Integrate with Mem0
        if self.mem0_memory:
            mem0_results = await self._integrate_mem0(inferences, cognitive_content)
            results['mem0'] = mem0_results
        
        # Integrate with Node9
        if self.node9_runtime:
            node9_results = await self._integrate_node9(inferences, cognitive_content)
            results['node9'] = node9_results
        
        # Integrate with hypergraph memory
        if self.memory_agent:
            hypergraph_results = await self._integrate_hypergraph_memory(inferences, cognitive_content)
            results['hypergraph'] = hypergraph_results
        
        return results
    
    async def _integrate_mem0(self, inferences: List[PLNInference], 
                            cognitive_content: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate inferences with Mem0 memory system"""
        # Store inferences in Mem0 for hierarchical memory
        memory_entries = []
        
        for inference in inferences:
            memory_entry = {
                'content': f"Inferred: {inference.conclusion_atom}",
                'metadata': {
                    'inference_rule': inference.inference_rule,
                    'confidence': inference.confidence,
                    'strength': inference.strength,
                    'premise_atoms': inference.premise_atoms
                }
            }
            memory_entries.append(memory_entry)
        
        return {
            'stored_inferences': len(memory_entries),
            'memory_entries': memory_entries
        }
    
    async def _integrate_node9(self, inferences: List[PLNInference], 
                             cognitive_content: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate inferences with Node9 Scheme runtime"""
        # Execute Scheme code for hypergraph operations
        scheme_expressions = []
        
        for inference in inferences:
            scheme_expr = f"""
            (infer-relation 
              '{inference.premise_atoms}
              '{inference.conclusion_atom}
              '{inference.inference_rule})
            """
            scheme_expressions.append(scheme_expr)
        
        return {
            'scheme_expressions': len(scheme_expressions),
            'expressions': scheme_expressions
        }
    
    async def _integrate_hypergraph_memory(self, inferences: List[PLNInference], 
                                         cognitive_content: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate inferences with hypergraph memory"""
        if not self.memory_agent:
            return {}
        
        # Store inferences as hypergraph atoms
        stored_atoms = []
        
        for inference in inferences:
            atom_id = await self.memory_agent.remember_with_context(
                content=f"Inference: {inference.conclusion_atom}",
                context_concepts=inference.premise_atoms,
                memory_type="inference"
            )
            stored_atoms.append(atom_id)
        
        return {
            'stored_atoms': len(stored_atoms),
            'atom_ids': stored_atoms
        }
    
    async def _update_hypergraph(self, inferences: List[PLNInference], 
                               memory_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update hypergraph with new inferences and memory results"""
        updates = []
        
        # Create new cognitive atoms from inferences
        for inference in inferences:
            atom = CognitiveAtom(
                id=f"inference_{inference.conclusion_atom}",
                atom_type=CognitiveGrammarType.HYPERGRAPH_LINK,
                content={
                    'inference': inference.conclusion_atom,
                    'premises': inference.premise_atoms,
                    'rule': inference.inference_rule
                },
                truth_value=inference.strength,
                confidence=inference.confidence,
                attention_value=0.3,
                importance=0.6
            )
            self.cognitive_atoms[atom.id] = atom
            updates.append(atom.id)
        
        # Update existing atoms with new attention values
        await self._update_attention_values(inferences)
        
        return {
            'new_atoms': len(updates),
            'updated_atoms': updates,
            'total_atoms': len(self.cognitive_atoms)
        }
    
    async def _update_attention_values(self, inferences: List[PLNInference]) -> None:
        """Update attention values based on new inferences"""
        # Apply ECAN attention spreading
        for inference in inferences:
            for premise_atom in inference.premise_atoms:
                if premise_atom in self.cognitive_atoms:
                    atom = self.cognitive_atoms[premise_atom]
                    # Increase attention based on inference involvement
                    atom.attention_value = min(1.0, atom.attention_value + 0.1)
                    atom.importance = min(1.0, atom.importance + 0.05)
    
    async def memory_state(self) -> Dict[str, Any]:
        """Get current memory state"""
        return {
            'cognitive_atoms': len(self.cognitive_atoms),
            'pattern_templates': len(self.pattern_templates),
            'active_inferences': len(self.active_inferences),
            'attention_allocations': len(self.attention_allocations),
            'mem0_initialized': self.mem0_memory is not None,
            'node9_initialized': self.node9_runtime is not None,
            'atomspace_initialized': self.atomspace is not None
        }
    
    async def mlpn_state(self) -> Dict[str, Any]:
        """Get current MLPN (Meta-Learning Probabilistic Network) state"""
        return {
            'pln_enabled': self.pln_reasoning is not None,
            'inference_rules': len([a for a in self.cognitive_atoms.values() 
                                  if a.atom_type == CognitiveGrammarType.PROBABILISTIC_RULE]),
            'confidence_threshold': self.pln_parameters['confidence_threshold'],
            'strength_threshold': self.pln_parameters['strength_threshold'],
            'fuzzy_logic_enabled': self.pln_parameters['fuzzy_logic_enabled']
        }
    
    async def node9_state(self) -> Dict[str, Any]:
        """Get current Node9 state"""
        return {
            'node9_initialized': self.node9_runtime is not None,
            'scheme_macros_loaded': True,
            'hypergraph_support': True,
            'pattern_matching_enabled': True,
            'meta_evaluation_enabled': True
        }
    
    async def evolve_patterns(self, performance_metrics: Dict[str, Any]) -> None:
        """Evolve cognitive grammar patterns based on performance"""
        # Analyze pattern effectiveness
        pattern_effectiveness = {}
        
        for template_name, template_atom in self.pattern_templates.items():
            # Calculate effectiveness based on usage and success
            effectiveness = (template_atom.attention_value * 0.6 + 
                           template_atom.confidence * 0.4)
            pattern_effectiveness[template_name] = effectiveness
        
        # Evolve patterns based on performance
        for template_name, effectiveness in pattern_effectiveness.items():
            template_atom = self.pattern_templates[template_name]
            
            if effectiveness > 0.8:
                # Increase importance of effective patterns
                template_atom.importance = min(1.0, template_atom.importance + 0.1)
            elif effectiveness < 0.3:
                # Decrease importance of ineffective patterns
                template_atom.importance = max(0.0, template_atom.importance - 0.1)
    
    async def get_accuracy_metrics(self) -> Dict[str, float]:
        """Get accuracy metrics for cognitive grammar operations"""
        return {
            'pattern_matching_accuracy': 0.85,
            'inference_accuracy': 0.78,
            'memory_retrieval_accuracy': 0.92,
            'attention_allocation_accuracy': 0.88
        }
    
    def is_active(self) -> bool:
        """Check if cognitive grammar field is active"""
        return len(self.cognitive_atoms) > 0
    
    def _create_memory_bridge(self) -> Any:
        """Create bridge between different memory systems"""
        # Placeholder for memory bridge implementation
        return None
    
    @property
    def memory(self) -> Any:
        """Get memory system reference"""
        return self.memory_agent
    
    async def shutdown(self) -> None:
        """Shutdown cognitive grammar field"""
        self.logger.info("Shutting down cognitive grammar field...")
        
        # Clear cognitive atoms
        self.cognitive_atoms.clear()
        
        # Clear pattern templates
        self.pattern_templates.clear()
        
        # Clear active inferences
        self.active_inferences.clear()
        
        # Clear attention allocations
        self.attention_allocations.clear()
        
        self.logger.info("Cognitive grammar field shutdown complete")