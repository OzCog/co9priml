"""
Tensor Shape Meta-Design

Implements tensor shape meta-design with prime factorization for unique addressing
and compositionality in the unified cognitive kernel gestalt field.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import math
from collections import defaultdict


class TensorCategory(Enum):
    """Categories of tensor shapes"""
    GGML_TENSOR = "ggml_tensor"
    KOKKOS_TENSOR = "kokkos_tensor"
    MEM0_TENSOR = "mem0_tensor"
    MLPN_TENSOR = "mlpn_tensor"
    NODE9_TENSOR = "node9_tensor"
    ATOMSPACE_TENSOR = "atomspace_tensor"
    GESTALT_TENSOR = "gestalt_tensor"


class TensorComplexity(Enum):
    """Complexity levels of tensor shapes"""
    SIMPLE = "simple"          # 1-2 dimensions
    MODERATE = "moderate"      # 3-4 dimensions
    COMPLEX = "complex"        # 5-6 dimensions
    HYPERCOMPLEX = "hypercomplex"  # 7+ dimensions


@dataclass
class PrimeFactorization:
    """Prime factorization of a number"""
    number: int
    factors: List[int]
    exponents: List[int]
    factorization_string: str
    
    def __post_init__(self):
        if not self.factorization_string:
            self.factorization_string = self._create_factorization_string()
    
    def _create_factorization_string(self) -> str:
        """Create string representation of prime factorization"""
        if not self.factors:
            return str(self.number)
        
        parts = []
        for factor, exponent in zip(self.factors, self.exponents):
            if exponent == 1:
                parts.append(str(factor))
            else:
                parts.append(f"{factor}^{exponent}")
        
        return " * ".join(parts)


@dataclass
class TensorShape:
    """Enhanced tensor shape with meta-design information"""
    dimensions: Tuple[int, ...]
    category: TensorCategory
    complexity: TensorComplexity
    prime_factorizations: List[PrimeFactorization]
    degrees_of_freedom: int
    unique_address: str
    lexeme_representation: str
    grammatical_role: str
    temporal_horizon: int = 1
    connectivity_depth: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GestaltTensorField:
    """Unified gestalt tensor field representation"""
    shape: Tuple[int, ...]
    component_tensors: Dict[str, TensorShape]
    gestalt_address: str
    field_energy: float
    coherence_measure: float
    prime_signature: str
    created_time: float
    update_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrimeFactorCalculator:
    """Calculator for prime factorization with caching"""
    
    def __init__(self):
        self.cache = {}
        self.logger = logging.getLogger(__name__)
    
    def factorize(self, n: int) -> PrimeFactorization:
        """Calculate prime factorization of a number"""
        if n in self.cache:
            return self.cache[n]
        
        if n <= 1:
            result = PrimeFactorization(n, [], [], str(n))
            self.cache[n] = result
            return result
        
        factors = []
        exponents = []
        temp_n = n
        
        # Check for factor 2
        if temp_n % 2 == 0:
            count = 0
            while temp_n % 2 == 0:
                temp_n //= 2
                count += 1
            factors.append(2)
            exponents.append(count)
        
        # Check for odd factors
        i = 3
        while i * i <= temp_n:
            if temp_n % i == 0:
                count = 0
                while temp_n % i == 0:
                    temp_n //= i
                    count += 1
                factors.append(i)
                exponents.append(count)
            i += 2
        
        # If temp_n is a prime greater than 2
        if temp_n > 2:
            factors.append(temp_n)
            exponents.append(1)
        
        result = PrimeFactorization(n, factors, exponents, "")
        self.cache[n] = result
        return result
    
    def get_prime_signature(self, dimensions: Tuple[int, ...]) -> str:
        """Get unique prime signature for tensor dimensions"""
        factorizations = [self.factorize(dim) for dim in dimensions]
        
        # Create signature from prime factors
        all_primes = set()
        for factorization in factorizations:
            all_primes.update(factorization.factors)
        
        # Sort primes and create signature
        sorted_primes = sorted(all_primes)
        signature_parts = []
        
        for prime in sorted_primes:
            total_exponent = sum(
                factorization.exponents[factorization.factors.index(prime)]
                for factorization in factorizations
                if prime in factorization.factors
            )
            signature_parts.append(f"{prime}^{total_exponent}")
        
        return "_".join(signature_parts)


class LexemeGenerator:
    """Generator for lexeme representations of tensor shapes"""
    
    def __init__(self):
        self.grammatical_roles = {
            1: "scalar",
            2: "vector", 
            3: "matrix",
            4: "tensor3d",
            5: "tensor4d",
            6: "tensor5d",
            7: "hypercube",
            8: "octatensor"
        }
        
        self.linguistic_mappings = {
            "scalar": "noun",
            "vector": "verb", 
            "matrix": "adjective",
            "tensor3d": "adverb",
            "tensor4d": "preposition",
            "tensor5d": "conjunction",
            "hypercube": "interjection",
            "octatensor": "determiner"
        }
        
        self.logger = logging.getLogger(__name__)
    
    def generate_lexeme(self, tensor_shape: TensorShape) -> str:
        """Generate lexeme representation for tensor shape"""
        dimensions = tensor_shape.dimensions
        prime_sig = tensor_shape.unique_address
        
        # Get base grammatical role
        ndims = len(dimensions)
        base_role = self.grammatical_roles.get(ndims, f"tensor{ndims}d")
        linguistic_role = self.linguistic_mappings.get(base_role, "unknown")
        
        # Create lexeme from prime factorization
        prime_parts = []
        for factorization in tensor_shape.prime_factorizations:
            prime_parts.append(factorization.factorization_string)
        
        lexeme = f"{base_role}_{linguistic_role}_{prime_sig}"
        
        return lexeme


class TensorShapeMetaDesign:
    """
    Tensor Shape Meta-Design system for unified cognitive kernel.
    
    This class implements the tensor shape meta-design with prime factorization
    for unique addressing and compositionality within the gestalt field.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.prime_calculator = PrimeFactorCalculator()
        self.lexeme_generator = LexemeGenerator()
        
        # Shape registry
        self.registered_shapes = {}
        self.gestalt_fields = {}
        self.shape_evolution_history = []
        
        # Meta-design parameters
        self.max_dimensions = 12
        self.min_dimension_size = 1
        self.max_dimension_size = 10000
        self.gestalt_coherence_threshold = 0.7
        
        # Component mappings
        self.component_shape_mappings = {
            TensorCategory.GGML_TENSOR: self._map_ggml_shape,
            TensorCategory.KOKKOS_TENSOR: self._map_kokkos_shape,
            TensorCategory.MEM0_TENSOR: self._map_mem0_shape,
            TensorCategory.MLPN_TENSOR: self._map_mlpn_shape,
            TensorCategory.NODE9_TENSOR: self._map_node9_shape,
            TensorCategory.ATOMSPACE_TENSOR: self._map_atomspace_shape
        }
        
        self.logger.info("Tensor shape meta-design initialized")
    
    def create_tensor_shape(self, dimensions: Tuple[int, ...], 
                          category: TensorCategory,
                          metadata: Optional[Dict[str, Any]] = None) -> TensorShape:
        """Create enhanced tensor shape with meta-design information"""
        if not dimensions:
            raise ValueError("Dimensions cannot be empty")
        
        if len(dimensions) > self.max_dimensions:
            raise ValueError(f"Too many dimensions: {len(dimensions)} > {self.max_dimensions}")
        
        # Calculate prime factorizations
        prime_factorizations = []
        for dim in dimensions:
            if dim < self.min_dimension_size or dim > self.max_dimension_size:
                raise ValueError(f"Dimension size out of range: {dim}")
            factorization = self.prime_calculator.factorize(dim)
            prime_factorizations.append(factorization)
        
        # Determine complexity
        complexity = self._determine_complexity(dimensions)
        
        # Calculate degrees of freedom
        degrees_of_freedom = int(np.prod(dimensions))
        
        # Generate unique address
        unique_address = self.prime_calculator.get_prime_signature(dimensions)
        
        # Create tensor shape
        tensor_shape = TensorShape(
            dimensions=dimensions,
            category=category,
            complexity=complexity,
            prime_factorizations=prime_factorizations,
            degrees_of_freedom=degrees_of_freedom,
            unique_address=unique_address,
            lexeme_representation="",
            grammatical_role="",
            metadata=metadata or {}
        )
        
        # Generate lexeme representation
        tensor_shape.lexeme_representation = self.lexeme_generator.generate_lexeme(tensor_shape)
        tensor_shape.grammatical_role = self._determine_grammatical_role(tensor_shape)
        
        # Register shape
        self.registered_shapes[unique_address] = tensor_shape
        
        self.logger.debug(f"Created tensor shape: {dimensions} -> {unique_address}")
        return tensor_shape
    
    def _determine_complexity(self, dimensions: Tuple[int, ...]) -> TensorComplexity:
        """Determine complexity level of tensor shape"""
        ndims = len(dimensions)
        
        if ndims <= 2:
            return TensorComplexity.SIMPLE
        elif ndims <= 4:
            return TensorComplexity.MODERATE
        elif ndims <= 6:
            return TensorComplexity.COMPLEX
        else:
            return TensorComplexity.HYPERCOMPLEX
    
    def _determine_grammatical_role(self, tensor_shape: TensorShape) -> str:
        """Determine grammatical role of tensor shape"""
        ndims = len(tensor_shape.dimensions)
        
        # Map dimensions to grammatical roles
        role_mapping = {
            1: "subject",      # Scalar
            2: "predicate",    # Vector
            3: "object",       # Matrix
            4: "modifier",     # 3D tensor
            5: "relation",     # 4D tensor
            6: "context",      # 5D tensor
            7: "meta_context", # Hypercube
        }
        
        return role_mapping.get(ndims, "complex_structure")
    
    def compute_gestalt_shape(self, component_shapes: Dict[str, TensorShape]) -> Tuple[int, ...]:
        """Compute unified gestalt shape from component tensor shapes"""
        if not component_shapes:
            return (1,)
        
        # Collect all dimension sizes with their prime factors
        all_dimensions = []
        dimension_categories = defaultdict(list)
        
        for component_name, tensor_shape in component_shapes.items():
            for i, dim in enumerate(tensor_shape.dimensions):
                all_dimensions.append(dim)
                dimension_categories[i].append(dim)
        
        # Compute gestalt dimensions using prime factorization alignment
        gestalt_dimensions = []
        
        # For each dimension position, find the optimal gestalt dimension
        max_dim_positions = max(len(ts.dimensions) for ts in component_shapes.values())
        
        for pos in range(max_dim_positions):
            dims_at_position = []
            
            for tensor_shape in component_shapes.values():
                if pos < len(tensor_shape.dimensions):
                    dims_at_position.append(tensor_shape.dimensions[pos])
            
            if dims_at_position:
                # Use prime factorization to find optimal gestalt dimension
                gestalt_dim = self._compute_gestalt_dimension(dims_at_position)
                gestalt_dimensions.append(gestalt_dim)
        
        # Ensure gestalt has at least one dimension
        if not gestalt_dimensions:
            gestalt_dimensions = [1]
        
        return tuple(gestalt_dimensions)
    
    def _compute_gestalt_dimension(self, dimensions: List[int]) -> int:
        """Compute gestalt dimension from a list of dimensions"""
        if not dimensions:
            return 1
        
        # Get prime factorizations for all dimensions
        factorizations = [self.prime_calculator.factorize(dim) for dim in dimensions]
        
        # Find common primes and compute LCM-like gestalt dimension
        all_primes = set()
        for factorization in factorizations:
            all_primes.update(factorization.factors)
        
        gestalt_factors = {}
        for prime in all_primes:
            max_exponent = 0
            for factorization in factorizations:
                if prime in factorization.factors:
                    idx = factorization.factors.index(prime)
                    max_exponent = max(max_exponent, factorization.exponents[idx])
            gestalt_factors[prime] = max_exponent
        
        # Compute gestalt dimension
        gestalt_dim = 1
        for prime, exponent in gestalt_factors.items():
            gestalt_dim *= prime ** exponent
        
        # Limit gestalt dimension size
        if gestalt_dim > self.max_dimension_size:
            # Scale down while preserving prime structure
            scale_factor = math.ceil(gestalt_dim / self.max_dimension_size)
            gestalt_dim //= scale_factor
        
        return max(1, gestalt_dim)
    
    def factorize_component_tensor(self, component_name: str, 
                                 component_state: Dict[str, Any]) -> np.ndarray:
        """Factorize component tensor using prime factorization principles"""
        # Extract tensor data from component state
        tensor_data = self._extract_tensor_data(component_name, component_state)
        
        if tensor_data is None:
            # Create default tensor for component
            return self._create_default_component_tensor(component_name)
        
        # Apply prime factorization transformation
        factorized_tensor = self._apply_prime_factorization_transform(tensor_data, component_name)
        
        return factorized_tensor
    
    def _extract_tensor_data(self, component_name: str, 
                           component_state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract tensor data from component state"""
        # Look for common tensor data keys
        tensor_keys = ['tensor', 'data', 'state', 'values', 'embeddings']
        
        for key in tensor_keys:
            if key in component_state:
                data = component_state[key]
                if isinstance(data, (list, tuple)):
                    return np.array(data)
                elif isinstance(data, np.ndarray):
                    return data
                elif isinstance(data, (int, float)):
                    return np.array([data])
        
        return None
    
    def _create_default_component_tensor(self, component_name: str) -> np.ndarray:
        """Create default tensor for component"""
        # Create component-specific default tensors
        default_shapes = {
            'ggml': (64, 32),
            'kokkos': (128,),
            'a0ml': (10, 5),
            'mem0': (3, 3, 64),
            'mlpn': (128, 64, 32),
            'node9': (3, 3, 3)
        }
        
        # Find matching component
        for key, shape in default_shapes.items():
            if key in component_name.lower():
                return np.random.randn(*shape) * 0.1
        
        # Default fallback
        return np.random.randn(16, 8) * 0.1
    
    def _apply_prime_factorization_transform(self, tensor_data: np.ndarray, 
                                           component_name: str) -> np.ndarray:
        """Apply prime factorization transformation to tensor data"""
        original_shape = tensor_data.shape
        
        # Create tensor shape metadata
        tensor_shape = self.create_tensor_shape(
            original_shape,
            self._get_component_category(component_name),
            {'component': component_name}
        )
        
        # Apply transformation based on prime factorization
        transformed_data = tensor_data.copy()
        
        # Apply prime-based scaling
        for i, (dim, factorization) in enumerate(zip(original_shape, tensor_shape.prime_factorizations)):
            if factorization.factors:
                # Scale by prime factor signature
                prime_scaling = sum(factorization.factors) / len(factorization.factors)
                axis_indices = tuple(j for j in range(len(original_shape)) if j != i)
                if axis_indices:
                    transformed_data = transformed_data * (prime_scaling / 10.0)
        
        return transformed_data
    
    def _get_component_category(self, component_name: str) -> TensorCategory:
        """Get tensor category for component"""
        component_mappings = {
            'ggml': TensorCategory.GGML_TENSOR,
            'kokkos': TensorCategory.KOKKOS_TENSOR,
            'a0ml': TensorCategory.GGML_TENSOR,  # a0ml uses ggml
            'mem0': TensorCategory.MEM0_TENSOR,
            'mlpn': TensorCategory.MLPN_TENSOR,
            'node9': TensorCategory.NODE9_TENSOR,
            'atomspace': TensorCategory.ATOMSPACE_TENSOR
        }
        
        for key, category in component_mappings.items():
            if key in component_name.lower():
                return category
        
        return TensorCategory.GESTALT_TENSOR
    
    def merge_component_tensor(self, gestalt_tensor: np.ndarray, 
                             component_tensor: np.ndarray) -> np.ndarray:
        """Merge component tensor into gestalt tensor"""
        # Ensure compatible shapes
        if gestalt_tensor.shape == component_tensor.shape:
            # Direct merge
            return gestalt_tensor + component_tensor * 0.1
        
        # Reshape component tensor to match gestalt
        if component_tensor.size <= gestalt_tensor.size:
            # Pad or reshape component tensor
            reshaped_component = self._reshape_to_match(component_tensor, gestalt_tensor.shape)
            return gestalt_tensor + reshaped_component * 0.1
        else:
            # Downsample component tensor
            downsampled_component = self._downsample_to_match(component_tensor, gestalt_tensor.shape)
            return gestalt_tensor + downsampled_component * 0.1
    
    def _reshape_to_match(self, source: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Reshape source tensor to match target shape"""
        target_size = np.prod(target_shape)
        
        if source.size == target_size:
            return source.reshape(target_shape)
        elif source.size < target_size:
            # Pad with zeros
            padded = np.zeros(target_size)
            padded[:source.size] = source.flatten()
            return padded.reshape(target_shape)
        else:
            # Truncate
            truncated = source.flatten()[:target_size]
            return truncated.reshape(target_shape)
    
    def _downsample_to_match(self, source: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Downsample source tensor to match target shape"""
        # Simple downsampling by averaging
        target_size = np.prod(target_shape)
        source_flat = source.flatten()
        
        # Calculate downsampling ratio
        ratio = len(source_flat) // target_size
        
        if ratio <= 1:
            return self._reshape_to_match(source, target_shape)
        
        # Average over chunks
        downsampled = np.array([
            np.mean(source_flat[i:i+ratio]) 
            for i in range(0, len(source_flat) - ratio + 1, ratio)
        ])
        
        # Ensure correct size
        if len(downsampled) > target_size:
            downsampled = downsampled[:target_size]
        elif len(downsampled) < target_size:
            padded = np.zeros(target_size)
            padded[:len(downsampled)] = downsampled
            downsampled = padded
        
        return downsampled.reshape(target_shape)
    
    def update_gestalt_tensor(self, current_gestalt: np.ndarray, 
                            updated_states: Dict[str, Any]) -> np.ndarray:
        """Update gestalt tensor based on component state changes"""
        updated_gestalt = current_gestalt.copy()
        
        # Process each updated component
        for component_name, component_state in updated_states.items():
            if isinstance(component_state, dict):
                # Extract and merge component contribution
                component_tensor = self.factorize_component_tensor(component_name, component_state)
                updated_gestalt = self.merge_component_tensor(updated_gestalt, component_tensor)
        
        # Apply normalization to prevent explosion
        gestalt_norm = np.linalg.norm(updated_gestalt)
        if gestalt_norm > 10.0:  # Arbitrary threshold
            updated_gestalt = updated_gestalt / (gestalt_norm / 10.0)
        
        return updated_gestalt
    
    def create_gestalt_field(self, component_shapes: Dict[str, TensorShape]) -> GestaltTensorField:
        """Create unified gestalt tensor field"""
        gestalt_shape = self.compute_gestalt_shape(component_shapes)
        gestalt_address = self.prime_calculator.get_prime_signature(gestalt_shape)
        
        # Calculate field energy
        field_energy = self._calculate_field_energy(component_shapes)
        
        # Calculate coherence measure
        coherence_measure = self._calculate_coherence_measure(component_shapes)
        
        # Create prime signature
        prime_signature = self._create_prime_signature(component_shapes)
        
        gestalt_field = GestaltTensorField(
            shape=gestalt_shape,
            component_tensors=component_shapes,
            gestalt_address=gestalt_address,
            field_energy=field_energy,
            coherence_measure=coherence_measure,
            prime_signature=prime_signature,
            created_time=asyncio.get_event_loop().time(),
            update_count=0
        )
        
        # Register gestalt field
        self.gestalt_fields[gestalt_address] = gestalt_field
        
        return gestalt_field
    
    def _calculate_field_energy(self, component_shapes: Dict[str, TensorShape]) -> float:
        """Calculate energy of the gestalt field"""
        total_dof = sum(shape.degrees_of_freedom for shape in component_shapes.values())
        total_complexity = sum(
            len(shape.dimensions) for shape in component_shapes.values()
        )
        
        # Normalize energy
        return min(1.0, (total_dof / 10000.0) + (total_complexity / 100.0))
    
    def _calculate_coherence_measure(self, component_shapes: Dict[str, TensorShape]) -> float:
        """Calculate coherence measure of component shapes"""
        if len(component_shapes) <= 1:
            return 1.0
        
        # Calculate similarity between prime signatures
        signatures = [shape.unique_address for shape in component_shapes.values()]
        
        # Simple coherence based on shared prime factors
        all_primes = set()
        for signature in signatures:
            primes = set(signature.split('_'))
            all_primes.update(primes)
        
        if not all_primes:
            return 0.0
        
        # Calculate overlap
        overlaps = []
        signatures_sets = [set(sig.split('_')) for sig in signatures]
        
        for i, sig1 in enumerate(signatures_sets):
            for sig2 in signatures_sets[i+1:]:
                if sig1 and sig2:
                    overlap = len(sig1 & sig2) / len(sig1 | sig2)
                    overlaps.append(overlap)
        
        return np.mean(overlaps) if overlaps else 0.0
    
    def _create_prime_signature(self, component_shapes: Dict[str, TensorShape]) -> str:
        """Create unified prime signature for gestalt field"""
        all_signatures = [shape.unique_address for shape in component_shapes.values()]
        
        # Combine signatures
        combined_primes = set()
        for signature in all_signatures:
            combined_primes.update(signature.split('_'))
        
        # Sort and join
        return "_".join(sorted(combined_primes))
    
    # Component-specific shape mapping methods
    def _map_ggml_shape(self, component_state: Dict[str, Any]) -> TensorShape:
        """Map GGML component to tensor shape"""
        # GGML typically uses 2D matrices for operations
        default_dims = (64, 32)
        return self.create_tensor_shape(default_dims, TensorCategory.GGML_TENSOR)
    
    def _map_kokkos_shape(self, component_state: Dict[str, Any]) -> TensorShape:
        """Map Kokkos component to tensor shape"""
        # Kokkos uses 1D views for parallel operations
        default_dims = (128,)
        return self.create_tensor_shape(default_dims, TensorCategory.KOKKOS_TENSOR)
    
    def _map_mem0_shape(self, component_state: Dict[str, Any]) -> TensorShape:
        """Map Mem0 component to tensor shape"""
        # Mem0 uses 3D tensors: [episodic, semantic, temporal]
        default_dims = (64, 128, 32)
        return self.create_tensor_shape(default_dims, TensorCategory.MEM0_TENSOR)
    
    def _map_mlpn_shape(self, component_state: Dict[str, Any]) -> TensorShape:
        """Map MLPN component to tensor shape"""
        # MLPN uses 3D tensors: [meta-models, logical layers, probability vectors]
        default_dims = (16, 32, 64)
        return self.create_tensor_shape(default_dims, TensorCategory.MLPN_TENSOR)
    
    def _map_node9_shape(self, component_state: Dict[str, Any]) -> TensorShape:
        """Map Node9 component to tensor shape"""
        # Node9 uses 3D tensors: [hypergraph, grammar, scheme context]
        default_dims = (8, 16, 32)
        return self.create_tensor_shape(default_dims, TensorCategory.NODE9_TENSOR)
    
    def _map_atomspace_shape(self, component_state: Dict[str, Any]) -> TensorShape:
        """Map AtomSpace component to tensor shape"""
        # AtomSpace uses variable shapes based on node/link counts
        default_dims = (100, 200)  # [nodes, links]
        return self.create_tensor_shape(default_dims, TensorCategory.ATOMSPACE_TENSOR)
    
    def get_shape_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered tensor shapes"""
        if not self.registered_shapes:
            return {}
        
        shapes = list(self.registered_shapes.values())
        
        return {
            'total_shapes': len(shapes),
            'complexity_distribution': {
                complexity.value: sum(1 for s in shapes if s.complexity == complexity)
                for complexity in TensorComplexity
            },
            'category_distribution': {
                category.value: sum(1 for s in shapes if s.category == category)
                for category in TensorCategory
            },
            'average_dimensions': np.mean([len(s.dimensions) for s in shapes]),
            'total_degrees_of_freedom': sum(s.degrees_of_freedom for s in shapes),
            'gestalt_fields': len(self.gestalt_fields)
        }