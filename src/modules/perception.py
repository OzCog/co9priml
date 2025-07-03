import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
import math

@dataclass
class SensoryInput:
    """Represents different types of sensory inputs"""
    visual: torch.Tensor = None
    auditory: torch.Tensor = None
    proprioceptive: torch.Tensor = None
    text: str = None
    timestamp: float = 0.0  # New: when this input was received
    novelty: float = 0.0    # New: how novel this input is
    importance: float = 1.0  # New: importance weight

@dataclass
class AttentionState:
    """Represents the current state of attention allocation"""
    focus_weights: torch.Tensor
    modality_preferences: Dict[str, float]
    attention_energy: float
    focus_history: List[Tuple[str, float]]  # (modality, strength, timestamp)
    adaptation_rate: float

class AdaptiveAttentionAllocator:
    """Manages dynamic attention allocation across modalities and features"""
    
    def __init__(self, feature_dim: int = 512, num_modalities: int = 4):
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        
        # Attention networks
        self.saliency_detector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.novelty_detector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.relevance_estimator = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),  # Current + context
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Attention state
        self.attention_state = AttentionState(
            focus_weights=torch.ones(feature_dim) / feature_dim,
            modality_preferences={'visual': 0.4, 'auditory': 0.3, 'proprioceptive': 0.2, 'text': 0.1},
            attention_energy=1.0,
            focus_history=[],
            adaptation_rate=0.1
        )
        
        # History tracking
        self.input_history = deque(maxlen=50)
        self.attention_history = deque(maxlen=50)
        self.performance_feedback = deque(maxlen=20)
        
        # Novelty tracking
        self.feature_prototypes = {}  # Store typical features for each modality
        self.prototype_update_rate = 0.05
    
    def allocate_attention(self, sensory_input: SensoryInput, 
                          context: torch.Tensor = None) -> Tuple[torch.Tensor, AttentionState]:
        """Dynamically allocate attention based on input characteristics"""
        
        # Encode all available modalities
        encoded_features = self._encode_all_modalities(sensory_input)
        
        # Calculate saliency for each modality
        modality_saliencies = {}
        for modality, features in encoded_features.items():
            if features is not None:
                saliency = self.saliency_detector(features)
                modality_saliencies[modality] = float(saliency)
        
        # Calculate novelty
        modality_novelties = self._calculate_novelty(encoded_features)
        
        # Calculate relevance (if context provided)
        modality_relevances = {}
        if context is not None:
            for modality, features in encoded_features.items():
                if features is not None:
                    combined = torch.cat([features, context])
                    relevance = self.relevance_estimator(combined)
                    modality_relevances[modality] = float(relevance)
        else:
            # Default relevance based on modality preferences
            modality_relevances = self.attention_state.modality_preferences.copy()
        
        # Combine factors to determine attention allocation
        attention_scores = self._combine_attention_factors(
            modality_saliencies, modality_novelties, modality_relevances
        )
        
        # Apply attention energy constraints
        attention_scores = self._apply_energy_constraints(attention_scores)
        
        # Generate final attention weights
        final_weights = self._generate_attention_weights(encoded_features, attention_scores)
        
        # Update attention state
        self._update_attention_state(attention_scores, encoded_features)
        
        # Apply attention to features
        attended_features = self._apply_attention(encoded_features, final_weights)
        
        return attended_features, self.attention_state
    
    def _encode_all_modalities(self, sensory_input: SensoryInput) -> Dict[str, torch.Tensor]:
        """Encode all available sensory modalities"""
        encoded = {}
        
        if sensory_input.visual is not None:
            # Ensure visual input is the right size
            if sensory_input.visual.numel() != self.feature_dim:
                if sensory_input.visual.numel() > self.feature_dim:
                    encoded['visual'] = sensory_input.visual.flatten()[:self.feature_dim]
                else:
                    padding = torch.zeros(self.feature_dim - sensory_input.visual.numel())
                    encoded['visual'] = torch.cat([sensory_input.visual.flatten(), padding])
            else:
                encoded['visual'] = sensory_input.visual
                
        if sensory_input.auditory is not None:
            # Ensure auditory input is the right size
            if sensory_input.auditory.numel() != self.feature_dim:
                if sensory_input.auditory.numel() > self.feature_dim:
                    encoded['auditory'] = sensory_input.auditory.flatten()[:self.feature_dim]
                else:
                    padding = torch.zeros(self.feature_dim - sensory_input.auditory.numel())
                    encoded['auditory'] = torch.cat([sensory_input.auditory.flatten(), padding])
            else:
                encoded['auditory'] = sensory_input.auditory
                
        if sensory_input.proprioceptive is not None:
            if sensory_input.proprioceptive.numel() != self.feature_dim:
                if sensory_input.proprioceptive.numel() > self.feature_dim:
                    encoded['proprioceptive'] = sensory_input.proprioceptive.flatten()[:self.feature_dim]
                else:
                    padding = torch.zeros(self.feature_dim - sensory_input.proprioceptive.numel())
                    encoded['proprioceptive'] = torch.cat([sensory_input.proprioceptive.flatten(), padding])
            else:
                encoded['proprioceptive'] = sensory_input.proprioceptive
                
        if sensory_input.text is not None:
            # Simple text encoding - in practice would use proper embeddings
            text_tensor = torch.zeros(self.feature_dim)
            text_hash = hash(sensory_input.text) % self.feature_dim
            text_tensor[text_hash] = 1.0
            encoded['text'] = text_tensor
        
        return encoded
    
    def _calculate_novelty(self, encoded_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate novelty for each modality based on feature prototypes"""
        novelties = {}
        
        for modality, features in encoded_features.items():
            if features is None:
                continue
                
            if modality in self.feature_prototypes:
                # Calculate distance from prototype
                prototype = self.feature_prototypes[modality]
                novelty = torch.norm(features - prototype, p=2)
                novelties[modality] = float(novelty)
                
                # Update prototype with exponential moving average
                self.feature_prototypes[modality] = (
                    (1 - self.prototype_update_rate) * prototype +
                    self.prototype_update_rate * features
                )
            else:
                # First time seeing this modality
                novelties[modality] = 1.0
                self.feature_prototypes[modality] = features.clone()
        
        return novelties
    
    def _combine_attention_factors(self, saliencies: Dict[str, float], 
                                 novelties: Dict[str, float],
                                 relevances: Dict[str, float]) -> Dict[str, float]:
        """Combine saliency, novelty, and relevance into attention scores"""
        combined_scores = {}
        
        # Weights for different factors
        saliency_weight = 0.4
        novelty_weight = 0.3
        relevance_weight = 0.3
        
        all_modalities = set(saliencies.keys()) | set(novelties.keys()) | set(relevances.keys())
        
        for modality in all_modalities:
            saliency = saliencies.get(modality, 0.0)
            novelty = novelties.get(modality, 0.0)
            relevance = relevances.get(modality, 0.0)
            
            combined_score = (
                saliency * saliency_weight +
                novelty * novelty_weight +
                relevance * relevance_weight
            )
            
            combined_scores[modality] = combined_score
        
        return combined_scores
    
    def _apply_energy_constraints(self, attention_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply attention energy constraints and fatigue effects"""
        
        # Total available attention energy
        total_energy = self.attention_state.attention_energy
        
        # Apply fatigue based on recent attention history
        if len(self.attention_history) > 5:
            recent_history = list(self.attention_history)[-5:]  # Convert deque to list
            recent_activity = sum(sum(weights.values()) for weights in recent_history)
            fatigue_factor = max(0.3, 1.0 - (recent_activity / 5.0) * 0.3)
            total_energy *= fatigue_factor
        
        # Normalize scores to respect energy constraints
        total_demand = sum(attention_scores.values())
        if total_demand > total_energy:
            scale_factor = total_energy / total_demand
            attention_scores = {k: v * scale_factor for k, v in attention_scores.items()}
        
        # Update attention energy (recovery over time)
        self.attention_state.attention_energy = min(1.0, 
            self.attention_state.attention_energy + 0.1 - sum(attention_scores.values()) * 0.5
        )
        
        return attention_scores
    
    def _generate_attention_weights(self, encoded_features: Dict[str, torch.Tensor],
                                   attention_scores: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Generate final attention weight tensors for each modality"""
        weights = {}
        
        for modality, features in encoded_features.items():
            if features is None:
                continue
                
            score = attention_scores.get(modality, 0.0)
            
            # Create attention weights based on score and feature importance
            if score > 0:
                # Use softmax to create focused attention pattern
                feature_importance = torch.abs(features)
                attention_pattern = torch.softmax(feature_importance * score * 10, dim=0)
                weights[modality] = attention_pattern * score
            else:
                weights[modality] = torch.zeros_like(features)
        
        return weights
    
    def _apply_attention(self, encoded_features: Dict[str, torch.Tensor],
                        attention_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply attention weights and combine modalities"""
        
        attended_modalities = []
        
        for modality, features in encoded_features.items():
            if modality in attention_weights:
                attended_features = features * attention_weights[modality]
                attended_modalities.append(attended_features)
        
        if attended_modalities:
            # Combine attended features from all modalities
            return torch.stack(attended_modalities).sum(dim=0)
        else:
            return torch.zeros(self.feature_dim)
    
    def _update_attention_state(self, attention_scores: Dict[str, float],
                               encoded_features: Dict[str, torch.Tensor]):
        """Update the attention state with current allocation"""
        
        # Update modality preferences based on performance feedback
        if self.performance_feedback:
            recent_performance = np.mean(self.performance_feedback)
            if recent_performance > 0.7:  # Good performance
                # Reinforce current preferences
                for modality, score in attention_scores.items():
                    if modality in self.attention_state.modality_preferences:
                        current_pref = self.attention_state.modality_preferences[modality]
                        self.attention_state.modality_preferences[modality] = (
                            current_pref * 0.9 + score * 0.1
                        )
            else:  # Poor performance
                # Increase exploration
                self.attention_state.adaptation_rate = min(0.3, 
                    self.attention_state.adaptation_rate * 1.1
                )
        
        # Update focus history
        timestamp = len(self.attention_history)
        for modality, score in attention_scores.items():
            self.attention_state.focus_history.append((modality, score, timestamp))
        
        # Keep history limited
        if len(self.attention_state.focus_history) > 100:
            self.attention_state.focus_history = self.attention_state.focus_history[-50:]
        
        # Record attention allocation
        self.attention_history.append(attention_scores.copy())
    
    def provide_feedback(self, performance_score: float):
        """Provide performance feedback to adapt attention allocation"""
        self.performance_feedback.append(performance_score)
    
    def get_attention_report(self) -> Dict[str, Any]:
        """Get a comprehensive report on attention allocation"""
        report = {
            'current_energy': self.attention_state.attention_energy,
            'modality_preferences': self.attention_state.modality_preferences.copy(),
            'adaptation_rate': self.attention_state.adaptation_rate,
            'recent_focus': {}
        }
        
        # Analyze recent focus patterns
        if self.attention_history:
            recent_attention = list(self.attention_history)[-10:]  # Convert deque to list
            for modality in ['visual', 'auditory', 'proprioceptive', 'text']:
                modality_attention = [att.get(modality, 0.0) for att in recent_attention]
                if modality_attention:
                    report['recent_focus'][modality] = {
                        'average': np.mean(modality_attention),
                        'variability': np.std(modality_attention),
                        'trend': np.polyfit(range(len(modality_attention)), 
                                          modality_attention, 1)[0] if len(modality_attention) > 1 else 0
                    }
        
        return report

class SensoryEncoder(nn.Module):
    """Enhanced sensory encoder with cross-modal integration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
        
        # Enhanced visual processing pathway
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.config.get('visual_dim', 784), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(384, 256),
            nn.ReLU()
        )
        
        # Enhanced auditory processing pathway
        self.auditory_encoder = nn.Sequential(
            nn.Linear(self.config.get('audio_dim', 256), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 128),
            nn.ReLU()
        )
        
        # Proprioceptive processing pathway
        self.proprioceptive_encoder = nn.Sequential(
            nn.Linear(self.config.get('proprio_dim', 64), 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU()
        )
        
        # Text processing pathway (simple embedding)
        self.text_encoder = nn.Sequential(
            nn.Linear(512, 256),  # Assume text is pre-embedded to 512
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Cross-modal integration networks
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )
        
        # Enhanced fusion layer with adaptive combination
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 128, 512),  # All modalities combined
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Modality importance predictor
        self.modality_importance = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),  # One score per modality
            nn.Softmax(dim=0)
        )
        
    def forward(self, sensory_input: SensoryInput) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced processing with cross-modal integration"""
        encoded_features = []
        modality_outputs = {}
        
        # Process each modality
        if sensory_input.visual is not None:
            visual_features = self.visual_encoder(sensory_input.visual)
            encoded_features.append(visual_features)
            modality_outputs['visual'] = visual_features
            
        if sensory_input.auditory is not None:
            audio_features = self.auditory_encoder(sensory_input.auditory)
            encoded_features.append(audio_features)
            modality_outputs['auditory'] = audio_features
            
        if sensory_input.proprioceptive is not None:
            proprio_features = self.proprioceptive_encoder(sensory_input.proprioceptive)
            encoded_features.append(proprio_features)
            modality_outputs['proprioceptive'] = proprio_features
            
        if sensory_input.text is not None:
            # Simple text encoding - in practice would use proper embeddings
            text_tensor = torch.zeros(512)
            text_hash = hash(sensory_input.text) % 512
            text_tensor[text_hash] = 1.0
            text_features = self.text_encoder(text_tensor)
            encoded_features.append(text_features)
            modality_outputs['text'] = text_features
        
        # Cross-modal integration if multiple modalities present
        if len(encoded_features) > 1:
            # Ensure all features are the same size for attention (128)
            target_size = 128
            normalized_features = []
            
            for feat in encoded_features:
                if feat.numel() > target_size:
                    # Compress larger features
                    compressed = feat.flatten()[:target_size]
                elif feat.numel() < target_size:
                    # Pad smaller features
                    padding = torch.zeros(target_size - feat.numel())
                    compressed = torch.cat([feat.flatten(), padding])
                else:
                    compressed = feat.flatten()
                normalized_features.append(compressed)
            
            # Apply cross-modal attention
            feature_stack = torch.stack(normalized_features)
            attended_features, _ = self.cross_modal_attention(
                feature_stack.unsqueeze(0),
                feature_stack.unsqueeze(0), 
                feature_stack.unsqueeze(0)
            )
            integrated_features = attended_features.squeeze(0).mean(dim=0)
            
            # Expand back to 512 dimensions
            if integrated_features.numel() < 512:
                padding = torch.zeros(512 - integrated_features.numel())
                integrated_features = torch.cat([integrated_features, padding])
        else:
            integrated_features = encoded_features[0] if encoded_features else torch.zeros(512)
        
        # Adaptive fusion
        if len(encoded_features) >= 2:
            # Ensure all features have the expected sizes
            visual_feat = modality_outputs.get('visual', torch.zeros(256))
            audio_feat = modality_outputs.get('auditory', torch.zeros(128))
            proprio_feat = modality_outputs.get('proprioceptive', torch.zeros(64))
            text_feat = modality_outputs.get('text', torch.zeros(128))
            
            # Resize features if needed
            if visual_feat.numel() != 256:
                visual_feat = torch.cat([visual_feat.flatten(), 
                                       torch.zeros(256 - visual_feat.numel())])[:256]
            if audio_feat.numel() != 128:
                audio_feat = torch.cat([audio_feat.flatten(), 
                                      torch.zeros(128 - audio_feat.numel())])[:128]
            if proprio_feat.numel() != 64:
                proprio_feat = torch.cat([proprio_feat.flatten(), 
                                        torch.zeros(64 - proprio_feat.numel())])[:64]
            if text_feat.numel() != 128:
                text_feat = torch.cat([text_feat.flatten(), 
                                     torch.zeros(128 - text_feat.numel())])[:128]
            
            combined = torch.cat([visual_feat, audio_feat, proprio_feat, text_feat])
            fused_output = self.adaptive_fusion(combined)
        else:
            fused_output = integrated_features
        
        return fused_output, modality_outputs

class PerceptionModule:
    """Enhanced perception module with adaptive attention allocation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Enhanced sensory encoder
        self.encoder = SensoryEncoder(config)
        
        # Adaptive attention allocator
        self.attention_allocator = AdaptiveAttentionAllocator(
            feature_dim=self.config.get('feature_dim', 512),
            num_modalities=4
        )
        
        # Default attention weights (will be dynamically updated)
        self.attention_weights = torch.ones(512) / 512
        
        # Performance tracking for attention adaptation
        self.processing_quality_history = deque(maxlen=20)
        
    def process_input(self, sensory_input: SensoryInput, 
                     context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Enhanced processing with adaptive attention allocation and cross-modal integration
        """
        
        # Calculate input novelty and importance
        sensory_input.novelty = self._calculate_novelty(sensory_input)
        sensory_input.importance = self._calculate_importance(sensory_input)
        
        # Adaptive attention allocation
        attended_features, attention_state = self.attention_allocator.allocate_attention(
            sensory_input, context
        )
        
        # Enhanced encoding with cross-modal integration
        encoded_input, modality_outputs = self.encoder(sensory_input)
        
        # Apply adaptive attention to encoded features
        if encoded_input.numel() == self.attention_weights.numel():
            final_attended_features = encoded_input * self.attention_weights
        else:
            # Resize attention weights to match encoded input
            if encoded_input.numel() > self.attention_weights.numel():
                padding = torch.zeros(encoded_input.numel() - self.attention_weights.numel())
                adjusted_weights = torch.cat([self.attention_weights, padding])
            else:
                adjusted_weights = self.attention_weights[:encoded_input.numel()]
            final_attended_features = encoded_input * adjusted_weights
            self.attention_weights = adjusted_weights
        
        # Calculate processing quality for adaptation
        processing_quality = self._assess_processing_quality(
            final_attended_features, modality_outputs, attention_state
        )
        self.processing_quality_history.append(processing_quality)
        
        # Provide feedback to attention allocator
        self.attention_allocator.provide_feedback(processing_quality)
        
        # Prepare enhanced output information
        processing_info = {
            'attention_state': attention_state,
            'modality_outputs': modality_outputs,
            'processing_quality': processing_quality,
            'novelty': sensory_input.novelty,
            'importance': sensory_input.importance,
            'cross_modal_integration': len(modality_outputs) > 1
        }
        
        return final_attended_features, self.attention_weights, processing_info
    
    def _calculate_novelty(self, sensory_input: SensoryInput) -> float:
        """Calculate novelty of the input"""
        novelty_score = 0.0
        modality_count = 0
        
        # Simple novelty based on feature variance
        if sensory_input.visual is not None:
            visual_novelty = float(torch.std(sensory_input.visual))
            novelty_score += visual_novelty
            modality_count += 1
            
        if sensory_input.auditory is not None:
            audio_novelty = float(torch.std(sensory_input.auditory))
            novelty_score += audio_novelty
            modality_count += 1
            
        if sensory_input.proprioceptive is not None:
            proprio_novelty = float(torch.std(sensory_input.proprioceptive))
            novelty_score += proprio_novelty
            modality_count += 1
            
        if sensory_input.text is not None:
            # Text novelty based on length and character diversity
            text_novelty = min(1.0, len(set(sensory_input.text)) / max(1, len(sensory_input.text)))
            novelty_score += text_novelty
            modality_count += 1
        
        return novelty_score / max(1, modality_count)
    
    def _calculate_importance(self, sensory_input: SensoryInput) -> float:
        """Calculate importance of the input"""
        importance_score = 1.0  # Base importance
        
        # Boost importance based on multiple modalities
        modality_count = sum([
            sensory_input.visual is not None,
            sensory_input.auditory is not None,
            sensory_input.proprioceptive is not None,
            sensory_input.text is not None
        ])
        
        importance_score *= (1 + modality_count * 0.2)
        
        # Boost based on signal strength
        if sensory_input.visual is not None:
            signal_strength = float(torch.norm(sensory_input.visual))
            importance_score *= (1 + signal_strength * 0.1)
            
        return min(2.0, importance_score)  # Cap at 2.0
    
    def _assess_processing_quality(self, attended_features: torch.Tensor,
                                  modality_outputs: Dict[str, torch.Tensor],
                                  attention_state: AttentionState) -> float:
        """Assess the quality of current processing for adaptation"""
        
        quality_score = 0.5  # Base quality
        
        # Quality based on feature coherence
        feature_coherence = 1.0 / (1.0 + float(torch.std(attended_features)))
        quality_score += feature_coherence * 0.3
        
        # Quality based on attention energy utilization
        energy_efficiency = attention_state.attention_energy
        quality_score += energy_efficiency * 0.2
        
        # Quality based on modality integration
        if len(modality_outputs) > 1:
            # Reward successful multi-modal integration
            quality_score += 0.2
        
        # Quality based on recent performance trend
        if len(self.processing_quality_history) > 3:
            recent_trend = np.polyfit(
                range(len(self.processing_quality_history)),
                list(self.processing_quality_history),
                1
            )[0]
            quality_score += recent_trend * 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def reset_attention(self) -> None:
        """Reset attention weights and clear adaptation history"""
        self.attention_weights = torch.ones(512) / 512
        self.processing_quality_history.clear()
        
        # Reset attention allocator state
        self.attention_allocator.attention_state = AttentionState(
            focus_weights=torch.ones(512) / 512,
            modality_preferences={'visual': 0.4, 'auditory': 0.3, 'proprioceptive': 0.2, 'text': 0.1},
            attention_energy=1.0,
            focus_history=[],
            adaptation_rate=0.1
        )
    
    def get_attention_report(self) -> Dict[str, Any]:
        """Get comprehensive attention and processing report"""
        report = self.attention_allocator.get_attention_report()
        
        # Add perception-specific metrics
        if self.processing_quality_history:
            report['processing_quality'] = {
                'current': self.processing_quality_history[-1],
                'average': np.mean(self.processing_quality_history),
                'trend': np.polyfit(range(len(self.processing_quality_history)), 
                                  list(self.processing_quality_history), 1)[0]
                if len(self.processing_quality_history) > 1 else 0.0
            }
        
        report['attention_focus_entropy'] = float(-torch.sum(
            self.attention_weights * torch.log(self.attention_weights + 1e-8)
        ))
        
        return report 