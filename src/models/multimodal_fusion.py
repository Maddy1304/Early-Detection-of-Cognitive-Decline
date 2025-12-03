"""
Multimodal fusion models for cognitive decline detection.

This module implements various fusion strategies for combining
information from different modalities (speech, gait, facial).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import math
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class MultimodalFusionModel(BaseModel):
    """Base multimodal fusion model."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],  # {'speech': 256, 'gait': 128, 'facial': 512}
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        fusion_method: str = 'concatenation'  # concatenation, attention, cross_modal
    ):
        """
        Initialize multimodal fusion model.
        
        Args:
            input_dims: Dictionary of input dimensions for each modality
            output_dim: Output dimension (number of classes)
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            activation: Activation function name
            device: Device to run on
            fusion_method: Method for fusing modalities
        """
        self.input_dims = input_dims
        self.fusion_method = fusion_method
        self.modalities = list(input_dims.keys())
        
        # Calculate total input dimension based on fusion method
        if fusion_method == 'concatenation':
            total_input_dim = sum(input_dims.values())
        elif fusion_method == 'attention':
            total_input_dim = hidden_dim  # Will be projected to hidden_dim
        elif fusion_method == 'cross_modal':
            total_input_dim = hidden_dim
        else:
            total_input_dim = sum(input_dims.values())
        
        super().__init__(
            total_input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build multimodal fusion model architecture."""
        if self.fusion_method == 'concatenation':
            self._build_concatenation_model()
        elif self.fusion_method == 'attention':
            self._build_attention_model()
        elif self.fusion_method == 'cross_modal':
            self._build_cross_modal_model()
        else:
            self._build_concatenation_model()
    
    def _build_concatenation_model(self):
        """Build concatenation-based fusion model."""
        # Input projections for each modality
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, self.hidden_dim)
            for modality, dim in self.input_dims.items()
        })
        
        # Fusion layers
        total_dim = len(self.modalities) * self.hidden_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_dim, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def _build_attention_model(self):
        """Build attention-based fusion model."""
        # Input projections for each modality
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, self.hidden_dim)
            for modality, dim in self.input_dims.items()
        })
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def _build_cross_modal_model(self):
        """Build cross-modal fusion model."""
        # Input projections for each modality
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, self.hidden_dim)
            for modality, dim in self.input_dims.items()
        })
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.cross_modal_layers.append(
                CrossModalAttentionLayer(self.hidden_dim, self.dropout, self.activation_fn)
            )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multimodal fusion model.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if self.fusion_method == 'concatenation':
            return self._forward_concatenation(x)
        elif self.fusion_method == 'attention':
            return self._forward_attention(x)
        elif self.fusion_method == 'cross_modal':
            return self._forward_cross_modal(x)
        else:
            return self._forward_concatenation(x)
    
    def _forward_concatenation(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for concatenation fusion."""
        # Project each modality to hidden dimension
        projected_features = []
        for modality in self.modalities:
            if modality in x:
                projected = self.modality_projections[modality](x[modality])
                projected_features.append(projected)
        
        # Concatenate features
        fused_features = torch.cat(projected_features, dim=-1)
        
        # Apply fusion layers
        features = self.fusion_layers(fused_features)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def _forward_attention(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for attention fusion."""
        # Project each modality to hidden dimension
        projected_features = []
        for modality in self.modalities:
            if modality in x:
                projected = self.modality_projections[modality](x[modality])
                projected_features.append(projected)
        
        # Stack features for attention
        stacked_features = torch.stack(projected_features, dim=1)  # (batch_size, num_modalities, hidden_dim)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Residual connection and layer norm
        attended_features = self.layer_norm(attended_features + stacked_features)
        
        # Global average pooling over modalities
        fused_features = torch.mean(attended_features, dim=1)
        
        # Apply fusion layers
        features = self.fusion_layers(fused_features)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def _forward_cross_modal(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for cross-modal fusion."""
        # Project each modality to hidden dimension
        projected_features = {}
        for modality in self.modalities:
            if modality in x:
                projected_features[modality] = self.modality_projections[modality](x[modality])
        
        # Apply cross-modal attention layers
        for cross_modal_layer in self.cross_modal_layers:
            projected_features = cross_modal_layer(projected_features)
        
        # Average features across modalities
        fused_features = torch.mean(torch.stack(list(projected_features.values())), dim=0)
        
        # Apply fusion layers
        features = self.fusion_layers(fused_features)
        
        # Classify
        output = self.classifier(features)
        
        return output


class CrossModalAttentionLayer(nn.Module):
    """Cross-modal attention layer."""
    
    def __init__(
        self, 
        hidden_dim: int, 
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize cross-modal attention layer.
        
        Args:
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        # Self-attention for each modality
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cross-modal attention layer.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Updated modality features
        """
        updated_features = {}
        
        for modality, features in modality_features.items():
            # Self-attention
            self_attended, _ = self.self_attention(features, features, features)
            features = self.layer_norm1(features + self_attended)
            
            # Cross-modal attention with other modalities
            other_features = [f for mod, f in modality_features.items() if mod != modality]
            if other_features:
                # Concatenate other modalities
                other_concat = torch.cat(other_features, dim=1)
                
                # Cross-attention
                cross_attended, _ = self.cross_attention(features, other_concat, other_concat)
                features = self.layer_norm2(features + cross_attended)
            
            # Feedforward network
            ff_output = self.feedforward(features)
            features = self.layer_norm3(features + ff_output)
            
            updated_features[modality] = features
        
        return updated_features


class AttentionFusion(nn.Module):
    """Attention-based fusion module."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize attention fusion module.
        
        Args:
            input_dims: Dictionary of input dimensions for each modality
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.modalities = list(input_dims.keys())
        
        # Input projections
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_dim)
            for modality, dim in input_dims.items()
        })
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through attention fusion.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Fused features tensor
        """
        # Project each modality to hidden dimension
        projected_features = []
        for modality in self.modalities:
            if modality in x:
                projected = self.modality_projections[modality](x[modality])
                projected_features.append(projected)
        
        # Stack features for attention
        stacked_features = torch.stack(projected_features, dim=1)  # (batch_size, num_modalities, hidden_dim)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Residual connection and layer norm
        attended_features = self.layer_norm(attended_features + stacked_features)
        
        # Global average pooling over modalities
        fused_features = torch.mean(attended_features, dim=1)
        
        # Output projection
        output = self.output_projection(fused_features)
        
        return output
    
    def get_attention_weights(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get attention weights.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Attention weights tensor
        """
        # Project each modality to hidden dimension
        projected_features = []
        for modality in self.modalities:
            if modality in x:
                projected = self.modality_projections[modality](x[modality])
                projected_features.append(projected)
        
        # Stack features for attention
        stacked_features = torch.stack(projected_features, dim=1)
        
        # Apply attention and get weights
        _, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        return attention_weights


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize cross-modal fusion module.
        
        Args:
            input_dims: Dictionary of input dimensions for each modality
            hidden_dim: Hidden dimension size
            num_layers: Number of cross-modal layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.modalities = list(input_dims.keys())
        
        # Input projections
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, hidden_dim)
            for modality, dim in input_dims.items()
        })
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttentionLayer(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through cross-modal fusion.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Fused features tensor
        """
        # Project each modality to hidden dimension
        projected_features = {}
        for modality in self.modalities:
            if modality in x:
                projected_features[modality] = self.modality_projections[modality](x[modality])
        
        # Apply cross-modal attention layers
        for cross_modal_layer in self.cross_modal_layers:
            projected_features = cross_modal_layer(projected_features)
        
        # Average features across modalities
        fused_features = torch.mean(torch.stack(list(projected_features.values())), dim=0)
        
        # Output projection
        output = self.output_projection(fused_features)
        
        return output


class HierarchicalFusionModel(BaseModel):
    """Hierarchical fusion model for edge-fog-cloud architecture."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        fusion_levels: List[str] = ['edge', 'fog', 'cloud']
    ):
        """
        Initialize hierarchical fusion model.
        
        Args:
            fusion_levels: List of fusion levels (edge, fog, cloud)
        """
        self.input_dims = input_dims
        self.fusion_levels = fusion_levels
        self.modalities = list(input_dims.keys())
        
        super().__init__(
            sum(input_dims.values()), output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build hierarchical fusion model architecture."""
        # Edge-level fusion (local modalities)
        self.edge_fusion = nn.ModuleDict({
            'speech_gait': AttentionFusion(
                {'speech': self.input_dims.get('speech', 256), 
                 'gait': self.input_dims.get('gait', 128)},
                self.hidden_dim // 2
            ),
            'facial': nn.Linear(self.input_dims.get('facial', 512), self.hidden_dim // 2)
        })
        
        # Fog-level fusion (clinic-level aggregation)
        self.fog_fusion = AttentionFusion(
            {'speech_gait': self.hidden_dim // 2, 'facial': self.hidden_dim // 2},
            self.hidden_dim
        )
        
        # Cloud-level fusion (global aggregation)
        self.cloud_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through hierarchical fusion model.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Edge-level fusion
        edge_features = {}
        
        # Fuse speech and gait at edge level
        if 'speech' in x and 'gait' in x:
            edge_features['speech_gait'] = self.edge_fusion['speech_gait']({
                'speech': x['speech'],
                'gait': x['gait']
            })
        
        # Process facial features at edge level
        if 'facial' in x:
            edge_features['facial'] = self.edge_fusion['facial'](x['facial'])
        
        # Fog-level fusion
        fog_features = self.fog_fusion(edge_features)
        
        # Cloud-level fusion
        cloud_features = self.cloud_fusion(fog_features)
        
        # Classify
        output = self.classifier(cloud_features)
        
        return output


class AdaptiveFusionModel(BaseModel):
    """Adaptive fusion model that learns optimal fusion weights."""
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        temperature: float = 0.07
    ):
        """
        Initialize adaptive fusion model.
        
        Args:
            temperature: Temperature for softmax in adaptive weighting
        """
        self.input_dims = input_dims
        self.temperature = temperature
        self.modalities = list(input_dims.keys())
        
        super().__init__(
            sum(input_dims.values()), output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build adaptive fusion model architecture."""
        # Input projections for each modality
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(dim, self.hidden_dim)
            for modality, dim in self.input_dims.items()
        })
        
        # Adaptive weighting network
        self.weighting_network = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.modalities), self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, len(self.modalities))
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through adaptive fusion model.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Project each modality to hidden dimension
        projected_features = []
        for modality in self.modalities:
            if modality in x:
                projected = self.modality_projections[modality](x[modality])
                projected_features.append(projected)
        
        # Concatenate features for weighting network
        concat_features = torch.cat(projected_features, dim=-1)
        
        # Compute adaptive weights
        raw_weights = self.weighting_network(concat_features)
        weights = F.softmax(raw_weights / self.temperature, dim=-1)
        
        # Weighted fusion
        weighted_features = torch.zeros_like(projected_features[0])
        for i, features in enumerate(projected_features):
            weighted_features += weights[:, i:i+1] * features
        
        # Apply fusion layers
        fused_features = self.fusion_layers(weighted_features)
        
        # Classify
        output = self.classifier(fused_features)
        
        return output
    
    def get_fusion_weights(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get fusion weights for each modality.
        
        Args:
            x: Dictionary of input tensors for each modality
            
        Returns:
            Fusion weights tensor
        """
        # Project each modality to hidden dimension
        projected_features = []
        for modality in self.modalities:
            if modality in x:
                projected = self.modality_projections[modality](x[modality])
                projected_features.append(projected)
        
        # Concatenate features for weighting network
        concat_features = torch.cat(projected_features, dim=-1)
        
        # Compute adaptive weights
        raw_weights = self.weighting_network(concat_features)
        weights = F.softmax(raw_weights / self.temperature, dim=-1)
        
        return weights
