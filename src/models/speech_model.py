"""
Speech analysis models for cognitive decline detection.

This module implements neural network architectures for processing
audio data including speech recognition and emotion analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SpeechModel(BaseModel):
    """Base speech model for audio processing."""
    
    def __init__(
        self,
        input_dim: int = 80,  # MFCC features
        output_dim: int = 4,  # Cognitive decline severity levels
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 100,
        num_mfcc: int = 13
    ):
        """
        Initialize speech model.
        
        Args:
            input_dim: Input dimension (MFCC features)
            output_dim: Output dimension (number of classes)
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            activation: Activation function name
            device: Device to run on
            sequence_length: Length of input sequences
            num_mfcc: Number of MFCC coefficients
        """
        self.sequence_length = sequence_length
        self.num_mfcc = num_mfcc
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build speech model architecture."""
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through speech model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Reshape for processing
        x = x.view(batch_size * seq_len, features)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # Global average pooling over sequence dimension
        features = torch.mean(features, dim=1)
        
        # Classify
        output = self.classifier(features)
        
        return output


class CNNLSTMSpeechModel(BaseModel):
    """CNN-LSTM model for speech analysis."""
    
    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 100,
        num_mfcc: int = 13,
        cnn_filters: List[int] = [32, 64, 128],
        lstm_units: int = 128
    ):
        """
        Initialize CNN-LSTM speech model.
        
        Args:
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.num_mfcc = num_mfcc
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build CNN-LSTM model architecture."""
        # CNN layers for local feature extraction
        self.cnn_layers = nn.ModuleList()
        
        # First CNN layer
        self.cnn_layers.append(nn.Sequential(
            nn.Conv1d(self.input_dim, self.cnn_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(self.cnn_filters[0]),
            self.activation_fn,
            nn.MaxPool1d(kernel_size=2)
        ))
        
        # Additional CNN layers
        for i in range(1, len(self.cnn_filters)):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(self.cnn_filters[i-1], self.cnn_filters[i], kernel_size=3, padding=1),
                nn.BatchNorm1d(self.cnn_filters[i]),
                self.activation_fn,
                nn.MaxPool1d(kernel_size=2)
            ))
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_filters[-1],
            hidden_size=self.lstm_units,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_units * 2, self.hidden_dim),  # *2 for bidirectional
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Transpose for CNN: (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Transpose back for LSTM: (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Classify
        output = self.classifier(lstm_out)
        
        return output


class TransformerSpeechModel(BaseModel):
    """Transformer model for speech analysis."""
    
    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = 'gelu',
        device: str = 'cpu',
        sequence_length: int = 100,
        num_mfcc: int = 13,
        num_heads: int = 8,
        feedforward_dim: int = 1024
    ):
        """
        Initialize Transformer speech model.
        
        Args:
            num_heads: Number of attention heads
            feedforward_dim: Feedforward dimension
        """
        self.sequence_length = sequence_length
        self.num_mfcc = num_mfcc
        self.num_heads = num_heads
        self.feedforward_dim = feedforward_dim
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build Transformer model architecture."""
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.hidden_dim, self.sequence_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, sequence_length)
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # Classify
        output = self.classifier(x)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class Wav2Vec2SpeechModel(BaseModel):
    """Wav2Vec2-based model for speech analysis."""
    
    def __init__(
        self,
        input_dim: int = 16000,  # Raw audio samples
        output_dim: int = 4,
        hidden_dim: int = 768,  # Wav2Vec2 hidden dimension
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        device: str = 'cpu',
        sequence_length: int = 160000,  # 10 seconds at 16kHz
        pretrained: bool = True
    ):
        """
        Initialize Wav2Vec2 speech model.
        
        Args:
            pretrained: Whether to use pretrained Wav2Vec2
        """
        self.sequence_length = sequence_length
        self.pretrained = pretrained
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build Wav2Vec2 model architecture."""
        # Wav2Vec2 feature extractor (simplified)
        if self.pretrained:
            # In practice, load pretrained Wav2Vec2 model
            logger.warning("Pretrained Wav2Vec2 not implemented - using simplified version")
        
        # Simplified Wav2Vec2-like feature extractor
        self.feature_extractor = nn.Sequential(
            # Convolutional feature extraction
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3),
            nn.GELU(),
            nn.LayerNorm(512),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(512),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(512),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(512),
            
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(512),
            
            nn.Conv1d(512, self.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Transformer encoder for contextual representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=12,
            dim_feedforward=3072,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 4, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Wav2Vec2 model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length) or (batch_size, 1, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        batch_size, channels, seq_len = x.shape
        
        # Feature extraction
        features = self.feature_extractor(x)  # (batch_size, hidden_dim, reduced_seq_len)
        
        # Transpose for transformer: (batch_size, reduced_seq_len, hidden_dim)
        features = features.transpose(1, 2)
        
        # Apply transformer encoder
        features = self.transformer_encoder(features)
        
        # Global average pooling
        features = torch.mean(features, dim=1)  # (batch_size, hidden_dim)
        
        # Classify
        output = self.classifier(features)
        
        return output


class AttentionSpeechModel(BaseModel):
    """Attention-based model for speech analysis."""
    
    def __init__(
        self,
        input_dim: int = 80,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 100,
        num_mfcc: int = 13,
        num_heads: int = 8
    ):
        """
        Initialize attention-based speech model.
        
        Args:
            num_heads: Number of attention heads
        """
        self.sequence_length = sequence_length
        self.num_mfcc = num_mfcc
        self.num_heads = num_heads
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build attention-based model architecture."""
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # Feedforward networks
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                self.activation_fn,
                self.dropout_layer,
                nn.Linear(self.hidden_dim * 4, self.hidden_dim)
            )
            for _ in range(self.num_layers)
        ])
        
        # Global attention for final classification
        self.global_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply attention layers
        for i in range(self.num_layers):
            # Multi-head attention
            attn_output, _ = self.attention_layers[i](x, x, x)
            
            # Residual connection and layer norm
            x = self.layer_norms[i](x + attn_output)
            
            # Feedforward network
            ff_output = self.feedforward_layers[i](x)
            
            # Residual connection
            x = x + ff_output
        
        # Global attention for classification
        # Create a learnable query for global attention
        global_query = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, hidden_dim)
        
        global_output, attention_weights = self.global_attention(
            global_query, x, x
        )
        
        # Flatten for classification
        global_output = global_output.squeeze(1)  # (batch_size, hidden_dim)
        
        # Classify
        output = self.classifier(global_output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights from the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights tensor
        """
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply attention layers and collect weights
        attention_weights = []
        
        for i in range(self.num_layers):
            # Multi-head attention
            attn_output, attn_weights = self.attention_layers[i](x, x, x)
            attention_weights.append(attn_weights)
            
            # Residual connection and layer norm
            x = self.layer_norms[i](x + attn_output)
            
            # Feedforward network
            ff_output = self.feedforward_layers[i](x)
            x = x + ff_output
        
        return attention_weights
