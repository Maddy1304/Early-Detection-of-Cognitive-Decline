"""
Gait analysis models for cognitive decline detection.

This module implements neural network architectures for processing
motion sensor data including accelerometer and gyroscope readings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class GaitModel(BaseModel):
    """Base gait model for motion sensor data processing."""
    
    def __init__(
        self,
        input_dim: int = 9,  # 3D accelerometer + 3D gyroscope + 3D magnetometer
        output_dim: int = 4,  # Cognitive decline severity levels
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 250,  # 5 seconds at 50Hz
        sampling_rate: int = 50
    ):
        """
        Initialize gait model.
        
        Args:
            input_dim: Input dimension (sensor channels)
            output_dim: Output dimension (number of classes)
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            activation: Activation function name
            device: Device to run on
            sequence_length: Length of input sequences
            sampling_rate: Sampling rate of sensor data
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build gait model architecture."""
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
        Forward pass through gait model.
        
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


class CNNLSTMGaitModel(BaseModel):
    """CNN-LSTM model for gait analysis."""
    
    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 250,
        sampling_rate: int = 50,
        cnn_filters: List[int] = [32, 64, 128],
        lstm_units: int = 128,
        kernel_size: int = 3
    ):
        """
        Initialize CNN-LSTM gait model.
        
        Args:
            cnn_filters: Number of filters for each CNN layer
            lstm_units: Number of LSTM units
            kernel_size: CNN kernel size
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.cnn_filters = cnn_filters
        self.lstm_units = lstm_units
        self.kernel_size = kernel_size
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build CNN-LSTM model architecture."""
        # CNN layers for local feature extraction
        self.cnn_layers = nn.ModuleList()
        
        # First CNN layer
        self.cnn_layers.append(nn.Sequential(
            nn.Conv1d(self.input_dim, self.cnn_filters[0], kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm1d(self.cnn_filters[0]),
            self.activation_fn,
            nn.MaxPool1d(kernel_size=2)
        ))
        
        # Additional CNN layers
        for i in range(1, len(self.cnn_filters)):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(self.cnn_filters[i-1], self.cnn_filters[i], kernel_size=self.kernel_size, padding=1),
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


class TransformerGaitModel(BaseModel):
    """Transformer model for gait analysis."""
    
    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = 'gelu',
        device: str = 'cpu',
        sequence_length: int = 250,
        sampling_rate: int = 50,
        num_heads: int = 8,
        feedforward_dim: int = 1024
    ):
        """
        Initialize Transformer gait model.
        
        Args:
            num_heads: Number of attention heads
            feedforward_dim: Feedforward dimension
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
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


class ResNetGaitModel(BaseModel):
    """ResNet-based model for gait analysis."""
    
    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 250,
        sampling_rate: int = 50,
        num_blocks: int = 3
    ):
        """
        Initialize ResNet gait model.
        
        Args:
            num_blocks: Number of ResNet blocks
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.num_blocks = num_blocks
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build ResNet model architecture."""
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            self.activation_fn,
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet blocks
        self.res_blocks = nn.ModuleList()
        
        # First block
        self.res_blocks.append(self._make_res_block(64, 64, stride=1))
        
        # Additional blocks
        for i in range(1, self.num_blocks):
            self.res_blocks.append(self._make_res_block(64, 64, stride=1))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def _make_res_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a ResNet block."""
        return ResNetBlock(in_channels, out_channels, stride, self.activation_fn, self.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Transpose for CNN: (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Apply ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, 64, 1)
        x = x.squeeze(-1)  # (batch_size, 64)
        
        # Classify
        output = self.classifier(x)
        
        return output


class ResNetBlock(nn.Module):
    """ResNet block for 1D convolutions."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.1
    ):
        """
        Initialize ResNet block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet block."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out


class AttentionGaitModel(BaseModel):
    """Attention-based model for gait analysis."""
    
    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 250,
        sampling_rate: int = 50,
        num_heads: int = 8
    ):
        """
        Initialize attention-based gait model.
        
        Args:
            num_heads: Number of attention heads
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
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


class TemporalConvNetGaitModel(BaseModel):
    """Temporal Convolutional Network for gait analysis."""
    
    def __init__(
        self,
        input_dim: int = 9,
        output_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu',
        sequence_length: int = 250,
        sampling_rate: int = 50,
        num_channels: List[int] = [64, 128, 256],
        kernel_size: int = 3
    ):
        """
        Initialize TCN gait model.
        
        Args:
            num_channels: Number of channels for each TCN layer
            kernel_size: Kernel size for temporal convolutions
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        
        super().__init__(
            input_dim, output_dim, hidden_dim, num_layers, dropout, activation, device
        )
    
    def _build_model(self):
        """Build TCN model architecture."""
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.num_channels[0])
        
        # TCN layers
        self.tcn_layers = nn.ModuleList()
        
        for i in range(len(self.num_channels) - 1):
            self.tcn_layers.append(
                TemporalBlock(
                    self.num_channels[i],
                    self.num_channels[i + 1],
                    self.kernel_size,
                    dropout=self.dropout,
                    activation=self.activation_fn
                )
            )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_channels[-1], self.hidden_dim),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            self.dropout_layer,
            
            nn.Linear(self.hidden_dim // 2, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, features = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Transpose for TCN: (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Classify
        output = self.classifier(x)
        
        return output


class TemporalBlock(nn.Module):
    """Temporal block for TCN."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize temporal block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolution
            dropout: Dropout rate
            activation: Activation function
        """
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.activation(out)
        
        return out
