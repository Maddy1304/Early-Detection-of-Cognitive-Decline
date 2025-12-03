"""
Base model class for multimodal cognitive decline detection.

This module provides the base class for all neural network models
with common functionality and interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """Base class for all neural network models."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = 'relu',
        device: str = 'cpu'
    ):
        """
        Initialize base model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (number of classes)
            hidden_dim: Hidden dimension size
            num_layers: Number of layers
            dropout: Dropout rate
            activation: Activation function name
            device: Device to run on
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.device = device
        
        # Model components
        self.activation_fn = self._get_activation_function()
        self.dropout_layer = nn.Dropout(dropout)
        
        # Model state
        self.is_training = True
        self.feature_extractor = None
        self.classifier = None
        
        # Initialize model
        self._build_model()
    
    def _get_activation_function(self):
        """Get activation function based on name."""
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),
            'mish': self._mish_activation
        }
        
        return activation_map.get(self.activation.lower(), nn.ReLU())
    
    def _mish_activation(self, x):
        """Mish activation function."""
        return x * torch.tanh(F.softplus(x))
    
    @abstractmethod
    def _build_model(self):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        if self.feature_extractor is not None:
            return self.feature_extractor(x)
        else:
            return x
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify features.
        
        Args:
            features: Feature tensor
            
        Returns:
            Classification output
        """
        if self.classifier is not None:
            return self.classifier(features)
        else:
            return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'activation': self.activation,
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def freeze_parameters(self, freeze: bool = True):
        """Freeze or unfreeze model parameters."""
        for param in self.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"Model parameters {'frozen' if freeze else 'unfrozen'}")
    
    def freeze_layers(self, layer_names: List[str], freeze: bool = True):
        """
        Freeze or unfreeze specific layers.
        
        Args:
            layer_names: List of layer names to freeze/unfreeze
            freeze: Whether to freeze (True) or unfreeze (False)
        """
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                for param in module.parameters():
                    param.requires_grad = not freeze
                
                logger.info(f"Layer {name} {'frozen' if freeze else 'unfrozen'}")
    
    def get_layer_outputs(self, x: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        Get outputs from specific layers.
        
        Args:
            x: Input tensor
            layer_names: List of layer names to extract outputs from
            
        Returns:
            Dictionary of layer outputs
        """
        outputs = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                outputs[name] = output
            return hook
        
        hooks = []
        for name, module in self.named_modules():
            if any(layer_name in name for layer_name in layer_names):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return outputs
    
    def save_model(self, filepath: str, include_optimizer: bool = False, optimizer=None):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            include_optimizer: Whether to include optimizer state
            optimizer: Optimizer to save (if include_optimizer is True)
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'model_config': {
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'activation': self.activation
            }
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu', **kwargs):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            device: Device to load model on
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Get model config
        model_config = checkpoint.get('model_config', {})
        model_config.update(kwargs)
        
        # Create model instance
        model = cls(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Get gradients for all parameters."""
        gradients = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients
    
    def set_gradients(self, gradients: Dict[str, torch.Tensor]):
        """Set gradients for parameters."""
        for name, param in self.named_parameters():
            if name in gradients:
                param.grad = gradients[name].clone()
    
    def compute_gradient_norm(self) -> float:
        """Compute L2 norm of gradients."""
        total_norm = 0.0
        
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** (1. / 2)
    
    def apply_gradient_clipping(self, max_norm: float = 1.0):
        """
        Apply gradient clipping.
        
        Args:
            max_norm: Maximum gradient norm
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
    
    def get_parameter_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for model parameters."""
        stats = {}
        
        for name, param in self.named_parameters():
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'norm': param.data.norm().item()
            }
        
        return stats
    
    def initialize_weights(self, method: str = 'xavier_uniform'):
        """
        Initialize model weights.
        
        Args:
            method: Weight initialization method
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight)
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight)
                elif method == 'normal':
                    nn.init.normal_(module.weight, 0, 0.01)
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
        logger.info(f"Model weights initialized using {method}")
    
    def set_training_mode(self, training: bool = True):
        """Set training mode."""
        self.is_training = training
        self.train(training)
    
    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention weights if the model uses attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights tensor or None
        """
        # Override in subclasses that use attention
        return None
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Embeddings tensor
        """
        return self.extract_features(x)
    
    def predict(self, x: torch.Tensor, return_probabilities: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions tensor, optionally with probabilities
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
            
            if return_probabilities:
                probabilities = F.softmax(logits, dim=-1)
                return predictions, probabilities
            else:
                return predictions
    
    def get_feature_importance(self, x: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """
        Get feature importance using gradient-based methods.
        
        Args:
            x: Input tensor
            target_class: Target class for importance calculation
            
        Returns:
            Feature importance tensor
        """
        x.requires_grad_(True)
        
        # Forward pass
        output = self.forward(x)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=-1)
        
        # Compute gradients
        if output.dim() > 1:
            target_output = output[range(len(output)), target_class]
        else:
            target_output = output
        
        gradients = torch.autograd.grad(
            outputs=target_output,
            inputs=x,
            grad_outputs=torch.ones_like(target_output),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute importance as absolute gradient values
        importance = torch.abs(gradients)
        
        return importance
