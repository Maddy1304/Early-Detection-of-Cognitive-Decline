"""
Federated learning client implementation.

This module implements the federated learning client that runs on edge devices,
handling local training and model updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)


class FederatedClient(ABC):
    """Base class for federated learning clients."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        local_epochs: int = 5,
        batch_size: int = 32,
        optimizer: str = 'adam',
        loss_function: str = 'cross_entropy',
        privacy_config: Optional[Dict] = None
    ):
        """
        Initialize federated client.
        
        Args:
            client_id: Unique identifier for the client
            model: Neural network model
            train_data: Training data loader
            val_data: Validation data loader
            device: Device to run on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimization
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            loss_function: Loss function type
            privacy_config: Privacy configuration
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.optimizer_type = optimizer
        self.loss_function_type = loss_function
        self.privacy_config = privacy_config or {}
        
        # Initialize optimizer and loss function
        self.optimizer = self._create_optimizer()
        self.loss_function = self._create_loss_function()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'communication_rounds': 0,
            'total_samples': len(train_data.dataset) if train_data else 0
        }
        
        # Privacy components
        self.privacy_mechanism = None
        if privacy_config.get('differential_privacy', False):
            from .privacy import DifferentialPrivacy
            self.privacy_mechanism = DifferentialPrivacy(
                epsilon=privacy_config.get('epsilon', 1.0),
                delta=privacy_config.get('delta', 1e-5),
                clipping_threshold=privacy_config.get('clipping_threshold', 1.0)
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.loss_function_type.lower() == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.loss_function_type.lower() == 'mse':
            return nn.MSELoss()
        elif self.loss_function_type.lower() == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function_type}")
    
    def train_local(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Train model locally on client data.
        
        Args:
            global_model_state: Global model state dict
            
        Returns:
            Training results including model updates
        """
        logger.info(f"Client {self.client_id}: Starting local training")
        
        # Load global model state
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        # Training metrics
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        # Local training loop
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.loss_function(output, target)
                
                # Backward pass
                loss.backward()
                
                # Apply privacy mechanism if enabled
                if self.privacy_mechanism:
                    self.privacy_mechanism.apply_gradient_clipping(self.model)
                
                # Update parameters
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_samples += data.size(0)
                
                # Calculate accuracy
                if len(output.shape) > 1:  # Classification
                    pred = output.argmax(dim=1, keepdim=True)
                    epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.debug(f"Client {self.client_id}: Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Update total metrics
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct_predictions += epoch_correct
            
            logger.info(f"Client {self.client_id}: Epoch {epoch} completed, Loss: {epoch_loss/len(self.train_data):.4f}")
        
        # Calculate final metrics
        avg_loss = total_loss / (self.local_epochs * len(self.train_data))
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Update training history
        self.training_history['loss'].append(avg_loss)
        self.training_history['accuracy'].append(accuracy)
        self.training_history['communication_rounds'] += 1
        
        # Get model updates
        model_updates = self._get_model_updates(global_model_state)
        
        # Apply privacy mechanism to updates
        if self.privacy_mechanism:
            model_updates = self.privacy_mechanism.add_noise(model_updates)
        
        logger.info(f"Client {self.client_id}: Local training completed, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'client_id': self.client_id,
            'model_updates': model_updates,
            'num_samples': total_samples,
            'loss': avg_loss,
            'accuracy': accuracy,
            'training_time': time.time()
        }
    
    def _get_model_updates(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get model updates (difference between current and global model)."""
        current_state = self.model.state_dict()
        updates = {}
        
        for key in current_state:
            updates[key] = current_state[key] - global_model_state[key]
        
        return updates
    
    def evaluate(self, test_data: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data loader (uses validation data if None)
            
        Returns:
            Evaluation metrics
        """
        if test_data is None:
            test_data = self.val_data
        
        if test_data is None:
            logger.warning(f"Client {self.client_id}: No test data available")
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.loss_function(output, target)
                
                total_loss += loss.item()
                total_samples += data.size(0)
                
                # Calculate accuracy
                if len(output.shape) > 1:  # Classification
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / len(test_data)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_samples
        }
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current model state."""
        return self.model.state_dict()
    
    def set_model_state(self, model_state: Dict[str, torch.Tensor]):
        """Set model state."""
        self.model.load_state_dict(model_state)
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.training_history.copy()
    
    def reset_training_history(self):
        """Reset training history."""
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'communication_rounds': 0,
            'total_samples': len(self.train_data.dataset) if self.train_data else 0
        }
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get client information."""
        return {
            'client_id': self.client_id,
            'device': self.device,
            'learning_rate': self.learning_rate,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer_type,
            'loss_function': self.loss_function_type,
            'total_samples': self.training_history['total_samples'],
            'communication_rounds': self.training_history['communication_rounds'],
            'privacy_enabled': self.privacy_mechanism is not None
        }


class EdgeClient(FederatedClient):
    """Edge device client implementation."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        local_epochs: int = 3,  # Fewer epochs for edge devices
        batch_size: int = 16,   # Smaller batch size for edge devices
        optimizer: str = 'adam',
        loss_function: str = 'cross_entropy',
        privacy_config: Optional[Dict] = None,
        resource_limits: Optional[Dict] = None
    ):
        """
        Initialize edge client.
        
        Args:
            resource_limits: Resource limits for edge device
        """
        super().__init__(
            client_id, model, train_data, val_data, device,
            learning_rate, local_epochs, batch_size, optimizer, loss_function, privacy_config
        )
        
        self.resource_limits = resource_limits or {
            'max_memory': 1024,  # MB
            'max_cpu_usage': 80,  # percentage
            'max_training_time': 300  # seconds
        }
        
        # Resource monitoring
        self.resource_usage = {
            'memory_usage': 0,
            'cpu_usage': 0,
            'training_time': 0
        }
    
    def train_local(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Train model locally with resource constraints."""
        start_time = time.time()
        
        # Check resource constraints
        if not self._check_resource_constraints():
            logger.warning(f"Edge client {self.client_id}: Resource constraints not met, skipping training")
            return {
                'client_id': self.client_id,
                'model_updates': {},
                'num_samples': 0,
                'loss': float('inf'),
                'accuracy': 0.0,
                'training_time': 0,
                'skipped': True
            }
        
        # Perform training
        result = super().train_local(global_model_state)
        
        # Update resource usage
        self.resource_usage['training_time'] = time.time() - start_time
        
        return result
    
    def _check_resource_constraints(self) -> bool:
        """Check if resource constraints are met."""
        import psutil
        
        # Check memory usage
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > self.resource_limits['max_cpu_usage']:
            return False
        
        # Check CPU usage
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > self.resource_limits['max_cpu_usage']:
            return False
        
        return True
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        import psutil
        
        return {
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent(),
            'training_time': self.resource_usage['training_time']
        }


class FogClient(FederatedClient):
    """Fog node client implementation."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        local_epochs: int = 5,
        batch_size: int = 32,
        optimizer: str = 'adam',
        loss_function: str = 'cross_entropy',
        privacy_config: Optional[Dict] = None,
        aggregation_config: Optional[Dict] = None
    ):
        """
        Initialize fog client.
        
        Args:
            aggregation_config: Configuration for local aggregation
        """
        super().__init__(
            client_id, model, train_data, val_data, device,
            learning_rate, local_epochs, batch_size, optimizer, loss_function, privacy_config
        )
        
        self.aggregation_config = aggregation_config or {
            'min_clients': 3,
            'max_clients': 10,
            'aggregation_interval': 600  # seconds
        }
        
        # Local aggregation state
        self.local_updates = []
        self.last_aggregation_time = time.time()
    
    def add_local_update(self, update: Dict[str, torch.Tensor], client_id: str):
        """Add local update from edge client."""
        self.local_updates.append({
            'update': update,
            'client_id': client_id,
            'timestamp': time.time()
        })
    
    def should_aggregate(self) -> bool:
        """Check if local aggregation should be performed."""
        current_time = time.time()
        
        # Check time interval
        if current_time - self.last_aggregation_time < self.aggregation_config['aggregation_interval']:
            return False
        
        # Check minimum number of updates
        if len(self.local_updates) < self.aggregation_config['min_clients']:
            return False
        
        return True
    
    def aggregate_local_updates(self) -> Dict[str, torch.Tensor]:
        """Aggregate local updates from edge clients."""
        if not self.local_updates:
            return {}
        
        # Simple averaging (can be enhanced with more sophisticated methods)
        aggregated_update = {}
        
        for key in self.local_updates[0]['update']:
            aggregated_update[key] = torch.zeros_like(self.local_updates[0]['update'][key])
            
            for update_info in self.local_updates:
                aggregated_update[key] += update_info['update'][key]
            
            aggregated_update[key] /= len(self.local_updates)
        
        # Clear local updates
        self.local_updates = []
        self.last_aggregation_time = time.time()
        
        return aggregated_update
    
    def train_local(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Train model locally with local aggregation."""
        # Check if local aggregation should be performed
        if self.should_aggregate():
            local_aggregated_update = self.aggregate_local_updates()
            
            # Apply local aggregated update to global model
            for key in global_model_state:
                if key in local_aggregated_update:
                    global_model_state[key] += local_aggregated_update[key]
        
        # Perform regular local training
        return super().train_local(global_model_state)
