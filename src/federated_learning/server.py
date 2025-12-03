"""
Federated learning server implementation.

This module implements the federated learning server that coordinates
the training process and aggregates model updates from clients.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import copy
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class FederatedServer(ABC):
    """Base class for federated learning servers."""
    
    def __init__(
        self,
        global_model: nn.Module,
        aggregation_algorithm: str = 'fedavg',
        num_clients: int = 10,
        num_rounds: int = 100,
        client_fraction: float = 1.0,
        min_clients: int = 2,
        device: str = 'cpu',
        aggregation_config: Optional[Dict] = None
    ):
        """
        Initialize federated server.
        
        Args:
            global_model: Global model to be trained
            aggregation_algorithm: Algorithm for model aggregation
            num_clients: Total number of clients
            num_rounds: Number of communication rounds
            client_fraction: Fraction of clients to select per round
            min_clients: Minimum number of clients required
            device: Device to run on
            aggregation_config: Configuration for aggregation algorithm
        """
        self.global_model = global_model.to(device)
        self.aggregation_algorithm = aggregation_algorithm
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.client_fraction = client_fraction
        self.min_clients = min_clients
        self.device = device
        self.aggregation_config = aggregation_config or {}
        
        # Initialize aggregation algorithm
        self.aggregator = self._create_aggregator()
        
        # Training state
        self.current_round = 0
        self.global_model_state = self.global_model.state_dict()
        
        # Training history
        self.training_history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'num_clients': [],
            'communication_time': [],
            'aggregation_time': []
        }
        
        # Client management
        self.clients = {}
        self.client_updates = {}
        self.client_metrics = {}
    
    def _create_aggregator(self):
        """Create aggregation algorithm instance."""
        from .aggregation import ModelAggregator, FedAvg, FedProx, SCAFFOLD
        
        if self.aggregation_algorithm.lower() == 'fedavg':
            return FedAvg(self.aggregation_config)
        elif self.aggregation_algorithm.lower() == 'fedprox':
            return FedProx(self.aggregation_config)
        elif self.aggregation_algorithm.lower() == 'scaffold':
            return SCAFFOLD(self.aggregation_config)
        else:
            logger.warning(f"Unknown aggregation algorithm: {self.aggregation_algorithm}, using FedAvg")
            return FedAvg(self.aggregation_config)
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]):
        """
        Register a client with the server.
        
        Args:
            client_id: Unique client identifier
            client_info: Client information
        """
        self.clients[client_id] = client_info
        logger.info(f"Registered client: {client_id}")
    
    def unregister_client(self, client_id: str):
        """
        Unregister a client from the server.
        
        Args:
            client_id: Client identifier to unregister
        """
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Unregistered client: {client_id}")
    
    def select_clients(self) -> List[str]:
        """
        Select clients for the current round.
        
        Returns:
            List of selected client IDs
        """
        available_clients = list(self.clients.keys())
        
        if len(available_clients) < self.min_clients:
            logger.warning(f"Not enough clients available: {len(available_clients)} < {self.min_clients}")
            return []
        
        # Calculate number of clients to select
        num_selected = max(
            self.min_clients,
            int(len(available_clients) * self.client_fraction)
        )
        
        # Randomly select clients
        selected_clients = np.random.choice(
            available_clients, 
            size=min(num_selected, len(available_clients)), 
            replace=False
        ).tolist()
        
        logger.info(f"Selected {len(selected_clients)} clients for round {self.current_round}")
        return selected_clients
    
    def send_global_model(self, client_id: str) -> Dict[str, torch.Tensor]:
        """
        Send global model to client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Global model state dict
        """
        return copy.deepcopy(self.global_model_state)
    
    def receive_client_update(
        self, 
        client_id: str, 
        update: Dict[str, Any]
    ) -> bool:
        """
        Receive model update from client.
        
        Args:
            client_id: Client identifier
            update: Client update containing model updates and metrics
            
        Returns:
            True if update was received successfully
        """
        try:
            self.client_updates[client_id] = update
            self.client_metrics[client_id] = {
                'loss': update.get('loss', 0.0),
                'accuracy': update.get('accuracy', 0.0),
                'num_samples': update.get('num_samples', 0),
                'training_time': update.get('training_time', 0.0)
            }
            
            logger.info(f"Received update from client {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error receiving update from client {client_id}: {e}")
            return False
    
    def aggregate_updates(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates to update global model.
        
        Returns:
            Aggregated model updates
        """
        if not self.client_updates:
            logger.warning("No client updates to aggregate")
            return {}
        
        start_time = time.time()
        
        # Prepare updates for aggregation
        updates = {}
        weights = {}
        
        for client_id, update in self.client_updates.items():
            if 'model_updates' in update and update['model_updates']:
                updates[client_id] = update['model_updates']
                weights[client_id] = update.get('num_samples', 1)
        
        if not updates:
            logger.warning("No valid model updates to aggregate")
            return {}
        
        # Perform aggregation
        aggregated_update = self.aggregator.aggregate(updates, weights)
        
        aggregation_time = time.time() - start_time
        
        # Update training history
        self.training_history['aggregation_time'].append(aggregation_time)
        
        logger.info(f"Aggregated updates from {len(updates)} clients in {aggregation_time:.2f}s")
        
        return aggregated_update
    
    def update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated updates.
        
        Args:
            aggregated_update: Aggregated model updates
        """
        if not aggregated_update:
            logger.warning("No aggregated updates to apply")
            return
        
        # Update global model state
        for key in self.global_model_state:
            if key in aggregated_update:
                self.global_model_state[key] += aggregated_update[key]
        
        # Load updated state into model
        self.global_model.load_state_dict(self.global_model_state)
        
        logger.info("Global model updated successfully")
    
    def evaluate_global_model(self, test_data) -> Dict[str, float]:
        """
        Evaluate global model on test data.
        
        Args:
            test_data: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Use appropriate loss function
        loss_function = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.global_model(data)
                loss = loss_function(output, target)
                
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
    
    def run_federated_training(
        self, 
        test_data=None,
        save_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run federated training process.
        
        Args:
            test_data: Test data for evaluation
            save_model_path: Path to save the final model
            
        Returns:
            Training results and history
        """
        logger.info(f"Starting federated training for {self.num_rounds} rounds")
        
        start_time = time.time()
        
        for round_num in range(self.num_rounds):
            self.current_round = round_num
            round_start_time = time.time()
            
            logger.info(f"Starting round {round_num + 1}/{self.num_rounds}")
            
            # Select clients for this round
            selected_clients = self.select_clients()
            if not selected_clients:
                logger.warning(f"No clients selected for round {round_num + 1}, skipping")
                continue
            
            # Send global model to selected clients
            for client_id in selected_clients:
                global_model_state = self.send_global_model(client_id)
                # In a real implementation, this would be sent over network
            
            # Simulate client training (in real implementation, clients would train locally)
            # For now, we'll skip the actual client training simulation
            
            # Clear previous round updates
            self.client_updates = {}
            self.client_metrics = {}
            
            # Aggregate updates (in real implementation, this would happen after receiving updates)
            aggregated_update = self.aggregate_updates()
            
            # Update global model
            self.update_global_model(aggregated_update)
            
            # Evaluate global model
            if test_data:
                eval_metrics = self.evaluate_global_model(test_data)
                logger.info(f"Round {round_num + 1} - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
                
                # Update training history
                self.training_history['round'].append(round_num + 1)
                self.training_history['loss'].append(eval_metrics['loss'])
                self.training_history['accuracy'].append(eval_metrics['accuracy'])
                self.training_history['num_clients'].append(len(selected_clients))
            
            round_time = time.time() - round_start_time
            self.training_history['communication_time'].append(round_time)
            
            logger.info(f"Round {round_num + 1} completed in {round_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Save final model
        if save_model_path:
            torch.save(self.global_model.state_dict(), save_model_path)
            logger.info(f"Final model saved to {save_model_path}")
        
        logger.info(f"Federated training completed in {total_time:.2f}s")
        
        return {
            'training_history': self.training_history,
            'final_model_state': self.global_model.state_dict(),
            'total_time': total_time,
            'total_rounds': self.num_rounds
        }
    
    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state."""
        return copy.deepcopy(self.global_model_state)
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history."""
        return self.training_history.copy()
    
    def get_client_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get client metrics."""
        return self.client_metrics.copy()
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            'aggregation_algorithm': self.aggregation_algorithm,
            'num_clients': self.num_clients,
            'num_rounds': self.num_rounds,
            'client_fraction': self.client_fraction,
            'min_clients': self.min_clients,
            'device': self.device,
            'current_round': self.current_round,
            'registered_clients': len(self.clients)
        }


class CloudServer(FederatedServer):
    """Cloud server implementation for global aggregation."""
    
    def __init__(
        self,
        global_model: nn.Module,
        aggregation_algorithm: str = 'fedavg',
        num_fog_nodes: int = 5,
        num_rounds: int = 100,
        fog_fraction: float = 1.0,
        min_fog_nodes: int = 2,
        device: str = 'cpu',
        aggregation_config: Optional[Dict] = None
    ):
        """
        Initialize cloud server.
        
        Args:
            num_fog_nodes: Number of fog nodes
            fog_fraction: Fraction of fog nodes to select per round
            min_fog_nodes: Minimum number of fog nodes required
        """
        super().__init__(
            global_model, aggregation_algorithm, num_fog_nodes, num_rounds,
            fog_fraction, min_fog_nodes, device, aggregation_config
        )
        
        self.num_fog_nodes = num_fog_nodes
        self.fog_nodes = {}
        self.fog_updates = {}
    
    def register_fog_node(self, fog_id: str, fog_info: Dict[str, Any]):
        """
        Register a fog node with the cloud server.
        
        Args:
            fog_id: Unique fog node identifier
            fog_info: Fog node information
        """
        self.fog_nodes[fog_id] = fog_info
        logger.info(f"Registered fog node: {fog_id}")
    
    def select_fog_nodes(self) -> List[str]:
        """
        Select fog nodes for the current round.
        
        Returns:
            List of selected fog node IDs
        """
        available_fog_nodes = list(self.fog_nodes.keys())
        
        if len(available_fog_nodes) < self.min_clients:  # min_fog_nodes
            logger.warning(f"Not enough fog nodes available: {len(available_fog_nodes)} < {self.min_clients}")
            return []
        
        # Calculate number of fog nodes to select
        num_selected = max(
            self.min_clients,
            int(len(available_fog_nodes) * self.client_fraction)  # fog_fraction
        )
        
        # Randomly select fog nodes
        selected_fog_nodes = np.random.choice(
            available_fog_nodes,
            size=min(num_selected, len(available_fog_nodes)),
            replace=False
        ).tolist()
        
        logger.info(f"Selected {len(selected_fog_nodes)} fog nodes for round {self.current_round}")
        return selected_fog_nodes
    
    def receive_fog_update(self, fog_id: str, update: Dict[str, Any]) -> bool:
        """
        Receive aggregated update from fog node.
        
        Args:
            fog_id: Fog node identifier
            update: Fog node update containing aggregated model updates
            
        Returns:
            True if update was received successfully
        """
        try:
            self.fog_updates[fog_id] = update
            logger.info(f"Received aggregated update from fog node {fog_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error receiving update from fog node {fog_id}: {e}")
            return False
    
    def aggregate_fog_updates(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates from fog nodes.
        
        Returns:
            Aggregated model updates
        """
        if not self.fog_updates:
            logger.warning("No fog node updates to aggregate")
            return {}
        
        start_time = time.time()
        
        # Prepare updates for aggregation
        updates = {}
        weights = {}
        
        for fog_id, update in self.fog_updates.items():
            if 'model_updates' in update and update['model_updates']:
                updates[fog_id] = update['model_updates']
                weights[fog_id] = update.get('num_samples', 1)
        
        if not updates:
            logger.warning("No valid model updates to aggregate")
            return {}
        
        # Perform aggregation
        aggregated_update = self.aggregator.aggregate(updates, weights)
        
        aggregation_time = time.time() - start_time
        
        # Update training history
        self.training_history['aggregation_time'].append(aggregation_time)
        
        logger.info(f"Aggregated updates from {len(updates)} fog nodes in {aggregation_time:.2f}s")
        
        return aggregated_update
    
    def run_cloud_training(
        self,
        test_data=None,
        save_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run cloud-level federated training process.
        
        Args:
            test_data: Test data for evaluation
            save_model_path: Path to save the final model
            
        Returns:
            Training results and history
        """
        logger.info(f"Starting cloud federated training for {self.num_rounds} rounds")
        
        start_time = time.time()
        
        for round_num in range(self.num_rounds):
            self.current_round = round_num
            round_start_time = time.time()
            
            logger.info(f"Starting cloud round {round_num + 1}/{self.num_rounds}")
            
            # Select fog nodes for this round
            selected_fog_nodes = self.select_fog_nodes()
            if not selected_fog_nodes:
                logger.warning(f"No fog nodes selected for round {round_num + 1}, skipping")
                continue
            
            # Send global model to selected fog nodes
            for fog_id in selected_fog_nodes:
                global_model_state = self.send_global_model(fog_id)
                # In a real implementation, this would be sent over network
            
            # Clear previous round updates
            self.fog_updates = {}
            
            # Aggregate fog node updates
            aggregated_update = self.aggregate_fog_updates()
            
            # Update global model
            self.update_global_model(aggregated_update)
            
            # Evaluate global model
            if test_data:
                eval_metrics = self.evaluate_global_model(test_data)
                logger.info(f"Cloud round {round_num + 1} - Loss: {eval_metrics['loss']:.4f}, Accuracy: {eval_metrics['accuracy']:.4f}")
                
                # Update training history
                self.training_history['round'].append(round_num + 1)
                self.training_history['loss'].append(eval_metrics['loss'])
                self.training_history['accuracy'].append(eval_metrics['accuracy'])
                self.training_history['num_clients'].append(len(selected_fog_nodes))
            
            round_time = time.time() - round_start_time
            self.training_history['communication_time'].append(round_time)
            
            logger.info(f"Cloud round {round_num + 1} completed in {round_time:.2f}s")
        
        total_time = time.time() - start_time
        
        # Save final model
        if save_model_path:
            torch.save(self.global_model.state_dict(), save_model_path)
            logger.info(f"Final cloud model saved to {save_model_path}")
        
        logger.info(f"Cloud federated training completed in {total_time:.2f}s")
        
        return {
            'training_history': self.training_history,
            'final_model_state': self.global_model.state_dict(),
            'total_time': total_time,
            'total_rounds': self.num_rounds
        }
