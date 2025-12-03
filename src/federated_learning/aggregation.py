"""
Model aggregation algorithms for federated learning.

This module implements various aggregation algorithms including:
- FedAvg (Federated Averaging)
- FedProx (Federated Proximal)
- SCAFFOLD (Stochastic Controlled Averaging)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)


class ModelAggregator(ABC):
    """Base class for model aggregation algorithms."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize aggregator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
    
    @abstractmethod
    def aggregate(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates from clients.
        
        Args:
            updates: Dictionary of client updates
            weights: Optional weights for each client
            
        Returns:
            Aggregated model updates
        """
        pass
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Equal weights if total is 0
            num_clients = len(weights)
            return {client_id: 1.0 / num_clients for client_id in weights.keys()}
        
        return {client_id: weight / total_weight for client_id, weight in weights.items()}


class FedAvg(ModelAggregator):
    """Federated Averaging (FedAvg) aggregation algorithm."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FedAvg aggregator.
        
        Args:
            config: Configuration parameters
        """
        super().__init__(config)
        self.algorithm_name = "FedAvg"
    
    def aggregate(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates using FedAvg.
        
        Args:
            updates: Dictionary of client updates
            weights: Optional weights for each client (default: equal weights)
            
        Returns:
            Aggregated model updates
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return {}
        
        # Use equal weights if not provided
        if weights is None:
            num_clients = len(updates)
            weights = {client_id: 1.0 / num_clients for client_id in updates.keys()}
        
        # Normalize weights
        weights = self._normalize_weights(weights)
        
        # Get model parameter keys from first client
        first_client = list(updates.keys())[0]
        param_keys = list(updates[first_client].keys())
        
        # Initialize aggregated updates
        aggregated_updates = {}
        for key in param_keys:
            aggregated_updates[key] = torch.zeros_like(updates[first_client][key])
        
        # Weighted average of updates
        for client_id, client_updates in updates.items():
            weight = weights[client_id]
            
            for key in param_keys:
                if key in client_updates:
                    aggregated_updates[key] += weight * client_updates[key]
        
        logger.info(f"FedAvg aggregated updates from {len(updates)} clients")
        return aggregated_updates


class FedProx(ModelAggregator):
    """Federated Proximal (FedProx) aggregation algorithm."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FedProx aggregator.
        
        Args:
            config: Configuration parameters
                - mu: Proximal term weight (default: 0.01)
        """
        super().__init__(config)
        self.algorithm_name = "FedProx"
        self.mu = self.config.get('mu', 0.01)  # Proximal term weight
    
    def aggregate(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates using FedProx.
        
        Args:
            updates: Dictionary of client updates
            weights: Optional weights for each client
            
        Returns:
            Aggregated model updates
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return {}
        
        # Use equal weights if not provided
        if weights is None:
            num_clients = len(updates)
            weights = {client_id: 1.0 / num_clients for client_id in updates.keys()}
        
        # Normalize weights
        weights = self._normalize_weights(weights)
        
        # Get model parameter keys from first client
        first_client = list(updates.keys())[0]
        param_keys = list(updates[first_client].keys())
        
        # Initialize aggregated updates
        aggregated_updates = {}
        for key in param_keys:
            aggregated_updates[key] = torch.zeros_like(updates[first_client][key])
        
        # Weighted average of updates (same as FedAvg for server-side aggregation)
        for client_id, client_updates in updates.items():
            weight = weights[client_id]
            
            for key in param_keys:
                if key in client_updates:
                    aggregated_updates[key] += weight * client_updates[key]
        
        logger.info(f"FedProx aggregated updates from {len(updates)} clients (mu={self.mu})")
        return aggregated_updates


class SCAFFOLD(ModelAggregator):
    """SCAFFOLD (Stochastic Controlled Averaging) aggregation algorithm."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SCAFFOLD aggregator.
        
        Args:
            config: Configuration parameters
                - learning_rate: Learning rate for control variates (default: 1.0)
        """
        super().__init__(config)
        self.algorithm_name = "SCAFFOLD"
        self.learning_rate = self.config.get('learning_rate', 1.0)
        
        # Control variates
        self.server_control_variate = {}
        self.client_control_variates = {}
    
    def aggregate(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates using SCAFFOLD.
        
        Args:
            updates: Dictionary of client updates
            weights: Optional weights for each client
            
        Returns:
            Aggregated model updates
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return {}
        
        # Use equal weights if not provided
        if weights is None:
            num_clients = len(updates)
            weights = {client_id: 1.0 / num_clients for client_id in updates.keys()}
        
        # Normalize weights
        weights = self._normalize_weights(weights)
        
        # Get model parameter keys from first client
        first_client = list(updates.keys())[0]
        param_keys = list(updates[first_client].keys())
        
        # Initialize aggregated updates
        aggregated_updates = {}
        for key in param_keys:
            aggregated_updates[key] = torch.zeros_like(updates[first_client][key])
        
        # SCAFFOLD aggregation with control variates
        for client_id, client_updates in updates.items():
            weight = weights[client_id]
            
            for key in param_keys:
                if key in client_updates:
                    # Apply SCAFFOLD correction
                    if key in self.server_control_variate:
                        # Subtract server control variate and add client control variate
                        corrected_update = client_updates[key] - self.server_control_variate[key]
                        if client_id in self.client_control_variates and key in self.client_control_variates[client_id]:
                            corrected_update += self.client_control_variates[client_id][key]
                        
                        aggregated_updates[key] += weight * corrected_update
                    else:
                        # First round: use regular updates
                        aggregated_updates[key] += weight * client_updates[key]
        
        # Update server control variate
        self._update_server_control_variate(aggregated_updates, weights)
        
        logger.info(f"SCAFFOLD aggregated updates from {len(updates)} clients")
        return aggregated_updates
    
    def _update_server_control_variate(
        self, 
        aggregated_updates: Dict[str, torch.Tensor], 
        weights: Dict[str, float]
    ):
        """Update server control variate."""
        for key in aggregated_updates:
            if key not in self.server_control_variate:
                self.server_control_variate[key] = torch.zeros_like(aggregated_updates[key])
            
            # Update server control variate
            self.server_control_variate[key] += self.learning_rate * aggregated_updates[key]
    
    def set_client_control_variate(self, client_id: str, control_variate: Dict[str, torch.Tensor]):
        """Set client control variate."""
        self.client_control_variates[client_id] = control_variate
    
    def get_server_control_variate(self) -> Dict[str, torch.Tensor]:
        """Get server control variate."""
        return copy.deepcopy(self.server_control_variate)


class AdaptiveAggregator(ModelAggregator):
    """Adaptive aggregation algorithm that adjusts weights based on client performance."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize adaptive aggregator.
        
        Args:
            config: Configuration parameters
                - performance_window: Window size for performance tracking (default: 5)
                - min_weight: Minimum weight for any client (default: 0.01)
                - max_weight: Maximum weight for any client (default: 0.5)
        """
        super().__init__(config)
        self.algorithm_name = "Adaptive"
        self.performance_window = self.config.get('performance_window', 5)
        self.min_weight = self.config.get('min_weight', 0.01)
        self.max_weight = self.config.get('max_weight', 0.5)
        
        # Performance tracking
        self.client_performance = {}
        self.client_weights_history = {}
    
    def aggregate(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates using adaptive weighting.
        
        Args:
            updates: Dictionary of client updates
            weights: Optional weights for each client
            
        Returns:
            Aggregated model updates
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return {}
        
        # Calculate adaptive weights based on performance
        adaptive_weights = self._calculate_adaptive_weights(updates, weights)
        
        # Get model parameter keys from first client
        first_client = list(updates.keys())[0]
        param_keys = list(updates[first_client].keys())
        
        # Initialize aggregated updates
        aggregated_updates = {}
        for key in param_keys:
            aggregated_updates[key] = torch.zeros_like(updates[first_client][key])
        
        # Weighted average of updates
        for client_id, client_updates in updates.items():
            weight = adaptive_weights[client_id]
            
            for key in param_keys:
                if key in client_updates:
                    aggregated_updates[key] += weight * client_updates[key]
        
        logger.info(f"Adaptive aggregated updates from {len(updates)} clients")
        return aggregated_updates
    
    def _calculate_adaptive_weights(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate adaptive weights based on client performance."""
        if base_weights is None:
            num_clients = len(updates)
            base_weights = {client_id: 1.0 / num_clients for client_id in updates.keys()}
        
        # Normalize base weights
        base_weights = self._normalize_weights(base_weights)
        
        # Calculate performance scores (simplified: based on update magnitude)
        performance_scores = {}
        for client_id, client_updates in updates.items():
            total_magnitude = 0.0
            for key, update in client_updates.items():
                total_magnitude += torch.norm(update).item()
            performance_scores[client_id] = total_magnitude
        
        # Normalize performance scores
        total_performance = sum(performance_scores.values())
        if total_performance > 0:
            performance_scores = {
                client_id: score / total_performance 
                for client_id, score in performance_scores.items()
            }
        else:
            # Equal performance if total is 0
            num_clients = len(performance_scores)
            performance_scores = {
                client_id: 1.0 / num_clients 
                for client_id in performance_scores.keys()
            }
        
        # Combine base weights with performance scores
        adaptive_weights = {}
        for client_id in base_weights:
            # Weighted combination of base weight and performance score
            adaptive_weight = 0.7 * base_weights[client_id] + 0.3 * performance_scores[client_id]
            
            # Apply min/max constraints
            adaptive_weight = max(self.min_weight, min(self.max_weight, adaptive_weight))
            
            adaptive_weights[client_id] = adaptive_weight
        
        # Normalize final weights
        adaptive_weights = self._normalize_weights(adaptive_weights)
        
        # Store weights history
        for client_id, weight in adaptive_weights.items():
            if client_id not in self.client_weights_history:
                self.client_weights_history[client_id] = []
            self.client_weights_history[client_id].append(weight)
        
        return adaptive_weights
    
    def get_client_weights_history(self) -> Dict[str, List[float]]:
        """Get client weights history."""
        return copy.deepcopy(self.client_weights_history)


class RobustAggregator(ModelAggregator):
    """Robust aggregation algorithm that handles Byzantine attacks."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize robust aggregator.
        
        Args:
            config: Configuration parameters
                - byzantine_threshold: Fraction of clients that can be Byzantine (default: 0.1)
                - method: Robust aggregation method ('coordinate_wise_median', 'krum', 'trimmed_mean')
        """
        super().__init__(config)
        self.algorithm_name = "Robust"
        self.byzantine_threshold = self.config.get('byzantine_threshold', 0.1)
        self.method = self.config.get('method', 'coordinate_wise_median')
    
    def aggregate(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate model updates using robust aggregation.
        
        Args:
            updates: Dictionary of client updates
            weights: Optional weights for each client
            
        Returns:
            Aggregated model updates
        """
        if not updates:
            logger.warning("No updates to aggregate")
            return {}
        
        # Get model parameter keys from first client
        first_client = list(updates.keys())[0]
        param_keys = list(updates[first_client].keys())
        
        # Initialize aggregated updates
        aggregated_updates = {}
        
        for key in param_keys:
            # Collect updates for this parameter
            param_updates = []
            for client_id, client_updates in updates.items():
                if key in client_updates:
                    param_updates.append(client_updates[key])
            
            if not param_updates:
                continue
            
            # Apply robust aggregation method
            if self.method == 'coordinate_wise_median':
                aggregated_updates[key] = self._coordinate_wise_median(param_updates)
            elif self.method == 'krum':
                aggregated_updates[key] = self._krum(param_updates, updates)
            elif self.method == 'trimmed_mean':
                aggregated_updates[key] = self._trimmed_mean(param_updates)
            else:
                # Fallback to median
                aggregated_updates[key] = self._coordinate_wise_median(param_updates)
        
        logger.info(f"Robust ({self.method}) aggregated updates from {len(updates)} clients")
        return aggregated_updates
    
    def _coordinate_wise_median(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """Compute coordinate-wise median of updates."""
        if len(updates) == 1:
            return updates[0]
        
        # Stack updates
        stacked_updates = torch.stack(updates, dim=0)
        
        # Compute median along client dimension
        median_update = torch.median(stacked_updates, dim=0)[0]
        
        return median_update
    
    def _krum(self, updates: List[torch.Tensor], all_updates: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Krum aggregation algorithm."""
        if len(updates) <= 2:
            return updates[0] if updates else torch.zeros_like(updates[0])
        
        # Calculate distances between all pairs of updates
        distances = []
        for i, update_i in enumerate(updates):
            dists = []
            for j, update_j in enumerate(updates):
                if i != j:
                    dist = torch.norm(update_i - update_j).item()
                    dists.append(dist)
            distances.append(dists)
        
        # Select update with minimum Krum score
        f = max(1, int(len(updates) * self.byzantine_threshold))  # Number of Byzantine clients
        krum_scores = []
        
        for i in range(len(updates)):
            # Sort distances for client i
            sorted_dists = sorted(distances[i])
            # Sum of f+1 smallest distances
            krum_score = sum(sorted_dists[:f+1])
            krum_scores.append(krum_score)
        
        # Select client with minimum Krum score
        best_client_idx = krum_scores.index(min(krum_scores))
        
        return updates[best_client_idx]
    
    def _trimmed_mean(self, updates: List[torch.Tensor]) -> torch.Tensor:
        """Trimmed mean aggregation algorithm."""
        if len(updates) <= 2:
            return updates[0] if updates else torch.zeros_like(updates[0])
        
        # Calculate number of updates to trim
        f = max(1, int(len(updates) * self.byzantine_threshold))
        
        # Stack updates
        stacked_updates = torch.stack(updates, dim=0)
        
        # Sort along client dimension
        sorted_updates, _ = torch.sort(stacked_updates, dim=0)
        
        # Trim f updates from each end
        trimmed_updates = sorted_updates[f:-f] if f > 0 else sorted_updates
        
        # Compute mean of trimmed updates
        mean_update = torch.mean(trimmed_updates, dim=0)
        
        return mean_update
