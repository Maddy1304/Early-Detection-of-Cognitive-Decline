"""
Federated learning framework for multimodal cognitive decline detection.

This module provides functionality for:
- Federated learning clients and servers
- Model aggregation algorithms
- Privacy-preserving techniques
- Edge-fog-cloud collaboration
"""

from .client import FederatedClient
from .server import FederatedServer
from .aggregation import ModelAggregator, FedAvg, FedProx, SCAFFOLD
from .privacy import DifferentialPrivacy, SecureAggregation

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "ModelAggregator",
    "FedAvg",
    "FedProx", 
    "SCAFFOLD",
    "DifferentialPrivacy",
    "SecureAggregation",
]
