"""
Cognitive Decline Detection System.

This package provides a comprehensive federated learning system for the early
detection of cognitive decline using multi-modal data (speech, gait, facial expressions)
with edge-fog-cloud collaboration.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "Early Detection of Cognitive Decline Using Multi-Modal Federated Learning with Edge–Fog Collaboration"

# Import main components
from .data import *
from .models import *
from .federated_learning import *
from .infrastructure import *
from .evaluation import *
from .utils import *

__all__ = [
    # Data modules
    'BaseMultimodalDataset', 'DAICWOZDataset', 'MPowerDataset', 'RAVDESSDataset',
    'AudioProcessor', 'GaitProcessor', 'FacialProcessor',
    
    # Model modules
    'BaseModel', 'SpeechModel', 'GaitModel', 'FacialModel', 'MultimodalFusionModel',
    
    # Federated learning modules
    'FLClient', 'FLServer', 'aggregate_weighted_average',
    
    # Infrastructure modules
    'EdgeDevice', 'Smartphone', 'WearableDevice',
    'FogNode', 'ClinicServer', 'LocalGateway',
    'CloudServer', 'GlobalAggregator', 'AnalyticsServer',
    'NetworkSimulator', 'WiFiNetwork', 'CellularNetwork', 'EthernetNetwork', 'FiberNetwork',
    'SimulationEnvironment',
    
    # Evaluation modules
    'EvaluationMetrics',
    
    # Utility modules
    'setup_logger', 'get_logger', 'PerformanceLogger', 'SecurityLogger',
    'FederatedLearningLogger', 'DataLogger', 'ModelLogger', 'NetworkLogger'
]