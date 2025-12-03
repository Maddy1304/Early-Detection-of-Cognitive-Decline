# API Reference

This document provides a comprehensive API reference for the Cognitive Decline Detection System.

## Table of Contents

1. [Data Processing](#data-processing)
2. [Models](#models)
3. [Federated Learning](#federated-learning)
4. [Infrastructure](#infrastructure)
5. [Evaluation](#evaluation)
6. [Utilities](#utilities)

## Data Processing

### BaseMultimodalDataset

Base class for multimodal datasets.

```python
class BaseMultimodalDataset(Dataset):
    def __init__(self, data_dir: str, processor_config: dict, mode: str = 'train')
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> dict
    def _load_data_paths(self) -> None
    def _initialize_processors(self) -> None
    def _get_label_mapping(self) -> dict
```

### AudioProcessor

Processes audio data for speech analysis.

```python
class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40, max_pad_len: int = 1000)
    def extract_features(self, audio_path: str) -> np.ndarray
    def __call__(self, audio_path: str) -> np.ndarray
```

### GaitProcessor

Processes gait data from accelerometer/gyroscope sensors.

```python
class GaitProcessor:
    def __init__(self, window_size: int = 128, overlap: float = 0.5, sample_rate: int = 100, features: list = None)
    def process_gait_data(self, gait_data_path: str) -> np.ndarray
    def _apply_filter(self, data: np.ndarray, order: int = 4, cutoff_freq: int = 5) -> np.ndarray
    def _extract_window_features(self, window: np.ndarray) -> np.ndarray
    def __call__(self, gait_data_path: str) -> np.ndarray
```

### FacialProcessor

Processes facial video data for expression analysis.

```python
class FacialProcessor:
    def __init__(self, image_size: tuple = (128, 128), num_frames: int = 10, face_detection_model: str = "haarcascade_frontalface_default.xml")
    def process_facial_data(self, video_path: str) -> np.ndarray
    def _detect_and_crop_face(self, frame: np.ndarray) -> np.ndarray
    def __call__(self, video_path: str) -> np.ndarray
```

## Models

### BaseModel

Base class for all neural network models.

```python
class BaseModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def get_output_dim(self) -> int
```

### SpeechModel

Neural network model for speech processing.

```python
class SpeechModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.3, model_type: str = "LSTM")
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### GaitModel

Neural network model for gait analysis.

```python
class GaitModel(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2, model_type: str = "CNN")
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### FacialModel

Neural network model for facial expression analysis.

```python
class FacialModel(BaseModel):
    def __init__(self, input_channels: int, image_size: tuple, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True, model_type: str = "ResNet18")
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### MultimodalFusionModel

Model for fusing multiple modalities.

```python
class MultimodalFusionModel(nn.Module):
    def __init__(self, speech_model: nn.Module, gait_model: nn.Module, facial_model: nn.Module, speech_output_dim: int, gait_output_dim: int, facial_output_dim: int, fusion_hidden_dim: int, output_dim: int, fusion_method: str = "concatenation", dropout: float = 0.4)
    def forward(self, audio_data: torch.Tensor, gait_data: torch.Tensor, facial_data: torch.Tensor) -> torch.Tensor
```

## Federated Learning

### FLClient

Federated learning client for edge devices.

```python
class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, config_path: str, dataset_name: str, data_dir: str)
    def get_parameters(self, config: dict) -> list
    def set_parameters(self, parameters: list) -> None
    def fit(self, parameters: list, config: dict) -> tuple
    def evaluate(self, parameters: list, config: dict) -> tuple
```

### FLServer

Federated learning server for fog nodes and cloud servers.

```python
class FLServer:
    def __init__(self, server_id: str, config_path: str, is_cloud: bool = False, dataset_name: str = None, test_data_dir: str = None)
    def start(self) -> None
    def get_evaluate_fn(self) -> callable
    def fit_config(self, server_round: int) -> dict
    def evaluate_config(self, server_round: int) -> dict
```

### Aggregation Functions

```python
def aggregate_weighted_average(results: List[Tuple[np.ndarray, int]]) -> np.ndarray
```

## Infrastructure

### EdgeDevice

Base class for edge devices.

```python
class EdgeDevice:
    def __init__(self, device_id: str, device_type: str, location: str, battery_level: float = 100.0, processing_power: str = "medium", memory_limit: int = 4096, storage_limit: int = 128000, network_bandwidth: float = 50.0, latency: float = 20.0)
    def collect_data(self) -> dict
    def process_data(self, data: Any) -> Any
    def send_data(self, destination: str, data: Any) -> bool
    def receive_data(self, source: str, data: Any) -> bool
    def get_device_info(self) -> dict
    def connect(self) -> None
    def disconnect(self) -> None
    def shutdown(self) -> None
```

### Smartphone

Smartphone implementation of edge device.

```python
class Smartphone(EdgeDevice):
    def __init__(self, device_id: str, location: str, battery_level: float = 100.0, processing_power: str = "medium", memory_limit: int = 4096, storage_limit: int = 128000, network_bandwidth: float = 50.0, latency: float = 20.0)
    def collect_audio_data(self) -> dict
    def collect_video_data(self) -> dict
    def collect_sensor_data(self) -> dict
```

### WearableDevice

Wearable device implementation of edge device.

```python
class WearableDevice(EdgeDevice):
    def __init__(self, device_id: str, location: str, battery_level: float = 100.0, processing_power: str = "low", memory_limit: int = 1024, storage_limit: int = 8000, network_bandwidth: float = 10.0, latency: float = 50.0)
    def collect_motion_data(self) -> dict
    def collect_physiological_data(self) -> dict
```

### FogNode

Base class for fog nodes.

```python
class FogNode:
    def __init__(self, node_id: str, node_type: str, location: str, capacity: int = 100, processing_power: str = "high", memory_limit: int = 16384, storage_limit: int = 100000, network_bandwidth: float = 100.0, latency: float = 10.0)
    def connect_device(self, device_id: str) -> bool
    def disconnect_device(self, device_id: str) -> None
    def receive_client_update(self, device_id: str, update: dict) -> bool
    def send_global_model(self, device_id: str, model_state: dict) -> bool
    def get_node_info(self) -> dict
    def connect(self) -> None
    def disconnect(self) -> None
    def shutdown(self) -> None
```

### ClinicServer

Clinic server implementation of fog node.

```python
class ClinicServer(FogNode):
    def __init__(self, node_id: str, location: str, capacity: int = 50, processing_power: str = "high", memory_limit: int = 32768, storage_limit: int = 200000, network_bandwidth: float = 200.0, latency: float = 5.0)
    def register_patient(self, patient_id: str, patient_info: dict) -> None
    def get_patient_info(self, patient_id: str) -> dict
    def apply_privacy_filter(self, data: Any, patient_id: str) -> Any
    def store_clinical_model(self, model_id: str, model: nn.Module, patient_id: str) -> None
    def get_clinical_model(self, model_id: str, patient_id: str) -> nn.Module
```

### LocalGateway

Local gateway implementation of fog node.

```python
class LocalGateway(FogNode):
    def __init__(self, node_id: str, location: str, capacity: int = 100, processing_power: str = "medium", memory_limit: int = 8192, storage_limit: int = 50000, network_bandwidth: float = 50.0, latency: float = 20.0)
    def register_device(self, device_id: str, device_info: dict) -> None
    def get_device_info(self, device_id: str) -> dict
    def update_routing_table(self, destination: str, next_hop: str) -> None
    def route_message(self, destination: str, message: Any) -> bool
    def balance_load(self, device_id: str) -> str
```

### CloudServer

Base class for cloud servers.

```python
class CloudServer:
    def __init__(self, server_id: str, server_type: str, location: str, capacity: int = 1000, processing_power: str = "ultra_high", memory_limit: int = 131072, storage_limit: int = 1000000, network_bandwidth: float = 1000.0, latency: float = 5.0)
    def connect_fog_node(self, fog_node_id: str) -> bool
    def disconnect_fog_node(self, fog_node_id: str) -> None
    def receive_fog_update(self, fog_node_id: str, update: dict) -> bool
    def send_global_model(self, fog_node_id: str, model_state: dict) -> bool
    def store_global_model(self, model_id: str, model: nn.Module) -> None
    def get_global_model(self, model_id: str) -> nn.Module
    def get_server_info(self) -> dict
    def connect(self) -> None
    def disconnect(self) -> None
    def shutdown(self) -> None
```

### GlobalAggregator

Global aggregator implementation of cloud server.

```python
class GlobalAggregator(CloudServer):
    def __init__(self, server_id: str, location: str, capacity: int = 1000, processing_power: str = "ultra_high", memory_limit: int = 262144, storage_limit: int = 2000000, network_bandwidth: float = 2000.0, latency: float = 2.0)
    def register_global_model(self, model_id: str, model: nn.Module, config: dict) -> None
    def get_global_model_config(self, model_id: str) -> dict
    def update_global_model(self, model_id: str, updates: dict) -> None
    def configure_federated_learning(self, config: dict) -> None
    def get_federated_learning_config(self) -> dict
    def configure_privacy_settings(self, settings: dict) -> None
    def get_privacy_settings(self) -> dict
```

### AnalyticsServer

Analytics server implementation of cloud server.

```python
class AnalyticsServer(CloudServer):
    def __init__(self, server_id: str, location: str, capacity: int = 500, processing_power: str = "high", memory_limit: int = 65536, storage_limit: int = 500000, network_bandwidth: float = 500.0, latency: float = 10.0)
    def register_analytics_model(self, model_id: str, model: nn.Module, config: dict) -> None
    def run_analytics(self, data: Any, model_id: str) -> dict
    def generate_report(self, report_type: str) -> dict
    def update_dashboard_data(self, data: dict) -> None
    def get_dashboard_data(self) -> dict
```

### NetworkSimulator

Base class for network simulators.

```python
class NetworkSimulator:
    def __init__(self, simulator_id: str, network_type: str, topology: str = "mesh", base_latency: float = 10.0, base_bandwidth: float = 100.0, packet_loss_rate: float = 0.001, jitter: float = 2.0)
    def add_node(self, node_id: str, node_type: str, location: tuple) -> None
    def remove_node(self, node_id: str) -> None
    def add_link(self, node1: str, node2: str, link_properties: dict) -> None
    def remove_link(self, node1: str, node2: str) -> None
    def calculate_path_latency(self, source: str, destination: str) -> float
    def calculate_path_bandwidth(self, source: str, destination: str) -> float
    def simulate_transmission(self, source: str, destination: str, data_size: float) -> dict
    def get_network_topology(self) -> dict
    def get_network_metrics(self) -> dict
    def shutdown(self) -> None
```

### WiFiNetwork

WiFi network simulator.

```python
class WiFiNetwork(NetworkSimulator):
    def __init__(self, simulator_id: str, frequency: str = "2.4GHz", standard: str = "802.11n", signal_strength: float = -50.0, interference_level: float = 0.1)
    def update_signal_strength(self, new_strength: float) -> None
```

### CellularNetwork

Cellular network simulator.

```python
class CellularNetwork(NetworkSimulator):
    def __init__(self, simulator_id: str, generation: str = "4G", carrier: str = "LTE", signal_strength: float = -70.0, congestion_level: float = 0.2)
    def update_congestion_level(self, new_level: float) -> None
```

### EthernetNetwork

Ethernet network simulator.

```python
class EthernetNetwork(NetworkSimulator):
    def __init__(self, simulator_id: str, speed: str = "1Gbps", cable_type: str = "Cat6", distance: float = 100.0)
```

### FiberNetwork

Fiber optic network simulator.

```python
class FiberNetwork(NetworkSimulator):
    def __init__(self, simulator_id: str, speed: str = "10Gbps", fiber_type: str = "Single-mode", distance: float = 1000.0)
```

### SimulationEnvironment

Main simulation environment.

```python
class SimulationEnvironment:
    def __init__(self, config_path: str)
    def start_simulation(self, duration: float = 3600.0) -> None
    def stop_simulation(self) -> None
    def get_simulation_status(self) -> dict
    def save_simulation_data(self, output_path: str) -> None
    def generate_report(self) -> dict
    def shutdown(self) -> None
```

## Evaluation

### EvaluationMetrics

Comprehensive evaluation metrics.

```python
class EvaluationMetrics:
    def __init__(self)
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, class_names: list = None) -> dict
    def calculate_federated_learning_metrics(self, communication_overhead: float, aggregation_time: float, model_accuracy: float, privacy_score: float, energy_consumption: float, latency: float, throughput: float) -> dict
    def calculate_privacy_metrics(self, epsilon: float, delta: float, noise_scale: float, data_sensitivity: float) -> dict
    def calculate_energy_metrics(self, cpu_usage: float, memory_usage: float, network_usage: float, processing_time: float, device_type: str) -> dict
    def calculate_communication_metrics(self, data_size: float, transmission_time: float, latency: float, bandwidth: float, packet_loss: float) -> dict
    def calculate_system_metrics(self, total_time: float, num_devices: int, num_rounds: int, model_size: float, data_processed: float) -> dict
    def compare_models(self, model_results: dict, baseline_model: str = "baseline") -> dict
    def generate_evaluation_report(self, classification_metrics: dict, fl_metrics: dict, privacy_metrics: dict, energy_metrics: dict, communication_metrics: dict, system_metrics: dict) -> dict
    def plot_metrics(self, metrics_history: list, output_path: str, title: str = "Metrics Over Time") -> None
    def save_metrics(self, metrics: dict, output_path: str) -> None
```

## Utilities

### Logging

```python
def setup_logger(level: str = "INFO", log_dir: str = None, log_file: str = None, console_output: bool = True) -> logging.Logger
def get_logger(name: str) -> logging.Logger

class PerformanceLogger:
    def __init__(self, logger: logging.Logger)
    def start_timer(self, operation: str) -> None
    def end_timer(self, operation: str) -> float
    def log_metric(self, metric_name: str, value: float, unit: str = "") -> None
    def log_system_info(self, info: dict) -> None

class SecurityLogger:
    def __init__(self, logger: logging.Logger)
    def log_authentication(self, user_id: str, success: bool, details: str = "") -> None
    def log_authorization(self, user_id: str, resource: str, success: bool) -> None
    def log_data_access(self, user_id: str, data_type: str, operation: str) -> None
    def log_privacy_event(self, event_type: str, details: str) -> None
    def log_security_alert(self, alert_type: str, severity: str, details: str) -> None

class FederatedLearningLogger:
    def __init__(self, logger: logging.Logger)
    def log_round_start(self, round_number: int, total_rounds: int) -> None
    def log_round_end(self, round_number: int, metrics: dict) -> None
    def log_client_update(self, client_id: str, update_size: float, accuracy: float) -> None
    def log_aggregation(self, server_id: str, num_clients: int, aggregation_time: float) -> None
    def log_model_distribution(self, server_id: str, num_clients: int, model_size: float) -> None
    def log_communication(self, source: str, destination: str, data_size: float, latency: float) -> None

class DataLogger:
    def __init__(self, logger: logging.Logger)
    def log_data_loading(self, dataset_name: str, num_samples: int, data_size: float) -> None
    def log_data_preprocessing(self, data_type: str, num_samples: int, processing_time: float) -> None
    def log_data_quality(self, dataset_name: str, quality_score: float, issues: list) -> None
    def log_data_split(self, dataset_name: str, train_size: int, val_size: int, test_size: int) -> None

class ModelLogger:
    def __init__(self, logger: logging.Logger)
    def log_model_creation(self, model_type: str, parameters: dict) -> None
    def log_model_training(self, model_id: str, epoch: int, loss: float, accuracy: float) -> None
    def log_model_evaluation(self, model_id: str, metrics: dict) -> None
    def log_model_save(self, model_id: str, file_path: str, model_size: float) -> None
    def log_model_load(self, model_id: str, file_path: str, load_time: float) -> None

class NetworkLogger:
    def __init__(self, logger: logging.Logger)
    def log_connection(self, node1: str, node2: str, connection_type: str) -> None
    def log_disconnection(self, node1: str, node2: str, reason: str = "") -> None
    def log_network_performance(self, network_id: str, latency: float, bandwidth: float, packet_loss: float) -> None
    def log_network_issue(self, network_id: str, issue_type: str, severity: str, details: str) -> None
```

## Configuration

### Configuration Files

The system uses YAML configuration files for different components:

- `config/edge_config.yaml` - Edge device configuration
- `config/fog_config.yaml` - Fog node configuration
- `config/cloud_config.yaml` - Cloud server configuration
- `config/model_config.yaml` - Model configuration
- `config/simulation_config.yaml` - Simulation configuration

### Example Configuration

```yaml
# Edge device configuration
edge_device:
  id: "edge_client_{}"
  data_path: "data/raw/edge_data/{}"
  processed_data_path: "data/processed/edge_data/{}"
  log_path: "results/logs/edge_client_{}.log"
  
  preprocessing:
    audio:
      sample_rate: 16000
      n_mfcc: 40
      max_pad_len: 1000
    gait:
      window_size: 128
      overlap: 0.5
      features: ["mean", "std", "min", "max", "fft"]
    facial:
      image_size: [128, 128]
      num_frames: 10
      face_detection_model: "haarcascade_frontalface_default.xml"
  
  training:
    epochs: 5
    batch_size: 32
    learning_rate: 0.001
    optimizer: "Adam"
    loss_function: "CrossEntropyLoss"
    model_save_path: "results/models/edge_client_{}_model.pth"
  
  federated_learning:
    server_address: "127.0.0.1:8080"
    num_rounds: 1
  
  device: "cpu"
```

## Usage Examples

### Basic Usage

```python
from src.infrastructure.simulation_environment import SimulationEnvironment
from src.evaluation.metrics import EvaluationMetrics

# Initialize simulation environment
sim_env = SimulationEnvironment("config/simulation_config.yaml")

# Start simulation
sim_env.start_simulation(duration=3600.0)  # 1 hour

# Wait for completion
while sim_env.is_running:
    time.sleep(1.0)

# Get results
results = sim_env.get_simulation_status()
print(f"Simulation completed with accuracy: {results['metrics']['model_accuracy']}")

# Generate report
report = sim_env.generate_report()
sim_env.save_simulation_data("results/simulation_data.json")

# Shutdown
sim_env.shutdown()
```

### Federated Learning Training

```python
from src.federated_learning.client import FLClient
from src.federated_learning.server import FLServer

# Create FL client
client = FLClient(
    client_id="client_0",
    config_path="config/edge_config.yaml",
    dataset_name="daic-woz",
    data_dir="data/raw/daic-woz_client_0"
)

# Create FL server
server = FLServer(
    server_id="server_0",
    config_path="config/cloud_config.yaml",
    is_cloud=True,
    dataset_name="daic-woz",
    test_data_dir="data/raw/daic-woz_test"
)

# Start server
server.start()
```

### Model Evaluation

```python
from src.evaluation.metrics import EvaluationMetrics
import numpy as np

# Initialize evaluator
evaluator = EvaluationMetrics()

# Calculate classification metrics
y_true = np.array([0, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1])
y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.8])

metrics = evaluator.calculate_classification_metrics(
    y_true, y_pred, y_prob, class_names=["healthy", "decline"]
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-score: {metrics['f1_score']:.3f}")
print(f"ROC AUC: {metrics['roc_auc']:.3f}")

# Generate comprehensive report
report = evaluator.generate_evaluation_report(
    classification_metrics=metrics,
    fl_metrics={},
    privacy_metrics={},
    energy_metrics={},
    communication_metrics={},
    system_metrics={}
)

# Save metrics
evaluator.save_metrics(report, "results/evaluation_report.json")
```

### Data Processing

```python
from src.data.preprocessing.audio_processor import AudioProcessor
from src.data.preprocessing.gait_processor import GaitProcessor
from src.data.preprocessing.facial_processor import FacialProcessor

# Initialize processors
audio_processor = AudioProcessor(sample_rate=16000, n_mfcc=40)
gait_processor = GaitProcessor(window_size=128, overlap=0.5)
facial_processor = FacialProcessor(image_size=(128, 128), num_frames=10)

# Process data
audio_features = audio_processor("path/to/audio.wav")
gait_features = gait_processor("path/to/gait_data.csv")
facial_features = facial_processor("path/to/video.mp4")

print(f"Audio features shape: {audio_features.shape}")
print(f"Gait features shape: {gait_features.shape}")
print(f"Facial features shape: {facial_features.shape}")
```

### Network Simulation

```python
from src.infrastructure.network_simulator import WiFiNetwork, CellularNetwork

# Create network simulators
wifi_network = WiFiNetwork(
    simulator_id="wifi_1",
    frequency="2.4GHz",
    standard="802.11n",
    signal_strength=-50.0
)

cellular_network = CellularNetwork(
    simulator_id="cellular_1",
    generation="4G",
    carrier="LTE",
    signal_strength=-70.0
)

# Add nodes
wifi_network.add_node("device_1", "edge_device", (0, 0))
wifi_network.add_node("fog_1", "fog_node", (100, 0))

# Simulate transmission
result = wifi_network.simulate_transmission("device_1", "fog_1", 10.0)  # 10 MB
print(f"Transmission successful: {result['success']}")
print(f"Latency: {result['latency']:.2f} ms")
print(f"Bandwidth: {result['bandwidth']:.2f} Mbps")
```

This API reference provides comprehensive documentation for all major components of the Cognitive Decline Detection System. For more detailed examples and usage patterns, refer to the individual module documentation and the main README file.
