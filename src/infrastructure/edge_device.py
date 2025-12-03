"""
Edge device simulation for cognitive decline detection.

This module implements various edge devices including smartphones,
wearables, and IoT sensors for local data processing and federated learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
import threading
import queue
from abc import ABC, abstractmethod
import psutil
import random

logger = logging.getLogger(__name__)


class EdgeDevice(ABC):
    """Base class for edge devices."""
    
    def __init__(
        self,
        device_id: str,
        device_type: str,
        location: str,
        processing_power: str = "medium",
        memory_limit: int = 4096,  # MB
        storage_limit: int = 8192,  # MB
        battery_level: int = 100,
        network_bandwidth: float = 10.0,  # Mbps
        latency: float = 50.0  # ms
    ):
        """
        Initialize edge device.
        
        Args:
            device_id: Unique device identifier
            device_type: Type of device (smartphone, wearable, iot_sensor)
            location: Device location
            processing_power: Processing power level (low, medium, high)
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            battery_level: Battery level (0-100)
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        self.device_id = device_id
        self.device_type = device_type
        self.location = location
        self.processing_power = processing_power
        self.memory_limit = memory_limit
        self.storage_limit = storage_limit
        self.battery_level = battery_level
        self.network_bandwidth = network_bandwidth
        self.latency = latency
        
        # Device state
        self.is_active = True
        self.is_connected = False
        self.current_task = None
        self.task_queue = queue.Queue()
        
        # Resource monitoring
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.storage_usage = 0.0
        self.network_usage = 0.0
        
        # Performance metrics
        self.processing_time = 0.0
        self.communication_time = 0.0
        self.energy_consumption = 0.0
        
        # Data storage
        self.local_data = {}
        self.model_cache = {}
        self.feature_cache = {}
        
        # Threading
        self.processing_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Initialize device
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize device-specific components."""
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Initialized {self.device_type} device: {self.device_id}")
    
    def _monitor_resources(self):
        """Monitor device resources."""
        while not self.stop_event.is_set():
            try:
                # Monitor CPU usage
                self.cpu_usage = psutil.cpu_percent()
                
                # Monitor memory usage
                memory_info = psutil.virtual_memory()
                self.memory_usage = memory_info.percent
                
                # Monitor storage usage
                disk_info = psutil.disk_usage('/')
                self.storage_usage = (disk_info.used / disk_info.total) * 100
                
                # Check resource limits
                if self.memory_usage > 90:
                    logger.warning(f"Device {self.device_id}: High memory usage: {self.memory_usage:.1f}%")
                
                if self.cpu_usage > 90:
                    logger.warning(f"Device {self.device_id}: High CPU usage: {self.cpu_usage:.1f}%")
                
                # Update battery level (simulate consumption)
                if self.is_active:
                    self.battery_level = max(0, self.battery_level - 0.01)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error monitoring resources for device {self.device_id}: {e}")
                time.sleep(5.0)
    
    @abstractmethod
    def process_data(self, data: Any) -> Any:
        """
        Process data locally on the device.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        pass
    
    @abstractmethod
    def collect_data(self) -> Any:
        """
        Collect data from device sensors.
        
        Returns:
            Collected sensor data
        """
        pass
    
    def train_model(self, model: nn.Module, data: Any, epochs: int = 5) -> Dict[str, Any]:
        """
        Train model locally on the device.
        
        Args:
            model: Model to train
            data: Training data
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        start_time = time.time()
        
        # Check if device can handle training
        if not self._can_handle_training():
            logger.warning(f"Device {self.device_id}: Insufficient resources for training")
            return {'success': False, 'reason': 'insufficient_resources'}
        
        try:
            # Simulate training process
            model.train()
            
            # Simulate training time based on processing power
            training_time = self._calculate_training_time(epochs)
            time.sleep(training_time)
            
            # Update energy consumption
            self.energy_consumption += training_time * 0.1  # Simulate energy consumption
            
            # Update processing time
            self.processing_time += training_time
            
            # Update battery level
            self.battery_level = max(0, self.battery_level - training_time * 0.5)
            
            logger.info(f"Device {self.device_id}: Training completed in {training_time:.2f}s")
            
            return {
                'success': True,
                'training_time': training_time,
                'energy_consumption': training_time * 0.1,
                'battery_level': self.battery_level
            }
            
        except Exception as e:
            logger.error(f"Device {self.device_id}: Training failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _can_handle_training(self) -> bool:
        """Check if device can handle training."""
        return (
            self.memory_usage < 80 and
            self.cpu_usage < 80 and
            self.battery_level > 20 and
            self.is_active
        )
    
    def _calculate_training_time(self, epochs: int) -> float:
        """Calculate training time based on device capabilities."""
        base_time = epochs * 2.0  # Base time per epoch
        
        # Adjust based on processing power
        if self.processing_power == "low":
            multiplier = 2.0
        elif self.processing_power == "medium":
            multiplier = 1.0
        else:  # high
            multiplier = 0.5
        
        return base_time * multiplier
    
    def communicate(self, data: Any, destination: str) -> bool:
        """
        Communicate with other devices or servers.
        
        Args:
            data: Data to send
            destination: Destination identifier
            
        Returns:
            True if communication successful
        """
        if not self.is_connected:
            logger.warning(f"Device {self.device_id}: Not connected to network")
            return False
        
        start_time = time.time()
        
        try:
            # Simulate communication delay
            communication_delay = self.latency / 1000.0  # Convert to seconds
            time.sleep(communication_delay)
            
            # Update communication time
            self.communication_time += communication_delay
            
            # Update network usage
            data_size = self._estimate_data_size(data)
            self.network_usage += data_size / self.network_bandwidth
            
            # Update energy consumption
            self.energy_consumption += communication_delay * 0.05
            
            logger.info(f"Device {self.device_id}: Communication to {destination} completed")
            return True
            
        except Exception as e:
            logger.error(f"Device {self.device_id}: Communication failed: {e}")
            return False
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate data size in MB."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size() / (1024 * 1024)
        elif isinstance(data, dict):
            return sum(self._estimate_data_size(v) for v in data.values())
        else:
            return 0.1  # Default estimate
    
    def store_data(self, key: str, data: Any):
        """Store data locally on the device."""
        if self.storage_usage < 90:  # Check storage limit
            self.local_data[key] = data
            logger.debug(f"Device {self.device_id}: Stored data with key {key}")
        else:
            logger.warning(f"Device {self.device_id}: Storage limit reached")
    
    def retrieve_data(self, key: str) -> Optional[Any]:
        """Retrieve data from local storage."""
        return self.local_data.get(key)
    
    def cache_model(self, model_id: str, model: nn.Module):
        """Cache model for faster access."""
        self.model_cache[model_id] = model
        logger.debug(f"Device {self.device_id}: Cached model {model_id}")
    
    def get_cached_model(self, model_id: str) -> Optional[nn.Module]:
        """Get cached model."""
        return self.model_cache.get(model_id)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'location': self.location,
            'processing_power': self.processing_power,
            'memory_limit': self.memory_limit,
            'storage_limit': self.storage_limit,
            'battery_level': self.battery_level,
            'network_bandwidth': self.network_bandwidth,
            'latency': self.latency,
            'is_active': self.is_active,
            'is_connected': self.is_connected,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'storage_usage': self.storage_usage,
            'network_usage': self.network_usage,
            'processing_time': self.processing_time,
            'communication_time': self.communication_time,
            'energy_consumption': self.energy_consumption
        }
    
    def connect(self):
        """Connect device to network."""
        self.is_connected = True
        logger.info(f"Device {self.device_id}: Connected to network")
    
    def disconnect(self):
        """Disconnect device from network."""
        self.is_connected = False
        logger.info(f"Device {self.device_id}: Disconnected from network")
    
    def shutdown(self):
        """Shutdown device."""
        self.is_active = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info(f"Device {self.device_id}: Shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.shutdown()


class SmartphoneDevice(EdgeDevice):
    """Smartphone device implementation."""
    
    def __init__(
        self,
        device_id: str,
        location: str,
        processing_power: str = "high",
        memory_limit: int = 8192,
        storage_limit: int = 16384,
        battery_level: int = 100,
        network_bandwidth: float = 50.0,
        latency: float = 30.0
    ):
        """
        Initialize smartphone device.
        
        Args:
            device_id: Unique device identifier
            location: Device location
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            battery_level: Battery level (0-100)
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            device_id, "smartphone", location, processing_power,
            memory_limit, storage_limit, battery_level, network_bandwidth, latency
        )
        
        # Smartphone-specific capabilities
        self.camera_available = True
        self.microphone_available = True
        self.accelerometer_available = True
        self.gyroscope_available = True
        self.gps_available = True
        
        # Data collection settings
        self.audio_sample_rate = 16000
        self.video_fps = 30
        self.sensor_sampling_rate = 50
    
    def collect_data(self) -> Dict[str, Any]:
        """Collect data from smartphone sensors."""
        data = {}
        
        if self.microphone_available:
            # Simulate audio data collection
            data['audio'] = self._collect_audio_data()
        
        if self.camera_available:
            # Simulate video data collection
            data['video'] = self._collect_video_data()
        
        if self.accelerometer_available and self.gyroscope_available:
            # Simulate motion sensor data collection
            data['motion'] = self._collect_motion_data()
        
        if self.gps_available:
            # Simulate location data
            data['location'] = self._collect_location_data()
        
        return data
    
    def _collect_audio_data(self) -> np.ndarray:
        """Collect audio data from microphone."""
        # Simulate 1 second of audio data
        duration = 1.0
        samples = int(duration * self.audio_sample_rate)
        audio_data = np.random.randn(samples).astype(np.float32)
        return audio_data
    
    def _collect_video_data(self) -> np.ndarray:
        """Collect video data from camera."""
        # Simulate 1 second of video data
        frames = self.video_fps
        height, width = 224, 224
        video_data = np.random.randint(0, 255, (frames, height, width, 3), dtype=np.uint8)
        return video_data
    
    def _collect_motion_data(self) -> np.ndarray:
        """Collect motion sensor data."""
        # Simulate 1 second of motion data
        duration = 1.0
        samples = int(duration * self.sensor_sampling_rate)
        motion_data = np.random.randn(samples, 6).astype(np.float32)  # 3D accel + 3D gyro
        return motion_data
    
    def _collect_location_data(self) -> Dict[str, float]:
        """Collect location data from GPS."""
        return {
            'latitude': random.uniform(-90, 90),
            'longitude': random.uniform(-180, 180),
            'altitude': random.uniform(0, 1000),
            'accuracy': random.uniform(1, 10)
        }
    
    def process_data(self, data: Any) -> Any:
        """Process data on smartphone."""
        if isinstance(data, dict):
            processed_data = {}
            
            if 'audio' in data:
                processed_data['audio'] = self._process_audio_data(data['audio'])
            
            if 'video' in data:
                processed_data['video'] = self._process_video_data(data['video'])
            
            if 'motion' in data:
                processed_data['motion'] = self._process_motion_data(data['motion'])
            
            return processed_data
        else:
            return data
    
    def _process_audio_data(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio data."""
        # Simulate audio processing (e.g., feature extraction)
        # In practice, this would involve MFCC extraction, noise reduction, etc.
        return audio_data
    
    def _process_video_data(self, video_data: np.ndarray) -> np.ndarray:
        """Process video data."""
        # Simulate video processing (e.g., face detection, feature extraction)
        # In practice, this would involve face detection, landmark extraction, etc.
        return video_data
    
    def _process_motion_data(self, motion_data: np.ndarray) -> np.ndarray:
        """Process motion sensor data."""
        # Simulate motion processing (e.g., gait analysis, step detection)
        # In practice, this would involve step detection, gait cycle analysis, etc.
        return motion_data


class WearableDevice(EdgeDevice):
    """Wearable device implementation."""
    
    def __init__(
        self,
        device_id: str,
        location: str,
        processing_power: str = "medium",
        memory_limit: int = 2048,
        storage_limit: int = 4096,
        battery_level: int = 100,
        network_bandwidth: float = 10.0,
        latency: float = 100.0
    ):
        """
        Initialize wearable device.
        
        Args:
            device_id: Unique device identifier
            location: Device location
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            battery_level: Battery level (0-100)
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            device_id, "wearable", location, processing_power,
            memory_limit, storage_limit, battery_level, network_bandwidth, latency
        )
        
        # Wearable-specific capabilities
        self.accelerometer_available = True
        self.gyroscope_available = True
        self.heart_rate_available = True
        self.temperature_available = True
        
        # Data collection settings
        self.sensor_sampling_rate = 50
        self.heart_rate_sampling_rate = 1
    
    def collect_data(self) -> Dict[str, Any]:
        """Collect data from wearable sensors."""
        data = {}
        
        if self.accelerometer_available and self.gyroscope_available:
            # Simulate motion sensor data collection
            data['motion'] = self._collect_motion_data()
        
        if self.heart_rate_available:
            # Simulate heart rate data collection
            data['heart_rate'] = self._collect_heart_rate_data()
        
        if self.temperature_available:
            # Simulate temperature data collection
            data['temperature'] = self._collect_temperature_data()
        
        return data
    
    def _collect_motion_data(self) -> np.ndarray:
        """Collect motion sensor data."""
        # Simulate 1 second of motion data
        duration = 1.0
        samples = int(duration * self.sensor_sampling_rate)
        motion_data = np.random.randn(samples, 6).astype(np.float32)  # 3D accel + 3D gyro
        return motion_data
    
    def _collect_heart_rate_data(self) -> float:
        """Collect heart rate data."""
        # Simulate heart rate measurement
        return random.uniform(60, 100)
    
    def _collect_temperature_data(self) -> float:
        """Collect temperature data."""
        # Simulate temperature measurement
        return random.uniform(36.0, 37.5)
    
    def process_data(self, data: Any) -> Any:
        """Process data on wearable device."""
        if isinstance(data, dict):
            processed_data = {}
            
            if 'motion' in data:
                processed_data['motion'] = self._process_motion_data(data['motion'])
            
            if 'heart_rate' in data:
                processed_data['heart_rate'] = self._process_heart_rate_data(data['heart_rate'])
            
            if 'temperature' in data:
                processed_data['temperature'] = self._process_temperature_data(data['temperature'])
            
            return processed_data
        else:
            return data
    
    def _process_motion_data(self, motion_data: np.ndarray) -> np.ndarray:
        """Process motion sensor data."""
        # Simulate motion processing (e.g., gait analysis, activity recognition)
        return motion_data
    
    def _process_heart_rate_data(self, heart_rate: float) -> float:
        """Process heart rate data."""
        # Simulate heart rate processing (e.g., anomaly detection)
        return heart_rate
    
    def _process_temperature_data(self, temperature: float) -> float:
        """Process temperature data."""
        # Simulate temperature processing (e.g., fever detection)
        return temperature


class IoTSensorDevice(EdgeDevice):
    """IoT sensor device implementation."""
    
    def __init__(
        self,
        device_id: str,
        location: str,
        sensor_type: str = "environmental",
        processing_power: str = "low",
        memory_limit: int = 512,
        storage_limit: int = 1024,
        battery_level: int = 100,
        network_bandwidth: float = 1.0,
        latency: float = 200.0
    ):
        """
        Initialize IoT sensor device.
        
        Args:
            device_id: Unique device identifier
            location: Device location
            sensor_type: Type of sensor (environmental, motion, audio)
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            battery_level: Battery level (0-100)
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            device_id, "iot_sensor", location, processing_power,
            memory_limit, storage_limit, battery_level, network_bandwidth, latency
        )
        
        self.sensor_type = sensor_type
        
        # IoT sensor-specific capabilities
        if sensor_type == "environmental":
            self.temperature_available = True
            self.humidity_available = True
            self.pressure_available = True
        elif sensor_type == "motion":
            self.accelerometer_available = True
            self.pir_available = True
        elif sensor_type == "audio":
            self.microphone_available = True
        
        # Data collection settings
        self.sampling_rate = 1  # Low sampling rate for IoT sensors
    
    def collect_data(self) -> Dict[str, Any]:
        """Collect data from IoT sensors."""
        data = {}
        
        if self.sensor_type == "environmental":
            if self.temperature_available:
                data['temperature'] = self._collect_temperature_data()
            if self.humidity_available:
                data['humidity'] = self._collect_humidity_data()
            if self.pressure_available:
                data['pressure'] = self._collect_pressure_data()
        
        elif self.sensor_type == "motion":
            if self.accelerometer_available:
                data['acceleration'] = self._collect_acceleration_data()
            if self.pir_available:
                data['motion_detected'] = self._collect_motion_detection_data()
        
        elif self.sensor_type == "audio":
            if self.microphone_available:
                data['audio'] = self._collect_audio_data()
        
        return data
    
    def _collect_temperature_data(self) -> float:
        """Collect temperature data."""
        return random.uniform(18.0, 25.0)
    
    def _collect_humidity_data(self) -> float:
        """Collect humidity data."""
        return random.uniform(30.0, 70.0)
    
    def _collect_pressure_data(self) -> float:
        """Collect pressure data."""
        return random.uniform(1000.0, 1020.0)
    
    def _collect_acceleration_data(self) -> np.ndarray:
        """Collect acceleration data."""
        return np.random.randn(3).astype(np.float32)
    
    def _collect_motion_detection_data(self) -> bool:
        """Collect motion detection data."""
        return random.choice([True, False])
    
    def _collect_audio_data(self) -> np.ndarray:
        """Collect audio data."""
        # Simulate 1 second of audio data at low sample rate
        duration = 1.0
        sample_rate = 8000  # Lower sample rate for IoT sensors
        samples = int(duration * sample_rate)
        audio_data = np.random.randn(samples).astype(np.float32)
        return audio_data
    
    def process_data(self, data: Any) -> Any:
        """Process data on IoT sensor."""
        # IoT sensors typically do minimal processing
        if isinstance(data, dict):
            processed_data = {}
            
            for key, value in data.items():
                # Simple processing (e.g., filtering, normalization)
                if isinstance(value, np.ndarray):
                    processed_data[key] = self._simple_filter(value)
                else:
                    processed_data[key] = value
            
            return processed_data
        else:
            return data
    
    def _simple_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply simple filtering to sensor data."""
        # Simple moving average filter
        if len(data) > 3:
            filtered_data = np.convolve(data, np.ones(3)/3, mode='valid')
            return filtered_data
        else:
            return data
