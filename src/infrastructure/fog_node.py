"""
Fog node simulation for cognitive decline detection.

This module implements fog nodes that act as intermediate aggregation points
between edge devices and cloud servers, providing local processing and aggregation.
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
from collections import defaultdict

logger = logging.getLogger(__name__)


class FogNode(ABC):
    """Base class for fog nodes."""
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        location: str,
        capacity: int = 100,  # Max number of edge devices
        processing_power: str = "high",
        memory_limit: int = 16384,  # MB
        storage_limit: int = 100000,  # MB
        network_bandwidth: float = 100.0,  # Mbps
        latency: float = 10.0  # ms
    ):
        """
        Initialize fog node.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of fog node (clinic_server, local_gateway)
            location: Node location
            capacity: Maximum number of edge devices
            processing_power: Processing power level (low, medium, high)
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        self.node_id = node_id
        self.node_type = node_type
        self.location = location
        self.capacity = capacity
        self.processing_power = processing_power
        self.memory_limit = memory_limit
        self.storage_limit = storage_limit
        self.network_bandwidth = network_bandwidth
        self.latency = latency
        
        # Node state
        self.is_active = True
        self.is_connected = False
        self.current_load = 0
        self.connected_devices = set()
        self.connected_cloud_servers = set()
        
        # Resource monitoring
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.storage_usage = 0.0
        self.network_usage = 0.0
        
        # Performance metrics
        self.aggregation_time = 0.0
        self.communication_time = 0.0
        self.processing_time = 0.0
        
        # Data storage
        self.local_data = {}
        self.model_cache = {}
        self.aggregated_updates = {}
        self.client_updates = defaultdict(list)
        
        # Aggregation settings
        self.aggregation_interval = 600  # seconds
        self.min_clients = 3
        self.max_clients = 50
        self.last_aggregation_time = time.time()
        
        # Threading
        self.aggregation_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Initialize node
        self._initialize_node()
    
    def _initialize_node(self):
        """Initialize fog node components."""
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
        
        logger.info(f"Initialized {self.node_type} fog node: {self.node_id}")
    
    def _monitor_resources(self):
        """Monitor fog node resources."""
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
                    logger.warning(f"Fog node {self.node_id}: High memory usage: {self.memory_usage:.1f}%")
                
                if self.cpu_usage > 90:
                    logger.warning(f"Fog node {self.node_id}: High CPU usage: {self.cpu_usage:.1f}%")
                
                # Update current load
                self.current_load = len(self.connected_devices)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error monitoring resources for fog node {self.node_id}: {e}")
                time.sleep(5.0)
    
    def _aggregation_loop(self):
        """Main aggregation loop."""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if aggregation should be performed
                if (current_time - self.last_aggregation_time >= self.aggregation_interval and
                    len(self.client_updates) >= self.min_clients):
                    
                    self._perform_aggregation()
                    self.last_aggregation_time = current_time
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in aggregation loop for fog node {self.node_id}: {e}")
                time.sleep(30.0)
    
    def _perform_aggregation(self):
        """Perform local aggregation of client updates."""
        if not self.client_updates:
            return
        
        start_time = time.time()
        
        try:
            # Aggregate updates from connected devices
            aggregated_update = self._aggregate_client_updates()
            
            if aggregated_update:
                # Store aggregated update
                self.aggregated_updates[time.time()] = aggregated_update
                
                # Send to cloud servers
                self._send_to_cloud_servers(aggregated_update)
                
                # Clear client updates
                self.client_updates.clear()
                
                aggregation_time = time.time() - start_time
                self.aggregation_time += aggregation_time
                
                logger.info(f"Fog node {self.node_id}: Aggregation completed in {aggregation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Fog node {self.node_id}: Aggregation failed: {e}")
    
    def _aggregate_client_updates(self) -> Optional[Dict[str, Any]]:
        """Aggregate updates from connected clients."""
        if not self.client_updates:
            return None
        
        # Simple averaging aggregation (can be enhanced with more sophisticated methods)
        aggregated_update = {}
        
        for client_id, updates in self.client_updates.items():
            if not updates:
                continue
            
            # Average updates from this client
            client_aggregated = {}
            for key in updates[0].keys():
                if isinstance(updates[0][key], torch.Tensor):
                    # Average tensor updates
                    stacked_updates = torch.stack([update[key] for update in updates])
                    client_aggregated[key] = torch.mean(stacked_updates, dim=0)
                else:
                    # Average scalar updates
                    client_aggregated[key] = np.mean([update[key] for update in updates])
            
            aggregated_update[client_id] = client_aggregated
        
        return aggregated_update
    
    def _send_to_cloud_servers(self, aggregated_update: Dict[str, Any]):
        """Send aggregated update to cloud servers."""
        for cloud_server_id in self.connected_cloud_servers:
            try:
                # Simulate communication delay
                communication_delay = self.latency / 1000.0
                time.sleep(communication_delay)
                
                # Update communication time
                self.communication_time += communication_delay
                
                logger.info(f"Fog node {self.node_id}: Sent aggregated update to cloud server {cloud_server_id}")
                
            except Exception as e:
                logger.error(f"Fog node {self.node_id}: Failed to send to cloud server {cloud_server_id}: {e}")
    
    def connect_device(self, device_id: str) -> bool:
        """
        Connect an edge device to the fog node.
        
        Args:
            device_id: Device identifier
            
        Returns:
            True if connection successful
        """
        if len(self.connected_devices) >= self.capacity:
            logger.warning(f"Fog node {self.node_id}: Capacity reached, cannot connect device {device_id}")
            return False
        
        self.connected_devices.add(device_id)
        logger.info(f"Fog node {self.node_id}: Connected device {device_id}")
        return True
    
    def disconnect_device(self, device_id: str):
        """
        Disconnect an edge device from the fog node.
        
        Args:
            device_id: Device identifier
        """
        if device_id in self.connected_devices:
            self.connected_devices.remove(device_id)
            logger.info(f"Fog node {self.node_id}: Disconnected device {device_id}")
    
    def connect_cloud_server(self, cloud_server_id: str):
        """
        Connect to a cloud server.
        
        Args:
            cloud_server_id: Cloud server identifier
        """
        self.connected_cloud_servers.add(cloud_server_id)
        logger.info(f"Fog node {self.node_id}: Connected to cloud server {cloud_server_id}")
    
    def disconnect_cloud_server(self, cloud_server_id: str):
        """
        Disconnect from a cloud server.
        
        Args:
            cloud_server_id: Cloud server identifier
        """
        if cloud_server_id in self.connected_cloud_servers:
            self.connected_cloud_servers.remove(cloud_server_id)
            logger.info(f"Fog node {self.node_id}: Disconnected from cloud server {cloud_server_id}")
    
    def receive_client_update(self, device_id: str, update: Dict[str, Any]) -> bool:
        """
        Receive update from a connected device.
        
        Args:
            device_id: Device identifier
            update: Update data
            
        Returns:
            True if update received successfully
        """
        if device_id not in self.connected_devices:
            logger.warning(f"Fog node {self.node_id}: Received update from unconnected device {device_id}")
            return False
        
        try:
            # Store client update
            self.client_updates[device_id].append(update)
            
            # Limit number of updates per client
            if len(self.client_updates[device_id]) > 10:
                self.client_updates[device_id] = self.client_updates[device_id][-10:]
            
            logger.debug(f"Fog node {self.node_id}: Received update from device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Fog node {self.node_id}: Error receiving update from device {device_id}: {e}")
            return False
    
    def send_global_model(self, device_id: str, model_state: Dict[str, torch.Tensor]) -> bool:
        """
        Send global model to a connected device.
        
        Args:
            device_id: Device identifier
            model_state: Global model state
            
        Returns:
            True if model sent successfully
        """
        if device_id not in self.connected_devices:
            logger.warning(f"Fog node {self.node_id}: Cannot send model to unconnected device {device_id}")
            return False
        
        try:
            # Simulate communication delay
            communication_delay = self.latency / 1000.0
            time.sleep(communication_delay)
            
            # Update communication time
            self.communication_time += communication_delay
            
            logger.info(f"Fog node {self.node_id}: Sent global model to device {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Fog node {self.node_id}: Error sending model to device {device_id}: {e}")
            return False
    
    def process_data(self, data: Any) -> Any:
        """
        Process data locally on the fog node.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        start_time = time.time()
        
        try:
            # Simulate data processing
            processing_time = self._calculate_processing_time(data)
            time.sleep(processing_time)
            
            # Update processing time
            self.processing_time += processing_time
            
            logger.debug(f"Fog node {self.node_id}: Processed data in {processing_time:.2f}s")
            return data
            
        except Exception as e:
            logger.error(f"Fog node {self.node_id}: Error processing data: {e}")
            return None
    
    def _calculate_processing_time(self, data: Any) -> float:
        """Calculate processing time based on data size and node capabilities."""
        # Estimate data size
        data_size = self._estimate_data_size(data)
        
        # Base processing time
        base_time = data_size * 0.001  # 1ms per MB
        
        # Adjust based on processing power
        if self.processing_power == "low":
            multiplier = 2.0
        elif self.processing_power == "medium":
            multiplier = 1.0
        else:  # high
            multiplier = 0.5
        
        return base_time * multiplier
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate data size in MB."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size() / (1024 * 1024)
        elif isinstance(data, dict):
            return sum(self._estimate_data_size(v) for v in data.values())
        else:
            return 0.1  # Default estimate
    
    def store_data(self, key: str, data: Any):
        """Store data locally on the fog node."""
        if self.storage_usage < 90:  # Check storage limit
            self.local_data[key] = data
            logger.debug(f"Fog node {self.node_id}: Stored data with key {key}")
        else:
            logger.warning(f"Fog node {self.node_id}: Storage limit reached")
    
    def retrieve_data(self, key: str) -> Optional[Any]:
        """Retrieve data from local storage."""
        return self.local_data.get(key)
    
    def cache_model(self, model_id: str, model: nn.Module):
        """Cache model for faster access."""
        self.model_cache[model_id] = model
        logger.debug(f"Fog node {self.node_id}: Cached model {model_id}")
    
    def get_cached_model(self, model_id: str) -> Optional[nn.Module]:
        """Get cached model."""
        return self.model_cache.get(model_id)
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get fog node information."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'location': self.location,
            'capacity': self.capacity,
            'processing_power': self.processing_power,
            'memory_limit': self.memory_limit,
            'storage_limit': self.storage_limit,
            'network_bandwidth': self.network_bandwidth,
            'latency': self.latency,
            'is_active': self.is_active,
            'is_connected': self.is_connected,
            'current_load': self.current_load,
            'connected_devices': len(self.connected_devices),
            'connected_cloud_servers': len(self.connected_cloud_servers),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'storage_usage': self.storage_usage,
            'network_usage': self.network_usage,
            'aggregation_time': self.aggregation_time,
            'communication_time': self.communication_time,
            'processing_time': self.processing_time
        }
    
    def connect(self):
        """Connect fog node to network."""
        self.is_connected = True
        logger.info(f"Fog node {self.node_id}: Connected to network")
    
    def disconnect(self):
        """Disconnect fog node from network."""
        self.is_connected = False
        logger.info(f"Fog node {self.node_id}: Disconnected from network")
    
    def shutdown(self):
        """Shutdown fog node."""
        self.is_active = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=5.0)
        
        logger.info(f"Fog node {self.node_id}: Shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.shutdown()


class ClinicServer(FogNode):
    """Clinic server implementation."""
    
    def __init__(
        self,
        node_id: str,
        location: str,
        capacity: int = 50,
        processing_power: str = "high",
        memory_limit: int = 32768,
        storage_limit: int = 200000,
        network_bandwidth: float = 200.0,
        latency: float = 5.0
    ):
        """
        Initialize clinic server.
        
        Args:
            node_id: Unique node identifier
            location: Clinic location
            capacity: Maximum number of edge devices
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            node_id, "clinic_server", location, capacity, processing_power,
            memory_limit, storage_limit, network_bandwidth, latency
        )
        
        # Clinic-specific capabilities
        self.patient_database = {}
        self.clinical_models = {}
        self.privacy_filters = {}
        
        # Aggregation settings
        self.aggregation_interval = 300  # 5 minutes
        self.min_clients = 5
        self.max_clients = 30
    
    def register_patient(self, patient_id: str, patient_info: Dict[str, Any]):
        """
        Register a patient in the clinic database.
        
        Args:
            patient_id: Patient identifier
            patient_info: Patient information
        """
        self.patient_database[patient_id] = patient_info
        logger.info(f"Clinic server {self.node_id}: Registered patient {patient_id}")
    
    def get_patient_info(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get patient information.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Patient information or None
        """
        return self.patient_database.get(patient_id)
    
    def apply_privacy_filter(self, data: Any, patient_id: str) -> Any:
        """
        Apply privacy filter to patient data.
        
        Args:
            data: Data to filter
            patient_id: Patient identifier
            
        Returns:
            Privacy-filtered data
        """
        # Simulate privacy filtering
        # In practice, this would involve data anonymization, encryption, etc.
        filtered_data = data
        logger.debug(f"Clinic server {self.node_id}: Applied privacy filter for patient {patient_id}")
        return filtered_data
    
    def store_clinical_model(self, model_id: str, model: nn.Module, patient_id: str):
        """
        Store clinical model for a patient.
        
        Args:
            model_id: Model identifier
            model: Clinical model
            patient_id: Patient identifier
        """
        if patient_id not in self.clinical_models:
            self.clinical_models[patient_id] = {}
        
        self.clinical_models[patient_id][model_id] = model
        logger.info(f"Clinic server {self.node_id}: Stored clinical model {model_id} for patient {patient_id}")
    
    def get_clinical_model(self, model_id: str, patient_id: str) -> Optional[nn.Module]:
        """
        Get clinical model for a patient.
        
        Args:
            model_id: Model identifier
            patient_id: Patient identifier
            
        Returns:
            Clinical model or None
        """
        if patient_id in self.clinical_models:
            return self.clinical_models[patient_id].get(model_id)
        return None


class LocalGateway(FogNode):
    """Local gateway implementation."""
    
    def __init__(
        self,
        node_id: str,
        location: str,
        capacity: int = 100,
        processing_power: str = "medium",
        memory_limit: int = 8192,
        storage_limit: int = 50000,
        network_bandwidth: float = 50.0,
        latency: float = 20.0
    ):
        """
        Initialize local gateway.
        
        Args:
            node_id: Unique node identifier
            location: Gateway location
            capacity: Maximum number of edge devices
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            node_id, "local_gateway", location, capacity, processing_power,
            memory_limit, storage_limit, network_bandwidth, latency
        )
        
        # Gateway-specific capabilities
        self.device_registry = {}
        self.routing_table = {}
        self.load_balancer = {}
        
        # Aggregation settings
        self.aggregation_interval = 600  # 10 minutes
        self.min_clients = 3
        self.max_clients = 50
    
    def register_device(self, device_id: str, device_info: Dict[str, Any]):
        """
        Register a device in the gateway registry.
        
        Args:
            device_id: Device identifier
            device_info: Device information
        """
        self.device_registry[device_id] = device_info
        logger.info(f"Local gateway {self.node_id}: Registered device {device_id}")
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get device information.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Device information or None
        """
        return self.device_registry.get(device_id)
    
    def update_routing_table(self, destination: str, next_hop: str):
        """
        Update routing table.
        
        Args:
            destination: Destination identifier
            next_hop: Next hop identifier
        """
        self.routing_table[destination] = next_hop
        logger.debug(f"Local gateway {self.node_id}: Updated routing table for {destination}")
    
    def route_message(self, destination: str, message: Any) -> bool:
        """
        Route message to destination.
        
        Args:
            destination: Destination identifier
            message: Message to route
            
        Returns:
            True if routing successful
        """
        if destination in self.routing_table:
            next_hop = self.routing_table[destination]
            logger.debug(f"Local gateway {self.node_id}: Routing message to {destination} via {next_hop}")
            return True
        else:
            logger.warning(f"Local gateway {self.node_id}: No route to {destination}")
            return False
    
    def balance_load(self, device_id: str) -> str:
        """
        Balance load for a device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Assigned fog node identifier
        """
        # Simple load balancing (can be enhanced with more sophisticated algorithms)
        if device_id in self.load_balancer:
            return self.load_balancer[device_id]
        
        # Assign to least loaded fog node
        assigned_node = f"fog_node_{len(self.load_balancer) % 3 + 1}"
        self.load_balancer[device_id] = assigned_node
        
        logger.info(f"Local gateway {self.node_id}: Assigned device {device_id} to {assigned_node}")
        return assigned_node
