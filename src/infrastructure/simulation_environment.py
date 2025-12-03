"""
Simulation environment for cognitive decline detection.

This module implements the main simulation environment that orchestrates
the edge-fog-cloud infrastructure and federated learning process.
"""

import time
import random
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
import yaml
from collections import defaultdict
import json
import os

from src.infrastructure.edge_device import EdgeDevice, SmartphoneDevice, WearableDevice
from src.infrastructure.fog_node import FogNode, ClinicServer, LocalGateway
from src.infrastructure.cloud_server import CloudServer, GlobalAggregator, AnalyticsServer
from src.infrastructure.network_simulator import NetworkSimulator, WiFiNetwork, CellularNetwork, EthernetNetwork, FiberNetwork

logger = logging.getLogger(__name__)


class SimulationEnvironment:
    """Main simulation environment for the cognitive decline detection system."""
    
    def __init__(self, config_path: str):
        """
        Initialize simulation environment.
        
        Args:
            config_path: Path to simulation configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Simulation state
        self.is_running = False
        self.simulation_time = 0.0
        self.start_time = None
        self.end_time = None
        
        # Infrastructure components
        self.edge_devices = {}
        self.fog_nodes = {}
        self.cloud_servers = {}
        self.networks = {}
        
        # Federated learning components
        self.fl_clients = {}
        self.fl_servers = {}
        self.global_model = None
        
        # Performance metrics
        self.metrics = {
            'communication_overhead': 0.0,
            'aggregation_time': 0.0,
            'model_accuracy': 0.0,
            'privacy_score': 0.0,
            'energy_consumption': 0.0,
            'latency': 0.0,
            'throughput': 0.0
        }
        
        # Data collection
        self.simulation_data = {
            'timestamps': [],
            'metrics_history': [],
            'events': [],
            'performance_logs': []
        }
        
        # Threading
        self.simulation_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Initialize simulation
        self._initialize_simulation()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load simulation configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded simulation configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_simulation(self):
        """Initialize simulation components."""
        try:
            # Initialize networks
            self._initialize_networks()
            
            # Initialize cloud servers
            self._initialize_cloud_servers()
            
            # Initialize fog nodes
            self._initialize_fog_nodes()
            
            # Initialize edge devices
            self._initialize_edge_devices()
            
            # Initialize federated learning components
            self._initialize_federated_learning()
            
            # Connect devices to fog nodes and networks
            self._connect_infrastructure()
            
            logger.info("Simulation environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize simulation: {e}")
            raise
    
    def _initialize_networks(self):
        """Initialize network simulators."""
        network_configs = self.config.get('networks', {})
        
        for network_id, network_config in network_configs.items():
            network_type = network_config.get('type', 'wifi')
            
            if network_type == 'wifi':
                network = WiFiNetwork(
                    simulator_id=network_id,
                    frequency=network_config.get('frequency', '2.4GHz'),
                    standard=network_config.get('standard', '802.11n'),
                    signal_strength=network_config.get('signal_strength', -50.0),
                    interference_level=network_config.get('interference_level', 0.1)
                )
            elif network_type == 'cellular':
                network = CellularNetwork(
                    simulator_id=network_id,
                    generation=network_config.get('generation', '4G'),
                    carrier=network_config.get('carrier', 'LTE'),
                    signal_strength=network_config.get('signal_strength', -70.0),
                    congestion_level=network_config.get('congestion_level', 0.2)
                )
            elif network_type == 'ethernet':
                network = EthernetNetwork(
                    simulator_id=network_id,
                    speed=network_config.get('speed', '1Gbps'),
                    cable_type=network_config.get('cable_type', 'Cat6'),
                    distance=network_config.get('distance', 100.0)
                )
            elif network_type == 'fiber':
                network = FiberNetwork(
                    simulator_id=network_id,
                    speed=network_config.get('speed', '10Gbps'),
                    fiber_type=network_config.get('fiber_type', 'Single-mode'),
                    distance=network_config.get('distance', 1000.0)
                )
            else:
                logger.warning(f"Unknown network type: {network_type}")
                continue
            
            self.networks[network_id] = network
            logger.info(f"Initialized {network_type} network: {network_id}")
    
    def _initialize_cloud_servers(self):
        """Initialize cloud servers."""
        cloud_configs = self.config.get('cloud_servers', {})
        
        for server_id, server_config in cloud_configs.items():
            server_type = server_config.get('type', 'global_aggregator')
            
            if server_type == 'global_aggregator':
                server = GlobalAggregator(
                    server_id=server_id,
                    location=server_config.get('location', 'data_center'),
                    capacity=server_config.get('capacity', 1000),
                    processing_power=server_config.get('processing_power', 'ultra_high'),
                    memory_limit=server_config.get('memory_limit', 262144),
                    storage_limit=server_config.get('storage_limit', 2000000),
                    network_bandwidth=server_config.get('network_bandwidth', 2000.0),
                    latency=server_config.get('latency', 2.0)
                )
            elif server_type == 'analytics_server':
                server = AnalyticsServer(
                    server_id=server_id,
                    location=server_config.get('location', 'data_center'),
                    capacity=server_config.get('capacity', 500),
                    processing_power=server_config.get('processing_power', 'high'),
                    memory_limit=server_config.get('memory_limit', 65536),
                    storage_limit=server_config.get('storage_limit', 500000),
                    network_bandwidth=server_config.get('network_bandwidth', 500.0),
                    latency=server_config.get('latency', 10.0)
                )
            else:
                logger.warning(f"Unknown cloud server type: {server_type}")
                continue
            
            self.cloud_servers[server_id] = server
            logger.info(f"Initialized {server_type} cloud server: {server_id}")
    
    def _initialize_fog_nodes(self):
        """Initialize fog nodes."""
        fog_configs = self.config.get('fog_nodes', {})
        
        for node_id, node_config in fog_configs.items():
            node_type = node_config.get('type', 'clinic_server')
            
            if node_type == 'clinic_server':
                node = ClinicServer(
                    node_id=node_id,
                    location=node_config.get('location', 'clinic'),
                    capacity=node_config.get('capacity', 50),
                    processing_power=node_config.get('processing_power', 'high'),
                    memory_limit=node_config.get('memory_limit', 32768),
                    storage_limit=node_config.get('storage_limit', 200000),
                    network_bandwidth=node_config.get('network_bandwidth', 200.0),
                    latency=node_config.get('latency', 5.0)
                )
            elif node_type == 'local_gateway':
                node = LocalGateway(
                    node_id=node_id,
                    location=node_config.get('location', 'gateway'),
                    capacity=node_config.get('capacity', 100),
                    processing_power=node_config.get('processing_power', 'medium'),
                    memory_limit=node_config.get('memory_limit', 8192),
                    storage_limit=node_config.get('storage_limit', 50000),
                    network_bandwidth=node_config.get('network_bandwidth', 50.0),
                    latency=node_config.get('latency', 20.0)
                )
            else:
                logger.warning(f"Unknown fog node type: {node_type}")
                continue
            
            self.fog_nodes[node_id] = node
            logger.info(f"Initialized {node_type} fog node: {node_id}")
    
    def _initialize_edge_devices(self):
        """Initialize edge devices."""
        device_configs = self.config.get('edge_devices', {})
        
        for device_id, device_config in device_configs.items():
            device_type = device_config.get('type', 'smartphone')
            
            if device_type == 'smartphone':
                device = SmartphoneDevice(
                    device_id=device_id,
                    location=device_config.get('location', 'user_location'),
                    battery_level=device_config.get('battery_level', 100.0),
                    processing_power=device_config.get('processing_power', 'medium'),
                    memory_limit=device_config.get('memory_limit', 4096),
                    storage_limit=device_config.get('storage_limit', 128000),
                    network_bandwidth=device_config.get('network_bandwidth', 50.0),
                    latency=device_config.get('latency', 20.0)
                )
            elif device_type == 'wearable':
                device = WearableDevice(
                    device_id=device_id,
                    location=device_config.get('location', 'user_location'),
                    battery_level=device_config.get('battery_level', 100.0),
                    processing_power=device_config.get('processing_power', 'low'),
                    memory_limit=device_config.get('memory_limit', 1024),
                    storage_limit=device_config.get('storage_limit', 8000),
                    network_bandwidth=device_config.get('network_bandwidth', 10.0),
                    latency=device_config.get('latency', 50.0)
                )
            else:
                logger.warning(f"Unknown edge device type: {device_type}")
                continue
            
            self.edge_devices[device_id] = device
            logger.info(f"Initialized {device_type} edge device: {device_id}")
    
    def _initialize_federated_learning(self):
        """Initialize federated learning components."""
        fl_config = self.config.get('federated_learning', {})
        
        # Initialize global model
        self.global_model = self._create_global_model(fl_config)
        
        # Initialize FL clients
        for device_id, device in self.edge_devices.items():
            client = self._create_fl_client(device_id, device, fl_config)
            self.fl_clients[device_id] = client
        
        # Initialize FL servers
        for node_id, node in self.fog_nodes.items():
            server = self._create_fl_server(node_id, node, fl_config)
            self.fl_servers[node_id] = server
        
        logger.info("Federated learning components initialized")
    
    def _create_global_model(self, fl_config: Dict[str, Any]):
        """Create global model for federated learning."""
        # This would create the actual neural network model
        # For now, we'll return a placeholder
        return {
            'model_id': 'global_model',
            'created_at': time.time(),
            'parameters': {},
            'accuracy': 0.0
        }
    
    def _create_fl_client(self, device_id: str, device: EdgeDevice, fl_config: Dict[str, Any]):
        """Create federated learning client for edge device."""
        return {
            'client_id': device_id,
            'device': device,
            'local_model': None,
            'training_data': [],
            'last_update': None,
            'update_count': 0
        }
    
    def _create_fl_server(self, node_id: str, node: FogNode, fl_config: Dict[str, Any]):
        """Create federated learning server for fog node."""
        return {
            'server_id': node_id,
            'node': node,
            'aggregated_model': None,
            'client_updates': [],
            'last_aggregation': None,
            'aggregation_count': 0
        }
    
    def _connect_infrastructure(self):
        """Connect devices to fog nodes, fog nodes to cloud, and all to networks."""
        # Connect edge devices to fog nodes
        for device_id, device in self.edge_devices.items():
            if self.fog_nodes:
                # Connect to a random fog node
                fog_node = random.choice(list(self.fog_nodes.values()))
                try:
                    fog_node.connect_device(device_id)
                    logger.debug(f"Device {device_id} connected to fog node {fog_node.node_id}")
                except Exception as e:
                    logger.debug(f"Device {device_id}: Connection failed - {e}")
        
        # Connect fog nodes to cloud servers
        for node_id, node in self.fog_nodes.items():
            if self.cloud_servers:
                # Connect to a random cloud server
                cloud_server = random.choice(list(self.cloud_servers.values()))
                try:
                    cloud_server.connect_fog_node(node_id)
                    logger.debug(f"Fog node {node_id} connected to cloud server {cloud_server.server_id}")
                except Exception as e:
                    logger.debug(f"Fog node {node_id}: Connection failed - {e}")
        
        # Add all nodes to networks
        for network in self.networks.values():
            # Add edge devices to WiFi or Cellular networks
            if network.network_type in ['wifi', 'cellular']:
                for device_id in self.edge_devices.keys():
                    try:
                        network.add_node(device_id, 'edge_device', (random.random() * 100, random.random() * 100))
                    except Exception as e:
                        logger.debug(f"Failed to add device {device_id} to network {network.simulator_id}: {e}")
                
                # Also add fog nodes to WiFi/Cellular so devices can communicate with them
                for node_id in self.fog_nodes.keys():
                    try:
                        network.add_node(node_id, 'fog_node', (random.random() * 100, random.random() * 100))
                    except Exception as e:
                        logger.debug(f"Failed to add fog node {node_id} to network {network.simulator_id}: {e}")
            
            # Add fog nodes to Ethernet or Fiber networks
            if network.network_type in ['ethernet', 'fiber']:
                for node_id in self.fog_nodes.keys():
                    try:
                        network.add_node(node_id, 'fog_node', (random.random() * 100, random.random() * 100))
                    except Exception as e:
                        logger.debug(f"Failed to add fog node {node_id} to network {network.simulator_id}: {e}")
                
                # Add cloud servers to Ethernet or Fiber networks
                for server_id in self.cloud_servers.keys():
                    try:
                        network.add_node(server_id, 'cloud_server', (random.random() * 100, random.random() * 100))
                    except Exception as e:
                        logger.debug(f"Failed to add cloud server {server_id} to network {network.simulator_id}: {e}")
        
        # Create network links/connections between nodes in the same network
        for network in self.networks.values():
            try:
                nodes = list(network.connected_nodes)
                # Create links between devices and fog nodes in WiFi/Cellular
                if network.network_type in ['wifi', 'cellular']:
                    devices = [n for n in nodes if n in self.edge_devices]
                    fog_nodes = [n for n in nodes if n in self.fog_nodes]
                    for device_id in devices:
                        for fog_id in fog_nodes:
                            try:
                                network.add_link(device_id, fog_id, {'latency': network.base_latency, 'bandwidth': network.base_bandwidth})
                            except:
                                pass  # Link might already exist
                
                # Create links between fog nodes and cloud servers in Ethernet/Fiber
                if network.network_type in ['ethernet', 'fiber']:
                    fog_nodes = [n for n in nodes if n in self.fog_nodes]
                    cloud_servers = [n for n in nodes if n in self.cloud_servers]
                    for fog_id in fog_nodes:
                        for cloud_id in cloud_servers:
                            try:
                                network.add_link(fog_id, cloud_id, {'latency': network.base_latency, 'bandwidth': network.base_bandwidth})
                            except:
                                pass  # Link might already exist
            except Exception as e:
                logger.debug(f"Failed to create network links for {network.simulator_id}: {e}")
    
    def start_simulation(self, duration: float = 3600.0):
        """
        Start the simulation.
        
        Args:
            duration: Simulation duration in seconds
        """
        if self.is_running:
            logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.end_time = self.start_time + duration
        self.simulation_time = 0.0
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Started simulation for {duration} seconds")
    
    def stop_simulation(self):
        """Stop the simulation."""
        if not self.is_running:
            logger.warning("Simulation is not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=10.0)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10.0)
        
        logger.info("Simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = time.time()
                self.simulation_time = current_time - self.start_time
                
                # Check if simulation should end
                if current_time >= self.end_time:
                    self.stop_simulation()
                    break
                
                # Run simulation step
                self._simulation_step()
                
                # Sleep for simulation step interval
                time.sleep(1.0)  # 1 second simulation step
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(5.0)
    
    def _monitoring_loop(self):
        """Monitoring loop for collecting metrics."""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Log performance
                self._log_performance()
                
                # Sleep for monitoring interval
                time.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30.0)
    
    def _simulation_step(self):
        """Execute one simulation step."""
        # Simulate edge device activities
        for device_id, device in self.edge_devices.items():
            self._simulate_device_activity(device_id, device)
        
        # Simulate fog node activities
        for node_id, node in self.fog_nodes.items():
            self._simulate_fog_activity(node_id, node)
        
        # Simulate cloud server activities
        for server_id, server in self.cloud_servers.items():
            self._simulate_cloud_activity(server_id, server)
        
        # Simulate network activities
        for network_id, network in self.networks.items():
            self._simulate_network_activity(network_id, network)
    
    def _simulate_device_activity(self, device_id: str, device: EdgeDevice):
        """Simulate edge device activity."""
        # Simulate data collection
        if random.random() < 0.3:  # 30% chance per step
            collected_data = device.collect_data()
            
            # Simulate local processing on collected data
            if collected_data and random.random() < 0.5:  # 50% chance to process
                try:
                    processed_data = device.process_data(collected_data)
                    logger.debug(f"Device {device_id}: Processed {len(str(processed_data))} bytes of data")
                except Exception as e:
                    logger.debug(f"Device {device_id}: Processing skipped - {e}")
        
        # Simulate communication with fog nodes
        if random.random() < 0.1:  # 10% chance per step
            self._simulate_device_communication(device_id, device)
    
    def _simulate_fog_activity(self, node_id: str, node: FogNode):
        """Simulate fog node activity."""
        # Simulate aggregation (more frequent)
        if random.random() < 0.05:  # 5% chance per step
            try:
                node._perform_aggregation()
                logger.info(f"Fog node {node_id}: Performed aggregation")
            except Exception as e:
                logger.debug(f"Fog node {node_id}: Aggregation skipped - {e}")
        
        # Simulate processing data from devices
        if random.random() < 0.1:  # 10% chance per step
            # Simulate receiving and processing data
            sample_data = {'data': 'sample', 'size': random.uniform(0.1, 5.0)}
            try:
                processed = node.process_data(sample_data)
                logger.debug(f"Fog node {node_id}: Processed data from device")
            except Exception as e:
                logger.debug(f"Fog node {node_id}: Processing skipped - {e}")
        
        # Simulate communication with cloud
        if random.random() < 0.05:  # 5% chance per step
            self._simulate_fog_communication(node_id, node)
    
    def _simulate_cloud_activity(self, server_id: str, server: CloudServer):
        """Simulate cloud server activity."""
        # Simulate global aggregation (more frequent)
        if random.random() < 0.02:  # 2% chance per step
            try:
                server._perform_global_aggregation()
                logger.info(f"Cloud server {server_id}: Performed global aggregation")
            except Exception as e:
                logger.debug(f"Cloud server {server_id}: Global aggregation skipped - {e}")
        
        # Simulate analytics (more frequent)
        if random.random() < 0.05:  # 5% chance per step
            try:
                server._perform_analytics()
                logger.info(f"Cloud server {server_id}: Performed analytics")
            except Exception as e:
                logger.debug(f"Cloud server {server_id}: Analytics skipped - {e}")
    
    def _simulate_network_activity(self, network_id: str, network: NetworkSimulator):
        """Simulate network activity."""
        # Simulate network traffic (more frequent)
        if random.random() < 0.3:  # 30% chance per step
            self._simulate_network_traffic(network_id, network)
    
    def _simulate_device_communication(self, device_id: str, device: EdgeDevice):
        """Simulate device communication."""
        # Find connected fog node
        connected_fog = self._find_connected_fog_node(device_id)
        if connected_fog:
            # Simulate sending update with actual data
            data_size = random.uniform(0.1, 2.0)  # MB
            update_data = {
                'device_id': device_id, 
                'timestamp': time.time(),
                'data_size': data_size,
                'model_update': {'weights': [random.random() for _ in range(10)]}
            }
            try:
                connected_fog.receive_client_update(device_id, update_data)
                # Simulate network transmission - find a network that has both device and fog node
                network = None
                for net in self.networks.values():
                    if device_id in net.connected_nodes and connected_fog.node_id in net.connected_nodes:
                        network = net
                        break
                
                if network:
                    try:
                        result = network.simulate_transmission(device_id, connected_fog.node_id, data_size)
                        if result and result.get('success'):
                            logger.info(f"Device {device_id} -> Fog {connected_fog.node_id}: Sent {data_size:.2f} MB via {network.simulator_id}")
                        else:
                            logger.debug(f"Device {device_id} -> Fog {connected_fog.node_id}: Transmission failed")
                    except Exception as e:
                        logger.debug(f"Device {device_id}: Network transmission error - {e}")
                else:
                    logger.debug(f"Device {device_id}: No network found connecting to fog {connected_fog.node_id}")
            except Exception as e:
                logger.debug(f"Device {device_id}: Communication failed - {e}")
        else:
            # Try to connect to a fog node
            if self.fog_nodes:
                fog_node = random.choice(list(self.fog_nodes.values()))
                try:
                    fog_node.connect_device(device_id)
                    logger.debug(f"Device {device_id} connected to fog node {fog_node.node_id}")
                except Exception as e:
                    logger.debug(f"Device {device_id}: Connection failed - {e}")
    
    def _simulate_fog_communication(self, node_id: str, node: FogNode):
        """Simulate fog node communication."""
        # Find connected cloud server
        connected_cloud = self._find_connected_cloud_server(node_id)
        if connected_cloud:
            # Simulate sending aggregated update with actual data
            data_size = random.uniform(0.5, 5.0)  # MB
            aggregated_update = {
                'fog_node_id': node_id, 
                'timestamp': time.time(),
                'data_size': data_size,
                'aggregated_model': {'weights': [random.random() for _ in range(20)]}
            }
            try:
                connected_cloud.receive_fog_update(node_id, aggregated_update)
                # Simulate network transmission - find a network that has both fog node and cloud server
                network = None
                for net in self.networks.values():
                    if node_id in net.connected_nodes and connected_cloud.server_id in net.connected_nodes:
                        network = net
                        break
                
                if network:
                    try:
                        result = network.simulate_transmission(node_id, connected_cloud.server_id, data_size)
                        if result and result.get('success'):
                            logger.info(f"Fog {node_id} -> Cloud {connected_cloud.server_id}: Sent {data_size:.2f} MB via {network.simulator_id}")
                        else:
                            logger.debug(f"Fog {node_id} -> Cloud {connected_cloud.server_id}: Transmission failed")
                    except Exception as e:
                        logger.debug(f"Fog {node_id}: Network transmission error - {e}")
                else:
                    logger.debug(f"Fog {node_id}: No network found connecting to cloud {connected_cloud.server_id}")
            except Exception as e:
                logger.debug(f"Fog {node_id}: Communication failed - {e}")
        else:
            # Try to connect to a cloud server
            if self.cloud_servers:
                cloud_server = random.choice(list(self.cloud_servers.values()))
                try:
                    cloud_server.connect_fog_node(node_id)
                    logger.debug(f"Fog node {node_id} connected to cloud server {cloud_server.server_id}")
                except Exception as e:
                    logger.debug(f"Fog {node_id}: Connection failed - {e}")
    
    def _simulate_network_traffic(self, network_id: str, network: NetworkSimulator):
        """Simulate network traffic."""
        # Simulate random data transmission
        if len(network.connected_nodes) >= 2:
            nodes = list(network.connected_nodes)
            source = random.choice(nodes)
            destination = random.choice([n for n in nodes if n != source])
            
            data_size = random.uniform(0.1, 5.0)  # MB
            try:
                result = network.simulate_transmission(source, destination, data_size)
                if result.get('success'):
                    logger.debug(f"Network {network_id}: {source} → {destination}, {data_size:.2f} MB, "
                              f"latency {result.get('latency', 0):.2f}ms, "
                              f"bandwidth {result.get('bandwidth', 0):.2f}Mbps")
            except Exception as e:
                logger.debug(f"Network {network_id}: Transmission failed - {e}")
    
    def _find_connected_fog_node(self, device_id: str) -> Optional[FogNode]:
        """Find fog node connected to device."""
        for node_id, node in self.fog_nodes.items():
            if device_id in node.connected_devices:
                return node
        return None
    
    def _find_connected_cloud_server(self, node_id: str) -> Optional[CloudServer]:
        """Find cloud server connected to fog node."""
        for server_id, server in self.cloud_servers.items():
            if node_id in server.connected_fog_nodes:
                return server
        return None
    
    def _get_device_network(self, device_id: str) -> Optional[NetworkSimulator]:
        """Get network for device."""
        # Try to find a network that has this device
        for network in self.networks.values():
            if device_id in network.connected_nodes:
                return network
        # If not found, return a random network (device will connect)
        if self.networks:
            return random.choice(list(self.networks.values()))
        return None
    
    def _get_fog_network(self, node_id: str) -> Optional[NetworkSimulator]:
        """Get network for fog node."""
        # Try to find a network that has this fog node
        for network in self.networks.values():
            if node_id in network.connected_nodes:
                return network
        # If not found, return a random network (fog node will connect)
        if self.networks:
            return random.choice(list(self.networks.values()))
        return None
    
    def _collect_metrics(self):
        """Collect simulation metrics."""
        # Communication overhead (convert bytes to MB)
        total_communication_bytes = 0.0
        for network in self.networks.values():
            total_communication_bytes += network.total_bytes_transferred
        
        self.metrics['communication_overhead'] = total_communication_bytes / (1024 * 1024)  # Convert to MB
        
        # Aggregation time (average, not sum)
        total_aggregation_time = 0.0
        count = 0
        for node in self.fog_nodes.values():
            if hasattr(node, 'aggregation_time') and node.aggregation_time > 0:
                total_aggregation_time += node.aggregation_time
                count += 1
        for server in self.cloud_servers.values():
            if hasattr(server, 'aggregation_time') and server.aggregation_time > 0:
                total_aggregation_time += server.aggregation_time
                count += 1
        
        self.metrics['aggregation_time'] = total_aggregation_time / count if count > 0 else 0.0
        
        # Latency (average across networks)
        total_latency = 0.0
        network_count = 0
        for network in self.networks.values():
            if network.avg_latency > 0:
                total_latency += network.avg_latency
                network_count += 1
        
        self.metrics['latency'] = total_latency / network_count if network_count > 0 else 0.0
        
        # Throughput (actual data throughput rate, not sum of capacities)
        # Calculate based on actual data transferred over time
        if self.simulation_time > 0 and total_communication_bytes > 0:
            # Actual throughput = total bytes transferred / time (in Mbps)
            total_bits = total_communication_bytes * 8  # Convert to bits
            throughput_mbps = (total_bits / self.simulation_time) / (1024 * 1024)  # Convert to Mbps
            self.metrics['throughput'] = throughput_mbps
        else:
            # No data transferred yet, throughput is 0
            self.metrics['throughput'] = 0.0
        
        # Energy consumption
        total_energy = 0.0
        for device in self.edge_devices.values():
            if hasattr(device, 'energy_consumption'):
                total_energy += device.energy_consumption
        
        self.metrics['energy_consumption'] = total_energy
    
    def _log_performance(self):
        """Log performance metrics."""
        timestamp = time.time()
        
        # Store metrics history
        self.simulation_data['timestamps'].append(timestamp)
        self.simulation_data['metrics_history'].append(self.metrics.copy())
        
        # Log performance with more details
        logger.info("=" * 60)
        logger.info(f"Simulation Status at {self.simulation_time:.1f}s")
        logger.info("=" * 60)
        logger.info(f"  Communication overhead: {self.metrics['communication_overhead']:.2f} MB")
        logger.info(f"  Aggregation time: {self.metrics['aggregation_time']:.2f} s")
        logger.info(f"  Average latency: {self.metrics['latency']:.2f} ms")
        logger.info(f"  Data throughput: {self.metrics['throughput']:.2f} Mbps")
        logger.info(f"  Energy consumption: {self.metrics['energy_consumption']:.2f} J")
        
        # Log infrastructure status
        active_devices = sum(1 for d in self.edge_devices.values() if hasattr(d, 'is_active') and d.is_active)
        active_fog = sum(1 for n in self.fog_nodes.values() if hasattr(n, 'is_active') and n.is_active)
        active_cloud = sum(1 for s in self.cloud_servers.values() if hasattr(s, 'is_active') and s.is_active)
        
        logger.info(f"  Active devices: {active_devices}/{len(self.edge_devices)}")
        logger.info(f"  Active fog nodes: {active_fog}/{len(self.fog_nodes)}")
        logger.info(f"  Active cloud servers: {active_cloud}/{len(self.cloud_servers)}")
        
        # Log network activity
        total_network_bytes = sum(n.total_bytes_transferred for n in self.networks.values())
        logger.info(f"  Total network traffic: {total_network_bytes / (1024*1024):.2f} MB")
        logger.info("=" * 60)
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        return {
            'is_running': self.is_running,
            'simulation_time': self.simulation_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'metrics': self.metrics.copy(),
            'infrastructure': {
                'edge_devices': len(self.edge_devices),
                'fog_nodes': len(self.fog_nodes),
                'cloud_servers': len(self.cloud_servers),
                'networks': len(self.networks)
            }
        }
    
    def save_simulation_data(self, output_path: str):
        """Save simulation data to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.simulation_data, f, indent=2)
            
            logger.info(f"Simulation data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save simulation data: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate simulation report."""
        return {
            'simulation_summary': {
                'duration': self.simulation_time,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'status': 'completed' if not self.is_running else 'running'
            },
            'infrastructure_summary': {
                'edge_devices': len(self.edge_devices),
                'fog_nodes': len(self.fog_nodes),
                'cloud_servers': len(self.cloud_servers),
                'networks': len(self.networks)
            },
            'performance_metrics': self.metrics.copy(),
            'data_collection': {
                'total_timestamps': len(self.simulation_data['timestamps']),
                'total_metrics': len(self.simulation_data['metrics_history']),
                'total_events': len(self.simulation_data['events'])
            }
        }
    
    def shutdown(self):
        """Shutdown simulation environment."""
        self.stop_simulation()
        
        # Shutdown all components
        for device in self.edge_devices.values():
            device.shutdown()
        
        for node in self.fog_nodes.values():
            node.shutdown()
        
        for server in self.cloud_servers.values():
            server.shutdown()
        
        for network in self.networks.values():
            network.shutdown()
        
        logger.info("Simulation environment shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.shutdown()
