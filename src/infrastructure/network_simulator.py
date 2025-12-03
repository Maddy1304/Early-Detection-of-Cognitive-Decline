"""
Network simulator for cognitive decline detection.

This module implements network simulation capabilities for the edge-fog-cloud
infrastructure, including latency, bandwidth, and connectivity modeling.
"""

import time
import random
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


class NetworkSimulator(ABC):
    """Base class for network simulators."""
    
    def __init__(
        self,
        simulator_id: str,
        network_type: str,
        topology: str = "mesh",
        base_latency: float = 10.0,  # ms
        base_bandwidth: float = 100.0,  # Mbps
        packet_loss_rate: float = 0.001,  # 0.1%
        jitter: float = 2.0  # ms
    ):
        """
        Initialize network simulator.
        
        Args:
            simulator_id: Unique simulator identifier
            network_type: Type of network (wifi, cellular, ethernet, fiber)
            topology: Network topology (mesh, star, tree, ring)
            base_latency: Base network latency in ms
            base_bandwidth: Base network bandwidth in Mbps
            packet_loss_rate: Packet loss rate (0.0 to 1.0)
            jitter: Network jitter in ms
        """
        self.simulator_id = simulator_id
        self.network_type = network_type
        self.topology = topology
        self.base_latency = base_latency
        self.base_bandwidth = base_bandwidth
        self.packet_loss_rate = packet_loss_rate
        self.jitter = jitter
        
        # Network state
        self.is_active = True
        self.connected_nodes = set()
        self.network_graph = nx.Graph()
        self.link_properties = {}
        
        # Performance metrics
        self.total_packets_sent = 0
        self.total_packets_received = 0
        self.total_packets_lost = 0
        self.total_bytes_transferred = 0
        self.avg_latency = base_latency  # Initialize with base latency
        self.avg_bandwidth = base_bandwidth  # Initialize with base bandwidth
        
        # Network monitoring
        self.latency_history = []
        self.bandwidth_history = []
        self.packet_loss_history = []
        
        # Threading
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Initialize network
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize network components."""
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_network)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info(f"Initialized {self.network_type} network simulator: {self.simulator_id}")
    
    def _monitor_network(self):
        """Monitor network performance."""
        while not self.stop_event.is_set():
            try:
                # Update network metrics
                self._update_network_metrics()
                
                # Check for network issues
                self._check_network_health()
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error monitoring network {self.simulator_id}: {e}")
                time.sleep(5.0)
    
    def _update_network_metrics(self):
        """Update network performance metrics."""
        # Calculate average latency
        if self.latency_history:
            self.avg_latency = np.mean(self.latency_history[-100:])  # Last 100 measurements
        
        # Calculate average bandwidth
        if self.bandwidth_history:
            self.avg_bandwidth = np.mean(self.bandwidth_history[-100:])  # Last 100 measurements
    
    def _check_network_health(self):
        """Check network health and identify issues."""
        # Check for high latency (only if we have actual measurements)
        if self.latency_history and self.avg_latency > self.base_latency * 2:
            logger.warning(f"Network {self.simulator_id}: High latency detected: {self.avg_latency:.2f}ms")
        
        # Check for low bandwidth (only if we have actual measurements)
        # If no measurements yet, use base_bandwidth for comparison
        current_bandwidth = self.avg_bandwidth if self.bandwidth_history else self.base_bandwidth
        if self.bandwidth_history and current_bandwidth < self.base_bandwidth * 0.5:
            logger.warning(f"Network {self.simulator_id}: Low bandwidth detected: {current_bandwidth:.2f}Mbps")
        
        # Check for high packet loss
        if self.packet_loss_history:
            recent_loss_rate = np.mean(self.packet_loss_history[-10:])
            if recent_loss_rate > self.packet_loss_rate * 5:
                logger.warning(f"Network {self.simulator_id}: High packet loss detected: {recent_loss_rate:.4f}")
    
    def add_node(self, node_id: str, node_type: str, location: Tuple[float, float]):
        """
        Add a node to the network.
        
        Args:
            node_id: Node identifier
            node_type: Type of node (edge_device, fog_node, cloud_server)
            location: Node location (x, y coordinates)
        """
        self.network_graph.add_node(node_id, type=node_type, location=location)
        self.connected_nodes.add(node_id)
        logger.info(f"Network {self.simulator_id}: Added node {node_id} of type {node_type}")
    
    def remove_node(self, node_id: str):
        """
        Remove a node from the network.
        
        Args:
            node_id: Node identifier
        """
        if node_id in self.network_graph:
            self.network_graph.remove_node(node_id)
            self.connected_nodes.discard(node_id)
            logger.info(f"Network {self.simulator_id}: Removed node {node_id}")
    
    def add_link(self, node1: str, node2: str, link_properties: Dict[str, Any]):
        """
        Add a link between two nodes.
        
        Args:
            node1: First node identifier
            node2: Second node identifier
            link_properties: Link properties (latency, bandwidth, etc.)
        """
        if node1 in self.network_graph and node2 in self.network_graph:
            self.network_graph.add_edge(node1, node2, **link_properties)
            self.link_properties[(node1, node2)] = link_properties
            logger.info(f"Network {self.simulator_id}: Added link between {node1} and {node2}")
        else:
            logger.warning(f"Network {self.simulator_id}: Cannot add link - nodes not found")
    
    def remove_link(self, node1: str, node2: str):
        """
        Remove a link between two nodes.
        
        Args:
            node1: First node identifier
            node2: Second node identifier
        """
        if self.network_graph.has_edge(node1, node2):
            self.network_graph.remove_edge(node1, node2)
            self.link_properties.pop((node1, node2), None)
            logger.info(f"Network {self.simulator_id}: Removed link between {node1} and {node2}")
    
    def calculate_path_latency(self, source: str, destination: str) -> float:
        """
        Calculate latency for a path between two nodes.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            
        Returns:
            Path latency in ms
        """
        try:
            if source == destination:
                return 0.0
            
            # Find shortest path
            path = nx.shortest_path(self.network_graph, source, destination)
            
            # Calculate total latency
            total_latency = 0.0
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                link_latency = self.link_properties.get((node1, node2), {}).get('latency', self.base_latency)
                total_latency += link_latency
            
            return total_latency
            
        except nx.NetworkXNoPath:
            logger.warning(f"Network {self.simulator_id}: No path found between {source} and {destination}")
            return float('inf')
    
    def calculate_path_bandwidth(self, source: str, destination: str) -> float:
        """
        Calculate bandwidth for a path between two nodes.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            
        Returns:
            Path bandwidth in Mbps
        """
        try:
            if source == destination:
                return float('inf')
            
            # Find shortest path
            path = nx.shortest_path(self.network_graph, source, destination)
            
            # Calculate minimum bandwidth (bottleneck)
            min_bandwidth = float('inf')
            for i in range(len(path) - 1):
                node1, node2 = path[i], path[i + 1]
                link_bandwidth = self.link_properties.get((node1, node2), {}).get('bandwidth', self.base_bandwidth)
                min_bandwidth = min(min_bandwidth, link_bandwidth)
            
            return min_bandwidth
            
        except nx.NetworkXNoPath:
            logger.warning(f"Network {self.simulator_id}: No path found between {source} and {destination}")
            return 0.0
    
    def simulate_transmission(self, source: str, destination: str, data_size: float) -> Dict[str, Any]:
        """
        Simulate data transmission between two nodes.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            data_size: Data size in MB
            
        Returns:
            Transmission simulation results
        """
        start_time = time.time()
        
        try:
            # Calculate path properties
            path_latency = self.calculate_path_latency(source, destination)
            path_bandwidth = self.calculate_path_bandwidth(source, destination)
            
            # Simulate transmission delay
            transmission_delay = (data_size * 8) / (path_bandwidth * 1024 * 1024)  # Convert to seconds
            total_delay = (path_latency / 1000.0) + transmission_delay
            
            # Add jitter
            jitter_delay = random.uniform(-self.jitter, self.jitter) / 1000.0
            total_delay += jitter_delay
            
            # Simulate packet loss
            packet_loss = random.random() < self.packet_loss_rate
            
            # Update metrics
            self.total_packets_sent += 1
            if not packet_loss:
                self.total_packets_received += 1
                self.total_bytes_transferred += data_size
            else:
                self.total_packets_lost += 1
            
            # Update history
            self.latency_history.append(path_latency)
            self.bandwidth_history.append(path_bandwidth)
            self.packet_loss_history.append(1.0 if packet_loss else 0.0)
            
            # Limit history size
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-1000:]
            if len(self.bandwidth_history) > 1000:
                self.bandwidth_history = self.bandwidth_history[-1000:]
            if len(self.packet_loss_history) > 1000:
                self.packet_loss_history = self.packet_loss_history[-1000:]
            
            # Simulate actual delay
            time.sleep(total_delay)
            
            return {
                'success': not packet_loss,
                'latency': path_latency,
                'bandwidth': path_bandwidth,
                'transmission_delay': transmission_delay,
                'total_delay': total_delay,
                'packet_loss': packet_loss,
                'data_size': data_size
            }
            
        except Exception as e:
            logger.error(f"Network {self.simulator_id}: Transmission simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data_size': data_size
            }
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information."""
        return {
            'nodes': list(self.network_graph.nodes()),
            'edges': list(self.network_graph.edges()),
            'topology': self.topology,
            'network_type': self.network_type,
            'connected_nodes': len(self.connected_nodes)
        }
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network performance metrics."""
        return {
            'total_packets_sent': self.total_packets_sent,
            'total_packets_received': self.total_packets_received,
            'total_packets_lost': self.total_packets_lost,
            'total_bytes_transferred': self.total_bytes_transferred,
            'packet_loss_rate': self.total_packets_lost / max(1, self.total_packets_sent),
            'avg_latency': self.avg_latency,
            'avg_bandwidth': self.avg_bandwidth,
            'network_utilization': self._calculate_network_utilization()
        }
    
    def _calculate_network_utilization(self) -> float:
        """Calculate network utilization."""
        if not self.bandwidth_history:
            return 0.0
        
        # Calculate utilization based on bandwidth usage
        current_bandwidth = self.avg_bandwidth
        utilization = min(1.0, current_bandwidth / self.base_bandwidth)
        return utilization
    
    def shutdown(self):
        """Shutdown network simulator."""
        self.is_active = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info(f"Network simulator {self.simulator_id}: Shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.shutdown()


class WiFiNetwork(NetworkSimulator):
    """WiFi network simulator."""
    
    def __init__(
        self,
        simulator_id: str,
        frequency: str = "2.4GHz",
        standard: str = "802.11n",
        signal_strength: float = -50.0,  # dBm
        interference_level: float = 0.1
    ):
        """
        Initialize WiFi network simulator.
        
        Args:
            simulator_id: Unique simulator identifier
            frequency: WiFi frequency band
            standard: WiFi standard
            signal_strength: Signal strength in dBm
            interference_level: Interference level (0.0 to 1.0)
        """
        # WiFi-specific parameters
        if standard == "802.11n":
            base_bandwidth = 150.0  # Mbps
            base_latency = 5.0  # ms
        elif standard == "802.11ac":
            base_bandwidth = 433.0  # Mbps
            base_latency = 3.0  # ms
        elif standard == "802.11ax":
            base_bandwidth = 600.0  # Mbps
            base_latency = 2.0  # ms
        else:
            base_bandwidth = 54.0  # Mbps
            base_latency = 10.0  # ms
        
        # Adjust for signal strength
        signal_factor = max(0.1, (signal_strength + 100) / 50)  # Normalize signal strength
        adjusted_bandwidth = base_bandwidth * signal_factor
        adjusted_latency = base_latency / signal_factor
        
        # Adjust for interference
        interference_factor = 1.0 - interference_level
        final_bandwidth = adjusted_bandwidth * interference_factor
        final_latency = adjusted_latency / interference_factor
        
        super().__init__(
            simulator_id, "wifi", "star", final_latency, final_bandwidth,
            packet_loss_rate=0.001 * (1.0 + interference_level),
            jitter=2.0 * (1.0 + interference_level)
        )
        
        self.frequency = frequency
        self.standard = standard
        self.signal_strength = signal_strength
        self.interference_level = interference_level
    
    def update_signal_strength(self, new_strength: float):
        """
        Update signal strength and recalculate network parameters.
        
        Args:
            new_strength: New signal strength in dBm
        """
        self.signal_strength = new_strength
        
        # Recalculate network parameters
        signal_factor = max(0.1, (new_strength + 100) / 50)
        interference_factor = 1.0 - self.interference_level
        
        self.base_bandwidth *= signal_factor * interference_factor
        self.base_latency /= signal_factor * interference_factor
        
        logger.info(f"WiFi network {self.simulator_id}: Updated signal strength to {new_strength}dBm")


class CellularNetwork(NetworkSimulator):
    """Cellular network simulator."""
    
    def __init__(
        self,
        simulator_id: str,
        generation: str = "4G",
        carrier: str = "LTE",
        signal_strength: float = -70.0,  # dBm
        congestion_level: float = 0.2
    ):
        """
        Initialize cellular network simulator.
        
        Args:
            simulator_id: Unique simulator identifier
            generation: Cellular generation (3G, 4G, 5G)
            carrier: Carrier technology
            signal_strength: Signal strength in dBm
            congestion_level: Network congestion level (0.0 to 1.0)
        """
        # Cellular-specific parameters
        if generation == "5G":
            base_bandwidth = 1000.0  # Mbps
            base_latency = 1.0  # ms
        elif generation == "4G":
            base_bandwidth = 100.0  # Mbps
            base_latency = 20.0  # ms
        elif generation == "3G":
            base_bandwidth = 10.0  # Mbps
            base_latency = 100.0  # ms
        else:
            base_bandwidth = 1.0  # Mbps
            base_latency = 500.0  # ms
        
        # Adjust for signal strength
        signal_factor = max(0.1, (signal_strength + 120) / 60)  # Normalize signal strength
        adjusted_bandwidth = base_bandwidth * signal_factor
        adjusted_latency = base_latency / signal_factor
        
        # Adjust for congestion
        congestion_factor = 1.0 - congestion_level
        final_bandwidth = adjusted_bandwidth * congestion_factor
        final_latency = adjusted_latency / congestion_factor
        
        super().__init__(
            simulator_id, "cellular", "mesh", final_latency, final_bandwidth,
            packet_loss_rate=0.005 * (1.0 + congestion_level),
            jitter=5.0 * (1.0 + congestion_level)
        )
        
        self.generation = generation
        self.carrier = carrier
        self.signal_strength = signal_strength
        self.congestion_level = congestion_level
    
    def update_congestion_level(self, new_level: float):
        """
        Update congestion level and recalculate network parameters.
        
        Args:
            new_level: New congestion level (0.0 to 1.0)
        """
        self.congestion_level = new_level
        
        # Recalculate network parameters
        congestion_factor = 1.0 - new_level
        signal_factor = max(0.1, (self.signal_strength + 120) / 60)
        
        self.base_bandwidth *= congestion_factor * signal_factor
        self.base_latency /= congestion_factor * signal_factor
        
        logger.info(f"Cellular network {self.simulator_id}: Updated congestion level to {new_level}")


class EthernetNetwork(NetworkSimulator):
    """Ethernet network simulator."""
    
    def __init__(
        self,
        simulator_id: str,
        speed: str = "1Gbps",
        cable_type: str = "Cat6",
        distance: float = 100.0  # meters
    ):
        """
        Initialize Ethernet network simulator.
        
        Args:
            simulator_id: Unique simulator identifier
            speed: Ethernet speed
            cable_type: Cable type
            distance: Cable distance in meters
        """
        # Ethernet-specific parameters
        if speed == "10Gbps":
            base_bandwidth = 10000.0  # Mbps
            base_latency = 0.1  # ms
        elif speed == "1Gbps":
            base_bandwidth = 1000.0  # Mbps
            base_latency = 0.5  # ms
        elif speed == "100Mbps":
            base_bandwidth = 100.0  # Mbps
            base_latency = 1.0  # ms
        else:
            base_bandwidth = 10.0  # Mbps
            base_latency = 5.0  # ms
        
        # Adjust for distance
        distance_factor = max(0.5, 1.0 - (distance / 1000.0))  # Degrade with distance
        final_bandwidth = base_bandwidth * distance_factor
        final_latency = base_latency / distance_factor
        
        super().__init__(
            simulator_id, "ethernet", "star", final_latency, final_bandwidth,
            packet_loss_rate=0.0001,  # Very low packet loss
            jitter=0.1  # Very low jitter
        )
        
        self.speed = speed
        self.cable_type = cable_type
        self.distance = distance


class FiberNetwork(NetworkSimulator):
    """Fiber optic network simulator."""
    
    def __init__(
        self,
        simulator_id: str,
        speed: str = "10Gbps",
        fiber_type: str = "Single-mode",
        distance: float = 1000.0  # meters
    ):
        """
        Initialize fiber optic network simulator.
        
        Args:
            simulator_id: Unique simulator identifier
            speed: Fiber speed
            fiber_type: Fiber type
            distance: Fiber distance in meters
        """
        # Fiber-specific parameters
        if speed == "100Gbps":
            base_bandwidth = 100000.0  # Mbps
            base_latency = 0.01  # ms
        elif speed == "10Gbps":
            base_bandwidth = 10000.0  # Mbps
            base_latency = 0.1  # ms
        elif speed == "1Gbps":
            base_bandwidth = 1000.0  # Mbps
            base_latency = 0.5  # ms
        else:
            base_bandwidth = 100.0  # Mbps
            base_latency = 1.0  # ms
        
        # Adjust for distance (fiber has very low attenuation)
        distance_factor = max(0.8, 1.0 - (distance / 10000.0))  # Minimal degradation
        final_bandwidth = base_bandwidth * distance_factor
        final_latency = base_latency / distance_factor
        
        super().__init__(
            simulator_id, "fiber", "mesh", final_latency, final_bandwidth,
            packet_loss_rate=0.00001,  # Extremely low packet loss
            jitter=0.01  # Extremely low jitter
        )
        
        self.speed = speed
        self.fiber_type = fiber_type
        self.distance = distance
