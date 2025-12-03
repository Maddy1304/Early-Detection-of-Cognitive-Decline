"""
Cloud server simulation for cognitive decline detection.

This module implements cloud servers that provide global aggregation,
model distribution, and analytics for the federated learning system.
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


class CloudServer(ABC):
    """Base class for cloud servers."""
    
    def __init__(
        self,
        server_id: str,
        server_type: str,
        location: str,
        capacity: int = 1000,  # Max number of fog nodes
        processing_power: str = "ultra_high",
        memory_limit: int = 131072,  # MB
        storage_limit: int = 1000000,  # MB
        network_bandwidth: float = 1000.0,  # Mbps
        latency: float = 5.0  # ms
    ):
        """
        Initialize cloud server.
        
        Args:
            server_id: Unique server identifier
            server_type: Type of cloud server (global_aggregator, analytics_server)
            location: Server location
            capacity: Maximum number of fog nodes
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        self.server_id = server_id
        self.server_type = server_type
        self.location = location
        self.capacity = capacity
        self.processing_power = processing_power
        self.memory_limit = memory_limit
        self.storage_limit = storage_limit
        self.network_bandwidth = network_bandwidth
        self.latency = latency
        
        # Server state
        self.is_active = True
        self.is_connected = False
        self.current_load = 0
        self.connected_fog_nodes = set()
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
        self.analytics_time = 0.0
        
        # Data storage
        self.global_models = {}
        self.aggregated_updates = {}
        self.fog_updates = defaultdict(list)
        self.analytics_data = {}
        
        # Global aggregation settings
        self.aggregation_interval = 3600  # seconds
        self.min_fog_nodes = 3
        self.max_fog_nodes = 100
        self.last_aggregation_time = time.time()
        
        # Threading
        self.aggregation_thread = None
        self.analytics_thread = None
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Initialize server
        self._initialize_server()
    
    def _initialize_server(self):
        """Initialize cloud server components."""
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
        
        # Start analytics thread
        self.analytics_thread = threading.Thread(target=self._analytics_loop)
        self.analytics_thread.daemon = True
        self.analytics_thread.start()
        
        logger.info(f"Initialized {self.server_type} cloud server: {self.server_id}")
    
    def _monitor_resources(self):
        """Monitor cloud server resources."""
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
                    logger.warning(f"Cloud server {self.server_id}: High memory usage: {self.memory_usage:.1f}%")
                
                if self.cpu_usage > 90:
                    logger.warning(f"Cloud server {self.server_id}: High CPU usage: {self.cpu_usage:.1f}%")
                
                # Update current load
                self.current_load = len(self.connected_fog_nodes)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error monitoring resources for cloud server {self.server_id}: {e}")
                time.sleep(5.0)
    
    def _aggregation_loop(self):
        """Main global aggregation loop."""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if global aggregation should be performed
                if (current_time - self.last_aggregation_time >= self.aggregation_interval and
                    len(self.fog_updates) >= self.min_fog_nodes):
                    
                    self._perform_global_aggregation()
                    self.last_aggregation_time = current_time
                
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in aggregation loop for cloud server {self.server_id}: {e}")
                time.sleep(60.0)
    
    def _analytics_loop(self):
        """Main analytics loop."""
        while not self.stop_event.is_set():
            try:
                # Perform analytics on collected data
                self._perform_analytics()
                
                time.sleep(300.0)  # Run analytics every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in analytics loop for cloud server {self.server_id}: {e}")
                time.sleep(300.0)
    
    def _perform_global_aggregation(self):
        """Perform global aggregation of fog node updates."""
        if not self.fog_updates:
            return
        
        start_time = time.time()
        
        try:
            # Aggregate updates from fog nodes
            global_update = self._aggregate_fog_updates()
            
            if global_update:
                # Update global model
                self._update_global_model(global_update)
                
                # Store aggregated update
                self.aggregated_updates[time.time()] = global_update
                
                # Distribute updated model to fog nodes
                self._distribute_global_model()
                
                # Clear fog updates
                self.fog_updates.clear()
                
                aggregation_time = time.time() - start_time
                self.aggregation_time += aggregation_time
                
                logger.info(f"Cloud server {self.server_id}: Global aggregation completed in {aggregation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Cloud server {self.server_id}: Global aggregation failed: {e}")
    
    def _aggregate_fog_updates(self) -> Optional[Dict[str, Any]]:
        """Aggregate updates from fog nodes."""
        if not self.fog_updates:
            return None
        
        # Simple averaging aggregation (can be enhanced with more sophisticated methods)
        global_update = {}
        
        for fog_node_id, updates in self.fog_updates.items():
            if not updates:
                continue
            
            # Average updates from this fog node
            fog_aggregated = {}
            for key in updates[0].keys():
                if isinstance(updates[0][key], torch.Tensor):
                    # Average tensor updates
                    stacked_updates = torch.stack([update[key] for update in updates])
                    fog_aggregated[key] = torch.mean(stacked_updates, dim=0)
                else:
                    # Average scalar updates
                    fog_aggregated[key] = np.mean([update[key] for update in updates])
            
            global_update[fog_node_id] = fog_aggregated
        
        return global_update
    
    def _update_global_model(self, global_update: Dict[str, Any]):
        """Update global model with aggregated updates."""
        # Simulate global model update
        # In practice, this would involve updating the actual global model
        logger.info(f"Cloud server {self.server_id}: Updated global model")
    
    def _distribute_global_model(self):
        """Distribute updated global model to fog nodes."""
        for fog_node_id in self.connected_fog_nodes:
            try:
                # Simulate communication delay
                communication_delay = self.latency / 1000.0
                time.sleep(communication_delay)
                
                # Update communication time
                self.communication_time += communication_delay
                
                logger.info(f"Cloud server {self.server_id}: Distributed global model to fog node {fog_node_id}")
                
            except Exception as e:
                logger.error(f"Cloud server {self.server_id}: Failed to distribute to fog node {fog_node_id}: {e}")
    
    def _perform_analytics(self):
        """Perform analytics on collected data."""
        start_time = time.time()
        
        try:
            # Simulate analytics processing
            analytics_time = random.uniform(1.0, 5.0)
            time.sleep(analytics_time)
            
            # Update analytics time
            self.analytics_time += analytics_time
            
            # Store analytics results
            self.analytics_data[time.time()] = {
                'performance_metrics': self._calculate_performance_metrics(),
                'system_health': self._assess_system_health(),
                'usage_statistics': self._calculate_usage_statistics()
            }
            
            logger.info(f"Cloud server {self.server_id}: Analytics completed in {analytics_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Cloud server {self.server_id}: Analytics failed: {e}")
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        return {
            'avg_aggregation_time': self.aggregation_time / max(1, len(self.aggregated_updates)),
            'avg_communication_time': self.communication_time / max(1, self.current_load),
            'avg_processing_time': self.processing_time / max(1, self.current_load),
            'avg_analytics_time': self.analytics_time / max(1, len(self.analytics_data)),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'storage_usage': self.storage_usage,
            'network_usage': self.network_usage
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess system health."""
        health_score = 100.0
        
        # Deduct points for high resource usage
        if self.cpu_usage > 80:
            health_score -= 20
        if self.memory_usage > 80:
            health_score -= 20
        if self.storage_usage > 80:
            health_score -= 20
        if self.network_usage > 80:
            health_score -= 20
        
        return {
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical',
            'issues': self._identify_issues()
        }
    
    def _identify_issues(self) -> List[str]:
        """Identify system issues."""
        issues = []
        
        if self.cpu_usage > 90:
            issues.append('High CPU usage')
        if self.memory_usage > 90:
            issues.append('High memory usage')
        if self.storage_usage > 90:
            issues.append('High storage usage')
        if self.network_usage > 90:
            issues.append('High network usage')
        if self.current_load > self.capacity * 0.9:
            issues.append('High load')
        
        return issues
    
    def _calculate_usage_statistics(self) -> Dict[str, Any]:
        """Calculate usage statistics."""
        return {
            'connected_fog_nodes': len(self.connected_fog_nodes),
            'total_updates_received': sum(len(updates) for updates in self.fog_updates.values()),
            'total_aggregations_performed': len(self.aggregated_updates),
            'total_analytics_runs': len(self.analytics_data),
            'uptime': time.time() - self.last_aggregation_time
        }
    
    def connect_fog_node(self, fog_node_id: str) -> bool:
        """
        Connect a fog node to the cloud server.
        
        Args:
            fog_node_id: Fog node identifier
            
        Returns:
            True if connection successful
        """
        if len(self.connected_fog_nodes) >= self.capacity:
            logger.warning(f"Cloud server {self.server_id}: Capacity reached, cannot connect fog node {fog_node_id}")
            return False
        
        self.connected_fog_nodes.add(fog_node_id)
        logger.info(f"Cloud server {self.server_id}: Connected fog node {fog_node_id}")
        return True
    
    def disconnect_fog_node(self, fog_node_id: str):
        """
        Disconnect a fog node from the cloud server.
        
        Args:
            fog_node_id: Fog node identifier
        """
        if fog_node_id in self.connected_fog_nodes:
            self.connected_fog_nodes.remove(fog_node_id)
            logger.info(f"Cloud server {self.server_id}: Disconnected fog node {fog_node_id}")
    
    def receive_fog_update(self, fog_node_id: str, update: Dict[str, Any]) -> bool:
        """
        Receive update from a connected fog node.
        
        Args:
            fog_node_id: Fog node identifier
            update: Update data
            
        Returns:
            True if update received successfully
        """
        if fog_node_id not in self.connected_fog_nodes:
            logger.warning(f"Cloud server {self.server_id}: Received update from unconnected fog node {fog_node_id}")
            return False
        
        try:
            # Store fog update
            self.fog_updates[fog_node_id].append(update)
            
            # Limit number of updates per fog node
            if len(self.fog_updates[fog_node_id]) > 20:
                self.fog_updates[fog_node_id] = self.fog_updates[fog_node_id][-20:]
            
            logger.debug(f"Cloud server {self.server_id}: Received update from fog node {fog_node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Cloud server {self.server_id}: Error receiving update from fog node {fog_node_id}: {e}")
            return False
    
    def send_global_model(self, fog_node_id: str, model_state: Dict[str, torch.Tensor]) -> bool:
        """
        Send global model to a connected fog node.
        
        Args:
            fog_node_id: Fog node identifier
            model_state: Global model state
            
        Returns:
            True if model sent successfully
        """
        if fog_node_id not in self.connected_fog_nodes:
            logger.warning(f"Cloud server {self.server_id}: Cannot send model to unconnected fog node {fog_node_id}")
            return False
        
        try:
            # Simulate communication delay
            communication_delay = self.latency / 1000.0
            time.sleep(communication_delay)
            
            # Update communication time
            self.communication_time += communication_delay
            
            logger.info(f"Cloud server {self.server_id}: Sent global model to fog node {fog_node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Cloud server {self.server_id}: Error sending model to fog node {fog_node_id}: {e}")
            return False
    
    def store_global_model(self, model_id: str, model: nn.Module):
        """Store global model."""
        self.global_models[model_id] = model
        logger.info(f"Cloud server {self.server_id}: Stored global model {model_id}")
    
    def get_global_model(self, model_id: str) -> Optional[nn.Module]:
        """Get global model."""
        return self.global_models.get(model_id)
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data."""
        return self.analytics_data.copy()
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get cloud server information."""
        return {
            'server_id': self.server_id,
            'server_type': self.server_type,
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
            'connected_fog_nodes': len(self.connected_fog_nodes),
            'connected_cloud_servers': len(self.connected_cloud_servers),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'storage_usage': self.storage_usage,
            'network_usage': self.network_usage,
            'aggregation_time': self.aggregation_time,
            'communication_time': self.communication_time,
            'processing_time': self.processing_time,
            'analytics_time': self.analytics_time
        }
    
    def connect(self):
        """Connect cloud server to network."""
        self.is_connected = True
        logger.info(f"Cloud server {self.server_id}: Connected to network")
    
    def disconnect(self):
        """Disconnect cloud server from network."""
        self.is_connected = False
        logger.info(f"Cloud server {self.server_id}: Disconnected from network")
    
    def shutdown(self):
        """Shutdown cloud server."""
        self.is_active = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=5.0)
        
        if self.analytics_thread and self.analytics_thread.is_alive():
            self.analytics_thread.join(timeout=5.0)
        
        logger.info(f"Cloud server {self.server_id}: Shutdown complete")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.shutdown()


class GlobalAggregator(CloudServer):
    """Global aggregator implementation."""
    
    def __init__(
        self,
        server_id: str,
        location: str,
        capacity: int = 1000,
        processing_power: str = "ultra_high",
        memory_limit: int = 262144,
        storage_limit: int = 2000000,
        network_bandwidth: float = 2000.0,
        latency: float = 2.0
    ):
        """
        Initialize global aggregator.
        
        Args:
            server_id: Unique server identifier
            location: Server location
            capacity: Maximum number of fog nodes
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            server_id, "global_aggregator", location, capacity, processing_power,
            memory_limit, storage_limit, network_bandwidth, latency
        )
        
        # Global aggregator-specific capabilities
        self.global_model_registry = {}
        self.federated_learning_config = {}
        self.privacy_settings = {}
        
        # Aggregation settings
        self.aggregation_interval = 1800  # 30 minutes
        self.min_fog_nodes = 5
        self.max_fog_nodes = 100
    
    def register_global_model(self, model_id: str, model: nn.Module, config: Dict[str, Any]):
        """
        Register a global model for federated learning.
        
        Args:
            model_id: Model identifier
            model: Global model
            config: Model configuration
        """
        self.global_model_registry[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time(),
            'last_updated': time.time()
        }
        logger.info(f"Global aggregator {self.server_id}: Registered global model {model_id}")
    
    def get_global_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get global model configuration.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model configuration or None
        """
        if model_id in self.global_model_registry:
            return self.global_model_registry[model_id]['config']
        return None
    
    def update_global_model(self, model_id: str, updates: Dict[str, Any]):
        """
        Update global model with aggregated updates.
        
        Args:
            model_id: Model identifier
            updates: Aggregated updates
        """
        if model_id in self.global_model_registry:
            # Simulate model update
            self.global_model_registry[model_id]['last_updated'] = time.time()
            logger.info(f"Global aggregator {self.server_id}: Updated global model {model_id}")
        else:
            logger.warning(f"Global aggregator {self.server_id}: Model {model_id} not found")
    
    def configure_federated_learning(self, config: Dict[str, Any]):
        """
        Configure federated learning parameters.
        
        Args:
            config: Federated learning configuration
        """
        self.federated_learning_config.update(config)
        logger.info(f"Global aggregator {self.server_id}: Updated federated learning configuration")
    
    def get_federated_learning_config(self) -> Dict[str, Any]:
        """Get federated learning configuration."""
        return self.federated_learning_config.copy()
    
    def configure_privacy_settings(self, settings: Dict[str, Any]):
        """
        Configure privacy settings.
        
        Args:
            settings: Privacy settings
        """
        self.privacy_settings.update(settings)
        logger.info(f"Global aggregator {self.server_id}: Updated privacy settings")
    
    def get_privacy_settings(self) -> Dict[str, Any]:
        """Get privacy settings."""
        return self.privacy_settings.copy()


class AnalyticsServer(CloudServer):
    """Analytics server implementation."""
    
    def __init__(
        self,
        server_id: str,
        location: str,
        capacity: int = 500,
        processing_power: str = "high",
        memory_limit: int = 65536,
        storage_limit: int = 500000,
        network_bandwidth: float = 500.0,
        latency: float = 10.0
    ):
        """
        Initialize analytics server.
        
        Args:
            server_id: Unique server identifier
            location: Server location
            capacity: Maximum number of fog nodes
            processing_power: Processing power level
            memory_limit: Memory limit in MB
            storage_limit: Storage limit in MB
            network_bandwidth: Network bandwidth in Mbps
            latency: Network latency in ms
        """
        super().__init__(
            server_id, "analytics_server", location, capacity, processing_power,
            memory_limit, storage_limit, network_bandwidth, latency
        )
        
        # Analytics server-specific capabilities
        self.analytics_models = {}
        self.reporting_config = {}
        self.dashboard_data = {}
        
        # Analytics settings
        self.analytics_interval = 300  # 5 minutes
        self.reporting_interval = 3600  # 1 hour
    
    def register_analytics_model(self, model_id: str, model: nn.Module, config: Dict[str, Any]):
        """
        Register an analytics model.
        
        Args:
            model_id: Model identifier
            model: Analytics model
            config: Model configuration
        """
        self.analytics_models[model_id] = {
            'model': model,
            'config': config,
            'created_at': time.time(),
            'last_used': time.time()
        }
        logger.info(f"Analytics server {self.server_id}: Registered analytics model {model_id}")
    
    def run_analytics(self, data: Any, model_id: str) -> Dict[str, Any]:
        """
        Run analytics on data using specified model.
        
        Args:
            data: Data to analyze
            model_id: Analytics model identifier
            
        Returns:
            Analytics results
        """
        if model_id not in self.analytics_models:
            logger.warning(f"Analytics server {self.server_id}: Model {model_id} not found")
            return {}
        
        start_time = time.time()
        
        try:
            # Simulate analytics processing
            processing_time = random.uniform(2.0, 10.0)
            time.sleep(processing_time)
            
            # Update model usage
            self.analytics_models[model_id]['last_used'] = time.time()
            
            # Generate analytics results
            results = {
                'model_id': model_id,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'results': self._generate_analytics_results(data)
            }
            
            logger.info(f"Analytics server {self.server_id}: Analytics completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Analytics server {self.server_id}: Analytics failed: {e}")
            return {}
    
    def _generate_analytics_results(self, data: Any) -> Dict[str, Any]:
        """Generate analytics results."""
        # Simulate analytics results
        return {
            'data_quality_score': random.uniform(0.7, 1.0),
            'anomaly_detection': random.choice([True, False]),
            'performance_metrics': {
                'accuracy': random.uniform(0.8, 0.95),
                'precision': random.uniform(0.75, 0.9),
                'recall': random.uniform(0.7, 0.85),
                'f1_score': random.uniform(0.72, 0.88)
            },
            'recommendations': [
                'Increase data collection frequency',
                'Improve data quality',
                'Update model parameters'
            ]
        }
    
    def generate_report(self, report_type: str) -> Dict[str, Any]:
        """
        Generate analytics report.
        
        Args:
            report_type: Type of report to generate
            
        Returns:
            Generated report
        """
        start_time = time.time()
        
        try:
            # Simulate report generation
            report_time = random.uniform(5.0, 15.0)
            time.sleep(report_time)
            
            report = {
                'report_type': report_type,
                'generated_at': time.time(),
                'generation_time': report_time,
                'data': self._generate_report_data(report_type)
            }
            
            logger.info(f"Analytics server {self.server_id}: Generated {report_type} report in {report_time:.2f}s")
            return report
            
        except Exception as e:
            logger.error(f"Analytics server {self.server_id}: Report generation failed: {e}")
            return {}
    
    def _generate_report_data(self, report_type: str) -> Dict[str, Any]:
        """Generate report data."""
        if report_type == 'performance':
            return {
                'system_performance': self._calculate_performance_metrics(),
                'model_performance': self._calculate_model_performance(),
                'resource_utilization': self._calculate_resource_utilization()
            }
        elif report_type == 'usage':
            return {
                'usage_statistics': self._calculate_usage_statistics(),
                'user_activity': self._calculate_user_activity(),
                'data_volume': self._calculate_data_volume()
            }
        else:
            return {'message': f'Report type {report_type} not supported'}
    
    def _calculate_model_performance(self) -> Dict[str, float]:
        """Calculate model performance metrics."""
        return {
            'avg_accuracy': random.uniform(0.8, 0.95),
            'avg_precision': random.uniform(0.75, 0.9),
            'avg_recall': random.uniform(0.7, 0.85),
            'avg_f1_score': random.uniform(0.72, 0.88),
            'model_uptime': random.uniform(0.95, 0.99)
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization."""
        return {
            'cpu_utilization': self.cpu_usage,
            'memory_utilization': self.memory_usage,
            'storage_utilization': self.storage_usage,
            'network_utilization': self.network_usage
        }
    
    def _calculate_user_activity(self) -> Dict[str, int]:
        """Calculate user activity metrics."""
        return {
            'active_users': random.randint(100, 1000),
            'total_sessions': random.randint(1000, 10000),
            'avg_session_duration': random.randint(300, 1800),
            'peak_usage_hour': random.randint(9, 17)
        }
    
    def _calculate_data_volume(self) -> Dict[str, float]:
        """Calculate data volume metrics."""
        return {
            'total_data_processed': random.uniform(1000, 10000),  # GB
            'daily_data_volume': random.uniform(100, 1000),  # GB
            'data_growth_rate': random.uniform(0.1, 0.5),  # %
            'storage_efficiency': random.uniform(0.8, 0.95)
        }
    
    def update_dashboard_data(self, data: Dict[str, Any]):
        """Update dashboard data."""
        self.dashboard_data.update(data)
        logger.debug(f"Analytics server {self.server_id}: Updated dashboard data")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.dashboard_data.copy()
