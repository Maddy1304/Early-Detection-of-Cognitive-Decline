"""
Infrastructure simulation for edge-fog-cloud architecture.

This module provides simulation components for:
- Edge devices (smartphones, wearables, IoT sensors)
- Fog nodes (clinic servers, local gateways)
- Cloud servers (global aggregation, model distribution)
- Network simulation and communication
"""

from .edge_device import EdgeDevice, SmartphoneDevice, WearableDevice, IoTSensorDevice
from .fog_node import FogNode, ClinicServer, LocalGateway
from .cloud_server import CloudServer, GlobalAggregator, AnalyticsServer
from .network_simulator import NetworkSimulator, WiFiNetwork, CellularNetwork, EthernetNetwork, FiberNetwork
from .simulation_environment import SimulationEnvironment

__all__ = [
    "EdgeDevice",
    "SmartphoneDevice",
    "WearableDevice",
    "IoTSensorDevice",
    "FogNode",
    "ClinicServer",
    "LocalGateway",
    "CloudServer",
    "GlobalAggregator",
    "AnalyticsServer",
    "NetworkSimulator",
    "WiFiNetwork",
    "CellularNetwork",
    "EthernetNetwork",
    "FiberNetwork",
    "SimulationEnvironment",
]
