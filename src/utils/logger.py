"""
Logging utilities for the cognitive decline detection system.

This module provides centralized logging configuration and utilities
for the entire system.
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime
import logging.handlers


def setup_logger(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup centralized logger for the system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory to store log files
        log_file: Specific log file name
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
        if log_file:
            log_path = os.path.join(log_dir, log_file)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"cognitive_decline_detection_{timestamp}.log")
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
        self.logger.info(f"Started operation: {operation}")
    
    def end_timer(self, operation: str):
        """End timing an operation and log the duration."""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            self.logger.info(f"Completed operation: {operation} in {duration:.2f} seconds")
            del self.start_times[operation]
            return duration
        else:
            self.logger.warning(f"Timer for operation {operation} was not started")
            return 0.0
    
    def log_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log a performance metric."""
        self.logger.info(f"Metric - {metric_name}: {value:.3f} {unit}")
    
    def log_system_info(self, info: dict):
        """Log system information."""
        for key, value in info.items():
            self.logger.info(f"System - {key}: {value}")


class SecurityLogger:
    """Logger for security-related events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_authentication(self, user_id: str, success: bool, details: str = ""):
        """Log authentication events."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Authentication {status} - User: {user_id}, Details: {details}")
    
    def log_authorization(self, user_id: str, resource: str, success: bool):
        """Log authorization events."""
        status = "GRANTED" if success else "DENIED"
        self.logger.info(f"Authorization {status} - User: {user_id}, Resource: {resource}")
    
    def log_data_access(self, user_id: str, data_type: str, operation: str):
        """Log data access events."""
        self.logger.info(f"Data Access - User: {user_id}, Type: {data_type}, Operation: {operation}")
    
    def log_privacy_event(self, event_type: str, details: str):
        """Log privacy-related events."""
        self.logger.info(f"Privacy Event - Type: {event_type}, Details: {details}")
    
    def log_security_alert(self, alert_type: str, severity: str, details: str):
        """Log security alerts."""
        self.logger.warning(f"Security Alert - Type: {alert_type}, Severity: {severity}, Details: {details}")


class FederatedLearningLogger:
    """Logger for federated learning events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_round_start(self, round_number: int, total_rounds: int):
        """Log the start of a federated learning round."""
        self.logger.info(f"FL Round {round_number}/{total_rounds} started")
    
    def log_round_end(self, round_number: int, metrics: dict):
        """Log the end of a federated learning round."""
        self.logger.info(f"FL Round {round_number} completed - Metrics: {metrics}")
    
    def log_client_update(self, client_id: str, update_size: float, accuracy: float):
        """Log client update."""
        self.logger.info(f"Client {client_id} update - Size: {update_size:.2f} MB, Accuracy: {accuracy:.3f}")
    
    def log_aggregation(self, server_id: str, num_clients: int, aggregation_time: float):
        """Log model aggregation."""
        self.logger.info(f"Server {server_id} aggregated {num_clients} clients in {aggregation_time:.2f}s")
    
    def log_model_distribution(self, server_id: str, num_clients: int, model_size: float):
        """Log model distribution."""
        self.logger.info(f"Server {server_id} distributed model ({model_size:.2f} MB) to {num_clients} clients")
    
    def log_communication(self, source: str, destination: str, data_size: float, latency: float):
        """Log communication events."""
        self.logger.info(f"Communication - {source} -> {destination}, Size: {data_size:.2f} MB, Latency: {latency:.2f} ms")


class DataLogger:
    """Logger for data processing events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_data_loading(self, dataset_name: str, num_samples: int, data_size: float):
        """Log data loading events."""
        self.logger.info(f"Data Loading - Dataset: {dataset_name}, Samples: {num_samples}, Size: {data_size:.2f} MB")
    
    def log_data_preprocessing(self, data_type: str, num_samples: int, processing_time: float):
        """Log data preprocessing events."""
        self.logger.info(f"Data Preprocessing - Type: {data_type}, Samples: {num_samples}, Time: {processing_time:.2f}s")
    
    def log_data_quality(self, dataset_name: str, quality_score: float, issues: list):
        """Log data quality assessment."""
        self.logger.info(f"Data Quality - Dataset: {dataset_name}, Score: {quality_score:.3f}, Issues: {len(issues)}")
    
    def log_data_split(self, dataset_name: str, train_size: int, val_size: int, test_size: int):
        """Log data splitting."""
        total_size = train_size + val_size + test_size
        self.logger.info(f"Data Split - Dataset: {dataset_name}, Train: {train_size}, Val: {val_size}, Test: {test_size}, Total: {total_size}")


class ModelLogger:
    """Logger for model-related events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_model_creation(self, model_type: str, parameters: dict):
        """Log model creation."""
        self.logger.info(f"Model Created - Type: {model_type}, Parameters: {parameters}")
    
    def log_model_training(self, model_id: str, epoch: int, loss: float, accuracy: float):
        """Log model training progress."""
        self.logger.info(f"Model Training - ID: {model_id}, Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def log_model_evaluation(self, model_id: str, metrics: dict):
        """Log model evaluation."""
        self.logger.info(f"Model Evaluation - ID: {model_id}, Metrics: {metrics}")
    
    def log_model_save(self, model_id: str, file_path: str, model_size: float):
        """Log model saving."""
        self.logger.info(f"Model Saved - ID: {model_id}, Path: {file_path}, Size: {model_size:.2f} MB")
    
    def log_model_load(self, model_id: str, file_path: str, load_time: float):
        """Log model loading."""
        self.logger.info(f"Model Loaded - ID: {model_id}, Path: {file_path}, Time: {load_time:.2f}s")


class NetworkLogger:
    """Logger for network-related events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_connection(self, node1: str, node2: str, connection_type: str):
        """Log network connections."""
        self.logger.info(f"Network Connection - {node1} <-> {node2}, Type: {connection_type}")
    
    def log_disconnection(self, node1: str, node2: str, reason: str = ""):
        """Log network disconnections."""
        self.logger.info(f"Network Disconnection - {node1} <-> {node2}, Reason: {reason}")
    
    def log_network_performance(self, network_id: str, latency: float, bandwidth: float, packet_loss: float):
        """Log network performance metrics."""
        self.logger.info(f"Network Performance - ID: {network_id}, Latency: {latency:.2f} ms, Bandwidth: {bandwidth:.2f} Mbps, Packet Loss: {packet_loss:.4f}")
    
    def log_network_issue(self, network_id: str, issue_type: str, severity: str, details: str):
        """Log network issues."""
        self.logger.warning(f"Network Issue - ID: {network_id}, Type: {issue_type}, Severity: {severity}, Details: {details}")


def create_logger_hierarchy():
    """Create a hierarchy of loggers for different components."""
    # Main logger
    main_logger = get_logger("cognitive_decline_detection")
    
    # Component loggers
    component_loggers = {
        "infrastructure": get_logger("cognitive_decline_detection.infrastructure"),
        "federated_learning": get_logger("cognitive_decline_detection.federated_learning"),
        "models": get_logger("cognitive_decline_detection.models"),
        "data": get_logger("cognitive_decline_detection.data"),
        "evaluation": get_logger("cognitive_decline_detection.evaluation"),
        "utils": get_logger("cognitive_decline_detection.utils")
    }
    
    return main_logger, component_loggers


def log_system_startup():
    """Log system startup information."""
    logger = get_logger("cognitive_decline_detection")
    logger.info("=" * 60)
    logger.info("COGNITIVE DECLINE DETECTION SYSTEM")
    logger.info("=" * 60)
    logger.info("System starting up...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("=" * 60)


def log_system_shutdown():
    """Log system shutdown information."""
    logger = get_logger("cognitive_decline_detection")
    logger.info("=" * 60)
    logger.info("System shutting down...")
    logger.info("=" * 60)
