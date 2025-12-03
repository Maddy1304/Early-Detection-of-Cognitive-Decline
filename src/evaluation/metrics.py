"""
Evaluation metrics for cognitive decline detection.

This module provides comprehensive evaluation metrics for assessing
the performance of the federated learning system and models.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Comprehensive evaluation metrics for cognitive decline detection."""
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.metrics_history = []
        self.baseline_metrics = {}
        self.threshold_metrics = {}
        
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: Class names for reporting
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        if class_names:
            for i, class_name in enumerate(class_names):
                metrics[f'precision_{class_name}'] = precision_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
                metrics[f'recall_{class_name}'] = recall_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
                metrics[f'f1_score_{class_name}'] = f1_score(y_true, y_pred, labels=[i], average='micro', zero_division=0)
        
        # ROC AUC and PR AUC
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob)
                else:  # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob, average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Additional metrics
        metrics['specificity'] = self._calculate_specificity(cm)
        metrics['sensitivity'] = self._calculate_sensitivity(cm)
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def _calculate_specificity(self, cm: np.ndarray) -> float:
        """Calculate specificity from confusion matrix."""
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:  # Multi-class classification
            # Calculate average specificity across classes
            specificities = []
            for i in range(cm.shape[0]):
                tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(specificity)
            return np.mean(specificities)
    
    def _calculate_sensitivity(self, cm: np.ndarray) -> float:
        """Calculate sensitivity (recall) from confusion matrix."""
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:  # Multi-class classification
            # Calculate average sensitivity across classes
            sensitivities = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - cm[i, i]
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                sensitivities.append(sensitivity)
            return np.mean(sensitivities)
    
    def calculate_federated_learning_metrics(
        self,
        communication_overhead: float,
        aggregation_time: float,
        model_accuracy: float,
        privacy_score: float,
        energy_consumption: float,
        latency: float,
        throughput: float
    ) -> Dict[str, float]:
        """
        Calculate federated learning specific metrics.
        
        Args:
            communication_overhead: Communication overhead in MB
            aggregation_time: Time for model aggregation in seconds
            model_accuracy: Model accuracy
            privacy_score: Privacy preservation score (0-1)
            energy_consumption: Energy consumption in Joules
            latency: Communication latency in ms
            throughput: Data throughput in Mbps
            
        Returns:
            Dictionary of FL metrics
        """
        metrics = {}
        
        # Communication efficiency
        metrics['communication_overhead'] = communication_overhead
        metrics['communication_efficiency'] = model_accuracy / max(communication_overhead, 1.0)
        
        # Aggregation efficiency
        metrics['aggregation_time'] = aggregation_time
        metrics['aggregation_efficiency'] = model_accuracy / max(aggregation_time, 1.0)
        
        # Privacy-utility trade-off
        metrics['privacy_score'] = privacy_score
        metrics['privacy_utility_tradeoff'] = (model_accuracy + privacy_score) / 2.0
        
        # Energy efficiency
        metrics['energy_consumption'] = energy_consumption
        metrics['energy_efficiency'] = model_accuracy / max(energy_consumption, 1.0)
        
        # Network performance
        metrics['latency'] = latency
        metrics['throughput'] = throughput
        metrics['network_efficiency'] = throughput / max(latency, 1.0)
        
        # Overall system efficiency
        metrics['system_efficiency'] = (
            metrics['communication_efficiency'] +
            metrics['aggregation_efficiency'] +
            metrics['privacy_utility_tradeoff'] +
            metrics['energy_efficiency'] +
            metrics['network_efficiency']
        ) / 5.0
        
        return metrics
    
    def calculate_privacy_metrics(
        self,
        epsilon: float,
        delta: float,
        noise_scale: float,
        data_sensitivity: float
    ) -> Dict[str, float]:
        """
        Calculate privacy preservation metrics.
        
        Args:
            epsilon: Privacy budget (epsilon)
            delta: Privacy parameter (delta)
            noise_scale: Noise scale parameter
            data_sensitivity: Data sensitivity measure
            
        Returns:
            Dictionary of privacy metrics
        """
        metrics = {}
        
        # Differential privacy metrics
        metrics['epsilon'] = epsilon
        metrics['delta'] = delta
        metrics['noise_scale'] = noise_scale
        metrics['data_sensitivity'] = data_sensitivity
        
        # Privacy score (higher is better)
        metrics['privacy_score'] = 1.0 / (1.0 + epsilon)  # Lower epsilon = higher privacy
        
        # Privacy loss
        metrics['privacy_loss'] = epsilon * data_sensitivity
        
        # Utility-privacy trade-off
        metrics['utility_privacy_tradeoff'] = 1.0 / (1.0 + epsilon * noise_scale)
        
        # Privacy budget utilization
        metrics['privacy_budget_utilization'] = epsilon / 10.0  # Assuming max epsilon of 10
        
        return metrics
    
    def calculate_energy_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        network_usage: float,
        processing_time: float,
        device_type: str
    ) -> Dict[str, float]:
        """
        Calculate energy consumption metrics.
        
        Args:
            cpu_usage: CPU usage percentage
            memory_usage: Memory usage percentage
            network_usage: Network usage percentage
            processing_time: Processing time in seconds
            device_type: Type of device (smartphone, wearable, etc.)
            
        Returns:
            Dictionary of energy metrics
        """
        metrics = {}
        
        # Device-specific energy coefficients
        energy_coefficients = {
            'smartphone': {'cpu': 0.1, 'memory': 0.05, 'network': 0.2, 'base': 0.5},
            'wearable': {'cpu': 0.05, 'memory': 0.02, 'network': 0.1, 'base': 0.2},
            'fog_node': {'cpu': 0.2, 'memory': 0.1, 'network': 0.3, 'base': 1.0},
            'cloud_server': {'cpu': 0.5, 'memory': 0.2, 'network': 0.4, 'base': 5.0}
        }
        
        coeffs = energy_coefficients.get(device_type, energy_coefficients['smartphone'])
        
        # Calculate energy consumption
        cpu_energy = cpu_usage * coeffs['cpu'] * processing_time
        memory_energy = memory_usage * coeffs['memory'] * processing_time
        network_energy = network_usage * coeffs['network'] * processing_time
        base_energy = coeffs['base'] * processing_time
        
        total_energy = cpu_energy + memory_energy + network_energy + base_energy
        
        metrics['cpu_energy'] = cpu_energy
        metrics['memory_energy'] = memory_energy
        metrics['network_energy'] = network_energy
        metrics['base_energy'] = base_energy
        metrics['total_energy'] = total_energy
        
        # Energy efficiency
        metrics['energy_efficiency'] = 1.0 / max(total_energy, 0.001)
        
        # Energy breakdown
        metrics['cpu_energy_ratio'] = cpu_energy / max(total_energy, 0.001)
        metrics['memory_energy_ratio'] = memory_energy / max(total_energy, 0.001)
        metrics['network_energy_ratio'] = network_energy / max(total_energy, 0.001)
        metrics['base_energy_ratio'] = base_energy / max(total_energy, 0.001)
        
        return metrics
    
    def calculate_communication_metrics(
        self,
        data_size: float,
        transmission_time: float,
        latency: float,
        bandwidth: float,
        packet_loss: float
    ) -> Dict[str, float]:
        """
        Calculate communication performance metrics.
        
        Args:
            data_size: Data size in MB
            transmission_time: Transmission time in seconds
            latency: Network latency in ms
            bandwidth: Network bandwidth in Mbps
            packet_loss: Packet loss rate (0-1)
            
        Returns:
            Dictionary of communication metrics
        """
        metrics = {}
        
        # Basic communication metrics
        metrics['data_size'] = data_size
        metrics['transmission_time'] = transmission_time
        metrics['latency'] = latency
        metrics['bandwidth'] = bandwidth
        metrics['packet_loss'] = packet_loss
        
        # Throughput
        metrics['throughput'] = (data_size * 8) / max(transmission_time, 0.001)  # Mbps
        
        # Efficiency metrics
        metrics['bandwidth_utilization'] = metrics['throughput'] / max(bandwidth, 0.001)
        metrics['latency_efficiency'] = 1.0 / max(latency, 0.001)
        metrics['reliability'] = 1.0 - packet_loss
        
        # Communication cost
        metrics['communication_cost'] = data_size * (1.0 + packet_loss)  # Account for retransmissions
        
        # Quality of service
        metrics['qos_score'] = (
            metrics['bandwidth_utilization'] * 0.3 +
            metrics['latency_efficiency'] * 0.3 +
            metrics['reliability'] * 0.4
        )
        
        return metrics
    
    def calculate_system_metrics(
        self,
        total_time: float,
        num_devices: int,
        num_rounds: int,
        model_size: float,
        data_processed: float
    ) -> Dict[str, float]:
        """
        Calculate overall system performance metrics.
        
        Args:
            total_time: Total execution time in seconds
            num_devices: Number of participating devices
            num_rounds: Number of federated learning rounds
            model_size: Model size in MB
            data_processed: Total data processed in MB
            
        Returns:
            Dictionary of system metrics
        """
        metrics = {}
        
        # Basic system metrics
        metrics['total_time'] = total_time
        metrics['num_devices'] = num_devices
        metrics['num_rounds'] = num_rounds
        metrics['model_size'] = model_size
        metrics['data_processed'] = data_processed
        
        # Scalability metrics
        metrics['devices_per_second'] = num_devices / max(total_time, 0.001)
        metrics['rounds_per_second'] = num_rounds / max(total_time, 0.001)
        metrics['data_processing_rate'] = data_processed / max(total_time, 0.001)  # MB/s
        
        # Efficiency metrics
        metrics['time_per_round'] = total_time / max(num_rounds, 1)
        metrics['time_per_device'] = total_time / max(num_devices, 1)
        metrics['data_per_device'] = data_processed / max(num_devices, 1)
        
        # Resource utilization
        metrics['model_efficiency'] = 1.0 / max(model_size, 0.001)
        metrics['throughput_efficiency'] = data_processed / max(total_time * model_size, 0.001)
        
        return metrics
    
    def compare_models(
        self,
        model_results: Dict[str, Dict[str, float]],
        baseline_model: str = "baseline"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models and calculate relative improvements.
        
        Args:
            model_results: Dictionary of model results
            baseline_model: Name of baseline model
            
        Returns:
            Dictionary of comparison results
        """
        if baseline_model not in model_results:
            logger.warning(f"Baseline model {baseline_model} not found in results")
            return {}
        
        baseline_metrics = model_results[baseline_model]
        comparisons = {}
        
        for model_name, model_metrics in model_results.items():
            if model_name == baseline_model:
                continue
            
            comparison = {}
            for metric_name, metric_value in model_metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]
                    if baseline_value != 0:
                        improvement = ((metric_value - baseline_value) / baseline_value) * 100
                        comparison[f'{metric_name}_improvement'] = improvement
                        comparison[f'{metric_name}_relative'] = metric_value / baseline_value
                    else:
                        comparison[f'{metric_name}_improvement'] = 0.0
                        comparison[f'{metric_name}_relative'] = 1.0
            
            comparisons[model_name] = comparison
        
        return comparisons
    
    def generate_evaluation_report(
        self,
        classification_metrics: Dict[str, float],
        fl_metrics: Dict[str, float],
        privacy_metrics: Dict[str, float],
        energy_metrics: Dict[str, float],
        communication_metrics: Dict[str, float],
        system_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            classification_metrics: Classification performance metrics
            fl_metrics: Federated learning metrics
            privacy_metrics: Privacy preservation metrics
            energy_metrics: Energy consumption metrics
            communication_metrics: Communication performance metrics
            system_metrics: System performance metrics
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'timestamp': time.time(),
            'classification_performance': classification_metrics,
            'federated_learning_performance': fl_metrics,
            'privacy_preservation': privacy_metrics,
            'energy_consumption': energy_metrics,
            'communication_performance': communication_metrics,
            'system_performance': system_metrics,
            'summary': self._generate_summary(classification_metrics, fl_metrics, privacy_metrics)
        }
        
        return report
    
    def _generate_summary(
        self,
        classification_metrics: Dict[str, float],
        fl_metrics: Dict[str, float],
        privacy_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate summary of evaluation results."""
        summary = {
            'overall_score': 0.0,
            'performance_grade': 'F',
            'recommendations': [],
            'strengths': [],
            'weaknesses': []
        }
        
        # Calculate overall score
        accuracy = classification_metrics.get('accuracy', 0.0)
        privacy_score = privacy_metrics.get('privacy_score', 0.0)
        system_efficiency = fl_metrics.get('system_efficiency', 0.0)
        
        overall_score = (accuracy * 0.4 + privacy_score * 0.3 + system_efficiency * 0.3)
        summary['overall_score'] = overall_score
        
        # Determine performance grade
        if overall_score >= 0.9:
            summary['performance_grade'] = 'A'
        elif overall_score >= 0.8:
            summary['performance_grade'] = 'B'
        elif overall_score >= 0.7:
            summary['performance_grade'] = 'C'
        elif overall_score >= 0.6:
            summary['performance_grade'] = 'D'
        else:
            summary['performance_grade'] = 'F'
        
        # Generate recommendations
        if accuracy < 0.8:
            summary['recommendations'].append("Improve model accuracy through better feature engineering or model architecture")
        if privacy_score < 0.7:
            summary['recommendations'].append("Enhance privacy preservation techniques")
        if system_efficiency < 0.6:
            summary['recommendations'].append("Optimize system efficiency and resource utilization")
        
        # Identify strengths and weaknesses
        if accuracy >= 0.85:
            summary['strengths'].append("High model accuracy")
        if privacy_score >= 0.8:
            summary['strengths'].append("Strong privacy preservation")
        if system_efficiency >= 0.7:
            summary['strengths'].append("Good system efficiency")
        
        if accuracy < 0.7:
            summary['weaknesses'].append("Low model accuracy")
        if privacy_score < 0.6:
            summary['weaknesses'].append("Weak privacy preservation")
        if system_efficiency < 0.5:
            summary['weaknesses'].append("Poor system efficiency")
        
        return summary
    
    def plot_metrics(
        self,
        metrics_history: List[Dict[str, float]],
        output_path: str,
        title: str = "Metrics Over Time"
    ):
        """
        Plot metrics over time.
        
        Args:
            metrics_history: List of metrics dictionaries
            output_path: Path to save the plot
            title: Plot title
        """
        if not metrics_history:
            logger.warning("No metrics history provided for plotting")
            return
        
        # Extract metrics
        timestamps = list(range(len(metrics_history)))
        metric_names = list(metrics_history[0].keys())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot accuracy
        if 'accuracy' in metric_names:
            accuracy_values = [m.get('accuracy', 0.0) for m in metrics_history]
            axes[0, 0].plot(timestamps, accuracy_values, 'b-', linewidth=2)
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_xlabel('Time Step')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True)
        
        # Plot loss
        if 'loss' in metric_names:
            loss_values = [m.get('loss', 0.0) for m in metrics_history]
            axes[0, 1].plot(timestamps, loss_values, 'r-', linewidth=2)
            axes[0, 1].set_title('Loss Over Time')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Plot communication overhead
        if 'communication_overhead' in metric_names:
            comm_values = [m.get('communication_overhead', 0.0) for m in metrics_history]
            axes[1, 0].plot(timestamps, comm_values, 'g-', linewidth=2)
            axes[1, 0].set_title('Communication Overhead Over Time')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Communication Overhead (MB)')
            axes[1, 0].grid(True)
        
        # Plot privacy score
        if 'privacy_score' in metric_names:
            privacy_values = [m.get('privacy_score', 0.0) for m in metrics_history]
            axes[1, 1].plot(timestamps, privacy_values, 'm-', linewidth=2)
            axes[1, 1].set_title('Privacy Score Over Time')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Privacy Score')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics plot saved to {output_path}")
    
    def save_metrics(self, metrics: Dict[str, Any], output_path: str):
        """Save metrics to file."""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
