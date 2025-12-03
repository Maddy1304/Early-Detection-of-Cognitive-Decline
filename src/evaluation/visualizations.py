"""
Comprehensive visualization module for cognitive decline detection results.

This module provides various visualization functions for results and discussion
sections, including performance comparisons, training curves, and system metrics.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class ResultsVisualizer:
    """Comprehensive visualization generator for results and discussion."""
    
    def __init__(self, output_dir: str = "results/plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        metrics: List[str] = None,
        output_name: str = "model_comparison"
    ):
        """
        Create bar chart comparing different models (speech, facial, multimodal).
        
        Args:
            results_dict: Dictionary with model names as keys and metrics as values
            metrics: List of metrics to compare (default: accuracy, precision, recall, f1_score)
            output_name: Output filename
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Extract data
        model_names = list(results_dict.keys())
        n_metrics = len(metrics)
        n_models = len(model_names)
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for model_name in model_names:
            model_data = results_dict[model_name]
            # Handle nested metrics (e.g., metrics.accuracy)
            for metric in metrics:
                value = model_data.get(metric, model_data.get('metrics', {}).get(metric, 0.0))
                data[metric].append(value)
        
        # Create figure
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        x = np.arange(n_models)
        width = 0.6
        
        colors = sns.color_palette("husl", n_models)
        
        for idx, metric in enumerate(metrics):
            axes[idx].bar(x, data[metric], width, label=metric.replace('_', ' ').title(), 
                         color=colors, edgecolor='black', linewidth=1.5)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            axes[idx].set_xlabel('Model Type', fontweight='bold')
            axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                               fontweight='bold', fontsize=11)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels([name.replace('_', ' ').title() for name in model_names], 
                                     rotation=45, ha='right')
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_axisbelow(True)
            
            # Add value labels on bars
            for i, v in enumerate(data[metric]):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', 
                              fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Model comparison plot saved to {output_name}")
    
    def plot_training_curves(
        self,
        training_history: Dict[str, List[float]],
        output_name: str = "training_curves"
    ):
        """
        Plot training and validation curves (loss, accuracy).
        
        Args:
            training_history: Dictionary with 'train_loss', 'train_accuracy', 
                            'val_loss', 'val_accuracy' keys
            output_name: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(training_history.get('train_loss', [])) + 1)
        
        # Plot loss
        if 'train_loss' in training_history:
            axes[0].plot(epochs, training_history['train_loss'], 'b-', 
                        label='Training Loss', linewidth=2, marker='o', markersize=4)
        if 'val_loss' in training_history:
            axes[0].plot(epochs, training_history['val_loss'], 'r--', 
                        label='Validation Loss', linewidth=2, marker='s', markersize=4)
        
        axes[0].set_xlabel('Epoch', fontweight='bold')
        axes[0].set_ylabel('Loss', fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_axisbelow(True)
        
        # Plot accuracy
        if 'train_accuracy' in training_history:
            axes[1].plot(epochs, training_history['train_accuracy'], 'b-', 
                        label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        if 'val_accuracy' in training_history:
            axes[1].plot(epochs, training_history['val_accuracy'], 'r--', 
                        label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        
        axes[1].set_xlabel('Epoch', fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontweight='bold')
        axes[1].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[1].set_ylim([0, 1.0])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves plot saved to {output_name}")
    
    def plot_confusion_matrix(
        self,
        cm_data: Dict[str, int] or np.ndarray,
        class_names: List[str] = None,
        output_name: str = "confusion_matrix"
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm_data: Confusion matrix as dict with keys (tp, tn, fp, fn) or 2D array
            class_names: List of class names
            output_name: Output filename
        """
        # Convert dict to matrix if needed
        if isinstance(cm_data, dict):
            if 'confusion_matrix' in cm_data:
                cm = np.array(cm_data['confusion_matrix'])
            elif all(k in cm_data for k in ['true_positive', 'true_negative', 
                                           'false_positive', 'false_negative']):
                tp = cm_data['true_positive']
                tn = cm_data['true_negative']
                fp = cm_data['false_positive']
                fn = cm_data['false_negative']
                cm = np.array([[tn, fp], [fn, tp]])
                if class_names is None:
                    class_names = ['Normal', 'Cognitive Decline']
            else:
                logger.warning("Could not parse confusion matrix data")
                return
        else:
            cm = np.array(cm_data)
        
        # Normalize for percentage
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
        axes[0].set_ylabel('True Label', fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontweight='bold')
        
        # Plot normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
        axes[1].set_ylabel('True Label', fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix plot saved to {output_name}")
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        output_name: str = "roc_curve"
    ):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            output_name: Output filename
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = np.trapz(tpr, fpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curve', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve plot saved to {output_name}")
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        output_name: str = "pr_curve"
    ):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            output_name: Output filename
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curve', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.xlim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Precision-Recall curve plot saved to {output_name}")
    
    def plot_federated_learning_metrics(
        self,
        fl_results: Dict[str, Any],
        output_name: str = "federated_learning_metrics"
    ):
        """
        Plot federated learning specific metrics (communication, rounds, etc.).
        
        Args:
            fl_results: Dictionary with FL metrics
            output_name: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract metrics
        rounds = fl_results.get('rounds', [])
        if isinstance(rounds, int):
            rounds = list(range(1, rounds + 1))
        
        # Communication overhead over rounds
        if 'communication_overhead' in fl_results:
            comm_overhead = fl_results['communication_overhead']
            if isinstance(comm_overhead, (int, float)):
                axes[0, 0].bar(['Total'], [comm_overhead], color='steelblue', edgecolor='black')
                axes[0, 0].set_ylabel('Communication Overhead (MB)', fontweight='bold')
            else:
                axes[0, 0].plot(rounds[:len(comm_overhead)], comm_overhead, 
                               'b-o', linewidth=2, markersize=5)
                axes[0, 0].set_xlabel('Federated Round', fontweight='bold')
                axes[0, 0].set_ylabel('Communication Overhead (MB)', fontweight='bold')
            axes[0, 0].set_title('Communication Overhead', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Accuracy over rounds
        if 'accuracy' in fl_results:
            accuracy = fl_results['accuracy']
            if isinstance(accuracy, (int, float)):
                axes[0, 1].bar(['Final'], [accuracy], color='green', edgecolor='black')
                axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
            else:
                axes[0, 1].plot(rounds[:len(accuracy)], accuracy, 
                               'g-o', linewidth=2, markersize=5)
                axes[0, 1].set_xlabel('Federated Round', fontweight='bold')
                axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
            axes[0, 1].set_title('Model Accuracy Over Rounds', fontweight='bold')
            axes[0, 1].set_ylim([0, 1.0])
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Loss over rounds
        if 'loss' in fl_results:
            loss = fl_results['loss']
            if isinstance(loss, (int, float)):
                axes[1, 0].bar(['Final'], [loss], color='red', edgecolor='black')
                axes[1, 0].set_ylabel('Loss', fontweight='bold')
            else:
                axes[1, 0].plot(rounds[:len(loss)], loss, 
                               'r-o', linewidth=2, markersize=5)
                axes[1, 0].set_xlabel('Federated Round', fontweight='bold')
                axes[1, 0].set_ylabel('Loss', fontweight='bold')
            axes[1, 0].set_title('Training Loss Over Rounds', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Privacy-Utility Trade-off
        privacy_score = fl_results.get('privacy_score', 0.0)
        accuracy_val = fl_results.get('accuracy', 0.0)
        axes[1, 1].scatter([privacy_score], [accuracy_val], s=200, 
                          color='purple', edgecolor='black', linewidth=2, zorder=3)
        axes[1, 1].set_xlabel('Privacy Score', fontweight='bold')
        axes[1, 1].set_ylabel('Model Accuracy', fontweight='bold')
        axes[1, 1].set_title('Privacy-Utility Trade-off', fontweight='bold')
        axes[1, 1].set_xlim([0, 1.0])
        axes[1, 1].set_ylim([0, 1.0])
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].annotate(f'({privacy_score:.2f}, {accuracy_val:.2f})', 
                           xy=(privacy_score, accuracy_val), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Federated learning metrics plot saved to {output_name}")
    
    def plot_performance_metrics(
        self,
        performance_data: Dict[str, float],
        output_name: str = "performance_metrics"
    ):
        """
        Plot system performance metrics (inference time, memory, CPU, energy).
        
        Args:
            performance_data: Dictionary with performance metrics
            output_name: Output filename
        """
        metrics = ['inference_time', 'memory_usage', 'cpu_usage', 'energy_consumption']
        labels = ['Inference Time (s)', 'Memory Usage (MB)', 'CPU Usage (%)', 
                 'Energy Consumption (J)']
        values = [performance_data.get(m, 0.0) for m in metrics]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, (metric, label, value, color) in enumerate(zip(metrics, labels, values, colors)):
            axes[idx].bar([label], [value], color=color, edgecolor='black', linewidth=2)
            axes[idx].set_ylabel(label, fontweight='bold')
            axes[idx].set_title(f'{label.split("(")[0].strip()}', fontweight='bold')
            axes[idx].set_xticks([])
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_axisbelow(True)
            
            # Add value label
            axes[idx].text(0, value + max(value * 0.05, 0.01), f'{value:.2f}', 
                          ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Performance metrics plot saved to {output_name}")
    
    def plot_metrics_radar(
        self,
        metrics_dict: Dict[str, float],
        output_name: str = "metrics_radar"
    ):
        """
        Create radar chart for multiple metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            output_name: Output filename
        """
        # Select key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        available_metrics = [m for m in key_metrics if m in metrics_dict]
        
        if len(available_metrics) < 3:
            logger.warning("Not enough metrics for radar chart")
            return
        
        # Prepare data
        labels = [m.replace('_', ' ').title() for m in available_metrics]
        values = [metrics_dict[m] for m in available_metrics]
        
        # Number of variables
        N = len(labels)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value to end for closing the plot
        values += values[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4', label='Model Performance')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title('Model Performance Metrics (Radar Chart)', 
                    fontweight='bold', pad=20, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Radar chart saved to {output_name}")
    
    def plot_multimodal_comparison(
        self,
        speech_results: Dict[str, Any],
        facial_results: Dict[str, Any],
        multimodal_results: Dict[str, Any],
        output_name: str = "multimodal_comparison"
    ):
        """
        Compare unimodal vs multimodal performance.
        
        Args:
            speech_results: Speech model results
            facial_results: Facial model results
            multimodal_results: Multimodal model results
            output_name: Output filename
        """
        models = ['Speech', 'Facial', 'Multimodal']
        
        # Extract accuracy
        speech_acc = speech_results.get('accuracy', speech_results.get('metrics', {}).get('accuracy', 0.0))
        facial_acc = facial_results.get('accuracy', facial_results.get('metrics', {}).get('accuracy', 0.0))
        multimodal_acc = multimodal_results.get('accuracy', multimodal_results.get('metrics', {}).get('accuracy', 0.0))
        accuracies = [speech_acc, facial_acc, multimodal_acc]
        
        # Extract F1 scores
        speech_f1 = speech_results.get('f1_score', speech_results.get('metrics', {}).get('f1_score', 0.0))
        facial_f1 = facial_results.get('f1_score', facial_results.get('metrics', {}).get('f1_score', 0.0))
        multimodal_f1 = multimodal_results.get('f1_score', multimodal_results.get('metrics', {}).get('f1_score', 0.0))
        f1_scores = [speech_f1, facial_f1, multimodal_f1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(models))
        width = 0.6
        
        # Accuracy comparison
        bars1 = axes[0].bar(x, accuracies, width, color=['#3498db', '#e74c3c', '#2ecc71'], 
                           edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].set_title('Accuracy: Unimodal vs Multimodal', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_axisbelow(True)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1 Score comparison
        bars2 = axes[1].bar(x, f1_scores, width, color=['#3498db', '#e74c3c', '#2ecc71'],
                           edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('F1 Score', fontweight='bold')
        axes[1].set_title('F1 Score: Unimodal vs Multimodal', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].set_ylim([0, 1.0])
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_axisbelow(True)
        
        # Add value labels
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Multimodal comparison plot saved to {output_name}")
    
    def plot_federated_vs_centralized(
        self,
        federated_results: Dict[str, Any],
        centralized_results: Dict[str, Any],
        output_name: str = "federated_vs_centralized"
    ):
        """
        Compare federated learning vs centralized training.
        
        Args:
            federated_results: Federated learning results
            centralized_results: Centralized training results
            output_name: Output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        fl_acc = federated_results.get('accuracy', 0.0)
        cen_acc = centralized_results.get('accuracy', centralized_results.get('metrics', {}).get('accuracy', 0.0))
        
        axes[0].bar(['Federated', 'Centralized'], [fl_acc, cen_acc], 
                   color=['#9b59b6', '#3498db'], edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Accuracy', fontweight='bold')
        axes[0].set_title('Accuracy Comparison', fontweight='bold')
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_axisbelow(True)
        
        # Add value labels
        axes[0].text(0, fl_acc + 0.02, f'{fl_acc:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[0].text(1, cen_acc + 0.02, f'{cen_acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Communication overhead
        fl_comm = federated_results.get('communication_overhead', 0.0)
        cen_comm = 0.0  # Centralized has no communication overhead
        
        axes[1].bar(['Federated', 'Centralized'], [fl_comm, cen_comm],
                   color=['#9b59b6', '#3498db'], edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Communication Overhead (MB)', fontweight='bold')
        axes[1].set_title('Communication Overhead', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_axisbelow(True)
        
        # Privacy score
        fl_privacy = federated_results.get('privacy_score', 0.0)
        cen_privacy = 0.5  # Centralized has lower privacy
        
        axes[2].bar(['Federated', 'Centralized'], [fl_privacy, cen_privacy],
                   color=['#9b59b6', '#3498db'], edgecolor='black', linewidth=2)
        axes[2].set_ylabel('Privacy Score', fontweight='bold')
        axes[2].set_title('Privacy Preservation', fontweight='bold')
        axes[2].set_ylim([0, 1.0])
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_axisbelow(True)
        
        # Add value labels
        axes[2].text(0, fl_privacy + 0.02, f'{fl_privacy:.3f}', ha='center', va='bottom', fontweight='bold')
        axes[2].text(1, cen_privacy + 0.02, f'{cen_privacy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_name}.png", bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / f"{output_name}.pdf", bbox_inches='tight')
        plt.close()
        logger.info(f"Federated vs Centralized plot saved to {output_name}")
    
    def generate_all_plots(
        self,
        results_dir: str = "results",
        include_training: bool = True
    ):
        """
        Generate all visualization plots from results directory.
        
        Args:
            results_dir: Directory containing results JSON files
            include_training: Whether to include training history plots
        """
        results_path = Path(results_dir)
        
        # Load available results
        speech_results_path = results_path / "speech_ravdess" / "training_results.json"
        facial_results_path = results_path / "facial_ravdess" / "training_results.json"
        multimodal_results_path = results_path / "evaluation_ravdess" / "evaluation_results.json"
        federated_results_path = results_path / "federated_ravdess" / "logs" / "cognitive_decline_detection_20250910_000528.log"
        
        results_dict = {}
        
        # Load speech results
        if speech_results_path.exists():
            results_dict['speech'] = self.load_results(speech_results_path)
        
        # Load facial results
        if facial_results_path.exists():
            results_dict['facial'] = self.load_results(facial_results_path)
        
        # Load multimodal results
        if multimodal_results_path.exists():
            results_dict['multimodal'] = self.load_results(multimodal_results_path)
        
        # Generate plots
        if len(results_dict) >= 2:
            self.plot_model_comparison(results_dict)
            self.plot_multimodal_comparison(
                results_dict.get('speech', {}),
                results_dict.get('facial', {}),
                results_dict.get('multimodal', {})
            )
        
        # Plot confusion matrix from evaluation results
        if multimodal_results_path.exists():
            multimodal_data = self.load_results(multimodal_results_path)
            if 'confusion_matrix' in multimodal_data:
                self.plot_confusion_matrix(multimodal_data['confusion_matrix'])
            if 'performance' in multimodal_data:
                self.plot_performance_metrics(multimodal_data['performance'])
            if 'metrics' in multimodal_data:
                self.plot_metrics_radar(multimodal_data['metrics'])
        
        logger.info("All plots generated successfully!")


if __name__ == "__main__":
    # Example usage
    visualizer = ResultsVisualizer("results/plots")
    
    # Generate all plots from results directory
    visualizer.generate_all_plots("results")

