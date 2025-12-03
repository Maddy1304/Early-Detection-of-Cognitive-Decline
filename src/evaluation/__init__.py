"""
Evaluation module for cognitive decline detection.

This module provides comprehensive evaluation capabilities for assessing
the performance of the federated learning system and models.
"""

from .metrics import EvaluationMetrics
from .visualizations import ResultsVisualizer

__all__ = ['EvaluationMetrics', 'ResultsVisualizer']
