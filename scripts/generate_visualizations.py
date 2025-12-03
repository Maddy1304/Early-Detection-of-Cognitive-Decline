#!/usr/bin/env python3
"""
Script to generate all visualization plots for results and discussion sections.

Usage:
    python scripts/generate_visualizations.py --results-dir results --output-dir results/plots
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.visualizations import ResultsVisualizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate visualization plots for results')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory containing results JSON files')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                       help='Output directory for plots')
    parser.add_argument('--speech-results', type=str, 
                       default='results/speech_ravdess/training_results.json',
                       help='Path to speech model results')
    parser.add_argument('--facial-results', type=str,
                       default='results/facial_ravdess/training_results.json',
                       help='Path to facial model results')
    parser.add_argument('--multimodal-results', type=str,
                       default='results/evaluation_ravdess/evaluation_results.json',
                       help='Path to multimodal evaluation results')
    parser.add_argument('--federated-results', type=str,
                       default='results/federated_ravdess',
                       help='Path to federated learning results directory')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ResultsVisualizer(args.output_dir)
    
    logger.info("Generating visualization plots...")
    
    # Load results
    results_dict = {}
    
    # Load speech results
    speech_path = Path(args.speech_results)
    if speech_path.exists():
        results_dict['speech'] = visualizer.load_results(str(speech_path))
        logger.info(f"Loaded speech results from {speech_path}")
    else:
        logger.warning(f"Speech results not found at {speech_path}")
    
    # Load facial results
    facial_path = Path(args.facial_results)
    if facial_path.exists():
        results_dict['facial'] = visualizer.load_results(str(facial_path))
        logger.info(f"Loaded facial results from {facial_path}")
    else:
        logger.warning(f"Facial results not found at {facial_path}")
    
    # Load multimodal results
    multimodal_path = Path(args.multimodal_results)
    if multimodal_path.exists():
        results_dict['multimodal'] = visualizer.load_results(str(multimodal_path))
        logger.info(f"Loaded multimodal results from {multimodal_path}")
    else:
        logger.warning(f"Multimodal results not found at {multimodal_path}")
    
    # Generate plots
    if len(results_dict) >= 2:
        logger.info("Generating model comparison plot...")
        visualizer.plot_model_comparison(results_dict, output_name="model_comparison")
        
        if 'speech' in results_dict and 'facial' in results_dict and 'multimodal' in results_dict:
            logger.info("Generating multimodal comparison plot...")
            visualizer.plot_multimodal_comparison(
                results_dict['speech'],
                results_dict['facial'],
                results_dict['multimodal'],
                output_name="multimodal_comparison"
            )
    
    # Generate plots from multimodal evaluation results
    if 'multimodal' in results_dict:
        multimodal_data = results_dict['multimodal']
        
        # Confusion matrix
        if 'confusion_matrix' in multimodal_data:
            logger.info("Generating confusion matrix...")
            visualizer.plot_confusion_matrix(
                multimodal_data['confusion_matrix'],
                output_name="confusion_matrix"
            )
        
        # Performance metrics
        if 'performance' in multimodal_data:
            logger.info("Generating performance metrics plot...")
            visualizer.plot_performance_metrics(
                multimodal_data['performance'],
                output_name="performance_metrics"
            )
        
        # Metrics radar chart
        if 'metrics' in multimodal_data:
            logger.info("Generating metrics radar chart...")
            visualizer.plot_metrics_radar(
                multimodal_data['metrics'],
                output_name="metrics_radar"
            )
    
    # Generate federated learning plots if available
    federated_path = Path(args.federated_results)
    if federated_path.exists():
        # Try to find training results
        training_results_path = federated_path / "training_results.json"
        if training_results_path.exists():
            fl_results = visualizer.load_results(str(training_results_path))
            logger.info("Generating federated learning metrics plot...")
            visualizer.plot_federated_learning_metrics(
                fl_results,
                output_name="federated_learning_metrics"
            )
    
    logger.info(f"All plots generated and saved to {args.output_dir}")
    logger.info("Generated plots:")
    output_path = Path(args.output_dir)
    for plot_file in sorted(output_path.glob("*.png")):
        logger.info(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()

