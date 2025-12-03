# Quick Reference: Available Graphs for Results & Discussion

This is a quick reference guide for all available visualization graphs in your cognitive decline detection project.

## 📊 Available Graph Types

### 1. **Model Comparison Bar Chart**
- **Function**: `plot_model_comparison()`
- **Shows**: Accuracy, Precision, Recall, F1-Score across models
- **Use in**: Results section - Model comparison
- **Output**: `model_comparison.png/pdf`

### 2. **Training Curves**
- **Function**: `plot_training_curves()`
- **Shows**: Training/Validation loss and accuracy over epochs
- **Use in**: Results section - Training analysis
- **Output**: `training_curves.png/pdf`

### 3. **Confusion Matrix**
- **Function**: `plot_confusion_matrix()`
- **Shows**: Classification errors (counts and percentages)
- **Use in**: Results section - Classification evaluation
- **Output**: `confusion_matrix.png/pdf`

### 4. **ROC Curve**
- **Function**: `plot_roc_curve()`
- **Shows**: True Positive Rate vs False Positive Rate
- **Use in**: Results section - Model discrimination
- **Output**: `roc_curve.png/pdf`

### 5. **Precision-Recall Curve**
- **Function**: `plot_precision_recall_curve()`
- **Shows**: Precision vs Recall at different thresholds
- **Use in**: Results section - Class imbalance analysis
- **Output**: `pr_curve.png/pdf`

### 6. **Federated Learning Metrics**
- **Function**: `plot_federated_learning_metrics()`
- **Shows**: Communication overhead, accuracy over rounds, privacy-utility trade-off
- **Use in**: Discussion section - FL analysis
- **Output**: `federated_learning_metrics.png/pdf`

### 7. **Performance Metrics**
- **Function**: `plot_performance_metrics()`
- **Shows**: Inference time, memory, CPU usage, energy consumption
- **Use in**: Discussion section - System performance
- **Output**: `performance_metrics.png/pdf`

### 8. **Metrics Radar Chart**
- **Function**: `plot_metrics_radar()`
- **Shows**: Comprehensive metrics in polar chart
- **Use in**: Results section - Overall performance
- **Output**: `metrics_radar.png/pdf`

### 9. **Multimodal Comparison**
- **Function**: `plot_multimodal_comparison()`
- **Shows**: Speech vs Facial vs Multimodal performance
- **Use in**: Results section - Main contribution
- **Output**: `multimodal_comparison.png/pdf`

### 10. **Federated vs Centralized**
- **Function**: `plot_federated_vs_centralized()`
- **Shows**: FL vs centralized accuracy, communication, privacy
- **Use in**: Discussion section - Approach comparison
- **Output**: `federated_vs_centralized.png/pdf`

## 🚀 Quick Start

### Generate All Plots:
```bash
python scripts/generate_visualizations.py
```

### Generate Specific Plot (Python):
```python
from src.evaluation import ResultsVisualizer
import json

# Load results
with open('results/evaluation_ravdess/evaluation_results.json') as f:
    results = json.load(f)

# Create visualizer
viz = ResultsVisualizer("results/plots")

# Generate plot
viz.plot_confusion_matrix(results['confusion_matrix'])
```

## 📝 Recommended Graph Sequence for Paper

### Results Section:
1. Model Comparison Bar Chart
2. Multimodal Comparison
3. Training Curves
4. Confusion Matrix
5. ROC Curve
6. Metrics Radar Chart

### Discussion Section:
1. Federated Learning Metrics
2. Privacy-Utility Trade-off
3. Federated vs Centralized
4. Performance Metrics

## 💡 Tips

- **High Quality**: All plots are saved at 300 DPI for publication
- **Dual Format**: Both PNG (presentations) and PDF (papers) are generated
- **Color Blind Friendly**: Uses accessible color palettes
- **Professional Style**: Uses seaborn for publication-ready appearance

## 📁 Output Location

All plots are saved to: `results/plots/`

