# Early Detection of Cognitive Decline Using Multi-Modal Federated Learning with Edge–Fog Collaboration

## 🧠 Project Overview

This project implements a privacy-preserving, multi-modal federated learning system for early detection of cognitive decline (Alzheimer's and Parkinson's diseases) using edge-fog-cloud collaboration. The system processes speech, gait, and facial expression data locally on edge devices while maintaining patient privacy through federated learning.

## 🎯 Key Features

- **Multi-Modal Data Processing**: Integrates speech, gait, and facial expression analysis
- **Federated Learning**: Privacy-preserving distributed training without sharing raw data
- **Edge-Fog-Cloud Architecture**: Hierarchical computing for low latency and scalability
- **Real-World Datasets**: Uses DAIC-WOZ, mPower, and RAVDESS datasets
- **Simulation Environment**: Complete simulation of healthcare infrastructure

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Devices  │    │   Edge Devices  │    │   Edge Devices  │
│  (Smartphones/  │    │  (Wearables/    │    │  (IoT Sensors/  │
│   Tablets)      │    │   Smartwatches) │    │   Cameras)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Fog Nodes            │
                    │   (Clinic Servers)        │
                    │  - Model Aggregation      │
                    │  - Local Processing       │
                    │  - Privacy Filtering      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │      Cloud Server         │
                    │   (Global Model)          │
                    │  - Global Aggregation     │
                    │  - Model Distribution     │
                    │  - Analytics & Reports    │
                    └───────────────────────────┘
```

## 📁 Project Structure

```
cognitive-decline-detection/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── edge_config.yaml
│   ├── fog_config.yaml
│   ├── cloud_config.yaml
│   └── model_config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── audio_processor.py
│   │   │   ├── gait_processor.py
│   │   │   └── facial_processor.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── daic_woz.py
│   │   │   ├── mpower.py
│   │   │   └── ravdess.py
│   │   └── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── speech_model.py
│   │   ├── gait_model.py
│   │   ├── facial_model.py
│   │   ├── multimodal_fusion.py
│   │   └── base_model.py
│   ├── federated_learning/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── server.py
│   │   ├── aggregation.py
│   │   └── privacy.py
│   ├── infrastructure/
│   │   ├── __init__.py
│   │   ├── edge_device.py
│   │   ├── fog_node.py
│   │   ├── cloud_server.py
│   │   └── network_simulator.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── benchmarking.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── config_loader.py
│       └── helpers.py
├── experiments/
│   ├── baseline_experiments.py
│   ├── federated_experiments.py
│   ├── privacy_analysis.py
│   └── scalability_tests.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_development.ipynb
│   ├── federated_learning_demo.ipynb
│   └── results_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_models.py
│   ├── test_federated_learning.py
│   └── test_infrastructure.py
├── docs/
│   ├── api_reference.md
│   ├── deployment_guide.md
│   ├── privacy_analysis.md
│   └── performance_benchmarks.md
├── scripts/
│   ├── download_datasets.py
│   ├── setup_environment.py
│   ├── run_experiments.py
│   └── deploy_simulation.py
└── results/
    ├── logs/
    ├── models/
    ├── plots/
    └── reports/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd cognitive-decline-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets**
```bash
python scripts/download_datasets.py
```

5. **Run simulation**
```bash
python scripts/run_experiments.py --experiment baseline
```

## 📊 Datasets

> **Note**: Datasets are not included in this repository due to their large size. Please download them separately using the instructions below.

### DAIC-WOZ (Depression and Anxiety in Context)
- **Purpose**: Speech and facial expression analysis
- **Size**: ~189 hours of audio-visual data
- **Features**: Audio, video, transcriptions, PHQ-8 scores
- **Download**: Use `python scripts/download_datasets.py --dataset daic_woz`


### RAVDESS (Ryerson Audio-Visual Database)
- **Purpose**: Emotional speech recognition
- **Size**: 7,356 files
- **Features**: Audio files with emotional labels
- **Download**: Use `python scripts/download_datasets.py --dataset ravdess`
- **Manual Setup**: 
  1. Download from: https://zenodo.org/record/1188976
  2. Extract to `data/ravdess/`
  3. Expected structure:
     ```
     data/ravdess/
     ├── Audio_Speech_Actors_01-24/
     ├── Video_Speech_Actors_01-24/
     └── README.txt
     ```

## 🔬 Experiments

### Baseline Experiments
- Centralized training on each modality
- Performance comparison across datasets
- Model architecture optimization

### Federated Learning Experiments
- Privacy-preserving distributed training
- Communication efficiency analysis
- Convergence behavior study

### Privacy Analysis
- Differential privacy implementation
- Privacy-utility trade-off evaluation
- Attack resistance testing

### Scalability Tests
- Edge device simulation (10-1000 devices)
- Fog node performance analysis
- Network latency impact study

## 📈 Key Metrics

- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Communication Overhead**: Data transfer efficiency
- **Latency**: End-to-end processing time
- **Privacy Budget**: Differential privacy cost

## 🔒 Privacy Features

- **Federated Learning**: No raw data sharing
- **Differential Privacy**: Mathematical privacy guarantees
- **Secure Aggregation**: Cryptographic model update protection
- **Local Processing**: Edge device data processing

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

### Documentation
```bash
sphinx-build docs/ docs/_build/
```

## 📚 Research Background

This project addresses critical gaps in healthcare AI:

1. **Limited Multimodal Integration**: Most FL systems use single data types
2. **Cognitive Disorder Focus**: Early detection of Alzheimer's/Parkinson's
3. **Real-World Deployment**: Practical edge-fog-cloud implementation
4. **Privacy-Latency Balance**: Optimized trade-offs for healthcare

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions and collaboration, please contact [your-email@domain.com]

## 🙏 Acknowledgments

- DAIC-WOZ dataset contributors
- mPower study participants
- RAVDESS dataset creators
- Federated learning research community

---

**Note**: This is a research prototype for simulation purposes. Not intended for clinical use without proper validation and regulatory approval.
