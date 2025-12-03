# Installation Commands for Cognitive Decline Detection Project

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

### 1. Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Upgrade pip (Recommended)
```bash
python -m pip install --upgrade pip
```

### 3. Install All Required Libraries

**Option A: Install from requirements.txt (Easiest)**
```bash
pip install -r requirements.txt
```

**Option B: Install packages individually (if needed)**

```bash
# Core ML and Deep Learning
pip install torch>=1.10.0 torchvision>=0.11.0 torchaudio>=0.10.0
pip install scikit-learn>=1.0.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0

# Audio Processing
pip install librosa>=0.8.0
pip install soundfile>=0.10.0

# Computer Vision and Image Processing
pip install opencv-python>=4.5.0
pip install Pillow>=8.0.0

# Signal Processing and Visualization
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install seaborn>=0.11.0

# Federated Learning
pip install flwr>=1.0.0

# Privacy and Security
pip install cryptography>=3.4.0

# Data Handling and Storage
pip install h5py>=3.1.0
pip install tqdm>=4.62.0
pip install joblib>=1.0.0

# Network and Communication
pip install requests>=2.25.0
pip install flask>=2.0.0

# Configuration and Logging
pip install PyYAML>=5.4.1
pip install python-dotenv>=0.19.0

# Testing and Quality
pip install pytest>=6.0.0
pip install pytest-cov>=2.12.0

# Jupyter and Development
pip install jupyter>=1.0.0
pip install ipykernel>=6.0.0

# System and Performance
pip install psutil>=5.8.0

# Additional Utilities
pip install click>=8.0.0
pip install pydantic>=1.8.0

# Edge Computing Simulation
pip install networkx>=2.6.0

# Time Series Analysis
pip install statsmodels>=0.12.0

# Model Interpretability
pip install shap>=0.40.0

# Optimization
pip install optuna>=2.10.0
```

### 4. Install Development Dependencies (Optional)

If you want to install development tools (from setup.py):
```bash
pip install -e ".[dev]"
```

Or install individually:
```bash
pip install pytest>=7.4.0
pip install pytest-cov>=4.1.0
pip install black>=23.7.0
pip install flake8>=6.0.0
pip install mypy>=1.5.0
pip install pre-commit>=3.3.0
```

### 5. Install Documentation Dependencies (Optional)

If you want to build documentation:
```bash
pip install sphinx>=7.1.0
pip install sphinx-rtd-theme>=1.3.0
pip install myst-parser>=2.0.0
```

### 6. Install GPU Support (Optional - if you have CUDA-enabled GPU)

**For PyTorch with CUDA (Windows):**
```bash
# Check your CUDA version first, then install appropriate PyTorch
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU-only (default):**
```bash
# Already included in requirements.txt
```

### 7. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test imports
python -c "import torch; import librosa; import cv2; import flwr; print('All imports successful!')"
```

### 8. Install Project in Development Mode (Optional)

```bash
pip install -e .
```

## Quick Start Command (All-in-One)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues:

1. **librosa installation fails on Windows:**
   - Install Microsoft Visual C++ Redistributable
   - Or use: `pip install librosa --no-cache-dir`

2. **soundfile installation fails:**
   - Install CFFI: `pip install cffi`
   - Or install from conda-forge if using conda

3. **opencv-python installation fails:**
   - Try: `pip install opencv-python-headless` as alternative

4. **PyTorch installation (GPU support):**
   - Visit https://pytorch.org/get-started/locally/ for the correct command based on your CUDA version

5. **Permission errors:**
   - Use `--user` flag: `pip install --user -r requirements.txt`
   - Or ensure virtual environment is activated

## Additional Notes

- The project requires Python 3.8 or higher
- Some packages may take time to install (especially PyTorch, librosa, opencv-python)
- If you encounter dependency conflicts, consider using `pip install --upgrade` for conflicting packages
- For production environments, consider using `pip freeze > requirements-lock.txt` to pin exact versions

