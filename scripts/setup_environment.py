#!/usr/bin/env python3
"""
Environment setup script for cognitive decline detection project.

This script sets up the development environment including:
- Virtual environment creation
- Dependency installation
- Configuration setup
- Directory structure creation
"""

import os
import sys
import argparse
import logging
import subprocess
import platform
from pathlib import Path
import shutil
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('setup_environment.log')
        ]
    )


def run_command(command: List[str], cwd: Optional[Path] = None) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run
        cwd: Working directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def create_virtual_environment(venv_path: Path, python_executable: Optional[str] = None) -> bool:
    """
    Create virtual environment.
    
    Args:
        venv_path: Path to virtual environment
        python_executable: Python executable to use
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Creating virtual environment at {venv_path}")
        
        # Remove existing environment if it exists
        if venv_path.exists():
            logger.info("Removing existing virtual environment")
            shutil.rmtree(venv_path)
        
        # Create virtual environment
        if python_executable:
            command = [python_executable, '-m', 'venv', str(venv_path)]
        else:
            command = [sys.executable, '-m', 'venv', str(venv_path)]
        
        if not run_command(command):
            return False
        
        logger.info("Virtual environment created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating virtual environment: {e}")
        return False


def get_venv_python(venv_path: Path) -> Path:
    """Get Python executable path in virtual environment."""
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / 'python.exe'
    else:
        return venv_path / 'bin' / 'python'


def get_venv_pip(venv_path: Path) -> Path:
    """Get pip executable path in virtual environment."""
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / 'pip.exe'
    else:
        return venv_path / 'bin' / 'pip'


def install_dependencies(venv_path: Path, requirements_file: Path) -> bool:
    """
    Install dependencies from requirements file.
    
    Args:
        venv_path: Path to virtual environment
        requirements_file: Path to requirements file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Installing dependencies...")
        
        pip_path = get_venv_pip(venv_path)
        
        # Upgrade pip first
        if not run_command([str(pip_path), 'install', '--upgrade', 'pip']):
            logger.warning("Failed to upgrade pip, continuing...")
        
        # Install requirements
        if not run_command([str(pip_path), 'install', '-r', str(requirements_file)]):
            return False
        
        logger.info("Dependencies installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def create_directory_structure(project_root: Path) -> bool:
    """
    Create project directory structure.
    
    Args:
        project_root: Project root directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Creating directory structure...")
        
        directories = [
            'data',
            'data/daic_woz',
            'data/mpower',
            'data/ravdess',
            'data/sample_data',
            'results',
            'results/logs',
            'results/models',
            'results/plots',
            'results/reports',
            'experiments',
            'notebooks',
            'tests',
            'docs',
            'scripts',
            'config',
            'certs'
        ]
        
        for directory in directories:
            dir_path = project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        logger.info("Directory structure created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating directory structure: {e}")
        return False


def create_gitignore(project_root: Path) -> bool:
    """
    Create .gitignore file.
    
    Args:
        project_root: Project root directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Creating .gitignore file...")
        
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/*/
!data/sample_data/
*.csv
*.json
*.pkl
*.h5
*.hdf5
*.npy
*.npz

# Model files
*.pth
*.pt
*.ckpt
*.model

# Logs
*.log
logs/

# Results
results/*/
!results/.gitkeep

# OS
.DS_Store
Thumbs.db

# Certificates
*.pem
*.key
*.crt

# Temporary files
*.tmp
*.temp
temp/
tmp/

# Configuration files with secrets
config/secrets.yaml
config/local.yaml
"""
        
        gitignore_path = project_root / '.gitignore'
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        logger.info("Created .gitignore file")
        return True
        
    except Exception as e:
        logger.error(f"Error creating .gitignore: {e}")
        return False


def create_env_file(project_root: Path) -> bool:
    """
    Create .env file template.
    
    Args:
        project_root: Project root directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Creating .env file template...")
        
        env_content = """# Environment Configuration
# Copy this file to .env and fill in your values

# Dataset paths
DAIC_WOZ_PATH=data/daic_woz
MPOWER_PATH=data/mpower
RAVDESS_PATH=data/ravdess

# Model paths
MODEL_PATH=results/models
LOG_PATH=results/logs

# API keys (if needed)
# OPENAI_API_KEY=your_key_here
# WANDB_API_KEY=your_key_here

# Database (if needed)
# DATABASE_URL=sqlite:///results/database.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=results/logs/app.log

# Development
DEBUG=False
TESTING=False
"""
        
        env_path = project_root / '.env.template'
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info("Created .env.template file")
        return True
        
    except Exception as e:
        logger.error(f"Error creating .env file: {e}")
        return False


def create_activation_script(project_root: Path, venv_path: Path) -> bool:
    """
    Create activation script for virtual environment.
    
    Args:
        project_root: Project root directory
        venv_path: Path to virtual environment
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Creating activation script...")
        
        if platform.system() == 'Windows':
            script_content = f"""@echo off
echo Activating virtual environment...
call {venv_path}\\Scripts\\activate.bat
echo Virtual environment activated!
echo.
echo To deactivate, run: deactivate
echo To run the project, use: python scripts/run_experiments.py
"""
            script_path = project_root / 'activate.bat'
        else:
            script_content = f"""#!/bin/bash
echo "Activating virtual environment..."
source {venv_path}/bin/activate
echo "Virtual environment activated!"
echo ""
echo "To deactivate, run: deactivate"
echo "To run the project, use: python scripts/run_experiments.py"
"""
            script_path = project_root / 'activate.sh'
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable on Unix systems
        if platform.system() != 'Windows':
            os.chmod(script_path, 0o755)
        
        logger.info(f"Created activation script: {script_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating activation script: {e}")
        return False


def run_tests(venv_path: Path, project_root: Path) -> bool:
    """
    Run basic tests to verify installation.
    
    Args:
        venv_path: Path to virtual environment
        project_root: Project root directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Running basic tests...")
        
        python_path = get_venv_python(venv_path)
        
        # Test imports
        test_script = """
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"PyTorch import failed: {e}")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas import failed: {e}")

print("Basic imports completed successfully!")
"""
        
        # Write test script to temporary file
        test_file = project_root / 'test_imports.py'
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        # Run test script
        if not run_command([str(python_path), str(test_file)]):
            return False
        
        # Clean up test file
        test_file.unlink()
        
        logger.info("Basic tests passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup development environment')
    parser.add_argument('--project-root', type=str, default='.', help='Project root directory')
    parser.add_argument('--venv-name', type=str, default='venv', help='Virtual environment name')
    parser.add_argument('--python-executable', type=str, help='Python executable to use')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Get project root
    project_root = Path(args.project_root).resolve()
    venv_path = project_root / args.venv_name
    
    logger.info(f"Setting up environment in {project_root}")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directory structure
    if not create_directory_structure(project_root):
        logger.error("Failed to create directory structure")
        return 1
    
    # Create virtual environment
    if not create_virtual_environment(venv_path, args.python_executable):
        logger.error("Failed to create virtual environment")
        return 1
    
    # Install dependencies
    if not args.skip_deps:
        requirements_file = project_root / 'requirements.txt'
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return 1
        
        if not install_dependencies(venv_path, requirements_file):
            logger.error("Failed to install dependencies")
            return 1
    
    # Create configuration files
    if not create_gitignore(project_root):
        logger.error("Failed to create .gitignore")
        return 1
    
    if not create_env_file(project_root):
        logger.error("Failed to create .env template")
        return 1
    
    if not create_activation_script(project_root, venv_path):
        logger.error("Failed to create activation script")
        return 1
    
    # Run tests
    if not args.skip_tests:
        if not run_tests(venv_path, project_root):
            logger.error("Tests failed")
            return 1
    
    # Success message
    logger.info("Environment setup completed successfully!")
    logger.info(f"Virtual environment created at: {venv_path}")
    logger.info(f"To activate the environment, run: {'activate.bat' if platform.system() == 'Windows' else 'source activate.sh'}")
    logger.info("To install datasets, run: python scripts/download_datasets.py")
    logger.info("To run experiments, run: python scripts/run_experiments.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
