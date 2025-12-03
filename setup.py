"""
Setup script for Early Detection of Cognitive Decline Using Multi-Modal Federated Learning
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cognitive-decline-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Multi-Modal Federated Learning for Early Detection of Cognitive Decline",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cognitive-decline-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "tensorflow[and-cuda]>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cognitive-decline=src.cli:main",
            "cdd-experiment=scripts.run_experiments:main",
            "cdd-download=scripts.download_datasets:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    keywords=[
        "federated-learning",
        "cognitive-decline",
        "alzheimer",
        "parkinson",
        "multimodal",
        "edge-computing",
        "fog-computing",
        "privacy-preserving",
        "healthcare-ai",
        "machine-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cognitive-decline-detection/issues",
        "Source": "https://github.com/yourusername/cognitive-decline-detection",
        "Documentation": "https://cognitive-decline-detection.readthedocs.io/",
    },
)
