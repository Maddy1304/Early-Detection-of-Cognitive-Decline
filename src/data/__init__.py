"""
Data processing and dataset handling modules.

This module provides functionality for:
- Loading and preprocessing multimodal datasets (DAIC-WOZ, mPower, RAVDESS)
- Audio, gait, and facial expression data processing
- Feature extraction and normalization
- Data augmentation and validation
"""

from .preprocessing import AudioProcessor, GaitProcessor, FacialProcessor
from .datasets import DAICWOZDataset, MPowerDataset, RAVDESSDataset
from .utils import DataLoader, DataValidator, FeatureExtractor

__all__ = [
    "AudioProcessor",
    "GaitProcessor", 
    "FacialProcessor",
    "DAICWOZDataset",
    "MPowerDataset",
    "RAVDESSDataset",
    "DataLoader",
    "DataValidator",
    "FeatureExtractor",
]
