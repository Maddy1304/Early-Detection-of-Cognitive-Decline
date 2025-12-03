"""
Data utilities for multimodal cognitive decline detection.

This module provides utility functions for data loading, validation, and processing.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """Custom data loader for multimodal datasets."""
    
    def __init__(
        self,
        datasets: List[Dataset],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        collate_fn: Optional[callable] = None
    ):
        """
        Initialize data loader.
        
        Args:
            datasets: List of datasets to load
            batch_size: Batch size for loading
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            collate_fn: Custom collate function
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        
        # Create data loaders for each dataset
        self.dataloaders = []
        for dataset in datasets:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn or dataset._collate_fn if hasattr(dataset, '_collate_fn') else None
            )
            self.dataloaders.append(dataloader)
    
    def __iter__(self):
        """Iterate over data loaders."""
        for dataloader in self.dataloaders:
            yield from dataloader
    
    def __len__(self) -> int:
        """Return total number of batches."""
        return sum(len(dataloader) for dataloader in self.dataloaders)


class DataValidator:
    """Data validator for multimodal datasets."""
    
    def __init__(self, validation_rules: Optional[Dict] = None):
        """
        Initialize data validator.
        
        Args:
            validation_rules: Custom validation rules
        """
        self.validation_rules = validation_rules or self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """Get default validation rules."""
        return {
            'audio': {
                'min_length': 1.0,  # seconds
                'max_length': 300.0,  # seconds
                'sample_rate': 16000,
                'channels': 1
            },
            'video': {
                'min_frames': 10,
                'max_frames': 9000,  # 5 minutes at 30fps
                'fps': 30,
                'channels': 3
            },
            'gait': {
                'min_duration': 5.0,  # seconds
                'max_duration': 120.0,  # seconds
                'sampling_rate': 50,
                'required_sensors': ['acc_x', 'acc_y', 'acc_z']
            },
            'text': {
                'min_length': 10,  # characters
                'max_length': 10000,  # characters
                'encoding': 'utf-8'
            }
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a data sample.
        
        Args:
            sample: Data sample to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required keys
        required_keys = ['label']
        for key in required_keys:
            if key not in sample:
                errors.append(f"Missing required key: {key}")
        
        # Validate each modality
        for modality, data in sample.items():
            if modality in self.validation_rules and modality != 'label':
                is_valid, modality_errors = self._validate_modality(modality, data)
                if not is_valid:
                    errors.extend(modality_errors)
        
        return len(errors) == 0, errors
    
    def _validate_modality(self, modality: str, data: Any) -> Tuple[bool, List[str]]:
        """Validate a specific modality."""
        errors = []
        rules = self.validation_rules.get(modality, {})
        
        if modality == 'audio':
            if isinstance(data, np.ndarray):
                if len(data.shape) != 1:
                    errors.append("Audio data should be 1D array")
                
                if 'min_length' in rules:
                    min_samples = int(rules['min_length'] * rules['sample_rate'])
                    if len(data) < min_samples:
                        errors.append(f"Audio too short: {len(data)} samples < {min_samples}")
                
                if 'max_length' in rules:
                    max_samples = int(rules['max_length'] * rules['sample_rate'])
                    if len(data) > max_samples:
                        errors.append(f"Audio too long: {len(data)} samples > {max_samples}")
            else:
                errors.append("Audio data should be numpy array")
        
        elif modality == 'video':
            if isinstance(data, np.ndarray):
                if len(data.shape) != 4:
                    errors.append("Video data should be 4D array (frames, height, width, channels)")
                
                if 'min_frames' in rules and len(data) < rules['min_frames']:
                    errors.append(f"Video too short: {len(data)} frames < {rules['min_frames']}")
                
                if 'max_frames' in rules and len(data) > rules['max_frames']:
                    errors.append(f"Video too long: {len(data)} frames > {rules['max_frames']}")
            else:
                errors.append("Video data should be numpy array")
        
        elif modality == 'gait':
            if isinstance(data, pd.DataFrame):
                required_sensors = rules.get('required_sensors', [])
                for sensor in required_sensors:
                    if sensor not in data.columns:
                        errors.append(f"Missing required sensor: {sensor}")
                
                if 'min_duration' in rules:
                    min_samples = int(rules['min_duration'] * rules['sampling_rate'])
                    if len(data) < min_samples:
                        errors.append(f"Gait data too short: {len(data)} samples < {min_samples}")
                
                if 'max_duration' in rules:
                    max_samples = int(rules['max_duration'] * rules['sampling_rate'])
                    if len(data) > max_samples:
                        errors.append(f"Gait data too long: {len(data)} samples > {max_samples}")
            else:
                errors.append("Gait data should be pandas DataFrame")
        
        elif modality == 'text':
            if isinstance(data, str):
                if 'min_length' in rules and len(data) < rules['min_length']:
                    errors.append(f"Text too short: {len(data)} characters < {rules['min_length']}")
                
                if 'max_length' in rules and len(data) > rules['max_length']:
                    errors.append(f"Text too long: {len(data)} characters > {rules['max_length']}")
            else:
                errors.append("Text data should be string")
        
        return len(errors) == 0, errors
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Validate entire dataset.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Validation results
        """
        results = {
            'total_samples': len(dataset),
            'valid_samples': 0,
            'invalid_samples': 0,
            'errors': [],
            'modality_errors': {}
        }
        
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                is_valid, errors = self.validate_sample(sample)
                
                if is_valid:
                    results['valid_samples'] += 1
                else:
                    results['invalid_samples'] += 1
                    results['errors'].extend([f"Sample {idx}: {error}" for error in errors])
                    
                    # Count errors by modality
                    for error in errors:
                        modality = error.split(':')[0].split()[-1] if ':' in error else 'unknown'
                        results['modality_errors'][modality] = results['modality_errors'].get(modality, 0) + 1
                        
            except Exception as e:
                results['invalid_samples'] += 1
                results['errors'].append(f"Sample {idx}: {str(e)}")
        
        return results


class FeatureExtractor:
    """Feature extractor for multimodal data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration for feature extraction
        """
        self.config = config or self._get_default_config()
        self._initialize_processors()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'audio': {
                'mfcc': True,
                'spectral': True,
                'prosodic': True,
                'sample_rate': 16000
            },
            'video': {
                'geometric': True,
                'texture': True,
                'emotion': True,
                'fps': 30
            },
            'gait': {
                'temporal': True,
                'frequency': True,
                'gait_specific': True,
                'balance': True,
                'sampling_rate': 50
            },
            'text': {
                'basic': True,
                'linguistic': True,
                'semantic': False
            }
        }
    
    def _initialize_processors(self):
        """Initialize data processors."""
        try:
            from .preprocessing.audio_processor import AudioProcessor
            from .preprocessing.facial_processor import FacialProcessor
            from .preprocessing.gait_processor import GaitProcessor
            
            self.audio_processor = AudioProcessor(
                sample_rate=self.config['audio']['sample_rate']
            )
            self.facial_processor = FacialProcessor()
            self.gait_processor = GaitProcessor(
                sampling_rate=self.config['gait']['sampling_rate']
            )
        except ImportError as e:
            logger.warning(f"Could not initialize processors: {e}")
            self.audio_processor = None
            self.facial_processor = None
            self.gait_processor = None
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from multimodal data.
        
        Args:
            data: Multimodal data dictionary
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract audio features
        if 'audio' in data and self.audio_processor:
            audio_features = self._extract_audio_features(data['audio'])
            features['audio'] = audio_features
        
        # Extract video features
        if 'video' in data and self.facial_processor:
            video_features = self._extract_video_features(data['video'])
            features['video'] = video_features
        
        # Extract gait features
        if 'gait' in data and self.gait_processor:
            gait_features = self._extract_gait_features(data['gait'])
            features['gait'] = gait_features
        
        # Extract text features
        if 'text' in data:
            text_features = self._extract_text_features(data['text'])
            features['text'] = text_features
        
        return features
    
    def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract audio features."""
        features = {}
        
        if self.config['audio']['mfcc']:
            features['mfcc'] = self.audio_processor.extract_mfcc(audio)
        
        if self.config['audio']['spectral']:
            features['spectral'] = self.audio_processor.extract_spectral_features(audio)
        
        if self.config['audio']['prosodic']:
            features['prosodic'] = self.audio_processor.extract_prosodic_features(audio)
        
        return features
    
    def _extract_video_features(self, video: np.ndarray) -> Dict[str, Any]:
        """Extract video features."""
        features = []
        
        for frame in video:
            frame_features = {}
            
            if self.config['video']['geometric']:
                # Extract geometric features (requires landmarks)
                pass
            
            if self.config['video']['texture']:
                # Extract texture features
                pass
            
            if self.config['video']['emotion']:
                # Extract emotion features
                pass
            
            features.append(frame_features)
        
        return features
    
    def _extract_gait_features(self, gait_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract gait features."""
        features = {}
        
        if self.config['gait']['temporal']:
            features['temporal'] = self.gait_processor.extract_temporal_features(gait_data.values)
        
        if self.config['gait']['frequency']:
            features['frequency'] = self.gait_processor.extract_frequency_features(gait_data.values)
        
        if self.config['gait']['gait_specific']:
            # Extract gait-specific features (requires step detection)
            pass
        
        if self.config['gait']['balance']:
            features['balance'] = self.gait_processor.extract_balance_features(gait_data.values)
        
        return features
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract text features."""
        features = {}
        
        if self.config['text']['basic']:
            words = text.split()
            sentences = text.split('.')
            
            features['basic'] = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0,
                'text_length': len(text)
            }
        
        if self.config['text']['linguistic']:
            # Extract linguistic features
            pass
        
        if self.config['text']['semantic']:
            # Extract semantic features
            pass
        
        return features


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return {}


def save_config(config: Dict, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved config to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config {config_path}: {e}")


def create_data_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/validation/test splits for dataset.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        test_ratio: Ratio of test samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total_samples = len(dataset)
    indices = list(range(total_samples))
    
    # Shuffle indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    # Create splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices


def create_stratified_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/validation/test splits for dataset.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
        test_ratio: Ratio of test samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get labels for stratification
    labels = []
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            label = sample['label']
            if isinstance(label, (np.ndarray, torch.Tensor)):
                label = int(label.item())
            else:
                label = int(label)
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error getting label for sample {idx}: {e}")
            labels.append(0)  # Default label
    
    # Create splits
    indices = list(range(len(dataset)))
    
    # First split: train vs (val + test)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=(val_ratio + test_ratio), random_state=random_seed, stratify=labels
    )
    
    # Second split: val vs test
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=(test_ratio / (val_ratio + test_ratio)), 
        random_state=random_seed, stratify=temp_labels
    )
    
    return train_indices, val_indices, test_indices


def normalize_features(features: np.ndarray, method: str = 'z_score') -> np.ndarray:
    """
    Normalize features.
    
    Args:
        features: Input features
        method: Normalization method ('z_score', 'min_max', 'robust')
        
    Returns:
        Normalized features
    """
    if method == 'z_score':
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        return (features - mean) / (std + 1e-8)
    
    elif method == 'min_max':
        min_val = np.min(features, axis=0, keepdims=True)
        max_val = np.max(features, axis=0, keepdims=True)
        return (features - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'robust':
        median = np.median(features, axis=0, keepdims=True)
        mad = np.median(np.abs(features - median), axis=0, keepdims=True)
        return (features - median) / (mad + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def augment_data(data: Dict[str, Any], augmentation_config: Dict) -> Dict[str, Any]:
    """
    Apply data augmentation to multimodal data.
    
    Args:
        data: Input data
        augmentation_config: Augmentation configuration
        
    Returns:
        Augmented data
    """
    augmented_data = data.copy()
    
    # Apply audio augmentation
    if 'audio' in data and 'audio' in augmentation_config:
        try:
            from .preprocessing.audio_processor import AudioProcessor
            processor = AudioProcessor()
            augmented_data['audio'] = processor.augment_audio(
                data['audio'], 
                augmentation_config['audio'].get('techniques', [])
            )
        except Exception as e:
            logger.warning(f"Error augmenting audio: {e}")
    
    # Apply video augmentation
    if 'video' in data and 'video' in augmentation_config:
        try:
            from .preprocessing.facial_processor import FacialProcessor
            processor = FacialProcessor()
            augmented_data['video'] = processor.augment_facial_data(
                data['video'], 
                augmentation_config['video'].get('techniques', [])
            )
        except Exception as e:
            logger.warning(f"Error augmenting video: {e}")
    
    # Apply gait augmentation
    if 'gait' in data and 'gait' in augmentation_config:
        try:
            from .preprocessing.gait_processor import GaitProcessor
            processor = GaitProcessor()
            augmented_data['gait'] = processor.augment_gait_data(
                data['gait'].values, 
                augmentation_config['gait'].get('techniques', [])
            )
        except Exception as e:
            logger.warning(f"Error augmenting gait: {e}")
    
    return augmented_data
