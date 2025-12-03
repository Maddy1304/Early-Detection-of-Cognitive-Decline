"""
Base dataset class for multimodal cognitive decline detection.

This module provides the base class for all dataset implementations,
defining common interfaces and functionality.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """Base class for all multimodal datasets."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        modalities: List[str] = None,
        preprocess: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        """
        Initialize base dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split to use
            modalities: List of modalities to load
            preprocess: Whether to preprocess the data
            transform: Transform to apply to input data
            target_transform: Transform to apply to target labels
        """
        self.data_dir = data_dir
        self.split = split
        self.modalities = modalities or []
        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        
        # Dataset metadata
        self.dataset_name = "BaseDataset"
        self.version = "1.0.0"
        self.description = "Base dataset for multimodal cognitive decline detection"
        
        # Initialize dataset
        self._initialize_dataset()
    
    @abstractmethod
    def _initialize_dataset(self):
        """Initialize dataset-specific components."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'dataset_name': self.dataset_name,
            'version': self.version,
            'description': self.description,
            'split': self.split,
            'modalities': self.modalities,
            'total_samples': len(self)
        }
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a sample for completeness and correctness."""
        try:
            # Check required keys
            required_keys = ['label']
            for key in required_keys:
                if key not in sample:
                    logger.warning(f"Missing required key: {key}")
                    return False
            
            # Check modality data
            for modality in self.modalities:
                if modality not in sample:
                    logger.warning(f"Missing modality: {modality}")
                    return False
            
            # Check data types and shapes
            if 'label' in sample:
                if not isinstance(sample['label'], (int, float, np.ndarray, torch.Tensor)):
                    logger.warning(f"Invalid label type: {type(sample['label'])}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating sample: {e}")
            return False
    
    def filter_samples(self, filter_func: callable) -> List[int]:
        """Filter samples based on a function."""
        valid_indices = []
        
        for idx in range(len(self)):
            try:
                sample = self[idx]
                if filter_func(sample):
                    valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        return valid_indices
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in the dataset."""
        class_counts = {}
        
        for idx in range(len(self)):
            try:
                sample = self[idx]
                label = sample['label']
                
                if isinstance(label, (np.ndarray, torch.Tensor)):
                    label = int(label.item())
                else:
                    label = int(label)
                
                class_counts[label] = class_counts.get(label, 0) + 1
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        return class_counts
    
    def get_modality_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each modality."""
        modality_stats = {}
        
        for modality in self.modalities:
            modality_stats[modality] = {
                'count': 0,
                'shapes': [],
                'dtypes': [],
                'ranges': []
            }
        
        for idx in range(len(self)):
            try:
                sample = self[idx]
                
                for modality in self.modalities:
                    if modality in sample:
                        data = sample[modality]
                        modality_stats[modality]['count'] += 1
                        
                        if isinstance(data, (np.ndarray, torch.Tensor)):
                            modality_stats[modality]['shapes'].append(data.shape)
                            modality_stats[modality]['dtypes'].append(str(data.dtype))
                            
                            if data.size > 0:
                                modality_stats[modality]['ranges'].append({
                                    'min': float(data.min()),
                                    'max': float(data.max()),
                                    'mean': float(data.mean())
                                })
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue
        
        return modality_stats
    
    def create_subset(self, indices: List[int]) -> 'BaseDataset':
        """Create a subset of the dataset."""
        # This is a simplified implementation
        # Subclasses should override this method for better performance
        class Subset(BaseDataset):
            def __init__(self, parent_dataset, indices):
                self.parent = parent_dataset
                self.indices = indices
                super().__init__(
                    parent_dataset.data_dir,
                    parent_dataset.split,
                    parent_dataset.modalities,
                    parent_dataset.preprocess,
                    parent_dataset.transform,
                    parent_dataset.target_transform
                )
            
            def _initialize_dataset(self):
                pass
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.parent[self.indices[idx]]
        
        return Subset(self, indices)
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
        """Split dataset into train/val/test sets."""
        total_samples = len(self)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        # Create indices
        indices = list(range(total_samples))
        
        # Shuffle indices
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices
    
    def save_metadata(self, filepath: str):
        """Save dataset metadata to file."""
        import json
        
        metadata = {
            'dataset_name': self.dataset_name,
            'version': self.version,
            'description': self.description,
            'split': self.split,
            'modalities': self.modalities,
            'total_samples': len(self),
            'statistics': self.get_statistics(),
            'class_distribution': self.get_class_distribution(),
            'modality_statistics': self.get_modality_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {filepath}")
    
    def load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Load dataset metadata from file."""
        import json
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata from {filepath}")
        return metadata
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"{self.__class__.__name__}(dataset_name='{self.dataset_name}', split='{self.split}', total_samples={len(self)})"
    
    def __str__(self) -> str:
        """String representation of the dataset."""
        return self.__repr__()
