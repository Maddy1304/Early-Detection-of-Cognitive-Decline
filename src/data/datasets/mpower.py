"""
mPower (Mobile Parkinson's Disease) dataset handler.

This module provides functionality for loading and processing the mPower dataset,
which contains sensor data for Parkinson's disease detection and monitoring.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MPowerDataset(BaseDataset):
    """mPower dataset for Parkinson's disease detection and monitoring."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # train, val, test
        modalities: List[str] = None,
        preprocess: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False
    ):
        """
        Initialize mPower dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split to use
            modalities: List of modalities to load ('gait', 'tapping', 'voice', 'memory')
            preprocess: Whether to preprocess the data
            transform: Transform to apply to input data
            target_transform: Transform to apply to target labels
            download: Whether to download the dataset if not present
        """
        super().__init__(data_dir, split, modalities, preprocess, transform, target_transform)
        
        self.dataset_name = "mPower"
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ['gait', 'tapping', 'voice', 'memory']
        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        
        # Dataset-specific parameters
        self.sampling_rate = 50  # Hz for sensor data
        self.voice_sampling_rate = 16000  # Hz for voice data
        self.max_gait_duration = 120  # seconds
        self.max_tapping_duration = 20  # seconds
        self.max_voice_duration = 10  # seconds
        
        # Initialize dataset
        self._initialize_dataset()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create data index
        self.data_index = self._create_data_index()
        
        logger.info(f"Initialized mPower dataset with {len(self.data_index)} samples")
    
    def _initialize_dataset(self):
        """Initialize dataset structure and check for required files."""
        # Check if data directory exists
        if not self.data_dir.exists():
            if self.download:
                self._download_dataset()
            else:
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check for required subdirectories
        required_dirs = ['gait', 'tapping', 'voice', 'memory', 'metadata']
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                logger.warning(f"Required directory not found: {dir_path}")
    
    def _download_dataset(self):
        """Download mPower dataset."""
        logger.info("Downloading mPower dataset...")
        # Note: mPower dataset is available through Sage Bionetworks
        # This is a placeholder for the download process
        raise NotImplementedError(
            "mPower dataset download requires registration. "
            "Please visit https://www.synapse.org/#!Synapse:syn4993293 to register and download the dataset."
        )
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        metadata_file = self.data_dir / 'metadata' / 'participants.csv'
        
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file)
        else:
            # Create metadata from available files
            metadata = self._create_metadata_from_files()
        
        # Filter by split if specified
        if 'split' in metadata.columns:
            metadata = metadata[metadata['split'] == self.split]
        
        return metadata
    
    def _create_metadata_from_files(self) -> pd.DataFrame:
        """Create metadata from available files."""
        metadata = []
        
        # Get all participant directories
        for participant_dir in self.data_dir.iterdir():
            if participant_dir.is_dir() and participant_dir.name.startswith('P'):
                participant_id = participant_dir.name
                
                # Check for available modalities
                gait_path = self.data_dir / 'gait' / f"{participant_id}.csv"
                tapping_path = self.data_dir / 'tapping' / f"{participant_id}.csv"
                voice_path = self.data_dir / 'voice' / f"{participant_id}.wav"
                memory_path = self.data_dir / 'memory' / f"{participant_id}.json"
                
                metadata.append({
                    'participant_id': participant_id,
                    'gait_path': str(gait_path) if gait_path.exists() else None,
                    'tapping_path': str(tapping_path) if tapping_path.exists() else None,
                    'voice_path': str(voice_path) if voice_path.exists() else None,
                    'memory_path': str(memory_path) if memory_path.exists() else None,
                    'split': self._assign_split(participant_id)
                })
        
        return pd.DataFrame(metadata)
    
    def _assign_split(self, participant_id: str) -> str:
        """Assign split based on participant ID."""
        # Simple split assignment (can be customized)
        participant_num = int(participant_id[1:])  # Remove 'P' prefix
        
        if participant_num % 10 < 7:
            return 'train'
        elif participant_num % 10 < 9:
            return 'val'
        else:
            return 'test'
    
    def _create_data_index(self) -> List[Dict]:
        """Create index of available data samples."""
        data_index = []
        
        for _, row in self.metadata.iterrows():
            sample = {
                'participant_id': row['participant_id'],
                'gait_path': row.get('gait_path'),
                'tapping_path': row.get('tapping_path'),
                'voice_path': row.get('voice_path'),
                'memory_path': row.get('memory_path'),
                'split': row.get('split', self.split)
            }
            
            # Check if required modalities are available
            available_modalities = []
            if sample['gait_path'] and os.path.exists(sample['gait_path']):
                available_modalities.append('gait')
            if sample['tapping_path'] and os.path.exists(sample['tapping_path']):
                available_modalities.append('tapping')
            if sample['voice_path'] and os.path.exists(sample['voice_path']):
                available_modalities.append('voice')
            if sample['memory_path'] and os.path.exists(sample['memory_path']):
                available_modalities.append('memory')
            
            # Only include samples with at least one required modality
            if any(mod in available_modalities for mod in self.modalities):
                sample['available_modalities'] = available_modalities
                data_index.append(sample)
        
        return data_index
    
    def _load_gait_data(self, gait_path: str) -> pd.DataFrame:
        """Load gait sensor data."""
        try:
            data = pd.read_csv(gait_path)
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'accel_x': 'acc_x',
                'accel_y': 'acc_y', 
                'accel_z': 'acc_z',
                'gyro_x': 'gyro_x',
                'gyro_y': 'gyro_y',
                'gyro_z': 'gyro_z'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data = data.rename(columns={old_name: new_name})
            
            # Ensure required columns exist
            required_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 0.0
            
            # Truncate to max duration
            max_samples = self.max_gait_duration * self.sampling_rate
            if len(data) > max_samples:
                data = data.iloc[:max_samples]
            
            return data
        except Exception as e:
            logger.error(f"Error loading gait data {gait_path}: {e}")
            return pd.DataFrame()
    
    def _load_tapping_data(self, tapping_path: str) -> pd.DataFrame:
        """Load tapping data."""
        try:
            data = pd.read_csv(tapping_path)
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'tap_x': 'tap_x',
                'tap_y': 'tap_y',
                'pressure': 'pressure'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns:
                    data = data.rename(columns={old_name: new_name})
            
            # Ensure required columns exist
            required_columns = ['tap_x', 'tap_y', 'pressure']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = 0.0
            
            # Truncate to max duration
            max_samples = self.max_tapping_duration * self.sampling_rate
            if len(data) > max_samples:
                data = data.iloc[:max_samples]
            
            return data
        except Exception as e:
            logger.error(f"Error loading tapping data {tapping_path}: {e}")
            return pd.DataFrame()
    
    def _load_voice_data(self, voice_path: str) -> np.ndarray:
        """Load voice data."""
        try:
            import librosa
            audio, sr = librosa.load(voice_path, sr=self.voice_sampling_rate)
            
            # Truncate to max duration
            max_samples = self.max_voice_duration * self.voice_sampling_rate
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            logger.error(f"Error loading voice data {voice_path}: {e}")
            return np.zeros(self.max_voice_duration * self.voice_sampling_rate)
    
    def _load_memory_data(self, memory_path: str) -> Dict:
        """Load memory test data."""
        try:
            with open(memory_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading memory data {memory_path}: {e}")
            return {}
    
    def _extract_gait_features(self, gait_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract gait features."""
        try:
            from ..preprocessing.gait_processor import GaitProcessor
            
            processor = GaitProcessor(sampling_rate=self.sampling_rate)
            features = processor.process_gait_data(gait_data, extract_features=True)
            
            return {
                'temporal_features': features.get('temporal_features', {}),
                'frequency_features': features.get('frequency_features', {}),
                'gait_features': features.get('gait_features', {}),
                'balance_features': features.get('balance_features', {})
            }
        except Exception as e:
            logger.error(f"Error extracting gait features: {e}")
            return {
                'temporal_features': {},
                'frequency_features': {},
                'gait_features': {},
                'balance_features': {}
            }
    
    def _extract_tapping_features(self, tapping_data: pd.DataFrame) -> Dict[str, float]:
        """Extract tapping features."""
        try:
            features = {}
            
            if len(tapping_data) == 0:
                return features
            
            # Basic tapping features
            features['tap_count'] = len(tapping_data)
            features['tap_duration'] = (tapping_data['timestamp'].iloc[-1] - tapping_data['timestamp'].iloc[0]) / 1000.0  # Convert to seconds
            
            if features['tap_count'] > 0:
                features['tap_frequency'] = features['tap_count'] / features['tap_duration']
            else:
                features['tap_frequency'] = 0.0
            
            # Spatial features
            if 'tap_x' in tapping_data.columns and 'tap_y' in tapping_data.columns:
                features['tap_x_mean'] = tapping_data['tap_x'].mean()
                features['tap_y_mean'] = tapping_data['tap_y'].mean()
                features['tap_x_std'] = tapping_data['tap_x'].std()
                features['tap_y_std'] = tapping_data['tap_y'].std()
                features['tap_spread'] = np.sqrt(features['tap_x_std']**2 + features['tap_y_std']**2)
            
            # Pressure features
            if 'pressure' in tapping_data.columns:
                features['pressure_mean'] = tapping_data['pressure'].mean()
                features['pressure_std'] = tapping_data['pressure'].std()
                features['pressure_max'] = tapping_data['pressure'].max()
                features['pressure_min'] = tapping_data['pressure'].min()
            
            # Timing features
            if len(tapping_data) > 1:
                time_diffs = tapping_data['timestamp'].diff().dropna()
                features['tap_interval_mean'] = time_diffs.mean() / 1000.0  # Convert to seconds
                features['tap_interval_std'] = time_diffs.std() / 1000.0
                features['tap_regularity'] = 1.0 / (1.0 + features['tap_interval_std'] / features['tap_interval_mean']) if features['tap_interval_mean'] > 0 else 0.0
            
            return features
        except Exception as e:
            logger.error(f"Error extracting tapping features: {e}")
            return {}
    
    def _extract_voice_features(self, voice_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract voice features."""
        try:
            from ..preprocessing.audio_processor import AudioProcessor
            
            processor = AudioProcessor(sample_rate=self.voice_sampling_rate)
            features = processor.process_audio(voice_data, extract_features=True)
            
            return {
                'mfcc': features.get('mfcc', np.zeros((39, 100))),
                'spectral_features': features.get('spectral_features', {}),
                'prosodic_features': features.get('prosodic_features', {})
            }
        except Exception as e:
            logger.error(f"Error extracting voice features: {e}")
            return {
                'mfcc': np.zeros((39, 100)),
                'spectral_features': {},
                'prosodic_features': {}
            }
    
    def _extract_memory_features(self, memory_data: Dict) -> Dict[str, float]:
        """Extract memory test features."""
        try:
            features = {}
            
            if not memory_data:
                return features
            
            # Basic memory features
            features['total_questions'] = memory_data.get('total_questions', 0)
            features['correct_answers'] = memory_data.get('correct_answers', 0)
            features['incorrect_answers'] = memory_data.get('incorrect_answers', 0)
            
            if features['total_questions'] > 0:
                features['accuracy'] = features['correct_answers'] / features['total_questions']
            else:
                features['accuracy'] = 0.0
            
            # Timing features
            features['total_time'] = memory_data.get('total_time', 0)
            features['avg_response_time'] = memory_data.get('avg_response_time', 0)
            features['response_time_std'] = memory_data.get('response_time_std', 0)
            
            # Difficulty features
            features['easy_accuracy'] = memory_data.get('easy_accuracy', 0)
            features['medium_accuracy'] = memory_data.get('medium_accuracy', 0)
            features['hard_accuracy'] = memory_data.get('hard_accuracy', 0)
            
            return features
        except Exception as e:
            logger.error(f"Error extracting memory features: {e}")
            return {}
    
    def _get_label(self, participant_id: str) -> int:
        """Get label for participant (Parkinson's disease severity)."""
        # This is a placeholder - actual labels would come from clinical assessments
        # In the real mPower dataset, labels are based on UPDRS scores
        
        # Simple label assignment based on participant ID (for demonstration)
        participant_num = int(participant_id[1:])  # Remove 'P' prefix
        
        # Simulate Parkinson's severity (0: normal, 1: mild, 2: moderate, 3: severe)
        if participant_num % 4 == 0:
            return 0  # Normal
        elif participant_num % 4 == 1:
            return 1  # Mild
        elif participant_num % 4 == 2:
            return 2  # Moderate
        else:
            return 3  # Severe
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        sample_info = self.data_index[idx]
        
        # Load data for available modalities
        data = {
            'participant_id': sample_info['participant_id'],
            'split': sample_info['split']
        }
        
        # Load gait data
        if 'gait' in self.modalities and sample_info['gait_path']:
            gait_data = self._load_gait_data(sample_info['gait_path'])
            data['gait'] = gait_data
            
            if self.preprocess:
                data['gait_features'] = self._extract_gait_features(gait_data)
        
        # Load tapping data
        if 'tapping' in self.modalities and sample_info['tapping_path']:
            tapping_data = self._load_tapping_data(sample_info['tapping_path'])
            data['tapping'] = tapping_data
            
            if self.preprocess:
                data['tapping_features'] = self._extract_tapping_features(tapping_data)
        
        # Load voice data
        if 'voice' in self.modalities and sample_info['voice_path']:
            voice_data = self._load_voice_data(sample_info['voice_path'])
            data['voice'] = voice_data
            
            if self.preprocess:
                data['voice_features'] = self._extract_voice_features(voice_data)
        
        # Load memory data
        if 'memory' in self.modalities and sample_info['memory_path']:
            memory_data = self._load_memory_data(sample_info['memory_path'])
            data['memory'] = memory_data
            
            if self.preprocess:
                data['memory_features'] = self._extract_memory_features(memory_data)
        
        # Get label
        label = self._get_label(sample_info['participant_id'])
        data['label'] = label
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        if self.target_transform:
            data['label'] = self.target_transform(data['label'])
        
        return data
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            'total_samples': len(self.data_index),
            'modalities': self.modalities,
            'split': self.split,
            'sampling_rate': self.sampling_rate,
            'voice_sampling_rate': self.voice_sampling_rate,
            'max_gait_duration': self.max_gait_duration,
            'max_tapping_duration': self.max_tapping_duration,
            'max_voice_duration': self.max_voice_duration
        }
        
        # Count samples by modality
        modality_counts = {}
        for modality in self.modalities:
            count = sum(1 for sample in self.data_index 
                       if modality in sample['available_modalities'])
            modality_counts[modality] = count
        
        stats['modality_counts'] = modality_counts
        
        # Count samples by label
        label_counts = {}
        for sample in self.data_index:
            label = self._get_label(sample['participant_id'])
            label_counts[label] = label_counts.get(label, 0) + 1
        
        stats['label_counts'] = label_counts
        
        return stats
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create DataLoader for the dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function for batching."""
        collated = {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['participant_id', 'split']:
                # Keep as list of strings
                collated[key] = [sample[key] for sample in batch]
            elif key == 'label':
                # Convert to tensor
                collated[key] = torch.tensor([sample[key] for sample in batch])
            elif key in ['gait', 'tapping']:
                # Keep as list of DataFrames
                collated[key] = [sample[key] for sample in batch]
            elif key in ['voice']:
                # Stack arrays
                collated[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
            elif key in ['memory']:
                # Keep as list of dictionaries
                collated[key] = [sample[key] for sample in batch]
            elif key.endswith('_features'):
                # Keep as list of dictionaries
                collated[key] = [sample[key] for sample in batch]
            else:
                # Keep as list
                collated[key] = [sample[key] for sample in batch]
        
        return collated
