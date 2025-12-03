"""
DAIC-WOZ (Depression and Anxiety in Context) dataset handler.

This module provides functionality for loading and processing the DAIC-WOZ dataset,
which contains audio-visual data for depression and anxiety detection.
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


class DAICWOZDataset(BaseDataset):
    """DAIC-WOZ dataset for depression and anxiety detection."""
    
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
        Initialize DAIC-WOZ dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split to use
            modalities: List of modalities to load ('audio', 'video', 'text')
            preprocess: Whether to preprocess the data
            transform: Transform to apply to input data
            target_transform: Transform to apply to target labels
            download: Whether to download the dataset if not present
        """
        super().__init__(data_dir, split, modalities, preprocess, transform, target_transform)
        
        self.dataset_name = "DAIC-WOZ"
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or ['audio', 'video', 'text']
        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        
        # Dataset-specific parameters
        self.sample_rate = 16000
        self.video_fps = 30
        self.max_audio_length = 300  # 5 minutes in seconds
        self.max_video_length = 300  # 5 minutes in seconds
        
        # Initialize dataset
        self._initialize_dataset()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create data index
        self.data_index = self._create_data_index()
        
        logger.info(f"Initialized DAIC-WOZ dataset with {len(self.data_index)} samples")
    
    def _initialize_dataset(self):
        """Initialize dataset structure and check for required files."""
        # Check if data directory exists
        if not self.data_dir.exists():
            if self.download:
                self._download_dataset()
            else:
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check for required subdirectories
        required_dirs = ['audio', 'video', 'transcripts', 'metadata']
        for dir_name in required_dirs:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                logger.warning(f"Required directory not found: {dir_path}")
    
    def _download_dataset(self):
        """Download DAIC-WOZ dataset."""
        logger.info("Downloading DAIC-WOZ dataset...")
        # Note: DAIC-WOZ requires registration and agreement to terms
        # This is a placeholder for the download process
        raise NotImplementedError(
            "DAIC-WOZ dataset download requires registration. "
            "Please visit https://dcapswoz.ict.usc.edu/ to register and download the dataset."
        )
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata."""
        metadata_file = self.data_dir / 'metadata' / 'metadata.csv'
        
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
                
                # Get session files
                for session_file in participant_dir.glob('*'):
                    if session_file.is_file():
                        session_id = session_file.stem
                        
                        # Check for available modalities
                        audio_path = self.data_dir / 'audio' / f"{session_id}.wav"
                        video_path = self.data_dir / 'video' / f"{session_id}.mp4"
                        transcript_path = self.data_dir / 'transcripts' / f"{session_id}.txt"
                        
                        metadata.append({
                            'participant_id': participant_id,
                            'session_id': session_id,
                            'audio_path': str(audio_path) if audio_path.exists() else None,
                            'video_path': str(video_path) if video_path.exists() else None,
                            'transcript_path': str(transcript_path) if transcript_path.exists() else None,
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
                'session_id': row['session_id'],
                'audio_path': row.get('audio_path'),
                'video_path': row.get('video_path'),
                'transcript_path': row.get('transcript_path'),
                'split': row.get('split', self.split)
            }
            
            # Check if required modalities are available
            available_modalities = []
            if sample['audio_path'] and os.path.exists(sample['audio_path']):
                available_modalities.append('audio')
            if sample['video_path'] and os.path.exists(sample['video_path']):
                available_modalities.append('video')
            if sample['transcript_path'] and os.path.exists(sample['transcript_path']):
                available_modalities.append('text')
            
            # Only include samples with at least one required modality
            if any(mod in available_modalities for mod in self.modalities):
                sample['available_modalities'] = available_modalities
                data_index.append(sample)
        
        return data_index
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio data."""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Truncate or pad to max length
            max_samples = self.max_audio_length * self.sample_rate
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            elif len(audio) < max_samples:
                audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return np.zeros(self.max_audio_length * self.sample_rate)
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video data."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                return np.zeros((self.max_video_length * self.video_fps, 224, 224, 3))
            
            # Convert to numpy array
            video = np.array(frames)
            
            # Truncate or pad to max length
            max_frames = self.max_video_length * self.video_fps
            if len(video) > max_frames:
                video = video[:max_frames]
            elif len(video) < max_frames:
                # Pad with last frame
                last_frame = video[-1] if len(video) > 0 else np.zeros((224, 224, 3))
                padding = np.tile(last_frame, (max_frames - len(video), 1, 1, 1))
                video = np.concatenate([video, padding], axis=0)
            
            return video
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return np.zeros((self.max_video_length * self.video_fps, 224, 224, 3))
    
    def _load_transcript(self, transcript_path: str) -> str:
        """Load transcript text."""
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            return transcript
        except Exception as e:
            logger.error(f"Error loading transcript {transcript_path}: {e}")
            return ""
    
    def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract audio features."""
        try:
            from ..preprocessing.audio_processor import AudioProcessor
            
            processor = AudioProcessor(sample_rate=self.sample_rate)
            features = processor.process_audio(audio, extract_features=True)
            
            return {
                'mfcc': features.get('mfcc', np.zeros((39, 100))),
                'spectral_features': features.get('spectral_features', {}),
                'prosodic_features': features.get('prosodic_features', {})
            }
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {
                'mfcc': np.zeros((39, 100)),
                'spectral_features': {},
                'prosodic_features': {}
            }
    
    def _extract_video_features(self, video: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract video features."""
        try:
            from ..preprocessing.facial_processor import FacialProcessor
            
            processor = FacialProcessor()
            features = []
            
            # Process each frame
            for frame in video:
                frame_features = processor.process_facial_data(frame, extract_features=True)
                features.append(frame_features)
            
            return {
                'geometric_features': [f.get('geometric_features', {}) for f in features],
                'texture_features': [f.get('texture_features', {}) for f in features],
                'emotions': [f.get('emotions', {}) for f in features]
            }
        except Exception as e:
            logger.error(f"Error extracting video features: {e}")
            return {
                'geometric_features': [],
                'texture_features': [],
                'emotions': []
            }
    
    def _extract_text_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract text features."""
        try:
            # Simple text features (can be enhanced with more sophisticated NLP)
            words = text.split()
            sentences = text.split('.')
            
            features = {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'avg_sentence_length': np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0,
                'text_length': len(text)
            }
            
            return features
        except Exception as e:
            logger.error(f"Error extracting text features: {e}")
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'avg_sentence_length': 0,
                'text_length': 0
            }
    
    def _get_label(self, participant_id: str) -> int:
        """Get label for participant (depression/anxiety severity)."""
        # This is a placeholder - actual labels would come from clinical assessments
        # In the real DAIC-WOZ dataset, labels are based on PHQ-8 scores
        
        # Simple label assignment based on participant ID (for demonstration)
        participant_num = int(participant_id[1:])  # Remove 'P' prefix
        
        # Simulate depression severity (0: normal, 1: mild, 2: moderate, 3: severe)
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
            'session_id': sample_info['session_id'],
            'split': sample_info['split']
        }
        
        # Load audio
        if 'audio' in self.modalities and sample_info['audio_path']:
            audio = self._load_audio(sample_info['audio_path'])
            data['audio'] = audio
            
            if self.preprocess:
                data['audio_features'] = self._extract_audio_features(audio)
        
        # Load video
        if 'video' in self.modalities and sample_info['video_path']:
            video = self._load_video(sample_info['video_path'])
            data['video'] = video
            
            if self.preprocess:
                data['video_features'] = self._extract_video_features(video)
        
        # Load transcript
        if 'text' in self.modalities and sample_info['transcript_path']:
            transcript = self._load_transcript(sample_info['transcript_path'])
            data['text'] = transcript
            
            if self.preprocess:
                data['text_features'] = self._extract_text_features(transcript)
        
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
            'sample_rate': self.sample_rate,
            'video_fps': self.video_fps,
            'max_audio_length': self.max_audio_length,
            'max_video_length': self.max_video_length
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
            if key in ['participant_id', 'session_id', 'split']:
                # Keep as list of strings
                collated[key] = [sample[key] for sample in batch]
            elif key == 'label':
                # Convert to tensor
                collated[key] = torch.tensor([sample[key] for sample in batch])
            elif key in ['audio', 'video']:
                # Stack arrays
                collated[key] = torch.stack([torch.from_numpy(sample[key]) for sample in batch])
            elif key.endswith('_features'):
                # Keep as list of dictionaries
                collated[key] = [sample[key] for sample in batch]
            else:
                # Keep as list
                collated[key] = [sample[key] for sample in batch]
        
        return collated
