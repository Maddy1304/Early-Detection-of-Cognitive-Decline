"""
RAVDESS (Ryerson Audio-Visual Database) dataset handler.

This module provides functionality for loading and processing the RAVDESS dataset,
which contains audio-visual emotional speech data.
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


class RAVDESSDataset(BaseDataset):
    """RAVDESS dataset for emotional speech recognition."""
    
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
        Initialize RAVDESS dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split to use
            modalities: List of modalities to load ('audio', 'video')
            preprocess: Whether to preprocess the data
            transform: Transform to apply to input data
            target_transform: Transform to apply to target labels
            download: Whether to download the dataset if not present
        """
        # Convert to Path first and override base class data_dir immediately
        data_dir_path = Path(data_dir)
        super().__init__(str(data_dir), split, modalities, preprocess, transform, target_transform)
        
        # Override data_dir to be Path object BEFORE _initialize_dataset is called
        self.data_dir = data_dir_path
        self.dataset_name = "RAVDESS"
        self.split = split
        self.modalities = modalities or ['audio', 'video']
        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        
        # Dataset-specific parameters
        self.sample_rate = 16000
        self.video_fps = 30
        self.max_audio_length = 4  # seconds
        self.max_video_length = 4  # seconds
        
        # Initialize processors once (reuse for all samples)
        self.audio_processor = None
        self.facial_processor = None
        if preprocess:
            if 'audio' in self.modalities:
                try:
                    from src.data.preprocessing.audio_processor import AudioProcessor as AP
                    self.audio_processor = AP(sample_rate=self.sample_rate)
                    logger.info("AudioProcessor initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize AudioProcessor: {e}")
            
            if 'video' in self.modalities:
                try:
                    from src.data.preprocessing.facial_processor import FacialProcessor as FP
                    self.facial_processor = FP()
                    logger.info("FacialProcessor initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize FacialProcessor: {e}")
        
        # Emotion labels
        self.emotion_labels = {
            1: 'neutral',
            2: 'calm',
            3: 'happy',
            4: 'sad',
            5: 'angry',
            6: 'fearful',
            7: 'disgust',
            8: 'surprised'
        }
        
        # Intensity labels
        self.intensity_labels = {
            1: 'normal',
            2: 'strong'
        }
        
        # Statement labels
        self.statement_labels = {
            1: 'kids are talking by the door',
            2: 'dogs are sitting by the door'
        }
        
        # Repetition labels
        self.repetition_labels = {
            1: 'first',
            2: 'second'
        }
        
        # Initialize dataset (data_dir is now Path object)
        self._initialize_dataset()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create data index
        self.data_index = self._create_data_index()
        
        logger.info(f"Initialized RAVDESS dataset with {len(self.data_index)} samples")
    
    def _initialize_dataset(self):
        """Initialize dataset structure and check for required files."""
        # Ensure data_dir is a Path object
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        
        # Check if data directory exists
        if not self.data_dir.exists():
            if self.download:
                self._download_dataset()
            else:
                raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check for dataset structure - support both old and new formats
        # New format: Audio_Speech_Actors_01-24/, Video_Speech_Actors_01-24/
        # Old format: audio/, video/, metadata/
        audio_dirs = [
            self.data_dir / 'Audio_Speech_Actors_01-24',
            self.data_dir / 'audio'
        ]
        video_dirs = [
            self.data_dir / 'Video_Speech_Actors_01-24',
            self.data_dir / 'video'
        ]
        
        # Check if any audio directory exists and has files
        audio_found = False
        for audio_dir in audio_dirs:
            if audio_dir.exists() and list(audio_dir.glob('*.wav')):
                audio_found = True
                logger.info(f"Found audio files in: {audio_dir}")
                break
        
        # Check if any video directory exists and has files
        video_found = False
        for video_dir in video_dirs:
            if video_dir.exists() and (list(video_dir.glob('*.mp4')) or list(video_dir.glob('*.avi')) or list(video_dir.glob('*.mov'))):
                video_found = True
                logger.info(f"Found video files in: {video_dir}")
                break
        
        # Only warn if neither structure is found
        if not audio_found and 'audio' in self.modalities:
            logger.warning(f"No audio files found. Checked: {[str(d) for d in audio_dirs]}")
        if not video_found and 'video' in self.modalities:
            logger.warning(f"No video files found. Checked: {[str(d) for d in video_dirs]}")
    
    def _download_dataset(self):
        """Download RAVDESS dataset."""
        logger.info("Downloading RAVDESS dataset...")
        # Note: RAVDESS dataset is available through Zenodo
        # This is a placeholder for the download process
        raise NotImplementedError(
            "RAVDESS dataset download requires registration. "
            "Please visit https://zenodo.org/record/1188976 to download the dataset."
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
        
        # Track processed filenames to avoid duplicates
        processed_files = set()
        
        # Get all audio files - check multiple possible locations
        audio_dirs = [
            self.data_dir / 'audio',
            self.data_dir / 'Audio_Speech_Actors_01-24',
            self.data_dir
        ]
        
        audio_dir = None
        for ad in audio_dirs:
            if ad.exists() and list(ad.glob('*.wav')):
                audio_dir = ad
                break
        
        # Get all video files - check multiple possible locations
        video_dirs = [
            self.data_dir / 'video',
            self.data_dir / 'Video_Speech_Actors_01-24',
            self.data_dir
        ]
        
        video_dir = None
        for vd in video_dirs:
            if vd.exists() and (list(vd.glob('*.mp4')) or list(vd.glob('*.avi')) or list(vd.glob('*.mov'))):
                video_dir = vd
                break
        
        # Process audio files first (if audio modality is needed)
        if audio_dir and audio_dir.exists() and ('audio' in self.modalities or len(self.modalities) == 0):
            for audio_file in audio_dir.glob('*.wav'):
                filename = audio_file.stem
                if filename in processed_files:
                    continue
                
                # Parse filename: 03-01-01-01-01-01-01.wav
                # Format: Modality(01)-Vocal(01)-Emotion(01-08)-Intensity(01-02)-Statement(01-02)-Repetition(01-02)-Actor(01-24)
                parts = filename.split('-')
                
                if len(parts) >= 7:
                    try:
                        modality = int(parts[0])
                        vocal = int(parts[1])
                        emotion = int(parts[2])
                        intensity = int(parts[3])
                        statement = int(parts[4])
                        repetition = int(parts[5])
                        actor = int(parts[6])
                        
                        # Check for video file
                        video_path = None
                        if video_dir:
                            for ext in ['.mp4', '.avi', '.mov']:
                                video_file = video_dir / f"{filename}{ext}"
                                if video_file.exists():
                                    video_path = str(video_file)
                                    break
                        
                        metadata.append({
                            'filename': filename,
                            'modality': modality,
                            'vocal': vocal,
                            'emotion': emotion,
                            'intensity': intensity,
                            'statement': statement,
                            'repetition': repetition,
                            'actor': actor,
                            'audio_path': str(audio_file),
                            'video_path': video_path,
                            'split': self._assign_split(actor)
                        })
                        processed_files.add(filename)
                    except ValueError:
                        logger.warning(f"Could not parse filename: {filename}")
        
        # Process video files (if only video modality is needed or video wasn't found via audio)
        if video_dir and video_dir.exists() and 'video' in self.modalities:
            for video_file in video_dir.glob('*.mp4'):
                filename = video_file.stem
                if filename in processed_files:
                    continue
                
                # Parse filename
                parts = filename.split('-')
                
                if len(parts) >= 7:
                    try:
                        modality = int(parts[0])
                        vocal = int(parts[1])
                        emotion = int(parts[2])
                        intensity = int(parts[3])
                        statement = int(parts[4])
                        repetition = int(parts[5])
                        actor = int(parts[6])
                        
                        # Check for audio file
                        audio_path = None
                        if audio_dir:
                            audio_file = audio_dir / f"{filename}.wav"
                            if audio_file.exists():
                                audio_path = str(audio_file)
                        
                        metadata.append({
                            'filename': filename,
                            'modality': modality,
                            'vocal': vocal,
                            'emotion': emotion,
                            'intensity': intensity,
                            'statement': statement,
                            'repetition': repetition,
                            'actor': actor,
                            'audio_path': audio_path,
                            'video_path': str(video_file),
                            'split': self._assign_split(actor)
                        })
                        processed_files.add(filename)
                    except ValueError:
                        logger.warning(f"Could not parse filename: {filename}")
        
        logger.info(f"Created metadata for {len(metadata)} samples from files")
        return pd.DataFrame(metadata)
    
    def _assign_split(self, actor: int) -> str:
        """Assign split based on actor ID."""
        # Simple split assignment (can be customized)
        if actor % 10 < 7:
            return 'train'
        elif actor % 10 < 9:
            return 'val'
        else:
            return 'test'
    
    def _create_data_index(self) -> List[Dict]:
        """Create index of available data samples."""
        data_index = []
        
        for _, row in self.metadata.iterrows():
            sample = {
                'filename': row['filename'],
                'modality': row['modality'],
                'vocal': row['vocal'],
                'emotion': row['emotion'],
                'intensity': row['intensity'],
                'statement': row['statement'],
                'repetition': row['repetition'],
                'actor': row['actor'],
                'audio_path': row['audio_path'],
                'video_path': row.get('video_path'),
                'split': row.get('split', self.split)
            }
            
            # Check if required modalities are available
            available_modalities = []
            if sample['audio_path'] and os.path.exists(sample['audio_path']):
                available_modalities.append('audio')
            if sample['video_path'] and os.path.exists(sample['video_path']):
                available_modalities.append('video')
            
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
    
    def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract audio features."""
        try:
            # Use pre-initialized processor or create one if needed
            if self.audio_processor is None:
                from src.data.preprocessing.audio_processor import AudioProcessor as AP
                self.audio_processor = AP(sample_rate=self.sample_rate)
            
            features = self.audio_processor.process_audio(audio, extract_features=True)
            
            # Get MFCC features - shape should be (n_mfcc_features, time_steps)
            mfcc = features.get('mfcc', np.zeros((39, 100)))
            
            # Ensure proper shape: (features, time) -> transpose to (time, features) for models
            if isinstance(mfcc, np.ndarray):
                if mfcc.ndim == 2:
                    # Transpose from (features, time) to (time, features)
                    mfcc = mfcc.T
                    # Pad or truncate to fixed sequence length
                    max_seq_len = 100  # Adjust based on your model's expected sequence length
                    if mfcc.shape[0] > max_seq_len:
                        mfcc = mfcc[:max_seq_len, :]
                    elif mfcc.shape[0] < max_seq_len:
                        padding = np.zeros((max_seq_len - mfcc.shape[0], mfcc.shape[1]))
                        mfcc = np.vstack([mfcc, padding])
            
            # Normalize MFCC features to help training
            if mfcc.ndim == 2 and mfcc.size > 0:
                # Normalize per feature dimension (z-score normalization)
                mfcc_mean = np.mean(mfcc, axis=0, keepdims=True)
                mfcc_std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
                mfcc = (mfcc - mfcc_mean) / mfcc_std
            
            return {
                'mfcc': mfcc,  # Now shape is (time, features) - (100, 39), normalized
                'spectral_features': features.get('spectral_features', {}),
                'prosodic_features': features.get('prosodic_features', {})
            }
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}", exc_info=True)
            # Return properly shaped zero features
            return {
                'mfcc': np.zeros((100, 39)),  # (time, features)
                'spectral_features': {},
                'prosodic_features': {}
            }
    
    def _extract_video_features(self, video: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract video features."""
        try:
            # Use pre-initialized processor or create one if needed
            if self.facial_processor is None:
                from src.data.preprocessing.facial_processor import FacialProcessor as FP
                self.facial_processor = FP()
            
            features = []
            
            # Process each frame (or just first frame for efficiency)
            # For now, process only first frame to speed up
            if len(video) > 0:
                frame_features = self.facial_processor.process_facial_data(video[0] if video.ndim > 2 else video, extract_features=True)
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
    
    def _get_label(self, emotion: int) -> int:
        """Get label for emotion (0-7 for 8 emotions)."""
        return emotion - 1  # Convert from 1-8 to 0-7
    
    def _get_emotion_name(self, emotion: int) -> str:
        """Get emotion name from emotion ID."""
        return self.emotion_labels.get(emotion, 'unknown')
    
    def _get_intensity_name(self, intensity: int) -> str:
        """Get intensity name from intensity ID."""
        return self.intensity_labels.get(intensity, 'unknown')
    
    def _get_statement_name(self, statement: int) -> str:
        """Get statement name from statement ID."""
        return self.statement_labels.get(statement, 'unknown')
    
    def _get_repetition_name(self, repetition: int) -> str:
        """Get repetition name from repetition ID."""
        return self.repetition_labels.get(repetition, 'unknown')
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        sample_info = self.data_index[idx]
        
        # Load data for available modalities
        data = {
            'filename': sample_info['filename'],
            'modality': sample_info['modality'],
            'vocal': sample_info['vocal'],
            'emotion': sample_info['emotion'],
            'intensity': sample_info['intensity'],
            'statement': sample_info['statement'],
            'repetition': sample_info['repetition'],
            'actor': sample_info['actor'],
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
        
        # Get label
        label = self._get_label(sample_info['emotion'])
        data['label'] = label
        
        # Add metadata
        data['emotion_name'] = self._get_emotion_name(sample_info['emotion'])
        data['intensity_name'] = self._get_intensity_name(sample_info['intensity'])
        data['statement_name'] = self._get_statement_name(sample_info['statement'])
        data['repetition_name'] = self._get_repetition_name(sample_info['repetition'])
        
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
            'max_video_length': self.max_video_length,
            'emotion_labels': self.emotion_labels,
            'intensity_labels': self.intensity_labels,
            'statement_labels': self.statement_labels,
            'repetition_labels': self.repetition_labels
        }
        
        # Count samples by modality
        modality_counts = {}
        for modality in self.modalities:
            count = sum(1 for sample in self.data_index 
                       if modality in sample['available_modalities'])
            modality_counts[modality] = count
        
        stats['modality_counts'] = modality_counts
        
        # Count samples by emotion
        emotion_counts = {}
        for sample in self.data_index:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        stats['emotion_counts'] = emotion_counts
        
        # Count samples by actor
        actor_counts = {}
        for sample in self.data_index:
            actor = sample['actor']
            actor_counts[actor] = actor_counts.get(actor, 0) + 1
        
        stats['actor_counts'] = actor_counts
        
        # Count samples by intensity
        intensity_counts = {}
        for sample in self.data_index:
            intensity = sample['intensity']
            intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1
        
        stats['intensity_counts'] = intensity_counts
        
        return stats
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """Get distribution of emotions in the dataset."""
        emotion_dist = {}
        
        for sample in self.data_index:
            emotion_name = self._get_emotion_name(sample['emotion'])
            emotion_dist[emotion_name] = emotion_dist.get(emotion_name, 0) + 1
        
        return emotion_dist
    
    def get_actor_distribution(self) -> Dict[int, int]:
        """Get distribution of actors in the dataset."""
        actor_dist = {}
        
        for sample in self.data_index:
            actor = sample['actor']
            actor_dist[actor] = actor_dist.get(actor, 0) + 1
        
        return actor_dist
    
    def create_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """Create DataLoader for the dataset."""
        if len(self) == 0:
            raise ValueError(
                f"Cannot create DataLoader: dataset is empty (0 samples).\n"
                f"Please ensure the dataset is properly loaded and contains data.\n"
                f"Dataset path: {self.data_dir}"
            )
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
        if not batch:
            return {}
        
        collated = {}
        
        # Get all unique keys from all samples in batch
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())
        
        # For multimodal models, ensure we check for both audio and video
        if 'audio' in self.modalities or 'video' in self.modalities:
            if 'audio' in self.modalities:
                all_keys.add('audio')
            if 'video' in self.modalities:
                all_keys.add('video')
        
        for key in all_keys:
            if key in ['filename', 'emotion_name', 'intensity_name', 'statement_name', 'repetition_name', 'split']:
                # Keep as list of strings
                collated[key] = [sample.get(key, '') for sample in batch]
            elif key in ['modality', 'vocal', 'emotion', 'intensity', 'statement', 'repetition', 'actor']:
                # Convert to tensor - use default value if missing
                collated[key] = torch.tensor([sample.get(key, 0) for sample in batch])
            elif key == 'label':
                # Convert to tensor - use default value if missing
                collated[key] = torch.tensor([sample.get(key, 0) for sample in batch])
            elif key in ['audio', 'video']:
                # Stack arrays - only include samples that have this key
                samples_with_key = [sample for sample in batch if key in sample and sample[key] is not None]
                if samples_with_key:
                    try:
                        collated[key] = torch.stack([torch.from_numpy(sample[key]) for sample in samples_with_key])
                    except Exception as e:
                        logger.warning(f"Error stacking {key}: {e}. Creating zero tensor.")
                        # Create zero tensor with expected shape
                        first_sample = samples_with_key[0][key]
                        if isinstance(first_sample, np.ndarray):
                            shape = first_sample.shape
                            collated[key] = torch.zeros((len(samples_with_key),) + shape)
                        else:
                            collated[key] = torch.zeros((len(samples_with_key),))
                else:
                    # No samples have this key, skip it
                    logger.debug(f"No samples have {key} key. Skipping.")
                    continue
            elif key.endswith('_features'):
                # Keep as list of dictionaries
                collated[key] = [sample.get(key, {}) for sample in batch]
            else:
                # Keep as list
                collated[key] = [sample.get(key, None) for sample in batch]
        
        return collated
