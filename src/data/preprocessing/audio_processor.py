"""
Audio data preprocessing for speech analysis.

This module handles audio data processing including:
- Noise reduction and filtering
- Feature extraction (MFCC, spectral features, etc.)
- Voice activity detection
- Audio augmentation techniques
"""

import librosa
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import signal
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio data processor for speech analysis and cognitive decline detection."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        preemphasis: float = 0.97,
        noise_reduction: bool = True,
        vad_threshold: float = 0.01
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of mel filter banks
            fmin: Minimum frequency for mel filters
            fmax: Maximum frequency for mel filters
            preemphasis: Preemphasis coefficient
            noise_reduction: Whether to apply noise reduction
            vad_threshold: Voice activity detection threshold
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.preemphasis = preemphasis
        self.enable_noise_reduction = noise_reduction  # Renamed to avoid conflict with method
        self.vad_threshold = vad_threshold
        
        # Initialize mel filter bank
        self.mel_filter_bank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax
        )
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio signal as numpy array
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {file_path}, shape: {audio.shape}, sr: {sr}")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise
    
    def preemphasis_filter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preemphasis filter to audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Preemphasized audio signal
        """
        return np.append(audio[0], audio[1:] - self.preemphasis * audio[:-1])
    
    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Noise-reduced audio signal
        """
            
        # Estimate noise from first 0.5 seconds
        noise_samples = int(0.5 * self.sample_rate)
        noise_spectrum = np.abs(librosa.stft(audio[:noise_samples]))
        noise_power = np.mean(noise_spectrum ** 2, axis=1, keepdims=True)
        
        # Apply spectral subtraction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor factor
        enhanced_magnitude = magnitude - alpha * np.sqrt(noise_power)
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
        
        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def voice_activity_detection(self, audio: np.ndarray) -> np.ndarray:
        """
        Detect voice activity in audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Boolean array indicating voice activity
        """
        # Calculate energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.010 * self.sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Apply threshold
        vad = energy > self.vad_threshold
        
        # Smooth VAD decisions
        vad = signal.medfilt(vad.astype(float), kernel_size=5).astype(bool)
        
        return vad
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Add delta and delta-delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate features
        mfcc_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        
        return mfcc_features
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Zero crossing rate
        features['zcr'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Mel spectrogram
        features['mel_spectrogram'] = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        return features
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features from audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary of prosodic features
        """
        features = {}
        
        # Fundamental frequency (F0) - use faster method
        try:
            # Use faster pitch estimation (yin is much faster than pyin)
            f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=self.sample_rate)
            voiced_flag = f0 > 0
            voiced_probs = (f0 > 0).astype(float)
        except Exception as e:
            # Fallback: skip F0 extraction for speed
            logger.warning(f"Error in F0 extraction, using fallback: {e}")
            f0 = np.zeros(max(100, len(audio) // self.hop_length))
            voiced_flag = np.zeros(len(f0), dtype=bool)
            voiced_probs = np.zeros(len(f0))
        
        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) > 0:
            features['f0_mean'] = np.mean(f0_voiced)
            features['f0_std'] = np.std(f0_voiced)
            features['f0_min'] = np.min(f0_voiced)
            features['f0_max'] = np.max(f0_voiced)
            features['f0_range'] = features['f0_max'] - features['f0_min']
        else:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
            features['f0_min'] = 0.0
            features['f0_max'] = 0.0
            features['f0_range'] = 0.0
        
        # Speaking rate (syllables per second)
        # This is a simplified estimation
        energy = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        energy_threshold = np.mean(energy) + np.std(energy)
        speech_frames = np.sum(energy > energy_threshold)
        speaking_rate = speech_frames * self.hop_length / self.sample_rate / len(audio)
        features['speaking_rate'] = speaking_rate
        
        # Pause ratio
        silence_threshold = np.percentile(energy, 20)
        silence_frames = np.sum(energy < silence_threshold)
        features['pause_ratio'] = silence_frames / len(energy)
        
        return features
    
    def augment_audio(self, audio: np.ndarray, techniques: List[str] = None) -> np.ndarray:
        """
        Apply audio augmentation techniques.
        
        Args:
            audio: Input audio signal
            techniques: List of augmentation techniques to apply
            
        Returns:
            Augmented audio signal
        """
        if techniques is None:
            techniques = ['time_shift', 'pitch_shift', 'noise_injection', 'speed_change']
        
        augmented_audio = audio.copy()
        
        for technique in techniques:
            if technique == 'time_shift':
                # Random time shift
                shift = np.random.randint(-int(0.1 * self.sample_rate), 
                                        int(0.1 * self.sample_rate))
                augmented_audio = np.roll(augmented_audio, shift)
                
            elif technique == 'pitch_shift':
                # Random pitch shift
                n_steps = np.random.uniform(-2, 2)
                augmented_audio = librosa.effects.pitch_shift(
                    augmented_audio, sr=self.sample_rate, n_steps=n_steps
                )
                
            elif technique == 'noise_injection':
                # Add random noise
                noise_factor = np.random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_factor, len(augmented_audio))
                augmented_audio = augmented_audio + noise
                
            elif technique == 'speed_change':
                # Random speed change
                speed_factor = np.random.uniform(0.9, 1.1)
                augmented_audio = librosa.effects.time_stretch(augmented_audio, rate=speed_factor)
        
        return augmented_audio
    
    def process_audio(self, audio: np.ndarray, extract_features: bool = True) -> Dict:
        """
        Complete audio processing pipeline.
        
        Args:
            audio: Input audio signal
            extract_features: Whether to extract features
            
        Returns:
            Dictionary containing processed audio and features
        """
        result = {'raw_audio': audio}
        
        # Preprocessing
        audio = self.preemphasis_filter(audio)
        # Apply noise reduction if enabled
        if self.enable_noise_reduction:
            audio = self._apply_noise_reduction(audio)
        
        result['processed_audio'] = audio
        
        # Voice activity detection
        vad = self.voice_activity_detection(audio)
        result['vad'] = vad
        
        if extract_features:
            # Extract features
            result['mfcc'] = self.extract_mfcc(audio)
            result['spectral_features'] = self.extract_spectral_features(audio)
            result['prosodic_features'] = self.extract_prosodic_features(audio)
        
        return result
    
    def normalize_features(self, features: np.ndarray, method: str = 'z_score') -> np.ndarray:
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
    
    def create_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Create mel spectrogram from audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
