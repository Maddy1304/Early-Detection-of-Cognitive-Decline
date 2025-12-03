"""
Gait data preprocessing for motion analysis.

This module handles gait data processing including:
- Sensor data preprocessing (accelerometer, gyroscope, magnetometer)
- Gait cycle detection and segmentation
- Feature extraction (temporal, frequency, statistical)
- Motion artifact removal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis
import warnings

logger = logging.getLogger(__name__)


class GaitProcessor:
    """Gait data processor for motion analysis and cognitive decline detection."""
    
    def __init__(
        self,
        sampling_rate: int = 50,
        window_size: float = 5.0,
        overlap: float = 0.5,
        lowpass_cutoff: float = 10.0,
        highpass_cutoff: float = 0.1,
        gravity_threshold: float = 9.5,
        step_detection_threshold: float = 0.5
    ):
        """
        Initialize gait processor.
        
        Args:
            sampling_rate: Sampling rate of sensor data (Hz)
            window_size: Window size for analysis (seconds)
            overlap: Overlap between windows (0-1)
            lowpass_cutoff: Low-pass filter cutoff frequency (Hz)
            highpass_cutoff: High-pass filter cutoff frequency (Hz)
            gravity_threshold: Threshold for gravity detection (m/s²)
            step_detection_threshold: Threshold for step detection
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.lowpass_cutoff = lowpass_cutoff
        self.highpass_cutoff = highpass_cutoff
        self.gravity_threshold = gravity_threshold
        self.step_detection_threshold = step_detection_threshold
        
        # Calculate window parameters
        self.window_samples = int(window_size * sampling_rate)
        self.hop_samples = int(self.window_samples * (1 - overlap))
        
        # Design filters
        self._design_filters()
    
    def _design_filters(self):
        """Design Butterworth filters for signal preprocessing."""
        nyquist = self.sampling_rate / 2
        
        # Low-pass filter
        self.lowpass_b, self.lowpass_a = butter(
            4, self.lowpass_cutoff / nyquist, btype='low'
        )
        
        # High-pass filter
        self.highpass_b, self.highpass_a = butter(
            4, self.highpass_cutoff / nyquist, btype='high'
        )
    
    def load_sensor_data(self, file_path: str) -> pd.DataFrame:
        """
        Load sensor data from file.
        
        Args:
            file_path: Path to sensor data file
            
        Returns:
            DataFrame with sensor data
        """
        try:
            # Try different file formats
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                data = pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Loaded sensor data: {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading sensor data {file_path}: {e}")
            raise
    
    def preprocess_sensor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess sensor data (filtering, calibration, etc.).
        
        Args:
            data: Raw sensor data
            
        Returns:
            Preprocessed sensor data
        """
        processed_data = data.copy()
        
        # Standardize column names
        column_mapping = {
            'acc_x': 'acc_x', 'acc_y': 'acc_y', 'acc_z': 'acc_z',
            'gyro_x': 'gyro_x', 'gyro_y': 'gyro_y', 'gyro_z': 'gyro_z',
            'mag_x': 'mag_x', 'mag_y': 'mag_y', 'mag_z': 'mag_z'
        }
        
        # Rename columns if needed
        for old_name, new_name in column_mapping.items():
            if old_name in processed_data.columns:
                processed_data = processed_data.rename(columns={old_name: new_name})
        
        # Apply filters
        sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        for col in sensor_columns:
            if col in processed_data.columns:
                # Remove DC component
                processed_data[col] = filtfilt(
                    self.highpass_b, self.highpass_a, processed_data[col]
                )
                
                # Apply low-pass filter
                processed_data[col] = filtfilt(
                    self.lowpass_b, self.lowpass_a, processed_data[col]
                )
        
        return processed_data
    
    def remove_gravity(self, acc_data: np.ndarray) -> np.ndarray:
        """
        Remove gravity component from accelerometer data.
        
        Args:
            acc_data: Accelerometer data (N x 3)
            
        Returns:
            Gravity-removed accelerometer data
        """
        # Estimate gravity using low-pass filter
        gravity = np.zeros_like(acc_data)
        
        for i in range(3):  # x, y, z components
            # Use a very low cutoff frequency to estimate gravity
            b, a = butter(2, 0.1 / (self.sampling_rate / 2), btype='low')
            gravity[:, i] = filtfilt(b, a, acc_data[:, i])
        
        # Remove gravity
        acc_no_gravity = acc_data - gravity
        
        return acc_no_gravity
    
    def detect_steps(self, acc_data: np.ndarray, axis: int = 2) -> np.ndarray:
        """
        Detect steps from accelerometer data.
        
        Args:
            acc_data: Accelerometer data (N x 3)
            axis: Axis to use for step detection (0=x, 1=y, 2=z)
            
        Returns:
            Array of step timestamps
        """
        # Use vertical acceleration (z-axis) for step detection
        vertical_acc = acc_data[:, axis]
        
        # Calculate magnitude of acceleration
        acc_magnitude = np.sqrt(np.sum(acc_data ** 2, axis=1))
        
        # Find peaks in acceleration magnitude
        peaks, properties = find_peaks(
            acc_magnitude,
            height=self.step_detection_threshold,
            distance=int(0.3 * self.sampling_rate)  # Minimum 0.3s between steps
        )
        
        return peaks
    
    def segment_gait_cycles(self, acc_data: np.ndarray, steps: np.ndarray) -> List[np.ndarray]:
        """
        Segment data into individual gait cycles.
        
        Args:
            acc_data: Accelerometer data (N x 3)
            steps: Array of step timestamps
            
        Returns:
            List of gait cycle segments
        """
        gait_cycles = []
        
        if len(steps) < 2:
            return gait_cycles
        
        # Create gait cycles between consecutive steps
        for i in range(len(steps) - 1):
            start_idx = steps[i]
            end_idx = steps[i + 1]
            
            if end_idx - start_idx > int(0.5 * self.sampling_rate):  # Minimum 0.5s cycle
                cycle = acc_data[start_idx:end_idx]
                gait_cycles.append(cycle)
        
        return gait_cycles
    
    def extract_temporal_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features from sensor data.
        
        Args:
            data: Sensor data (N x 3)
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Calculate magnitude
        magnitude = np.sqrt(np.sum(data ** 2, axis=1))
        
        # Basic statistics
        features['mean'] = np.mean(magnitude)
        features['std'] = np.std(magnitude)
        features['min'] = np.min(magnitude)
        features['max'] = np.max(magnitude)
        features['range'] = features['max'] - features['min']
        features['skewness'] = skew(magnitude)
        features['kurtosis'] = kurtosis(magnitude)
        
        # Percentiles
        features['p25'] = np.percentile(magnitude, 25)
        features['p50'] = np.percentile(magnitude, 50)
        features['p75'] = np.percentile(magnitude, 75)
        features['p95'] = np.percentile(magnitude, 95)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(magnitude - np.mean(magnitude))) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(magnitude)
        
        # Root mean square
        features['rms'] = np.sqrt(np.mean(magnitude ** 2))
        
        return features
    
    def extract_frequency_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from sensor data.
        
        Args:
            data: Sensor data (N x 3)
            
        Returns:
            Dictionary of frequency features
        """
        features = {}
        
        # Calculate magnitude
        magnitude = np.sqrt(np.sum(data ** 2, axis=1))
        
        # Compute power spectral density
        freqs, psd = signal.welch(magnitude, fs=self.sampling_rate, nperseg=min(256, len(magnitude)//4))
        
        # Spectral features
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / np.sum(psd))
        features['spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
        
        # Power in different frequency bands
        low_freq_mask = (freqs >= 0.1) & (freqs <= 2.0)
        mid_freq_mask = (freqs > 2.0) & (freqs <= 5.0)
        high_freq_mask = (freqs > 5.0) & (freqs <= 10.0)
        
        features['low_freq_power'] = np.sum(psd[low_freq_mask])
        features['mid_freq_power'] = np.sum(psd[mid_freq_mask])
        features['high_freq_power'] = np.sum(psd[high_freq_mask])
        features['total_power'] = np.sum(psd)
        
        # Power ratios
        features['low_freq_ratio'] = features['low_freq_power'] / features['total_power']
        features['mid_freq_ratio'] = features['mid_freq_power'] / features['total_power']
        features['high_freq_ratio'] = features['high_freq_power'] / features['total_power']
        
        return features
    
    def extract_gait_specific_features(self, acc_data: np.ndarray, steps: np.ndarray) -> Dict[str, float]:
        """
        Extract gait-specific features.
        
        Args:
            acc_data: Accelerometer data (N x 3)
            steps: Array of step timestamps
            
        Returns:
            Dictionary of gait-specific features
        """
        features = {}
        
        if len(steps) < 2:
            # Return default values if insufficient steps
            features.update({
                'step_frequency': 0.0,
                'stride_length': 0.0,
                'walking_speed': 0.0,
                'step_regularity': 0.0,
                'stride_regularity': 0.0,
                'step_time_variability': 0.0,
                'stride_time_variability': 0.0
            })
            return features
        
        # Step frequency (steps per second)
        step_intervals = np.diff(steps) / self.sampling_rate
        features['step_frequency'] = 1.0 / np.mean(step_intervals)
        
        # Step time variability
        features['step_time_variability'] = np.std(step_intervals) / np.mean(step_intervals)
        
        # Stride features (every other step)
        if len(steps) >= 3:
            stride_intervals = step_intervals[::2]  # Every other step
            features['stride_time_variability'] = np.std(stride_intervals) / np.mean(stride_intervals)
        else:
            features['stride_time_variability'] = 0.0
        
        # Step regularity (autocorrelation of step intervals)
        if len(step_intervals) > 1:
            autocorr = np.correlate(step_intervals, step_intervals, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            if len(autocorr) > 1:
                features['step_regularity'] = autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0.0
            else:
                features['step_regularity'] = 0.0
        else:
            features['step_regularity'] = 0.0
        
        # Stride regularity
        if len(steps) >= 3:
            stride_intervals = step_intervals[::2]
            if len(stride_intervals) > 1:
                autocorr = np.correlate(stride_intervals, stride_intervals, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                if len(autocorr) > 1:
                    features['stride_regularity'] = autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0.0
                else:
                    features['stride_regularity'] = 0.0
            else:
                features['stride_regularity'] = 0.0
        else:
            features['stride_regularity'] = 0.0
        
        # Estimate stride length and walking speed (simplified)
        # These are rough estimates and would need calibration for real-world use
        avg_step_time = np.mean(step_intervals)
        features['stride_length'] = 0.7  # Average stride length in meters (rough estimate)
        features['walking_speed'] = features['stride_length'] / (2 * avg_step_time)  # m/s
        
        return features
    
    def extract_balance_features(self, acc_data: np.ndarray) -> Dict[str, float]:
        """
        Extract balance-related features from accelerometer data.
        
        Args:
            acc_data: Accelerometer data (N x 3)
            
        Returns:
            Dictionary of balance features
        """
        features = {}
        
        # Calculate center of mass displacement (simplified)
        # Using accelerometer data as proxy for center of mass
        com_x = np.cumsum(acc_data[:, 0]) / self.sampling_rate
        com_y = np.cumsum(acc_data[:, 1]) / self.sampling_rate
        
        # Balance metrics
        features['com_displacement_x'] = np.std(com_x)
        features['com_displacement_y'] = np.std(com_y)
        features['com_displacement_total'] = np.sqrt(features['com_displacement_x']**2 + features['com_displacement_y']**2)
        
        # Sway area (approximate)
        features['sway_area'] = np.pi * features['com_displacement_x'] * features['com_displacement_y']
        
        # Velocity of center of mass
        com_velocity_x = np.diff(com_x) * self.sampling_rate
        com_velocity_y = np.diff(com_y) * self.sampling_rate
        
        features['com_velocity_x'] = np.std(com_velocity_x)
        features['com_velocity_y'] = np.std(com_velocity_y)
        features['com_velocity_total'] = np.sqrt(features['com_velocity_x']**2 + features['com_velocity_y']**2)
        
        return features
    
    def window_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Window sensor data for analysis.
        
        Args:
            data: Sensor data DataFrame
            
        Returns:
            List of windowed data frames
        """
        windows = []
        
        for start_idx in range(0, len(data) - self.window_samples + 1, self.hop_samples):
            end_idx = start_idx + self.window_samples
            window = data.iloc[start_idx:end_idx].copy()
            windows.append(window)
        
        return windows
    
    def process_gait_data(self, data: pd.DataFrame, extract_features: bool = True) -> Dict:
        """
        Complete gait data processing pipeline.
        
        Args:
            data: Raw sensor data
            extract_features: Whether to extract features
            
        Returns:
            Dictionary containing processed data and features
        """
        result = {'raw_data': data}
        
        # Preprocess data
        processed_data = self.preprocess_sensor_data(data)
        result['processed_data'] = processed_data
        
        # Convert to numpy array for processing
        sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        available_columns = [col for col in sensor_columns if col in processed_data.columns]
        
        if not available_columns:
            logger.warning("No sensor columns found in data")
            return result
        
        sensor_data = processed_data[available_columns].values
        
        # Remove gravity from accelerometer data
        if 'acc_x' in available_columns and 'acc_y' in available_columns and 'acc_z' in available_columns:
            acc_indices = [available_columns.index(col) for col in ['acc_x', 'acc_y', 'acc_z']]
            acc_data = sensor_data[:, acc_indices]
            acc_no_gravity = self.remove_gravity(acc_data)
            result['acc_no_gravity'] = acc_no_gravity
            
            # Detect steps
            steps = self.detect_steps(acc_no_gravity)
            result['steps'] = steps
            
            # Segment gait cycles
            gait_cycles = self.segment_gait_cycles(acc_no_gravity, steps)
            result['gait_cycles'] = gait_cycles
        
        if extract_features:
            # Extract features
            result['temporal_features'] = self.extract_temporal_features(sensor_data)
            result['frequency_features'] = self.extract_frequency_features(sensor_data)
            
            if 'steps' in result:
                result['gait_features'] = self.extract_gait_specific_features(
                    acc_no_gravity, result['steps']
                )
                result['balance_features'] = self.extract_balance_features(acc_no_gravity)
        
        return result
    
    def augment_gait_data(self, data: np.ndarray, techniques: List[str] = None) -> np.ndarray:
        """
        Apply data augmentation techniques to gait data.
        
        Args:
            data: Input sensor data
            techniques: List of augmentation techniques to apply
            
        Returns:
            Augmented sensor data
        """
        if techniques is None:
            techniques = ['time_warping', 'magnitude_warping', 'jittering', 'scaling']
        
        augmented_data = data.copy()
        
        for technique in techniques:
            if technique == 'time_warping':
                # Random time warping
                warp_factor = np.random.uniform(0.8, 1.2)
                new_length = int(len(augmented_data) * warp_factor)
                if new_length > 0:
                    indices = np.linspace(0, len(augmented_data) - 1, new_length)
                    augmented_data = np.interp(indices, np.arange(len(augmented_data)), augmented_data)
                
            elif technique == 'magnitude_warping':
                # Random magnitude warping
                warp_factor = np.random.uniform(0.9, 1.1)
                augmented_data = augmented_data * warp_factor
                
            elif technique == 'jittering':
                # Add random noise
                noise_factor = np.random.uniform(0.01, 0.05)
                noise = np.random.normal(0, noise_factor, augmented_data.shape)
                augmented_data = augmented_data + noise
                
            elif technique == 'scaling':
                # Random scaling
                scale_factor = np.random.uniform(0.95, 1.05)
                augmented_data = augmented_data * scale_factor
        
        return augmented_data
