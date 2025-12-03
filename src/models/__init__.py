"""
Neural network models for multimodal cognitive decline detection.

This module provides model architectures for:
- Speech analysis (audio processing)
- Gait analysis (motion sensor data)
- Facial expression analysis (image processing)
- Multimodal fusion
"""

from .base_model import BaseModel
from .speech_model import SpeechModel, TransformerSpeechModel, CNNLSTMSpeechModel
from .gait_model import GaitModel, CNNLSTMGaitModel, TransformerGaitModel
from .facial_model import FacialModel, ResNetFacialModel, VisionTransformerFacialModel
from .multimodal_fusion import MultimodalFusionModel, AttentionFusion, CrossModalFusion

__all__ = [
    "BaseModel",
    "SpeechModel",
    "TransformerSpeechModel", 
    "CNNLSTMSpeechModel",
    "GaitModel",
    "CNNLSTMGaitModel",
    "TransformerGaitModel",
    "FacialModel",
    "ResNetFacialModel",
    "VisionTransformerFacialModel",
    "MultimodalFusionModel",
    "AttentionFusion",
    "CrossModalFusion",
]
