"""
Main entry point for the cognitive decline detection system.

This module provides the main interface for running simulations,
training models, and evaluating the federated learning system.
"""

import argparse
import logging
import sys
import os
import time
from typing import Dict, Any, Optional, List
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.infrastructure.simulation_environment import SimulationEnvironment
from src.evaluation.metrics import EvaluationMetrics
from src.utils.logger import setup_logger
from src.data.datasets.ravdess import RAVDESSDataset
from src.models.speech_model import SpeechModel, TransformerSpeechModel, CNNLSTMSpeechModel
from src.models.facial_model import FacialModel, ResNetFacialModel
from src.models.multimodal_fusion import MultimodalFusionModel

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get available device (CUDA or CPU)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def create_model_from_config(model_type: str, model_config: Dict[str, Any], device: str) -> nn.Module:
    """
    Create model instance from configuration.
    
    Args:
        model_type: Type of model ('speech', 'facial', 'multimodal')
        model_config: Model configuration dictionary
        device: Device to run on
        
    Returns:
        Model instance
    """
    if model_type == 'speech':
        arch = model_config.get('architecture', 'transformer')
        # MFCC has 39 features (13 base + 13 delta + 13 delta-delta)
        mfcc_dim = 39
        if arch == 'transformer':
            return TransformerSpeechModel(
                input_dim=mfcc_dim,  # MFCC features = 39
                output_dim=8,  # RAVDESS has 8 emotions
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 6),
                num_heads=model_config.get('num_heads', 8),
                dropout=model_config.get('dropout', 0.1),
                activation=model_config.get('activation', 'gelu'),
                device=device,
                sequence_length=100  # Fixed sequence length
            )
        elif arch == 'cnn_lstm':
            return CNNLSTMSpeechModel(
                input_dim=mfcc_dim,  # MFCC features = 39
                output_dim=8,
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.1),
                device=device,
                sequence_length=100
            )
        else:
            return SpeechModel(
                input_dim=mfcc_dim,  # MFCC features = 39
                output_dim=8,
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 3),
                dropout=model_config.get('dropout', 0.1),
                device=device,
                sequence_length=100
            )
    
    elif model_type == 'facial':
        arch = model_config.get('architecture', 'resnet')
        if arch == 'resnet':
            return ResNetFacialModel(
                input_dim=224 * 224 * 3,
                output_dim=model_config.get('num_classes', 8),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 3),
                dropout=model_config.get('dropout', 0.3),
                device=device,
                image_size=(224, 224),
                num_channels=3,
                pretrained=model_config.get('pretrained', True),
                freeze_layers=model_config.get('freeze_layers', 10)
            )
        else:
            return FacialModel(
                input_dim=224 * 224 * 3,
                output_dim=model_config.get('num_classes', 8),
                hidden_dim=model_config.get('hidden_dim', 256),
                num_layers=model_config.get('num_layers', 3),
                dropout=model_config.get('dropout', 0.3),
                device=device,
                image_size=(224, 224),
                num_channels=3
            )
    
    elif model_type == 'multimodal':
        # Get individual model configs
        speech_config = model_config.get('speech_model', {})
        facial_config = model_config.get('facial_model', {})
        fusion_config = model_config.get('multimodal_fusion', {})
        
        # Create speech and facial models
        speech_model = create_model_from_config('speech', speech_config, device)
        facial_model = create_model_from_config('facial', facial_config, device)
        
        # Create fusion model
        return MultimodalFusionModel(
            input_dims={'speech': 256, 'facial': 256},  # Output dims from individual models
            output_dim=8,
            hidden_dim=fusion_config.get('hidden_dim', 512),
            num_layers=3,
            dropout=fusion_config.get('dropout', 0.1),
            device=device,
            fusion_method=fusion_config.get('fusion_method', 'late')
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_ravdess_data_path(dataset: str) -> str:
    """Get RAVDESS dataset path."""
    base_dir = os.path.join(project_root, 'data', dataset)
    return base_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Early Detection of Cognitive Decline Using Multi-Modal Federated Learning"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/simulation_config.yaml",
        help="Path to simulation configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "training", "evaluation", "demo"],
        default="simulation",
        help="Mode to run the system in"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="Simulation duration in seconds"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["daic-woz", "mpower", "ravdess", "all"],
        default="all",
        help="Dataset to use for training/evaluation"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["speech", "gait", "facial", "multimodal"],
        default="multimodal",
        help="Model type to train/evaluate"
    )
    
    parser.add_argument(
        "--privacy",
        action="store_true",
        help="Enable privacy-preserving techniques"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    return parser.parse_args()


def run_simulation(config_path: str, duration: float, output_dir: str) -> Dict[str, Any]:
    """
    Run the simulation environment.
    
    Args:
        config_path: Path to simulation configuration
        duration: Simulation duration in seconds
        output_dir: Output directory for results
        
    Returns:
        Simulation results
    """
    logger.info("Starting simulation...")
    
    try:
        # Initialize simulation environment
        sim_env = SimulationEnvironment(config_path)
        
        # Start simulation
        sim_env.start_simulation(duration)
        
        # Wait for simulation to complete
        while sim_env.is_running:
            time.sleep(1.0)
        
        # Get simulation results
        results = sim_env.get_simulation_status()
        
        # Save simulation data
        os.makedirs(output_dir, exist_ok=True)
        sim_env.save_simulation_data(os.path.join(output_dir, "simulation_data.json"))
        
        # Generate report
        report = sim_env.generate_report()
        with open(os.path.join(output_dir, "simulation_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Shutdown simulation
        sim_env.shutdown()
        
        logger.info("Simulation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


def run_training(config_path: str, dataset: str, model_type: str, output_dir: str) -> Dict[str, Any]:
    """
    Run real model training on RAVDESS dataset.
    
    Args:
        config_path: Path to configuration file
        dataset: Dataset to use for training (should be 'ravdess')
        model_type: Type of model to train ('speech', 'facial', 'multimodal')
        output_dir: Output directory for results
        
    Returns:
        Training results
    """
    logger.info(f"Starting real training with {dataset} dataset and {model_type} model...")
    start_time = time.time()
    
    try:
        # Load configuration
        model_config_path = os.path.join(project_root, 'config', 'model_config.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Determine modalities needed
        if model_type == 'speech':
            modalities = ['audio']
        elif model_type == 'facial':
            modalities = ['video']
        elif model_type == 'multimodal':
            modalities = ['audio', 'video']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get data path
        data_dir = get_ravdess_data_path(dataset)
        logger.info(f"Loading dataset from: {data_dir}")
        
        # Load datasets
        logger.info("Loading training dataset...")
        train_dataset = RAVDESSDataset(
            data_dir=data_dir,
            split='train',
            modalities=modalities,
            preprocess=True
        )
        
        logger.info("Loading validation dataset...")
        val_dataset = RAVDESSDataset(
            data_dir=data_dir,
            split='val',
            modalities=modalities,
            preprocess=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Validate dataset has samples
        if len(train_dataset) == 0:
            error_msg = (
                f"No training samples found! Dataset is empty.\n"
                f"Please ensure the RAVDESS dataset is properly downloaded and extracted.\n"
                f"Expected data directory: {data_dir}\n"
                f"Required subdirectories: audio/, video/, metadata/\n"
                f"Run: python scripts/download_datasets.py --dataset ravdess"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(val_dataset) == 0:
            logger.warning("No validation samples found. Using training set for validation.")
            val_dataset = train_dataset
        
        # Get training config
        train_config = model_config.get('training', {})
        batch_size = train_config.get('batch_size', 32) if 'batch_size' in train_config else 32
        
        # Create data loaders
        train_loader = train_dataset.create_dataloader(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=False
        )
        
        val_loader = val_dataset.create_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Create model
        logger.info(f"Creating {model_type} model...")
        if model_type == 'multimodal':
            model = create_model_from_config(model_type, model_config.get('models', {}), device)
        else:
            model_config_dict = model_config.get('models', {}).get(f'{model_type}_model', {})
            model = create_model_from_config(model_type, model_config_dict, device)
        
        model = model.to(device)
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Setup optimizer and loss
        optimizer_config = train_config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adamw').lower()
        learning_rate = optimizer_config.get('learning_rate', 0.0001)  # Default reduced
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        loss_fn = nn.CrossEntropyLoss()
        
        # Training config
        num_epochs = model_config.get('training', {}).get('epochs', 10) if 'epochs' in model_config.get('training', {}) else 10
        
        # Setup learning rate scheduler
        scheduler_config = train_config.get('scheduler', {})
        scheduler_type = scheduler_config.get('name', 'cosine').lower()
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        elif scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = None
        
        # Training history
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        
        # Training loop
        logger.info(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch in train_pbar:
                # Extract data based on model type
                if model_type == 'speech':
                    if 'audio_features' not in batch:
                        continue
                    # Use MFCC features
                    audio_features = batch['audio_features']
                    if isinstance(audio_features, list):
                        # Extract MFCC from features dict
                        mfcc_features = []
                        for feat_dict in audio_features:
                            if isinstance(feat_dict, dict) and 'mfcc' in feat_dict:
                                mfcc = feat_dict['mfcc']
                                if isinstance(mfcc, np.ndarray):
                                    mfcc = torch.from_numpy(mfcc).float()
                                elif not isinstance(mfcc, torch.Tensor):
                                    continue
                                # MFCC shape should be (time, features) = (100, 39)
                                if mfcc.dim() == 2:
                                    mfcc_features.append(mfcc)
                        if not mfcc_features:
                            continue
                        # Stack to get (batch, time, features)
                        x = torch.stack(mfcc_features)  # (batch, time, features)
                    else:
                        # Direct tensor
                        x = audio_features
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x).float()
                        if x.dim() == 2:
                            x = x.unsqueeze(0)  # Add batch dimension
                    
                    if not isinstance(batch['label'], torch.Tensor):
                        labels = torch.tensor(batch['label'], dtype=torch.long)
                    else:
                        labels = batch['label']
                    
                elif model_type == 'facial':
                    if 'video' not in batch:
                        continue
                    # Use video frames - take first frame or average
                    video = batch['video']
                    if isinstance(video, torch.Tensor):
                        if video.dim() == 5:  # (batch, frames, H, W, C)
                            x = video[:, 0, :, :, :]  # Take first frame
                            x = x.permute(0, 3, 1, 2)  # (batch, C, H, W)
                        else:
                            x = video
                    else:
                        continue
                    labels = batch['label']
                    
                elif model_type == 'multimodal':
                    # Need both audio and video
                    if 'audio_features' not in batch or 'video' not in batch:
                        continue
                    # This will need custom handling in multimodal model
                    # For now, use audio only as placeholder
                    audio_features = batch['audio_features']
                    if isinstance(audio_features, list):
                        mfcc_features = []
                        for feat_dict in audio_features:
                            if 'mfcc' in feat_dict:
                                mfcc = feat_dict['mfcc']
                                if isinstance(mfcc, np.ndarray):
                                    mfcc = torch.from_numpy(mfcc).float()
                                mfcc_features.append(mfcc)
                        if mfcc_features:
                            x = torch.stack(mfcc_features)
                            if x.dim() == 3:
                                x = x.transpose(1, 2)
                        else:
                            continue
                    labels = batch['label']
                
                # Move to device
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                else:
                    continue
                    
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                else:
                    labels = torch.tensor(labels).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                try:
                    outputs = model(x)
                    loss = loss_fn(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total if train_total > 0 else 0})
                except Exception as e:
                    logger.warning(f"Error in forward pass: {e}")
                    continue
            
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            train_acc = train_correct / train_total if train_total > 0 else 0.0
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_accuracy'].append(train_acc)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch in val_pbar:
                    # Extract data (same as training)
                    if model_type == 'speech':
                        if 'audio_features' not in batch:
                            continue
                        audio_features = batch['audio_features']
                        if isinstance(audio_features, list):
                            # Extract MFCC from features dict
                            mfcc_features = []
                            for feat_dict in audio_features:
                                if isinstance(feat_dict, dict) and 'mfcc' in feat_dict:
                                    mfcc = feat_dict['mfcc']
                                    if isinstance(mfcc, np.ndarray):
                                        mfcc = torch.from_numpy(mfcc).float()
                                    elif not isinstance(mfcc, torch.Tensor):
                                        continue
                                    # MFCC shape should be (time, features) = (100, 39)
                                    if mfcc.dim() == 2:
                                        mfcc_features.append(mfcc)
                            if not mfcc_features:
                                continue
                            # Stack to get (batch, time, features)
                            x = torch.stack(mfcc_features)  # (batch, time, features)
                        else:
                            # Direct tensor
                            x = audio_features
                            if isinstance(x, np.ndarray):
                                x = torch.from_numpy(x).float()
                            if x.dim() == 2:
                                x = x.unsqueeze(0)  # Add batch dimension
                        labels = batch['label']
                        
                    elif model_type == 'facial':
                        if 'video' not in batch:
                            continue
                        video = batch['video']
                        if isinstance(video, torch.Tensor):
                            if video.dim() == 5:
                                x = video[:, 0, :, :, :]
                                x = x.permute(0, 3, 1, 2)
                            else:
                                x = video
                        else:
                            continue
                        labels = batch['label']
                    
                    # Move to device
                    if isinstance(x, torch.Tensor):
                        x = x.to(device)
                    else:
                        continue
                        
                    if isinstance(labels, torch.Tensor):
                        labels = labels.to(device)
                    else:
                        labels = torch.tensor(labels).to(device)
                    
                    try:
                        outputs = model(x)
                        loss = loss_fn(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total if val_total > 0 else 0})
                    except Exception as e:
                        logger.warning(f"Error in validation forward pass: {e}")
                        continue
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_accuracy'].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                logger.debug(f"Learning rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(output_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
        
        training_time = time.time() - start_time
        
        # Prepare results
        training_results = {
            'dataset': dataset,
            'model_type': model_type,
            'num_epochs': num_epochs,
            'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else 0.0,
            'final_train_accuracy': training_history['train_accuracy'][-1] if training_history['train_accuracy'] else 0.0,
            'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else 0.0,
            'final_val_accuracy': training_history['val_accuracy'][-1] if training_history['val_accuracy'] else 0.0,
            'best_val_accuracy': best_val_acc,
            'training_time': training_time,
            'training_history': training_history,
            'model_path': best_model_path,
            'device': device
        }
        
        # Save training results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_results.json"), 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        return training_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


def run_evaluation(config_path: str, dataset: str, model_type: str, output_dir: str) -> Dict[str, Any]:
    """
    Run real model evaluation on RAVDESS test dataset.
    
    Args:
        config_path: Path to configuration file
        dataset: Dataset to use for evaluation (should be 'ravdess')
        model_type: Type of model to evaluate ('speech', 'facial', 'multimodal')
        output_dir: Output directory for results (should contain trained model)
        
    Returns:
        Evaluation results
    """
    logger.info(f"Starting real model evaluation with {dataset} dataset and {model_type} model...")
    start_time = time.time()
    
    # Get project root for paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Load configuration
        model_config_path = os.path.join(project_root, 'config', 'model_config.yaml')
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Determine modalities needed
        if model_type == 'speech':
            modalities = ['audio']
        elif model_type == 'facial':
            modalities = ['video']
        elif model_type == 'multimodal':
            modalities = ['audio', 'video']
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Get data path
        data_dir = get_ravdess_data_path(dataset)
        logger.info(f"Loading test dataset from: {data_dir}")
        
        # Load test dataset
        test_dataset = RAVDESSDataset(
            data_dir=data_dir,
            split='test',
            modalities=modalities,
            preprocess=True
        )
        
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Get training config for batch size
        train_config = model_config.get('training', {})
        batch_size = train_config.get('batch_size', 32) if 'batch_size' in train_config else 32
        
        # Create test data loader
        test_loader = test_dataset.create_dataloader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=False
        )
        
        # Create model
        logger.info(f"Creating {model_type} model...")
        if model_type == 'multimodal':
            model = create_model_from_config(model_type, model_config.get('models', {}), device)
        else:
            model_config_dict = model_config.get('models', {}).get(f'{model_type}_model', {})
            model = create_model_from_config(model_type, model_config_dict, device)
        
        model = model.to(device)
        
        # Load trained model weights
        # Model should be in the training output directory, not evaluation directory
        # Try multiple possible locations with both relative and absolute paths
        possible_paths = []
        
        # Normalize paths to handle both forward and backward slashes
        def normalize_path(p):
            """Normalize path to use os.sep and resolve to absolute path."""
            if not p:
                return None
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(os.path.normpath(p))
            return abs_path if os.path.exists(abs_path) else None
        
        # Build list of possible paths
        # 1. Model in training directory matching model type (relative to output_dir parent)
        parent_dir = os.path.dirname(output_dir) if os.path.dirname(output_dir) else 'results'
        possible_paths.append(os.path.join(parent_dir, model_type, 'best_model.pth'))
        possible_paths.append(os.path.abspath(os.path.join(parent_dir, model_type, 'best_model.pth')))
        
        # 2. Model in demo training directory (relative and absolute)
        demo_path = os.path.join('results', 'demo', model_type, 'best_model.pth')
        possible_paths.append(demo_path)
        possible_paths.append(os.path.abspath(demo_path))
        possible_paths.append(os.path.join(project_root, 'results', 'demo', model_type, 'best_model.pth'))
        
        # 3. Model in output directory (if it's actually a training directory)
        possible_paths.append(os.path.join(output_dir, 'best_model.pth'))
        possible_paths.append(os.path.abspath(os.path.join(output_dir, 'best_model.pth')))
        
        # 4. Model in parent directory
        possible_paths.append(os.path.join(os.path.dirname(output_dir), 'best_model.pth'))
        possible_paths.append(os.path.abspath(os.path.join(os.path.dirname(output_dir), 'best_model.pth')))
        
        # 5. Check results directory structure
        results_base = os.path.join(project_root, 'results')
        possible_paths.append(os.path.join(results_base, 'demo', model_type, 'best_model.pth'))
        possible_paths.append(os.path.join(results_base, f'{model_type}_ravdess', 'best_model.pth'))
        possible_paths.append(os.path.join(results_base, f'train_{model_type}', 'best_model.pth'))
        
        # 6. Search recursively in results directory for best_model.pth matching model_type
        results_dir = os.path.join(project_root, 'results')
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                if 'best_model.pth' in files:
                    # Only include if the directory name matches the model type
                    dir_name = os.path.basename(root)
                    if model_type in dir_name.lower() or dir_name == model_type:
                        possible_paths.append(os.path.join(root, 'best_model.pth'))
        
        # Normalize all paths and remove duplicates/None values
        normalized_paths = []
        seen = set()
        for path in possible_paths:
            if path:
                norm_path = os.path.abspath(os.path.normpath(path))
                if norm_path not in seen:
                    normalized_paths.append(norm_path)
                    seen.add(norm_path)
        
        possible_paths = normalized_paths
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                model_path = path
                logger.info(f"Found model at: {model_path}")
                break
        
        if model_path is None:
            # Provide helpful error message with suggestions
            training_dir = os.path.join('results', 'demo', model_type)
            expected_path = os.path.join(project_root, 'results', 'demo', model_type, 'best_model.pth')
            
            # Show first 10 checked paths (normalized)
            checked_paths = '\n'.join([f"  - {p}" for p in possible_paths[:10]])
            
            # Check if any model files exist in results directory
            found_models = []
            results_dir = os.path.join(project_root, 'results')
            if os.path.exists(results_dir):
                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        if file.endswith('.pth') or file.endswith('.pt'):
                            found_models.append(os.path.join(root, file))
            
            error_msg = (
                f"Trained model not found. Please train the model first.\n"
                f"Expected location: {expected_path}\n"
                f"Checked {len(possible_paths)} locations (showing first 10):\n{checked_paths}\n"
            )
            
            if found_models:
                error_msg += f"\nFound {len(found_models)} model file(s) in results directory, but none match '{model_type}':\n"
                error_msg += '\n'.join([f"  - {m}" for m in found_models[:5]])
                error_msg += "\n"
            
            error_msg += (
                f"\nTo train the model, run:\n"
                f"  python src/main.py --mode training --dataset ravdess --model {model_type} --output {training_dir}"
            )
            
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading trained model from: {model_path}")
        
        # Load checkpoint and validate it matches the expected model type
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Check if this is a state_dict or a full checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Validate model architecture matches
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            # Check for key indicators of model type
            has_multimodal_keys = any('modality_projections' in k or 'fusion_layers' in k for k in checkpoint_keys)
            has_speech_keys = any('transformer_encoder' in k or 'input_projection' in k for k in checkpoint_keys)
            has_facial_keys = any('resnet' in k.lower() or 'facial' in k.lower() for k in checkpoint_keys)
            
            # Validate based on expected model type
            if model_type == 'multimodal':
                if not has_multimodal_keys:
                    error_msg = (
                        f"Model architecture mismatch!\n"
                        f"Expected: Multimodal model (with modality_projections, fusion_layers)\n"
                        f"Found: Different model architecture in {model_path}\n"
                        f"Checkpoint keys suggest: "
                    )
                    if has_speech_keys:
                        error_msg += "Speech model"
                    elif has_facial_keys:
                        error_msg += "Facial model"
                    else:
                        error_msg += "Unknown model type"
                    error_msg += (
                        f"\n\nPlease train a multimodal model first:\n"
                        f"  python src/main.py --mode training --dataset ravdess --model multimodal --output results/demo/multimodal"
                    )
                    raise ValueError(error_msg)
            elif model_type == 'speech':
                if not has_speech_keys:
                    error_msg = (
                        f"Model architecture mismatch!\n"
                        f"Expected: Speech model (with transformer_encoder)\n"
                        f"Found: Different model architecture in {model_path}\n"
                        f"Please train a speech model first:\n"
                        f"  python src/main.py --mode training --dataset ravdess --model speech --output results/demo/speech"
                    )
                    raise ValueError(error_msg)
            elif model_type == 'facial':
                if not has_facial_keys:
                    error_msg = (
                        f"Model architecture mismatch!\n"
                        f"Expected: Facial model\n"
                        f"Found: Different model architecture in {model_path}\n"
                        f"Please train a facial model first:\n"
                        f"  python src/main.py --mode training --dataset ravdess --model facial --output results/demo/facial"
                    )
                    raise ValueError(error_msg)
            
            # Try to load with strict=False first to see what's missing
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                # If strict loading fails, try with strict=False and report issues
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    logger.warning(f"Model loading had issues:")
                    if missing_keys:
                        logger.warning(f"  Missing keys: {len(missing_keys)} (showing first 5)")
                        for key in missing_keys[:5]:
                            logger.warning(f"    - {key}")
                    if unexpected_keys:
                        logger.warning(f"  Unexpected keys: {len(unexpected_keys)} (showing first 5)")
                        for key in unexpected_keys[:5]:
                            logger.warning(f"    - {key}")
                    # If too many keys are missing, it's probably the wrong model
                    if len(missing_keys) > len(model.state_dict()) * 0.3:  # More than 30% missing
                        raise ValueError(
                            f"Model architecture mismatch! Too many missing keys ({len(missing_keys)}). "
                            f"This checkpoint doesn't match the {model_type} model architecture. "
                            f"Please train the correct model type first."
                        )
        except ValueError as e:
            # Re-raise ValueError as-is (our custom error messages)
            raise
        except Exception as e:
            # For other errors, provide context
            raise RuntimeError(
                f"Failed to load model from {model_path}: {e}\n"
                f"This might be due to architecture mismatch. Please ensure you're loading the correct model type."
            ) from e
        
        model.eval()
        
        # Initialize evaluation metrics
        evaluator = EvaluationMetrics()
        
        # Emotion class names for RAVDESS
        emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        
        # Run inference
        all_predictions = []
        all_labels = []
        all_probabilities = []
        inference_times = []
        
        logger.info("Running inference on test set...")
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Evaluating")
            for batch in test_pbar:
                batch_start_time = time.time()
                
                # Extract data based on model type
                if model_type == 'speech':
                    if 'audio_features' not in batch:
                        continue
                    audio_features = batch['audio_features']
                    if isinstance(audio_features, list):
                        # Extract MFCC from features dict
                        mfcc_features = []
                        for feat_dict in audio_features:
                            if isinstance(feat_dict, dict) and 'mfcc' in feat_dict:
                                mfcc = feat_dict['mfcc']
                                if isinstance(mfcc, np.ndarray):
                                    mfcc = torch.from_numpy(mfcc).float()
                                elif not isinstance(mfcc, torch.Tensor):
                                    continue
                                # MFCC shape should be (time, features) = (100, 39)
                                if mfcc.dim() == 2:
                                    mfcc_features.append(mfcc)
                        if not mfcc_features:
                            continue
                        # Stack to get (batch, time, features)
                        x = torch.stack(mfcc_features)  # (batch, time, features)
                    else:
                        # Direct tensor
                        x = audio_features
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x).float()
                        if x.dim() == 2:
                            x = x.unsqueeze(0)  # Add batch dimension
                    labels = batch['label']
                    
                elif model_type == 'facial':
                    if 'video' not in batch:
                        continue
                    video = batch['video']
                    if isinstance(video, torch.Tensor):
                        if video.dim() == 5:
                            x = video[:, 0, :, :, :]
                            x = x.permute(0, 3, 1, 2)
                        else:
                            x = video
                    else:
                        continue
                    labels = batch['label']
                    
                # Move to device
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
                else:
                    continue
                    
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                else:
                    labels = torch.tensor(labels).to(device)
                
                try:
                    # Forward pass
                    outputs = model(x)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Collect results
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    batch_time = time.time() - batch_start_time
                    inference_times.append(batch_time)
                    
                    test_pbar.set_postfix({'acc': np.mean(np.array(all_predictions) == np.array(all_labels)) if len(all_predictions) > 0 else 0})
                except Exception as e:
                    logger.warning(f"Error in evaluation forward pass: {e}")
                    continue
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        if len(all_predictions) == 0:
            raise ValueError("No predictions made. Check data loading and model compatibility.")
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        classification_metrics = evaluator.calculate_classification_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities,
            class_names=emotion_classes
        )
        
        # Calculate performance metrics
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        total_inference_time = np.sum(inference_times) if inference_times else 0.0
        
        evaluation_time = time.time() - start_time
        
        # Prepare evaluation results
        evaluation_results = {
            'dataset': dataset,
            'model_type': model_type,
            'model_path': model_path,
            'test_samples': len(all_labels),
            'metrics': {
                'accuracy': float(classification_metrics['accuracy']),
                'precision': float(classification_metrics['precision']),
                'recall': float(classification_metrics['recall']),
                'f1_score': float(classification_metrics['f1_score']),
                'roc_auc': float(classification_metrics.get('roc_auc', 0.0)),
                'pr_auc': float(classification_metrics.get('pr_auc', 0.0)),
                'balanced_accuracy': float(classification_metrics.get('balanced_accuracy', 0.0))
            },
            'confusion_matrix': classification_metrics.get('confusion_matrix', []),
            'per_class_metrics': {
                emotion_classes[i]: {
                    'precision': float(classification_metrics.get(f'precision_{emotion_classes[i]}', 0.0)),
                    'recall': float(classification_metrics.get(f'recall_{emotion_classes[i]}', 0.0)),
                    'f1_score': float(classification_metrics.get(f'f1_score_{emotion_classes[i]}', 0.0))
                }
                for i in range(len(emotion_classes))
            },
            'performance': {
                'inference_time_per_batch': float(avg_inference_time),
                'total_inference_time': float(total_inference_time),
                'evaluation_time': float(evaluation_time),
                'samples_per_second': len(all_labels) / total_inference_time if total_inference_time > 0 else 0.0
            },
            'device': device
        }
        
        # Save evaluation results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Test Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {evaluation_results['metrics']['f1_score']:.4f}")
        logger.info(f"Total evaluation time: {evaluation_time:.2f} seconds")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


def run_demo(config_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Run a demonstration of the system.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        Demo results
    """
    logger.info("Starting system demonstration...")
    
    try:
        # Run a short simulation
        sim_results = run_simulation(config_path, 300.0, output_dir)  # 5 minutes
        
        # Run training demo
        train_results = run_training(config_path, "daic-woz", "multimodal", output_dir)
        
        # Run evaluation demo
        eval_results = run_evaluation(config_path, "daic-woz", "multimodal", output_dir)
        
        # Combine results
        demo_results = {
            'simulation': sim_results,
            'training': train_results,
            'evaluation': eval_results,
            'demo_summary': {
                'total_time': 3600.0,  # 1 hour
                'components_tested': ['simulation', 'training', 'evaluation'],
                'status': 'completed'
            }
        }
        
        # Save demo results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "demo_results.json"), 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        logger.info("Demo completed successfully")
        return demo_results
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logger(args.log_level, os.path.join(args.output, "logs"))
    
    logger.info("Starting Cognitive Decline Detection System")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Run based on mode
        if args.mode == "simulation":
            results = run_simulation(args.config, args.duration, args.output)
        elif args.mode == "training":
            results = run_training(args.config, args.dataset, args.model, args.output)
        elif args.mode == "evaluation":
            results = run_evaluation(args.config, args.dataset, args.model, args.output)
        elif args.mode == "demo":
            results = run_demo(args.config, args.output)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Status: Completed successfully")
        logger.info(f"Results saved to: {args.output}")
        
        if args.mode == "simulation":
            logger.info(f"Simulation duration: {args.duration} seconds")
            logger.info(f"Communication overhead: {results.get('metrics', {}).get('communication_overhead', 0):.2f} MB")
            logger.info(f"Average latency: {results.get('metrics', {}).get('latency', 0):.2f} ms")
        
        elif args.mode == "training":
            logger.info(f"Training accuracy: {results.get('accuracy', 0):.3f}")
            logger.info(f"Training loss: {results.get('loss', 0):.3f}")
            logger.info(f"Training time: {results.get('training_time', 0):.1f} seconds")
        
        elif args.mode == "evaluation":
            logger.info(f"Evaluation accuracy: {results.get('metrics', {}).get('accuracy', 0):.3f}")
            logger.info(f"F1-score: {results.get('metrics', {}).get('f1_score', 0):.3f}")
            logger.info(f"Inference time: {results.get('performance', {}).get('inference_time', 0):.3f} seconds")
        
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
