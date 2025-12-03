#!/usr/bin/env python3
"""
Dataset download script for cognitive decline detection project.

This script downloads and prepares the required datasets:
- DAIC-WOZ (Depression and Anxiety in Context)
- mPower (Mobile Parkinson's Disease)
- RAVDESS (Ryerson Audio-Visual Database)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import requests
import zipfile
import tarfile
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('download_datasets.log')
        ]
    )


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to destination.
    
    Args:
        url: URL to download from
        destination: Destination path
        chunk_size: Chunk size for download
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {destination}")
        
        # Create destination directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"Successfully downloaded {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract archive file.
    
    Args:
        archive_path: Path to archive file
        extract_to: Destination directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting {archive_path} to {extract_to}")
        
        # Create destination directory
        extract_to.mkdir(parents=True, exist_ok=True)
        
        # Extract based on file extension
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"Successfully extracted {archive_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {archive_path}: {e}")
        return False


def download_daic_woz(data_dir: Path) -> bool:
    """
    Download DAIC-WOZ dataset.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading DAIC-WOZ dataset...")
    
    # Note: DAIC-WOZ requires registration and agreement to terms
    # This is a placeholder for the download process
    logger.warning(
        "DAIC-WOZ dataset requires registration. "
        "Please visit https://dcapswoz.ict.usc.edu/ to register and download the dataset manually."
    )
    
    # Create placeholder structure
    daic_woz_dir = data_dir / 'daic_woz'
    daic_woz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files
    (daic_woz_dir / 'README.txt').write_text(
        "DAIC-WOZ dataset placeholder. "
        "Please download the dataset from https://dcapswoz.ict.usc.edu/ and place it here."
    )
    
    return True


def download_mpower(data_dir: Path) -> bool:
    """
    Download mPower dataset.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading mPower dataset...")
    
    # Note: mPower dataset is available through Sage Bionetworks
    # This is a placeholder for the download process
    logger.warning(
        "mPower dataset requires registration. "
        "Please visit https://www.synapse.org/#!Synapse:syn4993293 to register and download the dataset manually."
    )
    
    # Create placeholder structure
    mpower_dir = data_dir / 'mpower'
    mpower_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files
    (mpower_dir / 'README.txt').write_text(
        "mPower dataset placeholder. "
        "Please download the dataset from https://www.synapse.org/#!Synapse:syn4993293 and place it here."
    )
    
    return True


def download_ravdess(data_dir: Path) -> bool:
    """
    Download RAVDESS dataset.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Downloading RAVDESS dataset...")
    
    # Note: RAVDESS dataset is available through Zenodo
    # This is a placeholder for the download process
    logger.warning(
        "RAVDESS dataset requires registration. "
        "Please visit https://zenodo.org/record/1188976 to download the dataset manually."
    )
    
    # Create placeholder structure
    ravdess_dir = data_dir / 'ravdess'
    ravdess_dir.mkdir(parents=True, exist_ok=True)
    
    # Create placeholder files
    (ravdess_dir / 'README.txt').write_text(
        "RAVDESS dataset placeholder. "
        "Please download the dataset from https://zenodo.org/record/1188976 and place it here."
    )
    
    return True


def create_sample_data(data_dir: Path) -> bool:
    """
    Create sample data for testing and development.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating sample data...")
    
    import numpy as np
    import pandas as pd
    from PIL import Image
    
    # Create sample data directory
    sample_dir = data_dir / 'sample_data'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample audio data
    audio_dir = sample_dir / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):
        # Generate random audio data
        audio_data = np.random.randn(16000)  # 1 second at 16kHz
        np.save(audio_dir / f'sample_audio_{i:03d}.npy', audio_data)
    
    # Create sample video data
    video_dir = sample_dir / 'video'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):
        # Generate random image data
        image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_data)
        image.save(video_dir / f'sample_video_{i:03d}.png')
    
    # Create sample gait data
    gait_dir = sample_dir / 'gait'
    gait_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(10):
        # Generate random sensor data
        gait_data = pd.DataFrame({
            'timestamp': np.arange(0, 5, 0.02),  # 5 seconds at 50Hz
            'acc_x': np.random.randn(250),
            'acc_y': np.random.randn(250),
            'acc_z': np.random.randn(250),
            'gyro_x': np.random.randn(250),
            'gyro_y': np.random.randn(250),
            'gyro_z': np.random.randn(250)
        })
        gait_data.to_csv(gait_dir / f'sample_gait_{i:03d}.csv', index=False)
    
    # Create sample text data
    text_dir = sample_dir / 'text'
    text_dir.mkdir(parents=True, exist_ok=True)
    
    sample_texts = [
        "This is a sample text for testing.",
        "Another sample text with different content.",
        "A third sample text for validation.",
        "Sample text number four for testing.",
        "Fifth sample text for development."
    ]
    
    for i, text in enumerate(sample_texts):
        with open(text_dir / f'sample_text_{i:03d}.txt', 'w') as f:
            f.write(text)
    
    # Create metadata
    metadata = {
        'dataset_name': 'sample_data',
        'description': 'Sample data for testing and development',
        'samples': {
            'audio': 10,
            'video': 10,
            'gait': 10,
            'text': 5
        },
        'created_by': 'download_datasets.py'
    }
    
    import json
    with open(sample_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Successfully created sample data")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download datasets for cognitive decline detection')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--datasets', nargs='+', choices=['daic_woz', 'mpower', 'ravdess', 'sample'], 
                       default=['sample'], help='Datasets to download')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading datasets to {data_dir}")
    
    # Download datasets
    success_count = 0
    total_count = len(args.datasets)
    
    for dataset in args.datasets:
        logger.info(f"Processing dataset: {dataset}")
        
        if dataset == 'daic_woz':
            success = download_daic_woz(data_dir)
        elif dataset == 'mpower':
            success = download_mpower(data_dir)
        elif dataset == 'ravdess':
            success = download_ravdess(data_dir)
        elif dataset == 'sample':
            success = create_sample_data(data_dir)
        else:
            logger.error(f"Unknown dataset: {dataset}")
            success = False
        
        if success:
            success_count += 1
            logger.info(f"Successfully processed {dataset}")
        else:
            logger.error(f"Failed to process {dataset}")
    
    # Summary
    logger.info(f"Download complete: {success_count}/{total_count} datasets processed successfully")
    
    if success_count == total_count:
        logger.info("All datasets processed successfully!")
        return 0
    else:
        logger.warning("Some datasets failed to process. Check logs for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
