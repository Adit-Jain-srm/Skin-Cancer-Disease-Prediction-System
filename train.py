"""
train.py - Model Training Script
Trains the CNN model on HAM10000 dataset.

Usage:
    python train.py --model baseline --epochs 50 --batch_size 32 --lr 0.001
    python train.py --model transfer --backbone resnet50 --epochs 30
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import setup_logging, ensure_directory
from src.dataset import DatasetManager
from src.model import CNNModel, TransferLearningModel

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Train skin cancer classification model"
    )
    parser.add_argument(
        "--model", 
        choices=["baseline", "transfer"], 
        default="baseline",
        help="Model type to train"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Pre-trained backbone for transfer learning"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging("INFO")
    logger.info("Starting model training...")
    
    # Prepare directories
    ensure_directory("models")
    ensure_directory("reports")
    
    # Load dataset
    dataset_mgr = DatasetManager("Dataset/", target_size=(224, 224))
    # TODO: metadata = dataset_mgr.load_metadata("HAM10000_metadata.csv")
    
    # Create model
    if args.model == "baseline":
        model = CNNModel(num_classes=7)
    else:
        model = TransferLearningModel(backbone=args.backbone, num_classes=7)
    
    # TODO: Build and train
    logger.info("Training completed")

if __name__ == "__main__":
    main()
