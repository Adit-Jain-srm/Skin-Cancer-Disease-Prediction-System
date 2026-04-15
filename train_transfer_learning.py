"""
Phase 5: Transfer Learning Training Script

Unified training orchestration for ResNet50 and EfficientNet-B3
with enhanced augmentation, class balancing, and regularization.

Usage:
    python train_transfer_learning.py --model resnet50 --lr 1e-3 --batch-size 32
    python train_transfer_learning.py --model efficientnet_b3 --augmentation strong --epochs 100
"""

import torch
import torch.nn as nn
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np

# Try to import from src modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.transfer_learning import TransferLearningModel
from src.enhanced_trainer import EnhancedTrainer
from src.enhanced_augmentation import AugmentationPipeline, AugmentedDataset, BalancedDataLoader
from src.model_manager import ModelManager
from src.dataset import DatasetManager
from src.data_loader import HAM10000DataLoader
from src.metrics import MetricComputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransferLearningTrainer:
    """Main training orchestrator for transfer learning models."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize trainer with configuration."""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("=" * 70)
        logger.info("PHASE 5: TRANSFER LEARNING TRAINING")
        logger.info("=" * 70)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Augmentation: {args.augmentation}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info("=" * 70)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create results directory
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> Tuple:
        """Load and prepare HAM10000 dataset with augmentation."""
        logger.info("\nLoading HAM10000 dataset...")
        
        # Initialize DatasetManager
        dm = DatasetManager(
            dataset_dir=self.args.data_dir,
            target_size=(self.args.image_size, self.args.image_size)
        )
        dm.load_metadata('HAM10000_metadata.csv')
        
        # Load dataset using HAM10000DataLoader
        data_loader = HAM10000DataLoader(
            dm,
            train_split=0.70,
            val_split=0.15,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            random_state=42
        )
        
        # Get dataloaders directly (no separate augmentation wrapping needed)
        train_loader = data_loader.get_train_loader()
        val_loader = data_loader.get_val_loader()
        test_loader = data_loader.get_test_loader()
        
        # Get metadata for labels and class names
        train_labels = data_loader.train_metadata['dx'].values
        val_labels = data_loader.val_metadata['dx'].values
        test_labels = data_loader.test_metadata['dx'].values
        class_names = data_loader.unique_classes
        
        logger.info(f"Train samples: {len(train_labels)}")
        logger.info(f"Val samples: {len(val_labels)}")
        logger.info(f"Test samples: {len(test_labels)}")
        
        return train_loader, val_loader, test_loader, class_names
    
    def create_model(self) -> nn.Module:
        """Create transfer learning model."""
        logger.info(f"\nCreating {self.args.model} model...")
        
        if self.args.model == 'resnet50':
            model = TransferLearningModel.create_resnet50(
                num_classes=7,
                pretrained=True,
                freeze_backbone=True
            )
        elif self.args.model == 'efficientnet_b3':
            model = TransferLearningModel.create_efficientnet_b3(
                num_classes=7,
                pretrained=True,
                freeze_backbone=True
            )
        else:
            raise ValueError(f"Unknown model: {self.args.model}")
        
        model = model.to(self.device)
        
        # Count parameters
        trainable, total = TransferLearningModel.count_parameters(model)
        logger.info(f"Trainable params: {trainable:,} / {total:,}")
        
        return model
    
    def create_loss_fn(self) -> nn.Module:
        """Create loss function with class weights."""
        logger.info("\nCreating loss function...")
        
        # Compute class weights
        if hasattr(self, 'train_labels'):
            class_counts_list = [0] * 7
            for label in self.train_labels:
                class_counts_list[int(label)] += 1
            
            # Weights: inverse of class frequency
            weights = torch.tensor([
                1.0 / max(count, 1) for count in class_counts_list
            ], dtype=torch.float32)
            weights = weights / weights.sum() * 7  # Normalize
            
            logger.info(f"Class weights: {weights.tolist()}")
            loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device))
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        return loss_fn
    
    def train(self):
        """Execute full training pipeline."""
        
        # Load data
        train_loader, val_loader, test_loader, class_names = self.load_data()
        
        # Get train labels from the data_loader stored during load_data
        # We need to reaccess it to get class weights
        dm = DatasetManager(
            dataset_dir=self.args.data_dir,
            target_size=(224, 224)
        )
        dm.load_metadata('HAM10000_metadata.csv')
        data_loader = HAM10000DataLoader(
            dm,
            train_split=0.70,
            val_split=0.15,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            random_state=42
        )
        
        # Convert string labels to indices
        train_labels_str = data_loader.train_metadata['dx'].values
        self.train_labels = [data_loader.class_to_id[label] for label in train_labels_str]
        
        # Create model
        model = self.create_model()
        
        # Create loss function
        loss_fn = self.create_loss_fn()
        
        # Create trainer
        trainer = EnhancedTrainer(
            model=model,
            device=self.device,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            use_ema=True,
            gradient_clip=self.args.gradient_clip,
            use_amp=self.args.use_amp,
            checkpoint_dir=str(self.checkpoint_dir)
        )
        
        # Training loop
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING START")
        logger.info("=" * 70)
        
        best_val_acc = 0.0
        final_epoch = 0
        
        for epoch in range(1, self.args.epochs + 1):
            final_epoch = epoch
            # Train
            train_loss = trainer.train_epoch(train_loader, loss_fn, epoch)
            
            # Validate
            val_loss, val_acc = trainer.validate(val_loader, loss_fn)
            
            # Log metrics
            logger.info(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                trainer.save_checkpoint(
                    epoch,
                    {
                        'val_accuracy': val_acc,
                        'val_loss': val_loss,
                        'train_loss': train_loss
                    }
                )
            
            # Early stopping
            if trainer.early_stop_check(val_loss, patience=self.args.patience):
                logger.info(f"\n✓ Early stopping triggered at epoch {epoch}")
                break
        
        # Save best model
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE - EVALUATING ON TEST SET")
        logger.info("=" * 70)
        
        trainer.save_best_model(str(self.checkpoint_dir / 'best_model.pt'))
        trainer.save_history(str(self.results_dir / 'training_history.json'))
        
        # Evaluate on test set
        test_loss, test_acc = trainer.validate(test_loader, loss_fn)
        
        logger.info(f"\nTest Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        
        # Save summary
        summary = {
            'model': self.args.model,
            'augmentation': self.args.augmentation,
            'learning_rate': self.args.lr,
            'batch_size': self.args.batch_size,
            'weight_decay': self.args.weight_decay,
            'epochs_trained': final_epoch,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = self.results_dir / f'{self.args.model}_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Training summary saved to {summary_path}")
        logger.info(f"✓ Best model saved to {self.checkpoint_dir / 'best_model.pt'}")
        logger.info("=" * 70)
        
        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Transfer Learning Training for HAM10000'
    )
    
    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'efficientnet_b3'],
                       help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='L2 weight decay')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='Dataset',
                       help='Path to HAM10000 dataset')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--augmentation', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='Augmentation level')
    parser.add_argument('--balance-classes', type=bool, default=True,
                       help='Use class-balanced sampling')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    
    # System
    parser.add_argument('--use-amp', type=bool, default=False,
                       help='Use automatic mixed precision')
    
    args = parser.parse_args()
    
    # Run training
    trainer = TransferLearningTrainer(args)
    summary = trainer.train()
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ PHASE 5 TRANSFER LEARNING TRAINING COMPLETE")
    logger.info(f"  Model: {summary['model']}")
    logger.info(f"  Test Accuracy: {summary['test_accuracy']:.2f}%")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
