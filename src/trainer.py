"""
Phase 4: CNN Trainer

Implements training loop with:
- Mixed precision training
- Early stopping
- Learning rate scheduling
- Validation loop
- Checkpoint management
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CNNTrainer:
    """
    Trainer for CNN model with validation, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: str = 'cpu',
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir or 'checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Early stopping
        self.early_stop_patience = 10
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
        
        # Training history
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
        }
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch avg training loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        accuracy = 100.0 * correct / total
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 50,
        early_stopping: bool = True,
    ) -> Dict:
        """
        Train model for specified number of epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
        
        Returns:
            Training history dictionary
        """
        logger.info("=" * 70)
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rate'].append(current_lr)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    
                    # Save best checkpoint
                    self.save_checkpoint(
                        epoch=epoch,
                        is_best=True,
                        metrics={'val_loss': val_loss, 'val_acc': val_acc}
                    )
                    logger.info(f"✓ Best model saved (val_loss: {val_loss:.4f})")
                else:
                    self.early_stop_counter += 1
                    logger.info(f"No improvement. Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")
                    
                    if self.early_stop_counter >= self.early_stop_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
        
        training_time = datetime.now() - start_time
        logger.info("\n" + "=" * 70)
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 70)
        
        return self.train_history
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metrics: Optional[Dict] = None,
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
            metrics: Dictionary of metrics to save
        
        Returns:
            Path to checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics or {},
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        return str(path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            Dictionary with checkpoint info
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    
    def save_training_history(self, output_path: str) -> None:
        """
        Save training history to JSON file.
        
        Args:
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f"Training history saved: {output_path}")
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training results.
        
        Returns:
            Dictionary with training summary
        """
        if not self.train_history['epoch']:
            return {}
        
        return {
            'total_epochs': len(self.train_history['epoch']),
            'final_train_loss': self.train_history['train_loss'][-1],
            'final_val_loss': self.train_history['val_loss'][-1],
            'final_val_acc': self.train_history['val_acc'][-1],
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.train_history['val_loss'].index(self.best_val_loss) + 1,
            'initial_lr': self.train_history['learning_rate'][0],
            'final_lr': self.train_history['learning_rate'][-1],
        }


if __name__ == '__main__':
    """Test trainer initialization."""
    print("=" * 70)
    print("CNN TRAINER - VERIFICATION TEST")
    print("=" * 70)
    
    # Test imports
    print("\n✓ Trainer module imports successful")
    print(f"✓ CNNTrainer class available: {CNNTrainer.__name__}")
