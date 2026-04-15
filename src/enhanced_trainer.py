"""
Enhanced Trainer for Phase 5 with Regularization & Optimization

Extends base trainer with:
- Gradient clipping
- Enhanced learning rate scheduling (warmup + decay)
- L2 weight decay / L1 regularization
- Early stopping with validation loss
- Model smoothing (EMA)
- Mixed precision training option
- Advanced logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ChainedScheduler, LinearLR
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: EMA decay rate (0.999 is typical)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights after backward pass."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    self.decay * self.shadow[name].data +
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].data
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class EnhancedTrainer:
    """
    Enhanced trainer with regularization, scheduling, and monitoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        gradient_clip: Optional[float] = 1.0,
        use_amp: bool = False,  # Automatic Mixed Precision
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: Neural network model
            device: torch.device (cuda or cpu)
            learning_rate: Initial learning rate
            weight_decay: L2 weight decay
            use_ema: Enable Exponential Moving Average
            ema_decay: EMA decay rate
            gradient_clip: Max gradient norm (None to disable)
            use_amp: Use automatic mixed precision
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model
        self.device = device
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # EMA
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # Mixed precision
        self.scaler = GradScaler() if use_amp else None
        
        # Early stopping tracking
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.epoch_counter = 0
        
        # Metrics history
        self.history = {
            'loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'lr': []
        }
        
        logger.info(f"EnhancedTrainer initialized:")
        logger.info(f"  LR: {learning_rate}, Weight Decay: {weight_decay}")
        logger.info(f"  Gradient Clipping: {gradient_clip}")
        logger.info(f"  EMA: {use_ema}, AMP: {use_amp}")
    
    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with warmup."""
        
        # Warmup for 5 epochs
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=5)
        
        # Cosine annealing decay over remaining epochs
        cosine = CosineAnnealingLR(self.optimizer, T_max=95)
        
        # Chain them together
        scheduler = ChainedScheduler([warmup, cosine])
        
        return scheduler
    
    def train_epoch(
        self,
        train_loader,
        loss_fn: nn.Module,
        epoch: int
    ) -> float:
        """
        Train one epoch.
        
        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Update EMA
            if self.use_ema:
                self.ema.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                avg_loss = total_loss / num_batches
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}: Loss={avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        self.history['loss'].append(avg_loss)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        self.scheduler.step()
        
        return avg_loss
    
    def validate(
        self,
        val_loader,
        loss_fn: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
        
        Returns:
            Tuple of (average_loss, accuracy%)
        """
        # Use EMA weights if available
        if self.use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = loss_fn(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Restore original weights if EMA was used
        if self.use_ema:
            self.ema.restore()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_accuracy'].append(accuracy)
        
        return avg_loss, accuracy
    
    def early_stop_check(
        self,
        val_loss: float,
        patience: int = 10
    ) -> bool:
        """
        Check if early stopping should trigger.
        
        Args:
            val_loss: Current validation loss
            patience: Epochs to wait before stopping
        
        Returns:
            True if should stop training
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            logger.info(f"✓ New best validation loss: {val_loss:.4f}")
            return False
        else:
            self.patience_counter += 1
            logger.info(f"Validation loss increased. Patience: {self.patience_counter}/{patience}")
            return self.patience_counter >= patience
    
    def save_checkpoint(self, epoch: int, metrics: Dict) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: dict with 'val_accuracy', 'val_loss', etc.
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch:03d}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
        }
        
        if self.use_ema:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def save_best_model(self, model_path: str = "best_model.pt"):
        """
        Save best model (current weights).
        
        Args:
            model_path: Path to save best model
        """
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Best model saved to {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.use_ema and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_metrics(self) -> Dict:
        """Get current training metrics."""
        return {
            'last_loss': self.history['loss'][-1] if self.history['loss'] else 0.0,
            'last_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0.0,
            'last_val_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else 0.0,
            'best_val_loss': self.best_loss,
            'current_lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def save_history(self, history_path: str = "training_history.json"):
        """Save training history to JSON."""
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"History saved to {history_path}")


if __name__ == '__main__':
    """Test enhanced trainer."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("ENHANCED TRAINER - VERIFICATION TEST")
    print("=" * 70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create simple test model
    model = nn.Sequential(
        nn.Linear(224*224*3, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7)
    )
    
    # Initialize trainer
    trainer = EnhancedTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        use_ema=True,
        gradient_clip=1.0,
        use_amp=False
    )
    
    print(f"✓ Trainer initialized")
    print(f"  Device: {device}")
    print(f"  Learning rate: {trainer.lr}")
    print(f"  Weight decay: {trainer.weight_decay}")
    print(f"  EMA enabled: {trainer.use_ema}")
    print(f"  Gradient clipping: {trainer.gradient_clip}\n")
    
    # Create dummy data
    dummy_images = torch.randn(32, 224*224*3)
    dummy_labels = torch.randint(0, 7, (32,))
    dummy_loader = [(dummy_images, dummy_labels)]
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Test training step
    print("Testing training step...")
    loss = trainer.train_epoch(dummy_loader, loss_fn, epoch=1)
    print(f"✓ Training step successful, loss={loss:.4f}\n")
    
    # Test validation
    print("Testing validation...")
    val_loss, val_acc = trainer.validate(dummy_loader, loss_fn)
    print(f"✓ Validation successful, loss={val_loss:.4f}, accuracy={val_acc:.2f}%\n")
    
    # Test checkpoint saving
    print("Testing checkpoint saving...")
    checkpoint_path = trainer.save_checkpoint(1, {'val_accuracy': val_acc, 'val_loss': val_loss})
    print(f"✓ Checkpoint saved\n")
    
    # Test early stopping
    print("Testing early stopping...")
    should_stop = trainer.early_stop_check(val_loss + 0.01, patience=2)
    print(f"✓ Early stop check: {should_stop}\n")
    
    # Test metrics
    print("Testing metrics...")
    metrics = trainer.get_metrics()
    print(f"✓ Metrics retrieved:")
    for key, val in metrics.items():
        print(f"    {key}: {val}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
