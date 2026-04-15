"""
Phase 4: Integration Test

Verify all components work together:
1. DataLoader creates batches
2. Model forward pass
3. Trainer training step
4. Metrics computation
"""

import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import DatasetManager
from data_loader import HAM10000DataLoader
from model import CNNBaseline
from trainer import CNNTrainer
from metrics import MetricComputer


def test_full_integration():
    """Test full training pipeline."""
    print("=" * 70)
    print("PHASE 4 - INTEGRATION TEST")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Step 1: Setup dataset
    print("\n[1/5] Loading dataset...")
    dm = DatasetManager(dataset_dir='Dataset', target_size=(224, 224))
    dm.load_metadata('HAM10000_metadata.csv')
    loader = HAM10000DataLoader(dm, batch_size=8, shuffle=False)
    logger.info("✓ Dataset loaded")
    
    # Step 2: Get loaders
    print("\n[2/5] Creating data loaders...")
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    logger.info(f"✓ Train loader: {len(train_loader)} batches")
    logger.info(f"✓ Val loader: {len(val_loader)} batches")
    
    # Step 3: Create model and trainer
    print("\n[3/5] Creating model and trainer...")
    model = CNNBaseline(num_classes=7, dropout=0.5)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=loader.get_class_weights().to(device))
    trainer = CNNTrainer(model, criterion, optimizer, device=device)
    logger.info("✓ Model and trainer created")
    
    # Step 4: Single training epoch
    print("\n[4/5] Running single training epoch...")
    train_loss = trainer.train_epoch(train_loader)
    logger.info(f"✓ Training epoch complete, loss: {train_loss:.4f}")
    
    # Step 5: Validation and metrics
    print("\n[5/5] Validation and metrics...")
    val_loss, val_acc = trainer.validate(val_loader)
    
    # Compute detailed metrics
    metric_computer = MetricComputer(num_classes=7)
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            metric_computer.update(outputs, labels)
    
    metrics = metric_computer.compute_metrics()
    metric_computer.log_metrics(metrics)
    
    print("\n" + "=" * 70)
    print("✅ INTEGRATION TEST PASSED")
    print("=" * 70)
    print(f"✓ Training loss: {train_loss:.4f}")
    print(f"✓ Validation loss: {val_loss:.4f}")
    print(f"✓ Validation accuracy: {val_acc:.2f}%")
    print(f"✓ Test metrics accuracy: {metrics['accuracy']:.4f}")
    print("\nAll components working correctly. Ready for Phase 4 training!")
    print("=" * 70)


if __name__ == '__main__':
    try:
        test_full_integration()
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
