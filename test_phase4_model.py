"""
Test Phase 4: CNN Baseline Model Integration

Verify:
1. Model instantiation and forward pass
2. Model compatibility with DataLoader output shapes
3. Loss computation with class weights
4. Batch processing through full pipeline
"""

import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dataset import DatasetManager
from data_loader import HAM10000DataLoader
from model import CNNBaseline


def test_model_forward_pass():
    """Test 1: Model forward pass with random input."""
    print("\n[1/4] Testing model forward pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNNBaseline(num_classes=7, dropout=0.5)
    model = model.to(device)
    model.eval()
    
    # Create dummy batch
    dummy_batch = torch.randn(8, 3, 224, 224, device=device)
    
    with torch.no_grad():
        output = model(dummy_batch)
    
    assert output.shape == (8, 7), f"Expected output shape (8, 7), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    print(f"✓ Forward pass successful")
    print(f"  Input shape:  {dummy_batch.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")


def test_model_with_dataloader():
    """Test 2: Model batch processing with real DataLoader."""
    print("\n[2/4] Testing model with DataLoader...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize DatasetManager and DataLoader
    dm = DatasetManager(dataset_dir='Dataset', target_size=(224, 224))
    dm.load_metadata('HAM10000_metadata.csv')
    loader = HAM10000DataLoader(dm, batch_size=8, shuffle=False)
    
    train_loader = loader.get_train_loader()
    
    # Create model
    model = CNNBaseline(num_classes=7, dropout=0.5)
    model = model.to(device)
    model.eval()
    
    # Get first batch
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"  Batch images shape: {images.shape}")
    print(f"  Batch labels shape: {labels.shape}")
    print(f"  Labels unique: {labels.unique().cpu().numpy()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    assert outputs.shape[0] == images.shape[0], "Batch size mismatch"
    assert outputs.shape[1] == 7, "Output classes mismatch"
    print(f"✓ Model batch processing successful")
    print(f"  Output shape: {outputs.shape}")


def test_loss_computation():
    """Test 3: Loss computation with class weights."""
    print("\n[3/4] Testing loss computation with class weights...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize DatasetManager and DataLoader
    dm = DatasetManager(dataset_dir='Dataset', target_size=(224, 224))
    dm.load_metadata('HAM10000_metadata.csv')
    loader = HAM10000DataLoader(dm, batch_size=8, shuffle=False)
    
    # Get class weights
    class_weights = loader.get_class_weights()
    class_weights = class_weights.to(device)
    
    print(f"  Class weights: {class_weights}")
    print(f"  Sum: {class_weights.sum():.4f}")
    print(f"  Dtype: {class_weights.dtype}")
    
    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create model
    model = CNNBaseline(num_classes=7, dropout=0.5)
    model = model.to(device)
    model.train()
    
    # Get batch from loader
    train_loader = loader.get_train_loader()
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(images)
    
    # Compute loss
    loss = criterion(outputs, labels)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Loss computation successful")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Loss dtype: {loss.dtype}")


def test_model_training_step():
    """Test 4: Single training step."""
    print("\n[4/4] Testing single training step...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize DatasetManager and DataLoader
    dm = DatasetManager(dataset_dir='Dataset', target_size=(224, 224))
    dm.load_metadata('HAM10000_metadata.csv')
    loader = HAM10000DataLoader(dm, batch_size=8, shuffle=False)
    
    # Create model and optimizer
    model = CNNBaseline(num_classes=7, dropout=0.5)
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=loader.get_class_weights().to(device))
    
    # Get batch
    train_loader = loader.get_train_loader()
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Training step
    initial_params = [p.clone() for p in model.parameters()]
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Verify parameters changed
    params_changed = False
    for p_init, p_curr in zip(initial_params, model.parameters()):
        if not torch.equal(p_init, p_curr):
            params_changed = True
            break
    
    assert params_changed, "Model parameters did not change after training step"
    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Parameters updated: ✓")


if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 4 - TASK 4.2: CNN Baseline Model Tests")
    print("=" * 70)
    
    try:
        test_model_forward_pass()
        test_model_with_dataloader()
        test_loss_computation()
        test_model_training_step()
        
        print("\n" + "=" * 70)
        print("✅ ALL MODEL TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
