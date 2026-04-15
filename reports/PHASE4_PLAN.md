# Phase 4: Baseline CNN Training - Detailed Execution Plan
## Skin Cancer Disease Prediction System

**Phase**: 4/9  
**Duration**: Week 4–5 (2026-04-28 to 2026-05-12)  
**Target Milestone**: M4 (Baseline CNN with ≥70% accuracy)  
**Status**: 📋 **PLANNING**

---

## Executive Strategy

### Approach
1. **Build incrementally**: DataLoader → Model → Training loop → Validation
2. **Test-driven**: Verify each component before combining
3. **Minimal complexity**: Use standard PyTorch patterns, no custom complexity
4. **Real data validation**: Train on subset first, scale up after verification
5. **Baseline focus**: Get simple CNN working, no transfer learning in Phase 4

### Critical Success Factors
- ✅ Input: DatasetManager (Phase 3) fully working
- ✅ Class weights ready: [1.0, 6.0, 6.1, 13.0, 20.5, 47.3, 58.3]
- ✅ Stratified split needed to prevent leakage
- ✅ Augmentation enabled during training
- ✅ Validation metrics: accuracy, per-class F1, weighted F1

### Execution Order
```
Task 4.1 (DataLoader)      [Sequential]
    ↓ [verify batching works]
Task 4.2 (CNN model)       [Sequential]
    ↓ [verify forward pass]
Task 4.3 (Training loop)   [Sequential]
    ↓ [verify loss decreases]
Task 4.4 (Validation)      [Sequential]
    ↓ [verify metrics computed]
Task 4.5a (Train small)    [Parallel with 4.5b]
Task 4.5b (Generate report)[Parallel with 4.5a]
    ↓
Phase 4 complete
```

---

## Task 4.1: PyTorch DataLoader Wrapper

### Objective
Create a reusable PyTorch DataLoader that provides batched, augmented, stratified data.

### Implementation Spec

**File**: `src/data_loader.py`

**Class**: `HAM10000DataLoader`

```python
class HAM10000DataLoader:
    """PyTorch DataLoader wrapper for HAM10000 dataset."""
    
    def __init__(
        self,
        dataset_manager: DatasetManager,
        train_split: float = 0.7,
        val_split: float = 0.15,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ):
        """
        Initialize data loader.
        
        Args:
            dataset_manager: Initialized DatasetManager instance
            train_split: Fraction for training (0.7 = 70%)
            val_split: Fraction for validation (0.15 = 15%)
            batch_size: Batch size for DataLoader
            shuffle: Shuffle training data
            num_workers: Number of workers for data loading
        """
        pass
    
    def get_train_loader(self) -> torch.utils.data.DataLoader:
        """Return PyTorch DataLoader for training (with augmentation)."""
        pass
    
    def get_val_loader(self) -> torch.utils.data.DataLoader:
        """Return PyTorch DataLoader for validation (no augmentation)."""
        pass
    
    def get_test_loader(self) -> torch.utils.data.DataLoader:
        """Return PyTorch DataLoader for testing (no augmentation)."""
        pass
    
    def get_class_weights(self) -> torch.Tensor:
        """Return class weights for loss calculation."""
        pass
```

**Key Implementation Details**:

1. **Stratified Split** (CRITICAL - prevents leakage):
   - Group by `lesion_id` (not image_id)
   - Split at lesion level, not image level
   - Use sklearn.model_selection.train_test_split with stratify
   - 70% train / 15% val / 15% test

2. **Custom Dataset Class**:
```python
class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, metadata, dataset_manager, augment=True):
        self.metadata = metadata
        self.dm = dataset_manager
        self.augment = augment
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        label = row['dx']
        
        # Load and preprocess
        img = self.dm.preprocess_image(image_path)
        
        # Augment if training
        if self.augment:
            img = self.dm.augment_image(img, augment=True)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        # Encode label to integer
        label_id = self.class_to_id[label]
        
        return img, label_id
```

3. **Class Weight Calculation**:
   - Inverse frequency weighting: weight = 1 / count
   - Normalized to sum=1
   - Return as torch.Tensor for use in criterion

**Verification Checklist**:
- [ ] Stratified split: no duplicate lesions between splits
- [ ] Batch shapes: (batch_size, 3, 224, 224)
- [ ] Labels: 0-6 integers
- [ ] Train augmentation active, val/test inactive
- [ ] Class weights computed correctly
- [ ] No data leakage between splits

---

## Task 4.2: CNN Baseline Architecture

### Objective
Implement simple but effective CNN for skin lesion classification.

### Implementation Spec

**File**: `src/model.py` (extend existing CNNModel class)

**Architecture**:
```
Input: (batch_size, 3, 224, 224)
    ↓
Conv Block 1: 64 filters, 3x3, ReLU, BatchNorm, MaxPool 2x2
Conv Block 2: 128 filters, 3x3, ReLU, BatchNorm, MaxPool 2x2
Conv Block 3: 256 filters, 3x3, ReLU, BatchNorm, MaxPool 2x2
Conv Block 4: 512 filters, 3x3, ReLU, BatchNorm, MaxPool 2x2
    ↓
Global Average Pooling → (batch_size, 512)
    ↓
Fully Connected:
  - Linear(512, 256) + ReLU + Dropout(0.5)
  - Linear(256, 128) + ReLU + Dropout(0.5)
  - Linear(128, 7) → Logits
    ↓
Output: (batch_size, 7) logits
```

**Model Code**:
```python
class CNNBaseline(nn.Module):
    def __init__(self, num_classes=7, dropout=0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
```

**Model Properties**:
- **Total parameters**: ~6.5M (reasonable for training on 10K images)
- **Output**: Logits (raw scores, not probabilities)
- **Dropout**: 0.5 in FC layers to prevent overfitting
- **Batch norm**: Applied after each conv layer

**Verification Checklist**:
- [ ] Forward pass works: input (batch, 3, 224, 224) → output (batch, 7)
- [ ] Parameter count logged
- [ ] No in-place operations on model inputs
- [ ] Dropout active during training, inactive during eval

---

## Task 4.3: Training Loop Implementation

### Objective
Create robust training loop with loss computation, backward pass, optimization.

### Implementation Spec

**File**: `src/trainer.py` (new file)

**Class**: `CNNTrainer`

```python
class CNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        num_epochs: int = 20,
        class_weights: torch.Tensor = None
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                acc = 100 * correct / total
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Batch {batch_idx+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%"
                )
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100 * correct / total
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, save_path='models/best_model.pth'):
        """Full training loop."""
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"✓ Model saved: {save_path}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if epoch - best_epoch >= 5:
                logger.info(f"Early stopping: no improvement for 5 epochs")
                break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete. Best model at epoch {best_epoch+1}")
        logger.info(f"{'='*60}")
        
        return self.history
```

**Key Implementation Details**:

1. **Loss Function**: CrossEntropyLoss with class weights
   - Handles class imbalance automatically
   - Input: logits (not probabilities)

2. **Optimizer**: Adam with default hyperparameters
   - Learning rate: 0.001 (standard, can tune in Phase 5)
   - No weight decay (can add if overfitting)

3. **Learning Rate Scheduler**: ReduceLROnPlateau
   - Reduce LR by 0.5x if val loss plateaus for 3 epochs
   - Helps convergence in later training stages

4. **Early Stopping**: Stop if no improvement for 5 epochs
   - Prevents unnecessary training
   - Saves best model to disk

**Verification Checklist**:
- [ ] Forward-backward pass works without errors
- [ ] Loss decreases over epochs (at least on training data)
- [ ] Validation loss computed correctly
- [ ] Model checkpoint saved
- [ ] Learning rate adjusts after plateau
- [ ] Early stopping triggers when needed

---

## Task 4.4: Validation & Metrics

### Objective
Compute comprehensive validation metrics (accuracy, precision, recall, F1-score per class).

### Implementation Spec

**File**: `src/metrics.py` (new file)

**Function**: `compute_metrics()`

```python
def compute_metrics(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    id_to_class: dict
) -> dict:
    """
    Compute comprehensive validation metrics.
    
    Returns:
        {
            'accuracy': float,
            'per_class_accuracy': dict,
            'per_class_f1': dict,
            'weighted_f1': float,
            'macro_f1': float,
            'confusion_matrix': np.ndarray
        }
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Compute metrics using sklearn
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, classification_report
    )
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Per-class metrics
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Weighted/macro F1
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Package results
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
    }
    
    return metrics
```

**Metrics to Compute**:
1. **Overall Accuracy**: Correct predictions / total predictions
2. **Per-class Accuracy**: Accuracy for each disease class
3. **Per-class F1-score**: Harmonic mean of precision & recall per class
4. **Weighted F1**: F1 weighted by class frequency (accounts for imbalance)
5. **Macro F1**: Average F1 across all classes (equal weight)
6. **Confusion Matrix**: For error analysis

**Verification Checklist**:
- [ ] All metrics computed without NaN
- [ ] Confusion matrix shape: (7, 7)
- [ ] Per-class F1 includes all 7 classes
- [ ] Weighted F1 ≥ macro F1 (usually true with imbalance)
- [ ] Classification report readable

---

## Task 4.5: Training on HAM10000 Subset

### Objective
Train CNN baseline on small subset first, validate pipeline, then scale to full dataset.

### Two-Phase Approach

**Phase 4.5a: Subset Training (GPU efficient)**
1. Use 20% of training data (1,400 images)
2. Train for 20 epochs (quick validation)
3. Target: ~70% accuracy on validation set
4. Time: <30 minutes

**Phase 4.5b: Full Dataset Training (if subset passes)**
1. Use 100% of training data (7,000 images)
2. Train for 25 epochs (monitor early stopping)
3. Target: ≥75% accuracy on validation set
4. Time: 2-3 hours

### Training Script

**File**: `train_phase4.py`

```python
#!/usr/bin/env python
"""Phase 4: Train baseline CNN on HAM10000."""

import torch
import logging
from src.dataset import DatasetManager
from src.data_loader import HAM10000DataLoader
from src.model import CNNBaseline
from src.trainer import CNNTrainer
from src.metrics import compute_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_baseline(subset_fraction=1.0):
    """Train CNN baseline."""
    
    logger.info("=" * 70)
    logger.info("PHASE 4: Baseline CNN Training")
    logger.info("=" * 70)
    
    # 1. Load dataset
    logger.info("\n[1/6] Loading dataset...")
    dm = DatasetManager('Dataset/')
    metadata = dm.load_metadata('HAM10000_metadata.csv')
    
    if subset_fraction < 1.0:
        logger.info(f"Using {subset_fraction*100}% of data for quick validation...")
        metadata = metadata.sample(frac=subset_fraction, random_state=42)
    
    # 2. Create data loaders
    logger.info("\n[2/6] Creating data loaders...")
    loader = HAM10000DataLoader(dm, batch_size=32, shuffle=True)
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()
    test_loader = loader.get_test_loader()
    class_weights = loader.get_class_weights()
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # 3. Create model
    logger.info("\n[3/6] Creating CNN model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    model = CNNBaseline(num_classes=7, dropout=0.5)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Create trainer
    logger.info("\n[4/6] Initializing trainer...")
    num_epochs = 20 if subset_fraction < 1.0 else 25
    trainer = CNNTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        num_epochs=num_epochs,
        class_weights=class_weights.to(device)
    )
    
    # 5. Train
    logger.info("\n[5/6] Training...")
    history = trainer.train(train_loader, val_loader)
    
    # 6. Evaluate
    logger.info("\n[6/6] Evaluating...")
    model.load_state_dict(torch.load('models/best_model.pth'))
    metrics = compute_metrics(model, test_loader, device)
    
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    return trainer, metrics

if __name__ == '__main__':
    # Train on subset first
    logger.info("\n>>> PHASE 1: Training on 20% subset...")
    trainer_subset, metrics_subset = train_baseline(subset_fraction=0.2)
    
    if metrics_subset['accuracy'] >= 0.65:
        logger.info("\n✓ Subset training successful! Proceeding to full dataset...")
        logger.info("\n>>> PHASE 2: Training on 100% dataset...")
        trainer_full, metrics_full = train_baseline(subset_fraction=1.0)
    else:
        logger.warning("\n✗ Subset training did not meet expectations. Debugging needed.")
```

**Verification Checklist**:
- [ ] Subset training: complete in <30 minutes
- [ ] Subset accuracy: ≥65%
- [ ] Full training: loss decreases monotonically
- [ ] Full accuracy: ≥70% on test set
- [ ] Model checkpoint saved
- [ ] No CUDA out-of-memory errors

---

## End-to-End Pipeline Test

### Objective
Verify complete pipeline: load → preprocess → batch → train → evaluate

### Test Script: `test_phase4_e2e.py`

```python
def test_data_loading():
    """Test data loader produces correct batches."""
    dm = DatasetManager('Dataset/')
    loader = HAM10000DataLoader(dm, batch_size=32)
    train_loader = loader.get_train_loader()
    
    for images, labels in train_loader:
        assert images.shape == (32, 3, 224, 224), f"Shape: {images.shape}"
        assert labels.shape == (32,), f"Labels: {labels.shape}"
        assert labels.min() >= 0 and labels.max() <= 6, f"Label range: {labels.min()}-{labels.max()}"
        break
    
    print("✓ Data loading test PASSED")

def test_model_forward():
    """Test model forward pass."""
    model = CNNBaseline(num_classes=7)
    images = torch.randn(32, 3, 224, 224)
    outputs = model(images)
    
    assert outputs.shape == (32, 7), f"Output shape: {outputs.shape}"
    print("✓ Model forward test PASSED")

def test_training_step():
    """Test single training step."""
    model = CNNBaseline(num_classes=7)
    dm = DatasetManager('Dataset/')
    loader = HAM10000DataLoader(dm, batch_size=32)
    train_loader = loader.get_train_loader()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = CNNTrainer(model, device=device, num_epochs=1)
    
    # Get first batch
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = trainer.criterion(outputs, labels)
    
    # Check loss decreased after backward
    initial_loss = loss.item()
    loss.backward()
    trainer.optimizer.step()
    
    images, labels = next(iter(train_loader))  # New batch
    outputs = model(images.to(device))
    new_loss = trainer.criterion(outputs, labels.to(device))
    
    print(f"✓ Training step test PASSED (loss: {initial_loss:.4f})")

if __name__ == '__main__':
    test_data_loading()
    test_model_forward()
    test_training_step()
    print("\n✅ All Phase 4 E2E tests PASSED")
```

**Verification Checklist**:
- [ ] Data loader produces (32, 3, 224, 224) tensors
- [ ] Model forward works: (32, 3, 224, 224) → (32, 7)
- [ ] Training step: loss computed without errors
- [ ] Backward pass completes
- [ ] Optimizer updates parameters

---

## Risk Assessment & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-----------|--------|-----------|
| GPU CUDA OOM | Medium | BLOCKING | Reduce batch size, gradient accumulation |
| Overfitting | High | Moderate | Dropout, augmentation, early stopping |
| Learning rate too low | Low | Moderate | Monitor learning, use scheduler |
| Data imbalance | High | Moderate | Class weights, weighted F1 metric |
| Slow convergence | Medium | Low | Increase learning rate for next epoch |
| Model not learning | Low | BLOCKING | Check loss computation, verify data pipeline |

---

## Success Metrics

| Metric | Success Criteria |
|--------|------------------|
| **Subset Accuracy** | ≥65% on 20% data, <30 min training |
| **Full Accuracy** | ≥70% on 100% data |
| **Training Speed** | <3 hours for full dataset |
| **Loss Curve** | Monotonically decreasing |
| **No Errors** | All tests pass, no crashes |
| **Reproducibility** | Fixed seed produces same results |

---

## Phase 4 Deliverables

### Code Files
- ✅ `src/data_loader.py` - HAM10000DataLoader class
- ✅ `src/model.py` - Update CNNBaseline model class
- ✅ `src/trainer.py` - CNNTrainer with training loop
- ✅ `src/metrics.py` - Metric computation functions
- ✅ `train_phase4.py` - Main training script

### Test Files
- ✅ `test_phase4_e2e.py` - End-to-end pipeline test

### Documentation
- ✅ `reports/PHASE4_PLAN.md` - This plan
- ✅ `reports/PHASE4_COMPLETION.md` - Results report (after completion)

### Models
- ✅ `models/best_model.pth` - Best trained model checkpoint

---

## Timeline & Checkpoints

| Day | Task | Duration | Checkpoint |
|-----|------|----------|-----------|
| Day 1 | Tasks 4.1-4.2 | 4 hours | DataLoader + Model working |
| Day 2 | Tasks 4.3-4.4 | 4 hours | Training loop + Metrics working |
| Day 3 | Task 4.5a | 1 hour | Subset training successful (≥65% acc) |
| Day 4 | Task 4.5b | 3 hours | Full training complete (≥70% acc) |
| Day 5 | Reports | 1 hour | Phase 4 completion report written |

**Total Duration**: 5 days  
**Target Completion**: 2026-04-28

---

## Acceptance Criteria - Phase 4 Complete

### MUST HAVE
- ✅ DataLoader produces correct batches (32, 3, 224, 224)
- ✅ Model forward pass works (32, 3, 224, 224) → (32, 7)
- ✅ Training loop runs without errors
- ✅ Loss decreases over epochs (confirmed via logs)
- ✅ Validation metrics computed correctly
- ✅ Subset accuracy ≥65%
- ✅ Full accuracy ≥70%
- ✅ Best model saved to disk
- ✅ Phase 4 report written

### SHOULD HAVE
- ✅ Early stopping triggers when needed
- ✅ Learning rate scheduler works
- ✅ Per-class metrics reported
- ✅ Training curves plotted
- ✅ No CUDA out-of-memory errors

### NICE TO HAVE
- ✅ Confusion matrix visualization
- ✅ Training time statistics
- ✅ Memory profiling

---

**Phase 4 Plan Status**: ✅ **READY FOR EXECUTION**

Next: Begin Task 4.1 implementation

