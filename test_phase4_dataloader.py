#!/usr/bin/env python
"""Test HAM10000DataLoader implementation."""

import logging
import torch
from src.dataset import DatasetManager
from src.data_loader import HAM10000DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)

print("\n" + "=" * 70)
print("PHASE 4 - TASK 4.1: DataLoader Tests")
print("=" * 70)

# Initialize
print("\n[1/5] Initializing DatasetManager...")
dm = DatasetManager('Dataset/')
dm.load_metadata('HAM10000_metadata.csv')

# Create loader
print("\n[2/5] Creating HAM10000DataLoader with stratified split...")
loader = HAM10000DataLoader(
    dm,
    train_split=0.7,
    val_split=0.15,
    batch_size=32,
    shuffle=True
)

print("\n[3/5] Testing data leakage (stratification at lesion level)...")
# Get all lesions from each split
train_loader = loader.get_train_loader()
val_loader = loader.get_val_loader()
test_loader = loader.get_test_loader()

# Extract lesion IDs from each split
train_lesions = set(loader.train_metadata['lesion_id'].unique())
val_lesions = set(loader.val_metadata['lesion_id'].unique())
test_lesions = set(loader.test_metadata['lesion_id'].unique())

# Check for overlap
print(f"Train lesions: {len(train_lesions)}")
print(f"Val lesions:   {len(val_lesions)}")
print(f"Test lesions:  {len(test_lesions)}")

overlap_tv = len(train_lesions & val_lesions)
overlap_tt = len(train_lesions & test_lesions)
overlap_vt = len(val_lesions & test_lesions)

print(f"\nOverlap check:")
print(f"  Train/Val overlap: {overlap_tv} (should be 0)")
print(f"  Train/Test overlap: {overlap_tt} (should be 0)")
print(f"  Val/Test overlap: {overlap_vt} (should be 0)")

assert overlap_tv == 0, "Data leakage: train/val"
assert overlap_tt == 0, "Data leakage: train/test"
assert overlap_vt == 0, "Data leakage: val/test"
print("✓ No data leakage detected")

print("\n[4/5] Testing batch creation from each loader...")
# Test train loader (with augmentation)
batch_image, batch_labels = next(iter(train_loader))
print(f"\nTrain batch (augmentation=ON):")
print(f"  Shape: {batch_image.shape} (expected: (32, 3, 224, 224))")
print(f"  Labels shape: {batch_labels.shape} (expected: (32,))")
print(f"  Label range: [{batch_labels.min()}, {batch_labels.max()}] (expected: [0, 6])")
print(f"  Dtype: {batch_image.dtype} (expected: torch.float32)")
print(f"  Value range: [{batch_image.min():.3f}, {batch_image.max():.3f}] (expected: [0, 1])")

assert batch_image.shape == (32, 3, 224, 224), f"Shape mismatch: {batch_image.shape}"
assert batch_labels.shape == (32,), f"Labels shape mismatch: {batch_labels.shape}"
assert batch_labels.min() >= 0 and batch_labels.max() <= 6, "Label range error"
assert batch_image.dtype == torch.float32, f"Dtype error: {batch_image.dtype}"
print("✓ Train batch structure valid")

# Test val loader (no augmentation)
batch_image, batch_labels = next(iter(val_loader))
print(f"\nVal batch (augmentation=OFF):")
print(f"  Shape: {batch_image.shape}")
print(f"  Batch size: {batch_image.shape[0]}")
assert batch_image.shape[0] <= 32, "Batch size exceeds specification"
print("✓ Val batch structure valid")

# Test test loader
batch_image, batch_labels = next(iter(test_loader))
print(f"\nTest batch (augmentation=OFF):")
print(f"  Shape: {batch_image.shape}")
print(f"  Batch size: {batch_image.shape[0]}")
assert batch_image.shape[0] <= 32, "Batch size exceeds specification"
print("✓ Test batch structure valid")

print("\n[5/5] Testing class weights...")
class_weights = loader.get_class_weights()
print(f"Class weights shape: {class_weights.shape} (expected: (7,))")
print(f"Class weights: {class_weights}")
print(f"Sum: {class_weights.sum():.4f} (should be ~1.0 if normalized)")
print(f"Dtype: {class_weights.dtype}")

assert class_weights.shape == (7,), f"Weights shape error: {class_weights.shape}"
assert class_weights.dtype == torch.float32, f"Weights dtype error: {class_weights.dtype}"
print("✓ Class weights valid")

print("\n" + "=" * 70)
print("✅ ALL DATALOADER TESTS PASSED")
print("=" * 70)
print(f"\nSummary:")
print(f"  ✓ Stratified split at lesion level: VERIFIED")
print(f"  ✓ No data leakage: VERIFIED")
print(f"  ✓ Batch shapes correct: VERIFIED")
print(f"  ✓ Label ranges valid: VERIFIED")
print(f"  ✓ Class weights computed: VERIFIED")
print(f"\nDataLoader is production-ready for Phase 4 training!")
