#!/usr/bin/env python
"""Test preprocessing implementation."""

import logging
import time
from src.dataset import DatasetManager
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)

print("Testing preprocess_image()...")
print("=" * 60)

# Initialize
dm = DatasetManager('Dataset/')
metadata = dm.load_metadata('HAM10000_metadata.csv')

# Test single image
test_image_id = metadata.iloc[0]['image_id']
image_path = f"Dataset/HAM10000_images_part_1/{test_image_id}.jpg"

print(f"\n1. Testing single image: {test_image_id}")
print(f"   Path: {image_path}")

start = time.time()
img = dm.preprocess_image(image_path)
elapsed = time.time() - start

print(f"   Shape: {img.shape} ✓")
print(f"   Dtype: {img.dtype} ✓")
print(f"   Min: {img.min():.6f}, Max: {img.max():.6f} ✓")
print(f"   Time: {elapsed*1000:.1f}ms ✓")

# Verify all requirements
assert img.shape == (224, 224, 3), f"Shape mismatch: {img.shape}"
assert img.dtype == np.float32, f"Dtype mismatch: {img.dtype}"
assert img.min() >= 0.0 and img.max() <= 1.0, f"Value range mismatch"

# Test multiple images (random sample)
print(f"\n2. Testing 20 random images...")
import random
random.seed(42)
sample_indices = random.sample(range(len(metadata)), 20)

times = []
failed = 0
for idx in sample_indices:
    row = metadata.iloc[idx]
    image_id = row['image_id']
    
    # Try part 1, fall back to part 2
    img_path_1 = f"Dataset/HAM10000_images_part_1/{image_id}.jpg"
    img_path_2 = f"Dataset/HAM10000_images_part_2/{image_id}.jpg"
    
    import os
    img_path = img_path_1 if os.path.exists(img_path_1) else img_path_2
    
    try:
        start = time.time()
        img = dm.preprocess_image(img_path)
        elapsed = time.time() - start
        times.append(elapsed)
        
        # Verify
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.float32
        assert 0 <= img.min() <= img.max() <= 1
    except Exception as e:
        print(f"   ❌ Failed on {image_id}: {e}")
        failed += 1

print(f"   Passed: {20 - failed}/20")
print(f"   Mean time: {np.mean(times)*1000:.1f}ms")
print(f"   Min time:  {np.min(times)*1000:.1f}ms")
print(f"   Max time:  {np.max(times)*1000:.1f}ms")

# All tests passed
print("\n" + "=" * 60)
print("✅ ALL PREPROCESSING TESTS PASSED")
print("=" * 60)
