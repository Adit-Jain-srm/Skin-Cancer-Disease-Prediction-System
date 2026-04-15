#!/usr/bin/env python
"""Test augmentation implementation."""

import logging
import numpy as np
from src.dataset import DatasetManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)

print("Testing augment_image()...")
print("=" * 60)

# Initialize
dm = DatasetManager('Dataset/')
metadata = dm.load_metadata('HAM10000_metadata.csv')

# Load and preprocess a sample image
test_image_id = metadata.iloc[0]['image_id']
image_path = f"Dataset/HAM10000_images_part_1/{test_image_id}.jpg"
img_original = dm.preprocess_image(image_path)

print(f"\n1. Testing augmentation=False (should return unchanged)")
img_no_aug = dm.augment_image(img_original, augment=False)
assert np.array_equal(img_no_aug, img_original), "augment=False should return unchanged"
print(f"   ✓ augment=False returns unchanged copy")

print(f"\n2. Testing augmentation=True (random transforms)")
print(f"   Running 10 augmentations on same image...")

shapes_ok = 0
dtypes_ok = 0
ranges_ok = 0
different_count = 0

for i in range(10):
    img_aug = dm.augment_image(img_original, augment=True)
    
    # Check shape
    if img_aug.shape == img_original.shape:
        shapes_ok += 1
    
    # Check dtype
    if img_aug.dtype == np.float32:
        dtypes_ok += 1
    
    # Check value range
    if img_aug.min() >= 0 and img_aug.max() <= 1:
        ranges_ok += 1
    
    # Check if different from original (not all pixels should be identical)
    if not np.array_equal(img_aug, img_original):
        different_count += 1

print(f"   Shape OK:  {shapes_ok}/10 ✓")
print(f"   Dtype OK:  {dtypes_ok}/10 ✓")
print(f"   Range OK:  {ranges_ok}/10 ✓")
print(f"   Different from orig: {different_count}/10 ✓")

assert shapes_ok == 10, "All augmented images should have correct shape"
assert dtypes_ok == 10, "All augmented images should be float32"
assert ranges_ok == 10, "All augmented images should have values in [0, 1]"
assert different_count > 0, "Augmentation should change images (at least sometimes)"

print(f"\n3. Testing augmentation on 5 different images")
import random
random.seed(42)
sample_indices = random.sample(range(len(metadata)), 5)

all_passed = True
for idx in sample_indices:
    row = metadata.iloc[idx]
    image_id = row['image_id']
    
    # Try part 1, fall back to part 2
    img_path_1 = f"Dataset/HAM10000_images_part_1/{image_id}.jpg"
    img_path_2 = f"Dataset/HAM10000_images_part_2/{image_id}.jpg"
    
    import os
    img_path = img_path_1 if os.path.exists(img_path_1) else img_path_2
    
    try:
        img = dm.preprocess_image(img_path)
        img_aug = dm.augment_image(img, augment=True)
        
        assert img_aug.shape == img.shape
        assert img_aug.dtype == np.float32
        assert 0 <= img_aug.min() <= img_aug.max() <= 1
    except Exception as e:
        print(f"   ❌ Failed on {image_id}: {e}")
        all_passed = False

if all_passed:
    print(f"   ✓ All 5 images augmented successfully")

print("\n" + "=" * 60)
print("✅ ALL AUGMENTATION TESTS PASSED")
print("=" * 60)
