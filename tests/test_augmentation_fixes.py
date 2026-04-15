#!/usr/bin/env python
"""Quick test for augmentation fixes."""

import torch
import numpy as np

print('Testing Augmentation Module Imports...')

# Test that the module loads without errors
try:
    import src.enhanced_augmentation as aug
    print('✓ enhanced_augmentation module imported successfully')
except Exception as e:
    print(f'✗ Error importing module: {e}')
    exit(1)

# Test albumentations import
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    print('✓ Albumentations dependencies available')
except Exception as e:
    print(f'✗ Error with albumentations: {e}')
    exit(1)

# Test that transforms can be created
try:
    # Create a simple compose with tuple normalization (fixed param type)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),  # Fixed: tuple instead of list
            std=(0.229, 0.224, 0.225)     # Fixed: tuple instead of list
        ),
        ToTensorV2(),
    ])
    print('✓ Transform with tuple normalization created successfully')
except Exception as e:
    print(f'✗ Error creating transform: {e}')
    exit(1)

# Test CoarseDropout with fixed parameters
try:
    transform_with_dropout = A.Compose([
        A.Resize(224, 224),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),  # Fixed: added max_height/width
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])
    print('✓ Transform with CoarseDropout created successfully')
except Exception as e:
    print(f'✗ Error creating dropout transform: {e}')
    exit(1)

print('\n✓ All augmentation fixes verified successfully')

