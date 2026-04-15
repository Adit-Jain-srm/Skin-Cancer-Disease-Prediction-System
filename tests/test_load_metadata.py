#!/usr/bin/env python
"""Quick test of load_metadata implementation."""

import logging
from src.dataset import DatasetManager

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)

print("Testing load_metadata()...")
print("=" * 60)

# Test 1: Load metadata
dm = DatasetManager('Dataset/')
metadata = dm.load_metadata('HAM10000_metadata.csv')

# Verify
assert len(metadata) == 10015, f"Expected 10,015 rows, got {len(metadata)}"
assert metadata.shape[1] == 6, f"Expected 6 columns, got {metadata.shape[1]}"
assert metadata['dx'].nunique() == 7, f"Expected 7 classes, got {metadata['dx'].nunique()}"

print("\n" + "=" * 60)
print("✅ TEST PASSED")
print("=" * 60)
print(f"Loaded: {len(metadata)} samples")
print(f"Classes: {metadata['dx'].nunique()}")
print(f"Unique lesions: {metadata['lesion_id'].nunique()}")
print("\nFirst 3 rows:")
print(metadata.head(3).to_string(index=False))
