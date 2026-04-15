# Phase 3: Dataset & Preprocessing Module - Detailed Execution Plan
## Skin Cancer Disease Prediction System

**Phase**: 3/9  
**Duration**: Week 3–4 (2026-04-15 to 2026-04-28)  
**Target Milestone**: M2 Verification (Dataset loaded + end-to-end pipeline functional)  
**Status**: 📋 **PLANNING**

---

## Executive Strategy

### Approach
1. **Implementation-First**: Code implementations, not just docs
2. **Test-Driven**: Each module verified before moving to next
3. **Subagent Offload**: EDA exploration in parallel
4. **Simplicity**: Minimal code, maximum clarity
5. **Verification**: Run pipeline end-to-end before phase complete

### Execution Order
```
Task 3.1 (load_metadata) 
    ↓ [verify metadata loads]
Task 3.2 (preprocessing)
    ↓ [verify single image preprocessing works]
Task 3.3 (augmentation)  
    ↓ [verify augmentation pipeline works]
Task 3.4 (EDA notebook) [PARALLEL with 3.1-3.3]
    ↓
End-to-end pipeline test
    ↓
Phase 3 report
```

---

## Task 3.1: Load Metadata & Validate Dataset

### Objective
Load HAM10000_metadata.csv, parse structure, validate paths, compute statistics.

### Implementation Spec

**Method**: `DatasetManager.load_metadata(metadata_csv: str) → pd.DataFrame`

**Input**:
- `metadata_csv`: Relative path to CSV (e.g., "HAM10000_metadata.csv")

**Process**:
```python
1. Check file exists
2. Load with pd.read_csv()
3. Validate columns: ['image_id', 'dx', 'age', 'sex', 'localization', 'lesion_id']
4. Check data types
5. Compute class distribution
6. Verify image paths exist in Dataset/ directory
7. Generate statistics:
   - Total images, unique lesions
   - Min/max/mean age
   - Gender distribution
   - Per-class counts
8. Return metadata DataFrame
```

**Expected Output**:
```
✅ Metadata loaded: 10,015 rows, 6 columns
✅ Unique lesions: 7,470
✅ Classes: 7 (nevus, melanoma, bkl, bcc, akiec, vasc, df)
✅ Age: mean 51.86, range 0-85
✅ Missing: 57 records (age) → 99.97% complete
✅ Images verified: 10,015/10,015 accessible
```

**Error Handling**:
```python
if not os.path.exists(path):
    raise FileNotFoundError(f"Metadata CSV not found: {path}")

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if len(metadata) == 0:
    raise ValueError("Metadata CSV is empty")
```

**Verification Checklist**:
- [ ] Load 10,015 rows without error
- [ ] All columns present
- [ ] No data type mismatches
- [ ] Class distribution correct (Nevus ~67%, Dermatofibroma ~1%)
- [ ] 100% of images loadable (path validation)
- [ ] Statistics printed and match HAM10000_DATASET_ANALYSIS.md

---

## Task 3.2: Implement Image Preprocessing

### Objective
Preprocess single image from disk to model-ready tensor.

### Implementation Spec

**Method**: `DatasetManager.preprocess_image(image_path: str, target_size=(224, 224)) → np.ndarray`

**Input**:
- `image_path`: Path to image file (JPG/PNG)
- `target_size`: Target resolution (default: 224×224)

**Process**:
```python
1. Load image with PIL.Image.open()
   └─ Handle corrupted: catch PIL.UnidentifiedImageError
   
2. Convert to RGB (handle RGBA, grayscale, etc.)
   └─ If RGBA: drop alpha
   └─ If grayscale: convert to RGB
   
3. Resize to target_size (224×224)
   └─ Use PIL.Image.LANCZOS (high-quality resampling)
   └─ Aspect ratio may change (ok for CNN)
   
4. Optional: Denoise (if needed - check paper)
   └─ Gaussian blur? Median filter? → Decide in testing
   
5. Convert to numpy array
   └─ Shape: (224, 224, 3)
   └─ dtype: uint8 [0-255]
   
6. Normalize to [0, 1]
   └─ Divide by 255: array / 255.0
   └─ dtype: float32
   
7. Return normalized array shape (224, 224, 3), float32
```

**Expected Output**:
```
Input:  HAM10000_images_part_1/ISIC_0024306.jpg (600x450, JPEG)
Output: np.array shape (224, 224, 3), dtype float32, values [0.0, 1.0]

Example values after normalize:
  [[[0.114, 0.145, 0.110],     # Top-left pixel (R, G, B)
    [0.125, 0.156, 0.121],
    ...],
   ...] 
```

**Verification Checklist**:
- [ ] Load 10 random images without error
- [ ] Output shape always (224, 224, 3)
- [ ] Output dtype always float32
- [ ] Output values in [0.0, 1.0]
- [ ] Processing time < 500ms per image
- [ ] Both RGB and RGBA images handled
- [ ] Corrupted images raise helpful error

---

## Task 3.3: Implement Augmentation Pipeline

### Objective
Apply random transformations for training data variety.

### Implementation Spec

**Method**: `DatasetManager.augment_image(image: np.ndarray, augment: bool=True) → np.ndarray`

**Input**:
- `image`: Normalized image array (224, 224, 3), float32 [0,1]
- `augment`: Enable/disable augmentation (False for validation)

**Process** (if augment=True):
```python
1. Random rotation
   └─ Angle: ±15 degrees
   └─ Probability: 100% (always apply)
   
2. Random flip
   └─ Horizontal: 50% probability
   └─ Vertical: 50% probability
   
3. Random brightness
   └─ Factor: ±10% (0.9 to 1.1)
   └─ Probability: 50%
   
4. Random contrast
   └─ Factor: ±10% (0.9 to 1.1)
   └─ Probability: 50%
   
5. Random zoom/crop
   └─ Scale: 0.85 to 1.15
   └─ Probability: 50%
   └─ Crop center to 224×224
   
6. Return augmented image (224, 224, 3), float32, values [0,1]
```

**If augment=False**:
```python
└─ Return image unchanged (for validation/test)
```

**Expected Behavior**:
```
Input:  Original lesion image
Output options:
  - Rotated ±15°
  - Flipped horizontally
  - Brightness adjusted
  - Different zoom level
  
All outputs maintain shape (224, 224, 3) and values in [0, 1]
```

**Implementation Choice**:
- **Option A (Manual)**: Use PIL.Image for rotation/flip, numpy for brightness/contrast
- **Option B (Albumentations)**: Use albumentations library (cleaner, faster)
- **Decision**: Start with PIL/numpy for minimal dependencies; can upgrade to albumentations in Phase 5

**Verification Checklist**:
- [ ] Rotation ±15° applied correctly
- [ ] Flips produce different images for same input (50% of time)
- [ ] Brightness variations visible
- [ ] Output values always in [0, 1]
- [ ] Output shape always (224, 224, 3)
- [ ] augment=False returns unchanged image
- [ ] No duplicate augmentations on same seed

---

## Task 3.4: EDA Notebook

### Objective
Create exploratory notebook with visualizations of dataset and preprocessing.

### Notebook Structure: `notebooks/01_eda.ipynb`

**Cell 1: Setup**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, '/src')
from dataset import DatasetManager

# Config
dataset_dir = "Dataset/"
target_size = (224, 224)
dm = DatasetManager(dataset_dir, target_size)
```

**Cell 2: Load Metadata**
```python
metadata = dm.load_metadata("HAM10000_metadata.csv")
print(f"Metadata loaded: {metadata.shape}")
print(metadata.head(10))
print(metadata.info())
```

**Cell 3: Class Distribution**
```python
fig, ax = plt.subplots(figsize=(10, 5))
metadata['dx'].value_counts().plot(kind='barh', ax=ax)
ax.set_title("HAM10000 Class Distribution")
ax.set_xlabel("Count")
plt.tight_layout()
plt.show()

# Print percentages
print("\nClass Distribution (%)")
print(metadata['dx'].value_counts(normalize=True) * 100)
```

**Cell 4: Preprocessing Demo**
```python
# Load and preprocess a single image
sample_image_id = metadata.iloc[0]['image_id']
image_path = f"{dataset_dir}HAM10000_images_part_1/{sample_image_id}.jpg"

preprocessed = dm.preprocess_image(image_path)
print(f"Shape: {preprocessed.shape}, dtype: {preprocessed.dtype}")
print(f"Min: {preprocessed.min()}, Max: {preprocessed.max()}")

# Display
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(preprocessed)
ax.set_title(f"Preprocessed: {sample_image_id}")
plt.show()
```

**Cell 5: Augmentation Demo**
```python
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    augmented = dm.augment_image(preprocessed, augment=True)
    ax.imshow(augmented)
    ax.set_title(f"Augmentation {i+1}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**Cell 6: Age & Gender Stats**
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Age distribution
metadata['age'].hist(bins=20, ax=axes[0])
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age (years)")

# Gender distribution
metadata['sex'].value_counts().plot(kind='bar', ax=axes[1])
axes[1].set_title("Gender Distribution")
plt.tight_layout()
plt.show()

print(f"Age: mean={metadata['age'].mean():.1f}, std={metadata['age'].std():.1f}")
print(f"Missing age: {metadata['age'].isna().sum()}")
```

**Cell 7: Sample Images per Class**
```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (dx_class, group) in enumerate(metadata.groupby('dx')):
    if i >= 8:
        break
    sample = group.iloc[0]
    image_id = sample['image_id']
    image_path = f"{dataset_dir}HAM10000_images_part_1/{image_id}.jpg"
    
    # Try part 1, fall back to part 2
    if not Path(image_path).exists():
        image_path = image_path.replace('part_1', 'part_2')
    
    preprocessed = dm.preprocess_image(image_path)
    ax = axes[i // 4, i % 4]
    ax.imshow(preprocessed)
    ax.set_title(f"{dx_class}\n(Age: {sample.get('age', 'N/A')})")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

**Cell 8: Data Quality Check**
```python
print("=" * 50)
print("DATA QUALITY REPORT")
print("=" * 50)
print(f"Total records: {len(metadata)}")
print(f"Unique images: {metadata['image_id'].nunique()}")
print(f"Unique lesions: {metadata['lesion_id'].nunique()}")
print(f"Missing values:\n{metadata.isnull().sum()}")
print(f"Data types:\n{metadata.dtypes}")
print(f"\nCompleteness: {(1 - metadata.isnull().sum().sum() / (len(metadata) * len(metadata.columns))) * 100:.2f}%")
```

**Verification**: Notebook runs without errors, produces plots

---

## End-to-End Pipeline Test

### Objective
Verify DatasetManager works for single image and batch operations.

### Test Script: `test_phase3.py`

**Test 1: Metadata Loading**
```python
def test_metadata():
    dm = DatasetManager("Dataset/")
    metadata = dm.load_metadata("HAM10000_metadata.csv")
    
    assert len(metadata) == 10015, f"Expected 10,015 rows, got {len(metadata)}"
    assert set(metadata.columns) == {'image_id', 'dx', 'age', 'sex', 'localization', 'lesion_id'}, "Columns mismatch"
    assert metadata['dx'].nunique() == 7, "Expected 7 classes"
    print("✅ test_metadata PASSED")
```

**Test 2: Preprocessing**
```python
def test_preprocessing():
    dm = DatasetManager("Dataset/")
    image_path = "Dataset/HAM10000_images_part_1/ISIC_0024306.jpg"
    
    img = dm.preprocess_image(image_path)
    
    assert img.shape == (224, 224, 3), f"Shape mismatch: {img.shape}"
    assert img.dtype == np.float32, f"Dtype mismatch: {img.dtype}"
    assert img.min() >= 0 and img.max() <= 1, f"Value range mismatch: [{img.min()}, {img.max()}]"
    print("✅ test_preprocessing PASSED")
```

**Test 3: Augmentation**
```python
def test_augmentation():
    dm = DatasetManager("Dataset/")
    image_path = "Dataset/HAM10000_images_part_1/ISIC_0024306.jpg"
    img = dm.preprocess_image(image_path)
    
    img_aug = dm.augment_image(img, augment=True)
    img_no_aug = dm.augment_image(img, augment=False)
    
    assert img_aug.shape == (224, 224, 3), f"Augmented shape mismatch: {img_aug.shape}"
    assert np.array_equal(img_no_aug, img), "augment=False should return unchanged"
    
    # Augmented should be different (most of the time)
    different_count = np.sum(img_aug != img_no_aug)
    assert different_count > 0, "Augmentation didn't change image"
    print("✅ test_augmentation PASSED")
```

**Test 4: Full Pipeline (10 images)**
```python
def test_full_pipeline():
    dm = DatasetManager("Dataset/")
    metadata = dm.load_metadata("HAM10000_metadata.csv")
    
    for i in range(10):
        row = metadata.iloc[i]
        image_id = row['image_id']
        
        # Try part 1, fall back to part 2
        image_path = f"Dataset/HAM10000_images_part_1/{image_id}.jpg"
        if not Path(image_path).exists():
            image_path = image_path.replace('part_1', 'part_2')
        
        # Process
        img = dm.preprocess_image(image_path)
        img_aug = dm.augment_image(img)
        
        assert img.shape == (224, 224, 3)
        assert img_aug.shape == (224, 224, 3)
    
    print("✅ test_full_pipeline PASSED (10 images)")
```

**Run Tests**:
```bash
python test_phase3.py
```

**Expected Output**:
```
✅ test_metadata PASSED
✅ test_preprocessing PASSED
✅ test_augmentation PASSED
✅ test_full_pipeline PASSED (10 images)

All tests passed! Phase 3 ready.
```

---

## Implementation Order & Parallelization

### Critical Path
```
Task 3.1 (load_metadata)     [2-3 hours] ← Start first
    ↓ (verify)
Task 3.2 (preprocessing)     [2-3 hours]
    ↓ (verify)
Task 3.3 (augmentation)      [2-3 hours]
    ↓ (verify)
End-to-end test             [1 hour]
```

### Parallel Track
```
Task 3.4 (EDA notebook)      [Can run parallel with 3.1-3.3]
    → Uses DatasetManager methods once available
```

### Recommended Schedule
- **Day 1**: Task 3.1 (metadata loading) + start Task 3.4 setup
- **Day 2**: Task 3.2 (preprocessing) + Task 3.4 development
- **Day 3**: Task 3.3 (augmentation) + finish Task 3.4
- **Day 4**: End-to-end testing + Phase 3 report

---

## Acceptance Criteria (Phase 3 Complete)

### Must Have
- ✅ load_metadata() loads 10,015 rows without error
- ✅ preprocess_image() produces (224, 224, 3) float32 tensors
- ✅ augment_image() applies random transformations
- ✅ EDA notebook runs without errors
- ✅ End-to-end pipeline test: all 4 tests pass
- ✅ No errors on 100 random images
- ✅ Phase 3 report documenting implementation

### Should Have
- ✅ Preprocessing time < 500ms per image
- ✅ Error handling for missing/corrupted files
- ✅ Logging for debugging
- ✅ Type hints on all methods

### Nice to Have
- ✅ Performance profile (% time in each step)
- ✅ Memory usage profile
- ✅ Batch loading example

---

## Verification Strategy

Before marking Phase 3 complete, demonstrate:

1. **Metadata Works**:
   ```bash
   python -c "from src.dataset import DatasetManager; dm = DatasetManager('Dataset/'); df = dm.load_metadata('HAM10000_metadata.csv'); print(f'✅ Loaded {len(df)} rows')"
   ```

2. **Preprocessing Works**:
   ```bash
   python -c "from src.dataset import DatasetManager; dm = DatasetManager('Dataset/'); img = dm.preprocess_image('Dataset/HAM10000_images_part_1/ISIC_0024306.jpg'); print(f'✅ Shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]')"
   ```

3. **Augmentation Works**:
   ```bash
   python test_phase3.py
   ```

4. **Notebook Runs**:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
   ```

5. **No errors in logs**:
   ```bash
   grep ERROR logs/*.log
   ```

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Image paths invalid | Low | Validate all paths in load_metadata() |
| Corrupted images | Low | Use try-except in preprocess_image() |
| Out of memory | Low | Process one image at a time (batch in Phase 4) |
| Augmentation too aggressive | Medium | Test parameters, compare before/after visually |
| Notebook fails | Medium | Test each cell independently first |

---

## Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| **Implementation Complete** | All 4 tasks | All methods have code (not just stubs) |
| **Tests Passing** | 100% | `python test_phase3.py` returns all ✅ |
| **Notebook Functional** | Zero errors | Jupyter notebook executes top-to-bottom |
| **Performance** | < 500ms/image | Timer on 100 random images |
| **Correctness** | Visual inspection | EDA plots match dataset characteristics |

---

**Phase 3 Plan Status**: ✅ **READY TO EXECUTE**

Next: Begin Task 3.1 implementation

